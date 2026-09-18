#!/usr/bin/env python3
"""Hermetic tests for `ci/scripts/jail_trace.py` — the jail arm's tolerant
`LD_TRACE_LOADED_OBJECTS` trace driver (see #534). No cu12 jail, no real
CUDA binary (the root arm builds a trivial loader-only jail), no `chroot` privilege assumed: `os.chroot` is exercised for
REAL (it is cheap and always available
as a syscall attempt), and its OUTCOME is asserted against the process's own
actual privilege level (`os.geteuid()`), never hardcoded — a non-root
process is GUARANTEED `EPERM` (exit 2); a root process (this repo's CI
containers commonly run as root) genuinely CAN `chroot`, so this suite
instead points the exec at a path that cannot exist inside ANY jail,
exercising the OTHER named failure arm (exit 3) for real.

Run: `python3 ci/scripts/test_jail_trace.py`
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "jail_trace.py"

spec = importlib.util.spec_from_file_location("jail_trace", MODULE_PATH)
jail_trace = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(jail_trace)


class TestJailTrace(unittest.TestCase):
    def test_module_parses(self) -> None:
        # `python3 -m py_compile` would also catch a syntax error; importing
        # it directly (already done at module load, above) is the same
        # property this repo's other hermetic suites check with `bash -n`.
        self.assertTrue(MODULE_PATH.exists())

    def test_child_env_is_pinned_and_minimal(self) -> None:
        """The traced child's environment is EXACTLY `LD_TRACE_LOADED_
        OBJECTS`/`PATH`, never a copy of this process's own ambient
        environment (an inherited `LD_PRELOAD` changes the trace's own
        report shape) — asserted against the literal dict this module
        execve's with, not merely "it probably works".
        """
        self.assertEqual(
            jail_trace._CHILD_ENV,
            {"LD_TRACE_LOADED_OBJECTS": "1", "PATH": "/usr/bin:/bin"},
        )
        # Poison the REAL ambient environment with an LD_PRELOAD-shaped
        # variable and confirm the pinned dict this module actually uses is
        # untouched by it -- the dict is a fixed literal, never derived from
        # `os.environ` at call time.
        os.environ["LD_PRELOAD"] = "/nonexistent/poison.so"
        try:
            self.assertNotIn("LD_PRELOAD", jail_trace._CHILD_ENV)
        finally:
            del os.environ["LD_PRELOAD"]

    def test_main_refuses_when_environment_already_carries_the_hazard_var(self) -> None:
        """A real, if narrower, defense — when this PROCESS's own
        `os.environ` already carries `LD_TRACE_LOADED_OBJECTS` (regardless
        of value, including empty), `main()` refuses immediately, before
        any fork, exit 3. This does NOT cover the case where `python3`
        itself was exec'd with the variable already set (see the module
        doc for why that case is unreachable from inside this file at all
        — checked separately, see
        `test_python3_itself_is_traced_when_the_hazard_var_
        is_already_set_at_exec_time` below).
        """
        os.environ["LD_TRACE_LOADED_OBJECTS"] = "1"
        try:
            rc = jail_trace.main(["jail_trace.py", "/tmp/jail", "/lib64/ld.so", "/bin"])
        finally:
            del os.environ["LD_TRACE_LOADED_OBJECTS"]
        self.assertEqual(rc, 3)

        # Empty-but-present must refuse too -- "present", not "truthy".
        os.environ["LD_TRACE_LOADED_OBJECTS"] = ""
        try:
            rc = jail_trace.main(["jail_trace.py", "/tmp/jail", "/lib64/ld.so", "/bin"])
        finally:
            del os.environ["LD_TRACE_LOADED_OBJECTS"]
        self.assertEqual(rc, 3)

    def test_python3_itself_is_traced_when_the_hazard_var_is_already_set_at_exec_time(self) -> None:
        """Proof that a Python-level check inside this file cannot defend
        against LD_TRACE_LOADED_OBJECTS being set BEFORE `python3` is
        exec'd: `python3`'s own dynamic loader intercepts the process and
        NO Python bytecode ever runs -- confirmed here by running a REAL
        `python3 -c` subprocess with the variable set and asserting its
        stdout is python3's own library trace, never the script's output,
        exit 0. This is exactly why the real guard for that case lives in
        the caller's shell (`release-binaries.yml`), not here.
        """
        if sys.platform != "linux":
            self.skipTest("LD_TRACE_LOADED_OBJECTS is glibc/ld.so-specific; meaningless under macOS's dyld")
        env = dict(os.environ)
        env["LD_TRACE_LOADED_OBJECTS"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", "print('SHOULD_NOT_PRINT')"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("SHOULD_NOT_PRINT", result.stdout)

    def test_main_refuses_wrong_argument_count(self) -> None:
        self.assertEqual(jail_trace.main(["jail_trace.py"]), 3)
        self.assertEqual(jail_trace.main(["jail_trace.py", "one", "two"]), 3)
        self.assertEqual(jail_trace.main(["jail_trace.py", "a", "b", "c", "d", "e"]), 3)

    def test_run_jail_trace_names_the_real_privilege_outcome(self) -> None:
        """The exit-code contract, exercised for REAL: `os.chroot` either
        raises EPERM (non-root — exit 2) or genuinely succeeds (root — exit
        3, from the deliberately-nonexistent loader path below, never from
        the chroot step itself). Never mocked: whichever branch this
        process's own privilege takes IS the property under test.
        """
        with tempfile.TemporaryDirectory() as jail_dir:
            report, rc = jail_trace.run_jail_trace(
                jail_dir,
                "/this-path-cannot-exist-inside-any-jail/ld.so",
                "/jammi-server",
                "/lib",
            )
            if os.geteuid() == 0:
                self.assertEqual(
                    rc,
                    3,
                    "running as root: os.chroot() should SUCCEED (root can chroot to an "
                    "empty, existing directory), so the failure must come from the exec "
                    "step naming a path that cannot exist, not from chroot itself",
                )
            else:
                self.assertEqual(
                    rc,
                    2,
                    "running as non-root: os.chroot() is REQUIRED to raise EPERM — if "
                    "this ever passes with a different code, the privilege model the "
                    "fail-closed property relies on has changed",
                )
            self.assertEqual(report, "", "the failure path (before any exec) must never carry a partial report")

    def test_run_jail_trace_root_can_actually_reach_the_loader(self) -> None:
        """Only meaningful as root (skipped otherwise): a REAL trace of a
        REAL loader against a REAL binary inside a REAL (if trivial) jail —
        proof this mechanism can reach exit 0 at all, not only its two
        failure arms. Uses the host's own `/bin/sh` (or `/bin/ls`) as a
        stand-in "binary" and the host's own dynamic loader at its real
        `PT_INTERP` path, resolved via the SAME `readelf -l` this repo's
        `bundle_binary_interp` uses (kept independent here rather than
        importing bash — this file stays hermetic Python).
        """
        if os.geteuid() != 0:
            self.skipTest("os.chroot requires root; exercised for real only when euid == 0")
        import shutil

        if shutil.which("readelf") is None:
            self.skipTest("no readelf on this host to resolve PT_INTERP")

        candidate = None
        for c in ("/bin/ls", "/usr/bin/ls", "/bin/sh"):
            if os.path.exists(c):
                candidate = c
                break
        if candidate is None:
            self.skipTest("no dynamically-linked binary found to use as the jail's own binary")

        interp_out = subprocess.run(
            ["readelf", "-l", candidate], capture_output=True, text=True, check=False
        ).stdout
        interp = None
        for line in interp_out.splitlines():
            if "Requesting program interpreter" in line:
                interp = line.split("Requesting program interpreter:", 1)[1].strip().rstrip("]").strip()
                break
        if not interp:
            self.skipTest(f"{candidate} carries no PT_INTERP (statically linked?)")

        with tempfile.TemporaryDirectory() as jail_dir:
            jail_lib = Path(jail_dir) / "lib"
            jail_lib.mkdir()
            interp_dest = Path(jail_dir) / interp.lstrip("/")
            interp_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(interp, interp_dest)
            shutil.copy(candidate, Path(jail_dir) / "jammi-server")
            # Best-effort: copy whatever the host loader itself resolves for
            # this binary so the trace has a real chance of completing
            # rather than merely proving the exec step was reached.
            ldd_out = subprocess.run(["ldd", candidate], capture_output=True, text=True, check=False).stdout
            for line in ldd_out.splitlines():
                if " => /" in line:
                    p = line.split(" => ", 1)[1].split(" (")[0].strip()
                    if os.path.exists(p):
                        try:
                            shutil.copy(p, jail_lib)
                        except OSError:
                            pass

            report, rc = jail_trace.run_jail_trace(jail_dir, interp, "/jammi-server", "/lib")
            self.assertEqual(rc, 0, f"expected the trace to complete; report so far:\n{report}")
            self.assertTrue(report.strip(), "expected a non-empty trace report on a successful run")


if __name__ == "__main__":
    unittest.main()
