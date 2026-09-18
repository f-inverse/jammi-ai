#!/usr/bin/env python3
"""Runs the cu12 tarball's jail-arm loader trace inside a `chroot` jail.

`bundle_cuda_libs.sh`'s jail arm cannot use `ld.so --list`: `--list` is
FATAL — exit 127, one error line, no report at all — on the FIRST missing
library, and the jail deliberately ships with the driver libraries
(`libcuda.so.1`, `libnvidia-*`) absent. A rule that needs to see "every
driver name is `not found`" cannot be driven by a mechanism that refuses to
produce a report at all the moment one name is missing.

The mechanism this script drives instead (working on glibc 2.28 and 2.39):
`fork`, `os.chroot` the CHILD into the jail, set
`LD_TRACE_LOADED_OBJECTS=1` in the CHILD's OWN environment AFTER the chroot
syscall, then `os.execve` the dynamic loader AT its real `PT_INTERP` path
(copied to any OTHER name, e.g. `/ld.so`, the loader's own self-named trace
line gains a `=>` the shipped `bundle_parse_loader_report`/`bundle_verify_
jail_report` would misread as "resolved from outside the jail" on an
otherwise CORRECT jail) with `--library-path <lib_search_path> <binary>`
as its arguments. This produces the SAME line shapes `bundle_parse_loader_
report` already parses (`soname => /path (0x...)`, `soname => not found`,
and the loader's own no-`=>` self line), tolerant of any number of `not
found` entries, exit 0.

Why this is a SEPARATE PYTHON PROCESS rather than a shell one-liner:
setting `LD_TRACE_LOADED_OBJECTS=1` on a `chroot <jail> ...` COMMAND LINE
(e.g. `LD_TRACE_LOADED_OBJECTS=1 chroot <jail> ld.so ...`) traces `chroot`'s
OWN dynamic loader, on the HOST, before the `chroot()` syscall ever runs —
a silent VACUOUS PASS (exit 0, a "report" that describes the host's chroot
binary, not the jail at all). Setting the variable only inside the forked
CHILD, strictly AFTER `os.chroot()` succeeds and strictly BEFORE `os.execve`
replaces the child's image, closes that failure mode: there is no separate
`chroot` process to trace, and the env var this script itself sets is never
visible to any process before the jail is entered.

A different, narrower instance of the same class of hazard exists one
level up from the one the fork/chroot ordering above closes: this script
is invoked as `python3 ci/scripts/jail_trace.py ...`, and `python3` ITSELF
is a dynamically-linked executable. If the CALLER's shell already has
`LD_TRACE_LOADED_OBJECTS` set (exported) in its OWN environment BEFORE
`python3` is ever exec'd, the loader that starts `python3` sees it and
traces `python3`'s OWN dependency list instead of ever transferring control
to `python3`'s interpreter: `LD_TRACE_LOADED_OBJECTS=1 python3 -c
"print(2)"` prints `python3`'s own `libc`/`libpython`/... trace, prints
NOTHING from the `-c` script, and exits 0. This means NO Python code in
this file — not even the very first line of `main()` — can ever run in
that exact scenario, because the Python interpreter itself never starts; a
check "at entry" inside this script is structurally unable to defend
against that specific instance, which is why the actual guard against it
lives OUTSIDE this file, in `release-binaries.yml`'s own invocation of it
(checked immediately before `python3` is exec'd, refusing with a named
error rather than silently proceeding). `main()` below still checks
`os.environ` for this variable as its OWN first statement — a real, if
narrower, defense: it catches every case where `python3` itself started
normally (the hazard variable was NOT set at `python3`'s own exec time)
but something set it in this process's `os.environ` before `main()` runs
(e.g. a test harness, or a future caller importing this module rather
than invoking it as a fresh `python3` subprocess) — refusing there is both
possible and worthwhile, it is simply not a COMPLETE defense on its own
the way the fork-scoped fix above is for the child.

The traced child execs with a PINNED, MINIMAL environment (`_CHILD_ENV`,
below) — `LD_TRACE_LOADED_OBJECTS=1` and a bare `PATH`, nothing else —
rather than the calling process's own ambient environment. An inherited
`LD_PRELOAD` changes the trace's own report shape (an extra resolved/
failed entry for whatever it names), and the committed fixture this report
becomes must have no environmental determinant at all — the SAME jail, run
twice on two differently-configured callers, must produce the identical
report.

Exit codes, distinguished so the caller can name the reason (fail closed,
naming the missing capability, never a silent fallback):
  0   the trace ran to completion inside the jail; stdout carries the
      report verbatim (which may still contain `not found` lines — judging
      those is `bundle_verify_jail_report`'s job, never this script's).
  2   `os.chroot` itself was denied (`OSError` with `errno.EPERM`) — the
      caller should report this as a missing capability (`CAP_SYS_CHROOT`),
      never fall back to the detection arm's already-passed result.
  3   any other failure before or during exec (a bad path, a missing
      loader, an exec failure, or `LD_TRACE_LOADED_OBJECTS` already present
      in this process's OWN environment when `main()` starts, refused
      immediately, before any fork) — distinct from a permission denial.

Usage: jail_trace.py <jail_dir> <loader_path_in_jail> <binary_path_in_jail> [lib_search_path]
  `loader_path_in_jail` and `binary_path_in_jail` are paths AS SEEN FROM
  INSIDE THE JAIL (after `chroot`), e.g. `/lib64/ld-linux-x86-64.so.2` and
  `/jammi-server` — never prefixed with `jail_dir`.
"""

from __future__ import annotations

import errno
import os
import sys

# The ONLY environment the traced child ever sees — never the calling
# process's own ambient `os.environ`. `PATH` is included only because glibc
# itself may consult it for nothing this trace needs, kept as a conservative
# minimal default rather than an empty environment some libc build might
# handle unexpectedly; `LD_TRACE_LOADED_OBJECTS` is the one variable this
# mechanism actually depends on.
_CHILD_ENV = {"LD_TRACE_LOADED_OBJECTS": "1", "PATH": "/usr/bin:/bin"}


def run_jail_trace(jail_dir: str, loader_path: str, binary_path: str, lib_search_path: str) -> tuple[str, int]:
    """Returns (captured_stdout, exit_code). Exit codes: see module doc."""
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        # Child: everything here either execve's away or calls os._exit —
        # never returns, never raises past this function, so the parent's
        # own control flow is unaffected by anything that happens here.
        os.close(read_fd)
        os.dup2(write_fd, 1)
        os.close(write_fd)
        try:
            os.chroot(jail_dir)
            os.chdir("/")
        except OSError as exc:
            os._exit(2 if exc.errno == errno.EPERM else 3)
        try:
            os.execve(
                loader_path,
                [loader_path, "--library-path", lib_search_path, binary_path],
                dict(_CHILD_ENV),
            )
        except OSError:
            os._exit(3)
        os._exit(3)  # unreachable: execve either replaces the image or raises

    os.close(write_fd)
    chunks = []
    while True:
        chunk = os.read(read_fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(read_fd)
    _, status = os.waitpid(pid, 0)
    report = b"".join(chunks).decode("utf-8", errors="replace")
    if os.WIFEXITED(status):
        return report, os.WEXITSTATUS(status)
    return report, 3


def main(argv: list[str]) -> int:
    # Refuse immediately, before any fork, if this process's OWN
    # environment already carries LD_TRACE_LOADED_OBJECTS -- a real, if
    # narrower, defense (see the module doc for why this cannot catch the
    # case where the hazard variable was already set before `python3`
    # itself was exec'd: this line of code never runs in that case,
    # because `python3` never reaches it).
    if "LD_TRACE_LOADED_OBJECTS" in os.environ:
        print(
            "jail_trace.py: LD_TRACE_LOADED_OBJECTS is already set in this process's "
            "environment -- refusing before forking. If python3 ITSELF was exec'd with this "
            "variable already present, this message never prints at all (python3's own loader "
            "traces python3, not this script -- see the module doc for why); the real "
            "guard for that case lives in the caller's own shell, before python3 is invoked.",
            file=sys.stderr,
        )
        return 3
    if len(argv) not in (4, 5):
        print(
            "usage: jail_trace.py <jail_dir> <loader_path_in_jail> <binary_path_in_jail> [lib_search_path]",
            file=sys.stderr,
        )
        return 3
    jail_dir, loader_path, binary_path = argv[1], argv[2], argv[3]
    # `bundle_cuda_libs.sh`'s own default (its bundle-able and platform
    # stage directories, colon-joined) — the caller always passes this
    # explicitly today; kept here only as this script's own standalone
    # default.
    lib_search_path = argv[4] if len(argv) == 5 else "/lib:/platform"

    report, rc = run_jail_trace(jail_dir, loader_path, binary_path, lib_search_path)
    sys.stdout.write(report)
    if rc == 2:
        print(
            "jail_trace.py: os.chroot() was denied (EPERM) -- CAP_SYS_CHROOT is likely "
            "missing from this job's container.",
            file=sys.stderr,
        )
    elif rc == 3:
        print(
            "jail_trace.py: the trace failed before/without producing a report "
            "(bad path, missing loader, or exec failure) -- see stderr above for the child's own diagnostic, if any.",
            file=sys.stderr,
        )
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
