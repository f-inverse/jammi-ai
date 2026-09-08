#!/usr/bin/env python3
"""Hermetic shape test for `fa2_ab.sh`: the `finetune-step` flash/block legs
this script runs each pass `--expect-kernels-disabled` explicitly (empty on
the flash leg, the disabled op key on the block leg), matching
`finetune_ab.sh:582`'s own convention, so the binary's own START check
(`finetune_step.rs::run`, before any device/checkpoint/tensor work) and END
check (`unmatched_disables`) gate the claim -- the "flash" leg really ran
with flash enabled and the "block" leg really disabled it, proven by the
binary's own refusal path rather than a human eyeballing a printed
`req`/`fired` line in the log. A refused leg also moves this script's own
exit status (`overall_rc`), never merely a `FAILED` line in scrollback.

`fa2_ab.sh` is a manual, exclusive-timing-box script (hardcoded `/root/...`
paths, an `nvidia-smi` call, a `cargo build --release` against a live
`perf/p6-fa2-dense` worktree) with no `DRY_RUN` support and nothing to gain
from one: there is no producer pipeline downstream of it to exercise
hermetically. So this test greps the actual command arrays in the committed
script text (never a re-implementation of its control flow) rather than
driving a dry run, plus a shellcheck pass on the script itself and a REAL
execution of the exit-status-propagation control flow: the per-leg step
body fa2_ab.sh's sweep loop calls lives in `fa2_ab_leg.sh`
(`fa2_ab_run_leg`), and this suite `source`s that file -- the exact code
fa2_ab.sh runs, never a hand-written stand-in of it -- against a stub
`finetune-step` replacement that succeeds on one call and fails on the
next, asserting the SOURCED function's own `overall_rc` moves.

Run: `python3 ci/scripts/perf/test_fa2_ab_sh.py`
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import tempfile
import unittest

PERF_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(PERF_DIR, "fa2_ab.sh")
LEG_SCRIPT = os.path.join(PERF_DIR, "fa2_ab_leg.sh")


def _read_script(path: str = SCRIPT) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class TestFa2AbShShape(unittest.TestCase):
    """The flash/block leg-dispatch shape now lives in `fa2_ab_leg.sh`
    (`fa2_ab_run_leg`), sourced by `fa2_ab.sh`'s sweep loop -- these tests
    grep the file that actually contains it."""

    def setUp(self) -> None:
        self.text = _read_script()
        self.leg_text = _read_script(LEG_SCRIPT)

    @staticmethod
    def _if_else_bodies(text: str) -> tuple[str, str]:
        m = re.search(
            r'if \[ "\$leg" = block \]; then\n(.*?)\n\s*else\n(.*?)\n\s*fi',
            text,
            re.DOTALL,
        )
        assert m is not None, "expected an `if [ \"$leg\" = block ]; then ... else ... fi` block"
        return m.group(1), m.group(2)

    def test_block_leg_names_its_own_disable_key(self) -> None:
        """The `if [ "$leg" = block ]` branch must run
        `JAMMI_KERNELS_DISABLE="$disable_key"` AND pass
        `--expect-kernels-disabled "$disable_key"` on the SAME command line
        -- the same op key, not just any non-empty expectation -- so the
        binary's own START check refuses before any step runs if the two
        ever disagree (a typo, a dropped env var, or an ambient
        `JAMMI_KERNELS_DISABLE` leaking in)."""
        block_body, _flash_body = self._if_else_bodies(self.leg_text)
        self.assertIn(
            'JAMMI_KERNELS_DISABLE="$disable_key"',
            block_body,
            "block leg must set JAMMI_KERNELS_DISABLE=\"$disable_key\"",
        )
        self.assertIn(
            '--expect-kernels-disabled "$disable_key"',
            block_body,
            "block leg must pass --expect-kernels-disabled \"$disable_key\" -- the "
            "SAME key it disables via JAMMI_KERNELS_DISABLE, not left "
            "unlabeled",
        )

    def test_flash_leg_passes_empty_expectation(self) -> None:
        """The `else` (flash) branch must NOT set `JAMMI_KERNELS_DISABLE` and
        must pass `--expect-kernels-disabled ""` -- an exact-set-equality
        guard against an ambient `JAMMI_KERNELS_DISABLE` leaking into this
        process from the calling shell/CI runner and silently turning the
        "flash" leg back into the block leg wearing a flash label."""
        _block_body, flash_body = self._if_else_bodies(self.leg_text)
        self.assertNotIn(
            "JAMMI_KERNELS_DISABLE=",
            flash_body,
            "flash leg must not set JAMMI_KERNELS_DISABLE",
        )
        self.assertIn(
            '--expect-kernels-disabled ""',
            flash_body,
            'flash leg must pass --expect-kernels-disabled "" so an '
            "ambient JAMMI_KERNELS_DISABLE cannot leak in unnoticed",
        )

    def test_both_legs_run_kernels_strict(self) -> None:
        """Both legs keep `JAMMI_KERNELS_STRICT=1` -- an eligible-but-failed
        fused op must ERROR, never silently fall back to eager numbers
        wearing a fused label (admission.rs's disable-wins-over-strict
        contract)."""
        block_body, flash_body = self._if_else_bodies(self.leg_text)
        self.assertIn("JAMMI_KERNELS_STRICT=1", block_body)
        self.assertIn("JAMMI_KERNELS_STRICT=1", flash_body)

    def test_fa2_ab_sh_sources_the_leg_file_and_calls_the_shared_function(self) -> None:
        """`fa2_ab.sh`'s sweep loop must call `fa2_ab_run_leg` (never keep
        its own copy of the step body inline) after sourcing `fa2_ab_leg.sh`
        -- otherwise the two files could drift into two different
        implementations of "run one leg"."""
        self.assertIn('. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fa2_ab_leg.sh"', self.text)
        self.assertIn("fa2_ab_run_leg ", self.text)
        self.assertIn("overall_rc=0", self.text)
        self.assertIn('echo "FA2AB_EXIT=$overall_rc', self.text)
        self.assertIn("exit $overall_rc", self.text)

    def test_leg_sh_carries_the_step_rc_parse_rc_overall_rc_bookkeeping(self) -> None:
        """The bookkeeping a refused leg (the binary's own START/END check,
        or a JSON-parse failure on the emitted report) rides to move
        `overall_rc` now lives in `fa2_ab_leg.sh`, not inline in
        `fa2_ab.sh` -- greeped off the committed text of the file that
        actually contains it."""
        leg_text = _read_script(LEG_SCRIPT)
        self.assertIn("step_rc=$?", leg_text)
        self.assertIn("parse_rc=$?", leg_text)
        self.assertIn(
            'if [ "$step_rc" -ne 0 ] || [ "$parse_rc" -ne 0 ]; then overall_rc=1; fi',
            leg_text,
        )

    def test_parse_only_arm_executed_for_real(self) -> None:
        """A stub `finetune-step` that EXITS 0 but writes a malformed
        report: `step_rc` is 0, so this exercises the parser's OWN
        `except` branch through the REAL sourced `fa2_ab_run_leg` -- never
        a grep for `sys.exit(1)` in the committed text, which would pass
        even if that branch were unreachable or its exit code wrong. A
        malformed/missing report must still move `overall_rc` to 1 via
        `parse_rc`, exactly as a genuinely broken binary would."""
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "out")
            os.makedirs(out_dir)
            stub_path = os.path.join(tmp, "finetune-step-stub.sh")
            with open(stub_path, "w", encoding="utf-8") as fh:
                fh.write(
                    "#!/usr/bin/env bash\n"
                    "echo 'not a json report'\n"
                    "exit 0\n"
                )
            os.chmod(stub_path, os.stat(stub_path).st_mode | stat.S_IEXEC)

            harness = f"""
set -uo pipefail
. "{LEG_SCRIPT}"
overall_rc=0
fa2_ab_run_leg "{stub_path}" "{out_dir}" K flash 8 128 r1 dummyarg
echo "STEP_RC=$step_rc PARSE_RC=$parse_rc OVERALL=$overall_rc"
"""
            result = subprocess.run(
                ["bash", "-c", harness], capture_output=True, text=True, check=False
            )
            self.assertIn(
                "STEP_RC=0 PARSE_RC=1 OVERALL=1",
                result.stdout,
                f"a malformed report from an otherwise-successful binary did not move "
                f"parse_rc/overall_rc as expected:\nstdout={result.stdout}\nstderr={result.stderr}",
            )

    def test_sourcing_the_real_leg_file_moves_overall_rc_on_a_failing_leg(self) -> None:
        """Not a re-implementation: this `source`s the ACTUAL
        `fa2_ab_leg.sh` (the exact file `fa2_ab.sh` sources in its own
        sweep loop) and calls its `fa2_ab_run_leg` twice against a stub
        `finetune-step` replacement that succeeds on the first call and
        fails (nonzero exit, no report) on the second -- proving the
        SHIPPED code's `overall_rc` moves from 0 to 1 across that exact
        transition, not a hand-written stand-in of its control flow."""
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "out")
            os.makedirs(out_dir)
            counter_path = os.path.join(tmp, "calls")
            stub_path = os.path.join(tmp, "finetune-step-stub.sh")
            with open(stub_path, "w", encoding="utf-8") as fh:
                fh.write(
                    "#!/usr/bin/env bash\n"
                    f'n=0; [ -f "{counter_path}" ] && n=$(cat "{counter_path}")\n'
                    f'n=$((n + 1)); echo "$n" > "{counter_path}"\n'
                    'if [ "$n" -eq 1 ]; then\n'
                    "  cat <<'JSON'\n"
                    '{"tiers": {"finetune_step": {"s_per_step_p50": {"value": 0.1}}}}\n'
                    "JSON\n"
                    "  exit 0\n"
                    "else\n"
                    '  echo "stub refusal" >&2\n'
                    "  exit 7\n"
                    "fi\n"
                )
            os.chmod(stub_path, os.stat(stub_path).st_mode | stat.S_IEXEC)

            harness = f"""
set -uo pipefail
. "{LEG_SCRIPT}"
overall_rc=0
fa2_ab_run_leg "{stub_path}" "{out_dir}" K flash 8 128 r1 dummyarg
first="$overall_rc"
fa2_ab_run_leg "{stub_path}" "{out_dir}" K flash 8 128 r2 dummyarg
echo "FIRST=$first SECOND=$overall_rc"
"""
            result = subprocess.run(
                ["bash", "-c", harness], capture_output=True, text=True, check=False
            )
            self.assertIn(
                "FIRST=0 SECOND=1",
                result.stdout,
                f"sourced fa2_ab_run_leg's overall_rc did not move as expected:\n"
                f"stdout={result.stdout}\nstderr={result.stderr}",
            )

    def test_block_leg_forwards_env_and_flags_for_real(self) -> None:
        """Not a grep of the `if [ "$leg" = block ]` body: this `source`s
        the ACTUAL `fa2_ab_leg.sh` and calls `fa2_ab_run_leg` with
        `leg=block`, against a stub `finetune-step` that records its OWN
        environment and argv to a file before succeeding -- proving the
        SHIPPED code really sets `JAMMI_KERNELS_DISABLE`/
        `JAMMI_KERNELS_STRICT` in the stub's environment and really passes
        `--expect-kernels-disabled` with the SAME key, `--batch`, and
        `--seq` on its argv, not merely that the committed text contains
        those substrings somewhere."""
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "out")
            os.makedirs(out_dir)
            record_path = os.path.join(tmp, "record.txt")
            stub_path = os.path.join(tmp, "finetune-step-stub.sh")
            with open(stub_path, "w", encoding="utf-8") as fh:
                fh.write(
                    "#!/usr/bin/env bash\n"
                    "{\n"
                    '  echo "ENV_DISABLE=${JAMMI_KERNELS_DISABLE-<unset>}"\n'
                    '  echo "ENV_STRICT=${JAMMI_KERNELS_STRICT-<unset>}"\n'
                    '  echo "ARGS=$*"\n'
                    f'}} > "{record_path}"\n'
                    "cat <<'JSON'\n"
                    '{"tiers": {"finetune_step": {"s_per_step_p50": {"value": 0.2}}}}\n'
                    "JSON\n"
                    "exit 0\n"
                )
            os.chmod(stub_path, os.stat(stub_path).st_mode | stat.S_IEXEC)

            harness = f"""
set -uo pipefail
. "{LEG_SCRIPT}"
overall_rc=0
fa2_ab_run_leg "{stub_path}" "{out_dir}" MY_DISABLE_KEY block 8 128 r1 --some-config-flag
echo "OVERALL=$overall_rc"
"""
            result = subprocess.run(
                ["bash", "-c", harness], capture_output=True, text=True, check=False
            )
            self.assertIn(
                "OVERALL=0",
                result.stdout,
                f"a well-formed block-leg stub run must not refuse:\n"
                f"stdout={result.stdout}\nstderr={result.stderr}",
            )
            with open(record_path, encoding="utf-8") as fh:
                record = fh.read()
            self.assertIn("ENV_DISABLE=MY_DISABLE_KEY", record)
            self.assertIn("ENV_STRICT=1", record)
            self.assertIn(
                "ARGS=finetune-step --some-config-flag --batch 8 --seq 128 "
                "--expect-kernels-disabled MY_DISABLE_KEY",
                record,
            )

    def test_flash_leg_forwards_ambient_env_and_empty_expectation_for_real(self) -> None:
        """The mirror of `test_block_leg_forwards_env_and_flags_for_real`,
        exercising the `leg=flash` arm through the REAL sourced
        `fa2_ab_run_leg` (never a grep of the `else` body): an AMBIENT
        `JAMMI_KERNELS_DISABLE`, set in the calling shell before the harness
        runs (never set BY `fa2_ab_run_leg` itself -- the flash branch has
        no `JAMMI_KERNELS_DISABLE=` assignment of its own), must still reach
        the stub's environment UNCHANGED, because bash forwards an already-
        exported variable to a child process whether or not the command
        prefix names it. The stub also records `--expect-kernels-disabled
        ""` on argv exactly as the block-leg test records
        `--expect-kernels-disabled MY_DISABLE_KEY`, so this proves the same
        FOR-REAL property on the leg whose whole purpose is catching that
        ambient leak (see the module doc and `test_flash_leg_passes_empty_
        expectation`'s grep-level counterpart)."""
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "out")
            os.makedirs(out_dir)
            record_path = os.path.join(tmp, "record.txt")
            stub_path = os.path.join(tmp, "finetune-step-stub.sh")
            with open(stub_path, "w", encoding="utf-8") as fh:
                fh.write(
                    "#!/usr/bin/env bash\n"
                    "{\n"
                    '  echo "ENV_DISABLE=${JAMMI_KERNELS_DISABLE-<unset>}"\n'
                    '  echo "ENV_STRICT=${JAMMI_KERNELS_STRICT-<unset>}"\n'
                    '  echo "ARGS=$*"\n'
                    f'}} > "{record_path}"\n'
                    "cat <<'JSON'\n"
                    '{"tiers": {"finetune_step": {"s_per_step_p50": {"value": 0.2}}}}\n'
                    "JSON\n"
                    "exit 0\n"
                )
            os.chmod(stub_path, os.stat(stub_path).st_mode | stat.S_IEXEC)

            harness = f"""
set -uo pipefail
export JAMMI_KERNELS_DISABLE=AMBIENT_LEAK
. "{LEG_SCRIPT}"
overall_rc=0
fa2_ab_run_leg "{stub_path}" "{out_dir}" UNUSED_DISABLE_KEY flash 8 128 r1 --some-config-flag
echo "OVERALL=$overall_rc"
"""
            result = subprocess.run(
                ["bash", "-c", harness], capture_output=True, text=True, check=False
            )
            self.assertIn(
                "OVERALL=0",
                result.stdout,
                f"a well-formed flash-leg stub run must not refuse:\n"
                f"stdout={result.stdout}\nstderr={result.stderr}",
            )
            with open(record_path, encoding="utf-8") as fh:
                record = fh.read()
            self.assertIn(
                "ENV_DISABLE=AMBIENT_LEAK",
                record,
                "an ambient JAMMI_KERNELS_DISABLE set before the flash leg runs must reach "
                "the child process unchanged -- the flash branch never unsets it, it only "
                "declines to set it itself",
            )
            self.assertIn("ENV_STRICT=1", record)
            self.assertIn(
                "ARGS=finetune-step --some-config-flag --batch 8 --seq 128 "
                "--expect-kernels-disabled \n",
                record,
                f"expected the flash leg's argv to end in an EMPTY --expect-kernels-disabled "
                f"value (a trailing space, no key):\n{record!r}",
            )

    def test_syntax_is_valid_bash(self) -> None:
        """`bash -n` is a pure parse check -- no network, no GPU, no /root
        paths touched."""
        result = subprocess.run(
            ["bash", "-n", SCRIPT], capture_output=True, text=True, check=False
        )
        self.assertEqual(
            result.returncode, 0, f"bash -n {SCRIPT} failed:\n{result.stderr}"
        )

    @unittest.skipUnless(shutil.which("shellcheck"), "shellcheck not installed")
    def test_shellcheck_clean_at_warning_severity(self) -> None:
        """`-S warning` intentionally excludes the pre-existing style/info
        findings (SC1091 for /root/.jammi_env not existing on this host by
        design, SC2086 unquoted loop variables the script's own author
        chose) that are not part of this follow-up's scope; it still fails
        on any real warning/error this or a future edit introduces."""
        result = subprocess.run(
            ["shellcheck", "-S", "warning", "-x", SCRIPT],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"shellcheck -S warning flagged fa2_ab.sh:\n{result.stdout}\n{result.stderr}",
        )

    @unittest.skipUnless(shutil.which("shellcheck"), "shellcheck not installed")
    def test_leg_sh_is_shellcheck_clean_at_warning_severity(self) -> None:
        """`fa2_ab_leg.sh` is sourced, never executed standalone -- its own
        `SC2034` on `overall_rc` (assigned for the caller that sources it)
        is suppressed inline (see that file's own comment), so a clean
        shellcheck pass here catches any OTHER warning a future edit
        introduces."""
        result = subprocess.run(
            ["shellcheck", "-S", "warning", "-x", LEG_SCRIPT],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"shellcheck -S warning flagged fa2_ab_leg.sh:\n{result.stdout}\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
