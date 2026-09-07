#!/usr/bin/env python3
"""Hermetic shape test for `fa2_ab.sh` (re-audit advisory, folded into #421's
follow-ups): before this test, the `finetune-step` flash/block legs this
script runs were UNLABELED — no `--expect-kernels-disabled` on either
command line, so the only witness that the "flash" leg really ran with
flash enabled (and the "block" leg really disabled it) was a printed
`req`/`fired` line a human had to eyeball in the log. `finetune_ab.sh:582`'s
own convention passes `--expect-kernels-disabled` on EVERY leg, even the
empty-string case, so the binary's own START check
(`finetune_step.rs::run`, before any device/checkpoint/tensor work) and END
check (`unmatched_disables`) gate the claim instead of a human reading a
line of stdout.

`fa2_ab.sh` is a manual, exclusive-timing-box script (hardcoded `/root/...`
paths, an `nvidia-smi` call, a `cargo build --release` against a live
`perf/p6-fa2-dense` worktree) with no `DRY_RUN` support and nothing to gain
from one: there is no producer pipeline downstream of it to exercise
hermetically. So — per the follow-up brief — this test greps the actual
command arrays in the committed script text (never a re-implementation of
its control flow) rather than driving a dry run, plus a shellcheck pass on
the script itself.

Run: `python3 ci/scripts/perf/test_fa2_ab_sh.py`
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import unittest

PERF_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(PERF_DIR, "fa2_ab.sh")


def _read_script() -> str:
    with open(SCRIPT, encoding="utf-8") as fh:
        return fh.read()


class TestFa2AbShShape(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _read_script()

    def test_block_leg_names_its_own_disable_key(self) -> None:
        """The `if [ $leg = block ]` branch must run `JAMMI_KERNELS_DISABLE=$K`
        AND pass `--expect-kernels-disabled "$K"` on the SAME command line --
        the same op key, not just any non-empty expectation -- so the
        binary's own START check refuses before any step runs if the two
        ever disagree (a typo, a dropped env var, or an ambient
        `JAMMI_KERNELS_DISABLE` leaking in)."""
        m = re.search(
            r"^\s*if \[ \$leg = block \]; then (.*)$", self.text, re.MULTILINE
        )
        self.assertIsNotNone(m, "expected an `if [ $leg = block ]; then ...` line")
        block_line = m.group(1)
        self.assertIn(
            "JAMMI_KERNELS_DISABLE=$K",
            block_line,
            "block leg must set JAMMI_KERNELS_DISABLE=$K",
        )
        self.assertIn(
            '--expect-kernels-disabled "$K"',
            block_line,
            "block leg must pass --expect-kernels-disabled \"$K\" -- the "
            "SAME key it disables via JAMMI_KERNELS_DISABLE, not left "
            "unlabeled",
        )

    def test_flash_leg_passes_empty_expectation(self) -> None:
        """The `else` (flash) branch must NOT set `JAMMI_KERNELS_DISABLE` and
        must pass `--expect-kernels-disabled ""` -- an exact-set-equality
        guard against an ambient `JAMMI_KERNELS_DISABLE` leaking into this
        process from the calling shell/CI runner and silently turning the
        "flash" leg back into the block leg wearing a flash label."""
        m = re.search(r"^\s*else (.*); fi$", self.text, re.MULTILINE)
        self.assertIsNotNone(m, "expected an `else ...; fi` line")
        flash_line = m.group(1)
        self.assertNotIn(
            "JAMMI_KERNELS_DISABLE=",
            flash_line,
            "flash leg must not set JAMMI_KERNELS_DISABLE",
        )
        self.assertIn(
            '--expect-kernels-disabled ""',
            flash_line,
            'flash leg must pass --expect-kernels-disabled "" so an '
            "ambient JAMMI_KERNELS_DISABLE cannot leak in unnoticed",
        )

    def test_both_legs_run_kernels_strict(self) -> None:
        """Both legs keep `JAMMI_KERNELS_STRICT=1` -- an eligible-but-failed
        fused op must ERROR, never silently fall back to eager numbers
        wearing a fused label (admission.rs's disable-wins-over-strict
        contract)."""
        for line_re in (
            r"^\s*if \[ \$leg = block \]; then (.*)$",
            r"^\s*else (.*); fi$",
        ):
            m = re.search(line_re, self.text, re.MULTILINE)
            self.assertIsNotNone(m)
            self.assertIn("JAMMI_KERNELS_STRICT=1", m.group(1))

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


if __name__ == "__main__":
    unittest.main()
