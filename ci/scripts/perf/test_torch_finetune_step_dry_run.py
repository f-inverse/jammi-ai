#!/usr/bin/env python3
# lane: torch-host
# needs: torch-venv
"""Every `train-step` identity field, as `impl Payload for TrainStepPayload`
declares it, read off the report a real `torch_finetune_step.py --dry-run`
writes.

Whether a field's VALUE is non-null is decided at run time (a version string,
a probe result, a digest), so only a real run can hold that every `NonNull`
field is non-null and every `NullMeans` field is at least present.

REQUIRES the torch venv `torch_venv.py` resolves. This suite is in the
`torch-host` lane (its `# lane:` line): nothing installs the venv, so the CI
image's run does not select this suite, and where it is selected a missing
venv fails naming it.

Run: `python3 ci/scripts/run_script_tests.py --lane torch-host`
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch_venv  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from check_cuda_run_artifacts import build_identity_tuples  # noqa: E402

REFERENCE_DIR = torch_venv.REPO_ROOT / "crates" / "jammi-bench" / "reference"

# The report's block the twin files the payload's identity in, beside the
# blocks a value may also stand in.
REPORT_BLOCKS = ("provenance", "args", "finetune_step")


def step_identity_fields():
    """The `train-step` identity as the Rust side declares it: the field
    names in order, and the names whose `null` is a value."""
    fields = [f for f in build_identity_tuples()[("finetune_step", "torch")]["fields"] if f[1] == "tier"]
    return tuple(f[0] for f in fields), frozenset(f[0] for f in fields if f[2] == "NullMeans")


def block_statuses(dump: dict, field: str) -> dict[str, str]:
    """Where `field` stands in each report block: `absent`, `null` or
    `present`. Every block is read, so a null in one block never masks a
    value in another, and "absent everywhere" stays distinct from "null in
    the one block that has it"."""

    def status(block) -> str:
        if not isinstance(block, dict) or field not in block:
            return "absent"
        return "null" if block[field] is None else "present"

    return {name: status(dump.get(name)) for name in REPORT_BLOCKS}


def satisfied(field: str, statuses: dict[str, str], null_is_a_value: frozenset[str]) -> bool:
    """A `NonNull` field needs a value in some block; a `NullMeans` field
    needs only presence, null being its declared state."""
    wanted = {"present", "null"} if field in null_is_a_value else {"present"}
    return bool(wanted & set(statuses.values()))


class TrainStepIdentityInADryRunReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if why := torch_venv.missing():
            raise AssertionError(f"{why} (`torch-venv` in ci/needs.toml)")
        # The producer writes its report to standard output.
        cls.dump = json.loads(
            torch_venv.run(
                REFERENCE_DIR / "torch_finetune_step.py",
                "--dry-run",
                "--batch", "2",
                "--seq", "6",
                "--steps", "1",
                "--warmup", "0",
                timeout=600,
            )
        )

    def test_every_declared_field_is_satisfied(self):
        names, null_is_a_value = step_identity_fields()
        self.assertTrue(names, "the train-step identity declares fields")
        statuses = {f: block_statuses(self.dump, f) for f in names}
        unsatisfied = {f: s for f, s in statuses.items() if not satisfied(f, s, null_is_a_value)}
        self.assertFalse(
            unsatisfied,
            f"train-step identity fields unsatisfied in a real --dry-run report, per block "
            f"({'/'.join(REPORT_BLOCKS)}): {unsatisfied}",
        )


if __name__ == "__main__":
    unittest.main()
