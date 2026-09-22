#!/usr/bin/env python3
"""Every `torch_finetune_step.py::TORCH_IDENTITY_FIELDS` entry, read off the
report a real `--dry-run` writes.

Whether a field's VALUE is non-null is decided at run time (a version string, a probe result, a digest), so only a real run can
hold that every `NonNull` field is non-null and every
`TORCH_IDENTITY_FIELDS_NULL_MEANS` field is at least present.

REQUIRES the torch venv `torch_venv.py` resolves. It is the `torch-venv` need
of this suite's guard in `ci/guards.toml`, which is in the `torch-host` lane:
nothing installs it, so the CI image's lane does not select this suite, and
where it is selected a missing venv fails naming it.

Run: `python3 ci/scripts/run_guards.py --lane torch-host`
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch_venv  # noqa: E402

REFERENCE_DIR = torch_venv.REPO_ROOT / "crates" / "jammi-bench" / "reference"
sys.path.insert(0, str(REFERENCE_DIR))
import torch_finetune_step as tfs  # noqa: E402

REPORT_BLOCKS = ("provenance", "args", "finetune_step")


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


def satisfied(field: str, statuses: dict[str, str]) -> bool:
    """A `NonNull` field needs a value in some block; a `NULL_MEANS` field
    needs only presence, null being its declared state."""
    wanted = {"present", "null"} if field in tfs.TORCH_IDENTITY_FIELDS_NULL_MEANS else {"present"}
    return bool(wanted & set(statuses.values()))


class TorchIdentityFieldsInADryRunReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if why := torch_venv.missing():
            raise AssertionError(f"{why} (`torch-venv` in ci/guards.toml)")
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
        statuses = {f: block_statuses(self.dump, f) for f in tfs.TORCH_IDENTITY_FIELDS}
        unsatisfied = {f: s for f, s in statuses.items() if not satisfied(f, s)}
        self.assertFalse(
            unsatisfied,
            f"TORCH_IDENTITY_FIELDS entries unsatisfied in a real --dry-run report, per block "
            f"({'/'.join(REPORT_BLOCKS)}): {unsatisfied}",
        )


if __name__ == "__main__":
    unittest.main()
