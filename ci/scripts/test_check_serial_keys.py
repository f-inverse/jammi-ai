#!/usr/bin/env python3
"""Tests for `check_serial_keys.py`, over a scratch tree holding the real
nextest config and every real source file, plus the one file each case adds.

Run directly: `python3 ci/scripts/test_check_serial_keys.py`
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_serial_keys as csk  # noqa: E402


class SerialKeysTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        (self.root / ".config").mkdir()
        shutil.copy(csk.REPO_ROOT / csk.NEXTEST_CONFIG, self.root / csk.NEXTEST_CONFIG)
        for path in csk.sources(csk.REPO_ROOT):
            dest = self.root / path.relative_to(csk.REPO_ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, dest)

    def tearDown(self):
        self._dir.cleanup()

    def add(self, text: str, rel: str = "crates/demo/tests/demo.rs"):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def test_the_real_tree_is_classified(self):
        self.assertEqual(csk.check(self.root), [])

    def test_an_unclassified_key_is_a_finding(self):
        self.add("#[serial(fixed_port)]\n#[test]\nfn t() {}\n")
        findings = csk.check(self.root)
        self.assertTrue(any("`fixed_port`" in f for f in findings), findings)

    def test_a_key_that_names_a_test_group_is_classified(self):
        self.add("#[serial_test::serial(training_set_stream)]\n#[test]\nfn t() {}\n")
        self.assertEqual(csk.check(self.root), [])

    def test_a_keyless_attribute_is_a_finding(self):
        self.add("#[serial]\n#[test]\nfn t() {}\n")
        findings = csk.check(self.root)
        self.assertTrue(any("keyless" in f for f in findings), findings)

    def test_a_commented_attribute_is_not_a_key(self):
        self.add("// #[serial(fixed_port)] is how it once read\n")
        self.assertEqual(csk.check(self.root), [])

    def test_an_unclassified_lock_is_a_finding(self):
        self.add("static PORT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());\n")
        findings = csk.check(self.root)
        self.assertTrue(any("PORT_LOCK" in f for f in findings), findings)

    def test_a_lock_that_no_longer_exists_is_a_finding(self):
        (self.root / "crates/jammi-test-utils/src/child.rs").write_text("// gone\n")
        findings = csk.check(self.root)
        self.assertTrue(any("SPAWN_LOCK" in f and "does not exist" in f for f in findings), findings)


if __name__ == "__main__":
    unittest.main()
