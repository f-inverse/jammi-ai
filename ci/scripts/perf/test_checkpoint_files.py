#!/usr/bin/env python3
"""`checkpoint_files.py` names what a checkpoint directory lacks.

Run: `python3 ci/scripts/perf/test_checkpoint_files.py`
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkpoint_files  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]


def write_checkpoint(directory: Path, payload: bytes = b"\x00" * 8) -> None:
    header = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)
    (directory / "config.json").write_text("{}")
    (directory / "tokenizer.json").write_text("{}")


class CheckpointFiles(unittest.TestCase):
    def test_a_committed_fixture_is_a_whole_checkpoint(self):
        fixture = REPO_ROOT / "cookbook" / "fixtures" / "tiny_bert_head64"
        self.assertEqual(checkpoint_files.defects(fixture), [])

    def test_weights_without_a_tokenizer_are_refused_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_checkpoint(Path(tmp))
            (Path(tmp) / "tokenizer.json").unlink()
            self.assertEqual(checkpoint_files.defects(Path(tmp)), ["tokenizer.json: missing"])

    def test_a_truncated_weights_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_checkpoint(Path(tmp), payload=b"\x00" * 3)
            (found,) = checkpoint_files.defects(Path(tmp))
            self.assertIn("interrupted download", found)

    def test_an_empty_file_and_a_non_json_config_are_each_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_checkpoint(Path(tmp))
            (Path(tmp) / "tokenizer.json").write_text("")
            (Path(tmp) / "config.json").write_text("<html>")
            found = checkpoint_files.defects(Path(tmp))
            self.assertEqual(len(found), 2)
            self.assertTrue(found[0].startswith("config.json: not JSON"))
            self.assertEqual(found[1], "tokenizer.json: empty")



class Fetch(unittest.TestCase):
    def test_a_directory_not_yet_whole_is_downloaded_until_it_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "ckpt"
            calls = []

            def download(repo, directory):
                calls.append(repo)
                if len(calls) == 2:
                    write_checkpoint(directory)
                else:
                    (directory / "config.json").write_text("{}")

            self.assertEqual(checkpoint_files.fetch("org/model", target, download), [])
            self.assertEqual(calls, ["org/model", "org/model"], "retried once, then whole")

    def test_a_whole_directory_is_never_downloaded_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_checkpoint(Path(tmp))
            self.assertEqual(checkpoint_files.fetch("org/model", Path(tmp), lambda *_: self.fail("downloaded")), [])

    def test_every_attempt_failing_names_the_last_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            def download(repo, directory):
                raise OSError("network unreachable")

            found = checkpoint_files.fetch("org/model", Path(tmp) / "ckpt", download)
            self.assertEqual(found, ["download of org/model failed: network unreachable"])


if __name__ == "__main__":
    unittest.main()
