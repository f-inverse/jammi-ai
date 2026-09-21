#!/usr/bin/env python3
"""The pure mirrors in `torch_finetune_run.py` — the pieces of jammi's trainer
and tier it re-implements in Python rather than calls — held to what jammi
computes.

Each is a rule whose silent drift would make a torch leg train on different
data, under a different partition, or under a different identity than the
jammi leg it is set beside: the validation split boundary, the bucket ladder,
JSONL line splitting, the two digests both producers emit, and the adapter
tensor names. None touches torch (the module imports it lazily, inside the
functions that need it), so this suite runs anywhere Python does.

Where a value is pinned, it is one jammi produced: the partition digest is
read off a committed `jammi-bench finetune-run` report, and the token-batch
digests are the same three values
`finetune_run.rs::tests::token_batches_sha256_*` pin on the Rust side.

Run directly: `python3 crates/jammi-bench/reference/test_torch_finetune_run_mirrors.py`
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch_finetune_run as tfr  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "ci", "scripts", "perf"))
import identity_fields  # noqa: E402

GOLDEN = os.path.join(REPO_ROOT, "ci", "scripts", "perf", "fixtures", "finetune_run_golden", "bert_fused.json")


class IdentityTupleTests(unittest.TestCase):
    def test_the_twin_declares_the_tuple_a_jammi_leg_is_compared_on(self):
        self.assertEqual(tfr.RUN_IDENTITY_FIELDS, identity_fields.FINETUNE_RUN_IDENTITY_FIELDS)


class SplitIndexTests(unittest.TestCase):
    def test_the_last_rounded_fraction_of_rows_is_validation(self):
        # The committed fixture's shape: 1372 rows at 0.1 hold out 137.
        self.assertEqual(tfr.split_index(1372, 0.1), 1235)
        self.assertEqual(tfr.split_index(96, 0.1), 86)
        self.assertEqual(tfr.split_index(4, 0.1), 4)
        self.assertEqual(tfr.split_index(10, 0.0), 10)

    def test_a_half_rounds_away_from_zero_as_rust_does(self):
        # 5 * 0.1 == 0.5 and 25 * 0.1 == 2.5: Rust's `f64::round` gives 1 and
        # 3, Python's `round` (half to even) would give 0 and 2.
        self.assertEqual(tfr.split_index(5, 0.1), 4)
        self.assertEqual(tfr.split_index(25, 0.1), 22)


class BucketLadderTests(unittest.TestCase):
    def test_widths_round_up_a_doubling_ladder_floored_at_eight(self):
        ladder = {1: 8, 8: 8, 9: 16, 16: 16, 17: 32, 33: 64, 64: 64}
        for natural, bucket in ladder.items():
            self.assertEqual(tfr.bucket_seq_len(natural, 64), bucket, natural)

    def test_the_ladder_is_capped_at_the_effective_max_length(self):
        self.assertEqual(tfr.bucket_seq_len(40, 48), 48)
        self.assertEqual(tfr.bucket_seq_len(3, 4), 4)

    def test_a_zero_width_passes_through(self):
        self.assertEqual(tfr.bucket_seq_len(0, 64), 0)
        self.assertEqual(tfr.bucket_seq_len(5, 0), 5)


class RustLinesTests(unittest.TestCase):
    def test_only_a_line_feed_ends_a_line(self):
        # U+2028 and a form feed are legal inside a JSON string; Python's
        # `splitlines` would cut the row there and Rust's `lines` does not.
        text = 'a b\nc\x0cd\r\ne'
        self.assertEqual(tfr.rust_lines(text), ["a b", "c\x0cd", "e"])

    def test_a_trailing_newline_adds_no_empty_line(self):
        self.assertEqual(tfr.rust_lines("a\nb\n"), ["a", "b"])
        self.assertEqual(tfr.rust_lines(""), [])


class DigestTests(unittest.TestCase):
    def test_the_partition_digest_is_the_one_a_jammi_leg_reports(self):
        with open(GOLDEN) as fh:
            tier = json.load(fh)["tiers"]["finetune_run"]
        # The golden's held-out fixture is `finetune_run_smoke.rs`'s: two rows,
        # ids `a100`/`a101`, one batch at `--batch 2`.
        self.assertEqual(tier["batch"], 2)
        self.assertEqual(tier["held_out_count"], 2)
        self.assertEqual(tfr.partition_sha256([["a100", "a101"]]), tier["heldout_batch_partition_sha256"])

    def test_the_token_batch_digest_is_the_documented_layout(self):
        padded = ([[2, 7, 3], [2, 9, 0]], [[1, 1, 1], [1, 1, 0]])
        single = ([[2, 5, 6, 3]], [[1, 1, 1, 1]])
        self.assertEqual(
            tfr.token_batches_sha256([padded, single]),
            "365019a4cf7bd591cb4801106620245616e8549e382bcaf1cdc5091d75681534",
        )
        self.assertEqual(
            tfr.token_batches_sha256([]),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )

    def test_the_same_ids_under_another_partition_are_another_stream(self):
        merged = ([[2, 7, 3, 2, 9, 0]], [[1, 1, 1, 1, 1, 0]])
        single = ([[2, 5, 6, 3]], [[1, 1, 1, 1]])
        self.assertEqual(
            tfr.token_batches_sha256([merged, single]),
            "b3d6b6770ac17ef7b9f80bc1503901c991eeed8692f10244fa48dc040316dc47",
        )


class AdapterTensorNameTests(unittest.TestCase):
    def test_modernbert_names_are_the_gradient_oracles(self):
        name = "base_model.model.layers.3.mlp.Wo.lora_B.default.weight"
        self.assertEqual(tfr.jammi_tensor_name("modernbert", name), "layer.3.mlp.Wo.lora_b")

    def test_bert_names_follow_the_wrapped_module_path(self):
        cases = {
            "attention.self.query": "query",
            "attention.self.key": "key",
            "attention.self.value": "value",
            "attention.output.dense": "dense",
            "intermediate.dense": "intermediate_dense",
            "output.dense": "output_dense",
        }
        for path, site in cases.items():
            for peft_leaf, jammi_leaf in (("lora_A", "lora_a"), ("lora_B", "lora_b")):
                peft_name = f"base_model.model.encoder.layer.11.{path}.{peft_leaf}.default.weight"
                self.assertEqual(
                    tfr.jammi_tensor_name("bert", peft_name), f"layer.11.{site}.{jammi_leaf}"
                )

    def test_a_linear_jammi_does_not_wrap_has_no_name(self):
        # PEFT's suffix match on `dense` also reaches BERT's pooler, which
        # jammi's encoder does not carry; the twin refuses on the `None`.
        self.assertIsNone(tfr.jammi_tensor_name("bert", "base_model.model.pooler.dense.lora_A.default.weight"))
        self.assertIsNone(tfr.jammi_tensor_name("distilbert", "base_model.model.transformer.layer.0.attention.q_lin.lora_A.default.weight"))


if __name__ == "__main__":
    unittest.main()
