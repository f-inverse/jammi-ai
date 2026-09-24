#!/usr/bin/env python3
"""The ladder twins' run defaults, held to the Rust declarations they mirror.

A twin's default run is one its ladder can judge: as many timed iterations
as the judge settles (`SpeedInstrument::MIN_RUN`), as many repeats as it
measures a rung against itself with (`SpeedInstrument::MIN_REPEATS`), and —
for a seeded workload — as many seeds as its learning rule is stated for
(`SEEDED_LOSS_SEEDS`). The engine producers read those constants; a twin
cannot, so its literals are read here against the same declarations. Reads
source text only, so it runs anywhere Python does.

Run: `python3 crates/jammi-bench/reference/test_ladder_twin_defaults.py`
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REFERENCE = Path(__file__).resolve().parent
DEFINITION = REFERENCE.parent / "src" / "ladder" / "definition.rs"


def declared(name: str) -> int:
    """A `usize` constant of the ladder's definition: a literal, or a literal
    times another constant declared there."""
    text = DEFINITION.read_text()
    found = re.search(rf"pub const {name}: usize = ([^;]+);", text)
    assert found, f"{DEFINITION} no longer declares {name}"
    expression = found.group(1).strip()
    if expression.isdigit():
        return int(expression)
    product = re.fullmatch(r"(\d+) \* Self::(\w+)", expression)
    assert product, f"{name} = {expression} is neither a literal nor a literal times a constant"
    return int(product.group(1)) * declared(product.group(2))


def default_of(twin: str, flag: str):
    text = (REFERENCE / twin).read_text()
    found = re.search(rf'parser\.add_argument\("{re.escape(flag)}"[^)]*?default=([^,)]+)', text)
    return None if found is None else found.group(1).strip()


class TwinDefaultsTest(unittest.TestCase):
    TIMED = {
        "torch_graph_sample.py": "--iterations",
        "torch_propagate.py": "--iterations",
        "torch_structure.py": "--iterations",
        "torch_encode.py": "--iters",
    }
    REPEATED = [*TIMED, "torch_context_predictor.py"]

    def test_a_twins_default_run_is_as_long_as_the_judge_settles(self):
        for twin, flag in self.TIMED.items():
            with self.subTest(twin=twin):
                self.assertEqual(int(default_of(twin, flag)), declared("MIN_RUN"))

    def test_a_twins_default_repeats_are_the_ones_a_noise_band_needs(self):
        text = (REFERENCE / "ladder_leg.py").read_text()
        found = re.search(r"^MIN_REPEATS = (\d+)$", text, re.MULTILINE)
        self.assertIsNotNone(found, "ladder_leg.py no longer declares MIN_REPEATS")
        self.assertEqual(int(found.group(1)), declared("MIN_REPEATS"))
        for twin in self.REPEATED:
            with self.subTest(twin=twin):
                text = (REFERENCE / twin).read_text()
                self.assertIn("ll.add_take_argument(parser)", text)
                self.assertNotRegex(text, r'add_argument\("--takes?"')

    def test_the_seeded_twins_default_seeds_are_the_count_the_rule_is_stated_for(self):
        text = (REFERENCE / "torch_context_predictor.py").read_text()
        found = re.search(r"^DEFAULT_SEEDS = list\(range\(1, (\d+)\)\)$", text, re.MULTILINE)
        self.assertIsNotNone(found, "the predictor twin no longer declares DEFAULT_SEEDS")
        self.assertEqual(int(found.group(1)) - 1, declared("SEEDED_LOSS_SEEDS"))


if __name__ == "__main__":
    unittest.main()
