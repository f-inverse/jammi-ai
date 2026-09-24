#!/usr/bin/env python3
"""`cpu_ladders_ab.sh` under its dry-run flag: each workload's legs run in a
palindrome per unit, each leg one take of one point, the twin's inputs named
by the engine's first leg, and the ladder spans torch to the top rung run.
Its defaults are held to the ladder's own declarations. Nothing is built or
measured.

Run: `python3 ci/scripts/perf/test_cpu_ladders_ab_sh_dry_run.py`
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PERF_DIR = Path(__file__).resolve().parent
SCRIPT = PERF_DIR / "cpu_ladders_ab.sh"
REFERENCE = PERF_DIR.parents[2] / "crates" / "jammi-bench" / "reference"
DEFINITION = REFERENCE.parent / "src" / "ladder" / "definition.rs"
sys.path.insert(0, str(REFERENCE))

from test_ladder_twin_defaults import declared  # noqa: E402

TWIN_RUNGS = {"torch", "torch-geometric"}


def run(**env: str) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as out:
        return subprocess.run(
            ["bash", str(SCRIPT)],
            capture_output=True,
            text=True,
            env={**os.environ, "CPU_AB_DRY_RUN": "1", "CPU_AB_OUT_DIR": out, **env},
            timeout=60,
        )


def legs(**env: str) -> list[tuple[str, str]]:
    """The `(label, command)` of every leg, in run order."""
    done = run(**env)
    assert done.returncode == 0, done.stderr
    return re.findall(r"^--- (\S+): (.*)$", done.stdout, flags=re.M)


def ladder_rungs(workload: str) -> list[str]:
    """The engine rungs `definition.rs` declares for `workload`, in order."""
    text = DEFINITION.read_text()
    body = text[text.index(f"fn {workload.replace('-', '_')}_ladder(") :]
    body = body[: body.index("\n}\n")]
    return [r for r in re.findall(r'Rung::new\("([^"]+)"', body) if r not in TWIN_RUNGS]


class CpuLaddersDryRun(unittest.TestCase):
    def test_a_units_legs_are_a_palindrome_over_the_rungs(self):
        labels = [label for label, _ in legs(CPU_AB_WORKLOAD="propagate", CPU_AB_UNITS="2048")]
        self.assertEqual(
            labels,
            [
                "plan__2048__r1",
                "plan-partitioned__2048__r1",
                "torch__2048__r1",
                "torch-geometric__2048__r1",
                "torch-geometric__2048__r2",
                "torch__2048__r2",
                "plan-partitioned__2048__r2",
                "plan__2048__r2",
            ],
        )

    def test_every_leg_is_one_take_of_one_point_pinned_where_asked(self):
        for label, command in legs(CPU_AB_WORKLOAD="structure", CPU_AB_UNITS="2048,8192", CPU_AB_CPUS="0-7"):
            with self.subTest(leg=label):
                take = label.rsplit("__r", 1)[1]
                self.assertTrue(command.startswith("taskset -c 0-7 "), command)
                self.assertEqual(re.findall(r"--take (\S+)", command), [take])

    def test_the_twin_reads_the_unit_the_engine_filed(self):
        commands = dict(legs(CPU_AB_WORKLOAD="propagate", CPU_AB_UNITS="2048"))
        self.assertIn("--unit unit-of-2048 --impl exact", commands["torch__2048__r1"])
        self.assertIn("--unit unit-of-2048 --impl pyg", commands["torch-geometric__2048__r1"])

    def test_the_predictor_twin_trains_the_engine_legs_configuration(self):
        commands = dict(legs(CPU_AB_WORKLOAD="predictor-train-run", CPU_AB_UNITS="3"))
        self.assertIn("--seeds 3 --rung in-process", commands["in-process__3__r1"])
        self.assertIn("--seeds 3 --arch of-engine-leg --epochs of-engine-leg --learning-rate of-engine-leg", commands["torch__3__r1"])

    def test_graph_sample_samples_the_committed_shape_and_the_ladder_reads_its_law(self):
        done = run(CPU_AB_WORKLOAD="graph-sample", CPU_AB_UNITS="64")
        self.assertIn("graph-fixture --nodes-per 64 --out", done.stdout)
        self.assertRegex(done.stdout, r"ladder graph-sample \S+ --from torch --to sampler --out \S+ --law-dir \S+/legs/law")

    def test_the_planes_rungs_run_on_the_fleet_and_the_verdict_spans_them(self):
        done = run(CPU_AB_WORKLOAD="predictor-train-run", CPU_AB_UNITS="1", CPU_AB_RUNGS="in-process,placed,shape-d")
        commands = dict(re.findall(r"^--- (\S+): (.*)$", done.stdout, flags=re.M))
        self.assertIn("--server-bin", commands["placed__1__r1"])
        self.assertIn("--server-bin", commands["shape-d__1__r2"])
        self.assertNotIn("--server-bin", commands["in-process__1__r1"])
        self.assertRegex(done.stdout, r"ladder predictor-train-run \S+ --from torch --to shape-d ")

    def test_the_engine_rungs_offered_are_the_ladders(self):
        for workload in ("propagate", "structure", "graph-sample", "predictor-train-run"):
            with self.subTest(workload=workload):
                done = run(CPU_AB_WORKLOAD=workload, CPU_AB_RUNGS="warp")
                self.assertEqual(done.returncode, 2)
                offered = re.search(r"engine rungs are (.*)\.$", done.stderr, flags=re.M).group(1).split()
                self.assertEqual(offered, ladder_rungs(workload))

    def test_the_predictors_default_seeds_are_the_count_its_rule_is_stated_for(self):
        seeds = {label.split("__")[1] for label, _ in legs(CPU_AB_WORKLOAD="predictor-train-run")}
        self.assertEqual(seeds, {str(s) for s in range(1, declared("SEEDED_LOSS_SEEDS") + 1)})

    def test_an_unknown_workload_is_refused(self):
        done = run(CPU_AB_WORKLOAD="encode")
        self.assertEqual(done.returncode, 2)
        self.assertIn("CPU_AB_WORKLOAD", done.stderr)


if __name__ == "__main__":
    unittest.main()
