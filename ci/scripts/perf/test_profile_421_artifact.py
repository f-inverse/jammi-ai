#!/usr/bin/env python3
"""Hermetic tests for `profile_421_artifact.py` (issue #421 close-out
artifact producer; contract `scratchpad/contract-421-profile.md` v2.5
`## Artifacts`).

Everything here is SYNTHETIC merge/attribution/identity/legs-dir JSON built
in a tempdir — no GPU, no pod, no `nsys`, no network, and (deliberately)
none of the real pod421 data: this suite exists to prove the MECHANISM
(cross-checks, arithmetic, refusals) generalizes, not to re-assert one
run's numbers. `SHA` below is a syntactically valid 40-hex git sha chosen
for the fixture; it need not (and, being invented, does not) resolve
against this checkout's own history — `profile_421_artifact.py` never
calls `git`, only `check_cuda_run_artifacts.py`'s ancestry rule does that,
against the REAL committed artifact, not this suite's fixtures.

Six legs (`{clip-text,clip-vision,htsat}-{A1,A2}`) are the minimum that
exercises every finding `compute_findings()` builds, since each finding
names specific leg ids. Findings are checked against an INDEPENDENT
arithmetic oracle computed in this file (never a value copied out of the
module under test), per the house "numpy-first oracle" convention scaled
down to plain-float arithmetic (the deltas here are simple enough that a
second float computation IS the independent check;
`profile_421_merge.py`'s own `test_profile_421_merge.py` is the numpy-first
precedent for a case where the arithmetic is less trivial).

Run: `python3 ci/scripts/perf/test_profile_421_artifact.py`
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PERF_DIR = Path(__file__).resolve().parent
ARTIFACT = PERF_DIR / "profile_421_artifact.py"

sys.path.insert(0, str(PERF_DIR))
import profile_421_artifact as art  # noqa: E402

SHA = "a" * 40
BOX = "testbox0001"

TOWER_FAMILY = {"clip-text": "clip", "clip-vision": "clip", "htsat": "clap"}
CLIP_SHA = "c" * 64
CLAP_SHA = "d" * 64
FAMILY_SHA = {"clip": CLIP_SHA, "clap": CLAP_SHA}

# One leg per (tower, arm); front_share_of_wall/busy/wall chosen to give
# each finding a non-trivial, easily-hand-checked range.
LEG_SPECS = {
    "clip-text-A1": {"tower": "clip-text", "dtype": "f32", "busy": 0.060, "wall": 0.100, "front_share": 0.0, "launches": 3673.0},
    "clip-text-A2": {"tower": "clip-text", "dtype": "bf16", "busy": 0.041, "wall": 0.096, "front_share": 0.0, "launches": 3722.0},
    "clip-vision-A1": {"tower": "clip-vision", "dtype": "f32", "busy": 0.065, "wall": 0.115, "front_share": 0.205, "launches": 3638.0},
    "clip-vision-A2": {"tower": "clip-vision", "dtype": "bf16", "busy": 0.039, "wall": 0.109, "front_share": 0.222, "launches": 3696.0},
    "htsat-A1": {"tower": "htsat", "dtype": "f32", "busy": 0.235, "wall": 1.550, "front_share": 0.807, "launches": 6913.0},
    "htsat-A2": {"tower": "htsat", "dtype": "bf16", "busy": 0.188, "wall": 1.500, "front_share": 0.834, "launches": 6950.0},
}
C_ATTN_HTSAT_BUSY = 0.3305
C_ATTN_HTSAT_WALL = 0.0501


def _write_legs_dir(root: Path, specs: dict = LEG_SPECS) -> Path:
    legs_dir = root / "legs"
    legs_dir.mkdir()
    for leg_id, spec in specs.items():
        leg_dir = legs_dir / leg_id
        leg_dir.mkdir()
        manifest = {
            "leg_id": leg_id,
            "git_sha": SHA,
            "box": BOX,
            "tower": spec["tower"],
            "dtype": spec["dtype"],
            "status": "ok",
        }
        (leg_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (leg_dir / "census.json").write_text(json.dumps({"launches_per_step": spec["launches"]}), encoding="utf-8")
    return legs_dir


def _merge_leg_row(leg_id: str, spec: dict) -> dict:
    return {
        "leg_id": leg_id,
        "tower": spec["tower"],
        "task": "text_embedding",
        "dtype": spec["dtype"],
        "arm": "A",
        "kernels_disabled": [],
        "verdict": "VALID",
        "reasons": [],
        "positive_proof": {"run_n": {}, "run_m": {}},
        "per_step": {
            "busy_s_per_step": spec["busy"],
            "wall_s_per_step": spec["wall"],
            "front_share_of_wall": spec["front_share"],
            "residual_s_per_step": spec["wall"] - spec["busy"],
        },
        "checkpoint_weights_sha256": FAMILY_SHA[TOWER_FAMILY[spec["tower"]]],
        "fusible_site_census": {"lora_sites_wrapped": 4, "layer_norms": 4, "gelu_seam_calls_per_forward": 0},
    }


def _attr_leg_row(leg_id: str, spec: dict) -> dict:
    chains = {"UNATTRIBUTED": {"status": "measured", "share_gpu_busy": 0.02, "share_wall": 0.01}}
    if spec["tower"] == "htsat":
        chains["C-ATTN-htsat"] = {"status": "measured", "share_gpu_busy": C_ATTN_HTSAT_BUSY, "share_wall": C_ATTN_HTSAT_WALL}
    return {
        "leg_id": leg_id,
        "tower": spec["tower"],
        "dtype": spec["dtype"],
        "verdict": "VALID",
        "reasons": [],
        "signatures": {},
        "chains": chains,
        "unknown_kernels": [],
        "decision_grade": True,
        "decision_grade_reason": None,
        "outside_signature_plausibly_attention": {"busy_us": 0.0, "share_wall": 0.0},
    }


def _write_fixture(root: Path, *, leg_specs: dict = LEG_SPECS, merge_override=None, attr_override=None) -> dict:
    legs_dir = _write_legs_dir(root, leg_specs)
    merge_report = {
        "tool": "profile_421_merge",
        "schema": 1,
        "legs": [_merge_leg_row(leg_id, spec) for leg_id, spec in leg_specs.items()],
        "p2_bf16": [{"tower": "clip-text", "verdict": "PASS", "reasons": []}],
        "summary": {"legs_total": len(leg_specs), "legs_valid": len(leg_specs), "legs_invalid": 0},
    }
    attribution_report = {
        "tool": "profile_421_attribute",
        "schema": 3,
        "legs": [_attr_leg_row(leg_id, spec) for leg_id, spec in leg_specs.items()],
        "candidate_decisions": [{"port": "C-ATTN-clip-text", "tower": "clip-text", "chain": "C-ATTN-clip-text", "verdict": "UNRESOLVED", "reason": "synthetic fixture reason string"}],
        "realized_gains": [{"chain": "C-LORA", "tower": "clip-text", "share_of_baseline_wall": 0.3}],
        "summary": {"legs_total": len(leg_specs), "legs_valid": len(leg_specs), "legs_invalid": 0, "legs_decision_grade": len(leg_specs)},
    }
    if merge_override:
        merge_override(merge_report)
    if attr_override:
        attr_override(attribution_report)
    identity = {
        "what": "synthetic fixture",
        "gpu": "FIXTURE-GPU",
        "driver": "0.0.0",
        "cpu": "FIXTURE-CPU",
        "nsys_human": "fixture nsys",
        "checkpoints": {
            "clip": {"repo": "fixture/clip-repo", "files": ["a.json"]},
            "clap": {"repo": "fixture/clap-repo", "files": ["b.json"]},
        },
        "recorded_deviations": ["fixture deviation"],
        "producer_invocation": "fixture invocation",
        "status": "GREEN",
    }
    merge_path = root / "merge.json"
    attr_path = root / "attr.json"
    identity_path = root / "identity.json"
    merge_path.write_text(json.dumps(merge_report), encoding="utf-8")
    attr_path.write_text(json.dumps(attribution_report), encoding="utf-8")
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    return {
        "legs_dir": legs_dir,
        "merge_path": merge_path,
        "attr_path": attr_path,
        "identity_path": identity_path,
        "merge_report": merge_report,
        "attribution_report": attribution_report,
        "identity": identity,
    }


def run_artifact_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ARTIFACT), *args], capture_output=True, text=True, timeout=120)


class BuildReportHappyPathTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.fixture = _write_fixture(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self):
        return art.build_report(
            self.fixture["legs_dir"],
            None,
            self.fixture["merge_report"],
            self.fixture["attribution_report"],
            self.fixture["identity"],
        )

    def test_top_level_shape(self):
        report = self._build()
        expected_keys = {
            "schema_version", "git_sha", "box", "producer", "status", "notes",
            "legs", "p2", "attribution", "realized_gains", "candidate_decisions", "findings",
        }
        self.assertEqual(set(report.keys()), expected_keys)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["git_sha"], SHA)
        self.assertEqual(report["box"], BOX)
        self.assertEqual(report["producer"]["path"], "ci/scripts/perf/profile_421_legs.sh")
        self.assertEqual(report["producer"]["kind"], "script")
        self.assertEqual(report["producer"]["gating"], "none")

    def test_legs_combine_merge_and_attribution(self):
        report = self._build()
        self.assertEqual(len(report["legs"]), 6)
        by_id = {row["leg_id"]: row for row in report["legs"]}
        row = by_id["clip-text-A1"]
        self.assertEqual(row["merge_verdict"], "VALID")
        self.assertEqual(row["attribution_verdict"], "VALID")
        self.assertEqual(row["per_step"]["busy_s_per_step"], 0.060)
        self.assertEqual(row["launches_per_step"], 3673.0)
        self.assertIn("UNATTRIBUTED", row["chains"])

    def test_checkpoint_sha256_attached_per_family(self):
        report = self._build()
        self.assertEqual(report["notes"]["checkpoints"]["clip"]["sha256"], CLIP_SHA)
        self.assertEqual(report["notes"]["checkpoints"]["clap"]["sha256"], CLAP_SHA)
        # Static, non-numeric identity is passed through verbatim from --identity.
        self.assertEqual(report["notes"]["checkpoints"]["clip"]["repo"], "fixture/clip-repo")

    def test_p2_and_realized_gains_and_candidate_decisions_are_verbatim(self):
        report = self._build()
        self.assertEqual(report["p2"], self.fixture["merge_report"]["p2_bf16"])
        self.assertEqual(report["realized_gains"], self.fixture["attribution_report"]["realized_gains"])
        self.assertEqual(report["candidate_decisions"], self.fixture["attribution_report"]["candidate_decisions"])
        # Byte-identical, not just equal-looking: the verdict-string object is
        # copied by value from the SAME source list this test built.
        self.assertIs(
            report["candidate_decisions"][0]["verdict"],
            self.fixture["attribution_report"]["candidate_decisions"][0]["verdict"],
        )

    def test_attribution_section_carries_every_leg(self):
        report = self._build()
        self.assertEqual(len(report["attribution"]), 6)


class FindingsIndependentOracleTests(unittest.TestCase):
    """Every finding's `evidence` (and the rounded numbers baked into its
    `text`) is checked against arithmetic computed HERE, independently of
    `compute_findings()` — never against a value the module under test
    itself produced."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.fixture = _write_fixture(self.root)
        self.report = art.build_report(
            self.fixture["legs_dir"], None, self.fixture["merge_report"], self.fixture["attribution_report"], self.fixture["identity"]
        )
        self.findings = {f["id"]: f for f in self.report["findings"]}

    def tearDown(self):
        self._tmp.cleanup()

    def test_htsat_front_end_share_oracle(self):
        oracle = {"htsat-A1": LEG_SPECS["htsat-A1"]["front_share"] * 100.0, "htsat-A2": LEG_SPECS["htsat-A2"]["front_share"] * 100.0}
        finding = self.findings["htsat-front-end-bound"]
        for leg_id, expected in oracle.items():
            self.assertAlmostEqual(finding["evidence"]["front_share_of_wall_pct"][leg_id], expected, places=9)
        lo, hi = min(oracle.values()), max(oracle.values())
        self.assertIn(f"{lo:.0f}-{hi:.0f}%", finding["text"])
        self.assertEqual((round(lo), round(hi)), (81, 83))

    def test_clip_vision_front_end_share_oracle(self):
        oracle = {
            "clip-vision-A1": LEG_SPECS["clip-vision-A1"]["front_share"] * 100.0,
            "clip-vision-A2": LEG_SPECS["clip-vision-A2"]["front_share"] * 100.0,
        }
        finding = self.findings["clip-vision-front-end-share"]
        for leg_id, expected in oracle.items():
            self.assertAlmostEqual(finding["evidence"]["front_share_of_wall_pct"][leg_id], expected, places=9)
        self.assertEqual((round(min(oracle.values())), round(max(oracle.values()))), (20, 22))

    def test_clip_launch_bound_and_bf16_deltas_oracle(self):
        finding = self.findings["clip-launch-bound-batch8"]
        expected_launches = {leg_id: LEG_SPECS[leg_id]["launches"] for leg_id in ("clip-text-A1", "clip-text-A2", "clip-vision-A1", "clip-vision-A2")}
        self.assertEqual(finding["evidence"]["launches_per_step"], expected_launches)

        def pct(a, b):
            return (b - a) / a * 100.0

        expected_busy = {
            "clip-text": pct(LEG_SPECS["clip-text-A1"]["busy"], LEG_SPECS["clip-text-A2"]["busy"]),
            "clip-vision": pct(LEG_SPECS["clip-vision-A1"]["busy"], LEG_SPECS["clip-vision-A2"]["busy"]),
        }
        expected_wall = {
            "clip-text": pct(LEG_SPECS["clip-text-A1"]["wall"], LEG_SPECS["clip-text-A2"]["wall"]),
            "clip-vision": pct(LEG_SPECS["clip-vision-A1"]["wall"], LEG_SPECS["clip-vision-A2"]["wall"]),
        }
        for tower, expected in expected_busy.items():
            self.assertAlmostEqual(finding["evidence"]["busy_delta_pct_bf16_vs_f32"][tower], expected, places=9)
        for tower, expected in expected_wall.items():
            self.assertAlmostEqual(finding["evidence"]["wall_delta_pct_bf16_vs_f32"][tower], expected, places=9)
        # Both deltas are negative here (BF16 cheaper): the sentence orders
        # the range by ascending MAGNITUDE, never a plain min-then-max pick.
        by_mag = sorted(expected_busy.values(), key=abs)
        self.assertIn(f"{by_mag[0]:.0f}...{by_mag[-1]:.0f}%", finding["text"])

    def test_c_attn_htsat_out_of_tier_oracle(self):
        finding = self.findings["c-attn-htsat-out-of-tier"]
        self.assertAlmostEqual(finding["evidence"]["share_gpu_busy"], C_ATTN_HTSAT_BUSY, places=9)
        self.assertAlmostEqual(finding["evidence"]["share_wall"], C_ATTN_HTSAT_WALL, places=9)
        self.assertIn(f"{C_ATTN_HTSAT_BUSY * 100.0:.0f}%", finding["text"])
        self.assertIn("out-of-tier".replace("-", ""), finding["id"].replace("-", ""))
        self.assertIn("OUTSIDE", finding["text"])
        self.assertIn("never decided under this issue", finding["text"])


class RefusalTests(unittest.TestCase):
    """Every way the input can be inconsistent or non-finite must refuse —
    never silently pick a value or propagate a NaN into the artifact."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self, fixture):
        return art.build_report(fixture["legs_dir"], None, fixture["merge_report"], fixture["attribution_report"], fixture["identity"])

    def test_refuses_git_sha_mismatch_across_manifests(self):
        fixture = _write_fixture(self.root)
        bad_manifest_path = fixture["legs_dir"] / "clip-text-A1" / "manifest.json"
        manifest = json.loads(bad_manifest_path.read_text())
        manifest["git_sha"] = "b" * 40
        bad_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("disagree", str(ctx.exception))

    def test_refuses_box_mismatch_across_manifests(self):
        fixture = _write_fixture(self.root)
        bad_manifest_path = fixture["legs_dir"] / "htsat-A2" / "manifest.json"
        manifest = json.loads(bad_manifest_path.read_text())
        manifest["box"] = "some-other-box"
        bad_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("disagree", str(ctx.exception))

    def test_refuses_checkpoint_sha256_disagreement_within_family(self):
        def corrupt(merge_report):
            merge_report["legs"][1]["checkpoint_weights_sha256"] = "e" * 64  # clip-text-A2, same family as A1

        fixture = _write_fixture(self.root, merge_override=corrupt)
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("checkpoint family", str(ctx.exception))

    def test_refuses_merge_leg_set_not_matching_legs_dir(self):
        def drop_a_leg(merge_report):
            merge_report["legs"] = [row for row in merge_report["legs"] if row["leg_id"] != "htsat-A2"]

        fixture = _write_fixture(self.root, merge_override=drop_a_leg)
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("--merge-json's leg set does not match", str(ctx.exception))

    def test_refuses_attribution_leg_set_not_matching_legs_dir(self):
        def drop_a_leg(attribution_report):
            attribution_report["legs"] = [row for row in attribution_report["legs"] if row["leg_id"] != "clip-vision-A1"]

        fixture = _write_fixture(self.root, attr_override=drop_a_leg)
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("--attribution-json's leg set does not match", str(ctx.exception))

    def test_refuses_nonfinite_per_step_value_used_by_a_finding(self):
        # NaN > c and NaN < c are both False — a naive min/max or threshold
        # check would silently let this through and poison the finding's
        # rounded output. This must be a hard refusal, not a quiet NaN.
        def poison(merge_report):
            for row in merge_report["legs"]:
                if row["leg_id"] == "htsat-A1":
                    row["per_step"]["front_share_of_wall"] = math.nan

        fixture = _write_fixture(self.root, merge_override=poison)
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("not finite", str(ctx.exception))

    def test_refuses_nonfinite_launches_per_step(self):
        fixture = _write_fixture(self.root)
        census_path = fixture["legs_dir"] / "clip-text-A2" / "census.json"
        census_path.write_text(json.dumps({"launches_per_step": math.inf}), encoding="utf-8")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("not finite", str(ctx.exception))

    def test_refuses_nonfinite_chain_share(self):
        def poison(attribution_report):
            for row in attribution_report["legs"]:
                if row["leg_id"] == "htsat-A1":
                    row["chains"]["C-ATTN-htsat"]["share_gpu_busy"] = math.nan

        fixture = _write_fixture(self.root, attr_override=poison)
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("not finite", str(ctx.exception))

    def test_refuses_missing_manifest_directory(self):
        with self.assertRaises(art.ArtifactBuildError):
            art.collect_identity(self.root / "does-not-exist", None)


class CliEndToEndTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.fixture = _write_fixture(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_cli_writes_a_schema_shaped_artifact(self):
        out_path = self.root / "artifact.json"
        result = run_artifact_cli(
            "--legs-dir", str(self.fixture["legs_dir"]),
            "--merge-json", str(self.fixture["merge_path"]),
            "--attribution-json", str(self.fixture["attr_path"]),
            "--identity", str(self.fixture["identity_path"]),
            "--out", str(out_path),
        )
        self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")
        report = json.loads(out_path.read_text())
        self.assertEqual(report["git_sha"], SHA)
        self.assertEqual(report["box"], BOX)
        self.assertEqual(len(report["legs"]), 6)
        self.assertEqual(len(report["findings"]), 4)

    def test_cli_refuses_and_exits_nonzero_on_bad_input(self):
        bad_manifest_path = self.fixture["legs_dir"] / "clip-text-A1" / "manifest.json"
        manifest = json.loads(bad_manifest_path.read_text())
        manifest["git_sha"] = "b" * 40
        bad_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result = run_artifact_cli(
            "--legs-dir", str(self.fixture["legs_dir"]),
            "--merge-json", str(self.fixture["merge_path"]),
            "--attribution-json", str(self.fixture["attr_path"]),
            "--identity", str(self.fixture["identity_path"]),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("disagree", result.stderr)


if __name__ == "__main__":
    unittest.main()
