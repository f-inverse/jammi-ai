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
precedent for a case where the arithmetic is less trivial). The media-
corpus pool constants (`MEDIA_FAMILIES`, `_DEFAULT_INSTANCES_PER_FAMILY`,
...) are re-derived here by a DIFFERENT parsing mechanism than the module
under test (`_independent_read_shell_int`/`_independent_read_py_int`
below), against the REAL committed `profile_421_legs.sh`/corpus-producer
scripts, so the two independently agree rather than one merely echoing
the other's regex.

Run: `python3 ci/scripts/perf/test_profile_421_artifact.py`
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PERF_DIR = Path(__file__).resolve().parent
ARTIFACT = PERF_DIR / "profile_421_artifact.py"
IDENTITY_SIDECAR = PERF_DIR / "profile_421_run2_identity.json"

sys.path.insert(0, str(PERF_DIR))
import profile_421_artifact as art  # noqa: E402
import profile_421_attribute as attribute_mod  # noqa: E402

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
UNATTRIBUTED_SHARE_GPU_BUSY = 0.02
UNATTRIBUTED_SHARE_WALL = 0.01

# `compute_kernel_identity_split_count`'s own fixture: TWO post-fix kernel
# names sharing one (grid, block) that a single pre-fix "Kernel2" bucket
# coalesced -- deliberately NOT 3 (the real committed identity sidecar's own
# number), so a test asserting this exact fixture value proves the module
# reads it LIVE rather than happening to agree with a hard-coded literal.
KERNEL_IDENTITY_SPLIT_COUNT = 2
_KERNEL2_GRID = [2, 1, 192]
_KERNEL2_BLOCK = [128, 1, 1]

IDENTITY_TEMPLATE_DEVIATIONS = [
    "fixture deviation, no placeholders.",
    (
        "htsat-A2 is {htsat_a2_merge_verdict} at the merge level but {htsat_a2_decision_grade_word} for "
        "attribution: UNATTRIBUTED share_gpu_busy is {htsat_a2_unattributed_share_gpu_busy_pct:.2f}%, "
        "over the {unattributed_decision_grade_limit_pct:.0f}% validity bound "
        "({unknown_kernel_share_limit_pct:.0f}% known-kernel-name gate)."
    ),
    "raw sqlite exports: {sqlite_raw_export_count} files.",
    "corpus: {corpus_total_files} files, {corpus_train_clips} distinct train clips, {corpus_rows_m} rows.",
    "kernel identity: {kernel_identity_split_count} distinct instantiations across {leg_count} legs.",
    "the live merge suite already carried {merge_suite_test_count} tests.",
]


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
        census: dict = {"launches_per_step": spec["launches"]}
        if leg_id == "clip-text-A2":
            # The one leg `compute_kernel_identity_split_count` reads by
            # name: `KERNEL_IDENTITY_SPLIT_COUNT` distinct post-fix kernel
            # names sharing (grid, block), collapsed pre-fix under a single
            # "Kernel2" bucket at the SAME (grid, block).
            census["by_kernel_and_grid"] = [
                {
                    "kernel": f"cutlass_fixture_{i}",
                    "grid": _KERNEL2_GRID,
                    "block": _KERNEL2_BLOCK,
                    "launches_per_step": 1.0,
                    "us_per_step": 1.0,
                    "share": 0.001,
                }
                for i in range(KERNEL_IDENTITY_SPLIT_COUNT)
            ]
            (leg_dir / "census.pre-demangle.json").write_text(
                json.dumps(
                    {
                        "by_kernel_and_grid": [
                            {
                                "kernel": "Kernel2",
                                "grid": _KERNEL2_GRID,
                                "block": _KERNEL2_BLOCK,
                                "launches_per_step": float(KERNEL_IDENTITY_SPLIT_COUNT),
                                "us_per_step": float(KERNEL_IDENTITY_SPLIT_COUNT),
                                "share": 0.001 * KERNEL_IDENTITY_SPLIT_COUNT,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
        (leg_dir / "census.json").write_text(json.dumps(census), encoding="utf-8")
    return legs_dir


def _write_p2_dir(root: Path, towers: tuple[str, ...], *, git_sha: str = SHA, box: str = BOX) -> Path:
    p2_dir = root / "p2-bf16"
    p2_dir.mkdir(exist_ok=True)
    for tower in towers:
        tower_dir = p2_dir / tower
        tower_dir.mkdir()
        manifest = {"tower": tower, "git_sha": git_sha, "box": box}
        (tower_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return p2_dir


def _merge_leg_row(leg_id: str, spec: dict) -> dict:
    front_s_per_step = spec.get("front_s_per_step", spec["front_share"] * spec["wall"])
    return {
        "leg_id": leg_id,
        "tower": spec["tower"],
        "task": "text_embedding",
        "dtype": spec["dtype"],
        "arm": spec.get("arm", "A"),
        "kernels_disabled": [],
        "verdict": "VALID",
        "reasons": [],
        "positive_proof": {"run_n": {}, "run_m": {}},
        "per_step": {
            "busy_s_per_step": spec["busy"],
            "wall_s_per_step": spec["wall"],
            "front_share_of_wall": spec["front_share"],
            "front_s_per_step": front_s_per_step,
            "residual_s_per_step": spec["wall"] - spec["busy"],
        },
        "checkpoint_weights_sha256": FAMILY_SHA[TOWER_FAMILY[spec["tower"]]],
        "fusible_site_census": {"lora_sites_wrapped": 4, "layer_norms": 4, "gelu_seam_calls_per_forward": 0},
    }


def _attr_leg_row(leg_id: str, spec: dict) -> dict:
    chains = {
        "UNATTRIBUTED": {
            "status": "measured",
            "share_gpu_busy": UNATTRIBUTED_SHARE_GPU_BUSY,
            "share_wall": UNATTRIBUTED_SHARE_WALL,
        }
    }
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


def _write_fixture(
    root: Path,
    *,
    leg_specs: dict = LEG_SPECS,
    merge_override=None,
    attr_override=None,
    identity_override=None,
    p2_towers: tuple[str, ...] = ("clip-text",),
) -> dict:
    legs_dir = _write_legs_dir(root, leg_specs)
    p2_dir = _write_p2_dir(root, p2_towers) if p2_towers else None
    merge_report = {
        "tool": "profile_421_merge",
        "schema": 1,
        "legs": [_merge_leg_row(leg_id, spec) for leg_id, spec in leg_specs.items()],
        "p2_bf16": [{"tower": tower, "verdict": "PASS", "reasons": []} for tower in p2_towers],
        "summary": {
            "legs_total": len(leg_specs),
            "legs_valid": len(leg_specs),
            "legs_invalid": 0,
            "p2_total": len(p2_towers),
            "p2_pass": len(p2_towers),
            "p2_fail": 0,
        },
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
        "recorded_deviations": list(IDENTITY_TEMPLATE_DEVIATIONS),
        "operator_recorded": {"raw_sqlite_export_total_size": "fixture: ~0 GB, no in-tree witness"},
        "producer_invocation": "fixture invocation",
    }
    if identity_override:
        identity_override(identity)
    merge_path = root / "merge.json"
    attr_path = root / "attr.json"
    identity_path = root / "identity.json"
    merge_path.write_text(json.dumps(merge_report), encoding="utf-8")
    attr_path.write_text(json.dumps(attribution_report), encoding="utf-8")
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    return {
        "legs_dir": legs_dir,
        "p2_dir": p2_dir,
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
            self.fixture["p2_dir"],
            self.fixture["merge_report"],
            self.fixture["attribution_report"],
            self.fixture["identity"],
        )

    def test_top_level_shape(self):
        report = self._build()
        expected_keys = {
            "schema_version", "git_sha", "box", "p2_witnessed", "producer", "status", "notes",
            "legs", "p2", "attribution", "realized_gains", "candidate_decisions", "findings",
            "suppressed_findings",
        }
        self.assertEqual(set(report.keys()), expected_keys)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["git_sha"], SHA)
        self.assertEqual(report["box"], BOX)
        self.assertEqual(report["producer"]["path"], "ci/scripts/perf/profile_421_legs.sh")
        self.assertEqual(report["producer"]["kind"], "script")
        self.assertEqual(report["producer"]["gating"], "none")
        self.assertEqual(report["suppressed_findings"], [])

    def test_status_is_green_when_merge_summary_is_clean(self):
        report = self._build()
        self.assertEqual(report["status"], "GREEN")

    def test_p2_witnessed_recorded_when_p2_rows_present(self):
        report = self._build()
        self.assertEqual(report["p2_witnessed"], {"towers": ["clip-text"], "git_sha": SHA, "box": BOX})

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

    def test_operator_recorded_passed_through_verbatim(self):
        report = self._build()
        self.assertEqual(
            report["notes"]["operator_recorded"],
            {"raw_sqlite_export_total_size": "fixture: ~0 GB, no in-tree witness"},
        )


class TemplatedDeviationTests(unittest.TestCase):
    """`notes.recorded_deviations` templates are filled from LIVE sources —
    every number checked here against an INDEPENDENT computation, never a
    value copied out of `profile_421_artifact.py` itself."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.fixture = _write_fixture(self.root)
        self.report = art.build_report(
            self.fixture["legs_dir"], self.fixture["p2_dir"], self.fixture["merge_report"], self.fixture["attribution_report"], self.fixture["identity"]
        )
        self.deviations = self.report["notes"]["recorded_deviations"]

    def tearDown(self):
        self._tmp.cleanup()

    def test_unfilled_template_passes_through_unchanged(self):
        self.assertEqual(self.deviations[0], "fixture deviation, no placeholders.")

    def test_htsat_a2_unattributed_share_and_contract_bounds_are_live(self):
        # Independent oracle: multiply the FIXTURE's own chain share (never
        # a value read back out of the module under test) by 100, and pull
        # the contract bounds straight off `profile_421_attribute`'s own
        # constants (imported directly, not re-typed).
        expected_pct = UNATTRIBUTED_SHARE_GPU_BUSY * 100.0
        expected_bound_pct = attribute_mod.UNATTRIBUTED_DECISION_GRADE_LIMIT * 100.0
        expected_gate_pct = attribute_mod.UNKNOWN_KERNEL_SHARE_LIMIT * 100.0
        self.assertEqual(expected_pct, 2.0)
        self.assertEqual(expected_bound_pct, 5.0)
        self.assertEqual(expected_gate_pct, 1.0)
        self.assertIn(f"{expected_pct:.2f}%", self.deviations[1])
        self.assertIn(f"{expected_bound_pct:.0f}%", self.deviations[1])
        self.assertIn(f"{expected_gate_pct:.0f}%", self.deviations[1])

    def test_htsat_a2_merge_verdict_and_decision_grade_word_are_live(self):
        # Both used to be hand-typed prose ("VALID" / "NOT decision-grade")
        # -- now read off the SAME fixture rows the chain-share number
        # above is read off, never independently asserted. The BASE fixture
        # has htsat-A2 both merge-VALID and (unlike the real committed run)
        # decision_grade True — the positive ("decision-grade", no "NOT")
        # word is exactly as live-derived as the negative one.
        self.assertEqual(
            {row["leg_id"]: row["verdict"] for row in self.fixture["merge_report"]["legs"]}["htsat-A2"], "VALID"
        )
        self.assertIn("VALID at the merge level", self.deviations[1])
        self.assertIn("but decision-grade for attribution", self.deviations[1])
        self.assertNotIn("NOT decision-grade", self.deviations[1])

    def test_htsat_a2_not_decision_grade_word_flips_live(self):
        def poison(attribution_report):
            for row in attribution_report["legs"]:
                if row["leg_id"] == "htsat-A2":
                    row["decision_grade"] = False

        alt_root = Path(tempfile.mkdtemp(dir=str(self.root)))
        fixture = _write_fixture(alt_root, attr_override=poison)
        report = art.build_report(
            fixture["legs_dir"], fixture["p2_dir"], fixture["merge_report"], fixture["attribution_report"], fixture["identity"]
        )
        self.assertIn("but NOT decision-grade for attribution", report["notes"]["recorded_deviations"][1])

    def test_htsat_a2_not_merge_valid_refuses_the_template_context(self):
        def poison(merge_report):
            for row in merge_report["legs"]:
                if row["leg_id"] == "htsat-A2":
                    row["verdict"] = "INVALID"

        alt_root = Path(tempfile.mkdtemp(dir=str(self.root)))
        fixture = _write_fixture(alt_root, merge_override=poison)
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            art.build_report(
                fixture["legs_dir"], fixture["p2_dir"], fixture["merge_report"], fixture["attribution_report"], fixture["identity"]
            )
        self.assertIn("not VALID", str(ctx.exception))
        self.assertIn("VALID at the merge level", str(ctx.exception))

    def test_merge_suite_test_count_matches_independent_unittest_discovery(self):
        # Independent oracle: `unittest`'s OWN loader, over the SAME real
        # committed `test_profile_421_merge.py`, called a SECOND time here
        # (never the module-under-test's own cached count).
        import importlib

        independent_module = importlib.import_module("test_profile_421_merge")
        expected = unittest.defaultTestLoader.loadTestsFromModule(independent_module).countTestCases()
        self.assertEqual(art.compute_merge_suite_test_count(), expected)
        self.assertIn(f"already carried {expected} tests", self.deviations[5])

    def test_sqlite_raw_export_count_is_two_per_leg_in_legs_dir(self):
        expected = len(LEG_SPECS) * 2  # run_n.sqlite + run_m.sqlite per leg
        self.assertEqual(expected, 12)
        self.assertIn(f"{expected} files", self.deviations[2])

    def test_corpus_pool_numbers_match_independent_parse_of_the_real_scripts(self):
        # A SEPARATE, simpler parser than `art._read_shell_int_const`/
        # `art._read_py_int_const` — deliberately not calling those
        # functions, so this is a genuine second computation, not the same
        # regex echoing itself.
        def independent_shell_int(path: Path, name: str) -> int:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith(f"{name}="):
                    rhs = line.split("=", 1)[1].split()[0].split("#")[0]
                    # `NAME="${ENV_VAR:-123}"` (env-defaulted) vs plain
                    # `NAME=123` — only the tail after `:-` carries the
                    # DEFAULT value; the env var's own name (e.g.
                    # `PROFILE_421_STEPS_M`) can itself contain digits.
                    tail = rhs.split(":-", 1)[1] if ":-" in rhs else rhs
                    digits = "".join(ch for ch in tail if ch.isdigit())
                    return int(digits)
            raise AssertionError(f"{name} not found in {path}")

        def independent_py_int(path: Path, name: str) -> int:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith(f"{name} ="):
                    return int(line.split("=", 1)[1].strip())
            raise AssertionError(f"{name} not found in {path}")

        media_families = independent_shell_int(art.LEGS_SH, "MEDIA_FAMILIES")
        media_heldout_families = independent_shell_int(art.LEGS_SH, "MEDIA_HELDOUT_FAMILIES")
        steps_m = independent_shell_int(art.LEGS_SH, "STEPS_M")
        batch = independent_shell_int(art.LEGS_SH, "BATCH")
        image_instances = independent_py_int(art.IMAGE_CORPUS_PY, "_DEFAULT_INSTANCES_PER_FAMILY")
        audio_instances = independent_py_int(art.AUDIO_CORPUS_PY, "_DEFAULT_INSTANCES_PER_FAMILY")
        self.assertEqual(image_instances, audio_instances)

        expected_total_files = media_families * image_instances
        expected_train_clips = (media_families - media_heldout_families) * image_instances
        expected_rows_m = batch * steps_m

        pool = art.compute_media_corpus_pool()
        self.assertEqual(pool["corpus_total_files"], expected_total_files)
        self.assertEqual(pool["corpus_train_clips"], expected_train_clips)
        self.assertEqual(pool["corpus_rows_m"], expected_rows_m)
        # Golden values as committed today (2026-09-07) — a change to any of
        # these constants should also update the identity sidecar's own
        # numbers via re-running the producer, never a hand-edit.
        self.assertEqual((expected_total_files, expected_train_clips, expected_rows_m), (24, 16, 4800))

        self.assertIn(f"{expected_total_files} files", self.deviations[3])
        self.assertIn(f"{expected_train_clips} distinct train clips", self.deviations[3])
        self.assertIn(f"{expected_rows_m} rows", self.deviations[3])

    def test_kernel_identity_split_count_and_leg_count_are_live(self):
        # Independent oracle: group the FIXTURE's own by_kernel_and_grid
        # rows for clip-text-A2 by (grid, block) and count distinct kernel
        # names, never a value read back out of the module under test.
        with open(self.fixture["legs_dir"] / "clip-text-A2" / "census.json", encoding="utf-8") as f:
            post = json.load(f)
        distinct_names = {row["kernel"] for row in post["by_kernel_and_grid"]}
        expected_split = len(distinct_names)
        expected_legs = len(LEG_SPECS)
        self.assertEqual(expected_split, KERNEL_IDENTITY_SPLIT_COUNT)
        self.assertEqual(expected_legs, 6)
        self.assertEqual(
            art.compute_kernel_identity_split_count(self.fixture["legs_dir"], "clip-text-A2", "Kernel2"),
            expected_split,
        )
        self.assertIn(f"{expected_split} distinct instantiations across {expected_legs} legs", self.deviations[4])

    def test_kernel_identity_split_count_refuses_when_coalesced_name_absent(self):
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            art.compute_kernel_identity_split_count(self.fixture["legs_dir"], "clip-text-A2", "NoSuchBucket")
        self.assertIn("no 'NoSuchBucket' bucket found", str(ctx.exception))

    def test_unknown_placeholder_refuses(self):
        def bad_identity(identity):
            identity["recorded_deviations"] = ["a fixture value that does not exist: {not_a_real_placeholder}."]

        alt_root = Path(tempfile.mkdtemp(dir=str(self.root)))
        fixture = _write_fixture(alt_root, identity_override=bad_identity)
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            art.build_report(fixture["legs_dir"], fixture["p2_dir"], fixture["merge_report"], fixture["attribution_report"], fixture["identity"])
        self.assertIn("unfillable template placeholder", str(ctx.exception))

    def test_recorded_deviations_must_be_a_list(self):
        def bad_identity(identity):
            identity["recorded_deviations"] = "not a list"

        alt_root = Path(tempfile.mkdtemp(dir=str(self.root)))
        fixture = _write_fixture(alt_root, identity_override=bad_identity)
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            art.build_report(fixture["legs_dir"], fixture["p2_dir"], fixture["merge_report"], fixture["attribution_report"], fixture["identity"])
        self.assertIn("must be a list", str(ctx.exception))


class IdentitySidecarNoBareMeasurementTests(unittest.TestCase):
    """The REAL committed `profile_421_run2_identity.json` carries no
    digit-bearing MEASUREMENT (a bare percentage, a bare count next to a
    unit word like files/rows/clips/GB/MB, a SPELLED-OUT numeral next to
    "DISTINCT", or a bare "All <N>" count) outside `operator_recorded` —
    every such number must instead be a `{placeholder}` this module fills
    live. Identifiers (git shas, issue/PR numbers, dates, `pass-4`-style
    version labels) are not measurements and are not what this regex
    targets."""

    _PERCENT_RE = re.compile(r"\d+(\.\d+)?\s*%")
    _UNIT_COUNT_RE = re.compile(r"\b\d+(\.\d+)?\b(?:\s+\S+){0,2}\s+(files?|rows?|clips?|tests?|GB|MB)\b")
    # "three DISTINCT ..." — a spelled-out numeral is exactly as much a
    # transcribed measurement as a digit; caught generically for any of the
    # small numerals this prose plausibly spells out.
    _SPELLED_NUMERAL_DISTINCT_RE = re.compile(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+DISTINCT\b"
    )
    # "All 12 censuses ..." — a bare count that the `{files?|rows?|...}`
    # unit-word regex above does not cover (the noun here is "censuses",
    # "legs", ...).
    _BARE_ALL_COUNT_RE = re.compile(r"\bAll \d+\b")

    def test_no_bare_percentage_or_unit_count_outside_operator_recorded(self):
        identity = json.loads(IDENTITY_SIDECAR.read_text(encoding="utf-8"))
        scrubbed = dict(identity)
        scrubbed.pop("operator_recorded", None)
        text = json.dumps(scrubbed)
        percent_hits = self._PERCENT_RE.findall(text)
        unit_hits = self._UNIT_COUNT_RE.findall(text)
        spelled_hits = self._SPELLED_NUMERAL_DISTINCT_RE.findall(text)
        all_count_hits = self._BARE_ALL_COUNT_RE.findall(text)
        self.assertEqual(percent_hits, [], f"bare percentage(s) found outside operator_recorded: {percent_hits}")
        self.assertEqual(unit_hits, [], f"bare unit-count(s) found outside operator_recorded: {unit_hits}")
        self.assertEqual(spelled_hits, [], f"spelled-out numeral(s) found outside operator_recorded: {spelled_hits}")
        self.assertEqual(all_count_hits, [], f"bare 'All <N>' count(s) found outside operator_recorded: {all_count_hits}")

    def test_unit_count_regex_catches_a_bare_test_count(self):
        # Regression for the gap this suite's own regex used to have: "45
        # tests" (CONTRACT.md's own stale, frozen figure) and "55 tests"
        # (the live count) both slipped through the old `_UNIT_COUNT_RE`
        # (which named `files?|rows?|clips?|GB|MB` but not `tests?`) — the
        # real committed sidecar no longer carries either as a bare digit
        # (the live one is `{merge_suite_test_count}`, the frozen one is
        # dropped in favor of a citation, see `test_no_bare_percentage_or_
        # unit_count_outside_operator_recorded` above), but the REGEX itself
        # must independently catch this shape so a FUTURE hand-typed count
        # cannot silently slip back in.
        hits = self._UNIT_COUNT_RE.findall("the suite already carried 55 tests as of this run")
        self.assertNotEqual(hits, [], "widened _UNIT_COUNT_RE failed to catch a bare 'N tests' count")

    def test_frozen_contract_test_count_is_never_quoted_as_a_bare_digit(self):
        identity = json.loads(IDENTITY_SIDECAR.read_text(encoding="utf-8"))
        text = json.dumps(identity)
        # CONTRACT.md's own frozen "45 hermetic tests" figure is quoted by
        # CITATION (path + section), never re-typed as a second,
        # independently-drifting digit this module cannot re-derive (it is
        # not this module's own file to read live -- see the deviation's own
        # prose).
        self.assertNotIn("45 tests", text)
        self.assertNotIn("45 hermetic tests", text)
        self.assertIn("CONTRACT.md", text)

    def test_operator_recorded_is_where_the_unwitnessed_size_lives(self):
        identity = json.loads(IDENTITY_SIDECAR.read_text(encoding="utf-8"))
        self.assertIn("operator_recorded", identity)
        self.assertIn("GB", identity["operator_recorded"]["raw_sqlite_export_total_size"])

    def test_status_field_no_longer_hand_declared_in_the_sidecar(self):
        identity = json.loads(IDENTITY_SIDECAR.read_text(encoding="utf-8"))
        self.assertNotIn("status", identity)


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
            self.fixture["legs_dir"], self.fixture["p2_dir"], self.fixture["merge_report"], self.fixture["attribution_report"], self.fixture["identity"]
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
        # the range by ascending MAGNITUDE, never a plain min-then-max pick,
        # and renders the ABSOLUTE magnitude (never a bare negative sign —
        # "cuts ... by -32%" would read as a GROWTH, not a cut).
        by_mag_busy = sorted(expected_busy.values(), key=abs)
        by_mag_wall = sorted(expected_wall.values(), key=abs)
        self.assertIn(f"by {abs(by_mag_busy[0]):.0f}-{abs(by_mag_busy[-1]):.0f}%", finding["text"])
        self.assertIn(f"by only {abs(by_mag_wall[0]):.0f}-{abs(by_mag_wall[-1]):.0f}%", finding["text"])
        self.assertNotIn("-32", finding["text"])
        self.assertNotIn("...", finding["text"])

    def test_c_attn_htsat_out_of_tier_oracle(self):
        finding = self.findings["c-attn-htsat-out-of-tier"]
        self.assertAlmostEqual(finding["evidence"]["share_gpu_busy"], C_ATTN_HTSAT_BUSY, places=9)
        self.assertAlmostEqual(finding["evidence"]["share_wall"], C_ATTN_HTSAT_WALL, places=9)
        self.assertIn(f"{C_ATTN_HTSAT_BUSY * 100.0:.0f}%", finding["text"])
        self.assertIn("out-of-tier".replace("-", ""), finding["id"].replace("-", ""))
        self.assertIn("OUTSIDE", finding["text"])
        self.assertIn("never decided under this issue", finding["text"])


class SignDerivedBf16WordingTests(unittest.TestCase):
    """`clip-launch-bound-batch8`'s BF16-vs-F32 wording is SIGN-DERIVED —
    never a hard-coded "cuts"/"drops" applied regardless of the measured
    direction. Covers: the committed (uniform-negative) case, a positive
    delta on ONE tower (mixed sign across towers), and a positive delta on
    BOTH towers."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _finding(self, merge_override=None):
        fixture = _write_fixture(self.root, merge_override=merge_override)
        report = art.build_report(
            fixture["legs_dir"], fixture["p2_dir"], fixture["merge_report"], fixture["attribution_report"], fixture["identity"]
        )
        return {f["id"]: f for f in report["findings"]}["clip-launch-bound-batch8"]

    def test_committed_negative_case_uses_a_single_cuts_drops_verb(self):
        finding = self._finding()
        self.assertLess(finding["evidence"]["busy_delta_pct_bf16_vs_f32"]["clip-text"], 0.0)
        self.assertLess(finding["evidence"]["busy_delta_pct_bf16_vs_f32"]["clip-vision"], 0.0)
        self.assertLess(finding["evidence"]["wall_delta_pct_bf16_vs_f32"]["clip-text"], 0.0)
        self.assertLess(finding["evidence"]["wall_delta_pct_bf16_vs_f32"]["clip-vision"], 0.0)
        self.assertIn("cuts GPU busy", finding["text"])
        self.assertIn("drops by only", finding["text"])
        self.assertNotIn("grows", finding["text"])

    def test_positive_busy_delta_on_one_tower_never_says_cuts_for_it(self):
        def bump(merge_report):
            for row in merge_report["legs"]:
                if row["leg_id"] == "clip-text-A2":
                    row["per_step"]["busy_s_per_step"] = 0.075  # > clip-text-A1's 0.060: positive delta

        finding = self._finding(bump)
        self.assertGreater(finding["evidence"]["busy_delta_pct_bf16_vs_f32"]["clip-text"], 0.0)
        self.assertLess(finding["evidence"]["busy_delta_pct_bf16_vs_f32"]["clip-vision"], 0.0)
        self.assertNotIn("clip-text BF16 cuts GPU busy", finding["text"])
        self.assertIn("clip-text BF16 grows GPU busy", finding["text"])
        self.assertIn("clip-vision BF16 cuts GPU busy", finding["text"])

    def test_positive_busy_delta_on_both_towers_never_says_cuts(self):
        def bump(merge_report):
            for row in merge_report["legs"]:
                if row["leg_id"] == "clip-text-A2":
                    row["per_step"]["busy_s_per_step"] = 0.075  # > clip-text-A1's 0.060
                if row["leg_id"] == "clip-vision-A2":
                    row["per_step"]["busy_s_per_step"] = 0.090  # > clip-vision-A1's 0.065

        finding = self._finding(bump)
        self.assertGreater(finding["evidence"]["busy_delta_pct_bf16_vs_f32"]["clip-text"], 0.0)
        self.assertGreater(finding["evidence"]["busy_delta_pct_bf16_vs_f32"]["clip-vision"], 0.0)
        self.assertNotIn("cuts GPU busy", finding["text"])
        self.assertIn("clip-text BF16 grows GPU busy", finding["text"])
        self.assertIn("clip-vision BF16 grows GPU busy", finding["text"])


class QualitativeWordRuleTests(unittest.TestCase):
    """Every qualitative word `compute_findings` can emit is gated behind a
    NAMED, numeric rule recorded in the finding's own `evidence` — for each
    rule this covers a probe where it HOLDS (the word appears, the numbers
    are still stated) and one where it does NOT (the word is absent, a
    neutral sentence states the SAME numbers, `evidence` still records the
    rule and its outcome). Uses `compute_findings` directly (never the full
    `build_report` pipeline / identity sidecar), since these are pure
    per-finding wording questions."""

    # Front-end time (seconds/step) close enough across A/D arms to clear
    # `ARM_INVARIANCE_REL_SPREAD_MAX` (relative spread ~1.6%, see the
    # arithmetic in the module doc of the test this feeds).
    HTSAT_D_LEGS_WITHIN_TOLERANCE = {
        "htsat-D1": {
            "tower": "htsat", "dtype": "f32", "arm": "D", "busy": 0.30, "wall": 1.60,
            "front_share": 0.775, "front_s_per_step": 1.24, "launches": 7000.0,
        },
        "htsat-D2": {
            "tower": "htsat", "dtype": "f32", "arm": "D", "busy": 0.28, "wall": 1.58,
            "front_share": 0.797, "front_s_per_step": 1.26, "launches": 6800.0,
        },
    }
    # Same shape, but front_s_per_step spread ~40% — well above the rule.
    HTSAT_D_LEGS_OUTSIDE_TOLERANCE = {
        "htsat-D1": {
            "tower": "htsat", "dtype": "f32", "arm": "D", "busy": 0.30, "wall": 1.60,
            "front_share": 0.625, "front_s_per_step": 1.00, "launches": 7000.0,
        },
        "htsat-D2": {
            "tower": "htsat", "dtype": "f32", "arm": "D", "busy": 0.28, "wall": 1.58,
            "front_share": 0.949, "front_s_per_step": 1.50, "launches": 6800.0,
        },
    }

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _findings(self, **kwargs):
        fixture = _write_fixture(self.root, **kwargs)
        return art.compute_findings(fixture["merge_report"], fixture["attribution_report"], fixture["legs_dir"])

    def _by_id(self, findings):
        return {f["id"]: f for f in findings}

    # -- "is CPU front-end-bound" ----------------------------------------
    def test_front_end_bound_word_present_when_rule_holds(self):
        findings, _ = self._findings()
        finding = self._by_id(findings)["htsat-front-end-bound"]
        self.assertIn("is CPU front-end-bound", finding["text"])
        self.assertTrue(finding["evidence"]["front_end_bound"])
        self.assertIn("front_share_of_wall >= 0.5", finding["evidence"]["front_end_bound_rule"])

    def test_front_end_bound_word_absent_when_rule_fails_on_one_leg(self):
        def lower_share(merge_report):
            for row in merge_report["legs"]:
                if row["leg_id"] == "htsat-A1":
                    row["per_step"]["front_share_of_wall"] = 0.30  # below FRONT_END_BOUND_SHARE_OF_WALL_MIN

        findings, _ = self._findings(merge_override=lower_share)
        finding = self._by_id(findings)["htsat-front-end-bound"]
        self.assertNotIn("is CPU front-end-bound", finding["text"])
        self.assertFalse(finding["evidence"]["front_end_bound"])
        # The numbers are still stated even though the word is dropped.
        self.assertIn("30-83%", finding["text"])

    # -- "dtype- and arm-invariant" ---------------------------------------
    def test_arm_invariant_clause_absent_when_d_legs_are_not_present(self):
        findings, _ = self._findings()  # the base fixture carries no D legs
        finding = self._by_id(findings)["htsat-front-end-bound"]
        self.assertNotIn("arm-invariant", finding["text"])
        self.assertNotIn("front_s_per_step", finding["evidence"])

    def test_arm_invariant_clause_present_when_d_legs_read_within_tolerance(self):
        specs = dict(LEG_SPECS)
        specs.update(self.HTSAT_D_LEGS_WITHIN_TOLERANCE)
        findings, _ = self._findings(leg_specs=specs)
        finding = self._by_id(findings)["htsat-front-end-bound"]
        self.assertIn("dtype- and arm-invariant", finding["text"])
        self.assertTrue(finding["evidence"]["arm_invariant"])
        self.assertLessEqual(finding["evidence"]["arm_invariance_relative_spread"], art.ARM_INVARIANCE_REL_SPREAD_MAX)
        self.assertEqual(set(finding["evidence"]["front_s_per_step"]), {"htsat-A1", "htsat-A2", "htsat-D1", "htsat-D2"})

    def test_arm_invariant_clause_dropped_when_spread_exceeds_tolerance(self):
        specs = dict(LEG_SPECS)
        specs.update(self.HTSAT_D_LEGS_OUTSIDE_TOLERANCE)
        findings, _ = self._findings(leg_specs=specs)
        finding = self._by_id(findings)["htsat-front-end-bound"]
        self.assertNotIn("dtype- and arm-invariant", finding["text"])
        self.assertIn("not treated as dtype-/arm-invariant here", finding["text"])
        self.assertFalse(finding["evidence"]["arm_invariant"])
        self.assertGreater(finding["evidence"]["arm_invariance_relative_spread"], art.ARM_INVARIANCE_REL_SPREAD_MAX)
        # The rule failing never suppresses the finding itself, nor the
        # PRIMARY (front_share_of_wall) clause -- only the invariance clause.
        self.assertIn("is CPU front-end-bound", finding["text"])

    def test_arm_invariant_clause_absent_when_one_d_leg_is_invalid(self):
        specs = dict(LEG_SPECS)
        specs.update(self.HTSAT_D_LEGS_WITHIN_TOLERANCE)

        def invalidate_one_d_leg(merge_report):
            for row in merge_report["legs"]:
                if row["leg_id"] == "htsat-D2":
                    row["verdict"] = "INVALID"

        findings, _ = self._findings(leg_specs=specs, merge_override=invalidate_one_d_leg)
        finding = self._by_id(findings)["htsat-front-end-bound"]
        self.assertNotIn("arm-invariant", finding["text"])
        # The REQUIRED (A-arm) finding is entirely unaffected by an
        # OPTIONAL D-arm leg going invalid.
        self.assertIn("is CPU front-end-bound", finding["text"])

    # -- "are launch-bound" -------------------------------------------------
    def test_launch_bound_word_present_when_rule_holds(self):
        findings, _ = self._findings()
        finding = self._by_id(findings)["clip-launch-bound-batch8"]
        self.assertIn("are launch-bound", finding["text"])
        self.assertTrue(finding["evidence"]["launch_bound"])

    def test_launch_bound_word_absent_when_residual_share_rule_fails(self):
        def starve_residual(merge_report):
            for row in merge_report["legs"]:
                if row["leg_id"] == "clip-text-A1":
                    # residual/wall = 0.001/0.100 = 1%, well below
                    # LAUNCH_BOUND_RESIDUAL_SHARE_OF_WALL_MIN.
                    row["per_step"]["residual_s_per_step"] = 0.001

        findings, _ = self._findings(merge_override=starve_residual)
        finding = self._by_id(findings)["clip-launch-bound-batch8"]
        self.assertNotIn("are launch-bound", finding["text"])
        self.assertIn("issue", finding["text"])
        self.assertFalse(finding["evidence"]["launch_bound"])
        # The launches/step numbers are still stated.
        self.assertIn("launches/step", finding["text"])

    # -- "only" (wall drops by ONLY X%) -------------------------------------
    def test_only_word_present_when_wall_drop_smaller_on_every_tower(self):
        findings, _ = self._findings()
        finding = self._by_id(findings)["clip-launch-bound-batch8"]
        self.assertIn("drops by only", finding["text"])
        self.assertTrue(finding["evidence"]["wall_drop_smaller_than_busy_drop_every_tower"])

    def test_only_word_dropped_when_wall_drop_is_not_smaller_on_a_tower(self):
        def big_wall_drop(merge_report):
            for row in merge_report["legs"]:
                if row["leg_id"] == "clip-text-A2":
                    row["per_step"]["wall_s_per_step"] = 0.056  # |wall delta| > |busy delta| for this tower
                    row["per_step"]["busy_s_per_step"] = 0.055

        findings, _ = self._findings(merge_override=big_wall_drop)
        finding = self._by_id(findings)["clip-launch-bound-batch8"]
        self.assertIn("drops by", finding["text"])
        self.assertNotIn("drops by only", finding["text"])
        self.assertFalse(finding["evidence"]["wall_drop_smaller_than_busy_drop_every_tower"])

    # -- the exact-zero arm gets its own clause shape ------------------------
    def test_zero_busy_delta_gets_its_own_clause_shape(self):
        def zero_busy_delta(merge_report):
            for row in merge_report["legs"]:
                if row["leg_id"] == "clip-text-A2":
                    row["per_step"]["busy_s_per_step"] = 0.060  # == clip-text-A1's own busy: exact-zero delta

        findings, _ = self._findings(merge_override=zero_busy_delta)
        finding = self._by_id(findings)["clip-launch-bound-batch8"]
        self.assertEqual(finding["evidence"]["busy_delta_pct_bf16_vs_f32"]["clip-text"], 0.0)
        self.assertIn("leaves GPU busy unchanged", finding["text"])
        self.assertNotIn("by 0%", finding["text"])

    # -- degenerate equal-after-rounding ranges collapse to a single value --
    def test_degenerate_equal_magnitude_range_collapses_to_a_single_value(self):
        def equalize_both_towers(merge_report):
            for row in merge_report["legs"]:
                if row["leg_id"] in ("clip-text-A1", "clip-vision-A1"):
                    row["per_step"]["busy_s_per_step"] = 0.100
                    row["per_step"]["wall_s_per_step"] = 0.100
                if row["leg_id"] in ("clip-text-A2", "clip-vision-A2"):
                    row["per_step"]["busy_s_per_step"] = 0.070
                    row["per_step"]["wall_s_per_step"] = 0.095

        findings, _ = self._findings(merge_override=equalize_both_towers)
        finding = self._by_id(findings)["clip-launch-bound-batch8"]
        self.assertAlmostEqual(finding["evidence"]["busy_delta_pct_bf16_vs_f32"]["clip-text"], -30.0, places=9)
        self.assertAlmostEqual(finding["evidence"]["busy_delta_pct_bf16_vs_f32"]["clip-vision"], -30.0, places=9)
        self.assertIn("by 30%", finding["text"])
        self.assertNotIn("30-30", finding["text"])
        self.assertIn("by only 5%", finding["text"])
        self.assertNotIn("5-5", finding["text"])


class SuppressedFindingsTests(unittest.TestCase):
    """A finding is computed ONLY from legs whose merge verdict is VALID
    (and, for a chain-share finding, whose attribution is decision_grade) —
    an INVALID leg suppresses (never silently taints) the findings that
    depend on it, and the suppression is named by finding id, leg(s), and
    reason."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _findings(self, **kwargs):
        # `compute_findings` directly — never the full `build_report`
        # pipeline, which ALSO runs `_identity_template_context` (gated on
        # htsat-A2 being merge-VALID, since the identity sidecar's own
        # "VALID at the merge level" deviation assumes exactly that). These
        # tests poison merge/attribution rows to exercise SUPPRESSION, a
        # `compute_findings`-only concern entirely independent of the
        # identity sidecar's own template gating.
        fixture = _write_fixture(self.root, **kwargs)
        return art.compute_findings(fixture["merge_report"], fixture["attribution_report"], fixture["legs_dir"])

    def test_invalid_htsat_legs_suppress_dependent_findings_not_emit_them_unqualified(self):
        def poison(merge_report):
            for row in merge_report["legs"]:
                if row["leg_id"] in ("htsat-A1", "htsat-A2"):
                    row["verdict"] = "INVALID"
                    row["reasons"] = ["synthetic: forced invalid for the suppression test"]

        findings, suppressed_findings = self._findings(merge_override=poison)
        finding_ids = {f["id"] for f in findings}
        self.assertNotIn("htsat-front-end-bound", finding_ids)
        self.assertNotIn("c-attn-htsat-out-of-tier", finding_ids)
        # Unaffected findings still build.
        self.assertIn("clip-vision-front-end-share", finding_ids)
        self.assertIn("clip-launch-bound-batch8", finding_ids)

        suppressed_by_id = {s["id"]: s for s in suppressed_findings}
        self.assertIn("htsat-front-end-bound", suppressed_by_id)
        entry = suppressed_by_id["htsat-front-end-bound"]
        self.assertEqual(set(entry["legs"]), {"htsat-A1", "htsat-A2"})
        self.assertIn("htsat-A1", entry["reason"])
        self.assertIn("htsat-A2", entry["reason"])
        self.assertIn("not VALID", entry["reason"])
        self.assertIn("c-attn-htsat-out-of-tier", suppressed_by_id)
        self.assertEqual(suppressed_by_id["c-attn-htsat-out-of-tier"]["legs"], ["htsat-A1"])

    def test_non_decision_grade_htsat_a1_suppresses_only_the_chain_share_finding(self):
        def poison(attribution_report):
            for row in attribution_report["legs"]:
                if row["leg_id"] == "htsat-A1":
                    row["decision_grade"] = False
                    row["decision_grade_reason"] = "synthetic: forced non-decision-grade"

        findings, suppressed_findings = self._findings(attr_override=poison)
        finding_ids = {f["id"] for f in findings}
        self.assertNotIn("c-attn-htsat-out-of-tier", finding_ids)
        # htsat-front-end-bound reads only merge per_step, not a chain share
        # or decision_grade, so it is UNAFFECTED by htsat-A1's attribution
        # status.
        self.assertIn("htsat-front-end-bound", finding_ids)

        suppressed_by_id = {s["id"]: s for s in suppressed_findings}
        self.assertIn("c-attn-htsat-out-of-tier", suppressed_by_id)
        self.assertIn("decision_grade", suppressed_by_id["c-attn-htsat-out-of-tier"]["reason"])


class RefusalTests(unittest.TestCase):
    """Every way the input can be inconsistent or non-finite must refuse —
    never silently pick a value or propagate a NaN into the artifact."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self, fixture):
        return art.build_report(fixture["legs_dir"], fixture["p2_dir"], fixture["merge_report"], fixture["attribution_report"], fixture["identity"])

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

    def test_refuses_leg_with_no_checkpoint_sha_at_all(self):
        def corrupt(merge_report):
            merge_report["legs"][0]["checkpoint_weights_sha256"] = None  # clip-text-A1

        fixture = _write_fixture(self.root, merge_override=corrupt)
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("no checkpoint_weights_sha256", str(ctx.exception))

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

    def test_refuses_duplicate_leg_id_in_merge_report(self):
        def duplicate(merge_report):
            merge_report["legs"].append(dict(merge_report["legs"][0]))

        fixture = _write_fixture(self.root, merge_override=duplicate)
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("duplicate leg_id", str(ctx.exception))

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

    def test_refuses_out_of_domain_share_above_one(self):
        def poison(merge_report):
            for row in merge_report["legs"]:
                if row["leg_id"] == "htsat-A1":
                    row["per_step"]["front_share_of_wall"] = 1.0001

        fixture = _write_fixture(self.root, merge_override=poison)
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("not in [0, 1]", str(ctx.exception))

    def test_refuses_out_of_domain_share_below_zero(self):
        def poison(merge_report):
            for row in merge_report["legs"]:
                if row["leg_id"] == "htsat-A1":
                    row["per_step"]["front_share_of_wall"] = -0.0001

        fixture = _write_fixture(self.root, merge_override=poison)
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("not in [0, 1]", str(ctx.exception))

    def test_boundary_shares_of_exactly_zero_and_one_are_accepted(self):
        # The boundary itself (`0.0`, `1.0`) is INSIDE the domain — this is
        # the oracle that proves `_finite_share`'s `<=`/`>=` are inclusive,
        # not off-by-one `</>` checks that would wrongly reject the edges.
        def boundary(merge_report):
            for row in merge_report["legs"]:
                if row["leg_id"] == "htsat-A1":
                    row["per_step"]["front_share_of_wall"] = 0.0
                if row["leg_id"] == "htsat-A2":
                    row["per_step"]["front_share_of_wall"] = 1.0

        fixture = _write_fixture(self.root, merge_override=boundary)
        report = self._build(fixture)
        finding = {f["id"]: f for f in report["findings"]}["htsat-front-end-bound"]
        self.assertEqual(finding["evidence"]["front_share_of_wall_pct"]["htsat-A1"], 0.0)
        self.assertEqual(finding["evidence"]["front_share_of_wall_pct"]["htsat-A2"], 100.0)

    def test_refuses_zero_denominator_busy_delta(self):
        def poison(merge_report):
            for row in merge_report["legs"]:
                if row["leg_id"] == "clip-text-A1":
                    row["per_step"]["busy_s_per_step"] = 0.0

        fixture = _write_fixture(self.root, merge_override=poison)
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("cannot compute a BF16-vs-F32 busy delta", str(ctx.exception))

    def test_refuses_zero_denominator_wall_delta(self):
        def poison(merge_report):
            for row in merge_report["legs"]:
                if row["leg_id"] == "clip-vision-A1":
                    row["per_step"]["wall_s_per_step"] = 0.0

        fixture = _write_fixture(self.root, merge_override=poison)
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("cannot compute a BF16-vs-F32 wall delta", str(ctx.exception))

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
            art.collect_identity(self.root / "does-not-exist", None, [])

    def test_collect_checkpoint_sha256_refuses_unknown_tower(self):
        def poison(merge_report):
            for row in merge_report["legs"]:
                if row["leg_id"] == "clip-text-A1":
                    row["tower"] = "unknown-tower"

        fixture = _write_fixture(self.root, merge_override=poison)
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("clip-text-A1", str(ctx.exception))
        self.assertIn("not one of this contract's known towers", str(ctx.exception))


class P2TowerNamesRefusalTests(unittest.TestCase):
    """`p2_tower_names` refuses a `p2_bf16` row with no non-empty string
    `tower` by name, rather than silently dropping it — a silent drop could
    shrink `p2_towers` to empty even though `--merge-json` genuinely carries
    `p2_bf16` rows, which would make `--p2-dir` wrongly optional and
    `p2_witnessed: null` wrongly claim "no p2_bf16 rows in the merge"."""

    def test_null_tower_refuses(self):
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            art.p2_tower_names({"p2_bf16": [{"tower": None, "verdict": "PASS"}]})
        self.assertIn("p2_bf16[0]", str(ctx.exception))

    def test_blank_tower_refuses(self):
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            art.p2_tower_names({"p2_bf16": [{"tower": "", "verdict": "PASS"}]})
        self.assertIn("p2_bf16[0]", str(ctx.exception))

    def test_all_rows_null_tower_refuses_never_treated_as_no_p2_rows(self):
        # The exact escape this closes: EVERY row invalid must still refuse,
        # never fall through to `p2_towers == []` (which would make
        # `--p2-dir` silently optional for a merge report that DID carry
        # p2_bf16 rows).
        with self.assertRaises(art.ArtifactBuildError):
            art.p2_tower_names({"p2_bf16": [{"tower": None}, {"tower": ""}]})

    def test_non_dict_row_refuses(self):
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            art.p2_tower_names({"p2_bf16": ["not-a-dict"]})
        self.assertIn("p2_bf16[0]", str(ctx.exception))

    def test_valid_rows_still_return_sorted_distinct_towers(self):
        towers = art.p2_tower_names(
            {"p2_bf16": [{"tower": "htsat"}, {"tower": "clip-text"}, {"tower": "clip-text"}]}
        )
        self.assertEqual(towers, ["clip-text", "htsat"])

    def test_build_report_refuses_end_to_end_when_every_p2_row_tower_is_blank(self):
        # The full pipeline, not just the unit function: a merge report that
        # DOES carry p2_bf16 rows (so --p2-dir must be witnessed) must never
        # silently fall through to "no p2_bf16 rows" just because every row
        # happened to carry a blank tower.
        def blank_p2_tower(merge_report):
            for row in merge_report["p2_bf16"]:
                row["tower"] = ""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _write_fixture(root, merge_override=blank_p2_tower)
            with self.assertRaises(art.ArtifactBuildError) as ctx:
                art.build_report(
                    fixture["legs_dir"], fixture["p2_dir"], fixture["merge_report"], fixture["attribution_report"], fixture["identity"]
                )
            self.assertIn("p2_bf16[0]", str(ctx.exception))


class StatusDerivationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_green_boundary_zero_invalid_zero_fail(self):
        fixture = _write_fixture(self.root)
        self.assertEqual(art.derive_status(fixture["merge_report"]), "GREEN")

    def test_one_invalid_leg_is_red(self):
        def poison(merge_report):
            merge_report["summary"]["legs_invalid"] = 1

        fixture = _write_fixture(self.root, merge_override=poison)
        status = art.derive_status(fixture["merge_report"])
        self.assertNotEqual(status, "GREEN")
        self.assertIn("1 leg(s) INVALID", status)

    def test_one_failed_p2_tower_is_red(self):
        def poison(merge_report):
            merge_report["summary"]["p2_fail"] = 1

        fixture = _write_fixture(self.root, merge_override=poison)
        status = art.derive_status(fixture["merge_report"])
        self.assertNotEqual(status, "GREEN")
        self.assertIn("1 P2 tower(s) FAIL", status)

    def test_refuses_missing_summary(self):
        fixture = _write_fixture(self.root, merge_override=lambda m: m.pop("summary"))
        with self.assertRaises(art.ArtifactBuildError):
            art.derive_status(fixture["merge_report"])

    def test_refuses_non_int_legs_invalid(self):
        def poison(merge_report):
            merge_report["summary"]["legs_invalid"] = "0"

        fixture = _write_fixture(self.root, merge_override=poison)
        with self.assertRaises(art.ArtifactBuildError):
            art.derive_status(fixture["merge_report"])


class P2WitnessRequiredTests(unittest.TestCase):
    """BLOCK fix: whenever `--merge-json` carries `p2_bf16` rows, `--p2-dir`
    is REQUIRED and every named P2 tower's own `manifest.json` must exist
    and agree — never a silent `continue` past a missing one, and never
    byte-identical to a run that skipped `--p2-dir` entirely."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_p2_dir_required_when_merge_has_p2_rows(self):
        fixture = _write_fixture(self.root, p2_towers=("clip-text",))
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            art.build_report(fixture["legs_dir"], None, fixture["merge_report"], fixture["attribution_report"], fixture["identity"])
        self.assertIn("--p2-dir was not given", str(ctx.exception))

    def test_regenerating_without_p2_dir_is_not_byte_identical(self):
        """The exact escape this fix closes: building WITH `--p2-dir` must
        differ from building WITHOUT it (a refusal, not a silently-identical
        report) whenever the merge report carries P2 rows."""
        fixture = _write_fixture(self.root, p2_towers=("clip-text",))
        with_p2 = art.build_report(
            fixture["legs_dir"], fixture["p2_dir"], fixture["merge_report"], fixture["attribution_report"], fixture["identity"]
        )
        with self.assertRaises(art.ArtifactBuildError):
            art.build_report(fixture["legs_dir"], None, fixture["merge_report"], fixture["attribution_report"], fixture["identity"])
        self.assertIsNotNone(with_p2["p2_witnessed"])

    def test_missing_manifest_for_a_named_p2_tower_refuses(self):
        fixture = _write_fixture(self.root, p2_towers=("clip-text", "clip-vision"))
        # clip-vision's manifest.json never got written for this tower.
        shutil.rmtree(fixture["p2_dir"] / "clip-vision")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            art.build_report(fixture["legs_dir"], fixture["p2_dir"], fixture["merge_report"], fixture["attribution_report"], fixture["identity"])
        self.assertIn("no manifest.json exists", str(ctx.exception))

    def test_disagreeing_p2_manifest_refuses(self):
        fixture = _write_fixture(self.root, p2_towers=("clip-text",))
        manifest_path = fixture["p2_dir"] / "clip-text" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["git_sha"] = "b" * 40
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            art.build_report(fixture["legs_dir"], fixture["p2_dir"], fixture["merge_report"], fixture["attribution_report"], fixture["identity"])
        self.assertIn("disagree", str(ctx.exception))

    def test_no_p2_rows_means_p2_dir_is_optional_and_p2_witnessed_is_none(self):
        fixture = _write_fixture(self.root, p2_towers=())
        report = art.build_report(fixture["legs_dir"], None, fixture["merge_report"], fixture["attribution_report"], fixture["identity"])
        self.assertIsNone(report["p2_witnessed"])
        self.assertEqual(report["p2"], [])


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
            "--p2-dir", str(self.fixture["p2_dir"]),
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
        self.assertEqual(report["suppressed_findings"], [])
        self.assertEqual(report["p2_witnessed"], {"towers": ["clip-text"], "git_sha": SHA, "box": BOX})
        self.assertEqual(report["status"], "GREEN")
        # Producer identity: sha256 of the exact input FILES, the sha256 of
        # this producer's OWN source (never a git commit sha — see the
        # module doc's "Producer identity" section), and this run's own
        # invocation — never present on the hermetic `build_report` output,
        # only on the CLI's.
        producer = report["producer"]
        self.assertEqual(
            set(producer["input_sha256"].keys()), {"merge_json", "attribution_json", "identity"}
        )
        for digest in producer["input_sha256"].values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertNotIn("tree_sha", producer)
        self.assertEqual(
            set(producer["source_sha256"].keys()),
            {
                "ci/scripts/perf/profile_421_artifact.py",
                "ci/scripts/perf/profile_421_attribute.py",
                "ci/scripts/perf/profile_421_legs.sh",
                "ci/scripts/perf/gen_fixed_shape_image_corpus.py",
                "ci/scripts/perf/gen_fixed_length_audio_corpus.py",
                "ci/scripts/perf/test_profile_421_merge.py",
            },
        )
        for digest in producer["source_sha256"].values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(producer["invocation_argv"][0], "profile_421_artifact.py")
        self.assertIn("--legs-dir", producer["invocation_argv"])

    def test_cli_source_sha256_matches_independent_hash_of_the_real_producer_files(self):
        import hashlib

        out_path = self.root / "artifact.json"
        result = run_artifact_cli(
            "--legs-dir", str(self.fixture["legs_dir"]),
            "--p2-dir", str(self.fixture["p2_dir"]),
            "--merge-json", str(self.fixture["merge_path"]),
            "--attribution-json", str(self.fixture["attr_path"]),
            "--identity", str(self.fixture["identity_path"]),
            "--out", str(out_path),
        )
        self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")
        report = json.loads(out_path.read_text())
        repo_root = PERF_DIR.parents[2]
        for relpath, digest in report["producer"]["source_sha256"].items():
            expected = hashlib.sha256((repo_root / relpath).read_bytes()).hexdigest()
            self.assertEqual(digest, expected, f"{relpath}: source_sha256 does not match the real file's own bytes")

    def test_cli_input_sha256_matches_independent_hash_of_the_same_files(self):
        import hashlib

        out_path = self.root / "artifact.json"
        result = run_artifact_cli(
            "--legs-dir", str(self.fixture["legs_dir"]),
            "--p2-dir", str(self.fixture["p2_dir"]),
            "--merge-json", str(self.fixture["merge_path"]),
            "--attribution-json", str(self.fixture["attr_path"]),
            "--identity", str(self.fixture["identity_path"]),
            "--out", str(out_path),
        )
        self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")
        report = json.loads(out_path.read_text())
        expected_merge_sha = hashlib.sha256(self.fixture["merge_path"].read_bytes()).hexdigest()
        expected_attr_sha = hashlib.sha256(self.fixture["attr_path"].read_bytes()).hexdigest()
        expected_identity_sha = hashlib.sha256(self.fixture["identity_path"].read_bytes()).hexdigest()
        self.assertEqual(report["producer"]["input_sha256"]["merge_json"], expected_merge_sha)
        self.assertEqual(report["producer"]["input_sha256"]["attribution_json"], expected_attr_sha)
        self.assertEqual(report["producer"]["input_sha256"]["identity"], expected_identity_sha)

    def test_cli_refuses_missing_p2_dir_when_merge_has_p2_rows(self):
        result = run_artifact_cli(
            "--legs-dir", str(self.fixture["legs_dir"]),
            "--merge-json", str(self.fixture["merge_path"]),
            "--attribution-json", str(self.fixture["attr_path"]),
            "--identity", str(self.fixture["identity_path"]),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--p2-dir was not given", result.stderr)

    def test_cli_refuses_and_exits_nonzero_on_bad_input(self):
        bad_manifest_path = self.fixture["legs_dir"] / "clip-text-A1" / "manifest.json"
        manifest = json.loads(bad_manifest_path.read_text())
        manifest["git_sha"] = "b" * 40
        bad_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result = run_artifact_cli(
            "--legs-dir", str(self.fixture["legs_dir"]),
            "--p2-dir", str(self.fixture["p2_dir"]),
            "--merge-json", str(self.fixture["merge_path"]),
            "--attribution-json", str(self.fixture["attr_path"]),
            "--identity", str(self.fixture["identity_path"]),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("disagree", result.stderr)


if __name__ == "__main__":
    unittest.main()
