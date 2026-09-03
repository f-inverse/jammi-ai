#!/usr/bin/env python3
"""Tests for `check_gpu_prove_once.py` (esc-084, issue #454; #454 follow-up,
operator direction 2026-09-03: every release publisher, not only the CUDA
lanes).

Drives the real `run_gate()`/`check_p1_p2()`/`check_promotion_table()`/
`check_p4()`/`check_p5()`/`check_p6_discovery()`/`check_promoting_if()`/
`reconstruct_if_expr()`/`split_top_level()`/`read_top_level_on_block()`/
`read_jobs_block_or_fail()`/`job_invokes_publish_primitive_recursive()`
entry points against synthetic fixture trees (never a hand-built stand-in
for the parsers themselves) — including a fixture reproducing the PRE-FIX
shape (esc-084: three publishers `uses:` a renting reusable), which must
fail naming every offending site, and a positive fixture (now covering
every `PROMOTION_TABLE` row: the CUDA lanes, the CI base-image callers,
crates.io, npm, and every PyPI dist) that must pass clean.

Run directly: `python3 ci/scripts/test_check_gpu_prove_once.py`
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_gpu_prove_once as cgo  # noqa: E402
import check_gpu_parity_matrix as gpu_parity_matrix  # noqa: E402
import gpu_prove_verdict  # noqa: E402

REAL_ARCHES = sorted(gpu_parity_matrix.load_shipped_cuda_silicon())
ARCH_LIST = ", ".join(REAL_ARCHES)
JOB_NAME_LINE = "GPU prove on RunPod (${{ matrix.arch }})"

MANIFEST_GOOD = {
    "lanes": {
        "cu12-image": {"cargo_features": ["cuda", "flash-attn"]},
        "cu12-tarball": {"cargo_features": ["cuda", "flash-attn"]},
        "cu12-wheel": {"cargo_features": ["cuda", "flash-attn"]},
    }
}

PROVE_YML_GOOD = f"""\
name: GPU prove (RunPod)

on:
  workflow_dispatch:
  pull_request:
    types: [labeled]
  schedule:
    - cron: "47 3 * * *"

permissions:
  contents: read

jobs:
  gpu-prove:
    name: {JOB_NAME_LINE}
    strategy:
      matrix:
        arch: [{ARCH_LIST}]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Prove GPU suites
        run: |
          bash ci/scripts/runpod_gpu_prove.sh
"""


PROOF_REQUIRED_YML_GOOD = """\
name: _gpu-proof-required

on:
  workflow_call: {}

permissions:
  contents: read
  actions: read

jobs:
  proof-required:
    name: GPU proof required
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - name: Check the commit's GPU-prove verdict (gpu-prove.yml job conclusions at github.sha)
        env:
          GITHUB_TOKEN: ${{ github.token }}
        run: |
          python3 ci/scripts/gpu_prove_verdict.py \\
            --repo "$GITHUB_REPOSITORY" \\
            --sha "$GITHUB_SHA"
"""


def _gate_job(gate_name: str = "gpu-proof", tag_family: str = "v") -> str:
    return f"""\
  {gate_name}:
    name: GPU proof required
    if: startsWith(github.ref, 'refs/tags/{tag_family}')
    uses: ./.github/workflows/_gpu-proof-required.yml
    permissions:
      contents: read
      actions: read
    secrets: inherit
"""


def _promoting_job(
    name: str,
    gate_name: str = "gpu-proof",
    if_expr: str | None = None,
    raw_if_block: str | None = None,
    raw_needs_block: str | None = None,
    tag_family: str = "v",
) -> str:
    """A job with `needs:`/`if:` gating the way every `"direct"`/`"chained"`
    PROMOTION_TABLE row expects. `raw_if_block`/`raw_needs_block`, when
    given, are inserted VERBATIM (already indented, trailing newline
    included) instead of the default single-line form -- used to drive a
    folded block scalar or a multi-line `needs:` list through the real
    parser."""
    needs_section = raw_needs_block if raw_needs_block is not None else f"    needs: [{gate_name}]\n"
    if raw_if_block is not None:
        if_section = raw_if_block
    else:
        if if_expr is None:
            if_expr = (
                f"always() && startsWith(github.ref, 'refs/tags/{tag_family}') && "
                f"needs.{gate_name}.result == 'success'"
            )
        if_section = f"    if: {if_expr}\n"
    return (
        f"  {name}:\n"
        f"{needs_section}"
        f"{if_section}"
        f"    runs-on: ubuntu-latest\n"
        f"    steps:\n"
        f"      - uses: actions/checkout@v4\n"
    )


def _ungated_job(name: str, if_expr: str) -> str:
    """The `gate_kind == "none"` shape: no `needs:`, no gate conjunct --
    just an `if:` that must structurally carry the exact
    `github.ref_type != 'tag'` conjunct (F3 audit fix)."""
    return f"""\
  {name}:
    if: {if_expr}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
"""


def _local_reusable_caller_yml(
    caller_job_name: str = "build",
    target: str = "_ci-base-image.yml",
    if_expr: str = "github.ref_type != 'tag'",
    on_block: str = 'on:\n  push:\n    branches: [main]\n  workflow_dispatch:\n',
) -> str:
    """The `image.yml`/`image-cuda.yml` shape: a job whose ENTIRE body is a
    job-level `uses: ./.github/workflows/<target>.yml` call, gated (or not)
    by its own `if:` -- used to drive the F1 recursive-discovery mechanism
    (a job that merely delegates to a local reusable which itself pushes is
    still a promoting job)."""
    return f"""\
name: caller

{on_block}
jobs:
  {caller_job_name}:
    if: {if_expr}
    uses: ./.github/workflows/{target}
"""


CI_BASE_IMAGE_YML = """\
name: _ci-base-image

on:
  workflow_call:
    inputs:
      image_suffix:
        type: string
        required: true

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/build-push-action@10e90e3645eae34f1e60eeb005ba3a3d33f178e8
        with:
          push: true
"""


def _step_gated_job(job_name: str, gate_name: str, step_name: str, step_if: str | None = None) -> str:
    """The npm.yml shape: the JOB always runs (`if: always()`, build+test
    unconditional), and the gate conjunct lives on one named STEP's own
    `if:` instead."""
    if step_if is None:
        step_if = f"always() && startsWith(github.ref, 'refs/tags/v') && needs.{gate_name}.result == 'success'"
    return f"""\
  {job_name}:
    needs: [{gate_name}]
    if: always()
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: {step_name}
        if: {step_if}
        run: echo publish
"""


def _wf(tag_pattern: str, jobs_text: str) -> str:
    return f'name: publisher\n\non:\n  push:\n    tags: ["{tag_pattern}"]\n\njobs:\n{jobs_text}'


def _server_image_yml(
    cu12_if: str | None = None,
    cpu_tag_if: str | None = None,
    main_if: str = (
        "github.event_name != 'pull_request' && github.ref == 'refs/heads/main' && github.ref_type != 'tag'"
    ),
    selfcontained_if: str = (
        "github.event_name == 'workflow_dispatch' && inputs.selfcontained && github.ref_type != 'tag'"
    ),
) -> str:
    jobs = (
        _gate_job("gpu-proof")
        + _promoting_job("build-and-push-cu12", if_expr=cu12_if)
        + _promoting_job("build-and-push", if_expr=cpu_tag_if)
        + _ungated_job("build-and-push-main", main_if)
        + _ungated_job("build-and-push-selfcontained", selfcontained_if)
    )
    return _wf("v*", jobs)


def _release_binaries_yml(
    cu12_if: str | None = None,
    cli_if: str | None = None,
    server_cpu_if: str | None = None,
    raw_cu12_if_block: str | None = None,
    raw_cu12_needs_block: str | None = None,
) -> str:
    jobs = (
        _gate_job("gpu-proof")
        + _promoting_job(
            "server-cu12-promote", if_expr=cu12_if, raw_if_block=raw_cu12_if_block, raw_needs_block=raw_cu12_needs_block
        )
        + _promoting_job("promote-binaries", if_expr=cli_if)
        + _promoting_job("server-cpu-promote", if_expr=server_cpu_if)
    )
    return _wf("v*", jobs)


def _crates_yml(publish_if: str | None = None, github_release_if: str | None = None) -> str:
    jobs = _gate_job("gpu-proof") + _promoting_job("publish", if_expr=publish_if) + _promoting_job(
        "github-release", gate_name="publish", if_expr=github_release_if
    )
    return _wf("v*", jobs)


def _npm_yml(step_if: str | None = None) -> str:
    jobs = _gate_job("gpu-proof") + _step_gated_job("publish", "gpu-proof", "Publish", step_if=step_if)
    return _wf("v*", jobs)


def _simple_publish_yml(
    tag_pattern: str = "py-v*", publish_if: str | None = None, tag_family: str = "py-v"
) -> str:
    jobs = _gate_job("gpu-proof", tag_family=tag_family) + _promoting_job(
        "publish", if_expr=publish_if, tag_family=tag_family
    )
    return _wf(tag_pattern, jobs)


def write_tree(root: Path, workflows: dict[str, str], manifest: dict) -> tuple[Path, Path]:
    wf_dir = root / ".github" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    for name, text in workflows.items():
        (wf_dir / name).write_text(text)
    manifest_path = root / "release-feature-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return wf_dir, manifest_path


def positive_workflows() -> dict[str, str]:
    """One fully-valid workflow file per `PROMOTION_TABLE` row's workflow --
    every row must find its promoting job, its gate, and a clean `if:` here,
    or `run_gate()`'s positive-fixture test below would not actually be
    positive."""
    return {
        "gpu-prove.yml": PROVE_YML_GOOD,
        "_gpu-proof-required.yml": PROOF_REQUIRED_YML_GOOD,
        "server-image.yml": _server_image_yml(),
        "release-binaries.yml": _release_binaries_yml(),
        "crates.yml": _crates_yml(),
        "npm.yml": _npm_yml(),
        "_ci-base-image.yml": CI_BASE_IMAGE_YML,
        "image.yml": _local_reusable_caller_yml("build", "_ci-base-image.yml"),
        "image-cuda.yml": _local_reusable_caller_yml("build", "_ci-base-image.yml"),
        "pypi.yml": _simple_publish_yml(),
        "pypi-client.yml": _simple_publish_yml(),
        "pypi-server.yml": _simple_publish_yml(),
        "pypi-server-cuda.yml": _simple_publish_yml(),
    }


def _positive_texts() -> dict[str, str]:
    return dict(positive_workflows())


class RunGatePositiveTest(unittest.TestCase):
    def test_positive_fixture_passes_clean(self):
        with tempfile.TemporaryDirectory() as td:
            wf_dir, manifest_path = write_tree(Path(td), positive_workflows(), MANIFEST_GOOD)
            findings = cgo.run_gate(wf_dir, manifest_path)
            self.assertEqual(findings, [])


class RealTreeTest(unittest.TestCase):
    """The actual `.github/workflows` tree and `ci/release-feature-
    manifest.json` this repo ships must themselves pass -- a synthetic
    fixture passing is necessary but not sufficient; the real thing must
    too."""

    def test_real_tree_passes(self):
        findings = cgo.run_gate(cgo.WORKFLOWS_DIR, cgo.MANIFEST_PATH)
        self.assertEqual(findings, [])


class PreFixShapeFixtureTest(unittest.TestCase):
    """Reproduces the PRE-FIX shape (esc-084's own wording): three
    publishers `uses:` a renting `_gpu-prove-gate.yml` which itself invokes
    `runpod_gpu_prove.sh` -- must FAIL naming all three publisher sites."""

    def test_pre_fix_shape_fails_naming_all_three_publishers(self):
        gate_renting = (
            "name: _gpu-prove-gate\n\non:\n  workflow_call:\n    inputs:\n      git_ref:\n"
            "        type: string\n        required: true\n\njobs:\n  prove:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - run: bash ci/scripts/runpod_gpu_prove.sh\n"
        )
        publisher = (
            "name: publisher\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  gpu-prove-cu12:\n    uses: ./.github/workflows/_gpu-prove-gate.yml\n"
            "    with:\n      git_ref: ${{ github.ref_name }}\n"
            "  promote:\n    needs: [gpu-prove-cu12]\n"
            "    if: always() && needs.gpu-prove-cu12.result == 'success'\n"
            "    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n"
        )
        workflows = {
            "gpu-prove.yml": PROVE_YML_GOOD,
            "_gpu-prove-gate.yml": gate_renting,
            "server-image.yml": publisher,
            "release-binaries.yml": publisher,
            "pypi-server-cuda.yml": publisher,
        }
        with tempfile.TemporaryDirectory() as td:
            wf_dir, manifest_path = write_tree(Path(td), workflows, MANIFEST_GOOD)
            findings = cgo.run_gate(wf_dir, manifest_path)
            self.assertTrue(findings)
            joined = "\n".join(findings)
            for site in ("server-image.yml", "release-binaries.yml", "pypi-server-cuda.yml"):
                self.assertIn(site, joined, f"{site} must be named among the findings")
            self.assertIn("_gpu-prove-gate.yml", joined)


class ProducerCountTest(unittest.TestCase):
    def test_zero_producers_fails(self):
        with tempfile.TemporaryDirectory() as td:
            wf_dir, manifest_path = write_tree(
                Path(td),
                {"gpu-prove.yml": PROVE_YML_GOOD.replace("bash ci/scripts/runpod_gpu_prove.sh", "echo nothing")},
                MANIFEST_GOOD,
            )
            findings = cgo.check_p1_p2(cgo.load_workflow_texts(wf_dir))
            self.assertTrue(any("zero workflows invoke" in f for f in findings))

    def test_two_producers_fails_naming_extra(self):
        second = PROVE_YML_GOOD.replace("name: GPU prove (RunPod)", "name: second-prover")
        with tempfile.TemporaryDirectory() as td:
            wf_dir, manifest_path = write_tree(
                Path(td), {"gpu-prove.yml": PROVE_YML_GOOD, "second-prover.yml": second}, MANIFEST_GOOD
            )
            findings = cgo.check_p1_p2(cgo.load_workflow_texts(wf_dir))
            joined = "\n".join(findings)
            self.assertIn("more than one workflow", joined)
            self.assertIn("second-prover.yml", joined)


class ManifestReconciliationTest(unittest.TestCase):
    """P3 (module doc): a SUBSET check only, one direction -- every manifest
    CUDA lane needs a table row; a table row naming a lane the manifest
    doesn't declare is expected and unflagged (most rows promote a
    non-CUDA, non-manifest surface)."""

    def test_cuda_lane_with_no_table_row_fails(self):
        manifest = dict(MANIFEST_GOOD)
        manifest["lanes"] = dict(MANIFEST_GOOD["lanes"])
        manifest["lanes"]["cu13-new-lane"] = {"cargo_features": ["cuda"]}
        findings = cgo.check_promotion_table(_positive_texts(), manifest)
        self.assertTrue(any("cu13-new-lane" in f and "no PROMOTION_TABLE row" in f for f in findings))

    def test_table_row_naming_lane_absent_from_manifest_is_not_flagged(self):
        manifest = {"lanes": {k: v for k, v in MANIFEST_GOOD["lanes"].items() if k != "cu12-wheel"}}
        findings = cgo.check_promotion_table(_positive_texts(), manifest)
        self.assertFalse(
            any("absent from the manifest" in f for f in findings),
            f"the reverse direction is intentionally unflagged now; got {findings}",
        )

    def test_missing_workflow_file_fails(self):
        texts = _positive_texts()
        del texts["pypi-server-cuda.yml"]
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertTrue(any("pypi-server-cuda.yml is missing" in f for f in findings))


class PromotingIfTest(unittest.TestCase):
    def _p3_for(self, if_expr: str) -> list[str]:
        texts = _positive_texts()
        texts["release-binaries.yml"] = _release_binaries_yml(cu12_if=if_expr)
        return cgo.check_promotion_table(texts, MANIFEST_GOOD)

    def test_missing_result_success_conjunct_fails(self):
        findings = self._p3_for("always() && startsWith(github.ref, 'refs/tags/v')")
        self.assertTrue(any("no top-level conjunct" in f for f in findings))

    def test_precedence_bypass_depth0_or_fails(self):
        findings = self._p3_for("github.event_name == 'push' || always() && needs.gpu-proof.result == 'success'")
        self.assertTrue(any("depth-0 `||`" in f for f in findings))

    def test_paren_string_hiding_or_still_caught(self):
        findings = self._p3_for("contains(needs.gpu-proof.outputs.v, '(') || needs.gpu-proof.result == 'success'")
        self.assertTrue(any("depth-0 `||`" in f for f in findings), findings)

    def test_wrapped_expression_reconstituted_positive(self):
        findings = self._p3_for(
            "${{ always() && startsWith(github.ref, 'refs/tags/v') && needs.gpu-proof.result == 'success' }}"
        )
        self.assertEqual(findings, [])

    def test_normalization_accepts_no_spaces_around_equals(self):
        findings = self._p3_for("always() && startsWith(github.ref, 'refs/tags/v') && needs.gpu-proof.result=='success'")
        self.assertEqual(findings, [])

    def test_duplicated_gate_term_under_different_job_name_fails(self):
        # names a DIFFERENT job's result -- must not satisfy the table's own gate job name.
        findings = self._p3_for("always() && needs.some-other-job.result == 'success'")
        self.assertTrue(any("no top-level conjunct" in f for f in findings))

    def test_folded_block_scalar_if_reconstituted_positive(self):
        raw_if_block = (
            "    if: >-\n"
            "      always() &&\n"
            "      startsWith(github.ref, 'refs/tags/v') &&\n"
            "      needs.gpu-proof.result == 'success'\n"
        )
        texts = _positive_texts()
        texts["release-binaries.yml"] = _release_binaries_yml(raw_cu12_if_block=raw_if_block)
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertEqual(findings, [])

    def test_unterminated_block_fails_loud(self):
        raw_if_block = "    if: >-\n"
        texts = _positive_texts()
        texts["release-binaries.yml"] = _release_binaries_yml(raw_cu12_if_block=raw_if_block)
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertTrue(any("unterminated block" in f for f in findings))


class GateKindTest(unittest.TestCase):
    """`"direct"`/`"chained"`/`"none"` -- each gate_kind's own structural
    rule (module doc's P3 description)."""

    def test_direct_gate_job_missing_the_reusable_uses_fails(self):
        # gpu-proof exists but never `uses: _gpu-proof-required.yml`.
        texts = _positive_texts()
        texts["release-binaries.yml"] = _wf(
            "v*",
            "  gpu-proof:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo not-the-reusable\n"
            + _promoting_job("server-cu12-promote")
            + _promoting_job("promote-binaries")
            + _promoting_job("server-cpu-promote"),
        )
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertTrue(any("does not `uses: ./.github/workflows/_gpu-proof-required.yml`" in f for f in findings))

    def test_chained_gate_job_not_another_row_promoting_job_fails(self):
        # `PROMOTION_TABLE` is a fixed, hand-reviewed constant -- a
        # workflow-text-only fixture cannot perturb WHAT a "chained" row
        # declares as its own `gate_job`, only whether the workflow tree
        # matches it. Exercising this structural rule means monkeypatching
        # the table itself: a "chained" row whose declared `gate_job` is not
        # some OTHER row's `promoting_job` in the SAME workflow (a reviewer
        # typo, or a row whose gate job was renamed/removed elsewhere).
        original = cgo.PROMOTION_TABLE
        try:
            broken = dict(original)
            broken["crates-github-release"] = cgo.PromotionRow(
                "crates.yml", "github-release", "nonexistent-job", "chained"
            )
            cgo.PROMOTION_TABLE = broken
            texts = _positive_texts()
            texts["crates.yml"] = texts["crates.yml"].replace(
                "  github-release:\n    needs: [publish]",
                "  github-release:\n    needs: [nonexistent-job]",
            ).replace(
                "needs.publish.result == 'success'", "needs.nonexistent-job.result == 'success'"
            )
            findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        finally:
            cgo.PROMOTION_TABLE = original
        self.assertTrue(
            any("is not some OTHER row's promoting_job" in f for f in findings), findings
        )

    def test_none_row_reachable_from_a_release_tag_fails(self):
        # An `if:` that names NO ref restriction at all lacks the exact
        # `github.ref_type != 'tag'` conjunct -- F3 audit fix.
        texts = _positive_texts()
        texts["server-image.yml"] = _server_image_yml(main_if="startsWith(github.ref, 'refs/tags/v')")
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertTrue(
            any("gate_kind='none'" in f and "build-and-push-main" in f for f in findings), findings
        )

    def test_none_row_real_leak_shape_no_ref_restriction_at_all_fails(self):
        # F3 audit fix (BLOCK B6b real leak, server-image.yml:121): the
        # PRE-FIX `selfcontained_if` shape -- gated only on the dispatch
        # input, no ref restriction whatsoever -- must FAIL now. Before this
        # fix, a `workflow_dispatch` against a `v*` tag ref with
        # `selfcontained=true` pushed this image entirely ungated.
        texts = _positive_texts()
        texts["server-image.yml"] = _server_image_yml(
            selfcontained_if="github.event_name == 'workflow_dispatch' && inputs.selfcontained"
        )
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertTrue(
            any(
                "gate_kind='none'" in f and "build-and-push-selfcontained" in f and "ref_type" in f
                for f in findings
            ),
            findings,
        )

    def test_none_row_missing_if_at_all_fails(self):
        texts = _positive_texts()
        texts["server-image.yml"] = _server_image_yml(main_if="true")
        # A trivial `if: true` still has no `github.ref_type != 'tag'` conjunct.
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertTrue(any("build-and-push-main" in f for f in findings), findings)

    def test_none_row_ungated_branch_only_if_passes(self):
        findings = cgo.check_promotion_table(_positive_texts(), MANIFEST_GOOD)
        self.assertEqual(findings, [])

    def test_direct_row_gate_job_missing_tag_guard_fails(self):
        # F7 audit fix: the GATE job's own `if:` must also carry the row's
        # exact tag-family conjunct -- a gate job reachable off no tag
        # restriction would let the verdict be consulted (and satisfied)
        # outside the release-tag path this row exists to gate.
        texts = _positive_texts()
        texts["release-binaries.yml"] = _wf(
            "v*",
            "  gpu-proof:\n    uses: ./.github/workflows/_gpu-proof-required.yml\n"
            + _promoting_job("server-cu12-promote")
            + _promoting_job("promote-binaries")
            + _promoting_job("server-cpu-promote"),
        )
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertTrue(
            any("gate job `gpu-proof`" in f and "F7 tag guard" in f for f in findings), findings
        )

    def test_promoting_job_missing_tag_guard_fails(self):
        # F7 audit fix: the PROMOTING job's own `if:` must carry the exact
        # tag-family conjunct too (distinct from the `needs.<gate>.result`
        # conjunct P3 already pinned) -- an `if:` naming the gate result but
        # no ref restriction at all would let the promotion run off any ref.
        texts = _positive_texts()
        texts["release-binaries.yml"] = _release_binaries_yml(
            cu12_if="always() && needs.gpu-proof.result == 'success'"
        )
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertTrue(
            any("F7 tag guard" in f and "refs/tags/v" in f for f in findings), findings
        )

    def test_wrong_tag_family_fails(self):
        # F7 audit fix: a py-v* row's promoting job carrying the WRONG
        # family's tag guard (v* instead of py-v*) must fail -- family is
        # per-row, never interchangeable.
        texts = _positive_texts()
        texts["pypi.yml"] = _simple_publish_yml(
            tag_pattern="py-v*",
            publish_if="always() && startsWith(github.ref, 'refs/tags/v') && needs.gpu-proof.result == 'success'",
            tag_family="py-v",
        )
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertTrue(any("F7 tag guard" in f for f in findings), findings)


class StepGatedTest(unittest.TestCase):
    """npm.yml's `publish` job always runs (build+test unconditional); the
    gate conjunct lives on its "Publish" STEP's own `if:` -- `PROMOTION_
    TABLE`'s `step_name` field."""

    def test_step_gated_positive_passes(self):
        findings = cgo.check_promotion_table(_positive_texts(), MANIFEST_GOOD)
        self.assertEqual(findings, [])

    def test_second_ungated_publishing_step_in_same_job_fails(self):
        # F4 audit fix: a step-gated row only pins the NAMED step's `if:` --
        # a SECOND step in the SAME job that itself invokes a publishing
        # primitive, with no `if:` of its own, used to sail through unseen.
        texts = _positive_texts()
        texts["npm.yml"] = _wf(
            "v*",
            _gate_job("gpu-proof")
            + "  publish:\n    needs: [gpu-proof]\n    if: always()\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - uses: actions/checkout@v4\n"
            "      - name: Publish\n"
            "        if: always() && startsWith(github.ref, 'refs/tags/v') && needs.gpu-proof.result == 'success'\n"
            "        run: npm publish --provenance --access public\n"
            "      - name: Sneak publish\n"
            "        run: npm publish --tag sneak\n",
        )
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertTrue(
            any("SECOND" in f and "Sneak publish" in f and "not the gated step" in f for f in findings),
            findings,
        )

    def test_missing_named_step_fails(self):
        texts = _positive_texts()
        texts["npm.yml"] = _wf(
            "v*",
            _gate_job("gpu-proof")
            + "  publish:\n    needs: [gpu-proof]\n    if: always()\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - uses: actions/checkout@v4\n      - name: Something Else\n        run: echo hi\n",
        )
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertTrue(any("step `Publish` does not exist" in f for f in findings), findings)

    def test_step_if_missing_gate_conjunct_fails(self):
        texts = _positive_texts()
        texts["npm.yml"] = _npm_yml(step_if="always() && startsWith(github.ref, 'refs/tags/v')")
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertTrue(any("no top-level conjunct" in f for f in findings), findings)

    def test_step_if_depth0_or_fails(self):
        texts = _positive_texts()
        texts["npm.yml"] = _npm_yml(
            step_if="github.event_name == 'push' || needs.gpu-proof.result == 'success'"
        )
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertTrue(any("depth-0 `||`" in f for f in findings), findings)

    def test_job_level_if_is_not_mistaken_for_the_step_if(self):
        # The job's OWN if: always() must never satisfy the gate conjunct
        # requirement -- only the named step's if: counts for a step_name row.
        texts = _positive_texts()
        texts["npm.yml"] = _wf(
            "v*",
            _gate_job("gpu-proof")
            + "  publish:\n    needs: [gpu-proof]\n"
            "    if: always() && needs.gpu-proof.result == 'success'\n"
            "    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n"
            "      - name: Publish\n        if: startsWith(github.ref, 'refs/tags/v')\n        run: echo publish\n",
        )
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertTrue(any("no top-level conjunct" in f for f in findings), findings)


class OnBlockDoctrineTest(unittest.TestCase):
    def test_push_trigger_on_prove_workflow_fails(self):
        bad = PROVE_YML_GOOD.replace(
            "on:\n  workflow_dispatch:", "on:\n  workflow_dispatch:\n  push:\n    tags: [\"v*\"]"
        )
        texts = _positive_texts()
        texts["gpu-prove.yml"] = bad
        findings = cgo.check_p1_p2(texts)
        self.assertTrue(any("push" in f and "on: block carries" in f for f in findings))

    def test_workflow_call_trigger_on_prove_workflow_fails(self):
        bad = PROVE_YML_GOOD.replace("on:\n  workflow_dispatch:", "on:\n  workflow_call:\n  workflow_dispatch:")
        texts = _positive_texts()
        texts["gpu-prove.yml"] = bad
        findings = cgo.check_p1_p2(texts)
        self.assertTrue(any("workflow_call" in f and "on: block carries" in f for f in findings))

    def test_quoted_on_block_is_unreadable_not_a_pass(self):
        bad = PROVE_YML_GOOD.replace("on:\n  workflow_dispatch:", '"on":\n  workflow_dispatch:')
        texts = _positive_texts()
        texts["gpu-prove.yml"] = bad
        findings = cgo.check_p1_p2(texts)
        self.assertTrue(any("cannot read" in f for f in findings))

    def test_flow_style_on_block_is_unreadable_not_a_pass(self):
        bad = PROVE_YML_GOOD.replace("on:\n  workflow_dispatch:\n  pull_request:\n    types: [labeled]\n  schedule:\n    - cron: \"47 3 * * *\"", "on: { workflow_dispatch: null }")
        texts = _positive_texts()
        texts["gpu-prove.yml"] = bad
        findings = cgo.check_p1_p2(texts)
        self.assertTrue(any("cannot read" in f for f in findings))

    def test_uses_local_reference_to_prove_workflow_fails(self):
        caller = "name: x\n\non:\n  workflow_dispatch:\n\njobs:\n  x:\n    uses: ./.github/workflows/gpu-prove.yml\n"
        texts = _positive_texts()
        texts["some-caller.yml"] = caller
        findings = cgo.check_p1_p2(texts)
        self.assertTrue(any("uses:` gpu-prove.yml" in f for f in findings))

    def test_uses_cross_repo_reference_to_prove_workflow_fails(self):
        caller = (
            "name: x\n\non:\n  workflow_dispatch:\n\njobs:\n  x:\n"
            "    uses: f-inverse/jammi-ai/.github/workflows/gpu-prove.yml@main\n"
        )
        texts = _positive_texts()
        texts["some-caller.yml"] = caller
        findings = cgo.check_p1_p2(texts)
        self.assertTrue(any("cross-repo reference" in f for f in findings))


class GateFileAbsentTest(unittest.TestCase):
    def test_gate_file_present_fails(self):
        with tempfile.TemporaryDirectory() as td:
            wf_dir, _ = write_tree(Path(td), positive_workflows(), MANIFEST_GOOD)
            (wf_dir / cgo.GATE_WORKFLOW).write_text("name: x\n")
            findings = cgo.check_gate_file_absent(wf_dir)
            self.assertTrue(any("must be deleted" in f for f in findings))

    def test_gate_file_absent_passes(self):
        with tempfile.TemporaryDirectory() as td:
            wf_dir, _ = write_tree(Path(td), positive_workflows(), MANIFEST_GOOD)
            self.assertEqual(cgo.check_gate_file_absent(wf_dir), [])


class P4NameArchAgreementTest(unittest.TestCase):
    def test_job_name_template_mismatch_fails(self):
        bad = PROVE_YML_GOOD.replace(JOB_NAME_LINE, "GPU prove for RunPod (${{ matrix.arch }})")
        findings = cgo.check_p4({"gpu-prove.yml": bad}, set(REAL_ARCHES))
        self.assertTrue(any("does not match" in f for f in findings))

    def test_arch_list_mismatch_fails(self):
        bad = PROVE_YML_GOOD.replace(f"arch: [{ARCH_LIST}]", "arch: [sm_80]")
        findings = cgo.check_p4({"gpu-prove.yml": bad}, set(REAL_ARCHES))
        self.assertTrue(any("!=" in f for f in findings))

    def test_positive_agreement_passes(self):
        findings = cgo.check_p4({"gpu-prove.yml": PROVE_YML_GOOD}, set(REAL_ARCHES))
        self.assertEqual(findings, [])


class SplitTopLevelTest(unittest.TestCase):
    def test_simple_and(self):
        tokens, ok = cgo.split_top_level("a && b && c")
        self.assertTrue(ok)
        self.assertEqual([t for t in tokens if t not in ("&&", "||")], ["a", "b", "c"])

    def test_or_inside_parens_is_not_top_level(self):
        tokens, ok = cgo.split_top_level("a && (b || c)")
        self.assertTrue(ok)
        self.assertNotIn("||", tokens)

    def test_or_inside_quotes_is_not_top_level(self):
        tokens, ok = cgo.split_top_level("contains('a||b', 'x') && needs.g.result == 'success'")
        self.assertTrue(ok)
        self.assertNotIn("||", tokens)

    def test_unbalanced_parens_reported(self):
        _, ok = cgo.split_top_level("a && (b")
        self.assertFalse(ok)

    def test_unterminated_string_reported(self):
        _, ok = cgo.split_top_level("a && 'unterminated")
        self.assertFalse(ok)

    def test_escaped_quote_inside_string(self):
        tokens, ok = cgo.split_top_level("contains('it''s', 'x') && b")
        self.assertTrue(ok)


class ProofRequiredConsultsVerdictTest(unittest.TestCase):
    """BLOCK B8 audit fix: P5 -- nothing pinned that
    `_gpu-proof-required.yml` actually CONSULTS the verdict; P3 only checks
    the gate job's `uses:` line, so gutting the reusable to `run: echo ok`
    left the gate green."""

    def test_real_file_passes(self):
        findings = cgo.check_p5({"_gpu-proof-required.yml": PROOF_REQUIRED_YML_GOOD})
        self.assertEqual(findings, [])

    def test_missing_reusable_fails(self):
        findings = cgo.check_p5({})
        self.assertTrue(any("is missing" in f for f in findings))

    def test_gutted_body_fails(self):
        gutted = (
            "name: _gpu-proof-required\n\non:\n  workflow_call: {}\n\n"
            "jobs:\n  proof-required:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - run: echo ok\n"
        )
        findings = cgo.check_p5({"_gpu-proof-required.yml": gutted})
        self.assertTrue(any("does not invoke" in f for f in findings))

    def test_literal_tag_sha_fails(self):
        bad = PROOF_REQUIRED_YML_GOOD.replace('--sha "$GITHUB_SHA"', "--sha v1.2.3")
        findings = cgo.check_p5({"_gpu-proof-required.yml": bad})
        self.assertTrue(any("not bound to" in f for f in findings), findings)

    def test_literal_hex_sha_fails(self):
        bad = PROOF_REQUIRED_YML_GOOD.replace('--sha "$GITHUB_SHA"', f"--sha {'a' * 40}")
        findings = cgo.check_p5({"_gpu-proof-required.yml": bad})
        self.assertTrue(any("not bound to" in f for f in findings), findings)

    def test_no_sha_argument_at_all_fails(self):
        bad = PROOF_REQUIRED_YML_GOOD.replace('--sha "$GITHUB_SHA"\n', "")
        findings = cgo.check_p5({"_gpu-proof-required.yml": bad})
        self.assertTrue(any("no --sha argument" in f for f in findings), findings)

    def test_workflow_call_expression_form_passes(self):
        good = PROOF_REQUIRED_YML_GOOD.replace('--sha "$GITHUB_SHA"', "--sha ${{ github.sha }}")
        findings = cgo.check_p5({"_gpu-proof-required.yml": good})
        self.assertEqual(findings, [])

    def test_not_workflow_call_only_fails(self):
        bad = PROOF_REQUIRED_YML_GOOD.replace(
            "on:\n  workflow_call: {}\n", "on:\n  workflow_call: {}\n  workflow_dispatch:\n"
        )
        findings = cgo.check_p5({"_gpu-proof-required.yml": bad})
        self.assertTrue(any("workflow_call`-only" in f for f in findings), findings)

    def test_no_repo_argument_at_all_fails(self):
        # F6 audit fix: --repo must be pinned to THIS repo too.
        bad = PROOF_REQUIRED_YML_GOOD.replace('--repo "$GITHUB_REPOSITORY" \\\n            ', "")
        findings = cgo.check_p5({"_gpu-proof-required.yml": bad})
        self.assertTrue(any("no --repo argument" in f for f in findings), findings)

    def test_literal_repo_argument_fails(self):
        bad = PROOF_REQUIRED_YML_GOOD.replace(
            '--repo "$GITHUB_REPOSITORY"', "--repo some-other-org/some-other-repo"
        )
        findings = cgo.check_p5({"_gpu-proof-required.yml": bad})
        self.assertTrue(any("not bound to `github.repository`" in f for f in findings), findings)

    def test_repo_expression_form_passes(self):
        good = PROOF_REQUIRED_YML_GOOD.replace('--repo "$GITHUB_REPOSITORY"', "--repo ${{ github.repository }}")
        findings = cgo.check_p5({"_gpu-proof-required.yml": good})
        self.assertEqual(findings, [])

    def test_workflow_override_to_something_else_fails(self):
        # F6 audit fix: a --workflow override may never name anything other
        # than gpu-prove.yml -- a pointed-elsewhere consumer could read a
        # DIFFERENT, unrelated workflow's runs as if they proved this one.
        bad = PROOF_REQUIRED_YML_GOOD.replace(
            '--sha "$GITHUB_SHA"', '--sha "$GITHUB_SHA" \\\n            --workflow some-other-workflow.yml'
        )
        findings = cgo.check_p5({"_gpu-proof-required.yml": bad})
        self.assertTrue(any("overrides --workflow" in f for f in findings), findings)

    def test_workflow_override_to_the_same_value_passes(self):
        good = PROOF_REQUIRED_YML_GOOD.replace(
            '--sha "$GITHUB_SHA"', '--sha "$GITHUB_SHA" \\\n            --workflow gpu-prove.yml'
        )
        findings = cgo.check_p5({"_gpu-proof-required.yml": good})
        self.assertEqual(findings, [])


def _proof_required_with_step(step_text: str) -> str:
    """A `_gpu-proof-required.yml`-shaped fixture whose SECOND step (the
    one that should invoke gpu_prove_verdict.py) is `step_text` verbatim --
    used to drive the seven round-2 F1 fail shapes through `check_p5`."""
    return (
        "name: _gpu-proof-required\n\non:\n  workflow_call: {}\n\n"
        "permissions:\n  contents: read\n  actions: read\n\n"
        "jobs:\n  proof-required:\n    name: GPU proof required\n"
        "    runs-on: ubuntu-latest\n    timeout-minutes: 15\n    steps:\n"
        "      - uses: actions/checkout@v4\n" + step_text
    )


class ProofRequiredMechanismEvasionTest(unittest.TestCase):
    """Round-2 adversarial audit (F1): P5 used to be a whole-file substring
    check plus a FIRST-`--sha`-match regex -- each of these seven shapes
    used to pass with zero findings while the job no longer really
    depended on the verdict. Every one must FAIL under the mechanism fix
    (parse the reusable's job -> steps; the invocation must be the actual
    `run:` command of some step, with no trailing control operator, no
    `continue-on-error:`/`if:` on that step or its job, and the LAST
    `--sha` on that command's line must be commit-bound)."""

    def test_invocation_named_only_in_step_name_fails(self):
        step = (
            "      - name: Check the commit's GPU-prove verdict via python3 ci/scripts/gpu_prove_verdict.py\n"
            "        run: echo ok\n"
        )
        findings = cgo.check_p5({"_gpu-proof-required.yml": _proof_required_with_step(step)})
        self.assertTrue(findings, findings)
        self.assertTrue(any("does not invoke" in f for f in findings), findings)

    def test_invocation_inside_quoted_echo_string_fails(self):
        step = (
            "      - name: Check the commit's GPU-prove verdict (gpu-prove.yml job conclusions at github.sha)\n"
            "        run: |\n"
            "          echo 'python3 ci/scripts/gpu_prove_verdict.py --sha \"$GITHUB_SHA\"'\n"
        )
        findings = cgo.check_p5({"_gpu-proof-required.yml": _proof_required_with_step(step)})
        self.assertTrue(findings, findings)
        self.assertTrue(any("does not invoke" in f for f in findings), findings)

    def test_real_invocation_followed_by_or_true_fails(self):
        step = (
            "      - name: Check the commit's GPU-prove verdict (gpu-prove.yml job conclusions at github.sha)\n"
            "        env:\n          GITHUB_TOKEN: ${{ github.token }}\n"
            "        run: |\n"
            "          python3 ci/scripts/gpu_prove_verdict.py \\\n"
            "            --repo \"$GITHUB_REPOSITORY\" \\\n"
            "            --sha \"$GITHUB_SHA\" || true\n"
        )
        findings = cgo.check_p5({"_gpu-proof-required.yml": _proof_required_with_step(step)})
        self.assertTrue(any("control operator" in f for f in findings), findings)

    def test_continue_on_error_step_fails(self):
        step = (
            "      - name: Check the commit's GPU-prove verdict (gpu-prove.yml job conclusions at github.sha)\n"
            "        continue-on-error: true\n"
            "        env:\n          GITHUB_TOKEN: ${{ github.token }}\n"
            "        run: |\n"
            "          python3 ci/scripts/gpu_prove_verdict.py \\\n"
            "            --repo \"$GITHUB_REPOSITORY\" \\\n"
            "            --sha \"$GITHUB_SHA\"\n"
        )
        findings = cgo.check_p5({"_gpu-proof-required.yml": _proof_required_with_step(step)})
        self.assertTrue(any("continue-on-error" in f for f in findings), findings)

    def test_if_false_step_fails(self):
        step = (
            "      - name: Check the commit's GPU-prove verdict (gpu-prove.yml job conclusions at github.sha)\n"
            "        if: false\n"
            "        env:\n          GITHUB_TOKEN: ${{ github.token }}\n"
            "        run: |\n"
            "          python3 ci/scripts/gpu_prove_verdict.py \\\n"
            "            --repo \"$GITHUB_REPOSITORY\" \\\n"
            "            --sha \"$GITHUB_SHA\"\n"
        )
        findings = cgo.check_p5({"_gpu-proof-required.yml": _proof_required_with_step(step)})
        self.assertTrue(any("if:" in f and "continue-on-error" in f for f in findings), findings)

    def test_second_trailing_sha_last_wins_fails(self):
        step = (
            "      - name: Check the commit's GPU-prove verdict (gpu-prove.yml job conclusions at github.sha)\n"
            "        env:\n          GITHUB_TOKEN: ${{ github.token }}\n"
            "        run: |\n"
            "          python3 ci/scripts/gpu_prove_verdict.py \\\n"
            "            --repo \"$GITHUB_REPOSITORY\" \\\n"
            "            --sha \"$GITHUB_SHA\" \\\n"
            "            --sha v1.2.3\n"
        )
        findings = cgo.check_p5({"_gpu-proof-required.yml": _proof_required_with_step(step)})
        self.assertTrue(any("not bound to" in f and "v1.2.3" in f for f in findings), findings)

    def test_sha_in_step_name_body_has_tag_fails(self):
        step = (
            "      - name: \"Check the commit's GPU-prove verdict --sha $GITHUB_SHA\"\n"
            "        env:\n          GITHUB_TOKEN: ${{ github.token }}\n"
            "        run: |\n"
            "          python3 ci/scripts/gpu_prove_verdict.py \\\n"
            "            --repo \"$GITHUB_REPOSITORY\" \\\n"
            "            --sha v1.2.3\n"
        )
        findings = cgo.check_p5({"_gpu-proof-required.yml": _proof_required_with_step(step)})
        self.assertTrue(any("not bound to" in f and "v1.2.3" in f for f in findings), findings)


class YamlExtensionTest(unittest.TestCase):
    """BLOCK B7 audit fix: GitHub Actions runs BOTH `.yml` and `.yaml`
    workflow files -- a `*.yml`-only glob is blind to a second producer, a
    `uses:` reference, or a resurrected renting reusable hiding under the
    `.yaml` spelling."""

    def test_yaml_second_producer_fails(self):
        second = PROVE_YML_GOOD.replace("name: GPU prove (RunPod)", "name: second-prover")
        with tempfile.TemporaryDirectory() as td:
            wf_dir, manifest_path = write_tree(
                Path(td), {**positive_workflows(), "second-prover.yaml": second}, MANIFEST_GOOD
            )
            findings = cgo.run_gate(wf_dir, manifest_path)
            self.assertTrue(any("second-prover.yaml" in f for f in findings), findings)

    def test_yaml_uses_of_the_prove_workflow_fails(self):
        caller = (
            "name: x\n\non:\n  workflow_dispatch:\n\njobs:\n  x:\n"
            "    uses: ./.github/workflows/gpu-prove.yaml\n"
        )
        with tempfile.TemporaryDirectory() as td:
            wf_dir, manifest_path = write_tree(
                Path(td), {**positive_workflows(), "some-caller.yml": caller}, MANIFEST_GOOD
            )
            findings = cgo.run_gate(wf_dir, manifest_path)
            self.assertTrue(any("gpu-prove.yaml" in f for f in findings), findings)

    def test_resurrected_yaml_gate_file_fails(self):
        with tempfile.TemporaryDirectory() as td:
            wf_dir, manifest_path = write_tree(
                Path(td), {**positive_workflows(), "_gpu-prove-gate.yaml": "name: x\n"}, MANIFEST_GOOD
            )
            findings = cgo.run_gate(wf_dir, manifest_path)
            self.assertTrue(any("_gpu-prove-gate.yaml" in f and "must be deleted" in f for f in findings), findings)

    def test_gpu_prove_yaml_named_caller_of_the_real_producer_fails(self):
        """F2 (round-2 adversarial audit): the `uses:` scan used to skip
        EVERY workflow whose file NAME matched a producer-name spelling
        (`gpu-prove.yml`/`gpu-prove.yaml`), not just the resolved producer
        itself -- so a sibling file literally named `gpu-prove.yaml` that
        `uses: ./.github/workflows/gpu-prove.yml` passed with zero
        findings, because its OWN name matched the skip set."""
        caller_named_like_the_producer = (
            "name: not-actually-the-prover\n\non:\n  workflow_dispatch:\n\n"
            "jobs:\n  x:\n    uses: ./.github/workflows/gpu-prove.yml\n"
        )
        with tempfile.TemporaryDirectory() as td:
            wf_dir, manifest_path = write_tree(
                Path(td),
                {**positive_workflows(), "gpu-prove.yaml": caller_named_like_the_producer},
                MANIFEST_GOOD,
            )
            findings = cgo.run_gate(wf_dir, manifest_path)
            self.assertTrue(
                any("gpu-prove.yaml" in f and "gpu-prove.yml" in f for f in findings), findings
            )


class NeedsMultilineFormTest(unittest.TestCase):
    """Advisory A7 fix: `[ \\t]*`, never `\\s*`, right after `needs:` -- the
    multi-line `needs:` list form (key alone on its line, `- item` entries
    below it) must PASS, not be misread as a single literal `- gate` name."""

    def test_multiline_needs_list_passes(self):
        raw_needs_block = "    needs:\n      - gpu-proof\n"
        texts = _positive_texts()
        texts["release-binaries.yml"] = _release_binaries_yml(raw_cu12_needs_block=raw_needs_block)
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertEqual(findings, [])


class P6DiscoveryTest(unittest.TestCase):
    """P6 (F2 audit fix): every workflow file is scanned, no trigger
    filtering at all -- a publishing-primitive-invoking job must be listed
    in `PROMOTION_TABLE` regardless of what triggers its own file. An
    unlisted one FAILS by name."""

    def test_real_tree_has_no_unlisted_promotion_job(self):
        findings = cgo.check_p6_discovery(cgo.load_workflow_texts(cgo.WORKFLOWS_DIR))
        self.assertEqual(findings, [])

    def test_unlisted_npm_publish_job_fails(self):
        rogue = (
            "name: rogue-npm\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak-publish:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - run: npm publish --provenance --access public\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue-npm-publisher.yml": rogue})
        self.assertTrue(
            any("rogue-npm-publisher.yml" in f and "sneak-publish" in f for f in findings), findings
        )

    def test_unlisted_gh_release_create_job_fails(self):
        rogue = (
            "name: rogue-release\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak-release:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - run: gh release create \"$TAG\" --generate-notes\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue-release.yml": rogue})
        self.assertTrue(any("sneak-release" in f for f in findings), findings)

    def test_docker_publish_with_push_false_is_not_a_promotion(self):
        # A build-only verification lane (push: "false") must never be
        # flagged as an unlisted promotion job.
        pr_lane = (
            "name: build-only\n\non:\n  push:\n    tags: [\"v*\"]\n  pull_request:\n\njobs:\n"
            "  build-only:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: ./.github/actions/docker-publish\n        with:\n          push: \"false\"\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "build-only.yml": pr_lane})
        self.assertEqual(findings, [], findings)

    def test_docker_publish_with_push_true_unlisted_fails(self):
        rogue = (
            "name: rogue-image\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak-image:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: ./.github/actions/docker-publish\n        with:\n          push: \"true\"\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue-image.yml": rogue})
        self.assertTrue(any("sneak-image" in f for f in findings), findings)

    def test_branches_only_workflow_with_a_publishing_primitive_is_still_discovered(self):
        # F2 audit fix: P6 used to skip any workflow whose `push:` sub-key
        # carried no `tags:` at all -- a `push: branches:`-only workflow
        # (the real-tree `image.yml`/`image-cuda.yml` shape) with an
        # UNLISTED publishing primitive used to sail through unseen. There
        # is no trigger filtering anymore: it must be discovered exactly
        # like a tag-triggered one.
        main_pusher = (
            "name: main-only\n\non:\n  push:\n    branches: [main]\n\njobs:\n"
            "  push-image:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: ./.github/actions/docker-publish\n        with:\n          push: \"true\"\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "main-only.yml": main_pusher})
        self.assertTrue(
            any("main-only.yml" in f and "push-image" in f for f in findings), findings
        )

    def test_listed_promoting_jobs_are_never_flagged(self):
        findings = cgo.check_p6_discovery(_positive_texts())
        self.assertEqual(findings, [], findings)


class PrimitivePatternShapesTest(unittest.TestCase):
    """F1 audit fix: PRIMITIVE_PATTERNS is a regex list over comment-
    stripped step bodies and `uses:` lines, whitespace-tolerant -- each
    shape gets its own unlisted-job FAIL fixture, not a grep for one known-
    bad string."""

    def _rogue(self, run_line: str, job_name: str = "sneak") -> dict[str, str]:
        rogue = (
            f"name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            f"  {job_name}:\n    runs-on: ubuntu-latest\n    steps:\n"
            f"      - run: {run_line}\n"
        )
        return {**_positive_texts(), "rogue.yml": rogue}

    def test_cargo_publish_two_spaces_unlisted_fails(self):
        findings = cgo.check_p6_discovery(self._rogue("cargo  publish --dry-run"))
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_twine_upload_unlisted_fails(self):
        findings = cgo.check_p6_discovery(self._rogue("twine upload dist/*"))
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_maturin_upload_unlisted_fails(self):
        findings = cgo.check_p6_discovery(self._rogue("maturin upload target/wheels/*"))
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_docker_push_shell_unlisted_fails(self):
        findings = cgo.check_p6_discovery(self._rogue("docker push ghcr.io/f-inverse/rogue:latest"))
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_gh_release_upload_unlisted_fails(self):
        findings = cgo.check_p6_discovery(self._rogue('gh release upload "$TAG" ./asset.bin'))
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_softprops_action_gh_release_unlisted_fails(self):
        rogue = (
            "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: softprops/action-gh-release@v2\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_bare_docker_build_push_action_unquoted_true_unlisted_fails(self):
        # No docker-publish composite in between -- a job that calls
        # docker/build-push-action DIRECTLY with an unquoted `push: true`.
        rogue = (
            "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: docker/build-push-action@v6\n        with:\n          push: true\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_bare_docker_build_push_action_push_false_is_not_a_promotion(self):
        clean = (
            "name: clean\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  build-only:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: docker/build-push-action@v6\n        with:\n          push: false\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "clean.yml": clean})
        self.assertEqual(findings, [], findings)

    def test_docker_build_push_action_expression_push_unlisted_fails(self):
        # A `${{ }}` expression push value MAY resolve to a push at runtime
        # -- never structurally exempt it just because it is not literally
        # `true`.
        rogue = (
            "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: docker/build-push-action@v6\n"
            "        with:\n          push: ${{ github.event_name != 'pull_request' }}\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_cross_repo_docker_publish_action_push_true_unlisted_fails(self):
        rogue = (
            "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: f-inverse/other-repo/.github/actions/docker-publish@main\n"
            "        with:\n          push: \"true\"\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_cross_repo_release_upload_unlisted_fails(self):
        rogue = (
            "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: f-inverse/other-repo/.github/actions/release-upload@main\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        self.assertTrue(any("sneak" in f for f in findings), findings)


class RecursiveLocalReusableDiscoveryTest(unittest.TestCase):
    """F1 audit fix: a job that merely `uses:` a LOCAL reusable workflow
    whose own jobs match a primitive is itself a promoting job too --
    `_ci-base-image.yml` pushes to GHCR; `image.yml`/`image-cuda.yml`'s
    `build` jobs (which each `uses:` it) must be discovered."""

    def test_real_tree_image_callers_are_tabled_not_double_counted(self):
        # positive_workflows() already includes _ci-base-image.yml,
        # image.yml, image-cuda.yml with the real (gated) shape and their
        # PROMOTION_TABLE rows -- confirms the recursion finds the caller,
        # never the reusable itself (which would be a bogus THIRD finding).
        findings = cgo.check_p6_discovery(_positive_texts())
        self.assertEqual(findings, [], findings)

    def test_new_untabled_local_reusable_caller_fails(self):
        texts = {**_positive_texts(), "rogue-caller.yml": _local_reusable_caller_yml(
            caller_job_name="sneak-build", target="_ci-base-image.yml"
        )}
        findings = cgo.check_p6_discovery(texts)
        self.assertTrue(
            any("rogue-caller.yml" in f and "sneak-build" in f for f in findings), findings
        )

    def test_reusable_only_workflow_itself_is_never_double_tabled(self):
        # _ci-base-image.yml's OWN `build-and-push` job (workflow_call-only
        # file) must never itself be required as a table row -- only its
        # caller's job is.
        findings = cgo.check_p6_discovery(_positive_texts())
        self.assertFalse(
            any("_ci-base-image.yml" in f and "build-and-push" in f for f in findings), findings
        )


class UnreadableOnOrJobsBlockFailsLoudTest(unittest.TestCase):
    """F2 audit fix: an unreadable `on:`/`jobs:` block is a FAIL LOUD, never
    a silent skip -- same doctrine P1 already holds `gpu-prove.yml`'s `on:`
    to, now applied across P6's full-tree scan."""

    def test_quoted_on_block_fails_loud(self):
        bad = '"on":\n  push:\n    tags: ["v*"]\n\njobs:\n  x:\n    runs-on: ubuntu-latest\n    steps:\n      - run: npm publish\n'
        findings = cgo.check_p6_discovery({**_positive_texts(), "bad.yml": "name: bad\n\n" + bad})
        self.assertTrue(any("bad.yml" in f and "cannot read" in f for f in findings), findings)

    def test_quoted_jobs_block_fails_loud(self):
        bad = (
            "name: bad\n\non:\n  push:\n    tags: [\"v*\"]\n\n"
            '"jobs":\n  x:\n    runs-on: ubuntu-latest\n    steps:\n      - run: npm publish\n'
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "bad.yml": bad})
        self.assertTrue(any("bad.yml" in f and "cannot read" in f for f in findings), findings)

    def test_flow_style_jobs_block_fails_loud(self):
        bad = "name: bad\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs: { x: { runs-on: ubuntu-latest } }\n"
        findings = cgo.check_p6_discovery({**_positive_texts(), "bad.yml": bad})
        self.assertTrue(any("bad.yml" in f and "flow-style" in f for f in findings), findings)

    def test_non_canonical_four_space_job_indent_fails_loud(self):
        bad = (
            "name: bad\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "    x:\n        runs-on: ubuntu-latest\n        steps:\n          - run: npm publish\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "bad.yml": bad})
        self.assertTrue(
            any("bad.yml" in f and "non-canonical indentation" in f for f in findings), findings
        )

    def test_on_with_trailing_comment_is_correctly_read_not_a_false_fail(self):
        # `on:  # comment` reads identically to a bare `on:` -- correctly
        # parsed, not flagged as unreadable, and its publishing primitive is
        # still discovered.
        ok = (
            "name: ok\n\non:  # release tags\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - run: npm publish --provenance --access public\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "ok.yml": ok})
        self.assertTrue(any("ok.yml" in f and "sneak" in f for f in findings), findings)
        self.assertFalse(any("cannot read" in f for f in findings), findings)

    def test_jobs_with_trailing_comment_is_correctly_read_not_a_false_fail(self):
        ok = (
            "name: ok\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:  # the jobs\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - run: npm publish --provenance --access public\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "ok.yml": ok})
        self.assertTrue(any("ok.yml" in f and "sneak" in f for f in findings), findings)
        self.assertFalse(any("non-canonical" in f or "cannot read" in f for f in findings), findings)


if __name__ == "__main__":
    unittest.main()
