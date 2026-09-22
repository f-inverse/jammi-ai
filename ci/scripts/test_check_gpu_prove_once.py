#!/usr/bin/env python3
"""Tests for `check_gpu_prove_once.py`: every release publisher, not only the
CUDA lanes, promotes a proof the prove lane recorded once and never rents a GPU
itself.

Drives the real `run_gate()`/`check_p1_p2()`/`check_promotion_table()`/
`check_p4()`/`check_p5()`/`check_p6_discovery()`/`check_promoting_if()`/
`reconstruct_if_expr()`/`split_top_level()`/`read_top_level_on_block()`/
`jobs_or_fail()`/`job_invokes_publish_primitive()`
entry points against synthetic fixture trees (never a hand-built stand-in
for the parsers themselves) — including a fixture where three publishers
`uses:` a renting reusable, which must fail naming every offending site, and a
positive fixture (covering every `PROMOTION_TABLE` row: the CUDA lanes, the CI base-image callers,
crates.io, npm, and every PyPI dist) that must pass clean.

Run directly: `python3 ci/scripts/test_check_gpu_prove_once.py`
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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



def _paid_lane_yml(name: str, job: str, script: str, label: str) -> str:
    """A minimal, VALID paid-pod-lane workflow (P7): label/dispatch-only
    triggers and exactly one step body invoking its own driver script."""
    return f"""\
name: {name}

on:
  workflow_dispatch:
  pull_request:
    types: [labeled]

permissions:
  contents: read

jobs:
  {job}:
    name: {name}
    if: github.event_name != 'pull_request' || github.event.label.name == '{label}'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Rent a pod and run this lane
        run: |
          bash ci/scripts/{script}
"""


GANG_YML_GOOD = _paid_lane_yml("GPU gang (RunPod)", "gpu-gang", "runpod_gpu_gang.sh", "run-gang")
HOWWELL_YML_GOOD = _paid_lane_yml(
    "GPU how-well (RunPod)", "gpu-howwell", "runpod_gpu_howwell.sh", "run-howwell"
)
CLUSTER_YML_GOOD = _paid_lane_yml(
    "GPU cluster (RunPod)", "gpu-cluster", "runpod_gpu_cluster.sh", "run-cluster"
)


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
    `github.ref_type != 'tag'` conjunct."""
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
    by its own `if:` -- drives the recursive-discovery mechanism (a job that merely delegates to a local reusable which itself pushes is
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
  merge-manifest:
    needs: [build-and-push]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker buildx imagetools create -t ghcr.io/f-inverse/jammi-ai-ci:latest a b
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
    merge_tag_if: str | None = None,
    merge_main_if: str = "needs.build-and-push-main.result == 'success' && github.ref_type != 'tag'",
) -> str:
    jobs = (
        _gate_job("gpu-proof")
        + _promoting_job("build-and-push-cu12", if_expr=cu12_if)
        + _promoting_job("build-and-push", if_expr=cpu_tag_if)
        + _ungated_job("build-and-push-main", main_if)
        + _ungated_job("build-and-push-selfcontained", selfcontained_if)
        # The two-arch CPU merge jobs -- `merge-cpu-tag` chains off
        # `build-and-push` (itself direct-gated), `merge-cpu-main` is a
        # second "none" row, same shape as `build-and-push-main` above.
        + _promoting_job("merge-cpu-tag", gate_name="build-and-push", if_expr=merge_tag_if)
        + _ungated_job("merge-cpu-main", merge_main_if)
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
        # P7's other PAID_POD_LANE_TABLE rows — the positive fixture must
        # carry every row, or "no P7 findings" would be vacuous (the
        # anti-vacuity leg in PaidPodLaneTest asserts exactly that).
        "gpu-gang.yml": GANG_YML_GOOD,
        "gpu-howwell.yml": HOWWELL_YML_GOOD,
        "gpu-cluster.yml": CLUSTER_YML_GOOD,
        # gpu-dev.sh's own row (whole-file scope gave it one: gpu-reap.yml
        # is its only invoker and carries RUNPOD_API_KEY at step scope).
        "gpu-reap.yml": REAP_YML,
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

    def test_main_exits_zero_on_the_real_tree(self):
        """`load_script_texts()` -- P7's own real-filesystem `git ls-files`
        read -- is exercised exactly once through the actual entry point
        `main()` calls (`run_gate()` with no `script_texts` override), never
        only through a fixture-injected map. Every other P7 test in this
        suite drives `check_p7_paid_pod_lanes` with an explicit
        `script_texts`, which never touches this function at all."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cgo.main()
        self.assertEqual(rc, 0, buf.getvalue())


class PaidPodLaneTest(unittest.TestCase):
    """P7: P1's doctrine over EVERY row of `PAID_POD_LANE_TABLE`, not only
    the prove lane. Every case below is driven against the gang row (the
    newest, priciest lane — 1 pod x 2 GPU), because a rule that only ever
    bit on the row it was written for is the escape P7 exists to close."""

    def _texts(self, **overrides: str) -> dict[str, str]:
        texts = _positive_texts()
        texts.update(overrides)
        return texts

    def test_positive_fixture_has_no_paid_lane_findings(self):
        self.assertEqual(cgo.check_p7_paid_pod_lanes(_positive_texts()), [])

    def test_every_table_row_is_exercised_by_the_positive_fixture(self):
        # Anti-vacuity: the positive fixture must actually CONTAIN every row
        # of the table, or "no findings" above would be free.
        texts = _positive_texts()
        for script, workflow in cgo.PAID_POD_LANE_TABLE.items():
            self.assertIn(workflow, texts, f"{workflow} missing from the positive fixture")
            self.assertIn(script, cgo.drop_comment_lines(texts[workflow]))

    def test_push_trigger_on_the_gang_workflow_fails(self):
        broken = GANG_YML_GOOD.replace(
            "on:\n  workflow_dispatch:", "on:\n  push:\n    branches: [main]\n  workflow_dispatch:"
        )
        findings = cgo.check_p7_paid_pod_lanes(self._texts(**{"gpu-gang.yml": broken}))
        joined = "\n".join(findings)
        self.assertIn("gpu-gang.yml's on: block carries ['push']", joined)

    def test_workflow_call_trigger_on_the_gang_workflow_fails(self):
        broken = GANG_YML_GOOD.replace(
            "on:\n  workflow_dispatch:", "on:\n  workflow_call:\n  workflow_dispatch:"
        )
        findings = cgo.check_p7_paid_pod_lanes(self._texts(**{"gpu-gang.yml": broken}))
        self.assertIn("workflow_call", "\n".join(findings))

    def test_quoted_on_block_is_read_exactly_like_bare_on(self):
        # A quoted "on": key is valid, unambiguous YAML (GitHub's own
        # documented way to avoid the YAML 1.1 boolean-resolution gotcha)
        # -- it must read IDENTICALLY to the bare form, never be refused:
        # the same fixture with the SAME triggers passes clean either way.
        broken = GANG_YML_GOOD.replace("\non:\n", '\n"on":\n')
        findings = cgo.check_p7_paid_pod_lanes(self._texts(**{"gpu-gang.yml": broken}))
        self.assertEqual(findings, [], findings)

    def test_quoted_push_key_still_fails_p7(self):
        # An unquoted-only CHILD key regex (`^  ([A-Za-z0-9_]+):`) drops a
        # quoted `"push":` trigger from the returned key list, and P7 never
        # sees it. The shared reader quote-normalizes child keys
        # (`push:`/`"push":`/`'push':` are the same key), so this must fail
        # exactly like the unquoted form in `test_push_trigger_on_the_gang_
        # workflow_fails` above.
        broken = GANG_YML_GOOD.replace(
            "on:\n  workflow_dispatch:", 'on:\n  "push":\n    branches: [main]\n  workflow_dispatch:'
        )
        findings = cgo.check_p7_paid_pod_lanes(self._texts(**{"gpu-gang.yml": broken}))
        joined = "\n".join(findings)
        self.assertIn("gpu-gang.yml's on: block carries ['push']", joined)

    def test_quoted_inline_on_push_fails_p7(self):
        # The inline-value arm strips flanking quotes off a single trigger
        # key, so `on: "push"` reads as the key `push`, the same bare
        # string P7 compares against -- an unnormalized `'"push"'` key
        # equals no bare trigger name and lets this exact shape evade
        # every check that gates on a specific trigger.
        broken = GANG_YML_GOOD.replace(
            "on:\n  workflow_dispatch:\n  pull_request:\n    types: [labeled]\n", 'on: "push"\n'
        )
        findings = cgo.check_p7_paid_pod_lanes(self._texts(**{"gpu-gang.yml": broken}))
        joined = "\n".join(findings)
        self.assertIn("gpu-gang.yml's on: block carries ['push']", joined)

    def test_merge_key_with_an_undefined_alias_is_refused_not_dropped(self):
        # A genuine YAML merge key (`<<: *anchors`) is refused loud through
        # the shared loader: GitHub Actions' own parser does not accept a
        # YAML alias in a workflow file at all (a merge key is always
        # defined via one), so this is a named refusal by construction --
        # never silently skipped, and never dependent on whether the
        # referenced anchor happens to be defined anywhere in the document.
        broken = GANG_YML_GOOD.replace(
            "on:\n  workflow_dispatch:", "on:\n  <<: *anchors\n  workflow_dispatch:"
        )
        findings = cgo.check_p7_paid_pod_lanes(self._texts(**{"gpu-gang.yml": broken}))
        self.assertIn("not accepted by GitHub Actions", "\n".join(findings))

    def test_two_invokers_of_the_gang_driver_fail(self):
        second = GANG_YML_GOOD.replace("name: GPU gang (RunPod)", "name: second-gang-renter")
        findings = cgo.check_p7_paid_pod_lanes(self._texts(**{"second-gang-renter.yml": second}))
        joined = "\n".join(findings)
        self.assertIn("more than one workflow", joined)
        self.assertIn("second-gang-renter.yml", joined)
        # The `extra site(s):` list's CONTENT, not just that the extra
        # workflow's name appears somewhere in the joined findings: it names
        # exactly the non-row invoker, never the row's own workflow.
        self.assertIn("extra site(s): ['second-gang-renter.yml']", joined)
        self.assertNotIn("extra site(s): ['gpu-gang.yml'", joined)

    def test_one_WRONG_invoker_is_a_failure_that_says_so(self):
        """Still a FAIL — and the finding describes the state it found: a
        single wrong invoker is never reported as 'more than one workflow'."""
        only_other = GANG_YML_GOOD.replace("name: GPU gang (RunPod)", "name: second-gang-renter")
        no_invoke = GANG_YML_GOOD.replace("bash ci/scripts/runpod_gpu_gang.sh", "echo nothing")
        findings = cgo.check_p7_paid_pod_lanes(
            self._texts(**{"gpu-gang.yml": no_invoke, "second-gang-renter.yml": only_other})
        )
        joined = "\n".join(findings)
        self.assertIn("none of which is gpu-gang.yml", joined)
        self.assertIn("second-gang-renter.yml", joined)
        self.assertNotIn("more than one workflow", joined)

    def test_zero_invokers_fails(self):
        broken = GANG_YML_GOOD.replace("bash ci/scripts/runpod_gpu_gang.sh", "echo nothing")
        findings = cgo.check_p7_paid_pod_lanes(self._texts(**{"gpu-gang.yml": broken}))
        self.assertIn("zero workflows invoke ci/scripts/runpod_gpu_gang.sh", "\n".join(findings))

    def test_missing_workflow_file_fails(self):
        texts = _positive_texts()
        del texts["gpu-gang.yml"]
        findings = cgo.check_p7_paid_pod_lanes(texts)
        joined = "\n".join(findings)
        self.assertIn("gpu-gang.yml is missing from the workflow tree", joined)

    def test_a_publisher_that_uses_the_gang_lane_fails(self):
        caller = (
            "name: publisher\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  gang:\n    uses: ./.github/workflows/gpu-gang.yml\n"
        )
        findings = cgo.check_p7_paid_pod_lanes(self._texts(**{"a-publisher.yml": caller}))
        self.assertIn("nothing may call a paid pod lane", "\n".join(findings))

    def test_cross_repo_uses_of_the_gang_lane_fails(self):
        caller = (
            "name: publisher\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  gang:\n    uses: f-inverse/jammi-ai/.github/workflows/gpu-gang.yml@main\n"
        )
        findings = cgo.check_p7_paid_pod_lanes(self._texts(**{"a-publisher.yml": caller}))
        self.assertIn("cross-repo reference", "\n".join(findings))

    def test_cross_repo_uses_of_an_unrelated_workflow_is_clean(self):
        """Negative fixture: a cross-repo `uses:` naming a DIFFERENT
        workflow entirely must not trip the gang row's own `uses:` scan --
        the pattern match is on the row's own workflow name, not on any
        cross-repo reference at all."""
        caller = (
            "name: publisher\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  other:\n    uses: some-org/some-repo/.github/workflows/totally-unrelated.yml@main\n"
        )
        findings = cgo.check_p7_paid_pod_lanes(self._texts(**{"a-publisher.yml": caller}))
        self.assertFalse(any("cross-repo reference" in f for f in findings), findings)

    def test_the_rule_bites_on_every_row_not_only_the_gang_one(self):
        # One `push:` trigger per row, each independently caught by name.
        for script, workflow in cgo.PAID_POD_LANE_TABLE.items():
            with self.subTest(lane=workflow):
                texts = _positive_texts()
                texts[workflow] = texts[workflow].replace(
                    "on:\n  workflow_dispatch:", "on:\n  push:\n    branches: [main]\n  workflow_dispatch:", 1
                )
                findings = cgo.check_p7_paid_pod_lanes(texts)
                self.assertIn(f"{workflow}'s on: block carries ['push']", "\n".join(findings), script)


class RpSshoRequiresRpInitTest(unittest.TestCase):
    """Every real
    `PAID_POD_LANE_TABLE` driver that references `runpod_lib.sh`'s own
    `RP_SSHO` array must call `rp_init` -- a static scan over the table's
    REAL drivers on disk, not a paraphrase. `RP_SSHO` is populated ONLY by
    `rp_init` (`-i "$RP_SSH_KEY"`/`IdentitiesOnly=yes`, among other
    options); a driver that reads it without ever calling `rp_init` runs
    every ssh/scp/rsync call with an EMPTY option array — a silent,
    wrong-key failure that reads exactly like 'not yet reachable'."""

    RP_INIT_CALL_RE = re.compile(r"(?m)^[ \t]*rp_init[ \t]*(?:#.*)?$")

    def test_every_real_rp_ssho_referencing_driver_calls_rp_init(self) -> None:
        offenders = []
        for script in cgo.PAID_POD_LANE_TABLE:
            text = (cgo.REPO_ROOT / script).read_text(encoding="utf-8")
            if "RP_SSHO[" not in text:
                continue  # this driver never reads the array at all -- nothing to bind.
            if not self.RP_INIT_CALL_RE.search(text):
                offenders.append(script)
        self.assertEqual(
            offenders,
            [],
            f"driver(s) reference RP_SSHO but never call rp_init: {offenders} -- every ssh/scp/rsync "
            "call in that driver would run with an EMPTY RP_SSHO array",
        )

    def test_a_driver_reading_rp_ssho_without_rp_init_is_caught_red_then_green(self) -> None:
        # RED: a fixture shaped exactly like the class this scan closes.
        bad_text = 'echo "${RP_SSHO[@]}"\nssh "${RP_SSHO[@]}" -p "$PORT" "root@$HOST" true\n'
        self.assertIn("RP_SSHO[", bad_text)
        self.assertIsNone(self.RP_INIT_CALL_RE.search(bad_text))
        # GREEN: rp_init called before the array is ever read.
        good_text = 'rp_init\necho "${RP_SSHO[@]}"\n'
        self.assertIsNotNone(self.RP_INIT_CALL_RE.search(good_text))
        # A mention of rp_init inside PROSE (a comment) is never mistaken
        # for a call -- the regex anchors the whole (stripped) line.
        prose_only = '# see rp_init for details\necho "${RP_SSHO[@]}"\n'
        self.assertIsNone(self.RP_INIT_CALL_RE.search(prose_only))


class DropCommentLinesTrailingCommentTest(unittest.TestCase):
    """`drop_comment_lines` also strips a TRAILING
    `# ...` comment, not only a full comment line -- a token that occurs
    only after a trailing `#` is prose, never code evidence."""

    def test_a_token_only_in_a_trailing_comment_does_not_resolve(self) -> None:
        text = 'echo hello  # mentions rp_cluster_create only in this comment\n'
        stripped = cgo.drop_comment_lines(text)
        self.assertNotIn("rp_cluster_create", stripped)
        self.assertIn("echo hello", stripped)

    def test_a_token_in_real_code_before_a_trailing_comment_still_resolves(self) -> None:
        text = "rp_cluster_create  # the real call, with a trailing comment\n"
        stripped = cgo.drop_comment_lines(text)
        self.assertIn("rp_cluster_create", stripped)

    def test_a_hash_inside_a_quoted_string_is_not_treated_as_a_comment(self) -> None:
        text = 'url="https://example.com/x#rp_cluster_create"\n'
        stripped = cgo.drop_comment_lines(text)
        self.assertIn("rp_cluster_create", stripped)

    def test_line_count_is_preserved_never_shifted(self) -> None:
        text = "a\nb  # c\nd\n"
        self.assertEqual(len(cgo.drop_comment_lines(text).splitlines()), len(text.splitlines()))

    def test_full_line_comments_are_still_blanked_as_before(self) -> None:
        text = "# a whole-line comment naming rp_cluster_create\nreal code\n"
        stripped = cgo.drop_comment_lines(text)
        self.assertNotIn("rp_cluster_create", stripped)
        self.assertIn("real code", stripped)

    def test_a_backslash_escaped_apostrophe_inside_a_single_quoted_string_is_not_a_toggle(self) -> None:
        # bash's own `'\''` idiom for embedding a literal apostrophe inside
        # a single-quoted string is THREE quote characters but only TWO
        # real quote-state toggles (close, then reopen) -- the middle one
        # is a backslash-escaped LITERAL character. A parser that toggles
        # on all three ends up believing it is still inside a string, and
        # fails to strip a REAL trailing comment that follows.
        text = "echo 'it'\\''s done'  # mentions rp_cluster_create only here\n"
        stripped = cgo.drop_comment_lines(text)
        self.assertNotIn("rp_cluster_create", stripped)
        self.assertIn("echo 'it'\\''s done'", stripped)


class P7UsesReadFromTheParsedDocumentTest(unittest.TestCase):
    """P7's 'nothing may call a paid pod lane' rule reads a job-level
    `uses:` (local or cross-repo) from the parsed document, never a text
    regex -- mirrors `UsesReadFromTheParsedDocumentTest` (P6) and
    `P1UsesReadFromTheParsedDocumentTest` one-for-one, driven against the
    gang row per this module's own convention. Each case below is a
    quoting/`+`-truncation/unexaminable-sibling shape a text regex would
    miss, GREEN once `uses:` is read from the parsed document via
    `_scan_uses_references`."""

    def _texts(self, **overrides: str) -> dict[str, str]:
        texts = _positive_texts()
        texts.update(overrides)
        return texts

    def test_single_quoted_local_uses_of_the_gang_lane_fails(self):
        caller = (
            "name: publisher\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  gang:\n    uses: './.github/workflows/gpu-gang.yml'\n"
        )
        findings = cgo.check_p7_paid_pod_lanes(self._texts(**{"a-publisher.yml": caller}))
        self.assertIn("nothing may call a paid pod lane", "\n".join(findings))

    def test_double_quoted_local_uses_of_the_gang_lane_fails(self):
        caller = (
            'name: publisher\n\non:\n  push:\n    tags: ["v*"]\n\njobs:\n'
            '  gang:\n    uses: "./.github/workflows/gpu-gang.yml"\n'
        )
        findings = cgo.check_p7_paid_pod_lanes(self._texts(**{"a-publisher.yml": caller}))
        self.assertIn("nothing may call a paid pod lane", "\n".join(findings))

    def test_quoted_cross_repo_pinned_ref_to_the_gang_lane_fails(self):
        caller = (
            "name: publisher\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            '  gang:\n    uses: "f-inverse/jammi-ai/.github/workflows/gpu-gang.yml@a1b2c3d4"\n'
        )
        findings = cgo.check_p7_paid_pod_lanes(self._texts(**{"a-publisher.yml": caller}))
        self.assertIn("cross-repo reference", "\n".join(findings))

    def test_plus_bearing_lane_name_is_never_truncated(self):
        # A character class like ([A-Za-z0-9_.-]+) stops at `+` -- it reads a
        # lane workflow named `pub+lish.yml` as `pub`, which never equals the
        # real target and so never matches.
        producer = _paid_lane_yml("pub+lish", "publish-job", "fake_plus.sh", "run-pub-plus")
        caller = (
            "name: some-caller\n\non:\n  push:\n    branches: [main]\n\n"
            "jobs:\n  x:\n    uses: ./.github/workflows/pub+lish.yml\n"
        )
        texts = self._texts(**{"pub+lish.yml": producer, "some-caller.yml": caller})
        with mock.patch.dict(cgo.PAID_POD_LANE_TABLE, {"ci/scripts/fake_plus.sh": "pub+lish.yml"}):
            findings = cgo.check_p7_paid_pod_lanes(texts)
        mine = [f for f in findings if "some-caller.yml" in f]
        self.assertGreaterEqual(len(mine), 1, findings)
        self.assertTrue(any("uses:` pub+lish.yml" in f for f in mine), mine)

    def test_unexaminable_other_workflow_is_a_finding_never_a_silent_skip(self):
        # A genuine YAML anchor makes the WHOLE document unexaminable
        # (`_assert_no_github_incompatible_yaml` refuses it outright,
        # regardless of whether the anchor is ever referenced) --
        # `_parsed_jobs_or_fail` fails here where flow-style alone would
        # not (flow-style is still valid, constructible YAML; only
        # `job_source_spans`'s own LINE-SPAN reader refuses it, and this
        # rule does not use that reader for `uses:` discovery).
        unexaminable = (
            "name: bad\n\non: &trig\n  push:\n    branches: [main]\n\n"
            "jobs:\n  x:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo hi\n"
        )
        findings = cgo.check_p7_paid_pod_lanes(self._texts(**{"bad.yml": unexaminable}))
        mine = [f for f in findings if "bad.yml" in f]
        self.assertGreaterEqual(len(mine), 1, findings)
        self.assertTrue(any("cannot examine" in f for f in mine), mine)

    def test_unexaminable_other_workflow_is_reported_once_not_once_per_row(self):
        # Reported ONCE across the whole scan, never once per
        # PAID_POD_LANE_TABLE row it happens to be compared against.
        # A genuine YAML anchor makes the WHOLE document unexaminable
        # (`_assert_no_github_incompatible_yaml` refuses it outright,
        # regardless of whether the anchor is ever referenced) --
        # `_parsed_jobs_or_fail` fails here where flow-style alone would
        # not (flow-style is still valid, constructible YAML; only
        # `job_source_spans`'s own LINE-SPAN reader refuses it, and this
        # rule does not use that reader for `uses:` discovery).
        unexaminable = (
            "name: bad\n\non: &trig\n  push:\n    branches: [main]\n\n"
            "jobs:\n  x:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo hi\n"
        )
        findings = cgo.check_p7_paid_pod_lanes(self._texts(**{"bad.yml": unexaminable}))
        mine = [f for f in findings if "bad.yml" in f and "cannot examine" in f]
        self.assertEqual(len(mine), 1, mine)


# --------------------------------------------------------------------------- #
# P7's DERIVED subject set (the deploy closure) — fixtures.
#
# Every case below drives the derivation through its PARAMETERS, so no case
# needs a file written into the real tree: `check_p7_paid_pod_lanes` takes
# the tracked-script map and the library text, the same way the workflow
# scan already takes `workflow_texts`.
# --------------------------------------------------------------------------- #
FIXTURE_LIB = """\
#!/usr/bin/env bash
rp_init() { : "${RUNPOD_API_KEY:?}"; }
# Deliberately payload-free. `test_pod_substrate.sh`'s own `(ab/gpuCount D5)`
# leg asserts a CLOSED set of tracked files naming the per-pod GPU-count
# variable (the library, the gang driver and that suite) with a plain grep
# that does not strip comments, so neither the fixture below NOR this comment
# may spell that variable out — either would read as "a second lane started
# moving its own gpuCount". Nothing here needs the real payload text: the
# closure derivation reads CALL structure, not payload contents.
_rp_deploy_payload() { # $1=cloudType $2=gpuTypeId
  echo "{}"
}
rp_deploy_live() {
  local body
  body="$(_rp_deploy_payload "$1" "$2")"
  echo "$body"
}
rp_deploy_arch() { # $1=arch
  rp_deploy_live "SECURE|NVIDIA A100 80GB PCIe"
}
rp_deploy_live_a100() { rp_deploy_arch a100; }
rp_sweep() { echo sweeping; }
# The SECOND renting root -- REST v2's cluster create entrypoint, with
# no internal caller of its own (real runpod_lib.sh shape: nothing else in
# the library calls it; every real caller is an external driver). Deliberately
# payload-free, same reason _rp_deploy_payload above is.
rp_cluster_create() { # $1=gpuTypeId $2=optional dataCenterIds
  echo "{}"
}
# The THIRD renting root -- the pods transport's own
# REST v2 `POST /v2/pods` entrypoint, with no internal caller of its own
# (real runpod_lib.sh shape, same as rp_cluster_create above). Deliberately
# payload-free.
rp_two_host_pod_create() { # $1=gpuTypeId $2=dataCenterId $3=rank
  echo "{}"
}
"""

# A one-line wrapper added to the library: the RED case for "the closure is
# COMPUTED, never a literal name list".
FIXTURE_LIB_WITH_WRAPPER = FIXTURE_LIB + "rp_deploy_h100() { rp_deploy_arch h100; }\n"


def _driver(call: str) -> str:
    return (
        "#!/usr/bin/env bash\n"
        "# a comment naming rp_deploy_live_a100 must NOT count as a call\n"
        'source "$DIR/runpod_lib.sh"\n'
        "rp_init\n"
        f"{call}\n"
    )


def fixture_scripts() -> dict[str, str]:
    """The tracked `ci/scripts/**` map the derivation ranges over: the four
    PAID_POD_LANE_TABLE drivers plus the two non-table deploy-capable
    scripts this tree really has (`gpu-dev.sh`, `test_pod_substrate.sh`)."""
    return {
        cgo.RUNPOD_LIB_REL: FIXTURE_LIB,
        "ci/scripts/runpod_gpu_prove.sh": _driver("rp_deploy_arch a100"),
        "ci/scripts/runpod_gpu_gang.sh": _driver("rp_deploy_arch a100"),
        "ci/scripts/runpod_gpu_howwell.sh": _driver("rp_deploy_live_a100"),
        "ci/scripts/gpu-dev.sh": _driver("rp_deploy_arch \"$ARCH\""),
        "ci/scripts/test_pod_substrate.sh": _driver("rp_deploy_live \"SECURE|X\""),
        # The cluster leg's own row -- calls BOTH the second root
        # (`rp_cluster_create`, the `cluster` transport) and the third
        # (`rp_two_host_pod_create`, the `pods` transport) directly (there
        # is no wrapper the way `_rp_deploy_payload` has
        # `rp_deploy_arch`/`rp_deploy_live`).
        "ci/scripts/runpod_gpu_cluster.sh": _driver(
            'rp_cluster_create "NVIDIA A100-SXM4-80GB"\n'
            'rp_two_host_pod_create "NVIDIA A100-SXM4-80GB" "dc-a" 0'
        ),
        # A tracked script that calls NOTHING in the closure: the derivation
        # must not sweep the whole directory in.
        "ci/scripts/check_something.py": "print('no deploy here')\n",
    }


REAP_YML = """\
name: GPU reap

on:
  workflow_dispatch:
  schedule:
    - cron: "0 * * * *"

jobs:
  reap:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Reap
        env:
          RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}
        run: bash ci/scripts/gpu-dev.sh reap ${{ inputs.force_hours }}
"""

GUARD_YML = """\
name: CI

on:
  pull_request:

jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: bash ci/scripts/test_pod_substrate.sh
"""


def _derived_texts(**overrides: str) -> dict[str, str]:
    texts = _positive_texts()
    texts["gpu-reap.yml"] = REAP_YML
    texts["ci.yml"] = GUARD_YML
    texts.update(overrides)
    return texts


class DerivedRentingDriverTest(unittest.TestCase):
    """P7's subject set is DERIVED from `runpod_lib.sh`'s own deploy
    closure; `PAID_POD_LANE_TABLE` is a completeness assertion over it. A
    rule that iterated the TABLE only would silently admit a fifth renting
    driver, a new deploy wrapper, and a row whose driver stopped renting."""

    def test_closure_is_computed_from_the_library(self):
        closure, findings = cgo.derive_deploy_closure(FIXTURE_LIB)
        self.assertEqual(findings, [])
        # The closure is each RENTING_ROOTS entry's transitive callers
        # PLUS the roots themselves -- `rp_cluster_create`/`rp_two_host_
        # pod_create` have no caller of their own in this fixture (matching
        # the real library), so each contributes only itself.
        self.assertEqual(
            set(closure),
            {
                "_rp_deploy_payload",
                "rp_cluster_create",
                "rp_two_host_pod_create",
                "rp_deploy_live",
                "rp_deploy_arch",
                "rp_deploy_live_a100",
            },
        )

    def test_a_new_deploy_wrapper_joins_the_closure_without_a_gate_edit(self):
        closure, findings = cgo.derive_deploy_closure(FIXTURE_LIB_WITH_WRAPPER)
        self.assertEqual(findings, [])
        self.assertIn("rp_deploy_h100", closure)

    def test_a_library_with_no_payload_builder_fails_closed(self):
        _closure, findings = cgo.derive_deploy_closure("#!/usr/bin/env bash\nrp_init() { :; }\n")
        self.assertTrue(any("cannot derive the renting closure" in f for f in findings), findings)
        # BOTH roots are missing here -- each gets its OWN named finding
        # (the closure fails closed per missing root, never with a single
        # collapsed message that only names one of the two).
        joined = "\n".join(findings)
        self.assertIn("_rp_deploy_payload", joined)
        self.assertIn("rp_cluster_create", joined)

    def test_a_library_missing_only_the_cluster_root_fails_closed(self):
        """The symmetric case: `_rp_deploy_payload` present, `rp_cluster_
        create` absent -- the SECOND root's own absence is caught
        independently, never masked by the first root's presence."""
        _closure, findings = cgo.derive_deploy_closure(
            "#!/usr/bin/env bash\n_rp_deploy_payload() { echo '{}'; }\n"
            "rp_deploy_live() { _rp_deploy_payload; }\nrp_init() { :; }\n"
        )
        self.assertTrue(
            any("cannot derive the renting closure" in f and "rp_cluster_create" in f for f in findings),
            findings,
        )

    def test_a_library_nothing_calls_the_payload_builder_from_fails_closed(self):
        _closure, findings = cgo.derive_deploy_closure(
            "#!/usr/bin/env bash\n_rp_deploy_payload() { echo '{}'; }\n"
            "rp_cluster_create() { echo '{}'; }\n"
            "rp_two_host_pod_create() { echo '{}'; }\nrp_init() { :; }\n"
        )
        self.assertTrue(any("carries no CALLERS" in f for f in findings), findings)

    def test_a_missing_library_in_the_script_map_fails_closed(self):
        scripts = fixture_scripts()
        del scripts[cgo.RUNPOD_LIB_REL]
        findings = cgo.check_p7_paid_pod_lanes(_derived_texts(), scripts)
        self.assertTrue(
            any("is not in the scanned ci/scripts set" in f for f in findings), findings
        )

    def test_derived_drivers_are_exactly_the_deploy_callers(self):
        closure, _ = cgo.derive_deploy_closure(FIXTURE_LIB)
        derived = cgo.derive_renting_drivers(fixture_scripts(), closure)
        self.assertEqual(
            sorted(derived),
            [
                "ci/scripts/gpu-dev.sh",
                "ci/scripts/runpod_gpu_cluster.sh",
                "ci/scripts/runpod_gpu_gang.sh",
                "ci/scripts/runpod_gpu_howwell.sh",
                "ci/scripts/runpod_gpu_prove.sh",
                "ci/scripts/test_pod_substrate.sh",
            ],
        )
        # The library itself DEFINES the closure; it is not a lane.
        self.assertNotIn(cgo.RUNPOD_LIB_REL, derived)
        # A comment naming a closure member is not a call.
        self.assertNotIn("rp_deploy_live_a100", derived["ci/scripts/gpu-dev.sh"])

    def test_the_positive_derived_fixture_is_clean(self):
        self.assertEqual(
            cgo.check_p7_paid_pod_lanes(_derived_texts(), fixture_scripts()), []
        )

    def test_a_fifth_renting_driver_with_no_table_row_fails(self):
        scripts = fixture_scripts()
        scripts["ci/scripts/runpod_gpu_new.sh"] = _driver("rp_deploy_arch a100")
        new_lane = (
            "name: new lane\n\non:\n  push:\n    branches: [main]\n  workflow_call:\n"
            "  workflow_dispatch:\n\njobs:\n  rent:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - name: Rent\n        env:\n"
            "          RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}\n"
            "        run: bash ci/scripts/runpod_gpu_new.sh\n"
        )
        findings = cgo.check_p7_paid_pod_lanes(
            _derived_texts(**{"new-lane.yml": new_lane, "new-lane-2.yml": new_lane}), scripts
        )
        joined = "\n".join(findings)
        self.assertIn("ci/scripts/runpod_gpu_new.sh", joined)
        self.assertIn("that workflow can RENT", joined)
        self.assertIn("new-lane.yml", joined)
        self.assertIn("new-lane-2.yml", joined)

    def test_a_driver_calling_the_cluster_root_directly_demands_a_row(self):
        """Both directions (part 1): a driver that calls `rp_cluster_
        create` DIRECTLY -- no wrapper needed, unlike `_rp_deploy_payload`'s
        `rp_deploy_arch`/`rp_deploy_live` -- is derived as a renting driver
        and demands a row exactly like any `_rp_deploy_payload` caller
        does, once it is MENTIONED in a secret-carrying workflow (below);
        one that is derived but mentioned nowhere draws only a note, never a
        finding (part 2, `test_the_positive_derived_fixture_is_clean`
        above -- no real PAID_POD_LANE_TABLE row names a driver of the
        second root on this tree)."""
        scripts = fixture_scripts()
        scripts["ci/scripts/runpod_gpu_new_cluster.sh"] = _driver(
            'rp_cluster_create "NVIDIA A100-SXM4-80GB"'
        )
        new_lane = (
            "name: new cluster lane\n\non:\n  workflow_dispatch:\n\njobs:\n  rent:\n"
            "    runs-on: ubuntu-latest\n    steps:\n      - name: Rent\n        env:\n"
            "          RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}\n"
            "        run: bash ci/scripts/runpod_gpu_new_cluster.sh\n"
        )
        findings = cgo.check_p7_paid_pod_lanes(
            _derived_texts(**{"new-cluster-lane.yml": new_lane}), scripts
        )
        joined = "\n".join(findings)
        self.assertIn("ci/scripts/runpod_gpu_new_cluster.sh", joined)
        self.assertIn("that workflow can RENT", joined)

    def test_a_driver_calling_a_NEW_deploy_wrapper_is_caught(self):
        # The whole point of computing the closure: the wrapper did not
        # exist when this rule was written, and no name list mentions it.
        scripts = fixture_scripts()
        scripts[cgo.RUNPOD_LIB_REL] = FIXTURE_LIB_WITH_WRAPPER
        scripts["ci/scripts/runpod_gpu_h100.sh"] = _driver("rp_deploy_h100")
        lane = (
            "name: h100 lane\n\non:\n  workflow_dispatch:\n\njobs:\n  rent:\n"
            "    runs-on: ubuntu-latest\n    steps:\n      - name: Rent\n        env:\n"
            "          RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}\n"
            "        run: bash ci/scripts/runpod_gpu_h100.sh\n"
        )
        findings = cgo.check_p7_paid_pod_lanes(
            _derived_texts(**{"h100.yml": lane}), scripts
        )
        self.assertIn("ci/scripts/runpod_gpu_h100.sh", "\n".join(findings))

    def test_a_table_row_whose_driver_stopped_renting_is_reported_as_rot(self):
        scripts = fixture_scripts()
        scripts["ci/scripts/runpod_gpu_gang.sh"] = "#!/usr/bin/env bash\necho 'no longer rents'\n"
        findings = cgo.check_p7_paid_pod_lanes(_derived_texts(), scripts)
        joined = "\n".join(findings)
        self.assertIn("PAID_POD_LANE_TABLE row `ci/scripts/runpod_gpu_gang.sh`", joined)
        self.assertIn("does NOT call", joined)

    def test_verb_parsing_is_gone_the_mention_and_the_secret_are_all_that_matter(self):
        """No verb list clears a driver and the secret is not read at JOB
        scope: a fresh, non-table derived driver invoked with a non-renting-
        sounding verb (`reap`) still gets a finding the moment its workflow
        carries the secret anywhere — no invocation-site parsing decides
        clearance."""
        scripts = fixture_scripts()
        scripts["ci/scripts/runpod_gpu_new.sh"] = _driver("rp_deploy_arch a100")
        lane = (
            "name: new-reap-verb-lane\n\non:\n  workflow_dispatch:\n\njobs:\n  reap:\n"
            "    runs-on: ubuntu-latest\n    steps:\n      - name: Reap\n        env:\n"
            "          RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}\n"
            "        run: bash ci/scripts/runpod_gpu_new.sh reap\n"
        )
        findings = cgo.check_p7_paid_pod_lanes(
            _derived_texts(**{"new-reap-verb-lane.yml": lane}), scripts
        )
        joined = "\n".join(findings)
        self.assertIn("ci/scripts/runpod_gpu_new.sh", joined)
        self.assertIn("that workflow can RENT", joined)

    def test_a_paths_only_mention_in_a_secret_free_workflow_is_neither_finding_nor_note(self):
        """The NOTE channel fires on ABSENCE, never on "not invoked". A
        derived driver named only by a `paths:` filter entry, in a workflow
        that carries no RUNPOD_API_KEY anywhere, is mentioned somewhere (so
        the absence NOTE does not fire) and that workflow cannot make it
        rent either (so there is nothing to condemn): the module doc and
        `_check_derived_driver_cannot_rent`'s own docstring state this
        silence as the rule's own outcome for this shape, not a gap."""
        scripts = fixture_scripts()
        scripts["ci/scripts/runpod_gpu_pathsonly.sh"] = _driver("rp_deploy_arch a100")
        paths_only = (
            "name: paths-only\n\non:\n  push:\n    paths:\n"
            "      - ci/scripts/runpod_gpu_pathsonly.sh\n\njobs:\n  g:\n"
            "    runs-on: ubuntu-latest\n    steps:\n      - run: echo hi\n"
        )
        notes: list[str] = []
        findings = cgo.check_p7_paid_pod_lanes(
            _derived_texts(**{"paths-only.yml": paths_only}), scripts, notes=notes
        )
        self.assertFalse(any("pathsonly" in f for f in findings), findings)
        self.assertFalse(any("pathsonly" in n for n in notes), notes)

    def test_a_same_basename_driver_in_a_different_directory_is_not_dropped(self):
        """Identity is the repo-relative PATH, never a basename. A prior
        revision keyed the completeness loop's own bookkeeping dict on
        `rel.rsplit("/", 1)[-1]`, so two distinct DERIVED drivers sharing one
        basename in different directories collapsed into a single dict
        entry and one of them was silently never checked at all — a renting
        `ci/scripts/perf/runpod_gpu_evil.sh` beside a renting flat
        `ci/scripts/runpod_gpu_evil.sh`, each invoked from its OWN
        secret-holding workflow, produced only ONE finding instead of two.
        Both must be caught independently."""
        scripts = fixture_scripts()
        scripts["ci/scripts/runpod_gpu_evil.sh"] = _driver("rp_deploy_arch a100")
        scripts["ci/scripts/perf/runpod_gpu_evil.sh"] = _driver("rp_deploy_arch a100")
        flat_lane = (
            "name: flat evil\n\non:\n  workflow_dispatch:\n\njobs:\n  rent:\n"
            "    runs-on: ubuntu-latest\n    steps:\n      - name: Rent\n        env:\n"
            "          RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}\n"
            "        run: bash ci/scripts/runpod_gpu_evil.sh\n"
        )
        perf_lane = (
            "name: perf evil\n\non:\n  workflow_dispatch:\n\njobs:\n  rent:\n"
            "    runs-on: ubuntu-latest\n    steps:\n      - name: Rent\n        env:\n"
            "          RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}\n"
            "        run: bash ci/scripts/perf/runpod_gpu_evil.sh\n"
        )
        findings = cgo.check_p7_paid_pod_lanes(
            _derived_texts(**{"flat-evil.yml": flat_lane, "perf-evil.yml": perf_lane}), scripts
        )
        joined = "\n".join(findings)
        self.assertIn("ci/scripts/runpod_gpu_evil.sh", joined)
        self.assertIn("ci/scripts/perf/runpod_gpu_evil.sh", joined)
        # Both are independent findings, not one message doing double duty.
        self.assertEqual(
            sum(1 for f in findings if "runpod_gpu_evil.sh" in f), 2, findings
        )

    def test_no_rsplit_anywhere_in_the_module(self):
        """The identity discipline this file's P7 section commits to
        (`ci/scripts/check_gpu_prove_once.py`'s own module doc): keying or
        re-keying any of this rule's identity comparisons on a basename via
        `.rsplit("/", ...)` is exactly the bug class `perf/`-collision
        fixture above reproduces. Checked on the actual AST, never a plain
        substring search, so this file's own PROSE describing the discipline
        (which necessarily spells `rsplit("/")` out to explain what must
        never appear) cannot trip its own assertion."""
        import ast

        src = Path(cgo.__file__).read_text()
        tree = ast.parse(src)
        rsplit_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "rsplit"
        ]
        self.assertEqual(rsplit_calls, [], [ast.dump(n) for n in rsplit_calls])

    def test_prove_script_identity_is_a_repo_relative_path(self):
        self.assertTrue(cgo.PROVE_SCRIPT.startswith("ci/scripts/"), cgo.PROVE_SCRIPT)
        for key in cgo.PAID_POD_LANE_TABLE:
            self.assertTrue(key.startswith("ci/scripts/"), key)

    def test_unresolved_row_with_another_invoker_names_the_row_not_resolving(self):
        """`:1100`'s `resolved_workflow is None` arm INSIDE the `producers`
        branch: a row's own workflow file is absent from the tree, but a
        DIFFERENT workflow's comment-stripped text still names the driver.
        `test_missing_workflow_file_fails` deletes the row's only invoker
        too, so it only ever reaches the `not producers` arm and the OUTER
        `resolved_workflow is None` arm (`gpu-gang.yml is missing`) -- never
        this one."""
        texts = _positive_texts()
        other = GANG_YML_GOOD.replace("name: GPU gang (RunPod)", "name: other-gang-invoker")
        del texts["gpu-gang.yml"]
        texts["other-gang-invoker.yml"] = other
        findings = cgo.check_p7_paid_pod_lanes(texts)
        joined = "\n".join(findings)
        self.assertIn(
            "invoked by ['other-gang-invoker.yml'], but its PAID_POD_LANE_TABLE row names "
            "gpu-gang.yml, which does not resolve",
            joined,
        )
        self.assertIn("gpu-gang.yml is missing from the workflow tree", joined)
        self.assertNotIn("zero workflows invoke", joined)

    def test_the_seed_loss_fail_propagates_through_the_gate_entry_point(self):
        """`:1061`'s `findings += closure_findings`: renaming
        `_rp_deploy_payload` in `runpod_lib.sh` is asserted at
        `derive_deploy_closure`'s own unit level elsewhere in this suite
        (`test_a_library_with_no_payload_builder_fails_closed`) but never
        through `check_p7_paid_pod_lanes` itself -- the actual gate entry
        point every other P7 fixture in this class drives through."""
        scripts = fixture_scripts()
        scripts[cgo.RUNPOD_LIB_REL] = FIXTURE_LIB.replace(
            "_rp_deploy_payload()", "_rp_deploy_payload_RENAMED()"
        )
        findings = cgo.check_p7_paid_pod_lanes(_derived_texts(), scripts)
        self.assertTrue(
            any("cannot derive the renting closure" in f and "_rp_deploy_payload" in f for f in findings),
            findings,
        )

    def test_the_secret_is_the_capability(self):
        # test_pod_substrate.sh is deploy-capable and IS invoked by the
        # guard job — clean only because that job passes no RUNPOD_API_KEY.
        self.assertFalse(
            any("test_pod_substrate.sh" in f for f in cgo.check_p7_paid_pod_lanes(
                _derived_texts(), fixture_scripts()
            ))
        )
        with_secret = GUARD_YML.replace(
            "    steps:\n      - uses: actions/checkout@v4\n",
            "    env:\n      RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}\n"
            "    steps:\n      - uses: actions/checkout@v4\n",
        )
        findings = cgo.check_p7_paid_pod_lanes(
            _derived_texts(**{"ci.yml": with_secret}), fixture_scripts()
        )
        self.assertIn("ci/scripts/test_pod_substrate.sh", "\n".join(findings))

    def test_a_workflow_level_env_secret_reaches_the_step(self):
        """The secret is read at EVERY scope of the invoking workflow. A
        top-level `env:` block is inherited by every job and every step in
        the file, so a deploy-capable driver invoked from any job in it can
        rent — reading only the job body (this rule's first revision) let
        that whole shape through, and 10+ workflows in this tree declare
        their env at the top level."""
        top_level_env = GUARD_YML.replace(
            "\njobs:\n",
            "\nenv:\n  RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}\n\njobs:\n",
        )
        self.assertIn("env:\n  RUNPOD_API_KEY", top_level_env)  # the fixture really moved it out
        self.assertNotIn("RUNPOD_API_KEY", top_level_env.split("jobs:", 1)[1])
        findings = cgo.check_p7_paid_pod_lanes(
            _derived_texts(**{"ci.yml": top_level_env}), fixture_scripts()
        )
        joined = "\n".join(findings)
        self.assertIn("ci/scripts/test_pod_substrate.sh", joined)
        self.assertIn("that workflow can RENT", joined)

    def test_a_with_block_secret_reaches_the_step(self):
        """Same capability, a different spelling site: a reusable-workflow
        `with:`/`secrets:` mapping is not a job `env:` block either. The
        `with:` mapping sits in a SEPARATE job from the one invoking the
        driver -- under whole-file scope there is no job attribution left to
        read at all (a prior revision scoped the secret per-job/per-step;
        that scoping is gone), so this fixture is the one that reds if a
        future revision ever reintroduces job-level attribution by mistake."""
        via_with = GUARD_YML.replace(
            "\njobs:\n  guard:\n",
            "\njobs:\n  other-job:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: ./.github/workflows/_x.yml\n"
            "        with:\n"
            "          RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}\n"
            "  guard:\n",
        )
        self.assertIn("other-job", via_with)
        self.assertNotIn("with:", GUARD_YML)  # the fixture really added a new job, not edited in place
        findings = cgo.check_p7_paid_pod_lanes(
            _derived_texts(**{"ci.yml": via_with}), fixture_scripts()
        )
        self.assertIn("ci/scripts/test_pod_substrate.sh", "\n".join(findings))

    def test_a_commented_out_secret_is_not_the_capability(self):
        """Whole-FILE scope, still comment-stripped: a commented secret is
        text, not a capability, so the widened scope does not turn every
        workflow that documents the variable into a paid lane."""
        commented = GUARD_YML.replace(
            "\njobs:\n",
            "\n# env:\n#   RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}\n\njobs:\n",
        )
        findings = cgo.check_p7_paid_pod_lanes(
            _derived_texts(**{"ci.yml": commented}), fixture_scripts()
        )
        self.assertFalse(any("test_pod_substrate.sh" in f for f in findings), findings)

    def test_a_driver_mentioned_in_no_workflow_at_all_is_reported_as_a_note(self):
        """The NOTE channel fires on ABSENCE: a derived driver whose path is
        mentioned in NO workflow file at all is REPORTED, never silently
        credited. It is a NOTE, not a finding: nothing in the committed text
        says it can rent, and nothing says it cannot."""
        scripts = fixture_scripts()
        scripts["ci/scripts/runpod_gpu_orphan.sh"] = _driver("rp_deploy_arch a100")
        notes: list[str] = []
        findings = cgo.check_p7_paid_pod_lanes(
            _derived_texts(), scripts, notes=notes
        )
        self.assertFalse(any("runpod_gpu_orphan.sh" in f for f in findings), findings)
        joined = "\n".join(notes)
        self.assertIn("ci/scripts/runpod_gpu_orphan.sh", joined)
        self.assertIn("mentioned in no workflow file in this tree", joined)
        self.assertNotIn("invoked", joined)
        # A driver MENTIONED somewhere (even if never truly invoked) produces
        # no note -- see the paths:-only fixture above for that shape.
        self.assertFalse(any("test_pod_substrate.sh" in n for n in notes), notes)

    def test_the_real_tree_derives_the_set_this_suite_claims(self):
        """Anti-vacuity, stated as the SET the property ranged over: the
        real `runpod_lib.sh` closure and the real derived-driver set, so a
        fixture that drifts from the tree is a failure here, not a silent
        loss of coverage."""
        scripts = cgo.load_script_texts()
        closure, findings = cgo.derive_deploy_closure(scripts[cgo.RUNPOD_LIB_REL])
        self.assertEqual(findings, [])
        # All three RENTING_ROOTS are members of the closure (a root is
        # a member of its own matched set), alongside `_rp_deploy_payload`'s
        # own real transitive callers.
        self.assertEqual(
            set(closure),
            {
                "_rp_deploy_payload",
                "rp_cluster_create",
                "rp_two_host_pod_create",
                "rp_deploy_live",
                "rp_deploy_arch",
                "rp_deploy_live_a100",
            },
        )
        derived = cgo.derive_renting_drivers(scripts, closure)
        self.assertEqual(
            sorted(derived),
            [
                # THIS file (see below) sorts before gpu-dev.sh.
                "ci/scripts/check_gpu_prove_once.py",
                "ci/scripts/gpu-dev.sh",
                "ci/scripts/runpod_gpu_cluster.sh",
                "ci/scripts/runpod_gpu_gang.sh",
                "ci/scripts/runpod_gpu_howwell.sh",
                "ci/scripts/runpod_gpu_prove.sh",
                # THIS file: its fixtures above spell the closure members
                # out in non-comment text, so the deliberately
                # over-approximating scan derives it too (see the gate's own
                # "WHAT THE DERIVATION DELIBERATELY DOES NOT DO"). It is
                # cleared by the machine predicate like any other derived
                # driver — ci.yml's guard job invokes it with no
                # RUNPOD_API_KEY — never by an exemption, which is the
                # point: nothing here gets a pass for being ours. Its own
                # docstrings and RENTING_ROOTS-mirroring constants are what
                # make `ci/scripts/check_gpu_prove_once.py` (above) derive
                # the same way, for the same reason.
                "ci/scripts/test_check_gpu_prove_once.py",
                # test_gpu_cluster_lane.sh's own EXIT-trap fixtures
                # source runpod_gpu_cluster.sh in a real
                # subshell and override `rp_cluster_delete`/`rp_cleanup` by
                # name, in non-comment text -- the deliberately
                # over-approximating scan derives it too. Cleared the
                # identical way: ci.yml's guard job invokes it with no
                # RUNPOD_API_KEY anywhere.
                "ci/scripts/test_gpu_cluster_lane.sh",
                "ci/scripts/test_pod_substrate.sh",
                # test_runpod_cluster_lib.sh calls `_rp_deploy_payload`
                # directly (Group 2 -- comparing the pod and cluster
                # entrypoint text byte-for-byte) AND `rp_cluster_create`
                # directly (Group 7); cleared the identical way, by ci.yml
                # carrying no RUNPOD_API_KEY anywhere.
                "ci/scripts/test_runpod_cluster_lib.sh",
            ],
        )
        # ... and the real tree is clean over exactly that derived set.
        self.assertEqual(cgo.run_gate(), [])


class ScheduleVisibilityTest(unittest.TestCase):
    """P8: a `schedule:` trigger on a paid pod lane (or on any OTHER
    workflow that mentions a RENTING_ROOTS-derived driver while carrying
    the secret) is a FINDING unless the workflow is a reviewed
    PAID_LANE_CRON_ALLOWLIST entry whose token resolves."""

    def test_real_tree_schedule_visibility_is_clean(self):
        self.assertEqual(
            cgo.check_p8_schedule_visibility(cgo.load_workflow_texts(cgo.WORKFLOWS_DIR)), []
        )

    def test_red_then_green_a_planted_cron_on_a_gang_shaped_workflow(self):
        """RED->GREEN: P7 alone (schedule-blind) does not refuse a planted
        cron on the gang lane; P8 does. The SAME fixture, two different
        checks, proves P8 -- not P7 -- is what catches this."""
        planted = GANG_YML_GOOD.replace(
            "on:\n  workflow_dispatch:\n  pull_request:\n    types: [labeled]\n",
            "on:\n  workflow_dispatch:\n  pull_request:\n    types: [labeled]\n"
            "  schedule:\n    - cron: \"0 5 * * *\"\n",
        )
        self.assertIn("schedule:", planted)  # the fixture really carries a cron
        texts = _positive_texts()
        texts["gpu-gang.yml"] = planted

        # RED absent (P7 alone): no finding names the planted schedule.
        p7_only = cgo.check_p7_paid_pod_lanes(texts)
        self.assertFalse(any("schedule" in f for f in p7_only), p7_only)

        # GREEN present (P8): the planted cron is a named finding.
        p8_findings = cgo.check_p8_schedule_visibility(texts)
        joined = "\n".join(p8_findings)
        self.assertIn("gpu-gang.yml", joined)
        self.assertIn("schedule", joined)
        self.assertIn("PAID_LANE_CRON_ALLOWLIST", joined)

    def test_the_allow_listed_prove_cron_is_clean(self):
        findings = cgo.check_p8_schedule_visibility(_positive_texts())
        self.assertFalse(any("gpu-prove.yml" in f for f in findings), findings)

    def test_a_second_cron_entry_on_an_allow_listed_lane_fails(self):
        """The allow-list review covers exactly ONE
        reviewed cadence; a SECOND `- cron:` under the same schedule: key is
        un-reviewed paid-lane exposure the token match alone cannot see."""
        two_crons = PROVE_YML_GOOD.replace(
            '  schedule:\n    - cron: "47 3 * * *"\n',
            '  schedule:\n    - cron: "47 3 * * *"\n    - cron: "0 0 * * *"\n',
        )
        texts = _positive_texts()
        texts["gpu-prove.yml"] = two_crons
        findings = cgo.check_p8_schedule_visibility(texts)
        joined = "\n".join(findings)
        self.assertIn("gpu-prove.yml", joined)
        self.assertIn("2 `- cron:` entries", joined)
        self.assertIn("un-reviewed paid-lane exposure", joined)

    def test_a_single_cron_entry_on_an_allow_listed_lane_stays_clean(self):
        # Control: the ordinary, single-cadence shape must not trip the
        # new check on its own.
        findings = cgo.check_p8_schedule_visibility(_positive_texts())
        self.assertFalse(any("cron:` entries" in f for f in findings), findings)

    def test_an_unresolvable_allow_list_token_fails(self):
        allow = dict(cgo.PAID_LANE_CRON_ALLOWLIST)
        allow["gpu-prove.yml"] = ("this-token-appears-nowhere-in-the-tree", "bogus")
        with mock.patch.object(cgo, "PAID_LANE_CRON_ALLOWLIST", allow):
            findings = cgo.check_p8_schedule_visibility(_positive_texts())
        joined = "\n".join(findings)
        self.assertIn("gpu-prove.yml", joined)
        self.assertIn("this-token-appears-nowhere-in-the-tree", joined)
        self.assertIn("neither its own", joined)
        self.assertIn("dead waiver", joined)

    def test_a_listed_workflow_with_no_cron_is_a_dead_waiver(self):
        texts = _positive_texts()
        no_cron = GANG_YML_GOOD  # carries no schedule: at all
        texts["gpu-gang.yml"] = no_cron
        allow = dict(cgo.PAID_LANE_CRON_ALLOWLIST)
        allow["gpu-gang.yml"] = ("run-gang", "bogus allow-list row for a lane with no cron")
        with mock.patch.object(cgo, "PAID_LANE_CRON_ALLOWLIST", allow):
            findings = cgo.check_p8_schedule_visibility(texts)
        joined = "\n".join(findings)
        self.assertIn("gpu-gang.yml", joined)
        self.assertIn("dead waiver", joined)

    def test_an_allow_list_entry_naming_a_nonexistent_workflow_fails(self):
        allow = dict(cgo.PAID_LANE_CRON_ALLOWLIST)
        allow["no-such-workflow.yml"] = ("whatever", "bogus")
        with mock.patch.object(cgo, "PAID_LANE_CRON_ALLOWLIST", allow):
            findings = cgo.check_p8_schedule_visibility(_positive_texts())
        joined = "\n".join(findings)
        self.assertIn("no-such-workflow.yml", joined)
        self.assertIn("does not exist", joined)

    def test_an_unreadable_on_block_fails_never_a_silent_skip(self):
        texts = _positive_texts()
        texts["gpu-gang.yml"] = GANG_YML_GOOD.replace("\non:\n", '\n"on":\n on:\n')  # duplicate key
        findings = cgo.check_p8_schedule_visibility(texts)
        self.assertTrue(any("gpu-gang.yml" in f for f in findings), findings)

    def test_a_derived_driver_with_a_schedule_and_the_secret_is_caught_even_off_table(self):
        """P8's subject set extends beyond PAID_POD_LANE_TABLE: a derived
        (RENTING_ROOTS-calling) driver mentioned in a secret-holding
        workflow is a subject even before it ever gets a table row."""
        scripts = fixture_scripts()
        scripts["ci/scripts/runpod_gpu_new.sh"] = _driver("rp_deploy_arch a100")
        cron_lane = (
            "name: new lane\n\non:\n  workflow_dispatch:\n  schedule:\n"
            "    - cron: \"0 6 * * *\"\n\njobs:\n  rent:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - name: Rent\n        env:\n"
            "          RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}\n"
            "        run: bash ci/scripts/runpod_gpu_new.sh\n"
        )
        findings = cgo.check_p8_schedule_visibility(
            _derived_texts(**{"new-lane.yml": cron_lane}), scripts
        )
        joined = "\n".join(findings)
        self.assertIn("new-lane.yml", joined)
        self.assertIn("PAID_LANE_CRON_ALLOWLIST", joined)


class PublishersCallingARentingReusableTest(unittest.TestCase):
    """Three publishers `uses:` a renting `_gpu-prove-gate.yml` which itself invokes
    `runpod_gpu_prove.sh` -- must FAIL naming all three publisher sites."""

    def test_three_renting_callers_fail_naming_all_three_publishers(self):
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
        # `github.ref_type != 'tag'` conjunct.
        texts = _positive_texts()
        texts["server-image.yml"] = _server_image_yml(main_if="startsWith(github.ref, 'refs/tags/v')")
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertTrue(
            any("gate_kind='none'" in f and "build-and-push-main" in f for f in findings), findings
        )

    def test_none_row_real_leak_shape_no_ref_restriction_at_all_fails(self):
        # A `selfcontained_if` gated only on the dispatch input, with no ref
        # restriction whatsoever, must FAIL: a `workflow_dispatch` against a
        # `v*` tag ref with `selfcontained=true` would push this image
        # entirely ungated.
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
        # The GATE job's own `if:` must also carry the row's
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
            any("gate job `gpu-proof`" in f and "tag guard" in f for f in findings), findings
        )

    def test_promoting_job_missing_tag_guard_fails(self):
        # The PROMOTING job's own `if:` must carry the exact
        # tag-family conjunct too (distinct from the `needs.<gate>.result`
        # conjunct P3 already pinned) -- an `if:` naming the gate result but
        # no ref restriction at all would let the promotion run off any ref.
        texts = _positive_texts()
        texts["release-binaries.yml"] = _release_binaries_yml(
            cu12_if="always() && needs.gpu-proof.result == 'success'"
        )
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertTrue(
            any("tag guard" in f and "refs/tags/v" in f for f in findings), findings
        )

    def test_wrong_tag_family_fails(self):
        # A py-v* row's promoting job carrying the WRONG
        # family's tag guard (v* instead of py-v*) must fail -- family is
        # per-row, never interchangeable.
        texts = _positive_texts()
        texts["pypi.yml"] = _simple_publish_yml(
            tag_pattern="py-v*",
            publish_if="always() && startsWith(github.ref, 'refs/tags/v') && needs.gpu-proof.result == 'success'",
            tag_family="py-v",
        )
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertTrue(any("tag guard" in f for f in findings), findings)


class StepGatedTest(unittest.TestCase):
    """npm.yml's `publish` job always runs (build+test unconditional); the
    gate conjunct lives on its "Publish" STEP's own `if:` -- `PROMOTION_
    TABLE`'s `step_name` field."""

    def test_step_gated_positive_passes(self):
        findings = cgo.check_promotion_table(_positive_texts(), MANIFEST_GOOD)
        self.assertEqual(findings, [])

    def test_second_ungated_publishing_step_in_same_job_fails(self):
        # A step-gated row only pins the NAMED step's `if:` -- a SECOND step
        # in the SAME job that itself invokes a publishing primitive, with no
        # `if:` of its own, must still be caught.
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

    def test_second_step_quoted_docker_publish_uses_is_caught(self):
        # `_other_publishing_steps` reads every step's `uses:` from the
        # PARSED document (the same reader P6 uses), never a hand-rolled
        # text-range scan, which a quoted `uses:` on the second (ungated)
        # step would escape entirely.
        texts = _positive_texts()
        texts["npm.yml"] = _wf(
            "v*",
            _gate_job("gpu-proof")
            + "  publish:\n    needs: [gpu-proof]\n    if: always()\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - uses: actions/checkout@v4\n"
            "      - name: Publish\n"
            "        if: always() && startsWith(github.ref, 'refs/tags/v') && needs.gpu-proof.result == 'success'\n"
            "        run: npm publish --provenance --access public\n"
            "      - name: Sneak docker publish\n"
            '        uses: "./.github/actions/docker-publish"\n'
            "        with:\n          push: true\n",
        )
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertTrue(
            any("SECOND" in f and "Sneak docker publish" in f and "not the gated step" in f for f in findings),
            findings,
        )

    def test_duplicate_gated_step_name_is_a_named_ambiguity_not_a_silent_first_match(self):
        # Two steps in the SAME job share the display name `Publish` -- the
        # first genuinely gated, the second NOT gated and itself publishing.
        # Neither `find_step_if_by_name` (returning the FIRST match) nor
        # `_other_publishing_steps` (excluding EVERY step sharing that name
        # from its own scan, hiding the second) may silently trust the
        # first: both refuse the ambiguity by name instead of guessing.
        texts = _positive_texts()
        texts["npm.yml"] = _wf(
            "v*",
            _gate_job("gpu-proof")
            + "  publish:\n    needs: [gpu-proof]\n    if: always()\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - uses: actions/checkout@v4\n"
            "      - name: Publish\n"
            "        if: always() && startsWith(github.ref, 'refs/tags/v') && needs.gpu-proof.result == 'success'\n"
            "        run: npm publish --provenance --access public\n"
            "      - name: Publish\n"
            "        run: npm publish --tag sneak\n",
        )
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertTrue(
            any("Publish" in f and "not unique" in f for f in findings),
            findings,
        )

    def test_a_steps_entry_that_is_not_a_mapping_is_a_named_finding(self):
        # A bare scalar list item under `steps:` (never valid GitHub
        # Actions, but not assumed here) makes `steps:` unparseable as a
        # list of step mappings -- a named finding, never a silent skip
        # of the whole second-step check. The GATED step's own `if:` is
        # still found correctly (that reader is unaffected, and unrelated
        # to this one).
        texts = _positive_texts()
        texts["npm.yml"] = _wf(
            "v*",
            _gate_job("gpu-proof")
            + "  publish:\n    needs: [gpu-proof]\n    if: always()\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - actions/checkout@v4\n"
            "      - name: Publish\n"
            "        if: always() && startsWith(github.ref, 'refs/tags/v') && needs.gpu-proof.result == 'success'\n"
            "        run: npm publish --provenance --access public\n",
        )
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertTrue(any("not a mapping" in f for f in findings), findings)
        self.assertFalse(any("step `Publish` does not exist" in f for f in findings), findings)

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

    def test_quoted_on_block_is_read_exactly_like_bare_on(self):
        # A quoted "on": key is valid, unambiguous YAML (quoting is GitHub's
        # own documented way to avoid the YAML 1.1 boolean-resolution
        # gotcha) -- it must read IDENTICALLY to the bare form, never be
        # refused. Swap in a push: trigger so a wrongly-silent misread
        # (treating it as carrying no triggers at all) would also go
        # undetected -- P1 must still catch it as a bad_trigger.
        bad = PROVE_YML_GOOD.replace(
            "on:\n  workflow_dispatch:", '"on":\n  workflow_dispatch:\n  push:\n    tags: ["v*"]'
        )
        texts = _positive_texts()
        texts["gpu-prove.yml"] = bad
        findings = cgo.check_p1_p2(texts)
        self.assertFalse(any("cannot read" in f for f in findings), findings)
        self.assertTrue(any("push" in f and "on: block carries" in f for f in findings), findings)

    def test_quoted_inline_on_push_fails_p1(self):
        # Same evasion as P7's gang-row sibling test, against the prove
        # workflow's own on: block instead.
        bad = PROVE_YML_GOOD.replace(
            "on:\n  workflow_dispatch:\n  pull_request:\n    types: [labeled]\n"
            '  schedule:\n    - cron: "47 3 * * *"\n',
            'on: "push"\n',
        )
        texts = _positive_texts()
        texts["gpu-prove.yml"] = bad
        findings = cgo.check_p1_p2(texts)
        self.assertTrue(any("push" in f and "on: block carries" in f for f in findings), findings)

    def test_flow_style_on_block_is_read_exactly_like_block_style(self):
        # A flow-style on: {...} mapping is valid, unambiguous YAML -- it
        # must be read correctly, never refused. Include a push: trigger so
        # a silent misread (crediting it with no triggers) would still be
        # caught by P1's own bad_trigger check.
        bad = PROVE_YML_GOOD.replace(
            "on:\n  workflow_dispatch:\n  pull_request:\n    types: [labeled]\n  schedule:\n    - cron: \"47 3 * * *\"",
            'on: { workflow_dispatch: null, push: { tags: ["v*"] } }',
        )
        texts = _positive_texts()
        texts["gpu-prove.yml"] = bad
        findings = cgo.check_p1_p2(texts)
        self.assertFalse(any("cannot read" in f for f in findings), findings)
        self.assertTrue(any("push" in f and "on: block carries" in f for f in findings), findings)

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


class P1UsesReadFromTheParsedDocumentTest(unittest.TestCase):
    """P1's 'nothing may call the prove lane' rule reads a job-level
    `uses:` (local or cross-repo) from the parsed document, never a text
    regex -- mirrors `UsesReadFromTheParsedDocumentTest` one-for-one.
    Each case below is a quoting/`+`-truncation/unexaminable-sibling
    shape a text regex would miss, GREEN once `uses:` is read from the
    parsed document via `_scan_uses_references`."""

    def test_single_quoted_local_uses_reference_to_prove_workflow_fails(self):
        caller = "name: x\n\non:\n  workflow_dispatch:\n\njobs:\n  x:\n    uses: './.github/workflows/gpu-prove.yml'\n"
        texts = _positive_texts()
        texts["some-caller.yml"] = caller
        findings = cgo.check_p1_p2(texts)
        self.assertTrue(any("uses:` gpu-prove.yml" in f for f in findings), findings)

    def test_double_quoted_local_uses_reference_to_prove_workflow_fails(self):
        caller = 'name: x\n\non:\n  workflow_dispatch:\n\njobs:\n  x:\n    uses: "./.github/workflows/gpu-prove.yml"\n'
        texts = _positive_texts()
        texts["some-caller.yml"] = caller
        findings = cgo.check_p1_p2(texts)
        self.assertTrue(any("uses:` gpu-prove.yml" in f for f in findings), findings)

    def test_quoted_cross_repo_pinned_ref_to_prove_workflow_fails(self):
        caller = (
            "name: x\n\non:\n  workflow_dispatch:\n\njobs:\n  x:\n"
            '    uses: "f-inverse/jammi-ai/.github/workflows/gpu-prove.yml@a1b2c3d4"\n'
        )
        texts = _positive_texts()
        texts["some-caller.yml"] = caller
        findings = cgo.check_p1_p2(texts)
        self.assertTrue(any("cross-repo reference" in f for f in findings), findings)

    def test_plus_bearing_producer_name_is_never_truncated(self):
        # A character class like ([A-Za-z0-9_.-]+) stops at `+` -- it reads a
        # producer named `pub+lish.yml` as `pub`, which never equals the real
        # target and so never matches.
        texts = _positive_texts()
        texts["pub+lish.yml"] = texts.pop("gpu-prove.yml")
        caller = "name: x\n\non:\n  workflow_dispatch:\n\njobs:\n  x:\n    uses: ./.github/workflows/pub+lish.yml\n"
        texts["some-caller.yml"] = caller
        with mock.patch.object(cgo, "PROVE_PRODUCER_WORKFLOW", "pub+lish.yml"):
            findings = cgo.check_p1_p2(texts)
        self.assertTrue(any("uses:` pub+lish.yml" in f for f in findings), findings)

    def test_unexaminable_other_workflow_is_a_finding_never_a_silent_skip(self):
        # A sibling workflow whose own jobs: cannot be parsed might be the
        # very one hiding a forbidden reference -- it is a named FAIL, not
        # simply excluded from the scan the way a raw text regex always
        # examines it (successfully or not) regardless of parseability.
        # A genuine YAML anchor makes the WHOLE document unexaminable
        # (`_assert_no_github_incompatible_yaml` refuses it outright,
        # regardless of whether the anchor is ever referenced) --
        # `_parsed_jobs_or_fail` fails here where flow-style alone would
        # not (flow-style is still valid, constructible YAML; only
        # `job_source_spans`'s own LINE-SPAN reader refuses it, and this
        # rule does not use that reader for `uses:` discovery).
        unexaminable = (
            "name: bad\n\non: &trig\n  push:\n    branches: [main]\n\n"
            "jobs:\n  x:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo hi\n"
        )
        texts = _positive_texts()
        texts["bad.yml"] = unexaminable
        findings = cgo.check_p1_p2(texts)
        mine = [f for f in findings if "bad.yml" in f]
        self.assertGreaterEqual(len(mine), 1, findings)
        self.assertTrue(any("cannot examine" in f for f in mine), mine)


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
    """P5: `_gpu-proof-required.yml` actually CONSULTS the verdict. P3 only
    checks the gate job's `uses:` line, so without P5 gutting the reusable to
    `run: echo ok` would leave the gate green."""

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
        # --repo must be pinned to THIS repo too.
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
        # A --workflow override may never name anything other
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
    drives the seven mechanism-evasion fail shapes through `check_p5`."""
    return (
        "name: _gpu-proof-required\n\non:\n  workflow_call: {}\n\n"
        "permissions:\n  contents: read\n  actions: read\n\n"
        "jobs:\n  proof-required:\n    name: GPU proof required\n"
        "    runs-on: ubuntu-latest\n    timeout-minutes: 15\n    steps:\n"
        "      - uses: actions/checkout@v4\n" + step_text
    )


class ProofRequiredMechanismEvasionTest(unittest.TestCase):
    """Each of these seven shapes passes a whole-file substring check plus a
    FIRST-`--sha`-match regex while the job does not really depend on the
    verdict. Every one must FAIL under P5's mechanism (parse the reusable's
    job -> steps; the invocation must be the actual
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
    """GitHub Actions runs BOTH `.yml` and `.yaml`
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
        """The `uses:` scan skips only the resolved producer itself, never
        EVERY workflow whose file NAME matches a producer-name spelling
        (`gpu-prove.yml`/`gpu-prove.yaml`) -- a sibling file literally named
        `gpu-prove.yaml` that `uses: ./.github/workflows/gpu-prove.yml` must
        fail even though its OWN name matches that spelling."""
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
    """`[ \\t]*`, never `\\s*`, right after `needs:` -- the
    multi-line `needs:` list form (key alone on its line, `- item` entries
    below it) must PASS, not be misread as a single literal `- gate` name."""

    def test_multiline_needs_list_passes(self):
        raw_needs_block = "    needs:\n      - gpu-proof\n"
        texts = _positive_texts()
        texts["release-binaries.yml"] = _release_binaries_yml(raw_cu12_needs_block=raw_needs_block)
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertEqual(findings, [])


class P6DiscoveryTest(unittest.TestCase):
    """P6: every workflow file is scanned, no trigger
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
        # A `push: branches:`-only workflow (the real-tree `image.yml`/
        # `image-cuda.yml` shape) with an UNLISTED publishing primitive
        # carries no `tags:` at all. P6 does no trigger filtering: it must
        # be discovered exactly like a tag-triggered one.
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
    """PRIMITIVE_PATTERNS is a regex list over comment-
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

    def test_imagetools_create_unlisted_fails(self):
        # An untabled `docker buildx imagetools create` merge job is a
        # promotion (it moves a real, consumer-facing tag) and must be caught
        # by name -- the same doctrine every other primitive above gets.
        findings = cgo.check_p6_discovery(
            self._rogue("docker buildx imagetools create -t ghcr.io/f-inverse/rogue:latest a b")
        )
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_imagetools_inspect_only_is_not_a_promotion(self):
        # `imagetools inspect` is read-only (used to assert a merged index's
        # platform set) -- never bare `imagetools`, and never flagged as a
        # promotion on its own.
        clean = (
            "name: clean\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  inspect-only:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - run: docker buildx imagetools inspect ghcr.io/f-inverse/rogue:latest\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "clean.yml": clean})
        self.assertEqual(findings, [], findings)


class JobLevelAndSequenceCarrierTest(unittest.TestCase):
    """The publish-primitive matcher's domain covers job-level
    `env:`/`strategy.matrix:` and a YAML SEQUENCE under `with:`/`env:` --
    carriers a `run:` step's own scalar never spells the primitive in at
    all (only a variable REFERENCE), and a step's own `with:` value that
    is a list, not a bare string."""

    def test_job_level_env_with_an_indirect_run_command_is_caught(self):
        rogue = (
            "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    env:\n      CMD: cargo publish\n"
            "    steps:\n"
            '      - run: bash -c "$CMD"\n'
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_strategy_matrix_with_an_indirect_run_command_is_caught(self):
        rogue = (
            "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n"
            '    strategy:\n      matrix:\n        cmd: ["cargo publish"]\n'
            "    steps:\n"
            "      - run: ${{ matrix.cmd }}\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_with_args_sequence_carrier_is_caught(self):
        rogue = (
            "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: docker://alpine\n"
            "        with:\n          args:\n            - -c\n            - cargo publish\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_job_level_env_carrier_on_a_step_gated_rows_second_step_is_caught(self):
        texts = _positive_texts()
        texts["npm.yml"] = _wf(
            "v*",
            _gate_job("gpu-proof")
            + "  publish:\n    needs: [gpu-proof]\n    if: always()\n    runs-on: ubuntu-latest\n"
            "    env:\n      SNEAK_CMD: npm publish --tag sneak\n"
            "    steps:\n      - uses: actions/checkout@v4\n"
            "      - name: Publish\n"
            "        if: always() && startsWith(github.ref, 'refs/tags/v') && needs.gpu-proof.result == 'success'\n"
            "        run: npm publish --provenance --access public\n",
        )
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertTrue(
            any("SECOND" in f and "job-level" in f and "not the gated step" in f for f in findings),
            findings,
        )

    def test_steps_present_but_not_a_list_is_a_named_finding_never_a_typeerror(self):
        # `steps: 5` would raise an uncaught TypeError from
        # `for step in job_node.get("steps") or []` (a non-empty int is
        # truthy, so `or []` never substitutes) -- it is a named finding.
        rogue = "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n  sneak:\n    runs-on: ubuntu-latest\n    steps: 5\n"
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        mine = [f for f in findings if "rogue.yml" in f and "sneak" in f]
        self.assertEqual(len(mine), 1, findings)
        self.assertIn("not a list", mine[0])


class RecursiveLocalReusableDiscoveryTest(unittest.TestCase):
    """A job that merely `uses:` a LOCAL reusable workflow
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
        # _ci-base-image.yml's OWN `build-and-push` AND `merge-manifest` jobs
        # (workflow_call-only file) must never themselves be required as
        # table rows -- only their caller's job is: the caller's
        # already-tabled row covers a SECOND promoting job in the same
        # reusable too -- no new PROMOTION_TABLE row for the merge job.
        findings = cgo.check_p6_discovery(_positive_texts())
        self.assertFalse(
            any("_ci-base-image.yml" in f and "build-and-push" in f for f in findings), findings
        )
        self.assertFalse(
            any("_ci-base-image.yml" in f and "merge-manifest" in f for f in findings), findings
        )


class DifferentialOracleAgainstTheBlanketUsesRuleTest(unittest.TestCase):
    """P6's per-delegate traversal (including composite-action resolution)
    is a STRICT REFINEMENT of the blanket fail-closed rule that flags EVERY
    unlisted job's job-level `uses:` unconditionally -- mere PRESENCE, never
    opened, never examined -- once the SAME two named exemptions
    (`REVIEWED_NONPUBLISHING_LOCAL_REUSABLES`, the `gate_job`/`PROOF_
    REQUIRED_WORKFLOW` pair) are applied. The traversal only ever CLEARS a
    finding by actually opening the delegate and finding the WHOLE
    reachable set clean; it never SILENCES a delegation the blanket rule
    would flag. This oracle implements the blanket rule directly (the
    module under test has no such function) and asserts, over the real tree
    AND every synthetic fixture defined in this file, that P6's own
    (workflow, job) finding pairs are a SUBSET of the blanket rule's."""

    @staticmethod
    def _blanket_rule_flagged_pairs(workflow_texts: dict[str, str]) -> set[tuple[str, str]]:
        listed = {(row.workflow, row.promoting_job) for row in cgo.PROMOTION_TABLE.values()}
        gate_jobs = {
            (row.workflow, row.gate_job) for row in cgo.PROMOTION_TABLE.values() if row.gate_job is not None
        }
        listed_resolved: set[tuple[str, str]] = set()
        for workflow, job in listed:
            resolved = cgo.resolve_workflow(workflow_texts, workflow)
            listed_resolved.add((resolved if resolved is not None else workflow, job))
        gate_jobs_resolved: set[tuple[str, str]] = set()
        for workflow, job in gate_jobs:
            resolved = cgo.resolve_workflow(workflow_texts, workflow)
            gate_jobs_resolved.add((resolved if resolved is not None else workflow, job))
        proof_required_variants = set(cgo._workflow_name_variants(cgo.PROOF_REQUIRED_WORKFLOW))
        out: set[tuple[str, str]] = set()
        for name, text in workflow_texts.items():
            on_keys, on_err = cgo.read_top_level_on_block(text)
            if on_err is not None or on_keys == ["workflow_call"]:
                continue
            jobs, jobs_err = cgo._parsed_jobs_or_fail(text)
            if jobs_err is not None:
                continue
            for job_name, job_node in jobs.items():
                if (name, job_name) in listed_resolved:
                    continue
                # The blanket rule's direct-primitive check calls the SAME
                # `job_invokes_publish_primitive` P6 does, so whatever that
                # matcher counts as a match widens both sides equally and
                # the subset property still holds.
                primitive, primitive_err = cgo.job_invokes_publish_primitive(job_node)
                if primitive_err is not None or primitive is not None:
                    out.add((name, job_name))
                    continue
                uses = job_node.get("uses")
                if not isinstance(uses, str):
                    continue
                if (name, job_name) in gate_jobs_resolved:
                    gate_target = cgo._local_reusable_workflow_target(job_node)
                    if gate_target is not None and gate_target in proof_required_variants:
                        continue
                if cgo._job_level_uses_is_reviewed_nonpublishing(job_node):
                    continue
                # The blanket rule: PRESENCE alone, unconditionally.
                out.add((name, job_name))
        return out

    @staticmethod
    def _rebuild_flagged_pairs(findings: list[str]) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        for f in findings:
            m = re.match(r"P6: (\S+)'s job `([^`]+)`", f)
            if m:
                out.add((m.group(1), m.group(2)))
        return out

    def _assert_strict_refinement(self, workflow_texts: dict[str, str]) -> None:
        old_shape = self._blanket_rule_flagged_pairs(workflow_texts)
        rebuild_findings = cgo.check_p6_discovery(workflow_texts)
        rebuild_pairs = self._rebuild_flagged_pairs(rebuild_findings)
        self.assertTrue(rebuild_pairs <= old_shape, rebuild_pairs - old_shape)

    def test_real_tree_is_a_strict_refinement(self):
        self._assert_strict_refinement(cgo.load_workflow_texts(cgo.WORKFLOWS_DIR))

    def test_positive_fixture_is_a_strict_refinement(self):
        self._assert_strict_refinement(_positive_texts())

    def test_diamond_fixture_is_a_strict_refinement(self):
        shared = LocalReusableTraversalTest._clean_reusable()
        texts = {
            **_positive_texts(),
            "_shared.yml": shared,
            "caller-a.yml": (
                "name: caller-a\n\non:\n  push:\n    branches: [main]\n\n"
                "jobs:\n  a:\n    uses: ./.github/workflows/_shared.yml\n"
            ),
            "caller-b.yml": (
                "name: caller-b\n\non:\n  push:\n    branches: [main]\n\n"
                "jobs:\n  b:\n    uses: ./.github/workflows/_shared.yml\n"
            ),
        }
        self._assert_strict_refinement(texts)

    def test_untabled_local_reusable_caller_fixture_is_a_strict_refinement(self):
        texts = {
            **_positive_texts(),
            "rogue-caller.yml": _local_reusable_caller_yml(caller_job_name="sneak-build", target="_ci-base-image.yml"),
        }
        self._assert_strict_refinement(texts)


class LocalCompositeActionResolutionTest(unittest.TestCase):
    """A step-level `uses: ./.github/actions/<x>` is RESOLVED and its
    `action.yml`'s own steps EXAMINED through the same readers, never keyed
    on a hand-named action list -- ANY local composite action whose own body
    invokes a primitive is caught. Uses a synthetic `_ACTIONS_DIR` (never the real
    `.github/actions/`, which `docker-publish`/`release-upload` fixtures
    elsewhere in this file already exercise for real)."""

    def _with_synthetic_actions_dir(self, actions: dict[str, str]):
        """`actions`: {name: action.yml text}. Returns a context manager
        that patches `cgo._ACTIONS_DIR` at a fresh temp directory holding
        exactly those actions, so no test here ever touches this repo's
        real `.github/actions/`."""
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        for name, text in actions.items():
            action_dir = root / name
            action_dir.mkdir(parents=True, exist_ok=True)
            (action_dir / "action.yml").write_text(text, encoding="utf-8")
        return tmp, mock.patch.object(cgo, "_ACTIONS_DIR", root)

    def test_local_composite_action_running_docker_push_unlisted_fails(self):
        # The RED fixture: a local composite action whose body runs
        # `docker push` via a plain `run:` command (never docker/build-
        # push-action, never a hand-named path) -- invisible to the
        # deleted name-keyed regex, which only ever recognised the two
        # literal paths `docker-publish`/`release-upload`.
        action_yml = (
            "name: sneaky-pusher\nruns:\n  using: composite\n  steps:\n"
            "    - name: Push it\n      shell: bash\n"
            "      run: docker push ghcr.io/f-inverse/rogue:latest\n"
        )
        tmp, patcher = self._with_synthetic_actions_dir({"sneaky-pusher": action_yml})
        with tmp, patcher:
            rogue = (
                "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
                "  check:\n    runs-on: ubuntu-latest\n    steps:\n"
                "      - uses: actions/checkout@v4\n"
                "      - uses: ./.github/actions/sneaky-pusher\n"
            )
            findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        self.assertTrue(any("check" in f and "docker push" in f for f in findings), findings)

    def test_local_composite_action_clean_body_is_not_flagged(self):
        action_yml = (
            "name: clean-noop\nruns:\n  using: composite\n  steps:\n"
            "    - name: Say hi\n      shell: bash\n      run: echo hi\n"
        )
        tmp, patcher = self._with_synthetic_actions_dir({"clean-noop": action_yml})
        with tmp, patcher:
            clean = (
                "name: clean\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
                "  build-only:\n    runs-on: ubuntu-latest\n    steps:\n"
                "      - uses: ./.github/actions/clean-noop\n"
            )
            findings = cgo.check_p6_discovery({**_positive_texts(), "clean.yml": clean})
        self.assertEqual(findings, [], findings)

    def test_missing_local_action_yml_is_a_named_refusal_never_a_silent_pass(self):
        tmp, patcher = self._with_synthetic_actions_dir({})
        with tmp, patcher:
            rogue = (
                "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
                "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
                "      - uses: ./.github/actions/does-not-exist\n"
            )
            findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        mine = [f for f in findings if "rogue.yml" in f and "sneak" in f]
        self.assertEqual(len(mine), 1, findings)
        self.assertIn("no action.yml", mine[0])

    def test_nested_local_action_running_docker_push_is_caught_via_the_action_it_calls(self):
        outer = (
            "name: outer\nruns:\n  using: composite\n  steps:\n"
            "    - name: delegate\n      uses: ./.github/actions/inner-pusher\n"
        )
        inner = (
            "name: inner-pusher\nruns:\n  using: composite\n  steps:\n"
            "    - name: Push it\n      shell: bash\n"
            "      run: docker push ghcr.io/f-inverse/rogue:latest\n"
        )
        tmp, patcher = self._with_synthetic_actions_dir({"outer-wrapper": outer, "inner-pusher": inner})
        with tmp, patcher:
            rogue = (
                "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
                "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
                "      - uses: ./.github/actions/outer-wrapper\n"
            )
            findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        self.assertTrue(any("sneak" in f and "docker push" in f for f in findings), findings)

    def test_cross_repo_local_shaped_action_is_refused_not_silently_passed(self):
        rogue = (
            "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: f-inverse/other-repo/.github/actions/some-action@main\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        mine = [f for f in findings if "rogue.yml" in f and "sneak" in f]
        self.assertEqual(len(mine), 1, findings)
        self.assertIn("CROSS-REPO", mine[0].upper())


class LocalReusableTraversalTest(unittest.TestCase):
    """The per-delegate traversal: diamond
    memoization, a cycle's named depth-bound refusal, self-mask (a
    sub-job's own direct match never masks examining its own further
    `uses:`), a dangling/cross-repo target reached MID-CHAIN (not only at
    the top-level caller) is its own named refusal, and memoized refusal
    precedence (every caller reaching the same unexaminable reusable
    sees the identical named refusal)."""

    @staticmethod
    def _clean_reusable() -> str:
        return (
            "name: clean-reusable\n\non:\n  workflow_call:\n\n"
            "jobs:\n  noop:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - run: echo hi\n"
        )

    def test_diamond_reusable_is_walked_once_not_once_per_caller(self):
        shared = self._clean_reusable()
        caller_a = (
            "name: caller-a\n\non:\n  push:\n    branches: [main]\n\n"
            "jobs:\n  a:\n    uses: ./.github/workflows/_shared.yml\n"
        )
        caller_b = (
            "name: caller-b\n\non:\n  push:\n    branches: [main]\n\n"
            "jobs:\n  b:\n    uses: ./.github/workflows/_shared.yml\n"
        )
        texts = {
            **_positive_texts(),
            "_shared.yml": shared,
            "caller-a.yml": caller_a,
            "caller-b.yml": caller_b,
        }
        original = cgo._parsed_jobs_or_fail
        calls: list[str] = []

        def counting(text):
            calls.append(text)
            return original(text)

        with mock.patch.object(cgo, "_parsed_jobs_or_fail", side_effect=counting):
            findings = cgo.check_p6_discovery(texts)
        self.assertEqual(findings, [], findings)
        self.assertEqual(calls.count(shared), 1, calls)

    def test_uses_cycle_terminates_in_a_named_depth_refusal_never_a_recursionerror(self):
        a = "name: a\n\non:\n  workflow_call:\n\njobs:\n  j:\n    uses: ./.github/workflows/_b.yml\n"
        b = "name: b\n\non:\n  workflow_call:\n\njobs:\n  k:\n    uses: ./.github/workflows/_a.yml\n"
        caller = (
            "name: caller\n\non:\n  push:\n    branches: [main]\n\n"
            "jobs:\n  top:\n    uses: ./.github/workflows/_a.yml\n"
        )
        texts = {**_positive_texts(), "_a.yml": a, "_b.yml": b, "caller.yml": caller}
        findings = cgo.check_p6_discovery(texts)
        mine = [f for f in findings if "caller.yml" in f]
        self.assertEqual(len(mine), 1, mine)
        self.assertIn("max examined depth", mine[0])

    def test_self_mask_direct_match_never_skips_examining_the_same_jobs_own_uses(self):
        # A single job carrying BOTH direct-match step content AND its
        # own job-level `uses:` is not valid GitHub Actions (`uses:` and
        # `steps:` are mutually exclusive there), but is not assumed
        # impossible here -- examining the job-level `uses:` must never
        # be skipped just because a direct match was already found.
        mid = (
            "name: mid\n\non:\n  workflow_call:\n\njobs:\n  j:\n"
            "    runs-on: ubuntu-latest\n"
            "    uses: ./.github/workflows/_does_not_exist_nested.yml\n"
            "    steps:\n      - run: npm publish\n"
        )
        caller = (
            "name: caller\n\non:\n  push:\n    branches: [main]\n\n"
            "jobs:\n  top:\n    uses: ./.github/workflows/_mid.yml\n"
        )
        texts = {**_positive_texts(), "_mid.yml": mid, "caller.yml": caller}
        findings = cgo.check_p6_discovery(texts)
        mine = [f for f in findings if "caller.yml" in f]
        self.assertEqual(len(mine), 1, mine)
        self.assertIn("_does_not_exist_nested.yml", mine[0])

    def test_nested_cross_repo_delegate_is_a_named_refusal_not_a_silent_pass(self):
        # A traversal that resolves only a `./`-prefixed LOCAL target
        # leaves a cross-repo delegate reached MID-CHAIN (not at the
        # top-level caller) silently invisible, never even a refusal.
        mid = (
            "name: mid\n\non:\n  workflow_call:\n\njobs:\n  j:\n"
            "    uses: f-inverse/other-repo/.github/workflows/_reuse.yml@main\n"
        )
        caller = (
            "name: caller\n\non:\n  push:\n    branches: [main]\n\n"
            "jobs:\n  top:\n    uses: ./.github/workflows/_mid.yml\n"
        )
        texts = {**_positive_texts(), "_mid.yml": mid, "caller.yml": caller}
        findings = cgo.check_p6_discovery(texts)
        mine = [f for f in findings if "caller.yml" in f]
        self.assertEqual(len(mine), 1, mine)
        self.assertIn("CROSS-REPO", mine[0])
        self.assertIn("_reuse.yml", mine[0])

    def test_dangling_nested_target_is_a_named_refusal(self):
        mid = (
            "name: mid\n\non:\n  workflow_call:\n\njobs:\n  j:\n"
            "    uses: ./.github/workflows/_does_not_exist_nested2.yml\n"
        )
        caller = (
            "name: caller\n\non:\n  push:\n    branches: [main]\n\n"
            "jobs:\n  top:\n    uses: ./.github/workflows/_mid2.yml\n"
        )
        texts = {**_positive_texts(), "_mid2.yml": mid, "caller.yml": caller}
        findings = cgo.check_p6_discovery(texts)
        mine = [f for f in findings if "caller.yml" in f]
        self.assertEqual(len(mine), 1, mine)
        self.assertIn("_does_not_exist_nested2.yml", mine[0])
        self.assertIn("dangling target", mine[0])

    def test_memoized_refusal_precedence_both_callers_see_the_identical_refusal(self):
        bad_mid = (
            "name: bad-mid\n\non:\n  workflow_call:\n\njobs:\n  j:\n"
            "    uses: ./.github/workflows/_does_not_exist_shared.yml\n"
        )
        caller_a = (
            "name: caller-a\n\non:\n  push:\n    branches: [main]\n\n"
            "jobs:\n  a:\n    uses: ./.github/workflows/_bad_mid.yml\n"
        )
        caller_b = (
            "name: caller-b\n\non:\n  push:\n    branches: [main]\n\n"
            "jobs:\n  b:\n    uses: ./.github/workflows/_bad_mid.yml\n"
        )
        texts = {
            **_positive_texts(),
            "_bad_mid.yml": bad_mid,
            "caller-a.yml": caller_a,
            "caller-b.yml": caller_b,
        }
        findings = cgo.check_p6_discovery(texts)
        mine_a = [f for f in findings if "caller-a.yml" in f]
        mine_b = [f for f in findings if "caller-b.yml" in f]
        self.assertEqual(len(mine_a), 1, findings)
        self.assertEqual(len(mine_b), 1, findings)

        def normalize(f: str) -> str:
            f = re.sub(r"^P6: caller-[ab]\.yml's", "P6: caller-X.yml's", f)
            return re.sub(r"job `[^`]+`", "job `X`", f, count=1)

        self.assertEqual(normalize(mine_a[0]), normalize(mine_b[0]))


def _quoted_job_level_caller(target: str, quote: str, job_name: str = "call-it") -> str:
    """`caller.yml`'s single job's job-level `uses:` wrapped in `quote`
    (`'"'` or `"'"`) -- GitHub reads a quoted `uses:` identically to a bare
    one; a text regex like `uses:\\s*\\./...` matches neither quote
    style."""
    return (
        "name: caller\n\non:\n  push:\n    branches: [main]\n\n"
        f"jobs:\n  {job_name}:\n    uses: {quote}./.github/workflows/{target}{quote}\n"
    )


class UsesReadFromTheParsedDocumentTest(unittest.TestCase):
    """Every `uses:` P6 reasons about -- job-level (local or cross-repo)
    and step-level action -- is read from the ONE parsed document, never
    a text regex. Fail-closed: a job-level `uses:` whose delegate cannot be
    opened and cleared is a FINDING unless the job itself is listed in
    `PROMOTION_TABLE` by name. A cross-repo reference, in particular, is
    found regardless of quoting, never skipped because only a `./`-prefixed
    LOCAL target resolves."""

    @staticmethod
    def _reusable(jobs_text: str) -> str:
        return f"name: reuse\n\non:\n  workflow_call:\n\n{jobs_text}"

    _PUBLISHING_JOBS = (
        "jobs:\n  publisher:\n    runs-on: ubuntu-latest\n    steps:\n      - run: npm publish\n"
    )

    def test_double_quoted_job_level_uses_is_a_finding_unless_listed(self):
        texts = {
            **_positive_texts(),
            "caller.yml": _quoted_job_level_caller("_reuse.yml", '"'),
            "_reuse.yml": self._reusable(self._PUBLISHING_JOBS),
        }
        findings = cgo.check_p6_discovery(texts)
        mine = [f for f in findings if "caller.yml" in f]
        self.assertEqual(len(mine), 1, mine)
        self.assertIn("not listed in PROMOTION_TABLE", mine[0])

    def test_single_quoted_job_level_uses_is_a_finding_unless_listed(self):
        texts = {
            **_positive_texts(),
            "caller.yml": _quoted_job_level_caller("_reuse.yml", "'"),
            "_reuse.yml": self._reusable(self._PUBLISHING_JOBS),
        }
        findings = cgo.check_p6_discovery(texts)
        mine = [f for f in findings if "caller.yml" in f]
        self.assertEqual(len(mine), 1, mine)
        self.assertIn("not listed in PROMOTION_TABLE", mine[0])

    def test_cross_repo_job_level_uses_is_a_finding_unless_listed(self):
        # A traversal resolving ONLY a `./`-prefixed local job-level
        # `uses:` never looks at a cross-repo
        # `owner/repo/.github/workflows/<f>@<ref>` reference, so a
        # delegating merge-path job calling one would look non-promoting no
        # matter what the callee did. The fail-closed rule reads only
        # PRESENCE, so this needs no callee content at all.
        caller = (
            "name: cross-repo-caller\n\non:\n  push:\n    branches: [main]\n\n"
            "jobs:\n  build:\n    uses: f-inverse/other-repo/.github/workflows/_reuse.yml@main\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "cross-repo-caller.yml": caller})
        mine = [f for f in findings if "cross-repo-caller.yml" in f]
        self.assertEqual(len(mine), 1, mine)
        self.assertIn("build", mine[0])
        self.assertIn("_reuse.yml", mine[0])

    def test_quoted_cross_repo_pinned_job_level_uses_is_a_finding_unless_listed(self):
        caller = (
            "name: cross-repo-caller\n\non:\n  push:\n    branches: [main]\n\n"
            'jobs:\n  build:\n    uses: "f-inverse/other-repo/.github/workflows/_reuse.yml@a1b2c3d4"\n'
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "cross-repo-caller.yml": caller})
        mine = [f for f in findings if "cross-repo-caller.yml" in f]
        self.assertEqual(len(mine), 1, mine)
        self.assertIn("build", mine[0])

    def test_plus_bearing_target_name_is_read_correctly_from_the_parsed_scalar(self):
        # No character-class truncation is possible here: the rule never
        # regex-extracts a substring out of `uses:`, it reads the whole
        # DECODED scalar the parser already produced.
        caller = (
            "name: plus-caller\n\non:\n  push:\n    branches: [main]\n\n"
            "jobs:\n  build:\n    uses: ./.github/workflows/pub+lish.yml\n"
        )
        texts = {
            **_positive_texts(),
            "plus-caller.yml": caller,
            "pub+lish.yml": self._reusable(self._PUBLISHING_JOBS),
        }
        findings = cgo.check_p6_discovery(texts)
        mine = [f for f in findings if "plus-caller.yml" in f]
        self.assertEqual(len(mine), 1, mine)
        self.assertIn("build", mine[0])
        self.assertIn("pub+lish.yml", mine[0])

    def test_dangling_job_level_target_is_a_finding_naming_job_and_target(self):
        # Fail-closed: this rule never resolves the target at all, so a
        # target that does not exist anywhere in the tree is exactly as
        # much a finding as one that does.
        caller = (
            "name: dangling-caller\n\non:\n  push:\n    branches: [main]\n\n"
            "jobs:\n  build:\n    uses: ./.github/workflows/_does_not_exist.yml\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "dangling-caller.yml": caller})
        mine = [f for f in findings if "dangling-caller.yml" in f]
        self.assertEqual(len(mine), 1, mine)
        self.assertIn("build", mine[0])
        self.assertIn("_does_not_exist.yml", mine[0])

    def test_listed_jobs_job_level_uses_is_never_flagged_regardless_of_target_shape(self):
        # image.yml's `build` row is REGISTERED (`PROMOTION_TABLE`'s
        # `ci-image-cpu`); once a job is listed, this rule never looks at
        # its `uses:` at all -- not its shape, not whether the target
        # exists, nothing. Repointed here at a target that does not even
        # exist, to prove the exemption is unconditional.
        texts = dict(_positive_texts())
        texts["image.yml"] = (
            "name: caller\n\non:\n  push:\n    branches: [main]\n  workflow_dispatch:\n\n"
            "jobs:\n  build:\n    if: github.ref_type != 'tag'\n"
            "    uses: ./.github/workflows/_does_not_exist_either.yml\n"
        )
        findings = cgo.check_p6_discovery(texts)
        mine = [f for f in findings if "image.yml" in f]
        self.assertEqual(mine, [])

    def test_reviewed_nonpublishing_local_reusable_is_never_flagged(self):
        # `REVIEWED_NONPUBLISHING_LOCAL_REUSABLES` exempts an UNLISTED
        # job's job-level `uses:` only when it names one of the small,
        # hand-reviewed set of local reusables confirmed to invoke no
        # primitive at all -- never a cross-repo or otherwise unreviewed
        # target (covered by the cases above).
        caller = (
            "name: summary-caller\n\non:\n  push:\n    branches: [main]\n\n"
            "jobs:\n  ci-summary:\n    if: always()\n"
            "    uses: ./.github/workflows/_summary.yml\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "summary-caller.yml": caller})
        mine = [f for f in findings if "summary-caller.yml" in f]
        self.assertEqual(mine, [])

    # -- step-level action matching (docker-publish / release-upload),
    # local and cross-repo, read from the parsed step scalar. -------------
    def test_quoted_double_docker_publish_push_true_unlisted_fails(self):
        rogue = (
            "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
            '      - uses: "./.github/actions/docker-publish"\n'
            "        with:\n          push: true\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_quoted_single_release_upload_unlisted_fails(self):
        rogue = (
            "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: './.github/actions/release-upload'\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_quoted_cross_repo_docker_publish_push_true_unlisted_fails(self):
        rogue = (
            "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
            '      - uses: "f-inverse/other-repo/.github/actions/docker-publish@main"\n'
            '        with:\n          push: "true"\n'
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        self.assertTrue(any("sneak" in f for f in findings), findings)

    # -- `push:` is decided from the parsed step's `with:` mapping (never
    # a line-anchored text regex). ----------------------------------------
    def test_flow_style_with_push_true_on_one_line_is_promoting(self):
        # A flow-style `with: { push: true }` never starts a line with
        # `push:` at all, so a line-anchored reader (`^[ \t]*push:`) would
        # miss it entirely.
        rogue = (
            "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: docker/build-push-action@v6\n"
            "        with: { push: true }\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_missing_with_mapping_entirely_is_promoting(self):
        rogue = (
            "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: docker/build-push-action@v6\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_with_mapping_present_but_no_push_key_is_promoting(self):
        rogue = (
            "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: docker/build-push-action@v6\n"
            "        with:\n          tags: ghcr.io/f-inverse/rogue:latest\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_literal_push_false_control_is_not_promoting(self):
        clean = (
            "name: clean\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  build-only:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: docker/build-push-action@v6\n"
            "        with:\n          push: false\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "clean.yml": clean})
        self.assertEqual(findings, [], findings)

    def test_quoted_false_control_is_not_promoting(self):
        clean = (
            "name: clean\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  build-only:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: docker/build-push-action@v6\n"
            '        with:\n          push: "false"\n'
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "clean.yml": clean})
        self.assertEqual(findings, [], findings)

    def test_bare_push_off_is_promoting_never_read_as_pyyaml_false(self):
        # PyYAML's SafeLoader constructs a bare `off` into the SAME
        # Python `False` a bare `false` constructs into -- but GitHub
        # Actions' own YAML-1.1 CORE-schema resolver treats only
        # `false`/`False`/`FALSE` as boolean-false; `off` resolves to the
        # STRING `"off"` there, which is not `false: it must stay
        # PROMOTING here (fail-closed), never silently cleared the way
        # PyYAML's own wider boolean set would read it.
        rogue = (
            "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: docker/build-push-action@v6\n"
            "        with:\n          push: off\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_bare_push_no_is_promoting_never_read_as_pyyaml_false(self):
        rogue = (
            "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: docker/build-push-action@v6\n"
            "        with:\n          push: no\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_quoted_false_capitalized_is_promoting_only_exact_lowercase_quoted_clears(self):
        # The quoted-string exemption is EXACTLY "false"/'false' (GitHub's
        # own two-form bare set has case variants; the quoted string form
        # does not) -- a quoted "False" is a plain string GitHub's own
        # resolver never treats as boolean-false either.
        rogue = (
            "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: docker/build-push-action@v6\n"
            '        with:\n          push: "False"\n'
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        self.assertTrue(any("sneak" in f for f in findings), findings)


class GateJobExemptionAndNameExemptedFilesAreScannedTest(unittest.TestCase):
    """A row's `gate_job` is exempt from P6's job-level `uses:` rule ONLY
    when its own parsed `uses:` resolves exactly to
    `_gpu-proof-required.yml` (never by name alone), and every
    exempted-by-name file's (`REVIEWED_NONPUBLISHING_LOCAL_REUSABLES`,
    `_gpu-proof-required.yml`) own jobs are scanned by the step-level rule
    directly -- a publish step added to either is a finding."""

    def test_gate_job_repointed_elsewhere_with_the_uses_text_in_its_name_is_a_finding(self):
        # A text-substring check on the
        # gate job's body would have been satisfied by this `name:`
        # value; the PARSED job-level `uses:` (what this rule reads
        # instead) resolves to `_evil.yml`, never `_gpu-proof-required.yml`.
        texts = _positive_texts()
        texts["npm.yml"] = _wf(
            "v*",
            "  gpu-proof:\n"
            '    name: "uses: ./.github/workflows/_gpu-proof-required.yml"\n'
            "    if: startsWith(github.ref, 'refs/tags/v')\n"
            "    uses: ./.github/workflows/_evil.yml\n"
            "    secrets: inherit\n" + _step_gated_job("publish", "gpu-proof", "Publish"),
        )
        texts["_evil.yml"] = (
            "name: evil\n\non:\n  workflow_call:\n\njobs:\n"
            "  x:\n    runs-on: ubuntu-latest\n    steps:\n      - run: npm publish\n"
        )
        findings = cgo.check_p6_discovery(texts)
        mine = [f for f in findings if "npm.yml" in f and "gpu-proof" in f]
        self.assertGreaterEqual(len(mine), 1, findings)

    def test_p3_gate_check_reads_the_parsed_uses_not_a_name_substring(self):
        texts = _positive_texts()
        texts["npm.yml"] = _wf(
            "v*",
            "  gpu-proof:\n"
            '    name: "uses: ./.github/workflows/_gpu-proof-required.yml"\n'
            "    if: startsWith(github.ref, 'refs/tags/v')\n"
            "    uses: ./.github/workflows/_evil.yml\n"
            "    secrets: inherit\n" + _step_gated_job("publish", "gpu-proof", "Publish"),
        )
        texts["_evil.yml"] = (
            "name: evil\n\non:\n  workflow_call:\n\njobs:\n"
            "  x:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo hi\n"
        )
        findings = cgo.check_promotion_table(texts, MANIFEST_GOOD)
        self.assertTrue(
            any("npm-publish" in f and "gate job `gpu-proof`" in f and "does not" in f for f in findings),
            findings,
        )

    def test_gpu_proof_required_itself_gaining_a_publish_step_is_a_finding(self):
        texts = _positive_texts()
        texts["_gpu-proof-required.yml"] = PROOF_REQUIRED_YML_GOOD.replace(
            "      - uses: actions/checkout@v4\n",
            "      - uses: actions/checkout@v4\n      - run: npm publish\n",
            1,
        )
        findings = cgo.check_p6_discovery(texts)
        mine = [f for f in findings if "_gpu-proof-required.yml" in f]
        self.assertGreaterEqual(len(mine), 1, findings)
        self.assertTrue(any("proof-required" in f for f in mine), mine)

    def test_summary_yml_gaining_a_publish_step_is_a_finding(self):
        texts = _positive_texts()
        texts["_summary.yml"] = (
            "name: Summary\n\non:\n  workflow_call:\n    inputs:\n"
            "      needs_json:\n        required: true\n        type: string\n\n"
            "jobs:\n  assert:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - run: npm publish\n"
        )
        caller = (
            "name: ci-caller\n\non:\n  push:\n    branches: [main]\n\n"
            "jobs:\n  ci-summary:\n    if: always()\n"
            "    uses: ./.github/workflows/_summary.yml\n"
            "    with:\n      needs_json: '{}'\n"
        )
        texts["ci-caller.yml"] = caller
        findings = cgo.check_p6_discovery(texts)
        mine = [f for f in findings if "_summary.yml" in f]
        self.assertGreaterEqual(len(mine), 1, findings)
        self.assertTrue(any("assert" in f for f in mine), mine)
        # The caller's own job-level uses: to the allowlisted file stays
        # exempt -- only the exempted file's OWN content is scanned.
        self.assertFalse(any("ci-caller.yml" in f for f in findings), findings)

    def test_with_command_carrier_is_caught(self):
        rogue = (
            "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: nick-fields/retry@v3\n"
            "        with:\n          command: cargo publish\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_with_script_carrier_is_caught(self):
        rogue = (
            "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: actions/github-script@v7\n"
            '        with:\n          script: "npm publish"\n'
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_with_args_carrier_is_caught(self):
        rogue = (
            "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: docker://alpine\n"
            '        with:\n          args: "npm publish"\n'
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_with_entrypoint_carrier_is_caught(self):
        rogue = (
            "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: docker://alpine\n"
            "        with:\n          entrypoint: npm publish\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_env_carrier_with_an_indirect_run_command_is_caught(self):
        # The `run:` line itself never spells the primitive out -- only
        # the `env:` value does.
        rogue = (
            "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - env:\n          CMD: npm publish\n"
            '        run: bash -c "$CMD"\n'
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        self.assertTrue(any("sneak" in f for f in findings), findings)

    def test_bare_string_step_entry_is_a_named_finding_never_a_silent_skip(self):
        rogue = (
            "name: rogue\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "  sneak:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - actions/checkout@v4\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "rogue.yml": rogue})
        mine = [f for f in findings if "rogue.yml" in f and "sneak" in f]
        self.assertGreaterEqual(len(mine), 1, findings)
        self.assertTrue(any("not a mapping" in f for f in mine), mine)

    def test_job_level_uses_that_is_a_list_not_a_string_is_a_finding(self):
        caller = (
            "name: list-caller\n\non:\n  push:\n    branches: [main]\n\n"
            "jobs:\n  build:\n    uses: [./.github/workflows/_evil.yml]\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "list-caller.yml": caller})
        mine = [f for f in findings if "list-caller.yml" in f]
        self.assertEqual(len(mine), 1, mine)
        self.assertIn("build", mine[0])
        self.assertIn("not a string", mine[0])


class UnreadableOnOrJobsBlockFailsLoudTest(unittest.TestCase):
    """An unreadable `on:`/`jobs:` block is a FAIL LOUD, never a silent skip
    -- the same doctrine P1 holds `gpu-prove.yml`'s `on:` to, applied across
    P6's full-tree scan. A quoted `"on":`/`"jobs":`
    key, a non-canonically-indented (but still consistent) `jobs:` block,
    and a flow-style `jobs: {...}` mapping are all real, valid YAML --
    P6 reads every one of them CORRECTLY (its own `jobs:` reader is the
    fully-parsed document, `_parsed_jobs_or_fail`, which does not care
    about block vs. flow style at all; only a line-span reader needs to
    refuse flow style, since a job's own line span is meaningless once
    every job lives on the same physical line). A
    genuinely unparseable document (duplicate keys, a YAML syntax error,
    an anchor/alias/tag, ...) is still a loud refusal."""

    def test_quoted_on_block_is_correctly_read_and_still_discovers_the_job(self):
        bad = '"on":\n  push:\n    tags: ["v*"]\n\njobs:\n  x:\n    runs-on: ubuntu-latest\n    steps:\n      - run: npm publish\n'
        findings = cgo.check_p6_discovery({**_positive_texts(), "bad.yml": "name: bad\n\n" + bad})
        self.assertFalse(any("bad.yml" in f and "cannot read" in f for f in findings), findings)
        self.assertTrue(any("bad.yml" in f and "x" in f for f in findings), findings)

    def test_quoted_jobs_block_is_correctly_read_and_still_discovers_the_job(self):
        bad = (
            "name: bad\n\non:\n  push:\n    tags: [\"v*\"]\n\n"
            '"jobs":\n  x:\n    runs-on: ubuntu-latest\n    steps:\n      - run: npm publish\n'
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "bad.yml": bad})
        self.assertFalse(any("bad.yml" in f and "cannot read" in f for f in findings), findings)
        self.assertTrue(any("bad.yml" in f and "x" in f for f in findings), findings)

    def test_flow_style_jobs_block_is_read_correctly_never_a_silent_or_misattributed_pass(self):
        # A line-span reader collapses the FIRST entry, the real publisher
        # `sneaky`, to a zero-length span and credits the LAST entry,
        # `tail`, with its text instead. P6's own `jobs:` reader is fully
        # parsed, so a flow-style mapping with
        # TWO entries is read exactly as correctly as a block-style one:
        # `sneaky`'s own real primitive is found and named; `tail` (inert)
        # is never mentioned.
        bad = (
            "name: bad\n\non:\n  push:\n    tags: [\"v*\"]\n\n"
            "jobs: { sneaky: { runs-on: ubuntu-latest, steps: [ { run: 'npm publish' } ] }, "
            "tail: { runs-on: ubuntu-latest } }\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "bad.yml": bad})
        self.assertTrue(any("bad.yml" in f and "sneaky" in f for f in findings), findings)
        self.assertFalse(any("bad.yml" in f and "tail" in f for f in findings), findings)

    def test_non_canonical_four_space_job_indent_is_correctly_read(self):
        bad = (
            "name: bad\n\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n"
            "    x:\n        runs-on: ubuntu-latest\n        steps:\n          - run: npm publish\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "bad.yml": bad})
        self.assertFalse(any("bad.yml" in f and "non-canonical" in f for f in findings), findings)
        self.assertTrue(any("bad.yml" in f and "x" in f for f in findings), findings)

    def test_duplicate_top_level_jobs_key_fails_loud(self):
        # A genuinely unparseable document -- a duplicate top-level `jobs:`
        # key -- IS the loud refusal this class actually guards; the four
        # cases above are all valid YAML and must never be confused with
        # this one.
        bad = (
            "name: bad\n\non:\n  push:\n    tags: [\"v*\"]\n\n"
            "jobs:\n  x:\n    runs-on: ubuntu-latest\n\n"
            "jobs:\n  y:\n    runs-on: ubuntu-latest\n    steps:\n      - run: npm publish\n"
        )
        findings = cgo.check_p6_discovery({**_positive_texts(), "bad.yml": bad})
        self.assertTrue(
            any("bad.yml" in f and "cannot parse YAML" in f and "duplicate key" in f for f in findings),
            findings,
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


class ReadOnBlockFromPathTest(unittest.TestCase):
    """`read_top_level_on_block_from_path`'s property: EVERY read error on
    the workflow file is a named FAIL ("cannot read file ..."), never
    `([], None)` / "no key" -- for EVERY euid, not only a non-root one. A
    directory path and a missing path raise `IsADirectoryError`/
    `FileNotFoundError` (both `OSError` subclasses) regardless of the
    calling user's privilege -- root does not bypass "this path is not a
    regular file" or "this path does not exist" the way it bypasses a
    `chmod 000` permission bit -- so these two cases prove the property on
    every CI runner, including the root container this repo's own lane
    runs in, where a chmod-000 fixture alone cannot."""

    def test_directory_path_is_a_named_cannot_read_fail(self):
        with tempfile.TemporaryDirectory() as d:
            keys, err = cgo.read_top_level_on_block_from_path(Path(d))
        self.assertIsNone(keys)
        self.assertIsNotNone(err)
        self.assertIn("cannot read file", err)

    def test_missing_path_is_a_named_cannot_read_fail(self):
        with tempfile.TemporaryDirectory() as d:
            missing = Path(d) / "does-not-exist.yml"
            keys, err = cgo.read_top_level_on_block_from_path(missing)
        self.assertIsNone(keys)
        self.assertIsNotNone(err)
        self.assertIn("cannot read file", err)

    def test_mode_000_file_is_a_named_cannot_read_fail_where_euid_cannot_bypass_it(self):
        # Guarded exactly like `test_gpu_gang_lane.sh`'s own G7 mode-000
        # fixture: root (or an ACL) bypasses a `chmod 000` permission bit
        # outright, so this case can only be asserted when `os.access`
        # itself reports the file unreadable post-chmod -- otherwise it is
        # skipped with a VISIBLE note, never silently reported as passing.
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "mode-000.yml"
            p.write_text("on:\n  workflow_dispatch:\n")
            os.chmod(p, 0o000)
            try:
                if os.access(p, os.R_OK):
                    print(
                        f"  * note: skipping the mode-000-file case -- euid {os.geteuid()} "
                        "can still read a mode-000 file (root or an ACL bypass), so this "
                        "fixture cannot establish the unreadable-file property here",
                        file=sys.stderr,
                    )
                    return
                keys, err = cgo.read_top_level_on_block_from_path(p)
            finally:
                os.chmod(p, 0o644)
        self.assertIsNone(keys)
        self.assertIsNotNone(err)
        self.assertIn("cannot read file", err)

    def test_non_utf8_file_is_a_named_cannot_read_fail_not_an_uncaught_traceback(self):
        # `read_top_level_on_block_from_path` catches `UnicodeDecodeError`
        # beside `OSError` (the former is not a subclass of the latter), so
        # a non-UTF-8 workflow file is the same named "cannot read file"
        # FAIL every other read error gets, never an uncaught traceback.
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "not-utf8.yml"
            p.write_bytes(b"on:\n  push:\n  \xff\xfe not valid utf-8 \x80\x81\n")
            keys, err = cgo.read_top_level_on_block_from_path(p)
        self.assertIsNone(keys)
        self.assertIsNotNone(err)
        self.assertIn("cannot read file", err)


if __name__ == "__main__":
    unittest.main()
