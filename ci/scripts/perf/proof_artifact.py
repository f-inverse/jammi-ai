#!/usr/bin/env python3
"""Turn `/root/proof-out/<tag>_*` into one committed-artifact JSON per tag.

This is the README-named producer for `crates/jammi-kernels/artifacts/
cuda-runs/*.json`'s `"artifact": "cuda-run"` shape (the `p1`/`p2`/`p3`-style
files) — the tool a pod session runs after a `cuda_parity`/crate-test/bench
sweep to fold the captured cargo logs into one committed JSON, so a green
artifact is machine-checked evidence of what actually ran, not a hand-typed
summary. `ci/scripts/check_cuda_run_artifacts.py` is the gate that gives that
evidence a schema and reconciles `git_sha` against `HEAD`'s ancestry (see the
maintainer guide's CI-guard-contracts list, and `docs/maintainer/
cuda-kernel-guide.md` §4).

usage: proof_artifact.py OUT_DIR TAG [TAG...]
       proof_artifact.py --self-test

Each artifact records WHAT ran (`schema_version`, `git_sha`, device/driver/
nvcc, the exact commands' outcomes), every test name with its status parsed
from the captured cargo-test log, and every bench leg's headline numbers +
dispatch counters. A tag with no `cuda_parity` log or with zero parsed tests
is written with `"status": "INVALID"` — zero-tests-matched is red, not a skip
(the execution-provenance principle: a machine-checked artifact of what ran,
never an unmeasured green).

No pod-local absolute path is required: `PROOF_SRC` (default `/root/
proof-out`, override-able) is the only path this script reads from besides
`OUT_DIR`, and both are caller-supplied, never hard-coded to a specific pod's
filesystem layout.

Not implemented: raw-leg persistence (unification contract C5.3 — writing the
per-leg raw inputs out alongside the folded artifact); this script emits only
the folded JSON today.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import datetime

SCHEMA_VERSION = 1
PRODUCER_PATH = "ci/scripts/perf/proof_artifact.py"


def sh(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as e:
        return f"ERR({e.returncode}): {e.output.strip()[:200]}"
    except FileNotFoundError as e:
        return f"ERR(not found): {e}"


def collect_env() -> dict:
    return {
        "gpu": sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"),
        "nvcc": sh("nvcc --version | tail -n 2 | head -n 1"),
        "rustc": sh("rustc -V"),
        "hostname": sh("hostname"),
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


TEST_LINE = re.compile(r"^test (\S+) \.\.\. (ok|FAILED|ignored)")
RESULT_LINE = re.compile(r"^test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored")
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def parse_test_log(path: str) -> tuple[list[dict], list[dict]]:
    tests: list[dict] = []
    results: list[dict] = []
    if not os.path.exists(path):
        return tests, results
    for line in open(path, errors="replace"):
        m = TEST_LINE.match(line.strip())
        if m:
            tests.append({"name": m.group(1), "status": m.group(2)})
        m = RESULT_LINE.match(line.strip())
        if m:
            results.append(
                {
                    "outcome": m.group(1),
                    "passed": int(m.group(2)),
                    "failed": int(m.group(3)),
                    "ignored": int(m.group(4)),
                }
            )
    return tests, results


def bench_leg(path: str, *, expected_git_sha: str | None = None) -> dict:
    try:
        d = json.load(open(path))
        t = d["tiers"]["finetune_step"]
    except Exception as e:  # noqa: BLE001 - surfacing any parse failure as INVALID is the point
        return {"file": os.path.basename(path), "status": f"INVALID ({e})"}
    row = {
        "file": os.path.basename(path),
        "s_per_step_p50": t["s_per_step_p50"]["value"],
        "triplets_per_s": t.get("triplets_per_s", {}).get("value"),
        "peak_vram_bytes": t["peak_vram_bytes"]["value"],
        "max_grad_norm": t.get("max_grad_norm"),
        "dispatch_counters": {k: v for k, v in t.items() if k.endswith("_dispatches")},
    }
    row["fused_proof"] = all(
        v > 0 for k, v in row["dispatch_counters"].items() if k.endswith("_fused_dispatches") and v
    ) and all(v == 0 for k, v in row["dispatch_counters"].items() if k.endswith("_eager_dispatches"))

    # Unification contract C5.1: cross-check the RAW leg's own baked
    # provenance (`report.provenance.build_sha`, phase 1) against the tag's
    # resolved `.ref`. Absent when the leg predates phase 1 (no `provenance`
    # key at all) — nothing to cross-check, not a finding of its own; a
    # PRESENT-but-mismatched/unknown/-dirty build_sha invalidates the leg
    # (never silently folded into a GREEN summary).
    build_sha = (d.get("provenance") or {}).get("build_sha") if isinstance(d, dict) else None
    if expected_git_sha is not None and build_sha is not None:
        if build_sha != expected_git_sha:
            row["status"] = (
                f"INVALID (leg provenance.build_sha={build_sha!r} != tag git_sha={expected_git_sha!r})"
            )
            row["fused_proof"] = False
    return row


def resolve_git_sha(ref: str | None, repo_root: str | None) -> tuple[str | None, str | None]:
    """Returns (git_sha, git_sha_unresolved) — exactly one is non-None (unless
    `ref` itself is None, in which case both are None and the caller must
    treat the artifact as INVALID/schema-incomplete).

    A full 40-hex ref is trusted as-is (no repo access needed — this keeps
    the common case, a pod session with `ref = $(git rev-parse HEAD)`
    already full-length, working with zero git calls). A short ref is
    expanded via `git rev-parse` IF a repo is available and the object
    resolves; otherwise it is kept as `git_sha_unresolved` so the artifact
    still records what it saw rather than fabricating a value.
    """
    if not ref:
        return None, None
    if FULL_SHA_RE.match(ref):
        return ref, None
    if repo_root:
        expanded = sh(f"git -C {repo_root} rev-parse {ref}")
        if FULL_SHA_RE.match(expanded):
            return expanded, None
    return None, ref


def build_artifact(src: str, tag: str, *, repo_root: str | None = None) -> dict:
    ref = open(f"{src}/{tag}.ref").read().strip() if os.path.exists(f"{src}/{tag}.ref") else None
    gates = open(f"{src}/{tag}.gates").read().strip() if os.path.exists(f"{src}/{tag}.gates") else None
    env = collect_env()
    parity_tests, parity_results = parse_test_log(f"{src}/{tag}_cuda_parity.log")
    crate_logs = {}
    for p in sorted(glob.glob(f"{src}/{tag}_test_*.log")):
        crate = os.path.basename(p)[len(tag) + 6 : -4]
        crate_logs[crate] = dict(zip(("tests", "results"), parse_test_log(p)))

    # Resolved BEFORE the leg loop (contract C5.1): each leg's own baked
    # provenance.build_sha is cross-checked against THIS tag's resolved sha.
    git_sha, git_sha_unresolved = resolve_git_sha(ref, repo_root)
    legs = [
        bench_leg(p, expected_git_sha=git_sha)
        for p in sorted(glob.glob(f"{src}/{tag}*_b*_s*_d*.json"))
    ]

    art: dict = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "cuda-run",
        "tag": tag,
        "env": env,
        "box": env.get("hostname") or "unknown",
        "gates": gates,
        "cuda_parity": {
            "command": "JAMMI_REQUIRE_CUDA=1 JAMMI_KERNELS_STRICT=1 cargo test -p jammi-kernels --features cuda --test cuda_parity",
            "results": parity_results,
            "tests": parity_tests,
        },
        "crate_tests_cuda": crate_logs,
        "bench_legs": legs,
        "producer": {
            "path": PRODUCER_PATH,
            "kind": "script",
            "invocation": f"python3 {PRODUCER_PATH} <out_dir> {tag}",
            "gating": "env:JAMMI_REQUIRE_CUDA",
        },
    }
    if git_sha is not None:
        art["git_sha"] = git_sha
    else:
        art["git_sha_unresolved"] = git_sha_unresolved
        art["producer"] = {"path": None, "kind": "none", "invocation": None, "gating": "none"}

    # `total_results` counts parsed "test result: ..." SUMMARY lines, not
    # passed-test count — a run where every test FAILED still has
    # `total_results == 1` (the harness ran and matched something) and must
    # report RED, not be conflated with a log that matched nothing at all
    # (`total_results == 0`, the genuine "zero tests parsed" INVALID case).
    total_results = len(parity_results)
    failed = sum(r["failed"] for r in parity_results)
    art["status"] = "INVALID (zero parity tests parsed)" if total_results == 0 else ("RED" if failed else "GREEN")
    return art


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()

    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    out_dir, tags = argv[0], argv[1:]
    os.makedirs(out_dir, exist_ok=True)
    src = os.environ.get("PROOF_SRC", "/root/proof-out")
    for tag in tags:
        art = build_artifact(src, tag)
        path = f"{out_dir}/{tag}.json"
        json.dump(art, open(path, "w"), indent=2)
        passed = sum(r["passed"] for r in art["cuda_parity"]["results"])
        failed = sum(r["failed"] for r in art["cuda_parity"]["results"])
        print(
            f"{path}: {art['status']} sha={art.get('git_sha', art.get('git_sha_unresolved'))} "
            f"parity_passed={passed} failed={failed} legs={len(art['bench_legs'])} gates={art['gates']}"
        )
    return 0


# --------------------------------------------------------------------------- #
# self-test — proves the schema/parsing logic against synthetic fixtures, no
# GPU, no real pod logs required.
# --------------------------------------------------------------------------- #
def self_test() -> int:
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        # RED case: a tag with no cuda_parity log at all must be INVALID
        # (zero parsed tests), never silently green.
        art_missing = build_artifact(tmp, "notag")
        if art_missing["status"] != "INVALID (zero parity tests parsed)":
            failures.append(
                f"self-test FAILED: a tag with no cuda_parity log did not come back INVALID: {art_missing['status']}"
            )

        # GREEN case: a well-formed cuda_parity log + a bench leg JSON.
        with open(f"{tmp}/happy.ref", "w") as f:
            f.write("5f29e3b87ba2305533e83d21223343a73100cb64\n")
        with open(f"{tmp}/happy.gates", "w") as f:
            f.write("clippy=0 parity=0 tests=0")
        with open(f"{tmp}/happy_cuda_parity.log", "w") as f:
            f.write("test some_parity_case ... ok\n")
            f.write("test result: ok. 1 passed; 0 failed; 0 ignored\n")
        with open(f"{tmp}/happy_test_jammi-lora.log", "w") as f:
            f.write("test a_lora_test ... ok\n")
            f.write("test result: ok. 1 passed; 0 failed; 0 ignored\n")
        with open(f"{tmp}/happy_b8_s128_d0.json", "w") as f:
            json.dump(
                {
                    "tiers": {
                        "finetune_step": {
                            "s_per_step_p50": {"value": 0.5},
                            "triplets_per_s": {"value": 10.0},
                            "peak_vram_bytes": {"value": 123456.0},
                            "max_grad_norm": None,
                            "lora_linear_fused_dispatches": 10,
                            "lora_linear_eager_dispatches": 0,
                        }
                    }
                },
                f,
            )
        art_happy = build_artifact(tmp, "happy")
        if art_happy["status"] != "GREEN":
            failures.append(f"self-test FAILED: a fully-green fixture did not report GREEN: {art_happy}")
        if art_happy.get("git_sha") != "5f29e3b87ba2305533e83d21223343a73100cb64":
            failures.append(f"self-test FAILED: full-length ref was not trusted as git_sha verbatim: {art_happy}")
        if art_happy.get("schema_version") != SCHEMA_VERSION:
            failures.append(f"self-test FAILED: schema_version missing/wrong: {art_happy.get('schema_version')}")
        if art_happy.get("producer", {}).get("kind") != "script":
            failures.append(f"self-test FAILED: a resolved-sha artifact should carry producer.kind == 'script': {art_happy.get('producer')}")
        if not art_happy["bench_legs"] or art_happy["bench_legs"][0].get("fused_proof") is not True:
            failures.append(f"self-test FAILED: dispatch-counter fused_proof did not compute True on an all-fused leg: {art_happy['bench_legs']}")

        # RED case: a bench leg missing the expected tiers.finetune_step shape
        # must come back INVALID at the LEG level, not crash the whole tag.
        with open(f"{tmp}/broken.ref", "w") as f:
            f.write("5f29e3b87ba2305533e83d21223343a73100cb64\n")
        with open(f"{tmp}/broken_cuda_parity.log", "w") as f:
            f.write("test some_parity_case ... ok\n")
            f.write("test result: ok. 1 passed; 0 failed; 0 ignored\n")
        with open(f"{tmp}/broken_b8_s128_d0.json", "w") as f:
            json.dump({"not_tiers": {}}, f)
        art_broken = build_artifact(tmp, "broken")
        if not art_broken["bench_legs"] or "INVALID" not in art_broken["bench_legs"][0].get("status", ""):
            failures.append(f"self-test FAILED: a malformed bench leg JSON was not reported INVALID at the leg level: {art_broken['bench_legs']}")

        # RED case: a short (sha7) ref that does not resolve (no repo_root
        # given) must be recorded as git_sha_unresolved, with producer
        # collapsed to kind == 'none' — never fabricated into a fake 40-hex.
        with open(f"{tmp}/shortsha.ref", "w") as f:
            f.write("5f29e3b\n")
        with open(f"{tmp}/shortsha_cuda_parity.log", "w") as f:
            f.write("test x ... ok\n")
            f.write("test result: ok. 1 passed; 0 failed; 0 ignored\n")
        art_short = build_artifact(tmp, "shortsha")
        if "git_sha" in art_short or art_short.get("git_sha_unresolved") != "5f29e3b":
            failures.append(f"self-test FAILED: an unresolvable short ref was not recorded as git_sha_unresolved: {art_short}")
        if art_short.get("producer", {}).get("kind") != "none":
            failures.append(f"self-test FAILED: an unresolved git_sha did not collapse producer.kind to 'none': {art_short.get('producer')}")

        # RED case (unification contract C5.1): a leg whose OWN baked
        # provenance.build_sha does not match the tag's resolved .ref must
        # be written INVALID — never silently folded into a GREEN summary
        # off a binary that was not proven at the sha the tag claims.
        with open(f"{tmp}/provmismatch.ref", "w") as f:
            f.write("5f29e3b87ba2305533e83d21223343a73100cb64\n")
        with open(f"{tmp}/provmismatch_cuda_parity.log", "w") as f:
            f.write("test some_parity_case ... ok\n")
            f.write("test result: ok. 1 passed; 0 failed; 0 ignored\n")
        with open(f"{tmp}/provmismatch_b8_s128_d0.json", "w") as f:
            json.dump(
                {
                    "provenance": {"build_sha": "f" * 40},  # != the tag's resolved ref above
                    "tiers": {
                        "finetune_step": {
                            "s_per_step_p50": {"value": 0.5},
                            "triplets_per_s": {"value": 10.0},
                            "peak_vram_bytes": {"value": 123456.0},
                            "max_grad_norm": None,
                            "lora_linear_fused_dispatches": 10,
                            "lora_linear_eager_dispatches": 0,
                        }
                    },
                },
                f,
            )
        art_provmismatch = build_artifact(tmp, "provmismatch")
        prov_leg = art_provmismatch["bench_legs"][0]
        if "INVALID" not in prov_leg.get("status", ""):
            failures.append(f"self-test FAILED: a leg whose provenance.build_sha != the tag's ref was not written INVALID: {prov_leg}")
        if prov_leg.get("fused_proof") is not False:
            failures.append(f"self-test FAILED: a provenance-mismatched leg must not also report fused_proof=True: {prov_leg}")

        # GREEN control: a leg whose provenance.build_sha MATCHES the tag's
        # resolved ref stays clean (the cross-check must not false-positive
        # on a genuinely matching leg).
        with open(f"{tmp}/provmatch.ref", "w") as f:
            f.write("5f29e3b87ba2305533e83d21223343a73100cb64\n")
        with open(f"{tmp}/provmatch_cuda_parity.log", "w") as f:
            f.write("test some_parity_case ... ok\n")
            f.write("test result: ok. 1 passed; 0 failed; 0 ignored\n")
        with open(f"{tmp}/provmatch_b8_s128_d0.json", "w") as f:
            json.dump(
                {
                    "provenance": {"build_sha": "5f29e3b87ba2305533e83d21223343a73100cb64"},
                    "tiers": {
                        "finetune_step": {
                            "s_per_step_p50": {"value": 0.5},
                            "triplets_per_s": {"value": 10.0},
                            "peak_vram_bytes": {"value": 123456.0},
                            "max_grad_norm": None,
                            "lora_linear_fused_dispatches": 10,
                            "lora_linear_eager_dispatches": 0,
                        }
                    },
                },
                f,
            )
        art_provmatch = build_artifact(tmp, "provmatch")
        prov_match_leg = art_provmatch["bench_legs"][0]
        if "INVALID" in prov_match_leg.get("status", ""):
            failures.append(f"self-test FAILED: a leg whose provenance.build_sha MATCHES the tag's ref was wrongly written INVALID: {prov_match_leg}")

        # RED case: a failing parity log must report RED, never GREEN.
        with open(f"{tmp}/failing.ref", "w") as f:
            f.write("5f29e3b87ba2305533e83d21223343a73100cb64\n")
        with open(f"{tmp}/failing_cuda_parity.log", "w") as f:
            f.write("test some_parity_case ... FAILED\n")
            f.write("test result: FAILED. 0 passed; 1 failed; 0 ignored\n")
        art_failing = build_artifact(tmp, "failing")
        if art_failing["status"] != "RED":
            failures.append(f"self-test FAILED: a failing parity log did not report RED: {art_failing['status']}")

    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        print("proof_artifact self-test: FAIL", file=sys.stderr)
        return 1
    print(
        "proof_artifact self-test: OK — zero-tests-parsed INVALID, a happy-path GREEN with schema fields "
        "and fused_proof, a malformed leg's INVALID isolation, an unresolved short-sha's producer collapse, "
        "a provenance.build_sha mismatch's leg-level INVALID (contract C5.1) plus its GREEN matching "
        "control, and a failing parity log's RED are all exercised."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
