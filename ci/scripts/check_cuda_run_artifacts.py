#!/usr/bin/env python3
"""CUDA-run-artifact schema + provenance gate — hermetic, static, no build, no GPU.

## The escape this closes (census, `.jammi/escapes.jsonl` row 215)

Nothing under `crates/jammi-kernels/artifacts/cuda-runs/` was read by any CI
script — the directory's own README claims "the file is the evidence, the
commit message is a pointer to it", but nothing ever checked the evidence was
shaped like evidence. Three concrete defects that absence hid:

  1. **4 of the 5 committed artifacts name a `git_sha` that is an ancestor of
     NO ref on this branch** (`5932520`, `e32ed90`, `eac63fd`, `80f02fb` — the
     branches that produced them were squash-merged, so the exact commit each
     artifact proved never survives in `HEAD`'s history). A green artifact
     whose sha is not an ancestor of the branch is evidence about the
     ORACLE, not the code (`docs/maintainer/cuda-kernel-guide.md` §4) — the
     artifact is proving a tree that no longer exists.
  2. **Five mutually incompatible top-level schemas coexist** across this
     repo's cuda-run artifacts (`git_sha` / `tip_sha` full / `tip_sha` sha7 /
     `deliverable_git_sha` / no build-ref field at all), so no script could
     have reconciled them even if one had tried.
  3. **Raw-run files under a `*-raw-runs/` subdirectory carry no build ref at
     all** — they are the per-leg raw `jammi-bench` JSON a parent artifact
     folds numbers from, and inherit nothing from that parent mechanically.

The fix is not "add a check for those four known-bad shas" (a grep for one
known-bad string is exactly the anti-pattern `check_doc_parity.py`'s own
docstring warns against, and `check_gpu_parity_matrix.py`'s docstring
restates for the device axis); it is a *schema* every artifact must satisfy,
checked by a script that recurses into `*-raw-runs/` too, so a newly
committed artifact with an unproven sha, an unresolvable producer, or a
silently-defaulted "no producer" cannot land unnoticed.

## The schema (documented for humans in `cuda-runs/README.md`)

Every `*.json` under `cuda-runs/` (including every `*-raw-runs/` leg file)
must carry:

  - `schema_version` (int)
  - `git_sha` (40-hex, lowercase) — OR, for a reviewed legacy artifact only,
    `git_sha_unresolved` (whatever short/malformed ref the pod session saw)
    paired with `producer.kind == "none"`.
  - `box` (string — the physical/pod identifier the run measured on)
  - `producer` — `{path, kind, invocation, gating}`:
      - `kind`: `"cargo-test"` (a single `#[test] fn`, invoked with
        `--exact <fn>`) | `"script"` (a tracked producer script) | `"none"`
        (reviewed legacy artifact — see `LEGACY_NONE_ALLOWLIST` below; a NEW
        file may never default to this).
      - `gating`: `"#[ignore]"` | `"env:<VAR>"` | `"required-features"` |
        `"none"` — how the named test/script stays out of a plain
        `cargo test`/CI run (mirrors `docs/maintainer/cuda-kernel-guide.md`
        §5's "no CI lane has a GPU" constraint).
  - `status` (string)
  - `merged_as` (40-hex, OPTIONAL) + `merged_via_pr` (int, OPTIONAL, required
    together with `merged_as`) — for a measured tip that was itself
    squash-merged (so `git_sha` is legitimately never an ancestor of
    anything again): the squash commit the SAME content landed on `main`
    as, and the PR that merged it. `git_sha` is kept VERBATIM (the tip that
    was actually measured); `merged_as` only ever supplements it, never
    replaces it, and is only valid alongside a resolved `git_sha` (never
    `git_sha_unresolved`).

## Fail-closed contract

  (a) Every required field is present and well-typed (including the
      `git_sha` XOR `git_sha_unresolved` split, the `git_sha_unresolved`
      ⇒ `producer.kind == "none"` consistency rule, and the
      `merged_as`/`merged_via_pr` pairing — `merged_as` requires
      `merged_via_pr` and a resolved `git_sha`, and vice versa).
  (b) `producer.path`, when non-null, exists on disk AND is `git
      ls-files`-tracked (an artifact cannot cite a producer CI's own
      checkout would not have).
  (c) `producer.kind == "cargo-test"` ⇒ the invocation names `--exact
      <fn>`, that `fn` is found by a static brace-balanced scan of
      `producer.path`, it is confirmed to sit under a `#[test]` attribute,
      and the STATED `gating` attribute genuinely appears there
      (`#[ignore]` immediately above the fn; the named env var — or the
      `cuda_device()` helper — inside the fn body; or a `required-features`
      key on the matching `[[test]]` section of the crate's `Cargo.toml`).
  (d) PASS if `git merge-base --is-ancestor <git_sha> HEAD`, OR — for a
      squash-merged tip — if `merged_as` is ALSO an ancestor of HEAD (with
      `merged_via_pr` present). Neither ancestor is a hard FAIL naming the
      guide's own sentence (never silently accepted as "was green once").
  (e) `cuda-runs/README.md`'s named producer script is itself
      `git ls-files`-tracked (the README cannot point at a file CI's
      checkout would not have either).
  (f) `producer.kind == "none"` is allowed ONLY for a path in the reviewed,
      in-script `LEGACY_NONE_ALLOWLIST` — a NEW artifact defaulting to
      `"none"` is a hard FAIL.

Run: `python3 ci/scripts/check_cuda_run_artifacts.py`
Self-test (RED cases for every rule above, on a throwaway `git init`'d
fixture repo — never the real checkout):
`python3 ci/scripts/check_cuda_run_artifacts.py --self-test`
Hermetic: reads the working tree (or an ephemeral tempdir git repo under
`--self-test`) and shells out only to `git`; no network, no cargo, no GPU.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CUDA_RUNS_DIR = REPO_ROOT / "crates" / "jammi-kernels" / "artifacts" / "cuda-runs"
README_PATH = CUDA_RUNS_DIR / "README.md"

GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
PRODUCER_KINDS = {"cargo-test", "script", "none"}
GATING_STATIC = {"#[ignore]", "required-features", "none"}
GATING_ENV_RE = re.compile(r"^env:[A-Za-z_][A-Za-z0-9_]*$")
README_PRODUCER_RE = re.compile(r"`([\w./-]*proof_artifact\.py)`")

# --------------------------------------------------------------------------- #
# LEGACY_NONE_ALLOWLIST — reviewed at gate-introduction time. Every artifact
# committed to `main` BEFORE this schema existed has no `producer` field to
# migrate honestly, so each is grandfathered here with a one-line reason. A
# NEW artifact is never added to this list; it must name a real producer.
# --------------------------------------------------------------------------- #
LEGACY_NONE_ALLOWLIST: dict[str, str] = {
    "2026-08-25-cast-w1-80f02fb-a100-sxm4.json": (
        "pre-schema artifact: the same-build forced-arm A/B was run by hand via "
        "jammi-bench finetune-step invocations on the lead's exclusive box, not a "
        "single #[test] fn or a tracked producer script."
    ),
    "2026-08-25-p1-5f29e3b-a100-sxm4.json": (
        "pre-schema artifact: produced by an earlier, untracked copy of "
        "proof_artifact.py, before this gate's PR tracked it at "
        "ci/scripts/perf/proof_artifact.py."
    ),
    "2026-08-25-p2-5932520-a100-sxm4.json": (
        "pre-schema artifact: same untracked-proof_artifact.py provenance as the "
        "p1 sibling artifact above."
    ),
    "2026-08-25-p3-e32ed90-a100-sxm4.json": (
        "pre-schema artifact: same untracked-proof_artifact.py provenance as the "
        "p1 sibling artifact above."
    ),
    "2026-08-25-p6a-eac63fd-a100-sxm4.json": (
        "pre-schema artifact: a flash_smoke execution-provenance dump captured by "
        "hand (16 test names from one binary run, not a single #[test] fn)."
    ),
}

ANCESTOR_MESSAGE = (
    "is not an ancestor of HEAD — a green artifact whose sha is not an ancestor "
    "of the branch is evidence about the ORACLE, not the code "
    "(docs/maintainer/cuda-kernel-guide.md §4)."
)


class ArtifactError(Exception):
    """Uncomputable input (parse failure, missing dir) — fails closed."""


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def git_ls_files(repo_root: Path) -> set[str]:
    proc = _run(["git", "ls-files"], repo_root)
    if proc.returncode != 0:
        raise ArtifactError(f"`git ls-files` failed in {repo_root}: {proc.stderr.strip()}")
    return set(proc.stdout.splitlines())


# --------------------------------------------------------------------------- #
# rule (a) — schema/typing
# --------------------------------------------------------------------------- #
def check_schema_types(data: dict) -> list[str]:
    failures: list[str] = []

    sv = data.get("schema_version")
    if not isinstance(sv, int) or isinstance(sv, bool):
        failures.append(f"schema_version must be an int, got {sv!r}")

    box = data.get("box")
    if not isinstance(box, str) or not box.strip():
        failures.append(f"box must be a non-empty string, got {box!r}")

    status = data.get("status")
    if not isinstance(status, str) or not status.strip():
        failures.append(f"status must be a non-empty string, got {status!r}")

    has_sha = "git_sha" in data
    has_unresolved = "git_sha_unresolved" in data
    if has_sha and has_unresolved:
        failures.append("carries BOTH git_sha and git_sha_unresolved — pick one")
    elif not has_sha and not has_unresolved:
        failures.append("missing git_sha (and no git_sha_unresolved fallback)")
    elif has_sha:
        sha = data["git_sha"]
        if not isinstance(sha, str) or not GIT_SHA_RE.match(sha):
            failures.append(f"git_sha must be 40 lowercase hex chars, got {sha!r}")
    else:
        unresolved = data["git_sha_unresolved"]
        if not isinstance(unresolved, str) or not unresolved.strip():
            failures.append(f"git_sha_unresolved must be a non-empty string, got {unresolved!r}")

    # `merged_as` / `merged_via_pr` — the optional squash-landing pair. A
    # branch tip a measurement ran on can be squash-merged, so `git_sha`
    # itself is legitimately never an ancestor of any ref again; `merged_as`
    # names the squash commit the SAME content landed on `main` as, kept
    # alongside (never instead of) the measured `git_sha`.
    has_merged_as = "merged_as" in data
    has_merged_via_pr = "merged_via_pr" in data
    if has_merged_as:
        merged_as = data["merged_as"]
        if not isinstance(merged_as, str) or not GIT_SHA_RE.match(merged_as):
            failures.append(f"merged_as must be 40 lowercase hex chars, got {merged_as!r}")
        if not has_merged_via_pr:
            failures.append("merged_as is present but merged_via_pr is missing")
        if not has_sha:
            failures.append(
                "merged_as requires git_sha (the measured tip, kept verbatim) — not valid "
                "alongside git_sha_unresolved"
            )
    if has_merged_via_pr:
        merged_via_pr = data["merged_via_pr"]
        if not isinstance(merged_via_pr, int) or isinstance(merged_via_pr, bool):
            failures.append(f"merged_via_pr must be an int, got {merged_via_pr!r}")
        if not has_merged_as:
            failures.append("merged_via_pr is present but merged_as is missing")

    producer = data.get("producer")
    if not isinstance(producer, dict):
        failures.append(f"producer must be an object, got {producer!r}")
        producer = {}
    else:
        for key in ("path", "kind", "invocation", "gating"):
            if key not in producer:
                failures.append(f"producer missing required key `{key}`")
        kind = producer.get("kind")
        if kind is not None and kind not in PRODUCER_KINDS:
            failures.append(f"producer.kind must be one of {sorted(PRODUCER_KINDS)}, got {kind!r}")
        gating = producer.get("gating")
        if gating is not None and gating not in GATING_STATIC and not (
            isinstance(gating, str) and GATING_ENV_RE.match(gating)
        ):
            failures.append(
                f"producer.gating must be '#[ignore]' | 'required-features' | 'none' | 'env:<VAR>', got {gating!r}"
            )
        for key in ("path", "invocation"):
            val = producer.get(key)
            if val is not None and not isinstance(val, str):
                failures.append(f"producer.{key} must be a string or null, got {val!r}")

    if has_unresolved and isinstance(producer, dict) and producer.get("kind") not in (None, "none"):
        failures.append(
            "git_sha_unresolved requires producer.kind == 'none' (an unresolvable sha cannot be "
            "attributed to a real producer)"
        )

    return failures


# --------------------------------------------------------------------------- #
# rule (b) — producer.path exists and is tracked
# --------------------------------------------------------------------------- #
def check_producer_path(producer: dict, repo_root: Path, tracked: set[str]) -> list[str]:
    path = producer.get("path")
    if not isinstance(path, str) or not path:
        return []
    failures: list[str] = []
    if not (repo_root / path).is_file():
        failures.append(f"producer.path `{path}` does not exist on disk")
    if path not in tracked:
        failures.append(f"producer.path `{path}` is not `git ls-files`-tracked")
    return failures


# --------------------------------------------------------------------------- #
# rule (c) — cargo-test producer static verification
# --------------------------------------------------------------------------- #
EXACT_RE = re.compile(r"--exact\s+(\S+)")


def _extract_fn_body(source: str, fn_kw_start: int) -> str:
    brace_start = source.find("{", fn_kw_start)
    if brace_start == -1:
        return ""
    depth = 0
    for i in range(brace_start, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[brace_start : i + 1]
    return source[brace_start:]


def _find_crate_root(repo_root: Path, rel_path: str) -> Path | None:
    p = (repo_root / rel_path).parent
    while True:
        if (p / "Cargo.toml").is_file():
            return p
        if p == repo_root or p.parent == p:
            return None
        p = p.parent


def _test_target_has_required_features(cargo_toml_text: str, test_stem: str) -> bool:
    blocks = re.split(r"(?m)^\[\[test\]\]\s*$", cargo_toml_text)[1:]
    for block in blocks:
        end = re.search(r"(?m)^\[", block)
        body = block[: end.start()] if end else block
        name_m = re.search(r'name\s*=\s*"([^"]+)"', body)
        if name_m and name_m.group(1) == test_stem:
            return "required-features" in body
    return False


def check_cargo_test_gating(data: dict, producer: dict, repo_root: Path) -> list[str]:
    path = producer.get("path")
    invocation = producer.get("invocation") or ""
    gating = producer.get("gating")
    if not isinstance(path, str) or not path:
        return ["producer.kind == 'cargo-test' requires a non-null producer.path"]
    if not (repo_root / path).is_file():
        return []  # rule (b) already reported the missing-file failure

    m = EXACT_RE.search(invocation)
    if not m:
        return [
            f"producer.kind == 'cargo-test' but invocation `{invocation}` lacks `--exact <fn_name>` "
            "— cannot statically verify which test this artifact proves ran"
        ]
    fn_full = m.group(1)
    fn_short = fn_full.rsplit("::", 1)[-1]

    source = (repo_root / path).read_text(encoding="utf-8", errors="replace")
    fn_re = re.compile(rf"\bfn\s+{re.escape(fn_short)}\s*\(")
    fn_m = fn_re.search(source)
    if not fn_m:
        return [f"named test fn `{fn_short}` not found by static scan in {path}"]

    fn_line_idx = source.count("\n", 0, fn_m.start())
    lines = source.splitlines()
    # Walk upward from the `fn` line while the line is blank, an attribute
    # (`#[...]`), or a doc/line comment — stop at the first line that is
    # none of those (e.g. the closing `}` of a PRECEDING, unrelated fn), so
    # a neighbour's `#[ignore]` a few lines up can never be mistaken for
    # this fn's own gating attribute.
    window_lines: list[str] = []
    i = fn_line_idx - 1
    while i >= 0:
        stripped = lines[i].strip()
        if stripped == "" or stripped.startswith("#[") or stripped.startswith("//"):
            window_lines.insert(0, lines[i])
            i -= 1
        else:
            break
    window_text = "\n".join(window_lines)

    failures: list[str] = []
    if "#[test]" not in window_text:
        failures.append(
            f"`{fn_short}` in {path} has no `#[test]` attribute in its contiguous "
            "attribute block — not confirmed to be a #[test] fn"
        )

    if gating == "#[ignore]":
        if "#[ignore]" not in window_text:
            failures.append(
                f"producer claims gating '#[ignore]' but `{fn_short}` in {path} has no "
                "#[ignore] attribute"
            )
    elif isinstance(gating, str) and gating.startswith("env:"):
        var = gating.split(":", 1)[1]
        body = _extract_fn_body(source, fn_m.start())
        if var not in body and "cuda_device(" not in body:
            failures.append(
                f"producer claims gating '{gating}' but neither `{var}` nor `cuda_device(` "
                f"appears in `{fn_short}`'s body in {path}"
            )
    elif gating == "required-features":
        crate_root = _find_crate_root(repo_root, path)
        if crate_root is None:
            failures.append(f"no Cargo.toml found above {path}; cannot verify required-features")
        else:
            test_stem = Path(path).stem
            cargo_toml = (crate_root / "Cargo.toml").read_text(encoding="utf-8", errors="replace")
            if not _test_target_has_required_features(cargo_toml, test_stem):
                failures.append(
                    f"producer claims gating 'required-features' but {crate_root / 'Cargo.toml'} "
                    f"has no `[[test]]` section named `{test_stem}` carrying `required-features`"
                )
    # gating == "none": nothing further to verify.

    return failures


# --------------------------------------------------------------------------- #
# rule (d) — ancestry
# --------------------------------------------------------------------------- #
def _is_ancestor(sha: str, repo_root: Path) -> bool:
    proc = _run(["git", "merge-base", "--is-ancestor", sha, "HEAD"], repo_root)
    return proc.returncode == 0


def check_ancestry(data: dict, repo_root: Path) -> list[str]:
    """PASS if `git_sha` is an ancestor of HEAD, OR — for a branch tip that
    was squash-merged, so `git_sha` itself can never be an ancestor of
    anything again — if `merged_as` is an ancestor of HEAD and the artifact
    also carries `git_sha` (the measured tip, kept verbatim) plus
    `merged_via_pr`. `merged_as`'s own well-typedness (40-hex, paired with
    `merged_via_pr`, only valid alongside a resolved `git_sha`) is rule (a)'s
    job (`check_schema_types`); this function only re-checks GIT_SHA_RE here
    so a malformed `merged_as` cannot be handed to `git merge-base` as a
    literal ref expression.
    """
    sha = data.get("git_sha")
    if not sha:
        return []  # git_sha_unresolved artifacts have nothing resolvable to check
    if _is_ancestor(sha, repo_root):
        return []

    merged_as = data.get("merged_as")
    merged_via_pr = data.get("merged_via_pr")
    if isinstance(merged_as, str) and GIT_SHA_RE.match(merged_as) and merged_via_pr is not None:
        if _is_ancestor(merged_as, repo_root):
            return []
        return [
            f"git_sha {sha} {ANCESTOR_MESSAGE} merged_as {merged_as} (PR #{merged_via_pr}) is "
            "ALSO not an ancestor of HEAD — the squash-landing claim does not hold either."
        ]

    return [f"git_sha {sha} {ANCESTOR_MESSAGE}"]


# --------------------------------------------------------------------------- #
# rule (e) — README's named producer is tracked
# --------------------------------------------------------------------------- #
def check_readme_producer(readme_path: Path, repo_root: Path, tracked: set[str]) -> list[str]:
    if not readme_path.is_file():
        return [f"README not found: {readme_path}"]
    text = readme_path.read_text(encoding="utf-8", errors="replace")
    m = README_PRODUCER_RE.search(text)
    if not m:
        return [
            f"{readme_path}: does not name a `...proof_artifact.py` producer in backticks — "
            "the schema doc's own producer citation is missing"
        ]
    named = m.group(1)
    if named not in tracked:
        return [f"{readme_path}: names producer `{named}` which is not `git ls-files`-tracked"]
    return []


# --------------------------------------------------------------------------- #
# rule (f) — kind == "none" only for the reviewed allow-list
# --------------------------------------------------------------------------- #
def check_none_allowlist(data: dict, relpath: str, allowlist: dict[str, str]) -> list[str]:
    producer = data.get("producer")
    if isinstance(producer, dict) and producer.get("kind") == "none" and relpath not in allowlist:
        return [
            f"producer.kind == 'none' but `{relpath}` is not in the reviewed LEGACY_NONE_ALLOWLIST — "
            "a NEW artifact must name a real producer (cargo-test or script), never default to 'none'"
        ]
    return []


# --------------------------------------------------------------------------- #
# orchestration — pure over (data, relpath, repo_root, tracked, allowlist) so
# `--self-test` can drive it against a synthetic fixture repo.
# --------------------------------------------------------------------------- #
def validate_artifact(
    data: dict,
    relpath: str,
    repo_root: Path,
    tracked: set[str],
    allowlist: dict[str, str],
) -> list[str]:
    failures = check_schema_types(data)

    producer = data.get("producer") if isinstance(data.get("producer"), dict) else {}
    failures += check_producer_path(producer, repo_root, tracked)
    if producer.get("kind") == "cargo-test":
        failures += check_cargo_test_gating(data, producer, repo_root)
    failures += check_none_allowlist(data, relpath, allowlist)
    failures += check_ancestry(data, repo_root)
    return failures


def run_gate(cuda_runs_dir: Path, repo_root: Path, allowlist: dict[str, str]) -> list[str]:
    if not cuda_runs_dir.is_dir():
        raise ArtifactError(f"cuda-runs dir not found: {cuda_runs_dir}")

    tracked = git_ls_files(repo_root)
    files = sorted(cuda_runs_dir.rglob("*.json"))
    if not files:
        raise ArtifactError(f"no *.json artifacts found under {cuda_runs_dir}")

    all_failures: list[str] = []
    for f in files:
        relpath = f.relative_to(cuda_runs_dir).as_posix()
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            all_failures.append(f"{relpath}: JSON parse error: {e}")
            continue
        if not isinstance(data, dict):
            all_failures.append(f"{relpath}: top-level JSON value is not an object")
            continue
        failures = validate_artifact(data, relpath, repo_root, tracked, allowlist)
        all_failures.extend(f"{relpath}: {msg}" for msg in failures)

    all_failures.extend(
        f"README.md: {msg}"
        for msg in check_readme_producer(cuda_runs_dir / "README.md", repo_root, tracked)
    )
    return all_failures


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()

    try:
        failures = run_gate(CUDA_RUNS_DIR, REPO_ROOT, LEGACY_NONE_ALLOWLIST)
    except ArtifactError as exc:
        print(f"cuda-run-artifacts: FAIL (uncomputable) — {exc}", file=sys.stderr)
        return 1

    if failures:
        print("cuda-run-artifacts: FAIL", file=sys.stderr)
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        print(f"\ncuda-run-artifacts: {len(failures)} finding(s).", file=sys.stderr)
        return 1

    print(
        f"cuda-run-artifacts: PASS — every *.json under "
        f"{CUDA_RUNS_DIR.relative_to(REPO_ROOT)} satisfies the schema, ancestry, and "
        "producer-provenance contract."
    )
    return 0


# --------------------------------------------------------------------------- #
# self-test — an ephemeral `git init`'d fixture repo, never the real
# checkout, proving each rule (a)-(f) actually bites.
# --------------------------------------------------------------------------- #
def self_test() -> int:
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        _run(["git", "init", "-q"], repo)
        _run(["git", "config", "user.email", "test@example.com"], repo)
        _run(["git", "config", "user.name", "Test"], repo)

        crate_dir = repo / "crates" / "fixture-crate"
        (crate_dir / "tests").mkdir(parents=True)
        (crate_dir / "tests" / "cuda_parity.rs").write_text(
            "#[test]\n"
            "#[ignore]\n"
            "fn some_gated_test() {\n"
            "    assert!(true);\n"
            "}\n"
            "\n"
            "#[test]\n"
            "fn env_gated_test() {\n"
            "    if std::env::var_os(\"JAMMI_REQUIRE_CUDA\").is_some() {\n"
            "        assert!(true);\n"
            "    }\n"
            "}\n"
        )
        (crate_dir / "Cargo.toml").write_text(
            '[package]\nname = "fixture-crate"\nversion = "0.0.0"\n\n'
            "[[test]]\n"
            'name = "cuda_parity"\n'
            'path = "tests/cuda_parity.rs"\n'
            'required-features = ["cuda"]\n'
        )

        cuda_runs = repo / "crates" / "jammi-kernels" / "artifacts" / "cuda-runs"
        cuda_runs.mkdir(parents=True)
        (cuda_runs / "README.md").write_text("# CUDA run artifacts\n\nProduced by `ci/scripts/perf/proof_artifact.py`.\n")
        perf_dir = repo / "ci" / "scripts" / "perf"
        perf_dir.mkdir(parents=True)
        (perf_dir / "proof_artifact.py").write_text("# stub producer\n")

        _run(["git", "add", "-A"], repo)
        _run(["git", "commit", "-q", "-m", "root"], repo)
        root_sha = _run(["git", "rev-parse", "HEAD"], repo).stdout.strip()

        (repo / "unrelated.txt").write_text("x\n")
        _run(["git", "add", "-A"], repo)
        _run(["git", "commit", "-q", "-m", "second"], repo)

        tracked = git_ls_files(repo)
        allowlist = {"legacy-none.json": "synthetic legacy fixture"}

        def baseline() -> dict:
            return {
                "schema_version": 1,
                "git_sha": root_sha,
                "box": "a100-fixture",
                "producer": {
                    "path": "crates/fixture-crate/tests/cuda_parity.rs",
                    "kind": "cargo-test",
                    "invocation": "cargo test -p fixture-crate --test cuda_parity -- --exact some_gated_test",
                    "gating": "#[ignore]",
                },
                "status": "GREEN",
            }

        def expect_clean(data: dict, relpath: str, label: str) -> None:
            got = validate_artifact(data, relpath, repo, tracked, allowlist)
            if got:
                failures.append(f"self-test FAILED: {label} expected clean, got {got}")

        def expect_hit(data: dict, relpath: str, needle: str, label: str) -> None:
            got = validate_artifact(data, relpath, repo, tracked, allowlist)
            if not any(needle in g for g in got):
                failures.append(f"self-test FAILED: {label} expected a finding containing {needle!r}, got {got}")

        # GREEN controls -----------------------------------------------------
        expect_clean(baseline(), "control-ignore.json", "cargo-test + #[ignore] baseline")

        env_variant = baseline()
        env_variant["producer"] = dict(env_variant["producer"])
        env_variant["producer"]["invocation"] = (
            "cargo test -p fixture-crate --test cuda_parity -- --exact env_gated_test"
        )
        env_variant["producer"]["gating"] = "env:JAMMI_REQUIRE_CUDA"
        expect_clean(env_variant, "control-env.json", "cargo-test + env:VAR baseline")

        rf_variant = baseline()
        rf_variant["producer"] = dict(rf_variant["producer"])
        rf_variant["producer"]["gating"] = "required-features"
        expect_clean(rf_variant, "control-required-features.json", "cargo-test + required-features baseline")

        none_variant = {
            "schema_version": 1,
            "git_sha_unresolved": "abc1234",
            "box": "a100-fixture",
            "producer": {"path": None, "kind": "none", "invocation": None, "gating": "none"},
            "status": "GREEN",
        }
        expect_clean(none_variant, "legacy-none.json", "allow-listed producer.kind == none baseline")

        # rule (a) — schema/typing ------------------------------------------
        bad = baseline()
        del bad["schema_version"]
        expect_hit(bad, "x.json", "schema_version must be", "rule (a): missing schema_version")

        bad = baseline()
        del bad["git_sha"]
        expect_hit(bad, "x.json", "missing git_sha", "rule (a): missing git_sha and git_sha_unresolved")

        bad = baseline()
        bad["git_sha"] = "not-hex"
        expect_hit(bad, "x.json", "git_sha must be 40", "rule (a): malformed git_sha")

        bad = baseline()
        bad["git_sha_unresolved"] = "abc1234"
        expect_hit(bad, "x.json", "BOTH git_sha and git_sha_unresolved", "rule (a): both sha fields present")

        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["kind"] = "bogus"
        expect_hit(bad, "x.json", "producer.kind must be one of", "rule (a): bogus producer.kind")

        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["gating"] = "bogus"
        expect_hit(bad, "x.json", "producer.gating must be", "rule (a): bogus gating")

        bad = {
            "schema_version": 1,
            "git_sha_unresolved": "abc1234",
            "box": "a100-fixture",
            "producer": {"path": None, "kind": "cargo-test", "invocation": None, "gating": "none"},
            "status": "GREEN",
        }
        expect_hit(bad, "x.json", "requires producer.kind == 'none'", "rule (a): unresolved sha with non-none producer")

        # rule (b) — producer.path exists + tracked ---------------------------
        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["path"] = "crates/fixture-crate/tests/does_not_exist.rs"
        expect_hit(bad, "x.json", "does not exist on disk", "rule (b): nonexistent producer.path")

        untracked_dir = crate_dir / "tests"
        (untracked_dir / "untracked.rs").write_text("#[test]\n#[ignore]\nfn some_gated_test() {}\n")
        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["path"] = "crates/fixture-crate/tests/untracked.rs"
        expect_hit(bad, "x.json", "is not `git ls-files`-tracked", "rule (b): untracked producer.path")

        # rule (c) — cargo-test static verification ---------------------------
        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["invocation"] = "cargo test -p fixture-crate --test cuda_parity"
        expect_hit(bad, "x.json", "lacks `--exact", "rule (c): invocation without --exact")

        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["invocation"] = "cargo test -p fixture-crate --test cuda_parity -- --exact no_such_fn"
        expect_hit(bad, "x.json", "not found by static scan", "rule (c): fn not found")

        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["invocation"] = "cargo test -p fixture-crate --test cuda_parity -- --exact env_gated_test"
        # env_gated_test has NO #[ignore] attribute — claiming '#[ignore]' must fail.
        expect_hit(bad, "x.json", "has no", "rule (c): claimed #[ignore] absent")

        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["invocation"] = "cargo test -p fixture-crate --test cuda_parity -- --exact some_gated_test"
        bad["producer"]["gating"] = "env:SOME_OTHER_VAR"
        # some_gated_test's body never mentions SOME_OTHER_VAR or cuda_device(.
        expect_hit(bad, "x.json", "neither `SOME_OTHER_VAR`", "rule (c): claimed env var absent from body")

        no_rf_dir = repo / "crates" / "no-rf-crate" / "tests"
        no_rf_dir.mkdir(parents=True)
        (no_rf_dir / "cuda_parity.rs").write_text("#[test]\n#[ignore]\nfn some_gated_test() {}\n")
        (repo / "crates" / "no-rf-crate" / "Cargo.toml").write_text(
            '[package]\nname = "no-rf-crate"\nversion = "0.0.0"\n\n'
            "[[test]]\n"
            'name = "cuda_parity"\n'
            'path = "tests/cuda_parity.rs"\n'
        )
        _run(["git", "add", "-A"], repo)
        _run(["git", "commit", "-q", "-m", "third"], repo)
        tracked = git_ls_files(repo)
        bad = baseline()
        bad["producer"] = dict(bad["producer"])
        bad["producer"]["path"] = "crates/no-rf-crate/tests/cuda_parity.rs"
        bad["producer"]["gating"] = "required-features"
        expect_hit(bad, "x.json", "has no `[[test]]` section", "rule (c): claimed required-features absent")

        # rule (d) — ancestry ---------------------------------------------------
        bad = baseline()
        bad["git_sha"] = "f" * 40
        expect_hit(bad, "x.json", "is not an ancestor of HEAD", "rule (d): non-ancestor git_sha")

        # GREEN control: a squash-merged tip whose OWN sha is not (and never
        # will be again) an ancestor of HEAD, but whose content landed on
        # HEAD via a real (here: the fixture repo's own) commit named by
        # merged_as + merged_via_pr.
        merged_ok = baseline()
        merged_ok["git_sha"] = "f" * 40
        merged_ok["merged_as"] = root_sha
        merged_ok["merged_via_pr"] = 363
        expect_clean(merged_ok, "control-merged-as.json", "non-ancestor git_sha rescued by an ancestor merged_as")

        # RED: merged_as ALSO not an ancestor of HEAD — the rescue must not
        # be granted just because the field is present and well-shaped.
        bad = baseline()
        bad["git_sha"] = "f" * 40
        bad["merged_as"] = "e" * 40
        bad["merged_via_pr"] = 999
        expect_hit(bad, "x.json", "is ALSO not an ancestor of HEAD", "rule (d): non-ancestor merged_as too")

        # RED: merged_as present but git_sha is NOT (only git_sha_unresolved)
        # — merged_as must never stand in for an actually-resolved git_sha.
        bad = {
            "schema_version": 1,
            "git_sha_unresolved": "abc1234",
            "box": "a100-fixture",
            "merged_as": root_sha,
            "merged_via_pr": 363,
            "producer": {"path": None, "kind": "none", "invocation": None, "gating": "none"},
            "status": "GREEN",
        }
        expect_hit(bad, "legacy-none.json", "merged_as requires git_sha", "rule (a): merged_as without a resolved git_sha")

        # RED: merged_as present without merged_via_pr, and vice versa.
        bad = baseline()
        bad["merged_as"] = root_sha
        expect_hit(bad, "x.json", "merged_as is present but merged_via_pr is missing", "rule (a): merged_as without merged_via_pr")

        bad = baseline()
        bad["merged_via_pr"] = 363
        expect_hit(bad, "x.json", "merged_via_pr is present but merged_as is missing", "rule (a): merged_via_pr without merged_as")

        # rule (e) — README's named producer is tracked -------------------------
        untracked_readme = repo / "untracked-readme-dir"
        untracked_readme.mkdir()
        (untracked_readme / "README.md").write_text(
            "Produced by `ci/scripts/perf/untracked/proof_artifact.py`.\n"
        )
        readme_failures = check_readme_producer(untracked_readme / "README.md", repo, tracked)
        if not any("not `git ls-files`-tracked" in f for f in readme_failures):
            failures.append(f"self-test FAILED: rule (e) untracked README producer not caught: {readme_failures}")

        missing_marker_readme = repo / "no-marker-dir"
        missing_marker_readme.mkdir()
        (missing_marker_readme / "README.md").write_text("No producer named here.\n")
        readme_failures2 = check_readme_producer(missing_marker_readme / "README.md", repo, tracked)
        if not any("does not name a" in f for f in readme_failures2):
            failures.append(f"self-test FAILED: rule (e) missing producer marker not caught: {readme_failures2}")

        # rule (f) — kind == none only for the allow-list ------------------------
        bad = {
            "schema_version": 1,
            "git_sha": root_sha,
            "box": "a100-fixture",
            "producer": {"path": None, "kind": "none", "invocation": None, "gating": "none"},
            "status": "GREEN",
        }
        expect_hit(bad, "brand-new-not-allowlisted.json", "not in the reviewed LEGACY_NONE_ALLOWLIST", "rule (f): new file defaulting to none")

    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        print("cuda-run-artifacts self-test: FAIL", file=sys.stderr)
        return 1
    print(
        "cuda-run-artifacts self-test: OK — every rule (a) schema/typing (including the "
        "merged_as/merged_via_pr pairing), (b) producer.path existence+tracking, (c) cargo-test "
        "static gating verification (#[ignore]/env:VAR/required-features), (d) ancestry (both the "
        "plain git_sha path and the merged_as squash-landing rescue, and its own non-ancestor RED "
        "case), (e) README producer tracking, and (f) the none-allowlist closure all bite on a "
        "throwaway fixture repo, and GREEN controls (ignore/env/required-features/merged_as-rescue) "
        "plus one allow-listed none control stay clean."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
