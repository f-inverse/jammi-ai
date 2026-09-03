#!/usr/bin/env python3
"""GPU-prove-once guard (esc-084, issue #454) — hermetic, static, no build,
no GPU, no PyYAML.

**Guarded property**: a release commit is proven ONCE per shipped arch, and
every CUDA release lane consumes that SAME verdict, never renting hardware
of its own. Operator direction (2026-09-03): the prove lane itself is a
manual dev run and must never sit in the critical path of any automated
workflow; a publisher gates on the SUMMARY of a prove execution already on
record for the commit it promotes.

Four positive-and-negative rules (F7, ask-6 — every rule has a fixture that
must PASS as well as fixtures that must FAIL, never a grep for one known-bad
string):

  P1 (exactly-once producer + the never-in-the-critical-path doctrine):
     exactly one workflow's comment-stripped step body invokes
     `ci/scripts/runpod_gpu_prove.sh`, and it is `gpu-prove.yml`; that
     workflow's own `on:` block carries neither `push:` nor `workflow_call:`
     (an unreadable — quoted or flow-style — `on:` block is itself a FAIL,
     never a silent skip); no OTHER workflow `uses:` it, local
     (`./.github/workflows/gpu-prove.yml`) or cross-repo
     (`<owner>/<repo>/.github/workflows/gpu-prove.yml@<ref>`) form.

  P2 (no renting reusable): no workflow `uses:` a `_gpu-prove-gate.yml`
     (any path), and that file must not exist in the tree at all.

  P3 (manifest lane -> promotion chain, reconciled both ways): every
     `ci/release-feature-manifest.json` lane whose `cargo_features` declares
     `cuda` has a reviewed row in `LANE_TABLE` naming (workflow, promoting
     job, gate job); the gate job `uses: ./.github/workflows/
     _gpu-proof-required.yml`; the promoting job's `needs:` lists the gate
     job; and the promoting job's `if:` is a PURE top-level conjunction
     (amendment O/X: parenthesis- and quote-aware, never a substring check)
     containing the exact conjunct `needs.<gate>.result == 'success'` and NO
     depth-0 `||` anywhere. A table row naming a lane the manifest no longer
     declares, a lane with no row, or a missing workflow file are each a
     FAIL, never a silent skip.

  P4 (consumer/producer name agreement): `gpu_prove_verdict.py`'s
     `JOB_NAME_TEMPLATE` matches `gpu-prove.yml`'s own matrix job `name:`
     line, and the required-arch set the consumer derives
     (`check_gpu_parity_matrix.py`'s `GENCODE_ARCHES` parser) equals the
     workflow's own matrix `arch:` list — never a hand-typed list on either
     side.

Mechanism: comment-stripped line scan plus a minimal indentation-based
`jobs:` block splitter (no PyYAML, this repo's own gate convention). Every
check function takes an explicit `workflows_dir`/`manifest_path` so
`test_check_gpu_prove_once.py` can drive them against synthetic fixture
trees, including a fixture reproducing TODAY's (pre-fix) shape.

Run: `python3 ci/scripts/check_gpu_prove_once.py`
Self-test: `python3 ci/scripts/test_check_gpu_prove_once.py`
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
MANIFEST_PATH = REPO_ROOT / "ci" / "release-feature-manifest.json"

sys.path.insert(0, str(REPO_ROOT / "ci" / "scripts"))
import check_gpu_parity_matrix as gpu_parity_matrix  # noqa: E402
import gpu_prove_verdict  # noqa: E402

PROVE_SCRIPT = "runpod_gpu_prove.sh"
PROVE_PRODUCER_WORKFLOW = "gpu-prove.yml"
GATE_WORKFLOW = "_gpu-prove-gate.yml"  # the DELETED renting reusable -- must stay gone.
PROOF_REQUIRED_WORKFLOW = "_gpu-proof-required.yml"

# P3's reviewed table -- WHICH job promotes WHICH manifest lane is a design
# decision, never recoverable from the YAML alone, so this is hand-
# maintained and reconciled against the manifest both ways by check_p3().
LANE_TABLE: dict[str, tuple[str, str, str]] = {
    "cu12-image": ("server-image.yml", "build-and-push-cu12", "gpu-proof"),
    "cu12-tarball": ("release-binaries.yml", "server-cu12-promote", "gpu-proof"),
    "cu12-wheel": ("pypi-server-cuda.yml", "publish", "gpu-proof"),
}

_USES_LOCAL_RE = re.compile(r"uses:\s*\./\.github/workflows/([A-Za-z0-9_.-]+)")
_USES_CROSS_REPO_RE = re.compile(r"uses:\s*[\w.-]+/[\w.-]+/\.github/workflows/([A-Za-z0-9_.-]+)@")


# --------------------------------------------------------------------------- #
# Line-level helpers (comment-vs-code, the same rule check_ci_guard_wiring.py
# and check_execution_surface_reachability.py both already apply).
# --------------------------------------------------------------------------- #
def drop_comment_lines(text: str) -> str:
    return "\n".join("" if line.strip().startswith("#") else line for line in text.splitlines())


def load_workflow_texts(workflows_dir: Path) -> dict[str, str]:
    if not workflows_dir.is_dir():
        return {}
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(workflows_dir.glob("*.yml"))}


# --------------------------------------------------------------------------- #
# jobs: block splitter -- minimal, indentation-based (no PyYAML).
# --------------------------------------------------------------------------- #
def split_top_level_jobs(text: str) -> dict[str, tuple[int, int]]:
    """{job_id: (start_line, end_line_exclusive)} of RAW 0-based line
    indices for each job directly under a top-level `jobs:` at 2-space
    indent. A job's body runs from its own header line through the line
    before the next 2-space-indented key, or EOF."""
    lines = text.splitlines()
    jobs_line = None
    for i, line in enumerate(lines):
        if line.rstrip() == "jobs:":
            jobs_line = i
            break
    if jobs_line is None:
        return {}
    job_re = re.compile(r"^  ([A-Za-z0-9_.-]+):\s*(#.*)?$")
    starts: list[tuple[str, int]] = []
    for i in range(jobs_line + 1, len(lines)):
        line = lines[i]
        if line.strip() == "" or line.lstrip().startswith("#"):
            continue
        if re.match(r"^\S", line):  # dedent back to a top-level (0-indent) key -- jobs: block ended
            break
        m = job_re.match(line)
        if m:
            starts.append((m.group(1), i))
    jobs: dict[str, tuple[int, int]] = {}
    for idx, (name, start) in enumerate(starts):
        end = starts[idx + 1][1] if idx + 1 < len(starts) else len(lines)
        jobs[name] = (start, end)
    return jobs


# --------------------------------------------------------------------------- #
# `if:` expression reconstitution + top-level conjunction scanner
# (amendment O/X).
# --------------------------------------------------------------------------- #
_IF_KEY_RE = re.compile(r"^    if:(.*)$")
_BLOCK_SCALAR_HEADS = {">", ">-", "|", "|-"}


def reconstruct_if_expr(lines: list[str], job_start: int, job_end: int) -> tuple[str | None, str | None]:
    """(expr, error). `expr` is `None` with `error` `None` when the job
    carries no `if:` at all (not itself an error). `error` is set (expr
    `None`) when a found `if:` cannot be read in full -- an unterminated
    block scalar, an empty inline value with no recognized block form."""
    for i in range(job_start, job_end):
        line = lines[i]
        if line.strip().startswith("#"):
            continue
        m = _IF_KEY_RE.match(line)
        if not m:
            continue
        rest = m.group(1).strip()
        if rest in _BLOCK_SCALAR_HEADS:
            body: list[str] = []
            j = i + 1
            while j < job_end:
                bl = lines[j]
                if bl.strip() == "":
                    j += 1
                    continue
                indent = len(bl) - len(bl.lstrip(" "))
                if indent <= 4:
                    break
                if not bl.strip().startswith("#"):
                    body.append(bl.strip())
                j += 1
            if not body:
                return None, f"if: (line {i + 1}): block scalar `{rest}` has no body -- unterminated block"
            expr = " ".join(body)
        elif rest == "":
            return None, f"if: (line {i + 1}): empty inline value, not a recognized block scalar"
        else:
            expr = rest
        if expr.startswith("${{") and expr.endswith("}}"):
            expr = expr[3:-2].strip()
        if len(expr) >= 2 and expr[0] == expr[-1] and expr[0] in ("'", '"'):
            expr = expr[1:-1]
        return expr, None
    return None, None


def split_top_level(expr: str) -> tuple[list[str], bool]:
    """Quote-aware (single-quoted GitHub strings, `''` escapes), paren-
    depth-aware split on top-level `&&`/`||`. Returns (tokens, balanced) --
    tokens alternate operand/operator/operand/...; `balanced` is False
    (FAIL LOUD, never silently accepted) if paren depth never returns to
    zero or a string is left open at end of input."""
    depth = 0
    in_str = False
    i = 0
    n = len(expr)
    tokens: list[str] = []
    buf: list[str] = []
    while i < n:
        c = expr[i]
        if in_str:
            if c == "'":
                if i + 1 < n and expr[i + 1] == "'":
                    buf.append("''")
                    i += 2
                    continue
                in_str = False
                buf.append(c)
                i += 1
                continue
            buf.append(c)
            i += 1
            continue
        if c == "'":
            in_str = True
            buf.append(c)
            i += 1
            continue
        if c == "(":
            depth += 1
            buf.append(c)
            i += 1
            continue
        if c == ")":
            depth -= 1
            buf.append(c)
            i += 1
            continue
        if depth == 0 and expr[i : i + 2] == "&&":
            tokens.append("".join(buf).strip())
            buf = []
            tokens.append("&&")
            i += 2
            continue
        if depth == 0 and expr[i : i + 2] == "||":
            tokens.append("".join(buf).strip())
            buf = []
            tokens.append("||")
            i += 2
            continue
        buf.append(c)
        i += 1
    tokens.append("".join(buf).strip())
    return tokens, (depth == 0 and not in_str)


def normalize_conjunct(c: str) -> str:
    c = re.sub(r"\s+", " ", c.strip())
    c = re.sub(r"\s*==\s*", " == ", c)
    return c


def gate_conjunct(gate_job: str) -> str:
    return f"needs.{gate_job}.result == 'success'"


def check_promoting_if(expr: str, gate_job: str) -> list[str]:
    """The pure-top-level-conjunction rule for one promoting job's already-
    reconstituted `if:` expression. Returns a (possibly empty) findings
    list -- never raises on a malformed-but-parseable expression (a
    genuinely unreadable one is handled by the caller via
    `reconstruct_if_expr`'s own error return)."""
    tokens, balanced = split_top_level(expr)
    if not balanced:
        return [f"if: `{expr}` has unbalanced parens/an unterminated string -- refusing to analyze a truncated expression"]
    operators = [t for t in tokens if t in ("&&", "||")]
    conjuncts = [t for t in tokens if t not in ("&&", "||")]
    if "||" in operators:
        return [f"if: `{expr}` carries a depth-0 `||` -- a promoting job's condition must be a PURE conjunction"]
    want = gate_conjunct(gate_job)
    normalized = [normalize_conjunct(c) for c in conjuncts]
    if want not in normalized:
        return [f"if: `{expr}` has no top-level conjunct equal to `{want}`"]
    return []


# --------------------------------------------------------------------------- #
# on: block reader (amendment Y).
# --------------------------------------------------------------------------- #
def read_top_level_on_block(text: str) -> tuple[list[str] | None, str | None]:
    """(trigger_keys, error). A quoted `"on":`/`'on':` or flow-style
    `on: {...}` is a "cannot read" FAIL, never a silent pass (amendment Y).
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if re.match(r'^"on":', stripped) or re.match(r"^'on':", stripped):
            return None, 'on: block is quoted ("on": / \'on\':) -- cannot read'
        if not line.startswith("on:"):
            continue
        rest = line[len("on:") :].strip()
        if rest == "":
            keys: list[str] = []
            for j in range(i + 1, len(lines)):
                l2 = lines[j]
                if l2.strip() == "" or l2.strip().startswith("#"):
                    continue
                if re.match(r"^\S", l2):
                    break
                m2 = re.match(r"^  ([A-Za-z0-9_]+):", l2)
                if m2:
                    keys.append(m2.group(1))
                elif not re.match(r"^  ", l2):
                    break
            return keys, None
        if rest.startswith("{") or rest.startswith("["):
            return None, "on: is flow-style -- cannot read"
        return [rest], None
    return None, "no top-level on: block found"


# --------------------------------------------------------------------------- #
# P1 + P2
# --------------------------------------------------------------------------- #
def check_p1_p2(workflow_texts: dict[str, str]) -> list[str]:
    findings: list[str] = []

    producers = sorted(
        name for name, text in workflow_texts.items() if PROVE_SCRIPT in drop_comment_lines(text)
    )
    if not producers:
        findings.append(f"P1: zero workflows invoke {PROVE_SCRIPT} -- the prove lane was deleted everywhere")
    else:
        if PROVE_PRODUCER_WORKFLOW not in producers:
            findings.append(
                f"P1: {PROVE_SCRIPT} is invoked by {producers}, none of which is {PROVE_PRODUCER_WORKFLOW}"
            )
        extra = [p for p in producers if p != PROVE_PRODUCER_WORKFLOW]
        if extra:
            findings.append(
                f"P1: {PROVE_SCRIPT} is invoked by more than one workflow ({producers}) -- "
                f"only {PROVE_PRODUCER_WORKFLOW} may; extra site(s): {extra}"
            )

    prove_text = workflow_texts.get(PROVE_PRODUCER_WORKFLOW)
    if prove_text is None:
        findings.append(f"P1: {PROVE_PRODUCER_WORKFLOW} is missing from the workflow tree")
    else:
        keys, err = read_top_level_on_block(prove_text)
        if err is not None:
            findings.append(f"P1: {PROVE_PRODUCER_WORKFLOW}: {err}")
        else:
            bad_triggers = [k for k in (keys or []) if k in ("push", "workflow_call")]
            if bad_triggers:
                findings.append(
                    f"P1: {PROVE_PRODUCER_WORKFLOW}'s on: block carries {bad_triggers} -- the prove lane "
                    "must never auto-start on a push or be `uses:`-callable (operator direction: never in "
                    "the critical path of an automated workflow)"
                )

    for name, text in workflow_texts.items():
        if name == PROVE_PRODUCER_WORKFLOW:
            continue
        stripped = drop_comment_lines(text)
        for m in _USES_LOCAL_RE.finditer(stripped):
            if m.group(1) == PROVE_PRODUCER_WORKFLOW:
                findings.append(f"P1: {name} `uses:` {PROVE_PRODUCER_WORKFLOW} -- nothing may call the prove lane")
        for m in _USES_CROSS_REPO_RE.finditer(stripped):
            if m.group(1) == PROVE_PRODUCER_WORKFLOW:
                findings.append(
                    f"P1: {name} `uses:` a cross-repo reference to {PROVE_PRODUCER_WORKFLOW} -- "
                    "nothing may call the prove lane"
                )
        if GATE_WORKFLOW in stripped:
            findings.append(f"P2: {name} references the deleted renting reusable {GATE_WORKFLOW}")

    return findings


def check_gate_file_absent(workflows_dir: Path) -> list[str]:
    if (workflows_dir / GATE_WORKFLOW).exists():
        return [f"P2: {GATE_WORKFLOW} still exists -- the renting reusable must be deleted, not merely unused"]
    return []


# --------------------------------------------------------------------------- #
# P3
# --------------------------------------------------------------------------- #
def cuda_lanes(manifest: dict) -> set[str]:
    return {
        lane
        for lane, spec in manifest.get("lanes", {}).items()
        if "cuda" in spec.get("cargo_features", [])
    }


def check_p3(workflow_texts: dict[str, str], manifest: dict) -> list[str]:
    findings: list[str] = []
    manifest_lanes = cuda_lanes(manifest)
    table_lanes = set(LANE_TABLE)

    for lane in sorted(manifest_lanes - table_lanes):
        findings.append(f"P3: manifest CUDA lane `{lane}` has no LANE_TABLE row (no promotion chain wired)")
    for lane in sorted(table_lanes - manifest_lanes):
        findings.append(f"P3: LANE_TABLE row `{lane}` names a lane absent from the manifest")

    for lane in sorted(table_lanes & manifest_lanes):
        workflow, promoting_job, gate_job = LANE_TABLE[lane]
        text = workflow_texts.get(workflow)
        if text is None:
            findings.append(f"P3: lane `{lane}`: workflow file {workflow} is missing")
            continue
        jobs = split_top_level_jobs(text)
        lines = text.splitlines()

        gate_range = jobs.get(gate_job)
        if gate_range is None:
            findings.append(f"P3: lane `{lane}`: {workflow} has no job `{gate_job}`")
        else:
            gate_body = drop_comment_lines("\n".join(lines[gate_range[0] : gate_range[1]]))
            if f"uses: ./.github/workflows/{PROOF_REQUIRED_WORKFLOW}" not in gate_body:
                findings.append(
                    f"P3: lane `{lane}`: {workflow}'s gate job `{gate_job}` does not "
                    f"`uses: ./.github/workflows/{PROOF_REQUIRED_WORKFLOW}`"
                )

        promo_range = jobs.get(promoting_job)
        if promo_range is None:
            findings.append(f"P3: lane `{lane}`: {workflow} has no job `{promoting_job}`")
            continue
        promo_body = drop_comment_lines("\n".join(lines[promo_range[0] : promo_range[1]]))
        needs_names: list[str] = []
        needs_m = re.search(r"^\s*needs:\s*(.*)$", promo_body, re.MULTILINE)
        if needs_m and needs_m.group(1).strip():
            rest = needs_m.group(1).strip()
            if rest.startswith("["):
                needs_names = [x.strip() for x in rest.strip("[]").split(",") if x.strip()]
            else:
                needs_names = [rest]
        elif needs_m:
            # multi-line `needs:` list form (`- x` items right after the key,
            # with no value on the `needs:` line itself).
            promo_lines = promo_body.splitlines()
            for idx, l2 in enumerate(promo_lines):
                if re.match(r"^\s*needs:\s*$", l2):
                    for l3 in promo_lines[idx + 1 :]:
                        m3 = re.match(r"^\s*-\s*([A-Za-z0-9_.-]+)\s*$", l3)
                        if m3:
                            needs_names.append(m3.group(1))
                        else:
                            break
                    break
        if gate_job not in needs_names:
            findings.append(
                f"P3: lane `{lane}`: {workflow}'s promoting job `{promoting_job}` does not `needs:` `{gate_job}` "
                f"(found needs={needs_names})"
            )

        expr, err = reconstruct_if_expr(lines, promo_range[0], promo_range[1])
        if err is not None:
            findings.append(f"P3: lane `{lane}`: {workflow}'s promoting job `{promoting_job}`: {err}")
        elif expr is None:
            findings.append(f"P3: lane `{lane}`: {workflow}'s promoting job `{promoting_job}` has no `if:` at all")
        else:
            findings.extend(
                f"P3: lane `{lane}`: {workflow}'s promoting job `{promoting_job}`: {f}"
                for f in check_promoting_if(expr, gate_job)
            )

    return findings


# --------------------------------------------------------------------------- #
# P4
# --------------------------------------------------------------------------- #
def check_p4(workflow_texts: dict[str, str], shipped_arches: set[str]) -> list[str]:
    findings: list[str] = []
    text = workflow_texts.get(PROVE_PRODUCER_WORKFLOW)
    if text is None:
        return [f"P4: {PROVE_PRODUCER_WORKFLOW} is missing -- cannot verify job-name/arch agreement"]

    name_lines = [l for l in text.splitlines() if re.match(r"^\s*name:\s*GPU prove on RunPod", l)]
    want_name = "name: " + gpu_prove_verdict.JOB_NAME_TEMPLATE.format(arch="${{ matrix.arch }}")
    if not any(l.strip() == want_name for l in name_lines):
        findings.append(
            f"P4: {PROVE_PRODUCER_WORKFLOW}'s matrix job `name:` does not match "
            f"gpu_prove_verdict.JOB_NAME_TEMPLATE (want `{want_name}`, found {name_lines})"
        )

    arch_m = re.search(r"arch:\s*\[([^\]]*)\]", text)
    if arch_m is None:
        findings.append(f"P4: {PROVE_PRODUCER_WORKFLOW} has no `arch: [...]` matrix list")
    else:
        workflow_arches = {a.strip() for a in arch_m.group(1).split(",") if a.strip()}
        if workflow_arches != shipped_arches:
            findings.append(
                f"P4: {PROVE_PRODUCER_WORKFLOW}'s matrix arch list {sorted(workflow_arches)} != "
                f"shipped GENCODE_ARCHES {sorted(shipped_arches)}"
            )
    return findings


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_gate(
    workflows_dir: Path = WORKFLOWS_DIR,
    manifest_path: Path = MANIFEST_PATH,
) -> list[str]:
    workflow_texts = load_workflow_texts(workflows_dir)
    if not workflow_texts:
        return ["no workflow files found -- cannot verify the gpu-prove-once property"]
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    findings: list[str] = []
    findings += check_p1_p2(workflow_texts)
    findings += check_gate_file_absent(workflows_dir)
    findings += check_p3(workflow_texts, manifest)
    findings += check_p4(workflow_texts, gpu_parity_matrix.load_shipped_cuda_silicon())
    return findings


def main() -> int:
    findings = run_gate()
    if findings:
        print("gpu-prove-once: FAIL", file=sys.stderr)
        for f in findings:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("gpu-prove-once: OK -- exactly one prove producer, no renting reusable, every CUDA lane's "
          "promotion gates on the shared verdict, consumer/producer names agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
