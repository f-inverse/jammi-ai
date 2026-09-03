#!/usr/bin/env python3
"""GPU-prove-once guard (esc-084, issue #454) — hermetic, static, no build,
no GPU, no PyYAML.

**Guarded property**: a release commit is proven ONCE per shipped arch, and
every CUDA release lane consumes that SAME verdict, never renting hardware
of its own. The prove lane itself is a manual dev run, never in the
critical path of any automated workflow -- see `gpu-prove.yml`'s own header
for the canonical statement of why; a publisher gates on the SUMMARY of a
prove execution already on record for the commit it promotes.

Five positive-and-negative rules (F7, ask-6 — every rule has a fixture that
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
     (parenthesis- and quote-aware structural scan, never a substring check)
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

  P5 (the reusable actually consults the verdict, BLOCK B8 audit fix): P3
     only checks a gate job's `uses:` line, so gutting
     `_gpu-proof-required.yml` to `run: echo ok` would otherwise leave P1-P4
     green with no real promotion gate behind it. `_gpu-proof-required.yml`
     must exist, its `on:` block must be `workflow_call`-only, and its
     comment-stripped step body must invoke `python3 ci/scripts/
     gpu_prove_verdict.py` with `--sha` bound to the commit being promoted
     (`github.sha`/`$GITHUB_SHA`) — a literal sha or a tag name FAILS.

Mechanism: comment-stripped line scan plus a minimal indentation-based
`jobs:` block splitter (no PyYAML, this repo's own gate convention). Every
check function takes an explicit `workflows_dir`/`manifest_path` so
`test_check_gpu_prove_once.py` can drive them against synthetic fixture
trees, including a fixture reproducing the PRE-FIX shape (esc-084: three
publishers `uses:` a renting reusable).

Disclosed limit (advisory A8): `LANE_TABLE` is a hand-REVIEWED table, not
derived from the workflow tree — P3 reconciles it against the manifest's
own CUDA lane set both ways, but a brand-NEW promoting job added inside an
EXISTING lane's workflow (rather than replacing the lane's one reviewed
row) is invisible to P3 unless `LANE_TABLE` is updated by hand alongside
it. This gate is a reviewed cross-check, not a discovery mechanism.

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
    # BLOCK B7 audit fix: GitHub Actions runs BOTH `.yml` and `.yaml`
    # workflow files -- a `*.yml`-only glob is blind to a second producer, a
    # renting reusable, or a `uses:` reference hiding under the `.yaml`
    # spelling. Glob both, deduplicated, sorted for deterministic iteration
    # (the same discipline `check_execution_surface_reachability.py`
    # already applies to its own workflow scan).
    if not workflows_dir.is_dir():
        return {}
    paths = sorted(set(workflows_dir.glob("*.yml")) | set(workflows_dir.glob("*.yaml")))
    return {p.name: p.read_text(encoding="utf-8") for p in paths}


def _workflow_name_variants(name: str) -> list[str]:
    """GitHub treats `.yml` and `.yaml` as the same workflow-file family;
    `LANE_TABLE` and `GATE_WORKFLOW` are hand-maintained with a canonical
    `.yml` spelling, so a lookup against the actually-discovered
    `workflow_texts` (BLOCK B7) must try both spellings rather than assume
    the file on disk matches the constant's own extension literally."""
    if name.endswith(".yml"):
        return [name, name[: -len(".yml")] + ".yaml"]
    if name.endswith(".yaml"):
        return [name, name[: -len(".yaml")] + ".yml"]
    return [name]


def resolve_workflow(workflow_texts: dict[str, str], name: str) -> str | None:
    """The discovered key in `workflow_texts` matching `name` under either
    the `.yml` or `.yaml` spelling, or `None` if neither is present."""
    for variant in _workflow_name_variants(name):
        if variant in workflow_texts:
            return variant
    return None


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
# `if:` expression reconstitution + top-level conjunction scanner (P3's
# structural, parenthesis- and quote-aware rule -- never a substring check).
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
# on: block reader (P1's fail-loud-on-unreadable rule).
# --------------------------------------------------------------------------- #
def read_top_level_on_block(text: str) -> tuple[list[str] | None, str | None]:
    """(trigger_keys, error). A quoted `"on":`/`'on':` or flow-style
    `on: {...}` is a "cannot read" FAIL, never a silent pass.
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
                    f"P1: {PROVE_PRODUCER_WORKFLOW}'s on: block carries {bad_triggers} -- the RULE this "
                    f"gate holds {PROVE_PRODUCER_WORKFLOW} to: the prove workflow carries no push:/"
                    "workflow_call: trigger and no workflow uses: it"
                )

    # BLOCK B7 audit fix: a `uses:` (or a bare GATE_WORKFLOW reference) can
    # name either the `.yml` or the `.yaml` spelling of the target file --
    # both must be caught, never just the constant's own literal extension.
    prove_producer_variants = set(_workflow_name_variants(PROVE_PRODUCER_WORKFLOW))
    gate_workflow_variants = _workflow_name_variants(GATE_WORKFLOW)

    # F2 audit fix (issue #454 round-2): the skip below used to exempt EVERY
    # workflow whose file NAME matched a producer-name spelling
    # (`gpu-prove.yml`/`gpu-prove.yaml`) from the `uses:` scan -- so a
    # sibling file literally named `gpu-prove.yaml` that itself `uses:
    # ./.github/workflows/gpu-prove.yml` passed with zero findings, because
    # ITS OWN name matched the skip set even though it is not the resolved
    # producer. The skip must exempt only the resolved producer file that
    # actually invokes `runpod_gpu_prove.sh` (computed above as
    # `producers`), never both name spellings unconditionally. The real
    # `gpu-prove.yml`'s only `uses:` is `actions/checkout@v4`, so it never
    # self-matches these patterns and needs no skip at all in practice.
    resolved_producer = producers[0] if len(producers) == 1 and producers[0] == PROVE_PRODUCER_WORKFLOW else None

    for name, text in workflow_texts.items():
        if resolved_producer is not None and name == resolved_producer:
            continue
        stripped = drop_comment_lines(text)
        for m in _USES_LOCAL_RE.finditer(stripped):
            if m.group(1) in prove_producer_variants:
                findings.append(f"P1: {name} `uses:` {m.group(1)} -- nothing may call the prove lane")
        for m in _USES_CROSS_REPO_RE.finditer(stripped):
            if m.group(1) in prove_producer_variants:
                findings.append(
                    f"P1: {name} `uses:` a cross-repo reference to {m.group(1)} -- "
                    "nothing may call the prove lane"
                )
        for gate_variant in gate_workflow_variants:
            if gate_variant in stripped:
                findings.append(f"P2: {name} references the deleted renting reusable {gate_variant}")

    return findings


def check_gate_file_absent(workflows_dir: Path) -> list[str]:
    # BLOCK B7 audit fix: check both spellings -- a resurrected
    # `_gpu-prove-gate.yaml` is exactly as real to GitHub as the `.yml` form.
    findings: list[str] = []
    for variant in _workflow_name_variants(GATE_WORKFLOW):
        if (workflows_dir / variant).exists():
            findings.append(f"P2: {variant} still exists -- the renting reusable must be deleted, not merely unused")
    return findings


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
        # BLOCK B7 audit fix: LANE_TABLE's workflow name is a canonical
        # `.yml` literal; resolve either spelling against what was actually
        # discovered on disk.
        resolved_name = resolve_workflow(workflow_texts, workflow)
        text = workflow_texts.get(resolved_name) if resolved_name is not None else None
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
        # Advisory A7 fix: `[ \t]*`, never `\s*`, right after `needs:` --
        # `\s` matches `\n` too, so `\s*` would swallow the newline AND the
        # next line's leading whitespace when `needs:` carries no inline
        # value, landing the cursor on the multi-line list's FIRST `- item`
        # and letting `(.*)$` capture `- gpu-proof` as a bogus single
        # literal "needs name" (dash and all) instead of falling through to
        # the multi-line-list branch below.
        needs_m = re.search(r"^[ \t]*needs:[ \t]*(.*)$", promo_body, re.MULTILINE)
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
# P5 (BLOCK B8 audit fix, round-2 F1 hardening): the reusable actually
# CONSULTS the verdict, as an un-bypassable step -- never a whole-file
# substring check.
# --------------------------------------------------------------------------- #
_SHA_ARG_RE = re.compile(r'--sha\s+(?:"(?P<q>[^"]*)"|(?P<u>\$\{\{[^}]*\}\}|\S+))')
_GPU_PROVE_VERDICT_INVOCATION = "python3 ci/scripts/gpu_prove_verdict.py"
_CONTROL_OPERATOR_RE = re.compile(r"\|\||;|&&")


def _sha_arg_is_commit_bound(value: str) -> bool:
    """`True` only for the two shapes that key the verdict by the exact
    commit a caller promotes: the bash env var `$GITHUB_SHA`, or the GitHub
    expression `${{ github.sha }}` (any internal whitespace). A literal sha
    or a tag name (`v1.2.3`, `${{ github.ref_name }}`, ...) is REFUSED --
    proof surface == shipped surface (esc-081/esc-084) means the verdict
    lookup itself must be bound to the identity being promoted, never a
    sibling ref."""
    value = value.strip()
    if value == "$GITHUB_SHA":
        return True
    if value.startswith("${{") and value.endswith("}}"):
        return value[3:-2].strip() == "github.sha"
    return False


def _find_step_ranges(lines: list[str], job_start: int, job_end: int) -> list[tuple[int, int]]:
    """Raw 0-based (start, end_exclusive) line ranges for each `- ` list
    item directly under this job's `steps:` key. Bullet column is taken
    from the FIRST bullet found after `steps:`; a dedent below that column
    (or a non-bullet line at that column) ends the list."""
    steps_line = None
    for i in range(job_start, job_end):
        if re.match(r"^\s*steps:\s*(#.*)?$", lines[i]):
            steps_line = i
            break
    if steps_line is None:
        return []
    bullet_col: int | None = None
    starts: list[int] = []
    end_of_list = job_end
    for i in range(steps_line + 1, job_end):
        line = lines[i]
        if line.strip() == "" or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.lstrip(" ")
        if bullet_col is None:
            bullet_col = indent
        if indent < bullet_col:
            end_of_list = i
            break
        if indent == bullet_col:
            if not stripped.startswith("-"):
                end_of_list = i
                break
            starts.append(i)
    if not starts:
        return []
    ranges: list[tuple[int, int]] = []
    for idx, s in enumerate(starts):
        e = starts[idx + 1] if idx + 1 < len(starts) else end_of_list
        ranges.append((s, e))
    return ranges


def _parse_step_keys(lines: list[str], s: int, e: int) -> dict[str, str]:
    """{key: value_text} of every top-level key directly inside this one
    step (the `- ` list item spanning [s, e)), whether the first key sits
    inline on the dash line (`- name: Foo`) or the dash is bare and the
    first key follows on its own line. A block-scalar value (`>`/`>-`/
    `|`/`|-`) is expanded to its full joined body (newline-separated, each
    physical line's own indentation stripped) so a multi-line `run:` is
    inspected whole, not just its `run: |` header line."""
    dash_line = lines[s]
    bullet_col = len(dash_line) - len(dash_line.lstrip(" "))
    after_dash = dash_line[bullet_col + 1 :]
    after_dash_stripped = after_dash.lstrip(" ")
    scan: list[tuple[int, str]] = []
    if after_dash_stripped.strip() != "":
        scan.append((s, after_dash_stripped))
    for i in range(s + 1, e):
        scan.append((i, lines[i]))

    # Key column: either the inline dash-line key's own column, or the
    # first subsequent line's indentation (bare-dash form).
    if after_dash_stripped.strip() != "":
        key_col = bullet_col + 1 + (len(after_dash) - len(after_dash_stripped))
    else:
        key_col = None
        for li, text in scan:
            if text.strip() == "" or text.strip().startswith("#"):
                continue
            key_col = len(text) - len(text.lstrip(" "))
            break
        if key_col is None:
            return {}

    results: dict[str, str] = {}
    idx = 0
    n = len(scan)
    while idx < n:
        li, text = scan[idx]
        is_inline_dash = li == s and after_dash_stripped.strip() != ""
        if text.strip() == "" or text.strip().startswith("#"):
            idx += 1
            continue
        if is_inline_dash:
            content = text
        else:
            indent = len(text) - len(text.lstrip(" "))
            if indent != key_col:
                idx += 1
                continue
            content = text.strip()
        m = re.match(r"^([A-Za-z0-9_.-]+):\s*(.*)$", content)
        if not m:
            idx += 1
            continue
        key, val = m.group(1), m.group(2).strip()
        if val in _BLOCK_SCALAR_HEADS:
            body: list[str] = []
            j = idx + 1
            while j < n:
                bli, btext = scan[j]
                if btext.strip() == "":
                    j += 1
                    continue
                bindent = len(btext) - len(btext.lstrip(" "))
                if bindent <= key_col:
                    break
                if not btext.strip().startswith("#"):
                    body.append(btext.strip())
                j += 1
            results[key] = "\n".join(body)
            idx = j
            continue
        results[key] = val
        idx += 1
    return results


def _join_shell_continuations(text: str) -> list[str]:
    """Logical (backslash-continuation-joined) lines of a shell `run:`
    body -- each physical line ending in a trailing `\\` is folded onto the
    next, so a multi-line invocation's arguments become one line to scan
    for a trailing control operator or the LAST `--sha`."""
    logical: list[str] = []
    buf = ""
    for line in text.splitlines():
        piece = line.strip()
        buf = f"{buf} {piece}".strip() if buf else piece
        if buf.endswith("\\"):
            buf = buf[:-1].rstrip()
            continue
        logical.append(buf)
        buf = ""
    if buf:
        logical.append(buf)
    return logical


def check_p5(workflow_texts: dict[str, str]) -> list[str]:
    """P3 only checks a gate job's `uses:` line -- gutting
    `_gpu-proof-required.yml` to `run: echo ok` would leave P1-P4 green
    while no promotion is actually conditioned on a real verdict lookup.
    P5 asserts the reusable ITSELF: it must exist, its `on:` block must be
    `workflow_call`-only (the same never-independently-starts doctrine P1
    holds the producer to), and it must contain a real STEP -- not a
    `name:`/`env:` mention, not a quoted echo string -- whose `run:` body
    invokes `python3 ci/scripts/gpu_prove_verdict.py` as an actual shell
    command, with no `||`/`;`/`&&` after the invocation on its
    continuation-joined logical line (a trailing `|| true` or a `--sha`
    that never runs would fail open), no `continue-on-error:`/`if:` on
    either that step or its job (either would let the verdict check be
    skipped or silenced), and whose LAST `--sha` argument (argparse's own
    last-wins semantics, never a first-match regex) is bound to the commit
    being promoted (`github.sha`/`$GITHUB_SHA`) -- never a literal sha or a
    tag name."""
    findings: list[str] = []
    resolved = resolve_workflow(workflow_texts, PROOF_REQUIRED_WORKFLOW)
    if resolved is None:
        return [f"P5: {PROOF_REQUIRED_WORKFLOW} is missing from the workflow tree"]
    text = workflow_texts[resolved]

    keys, err = read_top_level_on_block(text)
    if err is not None:
        findings.append(f"P5: {resolved}: {err}")
    elif keys != ["workflow_call"]:
        findings.append(
            f"P5: {resolved}'s on: block must be `workflow_call`-only (found {keys}) -- the reusable "
            "must never independently start anything"
        )

    stripped_text = drop_comment_lines(text)
    lines = stripped_text.splitlines()
    jobs = split_top_level_jobs(stripped_text)

    valid_found = False
    for job_start, job_end in jobs.values():
        job_if_present = any(
            re.match(r"^    if:", lines[i]) for i in range(job_start, job_end)
        )
        job_coe_present = any(
            re.match(r"^    continue-on-error:", lines[i]) for i in range(job_start, job_end)
        )
        for step_start, step_end in _find_step_ranges(lines, job_start, job_end):
            step_keys = _parse_step_keys(lines, step_start, step_end)
            run_text = step_keys.get("run")
            if run_text is None or _GPU_PROVE_VERDICT_INVOCATION not in run_text:
                continue
            logical_lines = _join_shell_continuations(run_text)
            invocation_line = next(
                (ll for ll in logical_lines if ll.startswith(_GPU_PROVE_VERDICT_INVOCATION)), None
            )
            if invocation_line is None:
                # The invocation text is present in this step's `run:` body
                # (e.g. inside a quoted `echo '...'`) but is not itself the
                # command that runs -- not a real invocation site.
                continue
            remainder = invocation_line[len(_GPU_PROVE_VERDICT_INVOCATION) :]
            if _CONTROL_OPERATOR_RE.search(remainder):
                findings.append(
                    f"P5: {resolved} invokes gpu_prove_verdict.py but a shell control operator "
                    f"(`||`/`;`/`&&`) follows it on its logical line (`{invocation_line}`) -- the "
                    "verdict check could fail open"
                )
                continue
            if "if" in step_keys or "continue-on-error" in step_keys or job_if_present or job_coe_present:
                findings.append(
                    f"P5: {resolved}'s step invoking gpu_prove_verdict.py (or its job) carries "
                    "`if:`/`continue-on-error:` -- the verdict check could be skipped or its "
                    "failure silenced"
                )
                continue
            sha_matches = list(_SHA_ARG_RE.finditer(invocation_line))
            if not sha_matches:
                findings.append(f"P5: {resolved} invokes gpu_prove_verdict.py with no --sha argument at all")
                continue
            # LAST occurrence wins, matching argparse's own last-flag-wins
            # semantics -- never the first match a naive regex would find.
            sha_m = sha_matches[-1]
            raw = sha_m.group("q") if sha_m.group("q") is not None else sha_m.group("u")
            if not _sha_arg_is_commit_bound(raw):
                findings.append(
                    f"P5: {resolved}'s --sha argument is `{raw}`, not bound to `github.sha`/`$GITHUB_SHA` -- "
                    "a literal sha or a tag name would key the verdict by the wrong identity"
                )
                continue
            valid_found = True

    if not valid_found and not findings:
        findings.append(
            f"P5: {resolved} does not invoke {_GPU_PROVE_VERDICT_INVOCATION} as a real step's `run:` "
            "command (a mention in `name:`/`env:`/a quoted echo string does not count)"
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
    findings += check_p5(workflow_texts)
    return findings


def main() -> int:
    findings = run_gate()
    if findings:
        print("gpu-prove-once: FAIL", file=sys.stderr)
        for f in findings:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("gpu-prove-once: OK -- exactly one prove producer, no renting reusable, every CUDA lane's "
          "promotion gates on the shared verdict, consumer/producer names agree, and the reusable "
          "actually consults the verdict keyed by the promoted commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
