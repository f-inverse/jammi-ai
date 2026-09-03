#!/usr/bin/env python3
"""GPU-prove-timings gate (esc-080/esc-082/esc-083): the watchdog defaults
and the two-term `RP_TIMEOUT` backstop are re-demanded against COMMITTED
EVIDENCE (`ci/artifacts/gpu-prove-timings/*.json`), never trusted as a
one-time derivation that can silently rot the moment the prove lane's own
walls move.

Hermetic: stdlib only (`re`, `json`, `subprocess` for `git` ancestry) --
no `cargo metadata`, no network, no GPU. Imports `ci/scripts/prove_surface`
(itself `tomllib`-only) and `ci/scripts/check_gpu_parity_matrix`'s own
`load_shipped_cuda_silicon()` (the ONE place `GENCODE_ARCHES` is parsed from
`crates/jammi-kernels/build.rs`, never a second hand-typed arch list here).

Rules:

  R1 -- the two `${VAR:-N}` DEFAULTS are actually SET at their three known
        sites (`runpod_lib.sh`'s inline `${RP_TIMEOUT:-3000}` used by BOTH
        `rp_run_remote` and `rp_run_remote_watched`, `runpod_lib.sh`'s own
        `RP_INACTIVITY="${RP_INACTIVITY:-600}"` assignment, and
        `runpod_gpu_prove.sh`'s own `export RP_TIMEOUT="${RP_TIMEOUT:-6000}"`)
        -- FAILS if any of the three is missing or its literal is
        unparseable, and FAILS if the two `${RP_TIMEOUT:-N}` occurrences in
        `runpod_lib.sh` ever disagree with each other. Separately, a
        stdlib-only, fail-closed SETTER-PREDICATE scan over comment-stripped
        `ci/scripts/**` (`^\\s*(export\\s+)?RP_(TIMEOUT|INACTIVITY)=`) and
        `.github/workflows/**` (`^\\s*RP_(TIMEOUT|INACTIVITY)\\s*:`) catches
        any OTHER committed setter (a workflow `env:`/`with:` key, a
        hardcoded override in some other script) outside the two legitimate
        assignment sites -- the library default (3000) is OUT OF R3's SCOPE
        (it still bounds `runpod_gpu_perf_ab.sh`/`gpu-dev.sh`, never a proof
        lane) and is never itself a violation.
  R2 -- `RP_INACTIVITY >= 3 * max(max_silent_gap_s over every HEALTHY
        artifact and every `slow-host`-disposed cut/kill)`.
  R3 -- the two-term backstop over HEALTHY walls only: `RP_TIMEOUT >= 1.5 *
        max_healthy_wall` AND `RP_TIMEOUT >= max_healthy_wall + 3 *
        RP_INACTIVITY`. VACUOUS (a FAIL, never a silent pass) unless every
        `GENCODE_ARCHES` arch has >= 1 healthy artifact (legacy or current).
  R4 -- every non-healthy (`budget-cut`/`watchdog-kill`) artifact carries a
        non-null `disposition` with `evidence.job_id` and
        `evidence.last_output_at`; `hang` additionally requires
        `evidence.issue`.
  R5 -- per `GENCODE_ARCHES` arch, at least one `prove-lane`-kind HEALTHY
        artifact whose `surface.expected_id` equals the CURRENT manifest's
        `prove_surface.current_expected_id()` -- else FAIL, unless a
        rot-checked waiver row in `gpu_prove_timings_allowlist.txt` covers
        it. Standing cost, disclosed: `expected_id` moves (and this rule
        reds again) whenever a lane feature is added/removed, a
        `prove_only` entry changes, or a crate stops/starts declaring a
        lane feature -- the fix is a fresh 4-pod dispatch, never a waiver.

Also enforces a schema-level integrity rule, independent of R1-R5: a
non-healthy artifact must never carry `wall_s` (a censored wall smuggled in
as if it were measured) -- only `wall_lower_bound_s`; and two `prove-lane`
artifacts sharing the same `git_sha` must never disagree on
`surface.expected_id` (the SAME commit cannot have proved two different
surfaces).

Run: `python3 ci/scripts/check_gpu_prove_timings.py`
Self-test: `python3 ci/scripts/check_gpu_prove_timings.py --self-test`
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ci" / "scripts"))
import prove_surface  # noqa: E402
import check_gpu_parity_matrix as gpu_parity_matrix  # noqa: E402

ARTIFACT_DIR_REL = "ci/artifacts/gpu-prove-timings"
ARTIFACT_DIR = REPO_ROOT / ARTIFACT_DIR_REL
ALLOWLIST_REL = "ci/scripts/gpu_prove_timings_allowlist.txt"
ALLOWLIST_PATH = REPO_ROOT / ALLOWLIST_REL
RUNPOD_LIB_REL = "ci/scripts/runpod_lib.sh"
RUNPOD_PROVE_REL = "ci/scripts/runpod_gpu_prove.sh"

# The CLOSED set of `outcome` values (matches the producer's own
# `--outcome` argparse `choices`) -- an unrecognized value (a typo, a
# renamed value on one side only) is a schema FAIL, never silently ignored.
# Of these, only `budget-cut`/`watchdog-kill` need an R4 disposition
# (a genuine cut/kill whose cause is undetermined without one); `healthy`,
# `suite-fail`, `capacity`, and `log-incomplete` need NONE, and a
# `disposition` present on any of those four is ITSELF a schema FINDING
# (round-2 audit fix: a disposition exists to explain an UNDETERMINED
# cut/kill cause -- attaching one to an outcome that already has a
# determined, self-explaining cause is either a bug in the producer or a
# hand-edited artifact papering over a real finding). `suite-fail` names its
# own cause in the leg's own suite output; `capacity` (exit 75) is a supply
# condition, not a hang/cut; `log-incomplete` (BLOCK B) is a truncated log,
# not a real outcome to disposition at all -- it needs a fresh run, not a
# reviewed explanation. `wrong-tree` (esc-084/#454 amendment N/U) is
# likewise a DETERMINED cause -- the ref moved under the clone, or a tag was
# moved -- self-explaining exactly like `capacity`, never dispositioned.
OUTCOME_VALUES = frozenset(
    {"healthy", "budget-cut", "watchdog-kill", "suite-fail", "capacity", "log-incomplete", "wrong-tree"}
)
OUTCOMES_NEEDING_DISPOSITION = frozenset({"budget-cut", "watchdog-kill"})
OUTCOMES_FORBIDDING_DISPOSITION = OUTCOME_VALUES - OUTCOMES_NEEDING_DISPOSITION

_GIT_NO_BACKGROUND_MAINTENANCE = ("-c", "gc.auto=0", "-c", "gc.autoDetach=false", "-c", "maintenance.auto=false")


def _run_git(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *_GIT_NO_BACKGROUND_MAINTENANCE, *cmd], cwd=cwd, capture_output=True, text=True)


def _is_ancestor(sha: str, repo_root: Path, target: str = "HEAD") -> bool:
    proc = _run_git(["merge-base", "--is-ancestor", sha, target], repo_root)
    return proc.returncode == 0


# --------------------------------------------------------------------------- #
# comment-stripped setter-predicate scan (R1's second half)
# --------------------------------------------------------------------------- #
_SHELL_SETTER_RE = re.compile(r"^\s*(?:export\s+)?RP_(TIMEOUT|INACTIVITY)=")
_YAML_SETTER_RE = re.compile(r"^\s*RP_(TIMEOUT|INACTIVITY)\s*:")

ALLOWED_SETTER_SITES = frozenset({(RUNPOD_LIB_REL, "INACTIVITY"), (RUNPOD_PROVE_REL, "TIMEOUT")})


def _strip_shell_comments(text: str) -> str:
    return "\n".join("" if line.strip().startswith("#") else line for line in text.splitlines())


def _strip_yaml_comments(text: str) -> str:
    return "\n".join("" if line.strip().startswith("#") else line for line in text.splitlines())


def _tracked_files(repo_root: Path) -> list[str]:
    out = _run_git(["ls-files"], repo_root)
    if out.returncode != 0:
        return []
    return out.stdout.splitlines()


def scan_setters(repo_root: Path = REPO_ROOT) -> list[tuple[str, str, int, str]]:
    """Every `(relpath, var, lineno, line_text)` matching the setter
    predicate, REGARDLESS of the allowlist -- callers filter against
    `ALLOWED_SETTER_SITES`. `line_text` (the comment-stripped line itself)
    lets a caller distinguish a genuine DEFAULT-setting assignment
    (`VAR="${VAR:-N}"`) from a same-variable NORMALIZATION reassignment
    (`VAR=$((10#$VAR))`, this file's own established base-10-parse
    convention for every numeric env var it validates) -- the latter is not
    a second source of truth for the default and must not count as one."""
    found: list[tuple[str, str, int, str]] = []
    for rel in _tracked_files(repo_root):
        if rel.startswith("ci/scripts/"):
            regex = _SHELL_SETTER_RE
            stripper = _strip_shell_comments
        elif rel.startswith(".github/workflows/"):
            regex = _YAML_SETTER_RE
            stripper = _strip_yaml_comments
        else:
            continue
        path = repo_root / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(stripper(text).splitlines(), start=1):
            m = regex.match(line)
            if m:
                found.append((rel, m.group(1), lineno, line))
    return found


# --------------------------------------------------------------------------- #
# R1: default extraction
# --------------------------------------------------------------------------- #
_LIB_TIMEOUT_DEFAULT_RE = re.compile(r"\$\{RP_TIMEOUT:-(\w+)\}")
_LIB_INACTIVITY_ASSIGN_RE = re.compile(r'^RP_INACTIVITY="\$\{RP_INACTIVITY:-(\w+)\}"', re.MULTILINE)
_PROVE_TIMEOUT_EXPORT_RE = re.compile(r'^export RP_TIMEOUT="\$\{RP_TIMEOUT:-(\w+)\}"', re.MULTILINE)


def check_r1(repo_root: Path = REPO_ROOT) -> tuple[list[str], dict[str, int]]:
    problems: list[str] = []
    defaults: dict[str, int] = {}

    lib_path = repo_root / RUNPOD_LIB_REL
    prove_path = repo_root / RUNPOD_PROVE_REL

    if not lib_path.is_file():
        problems.append(f"R1: {RUNPOD_LIB_REL} does not exist")
        return problems, defaults
    if not prove_path.is_file():
        problems.append(f"R1: {RUNPOD_PROVE_REL} does not exist")
        return problems, defaults

    lib_text = lib_path.read_text(encoding="utf-8")
    prove_text = prove_path.read_text(encoding="utf-8")

    lib_timeout_matches = _LIB_TIMEOUT_DEFAULT_RE.findall(lib_text)
    if not lib_timeout_matches:
        problems.append(f"R1: no `${{RP_TIMEOUT:-N}}` default found in {RUNPOD_LIB_REL} (allowed site missing)")
    else:
        values = set(lib_timeout_matches)
        if len(values) > 1:
            problems.append(f"R1: {RUNPOD_LIB_REL} has DISAGREEING `${{RP_TIMEOUT:-N}}` defaults: {sorted(values)}")
        else:
            raw = next(iter(values))
            if not raw.isdigit():
                problems.append(f"R1: {RUNPOD_LIB_REL}'s `${{RP_TIMEOUT:-{raw}}}` default is not a plain integer")
            else:
                defaults["lib_rp_timeout"] = int(raw)

    inact_m = _LIB_INACTIVITY_ASSIGN_RE.search(lib_text)
    if not inact_m:
        problems.append(f"R1: no `RP_INACTIVITY=\"${{RP_INACTIVITY:-N}}\"` assignment found in {RUNPOD_LIB_REL} (allowed site missing)")
    elif not inact_m.group(1).isdigit():
        problems.append(f"R1: {RUNPOD_LIB_REL}'s RP_INACTIVITY default `{inact_m.group(1)}` is not a plain integer")
    else:
        defaults["rp_inactivity"] = int(inact_m.group(1))

    prove_m = _PROVE_TIMEOUT_EXPORT_RE.search(prove_text)
    if not prove_m:
        problems.append(f"R1: no `export RP_TIMEOUT=\"${{RP_TIMEOUT:-N}}\"` found in {RUNPOD_PROVE_REL} (allowed site missing)")
    elif not prove_m.group(1).isdigit():
        problems.append(f"R1: {RUNPOD_PROVE_REL}'s RP_TIMEOUT default `{prove_m.group(1)}` is not a plain integer")
    else:
        defaults["rp_timeout"] = int(prove_m.group(1))

    allowed_site_counts: dict[tuple[str, str], int] = {}
    for rel, var, lineno, line_text in scan_setters(repo_root):
        if (rel, var) not in ALLOWED_SETTER_SITES:
            problems.append(
                f"R1: {rel}:{lineno}: a committed RP_{var} setter outside the two allowed "
                f"sites ({sorted(ALLOWED_SETTER_SITES)}) -- a hidden second source of truth"
            )
            continue
        # Only a genuine DEFAULT-setting shape (`VAR="${VAR:-N}"` or
        # `export VAR="${VAR:-N}"`) counts as an "assignment" for the
        # count==1 rule below -- a same-variable NORMALIZATION reassignment
        # (`VAR=$((10#$VAR))`, this file's own established base-10-parse
        # convention, e.g. RP_TTL_HOURS/RP_SSH_WAIT_SECS/RP_INACTIVITY all
        # do this) sets NO competing default and must not be miscounted as
        # a second source of truth.
        if re.search(rf'RP_{var}="\$\{{RP_{var}:-', line_text):
            allowed_site_counts[(rel, var)] = allowed_site_counts.get((rel, var), 0) + 1

    # A SECOND default-setting assignment of the same var in an otherwise-
    # allowed file is itself a hidden second source of truth (round-2 audit
    # fix): the allowlist names exactly ONE legitimate default-setting site
    # per (file, var), not "as many as happen to live in that file".
    for (rel, var), count in allowed_site_counts.items():
        if count != 1:
            problems.append(
                f"R1: {rel} has {count} RP_{var} DEFAULT-setting assignment(s) (expected "
                f"exactly 1) -- a second one is still a hidden second source of truth"
            )

    return problems, defaults


# --------------------------------------------------------------------------- #
# artifact loading + schema integrity
# --------------------------------------------------------------------------- #
def load_artifacts(artifact_dir: Path = ARTIFACT_DIR) -> list[dict]:
    if not artifact_dir.is_dir():
        return []
    out = []
    for p in sorted(artifact_dir.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError) as e:
            out.append({"_load_error": str(e), "_path": str(p)})
    return out


def check_schema(artifacts: list[dict]) -> list[str]:
    problems: list[str] = []
    by_sha: dict[str, set[str]] = {}
    for a in artifacts:
        if "_load_error" in a:
            problems.append(f"schema: {a['_path']}: unreadable/invalid JSON: {a['_load_error']}")
            continue
        outcome = a.get("outcome")
        if outcome not in OUTCOME_VALUES:
            problems.append(
                f"schema: run {a.get('run_id')}/{a.get('arch')}: outcome={outcome!r} is not one of "
                f"the closed set {sorted(OUTCOME_VALUES)} -- a typo or a renamed value on one side only"
            )
        if outcome != "healthy" and "wall_s" in a:
            problems.append(
                f"schema: run {a.get('run_id')}/{a.get('arch')}: outcome={outcome!r} but carries "
                f"`wall_s` -- a censored wall must be `wall_lower_bound_s`, never `wall_s`"
            )
        if outcome == "healthy" and "wall_lower_bound_s" in a:
            problems.append(
                f"schema: run {a.get('run_id')}/{a.get('arch')}: healthy outcome must not carry "
                f"`wall_lower_bound_s`"
            )
        if outcome in OUTCOMES_FORBIDDING_DISPOSITION and a.get("disposition") is not None:
            problems.append(
                f"schema: run {a.get('run_id')}/{a.get('arch')}: outcome={outcome!r} carries a "
                f"`disposition`, but only {sorted(OUTCOMES_NEEDING_DISPOSITION)} outcomes may -- "
                f"a determined-cause outcome does not need (or accept) one"
            )
        if outcome == "watchdog-kill":
            if "max_silent_gap_s" in a:
                problems.append(
                    f"schema: run {a.get('run_id')}/{a.get('arch')}: watchdog-kill carries "
                    f"`max_silent_gap_s`, but its own silence is right-censored at RP_INACTIVITY "
                    f"-- must be recorded as `silent_gap_lower_bound_s` instead"
                )
            if "silent_gap_lower_bound_s" not in a:
                problems.append(
                    f"schema: run {a.get('run_id')}/{a.get('arch')}: watchdog-kill is missing "
                    f"`silent_gap_lower_bound_s`"
                )
        elif "silent_gap_lower_bound_s" in a:
            problems.append(
                f"schema: run {a.get('run_id')}/{a.get('arch')}: outcome={outcome!r} carries "
                f"`silent_gap_lower_bound_s`, which only a watchdog-kill's right-censored "
                f"silence may use"
            )
        surface = a.get("surface", {})
        if surface.get("kind") == "prove-lane" and a.get("git_sha") and surface.get("expected_id"):
            seen = by_sha.setdefault(a["git_sha"], set())
            seen.add(surface["expected_id"])
    for sha, ids in by_sha.items():
        if len(ids) > 1:
            problems.append(f"schema: git_sha {sha} has DISAGREEING expected_id values across artifacts: {sorted(ids)}")
    return problems


# --------------------------------------------------------------------------- #
# R2 / R3
# --------------------------------------------------------------------------- #
def check_r2(artifacts: list[dict], defaults: dict[str, int]) -> list[str]:
    # Round-2 audit fix: R2 consumes HEALTHY artifacts ONLY. A
    # `watchdog-kill`'s own `silent_gap_lower_bound_s` is, by definition,
    # right-censored at RP_INACTIVITY (the kill fires exactly when that
    # threshold is crossed) -- feeding it back into the rule that SETS
    # RP_INACTIVITY would be circular, and a `slow-host` disposition's own
    # claim ("the host is just slow, not hung") is validated by its own
    # required follow-up healthy artifact (see R4 below), which is what
    # actually re-enters this margin -- never the kill's own censored value.
    problems: list[str] = []
    gaps = [
        a["max_silent_gap_s"]
        for a in artifacts
        if "_load_error" not in a and a.get("outcome") == "healthy" and "max_silent_gap_s" in a
    ]
    if not gaps:
        # VACUOUS, exactly like R3's own arch-coverage check -- a margin
        # with nothing to bound it against is never silently accepted as
        # clean; it is a FAIL demanding at least one healthy artifact
        # before this rule can mean anything.
        problems.append("R2: VACUOUS -- no healthy artifact to bound RP_INACTIVITY against")
        return problems
    max_gap = max(gaps)
    inactivity = defaults.get("rp_inactivity")
    if inactivity is None:
        return problems  # R1 already reported the missing default
    if inactivity < 3 * max_gap:
        problems.append(
            f"R2: RP_INACTIVITY={inactivity} is below 3x the largest observed HEALTHY "
            f"silent gap ({max_gap}s, need >= {3 * max_gap})"
        )
    return problems


def check_r3(artifacts: list[dict], defaults: dict[str, int], repo_root: Path = REPO_ROOT) -> list[str]:
    problems: list[str] = []
    try:
        arches = gpu_parity_matrix.load_shipped_cuda_silicon()
    except Exception as e:  # noqa: BLE001
        problems.append(f"R3: cannot load GENCODE_ARCHES: {e}")
        return problems

    healthy_by_arch: dict[str, list[float]] = {a: [] for a in arches}
    for art in artifacts:
        if "_load_error" in art:
            continue
        if art.get("outcome") == "healthy" and art.get("arch") in healthy_by_arch and "wall_s" in art:
            healthy_by_arch[art["arch"]].append(art["wall_s"])

    missing = [a for a, walls in healthy_by_arch.items() if not walls]
    if missing:
        problems.append(f"R3: VACUOUS -- no healthy artifact for arch(es) {sorted(missing)}; the backstop cannot be checked")
        return problems

    rp_timeout = defaults.get("rp_timeout")
    rp_inactivity = defaults.get("rp_inactivity")
    if rp_timeout is None or rp_inactivity is None:
        return problems  # R1 already reported

    max_healthy_wall = max(w for walls in healthy_by_arch.values() for w in walls)
    if rp_timeout < 1.5 * max_healthy_wall:
        problems.append(
            f"R3: RP_TIMEOUT={rp_timeout} < 1.5 * max_healthy_wall ({max_healthy_wall}) = {1.5 * max_healthy_wall}"
        )
    floor2 = max_healthy_wall + 3 * rp_inactivity
    if rp_timeout < floor2:
        problems.append(
            f"R3: RP_TIMEOUT={rp_timeout} < max_healthy_wall + 3*RP_INACTIVITY "
            f"({max_healthy_wall} + {3 * rp_inactivity}) = {floor2}"
        )
    return problems


# --------------------------------------------------------------------------- #
# R4 -- an arm for every non-healthy `outcome` (see OUTCOME_VALUES' own doc):
# `budget-cut`/`watchdog-kill` REQUIRE a disposition (this is the only arm
# below); `suite-fail`/`capacity` require NONE, by design, and are simply
# never visited by this loop -- stated here so that omission reads as a
# decision, not an oversight.
# --------------------------------------------------------------------------- #
def check_r4(artifacts: list[dict]) -> list[str]:
    problems: list[str] = []
    healthy_run_ids = {
        a.get("run_id") for a in artifacts if "_load_error" not in a and a.get("outcome") == "healthy"
    }
    for a in artifacts:
        if "_load_error" in a:
            continue
        if a.get("outcome") in OUTCOMES_NEEDING_DISPOSITION:
            disp = a.get("disposition")
            label = f"run {a.get('run_id')}/{a.get('arch')}"
            if not disp:
                problems.append(f"R4: {label}: {a['outcome']} artifact has no `disposition` -- undispositioned kill")
                continue
            if disp.get("kind") not in ("hang", "slow-host"):
                problems.append(f"R4: {label}: disposition.kind must be 'hang' or 'slow-host', got {disp.get('kind')!r}")
            evidence = disp.get("evidence") or {}
            if not evidence.get("job_id"):
                problems.append(f"R4: {label}: disposition.evidence missing `job_id`")
            if not evidence.get("last_output_at"):
                problems.append(f"R4: {label}: disposition.evidence missing `last_output_at`")
            if disp.get("kind") == "hang" and not evidence.get("issue"):
                problems.append(f"R4: {label}: a 'hang' disposition requires disposition.evidence.issue")
            if disp.get("kind") == "slow-host":
                # Round-2 audit advisory: "the host is just slow, not hung"
                # is a CLAIM about that host's real behavior -- it must cite
                # the follow-up run that actually RE-MEASURED it healthy,
                # never asserted on its own say-so.
                followup = evidence.get("followup_run_id")
                if not followup:
                    problems.append(
                        f"R4: {label}: a 'slow-host' disposition requires "
                        f"disposition.evidence.followup_run_id naming the healthy artifact that "
                        f"re-measured this host"
                    )
                elif followup not in healthy_run_ids:
                    problems.append(
                        f"R4: {label}: disposition.evidence.followup_run_id={followup!r} does not "
                        f"name any committed HEALTHY artifact's run_id"
                    )
    return problems


# --------------------------------------------------------------------------- #
# R5
# --------------------------------------------------------------------------- #
def load_waivers(path: Path = ALLOWLIST_PATH) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Returns `(rows, problems)` -- a malformed row (not exactly
    `<arch><TAB><reviewed_up_to_sha><TAB><reason>`, three tab-separated
    fields) is a NAMED FAIL in `problems`, never silently skipped: a typo'd
    tab count must not quietly drop a row from R5's coverage."""
    if not path.is_file():
        return [], []
    rows: list[tuple[str, str, str]] = []
    problems: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            problems.append(
                f"R5 waiver rot: {ALLOWLIST_REL}:{lineno}: malformed row (expected exactly 3 "
                f"tab-separated fields <arch>\\t<reviewed_up_to_sha>\\t<reason>, got {len(parts)}): {line!r}"
            )
            continue
        rows.append((parts[0], parts[1], parts[2]))
    return rows, problems


def check_r5(
    artifacts: list[dict],
    repo_root: Path = REPO_ROOT,
    manifest: dict | None = None,
    waivers: list[tuple[str, str, str]] | None = None,
) -> list[str]:
    problems: list[str] = []
    try:
        arches = gpu_parity_matrix.load_shipped_cuda_silicon()
    except Exception as e:  # noqa: BLE001
        problems.append(f"R5: cannot load GENCODE_ARCHES: {e}")
        return problems

    manifest = manifest if manifest is not None else prove_surface.load_manifest()
    current_id = prove_surface.current_expected_id(manifest, repo_root)
    if waivers is None:
        waivers, waiver_load_problems = load_waivers()
        problems.extend(waiver_load_problems)
    waived_arches = {w[0] for w in waivers}

    fresh_arches = set()
    for a in artifacts:
        if "_load_error" in a:
            continue
        if (
            a.get("outcome") == "healthy"
            and a.get("surface", {}).get("kind") == "prove-lane"
            and a.get("surface", {}).get("expected_id") == current_id
        ):
            fresh_arches.add(a.get("arch"))

    for arch in sorted(arches):
        if arch in fresh_arches:
            continue
        if arch in waived_arches:
            continue
        problems.append(
            f"R5: arch `{arch}` has no HEALTHY `prove-lane` artifact matching the CURRENT "
            f"expected_id ({current_id}) -- and no waiver row covers it"
        )

    # Waiver rot / dead-waiver mirror.
    for arch, reviewed_sha, reason in waivers:
        if arch not in arches:
            problems.append(f"R5 waiver rot: `{arch}` is not a current GENCODE_ARCHES entry")
            continue
        if not re.fullmatch(r"[0-9a-f]{7,40}", reviewed_sha):
            problems.append(f"R5 waiver rot: `{arch}` row's reviewed_up_to_sha `{reviewed_sha}` is malformed")
            continue
        if not _is_ancestor(reviewed_sha, repo_root):
            problems.append(f"R5 waiver rot: `{arch}` row's reviewed_up_to_sha `{reviewed_sha}` is not an ancestor of HEAD")
            continue
        if not reason.strip():
            problems.append(f"R5 waiver rot: `{arch}` row has an empty reason")
        if arch in fresh_arches:
            problems.append(f"R5 dead waiver: `{arch}` already has a fresh matching artifact -- delete the row")

    return problems


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_gate(repo_root: Path = REPO_ROOT, verbose: bool = True) -> int:
    problems: list[str] = []
    r1_problems, defaults = check_r1(repo_root)
    problems += r1_problems

    artifacts = load_artifacts(repo_root / ARTIFACT_DIR_REL)
    problems += check_schema(artifacts)
    problems += check_r2(artifacts, defaults)
    problems += check_r3(artifacts, defaults, repo_root)
    problems += check_r4(artifacts)
    problems += check_r5(artifacts, repo_root)

    if verbose:
        for p in problems:
            print(f"FAIL: {p}", file=sys.stderr)
        print(f"check_gpu_prove_timings: {'PASS' if not problems else f'FAIL ({len(problems)})'}")
    return 1 if problems else 0


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _write_fixture_repo(root: Path, *, lib_body: str, prove_body: str, workflow_body: str | None = None) -> None:
    (root / "ci" / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "ci" / "artifacts" / "gpu-prove-timings").mkdir(parents=True, exist_ok=True)
    (root / "crates" / "jammi-kernels").mkdir(parents=True, exist_ok=True)
    (root / "crates" / "jammi-kernels" / "build.rs").write_text(
        'pub(crate) const GENCODE_ARCHES: &[&str] = &[\n'
        '    "arch=compute_80,code=sm_80",\n'
        '];\n'
    )
    (root / "ci" / "scripts" / "runpod_lib.sh").write_text(lib_body)
    (root / "ci" / "scripts" / "runpod_gpu_prove.sh").write_text(prove_body)
    (root / "ci" / "release-feature-manifest.json").write_text(json.dumps({
        "lanes": {"cu12-tarball": {"cargo_features": ["cuda"]}},
        "server_only_cargo_features": {"features": []},
        "prove_lane": {"crates": {"jammi-kernels": {"kinds": ["default"], "prove_only": []}}},
    }))
    (root / "crates" / "jammi-kernels" / "Cargo.toml").write_text(
        '[package]\nname = "jammi-kernels"\nversion = "0.1.0"\n\n[features]\ncuda = []\ndefault = []\n'
    )
    if workflow_body is not None:
        (root / ".github" / "workflows").mkdir(parents=True, exist_ok=True)
        (root / ".github" / "workflows" / "gpu-prove.yml").write_text(workflow_body)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-q", "-m", "init"], cwd=root, check=True)


_GOOD_LIB = (
    'RP_INACTIVITY="${RP_INACTIVITY:-600}"\n'
    'rp_run_remote() { ssh "timeout ${RP_TIMEOUT:-3000} bash -s"; }\n'
    'rp_run_remote_watched() { ssh "timeout ${RP_TIMEOUT:-3000} bash -s"; }\n'
)
_GOOD_PROVE = 'export RP_TIMEOUT="${RP_TIMEOUT:-6000}"\n'


def _write_artifact(root: Path, name: str, body: dict) -> None:
    (root / "ci" / "artifacts" / "gpu-prove-timings" / name).write_text(json.dumps(body))


def _healthy_artifact(arch: str, wall_s: float, gap: float, git_sha: str = "a" * 40, expected_id: str | None = None) -> dict:
    return {
        "arch": arch,
        "outcome": "healthy",
        "wall_s": wall_s,
        "max_silent_gap_s": gap,
        "git_sha": git_sha,
        "run_id": "1",
        "job_id": "1",
        "groups": [],
        "disposition": None,
        "surface": {"kind": "prove-lane" if expected_id else "legacy-pre-d1", "expected_id": expected_id},
    }


def _self_test() -> int:
    import tempfile

    failures: list[str] = []
    total = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal total
        total += 1
        print(f"self-test[{name}]: " + ("ok" if cond else f"FAIL -- {detail}"))
        if not cond:
            failures.append(name)

    # --- good fixture: R1-R4 all green (R5 not checked here directly).
    # `load_shipped_cuda_silicon()` reads the REAL, tracked build.rs (not
    # parameterized by repo_root), so a healthy artifact is seeded for
    # every REAL current GENCODE_ARCHES entry, not just one. ---
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_fixture_repo(root, lib_body=_GOOD_LIB, prove_body=_GOOD_PROVE)
        for i, arch in enumerate(sorted(gpu_parity_matrix.load_shipped_cuda_silicon())):
            _write_artifact(root, f"{i}-{arch}.json", _healthy_artifact(arch, 2000.0, 100.0))
        r1, defaults = check_r1(root)
        check("good-fixture-r1-green", r1 == [], f"{r1}")
        arts = load_artifacts(root / ARTIFACT_DIR_REL)
        check("good-fixture-r2-green", check_r2(arts, defaults) == [], f"{check_r2(arts, defaults)}")
        r3 = check_r3(arts, defaults, root)
        check("good-fixture-r3-green", r3 == [], f"{r3}")
        check("good-fixture-r4-green", check_r4(arts) == [], f"{check_r4(arts)}")

    # --- lowered literal: RP_INACTIVITY too small for the observed gap. ---
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        bad_lib = _GOOD_LIB.replace('RP_INACTIVITY:-600', 'RP_INACTIVITY:-10')
        _write_fixture_repo(root, lib_body=bad_lib, prove_body=_GOOD_PROVE)
        _write_artifact(root, "1-sm_80.json", _healthy_artifact("sm_80", 2000.0, 100.0))
        r1, defaults = check_r1(root)
        arts = load_artifacts(root / ARTIFACT_DIR_REL)
        r2 = check_r2(arts, defaults)
        check("lowered-literal-r2-caught", any("below 3x" in p for p in r2), f"{r2}")

    # --- raised gap: an artifact's own silent gap exceeds the margin. ---
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_fixture_repo(root, lib_body=_GOOD_LIB, prove_body=_GOOD_PROVE)
        _write_artifact(root, "1-sm_80.json", _healthy_artifact("sm_80", 2000.0, 5000.0))
        r1, defaults = check_r1(root)
        arts = load_artifacts(root / ARTIFACT_DIR_REL)
        r2 = check_r2(arts, defaults)
        check("raised-gap-r2-caught", any("below 3x" in p for p in r2), f"{r2}")

    # --- missing arch: R3 vacuous. ---
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_fixture_repo(root, lib_body=_GOOD_LIB, prove_body=_GOOD_PROVE)
        r1, defaults = check_r1(root)
        arts: list[dict] = []
        r3 = check_r3(arts, defaults, root)
        check("missing-arch-r3-vacuous", any("VACUOUS" in p for p in r3), f"{r3}")

    # --- unparseable literal. ---
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        bad_prove = 'export RP_TIMEOUT="${RP_TIMEOUT:-notanumber}"\n'
        _write_fixture_repo(root, lib_body=_GOOD_LIB, prove_body=bad_prove)
        r1, defaults = check_r1(root)
        check("unparseable-literal-r1-caught", any("not a plain integer" in p for p in r1), f"{r1}")

    # --- undispositioned kill. ---
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_fixture_repo(root, lib_body=_GOOD_LIB, prove_body=_GOOD_PROVE)
        bad = _healthy_artifact("sm_80", 2000.0, 100.0)
        bad["outcome"] = "budget-cut"
        del bad["wall_s"]
        bad["wall_lower_bound_s"] = 3000.0
        bad["disposition"] = None
        _write_artifact(root, "1-sm_80.json", bad)
        arts = load_artifacts(root / ARTIFACT_DIR_REL)
        r4 = check_r4(arts)
        check("undispositioned-kill-r4-caught", any("undispositioned kill" in p for p in r4), f"{r4}")

    # `load_shipped_cuda_silicon()` is NOT parameterized by repo_root (it
    # reads the real, tracked `crates/jammi-kernels/build.rs`) -- so R5/R3
    # self-tests cover the REAL current GENCODE_ARCHES set, not a fixture
    # one.
    real_arches = sorted(gpu_parity_matrix.load_shipped_cuda_silicon())

    # --- stale surface without waiver (R5): every arch fresh EXCEPT sm_80. ---
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_fixture_repo(root, lib_body=_GOOD_LIB, prove_body=_GOOD_PROVE)
        manifest = json.loads((root / "ci" / "release-feature-manifest.json").read_text())
        current_id = prove_surface.current_expected_id(manifest, root)
        for i, arch in enumerate(real_arches):
            eid = "deadbeef" if arch == "sm_80" else current_id
            _write_artifact(root, f"{i}-{arch}.json", _healthy_artifact(arch, 2000.0, 100.0, expected_id=eid))
        arts = load_artifacts(root / ARTIFACT_DIR_REL)
        r5 = check_r5(arts, root, manifest, waivers=[])
        check("stale-surface-no-waiver-r5-caught", any("arch `sm_80`" in p and "no HEALTHY" in p for p in r5), f"{r5}")

        # covered by a waiver -> no longer a finding for sm_80 specifically.
        r5_waived = check_r5(arts, root, manifest, waivers=[("sm_80", "a" * 40, "reviewed")])
        check(
            "stale-surface-with-waiver-r5-clean",
            not any("arch `sm_80`" in p and "no HEALTHY" in p for p in r5_waived),
            f"{r5_waived}",
        )

    # --- dead waiver: every arch (including sm_80) is actually fresh. ---
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_fixture_repo(root, lib_body=_GOOD_LIB, prove_body=_GOOD_PROVE)
        manifest = json.loads((root / "ci" / "release-feature-manifest.json").read_text())
        current_id = prove_surface.current_expected_id(manifest, root)
        for i, arch in enumerate(real_arches):
            _write_artifact(root, f"{i}-{arch}.json", _healthy_artifact(arch, 2000.0, 100.0, expected_id=current_id))
        head_sha = _run_git(["rev-parse", "HEAD"], root).stdout.strip()
        arts = load_artifacts(root / ARTIFACT_DIR_REL)
        r5 = check_r5(arts, root, manifest, waivers=[("sm_80", head_sha, "no longer needed")])
        check("dead-waiver-r5-caught", any("dead waiver" in p for p in r5), f"{r5}")

    # --- malformed waiver row: a named FAIL, never silently skipped. ---
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        waivers_path = root / "waivers.txt"
        waivers_path.write_text("sm_80\tonly-two-fields\n# a comment, ignored\n\nsm_86\tdeadbeef\ttoo\tmany\tfields\n")
        rows, load_problems = load_waivers(waivers_path)
        check("malformed-waiver-row-caught", len(load_problems) == 2 and rows == [], f"{load_problems} {rows}")

    # --- committed setter: a shell assignment AND a YAML env key, both
    # OUTSIDE the two allowed sites; a comment mention must NOT fire.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rogue_lib = _GOOD_LIB + '\n# RP_TIMEOUT=999 (just a comment, must not fire)\n'
        (root / "ci" / "scripts").mkdir(parents=True, exist_ok=True)
        rogue_script = "RP_TIMEOUT=1234\n"
        _write_fixture_repo(
            root,
            lib_body=rogue_lib,
            prove_body=_GOOD_PROVE,
            workflow_body='jobs:\n  x:\n    env:\n      RP_INACTIVITY: "42"\n      # RP_TIMEOUT: "1" (comment, must not fire)\n',
        )
        (root / "ci" / "scripts" / "rogue.sh").write_text(rogue_script)
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        setters = scan_setters(root)
        rogue_hits = [s for s in setters if s[0] == "ci/scripts/rogue.sh"]
        yaml_hits = [s for s in setters if s[0] == ".github/workflows/gpu-prove.yml"]
        comment_hits_lib = [s for s in setters if s[0] == RUNPOD_LIB_REL and "RP_TIMEOUT=999" in ""]
        check("committed-setter-shell-caught", len(rogue_hits) == 1, f"{setters}")
        check("committed-setter-yaml-caught", len(yaml_hits) == 1 and yaml_hits[0][1] == "INACTIVITY", f"{setters}")
        r1, _ = check_r1(root)
        check(
            "committed-setter-r1-caught",
            any("rogue.sh" in p for p in r1) and any("gpu-prove.yml" in p for p in r1),
            f"{r1}",
        )
        check(
            "comment-mention-does-not-fire",
            not any("999" in p or "RP_TIMEOUT: \"1\"" in p for p in r1),
            f"{r1}",
        )

    # --- RP_WATCH_POLL_S (runpod_gpu_prove.sh's fixture/diagnostic-only
    # poll-interval escape hatch) is NOT in the RP_(TIMEOUT|INACTIVITY)
    # alternation and must never be flagged as a committed setter, even
    # when assigned right next to a real one. ---
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        prove_with_poll = _GOOD_PROVE + 'rp_run_remote_watched "" "${RP_WATCH_POLL_S:-5}" <<REMOTE\nREMOTE\n'
        _write_fixture_repo(root, lib_body=_GOOD_LIB, prove_body=prove_with_poll)
        setters = scan_setters(root)
        poll_hits = [s for s in setters if "WATCH_POLL" in s[1]]
        check("rp_watch_poll_s_never_flagged", poll_hits == [], f"{setters}")
        r1, _ = check_r1(root)
        check("rp_watch_poll_s_r1_clean", not any("WATCH_POLL" in p for p in r1), f"{r1}")

    # --- deleted export (allowed site missing). ---
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_fixture_repo(root, lib_body=_GOOD_LIB, prove_body="echo no export here\n")
        r1, _ = check_r1(root)
        check("deleted-export-r1-caught", any("allowed site missing" in p for p in r1), f"{r1}")

    # --- censored wall smuggled as wall_s. ---
    bad = _healthy_artifact("sm_80", 2000.0, 100.0)
    bad["outcome"] = "budget-cut"
    schema_problems = check_schema([bad])
    check("censored-wall-as-wall-s-caught", any("must be `wall_lower_bound_s`" in p for p in schema_problems), f"{schema_problems}")

    # --- same git_sha, disagreeing expected_id across two artifacts. ---
    a1 = _healthy_artifact("sm_80", 2000.0, 100.0, git_sha="b" * 40, expected_id="id-one")
    a2 = _healthy_artifact("sm_86", 2000.0, 100.0, git_sha="b" * 40, expected_id="id-two")
    schema_problems2 = check_schema([a1, a2])
    check("same-sha-disagreeing-expected-id-caught", any("DISAGREEING expected_id" in p for p in schema_problems2), f"{schema_problems2}")

    # --- round-2 audit: a SECOND assignment of the same var in an
    # otherwise-allowed file is still a hidden second source of truth. ---
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        double_lib = _GOOD_LIB + 'RP_INACTIVITY="${RP_INACTIVITY:-700}"\n'
        _write_fixture_repo(root, lib_body=double_lib, prove_body=_GOOD_PROVE)
        r1, _ = check_r1(root)
        check(
            "second-assignment-in-allowed-file-caught",
            any("has 2 RP_INACTIVITY DEFAULT-setting assignment(s)" in p for p in r1),
            f"{r1}",
        )

    # --- round-2 audit: outcome typo rejection. ---
    typo = _healthy_artifact("sm_80", 2000.0, 100.0)
    typo["outcome"] = "helthy"
    typo_problems = check_schema([typo])
    check("outcome-typo-rejected", any("is not one of the closed set" in p for p in typo_problems), f"{typo_problems}")

    # --- round-2 audit: R2 vacuity arm (no HEALTHY artifact at all --
    # only a budget-cut one, which R2 no longer consumes). ---
    only_cut = _healthy_artifact("sm_80", 2000.0, 100.0)
    only_cut["outcome"] = "budget-cut"
    del only_cut["wall_s"]
    only_cut["wall_lower_bound_s"] = 2000.0
    r2_vacuous = check_r2([only_cut], {"rp_inactivity": 600})
    check("r2-vacuous-with-no-healthy-artifact", any("VACUOUS" in p for p in r2_vacuous), f"{r2_vacuous}")

    # --- round-2 audit: a disposition on a healthy/suite-fail/capacity/
    # log-incomplete outcome is itself a schema finding. ---
    for bad_outcome in ("healthy", "suite-fail", "capacity", "log-incomplete", "wrong-tree"):
        art = _healthy_artifact("sm_80", 2000.0, 100.0)
        art["outcome"] = bad_outcome
        if bad_outcome != "healthy":
            del art["wall_s"]
            art["wall_lower_bound_s"] = 2000.0
        art["disposition"] = {"kind": "slow-host", "evidence": {"job_id": "1", "last_output_at": "x", "followup_run_id": "1"}}
        probs = check_schema([art])
        check(
            f"disposition-on-{bad_outcome}-rejected",
            any("carries a `disposition`" in p for p in probs),
            f"{probs}",
        )

    # --- round-2 audit: a slow-host disposition must cite a followup_run_id
    # naming a committed HEALTHY artifact. ---
    kill_no_followup = _healthy_artifact("sm_86", 2000.0, 100.0)
    kill_no_followup["outcome"] = "watchdog-kill"
    del kill_no_followup["wall_s"]
    kill_no_followup["wall_lower_bound_s"] = 2000.0
    del kill_no_followup["max_silent_gap_s"]
    kill_no_followup["silent_gap_lower_bound_s"] = 600.0
    kill_no_followup["disposition"] = {"kind": "slow-host", "evidence": {"job_id": "1", "last_output_at": "x"}}
    r4_no_followup = check_r4([kill_no_followup])
    check("slow-host-missing-followup-caught", any("followup_run_id" in p for p in r4_no_followup), f"{r4_no_followup}")

    followup_healthy = _healthy_artifact("sm_86", 1900.0, 90.0)
    followup_healthy["run_id"] = "followup-1"
    kill_with_followup = dict(kill_no_followup)
    kill_with_followup["disposition"] = {
        "kind": "slow-host",
        "evidence": {"job_id": "1", "last_output_at": "x", "followup_run_id": "followup-1"},
    }
    r4_with_followup = check_r4([kill_with_followup, followup_healthy])
    check("slow-host-with-real-followup-clean", not any("followup_run_id" in p for p in r4_with_followup), f"{r4_with_followup}")

    if failures:
        print(f"self-test: FAIL ({len(failures)}/{total} failing): {failures}", file=sys.stderr)
        return 1
    print(f"self-test: all {total} checks passed")
    return 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return _self_test()
    return run_gate()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
