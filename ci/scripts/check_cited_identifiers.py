#!/usr/bin/env python3
"""A backtick-quoted citation in a comment must resolve to a REAL identifier.

Three consecutive audit rounds in one session caught an agent citing a TEST
that does not exist in a code comment (e.g. "// see `softmax_scale_with_scale_
refuses_the_invalid_domain` below" — no such fn; the real one was
`with_scale_refuses_zero_negative_nan_and_infinite`). A citation resting on a
name that doesn't resolve is a fabricated proof: it reads as evidence to the
next human or agent who trusts the comment instead of re-deriving it. This gate
makes that a mechanical, fail-closed check instead of a human catch.

Scope: every `.rs` file under `crates/` (and `cookbook/` if it holds Rust).
Candidate citations are backtick-quoted, snake_case-shaped identifiers found in
comments (`//`, `///`, `//!`, `/* */`) — an optional `path::`-style prefix,
a final segment that looks like a function/test name (contains an underscore,
so `is_finite`-shaped false positives from plain two-word idioms are still
resolved, just via the method-call name pool below, not excluded up front),
and an optional trailing `()`. Examples that match: `foo_bar_baz`,
`ops::softmax::tests::foo_bar`, `foo_bar()`. Examples that do NOT match (no
underscore, or non-identifier characters inside the backticks): `NotRecomputable`,
`check_doc_parity.py`.

A cited identifier RESOLVES if its final segment (path prefix and trailing
`()` stripped) is, anywhere in the repo:
  - a `fn` / `const` / `static` / `struct` / `enum` / `mod` / `type` / `trait`
    definition, or a `macro_rules!` name, or
  - the receiver of a method call (`.<name>(`) — this is what resolves the
    plain-English two-segment false positives (`is_finite`, `to_vec`) that
    are real std/candle methods, not gate-internal exceptions, or
  - a struct/enum FIELD or named-argument declaration (`<name>: <type/value>`
    in real code, comments already blanked) — the empirical majority of
    backtick citations in this tree are prose naming a wire/proto field
    (`` `source_id` ``, `` `data_body` ``) or a config knob (`` `lora_rank` ``),
    not a fn/test name; that is a legitimate, different documentation pattern
    from the fabricated-test-name defect class this gate targets, so a real
    field/key identifier resolves here rather than drowning the fabricated-fn
    signal in thousands of false positives, or
  - the exact contents of a double-quoted string literal anywhere in the
    repo's `.rs` source (env var names via `env::var("...")`, `#[serde(rename
    = "...")]`, JSON/proto field-name string constants), or
  - (opportunistically, when the local cargo registry cache is populated —
    e.g. a developer box, never assumed in CI) a same-shaped definition or
    method call in the vendored source of a workspace DIRECT dependency
    (enumerated from every crate's `Cargo.toml` + pinned in `Cargo.lock`), or
  - listed in `ci/cited_identifiers_allow.txt` — the explicit, human-reviewed
    escape hatch for a deliberate reference to vendored/upstream internals
    (e.g. candle's `apply_op1_no_bwd`) that the CI runner's guard job (no
    toolchain, no cargo, no registry cache — see `.github/workflows/ci.yml`)
    cannot resolve on its own. This is the mechanism that actually makes the
    gate CI-green for legitimate vendor citations; the registry-cache lookup
    above is a local-dev convenience on top of it, never a CI dependency.

Comment/string extraction is a hand-rolled tokenizer (line comments, block
comments — including nested `/* /* */ */` — double-quoted strings with
backslash escapes, and single-quote char-literal-vs-lifetime disambiguation).
It does not special-case raw strings (`r"..."`/`r#"..."#`): a `//`-looking
substring inside a raw string could misparse as a comment. Acceptable, exact
for this tree today — no scanned file needs it — and a future false positive
there is a visible, cheap-to-fix cost of a heuristic tokenizer, the same
trade the sibling `check_sqlite_isms.py` and `check_doc_parity.py` gates make.

Run: `python3 ci/scripts/check_cited_identifiers.py`
Ratchet mode (only checks files changed vs a diff base — for burning down a
large pre-existing backlog without blocking unrelated PRs on it):
`python3 ci/scripts/check_cited_identifiers.py --changed-only`
Self-test (asserts the gate flags exactly a fabricated citation, and accepts a
resolving one, on synthetic in-memory fragments — the same style as the
sibling `check_sqlite_isms.py` self-test): `python3 ci/scripts/check_cited_identifiers.py --self-test`
Hermetic: reads only files in the working tree (plus, opportunistically, a
local cargo registry cache if present); no network, no build.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = ("crates", "cookbook")
ALLOWLIST_FILE = REPO_ROOT / "ci" / "cited_identifiers_allow.txt"

# --------------------------------------------------------------------------- #
# comment / string tokenizer
# --------------------------------------------------------------------------- #


def split_comments_and_code(source: str) -> tuple[list[tuple[int, str]], str]:
    """Return (per-physical-line comment text, code with comments blanked).

    Comments returned as `(line_no, single_physical_line_text)` — a multi-line
    block comment contributes one entry per physical line it spans, each with
    its own correct line number, so a citation anywhere inside a `/* ... */`
    block still gets an accurate report line.

    `code` is `source` with every comment span replaced by spaces of the same
    length (newlines preserved) — safe input for the definition/method-call
    regexes below, which would otherwise treat "// fn foo_bar" in a comment as
    a real definition.

    Known, accepted limitation: raw strings (`r"..."`, `r#"..."#`) are not
    tokenized specially, so a `//`/`/*` inside one could misparse. See the
    module docstring.
    """
    comments: list[tuple[int, str]] = []
    code_chars: list[str] = list(source)
    n = len(source)
    line = 1
    i = 0
    while i < n:
        c = source[i]
        if c == "\n":
            line += 1
            i += 1
            continue
        if c == "/" and i + 1 < n and source[i + 1] == "/":
            start = i
            end = source.find("\n", i)
            if end == -1:
                end = n
            text = source[start:end]
            for offset, part in enumerate(text.splitlines()):
                comments.append((line + offset, part))
            for j in range(start, end):
                if code_chars[j] != "\n":
                    code_chars[j] = " "
            i = end
            continue
        if c == "/" and i + 1 < n and source[i + 1] == "*":
            start = i
            start_line = line
            j = i + 2
            depth = 1
            while j < n and depth > 0:
                if source[j : j + 2] == "/*":
                    depth += 1
                    j += 2
                elif source[j : j + 2] == "*/":
                    depth -= 1
                    j += 2
                else:
                    if source[j] == "\n":
                        line += 1
                    j += 1
            text = source[start:j]
            for offset, part in enumerate(text.splitlines()):
                comments.append((start_line + offset, part))
            for k in range(start, j):
                if code_chars[k] != "\n":
                    code_chars[k] = " "
            i = j
            continue
        if c == '"':
            j = i + 1
            while j < n:
                if source[j] == "\\":
                    j += 2
                    continue
                if source[j] == "\n":
                    line += 1
                if source[j] == '"':
                    j += 1
                    break
                j += 1
            i = j
            continue
        if c == "'":
            # Char literal (`'x'`, `'\n'`) vs lifetime (`'a`): a char literal
            # closes with a `'` within a few characters; a lifetime does not.
            k = i + 1
            limit = min(n, i + 4)
            matched = False
            while k < limit:
                if source[k] == "\\":
                    k += 2
                    continue
                if source[k] == "'":
                    matched = True
                    k += 1
                    break
                k += 1
            i = k if matched else i + 1
            continue
        i += 1
    return comments, "".join(code_chars)


# --------------------------------------------------------------------------- #
# citation extraction
# --------------------------------------------------------------------------- #

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
# Final segment: snake_case-shaped — contains a literal underscore.
_SNAKE = rf"{_IDENT}_[A-Za-z0-9_]*"
BACKTICK_SPAN_RE = re.compile(r"`([^`\n]*)`")
CITATION_CONTENT_RE = re.compile(rf"^(?:{_IDENT}::)*({_SNAKE})(\(\))?$")


def citations_in_line(line: str) -> list[str]:
    """Backtick-quoted, snake_case-shaped citations found in one comment line."""
    out = []
    for span in BACKTICK_SPAN_RE.findall(line):
        if CITATION_CONTENT_RE.match(span):
            out.append(span)
    return out


def final_segment(citation: str) -> str:
    """The identifier a citation actually names: path/`()` stripped."""
    body = citation[:-2] if citation.endswith("()") else citation
    return body.rsplit("::", 1)[-1]


# --------------------------------------------------------------------------- #
# definition / method-call name pool
# --------------------------------------------------------------------------- #

DEF_RE = re.compile(
    r"\b(?:pub(?:\([^)]*\))?\s+)?(?:default\s+)?(?:async\s+)?(?:unsafe\s+)?"
    r"(?:extern\s+\"[^\"]*\"\s+)?"
    r"(?:fn|const|static(?:\s+mut)?|struct|enum|mod|type|trait)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)"
)
MACRO_RE = re.compile(r"macro_rules!\s*([A-Za-z_][A-Za-z0-9_]*)")
METHOD_CALL_RE = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*(?:::<[^>]*>)?\s*\(")
# Struct/enum field decl, named-struct-literal field, or named fn parameter:
# `<name>: <type-or-value>`. Broad by design — it also matches match-arm
# bindings and local block labels, which only widens acceptance, never hides
# a fabricated fn/test citation (see module docstring).
FIELD_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*:(?!:)")
# Exact contents of a double-quoted string literal, read from RAW source (not
# comment-blanked) — env-var names, `#[serde(rename = "...")]`, JSON/proto
# field-name string constants.
STRING_LITERAL_RE = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)"')
# `let [mut] <name>` local bindings — a large share of real, non-fabricated
# citations name a local variable ("the `grad_accum` buffer"), not an item.
LET_RE = re.compile(r"\blet\s+(?:mut\s+)?([A-Za-z_][A-Za-z0-9_]*)\b")


def names_from_source(source: str) -> set[str]:
    """Every def / macro / method-call / field / binding / literal name in `source`."""
    _comments, code = split_comments_and_code(source)
    names: set[str] = set(DEF_RE.findall(code))
    names |= set(MACRO_RE.findall(code))
    names |= set(METHOD_CALL_RE.findall(code))
    names |= set(FIELD_RE.findall(code))
    names |= set(LET_RE.findall(code))
    names |= set(STRING_LITERAL_RE.findall(source))
    return names


def crate_names() -> set[str]:
    """Every workspace crate's package name, as it appears in a Rust path (`-` → `_`)."""
    names: set[str] = set()
    for manifest in sorted((REPO_ROOT / "crates").glob("*/Cargo.toml")):
        for raw in manifest.read_text(errors="ignore").splitlines():
            m = PACKAGE_RE.match(raw.strip())
            if m:
                names.add(m.group(1).replace("-", "_"))
                break
    return names


def names_from_files(paths: list[Path]) -> set[str]:
    names: set[str] = set()
    for p in paths:
        try:
            source = p.read_text(errors="ignore")
        except OSError:
            continue
        names |= names_from_source(source)
    return names


# --------------------------------------------------------------------------- #
# repo file enumeration (git-tracked, deterministic)
# --------------------------------------------------------------------------- #


def tracked_rs_files() -> list[Path]:
    out: list[Path] = []
    for root in SCAN_DIRS:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--", f"{root}/**/*.rs", f"{root}/*.rs"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        for rel in result.stdout.split("\0"):
            if rel:
                out.append(REPO_ROOT / rel)
    return sorted(set(out))


def all_tracked_rs_files() -> list[Path]:
    """Every tracked `.rs` file in the repo (any crate) — the resolution universe."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.rs"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return sorted(REPO_ROOT / p for p in result.stdout.split("\0") if p)


def resolve_diff_base() -> str | None:
    candidates = [
        os.environ.get("SWARM_DIFF_BASE"),
        f"origin/{os.environ['GITHUB_BASE_REF']}" if os.environ.get("GITHUB_BASE_REF") else None,
        "origin/main",
        "main",
    ]
    for ref in candidates:
        if not ref:
            continue
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", ref],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return ref
    return None


def changed_rs_files(base: str) -> list[Path]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{base}...HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    out = []
    for rel in result.stdout.splitlines():
        if not rel.endswith(".rs"):
            continue
        if not any(rel == d or rel.startswith(d + "/") for d in SCAN_DIRS):
            continue
        p = REPO_ROOT / rel
        if p.is_file():
            out.append(p)
    return sorted(out)


# --------------------------------------------------------------------------- #
# optional bonus resolution source: vendored direct-dep registry sources
# --------------------------------------------------------------------------- #

DEP_SECTION_RE = re.compile(r"^\[(dependencies|dev-dependencies|build-dependencies)(\.[^\]]+)?\]$")
DEP_NAME_RE = re.compile(r'^([A-Za-z0-9_\-]+)\s*=')
PACKAGE_RE = re.compile(r'^name = "([^"]+)"$')
VERSION_RE = re.compile(r'^version = "([^"]+)"$')


def workspace_direct_dep_names() -> set[str]:
    """Every dependency name declared directly by the root or any crate's Cargo.toml."""
    names: set[str] = set()
    manifests = [REPO_ROOT / "Cargo.toml", *sorted((REPO_ROOT / "crates").glob("*/Cargo.toml"))]
    for manifest in manifests:
        if not manifest.is_file():
            continue
        in_section = False
        for raw in manifest.read_text(errors="ignore").splitlines():
            s = raw.strip()
            if DEP_SECTION_RE.match(s):
                in_section = True
                continue
            if s.startswith("["):
                in_section = False
                continue
            if in_section and s and not s.startswith("#"):
                m = DEP_NAME_RE.match(s)
                if m:
                    names.add(m.group(1))
    return names


def cargo_lock_versions() -> dict[str, str]:
    """`{package name: version}` from the top-level `Cargo.lock`, last-wins is fine."""
    lock = REPO_ROOT / "Cargo.lock"
    if not lock.is_file():
        return {}
    versions: dict[str, str] = {}
    name = None
    for raw in lock.read_text(errors="ignore").splitlines():
        s = raw.strip()
        m = PACKAGE_RE.match(s)
        if m:
            name = m.group(1)
            continue
        m = VERSION_RE.match(s)
        if m and name is not None:
            versions[name] = m.group(1)
            name = None
    return versions


def registry_src_roots() -> list[Path]:
    cargo_home = Path(os.environ.get("CARGO_HOME", str(Path.home() / ".cargo")))
    src = cargo_home / "registry" / "src"
    if not src.is_dir():
        return []
    try:
        return [p for p in src.iterdir() if p.is_dir()]
    except OSError:
        return []


def vendor_names() -> set[str]:
    """Bonus, opportunistic: def/method names from direct deps' vendored source.

    Never assumed present — the CI guard job that runs this gate has no
    toolchain/cargo step, so the registry cache is typically absent there; a
    legitimate vendor citation must go through `ci/cited_identifiers_allow.txt`
    to be CI-green regardless of whether this local lookup finds it.
    """
    roots = registry_src_roots()
    if not roots:
        return set()
    direct = workspace_direct_dep_names()
    if not direct:
        return set()
    versions = cargo_lock_versions()
    names: set[str] = set()
    for root in roots:
        try:
            children = list(root.iterdir())
        except OSError:
            continue
        for dep in direct:
            version = versions.get(dep)
            wanted = f"{dep}-{version}" if version else None
            for child in children:
                if not child.is_dir():
                    continue
                if wanted is not None:
                    if child.name != wanted:
                        continue
                elif not child.name.startswith(f"{dep}-"):
                    continue
                rs_files = list(child.rglob("*.rs"))
                names |= names_from_files(rs_files)
    return names


# --------------------------------------------------------------------------- #
# allowlist
# --------------------------------------------------------------------------- #


def load_allowlist() -> set[str]:
    if not ALLOWLIST_FILE.is_file():
        return set()
    names: set[str] = set()
    for raw in ALLOWLIST_FILE.read_text(errors="ignore").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        token = s.split("#", 1)[0].strip()
        if token:
            names.add(token)
    return names


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Failure:
    path: str
    line: int
    citation: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: `{self.citation}` does not resolve"


def scan_files(rs_files: list[Path], resolvable: set[str]) -> list[Failure]:
    failures: list[Failure] = []
    for path in rs_files:
        try:
            source = path.read_text(errors="ignore")
        except OSError:
            continue
        comments, _code = split_comments_and_code(source)
        rel = str(path.relative_to(REPO_ROOT)) if path.is_absolute() else str(path)
        for line_no, text in comments:
            for citation in citations_in_line(text):
                if final_segment(citation) not in resolvable:
                    failures.append(Failure(rel, line_no, citation))
    return sorted(failures, key=lambda f: (f.path, f.line, f.citation))


def main() -> int:
    argv = sys.argv[1:]
    if "--self-test" in argv:
        return self_test()

    changed_only = "--changed-only" in argv

    all_rs = all_tracked_rs_files()
    resolvable = names_from_files(all_rs)
    resolvable |= crate_names()
    resolvable |= vendor_names()
    resolvable |= load_allowlist()

    if changed_only:
        base = resolve_diff_base()
        if base is None:
            print(
                "check-cited-identifiers: --changed-only requested but no diff base "
                "available (SWARM_DIFF_BASE / origin/<base> / origin/main / main) — "
                "falling back to a full scan.",
                file=sys.stderr,
            )
            scan_targets = tracked_rs_files()
        else:
            scan_targets = changed_rs_files(base)
            print(f"check-cited-identifiers: --changed-only vs `{base}`.")
    else:
        scan_targets = tracked_rs_files()

    failures = scan_files(scan_targets, resolvable)

    if failures:
        print("check-cited-identifiers: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        print(
            f"\ncheck-cited-identifiers: {len(failures)} citation(s) do not resolve to "
            "a real definition/method/allowlisted-vendor-name. Fix the name, or add it "
            "to ci/cited_identifiers_allow.txt with a reason if it is a deliberate "
            "vendored/upstream reference.",
            file=sys.stderr,
        )
        return 1

    print(f"check-cited-identifiers: OK — {len(scan_targets)} file(s) scanned, all citations resolve.")
    return 0


# --------------------------------------------------------------------------- #
# self-test — synthetic in-memory fragments (same style as check_sqlite_isms.py)
# --------------------------------------------------------------------------- #

RESOLVING_FIXTURE_DEFN = """
fn with_scale_refuses_zero_negative_nan_and_infinite() {}
"""

RESOLVING_FIXTURE_COMMENT = """
// see `with_scale_refuses_zero_negative_nan_and_infinite` below for the
// exhaustive domain-refusal cases.
fn caller() {}
"""

FABRICATED_FIXTURE_COMMENT = """
// see `softmax_scale_with_scale_refuses_the_invalid_domain` below — no such
// fn exists anywhere in the repo.
fn other_caller() {}
"""

NON_CITATION_FIXTURE_COMMENT = """
// see `check_doc_parity.py` and `NotRecomputable` — neither is a citation
// this gate's regex should even attempt to resolve.
"""


def self_test() -> int:
    failures: list[str] = []

    resolvable = names_from_source(RESOLVING_FIXTURE_DEFN)
    if "with_scale_refuses_zero_negative_nan_and_infinite" not in resolvable:
        failures.append("self-test FAILED: definition scan did not find the fixture fn")

    good = scan_files_on_text(RESOLVING_FIXTURE_COMMENT, "<self-test: resolving>", resolvable)
    if good:
        failures.append(f"self-test FAILED: a resolving citation was flagged: {good}")

    bad = scan_files_on_text(FABRICATED_FIXTURE_COMMENT, "<self-test: fabricated>", resolvable)
    if len(bad) != 1 or bad[0].citation != "softmax_scale_with_scale_refuses_the_invalid_domain":
        failures.append(
            f"self-test FAILED: expected exactly the fabricated citation flagged, got: {bad}"
        )

    none_found = scan_files_on_text(
        NON_CITATION_FIXTURE_COMMENT, "<self-test: non-citation>", resolvable
    )
    if none_found:
        failures.append(
            f"self-test FAILED: a non-citation (no underscore / has '.py') was flagged: {none_found}"
        )

    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        print("check-cited-identifiers self-test: FAIL", file=sys.stderr)
        return 1

    print(
        "check-cited-identifiers self-test: OK — fabricated citation flagged, "
        "resolving and non-citation fragments clean."
    )
    return 0


def scan_files_on_text(text: str, label: str, resolvable: set[str]) -> list[Failure]:
    """`scan_files`'s core logic, applied to an in-memory fragment (no file I/O)."""
    comments, _code = split_comments_and_code(text)
    failures: list[Failure] = []
    for line_no, line in comments:
        for citation in citations_in_line(line):
            if final_segment(citation) not in resolvable:
                failures.append(Failure(label, line_no, citation))
    return failures


if __name__ == "__main__":
    sys.exit(main())
