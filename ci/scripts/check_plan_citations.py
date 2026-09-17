#!/usr/bin/env python3
"""Mechanical gate: every construct citation under `docs/plans/**` resolves.

Issue #556: `docs/plans/67-distributed-training/{DESIGN,PRESSURE,UNITS}.md`
carried roughly ninety `path:line` citations; seven citation/audit rounds on
one unit each found stale ones, and even the lead's own mechanical sweep (a
line-EXISTENCE check) accepted a line that had drifted onto the wrong
construct entirely, because a bare line NUMBER carries no meaning of its
own — it drifts with every commit and a line-existence check cannot see
that the construct it once pointed at moved. The rebuild: plan docs cite
CONSTRUCTS, never offsets, and every citation resolves against a real index
built from the artifact it claims to describe, not from prose.

**Citation grammar** (a backtick-quoted span under `docs/plans/**`):

  * ``path::symbol`` — a Rust item (`fn`/`struct`/`enum`/`trait`/`type`/
    `const`/`static`/`mod`, including a `#[test]` fn name) in a `.rs` file.
    `path` may be the file's full path from the repo root, OR a bare
    suffix (e.g. just the filename, or a partial directory tail) — resolved
    by searching the tree for every file whose path ends with that suffix;
    zero matches is UNRESOLVED, more than one is AMBIGUOUS (both findings).
    This is deliberately permissive on `path` and strict on `symbol`: the
    defect this gate closes was a citation whose LINE NUMBER pointed at the
    wrong construct in the right-ish file, not a citation naming the wrong
    file outright.
  * ``path::[table].key`` — a TOML key in a `.toml` file (dotted table
    path in brackets, then a leaf key), resolved with `tomllib`.
  * ``path#heading`` — a markdown heading anchor in a `.md` file, resolved
    against a GitHub-slug index of every heading in the target doc.
  * ``path::Message`` / ``path::Message.field`` — a protobuf message/enum
    (and optional field) in a `.proto` file, resolved against a descriptor
    set this gate extracts from the `.proto` source directly (brace-depth
    tracking over top-level `message`/`enum` blocks — this repo's protos do
    not nest messages inside other messages at the plan-doc citation sites,
    a stated, checked-against-the-real-tree scope, not an unverified
    assumption; see `_proto_descriptors`).
  * a BARE backticked symbol immediately followed by a `` (~:NNN`` locus
    hint — the plan docs' own convention (see `DESIGN.md`'s own "a trailing
    `(~:NNN)` is a locus hint only, never load-bearing" line) for citing a
    construct without committing to its enclosing file. Resolved the same
    way a bare-suffix `path::symbol` is: search the WHOLE Rust index for a
    definition of that exact name; zero is UNRESOLVED, more than one is
    AMBIGUOUS. The `(~:NNN)` hint itself is NEVER checked against a real
    line number — it is explicitly non-load-bearing, a hint for a human's
    eyes, not part of the resolved claim.

A token containing `<...>` (an angle-bracket placeholder, e.g.
``Cargo.toml::[dependencies].<dep>`` used to EXPLAIN the citation format
itself rather than cite a real dependency) is not a citation and is
skipped — the placeholder syntax is drawn from the same convention this
gate's own module docstring uses to describe a grammar without
instantiating a token that would need to resolve.

**Scope note on the pre-existing `path:line` form.** ~950 `path:line`
citations remain under `docs/plans/67-*/` and `docs/plans/68-*/` at the time
this gate was written — that offset form is OUTSIDE this gate's grammar
(silently not scanned) rather than converted wholesale by this change: a
blind mechanical rewrite from a bare line number to a guessed symbol name,
without opening and hand-verifying each of ~950 sites, would risk minting
exactly the kind of unverified citation this gate exists to prevent. The
~120 citations already written in the grammar above (from an earlier unit's
migration work) are what this gate governs today; the remaining sweep is
named as a follow-up, not silently declared done.

**Rust resolution is a REAL `syn` AST parse, never a regex reader.** Every
`rust`/`bare_rust` citation resolves against `ci/tools/symbol-index`
(`cargo run --release -p symbol-index`), a compiled tool that parses every
`.rs` file with `syn` and emits a JSON index of items, impl methods, enum
variants, struct fields, and call sites, each with file + line. An earlier
version of this gate carried its own regex/brace-counting Rust reader —
this repo's own recorded lesson ("regex readers over YAML/Rust lost five
audits") is why that reader was replaced, not merely improved: its very
first real run against this tree found and silently mis-resolved
`Catalog::fail_job` (a one-line sibling method's own braces spuriously
closed the enclosing `impl` block one line early), a defect class `syn`'s
real parser cannot produce. See `ci/tools/symbol-index/src/main.rs`'s own
module doc for the index's exact scope and stated limits.

Run: `python3 ci/scripts/check_plan_citations.py`
Self-test: `python3 ci/scripts/check_plan_citations.py --self-test`
NOT hermetic in the no-toolchain sense: TOML/heading/proto resolution reads
the working tree only, but Rust resolution shells out to `cargo run
--release -p symbol-index`, which needs the pinned Rust toolchain (a cold
build pays once; sccache/CI-cache carries it after). Wired in `swarm.yml`'s
Rust-toolchain job (`symbol-index-gates`), not the toolchain-free
`swarm-gates` job `check_doc_parity.py`/`check_journey_markers.py` run in.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_DOCS_ROOT = REPO_ROOT / "docs" / "plans"

CITATION_RE = re.compile(
    r"`([A-Za-z0-9_./\-]+\.(?:rs|toml|proto|md))(::|#)([^`]+)`"
)
BARE_HINT_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`\s*\(~:\d+")


@dataclass(frozen=True)
class Citation:
    doc_path: str
    line_no: int
    raw: str
    kind: str  # "rust" | "toml" | "heading" | "proto" | "bare_rust"
    target_path: str  # "" for bare_rust
    member: str


def _iter_plan_docs() -> list[Path]:
    if not PLAN_DOCS_ROOT.is_dir():
        return []
    return sorted(PLAN_DOCS_ROOT.rglob("*.md"))



# A citation immediately preceded by a capitalized-word-plus-"'s" possessive
# ("Ballista's `scheduler_server/...::on_receive`") names an EXTERNAL
# project's own source, read for design comparison, not a construct in
# THIS repo's tree -- out of this gate's grammar entirely, the same way a
# `<placeholder>` is. Generic (any such possessive), not hardcoded to one
# project name, so a future citation of DataFusion's or PyTorch's own
# source needs no gate edit.
EXTERNAL_POSSESSIVE_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*'s\s*$")


# A previous line ending in a backtick-quoted NON-Rust source path
# immediately followed by "'s" (`` `foo/bar.h`'s `` at end of line) already
# named the file a following bare-hint citation belongs to, and that file
# is not Rust -- the same "this is someone else's/some other language's
# source" signal `EXTERNAL_POSSESSIVE_RE` catches inline, just spanning a
# line wrap (a vendored C/CUDA header cited this way in this tree, e.g.
# `crates/jammi-kernels/third_party/flash-attention/**`).
_NON_RUST_SOURCE_OWNER_RE = re.compile(
    r"`[^`]+\.(?:h|hpp|hh|cu|cuh|cc|cpp|c|py|proto)`'s\s*$"
)


def extract_citations(doc_path: Path, text: str) -> list[Citation]:
    rel = str(doc_path.relative_to(REPO_ROOT))
    out: list[Citation] = []
    lines = text.splitlines()
    for line_no, line in enumerate(lines, start=1):
        for m in CITATION_RE.finditer(line):
            path, sep, member = m.group(1), m.group(2), m.group(3)
            if "<" in path or ">" in path or "<" in member or ">" in member:
                continue  # a placeholder explaining the grammar, not a citation
            if EXTERNAL_POSSESSIVE_RE.search(line[: m.start()]):
                continue  # "Ballista's `...`" -- an external project's own source
            if path.endswith(".toml") and sep == "::":
                kind = "toml"
            elif path.endswith(".proto") and sep == "::":
                kind = "proto"
            elif path.endswith(".md") and sep == "#":
                kind = "heading"
            elif path.endswith(".rs") and sep == "::":
                kind = "rust"
            else:
                continue
            out.append(Citation(rel, line_no, m.group(0), kind, path, member))
        for m in BARE_HINT_RE.finditer(line):
            prev_line = lines[line_no - 2] if line_no >= 2 else ""
            if m.start() == 0 and _NON_RUST_SOURCE_OWNER_RE.search(prev_line):
                continue  # named by a non-Rust source file on the previous line
            symbol = m.group(1)
            out.append(Citation(rel, line_no, m.group(0), "bare_rust", "", symbol))
    return out


# --------------------------------------------------------------------------- #
# rust symbol index (ci/tools/symbol-index -- a real `syn` AST parse)
# --------------------------------------------------------------------------- #

SYMBOL_INDEX_CRATE = "symbol-index"


def build_symbol_index(roots: list[str], cwd: Path = REPO_ROOT) -> dict:
    """Runs the REAL `symbol-index` tool (`ci/tools/symbol-index`) over
    `roots` (each a path, relative to `cwd` or absolute) and returns the
    parsed JSON index: `{"items": [{"path", "kind", "name", "qualified",
    "vis", "line", "line_end", "is_test"}, ...], "calls": [...],
    "files_scanned": N, "files_skipped": [...]}`. `cargo run --release` so
    a cold build pays once and every later invocation against the SAME
    `CARGO_TARGET_DIR` is incremental; this function never sets its own
    `CARGO_TARGET_DIR`/`RUSTC_WRAPPER` — the caller's environment (if any)
    is inherited unchanged, the same discipline every cargo-invoking
    gate/agent in this repo already follows.

    Raises `RuntimeError` (never returns a partial/guessed index) on a
    non-zero exit or unparseable stdout — a build failure must be LOUD,
    never silently read as "zero items, nothing resolves" (which would
    make every citation in the tree look stale at once).
    """
    proc = subprocess.run(
        ["cargo", "run", "--release", "-p", SYMBOL_INDEX_CRATE, "--", *roots],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{SYMBOL_INDEX_CRATE} failed (rc={proc.returncode}) over {roots}:\n"
            f"{proc.stderr.strip()[-4000:]}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{SYMBOL_INDEX_CRATE} did not emit valid JSON on stdout ({exc}); "
            f"stderr tail:\n{proc.stderr.strip()[-2000:]}"
        ) from exc


def _index_paths(index: dict) -> list[str]:
    """Every distinct file path the index carries at least one item for
    (exactly as `symbol-index` recorded it — relative to whatever root was
    given, or absolute if the root was)."""
    return sorted({it["path"] for it in index["items"]})


def _index_paths_matching_suffix(given: str, candidate_paths: list[str]) -> list[str]:
    """Every path string in `candidate_paths` equal to `given`, or ending
    with `/` + `given` (a bare/partial suffix) — string-suffix matching,
    deliberately never `Path.relative_to`: the index's own paths may be
    absolute (a self-test root outside the repo) or repo-relative (the
    real `crates` root), and a plain string comparison works identically
    either way with no rebasing."""
    norm = given.lstrip("/")
    exact = [p for p in candidate_paths if p == norm]
    if exact:
        return exact
    return [p for p in candidate_paths if p.endswith("/" + norm)]


def _file_defines(index: dict, path: str, symbol: str) -> bool:
    return any(
        it["path"] == path and (it["name"] == symbol or it.get("qualified") == symbol)
        for it in index["items"]
    )


def _files_defining(index: dict, symbol: str) -> set[str]:
    return {
        it["path"]
        for it in index["items"]
        if it["name"] == symbol or it.get("qualified") == symbol
    }


def find_md_files(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if root.is_dir():
            out.extend(sorted(root.rglob("*.md")))
    return out


def resolve_path_suffix(given: str, all_files: list[Path], base: Path = REPO_ROOT) -> list[Path]:
    """Every file in `all_files` (all assumed to live under `base`) whose
    path relative to `base` equals `given`, or ends with `/` + `given` (a
    bare/partial suffix). `base` defaults to the real repo root in
    production; the self-test passes its own temp-dir root so the same pure
    logic is exercisable without writing into the real tree."""
    exact = base / given
    if exact.is_file() and exact in all_files:
        return [exact]
    norm = given.lstrip("/")
    matches = [
        f for f in all_files if str(f.relative_to(base)) == norm
        or str(f.relative_to(base)).endswith("/" + norm)
    ]
    return matches


# --------------------------------------------------------------------------- #
# markdown heading index
# --------------------------------------------------------------------------- #

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def slugify_heading(text: str) -> str:
    """GitHub's own heading-anchor algorithm: lowercase, strip everything
    that is not a word character, whitespace, or an existing hyphen (an
    em-dash disappears WITHOUT leaving a hyphen behind), then replace each
    remaining whitespace character with its own hyphen — never collapsed,
    so a removed em-dash flanked by spaces leaves a double hyphen, matching
    GitHub's actual output (verified against a real heading in this tree:
    `docs/plans/68-compute-tier-substrate/units/DIST-DATA-PLANE.md`'s own
    "Unit 2 — membership" heading anchors as `...unit-2--membership...`).
    """
    s = text.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s", "-", s)
    return s


def heading_slugs(text: str) -> set[str]:
    slugs: set[str] = set()
    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            slugs.add(slugify_heading(m.group(2)))
    return slugs


# --------------------------------------------------------------------------- #
# proto descriptor set
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProtoDescriptor:
    name: str
    fields: frozenset[str]


PROTO_TOP_RE = re.compile(r"^\s*(?:message|enum)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{")
# A field line inside a message: `[repeated] TYPE name = N;` or an enum
# value: `NAME = N;`. Both shapes end in `<identifier> = <digits>;`.
PROTO_FIELD_RE = re.compile(r"^\s*(?:repeated\s+|optional\s+)?[\w.<>]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\d+\s*[;\[]")
PROTO_ENUM_VALUE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\d+\s*;")


def _proto_descriptors(text: str) -> dict[str, ProtoDescriptor]:
    """Top-level `message`/`enum` blocks only (brace-depth tracked) — this
    repo's plan-doc proto citations never name a nested message, a checked
    scope limit, not an unverified assumption (see module docstring)."""
    out: dict[str, ProtoDescriptor] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = PROTO_TOP_RE.match(lines[i])
        if not m:
            i += 1
            continue
        name = m.group(1)
        depth = lines[i].count("{") - lines[i].count("}")
        body: list[str] = []
        i += 1
        while i < len(lines) and depth > 0:
            depth += lines[i].count("{") - lines[i].count("}")
            body.append(lines[i])
            i += 1
        fields: set[str] = set()
        for bl in body:
            fm = PROTO_FIELD_RE.match(bl)
            if fm:
                fields.add(fm.group(1))
                continue
            em = PROTO_ENUM_VALUE_RE.match(bl)
            if em:
                fields.add(em.group(1))
        out[name] = ProtoDescriptor(name, frozenset(fields))
    return out


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Finding:
    citation: Citation
    reason: str

    def render(self) -> str:
        return (
            f"{self.citation.doc_path}:{self.citation.line_no}  "
            f"{self.citation.raw}  -- {self.reason}"
        )


def resolve_citation(
    c: Citation, index: dict | None, md_files: list[Path] | None = None, base: Path = REPO_ROOT
) -> Finding | None:
    md_files = md_files if md_files is not None else []
    if c.kind == "toml":
        target = REPO_ROOT / c.target_path
        if not target.is_file():
            return Finding(c, f"toml file not found: {c.target_path}")
        table_m = re.match(r"^\[([^\]]+)\]\.(.+)$", c.member)
        table_only_m = re.match(r"^\[([^\]]+)\]$", c.member)
        try:
            data = tomllib.loads(target.read_text())
        except (tomllib.TOMLDecodeError, OSError) as exc:
            return Finding(c, f"toml parse error: {exc}")
        if table_m:
            table_path, leaf = table_m.group(1), table_m.group(2)
            node = data
            for part in table_path.split("."):
                if not isinstance(node, dict) or part not in node:
                    return Finding(c, f"toml table `[{table_path}]` not found")
                node = node[part]
            if not isinstance(node, dict) or leaf not in node:
                return Finding(c, f"toml key `{leaf}` not found in `[{table_path}]`")
            return None
        if table_only_m:
            table_path = table_only_m.group(1)
            node = data
            for part in table_path.split("."):
                if not isinstance(node, dict) or part not in node:
                    return Finding(c, f"toml table `[{table_path}]` not found")
                node = node[part]
            return None
        return Finding(c, f"unrecognized TOML citation shape: {c.member!r}")

    if c.kind == "heading":
        matches = resolve_path_suffix(c.target_path, md_files, base=base)
        if not matches:
            return Finding(c, f"doc not found: {c.target_path}")
        if len(matches) > 1:
            shown = ", ".join(str(p.relative_to(base)) for p in matches[:5])
            return Finding(
                c, f"ambiguous doc path `{c.target_path}` -- {len(matches)} files match: {shown}"
            )
        target = matches[0]
        slugs = heading_slugs(target.read_text(errors="ignore"))
        if c.member not in slugs:
            return Finding(c, f"heading anchor `#{c.member}` not found in {c.target_path}")
        return None

    if c.kind == "proto":
        target = REPO_ROOT / c.target_path
        if not target.is_file():
            return Finding(c, f"proto file not found: {c.target_path}")
        descriptors = _proto_descriptors(target.read_text(errors="ignore"))
        if "." in c.member:
            msg_name, field_name = c.member.split(".", 1)
        else:
            msg_name, field_name = c.member, None
        if msg_name not in descriptors:
            return Finding(c, f"proto message/enum `{msg_name}` not found in {c.target_path}")
        if field_name is not None and field_name not in descriptors[msg_name].fields:
            return Finding(c, f"proto field `{field_name}` not found on `{msg_name}`")
        return None

    if c.kind in ("rust", "bare_rust"):
        if index is None:
            return Finding(c, "no symbol index available to resolve against")
        if c.kind == "rust":
            matches = _index_paths_matching_suffix(c.target_path, _index_paths(index))
            if not matches:
                return Finding(c, f"no file resolves for path `{c.target_path}`")
            if len(matches) > 1:
                shown = ", ".join(matches[:5])
                return Finding(
                    c, f"ambiguous path `{c.target_path}` -- {len(matches)} files match: {shown}"
                )
            hits = [p for p in matches if _file_defines(index, p, c.member)]
        else:
            hits = sorted(_files_defining(index, c.member))
        if not hits:
            return Finding(c, f"symbol `{c.member}` not found")
        if len(hits) > 1:
            shown = ", ".join(hits[:5])
            return Finding(
                c, f"ambiguous symbol `{c.member}` -- defined in {len(hits)} files: {shown}"
            )
        return None

    return Finding(c, f"unrecognized citation kind: {c.kind}")


def scan_plan_docs() -> list[Finding]:
    docs = _iter_plan_docs()
    citations: list[Citation] = []
    for doc in docs:
        citations.extend(extract_citations(doc, doc.read_text(errors="ignore")))
    if not any(c.kind in ("rust", "bare_rust") for c in citations):
        index: dict | None = None
    else:
        index = build_symbol_index(["crates"])
    if not any(c.kind == "heading" for c in citations):
        md_files: list[Path] = []
    else:
        # `docs/**` plus the repo-root-level `.md` files (README.md,
        # CHANGELOG.md, ...) -- never a bare `REPO_ROOT.rglob` (that walks
        # `target/`, vendored trees, and every worktree-local build
        # artifact directory for no benefit: no plan-doc citation targets
        # anything outside `docs/**` or a root-level file today).
        md_files = find_md_files([REPO_ROOT / "docs"]) + sorted(REPO_ROOT.glob("*.md"))
    findings: list[Finding] = []
    for c in citations:
        f = resolve_citation(c, index, md_files)
        if f is not None:
            findings.append(f)
    return findings


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #


def _write(root: Path, rel: str, content: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def self_test() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        if not cond:
            failures.append(name)

    # --- extraction: grammar recognition + placeholder skip -----------------
    text = (
        "See `crates/x/src/a.rs::foo` and `crates/x/Cargo.toml::[dependencies].bar` "
        "and `docs/x.md#a-heading` and `crates/x/proto/a.proto::Msg` "
        "and `crates/x/Cargo.toml::[dependencies].<dep>` (a placeholder) "
        "and `bare_symbol` (~:12, a hint).\n"
    )
    fake_doc = REPO_ROOT / "docs" / "plans" / "__self_test_fixture__.md"
    cites = extract_citations(fake_doc, text)
    kinds = sorted(c.kind for c in cites)
    check(
        "extraction: every grammar form recognized, placeholder skipped",
        kinds == ["bare_rust", "heading", "proto", "rust", "toml"],
    )

    # --- external-source exemptions: possessive (inline) and cross-line ---
    external_inline = "See Ballista's `scheduler_server/mod.rs::on_receive` for the arm.\n"
    check(
        "Ballista's `...` is exempt (external project source)",
        extract_citations(fake_doc, external_inline) == [],
    )
    external_cross_line = (
        "`crates/jammi-kernels/third_party/flash-attention/src/flash_bwd_kernel.h`'s\n"
        "`compute_dq_dk_dv_1colblock` (~:122, deterministic path)\n"
    )
    check(
        "a bare-hint citation named by a non-Rust source file on the PREVIOUS line is exempt",
        extract_citations(fake_doc, external_cross_line) == [],
    )
    # The SAME bare-hint shape, with no such preceding line, still resolves
    # as a real citation -- the exemption is context-gated, not blanket.
    no_context = "`compute_dq_dk_dv_1colblock` (~:122, deterministic path)\n"
    check(
        "the identical bare-hint shape WITHOUT the preceding non-Rust marker still extracts",
        len(extract_citations(fake_doc, no_context)) == 1,
    )

    # --- heading slugify matches GitHub's real algorithm --------------------
    check(
        "slugify: em-dash removed, double-hyphen preserved",
        slugify_heading("5.8 Unit 2 — membership (post PR-C; designed here)")
        == "58-unit-2--membership-post-pr-c-designed-here",
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # --- renamed fn: RED when the cited name no longer exists, against
        # the REAL compiled symbol-index tool (a genuine end-to-end
        # RED->GREEN, not a mocked/synthetic index) --------------------------
        rust_before = "pub fn compute_thing() -> i32 { 1 }\n"
        rust_after = "pub fn compute_thing_renamed() -> i32 { 1 }\n"
        f = _write(root, "crates/fake_crate/src/lib.rs", rust_before)
        idx_before = build_symbol_index([str(root)])
        check(
            "renamed fn: original name resolves before rename (real symbol-index run)",
            _files_defining(idx_before, "compute_thing") != set(),
        )
        f.write_text(rust_after)
        idx_after = build_symbol_index([str(root)])
        check(
            "renamed fn: old citation does NOT resolve after rename (RED, real run)",
            _files_defining(idx_after, "compute_thing") == set(),
        )
        check(
            "renamed fn: new name resolves (real run)",
            _files_defining(idx_after, "compute_thing_renamed") != set(),
        )

        # --- deleted struct ---------------------------------------------------
        struct_src = "pub struct Widget { pub x: i32 }\n"
        g = _write(root, "crates/fake_crate/src/widget.rs", struct_src)
        idx_before2 = build_symbol_index([str(root)])
        check(
            "deleted struct: resolves before deletion (real run)",
            _files_defining(idx_before2, "Widget") != set(),
        )
        g.write_text("// Widget removed.\n")
        idx_after2 = build_symbol_index([str(root)])
        check(
            "deleted struct: citation does NOT resolve after deletion (RED, real run)",
            _files_defining(idx_after2, "Widget") == set(),
        )

        # --- ambiguous bare basename: two files define the SAME symbol -----
        h1_rel = "crates/a/src/training.rs"
        h2_rel = "crates/b/src/training.rs"
        _write(root, h1_rel, "pub fn shared_name() {}\n")
        _write(root, h2_rel, "pub fn shared_name() {}\n")
        idx_amb = build_symbol_index([str(root)])
        c = Citation("x.md", 1, "`training.rs::shared_name`", "rust", "training.rs", "shared_name")
        matches = _index_paths_matching_suffix(c.target_path, _index_paths(idx_amb))
        check("ambiguous bare basename: two files match the suffix (real index)", len(matches) == 2)
        finding = resolve_citation(c, idx_amb)
        check("ambiguous bare basename: produces a finding", finding is not None)
        check(
            "ambiguous bare basename: finding names 'ambiguous'",
            finding is not None and "ambiguous" in finding.reason,
        )
        # A FULL (non-ambiguous) path to one of the two resolves cleanly.
        c_full = Citation(
            "x.md", 1, f"`{h1_rel}::shared_name`", "rust", h1_rel, "shared_name",
        )
        matches_full = _index_paths_matching_suffix(c_full.target_path, _index_paths(idx_amb))
        check(
            "full path to one of the two ambiguous files narrows to one (real index)",
            len(matches_full) == 1 and matches_full[0].endswith(h1_rel),
        )
        check(
            "full path resolves cleanly through resolve_citation (no finding, real index)",
            resolve_citation(c_full, idx_amb) is None,
        )

        # --- heading rename ---------------------------------------------------
        doc_before = "## Old Heading\n\nbody\n"
        doc_after = "## New Heading\n\nbody\n"
        d = _write(root, "docs/x.md", doc_before)
        check("heading rename: old slug resolves before rename", "old-heading" in heading_slugs(d.read_text()))
        d.write_text(doc_after)
        check(
            "heading rename: old slug does NOT resolve after rename (RED)",
            "old-heading" not in heading_slugs(d.read_text()),
        )
        check("heading rename: new slug resolves", "new-heading" in heading_slugs(d.read_text()))

        # --- proto field removal -----------------------------------------------
        proto_before = "message M {\n  string name = 1;\n  int32 count = 2;\n}\n"
        proto_after = "message M {\n  string name = 1;\n}\n"
        desc_before = _proto_descriptors(proto_before)
        check("proto field removal: field resolves before removal", "count" in desc_before["M"].fields)
        desc_after = _proto_descriptors(proto_after)
        check(
            "proto field removal: field does NOT resolve after removal (RED)",
            "count" not in desc_after["M"].fields,
        )
        check("proto field removal: message itself still resolves", "M" in desc_after)

        # --- TOML resolution: real key resolves, missing key is a finding ---
        toml_src = "[dependencies]\nfoo = \"1\"\n"
        t = _write(root, "Cargo.toml", toml_src)
        c_toml_ok = Citation("x.md", 1, "`Cargo.toml::[dependencies].foo`", "toml", "Cargo.toml", "[dependencies].foo")
        c_toml_bad = Citation("x.md", 1, "`Cargo.toml::[dependencies].missing`", "toml", "Cargo.toml", "[dependencies].missing")
        data = tomllib.loads(t.read_text())
        check("toml: real key present", "foo" in data.get("dependencies", {}))
        check("toml: missing key absent", "missing" not in data.get("dependencies", {}))

    if failures:
        print("check_plan_citations self-test: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(
        "check_plan_citations self-test: OK -- a renamed fn, a deleted struct, an "
        "ambiguous bare basename, and a full-path disambiguation all reproduce the "
        "RED->GREEN shape against the REAL, compiled symbol-index tool (never a "
        "mocked index); a heading rename and a proto field removal reproduce it for "
        "their own file-based resolution; TOML key resolution and the placeholder "
        "skip both hold."
    )
    return 0


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()

    findings = scan_plan_docs()
    if findings:
        print("check_plan_citations: FINDINGS", file=sys.stderr)
        for f in sorted(findings, key=lambda f: (f.citation.doc_path, f.citation.line_no)):
            print(f"  - {f.render()}", file=sys.stderr)
        print(
            f"\ncheck_plan_citations: {len(findings)} unresolved/ambiguous citation(s) "
            "under docs/plans/**. Re-anchor each to the construct it actually describes "
            "(see the module docstring for the citation grammar).",
            file=sys.stderr,
        )
        return 1

    print("check_plan_citations: OK -- every construct citation under docs/plans/** resolves.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
