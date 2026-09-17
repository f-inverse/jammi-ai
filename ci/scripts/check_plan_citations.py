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

Run: `python3 ci/scripts/check_plan_citations.py`
Self-test: `python3 ci/scripts/check_plan_citations.py --self-test`
Hermetic: reads the working tree only; no network, no build, no `cargo`.
Wired in `swarm.yml` beside `check_doc_parity.py`.
"""

from __future__ import annotations

import re
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
# rust symbol index
# --------------------------------------------------------------------------- #

# The same shape of regex-based item scan `check_no_consumer_names.py`'s own
# `PUB_DECL_RE` already uses for this repo's Rust source (precedent, not a
# full `syn`-AST parse — a real Rust AST walk would need a compiled helper
# this Python gate does not carry; see the contract deviation this scope
# note names). Any visibility (not `pub`-only, since a plan doc legitimately
# cites a crate-private helper), plus `#[test] fn NAME` bodies.
_PUB_PREFIX = r"(?:pub(?:\s*\([^)]*\))?\s+)?"
IMPL_RE = re.compile(
    r"^\s*impl(?:<[^>]*>)?\s+(?:[\w:]+(?:<[^>]*>)?\s+for\s+)?([A-Za-z_][\w:]*)"
)
ENUM_RE = re.compile(rf"^\s*{_PUB_PREFIX}enum\s+([A-Za-z_][A-Za-z0-9_]*)")
STRUCT_RE = re.compile(rf"^\s*{_PUB_PREFIX}struct\s+([A-Za-z_][A-Za-z0-9_]*)")
FN_RE = re.compile(rf"^\s*{_PUB_PREFIX}(?:async\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)")
OTHER_ITEM_RE = re.compile(
    rf"^\s*{_PUB_PREFIX}(?:unsafe\s+)?(?:trait|type|const|static|mod)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)"
)
VARIANT_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\b")
# A struct FIELD line: `[pub[(...)]] name: Type,` — scoped to a `struct`
# container only (see the char-scan loop), so it never mistakes a `match`
# arm or a block-local `let` binding for a field.
FIELD_RE = re.compile(rf"^\s*{_PUB_PREFIX}([A-Za-z_][A-Za-z0-9_]*)\s*:\s*[^:]")


def _rust_items_in_text(text: str) -> set[str]:
    """Every bare item name (`fn`/`struct`/`enum`/`trait`/`type`/`const`/
    `static`/`mod`, a `#[test]` fn included -- `#[test]` itself never
    matches `FN_RE`, but the plain `fn NAME` line right after it always
    does, `pub` or not), PLUS a qualified `Type::member` for every method
    defined directly inside an `impl Type { ... }` / `impl Trait for Type
    { ... }` block and every variant defined directly inside an `enum Name
    { ... }` body -- both of these are exactly the shapes this repo's plan
    docs actually cite (`Catalog::claim_next`, `AnchorKind::
    UnpinnedAtInstant`).

    A brace-depth-tracked pass, rustfmt-style-assuming (a block's opening
    `{` is on the SAME line as its `impl`/`enum` keyword — this repo's
    `cargo fmt --check` gate makes that a checked assumption, not an
    unverified one). Braces are walked CHARACTER BY CHARACTER, in true
    left-to-right order, never as a per-line "count every `}` then count
    every `{`" batch — a one-line method (`pub fn is_empty(&self) -> bool {
    self.len == 0 }`, common after `rustfmt`) carries both an open and a
    close on the SAME line, and batching would process that close BEFORE
    the open even though the open comes first in the text, spuriously
    popping the ENCLOSING `impl` block off the stack for the rest of the
    file (the exact defect an executed run against this repo's own
    `crates/jammi-db/src/catalog/jobs_repo.rs` found: every method after
    the first one-line method in `impl Catalog` silently stopped
    qualifying). Nested items past one level (a closure or a local `fn`
    inside a method body) are still recorded as bare names but never
    qualified — a stated, narrow limit: this repo's plan docs do not cite
    those.
    """
    names: set[str] = set()
    depth = 0
    stack: list[tuple[int, str, str]] = []  # (body_depth, kind, name)
    for raw_line in text.splitlines():
        # Declarations are always the FIRST token on their own line
        # (rustfmt), so the depth/container to classify them by is the
        # depth as of the START of this line -- untouched by any brace
        # this same line goes on to open or close.
        container = stack[-1] if stack and stack[-1][0] == depth else None

        m_impl = IMPL_RE.match(raw_line)
        m_enum = ENUM_RE.match(raw_line)
        m_struct = STRUCT_RE.match(raw_line)
        m_fn = FN_RE.match(raw_line)
        m_other = OTHER_ITEM_RE.match(raw_line)

        if m_impl:
            names.add(m_impl.group(1).split("::")[-1])
        elif m_enum:
            names.add(m_enum.group(1))
        elif m_struct:
            names.add(m_struct.group(1))
        elif m_fn:
            name = m_fn.group(1)
            names.add(name)
            if container is not None and container[1] == "impl":
                names.add(f"{container[2]}::{name}")
        elif m_other:
            names.add(m_other.group(1))
        elif container is not None and container[1] == "enum":
            vm = VARIANT_RE.match(raw_line)
            if vm:
                names.add(f"{container[2]}::{vm.group(1)}")
        elif container is not None and container[1] == "struct":
            fm = FIELD_RE.match(raw_line)
            if fm:
                names.add(f"{container[2]}::{fm.group(1)}")

        # Now walk this line's braces left to right, updating depth/stack.
        # `impl`/`enum`/`struct` push the block they open using the body
        # depth their OWN opening brace produces -- found via the same char
        # scan, not assumed to be "the only brace on the line". A
        # brace-less tuple/unit struct (`struct Foo(i32);`) never matches a
        # `{` on this line, so nothing is pushed for it -- correct, it has
        # no field-body to scan.
        pending_push: tuple[str, str] | None = None
        if m_impl:
            pending_push = ("impl", m_impl.group(1).split("::")[-1])
        elif m_enum:
            pending_push = ("enum", m_enum.group(1))
        elif m_struct:
            pending_push = ("struct", m_struct.group(1))
        for ch in raw_line:
            if ch == "{":
                depth += 1
                if pending_push is not None:
                    stack.append((depth, pending_push[0], pending_push[1]))
                    pending_push = None
            elif ch == "}":
                if stack and stack[-1][0] == depth:
                    stack.pop()
                depth -= 1
    return names


def rust_file_defines(file_text: str, symbol: str) -> bool:
    return symbol in _rust_items_in_text(file_text)


def find_rust_files(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if root.is_dir():
            out.extend(sorted(root.rglob("*.rs")))
    return out


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
    c: Citation, rust_files: list[Path], md_files: list[Path] | None = None, base: Path = REPO_ROOT
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
        if c.kind == "rust":
            matches = resolve_path_suffix(c.target_path, rust_files, base=base)
            if not matches:
                return Finding(c, f"no file resolves for path `{c.target_path}`")
            if len(matches) > 1:
                shown = ", ".join(str(p.relative_to(base)) for p in matches[:5])
                return Finding(
                    c, f"ambiguous path `{c.target_path}` -- {len(matches)} files match: {shown}"
                )
            candidates = matches
        else:
            candidates = rust_files
        hits = [
            f for f in candidates
            if rust_file_defines(f.read_text(errors="ignore"), c.member)
        ]
        if not hits:
            return Finding(c, f"symbol `{c.member}` not found")
        if len(hits) > 1:
            shown = ", ".join(str(p.relative_to(base)) for p in hits[:5])
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
        rust_files: list[Path] = []
    else:
        rust_files = find_rust_files([REPO_ROOT / "crates"])
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
        f = resolve_citation(c, rust_files, md_files)
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
        crate = root / "crates" / "fake_crate" / "src"

        # --- renamed fn: RED when the cited name no longer exists ----------
        rust_before = "pub fn compute_thing() -> i32 { 1 }\n"
        rust_after = "pub fn compute_thing_renamed() -> i32 { 1 }\n"
        f = _write(root, "crates/fake_crate/src/lib.rs", rust_before)
        check(
            "renamed fn: original name resolves before rename",
            rust_file_defines(f.read_text(), "compute_thing"),
        )
        f.write_text(rust_after)
        check(
            "renamed fn: old citation does NOT resolve after rename (RED)",
            not rust_file_defines(f.read_text(), "compute_thing"),
        )
        check(
            "renamed fn: new name resolves",
            rust_file_defines(f.read_text(), "compute_thing_renamed"),
        )

        # --- deleted struct ---------------------------------------------------
        struct_src = "pub struct Widget { pub x: i32 }\n"
        g = _write(root, "crates/fake_crate/src/widget.rs", struct_src)
        check("deleted struct: resolves before deletion", rust_file_defines(g.read_text(), "Widget"))
        g.write_text("// Widget removed.\n")
        check(
            "deleted struct: citation does NOT resolve after deletion (RED)",
            not rust_file_defines(g.read_text(), "Widget"),
        )

        # --- ambiguous bare basename: two files define the SAME symbol -----
        h1 = _write(root, "crates/a/src/training.rs", "pub fn shared_name() {}\n")
        h2 = _write(root, "crates/b/src/training.rs", "pub fn shared_name() {}\n")
        all_files = [h1, h2]
        c = Citation("x.md", 1, "`training.rs::shared_name`", "rust", "training.rs", "shared_name")
        matches = resolve_path_suffix(c.target_path, all_files, base=root)
        check("ambiguous bare basename: two files match the suffix", len(matches) == 2)
        finding = resolve_citation(c, all_files, base=root)
        check("ambiguous bare basename: produces a finding", finding is not None)
        check(
            "ambiguous bare basename: finding names 'ambiguous'",
            finding is not None and "ambiguous" in finding.reason,
        )
        # A FULL (non-ambiguous) path to one of the two resolves cleanly.
        c_full = Citation(
            "x.md", 1, "`crates/a/src/training.rs::shared_name`", "rust",
            "crates/a/src/training.rs", "shared_name",
        )
        matches_full = resolve_path_suffix(c_full.target_path, all_files, base=root)
        check(
            "full path to one of the two ambiguous files narrows to one",
            matches_full == [h1],
        )
        check(
            "full path resolves cleanly through resolve_citation (no finding)",
            resolve_citation(c_full, all_files, base=root) is None,
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
        "ambiguous bare basename, a heading rename, and a proto field removal all "
        "reproduce the RED->GREEN shape; TOML key resolution and the placeholder "
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
