#!/usr/bin/env python3
"""Bind a documented enum enumeration in the maintainer guide to the code it mirrors.

A prose enumeration of a code enum silently rots the moment the enum grows: the
guide keeps listing the old variants while the code has moved on. This gate makes
that drift a red CI check instead of a human catch. It is the mechanized
anti-staleness tripwire for the guide↔enum bindings registered below.

A binding is either:

  - **set-equality-only**: the guide's marked variant list == the enum's variant
    set, order-insensitive (`recompute_file` is `None`), or
  - **set-equality + replay-parity**: set-equality above, *and* every variant the
    guide annotates with its exception phrase (e.g. "no replay arm") is one whose
    arm in `recompute_file`'s dispatch match returns the binding's no-replay
    marker, and every non-annotated variant is one whose arm does *not* — i.e.
    the guide's recorded replay-exceptions match the code's actual no-replay arms
    exactly.

The replay-parity leg is optional per binding, not a special case bolted onto
`check_binding`: a binding either carries the three replay-exception fields
(`recompute_file`, `exception_phrase`, `no_replay_marker`) together, or carries
none of them, and `check_binding` runs the matching leg(s) accordingly.

Registered bindings:
  - `ProducingDescriptor` (the materialization replay key) — set-equality +
    replay-parity.
  - `StoragePrecision` (the ANN index's storage/compute precision) —
    set-equality-only; this enum has no replay-parity dimension.
  - `KernelOracleStandardIds` (the `KO-1`..`KO-8` stable ids
    `docs/maintainer/cuda-kernel-guide.md` §3's conformance list reports) —
    set-equality-only, code side is a PYTHON list constant
    (`ci/scripts/check_kernel_oracles.py`'s `STABLE_IDS`), not a Rust enum —
    see `variant_source`/`parse_python_list_variants` below, added ONLY as a
    second leaf-level source parser feeding the SAME `Binding`/`check_binding`
    machinery every other binding uses (no second reconciliation mechanism).

Any parse failure — a missing marker, an unparseable enum, an absent match block
— is a non-zero exit with a message naming what could not be resolved. A silent
pass on an uncomputable diff would defeat the gate, so it fails closed.

Run: `python3 ci/scripts/check_doc_parity.py`
Hermetic: reads only files in the working tree; no network, no build.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GUIDE = REPO_ROOT / "docs" / "maintainer" / "MAINTAINER-GUIDE.md"
MANIFEST = REPO_ROOT / "crates" / "jammi-db" / "src" / "store" / "manifest.rs"
RECOMPUTE = REPO_ROOT / "crates" / "jammi-ai" / "src" / "pipeline" / "recompute.rs"
CONFIG = REPO_ROOT / "crates" / "jammi-db" / "src" / "config.rs"
CUDA_KERNEL_GUIDE = REPO_ROOT / "docs" / "maintainer" / "cuda-kernel-guide.md"
KERNEL_ORACLES_SCRIPT = REPO_ROOT / "ci" / "scripts" / "check_kernel_oracles.py"


class ParityError(Exception):
    """A binding could not be resolved or is out of parity — the gate fails closed."""


@dataclass(frozen=True)
class Binding:
    """One registered guide↔enum parity binding.

    `recompute_file`/`exception_phrase`/`no_replay_marker` are the
    replay-parity leg's inputs; they are `None` together for a
    set-equality-only binding (no enum-specific fields ever hang off one
    without the others — see `check_binding`).
    """

    name: str
    enum_name: str
    enum_file: Path
    guide_begin: str
    guide_end: str
    recompute_file: Path | None = None
    # The prose phrase that marks a guide variant as a replay exception, and the
    # error the exception's `recompute_file` arm must return.
    exception_phrase: str | None = None
    no_replay_marker: str | None = None
    # Which guide file carries this binding's BEGIN/END markers — defaults to
    # the maintainer guide every pre-existing binding uses; a binding may
    # point at a different guide (e.g. `cuda-kernel-guide.md`) without this
    # module growing a second guide-parsing mechanism.
    guide_file: Path = GUIDE
    # "rust_enum" (default, `parse_enum_variants`) or "python_list"
    # (`parse_python_list_variants`) — which leaf parser reads `enum_file` to
    # produce the code-side variant set. Both feed the SAME `check_binding`
    # set-equality logic below; this is the one place a binding's *source
    # format* (Rust enum vs. a Python list constant) is a variable, not a
    # second reconciliation mechanism.
    variant_source: str = "rust_enum"
    # Extra python-list constants (same `enum_file`, `variant_source ==
    # "python_list"` only) whose elements are UNIONED into the code-side
    # set alongside `enum_name` — lets a binding's code side be the union of
    # several named lists (e.g. `MECHANICAL_HERE_IDS` + `MECHANICAL_
    # ELSEWHERE_IDS`) without concatenating them into a derived constant a
    # future edit could let drift from its own sources.
    extra_enum_names: tuple[str, ...] = ()
    # When set, `parse_guide_variants` only counts a marked-block LINE whose
    # text contains this substring (case-insensitive) — the guide's per-id
    # "(mechanical; ...)"/"(auditor-only; ...)" labels partition the SAME
    # `KERNEL-ORACLE-STANDARD-IDS` block `KernelOracleStandardIds` binds in
    # full; this is that block read a SECOND time, filtered by label,
    # reusing the identical set-equality/BEGIN-END-marker mechanism rather
    # than a bespoke label-aware parser.
    label_filter: str | None = None

    @property
    def has_replay_parity(self) -> bool:
        fields = (self.recompute_file, self.exception_phrase, self.no_replay_marker)
        if all(f is not None for f in fields):
            return True
        if all(f is None for f in fields):
            return False
        raise ParityError(
            f"binding `{self.name}`: replay-parity fields must be set together "
            "(recompute_file, exception_phrase, no_replay_marker) or not at all"
        )


PRODUCING_DESCRIPTOR = Binding(
    name="ProducingDescriptor",
    enum_name="ProducingDescriptor",
    enum_file=MANIFEST,
    guide_begin="<!-- BEGIN PRODUCING-DESCRIPTOR-VARIANTS -->",
    guide_end="<!-- END PRODUCING-DESCRIPTOR-VARIANTS -->",
    recompute_file=RECOMPUTE,
    exception_phrase="no replay arm",
    no_replay_marker="NotRecomputable",
)

STORAGE_PRECISION = Binding(
    name="StoragePrecision",
    enum_name="StoragePrecision",
    enum_file=CONFIG,
    guide_begin="<!-- BEGIN STORAGE-PRECISION-VARIANTS -->",
    guide_end="<!-- END STORAGE-PRECISION-VARIANTS -->",
)

KERNEL_ORACLE_STANDARD_IDS = Binding(
    name="KernelOracleStandardIds",
    enum_name="STABLE_IDS",
    enum_file=KERNEL_ORACLES_SCRIPT,
    guide_begin="<!-- BEGIN KERNEL-ORACLE-STANDARD-IDS -->",
    guide_end="<!-- END KERNEL-ORACLE-STANDARD-IDS -->",
    guide_file=CUDA_KERNEL_GUIDE,
    variant_source="python_list",
)

KERNEL_ORACLE_MECHANICAL_IDS = Binding(
    name="KernelOracleMechanicalIds",
    enum_name="MECHANICAL_HERE_IDS",
    extra_enum_names=("MECHANICAL_ELSEWHERE_IDS",),
    enum_file=KERNEL_ORACLES_SCRIPT,
    guide_begin="<!-- BEGIN KERNEL-ORACLE-STANDARD-IDS -->",
    guide_end="<!-- END KERNEL-ORACLE-STANDARD-IDS -->",
    guide_file=CUDA_KERNEL_GUIDE,
    variant_source="python_list",
    label_filter="mechanical",
)

KERNEL_ORACLE_AUDITOR_ONLY_IDS = Binding(
    name="KernelOracleAuditorOnlyIds",
    enum_name="AUDITOR_ONLY_IDS",
    enum_file=KERNEL_ORACLES_SCRIPT,
    guide_begin="<!-- BEGIN KERNEL-ORACLE-STANDARD-IDS -->",
    guide_end="<!-- END KERNEL-ORACLE-STANDARD-IDS -->",
    guide_file=CUDA_KERNEL_GUIDE,
    variant_source="python_list",
    label_filter="auditor-only",
)

BINDINGS = [
    PRODUCING_DESCRIPTOR,
    STORAGE_PRECISION,
    KERNEL_ORACLE_STANDARD_IDS,
    KERNEL_ORACLE_MECHANICAL_IDS,
    KERNEL_ORACLE_AUDITOR_ONLY_IDS,
]


def _strip_rust_comments(text: str) -> str:
    """Drop `//`/`///` line comments so brace counting never trips over comment braces.

    The enum bodies this gate parses carry no `//` inside a string literal, so a
    plain line-comment strip is exact for them. Block comments are not used in the
    parsed enum; treating a stray `/*` as code would only *add* apparent tokens,
    never drop a real variant, so the fail-closed contract holds.
    """
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def parse_enum_variants(source: str, enum_name: str) -> list[str]:
    """Top-level variant idents of `enum <enum_name>` — struct, tuple, or unit.

    Locates the enum, isolates its brace-balanced body, then reads only the text
    at brace depth 1 (variant bodies live at depth ≥2 and are skipped). Variants
    are the depth-1 comma-separated segments; each variant's name is its leading
    identifier, after dropping `#[...]` attribute lines.
    """
    body = _strip_rust_comments(source)
    match = re.search(rf"\benum\s+{re.escape(enum_name)}\b", body)
    if match is None:
        raise ParityError(f"enum `{enum_name}` not found in source")

    open_idx = body.find("{", match.end())
    if open_idx == -1:
        raise ParityError(f"enum `{enum_name}` has no opening brace")

    depth = 0
    depth1_chars: list[str] = []
    end_idx = None
    for i in range(open_idx, len(body)):
        ch = body[i]
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = i
                break
            continue
        if depth == 1:
            depth1_chars.append(ch)
    if end_idx is None:
        raise ParityError(f"enum `{enum_name}` body is not brace-balanced")

    depth1 = "".join(depth1_chars)
    variants: list[str] = []
    for segment in depth1.split(","):
        # Drop attribute lines (`#[...]`) and blank lines; the variant name is the
        # first identifier of what remains.
        cleaned = "\n".join(
            line for line in segment.splitlines() if not line.strip().startswith("#")
        )
        ident = re.search(r"[A-Za-z_][A-Za-z0-9_]*", cleaned)
        if ident:
            variants.append(ident.group(0))
    if not variants:
        raise ParityError(f"enum `{enum_name}` parsed to zero variants")
    return variants


def parse_python_list_variants(source: str, list_name: str) -> list[str]:
    """String-literal elements of a top-level Python list/tuple constant
    `<list_name> = [...]` or `<list_name>: <ty> = (...)` — the `python_list`
    sibling of `parse_enum_variants` above, for a binding whose code side is
    a Python source file rather than a Rust `enum`. Brace/bracket-balanced
    (mirrors `parse_enum_variants`'s brace-balanced enum-body isolation), so
    a nested collection inside the tuple would not prematurely end the scan
    — not needed by `STABLE_IDS` today, but keeps this parser as robust as
    its Rust-enum sibling rather than a narrower one-off.
    """
    match = re.search(rf"\b{re.escape(list_name)}\b\s*(?::[^=]+)?=\s*([\[\(])", source)
    if match is None:
        raise ParityError(f"python list constant `{list_name}` not found in source")
    open_ch = match.group(1)
    close_ch = "]" if open_ch == "[" else ")"
    open_idx = match.end() - 1
    depth = 0
    end_idx = None
    for i in range(open_idx, len(source)):
        ch = source[i]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    if end_idx is None:
        raise ParityError(f"python list constant `{list_name}` is not bracket-balanced")
    body = source[open_idx + 1 : end_idx]
    variants = re.findall(r"""["']([^"']+)["']""", body)
    if not variants:
        raise ParityError(f"python list constant `{list_name}` parsed to zero string elements")
    return variants


def parse_guide_variants(
    guide: str, binding: Binding
) -> tuple[list[str], set[str]]:
    """Parse the guide's marked variant list: (all names, exception-annotated names).

    Each line in the marked block names a variant in the first `backticked` token;
    a line carrying the binding's exception phrase (e.g. "no replay arm") marks
    that variant a documented replay exception. The identifier class includes
    `-` (hyphen) so a hyphenated id like `` `KO-7` `` resolves as ONE token —
    Rust variant names (the only prior users of this parser) never contain a
    hyphen, so this widening is behavior-preserving for every pre-existing
    binding.
    """
    begin = guide.find(binding.guide_begin)
    end = guide.find(binding.guide_end)
    if begin == -1:
        raise ParityError(
            f"guide is missing the `{binding.guide_begin}` parity marker for "
            f"`{binding.enum_name}`"
        )
    if end == -1:
        raise ParityError(
            f"guide is missing the `{binding.guide_end}` parity marker for "
            f"`{binding.enum_name}`"
        )
    if end < begin:
        raise ParityError(
            f"guide parity markers for `{binding.enum_name}` are out of order"
        )

    block = guide[begin + len(binding.guide_begin) : end]
    names: list[str] = []
    exceptions: set[str] = set()
    for line in block.splitlines():
        if not line.strip():
            continue
        if binding.label_filter is not None and binding.label_filter.lower() not in line.lower():
            continue
        ident = re.search(r"`([A-Za-z_][A-Za-z0-9_-]*)`", line)
        if ident is None:
            continue
        name = ident.group(1)
        names.append(name)
        if binding.exception_phrase is not None and binding.exception_phrase.lower() in line.lower():
            exceptions.add(name)
    if not names:
        raise ParityError(
            f"guide parity block for `{binding.enum_name}` lists no variants"
        )
    return names, exceptions


def parse_no_replay_variants(recompute: str, binding: Binding) -> set[str]:
    """Variants whose `replay_descriptor` match arm returns the no-replay error.

    Isolates the `match descriptor { ... }` block, splits it by
    `<enum>::<Variant>` arm heads, and reports each variant whose arm body
    contains the no-replay marker (`NotRecomputable`).

    Comments are stripped first (mirroring `parse_enum_variants`) so a comment
    that merely mentions `NotRecomputable` inside a *replayable* arm cannot
    trigger a false BLOCK. Structural assumption that remains: an arm is judged a
    no-replay arm only if the marker token appears inline in the arm body — an
    arm that refused replay via a helper returning the error out-of-line would not
    be detected. Acceptable today: every arm returns the error inline.
    """
    recompute = _strip_rust_comments(recompute)
    match = re.search(r"match\s+descriptor\s*\{", recompute)
    if match is None:
        raise ParityError(
            f"`match descriptor {{` block not found in {binding.recompute_file.name}"
        )

    open_idx = recompute.find("{", match.start())
    depth = 0
    end_idx = None
    for i in range(open_idx, len(recompute)):
        ch = recompute[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    if end_idx is None:
        raise ParityError(
            f"`match descriptor` block in {binding.recompute_file.name} is not "
            "brace-balanced"
        )

    block = recompute[open_idx + 1 : end_idx]
    arm_head = re.compile(rf"{re.escape(binding.enum_name)}::([A-Za-z_][A-Za-z0-9_]*)")
    heads = list(arm_head.finditer(block))
    if not heads:
        raise ParityError(
            f"no `{binding.enum_name}::<Variant>` arms found in the match block"
        )

    no_replay: set[str] = set()
    for idx, head in enumerate(heads):
        arm_start = head.end()
        arm_end = heads[idx + 1].start() if idx + 1 < len(heads) else len(block)
        arm_body = block[arm_start:arm_end]
        if binding.no_replay_marker in arm_body:
            no_replay.add(head.group(1))
    return no_replay


def check_binding(binding: Binding) -> list[str]:
    """Check one binding; return a list of human-readable parity failures (empty = OK).

    Every binding gets the set-equality leg. The replay-parity leg only runs
    when `binding.has_replay_parity` — a `StoragePrecision`-shaped binding
    (no replay dimension) skips it entirely rather than being special-cased
    around a `recompute_file` it doesn't have.
    """
    enum_src = binding.enum_file.read_text()
    guide_src = binding.guide_file.read_text()

    if binding.variant_source == "python_list":
        code_variants = set(parse_python_list_variants(enum_src, binding.enum_name))
        for extra_name in binding.extra_enum_names:
            code_variants |= set(parse_python_list_variants(enum_src, extra_name))
    elif binding.variant_source == "rust_enum":
        code_variants = set(parse_enum_variants(enum_src, binding.enum_name))
    else:
        raise ParityError(
            f"binding `{binding.name}`: unknown variant_source `{binding.variant_source}` "
            "(must be 'rust_enum' or 'python_list')"
        )
    guide_names, guide_exceptions = parse_guide_variants(guide_src, binding)
    guide_variants = set(guide_names)

    failures: list[str] = []

    # (1) Set equality between the guide's marked list and the code enum.
    missing_in_guide = sorted(code_variants - guide_variants)
    extra_in_guide = sorted(guide_variants - code_variants)
    if missing_in_guide:
        failures.append(
            f"guide is MISSING variant(s) present in `{binding.enum_name}`: "
            f"{', '.join(missing_in_guide)} — add them to the parity block."
        )
    if extra_in_guide:
        failures.append(
            f"guide lists variant(s) NOT in `{binding.enum_name}`: "
            f"{', '.join(extra_in_guide)} — remove them or re-register the binding."
        )

    if len(guide_names) != len(guide_variants):
        dupes = sorted({n for n in guide_names if guide_names.count(n) > 1})
        failures.append(
            f"guide parity block lists duplicate variant(s): {', '.join(dupes)}."
        )

    # (2) The guide's replay-exceptions must match the code's no-replay arms —
    # only for a binding that carries a replay dimension at all. A
    # set-equality-only binding (e.g. `StoragePrecision`) has no
    # `recompute_file` to read an arm from, so it skips this leg entirely
    # rather than being special-cased in.
    if binding.has_replay_parity:
        recompute_src = binding.recompute_file.read_text()
        code_no_replay = parse_no_replay_variants(recompute_src, binding)

        # Only over variants both sides agree exist (set-equality failures
        # above name the rest).
        shared = code_variants & guide_variants
        doc_says_exception_but_code_replays = sorted(
            v for v in shared if v in guide_exceptions and v not in code_no_replay
        )
        code_no_replay_but_doc_unmarked = sorted(
            v for v in shared if v in code_no_replay and v not in guide_exceptions
        )
        for v in doc_says_exception_but_code_replays:
            failures.append(
                f"guide annotates `{v}` as '{binding.exception_phrase}', but its "
                f"`replay_descriptor` arm does NOT return `{binding.no_replay_marker}` "
                "(it has a real replay arm) — fix the annotation or the code."
            )
        for v in code_no_replay_but_doc_unmarked:
            failures.append(
                f"`{v}`'s `replay_descriptor` arm returns `{binding.no_replay_marker}` "
                f"(no replay arm), but the guide does not annotate it "
                f"'{binding.exception_phrase}' — annotate it in the parity block."
            )
    # A set-equality-only binding has no `exception_phrase` registered, so
    # `parse_guide_variants` above can never populate `guide_exceptions` for
    # it — there is no dead "unmarked exception" state to check here.

    return failures


def main() -> int:
    any_failures = False
    for binding in BINDINGS:
        try:
            failures = check_binding(binding)
        except ParityError as exc:
            print(
                f"doc-parity[{binding.name}]: FAIL (uncomputable) — {exc}",
                file=sys.stderr,
            )
            any_failures = True
            continue
        except OSError as exc:
            print(
                f"doc-parity[{binding.name}]: FAIL (uncomputable) — {exc}",
                file=sys.stderr,
            )
            any_failures = True
            continue

        if failures:
            any_failures = True
            print(f"doc-parity[{binding.name}]: FAIL", file=sys.stderr)
            for failure in failures:
                print(f"  - {failure}", file=sys.stderr)
        else:
            print(f"doc-parity[{binding.name}]: OK")

    if any_failures:
        print(
            "\ndoc-parity: the maintainer guide is out of parity with the code it "
            "documents. See the failures above.",
            file=sys.stderr,
        )
        return 1
    print("doc-parity: all bindings in parity.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
