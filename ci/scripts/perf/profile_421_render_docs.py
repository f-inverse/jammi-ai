#!/usr/bin/env python3
"""Render #421 close-out doc blocks from the tower-profile artifact, never by hand.

THE CLASS THIS CLOSES. `docs/plans/66-tower-profile/README.md`'s measured-towers
table, its four candidate-port reasons, and its four verbatim findings were
hand-typed prose copied out of
`crates/jammi-kernels/artifacts/cuda-runs/2026-09-07-profile-421-towers-c1b0b0ba-a100-sxm4.json`
with NOTHING in this tree mechanically checking that the committed prose still
agrees with the artifact it claims to summarize (the audit's own finding on
this unit's close-out: `docs/maintainer/**` numeric tables get this via
`check_perf_claims.py`'s per-cell `<!-- claims: ... -->` binding, but
`docs/plans/66-tower-profile/**` is outside that gate's scope, and none of
these three blocks are pipe-table-shaped claims cells anyway -- the four
candidate reasons and four findings are whole verbatim STRINGS, not per-cell
numeric tokens). This script closes that the same way `gen_dep_dag.py` closes
the crate-dependency-graph class: render the block from its one source of
truth and splice it strictly between a pair of markers, so a later hand-edit
of the artifact (or the doc) is caught by `--check` diffing the two, not
trusted on prose alone.

BLOCKS. Three, each spliced between its own
`<!-- profile-421-generated: <block-id> -->` / `<!-- /profile-421-generated -->`
marker pair in `docs/plans/66-tower-profile/README.md`:

  - `measured-towers-table`: one pipe-table row per artifact `legs[]` entry,
    in artifact order (which is already tower-grouped, A1/A2/D1/D2 within
    each tower) -- `wall`/`front`/`busy`/`residual` at 4 decimal places
    (seconds), `front %`/`busy %` of wall at 1 decimal place, matching the
    doc's own printed precision exactly. The leg label is `A1`/`A2` for an
    `arm == "A"` leg, or `D1 (<chain>+<chain> eager)`/`D2 (<chain> eager)`
    for a `D` leg, built from `kernels_disabled` in a FIXED canonical order
    (LoRA, LN, GELU -- `_KERNEL_LABEL_ORDER`) rather than the JSON array's
    own order, which is not itself canonical.
  - `candidate-reasons`: one bullet per `candidate_decisions[]` entry,
    `` - **`<port>`** — <verdict>: "<reason>" `` -- the reason string
    verbatim, never elided (elision, where it appears in
    `docs/maintainer/fine-tune-performance-guide.md`'s OWN prose citing the
    same artifact, is a distinct, hand-authored economy this script does not
    attempt to reproduce).
  - `findings`: one bullet per `findings[]` entry, `` - **`<id>`**: "<text>" ``,
    verbatim.

Run: `python3 ci/scripts/perf/profile_421_render_docs.py` writes the current
render into the doc. `python3 ci/scripts/perf/profile_421_render_docs.py
--check` fails (non-zero, diff printed) the moment the committed block
differs from a fresh render -- the CI-wired form (`.github/workflows/ci.yml`,
next to the sibling perf checks).
Hermetic: reads only files in the working tree; no network, no build, no GPU.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[3]

ARTIFACT = (
    REPO_ROOT
    / "crates"
    / "jammi-kernels"
    / "artifacts"
    / "cuda-runs"
    / "2026-09-07-profile-421-towers-c1b0b0ba-a100-sxm4.json"
)
README = REPO_ROOT / "docs" / "plans" / "66-tower-profile" / "README.md"

_BEGIN_TEMPLATE = "<!-- profile-421-generated: {block_id} -->"
_END_MARKER = "<!-- /profile-421-generated -->"

_TOWER_DISPLAY = {
    "clip-text": "CLIP-text",
    "clip-vision": "CLIP-vision",
    "htsat": "HTSAT",
}

# The canonical label ORDER a D-leg's `kernels_disabled` set renders in --
# never the JSON array's own (unordered-by-convention) order. This matches
# every D-leg label already committed in the doc (`D1 (LoRA+LN eager)`,
# `D1 (LoRA+LN+GELU eager)`, `D2 (LoRA eager)`).
_KERNEL_LABEL_ORDER = (
    ("lora_linear_fused", "LoRA"),
    ("layer_norm_fused", "LN"),
    ("gelu_erf_fused", "GELU"),
)


def _leg_label(leg: dict) -> str:
    """The doc's own leg label: the bare `A1`/`A2` suffix for an `arm ==
    "A"` leg (README never annotates the A2 dtype inline -- that is the
    `dtype` column's own job), or `D1 (LoRA+LN eager)`-shaped for a `D` leg,
    built from `kernels_disabled` in `_KERNEL_LABEL_ORDER`.
    """
    suffix = leg["leg_id"].rsplit("-", 1)[-1]
    if leg["arm"] != "D":
        return suffix
    disabled = set(leg["kernels_disabled"])
    labels = [label for key, label in _KERNEL_LABEL_ORDER if key in disabled]
    return f"{suffix} ({'+'.join(labels)} eager)"


def render_measured_towers_table(artifact: dict) -> str:
    rows = []
    for leg in artifact["legs"]:
        ps = leg["per_step"]
        rows.append(
            "| {tower} | {leg} | {dtype} | {wall:.4f} | {front:.4f} | {busy:.4f} | "
            "{residual:.4f} | {front_pct:.1f} | {busy_pct:.1f} |".format(
                tower=_TOWER_DISPLAY[leg["tower"]],
                leg=_leg_label(leg),
                dtype=leg["dtype"],
                wall=ps["wall_s_per_step"],
                front=ps["front_s_per_step"],
                busy=ps["busy_s_per_step"],
                residual=ps["residual_s_per_step"],
                front_pct=ps["front_share_of_wall"] * 100,
                busy_pct=ps["busy_share_of_wall"] * 100,
            )
        )
    return "\n".join(rows)


# The doc's own prose-wrap width for a bulleted `**\`id\`** — ...` line
# (measured against the committed text, never guessed): a greedy
# `textwrap.fill` at this width, `- `-prefixed with a 2-space hanging
# indent, reproduces both bulleted blocks in this doc byte-for-byte. Never
# breaks on a hyphen (`s_wall+U_wall`/`decision-grade` must never split
# mid-token) or a long "word" (a run with no internal space, e.g. the
# `s_wall+U_wall=...` fragments, must never be force-split either).
_BULLET_WRAP_WIDTH = 91


def _bullet(body: str) -> str:
    return textwrap.fill(
        body,
        width=_BULLET_WRAP_WIDTH,
        initial_indent="- ",
        subsequent_indent="  ",
        break_long_words=False,
        break_on_hyphens=False,
    )


def render_candidate_reasons(artifact: dict) -> str:
    return "\n".join(
        _bullet(f'**`{cd["port"]}`** — {cd["verdict"]}: "{cd["reason"]}"')
        for cd in artifact["candidate_decisions"]
    )


def render_findings(artifact: dict) -> str:
    return "\n".join(
        _bullet(f'**`{finding["id"]}`**: "{finding["text"]}"') for finding in artifact["findings"]
    )


# `block_id -> (doc_path, render_fn)` -- every block this script owns. All
# three currently live in the same doc; the doc is still per-block (not a
# single shared constant) so a future block in a SECOND doc is a one-line
# addition, never a signature change.
BLOCKS: dict[str, tuple[Path, Callable[[dict], str]]] = {
    "measured-towers-table": (README, render_measured_towers_table),
    "candidate-reasons": (README, render_candidate_reasons),
    "findings": (README, render_findings),
}


class MarkerError(Exception):
    """A doc is missing one (or both) of a block's markers -- fails closed,
    never silently skips the block."""


def _split_on_markers(text: str, block_id: str, doc_path: Path) -> tuple[str, str, str]:
    begin = _BEGIN_TEMPLATE.format(block_id=block_id)
    if begin not in text:
        raise MarkerError(f"{doc_path}: missing begin marker for block {block_id!r}")
    before, rest = text.split(begin, 1)
    if _END_MARKER not in rest:
        raise MarkerError(f"{doc_path}: missing end marker for block {block_id!r}")
    current, after = rest.split(_END_MARKER, 1)
    return before + begin, current, _END_MARKER + after


def _current_block(text: str, block_id: str, doc_path: Path) -> str:
    _before, current, _after = _split_on_markers(text, block_id, doc_path)
    return current.strip("\n")


def _spliced(text: str, block_id: str, doc_path: Path, rendered: str) -> str:
    before, _current, after = _split_on_markers(text, block_id, doc_path)
    return f"{before}\n{rendered}\n{after}"


def main(argv: list[str]) -> int:
    check = "--check" in argv
    artifact = json.loads(ARTIFACT.read_text())

    mismatches: list[str] = []
    doc_texts: dict[Path, str] = {}
    for block_id, (doc_path, render_fn) in BLOCKS.items():
        if doc_path not in doc_texts:
            doc_texts[doc_path] = doc_path.read_text()
        rendered = render_fn(artifact)
        try:
            if check:
                current = _current_block(doc_texts[doc_path], block_id, doc_path)
                if current != rendered:
                    mismatches.append(block_id)
                    print(f"profile-421-render-docs: MISMATCH in block {block_id!r} ({doc_path})")
                    print("--- committed ---")
                    print(current)
                    print("--- rendered ---")
                    print(rendered)
            else:
                doc_texts[doc_path] = _spliced(doc_texts[doc_path], block_id, doc_path, rendered)
        except MarkerError as exc:
            print(f"profile-421-render-docs: FAIL (uncomputable) — {exc}", file=sys.stderr)
            return 1

    if check:
        if mismatches:
            print(f"profile-421-render-docs: FAIL — {len(mismatches)} block(s) drifted from the artifact.", file=sys.stderr)
            return 1
        print(f"profile-421-render-docs: {len(BLOCKS)} block(s) match the rendered artifact.")
        return 0

    for doc_path, text in doc_texts.items():
        doc_path.write_text(text)
    written = ", ".join(str(p) for p in doc_texts)
    print(f"profile-421-render-docs: {len(BLOCKS)} block(s) rendered into {written}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
