#!/usr/bin/env python3
"""Render #421 close-out doc blocks from the tower-profile artifact, never by hand.

THE CLASS THIS CLOSES. A pre-registered profile's close-out prose (README,
the maintainer guide's own condensed paragraph, the CHANGELOG entry) quotes
numbers and verbatim strings out of
`crates/jammi-kernels/artifacts/cuda-runs/2026-09-07-profile-421-towers-c1b0b0ba-a100-sxm4.json`
(and, for a handful of NON-artifact facts -- corpus pool sizes, a hermetic
test count, a pinned-tree basename-ambiguity count -- out of this tree's own
live source of truth for each). Nothing mechanically checks that committed
prose still agrees with the source it claims to summarize the moment it is
hand-typed instead of rendered: `docs/maintainer/**` numeric TABLES get this
via `check_perf_claims.py`'s per-cell `<!-- claims: ... -->` binding, but a
pipe-table cell is not the only shape a number takes in these three docs --
a realized-gain bullet, a DECLINE-band paragraph, a corpus-pool aside, and a
CHANGELOG entry are all prose, not table cells. This script closes that the
same way `gen_dep_dag.py` closes the crate-dependency-graph class: render
each block from its own one source of truth and splice it strictly between
a pair of markers, so a later hand-edit of the source (or the doc) is caught
by `--check` diffing the two, not trusted on prose alone.

BLOCKS. Each spliced between its own
`<!-- profile-421-generated: <block-id> -->` / `<!-- /profile-421-generated -->`
marker pair, in `docs/plans/66-tower-profile/README.md` unless noted:

  - `measured-summary`: the "## What was measured" opening paragraph's own
    leg count, tower count, device/driver/nsys identity and git sha --
    every one read live off `legs`/`p2_witnessed`/`status`/`notes`/`git_sha`,
    never a hand-typed literal that could drift the moment a re-run changes
    a count or an identity string.
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
  - `realized-gains`: the three C-LORA / C-LN / C-LN+C-GELU-HTSAT bullets,
    from the artifact's `realized_gains[]` array (grouped by `chain`; the
    HTSAT `C-LN` entry carries a `note` field marking it the JOINT
    C-LN+C-GELU delta, the signal this render keys the third bullet on).
  - `decision-grade-note`: the "decided on BOTH the F32 and BF16 decision
    legs" sentence, keyed off `attribution[].decision_grade` for the two
    CLIP-tower A2 legs and `htsat-A2`.
  - `decline-band-summary`: the "No port is licensed" paragraph -- every
    percentage in it is `attribution[<leg>].chains["C-GELU"].share_wall`/
    `share_gpu_busy` plus `UNATTRIBUTED`'s own two shares, summed here
    rather than hand-added.
  - `census-key-root-cause`: the Deviations bullet naming the cutlass
    template-wrapper split count and the known-kernel-name gate percentage
    -- read off the SAME rendered `notes.recorded_deviations` sentence
    `profile_421_artifact.py`'s own `_identity_template_context` already
    filled those two numbers into, never a second hand-typed copy.
  - `htsat-a2-deviation`: the Deviations bullet naming `htsat-A2`'s own
    UNATTRIBUTED share of GPU busy (`attribution["htsat-A2"].chains
    ["UNATTRIBUTED"].share_gpu_busy`), its "VALID"/"not decision-grade"
    words (`legs[htsat-A2].merge_verdict` / `attribution[htsat-A2].
    decision_grade`), its comparison word and "N %" validity bound (both
    PARSED off `attribution[htsat-A2].decision_grade_reason`'s own recorded
    `"UNATTRIBUTED share_gpu_busy=<share> > <bound>"` string, cross-checked
    against an actual `>` comparison of the two floats -- never a
    hard-coded "over" or a live import of the module constant that produced
    it), and its `htsat-A1` clause off `attribution["htsat-A1"]`
    (`decision_grade` plus its own UNATTRIBUTED share, verified to actually
    clear the same bound) -- refusing outright if the artifact's own
    `suppressed_findings` names `htsat-A1` as not decision-grade while
    `attribution["htsat-A1"].decision_grade` says otherwise.
  - `corpus-pool-note`: the Deviations bullet naming the media corpus pool
    size and the M-leg's train-clip cycle count -- sourced live from
    `profile_421_legs.sh`'s own `MEDIA_FAMILIES`/`MEDIA_HELDOUT_FAMILIES`/
    `STEPS_M`/`BATCH` constants and the audio corpus generator's own
    `_DEFAULT_INSTANCES_PER_FAMILY`, never a hand-copied product of the two.
  - `hermetic-test-count-note`: the Deviations bullet naming `test_
    profile_421_merge.py`'s own hermetic test count -- sourced live via
    `unittest.TestLoader` test discovery (no execution), never hand-counted.
  - `basename-ambiguity-note`: the Deviations bullet naming how many
    `layer_norm.rs`/`main.rs` files exist in the tree at `CONTRACT.md`'s own
    pinned citation epoch -- sourced live via `check_citations.py`'s own
    `_ls_tree_paths`, the exact machinery `check_citations.py` itself uses
    to decide the ambiguity `CONTRACT.md`'s basename map resolves.
  - `realized-gains-guide` / `decline-band-guide` / `findings-guide`: the
    maintainer guide's own condensed restatement of the same three artifact
    facts, in the guide's own (denser) wording -- a SEPARATE render function
    per block, never a re-use of the README wording, because the two docs'
    prose economy genuinely differs (elision, units, emphasis). Every number
    `findings-guide` needs is read off a finding's own structured `evidence`
    (never a digit regexed out of its `text`), and every QUALITATIVE word
    ("front-end-bound", "launch-bound", "dtype- and arm-invariant", "only")
    is gated on that finding's own `evidence.<rule>` flag (`_require_
    evidence_flag`) -- withheld or `False` fails this renderer CLOSED.
  - `changelog-421-entry`: the CHANGELOG's own close-out entry -- the
    per-tower headline numbers restated one more time, in the CHANGELOG's
    own terse style, gated on the SAME `evidence` flags `findings-guide`
    reads.

Run: `python3 ci/scripts/perf/profile_421_render_docs.py` writes the current
render into every doc. `python3 ci/scripts/perf/profile_421_render_docs.py
--check` fails (non-zero, diff printed) the moment a committed block
differs from a fresh render -- the CI-wired form (`.github/workflows/ci.yml`,
next to the sibling perf checks).
Hermetic: reads only files in the working tree; no network, no build, no GPU
(`unittest.TestLoader` DISCOVERS `test_profile_421_merge.py`'s tests -- it
imports the module and counts `Test*` methods -- it never RUNS one).
"""

from __future__ import annotations

import json
import re
import sys
import textwrap
import unittest
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_citations as _cc  # noqa: E402
# `profile_421_artifact`'s own `_fmt_range` (the EXACT rounding/range-collapse
# rule every finding's own text was rendered with) -- imported, never
# retyped, so a renderer that needs a NUMBER this module already computed
# reads it live off it (or off the artifact's own structured `evidence`),
# never a second, independently-drifting hand-typed copy or a literal. Every
# validity-gate NUMBER `profile_421_attribute.py` computes (e.g. the
# UNATTRIBUTED bound) is read back off the ARTIFACT's own recorded evidence
# (`decision_grade_reason`), never a live import of that module's constants
# -- a module constant could since have moved independently of the run that
# actually produced this artifact; the artifact's own recorded number is the
# one this run was actually judged against.
import profile_421_artifact as _pa  # noqa: E402

ARTIFACT = (
    REPO_ROOT
    / "crates"
    / "jammi-kernels"
    / "artifacts"
    / "cuda-runs"
    / "2026-09-07-profile-421-towers-c1b0b0ba-a100-sxm4.json"
)
README = REPO_ROOT / "docs" / "plans" / "66-tower-profile" / "README.md"
CONTRACT = REPO_ROOT / "docs" / "plans" / "66-tower-profile" / "CONTRACT.md"
GUIDE = REPO_ROOT / "docs" / "maintainer" / "fine-tune-performance-guide.md"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
PROFILE_421_LEGS_SH = REPO_ROOT / "ci" / "scripts" / "perf" / "profile_421_legs.sh"
AUDIO_CORPUS_PY = REPO_ROOT / "ci" / "scripts" / "perf" / "gen_fixed_length_audio_corpus.py"
PROFILE_421_MERGE_TEST = REPO_ROOT / "ci" / "scripts" / "perf" / "test_profile_421_merge.py"

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


_KNOWN_KERNEL_LABEL_KEYS = frozenset(key for key, _label in _KERNEL_LABEL_ORDER)


def _leg_label(leg: dict) -> str:
    """The doc's own leg label: the bare `A1`/`A2` suffix for an `arm ==
    "A"` leg (README never annotates the A2 dtype inline -- that is the
    `dtype` column's own job), or `D1 (LoRA+LN eager)`-shaped for a `D` leg,
    built from `kernels_disabled` in `_KERNEL_LABEL_ORDER`.

    Refuses BY NAME (like `_TOWER_DISPLAY`'s own plain dict lookup already
    does for an unrecognized `tower`) the moment `kernels_disabled` names a
    key `_KERNEL_LABEL_ORDER` does not know: the old
    `[label for key, label in _KERNEL_LABEL_ORDER if key in disabled]`
    comprehension only ever walked the KNOWN side, so a leg disabling some
    future kernel this table has not been taught about would silently drop
    it from the rendered label instead of surfacing the gap -- the exact
    silent-drift shape this whole doc-generation script exists to close.
    """
    suffix = leg["leg_id"].rsplit("-", 1)[-1]
    if leg["arm"] != "D":
        return suffix
    disabled = set(leg["kernels_disabled"])
    unknown = disabled - _KNOWN_KERNEL_LABEL_KEYS
    if unknown:
        raise ValueError(
            f"leg {leg['leg_id']!r} disables kernel key(s) {sorted(unknown)!r} that "
            "_KERNEL_LABEL_ORDER does not know -- add them there (refused by name, never "
            "silently dropped from the rendered D-leg label)"
        )
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


# -- The three towers, in the FIXED order every prose block below names them
# in (C-LORA's own bullet, the corpus-pool note, etc.) -- never the JSON
# array's own order, same reason `_KERNEL_LABEL_ORDER` is a fixed tuple.
_TOWER_ORDER = ("clip-text", "clip-vision", "htsat")


def _realized_gains_by_key(artifact: dict) -> dict[tuple[str, str], dict]:
    return {(g["chain"], g["tower"]): g for g in artifact["realized_gains"]}


def _ms(gain: dict) -> float:
    return gain["wall_delta_s_per_step"] * 1000


def _pct_of_baseline(gain: dict) -> float:
    return gain["share_of_baseline_wall"] * 100


def render_realized_gains(artifact: dict) -> str:
    """The three C-LORA / C-LN / C-LN+C-GELU-HTSAT bullets (README's
    "Realized gains" section) -- every millisecond and percentage figure
    read straight off `realized_gains[]`, keyed by `(chain, tower)`. The
    HTSAT `C-LN` entry is the one the contract's D1 disables `layer_norm_
    fused` AND `gelu_erf_fused` together on (module doc's own note field on
    that entry) -- rendered as its own third bullet, never folded into the
    two-tower C-LN bullet above it.
    """
    by_key = _realized_gains_by_key(artifact)
    lora = {t: by_key[("C-LORA", t)] for t in _TOWER_ORDER}
    ln_clip = {t: by_key[("C-LN", t)] for t in ("clip-text", "clip-vision")}
    joint = by_key[("C-LN", "htsat")]

    lora_bullet = (
        "**C-LORA** (`lora_linear_fused`, D2 minus its tower's A1): CLIP-text "
        f"+{_ms(lora['clip-text']):.1f} ms/step ({_pct_of_baseline(lora['clip-text']):.1f} % of "
        f"the A1 baseline wall), CLIP-vision +{_ms(lora['clip-vision']):.1f} ms/step "
        f"({_pct_of_baseline(lora['clip-vision']):.1f} %), HTSAT +{_ms(lora['htsat']):.1f} "
        f"ms/step ({_pct_of_baseline(lora['htsat']):.1f} %)."
    )
    ln_bullet = (
        "**C-LN** (`layer_norm_fused`, D1 minus D2, isolating the LayerNorm kernel on top of "
        f"the already-fused LoRA site): CLIP-text +{_ms(ln_clip['clip-text']):.1f} ms/step "
        f"({_pct_of_baseline(ln_clip['clip-text']):.1f} % of A1 baseline wall), CLIP-vision "
        f"+{_ms(ln_clip['clip-vision']):.1f} ms/step ({_pct_of_baseline(ln_clip['clip-vision']):.1f} %)."
    )
    joint_bullet = (
        "**C-LN + C-GELU-HTSAT joint** (HTSAT's D1 disables `layer_norm_fused` AND "
        "`gelu_erf_fused` together, so its D1-minus-D2 delta is the two chains combined, not "
        f"C-LN alone): +{_ms(joint):.1f} ms/step ({_pct_of_baseline(joint):.1f} % of the A1 "
        "baseline wall)."
    )
    return "\n".join(_bullet(b) for b in (lora_bullet, ln_bullet, joint_bullet))


# `(port, chain)` -> the attribution chain key that carries each candidate
# port's STANDALONE `share_wall`/`share_gpu_busy` (no `U` term) -- the
# artifact's own `candidate_decisions[].chain` field IS this mapping, kept
# here as a tuple only for the two towers' own fixed A1/A2 leg-id shape.
_CLIP_DECISION_TOWERS = ("clip-text", "clip-vision")


def _attribution_by_leg(artifact: dict) -> dict[str, dict]:
    return {a["leg_id"]: a for a in artifact["attribution"]}


def render_decision_grade_note(artifact: dict) -> str:
    """The "decided on BOTH the F32 (A1) and BF16 (A2) decision legs"
    sentence -- keyed off `attribution[<leg>].decision_grade` for the two
    CLIP-tower A2 legs (expected True, the pass-4 census-key fix's own
    effect) and `htsat-A2` (expected False, HTSAT has no candidate port so
    this never blocks a verdict either way)."""
    by_leg = _attribution_by_leg(artifact)
    clip_a2_grade = {t: by_leg[f"{t}-A2"]["decision_grade"] for t in _CLIP_DECISION_TOWERS}
    htsat_a2_grade = by_leg["htsat-A2"]["decision_grade"]
    if not all(clip_a2_grade.values()):
        raise ValueError(
            f"expected both CLIP-tower A2 legs decision-grade, got {clip_a2_grade!r} -- "
            "the README sentence this renders assumes this; rewrite it if the artifact's "
            "own attribution verdict changed"
        )
    if htsat_a2_grade:
        raise ValueError(
            "expected htsat-A2 non-decision-grade (its own Deviations bullet explains why) -- "
            "rewrite the README sentence this renders if the artifact's own verdict changed"
        )
    body = (
        "All four candidate ports the contract named are **UNRESOLVED** — decided on BOTH the "
        "F32 (A1) and BF16 (A2) decision legs of each tower (the pass-4 census-key fix, below, "
        "makes both CLIP-tower A2 legs decision-grade for attribution — `htsat-A2` stays VALID "
        "but non-decision-grade, see the deviation below; HTSAT has no candidate port under "
        "this contract, so that never blocks a candidate-port decision — no candidate is "
        "F32-only by consequence):"
    )
    return textwrap.fill(body, width=88, break_long_words=False, break_on_hyphens=False)


# The one non-`U` chain each CLIP `C-MLP-*` candidate port's decision reads
# (module doc's "decline-band-summary" block: `candidate_decisions[].chain`
# for both `C-MLP-clip-text`/`C-MLP-clip-vision` is `"C-GELU"`).
_C_MLP_CHAIN = "C-GELU"


def _chain_wall_pct(by_leg: dict, tower: str, arm: str, chain: str) -> float:
    return by_leg[f"{tower}-{arm}"]["chains"][chain]["share_wall"] * 100


def _chain_combined_pct(by_leg: dict, tower: str, arm: str, chain: str, kind: str) -> float:
    chains = by_leg[f"{tower}-{arm}"]["chains"]
    key = "share_wall" if kind == "wall" else "share_gpu_busy"
    return (chains[chain][key] + chains["UNATTRIBUTED"][key]) * 100


def _gelu_wall_pct(by_leg: dict, tower: str, arm: str) -> float:
    return _chain_wall_pct(by_leg, tower, arm, _C_MLP_CHAIN)


def _combined_pct(by_leg: dict, tower: str, arm: str, kind: str) -> float:
    return _chain_combined_pct(by_leg, tower, arm, _C_MLP_CHAIN, kind)


def render_decline_band_summary(artifact: dict) -> str:
    """The "No port is licensed under #421" paragraph -- every percentage
    is `attribution[<leg>].chains["C-GELU"].share_wall`/`share_gpu_busy`
    (the standalone, no-`U`-term measured share `C-MLP-<tower>`'s own
    candidate decision reads) plus `UNATTRIBUTED`'s own two shares at the
    same leg, summed here (never hand-added) to reproduce the combined
    `s_wall+U_wall`/`s_busy+U_busy` figures `candidate_decisions[].reason`
    already states verbatim in the block above this one.
    """
    by_leg = _attribution_by_leg(artifact)
    gelu_wall_pct = lambda tower, arm: _gelu_wall_pct(by_leg, tower, arm)  # noqa: E731
    combined_pct = lambda tower, arm, kind: _combined_pct(by_leg, tower, arm, kind)  # noqa: E731

    body = (
        "**No port is licensed under #421.** No candidate clears ACTIVATE "
        "(`s_wall>=10%` on any decision-grade leg) or DECLINE (both `s_wall+U_wall<5%` AND "
        "`s_busy+U_busy<5%` on every decision-grade leg) — the per-leg numbers are quoted "
        "verbatim above. This is not uniform across candidates or axes: `C-MLP`'s own measured "
        "`s_wall` (no `U` term) is only "
        f"{gelu_wall_pct('clip-text', 'A1'):.2f} %/{gelu_wall_pct('clip-text', 'A2'):.2f} % on "
        f"CLIP-text (A1/A2) and {gelu_wall_pct('clip-vision', 'A1'):.2f} %/"
        f"{gelu_wall_pct('clip-vision', 'A2'):.2f} % on CLIP-vision — well under the 5 % "
        "DECLINE floor on the wall axis, combined or not (`s_wall+U_wall` above is "
        f"{combined_pct('clip-text', 'A1', 'wall'):.2f} %/{combined_pct('clip-text', 'A2', 'wall'):.2f} % "
        f"and {combined_pct('clip-vision', 'A1', 'wall'):.2f} %/"
        f"{combined_pct('clip-vision', 'A2', 'wall'):.2f} %, still under 5 %) — it is the "
        "combined *busy* share (`s_busy+U_busy`, "
        f"{combined_pct('clip-text', 'A1', 'busy'):.2f} %/{combined_pct('clip-text', 'A2', 'busy'):.2f} % "
        f"CLIP-text, {combined_pct('clip-vision', 'A1', 'busy'):.2f} %/"
        f"{combined_pct('clip-vision', 'A2', 'busy'):.2f} % CLIP-vision) that lands in the "
        "contract's 5–10 % band and is what keeps DECLINE from firing. The two-sided rule does "
        "exactly what it was pre-registered to do: it refuses to manufacture a verdict a "
        "5–10 % share does not support, on either side."
    )
    return textwrap.fill(body, width=88, break_long_words=False, break_on_hyphens=False)


# The maintainer guide's own display capitalization for a candidate port
# name -- the artifact's own `port` field is lowercase-tower-shaped
# (`C-ATTN-clip-text`, README's own verbatim rendering); the guide's denser
# prose capitalizes the tower name instead (`C-ATTN-CLIP-text`). A small,
# stable display map, same shape as `_TOWER_DISPLAY` above -- never a
# string transform that could silently mis-capitalize a future port name.
_GUIDE_PORT_DISPLAY = {
    "C-ATTN-clip-text": "C-ATTN-CLIP-text",
    "C-MLP-clip-text": "C-MLP-CLIP-text",
    "C-ATTN-clip-vision": "C-ATTN-CLIP-vision",
    "C-MLP-clip-vision": "C-MLP-CLIP-vision",
}


def _reason_leg_shares(reason: str) -> str:
    """The per-leg `s_wall+U_wall=...`/`s_busy+U_busy=...` segment of a
    `candidate_decisions[].reason` string -- everything after its own
    " — " separator (the ACTIVATE/DECLINE clause every reason repeats
    verbatim), the exact substring the guide's own condensed prose quotes
    for every port after the first (elided prefix)."""
    return reason.split(" — ", 1)[1]


def _finding(artifact: dict, finding_id: str) -> dict:
    """The full `findings[]` entry (id/text/evidence) for `finding_id` --
    never just its `text` -- so a renderer that needs a QUALITATIVE word
    ("front-end-bound", "launch-bound", "dtype- and arm-invariant", "only")
    can check the licensing `evidence.<rule>` flag the word is GATED on
    (`profile_421_artifact.py`'s own `compute_findings`), rather than
    re-derive whether the word applies by regexing the sentence it does (or
    does not) appear in. A finding whose required leg(s) failed and was
    therefore SUPPRESSED (`suppressed_findings`) is refused here BY NAME,
    never silently treated as an absent/empty finding -- a renderer whose
    prose depends on this finding existing must fail closed the moment it
    does not, not render nothing.
    """
    for finding in artifact["findings"]:
        if finding["id"] == finding_id:
            return finding
    suppressed_ids = {s["id"] for s in artifact.get("suppressed_findings", [])}
    if finding_id in suppressed_ids:
        raise ValueError(
            f"finding {finding_id!r} is SUPPRESSED in this artifact (see suppressed_findings) "
            "-- the prose that depends on it must be rewritten, never rendered from a finding "
            "the producer never built"
        )
    raise ValueError(f"finding {finding_id!r} not present in artifact findings[]")


def _require_evidence_flag(finding: dict, flag: str, word: str) -> None:
    """Fails closed (`ValueError`, `--check` red) the moment the artifact
    WITHHOLDS the rule outcome a piece of FIXED prose asserts
    unconditionally -- `evidence[flag]` absent (the rule was never
    evaluated on this run, e.g. an optional arm-invariance clause whose
    D-arm legs were not all VALID) or `False` (evaluated and did NOT hold)
    are BOTH refusals, never a silent "render the word anyway". A renderer
    that hits this must be rewritten into a neutral sentence (the
    artifact's own finding `text` already carries one -- `compute_findings`
    drops the word for a neutral sentence stating the same numbers when its
    rule does not hold) rather than keep asserting a word this run's own
    evidence does not license.
    """
    value = finding.get("evidence", {}).get(flag)
    if value is not True:
        raise ValueError(
            f"finding {finding['id']!r} evidence[{flag!r}] is {value!r}, not True -- the "
            f"{word!r} wording this renderer's fixed prose asserts is NOT licensed by this "
            "artifact; rewrite the renderer (and the prose) into a neutral sentence instead "
            "of rendering a word the run's own evidence withholds"
        )


def _range_str(mapping: dict[str, float]) -> str:
    """`lo-hi` (en-dash, `profile_421_artifact.py`'s own `_fmt_range` --
    IDENTICAL rounding/collapse rule the finding's own `text` was rendered
    with) over an `evidence` mapping's values -- never a digit regexed back
    out of the finding's own prose sentence."""
    lo, hi = min(mapping.values()), max(mapping.values())
    return _pa._fmt_range(lo, hi).replace("-", "–")


def _magnitude_range(mapping: dict[str, float]) -> str:
    """The SAME magnitude-range computation `compute_findings`'s own
    uniform-negative branch builds the "cuts GPU busy by X%" / "wall drops
    by only Y%" clause from (sorted by `abs`, then `_fmt_range` over the
    two magnitude extremes, en-dash not hyphen -- same convention as
    `_range_str`) -- reused here rather than re-derived, so a renderer can
    never drift from the exact numbers the finding's own text was built
    from.

    `compute_findings` only ever builds that blanket "cuts"/"drops" clause
    when EVERY value in `mapping` is negative (its own uniform-negative
    branch) -- a mixed-sign or non-negative `mapping` instead gets a
    per-tower, sign-derived sentence (`_direction_clause`) that names each
    tower's own direction individually, never a single blanket verb. This
    producer-side branch choice is NOT itself recorded as a boolean in
    `evidence` (only the resulting numbers are), so every caller of this
    helper GATES the direction verbs on an explicit, live sign check of the
    SAME signed deltas `evidence` carries -- taking `abs()` unconditionally
    would keep asserting "cuts"/"drops" even once a future run's own
    deltas are not uniformly negative, which is exactly the "prose stops
    matching the sign the artifact actually recorded" drift this gate
    exists to make impossible. Refuses (`ValueError`, `--check` red) rather
    than render a directional verb the run's own signed evidence does not
    license.
    """
    if not all(v < 0.0 for v in mapping.values()):
        raise ValueError(
            f"{mapping!r} is not uniformly negative -- the 'cuts'/'drops' direction verbs this "
            "renderer's fixed prose asserts are only licensed when EVERY value in this evidence "
            "mapping decreases; rewrite the renderer (and the prose) into a per-tower, "
            "sign-derived sentence instead of a blanket magnitude range the run's own signed "
            "deltas do not license"
        )
    by_magnitude = sorted(mapping.values(), key=abs)
    return _pa._fmt_range(abs(by_magnitude[0]), abs(by_magnitude[-1])).replace("-", "–")


def render_realized_gains_guide(artifact: dict) -> str:
    """The guide's own condensed restatement of the realized-gain figures
    (denser wording than README's three bullets -- one prose sentence, no
    bullets), same `realized_gains[]` source."""
    by_key = _realized_gains_by_key(artifact)
    lora = {t: by_key[("C-LORA", t)] for t in _TOWER_ORDER}
    ln_clip = {t: by_key[("C-LN", t)] for t in ("clip-text", "clip-vision")}
    joint = by_key[("C-LN", "htsat")]
    body = (
        "The already-fused chains' realized gains (eager twin minus fused, per step): C-LORA "
        f"+{_ms(lora['clip-text']):.1f} ms wall on CLIP-text "
        f"({_pct_of_baseline(lora['clip-text']):.1f} % of the A1 baseline wall), "
        f"+{_ms(lora['clip-vision']):.1f} ms on CLIP-vision "
        f"({_pct_of_baseline(lora['clip-vision']):.1f} %), +{_ms(lora['htsat']):.1f} ms on "
        f"HTSAT ({_pct_of_baseline(lora['htsat']):.1f} %); C-LN +{_ms(ln_clip['clip-text']):.1f} "
        f"ms on CLIP-text, +{_ms(ln_clip['clip-vision']):.1f} ms on CLIP-vision; on HTSAT, D1 "
        "disables `layer_norm_fused` AND `gelu_erf_fused` together, so its D1-minus-D2 delta "
        f"(+{_ms(joint):.1f} ms, {_pct_of_baseline(joint):.1f} % of the A1 baseline wall) is "
        "the JOINT C-LN + C-GELU-HTSAT gain, not C-LN alone."
    )
    return textwrap.fill(body, width=91, break_long_words=False, break_on_hyphens=False)


def render_decline_band_guide(artifact: dict) -> str:
    """The guide's own condensed restatement of the DECLINE-band paragraph
    plus the four candidate ports' verbatim reasons (elided after the
    first) -- same source (`attribution[]`/`candidate_decisions[]`) as
    `render_decline_band_summary`, denser wording."""
    by_leg = _attribution_by_leg(artifact)
    cd = artifact["candidate_decisions"]
    port0, port1, port2, port3 = cd[0], cd[1], cd[2], cd[3]

    body = (
        "**All four candidate ports are UNRESOLVED — no port is licensed under #421.** No "
        "candidate clears ACTIVATE (`s_wall>=10%` on any decision-grade leg) or DECLINE (both "
        "`s_wall+U_wall<5%` AND `s_busy+U_busy<5%` on every decision-grade leg); the pass-4 "
        "`kernel_census.py` demangled-name fix (`docs/maintainer/MAINTAINER-GUIDE.md` §2.5) "
        "makes both CLIP-tower A2 legs decision-grade for attribution (`htsat-A2` stays "
        "non-decision-grade — HTSAT has no candidate port under this contract, so that never "
        "blocks a verdict; see `docs/plans/66-tower-profile/README.md`), so no verdict below "
        "is F32-only. This is not uniform across candidates or axes: `C-MLP`'s own measured "
        "`s_wall` (no `U` term) is only "
        f"{_gelu_wall_pct(by_leg, 'clip-text', 'A1'):.2f} %/"
        f"{_gelu_wall_pct(by_leg, 'clip-text', 'A2'):.2f} % on CLIP-text and "
        f"{_gelu_wall_pct(by_leg, 'clip-vision', 'A1'):.2f} %/"
        f"{_gelu_wall_pct(by_leg, 'clip-vision', 'A2'):.2f} % on CLIP-vision — well under the "
        "5 % DECLINE floor on wall, combined or not — it is the combined *busy* share "
        f"(`s_busy+U_busy`, {_combined_pct(by_leg, 'clip-text', 'A1', 'busy'):.2f} %/"
        f"{_combined_pct(by_leg, 'clip-text', 'A2', 'busy'):.2f} % CLIP-text, "
        f"{_combined_pct(by_leg, 'clip-vision', 'A1', 'busy'):.2f} %/"
        f"{_combined_pct(by_leg, 'clip-vision', 'A2', 'busy'):.2f} % CLIP-vision) that lands "
        "in the contract's 5–10 % band and keeps DECLINE from firing. Verbatim reasons "
        "(artifact `candidate_decisions[]`): "
        f"`{_GUIDE_PORT_DISPLAY[port0['port']]}` — {port0['verdict']}, \"{port0['reason']}\"; "
        f"`{_GUIDE_PORT_DISPLAY[port1['port']]}` — {port1['verdict']}, "
        f"\"…{_reason_leg_shares(port1['reason'])}\" (elided prefix identical to "
        f"`{_GUIDE_PORT_DISPLAY[port0['port']]}`'s above; full text at artifact "
        "`candidate_decisions[1].reason`); "
        f"`{_GUIDE_PORT_DISPLAY[port2['port']]}` — {port2['verdict']}, "
        f"\"…{_reason_leg_shares(port2['reason'])}\" (`candidate_decisions[2].reason`); "
        f"`{_GUIDE_PORT_DISPLAY[port3['port']]}` — {port3['verdict']}, "
        f"\"…{_reason_leg_shares(port3['reason'])}\" (`candidate_decisions[3].reason`). The "
        "two-sided rule refuses to manufacture a verdict a 5–10 % share does not support on "
        "either side — that refusal, not a missing signal, is why nothing ports."
    )
    return textwrap.fill(body, width=91, break_long_words=False, break_on_hyphens=False)


def render_findings_guide(artifact: dict) -> str:
    """The guide's own condensed restatement of the four `findings[]`
    entries -- one prose paragraph, no bullets, denser wording than
    README's `findings` block. Every NUMBER is read off each finding's own
    structured `evidence` (never a digit regexed back out of its `text`),
    and every QUALITATIVE word this paragraph's fixed prose asserts
    ("CPU front-end-bound", "dtype- and arm-invariant", "launch-bound",
    "only") is gated on that SAME finding's own `evidence.<rule>` flag via
    `_require_evidence_flag` -- a run whose evidence withholds one of those
    words fails this renderer CLOSED rather than render a claim the run's
    own numbers do not support.
    """
    total_files, train_clips, _m_leg_rows = corpus_pool_counts()

    htsat_finding = _finding(artifact, "htsat-front-end-bound")
    _require_evidence_flag(htsat_finding, "front_end_bound", "CPU front-end-bound")
    _require_evidence_flag(htsat_finding, "arm_invariant", "dtype- and arm-invariant")
    htsat_pct = _range_str(htsat_finding["evidence"]["front_share_of_wall_pct"]) + " %"

    clip_vision_finding = _finding(artifact, "clip-vision-front-end-share")
    clip_vision_pct = _range_str(clip_vision_finding["evidence"]["front_share_of_wall_pct"]) + " %"

    launch_finding = _finding(artifact, "clip-launch-bound-batch8")
    _require_evidence_flag(launch_finding, "launch_bound", "launch-bound")
    _require_evidence_flag(
        launch_finding, "wall_drop_smaller_than_busy_drop_every_tower", "only"
    )
    launch_evidence = launch_finding["evidence"]
    launches_range = _range_str(launch_evidence["launches_per_step"])
    busy_range = _magnitude_range(launch_evidence["busy_delta_pct_bf16_vs_f32"])
    wall_range = _magnitude_range(launch_evidence["wall_delta_pct_bf16_vs_f32"])

    attn_finding = _finding(artifact, "c-attn-htsat-out-of-tier")
    attn_busy_pct = f"{attn_finding['evidence']['share_gpu_busy'] * 100:.0f}"
    attn_wall_pct = f"{attn_finding['evidence']['share_wall'] * 100:.0f}"

    body = (
        f"**Findings.** The HTSAT training step is CPU front-end-bound: front-end share of "
        f"wall is {htsat_pct} across the F32/BF16 decision legs (audio "
        "decode/resample/STFT/mel dominating wall time), dtype- and arm-invariant — closed as "
        "its own follow-on unit on `perf/421-frontend` (parallelizing the media front end "
        "across rayon's global pool), not duplicated here. CLIP-vision's own image "
        f"decode/preprocess front end is {clip_vision_pct} of wall on the F32/BF16 decision "
        f"legs. Both media corpus producers cycle only {train_clips} distinct train clips "
        f"(families × instances = {total_files} files at any `--rows`) — a page-cached "
        "working set, not a realistic-corpus I/O cost — so both front-end numbers are a real "
        "per-item CPU decode/preprocess compute cost, never disk I/O (artifact "
        "`notes.recorded_deviations`; full caveat: `docs/plans/66-tower-profile/README.md`). "
        f"At batch 8 the CLIP training steps are launch-bound ({launches_range} "
        f"launches/step across the four F32/BF16 A-arm CLIP legs); BF16 cuts GPU busy "
        f"{busy_range} % per tower while wall drops only {wall_range} %. `C-ATTN-HTSAT` is "
        f"measured, not a candidate port: {attn_busy_pct} % of GPU busy (~{attn_wall_pct} % of "
        "wall) on the F32 decision leg — HTSAT's head_dim of 24 at every stage sits outside "
        "the fixed-head-dim port tier by the contract's own declaration, so this stays a "
        "measured, OPEN number, never folded into UNATTRIBUTED and never decided under this "
        "issue. Full per-leg table, deviations and the PR trail: "
        "`docs/plans/66-tower-profile/README.md`."
    )
    return textwrap.fill(body, width=91, break_long_words=False, break_on_hyphens=False)


def render_changelog_421_entry(artifact: dict) -> str:
    """The CHANGELOG's own close-out entry for #421 -- every headline
    number restated one more time, in the CHANGELOG's own terse style,
    all derived from the same `attribution[]`/`candidate_decisions[]`/
    `realized_gains[]`/`findings[]` this script's other blocks already
    read (plus `corpus_pool_counts` for the "16 distinct train clips"
    figure) -- never a fourth hand-typed copy of the same nine numbers.
    """
    by_leg = _attribution_by_leg(artifact)
    for leg in artifact["legs"]:
        if by_leg[leg["leg_id"]]["verdict"] != "VALID":
            raise ValueError(
                f"leg {leg['leg_id']!r} is not VALID -- rewrite the CHANGELOG entry this "
                "renders, its own headline claims all 12 legs VALID"
            )
    if len(artifact["legs"]) != 12:
        raise ValueError(
            f"expected 12 legs, got {len(artifact['legs'])} -- rewrite the CHANGELOG entry"
        )

    gelu_wall_vals = [
        _gelu_wall_pct(by_leg, t, a) for t in _CLIP_DECISION_TOWERS for a in ("A1", "A2")
    ]
    gelu_busy_vals = [
        _combined_pct(by_leg, t, a, "busy") for t in _CLIP_DECISION_TOWERS for a in ("A1", "A2")
    ]
    attn_busy_vals = [
        _chain_combined_pct(by_leg, t, a, f"C-ATTN-{t}", "busy")
        for t in _CLIP_DECISION_TOWERS
        for a in ("A1", "A2")
    ]

    by_key = _realized_gains_by_key(artifact)
    lora = {t: by_key[("C-LORA", t)] for t in _TOWER_ORDER}
    ln_clip = {t: by_key[("C-LN", t)] for t in ("clip-text", "clip-vision")}
    joint = by_key[("C-LN", "htsat")]

    _total_files, train_clips, _m_leg_rows = corpus_pool_counts()

    htsat_finding = _finding(artifact, "htsat-front-end-bound")
    _require_evidence_flag(htsat_finding, "front_end_bound", "CPU front-end-bound")
    _require_evidence_flag(htsat_finding, "arm_invariant", "dtype- and arm-invariant")
    htsat_pct = _range_str(htsat_finding["evidence"]["front_share_of_wall_pct"])

    clip_vision_finding = _finding(artifact, "clip-vision-front-end-share")
    clip_vision_pct = _range_str(clip_vision_finding["evidence"]["front_share_of_wall_pct"])

    launch_finding = _finding(artifact, "clip-launch-bound-batch8")
    _require_evidence_flag(launch_finding, "launch_bound", "launch-bound")
    _require_evidence_flag(
        launch_finding, "wall_drop_smaller_than_busy_drop_every_tower", "only"
    )
    launch_evidence = launch_finding["evidence"]
    launches_vals = list(launch_evidence["launches_per_step"].values())
    launches_k = f"{min(launches_vals) / 1000:.1f}–{max(launches_vals) / 1000:.1f}"
    busy_cut_pct = _magnitude_range(launch_evidence["busy_delta_pct_bf16_vs_f32"])
    wall_drop_pct = _magnitude_range(launch_evidence["wall_delta_pct_bf16_vs_f32"])

    attn_finding = _finding(artifact, "c-attn-htsat-out-of-tier")
    attn_htsat_busy_pct = f"{attn_finding['evidence']['share_gpu_busy'] * 100:.0f}"

    body = (
        "**The #421 tower training-step profile is closed out: driver, merge, attribution, "
        "and a committed close-out artifact (issue #421 step 3, \"PROFILE FIRST\").** All 12 "
        "legs (`A1`/`A2`/`D1`/`D2` × CLIP-text, OpenCLIP-vision, HTSAT) are VALID on "
        "`crates/jammi-kernels/artifacts/cuda-runs/2026-09-07-profile-421-towers-c1b0b0ba-a100-sxm4.json` "
        "(A100-SXM4-80GB); the BF16 pre-flight (P2) passes on all three towers. "
        "`profile_421_attribute.py` (new) reads `profile_421_merge.py`'s per-key equations "
        "and the kernel census into the contract's named chains and evaluates the two-sided "
        "ACTIVATE/DECLINE/UNRESOLVED rule per candidate port; `kernel_census.py` now keys "
        "each GPU-kernel bucket on `COALESCE(demangledName, shortName)` rather than "
        "`shortName` alone, a sum-preserving refinement that un-collapses cutlass's "
        "`Kernel2<...>` template wrapper's distinct bf16 GEMM tile instantiations -- rather "
        "than summing them into one anonymous row that could trip the attribution's "
        "known-kernel-name gate -- without moving any top-line "
        "`gpu_kernel_us_per_step`/wall/front/busy number. "
        "**No kernel port lands under #421**: all four candidate ports the contract named "
        "(`C-ATTN-CLIP-text`, `C-MLP-CLIP-text`, `C-ATTN-CLIP-vision`, `C-MLP-CLIP-vision`) "
        "resolve **UNRESOLVED** — none clears ACTIVATE (`s_wall≥10%` on any decision-grade "
        "leg) or DECLINE (combined share <5% on wall AND busy on every decision-grade leg), "
        "F32 and BF16 alike. This is not uniform across candidates or axes: only C-MLP's own "
        f"combined *busy* share (`s_busy+U_busy`, {min(gelu_busy_vals):.2f}–{max(gelu_busy_vals):.2f} % "
        "across the CLIP towers) lands in the contract's 5–10 % band — its wall-axis share "
        f"stays {min(gelu_wall_vals):.2f}–{max(gelu_wall_vals):.2f} % throughout, well under "
        f"the 5 % DECLINE floor — while C-ATTN's combined busy shares run "
        f"{round(min(attn_busy_vals))}–{round(max(attn_busy_vals))} %, entirely outside that "
        "band. The already-fused chains' realized gains, per step: C-LORA "
        f"+{_ms(lora['clip-text']):.1f} ms (CLIP-text, {_pct_of_baseline(lora['clip-text']):.1f} % of wall), "
        f"+{_ms(lora['clip-vision']):.1f} ms (CLIP-vision, {_pct_of_baseline(lora['clip-vision']):.1f} %), "
        f"+{_ms(lora['htsat']):.1f} ms (HTSAT, {_pct_of_baseline(lora['htsat']):.1f} %); "
        f"C-LN +{_ms(ln_clip['clip-text']):.1f} ms (CLIP-text), +{_ms(ln_clip['clip-vision']):.1f} ms "
        f"(CLIP-vision); the joint C-LN+C-GELU-HTSAT chain +{_ms(joint):.1f} ms "
        f"({_pct_of_baseline(joint):.1f} %). Findings: the HTSAT training step is CPU "
        f"front-end-bound (audio decode/resample/STFT/mel ≈ {htsat_pct} % of wall, dtype- and "
        f"arm-invariant); CLIP-vision's image front end is ≈ {clip_vision_pct} % of wall "
        f"(both corpora cycle only {train_clips} distinct train clips at any row count — a "
        "page-cached working set, never a realistic-corpus I/O cost — so both numbers are a "
        "real per-item CPU decode/preprocess compute cost; see "
        "`docs/plans/66-tower-profile/README.md`'s deviations); at batch 8 the CLIP steps "
        f"are launch-bound (≈ {launches_k} k launches/step; BF16 cuts GPU busy {busy_cut_pct} % "
        f"but wall only {wall_drop_pct} %); `C-ATTN-HTSAT` is measured (≈ {attn_htsat_busy_pct} % "
        "of GPU busy) and stays OUT OF TIER, a named-but-undecided chain, never folded into "
        "UNATTRIBUTED. See `docs/plans/66-tower-profile/README.md` and `CONTRACT.md` (the "
        "frozen v2.5 contract) for the full per-leg table and PR trail."
    )
    return textwrap.fill(
        body, width=88, initial_indent="- ", subsequent_indent="  ",
        break_long_words=False, break_on_hyphens=False,
    )


_DECISION_GRADE_REASON_RE = re.compile(
    r"^UNATTRIBUTED share_gpu_busy=([0-9.eE+-]+) > ([0-9.eE+-]+)$"
)


def _htsat_a1_named_not_decision_grade(artifact: dict) -> list[dict]:
    """Every `suppressed_findings[]` entry that names `htsat-A1` among its
    `legs` AND whose own `reason` string is about `decision_grade` -- the
    artifact's OWN record of a leg it built findings machinery around and
    judged non-decision-grade. Read structurally (leg id membership + the
    literal field name in the reason), never by re-deriving the judgment,
    so this catches the artifact contradicting itself regardless of which
    finding surfaced the contradiction."""
    return [
        s
        for s in artifact.get("suppressed_findings", [])
        if "htsat-A1" in s.get("legs", []) and "decision_grade" in s.get("reason", "")
    ]


def render_htsat_a2_deviation(artifact: dict) -> str:
    """The Deviations bullet naming `htsat-A2`'s own UNATTRIBUTED share of
    GPU busy. Every number AND word this bullet's fixed prose needs is read
    live, off the artifact's OWN attribution/evidence, never a hard-coded
    literal or a live import of a module constant that could since have
    moved independently of the run that produced this artifact:

    - the "VALID [at the merge level]" word from `legs[htsat-A2].
      merge_verdict` (never assumed);
    - the "not decision-grade" phrase from `attribution[htsat-A2].
      decision_grade` itself (never a hard-coded negation baked in ahead of
      the check);
    - the comparison word ("over") and the "N % validity bound" by PARSING
      `attribution[htsat-A2].decision_grade_reason` (`profile_421_attribute.
      py`'s own `leg_decision_grade` writes this exact
      `"UNATTRIBUTED share_gpu_busy=<share> > <bound>"` string into the
      artifact at run time) -- cross-checked against the same leg's own
      `chains.UNATTRIBUTED.share_gpu_busy` (the two must agree) and against
      an ACTUAL `>` comparison of the two floats (never assumed "over"
      because that is the only branch this bullet has ever seen); a `<`/`=`
      artifact, or a `decision_grade_reason` that is not this exact shape
      (e.g. a non-finite-share refusal), fails this renderer CLOSED rather
      than silently keep saying "over";
    - the `htsat-A1` clause off `attribution[htsat-A1]` itself
      (`decision_grade` for the True/False branch, `chains.UNATTRIBUTED.
      share_gpu_busy` to verify it ACTUALLY clears the same bound before
      the bullet is allowed to say so) -- and refuses outright the moment
      the artifact's own `suppressed_findings` names `htsat-A1` as not
      decision-grade (`_htsat_a1_named_not_decision_grade`) while
      `attribution[htsat-A1].decision_grade` is `True`: that is the
      artifact contradicting itself, not a case this bullet may paper over.

    Fails closed (never silently re-labels the bullet) the moment any of
    these stop matching what this bullet's own fixed prose assumes.
    """
    merge_by_leg = {leg["leg_id"]: leg for leg in artifact["legs"]}
    htsat_a2_merge_verdict = merge_by_leg["htsat-A2"]["merge_verdict"]
    if htsat_a2_merge_verdict != "VALID":
        raise ValueError(
            f"expected htsat-A2 merge_verdict VALID, got {htsat_a2_merge_verdict!r} -- this "
            "bullet's own fixed prose ('is VALID but not decision-grade') assumes this leg IS "
            "merge-VALID; rewrite it if that changed"
        )
    by_leg = _attribution_by_leg(artifact)
    a2 = by_leg["htsat-A2"]
    decision_grade = a2["decision_grade"]
    if decision_grade is not False:
        raise ValueError(
            f"expected htsat-A2 decision_grade False, got {decision_grade!r} -- this bullet's "
            "own fixed prose explains WHY it is non-decision-grade; rewrite it (and the "
            "README section around it) if the artifact's own verdict changed"
        )

    a2_share = a2["chains"]["UNATTRIBUTED"]["share_gpu_busy"]
    reason = a2.get("decision_grade_reason") or ""
    match = _DECISION_GRADE_REASON_RE.match(reason)
    if match is None:
        raise ValueError(
            f"htsat-A2 decision_grade_reason {reason!r} is not the "
            "'UNATTRIBUTED share_gpu_busy=<share> > <bound>' shape this bullet's own "
            "comparison and bound are parsed from -- rewrite this renderer (and the bullet) "
            "if the failing rule (or its reason string) changed"
        )
    reason_share, bound = float(match.group(1)), float(match.group(2))
    if round(reason_share, 4) != round(a2_share, 4):
        raise ValueError(
            f"htsat-A2 decision_grade_reason share ({reason_share!r}) disagrees with the same "
            f"leg's own chains['UNATTRIBUTED']['share_gpu_busy'] ({a2_share!r}) -- these two "
            "artifact fields must agree; refuse rather than silently pick one"
        )
    if a2_share > bound:
        comparison_word = "over"
    elif a2_share < bound:
        comparison_word = "under"
    else:
        comparison_word = "at"
    if comparison_word != "over":
        raise ValueError(
            f"htsat-A2 UNATTRIBUTED share is {comparison_word} its own bound "
            f"({a2_share!r} vs {bound!r}) -- this bullet's own fixed prose ('not "
            "decision-grade ... over the ... validity bound') assumes the share is OVER the "
            "bound; rewrite it if the artifact's own direction changed"
        )
    share_pct = a2_share * 100
    bound_pct = bound * 100.0

    a1 = by_leg["htsat-A1"]
    a1_decision_grade = a1["decision_grade"]
    a1_contradictions = _htsat_a1_named_not_decision_grade(artifact)
    if a1_contradictions and a1_decision_grade is True:
        raise ValueError(
            f"htsat-A1 attribution reports decision_grade=True but this artifact's own "
            f"suppressed_findings {a1_contradictions!r} names htsat-A1 as not decision-grade "
            "-- these two facts contradict each other in the SAME artifact; refuse rather "
            "than render a claim about htsat-A1 the artifact itself disputes"
        )
    if a1_decision_grade is not True:
        raise ValueError(
            f"expected htsat-A1 decision_grade True, got {a1_decision_grade!r} -- this "
            "bullet's own fixed prose ('clears the bound and is decision-grade') assumes "
            "htsat-A1 IS decision-grade; rewrite it if that changed"
        )
    a1_share = a1["chains"]["UNATTRIBUTED"]["share_gpu_busy"]
    if not a1_share < bound:
        raise ValueError(
            f"htsat-A1 UNATTRIBUTED share ({a1_share!r}) does not actually clear the "
            f"{bound!r} bound htsat-A2 was judged against -- this bullet's own fixed prose "
            "('clears the bound') assumes it does; rewrite it if that changed"
        )

    body = (
        f"**`htsat-A2` (bf16) is {htsat_a2_merge_verdict} but not decision-grade for "
        f"attribution**: its UNATTRIBUTED share of GPU busy is {share_pct:.2f} %, "
        f"{comparison_word} the contract's {bound_pct:.0f} % validity bound (window-partition "
        "copies and the audio front end's own activation are still undeclared chains at the "
        "identical element count as a generic residual-stream permute/reshape copy — the "
        "attribution module declares neither rather than guess). `htsat-A1` (f32) clears the "
        "bound and is decision-grade. HTSAT has no candidate port under this contract in the "
        "first place, so `htsat-A2`'s own non-decision-grade status never blocks a "
        "candidate-port decision."
    )
    return _bullet(body)


def _nsys_version(nsys_human: str) -> str:
    """The bare version token off the front of `notes.nsys` -- that field is
    a longer provenance sentence (install source, refused alternates), not
    a bare version string, so the README's own terse "nsys 2025.3.2.474"
    parenthetical reads only its leading `Nsight Systems <version>` token,
    never a hand-typed copy of the version alone."""
    m = re.match(r"Nsight Systems (\S+)", nsys_human)
    if m is None:
        raise ValueError(
            f"notes.nsys does not start with 'Nsight Systems <version>' ({nsys_human!r}) -- "
            "rewrite this extractor (and the README sentence it feeds) if the identity "
            "sidecar's own nsys field shape changed"
        )
    return m.group(1)


def render_measured_summary(artifact: dict) -> str:
    """The "## What was measured" opening paragraph -- its own leg count,
    tower count, device/driver/nsys identity and git sha are ALL read live
    off the artifact (`legs`, `p2_witnessed`, `status`, `notes`, `git_sha`),
    never hand-typed literals a re-run's own numbers could drift under
    without this block moving. Fails closed (the paragraph's fixed "All N
    legs are VALID" / "passes on all N towers" wording is unconditional)
    the moment `status` is not GREEN or `p2_witnessed` does not name every
    tower this run actually measured.
    """
    total_legs = len(artifact["legs"])
    status = artifact["status"]
    if status != "GREEN":
        raise ValueError(
            f"artifact status is {status!r}, not GREEN -- this paragraph's fixed 'All N legs "
            "are VALID; the BF16 pre-flight (P2) passes' wording assumes a clean pass; rewrite "
            "it to name the actual failure(s) instead"
        )
    towers = sorted({leg["tower"] for leg in artifact["legs"]})
    p2_witnessed = artifact.get("p2_witnessed") or {}
    if sorted(p2_witnessed.get("towers", [])) != towers:
        raise ValueError(
            f"expected p2_witnessed to name every measured tower {towers!r}, got "
            f"{p2_witnessed.get('towers')!r} -- rewrite the 'BF16 pre-flight (P2) passes on "
            "all N towers' sentence if this run's own P2 tower set differs from its legs"
        )
    tower_count = len(towers)
    if tower_count not in _NUMBER_WORDS:
        raise ValueError(f"{tower_count} towers has no spelled-out form registered in _NUMBER_WORDS -- add one")
    notes = artifact["notes"]
    artifact_rel = ARTIFACT.relative_to(REPO_ROOT).as_posix()
    body = (
        f"All {total_legs} legs are VALID; the BF16 pre-flight (P2) passes on all "
        f"{_NUMBER_WORDS[tower_count].lower()} towers. Source: `{artifact_rel}` "
        f"({notes['gpu']}, driver {notes['driver']}, nsys {_nsys_version(notes['nsys'])}, "
        f"git sha `{artifact['git_sha']}`)."
    )
    return textwrap.fill(body, width=97, break_long_words=False, break_on_hyphens=False)


def _root_cause_deviation_text(artifact: dict) -> str:
    """The ONE `notes.recorded_deviations` entry naming both the census-key
    split count and the known-kernel-name gate percentage -- identified by
    two stable substrings (never a positional index, which would silently
    start reading the WRONG deviation the day the sidecar's own entry order
    changes) -- so `render_census_key_root_cause_note` reads its two
    numbers off the SAME rendered sentence `profile_421_artifact.py`'s own
    `_identity_template_context` already filled in from a live source, not
    a second, independently-drifting copy computed here (this doc-render
    step never has the raw census files `kernel_identity_split_count`
    itself needs -- only `profile_421_artifact.py`, which reads `--legs-dir`
    directly, can compute that number from scratch)."""
    for text in artifact["notes"]["recorded_deviations"]:
        if "DISTINCT cutlass instantiations" in text and "known-kernel-name gate" in text:
            return text
    raise ValueError(
        "no notes.recorded_deviations entry names both 'DISTINCT cutlass instantiations' and "
        "'known-kernel-name gate' -- the census-key root-cause bullet's own split-count/gate-"
        "percentage numbers have no live source to read from; rewrite this extractor (and the "
        "bullet) if the identity sidecar's own deviation wording changed"
    )


def render_census_key_root_cause_note(artifact: dict) -> str:
    """The "census-key root cause" Deviations bullet -- its own split count
    ("N distinct cutlass instantiations") and known-kernel-name-gate
    percentage are the SAME two numbers already rendered into `notes.
    recorded_deviations` (`_root_cause_deviation_text`), read off THAT
    sentence rather than hand-retyped a second time.
    """
    deviation_text = _root_cause_deviation_text(artifact)
    split_m = re.search(r"has (\d+) DISTINCT cutlass instantiations", deviation_text)
    gate_m = re.search(r"known-kernel-name gate", deviation_text) and re.search(
        r"tripped profile_421_attribute\.py's (\d+)% known-kernel-name gate", deviation_text
    )
    if split_m is None or gate_m is None:
        raise ValueError(
            f"could not extract the split count / gate percentage from {deviation_text!r} -- "
            "rewrite this extractor (and the census-key root-cause bullet) if the identity "
            "sidecar's own deviation wording changed"
        )
    split_count_word = _NUMBER_WORDS.get(int(split_m.group(1)))
    if split_count_word is None:
        raise ValueError(
            f"{split_m.group(1)} has no spelled-out form registered in _NUMBER_WORDS -- add one"
        )
    gate_pct = gate_m.group(1)
    body = (
        "**The census-key root cause (pass-4, `perf/421-attribution`).** "
        "`kernel_census.py` keyed each GPU-kernel bucket on `shortName` alone; cutlass's "
        "`Kernel2<...>` template wrapper gives every bf16 GEMM tile instantiation the same "
        f"literal `shortName`, so {split_count_word.lower()} distinct cutlass instantiations "
        "on `clip-text-A2` collapsed into one anonymous row that tripped the attribution's "
        f"{gate_pct} % known-kernel-name gate. Fixed by keying on `COALESCE(demangledName, "
        "shortName)` instead — a strict, sum-preserving refinement (a bucket can only split, "
        "never merge two old buckets into fewer new ones): every top-line number "
        "(`gpu_kernel_us_per_step`, wall/front/busy per step) is unchanged; only the "
        "per-instantiation breakdown resplit. Both CLIP-tower A2 legs are decision-grade for "
        "attribution under the fix."
    )
    return _bullet(body)


def _int_constant(text: str, pattern: str, source: Path) -> int:
    m = re.search(pattern, text, re.MULTILINE)
    if m is None:
        raise ValueError(f"{source}: pattern {pattern!r} not found -- constant moved or renamed")
    return int(m.group(1))


def corpus_pool_counts() -> tuple[int, int, int]:
    """`(total_files, train_clips, m_leg_rows)` -- LIVE, never hand-copied:
    the driver's own pinned `MEDIA_FAMILIES`/`MEDIA_HELDOUT_FAMILIES`/
    `STEPS_M`/`BATCH` constants (`profile_421_legs.sh`) and the audio
    corpus generator's own `_DEFAULT_INSTANCES_PER_FAMILY` default
    (`gen_fixed_length_audio_corpus.py`) -- `total_files = families *
    instances`, `train_clips = (families - heldout_families) * instances`
    (the producer's own family-reservation rule), `m_leg_rows = steps_m *
    batch` (the driver's own `rows = 3B` convention divides out identically
    per triplet member, so the ROW count is exactly this product)."""
    sh_text = PROFILE_421_LEGS_SH.read_text()
    families = _int_constant(sh_text, r"^MEDIA_FAMILIES=(\d+)", PROFILE_421_LEGS_SH)
    heldout_families = _int_constant(
        sh_text, r"^MEDIA_HELDOUT_FAMILIES=(\d+)", PROFILE_421_LEGS_SH
    )
    steps_m = _int_constant(
        sh_text, r'STEPS_M="\$\{PROFILE_421_STEPS_M:-(\d+)\}"', PROFILE_421_LEGS_SH
    )
    batch = _int_constant(sh_text, r"^BATCH=(\d+)", PROFILE_421_LEGS_SH)
    py_text = AUDIO_CORPUS_PY.read_text()
    instances = _int_constant(
        py_text, r"^_DEFAULT_INSTANCES_PER_FAMILY = (\d+)", AUDIO_CORPUS_PY
    )
    total_files = families * instances
    train_clips = (families - heldout_families) * instances
    m_leg_rows = steps_m * batch
    return total_files, train_clips, m_leg_rows


def render_corpus_pool_note(_artifact: dict) -> str:
    total_files, train_clips, m_leg_rows = corpus_pool_counts()
    body = (
        f"Both media corpus producers emit families × instances = {total_files} files at any "
        f"`--rows`, so the M-leg's {m_leg_rows} rows cycle {train_clips} distinct train clips "
        "(a page-cached working set) — the HTSAT/vision front-end finding is a real per-item "
        "CPU decode/preprocess compute cost, not a realistic-corpus I/O cost."
    )
    return _bullet(body)


def hermetic_test_count() -> int:
    """`test_profile_421_merge.py`'s own hermetic test count -- DISCOVERED
    (`unittest.TestLoader.loadTestsFromModule`), never executed and never
    hand-counted: this only imports the module and counts `Test*` methods,
    so it stays hermetic (no GPU, no subprocess) and moves the moment a
    test is added or removed."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "profile_421_render_docs_test_profile_421_merge", PROFILE_421_MERGE_TEST
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return unittest.TestLoader().loadTestsFromModule(module).countTestCases()


def render_hermetic_test_count_note(_artifact: dict) -> str:
    count = hermetic_test_count()
    body = (
        "**`CONTRACT.md`'s own §D5 \"45 hermetic tests\" figure was already stale at freeze.** "
        f"`profile_421_merge.py`'s hermetic suite had grown to {count} tests by the freeze "
        f"commit (`perf/421-profile-p1` @ aace002f) and stays at {count} at `c1b0b0ba` "
        "(`python3 ci/scripts/perf/test_profile_421_merge.py` → "
        f"\"Ran {count} tests\"); the count was true earlier on `perf/421-profile-p1` but "
        "drifted before the freeze landed. Not corrected in the frozen body (a "
        "pre-registration's text is never edited after freezing — see `CONTRACT.md`'s own "
        "`citations-resolve-at` header note), recorded here instead: the count is descriptive "
        "prose about the suite's size, not a method parameter any decision rule reads, so this "
        "staleness never affected a verdict. `docs/maintainer/fine-tune-performance-guide.md` "
        "and `CHANGELOG.md` both already avoid citing a bare, drifting count for this suite."
    )
    return _bullet(body)


def basename_ambiguity_counts() -> tuple[str, int, int]:
    """`(epoch_sha, layer_norm_rs_count, main_rs_count)` -- LIVE, via
    `check_citations.py`'s own `_ls_tree_paths` (the exact machinery
    `check_citations.py` itself uses to decide whether a bare basename is
    ambiguous in `CONTRACT.md`'s own pinned tree), never a hand-counted
    number that can silently drift the moment either file is added,
    renamed, or removed."""
    text = CONTRACT.read_text()
    epoch_sha, error = _cc._file_citations_epoch(CONTRACT, text)
    if error is not None or epoch_sha is None:
        raise ValueError(f"CONTRACT.md's citations-resolve-at header is missing or malformed: {error}")
    tree = _cc._ls_tree_paths(epoch_sha)
    layer_norm_count = sum(1 for p in tree if p.rsplit("/", 1)[-1] == "layer_norm.rs")
    main_count = sum(1 for p in tree if p.rsplit("/", 1)[-1] == "main.rs")
    return epoch_sha, layer_norm_count, main_count


_NUMBER_WORDS = {1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR", 5: "FIVE", 6: "SIX",
                 7: "SEVEN", 8: "EIGHT", 9: "NINE", 10: "TEN", 11: "ELEVEN", 12: "TWELVE"}


def _spelled_out(n: int) -> str:
    if n not in _NUMBER_WORDS:
        raise ValueError(f"{n} has no spelled-out form registered in _NUMBER_WORDS -- add one")
    return _NUMBER_WORDS[n]


def render_basename_ambiguity_note(_artifact: dict) -> str:
    epoch_sha, layer_norm_count, main_count = basename_ambiguity_counts()
    short_sha = epoch_sha[:8]
    body = (
        "**`CONTRACT.md`'s Scope-facts section carries three bare-basename citations that are "
        f"mechanically AMBIGUOUS at its own pinned epoch (`{short_sha}`), not stale — resolved "
        "through a declared header map, never a frozen-body edit.** `check_citations.py`'s "
        "`docs/plans/66-tower-profile`-scoped citation form resolves a bare "
        "`` `<basename>.rs:<line>` `` by searching the pinned tree for a unique match; at "
        f"`{short_sha}` this repo already has {_spelled_out(layer_norm_count)} `layer_norm.rs` "
        "files (`crates/jammi-encoders/src/layer_norm.rs`, "
        "`crates/jammi-kernels/src/cuda/layer_norm.rs`, "
        "`crates/jammi-kernels/src/ops/layer_norm.rs`) and "
        f"{_spelled_out(main_count)} `main.rs` files across crate/test binaries, so "
        "`` `layer_norm.rs:129, 552-583` `` (Scope facts, para 1) and the two "
        "`` `main.rs:115-223` ``/`` `main.rs:1389-1400` `` citations (Scope facts, para 5) "
        "each resolve to more than one candidate by basename alone. The frozen body is never "
        "edited post-freeze (this same section's own `citations-resolve-at` header note) to "
        "spell them out as full paths — instead `CONTRACT.md`'s HEADER ZONE (never frozen) "
        "carries a second HTML comment, `<!-- citations-basename-map: "
        "layer_norm.rs=crates/jammi-encoders/src/layer_norm.rs; "
        "main.rs=crates/jammi-bench/src/main.rs -->`, naming the two intended targets. "
        "`check_citations.py` resolves each mapped basename to its declared path, validated "
        "against the pinned tree (the path must exist there and its own basename must match "
        "the map key) so the map can only disambiguate a genuine ambiguity, never silently "
        "re-point a citation — see `check_citations.py`'s own module doc for the full "
        "narrowing-not-asserting argument. An unmapped ambiguous basename anywhere else still "
        "fails closed exactly as before."
    )
    return _bullet(body)


# `block_id -> (doc_path, render_fn)` -- every block this script owns,
# across all three docs; still per-block (not a single shared constant) so
# a future block in a new doc is a one-line addition, never a signature
# change.
BLOCKS: dict[str, tuple[Path, Callable[[dict], str]]] = {
    "measured-summary": (README, render_measured_summary),
    "measured-towers-table": (README, render_measured_towers_table),
    "candidate-reasons": (README, render_candidate_reasons),
    "findings": (README, render_findings),
    "realized-gains": (README, render_realized_gains),
    "decision-grade-note": (README, render_decision_grade_note),
    "decline-band-summary": (README, render_decline_band_summary),
    "census-key-root-cause": (README, render_census_key_root_cause_note),
    "htsat-a2-deviation": (README, render_htsat_a2_deviation),
    "corpus-pool-note": (README, render_corpus_pool_note),
    "hermetic-test-count-note": (README, render_hermetic_test_count_note),
    "basename-ambiguity-note": (README, render_basename_ambiguity_note),
    "realized-gains-guide": (GUIDE, render_realized_gains_guide),
    "decline-band-guide": (GUIDE, render_decline_band_guide),
    "findings-guide": (GUIDE, render_findings_guide),
    "changelog-421-entry": (CHANGELOG, render_changelog_421_entry),
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
