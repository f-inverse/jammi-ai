#!/usr/bin/env python3
"""GPU-parity coverage-COMPLETENESS gate — hermetic, static, no build, no GPU.

This is the device-axis analog of the required Postgres/SQLite backend
it-matrix (`crates/jammi-db/tests/it`, tripwired syntactically by
`check_sqlite_isms.py`): where that matrix proves CPU↔GPU-shaped backend
parity is *tested*, this gate proves the parity *test suite itself has no
uncovered cell to hide a divergence in*.

## The escape this closes (esc-028)

ModernBERT×Classification silently produced empty scores on a real A100:
candle's CUDA matmul rejected the classifier's CLS row — a `narrow(seq=0)
.squeeze(1)` 2-D operand whose row stride is `seq·hidden` (not `hidden`) — a
layout its CPU matmul tolerated. `infer`'s per-row annotate semantics
swallowed the resulting `Err` into an empty score set instead of surfacing
it. This was latent because the gated `gpu_capability` suite (`crates/
jammi-ai/tests/gpu_capability/`) covered a handful of (architecture, verb)
cells with **no completeness check** — an uncovered cell cannot fail, so the
absence of a ModernBert×Classification parity test was invisible.

The fix is not "add that one test" (a grep for one known-bad string is
exactly the anti-pattern `check_doc_parity.py`'s docstring warns against);
it is a *property*: every SHIPPED (encoder architecture × GPU-dispatching
inference verb) pair is accounted for by name, in one of three REVIEWED,
in-repo sets — so a new architecture, a new verb, or a newly-discovered
serveable combination cannot silently ship with no GPU-parity accounting.

## The three reconciliation sets

  1. **COVERED** — parsed from a machine-readable `//! gpu-parity-cell:
     <Arch> × <Verb>` doc-comment marker, one per cell, in the module under
     `crates/jammi-ai/tests/gpu_capability/` that actually proves CPU↔GPU
     parity for it. A module that does not carry a marker contributes
     nothing (most of the suite's modules test a different property —
     precision admission, training-loop learning, device-independence for a
     kernel-free fold — not (arch × verb) forward parity; see the modules'
     own doc comments for why each does or does not carry a marker).
  2. **STRUCTURALLY_EXCLUDED** — cells that cannot exist: the served
     architecture never populates the model surface the verb's forward
     dispatch requires (e.g. a vision-only load has no `text_forward()`, so
     every text verb errors "Cannot run text task on a vision-only model";
     `ClipText`'s `forward_hidden` is a hard-coded `Err` — "OpenCLIP text
     encoder does not support forward_hidden (classification / NER are
     BERT-family only)" — so it can never serve Classification/Ner). Each
     entry carries a one-line reason grounded in the dispatch code, reviewed
     at PR time — this is NOT a place to silently wave off a real gap.
  3. **PENDING** — cells that ARE serveable (the code path exists and would
     run today) but have no GPU-parity test yet: tracked debt, each with a
     reason. This is what makes the gate landable now, with teeth: today
     only two cells are COVERED, but every one of the other 34 SHIPPED
     cells is accounted for by name in EXCLUDED or PENDING — none can
     silently fall through the cracks. Closing a PENDING cell means adding
     a `gpu-parity-cell` marker to a real parity test AND deleting the
     PENDING entry in the same PR — the two edits are the closure.

     TODO(follow-up issue): file a "fill the GPU parity matrix" tracking
     issue and knock out PENDING cells incrementally; each closure is a
     `gpu-parity-cell` marker + a live-gpu-tests test, landed with the
     PENDING entry removed in the same diff.

## SHIPPED architectures and verbs

Architectures are parsed from the `AnyEncoder` enum
(`crates/jammi-encoders/src/any.rs`: `Bert`, `DistilBert`, `ModernBert`,
`ClipText` today) UNION a small, reviewed `EXTRA_ARCHITECTURES` list for the
served architectures that dispatch a candle GPU forward but are not
`AnyEncoder` variants — the OpenCLIP vision tower and the HTSAT-CLAP audio
tower, each a distinct `Option<Box<dyn Candle*Forward>>` field on
`CandleModel` (`crates/jammi-ai/src/model/backend/candle.rs`) with its own
forward dispatch, exactly the shape the `AnyEncoder` doc comment's "if it's
a distinct served arch" carve-out describes. Each `EXTRA_ARCHITECTURES`
entry is anchored to a `rust_symbol` that must still appear in the backend
file — a rename or removal fails this gate rather than silently vanishing
the architecture from the matrix (the same anchor-resolution discipline
`check_constitution_anchors.py` runs for the constitution).

Verbs are parsed from `ModelTask::ALL`'s enum body
(`crates/jammi-db/src/model_task.rs`). All six current variants
(`TextEmbedding`, `ImageEmbedding`, `AudioEmbedding`, `Classification`,
`Ner`, `Regression`) dispatch through `CandleModel::forward`'s per-task
match to a real candle GPU forward — none is a CPU-only fold. The CPU-only
folds the retrospective calls out (graph propagation/SGC/APPNP, conformal
calibration, RRF fusion) are not `ModelTask` members at all: they live
entirely in `jammi-ai::pipeline` / `jammi-ai::predict` / `jammi-ai::query`
and never reach `CandleModel::forward`, so they are outside `ModelTask::ALL`
by construction and never enter this matrix — confirmed by reading the
dispatch match, not assumed.

## Fail-closed contract

  - Any SHIPPED (architecture × verb) cell in none of COVERED /
    STRUCTURALLY_EXCLUDED / PENDING is a non-zero exit naming it.
  - Any COVERED / EXCLUDED / PENDING entry naming an architecture or verb
    outside the parsed SHIPPED sets is a non-zero exit (the reviewed lists
    cannot silently rot to reference a renamed/removed identifier).
  - Any cell claimed by more than one of the three sets is a non-zero exit
    (contradictory bookkeeping).
  - Any `EXTRA_ARCHITECTURES` anchor that no longer resolves is a non-zero
    exit.
  - A parse failure (missing enum, no markers found, missing suite dir) is a
    non-zero exit naming what could not be resolved.

Run: `python3 ci/scripts/check_gpu_parity_matrix.py`
Self-test (proves the reconciliation bites — a dropped PENDING entry and a
bogus architecture reference are both caught): `python3 ci/scripts/check_gpu_parity_matrix.py --self-test`
Hermetic: reads only files in the working tree (or synthetic in-memory data
under `--self-test`); no network, no build, no GPU.

## A second, orthogonal completeness axis: SILICON

esc-028's parent principle — an uncovered cell cannot fail — is not specific
to the (architecture × verb) axis above; it applies to any matrix this repo
ships without a completeness check. The #351 GPU-validation review found a
second, independent instance of exactly that shape on the SILICON axis: the
`jammi-kernels` build compiles four real SASS binaries (`sm_80`/`sm_86`/
`sm_89`/`sm_90`, [`GENCODE_ARCHES`] in `crates/jammi-kernels/build.rs`) and
`jammi-ai` ships a `metal` feature, but execution proof (a real GPU run, not
"it compiled") existed for exactly ONE of those five shipped cells — and
because nothing enumerated "shipped silicon" against "silicon with a named
execution lane", that four-out-of-five gap was invisible to every gate in
this repo, the same way the empty ModernBERT×Classification scores were
invisible before this file existed.

The fix is the same shape as above, deliberately: two REVIEWED, in-repo sets
(SHIPPED, parsed from source; SILICON_ACCOUNTING, reviewed by name) that must
reconcile to sameness — never a grep for "sm_80 is proven", which only ever
catches the one string a human already thought to check.

  1. **SHIPPED silicon** — parsed from source, hermetically, no toolchain:
     the `code=sm_XX` suffixes inside the [`GENCODE_ARCHES`] const literal in
     `crates/jammi-kernels/build.rs` (parsed, not hardcoded, so a fifth
     `-gencode` pair trips this gate the moment it lands — the same anchor
     discipline [`EXTRA_ARCHITECTURES`] uses above), plus `"metal"` if
     `crates/jammi-ai/Cargo.toml` declares a `metal = [` feature.
  2. **SILICON_ACCOUNTING** — a reviewed declaration mapping each shipped
     silicon target to EXACTLY ONE of `ProvenBy(<named execution lane>)` (a
     real, recurring, named suite — not "it built once on someone's box") or
     `Deferred(reason, owner, date)` (a conscious, visible deferral, not a
     silent gap).
  3. **Fail-closed contract**: a shipped target with no accounting row, an
     accounting row naming a target no longer shipped (stale), or a target
     accounted for twice are each a non-zero exit naming the cell. Every
     failure states the same principle: "proof surface must equal shipped
     surface — add an execution lane binding or a reviewed deferral row;
     precedent is not an accounting."

Self-test: `python3 ci/scripts/check_gpu_parity_matrix.py --self-test` also
proves the silicon axis REDs on a synthetic shipped-but-unaccounted target
and a synthetic stale accounting row (plus a duplicated row) — the axis-1
self-test above is untouched by this addition.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ANY_ENCODER = REPO_ROOT / "crates" / "jammi-encoders" / "src" / "any.rs"
MODEL_TASK = REPO_ROOT / "crates" / "jammi-db" / "src" / "model_task.rs"
CANDLE_BACKEND = REPO_ROOT / "crates" / "jammi-ai" / "src" / "model" / "backend" / "candle.rs"
GPU_CAPABILITY_DIR = REPO_ROOT / "crates" / "jammi-ai" / "tests" / "gpu_capability"
KERNELS_BUILD_RS = REPO_ROOT / "crates" / "jammi-kernels" / "build.rs"
JAMMI_AI_CARGO_TOML = REPO_ROOT / "crates" / "jammi-ai" / "Cargo.toml"


class MatrixError(Exception):
    """Uncomputable input (parse failure, unresolved anchor) — fails closed."""


@dataclass(frozen=True)
class Cell:
    """One (architecture × verb) parity cell."""

    arch: str
    verb: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.arch} × {self.verb}"


@dataclass(frozen=True)
class ExtraArchitecture:
    """A served architecture that dispatches a candle GPU forward but is not
    an `AnyEncoder` variant — anchored to a `rust_symbol` so a rename or
    removal fails this gate instead of silently dropping the architecture
    from the matrix (mirrors `check_constitution_anchors.py`'s `rust_symbol`
    anchor kind).
    """

    name: str
    anchor_file: Path
    anchor_symbol: str


EXTRA_ARCHITECTURES = [
    ExtraArchitecture(
        "OpenClipVision", CANDLE_BACKEND, "OpenClipVisionTransformer"
    ),
    ExtraArchitecture("HtsatAudio", CANDLE_BACKEND, "HtsatAudio"),
]


# --------------------------------------------------------------------------- #
# SILICON axis — see the module docstring's "A second, orthogonal
# completeness axis: SILICON" section for the full rationale (esc-028 +
# the #351 GPU-validation review). Each shipped silicon target reconciles to
# EXACTLY ONE of ProvenBy / Deferred below, reviewed at PR time exactly like
# STRUCTURALLY_EXCLUDED / PENDING above.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProvenBy:
    """A silicon target with a real, named, recurring execution lane."""

    lane: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"proven_by: {self.lane}"


@dataclass(frozen=True)
class Deferred:
    """A silicon target with a conscious, visible deferral instead of proof."""

    reason: str
    owner: str
    date: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"deferred: reason={self.reason!r}, owner={self.owner}, date={self.date}"


SiliconAccountingEntry = ProvenBy | Deferred

# A reviewed list (not a dict) so a duplicated target is representable and
# this gate's own reconciliation catches it, rather than Python's dict
# literal semantics silently keeping only the last entry.
#
# sm_80/sm_86/sm_89/sm_90 are each proven by their own matrix leg of
# `gpu-prove.yml` (opt-in-PR/nightly/dispatch) and the reusable release gate
# `_gpu-prove-gate.yml` (every CUDA artifact promotion) — both drive
# `ci/scripts/runpod_gpu_prove.sh` with `GPU_PROVE_ARCH=<arch>`, which rents
# the matching device via `rp_deploy_arch` (runpod_lib.sh) and runs the same
# gated GPU suites (`grpc_embedding_gpu`, `gpu_capability`) real hardware ran
# for sm_80 alone before the #351 GPU-validation review widened this to all
# four shipped SASS targets.
SILICON_ACCOUNTING: list[tuple[str, SiliconAccountingEntry]] = [
    (
        "sm_80",
        ProvenBy(
            "gpu-prove.yml / _gpu-prove-gate.yml matrix leg sm_80 "
            "(ci/scripts/runpod_gpu_prove.sh GPU_PROVE_ARCH=sm_80 on a rented A100)"
        ),
    ),
    (
        "sm_86",
        ProvenBy(
            "gpu-prove.yml / _gpu-prove-gate.yml matrix leg sm_86 "
            "(ci/scripts/runpod_gpu_prove.sh GPU_PROVE_ARCH=sm_86 on a rented A40)"
        ),
    ),
    (
        "sm_89",
        ProvenBy(
            "gpu-prove.yml / _gpu-prove-gate.yml matrix leg sm_89 "
            "(ci/scripts/runpod_gpu_prove.sh GPU_PROVE_ARCH=sm_89 on a rented L40S)"
        ),
    ),
    (
        "sm_90",
        ProvenBy(
            "gpu-prove.yml / _gpu-prove-gate.yml matrix leg sm_90 "
            "(ci/scripts/runpod_gpu_prove.sh GPU_PROVE_ARCH=sm_90 on a rented H100)"
        ),
    ),
    (
        "metal",
        Deferred(
            reason=(
                "macos Metal execution lane pending the metal-gated oracles landing "
                "on feat/351-quantized-gguf-qlora (#430); flip when that lane exists"
            ),
            owner="maintainers",
            date="2026-08-31",
        ),
    ),
]


# --------------------------------------------------------------------------- #
# STRUCTURALLY_EXCLUDED — cells that cannot exist. Each reason is grounded in
# the dispatch code (`crates/jammi-ai/src/model/backend/candle.rs`) or the
# encoder's own doc comment (`crates/jammi-encoders/src/any.rs`,
# `crates/jammi-ai/src/model/backend/open_clip_text.rs`), reviewed at PR time.
# --------------------------------------------------------------------------- #
STRUCTURALLY_EXCLUDED: dict[Cell, str] = {
    # Text-only BERT-family architectures never populate `CandleModel::vision`
    # or `CandleModel::audio` — `forward_image_embedding` / `forward_audio_embedding`
    # error "No vision model loaded" / "No audio model loaded" for them.
    Cell("Bert", "ImageEmbedding"): "Bert is text-only; `forward_image_embedding` requires a loaded `vision` tower, which a text load never populates.",
    Cell("Bert", "AudioEmbedding"): "Bert is text-only; `forward_audio_embedding` requires a loaded `audio` tower, which a text load never populates.",
    Cell("DistilBert", "ImageEmbedding"): "DistilBert is text-only; `forward_image_embedding` requires a loaded `vision` tower, which a text load never populates.",
    Cell("DistilBert", "AudioEmbedding"): "DistilBert is text-only; `forward_audio_embedding` requires a loaded `audio` tower, which a text load never populates.",
    Cell("ModernBert", "ImageEmbedding"): "ModernBert is text-only; `forward_image_embedding` requires a loaded `vision` tower, which a text load never populates.",
    Cell("ModernBert", "AudioEmbedding"): "ModernBert is text-only; `forward_audio_embedding` requires a loaded `audio` tower, which a text load never populates.",
    Cell("ClipText", "ImageEmbedding"): "ClipText is the OpenCLIP TEXT tower; on an OpenCLIP model (`is_open_clip` co-loads BOTH towers, candle.rs:1520-1522) image embedding is served by the co-loaded vision tower and is accounted under `OpenClipVision × ImageEmbedding` — the ClipText row covers only text-tower verbs.",
    Cell("ClipText", "AudioEmbedding"): "An OpenCLIP model loads text+vision but no `audio` tower (`is_open_clip` sets `audio=None`, candle.rs:1523), so `forward_audio_embedding` errors \"No audio model loaded\".",
    # ClipText's `OpenClipTextForward::forward_hidden` is a hard-coded `Err`
    # ("OpenCLIP text encoder does not support forward_hidden (classification
    # / NER are BERT-family only)"); `forward_classification` / `forward_ner`
    # both call `text_forward()?.forward_hidden(..)`, so they cannot serve it.
    Cell("ClipText", "Classification"): "OpenClipTextForward::forward_hidden is a hard-coded Err (\"...classification / NER are BERT-family only\"); forward_classification needs per-token hidden states ClipText never exposes.",
    Cell("ClipText", "Ner"): "OpenClipTextForward::forward_hidden is a hard-coded Err (\"...classification / NER are BERT-family only\"); forward_ner needs per-token hidden states ClipText never exposes.",
    # OpenClipVision is vision-only: `CandleModel::text` is never populated,
    # so `text_forward()` — which every text verb routes through — errors
    # "Cannot run text task on a vision-only model".
    Cell("OpenClipVision", "TextEmbedding"): "An OpenCLIP model's text embedding is served by its co-loaded ClipText text tower (`is_open_clip` co-loads both towers, candle.rs:1520-1522) and is accounted under `ClipText × TextEmbedding`; the OpenClipVision (vision-tower) row covers only `ImageEmbedding`.",
    Cell("OpenClipVision", "AudioEmbedding"): "An OpenCLIP model loads text+vision but no `audio` tower (`is_open_clip` sets `audio=None`, candle.rs:1523), so `forward_audio_embedding` errors \"No audio model loaded\".",
    Cell("OpenClipVision", "Classification"): "Classification on an OpenCLIP model routes through the text tower, whose `OpenClipTextForward::forward_hidden` is a hard Err (open_clip_text.rs) — non-serveable, accounted under `ClipText × Classification`; the vision-tower row covers only `ImageEmbedding`.",
    Cell("OpenClipVision", "Ner"): "NER on an OpenCLIP model routes through the text tower, whose `OpenClipTextForward::forward_hidden` is a hard Err (open_clip_text.rs) — non-serveable, accounted under `ClipText × Ner`; the vision-tower row covers only `ImageEmbedding`.",
    Cell("OpenClipVision", "Regression"): "Regression on an OpenCLIP model is served by the co-loaded ClipText text tower's `forward_pooled` and is accounted under `ClipText × Regression`; the OpenClipVision (vision-tower) row covers only `ImageEmbedding`.",
    # HtsatAudio is audio-only: `CandleModel::text` is never populated either,
    # so the same `text_forward()` guard rejects every text verb; it has no
    # `vision` tower for ImageEmbedding.
    Cell("HtsatAudio", "TextEmbedding"): "HtsatAudio is audio-only; forward_embedding (TextEmbedding) calls text_forward(), which errors \"Cannot run text task on a vision-only model\" (the same text_forward() guard rejects every non-text load).",
    Cell("HtsatAudio", "ImageEmbedding"): "HtsatAudio is audio-only; forward_image_embedding requires a loaded `vision` tower, which an audio-only load never populates.",
    Cell("HtsatAudio", "Classification"): "HtsatAudio is audio-only; forward_classification calls text_forward(), which errors \"Cannot run text task on a vision-only model\".",
    Cell("HtsatAudio", "Ner"): "HtsatAudio is audio-only; forward_ner calls text_forward(), which errors \"Cannot run text task on a vision-only model\".",
    Cell("HtsatAudio", "Regression"): "HtsatAudio is audio-only; forward_regression calls text_forward(), which errors \"Cannot run text task on a vision-only model\".",
}


# --------------------------------------------------------------------------- #
# PENDING — serveable today, no GPU-parity test yet. Tracked debt; each entry
# is deleted the same PR that adds the closing `gpu-parity-cell` marker.
# --------------------------------------------------------------------------- #
_PENDING_REASON = (
    "shipped and serveable (the dispatch path exists and would run today); "
    "no GPU-parity test in crates/jammi-ai/tests/gpu_capability/ yet — "
    "tracked debt, see the 'fill the GPU parity matrix' TODO at the top of "
    "this file."
)

PENDING: dict[Cell, str] = {
    # BERT-family text embedding: only Bert×TextEmbedding is proven today
    # (embeddings_parity.rs); DistilBert and ModernBert share the same
    # generic `forward_embedding` path and are equally serveable.
    Cell("DistilBert", "TextEmbedding"): _PENDING_REASON,
    # Classification: only ModernBert×Classification is proven today
    # (classification_parity.rs, the esc-028 regression guard); Bert and
    # DistilBert have their own `*ClassificationForward` wrapper and are
    # equally serveable, and equally capable of hitting the same class of
    # CUDA strided-operand rejection.
    Cell("Bert", "Classification"): _PENDING_REASON,
    Cell("DistilBert", "Classification"): _PENDING_REASON,
    # NER: generic over `forward_hidden` for all three BERT-family
    # architectures (no per-arch wrapper family the way Classification has);
    # none is proven on GPU yet. cookbook/fixtures/tiny_modernbert_ner exists.
    Cell("Bert", "Ner"): _PENDING_REASON,
    Cell("DistilBert", "Ner"): _PENDING_REASON,
    # Regression: generic over `forward_pooled` for every text architecture,
    # including ClipText (whose `forward_pooled` override succeeds — only
    # `forward_hidden` is rejected); none is proven on GPU yet.
    Cell("Bert", "Regression"): _PENDING_REASON,
    Cell("DistilBert", "Regression"): _PENDING_REASON,
    Cell("ModernBert", "Regression"): _PENDING_REASON,
    Cell("ClipText", "Regression"): _PENDING_REASON + " (frozen-backbone ProjectionHead-style fine-tune is structurally supported by the trainer; forward_pooled succeeds for ClipText.)",
    # The two non-BERT served architectures: each has exactly one servable
    # verb, and neither is proven on GPU yet.
}


# --------------------------------------------------------------------------- #
# enum-variant parsing (mirrors `check_doc_parity.py`'s `parse_enum_variants`
# exactly — the same brace-balanced, depth-1, comment-stripped algorithm — so
# a struct/tuple/unit variant enum is parsed identically here; kept as an
# independent copy so this gate has no cross-script import dependency, the
# same standalone-script convention `check_sqlite_isms.py` follows).
# --------------------------------------------------------------------------- #
def _strip_rust_comments(text: str) -> str:
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def parse_enum_variants(source: str, enum_name: str) -> list[str]:
    body = _strip_rust_comments(source)
    match = re.search(rf"\benum\s+{re.escape(enum_name)}\b", body)
    if match is None:
        raise MatrixError(f"enum `{enum_name}` not found in source")

    open_idx = body.find("{", match.end())
    if open_idx == -1:
        raise MatrixError(f"enum `{enum_name}` has no opening brace")

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
        raise MatrixError(f"enum `{enum_name}` body is not brace-balanced")

    depth1 = "".join(depth1_chars)
    variants: list[str] = []
    for segment in depth1.split(","):
        cleaned = "\n".join(
            line for line in segment.splitlines() if not line.strip().startswith("#")
        )
        ident = re.search(r"[A-Za-z_][A-Za-z0-9_]*", cleaned)
        if ident:
            variants.append(ident.group(0))
    if not variants:
        raise MatrixError(f"enum `{enum_name}` parsed to zero variants")
    return variants


# --------------------------------------------------------------------------- #
# COVERED — parsed from `//! gpu-parity-cell: <Arch> × <Verb>` markers.
# --------------------------------------------------------------------------- #
MARKER_RE = re.compile(
    r"^\s*//!\s*gpu-parity-cell:\s*(?P<arch>[A-Za-z0-9_]+)\s*×\s*(?P<verb>[A-Za-z0-9_]+)\s*$"
)


def load_covered() -> dict[Cell, list[str]]:
    if not GPU_CAPABILITY_DIR.is_dir():
        raise MatrixError(f"gpu_capability suite dir not found: {GPU_CAPABILITY_DIR}")
    covered: dict[Cell, list[str]] = {}
    for path in sorted(GPU_CAPABILITY_DIR.glob("*.rs")):
        rel = str(path.relative_to(REPO_ROOT))
        for line in path.read_text(encoding="utf-8").splitlines():
            m = MARKER_RE.match(line)
            if m:
                cell = Cell(m.group("arch"), m.group("verb"))
                covered.setdefault(cell, []).append(rel)
    if not covered:
        raise MatrixError(
            f"no `gpu-parity-cell:` markers found under {GPU_CAPABILITY_DIR} "
            "— did the marker format change, or did every marker get deleted?"
        )
    return covered


def load_shipped() -> tuple[set[str], set[str]]:
    if not ANY_ENCODER.is_file():
        raise MatrixError(f"AnyEncoder source not found: {ANY_ENCODER}")
    if not MODEL_TASK.is_file():
        raise MatrixError(f"ModelTask source not found: {MODEL_TASK}")
    if not CANDLE_BACKEND.is_file():
        raise MatrixError(f"candle backend source not found: {CANDLE_BACKEND}")

    architectures = set(parse_enum_variants(ANY_ENCODER.read_text(encoding="utf-8"), "AnyEncoder"))
    verbs = set(parse_enum_variants(MODEL_TASK.read_text(encoding="utf-8"), "ModelTask"))

    candle_src = CANDLE_BACKEND.read_text(encoding="utf-8")
    for extra in EXTRA_ARCHITECTURES:
        if extra.anchor_symbol not in candle_src:
            raise MatrixError(
                f"EXTRA_ARCHITECTURES entry `{extra.name}` anchors to rust_symbol "
                f"`{extra.anchor_symbol}` in {extra.anchor_file.relative_to(REPO_ROOT)}, "
                "which no longer resolves — the served architecture may have been "
                "renamed or removed; update EXTRA_ARCHITECTURES."
            )
        architectures.add(extra.name)

    return architectures, verbs


# --------------------------------------------------------------------------- #
# SILICON — SHIPPED-side parsing. CUDA archs come from the `code=sm_XX`
# suffixes inside the [`GENCODE_ARCHES`] const literal in `build.rs` (comments
# stripped first, the same discipline `parse_enum_variants` uses, so the
# doc-comment's own worked example — "adding `arch=compute_100,code=sm_100`
# to GENCODE_ARCHES" — is never mistaken for a real shipped entry). Metal is a
# textual probe for the `metal = [` feature declaration.
# --------------------------------------------------------------------------- #
GENCODE_ARCHES_DECL_RE = re.compile(
    r"const\s+GENCODE_ARCHES\s*:.*?;", re.DOTALL
)
GENCODE_SM_RE = re.compile(r"code=sm_(\d+)")
METAL_FEATURE_RE = re.compile(r"^metal\s*=\s*\[", re.MULTILINE)


def load_shipped_cuda_silicon() -> set[str]:
    if not KERNELS_BUILD_RS.is_file():
        raise MatrixError(f"jammi-kernels build script not found: {KERNELS_BUILD_RS}")
    stripped = _strip_rust_comments(KERNELS_BUILD_RS.read_text(encoding="utf-8"))
    decl = GENCODE_ARCHES_DECL_RE.search(stripped)
    if decl is None:
        raise MatrixError(
            "GENCODE_ARCHES const declaration not found in "
            f"{KERNELS_BUILD_RS.relative_to(REPO_ROOT)} — was it renamed or removed?"
        )
    sms = GENCODE_SM_RE.findall(decl.group(0))
    if not sms:
        raise MatrixError(
            "GENCODE_ARCHES const declaration parsed to zero `code=sm_XX` gencode entries"
        )
    return {f"sm_{d}" for d in sms}


def load_shipped_metal_silicon() -> set[str]:
    if not JAMMI_AI_CARGO_TOML.is_file():
        raise MatrixError(f"jammi-ai Cargo.toml not found: {JAMMI_AI_CARGO_TOML}")
    text = JAMMI_AI_CARGO_TOML.read_text(encoding="utf-8")
    if METAL_FEATURE_RE.search(text):
        return {"metal"}
    return set()


def load_shipped_silicon() -> set[str]:
    return load_shipped_cuda_silicon() | load_shipped_metal_silicon()


SILICON_FAIL_PRINCIPLE = (
    "proof surface must equal shipped surface — add an execution lane binding "
    "or a reviewed deferral row; precedent is not an accounting."
)


def reconcile_silicon(
    shipped: set[str],
    accounting: list[tuple[str, SiliconAccountingEntry]],
) -> list[str]:
    failures: list[str] = []

    seen: dict[str, SiliconAccountingEntry] = {}
    for target, entry in accounting:
        if target in seen:
            failures.append(
                f"silicon target `{target}` appears twice in SILICON_ACCOUNTING — "
                f"pick one accounting row per target. {SILICON_FAIL_PRINCIPLE}"
            )
            continue
        seen[target] = entry

    accounted = set(seen)
    for target in sorted(shipped - accounted):
        failures.append(
            f"silicon target `{target}` is SHIPPED but has no SILICON_ACCOUNTING row. "
            f"{SILICON_FAIL_PRINCIPLE}"
        )
    for target in sorted(accounted - shipped):
        failures.append(
            f"SILICON_ACCOUNTING entry `{target}` names a target no longer shipped "
            f"(stale) — remove it or the source it was parsed from changed. {SILICON_FAIL_PRINCIPLE}"
        )

    return failures


def print_silicon_matrix(
    shipped: set[str],
    accounting: list[tuple[str, SiliconAccountingEntry]],
) -> None:
    by_target = dict(accounting)
    print("\nGPU-silicon matrix (shipped target -> accounting):")
    for target in sorted(shipped):
        entry = by_target.get(target)
        if entry is None:
            print(f"    !!!! UNACCOUNTED !!!! {target}")
        else:
            print(f"    {target:<8} {entry}")
    print(
        f"\nSilicon summary: {len(shipped)} SHIPPED target(s), "
        f"{len(accounting)} SILICON_ACCOUNTING row(s)."
    )


# --------------------------------------------------------------------------- #
# reconciliation — pure function over parsed/declared data, so `--self-test`
# can drive it with synthetic inputs and prove it actually bites.
# --------------------------------------------------------------------------- #
def reconcile(
    architectures: set[str],
    verbs: set[str],
    covered: dict[Cell, list[str]],
    excluded: dict[Cell, str],
    pending: dict[Cell, str],
) -> list[str]:
    failures: list[str] = []

    def check_refs(cells, label: str) -> None:
        for c in cells:
            if c.arch not in architectures:
                failures.append(
                    f"{label} entry {c} references unknown architecture `{c.arch}` "
                    "— not an AnyEncoder variant nor an EXTRA_ARCHITECTURES entry"
                )
            if c.verb not in verbs:
                failures.append(
                    f"{label} entry {c} references unknown verb `{c.verb}` "
                    "— not a ModelTask variant"
                )

    check_refs(covered.keys(), "COVERED")
    check_refs(excluded.keys(), "STRUCTURALLY_EXCLUDED")
    check_refs(pending.keys(), "PENDING")

    covered_set, excluded_set, pending_set = set(covered), set(excluded), set(pending)
    for c in sorted(covered_set & excluded_set, key=str):
        failures.append(f"cell {c} is claimed both COVERED and STRUCTURALLY_EXCLUDED — contradictory bookkeeping")
    for c in sorted(covered_set & pending_set, key=str):
        failures.append(f"cell {c} is claimed both COVERED and PENDING — remove the stale PENDING entry")
    for c in sorted(excluded_set & pending_set, key=str):
        failures.append(f"cell {c} is claimed both STRUCTURALLY_EXCLUDED and PENDING — pick one")

    shipped = {Cell(a, v) for a in architectures for v in verbs}
    accounted = covered_set | excluded_set | pending_set
    for c in sorted(shipped - accounted, key=str):
        failures.append(
            f"cell {c} is SHIPPED but appears in none of COVERED / STRUCTURALLY_EXCLUDED "
            "/ PENDING — add a `gpu-parity-cell` marker, a STRUCTURALLY_EXCLUDED entry "
            "with a grounded reason, or a PENDING entry with a reason."
        )

    return failures


def print_matrix(
    architectures: set[str],
    verbs: set[str],
    covered: dict[Cell, list[str]],
    excluded: dict[Cell, str],
    pending: dict[Cell, str],
) -> None:
    print("GPU-parity matrix (architecture × verb):")
    for arch in sorted(architectures):
        for verb in sorted(verbs):
            cell = Cell(arch, verb)
            if cell in covered:
                files = ", ".join(covered[cell])
                print(f"    COVERED             {cell}  <- {files}")
            elif cell in excluded:
                print(f"    STRUCTURALLY_EXCL   {cell}  — {excluded[cell]}")
            elif cell in pending:
                print(f"    PENDING             {cell}")
            else:
                print(f"    !!!! UNACCOUNTED !!!! {cell}")

    print(
        f"\nSummary: {len(covered)} COVERED, {len(excluded)} STRUCTURALLY_EXCLUDED, "
        f"{len(pending)} PENDING out of {len(architectures) * len(verbs)} SHIPPED cells."
    )
    if pending:
        print(f"\nPENDING debt ({len(pending)} cell(s) — never silent, tracked here):")
        for cell in sorted(pending, key=str):
            print(f"    {cell}")


# --------------------------------------------------------------------------- #
# self-test — proves the reconciliation actually bites, on synthetic data.
# --------------------------------------------------------------------------- #
def self_test() -> int:
    failures: list[str] = []

    archs = {"Alpha", "Beta"}
    verbs = {"Foo", "Bar"}
    covered = {Cell("Alpha", "Foo"): ["<synthetic>"]}
    excluded = {Cell("Beta", "Bar"): "synthetic structural exclusion"}
    pending = {
        Cell("Alpha", "Bar"): "synthetic pending debt",
        Cell("Beta", "Foo"): "synthetic pending debt",
    }

    clean = reconcile(archs, verbs, covered, excluded, pending)
    if clean:
        failures.append(f"self-test FAILED: a fully-accounted 2x2 matrix reported failures: {clean}")

    # Drop a PENDING entry — the now-unaccounted cell must be reported. This
    # is the "removing one PENDING entry makes it FAIL" acceptance check.
    pending_missing = dict(pending)
    del pending_missing[Cell("Alpha", "Bar")]
    dropped = reconcile(archs, verbs, covered, excluded, pending_missing)
    if not any("Alpha × Bar" in f and "SHIPPED but appears in none" in f for f in dropped):
        failures.append(f"self-test FAILED: dropping a PENDING entry did not surface the now-uncovered cell: {dropped}")

    # A COVERED/EXCLUDED/PENDING entry naming an architecture outside the
    # SHIPPED set — the "adding a fake new arch" acceptance check — must be
    # rejected as an unknown-architecture reference.
    covered_bogus = dict(covered)
    covered_bogus[Cell("Gamma", "Foo")] = ["<synthetic>"]
    bogus = reconcile(archs, verbs, covered_bogus, excluded, pending)
    if not any("unknown architecture `Gamma`" in f for f in bogus):
        failures.append(f"self-test FAILED: a bogus architecture reference was not rejected: {bogus}")

    # A cell claimed by two sets at once must be rejected.
    excluded_dup = dict(excluded)
    excluded_dup[Cell("Alpha", "Foo")] = "synthetic duplicate"
    dup = reconcile(archs, verbs, covered, excluded_dup, pending)
    if not any("claimed both COVERED and STRUCTURALLY_EXCLUDED" in f for f in dup):
        failures.append(f"self-test FAILED: a doubly-claimed cell was not rejected: {dup}")

    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        print("gpu-parity-matrix self-test: FAIL", file=sys.stderr)
        return 1
    print(
        "gpu-parity-matrix self-test: OK — a dropped PENDING entry, a bogus "
        "architecture reference, and a doubly-claimed cell are all caught."
    )
    return 0


# --------------------------------------------------------------------------- #
# SILICON self-test — proves the SILICON axis's reconciliation actually
# bites, on synthetic data. Independent of `self_test()` above; the axis-1
# self-test is untouched by this function's existence.
# --------------------------------------------------------------------------- #
def self_test_silicon() -> int:
    failures: list[str] = []

    shipped = {"sm_80", "sm_86"}
    accounting: list[tuple[str, SiliconAccountingEntry]] = [
        ("sm_80", ProvenBy("synthetic gpu-prove lane")),
        ("sm_86", Deferred(reason="synthetic deferral", owner="maintainers", date="2026-08-31")),
    ]

    clean = reconcile_silicon(shipped, accounting)
    if clean:
        failures.append(
            f"silicon self-test FAILED: a fully-accounted shipped set reported failures: {clean}"
        )

    # (a) synthetic shipped-but-unaccounted arch — a new arch ships with no
    # accounting row and must be surfaced by name.
    shipped_extra = shipped | {"sm_100"}
    unaccounted = reconcile_silicon(shipped_extra, accounting)
    if not any(
        "sm_100" in f and "has no SILICON_ACCOUNTING row" in f for f in unaccounted
    ):
        failures.append(
            f"silicon self-test FAILED: a shipped-but-unaccounted silicon target was not "
            f"surfaced: {unaccounted}"
        )

    # (b) stale accounting row — an accounting entry naming a target that is
    # no longer shipped must be surfaced by name.
    accounting_stale = accounting + [
        ("sm_120", Deferred(reason="synthetic stale row", owner="maintainers", date="2026-08-31"))
    ]
    stale = reconcile_silicon(shipped, accounting_stale)
    if not any("sm_120" in f and "no longer shipped" in f for f in stale):
        failures.append(
            f"silicon self-test FAILED: a stale accounting row was not surfaced: {stale}"
        )

    # A target accounted for twice must be rejected.
    accounting_dup = accounting + [
        ("sm_80", Deferred(reason="synthetic duplicate", owner="maintainers", date="2026-08-31"))
    ]
    dup = reconcile_silicon(shipped, accounting_dup)
    if not any("appears twice" in f for f in dup):
        failures.append(
            f"silicon self-test FAILED: a duplicated accounting row was not surfaced: {dup}"
        )

    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        print("gpu-silicon-matrix self-test: FAIL", file=sys.stderr)
        return 1
    print(
        "gpu-silicon-matrix self-test: OK — a shipped-but-unaccounted silicon target, a "
        "stale accounting row, and a duplicated accounting row are all caught."
    )
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        rc1 = self_test()
        rc2 = self_test_silicon()
        return 1 if (rc1 != 0 or rc2 != 0) else 0

    try:
        architectures, verbs = load_shipped()
        covered = load_covered()
    except MatrixError as exc:
        print(f"gpu-parity-matrix: FAIL (uncomputable) — {exc}", file=sys.stderr)
        return 1

    failures = reconcile(architectures, verbs, covered, STRUCTURALLY_EXCLUDED, PENDING)

    print_matrix(architectures, verbs, covered, STRUCTURALLY_EXCLUDED, PENDING)

    if failures:
        print("\ngpu-parity-matrix: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print(
            f"\ngpu-parity-matrix: {len(failures)} finding(s) — a shipped (architecture × "
            "verb) cell has no GPU-parity accounting, or the reviewed lists have rotted. "
            "See above.",
            file=sys.stderr,
        )
        return 1

    print(
        "\ngpu-parity-matrix: PASS — every SHIPPED (architecture × verb) cell is "
        "COVERED, STRUCTURALLY_EXCLUDED, or PENDING; no cell can hide a silent divergence."
    )

    try:
        shipped_silicon = load_shipped_silicon()
    except MatrixError as exc:
        print(f"gpu-silicon-matrix: FAIL (uncomputable) — {exc}", file=sys.stderr)
        return 1

    silicon_failures = reconcile_silicon(shipped_silicon, SILICON_ACCOUNTING)

    print_silicon_matrix(shipped_silicon, SILICON_ACCOUNTING)

    if silicon_failures:
        print("\ngpu-silicon-matrix: FAIL", file=sys.stderr)
        for f in silicon_failures:
            print(f"  - {f}", file=sys.stderr)
        print(
            f"\ngpu-silicon-matrix: {len(silicon_failures)} finding(s) — a shipped silicon "
            "target has no accounting, or SILICON_ACCOUNTING has rotted. See above.",
            file=sys.stderr,
        )
        return 1

    print(
        "\ngpu-silicon-matrix: PASS — every SHIPPED silicon target is proven or "
        "consciously deferred by name; no cell can hide a silent gap."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
