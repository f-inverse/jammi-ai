# Wave A — Extended RaBitQ B-bit codes

**Status:** ready to execute (non-contingent — next in the storage line).
**Depends on:** the asymmetric-binary base substrate (two-phase build, companion artifact,
bootstrap-CI gate).
**Risk:** Medium. Storage-side, model-agnostic, ownable end-to-end.

## Goal

A new `StoragePrecision` family giving **arbitrary B-bit scalar codes** (2-bit, 3-bit, …)
with RaBitQ's unbiased, asymptotically-optimal distance estimator — sitting alongside
int8/binary, hitting high recall at *less or no* rescore. This modernizes the storage axis
past int8/binary AND is the **target Wave B (QAMA) co-designs a model to quantize into**, so
it comes first.

## Grounding (cited — do not design from memory)

- **Extended RaBitQ** — arxiv 2409.09913 (SIGMOD'25) + the inventor's writeup
  (dev.to/gaoj0017/extended-rabitq...). Generalizes 1-bit RaBitQ to arbitrary B-bit with an
  unbiased estimator (Alon–Klartag). Paper: **>95% recall @~6.4× and >99% @~4.5×
  compression WITHOUT raw-vector rescore**; wins at moderate 8×/4× too.
- **SAQ** — arxiv 2509.12086 (SIGMOD'25). PCA-segment + DP bit-allocation + coordinate-descent
  refinement: up to **80% lower quantization error and ~80× faster encoding** than Extended
  RaBitQ. A fast-encode follow-on that makes re-index-per-fine-tune cheap.
- Competitive context: Qdrant ships RaBitQ + sub-byte 2/1.5-bit; Weaviate ships rotational
  quant. This is parity-turning-into-lead, not greenfield novelty.

## THE TRAP (read this twice)

RaBitQ's headline "orders of magnitude better" refers to **distance-ESTIMATION error, NOT
end-to-end retrieval recall.** Do not conflate them, and do not let the gate assert
estimation error. **The deliverable is a first-party END-TO-END recall benchmark** on
jammi's own corpora. Also: the deep-research adversarial pass **REFUTED** a blanket "PQ gives
<80% recall at ≥32×" claim — so do NOT dismiss product/optimized quantization (PQ/OPQ); it
must be in the A/B as a real baseline, not assumed inferior.

## Decisive experiment FIRST (GO/NO-GO before touching the production sidecar)

Prototype RaBitQ B-bit encode + estimator in numpy/python on **real ModernBERT-base
embeddings** (reuse the corpus already extracted at
`feat/qat-wave2-joint-abandoned:cookbook/book/artifacts/qat/baseline`, and the recall harness
from `cookbook/book/scripts/build_asymmetric_binary_cache.py`). Measure **end-to-end
recall@k** (k∈{1,10,100}, no-rescore AND retrieve-then-rescore) for:
- RaBitQ at B ∈ {2, 3, 4} bits/dim,
- vs the existing int8 and asymmetric-binary paths,
- vs a PQ/OPQ baseline (do not skip — the PQ-dismissal was refuted),
- vs f32 exact (the ceiling),
at matched **bytes-per-vector** (compression-controlled) AND at matched recall (bytes-controlled).

**GO** if RaBitQ B-bit gives a genuinely better recall/byte frontier than int8 and binary on
real embeddings — e.g. matches int8 recall at meaningfully fewer bytes, or beats binary
recall at similar bytes, *at the no-rescore coarse stage* (where the compression win lives;
rescore recovers everything anyway). **NO-GO / re-scope** if the frontier isn't better than
what int8+asymmetric-binary already give on jammi's data — report honestly and stop.

## Design (only after GO)

Reuse the base substrate's exact patterns:
- New `StoragePrecision::RaBitQ { bits }` (config.rs), Hamming/asymmetric-distance estimator
  in the sidecar coarse path. The RaBitQ **codebook/rotation** (and any per-vector scale) is
  a **companion artifact** fitted at `build()` — the same two-phase (accumulate → fit →
  pack) lifecycle and companion-save/load the τ threshold already uses. Fit on a bounded
  corpus sample (≤~100k rows), as τ does.
- **Rescore stays untouched** — the `.rawf32` companion remains the original f32 vectors;
  retrieve-then-rescore recomputes exact cosine on originals, byte-identical with/without the
  codebook. Only the coarse candidate set changes. (This is the load-bearing K4 property; the
  base's `rescore_is_byte_identical...` test is the template.)
- ANN manifest version bump (monotonic sidecar file-format field, NOT a catalog migration).
- Extend the **bootstrap-CI recall gate** to the RaBitQ precision (multi-seed / CI lower
  bound clears a committed floor). The committed measured recall floor on a B-bit code **is
  the moat artifact.**

## Invariants / gates

- **K4 embedded⇄remote byte-parity** — one `SidecarIndex`, codebook applied identically at
  build + query, companion saved/loaded on both surfaces via `sidecar_extensions`. (Same
  proof shape the τ companion already satisfies.)
- No wire/proto/public-API change — internal storage only.
- Cookbook chapter (nightly render leg) measuring the RaBitQ recall/byte frontier at
  ModernBERT scale, honest about where it wins.
- Fail-closed: merges only if the scale recall gate clears.

## Open questions (resolve during the wave)

- Which variant to own: base Extended RaBitQ vs add SAQ's PCA-segment + DP allocation. Start
  with Extended RaBitQ; add SAQ only if encoding cost is a real index-build cliff.
- Rotation cost at index build (SVD/Procrustes at D=768 over the sample) — bound it like the
  τ fit; freeze the companion + assert recall, not byte-equal codebook (BLAS nondeterminism).
- Does RaBitQ dominate PQ/OPQ across jammi's actual operating rates, or is there a crossover?
  The A/B answers this on first-party data.
- Interaction with Matryoshka truncation (a RaBitQ code of a truncated-then-renormed prefix)
  — likely a per-served-width index like binary; confirm, don't assume.

## Effort

Medium. The base substrate (two-phase build, companion artifact, CI gate, K4 test shape)
removes most of the plumbing risk; the new content is the RaBitQ encoder/estimator kernel and
its first-party recall benchmark.
