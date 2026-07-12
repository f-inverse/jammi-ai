# Precision Moat — roadmap master plan

The co-design moat: **quantization-aware model training co-designed with a B-bit
compressed-storage/rescore stage, under a statistically-committed recall floor — all in
one SQL-native binary.** No single-axis competitor can occupy this intersection (a vector
DB owns only storage; an inference server owns only the model; a measurement product owns
only the floor). This directory specs the waves that build it.

Evidence base: two adversarially-verified deep-research passes (mid-2026), captured in the
`jammi-frontier-opportunities` and `precision-wave-roadmap` auto-memories. Every numeric
claim here traces to a cited source in those notes or the per-wave specs.

## The base (already built, unshipped)

Branch `feat/asymmetric-binary-quant` carries the **base of this train**:
- **Asymmetric (median-centered) binary quantization** — fixes a real latent defect:
  transformer embeddings are anisotropic (‖μ‖/E‖v‖≈0.97 on ModernBERT), so a fixed-0
  `sign(v)` collapsed ~183/768 bits. `sign(v−τ)` with a per-dim learned threshold (τ
  companion artifact, two-phase sidecar build, ANN manifest v2→v3) eliminates the collapse;
  rescore (raw f32) is byte-identical, so it is cosine-free. Storage-side, benefits every
  binary index. Real but partial (no-rescore recall@10 +~0.06; near-neutral on the default
  rescored path — honest calibration).
- **Bootstrap-CI recall gate** — the binary recall gate now asserts a percentile-bootstrap
  95% CI lower bound clears a committed floor, closing the single-seed false-positive class
  (it caught a real n=100 variance gap on first run).

The base's two-phase build, companion-artifact pattern, per-dim centering, and CI gate are
the **substrate** every wave below reuses. It went through the full rigor chain (oracle PASS,
adversarial-audit PASS). It is NOT shipped — it is the train's foundation.

> Why the asymmetric-binary work is not "the moat": the original QAT-fine-tune moat was
> pursued rigorously and **rejected on evidence** — a naive sign-STE-in-contrastive-loss
> collapsed at real scale; an ITQ-rotation alternative also failed (a decisive A/B proved it
> can't fix a mean offset). The decisive A/B found the true root cause (anisotropy) and this
> simpler storage-side fix. That arc is the origin of the **decisive-A/B-first discipline**
> below — read `docs/plans/51-marathon-learnings/` and the `precision-wave-roadmap` memory
> for the full story before starting any wave.

## Release model (decided)

**N wave PRs → main (each merged UNSHIPPED) → ONE release at completion.** There are NO
per-wave releases. Rationale: greenfield, no production users; a demand-pulled consumer; a
published artifact nobody consumes yet buys nothing (bisection comes from main, CI from the
PR). Feature PRs do **not** each bump the lockstep version; a single final release PR bumps
it once and the orchestrator pushes tags (Model B).

**The one exception** — the only valid reason to publish mid-roadmap: the platform
(jammi-enterprise) **demand-pulls** a specific capability and needs it published to consume
it via its crates.io pin. That release is forced by consumer demand through the cross-repo
seam, not by the wave structure. Absent a real consumer pull, hold everything to one release.

Each wave is still its own **rigor-chained PR to main**: gap-analyze → pressure-test →
implement (domain agents) → adversarial-audit + oracle → verify → cookbook chapter. B6
atomicity is per-PR. A wave PR that does not clear its fail-closed gate does **not** merge.

## The discipline (non-negotiable, learned the hard way)

Every wave runs **decisive-A/B-first**:
1. **Ground before code** — the design must trace to verified references, not memory. (The
   QAT arc's naive design *looked* reasonable and was wrong; deep research found the real
   method.)
2. **Pressure-test the design** before implementation.
3. **Run the decisive experiment BEFORE full integration** — the cheap prototype/A/B that
   decides GO/NO-GO on real embeddings at real scale, not a toy fixture. (The ITQ-rotation
   A/B that killed a wrong storage integration in ~1 agent-run is the template. A hermetic
   toy gate is necessary, never sufficient — it produced a false positive here.)
4. **Fail-closed real-scale recall gate** — a wave ships only if it beats baseline on a
   statistically-sound (bootstrap-CI / multi-seed) recall measurement at credible scale.
5. **Cookbook-at-scale** validation is the co-evolution gate, not optional.

A wave GO/NO-GO is real: some waves (QAMA especially) may not beat baseline at scale, in
which case the moat is honestly smaller than hoped. Do not force-ship a wave that doesn't
clear its gate.

## Waves (sequence + dependencies)

| Wave | What | Depends on | Risk | Spec |
|---|---|---|---|---|
| **Base** | Asymmetric binary quant + bootstrap-CI gate | — | done | (this README) |
| **A — RaBitQ B-bit** | Arbitrary B-bit scalar codes (storage-side) | Base substrate | Med | `wave-a-rabitq.md` |
| **D — Reranker→embedder distillation** | Distill a larger teacher/reranker into the LoRA embedder (fine-tune-side) | — (independent) | Med | `later-waves.md` |
| **H — Hybrid / learned-sparse** | SPLADE + RRF fusion on the existing BM25 | — (independent) | Low-Med | `later-waves.md` |
| **B — QAMA co-training** | Quant-aware Matryoshka co-training against Wave-A codes | **A** (needs B-bit codes as the target) | **High** | `later-waves.md` |
| **L — Bounded late-interaction** | ConstBERT-style fixed-vector multi-vector | matched ANN backend spike | Med-High | `later-waves.md` |
| **M — Measurement moat** | Extend bootstrap-CI toward continuous/live-traffic floor | (ongoing) | Low | `later-waves.md` |

**Ordering logic:** A is the storage substrate B co-designs against, so A before B. D
(distillation) and H (hybrid) are independent of the quant spine and can run in parallel /
first — D is the highest value-per-effort and feeds B. B is highest-risk (joint training,
the exact class that failed) and comes after A with a reproduce-first gate. L is a distinct
later wave gated on a backend spike. M is connective tissue that every wave gates on.

You do not need to fully spec B/L before A/D/H land — their detailed design depends on the
earlier waves' *measured* results. `later-waves.md` specs them at goal + decisive-experiment
+ open-questions + gate level, to be refined when their predecessor lands.

## How a cold session picks this up

1. Read the `precision-wave-roadmap` + `jammi-frontier-opportunities` memories and this README.
2. Pick the next wave (A or D — both ready; A if continuing the storage line, D for the
   highest independent value).
3. Read that wave's spec, run its decisive experiment FIRST, and only then take it through
   the swarm as a rigor-chained PR to main (unshipped).
4. Update the roadmap memory when a wave lands; refine the next contingent spec against the
   measured results.
