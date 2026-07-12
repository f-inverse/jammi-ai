# Later waves — contingent specs

These are specced at **goal + grounding + decisive-experiment + gate + open-questions**
level, NOT full designs. Their detailed design depends on the *measured* results of the wave
they build on — writing a full design now would be the "plan from assumptions" mistake the
QAT arc punished. Refine each when its predecessor lands. All follow the decisive-A/B-first
discipline in `README.md` and are rigor-chained PRs to main (unshipped).

---

## Wave D — Reranker → embedder distillation (fine-tune-side)

**Highest value-per-effort; independent of the storage spine; feeds Wave B. Could run first
or in parallel with Wave A.**

**Goal:** distill a strictly-larger teacher (cross-encoder / LLM ranker) into jammi's
bi-encoder embedder at *training* time, so reranker-quality is baked into the single vector
— no runtime rerank stage. Improves the first-stage dense retriever itself, and the improved
embedding is what then gets quantized (so it compounds with Waves A/B).

**Grounding (cited):**
- **BiXSE** — arxiv 2508.06781. Pointwise-BCE over LLM-graded relevance scores. On
  ModernBERT-base TREC-DL: **+12.6% NDCG@10 (42.32→47.67)** over InfoNCE; transfers to BEIR
  (+10.1% on Qwen2.5-1.5B). Token-efficient (one labeled pair/query).
- **Listwise distillation + contrastive** — arxiv 2505.19274. Combined BEATS contrastive
  alone; **contrastive-alone can WORSEN a SOTA embedder** (BGE FiQA 44.3→39.5). KL-div from
  temperature-scaled cross-encoder distributions (soft labels), not hard labels.
- Ties to `jammi-frontier-opportunities`: distillation helps **only when the teacher is
  genuinely larger** (arxiv 2507.08336) — same-capacity distillation confers no gain. jina-v5
  distills Qwen3-Embedding-4B → 596M + task LoRA adapters (maps onto jammi's LoRA-per-task).

**Decisive experiment:** fine-tune jammi's LoRA embedder two ways on the same data —
InfoNCE-only vs InfoNCE + listwise-distill from a hosted larger teacher/reranker — and
measure NDCG@10 / recall@k at real scale. **GO** if distillation clears InfoNCE by a real
margin (literature: +3–12%). This is a training-loop change on the existing contrastive
trainer + a teacher-hosting/scoring path.

**Fits the moat:** one vector per row, quality baked in *before* quantization, zero runtime
cost, fine-tune-owning-engine play. Feeds Wave B (a better base embedder to co-train).

**Open:** teacher choice (host a cross-encoder vs call an LLM ranker); soft-label
(listwise/KL) vs pointwise-BCE (BiXSE) — BiXSE is cheaper, listwise may be stronger; measure.

---

## Wave H — Hybrid / learned-sparse (independent near-term win)

**Goal:** fuse dense (quantized ANN) with lexical/learned-sparse on jammi's **existing BM25**
surface — dense+lexical recall gains at low cost.

**Grounding (cited):**
- **SPLADE-v3** — semanticscholar SPLADE-v3. Stat-sig > BM25 and SPLADE++, **comparable to
  cross-encoder rerankers.**
- Two-stage **ESPLADE first-stage + ConstBERT32 rerank** (arxiv 2504.01818): nDCG@10 ~74.4
  (matches ColBERT) at **<6ms** vs PLAID's ~51ms.

**Decisive experiment:** RRF / score-fusion of jammi's dense (quantized) results with a
lexical/SPLADE result set; measure fused recall@k / nDCG vs dense-alone and lexical-alone at
scale. **GO** if fusion clears the better single arm by a real margin.

**Fits the moat:** reuses the existing lexical index; composes cleanly with the quantized
dense stage (fusion is post-retrieval). Independent of the quant spine — no Wave-A/B
dependency.

**Open:** learned-sparse (SPLADE) vs plain BM25 as the sparse arm (SPLADE needs a sparse
encoder + inverted-index serving — bigger); fusion method (RRF vs weighted score); does it
compose with a committed recall floor (gate the fused result).

---

## Wave B — QAMA quantization-aware co-training (HIGHEST RISK, after A)

**Goal:** joint Matryoshka + multi-level-quant + quant-aware loss so the *model* emits
embeddings engineered to survive Wave-A's B-bit codes — the structurally-unique co-design
moat. Recover most of FP32 quality at the compressed operating point.

**Grounding (cited, treat as HYPOTHESIS not spec):**
- **QAMA** — dl.acm 3746252.3761077 (CIKM'25). Joint MRL + 2-bit/hybrid quant + quant-aware
  loss; claims **95–98% of FP32 at >90% memory reduction.** SELF-REPORTED, only 2 encoders
  (ModernBERT, MiniLM), not independently replicated, search is brute-force Hamming (ANN
  integration needs design).
- **GOR** (Global Orthogonal Regularizer, jina-v5, arxiv 2602.15547) — the grounded
  anti-collapse ingredient: spreads embeddings uniformly, binary-quant MTEB degradation
  −3.08→−1.90.

**⚠ This is the exact class that already burned us** — joint training with a quantization
term in the loss is what caused sign-bit collapse in the abandoned QAT arc. So:
- QAMA's claim is a **hypothesis to REPRODUCE on jammi's stack at real scale before any
  integration.** Decisive A/B: co-train (QAMA-style, with GOR) vs baseline embedder, both
  measured through Wave-A's *actual* B-bit codes, at ModernBERT scale. **GO only if co-trained
  quantized recall clears the baseline embedder's quantized recall by a real margin** —
  fail-closed. A NO-GO here is a legitimate outcome; do not force it.
- Depends on Wave A (needs the B-bit codes as the training target) and benefits from Wave D
  (a stronger base embedder to co-train).

**Open (resolve after A lands):** which quant target(s) to co-train for (the B-bit level Wave
A actually ships); Matryoshka × quant-aware loss composition; whether it's a `FineTuneConfig`
knob (like Matryoshka) or heavier; the anti-collapse regularizer (GOR / balance+decorrelation
— now grounded, see the abandoned arc's research); ANN integration of QAMA's Hamming search.

---

## Wave L — Bounded late-interaction (distinct later wave, gated on a backend spike)

**Goal:** ConstBERT-style **fixed-vector** multi-vector retrieval — the only form of
late-interaction that doesn't break the one-vector-per-row compression moat.

**Grounding (cited):**
- Naive ColBERT/PLAID is per-token → order-of-magnitude storage blow-up (PLAID-compressed MS
  MARCO v1 = 21.6 GiB, v2 = 202 GiB; WARP 95 GiB/1.44B tokens). **BREAKS the compression
  moat.**
- **ConstBERT** (arxiv 2504.01818): fixed C vectors/doc (not per-token) → halves index (11G
  vs 22G), near-parity (MRR 39.04 vs 39.99), reduction **orthogonal to Matryoshka + quant.**
- Composes with quantization: PLAID/ColBERTv2/WARP all use centroid + b-bit residual (WARP
  b=4 → 8×). Token-pruning (LeapMV) is another orthogonal ~2× cut. Centroid-only first stage
  gets 99+% recall @10·k — maps onto retrieve-then-rescore.
- **CAVEAT (3-0 verified):** the ANN backend for late-interaction is first-class, NOT
  interchangeable — ConstBERT + PLAID-defaults **collapses to 30% MRR vs 39% with FAISS-IVF**
  (its 32 vectors hit only ~12 unique centroids). A real integration cost.

**Decisive spike FIRST:** does bounded multi-vector (ConstBERT, fixed C) + a *matched* ANN
backend beat single-vector dense (quantized) enough to justify the storage/complexity, on
jammi's data? **GO** only if the quality gain clears the multi-vector cost meaningfully.

**Why later:** biggest integration surface (a fixed-vector model + a matched multi-vector ANN
backend + per-vector quantization), and it composes-with rather than replaces the A→B spine.

---

## Wave M — Measurement moat (connective tissue, ongoing)

**Goal:** extend the bootstrap-CI recall gate (already built in the base) toward a
**continuously-measured, quantization-robustness-aware committed recall floor** — measured
over the *same* quantization the storage layer runs.

**Grounding (cited):** turbopuffer productized the bar — committed **90–95% recall@10 (incl.
filtered), auto-measured on 1% live traffic, per-namespace endpoint** (turbopuffer.com/docs).
MTEB-BR (arxiv 2607.04581): bootstrap-CI + paired-bootstrap significance for embedding
leaderboards.

**This is arguably the most defensible piece** — a competitor can copy a quantizer, not a
co-designed measured guarantee. Every wave above gates on it. Not a standalone wave so much
as the floor-quality bar each wave raises. Near-term increments: multi-seed CIs (done for
binary), extend to RaBitQ/int8; longer-term: live-traffic sampling if/when there's a serving
deployment to sample.
