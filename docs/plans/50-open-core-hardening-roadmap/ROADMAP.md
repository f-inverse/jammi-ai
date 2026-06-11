# Open-Core Hardening Roadmap — to a Mainstream-Ready Release, and the Path to 1.0

Status: living roadmap. Scope: the **open-core engine only**. Enterprise capabilities
(authentication, RBAC/SSO, governed/audited conformal, ownership & approval
semantics, cross-org federation policy) are **BYO** — out of scope here by design.
The engine's job is to provide the *primitives* and the *extension seams* a consumer
builds those on; it names no consumer and ships no security boundary of its own.

This document is evidence-graded: every gap below is anchored to something this
codebase actually proved or broke. It is not aspirational. The validation method
that produced this evidence — a bidirectional engine↔consumer-cookbook loop, standing
adversarial oracle sweeps, and a per-PR rigor chain — is itself part of the roadmap,
because *how* we harden is as load-bearing as *what* we harden.

---

## 0. Two milestones, defined

- **M1 — Mainstream-Ready ("serious ML/Search").** A team can build a real
  graph-conditioned retrieval + uncertainty workload on a **trusted network at real
  scale** and trust the results, the failure modes, and the operational story. Not an
  API-stability commitment. Target: **0.28.x**.
- **M2 — 1.0.** A deliberate, stable public surface with semver guarantees; every
  verb measured, oracle-tested, remote/embedded-parity-tested, and scale-benchmarked;
  the harder streaming/transaction guarantees stated and proven; storage/wire formats
  versioned. Target: **1.0.0**.

The discipline test gates every item entering the engine, unchanged: *would a user
who has never heard of any particular consumer reach for this on its own?* If it only
survives with a real consumer's name attached, it belongs in that consumer's repo.

---

## 1. Where we are (evidence-graded, end of 0.26.3)

**Proven solid — validated to an unusually high bar.**
- *Conformal prediction & calibration.* Coverage *validity* was validated by 28
  numpy-first oracle/Monte-Carlo tests (LAC/APS/RAPS marginal coverage ≥ 1−α at
  multiple α, exact finite-sample correction `⌈(n+1)(1−α)⌉/n`, CQR/Mondrian cover,
  proper-scoring/PIT/significance/RRF numerics correct); a consumer cookbook
  independently reproduced `engine == manual` for CRPS/NLL/APS coverage. This is a
  genuine differentiator — prediction sets with *proven* coverage, not just scores.
- *Embedding → neighbor graph → propagation → retrieval.* Real, reproducible lifts
  (recall 0.538→0.556 on a 4k-paper graph; 0.747→0.919 on a 3.5k-airport graph),
  APPNP recurrence hand-verified, exact search now deterministic and resolving
  `_row_id` under the engine's default schema.

**Solid but with sharp edges.**
- *Fine-tuning* works (LoRA/MNRL/graph-FT) but the honest finding is *supervision
  caps the gain, not the loss*; **hard-negative mining OOMs at ~1500 pairs** (memory
  not scale-bounded).
- *Search ergonomics:* `search(source)` is ambiguous once a source has multiple
  embedding tables — callers fall back to an explicit fold.
- *Multi-tenancy* is correct but not what most assume: catalog-row + discriminator-
  column isolation hold, but a **discriminator-less federated source is globally
  readable** — the engine does not authenticate; that boundary lives above it.

**Not yet battle-tested.**
- Streaming/CDC, mutable feature-store tables, and the full eval family were
  server-complete but lacked a published client surface until now; not yet driven
  end-to-end from a user's seat.
- **Scale is unproven** — all validation was at small scale (≤4k rows). The OOM is a
  warning that memory-scaling edges are unmapped.

**Why this is a mature 0.26.x, not a 1.0.** The consumer loop is *still* surfacing
real engine findings (the fine-tune vertical alone produced three). Several bugs fixed
this cycle were severe and would have hit real users immediately — exact search broken
for non-indexed tables, calibration erroring on *every* run, regression unusable on
high-offset targets (years/prices/counts), a cross-tenant model-overwrite leak. The
well-trodden paths are now solid; less-exercised corners still have edges.

---

## 2. Method — how to harden (the process is the product)

These mechanisms produced the evidence above. They become standing commitments, not
one-offs.

1. **The engine↔consumer-cookbook loop, made continuous.** Every public verb earns a
   *measured* chapter in a consumer cookbook: a real workflow that emits a golden
   number against committed fixtures. Authoring it is the integration test; the loop is
   bidirectional (authoring finds engine bugs; every fixed verb becomes a chapter).
   M1/M2 acceptance for any verb = it has such a chapter **and** the chapter runs in CI.
2. **Standing adversarial oracle sweeps.** For each subsystem, construct independently-
   computed correct answers plus degenerate/boundary inputs; a failing oracle is a
   repro. Run the sweep suite every release, not once. (This is the #44 model; it found
   the negative-LR, double-count, and standardization classes.)
3. **The rigor chain, non-negotiable per PR:** plan → **pressure-test** → implement →
   **independent adversarial audit** → CI → release. Pressure-test and audit are the
   load-bearing steps: this cycle they killed an over-engineered schema design, a
   mathematically-wrong standardization design (Adam normalizes per-parameter, so loss-
   rescaling can't move a raw parameter), a manufactured-crux risk, a tenancy overclaim,
   and two wrong spec premises — all *before* merge or *before* shipping a false result.
4. **Domain-validity as a review invariant.** The unifying root cause across the sweep
   was "the engine computes confidently past its valid input domain" (negative LR,
   unbounded mean, inflated degree, global identity). Institutionalize input-domain
   validation/clamp/normalize as a checklist item + property tests on every numeric and
   catalog path.
5. **Add the missing dimension: scale/load.** Correctness ≠ scale. A scale-tier in CI
   (documented row/graph/QPS sizes, bounded peak memory, perf-regression gates) is the
   single biggest gap between "validated" and "mainstream-ready."

---

## 3. Hardening workstreams to **M1 (mainstream-ready)**

Each workstream: the gap (with evidence), the approach, and an acceptance criterion
expressed as a *measured* result.

### 3.1 Scale & resource-bounding — **the #1 gap**
- **Gap.** Everything validated at ≤4k rows; hard-negative mining OOMs at 1500 pairs
  in the corpus-encode pass *before training*, independent of batch size — a pass that
  materializes embeddings without bounding memory.
- **Approach.** Audit every pass that materializes corpus/embeddings/edges (embed,
  mine-hard-negatives, propagate, fine-tune, exact-search fold) for chunked, bounded-
  memory streaming. Add a scale tier to CI at documented sizes (e.g. 100k → 1M rows /
  10⁵–10⁶ edges) running on a representative box, asserting bounded peak RSS and a
  perf-regression gate.
- **Acceptance.** A published benchmark (embed/s, search QPS, propagate on N-node
  graphs, fine-tune throughput) at named scales with bounded memory; documented limits;
  CI fails on >X% perf regression.

### 3.2 Search/retrieval completeness & ergonomics
- **Gap.** `search(source)` ambiguous with multiple embedding tables; ANN recall vs
  exact unquantified at scale; no first-class filtered (metadata-predicate + vector)
  search; hybrid/RRF not yet a measured first-class path.
- **Approach.** Add an explicit embedding-set/`table=` selector to `search`. Establish
  an ANN-quality benchmark (recall@k vs exact across index params + scale). Make
  pre-filtered vector search a first-class verb. Land hybrid + RRF as measured chapters.
- **Acceptance.** A retrieval benchmark chapter: dense/hybrid/RRF recall@k + nDCG and
  ANN-vs-exact recall at ≥3 documented scales, all asserted to golden.

### 3.3 Training robustness
- **Gap.** The Adam/standardization class (fixed twice this cycle — fine-tune head and
  context predictor) shows trainable heads need a standardization/domain contract;
  hard-negative OOM (#63) + a `hard_negative_refresh_every` default that silently
  requires `>0`; a graph fine-tune warmup-epoch question (#64); no documented
  determinism/resume story at scale.
- **Approach.** A standardization/domain-contract property test for **every** trainable
  head (high-offset, low-variance, large-magnitude oracle in CI). Fix hard-negative
  memory-bounding + the default mismatch (validate-with-clear-error or sane default).
  Confirm/document the warmup epoch. Specify + test checkpoint/resume.
- **Acceptance.** Every head passes the high-offset oracle in CI; hard-negative mining
  runs at the scale tier; a documented, tested resume-after-crash path.

### 3.4 Remote (data-plane) surface parity & streaming
- **Gap.** The published `RemoteDatabase` was a strict subset of the embedded surface;
  the cp9 substrate (mutable tables, topics/CDC, channels) and the eval family had
  server handlers but no client wrappers (now closing). Long-lived `subscribe` streaming
  semantics over gRPC (backpressure, reconnection, exactly-/at-least-once) are unproven
  at scale.
- **Approach.** Finish client parity (mutable/topic/pubsub → then channels/eval/infer).
  A **CI conformance guard** that `RemoteDatabase` surface == embedded surface name-for-
  name + signature, so the fault line cannot silently reopen. Specify subscribe stream
  semantics (replay+tail bounds, cancellation, reconnection) and test under load.
- **Acceptance.** Conformance guard green; a CDC chapter that publishes → predicate-
  filtered subscribe → backing-table replay end-to-end over the published client, at the
  scale tier, with stated delivery semantics.

### 3.5 Multi-tenant correctness as a stated, tested contract (open-core part)
- **Gap.** The two-layer model (catalog-row + discriminator-column isolation; the "no
  auth in the engine" boundary) is correct but under-documented and under-tested; the
  `with_tenant` ergonomics bind-in-place/return-None (#60).
- **Approach.** Publish the tenant contract explicitly (what isolates, what does not,
  where the consumer must gate). A standing isolation oracle: the property must hold
  across **every** verb (catalog reads, search, propagate, query, mutable, topics). A
  scope-safe `with_tenant` (context manager / scoped handle). A clean, documented
  **BYO-auth seam** (Flight SQL / gRPC interceptor) where a consumer plugs identity —
  the engine ships the seam, never the auth.
- **Acceptance.** A measured multi-tenant chapter (listing isolation + discriminator-
  column isolation as hard zeros + the honest global-source caveat) in CI across the
  verb surface; the BYO-auth seam documented with a worked example.

### 3.6 Catalog & lifecycle completeness
- **Gap.** Model retire shipped; no hard delete of unreferenced models, no
  versioning/promotion semantics; source schema-evolution and reload-at-scale unproven;
  migration-on-Postgres path exercised but not load-tested.
- **Approach.** Hard-delete-when-unreferenced (typed `ModelReferenced` error); model
  version/promotion verbs; source schema evolution; migration load tests on both
  backends.
- **Acceptance.** Lifecycle chapter covering register → promote → retire → delete with
  referential-integrity assertions; migrations validated at the scale tier on SQLite +
  Postgres.

### 3.7 Operability / production-readiness
- **Gap.** Observability is thin (the silent-non-TTY-logs bug was a canary); no
  published metrics/tracing surface; backpressure/quotas/rate-limiting unspecified;
  failure-mode behavior undocumented.
- **Approach.** Structured logs + Prometheus-style metrics + tracing across server and
  worker; health/readiness + graceful shutdown; documented backpressure and resource
  quotas; a failure-mode matrix (what happens when storage / broker / GPU / a worker
  dies). Build on the existing worker lease/recovery + crash-window work.
- **Acceptance.** An operability doc + a chaos test (kill storage/broker/worker mid-
  workload, assert recovery + no data loss/leak) in CI; metrics/trace exemplars.

### 3.8 API stability staging & error taxonomy
- **Gap.** The surface is good but not yet *deliberate*; some confusing failures (the
  `refresh_every`-default-None error is the anti-pattern).
- **Approach.** Make the grounded-API-reference contract test the engine's own (a
  consumer cookbook already does this — lift it in). A typed error taxonomy with clear,
  actionable messages and stable gRPC status mapping. A provisional-vs-stable annotation
  on every public verb (sets up the 1.0 freeze).
- **Acceptance.** Reference-contract test in CI; every public error is typed with a
  documented message + gRPC code; the provisional/stable split published.

---

## 4. The path to **1.0 (M2)** — beyond mainstream-ready

1. **API-stability commitment (semver).** A deliberate final pass to remove/rename
   anything provisional; no breaking change without a major. Requires every public verb
   annotated stable and frozen.
2. **Breadth coverage, uniformly.** Every public verb has: (a) a measured cookbook
   chapter, (b) an adversarial oracle in CI, (c) a remote+embedded parity test, (d) a
   scale benchmark. 1.0 is the point where that grid is fully green — no
   surface-with-an-edge ships unmarked.
3. **The harder guarantees, stated and proven.** Trigger-stream delivery
   (at-least-once / exactly-once, replay completeness after restart); transactional
   guarantees spanning catalog + result-table storage; crash-consistency of the
   mutable-companion-table path. Each with a property test / model-checked argument.
4. **Performance SLOs.** Documented, gated targets for the core verbs at named scales.
5. **Storage/wire format stability.** Parquet result-table + sidecar-index + proto
   schemas versioned and forward-compatible; a documented upgrade path.
6. **Security posture (open-core).** The trusted-network boundary fully documented + a
   published threat model + a stable BYO-auth seam. (Auth/RBAC/SSO impls themselves
   remain enterprise/BYO — the engine ships the seam and the contract, not the policy.)

---

## 5. Sequencing

- **H1 — finish what's in flight (→ 0.26.4 / 0.27).** Remote client parity (mutable/
  topic/pubsub → eval/infer), the cp9 verticals (mutable feature-store, CDC/triggers)
  authored against the published client, the open loop findings (`with_tenant`
  ergonomics, crates index-wait, hard-negative OOM, graph warmup epoch). Closes 3.4
  partially + the smaller 3.3/3.5 items.
- **H2 — scale & search (→ 0.27 / 0.28). _This is the M1 gate._** The scale tier
  (3.1), search completeness + ANN benchmarks (3.2), training memory-bounding (3.3).
  At the end of H2 the engine is **mainstream-ready for serious ML/Search on a trusted
  network at real scale.**
- **H3 — operability & contracts (→ 0.28 / 0.29).** Observability + chaos (3.7), the
  multi-tenant contract + BYO-auth seam (3.5), the API-stability staging + error
  taxonomy (3.8), catalog lifecycle completeness (3.6).
- **H4 — 1.0.** The breadth grid fully green, the harder guarantees proven, perf SLOs,
  format stability, the published security posture (§4).

**M1 ≈ end of H2 (~0.28). 1.0 ≈ end of H4.**

---

## 6. How to run it (commitments)

- The continuous engine↔cookbook loop is the acceptance harness: no M1/M2 item is
  "done" without a measured chapter **and** a CI oracle.
- One adversarial sweep per release, broadened each time toward the breadth grid.
- The scale tier runs on every release candidate.
- The rigor chain (plan → pressure-test → implement → independent audit → CI → release)
  on every PR — the pressure-test and audit are not optional; they are where this cycle
  repeatedly caught wrong designs and false results before they shipped.
- Honesty is the non-negotiable: a manufactured benchmark, a mismeasured fix, or an
  overclaimed guarantee is a release blocker, not a nuance. The whole reason the core is
  trustworthy today is that nothing untrue was allowed to ship.
