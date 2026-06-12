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

## Start here (fresh session)

A session continuing this roadmap with no prior context should, in order:

1. **Load the operational context.** Read §7 (Execution substrate) and the memory
   files it names — they hold the build/test/release/infra specifics this document
   deliberately keeps out of the strategy. Recalled memories reflect the state when
   written; verify any file/flag/path they name still exists before acting on it.
2. **Re-establish the baseline.** Run the engine CI gate (§7.2) and the standing
   adversarial sweep on a clean `main`, green, before changing anything — the sweep
   is the regression net, not a one-off.
3. **Pick the next phase from §5.** H1 (in flight at the time of writing) → **H2 is
   the M1 gate** (scale + search). Within a phase, take the workstream whose
   acceptance — a *measured cookbook chapter + a CI oracle* — is most load-bearing;
   §7.7 maps each workstream to its primary files.
4. **Run every change through the rigor chain** (§2.3): plan → pressure-test →
   implement → independent adversarial audit → CI → release. Authoring/extending the
   cookbook chapter named in the workstream's acceptance *is* the integration test.

Issue-ref note: `#NN` here are this project's internal task-tracker IDs (not GitHub
issues). Each maps to a merged fix in git history / the `CHANGELOG`, and the
surrounding prose is self-describing — treat the number as a label, not a lookup.

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

## 1. Where we are (evidence-graded, end of 0.26.4)

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
  caps the gain, not the loss*. **Hard-negative mining OOM and the `refresh_every`
  default anti-pattern are CLOSED as of 0.26.4** (#160 — mining is memory-bounded,
  defaults overlay correctly on the wire; see CHANGELOG v0.26.4). The scale gap
  remains open (see §3.1).
- *Search ergonomics:* `search(source)` is ambiguous once a source has multiple
  embedding tables — callers fall back to an explicit fold.
- *Multi-tenancy* is correct but not what most assume: catalog-row + discriminator-
  column isolation hold, but a **discriminator-less federated source is globally
  readable** — the engine does not authenticate; that boundary lives above it. **The
  `with_tenant` ergonomics gap is CLOSED as of 0.26.4** (#60/#161 — replaced by the
  unambiguous `set_tenant` setter and the `tenant_scope` block-scoped context manager,
  on both embedded and remote surfaces; see CHANGELOG v0.26.4).

**Not yet battle-tested.**
- Streaming/CDC, mutable feature-store tables, and the full eval family were
  server-complete but lacked a published client surface until now. **The cp9 client
  gap is CLOSED for mutable/topic/pubsub as of 0.26.4** (#58/#158 — `RemoteDatabase`
  gains `create_mutable_table`, `drop_mutable_table`, `list_mutable_tables`,
  `register_topic`, `drop_topic`, `list_topics`, `publish_topic`, `subscribe_collect`
  with a conformance guard; see CHANGELOG v0.26.4). **Eval/infer client parity is
  NOT closed:** the embedded `Database` exposes `infer`, `eval_embeddings`,
  `eval_per_query`, `eval_inference`, `eval_compare`, `register_channel`, and
  `add_channel_columns`; of that family only `eval_calibration` (T4, #119) and
  `predict_with_context_predictor` (T3, #115) are on `RemoteDatabase`. The other
  remaining gap is end-to-end cookbook chapters driving the landed verbs against
  the published client (C1/C2 are authored in the cookbook; CI drives them on CPU
  with the embedded engine — not yet at scale or against a published `grpc://`
  server).
- **Scale is unproven** — all validation was at small scale (≤4k rows). The
  memory-scaling edge is now partially addressed (mining bounded) but the broader
  scale tier (3.1) remains the #1 gap.

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
- **Gap.** Everything validated at ≤4k rows; hard-negative mining OOMed at 1500 pairs
  in the corpus-encode pass *before training*, independent of batch size — a pass that
  materialized embeddings without bounding memory. **Partial close (0.26.4):** mining
  is now memory-bounded (no second full corpus-embedding copy; anchors scored in
  batches; ANN over-fetch capped — #63/#160, CHANGELOG v0.26.4). The scale tier
  itself remains open.
- **Approach.** Audit every remaining pass that materializes corpus/embeddings/edges
  (embed, propagate, fine-tune, exact-search fold) for chunked, bounded-memory
  streaming. Add a scale tier to CI at documented sizes (e.g. 100k → 1M rows /
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
- **CLOSED (0.26.4) — hard-negative OOM + default anti-pattern (#63/#160,
  CHANGELOG v0.26.4):** mining is memory-bounded; wire defaults overlay correctly
  (zeros no longer ship as literal values that the engine rejects). The
  `refresh_every` silent-default anti-pattern (where a `None` default silently
  required a positive value at runtime) is resolved — the default overlay now applies
  engine defaults unconditionally. The warmup-epoch question (#64) resolved as
  **not a bug** (#160): there is exactly one `for epoch in 0..epochs` loop shared by
  the tabular and graph paths (no warmup/zeroth epoch); the epoch count is verified
  exact and pinned by a regression-guard oracle
  (`optimizer steps == epochs × ⌈batches/grad_accum⌉`).
- **Remaining open:** standardization/domain-contract property test for every
  trainable head (high-offset oracle in CI), checkpoint/resume story at scale.
- **Approach.** A standardization/domain-contract property test for **every** trainable
  head (high-offset, low-variance, large-magnitude oracle in CI). Specify + test
  checkpoint/resume.
- **Acceptance.** Every head passes the high-offset oracle in CI; hard-negative mining
  runs at the scale tier; a documented, tested resume-after-crash path.

### 3.4 Remote (data-plane) surface parity & streaming
- **Gap.** The published `RemoteDatabase` was a strict subset of the embedded surface;
  the cp9 substrate (mutable tables, topics/CDC, channels) and the eval family had
  server handlers but no client wrappers (now closing). Long-lived `subscribe` streaming
  semantics over gRPC (backpressure, reconnection, exactly-/at-least-once) are unproven
  at scale.
- **CLOSED (partial) — training/pipeline wire parity (in by 0.26.0) +
  mutable/topic/pubsub client parity (0.26.4):** T1–T4+N (#107–#119) landed the
  training verbs (T3, #115), the pipeline verbs + `eval_calibration` (T4, #119), and
  the shared conformal/RRF numerics (N, #110) on `RemoteDatabase` — all in by the
  0.26.0 release. #58/#158 (0.26.4, CHANGELOG v0.26.4) added the cp9 substrate:
  mutable-companion-table + topic + pub/sub verbs, including multi-chunk
  `publish_topic` parity (#158 — the remote client collapses multi-chunk tables via
  `combine_chunks` before the wire hop, matching the embedded `concat_batches`). The
  conformance guard in `crates/jammi-python/tests/test_conformance.py` pins the
  landed sets (`_TRAINING_VERBS`, `_PIPELINE_VERBS`, `_NUMERIC_VERBS`,
  `_MUTABLE_TOPIC_VERBS`) name-for-name + signature-for-signature across embedded and
  remote transports. Cookbook chapters C1 (feature-store, mutable tables) and C2
  (CDC, trigger topics) author the end-to-end workflows on CPU with the embedded
  engine.
- **Remaining open — eval/infer/channel client parity (the H1 residual):** the
  embedded `Database` exposes `infer`, `eval_embeddings`, `eval_per_query`,
  `eval_inference`, `eval_compare`, `register_channel`, `add_channel_columns`; none
  of these are on `RemoteDatabase` (the server-side RPCs exist in
  `eval.proto`/`inference.proto`/`catalog.proto`, but the published client carries no
  wrappers), they have no conformance-guard set, and no measured cookbook chapter
  drives them. Also open — a live client-parity *defect*, not just a gap: remote
  `fine_tune_graph` builds its `FineTuneConfig` but never attaches it to
  `StartTrainingRequest`, so the server silently runs engine defaults (#167; the
  cookbook's `epochs=1` → 3-epochs observation). Also open: subscribe stream semantics at scale (replay+tail bounds,
  cancellation, reconnection under load); channels/eval/infer cookbook chapters
  against a published `grpc://` server at the scale tier; delivery-semantics
  specification.
- **Approach.** Specify subscribe stream semantics (replay+tail bounds, cancellation,
  reconnection) and test under load.
- **Acceptance.** Conformance guard green; a CDC chapter that publishes → predicate-
  filtered subscribe → backing-table replay end-to-end over the published client, at the
  scale tier, with stated delivery semantics.

### 3.5 Multi-tenant correctness as a stated, tested contract (open-core part)
- **Gap.** The two-layer model (catalog-row + discriminator-column isolation; the "no
  auth in the engine" boundary) is correct but under-documented and under-tested; the
  `with_tenant` ergonomics bind-in-place/return-None (#60).
- **CLOSED (partial, 0.26.4) — `with_tenant` ergonomics (#60/#161, CHANGELOG v0.26.4):**
  `with_tenant` (bind-in-place, returned `None`, read like a builder) is replaced by
  `set_tenant` (unambiguous sticky setter, `-> None`) and `tenant_scope` (block-scoped
  context manager restoring the prior tenant on exit, yielding the same connection
  object). Both the embedded and remote surfaces are updated atomically; the conformance
  test asserts neither surface carries `with_tenant`.
- **Remaining open:** a standing isolation oracle across the full verb surface (catalog
  reads, search, propagate, query, mutable, topics); a documented, measured multi-tenant
  chapter in CI at the scale tier; the BYO-auth seam documented with a worked example.
- **Approach.** Publish the tenant contract explicitly (what isolates, what does not,
  where the consumer must gate). A standing isolation oracle across every verb. A clean,
  documented **BYO-auth seam** (Flight SQL / gRPC interceptor) where a consumer plugs
  identity — the engine ships the seam, never the auth.
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
- **CLOSED (partial, 0.26.4) — `refresh_every` silent-default anti-pattern (#63/#160,
  CHANGELOG v0.26.4):** the hard-negative config defaults now overlay at the engine
  level; the wire sends `optional` fields that fall back to `HardNegativeConfig::default()`
  rather than literal zeros that validation rejected with an opaque error. The
  broader API-stability staging (provisional/stable annotations, reference-contract
  test in engine CI, typed error taxonomy) remains open.
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

- **H1 — landed except one residual.** Landed: the T1–T4+N remote-ML-surface spec
  set (#107–#119 — training, pipeline + `eval_calibration`, conformal/RRF numerics on
  the client, conformance guard), in by the 0.26.0 release; remote client parity for
  mutable/topic/pubsub (3.4 partial, #58/#158), `with_tenant` → `set_tenant` +
  `tenant_scope` (3.5 partial, #60/#161), hard-negative OOM + default anti-pattern +
  the epoch-count oracle (3.3 partial + 3.8 partial, #63/#64/#160), and crates
  index-wait (CI fix, #62), all in 0.26.4. The cp9 cookbook chapters (C1 mutable
  feature-store, C2 CDC/triggers) authored and merged in `jammi-cookbook`.
  **The H1 RESIDUAL — eval/infer/channel client parity:** `infer`,
  `eval_embeddings`, `eval_per_query`, `eval_inference`, `eval_compare`,
  `register_channel`, `add_channel_columns` are still embedded-only — no
  `RemoteDatabase` wrappers, no conformance-guard coverage, no measured cookbook
  chapters (see §3.4). **Also open from H1's scope:** eval/infer/channels cookbook
  chapters against a published `grpc://` server at scale; subscribe stream semantics
  under load; BYO-auth seam documentation; scale tier (moves to H2). See §3
  workstreams for per-item status.
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

---

## 7. Execution substrate (operational reference)

§1–§6 are deliberately free of host/tooling specifics so they stay true as the infra
changes. Those specifics live here and in the linked memory files. **This host is not
the `CLAUDE.md` devcontainer** — the paths, toolchain, and constraints below are for the
SageMaker host this work has run on; re-verify them at the start of a new session.

### 7.1 Engine build/test env (Rust)
Use the `.cargo/config.toml` defaults (sccache + mold). Do **not** override
`RUSTC_WRAPPER`/`RUSTFLAGS` — any override changes the sccache key and re-misses the
cache (this once turned a gate into ~100 min of redundant compiles). Per shell:
```
export PATH=$HOME/.local/bin:$PATH          # prebuilt sccache 0.8.2 + mold 2.32.0
export PROTOC=/opt/conda/bin/protoc         # not on the minimal PATH
export CARGO_TARGET_DIR=/mnt/sagemaker-nvme/ct-<unique>   # NOT /home (100 GiB, fills)
unset RUSTC_WRAPPER RUSTFLAGS
```
`sccache --start-server` once per session (`--stop-server` if it wedges). mold silently
falls back to the host `ld` (host gcc predates `-fuse-ld=mold`) — sccache is the real
win; don't chase mold.

### 7.2 The engine CI gate (run ALL — not a subset)
```
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --exclude jammi-python                 # full hermetic lane
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --exclude jammi-python --no-deps
cargo test --workspace --exclude jammi-python --features live-hub-tests --no-run
```
Proto/client changes also need `make -C clients/python generate` + a `jammi-python`
rebuild + the TS client build. Embedded Python tests need `maturin develop` — beware the
cross-worktree `jammi_ai` shadowing trap: pin `PYTHONPATH` to the worktree's `python/` +
`clients/python` and confirm `jammi_ai.__file__` resolves into the worktree (and that its
`_native.abi3.so` is the fresh build). The **Postgres** lane (`test-pg`,
`--features live-postgres-tests`) is CI-only — compile-check `--no-run` locally; remote
CI is the authoritative gate. Background long runs and poll; iterate with
`cargo test … -- the::test::name` on the built binary (no recompile), never by re-running
with a different `-p`/`--features`. Memory: `[[engine-ci-gate-checklist]]`.

### 7.3 Infra constraints (these bit this project)
- **NVMe (`/mnt/sagemaker-nvme`, ~416 GiB) is the build volume**; each worktree `target/`
  is multi-GiB and it *does* fill (`df -h /mnt/sagemaker-nvme`). Remove merged-PR
  worktrees (`git worktree remove --force …`) and stale `ct-*` dirs. A worktree checked
  out while the disk was full once silently dropped a tracked fixture and produced a
  spurious test failure — keep headroom.
- **`docker --gpus` is blocked on SageMaker** — GPU work runs the binary directly on the
  A10G, not in a GPU container. The cookbook's heavy GPU emit is one-time; CI reads the
  committed cache on CPU.
- **The server's JSON log sink writes nothing to a non-TTY stdout** — to prove GPU use,
  rely on `nvidia-smi --query-compute-apps=…`, not the server log.

### 7.4 GPU / CUDA build (A10G)
Full recipe in `[[cuda-build-recipe-host]]`: a one-time `cudatk` conda env
(`cuda-toolkit=12.9`), `CUDA_ROOT` at the `targets/x86_64-linux` subdir,
`CUDA_COMPUTE_CAP=86`, then `cargo build -p jammi-cli --features cuda`. Or use the
published `jammi-server-cu12` wheel (bundles the `nvidia-*-cu12` runtime + an
`LD_LIBRARY_PATH` shim). Prove the GPU path with `nvidia-smi` compute-apps.

### 7.5 Release machinery (lockstep bump + 8 publishes)
A release is a PR bumping `0.X.Y → 0.X.(Y+1)` across **8 files** — `Cargo.toml`,
`Cargo.lock` (via `cargo update --workspace`), `CHANGELOG.md`, `pyproject.toml`,
`clients/python/pyproject.toml`, `clients/typescript/package.json`,
`packaging/server-cpu/pyproject.toml`, `packaging/server-cu12/pyproject.toml` — every
publishable crate ships at the same `workspace.package.version`. On merge, tag
**`vX.Y.Z`** (crates ×12 / npm / GHCR CPU+cu12 images / release binaries) **and
`py-vX.Y.Z`** (PyPI: `jammi-ai`, `jammi-client`, `jammi-server`, `jammi-server-cu12`) on
the merge commit. All channels use OIDC trusted publishing (no tokens); crates.yml waits
for sparse-index propagation between dependent publishes; release-binaries creates the
release if missing. Release pre-authorization is standing for engine patches; only the
cookbook is user-gated. Memories: `[[packaging-25c-plan]]`, `[[jammi-distribution-model]]`,
`[[release-authorization]]`.

### 7.6 Repos & the cookbook loop
- Engine: this repo (`f-inverse/jammi-ai`). `docs/plans/` is **gitignored** — this
  roadmap is committed with `git add -f`, so a fresh session sees it only after its PR
  merges.
- Cookbook: **`f-inverse/jammi-cookbook`** (private, Quarto), version-pinned to a released
  `jammi_ai`. Its gate: `ruff` · `check_api_reference.py` · `check_citations.py` ·
  `no_deferral_grep.sh` · `pytest` · `quarto render` · checksum verify. The heavy GPU emit
  is one-time and committed; chapters read the cache on CPU and assert against
  `golden_metrics.json`. A fixed source/artifact path collides on re-render — use a fresh
  `tempfile.mkdtemp` per render. Memories: `[[cookbook-repo-decisions]]`,
  `[[cookbook-env-readiness]]`.

### 7.7 Code map — workstream → primary files
Entry points for §3 (not exhaustive; confirm against the current tree).

| Workstream | Primary files |
|---|---|
| 3.1 Scale / bounded passes | `crates/jammi-ai/src/fine_tune/hard_negative_miner.rs`; `…/fine_tune/trainer.rs` (`mine_hard_negative_loader`); `…/pipeline/graph_propagation.rs`; `crates/jammi-db/src/index/{exact,sidecar}.rs`; the embed pass in `…/model/backend/candle.rs` |
| 3.2 Search / retrieval | `crates/jammi-db/src/index/{exact,sidecar}.rs`; the `search`/`assemble_context` session methods; `crates/jammi-numerics` (RRF) |
| 3.3 Training robustness | `crates/jammi-ai/src/fine_tune/{trainer,hard_negative_miner,regression_loss,target}.rs`; `…/pipeline/context_predictor.rs`; oracle tests `crates/jammi-ai/tests/it/ft_correctness_sweep.rs` |
| 3.4 Remote parity / streaming | `clients/python/jammi_client/_database.py`; conformance guard `crates/jammi-python/tests/test_conformance.py`; server handlers `crates/jammi-server/src/grpc/`; trigger framing `crates/jammi-wire/src/trigger.rs` |
| 3.5 Multi-tenant | `crates/jammi-db/src/tenant_scope.rs` (analyzer rule); `…/session.rs` (`bind_tenant`/`with_tenant_scoped`); BYO-auth seam = the Flight/gRPC interceptor in `crates/jammi-server/src/`; `docs/guide/src/multi-tenant.md` |
| 3.6 Catalog lifecycle | `crates/jammi-db/src/catalog/model_repo.rs` (`retire_model`, `model_pk`); the append-only migrations in `crates/jammi-db/src/catalog/backend*.rs` |
| 3.7 Operability | server + worker logging; `crates/jammi-ai/src/fine_tune/worker.rs` (lease / recovery / crash-window) |
| 3.8 API stability / errors | typed errors per crate; gRPC mapping `crates/jammi-server/src/grpc/wire.rs` (`map_engine_error`); the cookbook's `check_api_reference.py` (lift into engine CI) |
