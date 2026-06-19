# Maintainer roadmap: sharp edges, tech debt & first PRs

This is the journey-shaped companion to `docs/maintainer/MAINTAINER-GUIDE.md`. It
records where the system is heading — known gaps, deliberate non-goals, and the
roadmap state — plus a few safe starter PRs. Section references (`[§N]`) point into
the maintainer guide.

## Sharp edges & tech debt

### Architectural gaps

- **Rust `Target` is Local-only** (`crates/jammi-ai/src/jammi.rs`, the `Target`
  enum) — the "one front door serves both" property is fully realized only in
  **Python** (`connect`). A unified Rust front door needs a `Target::Remote` arm
  dispatching to `jammi-client` (designed carefully to preserve the candle split).
- **The index dispatcher leaks the concrete type** — `ResultStore::resolve_search_mode`
  returns `Option<SidecarIndex>` not `Option<Box<dyn VectorIndex>>`
  (`crates/jammi-db/src/store/mod.rs`); this is the single biggest barrier to a
  second ANN backend [§4.4].
- **`IndexType` / `DistanceMetric` config knobs are dead** — declared on
  `EmbeddingConfig` but read by no index/search code; a trap for anyone who assumes
  `default_index_type` switches backends. Metric is hardcoded `Cos`.
- **No `backend_version` enforcement on load** despite the manifest carrying it; a
  graph from a mismatched USearch could load wrong. Whether USearch's own header
  check catches a mismatch first is unconfirmed.
- **No vector mutation/delete/incremental update** — add-then-build then immutable;
  updating embeddings means a full rebuild (the recovery rebuild-from-parquet path is
  also the only "repair").
- **USearch FFI is path-based** — every cloud-backed load/save round-trips through a
  tempdir, a fresh full-bundle download per `AnnSearchExec::execute`; no in-process
  loaded-index cache (`AnnCache` caches *results* but is **not consulted** by the
  search path).

### Model lifecycle

- **Production runs `GpuScheduler::new_unlimited()`** (`crates/jammi-ai/src/session.rs`)
  — the entire memory-budget machinery (CAS admission, eviction, permits) is **inert
  in deployment**; it only bites in scheduling unit tests. Blocked on
  `detect_gpu_memory` (a stub that always errs, gated behind a future `cuda` feature).
- **GPU priority is a no-op** — `GpuPriority` is a label; real priority scheduling is a
  later phase. **ORT backend is a placeholder** (`load` always errors). **HTTP backend
  is off the cache contract** (`BackendType::Http` unreachable in `do_load`). **No CPU
  offload / residency tiering** (`_residency` is always `Gpu`, never read).
  **Activation memory not budgeted.** **No sharded-weight handling in catalog
  reconstruction** (HF download path does gather shards; catalog arm hardcodes
  single-file names).

### Training

- **NER is unimplemented end-to-end** (`encode_chunk` rejects `TextChunk::Ner`;
  `ner_loss` masking is approximate). **Audio LoRA-in-encoder is unsupported by
  design** (projection head on a frozen tower only). **Serve-path σ de-standardisation
  is not yet unified.** **`from_loaded` cannot represent RSLoRA scaling.** **GradCache &
  hard-negative mining only apply to MNRL** (silent no-op otherwise).
  `ClassificationLoss`/`FineTuneMethod` are degenerate single-arm enums. The trainer
  file is large (~5000 lines) with no sub-module split.

### Catalog / storage

- **`scan_after` materializes the entire qualifying row set into one batch** (the
  closure-passing tx API can't lazily stream across `poll_next`); memory is bounded
  only by topic retention. **`training_spec` column is reserved but unwritten.**
  **Postgres mutable backend** — full type coverage is unconfirmed; confirm
  `crates/jammi-db/src/store/mutable/postgres.rs` before relying on a new column type.
  **annotate loads the model at plan time** (`block_in_place`+`block_on`) — a schema
  cache would remove it; whether such a cache exists is unconfirmed.

### Server edge

- **Flight SQL multi-tenant concurrency (the biggest one)** — `TenantBoundProvider`
  mutates a process-global `TenantBinding`; concurrent multi-tenant Flight SQL can race
  on a stale binding. **Route multi-tenant through gRPC.** Planned fix: per-plan
  `ConfigExtension`.
- **`list_topics` pagination unimplemented** (returns everything in one response).
  **`SessionStore` is unbounded & in-process** (no eviction/TTL; `RwLock` panics on
  poison). **`InferenceService.Predict` is heavier than `Infer`** (reconstructs the
  context-serve source inline, the one handler not delegating to a single verb).
  **Embedded Python `Database` always runs a training worker** even for read-only use.

### Encoders

- **No shared encoder trait** — three structs with identical inherent methods + a
  hand-maintained `AnyEncoder` union (the main maintenance tax: the compiler catches
  missing union arms but not a method you forgot on one concrete encoder).
- **Per-family duplication** of `build`/`LoraSite`; **README drift**
  (`crates/jammi-encoders/README.md` predates the ClipText/audio/context families).
- **ONNX is not in `jammi-encoders`** — it's a separate backend
  (`crates/jammi-ai/src/model/backend/ort.rs` + resolver), selected by artifact
  detection.

## Roadmap state

The implementation roadmap is `docs/plans/50-open-core-hardening-roadmap/ROADMAP.md`.
Headline items:

- **Mainstream-ready milestone shipped**: scale tier, the `search(embedding_table=)`
  selector, training robustness (standardization oracle, seeded bit-reproducible CPU
  fine-tune, byte-exact checkpoint/resume), public `db.fine_tune(task="regression")`.
- **Next milestone**: operability/chaos, multi-tenant contract + BYO-auth seam,
  API-stability staging + error taxonomy, catalog lifecycle.
- **1.0 tag decoupled/deferred**: the 1.0 engineering bar ships as a 0.x release; the
  tag awaits real adoption.
- Flagged open items: channel gRPC error-taxonomy (`Code::Internal` instead of
  `AlreadyExists`/`NotFound`/`InvalidArgument`, file
  `crates/jammi-server/src/grpc/wire.rs`); a `fine_tune_graph` config-propagation
  defect; subscribe stream semantics at scale unproven; scale-tier bounded-memory audit
  on dependent paths; thin observability.

## Doctrine a maintainer must uphold

The rigor chain (`plan → pressure-test → implement → independent adversarial audit →
CI → merge`) is a bug-discovery mechanism. **Never merge on green CI alone** — green is
necessary, not sufficient (CI only checks what someone asserted). A delegated agent's
"done" is a claim to verify against the pushed artifact + the full gate. The
engine↔cookbook loop is the acceptance harness: no roadmap item is "done" without a
*measured* cookbook chapter **and** a CI oracle. Honesty is a release blocker.

## First-PR suggestions

Three concrete, safe starter changes, ordered by blast radius:

1. **Fix README drift in `jammi-encoders` (docs-only, teaches the encoder surface).**
   `crates/jammi-encoders/README.md` says "One crate, three encoders" and omits
   `ClipText`, the audio/context families, and the
   `dropout_positions`/`restore_dropout_positions` methods
   (`crates/jammi-encoders/src/lib.rs` is the authoritative surface). Bring the
   README's module list and API-surface list in line with `lib.rs`. Zero risk; forces
   you to read every encoder's method surface [§2.5] and the "no backwards compatibility
   / docs reflect current state" doctrine [§6]. Note: docs must not narrate journey.

2. **Add a new retrieval metric, e.g. MAP@k (small, end-to-end, teaches the atomic
   rule).** Follow [§4.8]: a field on `QueryMetrics`/`AggregateMetrics`
   (`crates/jammi-numerics/src/retrieval.rs`) computed inside `compute_query` **reusing**
   the already-built `top_k`/`relevant_set`/`grade_map`, the mean in `aggregate`, the
   `crates/jammi-wire/src/eval/report.rs` DTO update, and the `jammi-ai` runner update —
   *all in one PR* [§5]. Add the in-crate unit test + the contract test in
   `crates/jammi-numerics/tests/it/retrieval.rs`. Teaches the determinism contract (no
   RNG, `f64`, `total_cmp` for sorts) and the wire-coupling of metric structs.

3. **Add a new pooling strategy (self-contained, teaches the encoder dispatch).** Per
   [§4.5]: a `Pooling` variant (`crates/jammi-encoders/src/pooling.rs`), a match arm in
   `pool_and_normalize`, and a private `*_pool` fn following the `mask_f32` convention —
   automatically available to every encoder via the `.pooling()` builder knob.
   Self-contained in `jammi-encoders` (no parent-crate changes), and it teaches the
   **`Max` uses `-1e30` not `-inf`** and **L2 output is a hard contract** invariants
   [§5]. Add a unit test that asserts unit-norm rows.

Before opening any of these, run the local gate [§6] and remember: green CI is
necessary, not sufficient.
