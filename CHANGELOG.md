# Changelog

All notable changes to the Jammi AI workspace are recorded here. The
workspace ships every publishable crate at the same
`workspace.package.version`; PyPI `jammi-ai` mirrors that version.

## [Unreleased]

## [0.31.0] - 2026-06-18

**H4 — the 1.0 engineering bar, shipped as a terminal 0.x.** This release raises
the engine to its production-completeness bar: a published, enforced API-stability
and format-stability policy; harder operational guarantees with crash-consistency
and at-least-once proofs; CI-gated performance SLOs including a release-blocking
lane; a stated threat model with a dependency-audit lane; and the final engine
features — point-in-time joins, a verifiable materialization contract, and
incremental recompute with opt-in caching.

### Added
- **As-of temporal join — `asof_join` (SPEC-01).** A point-in-time join verb that
  matches each probe row against the most recent (or nearest) row of another
  relation at-or-before its timestamp, implemented as one verb-centric hand-built
  sort-merge operator. Typed `MatchDirection`/`Boundary`/`Tolerance`/`TieBreak`
  controls, SQL `NULL ≠ NULL` semantics, float temporal keys rejected at schema
  validation, and bit-reproducible tie-breaking. Available `embed == remote`
  through `Database.asof_join` and `RemoteDatabase.asof_join`.
- **Verifiable materialization contract + `verify_materialization` (SPEC-02).**
  Every produced result table now records a `.materialization.json` sidecar
  capturing a `ProducingDescriptor` (the verb and all output-affecting parameters)
  and a `MaterializationEnv` (identities and backend kinds, including compute
  device) folded into a `definition_hash`. A single `finalize_with_manifest` funnel
  is the sole `building → ready` transition, `recover()` reconciles the sidecar on
  startup (no new crash window), and the `verify_materialization` verb
  (`embed == remote`) confirms a table against its recorded determinants, returning
  a typed `MatchVerdict`. The descriptor folds every output-affecting parameter of
  all data variants (neighbor-graph, graph-propagation, context-set, as-of join),
  so a one-bit parameter change is a distinct identity.
- **Incremental-recompute sensing + recompute + caching (W-61).** A read-only
  sensing layer — `staleness`, `derives_from` lineage with a stack-safe transitive
  closure, and `lookup_cached` — over result-table lineage (`embed == remote`),
  with `Undecidable` returned honestly where no current-version surface exists.
  An opt-in `CachePolicy` (default `Bypass`) threads through the five result-table
  producers: under `Use`, a producer probes for a byte-identical recorded
  definition before computing and returns an observable `CacheOutcome`
  (`Computed`/`Reused`), so reuse is always requested and observable, never
  inferred — neighbor-graph and graph-propagation are genuinely cacheable, the
  unpinned-source producers honestly never hit. A `recompute(table, cascade)` verb
  replays the recorded `ProducingDescriptor` (always recomputing, `CachePolicy::
  Bypass`); `Cascade::Downstream` sweeps every transitive dependent once in a
  stack-safe topological order. A typed `NotRecomputable` refuses a pre-contract
  table and a typed `DependencyCycle` refuses a lineage cycle.
- **Breadth-grid scale benchmarks + rate-regression gate (W1).** Scale-benchmark
  cells for every scale-relevant public verb (training, conformal, eval, propagate,
  graph-train, context-predictor, model-inference, search/recall/RSS), each driving
  the real engine path with non-vacuous gates: fraction floors, metric goldens,
  in-process determinism + relative-perturbation digests, and a reusable same-box
  throughput regression-net harness with committed, re-derivable baselines.
- **API-stability policy + wire-surface freeze-guard (§4.1).** A published
  API-stability page documenting the three frozen stable surfaces — the conformance
  verb sets, the nine `jammi.v1.*` wire packages, and the persisted-format versions
  — plus the terminal-0.x semver commitment. The freeze is enforceable: a
  `jammi-server` freeze-guard decodes the live descriptor set into the served
  `(Service, Method)` and package set and asserts it equals a committed baseline, so
  removing or renaming a stable rpc reds CI and adding one requires an explicit
  baseline edit in the same change.
- **Security posture + threat model + dependency-audit lane (§4.6).** A published
  threat-model page stating what the engine defends (format-version reject-newer,
  tenant-scope filtering on every catalog query, typed error surfaces, the
  bring-your-own-auth interceptor seam) and what it explicitly does not
  (no authentication/authorization, `jammi-session-id` is a correlation id not a
  credential, no TLS/secrets), the trusted-network assumption, and the consumer's
  authenticate → authorize → bind responsibility. A `cargo-deny` CI lane
  (`deny.toml`) enforces RustSec advisories, a license allowlist, and the one-way
  dependency direction.

### Changed
- **Performance SLOs are CI-gated (W3-b, §4.4).** A release-tag-blocking `perf-gate`
  in the crates workflow runs the CPU-hermetic `*-scale` tiers on every `v*` tag;
  a structural throughput regression below the committed floor or a determinism
  drift reds the gate and blocks the publish + GitHub release. A nightly
  early-warning `perf.yml` lane runs the same tiers without gating any merge.
- **Format stability — version-stamp validation on load (W3-a, §4.5).** Sidecar
  formats now reject incompatible on-disk artifacts on load through one shared
  typed `IncompatibleFormat` error: the `.rowmap` and ANN manifest versions
  reject-newer, and the USearch `backend_version` is strict-compared so a backend
  format bump can no longer silently mis-deserialize a neighbor graph. The
  materialization manifest rejects any non-current `MANIFEST_VERSION` before use.

### Fixed
- **Trigger at-least-once across the replay/live seam (W2 T1).** The subscribe seam
  no longer conflates the engine `_offset` with the broker's native sequence:
  subscription keys the live tail in engine-offset space and dedups the merged
  replay+live stream by `_offset`, so a post-commit broker fan-out failure can no
  longer skew the counters and drop boundary events. At-least-once delivery and
  replay-completeness are proven (including a randomized-state property test).
- **Catalog ↔ result-table crash-consistency + tenant-scoped recovery (W2 T2).**
  Proven crash-recovery of the `building → ready/failed` status gate across both
  backends; the startup `recover()` sweep is now admin-scoped and preserves each
  orphan's `tenant_id` rather than running unscoped and stranding tenant-owned
  building rows.
- **Mutable-table and topic lifecycle crash-consistency (W2 T3).** `register`,
  `register_topic`, `drop_table`, and `drop_topic` each run as a single database
  transaction (catalog row + storage DDL share the catalog database), so a crash
  leaves an all-or-nothing result; proven with SIGKILL crash tests.

## [0.30.0] - 2026-06-16

**H3 — operability & contracts.** This release hardens the engine for production
operation and pins its contracts: structured observability with a documented
failure-mode matrix, a typed error taxonomy mapped to correct gRPC status codes,
catalog lifecycle completeness (hard delete with referential integrity), and a
stated, standing multi-tenant isolation contract across the full verb surface with
a bring-your-own-auth seam.

### Added
- **Operability surface (§3.7).** The previously-dead Prometheus counters now
  increment (one path-filtered tower layer); gRPC and worker tracing spans
  correlated by tenant/job; an operability guide + failure-mode matrix; a chaos
  lane in CI.
- **Catalog lifecycle (§3.6).** `delete_model` — hard delete gated by a four-edge
  referential scan that raises a typed `ModelReferenced` (→ `FailedPrecondition`)
  rather than letting a database FK reject the delete, with `ModelNotFound`
  (→ `NotFound`) for an absent or out-of-scope target. The engine model-verb
  surface (`list_models`/`describe_model`/`delete_model`) is on the embedded PyO3
  and remote Python clients, projected through a minimal `ModelDescriptor`: the
  engine catalog is for listing and cleaning up models with referential integrity,
  not lifecycle-stage management.
- **Multi-tenant contract (§3.5).** A standing tenant-isolation oracle that proves
  isolation across every verb and binds coverage to the compiled proto descriptor —
  a new rpc landing without an isolation case fails CI. A documented
  bring-your-own-auth seam with a worked custom-interceptor example (auth placed
  ahead of the unauthenticated session-correlation header).

### Changed
- **Channel error taxonomy (§3.8).** Evidence-channel operations return typed
  errors mapped to correct gRPC codes (`AlreadyExists`/`NotFound`/
  `FailedPrecondition`/`InvalidArgument`) instead of `Internal`.
- **Transport-parity collapse (§3.8).** The embedded-PyO3 and pure-client request
  assembly is collapsed onto a single proto seam (`jammi_client._assembly` +
  `jammi_ai::wire` decoders), so each verb's field map lives once.

### Fixed
- **Cross-tenant isolation (§3.5).** Closed cross-tenant read/delete on mutable
  tables, sources, result tables, and topics via a strict tenant predicate on every
  destructive path; topic ids are now server-minted, so a caller can no longer
  replay a UUID to collide another tenant's topic registration.
- **Model-not-found gRPC status.** The lifecycle verbs now return a typed
  `ModelNotFound` mapped to `NotFound` (previously `InvalidArgument`, contradicting
  the documented contract).

### Removed
- **Model promotion and retirement.** The engine catalog is for listing and
  cleaning up models; lifecycle-stage management (promotion, retirement, and the
  `retired` status) is out of scope. Dropped the `promote_model`/`retire_model`
  verbs across every surface (catalog, gRPC, CLI, embedded PyO3, remote client),
  the `ModelStatus::Retired` state and its `ModelRetired` error, the `promoted`
  projection field, and the model-promotion catalog migration.

## [0.29.0] - 2026-06-14

The regression fine-tune surface is now consumer-usable and scale-robust. A team
can call `db.fine_tune(task="regression")` on a numeric target at any scale —
default objective (`BetaNll{0.5}`) included — and read back calibrated
predictions through `Infer`: this release wires that public on-ramp end to end
(numeric-target detection, the Python `regression_loss`/`quantile_levels` kwargs,
and `DistributionForm`-correct serving for both Gaussian and non-crossing
quantile heads). It also moves the regression loss into standardized (z) space so
the default Gaussian-family objectives converge on high-variance targets where
they previously diverged out of the box, with served point estimates preserved
within a tight tolerance (not byte-equal) and served σ correctly σ_y-scaled.

### Added
- **Scale-robust regression fine-tune loss — the standardization contract on the
  variance axis (W5-PR5).** `db.fine_tune(task="regression")` now converges for
  ALL FOUR objectives (`GaussianNll`, `BetaNll` the default, `Crps`, `Pinball`)
  on realistic-variance targets (σ_y ≈ 19+). Previously the default
  `BetaNll{0.5}` and `GaussianNll` DIVERGED out of the box on high-variance data:
  the loss scored `(y−μ)²/σ²` in raw outcome units, so a tens-of-units residual
  against a zero-init σ (≈0.69) blew the loss past the trainer's divergence guard
  on the first step. A new in-crate high-variance oracle
  (`standardization_contract`, σ_y ≈ 19) proves each objective converges, the
  served point estimate fits the target, and the served σ recovers σ_y exactly
  (a per-row σ_raw/σ_z = σ_y identity against an independent reference, which
  catches a missing OR mis-scaled multiply) — plus a raw-vs-z served-preservation
  check (within a justified tolerance, since the non-scale-free AdamW perturbs the
  trajectory) for the scale-equivariant objectives, a constant-target (degenerate
  σ_y) arm, and a destructive non-vacuity guard. The bug fingerprint (raw-space
  NLL trips the production divergence guard while Crps stays bounded) is pinned as
  its own test.

### Changed
- **The regression fine-tune loss is now scored in standardized (z) space
  (W5-PR5).** The head forward (`head_forward`, was `regress`) emits its raw
  z-output with no de-standardisation, and the target is z-scored via
  `TargetScaler::standardize_value` where the loss is computed, so the optimizer
  sees O(1) residuals at any target scale. De-standardisation moved entirely to
  the serve path, applied uniformly across all four objectives: the mean/quantile
  affine (`μ_y + σ_y·z`) stays at the backend, and the σ-axis multiply lands on
  the **post-softplus** σ (`σ_y·σ_z`, re-floored) at the inference adapter — which
  now carries σ_y (`DistributionAdapter::gaussian_scaled`). This mirrors the
  proven in-context predictor (`destandardize_distribution`), via a single shared
  σ helper (`destandardize_sigma`) both serve paths call, making the two regression
  surfaces one σ rule. For the scale-equivariant objectives (Crps, Pinball) z and
  raw space share the same population minimizer, so the served point estimates are
  preserved across the change — but NOT byte-equal: the production AdamW is not
  scale-free (`eps = 1e-8`, decoupled `weight_decay`), so dividing the loss by σ_y
  perturbs the trajectory. A served-output test pins this to a stated, justified
  tolerance (measured ≈ 0.12·σ_y on the mean at σ_y ≈ 19), not to equality. The
  divergence guard's `>100` threshold is unchanged. In z-space the regression-arm
  losses are O(1), so the *numeric* `>100` branch rarely fires on finite
  divergence — it is LESS discriminating there, not more — and the `is_nan()`
  branch becomes the load-bearing backstop for those arms.

- **Public `db.fine_tune(task="regression")` on-ramp end to end (W5-PR4).** A
  consumer can now fine-tune and serve a regression head through the public
  surface. A `(text, target)` source with a numeric `target` column, fine-tuned
  with `task=regression`, trains a distribution head and serves its
  de-standardised prediction back through `Infer`. The worker's column→loader
  detector gained a task-gated regression arm (built via
  `TrainingDataLoader::from_regression`) and an `extract_numeric_column` helper
  that reads `Int64`/`Int32`/`Float64`/`Float32` (and casts the remaining numeric
  families) into `f32` targets — the int64 year path included — rejecting
  null/NaN targets with a typed, row-citing error rather than coercing them to
  `0.0` (which would corrupt the scaler's μ/σ). The Python `fine_tune` binding
  now exposes `regression_loss` (`gaussian_nll` / `beta_nll` with `regression_beta`
  / `crps` / `pinball`) and `quantile_levels`, so a Python consumer can reach the
  Quantile head and every non-default Gaussian objective.

- The classification training branch is now gated on `task != Regression`: a
  `(text, label)` source submitted with `task=regression` no longer falls into
  the classification path (which gathered a numeric outcome as a class index and
  triggered a CUDA device-side assert) — it now produces a typed
  "regression needs a numeric `target` column" error.

### Fixed
- **A quantile-trained regression head is no longer silently mis-served as
  Gaussian on the `Infer` read path.** `create_adapter(Regression)` and the
  schema-construction twin now select the output adapter from the served head's
  persisted `DistributionForm` (threaded through `InferenceExec` /
  `build_output_schema`), so a Pinball/Quantile head serves its `quantile_{level}`
  columns (non-crossing) instead of being decoded as `(predicted_mean,
  predicted_std)`. The all-error read path picks up the same form-aware width via
  a new `OutputAdapter::error_output`.
- **The regression `distribution` head layer is now reloaded and applied on
  serve.** Previously only the `projection` layer was reconstructed, so a served
  regression head emitted the pooled embedding (hidden-width) instead of its
  distribution parameters; serving now applies the `distribution` layer the
  trainer trains, reproducing the `(mean, raw_std)` / quantile output shape. The
  end-to-end tests prove this by **group separation**: a fixture of two topically
  distinct text groups mapped to well-separated year bands trains a head that, on
  held-out items, serves group B above group A (Gaussian-form ≈ 6.9 yr,
  quantile-form ≈ 12.9 yr), where an untrained μ-regurgitating head serves μ_y for
  both (~0 separation). A permanent destructive guard zeroes the trained head and
  asserts that collapse, locking the tests against a future head-serving
  regression. On a realistic-variance target (σ_y ≈ 19.5) the Gaussian NLL
  objectives diverge (the loss scores `(y-μ)²/σ²` in raw outcome units); the
  Gaussian-form e2e path uses the robust `Crps` objective, which still serves
  `predicted_mean`/`predicted_std`.

## [0.28.0] - 2026-06-14

M1 — the "mainstream-ready" milestone. This release lands the H2
scale-and-search tier alongside the training-robustness work: a team can now
build a graph-conditioned retrieval + uncertainty workload at real scale on a
trusted network and trust the results, the failure modes, and the operational
story. Vector search is bounded-memory and tunable through typed HNSW knobs,
recall-vs-cost is measurable against an exact oracle with a held-out gate, and
fine-tuning is deterministic, durably resumable, and standardisation-correct
across every offset-bearing head. The wire/proto/Python/TS surfaces move in
lockstep, pinned by conformance signatures.

### Added
- **Seeded determinism for CPU fine-tuning (W5-PR0b) — adapters are now
  bit-reproducible.** A LoRA fine-tune on `Device::Cpu` is a pure function of
  `(seed, source rows, config)`: two runs with the same seed produce
  byte-identical adapter weights, run-to-run and across separate processes.
  `FineTuneConfig` gains a `seed: u64` field (default `42`, fixed — never drawn
  from entropy), threaded through the proto and the Python/TS clients. The LoRA
  A/B initialisation (`jammi-lora`) no longer draws candle's unseedable global
  RNG: a jammi-owned SplitMix64 fills the host buffers from a stream keyed by
  `(seed, fully-qualified parameter name)` — so the per-parameter draw is
  independent of `VarMap`/`HashMap` iteration order and stable across processes —
  and the seeded values are written into the registered trainable `Var`s in
  place. LoRA dropout is likewise seeded (a run-owned per-layer Bernoulli mask
  stream replaces candle's unseeded `ops::dropout`). The tabular source read
  (`read_source_columns`) now appends a deterministic `ORDER BY` over the full
  projected column tuple, pinning row order (DataFusion gives no order guarantee
  without it); ties can only occur between rows byte-identical on every selected
  column, which are interchangeable for both batching and the `TargetScaler`
  μ/σ reduction, so the result is a pure function of the row multiset. Two
  hermetic CPU acceptance tests prove this: `tests/it/ft_determinism.rs` covers
  seeded-init byte-reproducibility, and an in-crate `determinism_through_forward`
  module drives the production fine-tune dispatch (`lora_dropout > 0`, the LoRA
  head's `forward` actually applied so the adapter trains off zero) to prove the
  seeded-dropout and trained-trajectory contract — same-seed byte-equality,
  different-seed divergence, and a non-zero `lora_b` dead-path guard.
- **Durable checkpoint/resume of LoRA fine-tuning, proven byte-exact (W5-PR2).**
  A fine-tune that dies mid-training now resumes from a durable checkpoint and
  continues the *exact* trajectory it would have taken without the crash — proven,
  not approximated. At each epoch boundary the trainer writes a `ResumeCheckpoint`
  bundle — adapter weights, the AdamW first/second moments **per parameter keyed
  by name**, `step_t`, the `TargetScaler`'s `(μ, σ)`, and each dropout stream's
  draw position — to a new attempt-shared, publish-exempt durable prefix
  `{job_id}/_resume/` via `ArtifactStore::{put,fetch,delete}_resume_checkpoint`
  (thin wrappers over the existing manifest-last machinery, never on the
  serving/CAS path). On (re)claim the worker `discover`s that checkpoint and the
  trainer restores it, starting at `last_completed_epoch + 1`; absent it, training
  starts from scratch as before. The durable save is gated on the lease (the
  `!cancel` epoch-boundary check), so a reclaimed zombie cannot regress the shared
  checkpoint to a stale epoch; the finalize-CAS winner GCs it once the job is
  `completed`. Three correctness fixes make the resume bit-exact: the trainer now
  snapshots `VarMap::all_vars()` **once** and correlates optimizer moments to
  parameter **names** (a `VarMap`'s HashMap order is not stable across processes,
  so serialising moments positionally would silently load the wrong parameter's
  trajectory); the dropout mask stream is replayable to a persisted draw position
  (so a resumed run draws the same masks, and a validation pass — dropout off —
  never perturbs it); and the scaler's `(μ, σ)` is *loaded* authoritatively on
  resume, never recomputed from possibly-changed source rows. Restore writes the
  weights **into the registered `Var`s in place** so the forward, the gradient,
  and the optimizer step stay bound to one tensor identity. An in-crate
  `resume_invariant` test proves the three-run invariant on `Device::Cpu`,
  bit-exact and multi-thread: restored state is byte-equal to the uninterrupted
  run's epoch-boundary snapshot AND the next steps produce byte-equal weights,
  with ≥3 LoRA layers (so the name-keying, not positional luck, is what holds),
  `lora_dropout > 0`, a lease-gate concurrency test, a source-mutation test (the
  persisted scaler is authoritative), and a destructive weights-only-restore
  control that diverges (so the next-steps assertion is non-vacuous). The
  bit-exact next-step guarantee is scoped to `Device::Cpu`; on CUDA/Metal the
  restored state is still byte-equal (load + compare) but subsequent steps match
  only within tolerance.
- **Standardisation-contract oracle + completeness guards for every
  offset-bearing distribution head.** The fine-tune "standardisation contract"
  (a zero-init distribution head reaches a high-offset / low-variance target —
  calendar years μ≈2017, σ≈2 — only by reparameterising through a single
  dataset-level `TargetScaler`, since AdamW's per-step move is ≈`lr·sign(grad)`
  and is loss-scale-independent) is now a type-checked, every-head property.
  A new `StandardizableHead` closed enum
  (`crates/jammi-ai/src/fine_tune/target.rs`) is the union of the four
  offset-bearing heads — fine-tune Gaussian/quantile and in-context
  Gaussian/quantile — with an `ALL` source-of-truth slice (the
  `ModelTask::ALL` idiom). Two complementary completeness guards bind it: an
  **exhaustive, no-wildcard** map from every `jammi_wire::RegressionLoss` arm
  *and* every `PredictiveHead` arm onto a `StandardizableHead` — so a new arm on
  either enum, including the cross-crate wire enum, **fails to compile** until it
  is given its standardisation contract — plus a `#[test]` asserting every arm of
  both enums lands on a head listed in `ALL` (catching a new loss that lands on
  an existing head). The classifier is load-bearing in production: the fine-tune
  trainer's `regression_form` and the context predictor's persisted-form tag both
  route their gaussian-vs-quantile decision through it. The oracle itself adds
  the genuinely-new coverage — an in-crate train-to-fit test that drives the
  **production** fine-tune dispatch (`TrainingLoop::regress` →
  `TargetScaler::destandardize` and `compute_loss` → `regression_loss` → the
  configured `RegressionLoss`) for the Gaussian (head 6) and quantile (head 7)
  fine-tune-regression heads, asserting the served mean fits μ_y within ±50 and
  the served σ stays off the floor. With this change **all four offset-bearing
  heads now carry a high-offset train-to-fit behavioural oracle**: the two
  fine-tune-regression heads above, plus the two in-context heads
  (`crates/jammi-ai/src/pipeline/context_predictor.rs`) — the pre-existing
  Gaussian in-context oracle and a new sibling for the **quantile** in-context
  head, which trains a pinball head to the same μ≈2017/σ≈2 target and asserts the
  served, de-standardised quantile set fits μ_y within ±50 and is non-crossing.
  Every oracle reads its assertion off the served/de-standardised parse the
  serving path runs, and each pairs the fit with a destructive un-standardised
  arm that fails that same bar — so each test would fail if the `TargetScaler`
  reparameterisation were bypassed (verified by experiment, not narration). The
  offset-bearing-head surfaces were confirmed exhaustively to be the complete
  set — no further surface exists. All four heads fit the high-offset target
  as-is; no head fix was needed.
- **`jammi-bench` held-out ANN-vs-exact recall gate.** The `arxiv` subcommand
  now measures recall over a **held-out** query set — a query parquet *disjoint*
  from the indexed corpus — rather than querying the corpus with its own rows.
  With held-out queries no query is its own nearest neighbour, so recall@k
  reflects how well the frozen sidecar recovers the exact neighbours of *unseen*
  points (the quantity a deployed index is judged on), not the structurally-1.0
  recall a corpus-as-query set yields. A small hermetic fixture ships under
  `crates/jammi-bench/fixtures/scale/` — a deterministic sorted-`_row_id` subset
  of the real 170k-embedding scale cache (corpus rows + a frozen sidecar built
  once over them + a separate held-out query slice), with a `floor.json` whose
  per-k floors are the recall *measured on that slice* minus a safety margin. A
  cargo-test gate loads the committed fixture, runs the held-out recall path,
  and asserts each recall@k clears its committed floor — proving the held-out
  gate works hermetically on real embeddings with no Git-LFS dependency. The
  full 168k held-out recall gate runs in the cookbook chapter over the LFS cache
  this fixture is subset from; the split keeps the engine repo LFS-free while
  still asserting a real floor on a provable projection. The corpus-as-query
  recall *mechanism* and its primitives (`mean_recall_at_k`, the
  set-intersection arithmetic, the deterministic subset) are retained and still
  tested.
- **`jammi-bench` ANN-vs-exact recall mechanism.** The harness now measures how
  well a frozen sidecar index recovers the exact nearest neighbours. The
  `arxiv` subcommand drives a recall path that reads a committed `(_row_id,
  vector)` corpus back through the engine's own vector-read path, derives a
  deterministic query-by-example set (the first rows by sorted `_row_id`), runs
  the engine's `exact_vector_search` as the ground-truth oracle and a
  **loaded — never rebuilt** `SidecarIndex` as the approximate retriever, and
  reports recall@k as a *set-intersection* fraction
  (`mean |ANN_topk ∩ EXACT_topk| / k`) for a curve of k∈{1, 10, 100}. The
  `ArxivTier` schema's `recall_at_10` scalar becomes that k-keyed recall curve;
  the perf metrics (embed/search QPS, propagate latency, peak RSS) stay explicit
  not-yet-measured markers. Loading the frozen index (rather than rebuilding) is
  deliberate: USearch's default HNSW build is nondeterministic, so the committed
  graph is the one whose recall is asserted. A hermetic test proves the
  mechanism over a tiny deterministic fixture (a sidecar frozen over the same
  vectors the oracle scores recovers them, recall@k == 1.0; the exact oracle
  reproduces a hand-checkable top-k; the set-intersection arithmetic is
  order-blind; the sorted-`_row_id` subset is the deterministic projection). The
  meaningful real-embedding recall *floor* (recall@k ≥ 0.95 over a committed
  170k-row corpus) is asserted by a committed-fixture gate added after the
  on-box emit, in a later change.
- **`jammi-bench` scale-measurement harness (`publish = false`).** A new
  workspace member that links the engine and drives its primitives at scale,
  emitting one machine-readable JSON report per run (`cargo run -p jammi-bench
  --release -- <subcommand>`). It is a measurement *consumer* of the engine,
  kept out of the published workspace so the engine stays a clean library while
  still being compile-checked by the workspace gate. Its first functional
  subcommand, `search-rss`, is the bounded-memory proof for the streamed
  `exact_vector_search`: over seeded synthetic vectors at two corpus sizes it
  measures the streamed path's peak RSS against a bench-only naive collect-all
  baseline (the negative control — the `O(N·d)` path the streaming rewrite
  removed), and asserts the streamed resident set stays flat as the corpus
  grows while the baseline grows linearly. The realistic-corpus perf tiers
  (embed throughput, ANN QPS, propagate latency, peak RSS) are scaffolded as
  explicit not-yet-measured stubs so the report schema is stable from the first
  emit.
- **`search` gains an `embedding_table=` selector.** A source can carry several
  embedding tables (a raw table, a propagated table, a fine-tuned table); the
  search verb now names which one to search. `search(source, query=…, k=…,
  embedding_table="<table>")` searches that table; `embedding_table=None` (the
  default) searches the source's most-recent ready table — today's behaviour,
  unchanged. The selector reuses the exact `embedding_table=` name and
  most-recent-default semantics `eval_embeddings` already ships. It rides the
  whole surface atomically: the typed engine verbs (`Session::search` and the
  query-by-example `Session::search_by_id`, so naming a table makes the example
  vector AND its neighbours come from that one table), the flattened wire
  `SearchRequest`, the gRPC `SearchRequest` (new `optional string
  embedding_table = 7`) and its handler, the data-plane client, and both Python
  bindings (embedded `Database.search` and remote `RemoteDatabase.search`) —
  pinned identical across wheels by a conformance signature test.
- **Typed `AnnIndexConfig` exposes the HNSW graph knobs.** The ANN sidecar index
  previously hard-coded USearch's built-in HNSW defaults, so a deployment could
  not trade recall against build/query cost. The three universal HNSW dials —
  `connectivity` (M), `build_expansion` (ef_construction), and `search_expansion`
  (ef_search) — are now a typed `AnnIndexConfig` on `EmbeddingConfig`, named for
  the HNSW primitive rather than the backing library. The mapping onto
  `usearch::IndexOptions` lives in exactly one function (`sidecar::index_options`),
  the sole place USearch field names appear. A `0` knob is the documented no-op
  (USearch substitutes its built-in default), so an unset config reproduces
  today's indexes byte-for-byte. `connectivity`/`build_expansion` are fixed at
  construction; `search_expansion` is a query-time dial USearch does not persist,
  so `SidecarIndex::load` re-applies it via `change_expansion_search`. The config
  threads from `EmbeddingConfig` through `ResultStore` to every build site (embed
  pipeline, recovery rebuild, derived-table materialization) and the query-time
  load path — one deployment knob with zero wire/proto/Python surface. A
  round-trip test proves non-default knobs reach the backing graph and that a
  default resolves to USearch's documented 16/128/64; a load-path test proves
  `search_expansion` is re-applied on load while the build knobs are inert on a
  frozen graph; pinned default constants trip a backend bump that would silently
  shift recall/cost.
- **`jammi-bench recall-sweep` — recall-vs-cost sweep over the HNSW knobs.** The
  scale tier proves ANN recall clears a floor at one knob setting; `recall-sweep`
  now measures how recall trades against cost as the knobs move, against the exact
  oracle over a held-out query set, on two axes: a **build** axis sweeping the
  construction knobs (`connectivity`, `build_expansion`) — each point a separately
  built graph, costed by build time and on-disk size, an on-box reference too
  large to commit — and a **search** axis sweeping `search_expansion` over one
  frozen, re-dialed graph, the re-derivable portable recall-floor curve the
  cookbook re-runs against its own oracle. The exact ground truth is independent
  of the ANN knobs, so the top-k is computed once per query and reused across
  every swept point; QPS is measured on the very searches the recall curve runs,
  at k=10. A smoke test runs the whole sweep over the committed 2000-row fixture
  and asserts the output is schema-valid (every grid point present, every recall a
  fraction in [0,1], every build point carrying a positive build time and a
  non-empty index), making no monotonicity claim at that sub-millisecond corpus
  scale. The linked USearch version (`jammi_db::index::backend_version()`) is now
  recorded in the sweep tier and in every sidecar manifest `save` writes — recall
  and the serialized graph format are backend-version-dependent.
- **jammi-owned AdamW with serializable optimizer state.** The trainer's optimizer
  is now `fine_tune::adamw::AdamW`, a reimplementation with the *identical* update
  (decoupled weight decay, bias-corrected moments — the same arithmetic candle
  ran) that adds `state()` / `load_state()` / `step_t()`: the per-parameter
  moments and step counter a checkpoint must carry. `candle_nn::AdamW` keeps those
  buffers private with no accessor, so a mid-flight resume through it would restart
  the moments at zero and diverge — this jammi-owned optimizer is the substrate
  that makes the byte-exact checkpoint/resume below possible. It is a drop-in for
  both construction sites (text trainer, parallel context-predictor loop) and the
  shared clip→step seam. `state()` deep-copies the moment tensors (a shallow
  snapshot would be overwritten in place by the next step); `load_state` rejects a
  parameter-count mismatch. Numerically faithful: the step-accounting oracle, all
  16 trainer convergence tests, the in-context high-offset oracle, and the
  distributional integration suite pass unchanged.

### Changed
- **`exact_vector_search` is now bounded-memory.** The no-sidecar brute-force
  fallback streams the scan one `RecordBatch` at a time and folds it into a
  bounded top-`k` heap that retains only `(row_id, distance)` pairs — never a
  vector — instead of collecting every vector before scoring. Peak memory is now
  `O(k + batch_rows · d)`, independent of the corpus size `N`, rather than the
  previous `O(N · d)`. The result is bit-identical: the kept set and its order
  are unchanged because the `(distance, unique _row_id)` total order makes the
  bounded top-`k` return exactly the same prefix as the prior
  sort-then-truncate, and the per-row distance fold is untouched. One
  `candidate_order` comparator drives both the heap and the final sort.

### Fixed
- **`RemoteDatabase.fine_tune_graph` now attaches its `FineTuneConfig` to the
  request (#167).** The remote graph fine-tune assembled a `FineTuneConfig` from
  the caller's hyperparameters (loss, epochs, batch_size, learning_rate,
  lora_rank, matryoshka_dims) but never attached it to the `StartTrainingRequest`,
  so the server read `config = None`, fell back to its built-in defaults, and
  silently dropped every hyperparameter the caller set. The embed fine-tune path
  already attached `config=config`; the graph path was missing the same line. Two
  regression tests capture the assembled request (stubbing `_start_training`, no
  channel dialed) and assert the config rides it — one with explicit
  hyperparameters, one confirming even the default-MNRL config attaches — both
  failing without the fix.

## v0.26.5 — 2026-06-12

### Added
- **`RemoteDatabase` gains the eval family.** The published gRPC client's
  `eval_embeddings`, `eval_per_query`, `eval_inference`, and `eval_compare`
  drive the engine's evaluation verbs server-side (`EvalService`) and return
  the same nested report dicts the embedded `Database` produces — tagged
  inference aggregates flatten to `{"task": …}` records, `recall_at_ks` rides
  as `[k, recall]` pairs, and absent options (`delta` for a baseline,
  `significance` for an unpairable run) are explicit `None`s. Together with the
  already-present `eval_calibration`, the whole eval vocabulary swaps
  transports without changing the call; the projection shape is pinned from
  both the Rust and Python sides against one shared golden fixture
  (`tests/fixtures/eval_report_projection.json`).
- **`RemoteDatabase` gains the bulk inference verb.** The published gRPC client's
  `infer(source=…, model=…, columns=…, task=…, key=…)` runs a model over a
  registered source server-side (`InferenceService.Infer`) and returns the output
  rows as a `pyarrow.Table` — the same call surface as the embedded
  `Database.infer`, so a caller swaps `connect("file://…")` for
  `connect("grpc://…")` without changing the call. The result rides back as one
  unary `ArrowBatch`, so gRPC's default 4 MB receive cap bounds the result size a
  default channel can carry.
- **`RemoteDatabase` gains the evidence-channel family.** The published gRPC
  client's `register_channel`, `add_channel_columns`, and `list_channels` drive
  the engine's provenance-channel registry server-side (`CatalogService`) and
  carry the same call surface as the embedded `Database`, so a caller swaps
  transports without changing the call. `list_channels` returns the same dict
  shape on both transports — a list of
  `{"channel_id", "priority", "columns": [{"name", "data_type"}]}` ordered by
  `(priority, channel_id)`, with `data_type` the canonical PascalCase token
  (`"Float32"`, `"Utf8"`, …) that `register_channel` accepts. The registry is
  tenant-scoped: each verb rides the session's bound tenant, so a channel
  registered under one tenant is invisible to another and both may hold a
  channel of the same id without collision, while an unbound session sees only
  the global seed channels.
- **`Database.list_channels` on the embedded binding.** The in-process engine
  now exposes `list_channels` alongside `register_channel` /
  `add_channel_columns`, returning the registry read-back in the same dict
  shape as the remote client — closing the read half of the channel registry on
  both transports.

### Changed
- **`Database.eval_embeddings` names its result-table selector for what it is.**
  The embedded verb's optional kwarg is `embedding_table=` — it names the
  embedding result table to evaluate (`None` resolves the source's most recent
  table), which is what the engine always did with the value; the former
  `model=` name misdescribed the lookup. The remote `eval_embeddings` carries
  the same signature, and the cross-wheel conformance pin holds on the new
  name.

### Fixed
- **`eval_compare` significance CIs no longer depend on per-query emission
  order.** `bootstrap_ci` resampled its input positionally under a fixed seed,
  so the same multiset of paired per-query differences in a different order
  selected different values and produced a different confidence interval. Since
  `per_query` carries no `ORDER BY`, two engine instances could emit the same
  records in different orders and diverge on `delta.significance.<metric>.ci_*`
  while every point metric, delta, and the Mann–Whitney p-value agreed exactly
  — a self-comparison (all differences zero) hid it. The bootstrap now
  canonicalizes its sample basis (sorts the input) before the seeded resample,
  making the interval a function of the sample *multiset*, not its order — the
  property a seeded resampler needs to be reproducible across instances. All
  three call sites compute the order-invariant mean, so the canonicalization is
  correct for every one. A `jammi-numerics` unit test pins order-invariance of
  `bootstrap_ci` directly and a `jammi-wire` test pins it through
  `delta_significance` on a non-degenerate paired set.
- **The evidence-channel catalog is now tenant-scoped (cross-tenant data leak,
  D1).** `evidence_channels.channel_name` was a global `TEXT PRIMARY KEY` and the
  channel repo carried no tenant predicate, so one tenant's `register`/`list`
  saw — and collided with — every other tenant's channels even though the gRPC
  handlers already wrapped the calls in a tenant scope. The same D1 class fixed
  for the model catalog (#140). The channel name is now unique *per tenant*:
  migration 020 reshapes `evidence_channels` and `evidence_channel_columns` to
  carry `tenant_id` with `UNIQUE (tenant_id, channel_name)` and a composite FK,
  and `register`/`add_channel_columns`/`get`/`list` read and write `tenant_id`
  (`tenant = None` → `IS NULL` only; a tenant sees its own channels plus the
  unshadowed global seeds, never another tenant's). Because both backends treat
  NULLs as distinct in a UNIQUE constraint, a partial unique index on
  `channel_name WHERE tenant_id IS NULL` enforces global-channel-name uniqueness
  atomically — closing the race where two concurrent unbound registrations of
  the same name could both commit. The embedded `register_channel` docstring is
  corrected to say per-tenant. An adversarial cross-tenant isolation test covers
  the leak.
- **`jammi-client`'s declared floors can no longer lie about its stubs.** The
  proto stubs are generated at wheel-build time, and an unpinned `grpcio-tools`
  baked import-time guards (`GRPC_GENERATED_VERSION`,
  `ValidateProtobufRuntimeVersion`) far above the wheel's declared
  `grpcio>=1.60` / `protobuf>=4.25` — a wheel installed at its own minima
  crashed on import. The generator is now pinned (`grpcio-tools==1.80.0` in the
  `dev` extra, consumed by every CI/publish lane), the runtime floors are
  raised to what that pin emits (`grpcio>=1.80.0`, `protobuf>=6.31.1`), and a
  hermetic test asserts the floors satisfy the guards in freshly generated
  stubs. `make generate` also cleans `_generated/` first, so a proto removed
  upstream can't leave an orphaned stub behind.

## v0.26.4 — 2026-06-12

Wire-parity for the trigger/mutable-table substrate, tenant-scope ergonomics, and
fine-tune robustness — the engine surfaces the cookbook's data-plane chapters
exercise.

### Added
- **`RemoteDatabase` reaches the cp9 substrate.** The published gRPC client gains
  the mutable-companion-table, trigger-topic, and publish/subscribe verbs
  (`create_mutable_table`/`drop_mutable_table`/`list_mutable_tables`,
  `register_topic`/`drop_topic`/`list_topics`, `publish_topic`,
  `subscribe_collect`), so a caller swaps `connect("file://…")` for
  `connect("grpc://…")` without changing the call. The embedded surface gains the
  matching `list_mutable_tables` peer so the two vocabularies stay in lockstep.
- **Scope-safe tenant context manager.** `with db.tenant_scope("t"): …` binds a
  tenant for the block and restores the prior scope on exit (embedded and remote
  alike). The in-place setter is now `set_tenant` — an unambiguous `-> None`
  setter — replacing the `with_tenant` method whose `None` return read like a
  builder.

### Fixed
- **Hard-negative mining defaults resolve on the wire.** A remote caller that
  enables mining without setting the count knobs now picks up the engine defaults
  (`k`/`exclude_hops`/`refresh_every` are `optional` in the proto and overlay onto
  `HardNegativeConfig::default()`), instead of shipping literal zeros that
  validation rejected.
- **Hard-negative mining is memory-bounded.** The miner no longer keeps a second
  full copy of the corpus embeddings (the sidecar index is their sole owner), and
  anchors are scored in batches; the per-anchor over-fetch caps its excluded
  headroom so a dense corpus cannot escalate the ANN query into a near-full scan.
- **Publish parity for multi-chunk tables.** The remote `publish_topic` collapses
  a multi-chunk `pyarrow.Table` to one batch before sending, matching the embedded
  path and the wire's one-batch contract.
- **Release tooling waits for crates.io index propagation** between dependent crate
  publishes, so a fresh release no longer needs a manual re-run when the sparse
  index lags an upload.

## v0.26.3 — 2026-06-11

Follow-up engine work: a model retire lifecycle, catalog/SQL hardening, and
release-tooling fixes.

### Added
- **Model retire lifecycle.** `RetireModel` (a control-plane RPC + `jammi models
  retire`) soft-retires a model: `list_models`/`describe_model` hide it and the
  serve/load path refuses it, while `get_model` still resolves it so a training
  job or eval that references it stays valid. Retire is tenant-strict — a tenant
  can retire only its own model, never a global one.

### Fixed
- **Per-source tenant discriminator persists** across `reload_sources` (carried
  in the source connection), so a federated source's row-level tenant scoping
  survives a restart.
- **Multi-part relation references are quoted part-wise** in the eval/annotate
  SQL (`"catalog"."schema"."table"`), so hyphenated catalog/schema names resolve.
- **`release-binaries` no longer races release creation** — each tarball-upload
  leg creates the GitHub release if missing, so it succeeds on the tag push
  without waiting on the crates publish.
- **`jammi-wire` vendors `protoc`** — a source build (e.g. `cargo install
  jammi-cli`) no longer requires a system `protoc`.

### Changed
- Documented the multi-threaded-runtime invariant of the SQLite catalog
  `transaction()` path.

## v0.26.2 — 2026-06-11

Completes the regression target-standardization fix: 0.26.1 standardized the
fine-tune projection head, but the amortized context predictor — a separate
subsystem — was not covered and still collapsed on high-offset targets.

### Fixed
- **Context-predictor target standardization.** `train_context_predictor` /
  `predict_with_context_predictor` now z-score the outcome — and the in-context
  members' outcomes — in data space with one train-derived scaler, train the
  Gaussian/quantile head in that space, and de-standardize the served
  distribution (the scaler is persisted with the model and reloaded). The
  amortized in-context regressor now fits high-offset, low-variance targets
  (e.g. calendar years, prices) instead of collapsing to a far-off mean with a
  floored variance. (0.26.1's standardization covered only the fine-tune
  projection head; loss-space rescaling alone cannot fix this under Adam, so the
  standardization is applied to the data the head conditions on and is scored
  against.)

## v0.26.1 — 2026-06-11

A correctness patch from a deliberate adversarial sweep of the training, graph,
search, and catalog surfaces. Each fix establishes a domain-validity invariant
where the engine previously computed past its valid input domain.

### Fixed
- **Tenant model isolation.** The model catalog primary key is tenant-qualified,
  so a model registered under one tenant can no longer be overwritten by another
  tenant registering the same name; per-tenant models of the same name coexist.
  (Read paths were already tenant-scoped.)
- **Fine-tune learning-rate schedule.** The LR horizon counts the realised
  optimizer steps — including each epoch's trailing gradient-accumulation flush —
  and `compute_lr` clamps progress to `[0, 1]` and floors the rate at zero, so the
  schedule can no longer return a negative learning rate past the horizon. The
  trailing partial-accumulation window scales its loss by its actual micro-batch
  count.
- **Regression target standardization.** Distributional and quantile regression
  heads learn in a standardized space and apply a persisted de-standardization
  affine in their forward pass, so they fit high-offset, low-variance targets
  (e.g. calendar years) instead of stalling near the zero-init mean. The served
  distribution is de-standardized by its declared form, not by head width.
- **Undirected graph propagation.** A symmetric edge list that declares both
  directions of an edge no longer double-counts: redundant reverse edges collapse
  to the same unordered-edge set the engine's other graph operators use.
- **Exact vector search.** Tied distances break deterministically on `_row_id`,
  and `_row_id` resolves under the engine's default schema (`Utf8View`), so exact
  search works for tables without an ANN sidecar index.
- **Calibration evaluation.** A calibration run records no model foreign key (it
  scores a held-out predictive distribution, not a registered model);
  `eval_runs.model_id` is nullable while keeping its foreign key.

## v0.26.0 — 2026-06-10

The client redesign and server packaging: a candle-free client substrate, a
control/data-plane split, and prebuilt server distributions (CPU + GPU).

### Added
- **Candle-free client substrate.** Three new crates — `jammi-wire` (the
  `jammi.v1` gRPC tonic stubs, the proto↔domain conversions, the IPC helpers,
  and the shared session transport), `jammi-admin` (the control-plane
  `CatalogService` client), and `jammi-client` (the data-plane typed-RPC +
  Flight SQL client). None pull the embedded ML/candle stack.
- **Strict-client CLI.** `jammi` is now a control-plane-only client built on
  `jammi-admin`; it no longer depends on `jammi-ai` and links no embedded ML —
  enforced by a CI guard on its compile graph.
- **Server distributions.** `jammi-server` (CPU, `manylinux_2_28_x86_64`) and
  `jammi-server-cu12` (CUDA 12, bundling the `nvidia-*-cu12` runtime wheels with
  an `LD_LIBRARY_PATH` entrypoint shim) PyPI wheels, prebuilt CPU/GPU server +
  CLI tarballs, and CPU/cu12 container images entrypointed at `jammi-server`.
- **GPU capability-correctness suite** (`live-gpu-tests`, proven on an A10G):
  CPU↔GPU output parity for embeddings / encode / predict, and on-device
  learning for fine-tune and graph fine-tune. Device selection gained a
  `require_gpu` knob — loud CPU-fallback by default, fail-fast when set.
- **Object-store model artifacts** and a gated multi-process
  distributed-validation lane (exactly-once claim, kill-9 reclaim, artifact
  crash-window, cross-tenant isolation), with a hollow-green CI guard.
- **Trusted-network security-posture documentation** for the server.

### Changed
- **Control/data-plane split.** Catalog/metadata administration moves to a
  single control-plane `CatalogService` gRPC surface; Flight SQL is now
  query + data-DML only (catalog DDL such as `CREATE TOPIC` goes through the
  typed control RPCs). Source providers hydrate across all tenants at startup.

### Fixed
- Quote source/table identifiers in generated read SQL (hyphenated names).
- SQLite `BEGIN` cancellation-safety and `BEGIN DEFERRED` write-deadlock under
  the always-on worker; typed `RegisterTopic` now registers the broker driver
  as well as the catalog.
- Context-predictor `base_model_id` foreign key; non-TTY server logging.

## v0.25.0 — 2026-06-08

Graph feature propagation (S12) — the **propagate** half of a decoupled GNN.

### Added
- **Graph feature propagation (S12).** `propagate_embeddings` runs the SGC/APPNP
  forward pass `ÂᵏX` over a declared graph as a deterministic data-plane
  operation (no autograd, no architecture), emitting a normal `kind=Model`
  embedding table. Self-loops (`Ã = A + I`) so an isolated node propagates to its
  own `X⁽⁰⁾`; the over-smoothing-safe default is degree-normalised `Â` with an
  `α`-teleport restart (PageRank-decay), 2 hops capped at 3. A
  `PropagationWeighting` enum (`Uniform` / `DegreeNormalized` / `EdgeSimilarity`,
  the last clamping negative edge weights and folding `Σ(w·x)/Σw`), a typed
  `PropagationOutput` (`Final` / `JumpingKnowledge` — the per-hop L2-normalised
  `(K+1)·d` concat), a `PropagateRequest` builder, the tenant-scoped edge scan
  (a cross-tenant endpoint is never aggregated), an `f64` deterministic fold
  (byte-identical across thread counts), and a row-count ceiling. Python binding
  + the `graph-propagation` cookbook page.

### Changed
- `ResultStore::materialize_embedding_table` now takes `derived_from` so a
  propagated table records the FK lineage to its source embedding table.

## v0.23.0 — 2026-06-07

The amortized in-context predictor (S19) and its training substrate (P5): a
database-native prior-fitted network that conditions a calibrated predictive
distribution on a retrieved context set in one forward pass, with no gradient
updates at inference.

### Added
- **Parallel non-text training substrate (P5).** A `train_loop` over precomputed
  feature/target batches that reuses the autograd/optimizer stack without the
  token-coupled text trainer; a differentiable `segment_aggregate`
  (`SegmentReduce::{Sum,Mean,Max}`) matching the data-plane vector-aggregation
  UDAF, with a documented empty-segment-zero convention; an extracted, shared
  clip→step optimizer seam.
- **Amortized in-context predictor (S19).** `AnyContextPredictor` — a curated,
  config-selectable `{Cnp, AttnCnp, Tnp}` family in `jammi-encoders` — trained by
  an episodic meta-training pipeline (`train_context_predictor`): per-target
  leakage-scoped context assembly (S16, `exclude_self` + same-task split),
  per-member vector reads over the generic SQL surface, a held-out-**task** split
  with a meta-overfitting guard, and S18's proper-scoring objectives (reused, no
  new loss code). Served inference-only via the S18 distribution adapter, with a
  composed S17 conformal wrap calibrated on a held-out-task split. Python
  bindings + the `train-context-predictor` cookbook.

## v0.22.0 — 2026-06-06

The graph-ML and neural-process substrate: construct, learn over, and retrieve
over similarity graphs, and condition calibrated predictions on a retrieved
context set. All data-plane primitives + offline eval — no governance.
(Feature **propagation** over the graph lands in v0.25.0.)

### Added
- **Shared prep primitives.** Paired distribution-free significance (bootstrap
  CI + Mann–Whitney U) on `eval_compare` per-metric deltas; `jammi_numerics::calibration`
  (coverage, ECE, CRPS, NLL, sharpness, PIT — pure functions); a vector-aggregation
  UDAF (element-wise mean/sum/max over `FixedSizeList<Float32>`, permutation-invariant);
  a kind-conditional sidecar-extension registry.
- **Similarity-graph materialization.** `build_neighbor_graph` writes the self-kNN
  edge relation of an embedding table as a queryable `result_table` (migration 013
  adds `kind`/`derived_from`); index-assisted + exact drivers; approximate-by-default
  with an `exact` mode; endpoints are source keys.
- **Lexical retrieval + RRF.** A tantivy BM25 sidecar (`bm25` evidence channel,
  migration 014) and reciprocal-rank fusion that fuses on rank, not score scale.
- **Conformal prediction** (OSS serving primitive): distribution-free prediction
  sets/intervals (APS/RAPS/LAC/CQR/abs-residual, weighted + Mondrian) with the
  finite-sample quantile and a `conformal` evidence channel.
- **Context-set assembly.** `assemble_context` pools a retrieval into a permutation-invariant
  context representation (the encode-and-aggregate half of a Neural Process), with
  self-exclusion + train-split leakage guards.
- **Distributional inference.** A genuine `ModelTask::Regression` with a
  `DistributionAdapter` ((mean, std) or quantiles), proper-scoring objectives
  (β-NLL, CRPS, pinball), monotone quantiles, and an `uncertainty` evidence channel.
- **Contrastive fine-tuning.** Multiple-Negatives-Ranking (in-batch negatives /
  InfoNCE) with GradCache, index-mined hard negatives (k-hop false-negative guard),
  and Matryoshka multi-resolution embeddings; AnglE and cosine-MSE objectives.
- **Graph-supervised fine-tuning.** A `TrainingFormat::Graph` that samples a graph
  (node2vec biased walks) into contrastive pairs driving the existing objective —
  genuine gain comes from declared/external edges, not self-similarity edges.
- **Evaluation recipes.** A graph-ML "did structure help?" recipe and a calibration
  eval harness (`eval_calibration`) headlining a proper score with coverage + sharpness.

## v0.21.0 — 2026-06-04

### Added
- **Authenticated channels in `jammi-client`.** `connect(target, credentials=…)`
  attaches credentials to the channel — composite call-credentials on TLS, a
  metadata interceptor on plaintext — so the client can reach a bearer-protected
  endpoint. A typed `ChannelCredentials` / `BearerCredentials` abstraction; the
  per-connection session-id header continues to ride alongside.
- **`SigningKeyStore` port for audit signing.** The audit-HMAC master key flows
  through an `Arc<dyn SigningKeyStore>` owned by the session; `EnvSigningKeyStore`
  reads `JAMMI_AUDIT_MASTER_KEY` (the default, byte-for-byte identical signatures).
  `JammiSession`/`InferenceSession` accept a caller-supplied store at construction,
  so a host can route both the sign and verify paths through its own key store.

### CI
- **Open-core boundary fitness functions.** A dependency-direction guard fails the
  build if any engine crate's resolved closure contains a consumer/proprietary
  crate; an OSS-only build guard proves the workspace builds hermetically.

## v0.20.0 — 2026-06-04

### Added
- **Service tiers (S8).** The server mounts a configurable set of gRPC service
  tiers — *core* (Session / Embedding / Inference + introspection / MutableTable /
  Channel / Audit) always, plus optional *train* (FineTune), *event* (Trigger),
  *tooling* (Eval) — selected via `[server] services`, layered on the compile
  features. A compiled-out tier named in config is a truthful `FeatureNotCompiled`
  startup error, never a silent drop; a serve-only deployment no longer advertises
  train verbs (it returns `Unimplemented`). `ServerInfo.services` now reports the
  mounted tier set — the runtime capability handshake a remote caller needs.
- **Compound query over the wire (S7).** `annotate(model, task, relation, …)` — a
  DataFusion table function exposing model inference inside SQL, registered once on
  the engine context and reachable over both Flight SQL and the in-process `sql`
  surface — so a caller composes search → join → annotate → filter in one round-trip.
  `RemoteDatabase.sql(...)` runs SQL over Flight SQL, tenant-scoped.
- **`jammi-ai-server-cu12`** — a CUDA build of the server image, published on `v*`.

### Changed
- `search` is now a single bounded primitive returning a table directly on **both**
  the embedded and remote Python surfaces (the `.run()` builder is gone); the fluent
  compound builder is `QueryBuilder` (`crates/jammi-ai/src/query/`, renamed from
  `src/search/`). Embedded and remote `search` agree by construction.

## v0.19.0 — 2026-06-03

**Breaking — packaging & client-API redesign (spec M2 Stages 2+3).**

### Added
- **`jammi-client`** — a new pure-Python (`py3-none-any`), proto-generated remote
  client; the lean Shape-C deploy package, peer to the npm `@f-inverse/jammi-client`.

### Changed (breaking)
- **Unified `connect(target)`** replaces `connect()` / `connect_remote()` — one
  operator over a target (`file://…` embedded, `https://…` / `grpc://…` remote),
  mirroring the Rust `Jammi::open(Target)`. Transport is configuration (env-drivable
  via `JAMMI_TARGET`); scaling local→remote is a config change, not a code change.
  Engine tuning (`gpu_device`, batch size) moves to env (`JAMMI_GPU__*`, `JAMMI_ENGINE__*`).
- **The `jammi-ai` wheel is now local-only** — it links no tonic/proto. Its remote
  arm is provided by the new `jammi-client` dependency (composition: jammi-ai's remote
  *is* jammi-client's, by construction).
- Per-modality method names dropped in favor of the unified `modality=` form
  (`encode_query` / `generate_embeddings`).

### Removed
- **`jammi-ai-cu12`** (the CUDA embed wheel) and its `py-cu-v*` lane — CUDA now lives
  only on the server image. The PyO3 `connect_remote` / `RemoteDatabase` binding
  (superseded by the pure-Python `jammi-client`).

## v0.18.0 — 2026-06-03

### Added
- `EmbeddingService.ListSources` / `DescribeSource` — source-registry introspection,
  returning a `SourceDescriptor { source_id, kind, status, result_tables }`. The typed
  home for "what sources are registered and what's each one's status," so consumers
  (and downstream tiers) build on the engine instead of reimplementing it. (`DescribeSource`
  returns `NotFound` for an absent id; the remote surface maps that to `None`.)
- `SessionService.GetServerInfo` — a capabilities handshake reporting `{ version,
  features, storage_backends }`, so clients negotiate availability instead of
  discovering it via a runtime error.

### Changed
- The wire `ResultTable` is now self-describing — it carries its own `task` (a
  `jammi.v1.inference.ModelTask`), so `GenerateEmbeddings` and `DescribeSource` share one
  shape and `result_table_from_proto` no longer needs an out-of-band `modality` argument.

## v0.17.0 — 2026-06-03

### Added
- `RemoteSession` (and the Python `RemoteDatabase` / `connect_remote`) now wire
  `add_source` over the typed `EmbeddingService.AddSource` RPC. A remote (Shape C)
  consumer can register sources over the wire — not just `generate_embeddings` /
  `encode_query` / `search` — so the full ingest path runs against a remote engine.
  `sql` / `read_vectors` remain on the Flight SQL lane (no typed RPC) and still
  return the truthful "not available on the remote transport" error.

### Changed
- A default-on `local` cargo feature on `jammi-ai` gates the embedded ML engine
  (candle / hf-hub / tokenizers / symphonia / jammi-encoders). A remote-only client
  builds with `--no-default-features --features wire` and links none of those heavy
  deps; the embedded / PyO3 build is byte-unchanged (`default = ["local"]`). A CI
  lane guards the thin build against dependency regressions.

## v0.16.0 — 2026-06-03

### Added

- **Python remote sessions.** `jammi_ai.connect_remote(endpoint=…)` returns a
  `RemoteDatabase` that drives the engine over gRPC via the single Rust
  `RemoteSession` — the Python SDK can now run jammi in a remote deployment, not
  just embedded. The wheel gains the gRPC client; embedded use is unchanged.
- **Cloud storage in the published server image.** The `jammi-ai-server` image
  is built with the `r2`/`s3`/`gcs`/`azure` object-store backends enabled, so it
  reads `r2://` / `s3://` / `gs://` / `azure://` sources out of the box — no
  rebuild. (The default library build keeps these features opt-in for embedders.)

### Fixed

- **gRPC-web typed errors reach Connect clients.** Engine errors over gRPC-web
  now carry a canonical `google.rpc.Status` envelope (the typed detail as its
  `Any`), so a Connect-ES client surfaces the real status + message + detail
  instead of `"missing message"` for a trailers-only unary error. Raw gRPC /
  Flight SQL / success responses are unaffected.

## v0.15.0 — 2026-06-03

### Added

- **Real HTSAT-Swin CLAP audio encoder.** The audio tower is now a faithful port
  of the HuggingFace `transformers` `ClapModel` audio branch (an HTSAT Swin
  transformer): batch-norm → bicubic time-resample → `reshape_mel2img` → fused
  patch-embed (Attentional Feature Fusion) → four hierarchical Swin stages
  (windowed / shifted-window MSA with relative-position bias, patch-merging) →
  group-2D pooling → projection, fed by a `ClapFeatureExtractor`-matching
  front-end. `laion/clap-htsat-fused` now loads and embeds audio, reproducing
  HF `get_audio_features` (live cosine 1.0000002). A hermetic per-boundary golden
  suite parity-tests every unit against PyTorch, and a weight-key coverage test
  proves the full checkpoint is consumed.

### Changed

- **Replaced the flat-ViT `ClapAudio` placeholder.** The previous CLAP audio
  encoder was a single-scale ViT that matched no public checkpoint and only
  loaded a synthetic fixture; it is removed in favor of the real HTSAT-Swin
  tower. HF `clap` architectures (`model_type = "clap_audio_model"`) dispatch to
  the new tower, and the synthetic `tiny_clap` fixture is retired for the
  real-key `htsat_clap_tiny` fixture.

## v0.14.0 — 2026-06-02

### Added

- **Transport-agnostic SDK.** `Session` is now `Local(LocalSession)` plus a
  `wire`-gated `Remote(RemoteSession)`, dispatched by enum match — the same
  surface drives an in-process engine or a remote server. `Jammi::open(Target)`
  is the one front door selecting the transport; `Target::Remote` and the remote
  arm are `#[cfg(feature = "wire")]`, so a build without `wire` cannot name a
  remote target.
- **Complete gRPC wire surface.** Every `Session` method (sans `ephemeral`) is
  reachable over typed gRPC verbs: embeddings / encode-query / source / search,
  inference, eval, fine-tune (+status), mutable tables, topics (publish /
  server-streaming subscribe / register / drop), provenance channels, and audit.
- **Faithful typed-error wire contract.** `JammiError`, `TriggerError`, and
  `AuditError` reconstruct to their exact variant + fields across the wire (a
  structured error detail in the gRPC `Status`), so `Remote` returns the same
  error `Local` does — never a lossy gRPC-code guess. Engine-owned wrapped
  errors (e.g. `MutableTableError`, `BackendError`) reconstruct faithfully; only
  genuinely-foreign source errors degrade to a faithful `Display` string.
- **`@f-inverse/jammi-client`.** The official TypeScript gRPC-web SDK, generated
  from the canonical proto (protobuf-es + Connect-ES), for V8/Workers consumers
  that cannot load native code. Published to npm in lockstep with the engine.
- **Config-driven cloud result-table storage.** `[storage]` selects a cloud
  object-store backend (R2 / S3 / GCS / Azure) for result tables, alongside the
  local default.

### Internal

- The gRPC proto, generated client+server stubs, and proto↔domain conversions
  live in `jammi-ai` behind a default-off `wire` feature (one conversion set
  shared by the server handlers and `RemoteSession`); the embeddable engine and
  the PyO3 wheel stay free of tonic/prost by default.

## v0.13.0 — 2026-06-01

### Added

- **Audio embedding modality.** An `AudioEmbedding` task plus a CLAP-style audio
  encoder: decode → resample → log-mel → audio tower, producing L2-normalized
  vectors alongside the existing text and image modalities.
- **Audio-encoder fine-tuning.** The LoRA / contrastive fine-tune path accepts the
  audio encoder via a projection head, so a domain can adapt audio embeddings the
  same way it adapts text ones.
- **`EmbeddingService` gRPC surface.** `AddSource`, `GenerateAudioEmbeddings`, and
  `EncodeAudioQuery` exposed as typed gRPC RPCs, served over gRPC-web (tonic-web)
  so HTTP/2-less runtimes can drive the audio-embedding path.
- **`Search` on the gRPC wire.** The engine's `search` is now an `EmbeddingService`
  RPC (query by vector or by an existing row via `search_by_id`, with SQL-predicate
  filter and column projection), reachable over gRPC-web — the consumption verb for
  embeddings without the Flight SQL (HTTP/2) surface.
- **First-class `r2://` object-store backend.** Cloudflare R2 joins `s3://`/`gs://`/
  `azure://` as a named scheme; `R2Config` derives R2's account-scoped endpoint and
  `region = "auto"` so a deployer cannot misconfigure them. Gated behind `storage-r2`.
- **Self-contained server image variant.** A deployable image that bakes a config and
  a small encoder, for container-sidecar deployments.
- **Design Philosophy guide** (`docs/guide/src/philosophy.md`) — the engine-vs-consumer
  boundary, the discipline test, and the one-binary/pluggable-backends deployment stance.

## v0.12.1 — 2026-05-30

### Fixed

- **Per-query audit log crashed for the second tenant onward** in multi-tenant
  deployments. The `topics` catalog table (migration `009`) enforced a *global*
  `UNIQUE(name)`, but the audit primitive — like every substrate-owned
  trigger-stream topic — registers a *per-tenant* `jammi.audit.search.v1` topic.
  The first tenant to call `session.audit().log(...)` claimed the topic name
  process-wide; every other tenant's first `log` failed with
  `UNIQUE constraint failed: topics.name`. Direct multi-tenant `jammi-ai`
  library users hit this on their second tenant.
  - Migration `012` rebuilds `topics` with a composite `UNIQUE(name, tenant_id)`
    so per-tenant topics sharing a logical name coexist. Existing catalogs pick
    up the new constraint on next open via the numbered migration runner. The
    fix is engine-side, so no consumer workaround is required and delivered
    audit events remain tenant-isolated.

### Added

- **Per-query eval persistence + cohort tagging** (`jammi_db::catalog::eval_repo`,
  wired through the `jammi_ai` eval runner). Embedding evals now persist a
  companion per-query row alongside the historical aggregate, so per-query
  results survive the call and can be re-aggregated by segment downstream
  without re-running the eval.
  - New reserved, tenant-scoped catalog table `_jammi_eval_per_query`
    (migration `011`): one row per `(eval_run_id, query_id)` carrying a metrics
    JSON (`recall@1/3/5/10`, `mrr`, `ndcg`, `distance`) and an opaque `cohorts`
    JSON object (`{}` when none).
  - `Catalog::record_eval_per_query` (bulk multi-row insert, tenant-asserted)
    and `Catalog::get_eval_per_query` (tenant-scoped read, ordered by
    `query_id`).
  - `RetrievalMetrics::recall_at_ks` extends the numerics kernel to emit
    Recall@k at several cutoffs without re-deriving the recall definition.
  - `eval_embeddings` accepts an optional per-`query_id` `cohorts:
    map<string,string>` (opaque — the substrate never interprets keys/values),
    persists per-query rows always-on (no opt-in flag), and surfaces the
    `eval_run_id` on `EmbeddingEvalReport`. `PerQueryRecord` additionally
    carries `recall_at_ks`, `distance`, and `cohorts` (additive; existing
    `metrics` fields unchanged).
  - `session.eval_per_query(eval_run_id)` (Rust) and `db.eval_per_query(...)`
    (Python, returning dicts with decoded `cohorts` + `metrics`); `cohorts=`
    kwarg on `db.eval_embeddings`.
  - Cookbook `eval_embeddings` recipe extended with per-query drill-down and a
    cohort-tag round-trip.

- **Per-query audit record primitive** (`jammi_db::audit`, re-exported from
  `jammi_ai`). A standardized, tenant-scoped, HMAC-signed record of *what was
  queried, with what model, what came back, and when*. It composes the existing
  substrate primitives — mutable tables (storage), tenant scope (auto-injected
  `tenant_id` + scoped reads), the trigger stream (publication), and the catalog
  (registration) — so audited-ML tenants no longer hand-roll an incompatible
  audit schema, signature scheme, and stream integration per project.
  - `PerQueryAudit` typed record with canonical (fixed field order, recursively
    sorted keys, no whitespace) serialization used as the signing input.
  - `session.audit().log([...])` resolves the session tenant, enforces the
    `query_lineage` size cap by construction (`JAMMI_AUDIT_MAX_LINEAGE_BYTES`,
    default 8 KiB), signs each record with a per-tenant HMAC-SHA256 secret
    derived via HKDF-SHA256 from `JAMMI_AUDIT_MASTER_KEY`, batch-inserts into the
    reserved `_jammi_search_audit` mutable table, and publishes the batch to the
    `jammi.audit.search.v1` trigger topic.
  - `audit::fetch_by_query_id` / `fetch_recent` typed reads; tenant scope
    auto-applied by the analyzer.
  - `audit::verify` / `verify_with_env` signature checks, deterministic across
    restarts; `ensure_master_key_present` server-startup gate (a missing or
    invalid key is fatal for any signing or verification).
  - `create_mutable_table` now rejects any reserved `_jammi_*` table name; the
    audit table is created via a substrate-internal unchecked path.
  - PyO3 bindings: `db.audit.log([...])`, `db.audit.fetch_by_query_id(...)`,
    `db.audit.fetch_recent(...)`, and a `PerQueryAudit` record class with a
    `.verify()` method.
  - Cookbook recipe `cookbook/recipes/search_audit/` + smoke-test entry.

- **Ephemeral session-storage primitive** (`jammi_db::ephemeral`, re-exported
  from `jammi_ai`). A tenant-scoped storage context whose mutable tables are
  auto-deleted when the session ends — on explicit `close()`, on `Drop`
  (best-effort), or when the timeout scanner force-closes a session past its
  deadline. It composes the existing substrate primitives directly: mutable
  tables (session-prefixed storage), tenant scope (tables created and read under
  the session's bound tenant), the trigger stream (lifecycle publication), and
  the catalog (registration). Satisfies the requirement to delete uploaded data
  and derived representations immediately on session end while keeping durable
  audit lineage that references only hashes.
  - `EphemeralSession::open` opens a session pinned to the parent's bound tenant
    (refusing to open without one); `create_ephemeral_table`, `insert`, `sql`,
    and `count_rows` operate on real federated mutable tables whose physical ids
    are namespaced `__eph_<session-uuid>_<name>`.
  - `close()` (the safe path) drops every table the session created, sums the
    deleted rows, and publishes a terminal lifecycle event; partial drop
    failures emit a `partial_deletion_failure` event listing survivors.
  - Lifecycle events (`opened`, `closed`, `timed_out`, `partial_deletion_failure`)
    publish to the new `jammi.audit.session_lifecycle.v1` trigger topic
    (registered lookup-or-create per tenant, mirroring the audit topic path),
    carrying session id, tenant, table count, and deleted-row count.
  - A process-shared `ActiveSessions` registry + `spawn_timeout_scanner`
    background task force-closes expired sessions on a 60-second interval;
    explicit close and the scanner coordinate through the registry so tables are
    never double-dropped.
  - PyO3 bindings: `db.ephemeral_session(timeout_seconds=...)` returns a
    context-manager `EphemeralSession` (`create_ephemeral_table`, `insert`,
    `sql`, `count_rows`, `physical_table_ref`, `close`); the in-process timeout
    scanner is spawned on first use.
  - Cookbook recipe `cookbook/recipes/session_lifecycle/` + smoke-test entry.

## v0.11.0 — 2026-05-27

### Changed

- `jammi_db::catalog::resolve_embedding_table` derives its embedding-task
  list from `ModelTask::ALL.iter().filter(|t| t.is_embedding())` instead
  of a hardcoded `task IN ('text_embedding', 'image_embedding')` literal.
  Adding a future embedding variant only requires extending `ModelTask` +
  its new `ALL` constant; the resolver recovers it automatically. No
  wire change — `as_db_str` / `try_from_db_str` continue to map the same
  four snake_case strings, persisted `task` columns and serde JSON
  round-trip identically.
- `eval_inference` `PerRecordPrediction` is now a serde-tagged enum
  (`{"task": "classification", ...}` / `{"task": "ner", ...}`) mirroring
  the existing `InferenceAggregate` shape. Classification per-record
  dicts gain a `"task": "classification"` tag (additive); NER per-record
  dicts gain a `"task": "ner"` tag carrying `predicted`/`gold` entity
  lists.

### Added

- `ModelTask::ALL: &'static [ModelTask]` — single source of truth for
  "every variant," consumed by the catalog SQL builders. An
  exhaustive-`match` test guards against `ALL` drifting from the enum
  body (adding a variant without extending `ALL` either fails to
  compile or fails the membership assertion).
- `EvalTask::Ner` is now implemented end-to-end through `eval_inference`.
  The runner loads per-span gold rows `(id, label, start, end)` from
  the registered golden source, runs NER inference, parses the
  `entities` JSON payload, and computes entity-level
  precision/recall/F1 + per-type breakdown via
  `jammi_numerics::ner::NerMetrics`. New cookbook recipe
  `cookbook/recipes/eval_inference_ner/` exercises the path against the
  shipped `tiny_modernbert_ner` model fixture (relocated from
  `tests/fixtures/` to `cookbook/fixtures/` for the same reason the
  classifier fixture lives there). `jammi_numerics::ner::Entity` gains
  a `Deserialize` derive so the round-trip from the NER inference
  adapter's JSON column back into typed entity sets uses the same serde
  contract as serialization.

## v0.10.0 — 2026-05-27

### Added

- `TriggerBroker::list_consumers(topic_id) -> Vec<ConsumerOffsetSnapshot>`
  returns one snapshot per consumer currently bound to the topic, carrying
  the broker's last-delivered and ack-floor stream sequences. Unblocks
  the OSS broker listing gap noted in jammi-enterprise's E5 CHANGELOG
  entry; the enterprise backup path will adopt it once this release
  publishes. Wired through both the JetStream driver (via
  `stream.consumers()`) and the in-memory broker (each subscription
  registers a tracker that's pruned when the subscription drops).

### Changed

- `jammi_server::runtime::CatalogPingProbe` now drives readiness through
  `Catalog::ping` (the backend-native reachability primitive) instead of a
  `SELECT 1` round-trip on the DataFusion `SessionContext`. The probe now
  takes an `Arc<InferenceSession>` at construction.

### Removed

- `jammi_numerics::retrieval::AggregateMetrics::field_by_name` — the
  transitional helper flagged for removal in the v0.9.0 entry below. The
  jammi-enterprise Gate now routes its metric selection through its typed
  `MetricName` enum, leaving only test consumers, which iterate over a
  `[(&'static str, f64); 4]` array built from the struct's fields directly.

## v0.9.0 — 2026-05-26

### Changed

- `eval_embeddings`, `eval_inference`, and `eval_compare` return typed
  reports (`EmbeddingEvalReport`, `InferenceEvalReport`,
  `CompareEvalReport`) instead of `serde_json::Value`. Each report carries
  both the aggregate metrics and the per-query / per-record arrays. The
  per-query data is what sample-based statistical rules (Welch's t,
  Mann-Whitney U) consume at gate time; the aggregate is what the catalog
  persists. `EmbeddingEvalReport.aggregate` is `AggregateMetrics` (same
  fields as before); `InferenceEvalReport.aggregate` is the new
  `InferenceAggregate` enum tagged by `task` (`"classification"` carries
  the existing `ClassificationResult` shape; `"ner"` is still gated by
  `EvalTask::Ner`'s not-yet-implemented error). `CompareEvalReport`
  exposes `per_table` — the first entry is the baseline with `delta:
  None`, and every subsequent entry carries `delta: Some(AggregateDelta)`
  with per-metric `absolute` / `relative` values.
- The Python `db.eval_embeddings`, `db.eval_inference`, and
  `db.eval_compare` bindings now return dicts with `aggregate` plus
  `per_query` / `per_record` / `per_table` keys (the JSON shape of the
  new Rust types).
- `jammi_python::convert` replaces `json_to_pydict(serde_json::Value)`
  with a generic `serializable_to_pydict<T: Serialize>` helper so every
  eval entry point routes its typed report through one converter.

### Added

- `AggregateMetrics::field_by_name(&str) -> Option<f64>` (`#[doc(hidden)]`)
  in `jammi-numerics::retrieval`. Transitional helper for the
  jammi-enterprise Gate config; removed in E2 once the Gate switches to a
  typed metric enum.
- `jammi_ai::eval::report` module exporting the new typed report types
  (`EmbeddingEvalReport`, `PerQueryRecord`, `InferenceEvalReport`,
  `InferenceAggregate`, `PerRecordPrediction`, `CompareEvalReport`,
  `TableEvalReport`, `AggregateDelta`, `MetricDelta`).
- `NerMetrics` and `TypeMetrics` now derive `Deserialize` so
  `InferenceAggregate` round-trips through serde.

### Removed

- `jammi_ai::eval::compare` (empty placeholder module with no consumers).
- `jammi_python::convert::json_to_pydict` (subsumed by
  `serializable_to_pydict`).

## v0.8.0 — 2026-05-26

### Added

- `JammiConfig::catalog` and `JammiConfig::broker` fields, both tagged enums:
  - `CatalogConfig::Sqlite { path: Option<PathBuf> }` (default; uses
    `{artifact_dir}/catalog.db` when `path` is `None`) and
    `CatalogConfig::Postgres { url, pool_size, max_lifetime_secs }`.
    `pool_size` defaults to 8; `max_lifetime_secs` is optional and, when
    `Some`, sets `sqlx::PgPool::max_lifetime` to limit per-connection
    lifetime behind connection-pooling proxies (PgBouncer, RDS Proxy).
  - `BrokerConfig::InMemory` (default) and
    `BrokerConfig::JetStream { url, retention_seconds, credentials_path }`.
    `retention_seconds` defaults to 7 days; `credentials_path` is optional
    and selects authenticated vs anonymous NATS connection.
- `JammiConfig::load` runs `${VAR}` env-var interpolation on the raw TOML
  source before parsing. A missing variable is a typed
  `JammiError::Config` (no silent empty substitution); `$$` escapes a
  literal `$`; unterminated `${` is a typed error.
- `CatalogBackend::ping(&self) -> Result<(), BackendError>` plus
  per-backend implementations and a `Catalog::ping` thin wrapper. The
  primitive runs `SELECT 1` against the connection pool and classifies
  pool failures as `BackendError::Unavailable`. Cost is microseconds
  against a warm pool. Consumed by the OSS server's `/readyz` route.
- `BackendImpl::sqlite_from_path` and
  `BackendImpl::postgres_from_url(url, pool_size, max_lifetime_secs)`
  factories. The session resolver (`JammiSession::new`,
  `JammiSession::with_broker`, `JammiSession::with_backend`) reaches for
  these so a caller that overrides one dimension keeps the other
  config-driven.
- `JetStreamBroker::connect_with_credentials(url, retention_seconds, &Path)`
  for SaaS deployments where the broker rejects anonymous connections.
  Internally a `from_client` helper DRYs the two constructors so they
  agree on the schemas-cache and `JetStreamContext` derivation.
- `crates/jammi-db/examples/sample-postgres.toml` demonstrating a
  Postgres + JetStream production config.
- `docs/guide/src/catalog-and-broker.md` covering the TOML schema, the
  env-var interpolation rules, the SQLite/Postgres trade-off matrix,
  and the broker selection rationale.
- New integration test file `crates/jammi-db/tests/it/catalog_ping.rs`
  exercising `Catalog::ping` for SQLite (happy path, idempotency, arc
  lifetime) plus a Postgres lane behind `live-postgres-tests` with a
  happy-path and an unreachable-URL negative test.
- **`jammi-server` OSS binary.** The `jammi-server` crate gains a
  `[[bin]]` target. The binary loads `JammiConfig` from
  `--config`/`JAMMI_CONFIG`/the platform default, initialises tracing
  per the resolved logging configuration, and hands control to
  `jammi_server::runtime::OssServer`. The orchestration is one module
  (`src/runtime.rs`) — the Axum side-channel and the Tonic chain are
  wired together with one `tokio::sync::broadcast` channel for graceful
  shutdown.
- **Container image.** A `Dockerfile` at the workspace root builds a
  stripped distroless image (`gcr.io/distroless/cc-debian12`) from the
  CI base toolchain. The image runs as the nonroot user (uid `65532`),
  exposes `:8080` (HTTP side-channel) and `:8081` (gRPC + Flight SQL),
  and declares `/var/lib/jammi` as a volume. CI publishes to
  `ghcr.io/f-inverse/jammi-ai-server` on every `v*` tag via
  `.github/workflows/server-image.yml`.
- **Health endpoints.** The HTTP side-channel exposes `/healthz`
  (liveness; returns `{"status":"ok","version":"<crate version>"}`),
  `/readyz` (readiness; pings the catalog backend via
  `Catalog::ping().await` and returns 503 on failure), and `/metrics`
  (Prometheus text-format snapshot of the substrate counters: gRPC
  requests, Flight SQL queries, eval invocations, and a search-latency
  histogram).
- **`runtime::serve_grpc_chain` test-fixture entry-point.** The same
  chain builder the binary uses, exposed for integration tests that
  need to wire pre-seeded sessions to a unified Flight SQL + gRPC
  server on one port.
- **`InferenceSession::tenant_binding_arc`** accessor — required by the
  OSS server so the Flight SQL `TenantBoundProvider` can bind tenants
  for the duration of each query.
- **`cookbook/` tree at the repo root** — OSS source of truth for runnable
  Python recipes against `jammi-ai`. Layout:
  - `cookbook/README.md` — index
  - `cookbook/quickstart/` — 5-minute walkthrough (`README.md`,
    `01_install.md` .. `04_vector_search.md`, runnable `quickstart.py`)
  - `cookbook/recipes/{mutable_tables, trigger_streams, eval_embeddings,
    eval_inference, fine_tune, flight_sql}/{README.md, example.py}`
  - `cookbook/fixtures/` — deterministic `tiny_corpus.parquet`,
    `tiny_golden.json`, `tiny_labels.csv`, `tiny_pairs.csv` plus
    `generate.py`; `tiny_bert/` and `tiny_modernbert_classifier/` model
    fixtures moved here from `tests/fixtures/`. Total tree < 250 KB.
  Every recipe runs against the local fixture model so CI is hermetic;
  every recipe's README has a "When to use this pattern" callout.
- `tests/cookbook_smoke.py` — smoke runner that times every recipe,
  fails the build if `quickstart.py` exceeds 60s wall-clock, excludes
  `fine_tune` and `flight_sql` by default, surfaces them behind
  `JAMMI_COOKBOOK_SLOW=1`.
- `.github/workflows/cookbook.yml` — per-PR fast lane plus nightly
  cron that sets `JAMMI_COOKBOOK_SLOW=1` and builds
  `target/release/jammi` for the Flight SQL recipe.
- `docs/guide/src/cookbook-recipes.md` — mdBook entry that
  `{{#include}}`s every recipe README so the rendered guide and the
  OSS recipes never drift apart.
- `jammi_test_utils::{cookbook_fixture, cookbook_fixture_url,
  cookbook_fixtures_dir}` — first-class helper for the cookbook
  fixtures path. Every integration test that consumed `tiny_bert` /
  `tiny_modernbert_classifier` now reads from `cookbook/fixtures/`.

### Changed

- The CI `test-pg` job now uses `postgres:16` (was `postgres:15`),
  matching the spec's pinned base image.
- `README.md` quickstart section collapsed to a 10-line inline example
  with a link to `cookbook/quickstart/` for the full walkthrough — the
  cookbook tree is the single source of truth.
- `docs/guide/src/quickstart-python.md` rewritten as a stub that
  `{{#include}}`s the cookbook quickstart README.

### Removed

- `tests/cookbook_smoke_test.py` (legacy file from before the cookbook
  tree existed; used a non-existent `add_source(path=...)` kwarg and was
  never wired into CI). Superseded by `tests/cookbook_smoke.py`.
- `tests/fixtures/tiny_bert/` — relocated to `cookbook/fixtures/tiny_bert/`.
- `tests/fixtures/tiny_modernbert_classifier/` — relocated to
  `cookbook/fixtures/tiny_modernbert_classifier/`.

### Fixed

- `ResultStore::create_table` now sanitises `.` characters in the model id
  alongside `/`, `:`, and spaces. A dot in the embedded model-id path
  (e.g. `local:/foo/.cache/model`) survived the previous sanitiser and
  produced a result-table name like `foo__model_.cache__...`, which
  `Path::with_extension("")` then mis-parsed when the sidecar layout
  derived the on-disk stem — the trailing `.cache__...` component was
  treated as an extension and stripped, so the `.usearch` / `.rowmap` /
  `.manifest.json` siblings were written under a truncated name.
  Affected any deployment whose model-id path contained a dot.
- `docs/guide/src/installation.md` — Python install snippet was still
  `pip install jammi`; corrected to `pip install jammi-ai` (post-S1
  rename).
- `docs/guide/src/introduction.md` — three-ways-to-use table row for
  Python was still `pip install jammi`; corrected to
  `pip install jammi-ai`.

### Breaking

- `CatalogBackend` trait grew a new required method `ping`. Any
  out-of-tree implementor must add it; the workspace has no such
  callers.
- `PostgresBackend::open(url)` is renamed to
  `PostgresBackend::open_with_options(url, pool_size, max_lifetime_secs)`.
  The previous signature hardcoded `max_connections = 8` and did not
  expose connection lifetime; both knobs are now caller-supplied. No
  shim is provided.
- **`/health` renamed to `/healthz`.** The HTTP side-channel's liveness
  endpoint moves to the Kubernetes convention. No shim is provided —
  callers update in lockstep.
- **`jammi_server::serve_grpc_with_shutdown` removed.** The function is
  superseded by `jammi_server::runtime::OssServer::run` (binary
  entry-point) and `jammi_server::runtime::serve_grpc_chain` (test
  fixture). Migrate via the `serve_grpc_chain` helper which takes the
  same arguments plus the Flight SQL `SessionContext` and
  `TenantBinding`.

### Migration

```rust
// before
use jammi_db::catalog::backend_postgres::PostgresBackend;
let pg = PostgresBackend::open("postgres://...").await?;
// after
let pg = PostgresBackend::open_with_options("postgres://...", 8, None).await?;
```

```toml
# before — JammiConfig::default() implicitly chose SQLite + InMemory
[artifact_dir]
# ...

# after — both selections are explicit (and a missing `[catalog]` / `[broker]`
# stanza still defaults to SQLite + InMemory)
[catalog]
kind = "postgres"
url = "${POSTGRES_URL}"
pool_size = 16

[broker]
kind = "jet_stream"
url = "nats://${NATS_HOST}:4222"
credentials_path = "/var/run/secrets/nats.creds"
```

Out-of-tree `CatalogBackend` impls must add the new method:

```rust
fn ping(&self) -> Pin<Box<dyn Future<Output = Result<(), BackendError>> + Send + '_>> {
    Box::pin(async move {
        sqlx::query("SELECT 1").execute(&self.pool).await.map_err(classify)?;
        Ok(())
    })
}
```

```rust
// before
jammi_server::serve_grpc_with_shutdown(addr, store, Some(trigger), shutdown).await?;
// after
jammi_server::runtime::serve_grpc_chain(
    addr,
    session.context().clone(),
    session.tenant_binding_arc(),
    store,
    Some(trigger),
    shutdown,
)
.await?;
```

```bash
# Liveness probe URL.
# before
curl http://localhost:8080/health
# after
curl http://localhost:8080/healthz
```

