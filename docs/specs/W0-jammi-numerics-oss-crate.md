# W0 — `jammi-numerics` OSS crate

**Status:** spec — pending review
**Owner:** TBD
**Estimated effort:** 2 weeks
**Workstream dependencies:** W00 (uses renamed `jammi-db` crate name in deps)
**Workstreams blocked by this:** W0.7, W4, W5, W5.5

## Motivation

Three downstream workstreams (W4 Rule engine, W5 Sampler BO+EI, W5.5 Monitor divergence refactor) need numerical primitives. Building them inline in `jammi-enterprise` then extracting later means writing the math twice, refactoring callers twice, and burying general-utility code in a commercial crate when it passes the OSS discipline test ("would a Jammi user who has never heard of jammi-enterprise/Lace/AccuRisk want this?").

Additionally, divergence kernels (currently in `jammi-enterprise/src/resilience/divergence.rs`) and eval metric kernels (currently in `jammi-ai/src/eval/metrics/*.rs`) are duplicated against the principle they violate today: the divergence math is consumed by both Monitor and Resilience (DRY violation in `jammi-enterprise`); the eval metric kernels are buried inside jammi-ai's EvalRunner orchestrator instead of being substrate primitives. W0 consolidates all pure-math kernels into one OSS crate with consistent naming and a single canonical home.

## Current state (verified at spec time)

**Existing math that moves into `jammi-numerics`:**

| Source location | Functions | LOC | License |
|---|---|---:|---|
| `jammi-enterprise/crates/jammi-enterprise/src/resilience/divergence.rs:266-450` | `project`, `project_all`, `mean_pairwise_cosine`, `vector_norm`, `cosine`, `bin_proportions`, `smooth_and_renormalise`, `padded_range`, `jensen_shannon`, `kl_divergence_log2`, `psi`, `wasserstein_1d`, `interpolate_to`, `cosine_similarity_shift`, `build_projection_vector` + `PROJECTION_SEED` and `NUM_BINS` constants | ~200 (math) | currently closed; **moves to OSS** |
| `jammi-enterprise/tests/it/divergence_unit.rs` | hermetic unit tests for the kernels | 189 | moves with the code |
| `jammi-ai/crates/jammi-ai/src/eval/metrics/retrieval.rs` | `RetrievalMetrics::compute_query`, `QueryMetrics`, `AggregateMetrics` | 121 | already OSS |
| `jammi-ai/crates/jammi-ai/src/eval/metrics/classification.rs` | `ClassificationMetrics::compute`, `ClassMetrics`, `ClassificationResult` | 91 | already OSS |
| `jammi-ai/crates/jammi-ai/src/eval/metrics/ner.rs` | `NerMetrics::compute`, `TypeMetrics`, `EvalEntity` | 257 | already OSS |
| `jammi-ai/crates/jammi-ai/src/inference/ner_decode.rs` | `decode_bio_spans`, `EntitySpan` | ~150 (full file) | already OSS |
| `jammi-ai/crates/jammi-engine/src/index/mod.rs:35-50` | `cosine_distance(a: &[f32], b: &[f32]) -> f32` | ~15 | already OSS |

**Existing math that does NOT move** (rejected by discipline test, recorded for posterity):

| Location | Functions | Why kept |
|---|---|---|
| `jammi-ai/crates/jammi-encoders/src/pooling.rs` | `pool_and_normalize`, `mean_pool`, `cls_pool`, `max_pool`, `weighted_mean_pool`, `l2_normalize` | Tensor-shaped (`candle Tensor` + attention mask); would force jammi-numerics to depend on candle, breaking its "no tensor-library dep" charter |
| `jammi-ai/crates/jammi-ai/src/fine_tune/trainer.rs` | `compute_lr`, `cosent_loss`, `cross_entropy_loss`, `ner_loss`, `triplet_loss`, `cosine_similarity` (Tensor variant) | Same tensor-shape problem; single consumer (trainer); premature extraction per `feedback_no_phantom_user_work` |
| `jammi-ai/crates/jammi-ai/src/model/backend/candle.rs` | numerical helpers | Backend-specific tensor wiring; not shared math |

**Net-new code in `jammi-numerics`:**

| Module | Purpose | Consumer |
|---|---|---|
| `gp/` | Gaussian process: RBF kernel, Cholesky with jitter retry, posterior mean/variance, Expected Improvement acquisition, hyperparameter MLE | W5 (Sampler BO+EI) |
| `stats/` | `welch_t_test(sample_a, sample_b) -> TestResult`, `mann_whitney_u(sample_a, sample_b) -> TestResult`, `bootstrap_ci(sample, statistic_fn, n_resamples, alpha) -> Interval` | W4 (Rule engine) |
| `pareto/` | `dominates(a, b)`, `frontier(points)` | W4 (Rule engine) |

## Crate layout

```
crates/jammi-numerics/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs               // re-exports + crate-level docs
│   ├── divergence/
│   │   ├── mod.rs           // module re-exports
│   │   ├── jensen_shannon.rs
│   │   ├── psi.rs
│   │   ├── wasserstein.rs
│   │   ├── cosine_shift.rs
│   │   └── projection.rs    // random projection helpers + PROJECTION_SEED
│   ├── distance/
│   │   ├── mod.rs
│   │   ├── cosine.rs        // cosine_distance + cosine_similarity (slice flavours)
│   │   └── euclidean.rs     // (future; not in W0)
│   ├── histogram/
│   │   ├── mod.rs
│   │   ├── binning.rs       // bin_proportions + padded_range + smoothing
│   │   └── interpolate.rs   // interpolate_to (used by wasserstein)
│   ├── stats/
│   │   ├── mod.rs
│   │   ├── welch.rs         // welch_t_test
│   │   ├── mannwhitney.rs   // mann_whitney_u
│   │   ├── bootstrap.rs     // bootstrap_ci
│   │   └── types.rs         // TestResult, Interval shared types
│   ├── gp/
│   │   ├── mod.rs
│   │   ├── kernel.rs        // RBF kernel
│   │   ├── cholesky.rs      // jitter-retry wrapper around nalgebra::Cholesky
│   │   ├── posterior.rs     // posterior mean/variance
│   │   ├── ei.rs            // Expected Improvement acquisition
│   │   └── mle.rs           // hyperparameter MLE
│   ├── retrieval/
│   │   ├── mod.rs
│   │   └── metrics.rs       // recall@k, precision@k, mrr, ndcg, QueryMetrics, AggregateMetrics
│   ├── classification/
│   │   ├── mod.rs
│   │   └── metrics.rs       // accuracy, precision, recall, f1, per_class
│   ├── ner/
│   │   ├── mod.rs
│   │   ├── metrics.rs       // entity-level P/R/F1, EvalEntity, TypeMetrics
│   │   └── decoding.rs      // decode_bio_spans, EntitySpan
│   ├── pareto/
│   │   ├── mod.rs
│   │   ├── dominates.rs
│   │   └── frontier.rs
│   └── error.rs             // NumericsError (compute failures, dim mismatch, etc.)
└── tests/
    └── it/
        ├── divergence.rs    // moved from jammi-enterprise/tests/it/divergence_unit.rs
        ├── distance.rs
        ├── histogram.rs
        ├── stats.rs         // Welch / MannWhitney / bootstrap on realistic samples
        ├── gp.rs            // determinism, Branin convergence, edge cases
        ├── retrieval.rs     // moved from jammi-ai metrics tests
        ├── classification.rs
        ├── ner_decoding.rs
        ├── ner_metrics.rs
        └── pareto.rs
```

## Cargo.toml

```toml
[package]
name = "jammi-numerics"
description = "Numerical kernels for vector search, retrieval, statistics, and Bayesian optimization. Pure math, no I/O, no tensor-library dependency."
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true
readme = "README.md"

[dependencies]
# Pure math only. NO candle, NO tonic, NO jammi-db, NO jammi-ai.
nalgebra = "0.34"            # Cholesky + matrix ops for GP
rand = "0.8"                 # seeded RNG for GP candidate generation + bootstrap
rand_distr = "0.4"           # Normal distribution sampling for stats tests
serde = { version = "1", features = ["derive"] }   # TestResult, QueryMetrics, etc. are serializable

[dev-dependencies]
approx = "0.5"                # float-equality assertions in tests
```

Workspace `Cargo.toml`:
- Add `"crates/jammi-numerics"` to `workspace.members` AND `workspace.default-members`
- Add `jammi-numerics = { version = "0.6.0", path = "crates/jammi-numerics" }` to `workspace.dependencies`

## Determinism contract

Every function in `jammi-numerics` is **single-architecture deterministic** given the same inputs (and, where applicable, the same seed). Document in crate-level rustdoc:

> Cross-architecture bit-equivalence is NOT promised. Floating-point summation order in `nalgebra::Cholesky` and accumulator order in our reduction helpers can differ between x86_64 and aarch64. This crate is designed for audit-grade computation — but the audit guarantee is "the same Rust binary on the same architecture always produces the same result," not "every architecture produces identical bits."

Seeded RNG (`StdRng::seed_from_u64`) is used everywhere randomness appears. Each function that consumes randomness takes either an explicit `&mut StdRng` or a `seed: u64` parameter — no implicit `thread_rng()`.

## Test pattern

- Hermetic. No live network. No `#[ignore]`.
- Tests use realistic inputs (real distributions, hand-built sample arrays, known-result benchmarks).
- **Convergence assertions for GP**: 1D quadratic objective `f(x) = -(x - 1.7)^2`, BO with seeded RNG converges within 0.3 of optimum in 30 iterations; 2D Branin function reaches within 5% of known optimum 0.397887 in 50 iterations.
- **Statistical test assertions**: hand-built samples from `Normal(0, 1)` vs `Normal(0.5, 1)` via seeded RNG; Welch's t and Mann-Whitney distinguish them with p < 0.05 at n=30.
- **Divergence assertions**: existing fixtures from `divergence_unit.rs` and `resilience_divergence.rs` move with the code; assertions unchanged (bit-equivalent because same functions).
- **Eval metric assertions**: existing test fixtures from jammi-ai eval tests move with the code.

## Migration plan (within W0's PR)

### Step 1: Create the crate skeleton

- New `crates/jammi-numerics/` with `Cargo.toml`, `README.md`, empty `src/lib.rs` exposing module structure
- Add workspace member + dep entry in root `Cargo.toml`
- Verify `cargo build -p jammi-numerics` succeeds with empty modules

### Step 2: Move divergence kernels from jammi-enterprise

- Copy `jammi-enterprise/crates/jammi-enterprise/src/resilience/divergence.rs:252-450` (the pure-math helpers) into `crates/jammi-numerics/src/divergence/*.rs` + `histogram/binning.rs` + `distance/cosine.rs`
- Copy `jammi-enterprise/tests/it/divergence_unit.rs` into `crates/jammi-numerics/tests/it/divergence.rs`
- Delete the moved functions from the original `divergence.rs`; replace with `pub use jammi_numerics::divergence::*` re-exports... **NO**, per CLAUDE.md no-shims rule: this is W5.5's job in jammi-enterprise. In W0 we only ADD to jammi-numerics; W5.5 deletes the originals and switches callers.

### Step 3: Move eval metric kernels from jammi-ai

- Copy `crates/jammi-ai/src/eval/metrics/{retrieval,classification,ner}.rs` content into `crates/jammi-numerics/src/{retrieval,classification,ner}/metrics.rs`
- Copy `crates/jammi-ai/src/inference/ner_decode.rs` into `crates/jammi-numerics/src/ner/decoding.rs` (rename `EntitySpan` to live alongside `EvalEntity`; both stay because they serve different test contracts)
- **Within W0's PR**, switch jammi-ai's eval runner to consume from `jammi-numerics` directly (delete the old `metrics/` files; update `eval/runner.rs` imports). This IS a jammi-ai workspace internal refactor and lands in the same PR.
- Same for `cosine_distance`: move from `jammi-engine/src/index/mod.rs:35` to `jammi-numerics/src/distance/cosine.rs`; engine's `index/mod.rs` switches to `use jammi_numerics::distance::cosine_distance`.

### Step 4: Implement net-new modules

- `stats/welch.rs`: Welch's t-test formula (degrees of freedom via Satterthwaite approximation; p-value via t-distribution CDF from `statrs` or hand-rolled)

  *Note: we may need `statrs = "0.18"` as a dep for distribution CDFs. Decision: add it. It's a pure-math crate with no tensor dep.*

- `stats/mannwhitney.rs`: U statistic + tie-corrected variance; p-value via normal approximation
- `stats/bootstrap.rs`: resample with replacement; user-provided `statistic_fn`; percentile interval at `alpha` level
- `pareto/`: straightforward implementations
- `gp/kernel.rs`: RBF kernel `k(x, y) = signal_var * exp(-||x-y||^2 / (2 * lengthscale^2))`
- `gp/cholesky.rs`: wrap `nalgebra::Cholesky::new`, retry with increasing jitter on `None` return (up to 5 retries with jitter `1e-10, 1e-9, 1e-8, 1e-7, 1e-6`; final failure → `NumericsError::IllConditioned`)
- `gp/posterior.rs`: posterior mean/variance from fitted GP + new x_star
- `gp/ei.rs`: Expected Improvement formula `EI(x) = (μ(x) - f*) Φ(z) + σ(x) φ(z)` where `z = (μ(x) - f*) / σ(x)`
- `gp/mle.rs`: hyperparameter MLE via grid search (initial implementation; can replace with L-BFGS later)

### Step 5: Tests

- Move existing tests for divergence/eval metrics; they pass byte-identical
- Write net-new tests for stats / GP / Pareto per the patterns above

### Step 6: Publication

- crates.io: publish `jammi-numerics v0.6.0` (matches workspace version after W00 bumps to 0.6.0)
- Update workspace `version` in root `Cargo.toml` to `0.6.0`

## Atomicity

One PR in jammi-ai workspace covering:
- New `jammi-numerics` crate
- Within-workspace cleanup (jammi-ai eval runner + jammi-engine cosine_distance switch to consume from jammi-numerics; delete the old `metrics/` files and the local `cosine_distance` function)

The jammi-enterprise-side consumer switch (delete `divergence.rs` math, replace with imports) lands in **W5.5**, not W0. W0 leaves the duplicate intact temporarily because the bit-equivalent kernels in both places do no harm during the brief window between W0 publishing and W5.5 landing.

## Success criteria

1. `cargo build -p jammi-numerics` succeeds in jammi-ai workspace
2. `cargo test -p jammi-numerics` passes (all moved tests + net-new tests)
3. `cargo test --workspace --exclude jammi-python` passes (jammi-ai's own tests still green; eval runner refactor doesn't regress)
4. `cargo clippy --workspace -- -D warnings` clean
5. No public `cosine_distance` remains in `jammi-engine` (verified via grep)
6. No `eval/metrics/*.rs` files remain in jammi-ai (verified via `find`)
7. crates.io: `jammi-numerics v0.6.0` published
8. Rustdoc on crate root explains the determinism contract and the discipline test that gated extraction choices

## Out of scope

- jammi-numerics PyO3 wheel — defer until there's a Python consumer outside jammi-ai (currently jammi-ai's Python wrapper consumes via the OSS engine; no need)
- L-BFGS for GP hyperparameter MLE — grid search is good enough for MVP
- Cross-architecture bit-equivalence — documented as not promised
- Streaming variants of any kernel — current callers use batch
- Tensor-shaped variants — kept in jammi-encoders / jammi-ai per discipline test

## CLAUDE.md self-check

- [x] Pure functions, no I/O — every kernel is `inputs in, outputs out`
- [x] Clean boundaries — `jammi-numerics` knows nothing about jammi-db / jammi-ai / jammi-enterprise; downstream consumers depend on it, not vice versa
- [x] DRY — divergence kernel duplication (Monitor + Resilience) is resolved; eval kernel "buried in orchestrator" is resolved
- [x] No backwards-compat shims (no re-export modules; consumers update imports atomically per W0 + W5.5)
- [x] Type-driven — `TestResult`, `Interval`, `QueryMetrics`, `EvalEntity` are typed structs with derived `Serialize`
- [x] No band-aids — Cholesky jitter retry is principled (well-known numerical technique), not a `unwrap_or(0.0)` hack
- [x] No tenant coupling — every kernel passes the discipline test ("would a user who has never heard of Lace/AccuRisk/jammi-enterprise want this?")
- [x] Atomic across the workspace — W0 PR includes jammi-ai's internal cleanup; jammi-enterprise consumer cleanup is W5.5 (deliberate split-by-capability per CLAUDE.md, not split-by-crate)
- [x] Scaffolding earned — every module has multiple files (e.g. `gp/{kernel,cholesky,posterior,ei,mle}.rs`); no single-file modules
- [x] Idiom matches surrounding files — `Result` propagation via `?`; `serde::{Serialize, Deserialize}` derive; no panics outside tests
