# Changelog

All notable changes to the Jammi AI workspace are recorded here. The
workspace ships every publishable crate at the same
`workspace.package.version`; PyPI `jammi-ai` mirrors that version.

## v0.7.0 — 2026-05-26

### Added

- `jammi-numerics` OSS crate (new workspace member, `crates/jammi-numerics/`),
  consolidating pure-math kernels into one canonical home:
  - **Stats** (`stats::{welch_t_test, mann_whitney_u, bootstrap_ci}`):
    Welch's two-sample t-test with Satterthwaite df + t-distribution
    p-value, Mann-Whitney U with tie-corrected normal-approximation
    p-value, percentile bootstrap CI. Shared `TestResult` and `Interval`
    types.
  - **GP + EI** (`gp::{rbf, rbf_gram, cholesky_with_jitter,
    posterior, expected_improvement, mle_hyperparams}`): RBF kernel,
    Cholesky factoriser with progressive diagonal-jitter retry ladder
    (`1e-10..1e-6` relative to the matrix's mean diagonal), GP
    posterior mean/variance, Expected Improvement acquisition
    (maximisation convention), grid-search hyperparameter MLE.
  - **Divergence** (`divergence::{jensen_shannon, psi, wasserstein_1d,
    cosine_similarity_shift, build_projection_vector, project_all,
    mean_pairwise_cosine, PROJECTION_SEED}`): Jensen-Shannon, PSI, 1D
    Wasserstein, cosine-similarity shift, and the seeded random
    projection helper used by drift monitors.
  - **Histogram** (`histogram::{bin_proportions, smooth_and_renormalise,
    padded_range, interpolate_to, NUM_BINS, RANGE_PADDING}`): binning,
    additive smoothing, padded range, 1-D linear interpolation.
  - **Distance** (`distance::{cosine_distance, cosine_similarity,
    vector_norm}`): vector primitives previously in
    `jammi_db::index::cosine_distance`.
  - **Retrieval / classification / NER eval metrics**
    (`retrieval::{RelevanceJudgment, RetrievalMetrics, QueryMetrics,
    AggregateMetrics}`,
    `classification::{ClassificationMetrics, ClassMetrics,
    ClassificationResult}`,
    `ner::{Entity, NerMetrics, TypeMetrics, decode_bio_spans}`):
    moved from `jammi_ai::eval::metrics::*` and
    `jammi_ai::inference::ner_decode`.
  - **Pareto** (`pareto::{dominates, frontier}`): Pareto-dominance
    primitives.
  - **Typed errors** (`error::NumericsError`): `IllConditioned`,
    `InvalidInput`, `DimensionMismatch`.

  The crate has zero dependency on `jammi-db`, `jammi-ai`,
  `jammi-server`, candle, or any tensor library. Consumers depend on
  it; it never depends on the workspace.

  Determinism contract: every function is single-architecture
  deterministic given the same inputs (and seed where applicable);
  cross-architecture bit-equivalence is not promised. Randomness is
  always seeded via `rand::rngs::StdRng::seed_from_u64`.

### Changed

- `jammi-db`'s in-crate `cosine_distance` function relocated to
  `jammi_numerics::distance::cosine_distance`. Consumers updated
  in-place (`jammi-db::index::exact`, `jammi-ai::tests::it::e2e_inference`).
  No behaviour change.
- `jammi-ai`'s `eval::metrics::{retrieval, classification, ner}` modules
  removed; the runner now imports `ClassificationMetrics` and
  `RetrievalMetrics` directly from `jammi_numerics`. The eval module's
  public `RelevanceJudgment` is re-exported via
  `pub use jammi_numerics::retrieval::RelevanceJudgment;` so external
  Rust callers (including `eval::compare`) continue to resolve.
- `jammi-ai::inference::ner_decode` module removed. The decoder is now
  `jammi_numerics::ner::decode_bio_spans`; the candle backend imports
  it from there.
- The decoder-output `EntitySpan` and the eval-side `EvalEntity` types
  are unified as `jammi_numerics::ner::Entity`. `Entity` carries the
  union of fields (`label`, `start`, `end`, `text`, `confidence`);
  equality and hashing are defined on `(label, start, end)` only so
  the eval metric's strict-matching contract is preserved (the eval
  side constructs gold entities with empty `text` and zero
  `confidence`, the decoder populates both).

### Breaking

These public Rust paths are gone — `jammi_db::index::cosine_distance`,
`jammi_ai::eval::metrics::{retrieval, classification, ner}`,
`jammi_ai::inference::ner_decode`. Their items moved into
`jammi-numerics`. The `EntitySpan` and `EvalEntity` types are also
gone; replaced by a single `jammi_numerics::ner::Entity`. JSON shape
of decoder output is unchanged (same field set).

### Migration

```rust
// before
use jammi_db::index::cosine_distance;
// after
use jammi_numerics::distance::cosine_distance;

// before
use jammi_ai::eval::metrics::retrieval::RetrievalMetrics;
use jammi_ai::eval::metrics::classification::ClassificationMetrics;
use jammi_ai::eval::metrics::ner::{NerMetrics, EvalEntity};
use jammi_ai::inference::ner_decode::{decode_bio_spans, EntitySpan};
// after
use jammi_numerics::retrieval::RetrievalMetrics;
use jammi_numerics::classification::ClassificationMetrics;
use jammi_numerics::ner::{NerMetrics, Entity, decode_bio_spans};
```

## v0.6.0 — 2026-05-25

### Breaking

- `jammi-engine` crate renamed to `jammi-db`. The crate's description
  was retold to reflect what it actually is: a vector database, SQL
  federation, mutable companion tables, and trigger broker. Update
  Cargo deps:

  ```toml
  # before
  jammi-engine = { version = "0.5.9", ... }
  # after
  jammi-db = { version = "0.6.0", ... }
  ```

  And every Rust use-path:

  ```rust
  // before
  use jammi_engine::{config::JammiConfig, session::JammiSession};
  // after
  use jammi_db::{config::JammiConfig, session::JammiSession};
  ```

- Python import: `import jammi` → `import jammi_ai`. The importable
  module name now matches the PyPI distribution name (`jammi-ai`).

  ```python
  # before
  import jammi
  db = jammi.connect(...)
  # after
  import jammi_ai
  db = jammi_ai.connect(...)
  ```

- The PyO3 native extension path moved from `jammi._native` to
  `jammi_ai._native`. No customer-facing impact unless you were
  reaching past the public `__init__.py` to import internals
  directly.

### Migration

```bash
# Cargo deps and Rust source
sed -i 's/jammi-engine/jammi-db/g; s/jammi_engine/jammi_db/g' \
  Cargo.toml $(find . -name '*.rs')

# Python imports
sed -i 's/^import jammi$/import jammi_ai/; s/^from jammi import/from jammi_ai import/' \
  $(find . -name '*.py')

# Python jammi.X attribute access
sed -i 's/jammi\.connect/jammi_ai.connect/g' $(find . -name '*.py')
```

### Notes

- The `0.5.x` line of releases continues to ship the old names on
  crates.io (`jammi-engine`) and PyPI (`jammi-ai<0.6` carries
  `import jammi`). Pinned consumers keep working; no
  backwards-compatibility shim is provided in `0.6.0` per the
  no-shims engineering rule.
- This is a pure rename. No behavioural changes, no schema migrations,
  no on-disk artifact-format changes.
- The DataFusion catalog schema name (`jammi`), the runtime
  `directories::ProjectDirs` identifier (`ai/jammi/jammi`), the
  protobuf wire packages (`jammi.v1.session`, `jammi.v1.trigger`), and
  the `jammi` CLI binary name are unchanged — those are runtime
  identifiers, not Rust crate or Python module names.
