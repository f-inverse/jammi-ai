# jammi-numerics

Numerical kernels for vector search, retrieval, statistics, and Bayesian
optimization. Part of [Jammi AI](https://github.com/f-inverse/jammi-ai).

Pure math, no I/O, no tensor-library dependency. Consumers depend on
this crate; this crate never depends on `jammi-db`, `jammi-ai`,
`jammi-server`, candle, or any tensor library.

## Modules

- `distance` — cosine distance / similarity, vector norm
- `divergence` — Jensen-Shannon, PSI, 1D Wasserstein, cosine-similarity
  shift, with the random-projection helper used by drift monitors
- `histogram` — binning, smoothing, padded range, interpolation
- `stats` — Welch's t-test, Mann-Whitney U test, bootstrap CIs
- `gp` — Gaussian processes: RBF kernel, Cholesky with jitter retry,
  posterior mean/variance, Expected Improvement acquisition,
  grid-search hyperparameter MLE
- `retrieval` — recall@k, precision@k, MRR, NDCG
- `classification` — accuracy, precision, recall, F1, per-class metrics
- `ner` — BIO span decoder and entity-level NER metrics, with a single
  `Entity` type that serves both decode and evaluation
- `pareto` — Pareto dominance and frontier extraction

## Determinism contract

Every function in this crate is single-architecture deterministic given
the same inputs (and, where applicable, the same seed).
Cross-architecture bit-equivalence is **not** promised — floating-point
summation and reduction order can differ between x86_64 and aarch64.
The guarantee is "the same Rust binary on the same architecture always
produces the same result given the same inputs and seed."

Randomness is always seeded via `rand::rngs::StdRng::seed_from_u64`.
Every function consuming randomness takes either an explicit
`&mut StdRng` or a `seed: u64` parameter — never an implicit
`thread_rng()`.

## License

Apache-2.0
