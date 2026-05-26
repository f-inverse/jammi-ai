//! Numerical kernels for vector search, retrieval, statistics, and Bayesian
//! optimization.
//!
//! # Determinism contract
//!
//! Every function in this crate is single-architecture deterministic given
//! the same inputs (and, where applicable, the same seed). Cross-architecture
//! bit-equivalence is NOT promised — floating-point summation order and
//! reduction order can differ between x86_64 and aarch64 builds. The
//! guarantee is: the same Rust binary on the same architecture always
//! produces the same result given the same inputs and seed.
//!
//! Seeded RNG (`rand::rngs::StdRng::seed_from_u64`) is used everywhere
//! randomness appears. Each function that consumes randomness takes either
//! an explicit `&mut StdRng` or a `seed: u64` parameter — never an implicit
//! `thread_rng()`.
//!
//! # Boundaries
//!
//! `jammi-numerics` depends on `nalgebra`, `rand`, `rand_distr`, `statrs`,
//! `serde`, and `thiserror` — no other workspace crate. Downstream consumers
//! (`jammi-db`, `jammi-ai`, `jammi-enterprise`) depend on this crate, never
//! the reverse.

pub mod classification;
pub mod distance;
pub mod divergence;
pub mod error;
pub mod gp;
pub mod histogram;
pub mod ner;
pub mod pareto;
pub mod retrieval;
pub mod stats;

pub use error::{NumericsError, Result};
