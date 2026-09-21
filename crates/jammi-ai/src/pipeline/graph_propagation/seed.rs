//! The structural seed — initial features generated from the graph itself, for
//! nodes that carry no content an encoder could read.
//!
//! A node's seed row is a row of a very sparse random projection matrix
//! (Achlioptas; Li et al.), scaled by a power of the node's degree:
//!
//! ```text
//! X⁽⁰⁾[j, i] = d̃ⱼ^β · rᵢ      rᵢ = +√s  with probability 1/(2s)
//!                                    0   with probability 1 − 1/s
//!                                  −√s  with probability 1/(2s)
//! ```
//!
//! # A row is a function of its key, never of its position
//!
//! `rᵢ` is drawn from a stream keyed by `(seed, node key)` alone — FNV-1a over
//! the key bytes, mixed with the seed, through one SplitMix64 round
//! ([`seed_for_target`]) into the state of a [`SplitMix64`] the row then reads
//! `dimensions` draws from, one per lane, in lane order. So the projection row
//! depends on `(seed, key, dimensions, sparsity)` and on nothing else: not the
//! row order, not the partitioning, not which other nodes exist. Adding a node
//! to the graph leaves every other node's projection row bit-identical, and a
//! `d`-wide row is a prefix of the wider one under the same seed.
//!
//! The derivation is sound for a projection because SplitMix64's output
//! function is a bijective finaliser over a Weyl sequence: two keys collide
//! only if their 64-bit states do (≈ `n²/2⁶⁵` for `n` keys), and two states
//! that merely fall near each other on the sequence yield rows that are lagged
//! copies — lane-wise independent, which is all a random projection's
//! distance-preservation argument asks of its rows. The stream is
//! domain-separated ([`STREAM_DOMAIN`]) from the neighbour-sampling streams
//! that share the mixer.
//!
//! # The degree scale
//!
//! `d̃ⱼ` is the node's augmented degree in the propagation's own adjacency
//! (its neighbours plus the self-loop), so the scale and the operator agree on
//! what a degree is. FastRP writes the scale as `(dⱼ/2m)^β`; the `(2m)^β`
//! factor is one positive scalar over the whole table — invisible to cosine
//! search and cancelled exactly by the readout's per-block normalisation — and
//! carrying it would make every row a function of the whole graph's size. It
//! is left out, which is what keeps a row local to its node.
//!
//! `d̃^β` is the one step here that is not an IEEE-754 basic operation: it is
//! the platform's `pow`. At `β = 0` it is exactly `1`.
//!
//! # References
//! - Achlioptas 2003, *Database-friendly random projections*:
//!   <https://doi.org/10.1016/S0022-0000(03)00025-4>
//! - Li, Hastie, Church 2006, *Very sparse random projections*:
//!   <https://doi.org/10.1145/1150402.1150436>
//! - Chen et al. 2019, *Fast and Accurate Network Embeddings via Very Sparse
//!   Random Projection (FastRP)*: <https://arxiv.org/abs/1908.11512>

use jammi_db::error::{JammiError, Result};

use crate::pipeline::graph_neighbourhood::{seed_for_target, SplitMix64};

/// Separates the structural-seed streams from every other stream derived
/// through [`seed_for_target`] under the same caller seed.
const STREAM_DOMAIN: u64 = 0x5EED_0F57_2C70_0001;

/// The parameters of a structural seed. Construct through [`SeedSpec::new`],
/// which validates them.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SeedSpec {
    seed: u64,
    dimensions: usize,
    sparsity: f64,
    beta: f64,
}

impl SeedSpec {
    /// Refuses ([`JammiError::Config`]) zero `dimensions`, a `sparsity` that is
    /// not a finite number `≥ 1` (at `s = 1` the projection is dense `±1`), and
    /// a non-finite `beta`.
    pub fn new(seed: u64, dimensions: usize, sparsity: f64, beta: f64) -> Result<Self> {
        if dimensions == 0 {
            return Err(JammiError::Config(
                "a structural seed needs at least one dimension".into(),
            ));
        }
        if !(sparsity.is_finite() && sparsity >= 1.0) {
            return Err(JammiError::Config(format!(
                "a structural seed's sparsity must be a finite number >= 1, got {sparsity}"
            )));
        }
        if !beta.is_finite() {
            return Err(JammiError::Config(format!(
                "a structural seed's degree exponent must be finite, got {beta}"
            )));
        }
        Ok(Self {
            seed,
            dimensions,
            sparsity,
            beta,
        })
    }

    /// The row width.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// The seed row of the node `key` whose augmented degree is `degree`,
    /// appended to `out`.
    pub(crate) fn row_into(&self, key: &str, degree: u64, out: &mut Vec<f64>) {
        let magnitude = self.sparsity.sqrt() * (degree as f64).powf(self.beta);
        let positive = 1.0 / (2.0 * self.sparsity);
        let nonzero = 1.0 / self.sparsity;
        let mut stream = SplitMix64::new(seed_for_target(self.seed ^ STREAM_DOMAIN, key));
        out.extend((0..self.dimensions).map(|_| {
            let u = stream.next_f64();
            if u < positive {
                magnitude
            } else if u < nonzero {
                -magnitude
            } else {
                0.0
            }
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(spec: &SeedSpec, key: &str, degree: u64) -> Vec<f64> {
        let mut out = Vec::new();
        spec.row_into(key, degree, &mut out);
        out
    }

    #[test]
    fn a_row_is_a_function_of_seed_and_key_and_a_prefix_of_the_wider_row() {
        let narrow = SeedSpec::new(7, 64, 3.0, 0.0).unwrap();
        let wide = SeedSpec::new(7, 256, 3.0, 0.0).unwrap();
        assert_eq!(row(&narrow, "acct-17", 4), row(&narrow, "acct-17", 4));
        assert_eq!(row(&narrow, "acct-17", 4), row(&wide, "acct-17", 4)[..64]);
        assert_ne!(row(&narrow, "acct-17", 4), row(&narrow, "acct-18", 4));
        let reseeded = SeedSpec::new(8, 64, 3.0, 0.0).unwrap();
        assert_ne!(row(&narrow, "acct-17", 4), row(&reseeded, "acct-17", 4));
    }

    #[test]
    fn the_degree_only_rescales_the_row() {
        let spec = SeedSpec::new(7, 128, 3.0, -0.5).unwrap();
        let (low, high) = (row(&spec, "n", 1), row(&spec, "n", 4));
        // d̃^β at β = −½: degree 4 halves the magnitude, the signs are the key's.
        for (l, h) in low.iter().zip(&high) {
            assert_eq!(*h, l * 0.5);
        }
    }

    #[test]
    fn entries_follow_the_very_sparse_distribution() {
        let s = 3.0_f64;
        let spec = SeedSpec::new(11, 4096, s, 0.0).unwrap();
        let mut all = Vec::new();
        for node in 0..64 {
            spec.row_into(&format!("node-{node}"), 1, &mut all);
        }
        let n = all.len() as f64;
        let nonzero = all.iter().filter(|v| **v != 0.0).count() as f64 / n;
        let positive = all.iter().filter(|v| **v > 0.0).count() as f64 / n;
        let variance = all.iter().map(|v| v * v).sum::<f64>() / n;
        assert!(all.iter().all(|v| *v == 0.0 || v.abs() == s.sqrt()));
        // 262 144 draws: a fraction's standard error is below 0.001.
        assert!((nonzero - 1.0 / s).abs() < 0.01, "nonzero share {nonzero}");
        assert!(
            (positive - 1.0 / (2.0 * s)).abs() < 0.01,
            "positive share {positive}"
        );
        assert!(
            (variance - 1.0).abs() < 0.03,
            "unit variance, got {variance}"
        );
    }

    #[test]
    fn invalid_parameters_are_refused() {
        for bad in [
            SeedSpec::new(0, 0, 3.0, 0.0),
            SeedSpec::new(0, 8, 0.5, 0.0),
            SeedSpec::new(0, 8, f64::NAN, 0.0),
            SeedSpec::new(0, 8, f64::INFINITY, 0.0),
            SeedSpec::new(0, 8, 3.0, f64::NAN),
        ] {
            assert!(matches!(bad, Err(JammiError::Config(_))));
        }
    }
}
