//! A [`VarBuilder`] whose initial weights are a pure function of a seed.
//!
//! candle's CPU `rand_uniform`/`randn` draw from the process-global
//! `rand::rng()` and `Device::set_seed` does not reach it, so a module built
//! over a plain [`VarMap`] starts from different weights in every process —
//! two runs of one training job publish different bytes. [`seeded_var_builder`]
//! is the same builder over the same map with the draws owned here instead:
//! every parameter a module asks for is filled on the host from a SplitMix64
//! stream keyed by `(seed, fully-qualified parameter name)`, then registered
//! in the map as the trainable `Var` the optimizer steps. Keying by name,
//! never by construction or map-iteration order, makes a parameter's initial
//! value independent of which other parameters exist or when they were built,
//! so the same job starts from byte-identical weights on whichever process
//! trains it.
//!
//! The distributions are the ones the module's own [`Init`] hints name —
//! constant, uniform, normal, Kaiming in either form — so a module
//! initialises as its author specified, only reproducibly.

use candle_core::{DType, Device, Shape, Tensor, Var};
use candle_nn::init::NormalOrUniform;
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Init, VarBuilder, VarMap};

/// A builder over `varmap` whose every initial draw is a pure function of
/// `(seed, parameter name)`. The parameters land in `varmap` exactly as a
/// plain `VarBuilder::from_varmap` would leave them.
pub(crate) fn seeded_var_builder(
    varmap: &VarMap,
    seed: u64,
    dtype: DType,
    device: &Device,
) -> VarBuilder<'static> {
    VarBuilder::from_backend(
        Box::new(SeededVarMap {
            varmap: varmap.clone(),
            seed,
        }),
        dtype,
        device.clone(),
    )
}

/// The backend behind [`seeded_var_builder`]: `varmap`'s parameters, created
/// on first request from the seeded stream.
struct SeededVarMap {
    varmap: VarMap,
    seed: u64,
}

impl SimpleBackend for SeededVarMap {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        init: Init,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let mut data = self
            .varmap
            .data()
            .lock()
            .map_err(|_| candle_core::Error::Msg("seeded varmap lock poisoned".into()))?;
        if let Some(existing) = data.get(name) {
            if existing.shape() != &shape {
                candle_core::bail!(
                    "shape mismatch for '{name}': requested {shape:?}, held {:?}",
                    existing.shape()
                );
            }
            return Ok(existing.as_tensor().clone());
        }
        let values = initial_values(init, &shape, param_seed(self.seed, name));
        let var = Var::from_tensor(&Tensor::from_vec(values, shape, device)?.to_dtype(dtype)?)?;
        let tensor = var.as_tensor().clone();
        data.insert(name.to_string(), var);
        Ok(tensor)
    }

    fn get_unchecked(
        &self,
        name: &str,
        _dtype: DType,
        _device: &Device,
    ) -> candle_core::Result<Tensor> {
        candle_core::bail!(
            "a seeded parameter is created from its shape and init; '{name}' names neither"
        )
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.varmap
            .data()
            .lock()
            .map(|data| data.contains_key(name))
            .unwrap_or(false)
    }
}

/// The host buffer `init` names for a parameter of `shape`, drawn from the
/// stream `seed` starts. Matches `Init::var`'s distributions one for one.
fn initial_values(init: Init, shape: &Shape, seed: u64) -> Vec<f32> {
    let len = shape.elem_count();
    let mut rng = SplitMix64(seed);
    match init {
        Init::Const(value) => vec![value as f32; len],
        Init::Uniform { lo, up } => (0..len).map(|_| rng.uniform(lo, up)).collect(),
        Init::Randn { mean, stdev } => (0..len).map(|_| rng.normal(mean, stdev)).collect(),
        Init::Kaiming {
            dist,
            fan,
            non_linearity,
        } => {
            let std = non_linearity.gain() / (fan.for_shape(shape) as f64).sqrt();
            match dist {
                NormalOrUniform::Uniform => {
                    let bound = 3f64.sqrt() * std;
                    (0..len).map(|_| rng.uniform(-bound, bound)).collect()
                }
                NormalOrUniform::Normal => (0..len).map(|_| rng.normal(0.0, std)).collect(),
            }
        }
    }
}

/// The draw seed of one parameter: FNV-1a over the name's bytes, mixed with
/// the run seed, through one SplitMix64 round.
fn param_seed(seed: u64, name: &str) -> u64 {
    let hash = name.bytes().fold(0xCBF2_9CE4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01B3)
    });
    SplitMix64(hash ^ seed).next_u64()
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform `f64` in `[0, 1)` from the top 53 bits (exact).
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn uniform(&mut self, lo: f64, up: f64) -> f32 {
        (lo + (up - lo) * self.unit()) as f32
    }

    /// Box–Muller's cosine variate; `u1` in `(0, 1]` keeps `ln` finite.
    fn normal(&mut self, mean: f64, stdev: f64) -> f32 {
        let u1 = 1.0 - self.unit();
        let u2 = self.unit();
        let standard = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        (mean + stdev * standard) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{linear, Module};

    fn weights(varmap: &VarMap) -> Vec<(String, Vec<f32>)> {
        let data = varmap.data().lock().unwrap();
        let mut named: Vec<(String, Vec<f32>)> = data
            .iter()
            .map(|(name, var)| {
                (
                    name.clone(),
                    var.as_tensor().flatten_all().unwrap().to_vec1().unwrap(),
                )
            })
            .collect();
        named.sort_by(|a, b| a.0.cmp(&b.0));
        named
    }

    /// Two builds from one seed hold byte-identical parameters whatever order
    /// the layers were built in; another seed holds different ones; every
    /// parameter draws its own stream.
    #[test]
    fn a_seed_fixes_every_parameter_independent_of_build_order() {
        let device = Device::Cpu;
        let build = |seed: u64, reversed: bool| {
            let varmap = VarMap::new();
            let vb = seeded_var_builder(&varmap, seed, DType::F32, &device);
            let mut names = ["phi.fc1", "phi.fc2", "rho.fc1"];
            if reversed {
                names.reverse();
            }
            for name in names {
                linear(8, 4, vb.pp(name)).unwrap();
            }
            varmap
        };

        let first = weights(&build(7, false));
        assert_eq!(first, weights(&build(7, true)));
        assert_ne!(first, weights(&build(8, false)));
        assert_eq!(first.len(), 6, "a weight and a bias per layer");
        assert!(
            first.iter().flat_map(|(_, v)| v).any(|x| *x != 0.0),
            "the draws are not degenerate"
        );
        let distinct: std::collections::BTreeSet<Vec<u32>> = first
            .iter()
            .filter(|(name, _)| name.ends_with(".weight"))
            .map(|(_, v)| v.iter().map(|x| x.to_bits()).collect())
            .collect();
        assert_eq!(distinct.len(), 3, "each parameter draws its own stream");
    }

    /// A layer built through the seeded builder runs, and asking for the same
    /// parameter twice returns the var already held rather than a new draw.
    #[test]
    fn a_held_parameter_is_returned_not_redrawn() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = seeded_var_builder(&varmap, 3, DType::F32, &device);
        let layer = linear(4, 2, vb.pp("head")).unwrap();
        let again = linear(4, 2, vb.pp("head")).unwrap();
        let x = Tensor::ones((1, 4), DType::F32, &device).unwrap();
        assert_eq!(
            layer.forward(&x).unwrap().to_vec2::<f32>().unwrap(),
            again.forward(&x).unwrap().to_vec2::<f32>().unwrap()
        );
        assert_eq!(varmap.all_vars().len(), 2);
        assert!(
            linear(5, 2, vb.pp("head")).is_err(),
            "a held name keeps its shape"
        );
    }
}
