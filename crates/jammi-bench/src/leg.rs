//! The leg: one run of one rung of a workload — the one type every producer
//! fills and the ladder's comparator reads.
//!
//! A leg is four parts. Its *payload* is what is specific to the workload —
//! the identity fields two legs must agree on to be comparable, and the
//! workload's own measurements beside them; each payload type declares its
//! identity once, in [`Payload::IDENTITY_FIELDS`], and the comparator holds
//! every producer's leg — a PyTorch script's included — to that one
//! declaration. Its *provenance* is recorded and never compared: the device,
//! the build, which kernel arm the leg claims. Its *measurements* are what
//! every axis of the ladder reads: the per-iteration time series, the peaks
//! of the two memory instruments, and the outcome in whichever form the
//! workload's edge kind pairs. Its *facts* are what a rung's premises check
//! the claim against.
//!
//! A `jammi-bench` producer serializes `Leg<ItsPayload>` flat under its tier
//! key, `tiers.<key>`; any other producer writes the same flat shape under
//! `<key>` at the top level of its JSON. The comparator reads either as
//! [`Leg<Fields>`], the payload as a map, and a leg's *role* — which rung,
//! which unit of the sweep, which take — is its file name,
//! `<rung>__<unit>__<take>.json`, because one producer can serve several
//! rungs and only the script that launched it knows which.

use std::borrow::Cow;
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::report::Nullable;

/// The workload-specific part of a leg, with its identity declared once.
pub trait Payload: Serialize {
    /// What two legs must agree on to be comparable, by field name, with
    /// what a null reading means where null is a value.
    const IDENTITY_FIELDS: &'static [(&'static str, Nullable)];
}

/// A measured quantity with its unit; `value: None` is "not measured". Read
/// back as either the slot or a bare number.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Measurement {
    pub value: Option<f64>,
    pub unit: Cow<'static, str>,
}

impl Measurement {
    pub fn not_yet_measured(unit: &'static str) -> Self {
        Self {
            value: None,
            unit: Cow::Borrowed(unit),
        }
    }

    pub fn measured(value: f64, unit: &'static str) -> Self {
        Self {
            value: Some(value),
            unit: Cow::Borrowed(unit),
        }
    }
}

impl<'de> Deserialize<'de> for Measurement {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Read {
            Bare(f64),
            Slot {
                value: Option<f64>,
                #[serde(default)]
                unit: Option<String>,
            },
        }
        Ok(match Read::deserialize(deserializer)? {
            Read::Bare(value) => Self {
                value: Some(value),
                unit: Cow::Borrowed(""),
            },
            Read::Slot { value, unit } => Self {
                value,
                unit: unit.map_or(Cow::Borrowed(""), Cow::Owned),
            },
        })
    }
}

/// A mutant leg's own statement of which patch produced it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MutantStamp {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mutant_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mutant_base_sha: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mutant_patch_sha256: Option<String>,
}

/// Recorded on every leg the engine produces, never compared.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// The concrete device sub-class the leg resolved to.
    pub device_name: String,
    /// The cargo features the binary was built with.
    pub build_features: Vec<String>,
    /// Whether the flash cascade was compiled in.
    pub flash_compiled: bool,
    /// What `JAMMI_KERNELS_DISABLE` named this process, sorted.
    pub kernels_disabled_requested: Vec<String>,
    /// Which of those keys disabled a live dispatch, sorted.
    pub kernels_disabled_fired: Vec<String>,
    /// The kernel arm the leg claims — `fused`, or `alloff` for an arm with
    /// families off — which its dispatch counters prove.
    pub arm: String,
    /// The attention reference class the leg resolved to: `eager` or
    /// `fused`.
    pub attention_arm: String,
    #[serde(flatten)]
    pub mutant: MutantStamp,
}

/// One held-out evaluation along a training run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryPoint {
    pub epoch: usize,
    pub held_out_mean: f64,
    /// Cumulative training wall seconds when this evaluation was taken.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub train_wall_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held_out_tie_fraction: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held_out_batch_partition_sha256: Option<String>,
}

fn unmeasured_bytes() -> Measurement {
    Measurement::not_yet_measured("bytes")
}

/// What every axis reads. A quantity a workload does not produce is absent,
/// never null.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Measured {
    /// Post-warmup wall seconds of each timed iteration, in run order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iter_wall_s: Option<Vec<f64>>,
    /// The size this leg's cost scales with — rows encoded or trained on,
    /// edges × hops propagated, edges walked — where a sweep varies it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub work: Option<f64>,
    /// The kernel's high-water mark for the process.
    #[serde(default = "unmeasured_bytes")]
    pub peak_rss_bytes: Measurement,
    /// The whole-device sampler's high-water mark above the resident
    /// baseline.
    #[serde(default = "unmeasured_bytes")]
    pub peak_vram_bytes: Measurement,
    /// Digest of the artifact, for exact and revision edges.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome_digest: Option<String>,
    /// Final held-out loss, for seed-paired edges.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held_out_example_mean: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trajectory: Vec<TrajectoryPoint>,
    /// Per-row vectors beside the leg: little-endian `f32`, row-major, in
    /// committed key order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vectors_file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector_dim: Option<usize>,
    /// Counts per category per cell of the workload's law, in the law
    /// file's order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub law_observed: Option<Vec<Vec<u64>>>,
}

impl Default for Measured {
    fn default() -> Self {
        Self {
            iter_wall_s: None,
            work: None,
            peak_rss_bytes: unmeasured_bytes(),
            peak_vram_bytes: unmeasured_bytes(),
            outcome_digest: None,
            held_out_example_mean: None,
            trajectory: vec![],
            vectors_file: None,
            vector_dim: None,
            law_observed: None,
        }
    }
}

/// The counted facts behind a training leg's kernel-arm claim: one
/// fused/fallback pair per counted family, the flash cascade's fallback
/// being a declined admission. A pair a leg does not carry reads as zero —
/// a tower with no such seam never dispatches it — and a leg that carries
/// no pair at all has no counters (see [`Facts::dispatch`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DispatchCounters {
    pub ln_fused_dispatches: u64,
    pub ln_eager_dispatches: u64,
    pub rope_fused_dispatches: u64,
    pub rope_eager_dispatches: u64,
    pub softmax_fused_dispatches: u64,
    pub softmax_eager_dispatches: u64,
    pub geglu_fused_dispatches: u64,
    pub geglu_eager_dispatches: u64,
    pub gelu_fused_dispatches: u64,
    pub gelu_eager_dispatches: u64,
    pub lora_epilogue_fused_dispatches: u64,
    pub lora_epilogue_eager_dispatches: u64,
    pub lora_linear_fused_dispatches: u64,
    pub lora_linear_eager_dispatches: u64,
    pub attention_block_fused_dispatches: u64,
    pub attention_block_eager_dispatches: u64,
    pub adamw_fused_dispatches: u64,
    pub adamw_eager_dispatches: u64,
    pub attention_block_flash_fused_dispatches: u64,
    pub attention_block_flash_declined_dispatches: u64,
}

impl DispatchCounters {
    /// Every counter base with its fused and fallback counts.
    pub fn pairs(&self) -> [(&'static str, u64, u64); 10] {
        [
            ("ln", self.ln_fused_dispatches, self.ln_eager_dispatches),
            (
                "rope",
                self.rope_fused_dispatches,
                self.rope_eager_dispatches,
            ),
            (
                "softmax",
                self.softmax_fused_dispatches,
                self.softmax_eager_dispatches,
            ),
            (
                "geglu",
                self.geglu_fused_dispatches,
                self.geglu_eager_dispatches,
            ),
            (
                "gelu",
                self.gelu_fused_dispatches,
                self.gelu_eager_dispatches,
            ),
            (
                "lora_epilogue",
                self.lora_epilogue_fused_dispatches,
                self.lora_epilogue_eager_dispatches,
            ),
            (
                "lora_linear",
                self.lora_linear_fused_dispatches,
                self.lora_linear_eager_dispatches,
            ),
            (
                "attention_block",
                self.attention_block_fused_dispatches,
                self.attention_block_eager_dispatches,
            ),
            (
                "adamw",
                self.adamw_fused_dispatches,
                self.adamw_eager_dispatches,
            ),
            (
                "attention_block_flash",
                self.attention_block_flash_fused_dispatches,
                self.attention_block_flash_declined_dispatches,
            ),
        ]
    }
}

/// What a rung's premises read.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Facts {
    /// The train-side probe loss, anchored at the untrained model: one entry
    /// before training and one per epoch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub train_probe_series: Option<Vec<f64>>,
    /// Whether variable-length rows took the dense transport.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub admission_is_dense: Option<bool>,
    /// The fraction of held-out examples whose loss tied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tie_fraction: Option<f64>,
    /// `None` on a leg that carries no `*_fused_dispatches` pair at all.
    #[serde(default, flatten, skip_serializing_if = "Option::is_none")]
    pub dispatch: Option<DispatchCounters>,
}

/// The suffix every dispatch counter's fused count is reported under.
pub const FUSED_DISPATCHES: &str = "_fused_dispatches";

/// One run of one rung. Serialized flat: the four parts share one object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Leg<P> {
    #[serde(flatten)]
    pub payload: P,
    /// Absent on a leg another framework produced.
    #[serde(flatten)]
    pub provenance: Option<Provenance>,
    #[serde(flatten)]
    pub measured: Measured,
    #[serde(flatten)]
    pub facts: Facts,
}

impl<P: Payload> Leg<P> {
    /// A jammi leg: every part filled.
    pub fn new(payload: P, provenance: Provenance, measured: Measured, facts: Facts) -> Self {
        Self {
            payload,
            provenance: Some(provenance),
            measured,
            facts,
        }
    }

    /// The leg as JSON, with every identity field present as its payload
    /// declares — a producer's own check on every emit.
    pub fn to_value(&self) -> Value {
        let value = serde_json::to_value(self).expect("serialize leg");
        crate::report::assert_identity_fields_present(&value, P::IDENTITY_FIELDS);
        value
    }
}

/// A payload read by the comparator: every field of the block, by name.
pub type Fields = BTreeMap<String, Value>;

impl Leg<Fields> {
    /// Read any producer's flat leg block. Every absent counter pair reads
    /// as zero, so a block carrying no pair at all is told apart here: it
    /// has no counters, not zero of each.
    pub fn read(block: &Value) -> Result<Self, serde_json::Error> {
        let mut leg: Self = serde_json::from_value(block.clone())?;
        if !leg
            .payload
            .keys()
            .any(|key| key.ends_with(FUSED_DISPATCHES))
        {
            leg.facts.dispatch = None;
        }
        Ok(leg)
    }
}

impl Payload for Fields {
    const IDENTITY_FIELDS: &'static [(&'static str, Nullable)] = &[];
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[derive(Serialize, Deserialize)]
    struct Batch {
        seed: u64,
        batch: usize,
    }

    impl Payload for Batch {
        const IDENTITY_FIELDS: &'static [(&'static str, Nullable)] =
            &[("seed", Nullable::NonNull), ("batch", Nullable::NonNull)];
    }

    fn provenance() -> Provenance {
        Provenance {
            device_name: "cpu".into(),
            build_features: vec!["cuda".into()],
            flash_compiled: false,
            kernels_disabled_requested: vec![],
            kernels_disabled_fired: vec![],
            arm: "fused".into(),
            attention_arm: "eager".into(),
            mutant: MutantStamp::default(),
        }
    }

    #[test]
    fn a_leg_serializes_flat_and_reads_back_as_fields() {
        let leg = Leg::new(
            Batch { seed: 7, batch: 16 },
            provenance(),
            Measured {
                iter_wall_s: Some(vec![0.1, 0.2]),
                work: Some(16.0),
                peak_rss_bytes: Measurement::measured(1024.0, "bytes"),
                outcome_digest: Some("d".into()),
                ..Default::default()
            },
            Facts::default(),
        );
        let value = leg.to_value();
        assert_eq!(value["seed"], 7);
        assert_eq!(value["device_name"], "cpu");
        assert_eq!(value["peak_rss_bytes"]["value"], 1024.0);
        assert_eq!(value["peak_vram_bytes"]["value"], Value::Null);
        assert!(value.get("mutant_id").is_none());
        assert!(value.get("ln_fused_dispatches").is_none());

        let read = Leg::<Fields>::read(&value).unwrap();
        assert_eq!(read.payload["seed"], 7);
        assert_eq!(read.provenance, Some(provenance()));
        assert_eq!(read.measured.iter_wall_s.as_deref(), Some(&[0.1, 0.2][..]));
        assert_eq!(read.measured.peak_rss_bytes.value, Some(1024.0));
        assert_eq!(read.measured.peak_vram_bytes.value, None);
        assert_eq!(read.facts.dispatch, None);
    }

    /// Another framework's leg carries the payload and measurements and no
    /// engine provenance; a bare number is a measurement too.
    #[test]
    fn a_leg_without_provenance_reads_and_a_bare_number_is_a_measurement() {
        let read: Leg<Fields> = serde_json::from_value(json!({
            "seed": 7, "batch": 16, "iter_wall_s": [0.1], "peak_rss_bytes": 2048
        }))
        .unwrap();
        assert_eq!(read.provenance, None);
        assert_eq!(read.measured.peak_rss_bytes.value, Some(2048.0));
    }

    #[test]
    fn a_dispatch_pair_a_leg_does_not_carry_reads_as_zero() {
        let read: Leg<Fields> = serde_json::from_value(json!({
            "ln_fused_dispatches": 9, "ln_eager_dispatches": 1
        }))
        .unwrap();
        let counters = read.facts.dispatch.unwrap();
        assert_eq!(
            (counters.ln_fused_dispatches, counters.ln_eager_dispatches),
            (9, 1)
        );
        assert_eq!(counters.gelu_fused_dispatches, 0);
    }
}
