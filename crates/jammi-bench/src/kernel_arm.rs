//! Fused-kernel families and the arm a rung runs with them.
//!
//! A *family* is one fused kernel and the admission keys that switch it: a
//! rung's kernel arm is the set of families it turns off, stated as data in
//! the ladder's definition, never as a list of op keys typed into a script.
//! The op keys `JAMMI_KERNELS_DISABLE` takes for an arm are *derived*: the
//! families' keys, intersected with the keys one training step on the
//! checkpoint actually consults — its census — so an arm never names an op
//! the checkpoint never dispatches (a checkpoint without a GELU seam has no
//! `gelu_erf_fused` to turn off), which the producer would refuse as a
//! disable that never fired.
//!
//! The census is read off the admission counters after one training step:
//! every call site consults its key before the device is looked at, so a key
//! consulted on the CPU build is a key a disable of it fires on any device.
//! Some call sites are only reached when another family's kernel is off —
//! RoPE and softmax inside the eager attention composition, behind the block
//! kernel; the block kernel and the memory-efficient cascade behind the flash
//! cascade — so the census is taken to a fixpoint: a step with nothing off,
//! then a step with every consulted family that absorbs another turned off,
//! until no new key appears. Each step is its own process, because the
//! disable list is read once per process. An arm that turns a family off
//! while its absorber stays on would name a key that never fires on the
//! device, and the derivation refuses it by name.

use std::collections::BTreeSet;

use jammi_kernels::admission::{self, ProbedOp};
use serde::Serialize;

/// One fused kernel, under the name a rung's arm is stated in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum KernelFamily {
    LayerNorm,
    /// The FlashAttention-2 cascade.
    FlashAttention,
    /// The memory-efficient attention cascade, consulted once flash declines.
    MemoryEfficientAttention,
    /// The whole-attention-block kernel, consulted once both cascades decline.
    AttentionBlock,
    /// RoPE, consulted only inside the eager attention composition.
    Rope,
    /// The last-dimension softmax, consulted only inside the eager attention
    /// composition.
    Softmax,
    Geglu,
    GeluErf,
    /// The fused LoRA site: its cast-boundary sub-kernels are reached only
    /// inside its own admitted branch and are switched with it.
    Lora,
    AdamW,
}

impl KernelFamily {
    pub const ALL: [Self; 10] = [
        Self::LayerNorm,
        Self::FlashAttention,
        Self::MemoryEfficientAttention,
        Self::AttentionBlock,
        Self::Rope,
        Self::Softmax,
        Self::Geglu,
        Self::GeluErf,
        Self::Lora,
        Self::AdamW,
    ];

    /// The admission table rows this family switches.
    pub fn ops(self) -> &'static [&'static ProbedOp] {
        match self {
            Self::LayerNorm => &[&admission::LAYER_NORM],
            Self::FlashAttention => &[&admission::ATTENTION_BLOCK_FLASH],
            Self::MemoryEfficientAttention => &[&admission::MEM_EFFICIENT_ATTENTION],
            Self::AttentionBlock => &[&admission::ATTENTION_BLOCK],
            Self::Rope => &[&admission::ROPE],
            Self::Softmax => &[&admission::SOFTMAX],
            Self::Geglu => &[&admission::GEGLU],
            Self::GeluErf => &[&admission::GELU_ERF],
            Self::Lora => &[&admission::LOW_RANK_RESIDUAL_LINEAR],
            Self::AdamW => &[&admission::ADAMW_STEP],
        }
    }

    /// The keys `JAMMI_KERNELS_DISABLE` takes for this family.
    pub fn keys(self) -> impl Iterator<Item = &'static str> {
        self.ops().iter().flat_map(|op| op.all_registry_keys())
    }

    /// The bases of the `<base>_fused_dispatches` counters a leg reports this
    /// family under. Empty for a family whose dispatches are not counted on
    /// a leg.
    pub fn counters(self) -> &'static [&'static str] {
        match self {
            Self::LayerNorm => &["ln"],
            Self::FlashAttention => &["attention_block_flash"],
            Self::MemoryEfficientAttention => &[],
            Self::AttentionBlock => &["attention_block"],
            Self::Rope => &["rope"],
            Self::Softmax => &["softmax"],
            Self::Geglu => &["geglu"],
            Self::GeluErf => &["gelu"],
            Self::Lora => &["lora_linear", "lora_epilogue"],
            Self::AdamW => &["adamw"],
        }
    }

    /// The family whose admitted kernel makes this family's call site
    /// unreachable.
    pub fn absorbed_by(self) -> Option<Self> {
        match self {
            Self::MemoryEfficientAttention | Self::AttentionBlock => Some(Self::FlashAttention),
            Self::Rope | Self::Softmax => Some(Self::AttentionBlock),
            _ => None,
        }
    }

    /// The family, then its absorber, then its absorber's absorber.
    pub fn absorbers(self) -> impl Iterator<Item = Self> {
        std::iter::successors(self.absorbed_by(), |f| f.absorbed_by())
    }

    /// The family an admission key switches.
    pub fn of_key(key: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|family| family.keys().any(|k| k == key))
    }
}

/// The families a rung turns off.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct KernelArm {
    pub off: BTreeSet<KernelFamily>,
}

/// Why an arm cannot be turned into a disable list.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ArmError {
    #[error("{family:?} is absorbed by {absorber:?}, which this arm leaves on: its key would never fire on the device")]
    Absorbed {
        family: KernelFamily,
        absorber: KernelFamily,
    },
}

impl KernelArm {
    /// Every family on: the fused arm.
    pub fn fused() -> Self {
        Self::default()
    }

    pub fn off(families: impl IntoIterator<Item = KernelFamily>) -> Self {
        Self {
            off: families.into_iter().collect(),
        }
    }

    /// Every family off: the reference arm of a step-level comparison.
    pub fn all_off() -> Self {
        Self::off(KernelFamily::ALL)
    }

    pub fn is_off(&self, family: KernelFamily) -> bool {
        self.off.contains(&family)
    }

    /// The label a leg states as its `arm`.
    pub fn label(&self) -> &'static str {
        if self.off.is_empty() {
            "fused"
        } else {
            "alloff"
        }
    }

    /// The `JAMMI_KERNELS_DISABLE` value for this arm on a checkpoint whose
    /// one training step consults `live`: the arm's keys, restricted to the
    /// live ones, sorted. Refused when the arm turns off a family whose
    /// absorber it leaves on.
    pub fn disable_list(&self, live: &BTreeSet<String>) -> Result<Vec<String>, ArmError> {
        for family in &self.off {
            if let Some(absorber) = family.absorbers().find(|a| !self.is_off(*a)) {
                return Err(ArmError::Absorbed {
                    family: *family,
                    absorber,
                });
            }
        }
        let keys: BTreeSet<String> = self
            .off
            .iter()
            .flat_map(|family| family.keys())
            .filter(|key| live.contains(*key))
            .map(str::to_owned)
            .collect();
        Ok(keys.into_iter().collect())
    }
}

/// Every admission key consulted at least once between two readings of the
/// counters.
fn consulted(before: &Census, after: &Census) -> BTreeSet<String> {
    after
        .0
        .iter()
        .filter(|(key, total)| before.0.get(*key).copied().unwrap_or(0) < **total)
        .map(|(key, _)| key.clone())
        .collect()
}

/// Every registered admission key with the number of decisions taken under
/// it, two-arm and cascade counters alike.
struct Census(std::collections::BTreeMap<String, u64>);

impl Census {
    fn read() -> Self {
        let two_arm = admission::snapshot_all()
            .into_iter()
            .map(|(key, s)| (key.to_owned(), s.fused + s.eager));
        // A cascade's counters live in their own registry, read key by key.
        let cascade = admission::PROBED_OPS
            .iter()
            .filter(|op| op.kind() == admission::ProbedOpKind::Cascade)
            .flat_map(|op| op.all_registry_keys())
            .map(|key| {
                let s = admission::cascade_counters_for(key).snapshot();
                (key.to_owned(), s.fused + s.eager + s.declined)
            });
        Self(two_arm.chain(cascade).collect())
    }
}

/// One training step — forward, backward, an optimizer update — over a
/// synthetic batch on the CPU build of a text checkpoint of any family,
/// with LoRA on `target_modules`: enough to reach every admission key the
/// checkpoint's tower consults.
fn training_step(
    model_dir: &std::path::Path,
    target_modules: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    use jammi_ai::fine_tune::adamw::AdamW;
    use jammi_ai::fine_tune::optimizer::sorted_trainable_vars;
    use jammi_ai::model::arch::EncoderFamily;
    use jammi_encoders::AnyEncoder;

    let config_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json"))?)?;
    let family = EncoderFamily::from_config(&config_json)
        .ok_or("config.json names no encoder family this census can build")?;
    let weights = model_dir.join("model.safetensors");
    let device = candle_core::Device::Cpu;
    let varmap = candle_nn::VarMap::new();
    let empty_ranks = std::collections::HashMap::new();
    let lora = jammi_lora::LoraBuildConfig {
        target_modules,
        layers_to_transform: &None,
        lora_rank: 2,
        lora_alpha: 4.0,
        use_rslora: false,
        lora_dropout: None,
        rank_pattern: &empty_ranks,
        init_mode: jammi_lora::LoraInitMode::ZerosB,
        seed: 0,
        dropout_seed: 0,
    };
    let dtype = candle_core::DType::F32;
    let pooling = jammi_encoders::Pooling::Mean;
    let mut encoder = match family {
        EncoderFamily::ModernBert => {
            let cfg: jammi_encoders::ModernBertConfig =
                serde_json::from_value(config_json.clone())?;
            AnyEncoder::ModernBert(
                jammi_encoders::ModernBert::builder()
                    .pooling(pooling)
                    .backbone_dtype(dtype)
                    .lora(lora)
                    .build(&[weights.as_path()], &cfg, &device, &varmap)?,
            )
        }
        EncoderFamily::Bert => {
            let cfg: jammi_encoders::BertConfig = serde_json::from_value(config_json.clone())?;
            AnyEncoder::Bert(
                jammi_encoders::Bert::builder()
                    .pooling(pooling)
                    .backbone_dtype(dtype)
                    .lora(lora)
                    .build(&[weights.as_path()], &cfg, &device, &varmap)?,
            )
        }
        EncoderFamily::DistilBert => {
            let cfg: jammi_encoders::DistilBertConfig =
                serde_json::from_value(config_json.clone())?;
            AnyEncoder::DistilBert(
                jammi_encoders::DistilBert::builder()
                    .pooling(pooling)
                    .backbone_dtype(dtype)
                    .lora(lora)
                    .build(&[weights.as_path()], &cfg, &device, &varmap)?,
            )
        }
        other => return Err(format!("{other:?} is not a text tower this census steps").into()),
    };
    encoder.set_training(true);
    let trainable = sorted_trainable_vars(&varmap);
    if trainable.is_empty() {
        return Err("target_modules matched no LoRA site".into());
    }
    let mut opt = AdamW::new(
        trainable,
        candle_nn::ParamsAdamW {
            lr: 2e-4,
            ..Default::default()
        },
    )?;
    let vocab = config_json["vocab_size"]
        .as_u64()
        .ok_or("config.json has no vocab_size")? as usize;
    let (batch, seq) = (2, 8);
    let ids = crate::finetune_step::synthetic_ids(batch, seq, vocab, 0, &device);
    let mask = candle_core::Tensor::ones((batch, seq), candle_core::DType::U32, &device)?;
    let pooled = encoder.forward(&ids, &mask)?;
    let loss = pooled.sqr()?.mean_all()?;
    let grads = loss.backward()?;
    opt.step(&grads)?;
    Ok(())
}

/// The admission keys one training step consults in this process, under
/// whatever `JAMMI_KERNELS_DISABLE` names: one pass of the census.
pub fn consulted_keys(
    model_dir: &std::path::Path,
    target_modules: &[String],
) -> Result<BTreeSet<String>, Box<dyn std::error::Error>> {
    let before = Census::read();
    training_step(model_dir, target_modules)?;
    Ok(consulted(&before, &Census::read()))
}

/// One pass of the census in a fresh process with `off` disabled.
fn census_pass(
    model_dir: &std::path::Path,
    target_modules: &[String],
    off: &BTreeSet<String>,
) -> Result<BTreeSet<String>, Box<dyn std::error::Error>> {
    let disable: Vec<&str> = off.iter().map(String::as_str).collect();
    let output = std::process::Command::new(std::env::current_exe()?)
        .arg("kernel-census")
        .arg("--model-dir")
        .arg(model_dir)
        .arg("--target-modules")
        .arg(target_modules.join(","))
        .env("JAMMI_KERNELS_DISABLE", disable.join(","))
        .env_remove("JAMMI_KERNELS_STRICT")
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "census pass with {disable:?} off failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

/// Every admission key any training step on `model_dir` can consult: the
/// checkpoint's census, taken to its fixpoint over absorption.
pub fn live_keys(
    model_dir: &std::path::Path,
    target_modules: Vec<String>,
) -> Result<BTreeSet<String>, Box<dyn std::error::Error>> {
    let requested = admission::disabled_ops_requested();
    if !requested.is_empty() {
        return Err(format!(
            "JAMMI_KERNELS_DISABLE={} is set: the census turns families off itself",
            requested.join(",")
        )
        .into());
    }
    let mut live: BTreeSet<String> = BTreeSet::new();
    let mut off = BTreeSet::new();
    loop {
        let pass = census_pass(model_dir, &target_modules, &off)?;
        let before = live.len();
        live.extend(pass);
        if live.is_empty() {
            return Err("the census consulted no admission key at all".into());
        }
        // Turn off every consulted family that hides another behind it, so
        // the next pass reaches what it absorbs.
        let absorbers: BTreeSet<String> = live
            .iter()
            .filter_map(|key| KernelFamily::of_key(key))
            .filter(|family| {
                KernelFamily::ALL
                    .iter()
                    .any(|f| f.absorbed_by() == Some(*family))
            })
            .flat_map(|family| family.keys())
            .filter(|key| live.contains(*key))
            .map(str::to_owned)
            .collect();
        if live.len() == before || absorbers == off {
            return Ok(live);
        }
        off = absorbers;
    }
}

/// `jammi-bench kernel-census`: one pass of the census, printed as JSON —
/// the child process of `kernel-arm`.
#[derive(Debug, clap::Args)]
pub struct KernelCensusArgs {
    #[arg(long)]
    pub model_dir: std::path::PathBuf,
    #[arg(long)]
    pub target_modules: String,
}

pub fn run_census(args: &KernelCensusArgs) -> std::process::ExitCode {
    match consulted_keys(&args.model_dir, &split(&args.target_modules)) {
        Ok(keys) => {
            println!(
                "{}",
                serde_json::to_string(&keys).expect("serialize census")
            );
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("kernel-census: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn split(target_modules: &str) -> Vec<String> {
    target_modules
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

/// `jammi-bench kernel-arm`: the `JAMMI_KERNELS_DISABLE` value of an arm on
/// a checkpoint.
#[derive(Debug, clap::Args)]
pub struct KernelArmArgs {
    #[arg(long)]
    pub model_dir: std::path::PathBuf,
    /// Families to turn off; `--all` turns off every family.
    #[arg(long, value_enum, value_delimiter = ',')]
    pub off: Vec<KernelFamily>,
    #[arg(long)]
    pub all: bool,
    /// The LoRA sites the census's one training step adapts: the same
    /// selector names the legs will use.
    #[arg(long)]
    pub target_modules: String,
    /// Print the census and the derivation as JSON instead of the bare
    /// disable value.
    #[arg(long)]
    pub json: bool,
}

#[derive(Serialize)]
struct Derivation<'a> {
    arm: &'a KernelArm,
    live: &'a BTreeSet<String>,
    disable: &'a [String],
}

pub fn run(args: &KernelArmArgs) -> std::process::ExitCode {
    let arm = if args.all {
        KernelArm::all_off()
    } else {
        KernelArm::off(args.off.iter().copied())
    };
    let derived = live_keys(&args.model_dir, split(&args.target_modules))
        .and_then(|live| Ok((arm.disable_list(&live)?, live)));
    match derived {
        Ok((disable, live)) => {
            if args.json {
                let derivation = Derivation {
                    arm: &arm,
                    live: &live,
                    disable: &disable,
                };
                println!(
                    "{}",
                    serde_json::to_string_pretty(&derivation).expect("serialize derivation")
                );
            } else {
                println!("{}", disable.join(","));
            }
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("kernel-arm: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(families: &[KernelFamily]) -> BTreeSet<String> {
        families
            .iter()
            .flat_map(|f| f.keys())
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn every_key_belongs_to_one_family_and_every_counter_base_to_one() {
        let mut seen = BTreeSet::new();
        for family in KernelFamily::ALL {
            for key in family.keys() {
                assert!(seen.insert(key), "{key} is switched by two families");
                assert_eq!(KernelFamily::of_key(key), Some(family));
            }
            for base in family.counters() {
                assert_eq!(
                    KernelFamily::ALL
                        .into_iter()
                        .filter(|f| f.counters().contains(base))
                        .count(),
                    1,
                    "{base} is reported under two families"
                );
            }
        }
    }

    #[test]
    fn a_family_absorbed_by_a_family_left_on_is_refused_by_name() {
        let live = keys(&KernelFamily::ALL);
        let arm = KernelArm::off([KernelFamily::Rope]);
        assert_eq!(
            arm.disable_list(&live),
            Err(ArmError::Absorbed {
                family: KernelFamily::Rope,
                absorber: KernelFamily::AttentionBlock,
            })
        );
        let arm = KernelArm::off([KernelFamily::Rope, KernelFamily::AttentionBlock]);
        assert_eq!(
            arm.disable_list(&live),
            Err(ArmError::Absorbed {
                family: KernelFamily::AttentionBlock,
                absorber: KernelFamily::FlashAttention,
            })
        );
        let arm = KernelArm::off([
            KernelFamily::Rope,
            KernelFamily::AttentionBlock,
            KernelFamily::FlashAttention,
        ]);
        assert_eq!(
            arm.disable_list(&live).unwrap(),
            [
                "attention_block_flash",
                "attention_block_fused",
                "rope_fused"
            ]
        );
    }

    #[test]
    fn the_disable_list_is_the_arm_restricted_to_the_live_keys() {
        let live: BTreeSet<String> = ["adamw_step_fused", "layer_norm_fused"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        assert_eq!(
            KernelArm::all_off().disable_list(&live).unwrap(),
            ["adamw_step_fused", "layer_norm_fused"]
        );
        assert!(KernelArm::fused().disable_list(&live).unwrap().is_empty());
    }
}
