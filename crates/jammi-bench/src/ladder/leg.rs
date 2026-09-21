//! A leg: one run of one rung, as the comparator sees it.
//!
//! Any producer may emit a leg — a `jammi-bench` report, a reference
//! script's JSON. This module is the one place a producer's report is mapped
//! onto the rung-agnostic [`Leg`]; nothing downstream reads a report field by
//! name. A leg's *role* — which rung, which unit of the sweep, which take —
//! is its file name, `<rung>__<unit>__<take>.json`, because one producer can
//! serve several rungs and only the script that launched it knows which.
//! What the file name claims, the rung's premises then check against the
//! leg's own contents.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;

use crate::report::Nullable;

use super::definition::Workload;
use super::refusal::Refusal;

/// One point of a sweep: a seed, a row count. Ordered by its trailing
/// number, so `seed2` sorts before `seed10`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Unit(String);

impl Unit {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn sort_key(&self) -> (&str, u64) {
        let digits = self
            .0
            .chars()
            .rev()
            .take_while(char::is_ascii_digit)
            .count();
        let (prefix, number) = self.0.split_at(self.0.len() - digits);
        (prefix, number.parse().unwrap_or(0))
    }
}

impl Ord for Unit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.sort_key()
            .cmp(&other.sort_key())
            .then_with(|| self.0.cmp(&other.0))
    }
}

impl PartialOrd for Unit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Which run of a `(rung, unit)` a leg is.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Take {
    /// `r<N>`: the N-th measured repeat. `r1` carries the outcome; every
    /// repeat carries time and memory.
    Repeat(u32),
    /// Any other tag: a control run, never counted into a statistic.
    Control(String),
}

impl Take {
    fn parse(tag: &str) -> Self {
        tag.strip_prefix('r')
            .and_then(|n| n.parse().ok())
            .filter(|n| *n >= 1)
            .map_or_else(|| Self::Control(tag.to_owned()), Self::Repeat)
    }
}

impl std::fmt::Display for Take {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Repeat(n) => write!(f, "r{n}"),
            Self::Control(tag) => f.write_str(tag),
        }
    }
}

/// A leg file's name, parsed: `<rung>__<unit>__<take>.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegName {
    pub rung: String,
    pub unit: Unit,
    pub take: Take,
}

impl LegName {
    pub fn parse(file_name: &str) -> Result<Self, Refusal> {
        let malformed = |reason: &str| Refusal::LegNameMalformed {
            file: file_name.to_owned(),
            reason: reason.to_owned(),
        };
        let stem = file_name
            .strip_suffix(".json")
            .ok_or_else(|| malformed("not a .json file"))?;
        match stem.split("__").collect::<Vec<_>>().as_slice() {
            [rung, unit, take] if [rung, unit, take].iter().all(|part| !part.is_empty()) => {
                Ok(Self {
                    rung: (*rung).to_owned(),
                    unit: Unit((*unit).to_owned()),
                    take: Take::parse(take),
                })
            }
            _ => Err(malformed(
                "expected exactly three non-empty parts separated by `__`",
            )),
        }
    }
}

impl std::fmt::Display for LegName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}__{}__{}", self.rung, self.unit.as_str(), self.take)
    }
}

/// A measured quantity as either a bare number or the report schema's
/// `{ value, unit }` slot, whose `value: null` is "not measured".
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Quantity {
    Bare(f64),
    Slot { value: Option<f64> },
}

impl Quantity {
    fn value(self) -> Option<f64> {
        match self {
            Self::Bare(v) => Some(v),
            Self::Slot { value } => value,
        }
    }
}

/// One held-out evaluation along a training run.
#[derive(Debug, Clone, Deserialize)]
pub struct TrajectoryPoint {
    pub held_out_mean: f64,
    /// Cumulative training wall seconds when this evaluation was taken.
    pub train_wall_s: Option<f64>,
}

/// A mutant leg's own statement of which patch produced it.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct MutantStamp {
    pub mutant_id: Option<String>,
    pub mutant_base_sha: Option<String>,
    pub mutant_patch_sha256: Option<String>,
}

/// What a leg says about how it ran — the inputs of the rung premises.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Facts {
    pub arm: Option<String>,
    pub schedule: Option<String>,
    pub admission_is_dense: Option<bool>,
    pub tie_fraction: Option<f64>,
    pub epochs: Option<usize>,
    pub train_probe_series: Option<Vec<f64>>,
    pub backbone_dtype: Option<String>,
    pub compute_precision: Option<String>,
    pub flash_compiled: Option<bool>,
    #[serde(default)]
    pub kernels_disabled_requested: Vec<String>,
    #[serde(default)]
    pub kernels_disabled_fired: Vec<String>,
    #[serde(flatten)]
    pub mutant: MutantStamp,
}

/// The fields this module reads off a producer's block, by their producer
/// names. Every name a leg is read by appears here and nowhere else.
#[derive(Debug, Deserialize)]
struct Block {
    iter_wall_s: Option<Vec<f64>>,
    work: Option<f64>,
    peak_rss_bytes: Option<Quantity>,
    peak_vram_bytes: Option<Quantity>,
    outcome_digest: Option<String>,
    held_out_example_mean: Option<f64>,
    #[serde(default)]
    trajectory: Vec<TrajectoryPoint>,
    vectors_file: Option<String>,
    vector_dim: Option<usize>,
    law_observed: Option<Vec<Vec<u64>>>,
    #[serde(flatten)]
    facts: Facts,
    #[serde(flatten)]
    rest: BTreeMap<String, Value>,
}

const FUSED_SUFFIX: &str = "_fused_dispatches";

/// A kernel's dispatch counters: how often the fused arm ran, and how often
/// its fallback did. `None` is a counter the producer did not emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchPair {
    pub fused: u64,
    pub fallback: Option<u64>,
}

/// The per-row vectors an encode leg produced: little-endian `f32`,
/// row-major, `rows × dim`, in the committed key order.
#[derive(Debug, Clone)]
pub struct VectorsFile {
    pub path: PathBuf,
    pub dim: usize,
}

#[derive(Debug, Clone)]
pub struct Leg {
    pub name: LegName,
    /// Each identity field's canonical value; `None` is absent, or null on a
    /// field whose null means nothing.
    pub identity: BTreeMap<&'static str, Option<String>>,
    /// Identity fields as numbers, where they are numbers.
    identity_numbers: BTreeMap<&'static str, f64>,
    pub facts: Facts,
    /// `base -> counters` for every `<base>_fused_dispatches` field.
    pub dispatch: BTreeMap<String, DispatchPair>,
    /// Post-warmup wall seconds of each timed iteration, in run order.
    pub iter_wall_s: Option<Vec<f64>>,
    /// The size this leg's cost scales with — rows encoded or trained on,
    /// edges × hops propagated, edges walked — where a sweep varies it.
    pub work: Option<f64>,
    pub peak_rss_bytes: Option<f64>,
    pub peak_vram_bytes: Option<f64>,
    pub outcome_digest: Option<String>,
    pub held_out: Option<f64>,
    pub trajectory: Vec<TrajectoryPoint>,
    pub vectors: Option<VectorsFile>,
    /// Counts observed in each category of each cell of the workload's law,
    /// in the law file's cell and category order.
    pub law_observed: Option<Vec<Vec<u64>>>,
}

impl Leg {
    /// Map one producer report onto a leg.
    pub fn from_report(
        workload: Workload,
        name: LegName,
        report: &Value,
        dir: &Path,
    ) -> Result<Self, Refusal> {
        let unreadable = |reason: String| Refusal::LegUnreadable {
            leg: name.to_string(),
            reason,
        };
        let key = workload.tier_key();
        let block = report
            .pointer(&format!("/tiers/{key}"))
            .or_else(|| report.get(key))
            .filter(|b| b.is_object())
            .ok_or_else(|| unreadable(format!("no `tiers.{key}` or top-level `{key}` object")))?;
        let parsed: Block = Block::deserialize(block).map_err(|e| unreadable(e.to_string()))?;

        let identity = workload
            .identity_fields()
            .iter()
            .map(|(field, nullable)| {
                let value = block.get(*field).filter(|v| match nullable {
                    Nullable::NonNull => !v.is_null(),
                    Nullable::NullMeans(_) => true,
                });
                (*field, value.map(canonical))
            })
            .collect();
        let identity_numbers = workload
            .identity_fields()
            .iter()
            .filter_map(|(field, _)| Some((*field, block.get(*field)?.as_f64()?)))
            .collect();

        let dispatch = parsed
            .rest
            .iter()
            .filter_map(|(field, value)| Some((field.strip_suffix(FUSED_SUFFIX)?, value.as_u64()?)))
            .map(|(base, fused)| {
                let fallback = ["_eager_dispatches", "_declined_dispatches"]
                    .iter()
                    .find_map(|suffix| parsed.rest.get(&format!("{base}{suffix}"))?.as_u64());
                (base.to_owned(), DispatchPair { fused, fallback })
            })
            .collect();

        let vectors = match (parsed.vectors_file, parsed.vector_dim) {
            (Some(file), Some(dim)) => Some(VectorsFile {
                path: dir.join(file),
                dim,
            }),
            (None, None) => None,
            _ => {
                return Err(unreadable(
                    "`vectors_file` and `vector_dim` must be given together".into(),
                ))
            }
        };

        Ok(Self {
            name,
            identity,
            identity_numbers,
            facts: parsed.facts,
            dispatch,
            iter_wall_s: parsed.iter_wall_s,
            work: parsed.work,
            peak_rss_bytes: parsed.peak_rss_bytes.and_then(Quantity::value),
            peak_vram_bytes: parsed.peak_vram_bytes.and_then(Quantity::value),
            outcome_digest: parsed.outcome_digest,
            held_out: parsed.held_out_example_mean,
            trajectory: parsed.trajectory,
            vectors,
            law_observed: parsed.law_observed,
        })
    }

    pub fn identity_number(&self, field: &str) -> Option<f64> {
        self.identity_numbers.get(field).copied()
    }

    /// The dtype the model's matmuls ran in, under either tier's name for it.
    pub fn compute_dtype(&self) -> Option<&str> {
        self.facts
            .compute_precision
            .as_deref()
            .or(self.facts.backbone_dtype.as_deref())
    }
}

/// A JSON value in the one spelling two producers must share to agree:
/// numbers as `f64` (so `16` and `16.0` are the same batch size), everything
/// else as written. A producer that spells a value differently disagrees.
fn canonical(value: &Value) -> String {
    match value {
        Value::Number(n) => n
            .as_f64()
            .map_or_else(|| n.to_string(), |f| format!("{f:?}")),
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(canonical).collect::<Vec<_>>().join(",")
        ),
        other => other.to_string(),
    }
}

/// Every leg of one session, by rung.
#[derive(Debug, Default)]
pub struct LegSet {
    by_rung: BTreeMap<String, RungLegs>,
    /// Files that could not become a leg.
    pub unreadable: Vec<Refusal>,
}

/// The legs of one rung, by unit.
#[derive(Debug, Default, Clone)]
pub struct RungLegs {
    pub units: BTreeMap<Unit, Vec<Leg>>,
}

impl RungLegs {
    pub fn repeats(&self, unit: &Unit) -> impl Iterator<Item = &Leg> {
        self.units
            .get(unit)
            .into_iter()
            .flatten()
            .filter(|leg| matches!(leg.name.take, Take::Repeat(_)))
    }

    pub fn primary(&self, unit: &Unit) -> Option<&Leg> {
        self.repeats(unit)
            .find(|leg| leg.name.take == Take::Repeat(1))
    }

    pub fn controls(&self) -> impl Iterator<Item = &Leg> {
        self.units
            .values()
            .flatten()
            .filter(|leg| matches!(leg.name.take, Take::Control(_)))
    }

    pub fn all(&self) -> impl Iterator<Item = &Leg> {
        self.units.values().flatten()
    }

    /// Units that carry at least one repeat.
    pub fn measured_units(&self) -> impl Iterator<Item = &Unit> {
        self.units
            .iter()
            .filter(|(_, legs)| legs.iter().any(|l| matches!(l.name.take, Take::Repeat(_))))
            .map(|(unit, _)| unit)
    }
}

impl LegSet {
    /// Read every `*.json` directly under `dir`. A missing directory is an
    /// empty set: whether that matters is the comparison's call.
    pub fn read(workload: Workload, dir: &Path) -> std::io::Result<Self> {
        let mut set = Self::default();
        if !dir.is_dir() {
            return Ok(set);
        }
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<std::io::Result<_>>()?;
        files.retain(|p| p.is_file() && p.extension().is_some_and(|e| e == "json"));
        files.sort();
        for path in files {
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            match Self::read_leg(workload, file_name, &path, dir) {
                Ok(leg) => set.insert(leg),
                Err(refusal) => set.unreadable.push(refusal),
            }
        }
        Ok(set)
    }

    fn read_leg(
        workload: Workload,
        file_name: &str,
        path: &Path,
        dir: &Path,
    ) -> Result<Leg, Refusal> {
        let name = LegName::parse(file_name)?;
        let unreadable = |reason: String| Refusal::LegUnreadable {
            leg: name.to_string(),
            reason,
        };
        let text = std::fs::read_to_string(path).map_err(|e| unreadable(e.to_string()))?;
        let report: Value = serde_json::from_str(&text).map_err(|e| unreadable(e.to_string()))?;
        Leg::from_report(workload, name, &report, dir)
    }

    pub fn insert(&mut self, leg: Leg) {
        self.by_rung
            .entry(leg.name.rung.clone())
            .or_default()
            .units
            .entry(leg.name.unit.clone())
            .or_default()
            .push(leg);
    }

    pub fn rung(&self, name: &str) -> RungLegs {
        self.by_rung.get(name).cloned().unwrap_or_default()
    }

    pub fn rung_names(&self) -> impl Iterator<Item = &str> {
        self.by_rung.keys().map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_leg_name_has_exactly_three_parts() {
        let name = LegName::parse("resident__seed10__r2.json").unwrap();
        assert_eq!(
            (name.rung.as_str(), name.unit.as_str()),
            ("resident", "seed10")
        );
        assert_eq!(name.take, Take::Repeat(2));
        assert_eq!(
            LegName::parse("resident__seed1__lr0.json").unwrap().take,
            Take::Control("lr0".into())
        );
        assert_eq!(
            LegName::parse("resident__seed1__r0.json").unwrap().take,
            Take::Control("r0".into())
        );
        for bad in [
            "resident__seed1.json",
            "a__b__c__d.json",
            "a____r1.json",
            "a__b__r1.txt",
        ] {
            assert!(
                matches!(LegName::parse(bad), Err(Refusal::LegNameMalformed { .. })),
                "{bad}"
            );
        }
    }

    #[test]
    fn units_sort_by_their_number() {
        let mut units: Vec<Unit> = ["seed10", "seed2", "seed1"]
            .iter()
            .map(|s| Unit((*s).into()))
            .collect();
        units.sort();
        let names: Vec<&str> = units.iter().map(Unit::as_str).collect();
        assert_eq!(names, ["seed1", "seed2", "seed10"]);
    }

    fn encode_block() -> Value {
        json!({
            "seed": 7, "batch": 16, "seq": 64, "row_lengths": [64, 64],
            "compute_precision": "f32",
            "checkpoint_config_sha256": "c", "checkpoint_weights_sha256": "w",
            "checkpoint_weights_size_bytes": 10, "checkpoint_tokenizer_sha256": "t",
            "pooling": "mean", "normalize": true, "warmup": 2, "iters_measured": 4,
            "device_requested": "cpu",
            "iter_wall_s": [0.1, 0.2, 0.1, 0.1],
            "work": 16,
            "peak_rss_bytes": {"value": 1024.0, "unit": "bytes"},
            "peak_vram_bytes": {"value": null, "unit": "bytes"},
            "outcome_digest": "abc"
        })
    }

    fn leg_from(report: &Value) -> Result<Leg, Refusal> {
        let name = LegName::parse("direct__rows16__r1.json").unwrap();
        Leg::from_report(Workload::Encode, name, report, Path::new("."))
    }

    #[test]
    fn a_tiered_report_and_a_flat_one_map_onto_the_same_leg() {
        let tiered = leg_from(&json!({"tiers": {"encode_step": encode_block()}})).unwrap();
        let flat = leg_from(&json!({"encode_step": encode_block()})).unwrap();
        assert_eq!(tiered.identity, flat.identity);
        assert_eq!(
            tiered.iter_wall_s.as_deref(),
            Some(&[0.1, 0.2, 0.1, 0.1][..])
        );
        assert_eq!(tiered.peak_rss_bytes, Some(1024.0));
        assert_eq!(tiered.peak_vram_bytes, None);
        assert_eq!(tiered.outcome_digest.as_deref(), Some("abc"));
        assert_eq!(tiered.compute_dtype(), Some("f32"));
    }

    #[test]
    fn integer_and_float_spellings_of_a_number_agree() {
        let mut other = encode_block();
        other["batch"] = json!(16.0);
        let a = leg_from(&json!({"encode_step": encode_block()})).unwrap();
        let b = leg_from(&json!({"encode_step": other})).unwrap();
        assert_eq!(a.identity["batch"], b.identity["batch"]);
    }

    #[test]
    fn a_null_identity_field_is_missing_unless_its_null_means_something() {
        let mut block = encode_block();
        block["pooling"] = Value::Null;
        let leg = leg_from(&json!({"encode_step": block})).unwrap();
        assert_eq!(leg.identity["pooling"], None);

        let name = LegName::parse("resident__seed1__r1.json").unwrap();
        let train = json!({"finetune_run": {"margin": null, "seed": 1}});
        let leg = Leg::from_report(Workload::TrainRun, name, &train, Path::new(".")).unwrap();
        assert_eq!(leg.identity["margin"].as_deref(), Some("null"));
        assert_eq!(leg.identity["task"], None);
    }

    #[test]
    fn a_report_without_the_workloads_block_is_unreadable() {
        assert!(matches!(
            leg_from(&json!({"tool": "dry-run"})),
            Err(Refusal::LegUnreadable { .. })
        ));
        assert!(matches!(
            leg_from(&json!({"encode_step": {"iter_wall_s": "fast"}})),
            Err(Refusal::LegUnreadable { .. })
        ));
    }

    #[test]
    fn dispatch_counters_pair_a_fused_count_with_its_fallback() {
        let name = LegName::parse("resident__seed1__r1.json").unwrap();
        let report = json!({"finetune_run": {
            "ln_fused_dispatches": 9, "ln_eager_dispatches": 0,
            "attention_block_flash_fused_dispatches": 3, "attention_block_flash_declined_dispatches": 1,
            "solo_fused_dispatches": 2
        }});
        let leg = Leg::from_report(Workload::TrainRun, name, &report, Path::new(".")).unwrap();
        assert_eq!(
            leg.dispatch["ln"],
            DispatchPair {
                fused: 9,
                fallback: Some(0)
            }
        );
        assert_eq!(
            leg.dispatch["attention_block_flash"],
            DispatchPair {
                fused: 3,
                fallback: Some(1)
            }
        );
        assert_eq!(
            leg.dispatch["solo"],
            DispatchPair {
                fused: 2,
                fallback: None
            }
        );
    }
}
