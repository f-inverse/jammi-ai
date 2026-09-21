//! Premises: facts a leg must show about itself before its numbers count.
//!
//! A leg's file name claims a rung. A premise checks the counted fact behind
//! the claim — a leg filed as the fused-kernel rung must have dispatched the
//! fused kernels, a leg filed as a training run must show that training
//! moved the model. A claim with no counted fact behind it is refused, never
//! assumed.

use serde::Serialize;

use super::leg::{DispatchPair, Leg};
use super::refusal::Refusal;

/// Which way the train-side probe loss must have moved over the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainDirection {
    /// The loss fell: the run learned.
    Descent,
    /// The loss rose: a mutant built to ascend did ascend.
    Ascent,
}

/// The floor the probe movement `|first − last|` must strictly exceed. A run
/// that cannot update a parameter moves the probe by exactly zero, and the
/// control legs prove this floor refuses it.
pub const LEARNING_FLOOR: f64 = 0.0;

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LegPremise {
    /// The learning-rate schedule is constant. A run chained from
    /// single-epoch legs recomputes a decaying schedule's horizon per leg,
    /// so only a constant schedule is the schedule it claims to be.
    ConstantSchedule,
    /// Variable-length rows took the padded transport, the one the fixture
    /// is scoped to.
    PaddedAdmission,
    /// The train-side probe moved by more than [`LEARNING_FLOOR`], in the
    /// declared direction: held-out movement is attributable to training
    /// only if training demonstrably happened.
    LearningHappened(TrainDirection),
    /// The held-out tie fraction is under this cap; a saturated loss does
    /// not discriminate between examples.
    TieFractionBelow(f64),
    /// The leg states this arm.
    Arm(&'static str),
    /// The fused kernels ran, the flash attention cascade among them.
    FusedDispatch,
    /// The kernels this leg asked to disable did not run fused, their
    /// fallbacks did, and attention fell through to the block kernel.
    ReferenceDispatch,
    /// The leg is stamped with the mutant patch it was built from, and the
    /// stamp names this patch.
    MutantStamp { patch_sha256: String },
}

impl LegPremise {
    pub fn name(&self) -> &'static str {
        match self {
            Self::ConstantSchedule => "constant_schedule",
            Self::PaddedAdmission => "padded_admission",
            Self::LearningHappened(_) => "learning_happened",
            Self::TieFractionBelow(_) => "tie_fraction",
            Self::Arm(_) => "arm",
            Self::FusedDispatch => "fused_dispatch",
            Self::ReferenceDispatch => "reference_dispatch",
            Self::MutantStamp { .. } => "mutant_stamp",
        }
    }

    /// Why `leg` fails this premise, if it does.
    fn violation(&self, leg: &Leg) -> Option<String> {
        let facts = &leg.facts;
        match self {
            Self::ConstantSchedule => (facts.schedule.as_deref() != Some("constant"))
                .then(|| format!("schedule is {:?}, not \"constant\"", facts.schedule)),
            Self::PaddedAdmission => (facts.admission_is_dense != Some(false))
                .then(|| format!("admission_is_dense is {:?}, not false", facts.admission_is_dense)),
            Self::LearningHappened(direction) => match probe_movement(leg) {
                Err(reason) => Some(reason),
                Ok(moved) if moved.abs() <= LEARNING_FLOOR => Some(format!(
                    "the train probe moved by {moved}, not more than {LEARNING_FLOOR}: no learning was observed"
                )),
                Ok(moved) => {
                    let actual = if moved > 0.0 {
                        TrainDirection::Descent
                    } else {
                        TrainDirection::Ascent
                    };
                    (actual != *direction)
                        .then(|| format!("the train probe moved by {moved} ({actual:?}); this leg is declared {direction:?}"))
                }
            },
            Self::TieFractionBelow(cap) => match facts.tie_fraction {
                Some(tie) if tie < *cap => None,
                other => Some(format!("tie_fraction is {other:?}, not below {cap}")),
            },
            Self::Arm(arm) => {
                (facts.arm.as_deref() != Some(arm)).then(|| format!("arm is {:?}, not {arm:?}", facts.arm))
            }
            Self::FusedDispatch => dispatch::fused_violation(leg),
            Self::ReferenceDispatch => dispatch::reference_violation(leg),
            Self::MutantStamp { patch_sha256 } => mutant_stamp_violation(leg, patch_sha256),
        }
    }
}

/// `probe[0] − probe[last]`: how far the train-side probe loss fell from the
/// untrained model to the final epoch. The series is anchored at the
/// untrained model — one entry before training, one per epoch — so the first
/// epoch's learning, usually the largest, is inside the difference.
pub fn probe_movement(leg: &Leg) -> Result<f64, String> {
    let series = leg
        .facts
        .train_probe_series
        .as_deref()
        .ok_or("train_probe_series is absent")?;
    if series.iter().any(|v| !v.is_finite()) {
        return Err("train_probe_series has a non-finite entry".into());
    }
    let expected = leg.facts.epochs.map_or(2, |epochs| epochs + 1);
    let exact = leg.facts.epochs.is_some();
    if series.len() < 2 || (exact && series.len() != expected) {
        return Err(format!(
            "train_probe_series has {} entries; an init-anchored series has one before training and one per epoch ({})",
            series.len(),
            if exact { format!("{expected} for this leg") } else { "at least 2".into() },
        ));
    }
    Ok(series[0] - series[series.len() - 1])
}

fn mutant_stamp_violation(leg: &Leg, patch_sha256: &str) -> Option<String> {
    let stamp = &leg.facts.mutant;
    let blank = |field: &Option<String>| field.as_deref().is_none_or(|v| v.trim().is_empty());
    let missing: Vec<&str> = [
        ("mutant_id", &stamp.mutant_id),
        ("mutant_base_sha", &stamp.mutant_base_sha),
        ("mutant_patch_sha256", &stamp.mutant_patch_sha256),
    ]
    .into_iter()
    .filter(|(_, value)| blank(value))
    .map(|(field, _)| field)
    .collect();
    if !missing.is_empty() {
        return Some(format!(
            "{missing:?} absent: the leg is not attributable to an auditable patch"
        ));
    }
    let stamped = stamp.mutant_patch_sha256.as_deref().unwrap_or_default();
    (!stamped.trim().eq_ignore_ascii_case(patch_sha256.trim()))
        .then(|| format!("stamped with patch {stamped:?}, filed under patch {patch_sha256:?}"))
}

/// Every premise of `premises` that `leg` fails.
pub fn violations(leg: &Leg, premises: &[LegPremise]) -> Vec<Refusal> {
    premises
        .iter()
        .filter_map(|premise| {
            premise
                .violation(leg)
                .map(|reason| Refusal::PremiseViolated {
                    leg: leg.name.to_string(),
                    premise: premise.name(),
                    reason,
                })
        })
        .collect()
}

/// The counted facts behind a leg's kernel-arm claim.
mod dispatch {
    use super::{DispatchPair, Leg};

    /// Kernels every fused training step dispatches.
    const REQUIRED: [&str; 3] = ["ln", "geglu", "adamw"];
    /// Kernels the whole-attention-block kernel absorbs when it runs.
    const ABSORBED_BY_ATTENTION: [&str; 2] = ["rope", "softmax"];
    /// The two LoRA kernels; a site dispatches exactly one of them.
    const LORA_SITES: [&str; 2] = ["lora_epilogue", "lora_linear"];
    const OPTIONAL: [&str; 1] = ["gelu"];
    const ATTENTION_BLOCK: &str = "attention_block";
    /// The flash cascade: its fallback is a *declined* admission, which
    /// falls through to [`ATTENTION_BLOCK`].
    const FLASH: &str = "attention_block_flash";
    /// Compute dtypes the flash cascade admits.
    const FLASH_DTYPES: [&str; 2] = ["bf16", "f16"];
    /// A disable request, and the kernel whose counters it governs.
    const DISABLE_OPS: [(&str, &str); 2] = [(FLASH, FLASH), ("adamw_step_fused", "adamw")];

    fn known(base: &str) -> bool {
        REQUIRED
            .iter()
            .chain(&ABSORBED_BY_ATTENTION)
            .chain(&LORA_SITES)
            .chain(&OPTIONAL)
            .chain(&[ATTENTION_BLOCK, FLASH])
            .any(|b| *b == base)
    }

    fn fused(leg: &Leg, base: &str) -> u64 {
        leg.dispatch.get(base).map_or(0, |p| p.fused)
    }

    /// Defects of the counter set itself, shared by both arms.
    fn schema_violation(leg: &Leg) -> Option<String> {
        if leg.dispatch.is_empty() {
            return Some(
                "no dispatch counters: the arm claim has no counted fact behind it".into(),
            );
        }
        if let Some((base, _)) = leg.dispatch.iter().find(|(_, p)| p.fallback.is_none()) {
            return Some(format!(
                "{base}_fused_dispatches has no fallback counter beside it"
            ));
        }
        if let Some(base) = leg.dispatch.keys().find(|base| !known(base)) {
            return Some(format!(
                "dispatch counter {base:?} is not a kernel this proof classifies"
            ));
        }
        let dtype = leg.compute_dtype().unwrap_or("<absent>");
        (fused(leg, FLASH) > 0 && !FLASH_DTYPES.contains(&dtype)).then(|| {
            format!(
                "{} flash dispatches at dtype {dtype}, which the flash cascade does not admit: the report contradicts itself",
                fused(leg, FLASH)
            )
        })
    }

    pub fn fused_violation(leg: &Leg) -> Option<String> {
        if let Some(defect) = schema_violation(leg) {
            return Some(defect);
        }
        let dtype = leg.compute_dtype().unwrap_or("<absent>");
        if !FLASH_DTYPES.contains(&dtype) {
            return Some(format!(
                "dtype {dtype} cannot run the flash cascade this rung is defined by"
            ));
        }
        if leg.facts.flash_compiled != Some(true) {
            return Some(format!(
                "flash_compiled is {:?}: this build cannot run the flash cascade",
                leg.facts.flash_compiled
            ));
        }
        let requested = &leg.facts.kernels_disabled_requested;
        let fired = &leg.facts.kernels_disabled_fired;
        let deliberate = |base: &str| {
            base == FLASH && requested.iter().any(|k| k == base) && fired.iter().any(|k| k == base)
        };
        if let Some((base, pair)) = leg
            .dispatch
            .iter()
            .find(|(base, p)| p.fallback.unwrap_or(0) > 0 && !deliberate(base))
        {
            return Some(format!(
                "{base} fell back {} time(s)",
                pair.fallback.unwrap_or(0)
            ));
        }
        if let Some(base) = REQUIRED.iter().find(|base| fused(leg, base) == 0) {
            return Some(format!("{base} never dispatched fused"));
        }
        let absent = |bases: &[&'static str]| {
            bases
                .iter()
                .copied()
                .find(|base| !leg.dispatch.contains_key(*base))
        };
        if let Some(base) = absent(&[ATTENTION_BLOCK])
            .or(absent(&ABSORBED_BY_ATTENTION))
            .or(absent(&LORA_SITES))
        {
            return Some(format!("{base} has no dispatch counters"));
        }
        let attention_ran = fused(leg, ATTENTION_BLOCK) > 0 || fused(leg, FLASH) > 0;
        if !attention_ran {
            return Some("neither attention kernel dispatched fused".into());
        }
        if LORA_SITES.iter().map(|base| fused(leg, base)).sum::<u64>() == 0 {
            return Some("no LoRA site dispatched fused".into());
        }
        (fused(leg, FLASH) == 0)
            .then(|| "the flash cascade never dispatched: the block kernel ran in its place".into())
    }

    pub fn reference_violation(leg: &Leg) -> Option<String> {
        if let Some(defect) = schema_violation(leg) {
            return Some(defect);
        }
        let requested = &leg.facts.kernels_disabled_requested;
        let disabled: Vec<&str> = DISABLE_OPS
            .iter()
            .filter(|(op, _)| requested.iter().any(|k| k == op))
            .map(|(_, base)| *base)
            .collect();
        if disabled.is_empty() {
            return Some(format!(
                "kernels_disabled_requested {requested:?} names no kernel this rung disables"
            ));
        }
        for base in disabled {
            match leg.dispatch.get(base) {
                None => return Some(format!("{base} was disabled but has no dispatch counters")),
                Some(DispatchPair { fused, .. }) if *fused > 0 => {
                    return Some(format!(
                        "{base} was disabled yet dispatched fused {fused} time(s)"
                    ))
                }
                Some(DispatchPair { fallback, .. }) if fallback.unwrap_or(0) == 0 => {
                    return Some(format!(
                        "{base} shows no fallback dispatch: nothing counted behind the disable"
                    ))
                }
                Some(_) => {}
            }
        }
        (fused(leg, ATTENTION_BLOCK) == 0)
            .then(|| "attention did not fall through to the block kernel".into())
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::ladder::definition::Workload;
    use crate::ladder::leg::LegName;
    use serde_json::{json, Value};

    /// Dispatch counters and arm facts of a fused-kernel training leg, in the
    /// shape a real one carries.
    pub fn fused_facts() -> Value {
        json!({
            "arm": "fused", "schedule": "constant", "admission_is_dense": false, "tie_fraction": 0.0,
            "epochs": 3, "train_probe_series": [3.32, 2.88, 2.74, 2.52],
            "backbone_dtype": "bf16", "flash_compiled": true,
            "kernels_disabled_requested": [], "kernels_disabled_fired": [],
            "ln_fused_dispatches": 6669, "ln_eager_dispatches": 0,
            "rope_fused_dispatches": 0, "rope_eager_dispatches": 0,
            "softmax_fused_dispatches": 0, "softmax_eager_dispatches": 0,
            "geglu_fused_dispatches": 3276, "geglu_eager_dispatches": 0,
            "lora_epilogue_fused_dispatches": 0, "lora_epilogue_eager_dispatches": 0,
            "lora_linear_fused_dispatches": 13104, "lora_linear_eager_dispatches": 0,
            "attention_block_fused_dispatches": 0, "attention_block_eager_dispatches": 0,
            "adamw_fused_dispatches": 26208, "adamw_eager_dispatches": 0,
            "attention_block_flash_fused_dispatches": 3276, "attention_block_flash_declined_dispatches": 0
        })
    }

    /// The same run with the flash cascade and fused AdamW disabled.
    pub fn reference_facts() -> Value {
        let mut facts = fused_facts();
        for (field, value) in [
            ("arm", json!("alloff")),
            (
                "kernels_disabled_requested",
                json!(["adamw_step_fused", "attention_block_flash"]),
            ),
            (
                "kernels_disabled_fired",
                json!(["adamw_step_fused", "attention_block_flash"]),
            ),
            ("adamw_fused_dispatches", json!(0)),
            ("adamw_eager_dispatches", json!(26208)),
            ("attention_block_flash_fused_dispatches", json!(0)),
            ("attention_block_flash_declined_dispatches", json!(3276)),
            ("attention_block_fused_dispatches", json!(3276)),
        ] {
            facts[field] = value;
        }
        facts
    }

    fn leg(facts: Value) -> Leg {
        let name = LegName::parse("resident__seed1__r1.json").unwrap();
        Leg::from_report(
            Workload::TrainRun,
            name,
            &json!({ "finetune_run": facts }),
            std::path::Path::new("."),
        )
        .unwrap()
    }

    fn with(mut facts: Value, field: &str, value: Value) -> Leg {
        facts[field] = value;
        leg(facts)
    }

    fn fails(leg: &Leg, premise: LegPremise) -> bool {
        !violations(leg, std::slice::from_ref(&premise)).is_empty()
    }

    /// Two legs the real binary emitted on a ModernBERT-large checkpoint, one
    /// per kernel arm: what a hand-typed counter set would forget, these
    /// carry.
    fn golden(name: &str) -> Leg {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../ci/scripts/perf/fixtures/finetune_run_golden")
            .join(format!("{name}.json"));
        let report: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let name = LegName::parse("resident__seed1__r1.json").unwrap();
        Leg::from_report(Workload::TrainRun, name, &report, std::path::Path::new(".")).unwrap()
    }

    #[test]
    fn legs_the_real_binary_emitted_clear_their_own_arm_and_fail_the_other() {
        let (fused, reference) = (golden("modernbert_fused"), golden("modernbert_alloff"));
        for leg in [&fused, &reference] {
            assert!(!fails(leg, LegPremise::ConstantSchedule));
            assert!(!fails(leg, LegPremise::PaddedAdmission));
            assert!(!fails(leg, LegPremise::TieFractionBelow(0.5)));
        }
        assert!(!fails(&fused, LegPremise::FusedDispatch));
        assert!(!fails(&fused, LegPremise::Arm("fused")));
        assert!(fails(&fused, LegPremise::ReferenceDispatch));
        assert!(!fails(&reference, LegPremise::ReferenceDispatch));
        assert!(!fails(&reference, LegPremise::Arm("alloff")));
        assert!(fails(&reference, LegPremise::FusedDispatch));
    }

    #[test]
    fn real_shaped_legs_clear_their_own_arm_and_fail_the_other() {
        let (fused, reference) = (leg(fused_facts()), leg(reference_facts()));
        assert!(!fails(&fused, LegPremise::FusedDispatch));
        assert!(!fails(&reference, LegPremise::ReferenceDispatch));
        assert!(fails(&reference, LegPremise::FusedDispatch));
        assert!(fails(&fused, LegPremise::ReferenceDispatch));
        assert!(fails(&fused, LegPremise::Arm("alloff")));
        assert!(!fails(&fused, LegPremise::Arm("fused")));
    }

    #[test]
    fn a_fused_leg_must_have_run_the_flash_cascade_itself() {
        // The block kernel picking up the slack is a valid fused step, and
        // still not the rung this leg is filed under.
        let mut facts = fused_facts();
        facts["attention_block_flash_fused_dispatches"] = json!(0);
        facts["attention_block_fused_dispatches"] = json!(3276);
        assert!(fails(&leg(facts), LegPremise::FusedDispatch));
        assert!(fails(
            &with(fused_facts(), "flash_compiled", json!(false)),
            LegPremise::FusedDispatch
        ));
        assert!(fails(
            &with(fused_facts(), "backbone_dtype", json!("f32")),
            LegPremise::FusedDispatch
        ));
        assert!(fails(
            &with(fused_facts(), "ln_eager_dispatches", json!(1)),
            LegPremise::FusedDispatch
        ));
        assert!(fails(
            &with(fused_facts(), "geglu_fused_dispatches", json!(0)),
            LegPremise::FusedDispatch
        ));
    }

    #[test]
    fn counters_that_are_absent_solo_or_unclassified_are_refused() {
        let bare = json!({"arm": "fused", "backbone_dtype": "bf16", "flash_compiled": true});
        assert!(fails(&leg(bare), LegPremise::FusedDispatch));
        assert!(fails(
            &with(fused_facts(), "novel_fused_dispatches", json!(4)),
            LegPremise::FusedDispatch
        ));
        let mut unclassified = fused_facts();
        unclassified["novel_fused_dispatches"] = json!(4);
        unclassified["novel_eager_dispatches"] = json!(0);
        assert!(fails(&leg(unclassified), LegPremise::FusedDispatch));
    }

    #[test]
    fn a_reference_leg_needs_a_counted_fallback_behind_each_disable() {
        assert!(fails(
            &with(reference_facts(), "adamw_fused_dispatches", json!(5)),
            LegPremise::ReferenceDispatch
        ));
        assert!(fails(
            &with(reference_facts(), "adamw_eager_dispatches", json!(0)),
            LegPremise::ReferenceDispatch
        ));
        assert!(fails(
            &with(
                reference_facts(),
                "attention_block_fused_dispatches",
                json!(0)
            ),
            LegPremise::ReferenceDispatch
        ));
        assert!(fails(
            &with(reference_facts(), "kernels_disabled_requested", json!([])),
            LegPremise::ReferenceDispatch
        ));
    }

    #[test]
    fn learning_must_be_observed_and_in_the_declared_direction() {
        let descent = LegPremise::LearningHappened(TrainDirection::Descent);
        let ascent = LegPremise::LearningHappened(TrainDirection::Ascent);
        assert!(!fails(&leg(fused_facts()), descent.clone()));
        assert!(fails(&leg(fused_facts()), ascent.clone()));
        let rose = with(
            fused_facts(),
            "train_probe_series",
            json!([3.32, 3.4, 3.6, 3.9]),
        );
        assert!(fails(&rose, descent.clone()));
        assert!(!fails(&rose, ascent));
        let flat = with(
            fused_facts(),
            "train_probe_series",
            json!([3.32, 3.32, 3.32, 3.32]),
        );
        assert!(fails(&flat, descent.clone()));
        // Not anchored at the untrained model: one entry per epoch only.
        let unanchored = with(
            fused_facts(),
            "train_probe_series",
            json!([2.88, 2.74, 2.52]),
        );
        assert!(fails(&unanchored, descent.clone()));
        let mut absent = fused_facts();
        absent.as_object_mut().unwrap().remove("train_probe_series");
        assert!(fails(&leg(absent), descent));
    }

    #[test]
    fn schedule_admission_and_ties_are_checked_as_stated() {
        assert!(fails(
            &with(fused_facts(), "schedule", json!("cosine")),
            LegPremise::ConstantSchedule
        ));
        assert!(fails(
            &with(fused_facts(), "admission_is_dense", json!(true)),
            LegPremise::PaddedAdmission
        ));
        assert!(fails(
            &with(fused_facts(), "tie_fraction", json!(0.5)),
            LegPremise::TieFractionBelow(0.5)
        ));
        assert!(!fails(
            &with(fused_facts(), "tie_fraction", json!(0.49)),
            LegPremise::TieFractionBelow(0.5)
        ));
    }

    #[test]
    fn a_mutant_leg_is_stamped_with_the_patch_it_is_filed_under() {
        let stamp = |sha: &str| LegPremise::MutantStamp {
            patch_sha256: sha.into(),
        };
        let mut facts = fused_facts();
        assert!(fails(&leg(facts.clone()), stamp("AB12")));
        facts["mutant_id"] = json!("m");
        facts["mutant_base_sha"] = json!("base");
        facts["mutant_patch_sha256"] = json!("ab12");
        assert!(!fails(&leg(facts.clone()), stamp(" AB12 ")));
        assert!(fails(&leg(facts.clone()), stamp("cd34")));
        facts["mutant_base_sha"] = json!("  ");
        assert!(fails(&leg(facts), stamp("ab12")));
    }
}
