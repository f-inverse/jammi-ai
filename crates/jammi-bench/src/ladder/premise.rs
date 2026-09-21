//! Premises: facts a leg must show about itself before its numbers count.
//!
//! A leg's file name claims a rung. A premise checks the counted fact behind
//! the claim — a leg filed as the fused-kernel rung must have dispatched the
//! fused kernels, a leg filed as a training run must show that training
//! moved the model. A claim with no counted fact behind it is refused, never
//! assumed.

use serde::Serialize;

use crate::kernel_arm::{KernelArm, KernelFamily};

use super::leg::Leg;
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
    /// The dispatch counters prove the arm: every key the leg disabled
    /// belongs to a family the arm turns off and fired; a family turned off
    /// never ran fused and its fallback ran; a family left on never fell
    /// back, and ran fused unless a family absorbing it did.
    KernelArm(KernelArm),
    /// This family's kernel ran fused at least once.
    Dispatched(KernelFamily),
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
            Self::KernelArm(_) => "kernel_arm",
            Self::Dispatched(_) => "dispatched",
            Self::MutantStamp { .. } => "mutant_stamp",
        }
    }

    /// Why `leg` fails this premise, if it does.
    fn violation(&self, leg: &Leg) -> Option<String> {
        let facts = &leg.facts;
        match self {
            Self::ConstantSchedule => (leg.field_str("schedule") != Some("constant"))
                .then(|| format!("schedule is {:?}, not \"constant\"", leg.field_str("schedule"))),
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
                let stated = leg.provenance.as_ref().map(|p| p.arm.as_str());
                (stated != Some(arm)).then(|| format!("arm is {stated:?}, not {arm:?}"))
            }
            Self::KernelArm(arm) => dispatch::arm_violation(leg, arm),
            Self::Dispatched(family) => dispatch::dispatched_violation(leg, *family),
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
    let epochs = leg.field_u64("epochs").map(|e| e as usize);
    let expected = epochs.map_or(2, |epochs| epochs + 1);
    let exact = epochs.is_some();
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
    let Some(stamp) = leg.provenance.as_ref().map(|p| &p.mutant) else {
        return Some("no provenance: the leg is not attributable to any build".into());
    };
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
    use crate::kernel_arm::{KernelArm, KernelFamily};

    use super::Leg;

    /// Compute dtypes the flash cascade admits.
    const FLASH_DTYPES: [&str; 2] = ["bf16", "f16"];

    fn fused(leg: &Leg, base: &str) -> u64 {
        leg.dispatch.get(base).map_or(0, |p| p.fused)
    }

    fn fallback(leg: &Leg, base: &str) -> u64 {
        leg.dispatch.get(base).map_or(0, |p| p.fallback)
    }

    /// The family's counters on this leg, summed: `(fused, fallback)`.
    fn family_counts(leg: &Leg, family: KernelFamily) -> (u64, u64) {
        family.counters().iter().fold((0, 0), |(f, e), base| {
            (f + fused(leg, base), e + fallback(leg, base))
        })
    }

    /// Defects of the counter set itself, whatever the arm.
    fn schema_violation(leg: &Leg) -> Option<String> {
        if leg.dispatch.is_empty() {
            return Some(
                "no dispatch counters: the arm claim has no counted fact behind it".into(),
            );
        }
        let dtype = leg.compute_dtype().unwrap_or("<absent>");
        let flash = family_counts(leg, KernelFamily::FlashAttention).0;
        (flash > 0 && !FLASH_DTYPES.contains(&dtype)).then(|| {
            format!(
                "{flash} flash dispatches at dtype {dtype}, which the flash cascade does not admit: the report contradicts itself"
            )
        })
    }

    pub fn arm_violation(leg: &Leg, arm: &KernelArm) -> Option<String> {
        if let Some(defect) = schema_violation(leg) {
            return Some(defect);
        }
        let Some(provenance) = leg.provenance.as_ref() else {
            return Some("no provenance: the arm claim names no build".into());
        };
        let requested = &provenance.kernels_disabled_requested;
        let fired = &provenance.kernels_disabled_fired;
        if let Some(key) = requested
            .iter()
            .find(|key| KernelFamily::of_key(key).is_none_or(|f| !arm.is_off(f)))
        {
            return Some(format!(
                "kernels_disabled_requested names {key:?}, which no family of this arm switches"
            ));
        }
        if let Some(key) = requested.iter().find(|key| !fired.contains(key)) {
            return Some(format!("{key} was requested off but never fired"));
        }
        let off_and_fired = |family: KernelFamily| {
            arm.is_off(family) && family.keys().any(|k| fired.iter().any(|f| f == k))
        };
        for family in KernelFamily::ALL {
            let (fused, fallback) = family_counts(leg, family);
            if arm.is_off(family) {
                if fused > 0 {
                    return Some(format!(
                        "{family:?} is off on this arm yet dispatched fused {fused} time(s)"
                    ));
                }
                if off_and_fired(family) && !family.counters().is_empty() && fallback == 0 {
                    return Some(format!(
                        "{family:?} was disabled, yet no fallback dispatch is counted behind it"
                    ));
                }
            } else {
                if fallback > 0 {
                    return Some(format!("{family:?} fell back {fallback} time(s)"));
                }
                let consulted = fused > 0 || fallback > 0;
                let absorbed = family
                    .absorbers()
                    .any(|absorber| !arm.is_off(absorber) && family_counts(leg, absorber).0 > 0);
                if !consulted
                    && !family.counters().is_empty()
                    && !absorbed
                    && family.absorbed_by().is_some()
                {
                    return Some(format!(
                        "{family:?} never dispatched, and nothing absorbing it ran fused"
                    ));
                }
            }
        }
        None
    }

    pub fn dispatched_violation(leg: &Leg, family: KernelFamily) -> Option<String> {
        if family == KernelFamily::FlashAttention {
            let dtype = leg.compute_dtype().unwrap_or("<absent>");
            if !FLASH_DTYPES.contains(&dtype) {
                return Some(format!(
                    "dtype {dtype} cannot run the flash cascade this rung is defined by"
                ));
            }
            let compiled = leg.provenance.as_ref().map(|p| p.flash_compiled);
            if compiled != Some(true) {
                return Some(format!(
                    "flash_compiled is {compiled:?}: this build cannot run the flash cascade"
                ));
            }
        }
        (family_counts(leg, family).0 == 0)
            .then(|| format!("{family:?} never dispatched fused on this leg"))
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::ladder::definition::Workload;
    use std::sync::LazyLock;

    static FUSED_ARM: LazyLock<LegPremise> =
        LazyLock::new(|| LegPremise::KernelArm(KernelArm::fused()));
    static REFERENCE_ARM: LazyLock<LegPremise> = LazyLock::new(|| {
        LegPremise::KernelArm(KernelArm::off([
            KernelFamily::FlashAttention,
            KernelFamily::AdamW,
        ]))
    });
    const FLASH_RAN: LegPremise = LegPremise::Dispatched(KernelFamily::FlashAttention);
    const BLOCK_RAN: LegPremise = LegPremise::Dispatched(KernelFamily::AttentionBlock);
    use crate::ladder::leg::LegName;
    use serde_json::{json, Value};

    /// Dispatch counters and arm facts of a fused-kernel training leg, in the
    /// shape a real one carries.
    pub fn fused_facts() -> Value {
        json!({
            "arm": "fused", "attention_arm": "fused", "device_name": "gpu", "build_features": ["cuda"],
            "schedule": "constant", "admission_is_dense": false, "tie_fraction": 0.0,
            "epochs": 3, "train_probe_series": [3.32, 2.88, 2.74, 2.52],
            "backbone_dtype": "bf16", "flash_compiled": true,
            "kernels_disabled_requested": [], "kernels_disabled_fired": [],
            "ln_fused_dispatches": 6669, "ln_eager_dispatches": 0,
            "gelu_fused_dispatches": 0, "gelu_eager_dispatches": 0,
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
        assert!(!fails(&fused, FUSED_ARM.clone()));
        assert!(!fails(&fused, FLASH_RAN));
        assert!(!fails(&fused, LegPremise::Arm("fused")));
        assert!(fails(&fused, REFERENCE_ARM.clone()));
        assert!(!fails(&reference, REFERENCE_ARM.clone()));
        assert!(!fails(&reference, BLOCK_RAN));
        assert!(!fails(&reference, LegPremise::Arm("alloff")));
        assert!(fails(&reference, FUSED_ARM.clone()));
        assert!(fails(&reference, FLASH_RAN));
    }

    #[test]
    fn real_shaped_legs_clear_their_own_arm_and_fail_the_other() {
        let (fused, reference) = (leg(fused_facts()), leg(reference_facts()));
        assert!(!fails(&fused, FUSED_ARM.clone()));
        assert!(!fails(&reference, REFERENCE_ARM.clone()));
        assert!(fails(&reference, FUSED_ARM.clone()));
        assert!(fails(&fused, REFERENCE_ARM.clone()));
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
        let block_instead = leg(facts);
        assert!(!fails(&block_instead, FUSED_ARM.clone()));
        assert!(fails(&block_instead, FLASH_RAN));
        assert!(fails(
            &with(fused_facts(), "flash_compiled", json!(false)),
            FLASH_RAN
        ));
        assert!(fails(
            &with(fused_facts(), "backbone_dtype", json!("f32")),
            FLASH_RAN
        ));
        assert!(fails(
            &with(fused_facts(), "ln_eager_dispatches", json!(1)),
            FUSED_ARM.clone()
        ));
        // RoPE never dispatching is fine while the attention block or the
        // flash cascade absorbed it; with neither running fused it is not.
        let mut bare_attention = fused_facts();
        bare_attention["attention_block_flash_fused_dispatches"] = json!(0);
        assert!(fails(&leg(bare_attention), FUSED_ARM.clone()));
    }

    #[test]
    fn a_leg_with_no_counters_at_all_is_refused() {
        let bare = json!({"arm": "fused", "backbone_dtype": "bf16", "flash_compiled": true});
        assert!(fails(&leg(bare), FUSED_ARM.clone()));
    }

    #[test]
    fn a_reference_leg_needs_a_counted_fallback_behind_each_disable() {
        assert!(fails(
            &with(reference_facts(), "adamw_fused_dispatches", json!(5)),
            REFERENCE_ARM.clone()
        ));
        assert!(fails(
            &with(reference_facts(), "adamw_eager_dispatches", json!(0)),
            REFERENCE_ARM.clone()
        ));
        assert!(fails(
            &with(
                reference_facts(),
                "attention_block_fused_dispatches",
                json!(0)
            ),
            BLOCK_RAN
        ));
        // A key requested off outside the arm's families, and a key the arm
        // turns off that never fired, are both refused by name.
        assert!(fails(
            &with(
                reference_facts(),
                "kernels_disabled_requested",
                json!(["adamw_step_fused", "attention_block_flash", "geglu_fused"])
            ),
            REFERENCE_ARM.clone()
        ));
        assert!(fails(
            &with(
                reference_facts(),
                "kernels_disabled_fired",
                json!(["adamw_step_fused"])
            ),
            REFERENCE_ARM.clone()
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
