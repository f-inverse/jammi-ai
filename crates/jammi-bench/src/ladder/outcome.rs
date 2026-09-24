//! The outcome axis: did the two rungs produce the same artifact?
//!
//! Exact edges ask for digest equality. A cross-stack edge pairs what it can:
//! by seed, it reads both rungs at the point fixed before any leg ran — where
//! the reference learned the most — and asks two questions of the paired
//! differences: is there a directional difference, and is the upper rung no
//! worse than the lower by more than the margin. It keeps them apart,
//! because not finding a difference is not finding parity. The margin is not
//! chosen: it is a fraction of the learning effect the reference itself
//! establishes in the same session, and a reference that establishes none
//! leaves no margin to judge against. The claim is one-sided: an upper rung
//! that is *better* by more than the margin has not failed "as good as", so
//! two-sided equivalence is reported beside the claim and never judged in its
//! place. By row, it holds each row to a bound derived from the compute
//! precision. By tensor, it holds each gradient's direction to a measured
//! floor. Where nothing can be paired, each side is tested against the
//! workload's analytic law instead.

use std::collections::BTreeSet;
use std::str::FromStr;

use jammi_numerics::stats::{
    bootstrap_ci, goodness_of_fit, mean, paired_margin_test, population_std_dev, sign_test,
    sign_test_critical_count, Better, FitCell, FitStatistic, GoodnessOfFit,
};
use jammi_numerics::ComputePrecision;
use sha2::{Digest, Sha256};

use crate::leg::GradientTensor;

use super::definition::{
    rule, ControlRule, CrossStackOutcome, JudgedPoint, Margin, RowMetric, Rule, RuleForce,
    SpeedInstrument, MARGIN_BOOTSTRAP_ITERATIONS,
};
use super::leg::{Leg, RungLegs, Take, Unit, VectorsFile};
use super::premise::{self, LegPremise, LEARNING_FLOOR};
use super::refusal::Refusal;
use super::verdict::{
    Assay, Bound, ControlVerdict, Direction, Judgement, OutcomeVerdict, RepeatFloor,
    TensorAgreement, TensorKind, UnitDifference, DIRECTION_RULE, NON_INFERIORITY_RULE,
};

/// What an axis hands back: its verdict, the rules it applied, and what it
/// refused.
pub struct AxisResult<V> {
    pub verdict: Option<V>,
    pub judgements: Vec<Judgement>,
    pub refusals: Vec<Refusal>,
}

impl<V> AxisResult<V> {
    pub fn refused(refusals: Vec<Refusal>) -> Self {
        Self {
            verdict: None,
            judgements: vec![],
            refusals,
        }
    }

    /// No verdict on `rule`: refused when `refusals` say why the quantity
    /// could not be measured, unjudged when there was nothing to measure.
    pub fn unjudged(
        rule: &'static str,
        force: RuleForce,
        bound: Bound,
        detail: impl Into<String>,
        refusals: Vec<Refusal>,
    ) -> Self {
        let judgement = if refusals.is_empty() {
            Judgement::new(rule, force, bound, None, detail)
        } else {
            Judgement::refused(rule, force, bound, detail)
        };
        Self {
            verdict: None,
            judgements: vec![judgement],
            refusals,
        }
    }
}

/// The two rungs of an edge and the units both were measured at.
pub struct Pair<'a> {
    pub edge: &'a str,
    pub lower: &'a RungLegs,
    pub upper: &'a RungLegs,
    pub units: &'a [Unit],
    /// Units whose legs failed a premise: measured and reported, never
    /// counted.
    pub unclean: &'a BTreeSet<Unit>,
}

/// Which of a seeded edge's claims a comparison judges. A mutant column — a
/// detection instrument standing in for the upper rung — judges direction
/// alone: it has no controls of its own and no margin to keep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Claims {
    pub controls: bool,
    pub margin: bool,
}

impl Claims {
    pub const ALL: Self = Self {
        controls: true,
        margin: true,
    };
    pub const DIRECTION_ONLY: Self = Self {
        controls: false,
        margin: false,
    };
}

const DIGEST_RULE: &str = "outcome_digests_equal";

/// Every repeat of both rungs in a unit carries one digest, and they agree.
/// On an exact edge a disagreement is refused: the engine is deterministic,
/// so two digests are two artifacts. Judged at `force` otherwise.
pub fn digests(pair: &Pair<'_>, force: RuleForce) -> AxisResult<OutcomeVerdict> {
    let mut missing = vec![];
    let mut mismatches = vec![];
    let mut units_equal = 0;
    for unit in pair.units {
        let legs: Vec<&Leg> = pair
            .lower
            .repeats(unit)
            .chain(pair.upper.repeats(unit))
            .collect();
        let digests: Vec<(String, String)> = legs
            .iter()
            .filter_map(|leg| Some((leg.name.to_string(), leg.measured.outcome_digest.clone()?)))
            .collect();
        if digests.len() < legs.len() {
            missing.extend(
                legs.iter()
                    .filter(|l| l.measured.outcome_digest.is_none())
                    .map(|l| Refusal::MeasurementMissing {
                        subject: l.name.to_string(),
                        measurement: "outcome_digest",
                    }),
            );
        } else if digests.iter().any(|(_, d)| *d != digests[0].1) {
            mismatches.push(Refusal::DigestMismatch {
                unit: unit.as_str().to_owned(),
                digests,
            });
        } else {
            units_equal += 1;
        }
    }
    let verdict = Some(OutcomeVerdict::Digest {
        units_equal,
        units: pair.units.len(),
    });
    match force {
        RuleForce::Hard => AxisResult {
            verdict,
            judgements: vec![],
            refusals: missing.into_iter().chain(mismatches).collect(),
        },
        RuleForce::Evidence => AxisResult {
            verdict,
            judgements: vec![Judgement::new(
                DIGEST_RULE,
                force,
                Bound::None,
                missing.is_empty().then_some(mismatches.is_empty()),
                if missing.is_empty() {
                    format!(
                        "digests equal on {units_equal} of {} unit(s)",
                        pair.units.len()
                    )
                } else {
                    format!("{} leg(s) carry no outcome_digest", missing.len())
                },
            )],
            refusals: vec![],
        },
    }
}

#[derive(Debug, Clone, Default)]
pub struct CrossStackOptions {
    /// The operator states the control was not run. Recorded in the verdict.
    pub waive_control: bool,
    /// Where a law workload's ground truth lives: `<unit>.json` per unit,
    /// `{"cells": [[probability, ...], ...]}`.
    pub law_dir: Option<std::path::PathBuf>,
}

pub fn cross_stack(
    pair: &Pair<'_>,
    rules: &CrossStackOutcome,
    lower_premises: &[LegPremise],
    upper_premises: &[LegPremise],
    options: &CrossStackOptions,
    claims: Claims,
) -> AxisResult<OutcomeVerdict> {
    match rules {
        CrossStackOutcome::RowAgreement { metric, force } => row_agreement(pair, *metric, *force),
        CrossStackOutcome::GradientAgreement {
            take,
            cosine_floor,
            structure,
        } => gradient_agreement(
            pair,
            take,
            cosine_floor,
            *structure,
            [lower_premises, upper_premises],
        ),
        CrossStackOutcome::Law {
            statistic,
            alpha,
            force,
        } => law(pair, *statistic, *alpha, *force, options.law_dir.as_deref()),
        CrossStackOutcome::SeededLoss {
            seeds,
            sign_alpha,
            direction_force,
            judged_at,
            margin,
            control,
        } => {
            let mut result = seeded_loss(
                pair,
                *seeds,
                *sign_alpha,
                *direction_force,
                *judged_at,
                claims.margin.then_some(*margin),
            );
            let control_verdict = control.as_ref().filter(|_| claims.controls).map(|rule| {
                let (verdict, refusals) =
                    controls(pair, rule, lower_premises, upper_premises, options);
                result.refusals.extend(refusals);
                verdict
            });
            if let Some(OutcomeVerdict::SeededLoss { control, .. }) = &mut result.verdict {
                *control = control_verdict;
            }
            result
        }
    }
}

/// The outcome of a seeded edge is a loss, and `d = upper − lower`: the upper
/// rung is no worse when `d` is not high.
const LOSS: Better = Better::Lower;

/// Per unit: the judged epoch and both rungs' held-out loss there, plus the
/// lower rung's improvement from its untrained loss.
fn unit_difference(pair: &Pair<'_>, unit: &Unit) -> UnitDifference {
    let reference = pair.lower.primary(unit);
    let judged = reference.and_then(Leg::held_out_minimum);
    let epoch = judged.map(|(epoch, _)| epoch);
    let lower = judged.map(|(_, loss)| loss);
    let upper = epoch.and_then(|e| pair.upper.primary(unit)?.held_out_at(e));
    let init = reference.and_then(|leg| leg.measured.held_out_at_init);
    UnitDifference {
        unit: unit.as_str().to_owned(),
        epoch,
        lower,
        upper,
        d: lower.zip(upper).map(|(l, u)| u - l),
        reference_improvement: init.zip(lower).map(|(i, l)| i - l),
        clean: !pair.unclean.contains(unit),
    }
}

/// What a unit's legs must carry to be read at the judged point.
fn judged_point_refusals(pair: &Pair<'_>, unit: &Unit, margin: bool) -> Vec<Refusal> {
    let (Some(lower), Some(upper)) = (pair.lower.primary(unit), pair.upper.primary(unit)) else {
        return vec![];
    };
    let missing = |subject: String, measurement: &'static str| Refusal::MeasurementMissing {
        subject,
        measurement,
    };
    let mut refusals = vec![];
    match lower.held_out_minimum() {
        None => refusals.push(missing(lower.name.to_string(), "trajectory")),
        Some((epoch, _)) if upper.held_out_at(epoch).is_none() => refusals.push(missing(
            format!("{} at epoch {epoch}", upper.name),
            "trajectory",
        )),
        Some(_) => {}
    }
    if margin && lower.measured.held_out_at_init.is_none() {
        refusals.push(missing(lower.name.to_string(), "held_out_at_init"));
    }
    refusals
}

fn seeded_loss(
    pair: &Pair<'_>,
    seeds: usize,
    sign_alpha: f64,
    direction_force: RuleForce,
    judged_at: JudgedPoint,
    margin: Option<Margin>,
) -> AxisResult<OutcomeVerdict> {
    let JudgedPoint::ReferenceMinimum = judged_at;
    let per_unit: Vec<UnitDifference> = pair
        .units
        .iter()
        .map(|unit| unit_difference(pair, unit))
        .collect();
    let mut refusals: Vec<Refusal> = pair
        .units
        .iter()
        .flat_map(|unit| judged_point_refusals(pair, unit, margin.is_some()))
        .collect();

    // How much the differences vary across units, from every difference
    // that could be computed: the spread is a fact about the measurement,
    // whether or not a unit's premises let it count.
    let all_d: Vec<f64> = per_unit.iter().filter_map(|u| u.d).collect();
    let spread = if all_d.len() >= 2 {
        population_std_dev(&all_d).unwrap_or(0.0)
    } else {
        0.0
    };
    let mut max_delta = 0.0_f64;
    for (rung_name, rung) in [("lower", pair.lower), ("upper", pair.upper)] {
        for (unit, difference) in pair.units.iter().zip(&per_unit) {
            let Some(epoch) = difference.epoch else {
                continue;
            };
            let outcomes: Vec<f64> = rung
                .repeats(unit)
                .filter_map(|leg| leg.held_out_at(epoch))
                .collect();
            let delta = outcomes
                .iter()
                .skip(1)
                .map(|v| (v - outcomes[0]).abs())
                .fold(0.0, f64::max);
            max_delta = max_delta.max(delta);
            if delta > spread {
                refusals.push(Refusal::RepeatExceedsSpread {
                    rung: rung
                        .primary(unit)
                        .map_or_else(|| rung_name.to_owned(), |leg| leg.name.rung.clone()),
                    unit: unit.as_str().to_owned(),
                    delta,
                    spread,
                });
            }
        }
    }

    let clean: Vec<&UnitDifference> = per_unit.iter().filter(|u| u.clean).collect();
    let clean_d: Vec<f64> = clean.iter().filter_map(|u| u.d).collect();
    if clean_d.len() != seeds {
        refusals.push(Refusal::WrongUnitCount {
            edge: pair.edge.to_owned(),
            clean: clean_d.len(),
            required: seeds,
        });
    }
    let context = || format!("edge {} sign test", pair.edge);
    let sign = (!clean_d.is_empty())
        .then(|| sign_test(&clean_d))
        .transpose()
        .unwrap_or_else(|e| {
            refusals.push(Refusal::statistics(context(), e));
            None
        });
    let critical_count = sign_test_critical_count(seeds, sign_alpha).unwrap_or_else(|e| {
        refusals.push(Refusal::statistics(context(), e));
        None
    });
    let mean_d = mean(&clean_d).ok();
    // The count is taken over the units the rule is stated for, not over
    // the untied ones: a unit that ties is a unit that did not concord.
    let direction = match (sign, critical_count, mean_d) {
        (Some(s), Some(k), Some(m)) if s.n_pos >= k && m > 0.0 => Direction::Degradation,
        (Some(s), Some(k), Some(m)) if s.n_neg >= k && m < 0.0 => Direction::Improvement,
        _ => Direction::None,
    };
    let mut judgements = vec![Judgement::directed(
        DIRECTION_RULE,
        direction_force,
        Bound::Level { alpha: sign_alpha },
        sign.map(|_| direction == Direction::None),
        direction,
        match sign {
            Some(s) => format!(
                "{} of {} higher, {} lower; {:?} needed either way at alpha {sign_alpha}",
                s.n_pos,
                clean_d.len(),
                s.n_neg,
                critical_count
            ),
            None => "no paired difference to test".to_owned(),
        },
    )];

    let mut assay = None;
    let mut margin_test = None;
    let improvements: Vec<f64> = clean
        .iter()
        .filter_map(|u| u.reference_improvement)
        .collect();
    // A unit whose reference did not record its untrained loss is refused
    // above; the effect is read only when every clean unit has one.
    if let Some(margin) = margin.filter(|_| !clean.is_empty() && improvements.len() == clean.len())
    {
        match reference_effect(&improvements, margin, pair.edge) {
            Err(refusal) => refusals.push(refusal),
            Ok(effect) => {
                if !effect.sensitive {
                    refusals.push(Refusal::AssayInsensitive {
                        edge: pair.edge.to_owned(),
                        mean_improvement: effect.mean_improvement,
                        lower_bound: effect.established_effect,
                    });
                } else {
                    let (test, judged) = margin_claims(&clean_d, effect.delta, margin, pair.edge)
                        .unwrap_or_else(|refusal| {
                            refusals.push(refusal);
                            (None, vec![])
                        });
                    judgements.extend(judged);
                    margin_test = test;
                }
                assay = Some(effect);
            }
        }
    }

    AxisResult {
        verdict: Some(OutcomeVerdict::SeededLoss {
            judged_at,
            clean_units: clean_d.len(),
            per_unit,
            sign_test: sign,
            critical_count,
            mean_d,
            direction,
            assay,
            margin_test,
            repeat_floor: RepeatFloor { max_delta, spread },
            control: None,
        }),
        judgements,
        refusals,
    }
}

/// The reference rung's learning effect over the clean units, lower-bounded
/// at the margin's level, and the margin derived from it. The effect is
/// established when the bound is positive; a bound at or below zero is a
/// reference that has not been shown to learn, and there is then no effect
/// a fraction of which could be a margin.
fn reference_effect(improvements: &[f64], margin: Margin, edge: &str) -> Result<Assay, Refusal> {
    let context = || format!("edge {edge} reference effect");
    let interval = bootstrap_ci(
        improvements,
        |s| s.iter().sum::<f64>() / s.len() as f64,
        MARGIN_BOOTSTRAP_ITERATIONS,
        2.0 * margin.alpha,
        SpeedInstrument::BOOTSTRAP_SEED,
    )
    .map_err(|e| Refusal::statistics(context(), e))?;
    let mean_improvement = mean(improvements).map_err(|e| Refusal::statistics(context(), e))?;
    let established_effect = interval.lower;
    Ok(Assay {
        mean_improvement,
        interval,
        established_effect,
        delta: margin.preserved_effect_fraction * established_effect,
        sensitive: established_effect > 0.0,
    })
}

/// Both one-sided tests of the mean paired difference against `±delta`:
/// non-inferiority as the claim, equivalence as evidence beside it.
type MarginClaims = (
    Option<jammi_numerics::stats::MarginTestResult>,
    Vec<Judgement>,
);

fn margin_claims(
    clean_d: &[f64],
    delta: f64,
    margin: Margin,
    edge: &str,
) -> Result<MarginClaims, Refusal> {
    if clean_d.len() < 2 {
        return Ok((None, vec![]));
    }
    let result = paired_margin_test(
        clean_d,
        delta,
        MARGIN_BOOTSTRAP_ITERATIONS,
        margin.alpha,
        SpeedInstrument::BOOTSTRAP_SEED,
    )
    .map_err(|e| Refusal::statistics(format!("edge {edge} margin test"), e))?;
    let detail = |bound: String| {
        format!(
            "mean d in [{:.6}, {:.6}] against {bound}",
            result.interval.lower, result.interval.upper
        )
    };
    let judgements = vec![
        Judgement::new(
            NON_INFERIORITY_RULE,
            margin.non_inferiority_force,
            Bound::Derived { value: delta },
            Some(result.non_inferior(LOSS)),
            detail(format!("an upper bound below +{delta:.6}")),
        ),
        Judgement::new(
            "equivalent_within_delta",
            RuleForce::Evidence,
            Bound::Derived { value: delta },
            Some(result.equivalent()),
            detail(format!("±{delta:.6}")),
        ),
    ];
    Ok((Some(result), judgements))
}

/// The control legs: runs that *cannot* learn, which the learning premise
/// must therefore refuse. A control that learns is a finding against the
/// premise's floor, not a training result.
fn controls(
    pair: &Pair<'_>,
    rule: &ControlRule,
    lower_premises: &[LegPremise],
    upper_premises: &[LegPremise],
    options: &CrossStackOptions,
) -> (ControlVerdict, Vec<Refusal>) {
    let of = |rung: &'_ RungLegs| -> Vec<Leg> {
        rung.controls()
            .filter(|leg| leg.name.take == Take::Control(rule.take.to_owned()))
            .cloned()
            .collect()
    };
    let (lower, upper) = (of(pair.lower), of(pair.upper));
    let units_of = |legs: &[Leg]| {
        legs.iter()
            .map(|l| l.name.unit.clone())
            .collect::<BTreeSet<_>>()
    };
    let units: Vec<Unit> = units_of(&lower)
        .intersection(&units_of(&upper))
        .cloned()
        .collect();

    let mut refusals = vec![];
    if units.len() < rule.required_units && !options.waive_control {
        refusals.push(Refusal::ControlMissing {
            edge: pair.edge.to_owned(),
            found: units.len(),
            required: rule.required_units,
        });
    }
    let except_learning = |premises: &[LegPremise]| -> Vec<LegPremise> {
        premises
            .iter()
            .filter(|p| !matches!(p, LegPremise::LearningHappened(_)))
            .cloned()
            .collect()
    };
    for (legs, premises) in [
        (&lower, except_learning(lower_premises)),
        (&upper, except_learning(upper_premises)),
    ] {
        for leg in legs {
            refusals.extend(premise::violations(leg, &premises));
            let invalid = |reason: String| Refusal::ControlInvalid {
                leg: leg.name.to_string(),
                reason,
            };
            if leg.identity_number(rule.field) != Some(rule.value) {
                refusals.push(invalid(format!(
                    "{} is {:?}, not {}: a control that never ran as one validates nothing",
                    rule.field,
                    leg.identity_number(rule.field),
                    rule.value
                )));
            }
            match premise::probe_movement(leg) {
                Err(reason) => refusals.push(invalid(reason)),
                Ok(moved) if moved.abs() > LEARNING_FLOOR => refusals.push(invalid(format!(
                    "the train probe moved by {moved} in a run that cannot learn: the learning floor {LEARNING_FLOOR} does not separate learning from none"
                ))),
                Ok(_) => {}
            }
        }
    }
    (
        ControlVerdict {
            units: units.iter().map(|u| u.as_str().to_owned()).collect(),
            waived: options.waive_control && units.len() < rule.required_units,
        },
        refusals,
    )
}

/// The relative perturbation two stacks' rows may differ by at a compute
/// precision: `√ε`. A well-conditioned computation carried out at machine
/// epsilon `ε` agrees with its twin to half its fraction bits; past that, the
/// two disagree in digits rounding alone does not reach.
fn perturbation_allowance(precision: ComputePrecision) -> f64 {
    precision.machine_epsilon().sqrt()
}

impl RowMetric {
    /// The bound a row is held to. A relative perturbation `η` of a vector
    /// costs `η² / 2` of cosine, so the one allowance gives both bounds.
    pub fn bound(self, precision: ComputePrecision) -> f64 {
        let eta = perturbation_allowance(precision);
        match self {
            Self::Cosine => 1.0 - eta * eta / 2.0,
            Self::RelativeError => eta,
        }
    }

    fn measure(self, lower: &[f64], upper: &[f64]) -> f64 {
        match self {
            Self::Cosine => cosine(lower, upper),
            Self::RelativeError => relative_error(lower, upper),
        }
    }

    fn beyond(self, value: f64, bound: f64) -> bool {
        match self {
            Self::Cosine => value < bound,
            Self::RelativeError => value > bound,
        }
    }

    /// The worse of two measurements.
    fn worse(self, a: f64, b: f64) -> f64 {
        match self {
            Self::Cosine => a.min(b),
            Self::RelativeError => a.max(b),
        }
    }
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    dot / (norm(a) * norm(b))
}

/// `‖b − a‖ ÷ ‖a‖`.
fn relative_error(a: &[f64], b: &[f64]) -> f64 {
    let difference: Vec<f64> = a.iter().zip(b).map(|(x, y)| y - x).collect();
    norm(&difference) / norm(a)
}

fn read_vectors(file: &VectorsFile) -> Result<Vec<Vec<f64>>, String> {
    let bytes = std::fs::read(&file.path).map_err(|e| format!("{}: {e}", file.path.display()))?;
    let row_bytes = file.dim * 4;
    if file.dim == 0 || bytes.is_empty() || bytes.len() % row_bytes != 0 {
        return Err(format!(
            "{}: {} bytes is not a whole number of {}-dimensional f32 rows",
            file.path.display(),
            bytes.len(),
            file.dim
        ));
    }
    Ok(bytes
        .chunks_exact(row_bytes)
        .map(|row| {
            row.chunks_exact(4)
                .map(|b| f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
                .collect()
        })
        .collect())
}

const ROW_AGREEMENT_RULE: &str = "row_agreement";

fn row_agreement(
    pair: &Pair<'_>,
    metric: RowMetric,
    force: RuleForce,
) -> AxisResult<OutcomeVerdict> {
    let mut refusals = vec![];
    let mut measured: Vec<f64> = vec![];
    let mut bound: Option<f64> = None;
    for unit in pair.units {
        let malformed = |reason: String| Refusal::VectorsMalformed {
            unit: unit.as_str().to_owned(),
            reason,
        };
        let (Some(lower), Some(upper)) = (pair.lower.primary(unit), pair.upper.primary(unit))
        else {
            continue;
        };
        let (Some(lower_file), Some(upper_file)) = (&lower.vectors, &upper.vectors) else {
            refusals.extend(
                [lower, upper]
                    .iter()
                    .filter(|l| l.vectors.is_none())
                    .map(|l| Refusal::MeasurementMissing {
                        subject: l.name.to_string(),
                        measurement: "vectors_file",
                    }),
            );
            continue;
        };
        match upper.compute_dtype().map(ComputePrecision::from_str) {
            Some(Ok(precision)) => {
                // Identity holds the dtype equal across units, so every unit
                // derives this same bound.
                bound = Some(metric.bound(precision));
            }
            other => {
                refusals.push(malformed(format!(
                    "compute dtype {other:?} names no precision to derive a bound from"
                )));
                continue;
            }
        }
        match (read_vectors(lower_file), read_vectors(upper_file)) {
            (Ok(a), Ok(b)) if a.len() == b.len() && lower_file.dim == upper_file.dim => {
                measured.extend(a.iter().zip(&b).map(|(x, y)| metric.measure(x, y)));
            }
            (Ok(a), Ok(b)) => refusals.push(malformed(format!(
                "{}x{} against {}x{}: the two stacks did not produce the same keys",
                a.len(),
                lower_file.dim,
                b.len(),
                upper_file.dim
            ))),
            (a, b) => refusals.extend([a.err(), b.err()].into_iter().flatten().map(malformed)),
        }
    }
    let (Some(bound), false) = (bound, measured.is_empty()) else {
        return AxisResult::unjudged(
            ROW_AGREEMENT_RULE,
            force,
            Bound::None,
            "no vectors to compare",
            refusals,
        );
    };
    if measured.iter().any(|m| !m.is_finite()) {
        refusals.push(Refusal::VectorsMalformed {
            unit: "*".to_owned(),
            reason: "a row has no direction (zero or non-finite vector)".to_owned(),
        });
    }
    let worst = measured
        .iter()
        .copied()
        .reduce(|a, b| metric.worse(a, b))
        .unwrap_or(f64::NAN);
    let rows_beyond_bound = measured
        .iter()
        .filter(|m| metric.beyond(**m, bound))
        .count();
    AxisResult {
        verdict: Some(OutcomeVerdict::RowAgreement {
            metric,
            bound,
            rows: measured.len(),
            worst,
            rows_beyond_bound,
        }),
        judgements: vec![Judgement::new(
            ROW_AGREEMENT_RULE,
            force,
            Bound::Derived { value: bound },
            Some(rows_beyond_bound == 0),
            format!("worst row {worst:.3e} against {bound:.3e}"),
        )],
        refusals,
    }
}

const GRADIENT_STRUCTURE_RULE: &str = "gradient_structure";

/// The leg of `rung` at `unit` filed under `take`.
fn take_leg<'a>(rung: &'a RungLegs, unit: &Unit, take: &Take) -> Option<&'a Leg> {
    rung.units
        .get(unit)?
        .iter()
        .find(|leg| leg.name.take == *take)
}

fn widened(v: &[f32]) -> Vec<f64> {
    v.iter().copied().map(f64::from).collect()
}

/// One tensor's pair of gradients, classified. A zero gradient is exactly
/// zero: `dL/dA` at a zero adapter factor `B` is a product with `B`, not a
/// small number. Weights are the same file loaded twice and widened to
/// `f32`, so they agree bit for bit or the two sides did not share them.
fn tensor_agreement(
    unit: &Unit,
    name: &str,
    lower: &GradientTensor,
    upper: &GradientTensor,
) -> Result<TensorAgreement, String> {
    if lower.weight != upper.weight {
        return Err(format!(
            "{}: {name}: the two sides did not load the same weights",
            unit.as_str()
        ));
    }
    if lower.shape != upper.shape || lower.grad.len() != upper.grad.len() {
        return Err(format!(
            "{}: {name}: gradient shapes {:?} and {:?} differ",
            unit.as_str(),
            lower.shape,
            upper.shape
        ));
    }
    let finite = |t: &GradientTensor| t.grad.iter().all(|x| x.is_finite());
    let zero = |t: &GradientTensor| t.grad.iter().all(|x| *x == 0.0);
    let kind = match (finite(lower) && finite(upper), zero(lower), zero(upper)) {
        (false, _, _) => TensorKind::NonFinite,
        (true, true, true) => TensorKind::Vacuous,
        (true, true, false) | (true, false, true) => TensorKind::OneSidedZero,
        (true, false, false) => TensorKind::Signal,
    };
    let (a, b) = (widened(&lower.grad), widened(&upper.grad));
    let signal = kind == TensorKind::Signal;
    Ok(TensorAgreement {
        unit: unit.as_str().to_owned(),
        name: name.to_owned(),
        kind,
        cosine: signal.then(|| cosine(&a, &b)),
        relative_error: signal.then(|| relative_error(&a, &b)),
    })
}

/// Gradient agreement at shared weights, over the units both rungs carry a
/// `take` leg for. A gradient leg claims its rung like any other and is
/// held to the rung's premises. Per tensor: both gradients zero is vacuous;
/// exactly one zero, a non-finite entry, a tensor one side lacks, or
/// weights that differ breaks the structure; a real pair's cosine is held
/// to the measured floor.
fn gradient_agreement(
    pair: &Pair<'_>,
    take: &'static str,
    cosine_floor: &Rule,
    structure: RuleForce,
    premises: [&[LegPremise]; 2],
) -> AxisResult<OutcomeVerdict> {
    let wanted = Take::Control(take.to_owned());
    let of = |rung, unit| take_leg(rung, unit, &wanted);
    let mut refusals = vec![];
    let mut breaks: Vec<String> = vec![];
    let mut tensors: Vec<TensorAgreement> = vec![];
    let mut paired_units = 0;
    for unit in pair.units {
        let (lower, upper) = (of(pair.lower, unit), of(pair.upper, unit));
        let (Some(lower), Some(upper)) = (lower, upper) else {
            if let Some(present) = lower.or(upper) {
                let absent = if lower.is_none() {
                    pair.lower
                } else {
                    pair.upper
                };
                refusals.push(Refusal::MissingLeg {
                    rung: absent
                        .all()
                        .next()
                        .map_or_else(|| present.name.rung.clone(), |l| l.name.rung.clone()),
                    unit: unit.as_str().to_owned(),
                    take: take.to_owned(),
                });
            }
            continue;
        };
        for (leg, rung_premises) in [lower, upper].into_iter().zip(premises) {
            refusals.extend(premise::violations(leg, rung_premises));
        }
        let (Some(a), Some(b)) = (&lower.measured.gradients, &upper.measured.gradients) else {
            refusals.extend(
                [lower, upper]
                    .into_iter()
                    .filter(|l| l.measured.gradients.is_none())
                    .map(|l| Refusal::MeasurementMissing {
                        subject: l.name.to_string(),
                        measurement: "gradients",
                    }),
            );
            continue;
        };
        paired_units += 1;
        for name in a.keys().chain(b.keys()).collect::<BTreeSet<_>>() {
            match (a.get(name), b.get(name)) {
                (Some(ga), Some(gb)) => match tensor_agreement(unit, name, ga, gb) {
                    Ok(tensor) => {
                        if !matches!(tensor.kind, TensorKind::Signal | TensorKind::Vacuous) {
                            breaks.push(format!(
                                "{}: {name}: {}",
                                unit.as_str(),
                                super::verdict::serde_plain(&tensor.kind)
                            ));
                        }
                        tensors.push(tensor);
                    }
                    Err(reason) => breaks.push(reason),
                },
                _ => breaks.push(format!("{}: {name} is on one side only", unit.as_str())),
            }
        }
    }
    let signal: Vec<&TensorAgreement> = tensors
        .iter()
        .filter(|t| t.kind == TensorKind::Signal)
        .collect();
    let worst_cosine = signal.iter().filter_map(|t| t.cosine).reduce(f64::min);
    let worst_relative_error = signal
        .iter()
        .filter_map(|t| t.relative_error)
        .reduce(f64::max);
    let mean_cosine = (!signal.is_empty())
        .then(|| signal.iter().filter_map(|t| t.cosine).sum::<f64>() / signal.len() as f64);
    let judgements = vec![
        Judgement::new(
            GRADIENT_STRUCTURE_RULE,
            structure,
            Bound::None,
            (paired_units > 0).then_some(breaks.is_empty()),
            if paired_units == 0 {
                format!("no unit carries a `{take}` leg on both rungs")
            } else if breaks.is_empty() {
                format!(
                    "{} tensor(s) paired over {paired_units} unit(s), {} vacuous (both gradients zero)",
                    tensors.len(),
                    tensors.iter().filter(|t| t.kind == TensorKind::Vacuous).count()
                )
            } else {
                breaks.join("; ")
            },
        ),
        Judgement::budgeted(
            rule::GRADIENT_COSINE_FLOOR,
            cosine_floor,
            worst_cosine,
            |worst, floor| worst >= floor,
            match (worst_cosine, worst_relative_error) {
                (Some(c), Some(e)) => format!(
                    "worst tensor cosine {c:.9}, worst relative error {e:.3e}, against {}",
                    Bound::of(cosine_floor)
                ),
                _ => "no tensor carries a signal on both sides".to_owned(),
            },
        ),
    ];
    AxisResult {
        verdict: (paired_units > 0).then_some(OutcomeVerdict::GradientAgreement {
            tensors,
            mean_cosine,
            worst_cosine,
            worst_relative_error,
        }),
        judgements,
        refusals,
    }
}

/// A workload's analytic ground truth for one unit: the probability of each
/// category of each cell.
#[derive(serde::Deserialize)]
struct LawFile {
    cells: Vec<Vec<f64>>,
}

/// The law for `unit`, checked against the digest both legs ran under: the
/// ground truth is a committed artifact, never a producer's claim.
fn read_law(dir: &std::path::Path, unit: &Unit, legs: [&Leg; 2]) -> Result<LawFile, Refusal> {
    let path = dir.join(format!("{}.json", unit.as_str()));
    let unusable = |reason: String| Refusal::LawUnusable {
        file: path.display().to_string(),
        reason,
    };
    let bytes = std::fs::read(&path).map_err(|e| unusable(e.to_string()))?;
    let digest = format!("\"{}\"", hex::encode(Sha256::digest(&bytes)));
    if let Some(leg) = legs
        .iter()
        .find(|leg| leg.identity.get("law_sha256") != Some(&Some(digest.clone())))
    {
        return Err(unusable(format!(
            "its sha256 is {digest}; leg {} ran under law_sha256 {:?}",
            leg.name,
            leg.identity.get("law_sha256").cloned().flatten()
        )));
    }
    serde_json::from_slice(&bytes).map_err(|e| unusable(e.to_string()))
}

const LAW_RULE: &str = "law_goodness_of_fit";

/// What a law fit tested of what it was given.
fn coverage(fit: &GoodnessOfFit) -> String {
    format!(
        "{} states tested at {} dof, {} categories pooled, {} states untested holding {} of {} steps",
        fit.cells_tested,
        fit.degrees_of_freedom,
        fit.categories_pooled,
        fit.cells_untested,
        fit.observations_untested,
        fit.observations + fit.observations_untested
    )
}

fn law(
    pair: &Pair<'_>,
    statistic: FitStatistic,
    alpha: f64,
    force: RuleForce,
    law_dir: Option<&std::path::Path>,
) -> AxisResult<OutcomeVerdict> {
    let level = || Bound::Level { alpha };
    let unjudged =
        |detail: &str, refusals| AxisResult::unjudged(LAW_RULE, force, level(), detail, refusals);
    let Some(dir) = law_dir else {
        return unjudged("no law directory was given", vec![]);
    };
    let mut refusals = vec![];
    let mut laws: Vec<(LawFile, [&Leg; 2])> = vec![];
    for unit in pair.units {
        let (Some(lower), Some(upper)) = (pair.lower.primary(unit), pair.upper.primary(unit))
        else {
            continue;
        };
        match read_law(dir, unit, [lower, upper]) {
            Ok(file) => laws.push((file, [lower, upper])),
            Err(refusal) => refusals.push(refusal),
        }
    }
    let fit = |side: usize, refusals: &mut Vec<Refusal>| -> Option<GoodnessOfFit> {
        let mut cells = vec![];
        for (file, legs) in &laws {
            let leg = legs[side];
            let Some(observed) = leg.measured.law_observed.as_deref() else {
                refusals.push(Refusal::MeasurementMissing {
                    subject: leg.name.to_string(),
                    measurement: "law_observed",
                });
                return None;
            };
            if observed.len() != file.cells.len() {
                refusals.push(Refusal::statistics(
                    leg.name.to_string(),
                    jammi_numerics::NumericsError::DimensionMismatch {
                        expected: file.cells.len(),
                        got: observed.len(),
                    },
                ));
                return None;
            }
            cells.extend(
                observed
                    .iter()
                    .zip(&file.cells)
                    .map(|(observed, probabilities)| FitCell {
                        observed,
                        probabilities,
                    }),
            );
        }
        goodness_of_fit(&cells, statistic)
            .map_err(|e| {
                refusals.push(Refusal::statistics(
                    format!("edge {} law fit", pair.edge),
                    e,
                ))
            })
            .ok()
    };
    let (lower, upper) = (fit(0, &mut refusals), fit(1, &mut refusals));
    let (Some(lower), Some(upper)) = (lower, upper) else {
        return unjudged("the law could not be tested", refusals);
    };
    AxisResult {
        verdict: Some(OutcomeVerdict::Law {
            alpha,
            lower,
            upper,
        }),
        judgements: vec![Judgement::new(
            LAW_RULE,
            force,
            level(),
            Some(lower.p_value >= alpha && upper.p_value >= alpha),
            format!(
                "p = {:.4} (lower: {}), {:.4} (upper: {}) against alpha {alpha}",
                lower.p_value,
                coverage(&lower),
                upper.p_value,
                coverage(&upper)
            ),
        )],
        refusals,
    }
}
