//! The outcome axis: did the two rungs produce the same artifact?
//!
//! Exact edges ask for digest equality. A cross-stack edge pairs what it can:
//! by seed, it asks two questions of the paired differences — is there a
//! directional difference, and is the upper rung no worse than the lower by
//! more than the margin — and keeps them apart, because not finding a
//! difference is not finding parity. The claim is one-sided: an upper rung
//! that is *better* by more than the margin has not failed "as good as", so
//! two-sided equivalence is reported beside the claim and never judged in its
//! place. By row, it holds each row to a bound derived from the compute
//! precision. Where nothing can be paired, each side is tested against the
//! workload's analytic law instead.

use std::collections::BTreeSet;
use std::str::FromStr;

use jammi_numerics::stats::{
    goodness_of_fit, mean, paired_margin_test, population_std_dev, sign_test,
    sign_test_critical_count, Better, FitCell, FitStatistic, GoodnessOfFit,
};
use jammi_numerics::ComputePrecision;
use sha2::{Digest, Sha256};

use super::definition::{
    ControlRule, CrossStackOutcome, Gate, RowMetric, SpeedInstrument, MARGIN_BOOTSTRAP_ITERATIONS,
};
use super::leg::{Leg, RungLegs, Take, Unit, VectorsFile};
use super::premise::{self, LegPremise, LEARNING_FLOOR};
use super::refusal::Refusal;
use super::verdict::{
    ControlVerdict, Direction, Judgement, OutcomeVerdict, RepeatFloor, UnitDifference,
    DIRECTION_RULE, NON_INFERIORITY_RULE,
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

const DIGEST_RULE: &str = "outcome_digests_equal";

/// Every repeat of both rungs in a unit carries one digest, and they agree.
/// On an exact edge a disagreement is refused: the engine is deterministic,
/// so two digests are two artifacts. Judged at `gate` otherwise.
pub fn digests(pair: &Pair<'_>, gate: Gate) -> AxisResult<OutcomeVerdict> {
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
    match gate {
        Gate::Hard => AxisResult {
            verdict,
            judgements: vec![],
            refusals: missing.into_iter().chain(mismatches).collect(),
        },
        Gate::Evidence => AxisResult {
            verdict,
            judgements: vec![Judgement::new(
                DIGEST_RULE,
                gate,
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
    judge_controls: bool,
) -> AxisResult<OutcomeVerdict> {
    match rules {
        CrossStackOutcome::None => AxisResult {
            verdict: None,
            judgements: vec![],
            refusals: vec![],
        },
        CrossStackOutcome::RowAgreement { metric, gate } => row_agreement(pair, *metric, *gate),
        CrossStackOutcome::Law {
            statistic,
            alpha,
            gate,
        } => law(pair, *statistic, *alpha, *gate, options.law_dir.as_deref()),
        CrossStackOutcome::SeededLoss {
            seeds,
            sign_alpha,
            direction_gate,
            delta,
            margin_alpha,
            non_inferiority_gate,
            control,
        } => {
            let mut result = seeded_loss(
                pair,
                *seeds,
                *sign_alpha,
                *direction_gate,
                delta.map(|d| (d, *margin_alpha, *non_inferiority_gate)),
            );
            let control_verdict = control.as_ref().filter(|_| judge_controls).map(|rule| {
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

fn seeded_loss(
    pair: &Pair<'_>,
    seeds: usize,
    sign_alpha: f64,
    direction_gate: Gate,
    margin: Option<(f64, f64, Gate)>,
) -> AxisResult<OutcomeVerdict> {
    let mut refusals = vec![];
    let per_unit: Vec<UnitDifference> = pair
        .units
        .iter()
        .map(|unit| {
            let held_out = |rung: &RungLegs| {
                rung.primary(unit)
                    .and_then(|leg| leg.measured.held_out_example_mean)
            };
            let (lower, upper) = (held_out(pair.lower), held_out(pair.upper));
            UnitDifference {
                unit: unit.as_str().to_owned(),
                lower,
                upper,
                d: lower.zip(upper).map(|(l, u)| u - l),
                clean: !pair.unclean.contains(unit),
            }
        })
        .collect();
    refusals.extend(
        pair.units
            .iter()
            .flat_map(|unit| [pair.lower.primary(unit), pair.upper.primary(unit)])
            .flatten()
            .filter(|leg| leg.measured.held_out_example_mean.is_none())
            .map(|leg| Refusal::MeasurementMissing {
                subject: leg.name.to_string(),
                measurement: "held_out_example_mean",
            }),
    );

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
        for unit in pair.units {
            let outcomes: Vec<f64> = rung
                .repeats(unit)
                .filter_map(|leg| leg.measured.held_out_example_mean)
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

    let clean_d: Vec<f64> = per_unit
        .iter()
        .filter(|u| u.clean)
        .filter_map(|u| u.d)
        .collect();
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
        direction_gate,
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

    let margin_test = match margin {
        None => {
            refusals.push(Refusal::DeltaNotFixed {
                edge: pair.edge.to_owned(),
            });
            None
        }
        Some((delta, alpha, gate)) => {
            let result = (clean_d.len() >= 2)
                .then(|| {
                    paired_margin_test(
                        &clean_d,
                        delta,
                        MARGIN_BOOTSTRAP_ITERATIONS,
                        alpha,
                        SpeedInstrument::BOOTSTRAP_SEED,
                    )
                })
                .transpose()
                .unwrap_or_else(|e| {
                    refusals.push(Refusal::statistics(
                        format!("edge {} margin test", pair.edge),
                        e,
                    ));
                    None
                });
            let detail = |bound: &str| {
                result.map_or_else(
                    || "not computed".to_owned(),
                    |r| {
                        format!(
                            "mean d in [{:.6}, {:.6}] against {bound}",
                            r.interval.lower, r.interval.upper
                        )
                    },
                )
            };
            judgements.push(Judgement::new(
                NON_INFERIORITY_RULE,
                gate,
                result.map(|r| r.non_inferior(LOSS)),
                detail(&format!("an upper bound below +{delta}")),
            ));
            judgements.push(Judgement::new(
                "equivalent_within_delta",
                Gate::Evidence,
                result.map(|r| r.equivalent()),
                detail(&format!("±{delta}")),
            ));
            result
        }
    };

    AxisResult {
        verdict: Some(OutcomeVerdict::SeededLoss {
            clean_units: clean_d.len(),
            per_unit,
            sign_test: sign,
            critical_count,
            mean_d,
            direction,
            margin_test,
            repeat_floor: RepeatFloor { max_delta, spread },
            control: None,
        }),
        judgements,
        refusals,
    }
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
        let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        match self {
            Self::Cosine => {
                let dot: f64 = lower.iter().zip(upper).map(|(x, y)| x * y).sum();
                dot / (norm(lower) * norm(upper))
            }
            Self::RelativeError => {
                let difference: Vec<f64> = lower.iter().zip(upper).map(|(x, y)| y - x).collect();
                norm(&difference) / norm(lower)
            }
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

fn row_agreement(pair: &Pair<'_>, metric: RowMetric, gate: Gate) -> AxisResult<OutcomeVerdict> {
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
        return AxisResult {
            verdict: None,
            judgements: vec![Judgement::new(
                ROW_AGREEMENT_RULE,
                gate,
                None,
                "no vectors to compare",
            )],
            refusals,
        };
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
            gate,
            Some(rows_beyond_bound == 0),
            format!("worst row {worst:.3e} against {bound:.3e}"),
        )],
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

fn law(
    pair: &Pair<'_>,
    statistic: FitStatistic,
    alpha: f64,
    gate: Gate,
    law_dir: Option<&std::path::Path>,
) -> AxisResult<OutcomeVerdict> {
    let unmeasured = |detail: &str, refusals| AxisResult {
        verdict: None,
        judgements: vec![Judgement::new(LAW_RULE, gate, None, detail)],
        refusals,
    };
    let Some(dir) = law_dir else {
        return unmeasured("no law directory was given", vec![]);
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
        return unmeasured("the law could not be tested", refusals);
    };
    AxisResult {
        verdict: Some(OutcomeVerdict::Law {
            alpha,
            lower,
            upper,
        }),
        judgements: vec![Judgement::new(
            LAW_RULE,
            gate,
            Some(lower.p_value >= alpha && upper.p_value >= alpha),
            format!(
                "p = {:.4} (lower), {:.4} (upper) against alpha {alpha}",
                lower.p_value, upper.p_value
            ),
        )],
        refusals,
    }
}
