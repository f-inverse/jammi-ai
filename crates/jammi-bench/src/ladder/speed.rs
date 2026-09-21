//! The speed axis: what a layer costs, as a ratio with an interval.
//!
//! # Location and interval
//!
//! Timing noise is one-sided — an iteration is delayed by the machine, never
//! hurried — so a leg's *fastest* iteration is the robust estimate of the
//! undisturbed run, and the cost computed from minima is the one reported
//! first. It cannot carry the interval. A resample's minimum is the sample's
//! minimum with probability approaching `1 − 1/e` and is never below it, so a
//! bootstrap of minima is a spike, not a sampling distribution (see
//! [`jammi_numerics::stats::block_bootstrap`]). The interval is therefore
//! taken on the ratio of *medians*, a smooth functional the bootstrap is
//! valid for, and every rule is judged on that interval. The two estimates
//! answer different questions — how fast the code can go, and how fast it
//! typically went in this session — and a verdict carries both.
//!
//! The legs of an edge run interleaved in one session, so the median's
//! sensitivity to machine noise is shared by both sides of the ratio. The
//! same rung measured against itself gives the width of what is not shared:
//! a cost inside that band is reported as indistinguishable from 1.

use jammi_numerics::stats::{
    block_bootstrap_ci, geometric_mean, linear_fit, mann_kendall, median, minimum, Interval,
    LinearFit,
};

use super::definition::{Gate, Judged, ShapeRules, SpeedInstrument};
use super::leg::{Leg, RungLegs, Unit};
use super::outcome::{AxisResult, Pair};
use super::refusal::Refusal;
use super::verdict::{Direction, Judgement, Ratio, ShapeVerdict, SpeedVerdict, TimeToQuality};

/// How an edge's cost is bounded.
#[derive(Debug, Clone, Copy)]
pub enum SpeedBound {
    /// Paired edge: the lower bound of `lower ÷ upper` must exceed the bar.
    NonInferiority(Judged<f64>),
    /// Exact edge: the upper bound of `upper ÷ lower` must stay under the
    /// budget.
    OverheadBudget(Judged<f64>),
    /// Revision edge: the cost must lie inside the rung's own noise band.
    WithinNoiseBand(Gate),
}

/// A leg's series, or why it cannot be used.
fn series(leg: &Leg) -> Result<Option<&[f64]>, Refusal> {
    let Some(series) = leg.measured.iter_wall_s.as_deref() else {
        return Ok(None);
    };
    let name = || leg.name.to_string();
    if series.len() < SpeedInstrument::MIN_SAMPLES {
        return Err(Refusal::TooFewSamples {
            leg: name(),
            got: series.len(),
            need: SpeedInstrument::MIN_SAMPLES,
        });
    }
    if series.iter().any(|t| !(t.is_finite() && *t > 0.0)) {
        return Err(Refusal::LegUnreadable {
            leg: name(),
            reason: "iter_wall_s has an entry that is not a positive finite duration".into(),
        });
    }
    let trend = mann_kendall(series).map_err(|e| Refusal::statistics(name(), e))?;
    let level = median(series).map_err(|e| Refusal::statistics(name(), e))?;
    let relative_drift = trend.sen_slope * (series.len() - 1) as f64 / level;
    if trend.p_value < SpeedInstrument::TREND_ALPHA
        && relative_drift.abs() > SpeedInstrument::MAX_RELATIVE_DRIFT
    {
        return Err(Refusal::NonStationary {
            leg: name(),
            p_value: trend.p_value,
            relative_drift: relative_drift * 100.0,
        });
    }
    Ok(Some(series))
}

/// Every repeat's series for each unit of one rung: `None` when any repeat
/// carries no series at all.
fn rung_series<'a>(
    rung: &'a RungLegs,
    units: &[Unit],
    refusals: &mut Vec<Refusal>,
) -> Option<Vec<Vec<&'a [f64]>>> {
    let per_unit: Vec<Option<Vec<&[f64]>>> = units
        .iter()
        .map(|unit| {
            rung.repeats(unit)
                .map(|leg| {
                    series(leg).unwrap_or_else(|refusal| {
                        refusals.push(refusal);
                        None
                    })
                })
                .collect()
        })
        .collect();
    per_unit.into_iter().collect()
}

/// `location(numerator) ÷ location(denominator)` per unit, pooled over a
/// unit's repeats, then the geometric mean over units.
fn pooled_ratio(
    numerator: &[Vec<&[f64]>],
    denominator: &[Vec<&[f64]>],
    location: fn(&[f64]) -> f64,
) -> f64 {
    let pooled = |repeats: &Vec<&[f64]>| location(&repeats.concat());
    let ratios: Vec<f64> = numerator
        .iter()
        .zip(denominator)
        .map(|(n, d)| pooled(n) / pooled(d))
        .collect();
    geometric_mean(&ratios).unwrap_or(f64::NAN)
}

fn median_of(values: &[f64]) -> f64 {
    median(values).unwrap_or(f64::NAN)
}

fn minimum_of(values: &[f64]) -> f64 {
    minimum(values).unwrap_or(f64::NAN)
}

/// The cost `numerator ÷ denominator` with its interval. The bootstrap
/// resamples every leg's series on its own and recomputes the whole
/// statistic — per-unit pooled medians, their ratio, the geometric mean — so
/// the interval belongs to the number it is printed beside.
fn ratio(
    numerator: &[Vec<&[f64]>],
    denominator: &[Vec<&[f64]>],
    context: &str,
) -> Result<Ratio, Refusal> {
    let flat: Vec<&[f64]> = numerator
        .iter()
        .chain(denominator)
        .flatten()
        .copied()
        .collect();
    let shape: Vec<usize> = numerator.iter().chain(denominator).map(Vec::len).collect();
    let units = numerator.len();
    let interval = block_bootstrap_ci(
        &flat,
        |resamples| {
            let mut rest = resamples;
            let grouped: Vec<Vec<&[f64]>> = shape
                .iter()
                .map(|n| {
                    let (head, tail) = rest.split_at(*n);
                    rest = tail;
                    head.iter().map(Vec::as_slice).collect()
                })
                .collect();
            pooled_ratio(&grouped[..units], &grouped[units..], median_of)
        },
        SpeedInstrument::BOOTSTRAP_ITERATIONS,
        SpeedInstrument::INTERVAL_ALPHA,
        SpeedInstrument::BOOTSTRAP_SEED,
    )
    .map_err(|e| Refusal::statistics(context, e))?;
    Ok(Ratio {
        of_minima: pooled_ratio(numerator, denominator, minimum_of),
        of_medians: pooled_ratio(numerator, denominator, median_of),
        interval,
    })
}

/// The half-width, as a ratio `>= 1`, of a rung measured against itself:
/// first repeat against the rest, per unit.
fn noise_band(rung: &[Vec<&[f64]>], context: &str) -> Result<Option<f64>, Refusal> {
    if rung.iter().any(|repeats| repeats.len() < 2) {
        return Ok(None);
    }
    let first: Vec<Vec<&[f64]>> = rung.iter().map(|r| vec![r[0]]).collect();
    let rest: Vec<Vec<&[f64]>> = rung.iter().map(|r| r[1..].to_vec()).collect();
    Ok(Some(half_width(ratio(&first, &rest, context)?.interval)))
}

/// A revision edge: the cost must lie inside the wider of the repeat noise
/// band and the A/A band — the interval of a second build of the lower side
/// against the lower side, when `rebuilt` legs were measured.
pub fn revision(
    pair: &Pair<'_>,
    gate: Gate,
    rebuilt: Option<&RungLegs>,
) -> AxisResult<SpeedVerdict> {
    let mut result = speed(pair, SpeedBound::WithinNoiseBand(gate), None, None);
    let Some(rebuilt) = rebuilt else {
        return result;
    };
    let null = Pair {
        edge: &format!("{} (A/A null)", pair.edge),
        lower: pair.lower,
        upper: rebuilt,
        units: pair.units,
        unclean: pair.unclean,
    };
    let aa = match measure_cost(&null) {
        Err(refusals) => {
            result.refusals.extend(refusals);
            return result;
        }
        Ok(None) => return result,
        Ok(Some((cost, _))) => cost,
    };
    let Some(verdict) = result.verdict.as_mut() else {
        return result;
    };
    let band = verdict
        .noise_band
        .map_or(half_width(aa.interval), |b| b.max(half_width(aa.interval)));
    verdict.noise_band = Some(band);
    verdict.aa_null = Some(aa);
    let inside = verdict.cost.of_medians.ln().abs() <= band.ln();
    verdict.indistinguishable_from_one = Some(inside);
    let cost = verdict.cost.of_medians;
    result.judgements = result
        .judgements
        .into_iter()
        .map(|j| {
            if j.rule != "speed_within_noise_band" {
                return j;
            }
            let direction = match (inside, cost < 1.0) {
                (true, _) => Direction::None,
                (false, true) => Direction::Improvement,
                (false, false) => Direction::Degradation,
            };
            Judgement::directed(
                j.rule,
                j.gate,
                Some(inside),
                direction,
                format!(
                    "upper/lower time {cost:.4} against the wider of the repeat band and the A/A band, x{band:.4}"
                ),
            )
        })
        .collect();
    result
}

/// The half-width, as a ratio `>= 1`, that covers an interval: the further
/// of its two bounds from 1.
fn half_width(interval: Interval) -> f64 {
    interval
        .lower
        .ln()
        .abs()
        .max(interval.upper.ln().abs())
        .exp()
}

/// The cost of `upper` over `lower`, measured from their series.
pub fn measure_cost(pair: &Pair<'_>) -> Result<Option<(Ratio, Option<f64>)>, Vec<Refusal>> {
    let mut refusals = vec![];
    let lower = rung_series(pair.lower, pair.units, &mut refusals);
    let upper = rung_series(pair.upper, pair.units, &mut refusals);
    if !refusals.is_empty() {
        return Err(refusals);
    }
    let (Some(lower), Some(upper)) = (lower, upper) else {
        return Ok(None);
    };
    if pair.units.is_empty() {
        return Ok(None);
    }
    let measured = ratio(&upper, &lower, pair.edge).and_then(|cost| {
        let bands = [
            noise_band(&lower, pair.edge)?,
            noise_band(&upper, pair.edge)?,
        ];
        Ok((cost, bands.into_iter().flatten().reduce(f64::max)))
    });
    measured.map(Some).map_err(|refusal| vec![refusal])
}

pub fn speed(
    pair: &Pair<'_>,
    bound: SpeedBound,
    shape_rules: Option<&ShapeRules>,
    time_to_quality: Option<(Judged<f64>, f64)>,
) -> AxisResult<SpeedVerdict> {
    let (rule, gate) = match bound {
        SpeedBound::NonInferiority(j) => ("speed_non_inferiority", j.gate),
        SpeedBound::OverheadBudget(j) => ("overhead_budget", j.gate),
        SpeedBound::WithinNoiseBand(gate) => ("speed_within_noise_band", gate),
    };
    let (cost, band) = match measure_cost(pair) {
        Err(refusals) => return AxisResult::refused(refusals),
        Ok(None) => {
            return AxisResult {
                verdict: None,
                judgements: vec![Judgement::new(
                    rule,
                    gate,
                    None,
                    "no leg carries iter_wall_s",
                )],
                refusals: vec![],
            }
        }
        Ok(Some(measured)) => measured,
    };
    let mut judgements = vec![match bound {
        SpeedBound::NonInferiority(j) => Judgement::new(
            rule,
            gate,
            Some(1.0 / cost.interval.upper > j.bound),
            format!(
                "lower/upper time at least {:.4}, bar {}",
                1.0 / cost.interval.upper,
                j.bound
            ),
        ),
        SpeedBound::OverheadBudget(j) => Judgement::new(
            rule,
            gate,
            Some(cost.interval.upper < j.bound),
            format!(
                "upper/lower time at most {:.4}, budget {}",
                cost.interval.upper, j.bound
            ),
        ),
        SpeedBound::WithinNoiseBand(gate) => {
            let inside = band.map(|b| cost.of_medians.ln().abs() <= b.ln());
            let direction = match inside {
                Some(false) if cost.of_medians < 1.0 => Direction::Improvement,
                Some(false) => Direction::Degradation,
                _ => Direction::None,
            };
            Judgement::directed(
                rule,
                gate,
                inside,
                direction,
                match band {
                    Some(b) => format!(
                        "upper/lower time {:.4} against the rung's own noise band x{b:.4}",
                        cost.of_medians
                    ),
                    None => "no rung carries two repeats, so it has no noise band".to_owned(),
                },
            )
        }
    }];
    let mut refusals = vec![];

    let shape_verdict = shape_rules.and_then(|rules| match shape(pair, rules) {
        Ok(Some((verdict, judged))) => {
            judgements.extend(judged);
            Some(verdict)
        }
        Ok(None) => {
            judgements.push(Judgement::new(
                "shape",
                rules.gate,
                None,
                "fewer than 3 sizes carry `work`",
            ));
            None
        }
        Err(refusal) => {
            refusals.push(refusal);
            None
        }
    });

    let ttq = time_to_quality.map(|(bar, slack)| {
        let verdict = time_to_target(pair, slack);
        judgements.push(Judgement::new(
            "time_to_quality",
            bar.gate,
            verdict
                .ratio
                .map(|r| r > bar.bound && verdict.unreached.is_empty()),
            format!(
                "lower/upper seconds to target {:?}, bar {}",
                verdict.ratio, bar.bound
            ),
        ));
        verdict
    });

    AxisResult {
        verdict: Some(SpeedVerdict {
            cost,
            noise_band: band,
            indistinguishable_from_one: band.map(|b| cost.of_medians.ln().abs() <= b.ln()),
            aa_null: None,
            shape: shape_verdict,
            time_to_quality: ttq,
        }),
        judgements,
        refusals,
    }
}

/// `time = fixed + per_work · work` for one rung: each unit's fastest
/// iteration against the work it did.
fn fit(
    rung: &RungLegs,
    units: &[Unit],
    rung_name: &str,
    rules: &ShapeRules,
) -> Result<Option<LinearFit>, Refusal> {
    let points: Vec<(f64, f64)> = units
        .iter()
        .filter_map(|unit| {
            let work = rung.primary(unit)?.measured.work?;
            let times: Vec<f64> = rung
                .repeats(unit)
                .filter_map(|leg| leg.measured.iter_wall_s.as_deref())
                .flatten()
                .copied()
                .collect();
            Some((work, minimum(&times).ok()?))
        })
        .collect();
    if points.len() < 3 {
        return Ok(None);
    }
    let (xs, ys): (Vec<f64>, Vec<f64>) = points.into_iter().unzip();
    let poor = |relative_residual: f64, reason: String| Refusal::ShapeFitPoor {
        rung: rung_name.to_owned(),
        relative_residual,
        limit: rules.max_relative_residual,
        reason,
    };
    let line = linear_fit(&xs, &ys).map_err(|e| poor(f64::NAN, e.to_string()))?;
    let relative_residual = line.residual_rms / (ys.iter().sum::<f64>() / ys.len() as f64);
    if relative_residual > rules.max_relative_residual {
        return Err(poor(
            relative_residual,
            "the residual is too large for two coefficients to describe".into(),
        ));
    }
    if line.slope <= 0.0 {
        return Err(poor(
            relative_residual,
            format!("per-work cost {} is not positive", line.slope),
        ));
    }
    Ok(Some(line))
}

fn shape(
    pair: &Pair<'_>,
    rules: &ShapeRules,
) -> Result<Option<(ShapeVerdict, Vec<Judgement>)>, Refusal> {
    let name = |rung: &RungLegs, side: &str| {
        rung.all()
            .next()
            .map_or_else(|| side.to_owned(), |leg| leg.name.rung.clone())
    };
    let lower = fit(pair.lower, pair.units, &name(pair.lower, "lower"), rules)?;
    let upper = fit(pair.upper, pair.units, &name(pair.upper, "upper"), rules)?;
    let (Some(lower), Some(upper)) = (lower, upper) else {
        return Ok(None);
    };
    let verdict = ShapeVerdict {
        fixed_work_equivalent: (upper.intercept - lower.intercept) / lower.slope,
        per_work_ratio: upper.slope / lower.slope,
        lower,
        upper,
    };
    let judgements = vec![
        Judgement::new(
            "fixed_cost",
            rules.gate,
            Some(verdict.fixed_work_equivalent <= rules.fixed_work_equivalent),
            format!(
                "the layer's fixed cost is worth {:.1} units of work, budget {}",
                verdict.fixed_work_equivalent, rules.fixed_work_equivalent
            ),
        ),
        Judgement::new(
            "per_work_cost",
            rules.gate,
            Some(verdict.per_work_ratio <= rules.per_work_ratio),
            format!(
                "per-work cost x{:.4}, budget x{}",
                verdict.per_work_ratio, rules.per_work_ratio
            ),
        ),
    ];
    Ok(Some((verdict, judgements)))
}

/// Seconds of training until the held-out loss first reaches `target`:
/// `None` when the leg does not time its evaluations, `Some(None)` when it
/// does and never gets there.
fn seconds_to(leg: &Leg, target: f64) -> Option<Option<f64>> {
    let timed: Vec<(f64, f64)> = leg
        .measured
        .trajectory
        .iter()
        .map(|point| Some((point.held_out_mean, point.train_wall_s?)))
        .collect::<Option<_>>()?;
    (!timed.is_empty()).then(|| {
        timed
            .iter()
            .find(|(loss, _)| *loss <= target)
            .map(|(_, seconds)| *seconds)
    })
}

/// Training wall time until the held-out loss first comes within `slack` of
/// the lower rung's own final loss, per unit. A stack that steps faster and
/// converges slower loses here.
fn time_to_target(pair: &Pair<'_>, slack: f64) -> TimeToQuality {
    let mut unreached = vec![];
    let ratios: Vec<f64> = pair
        .units
        .iter()
        .filter_map(|unit| {
            let (lower, upper) = (pair.lower.primary(unit)?, pair.upper.primary(unit)?);
            let target = lower.measured.held_out_example_mean? + slack;
            let reference = seconds_to(lower, target)??;
            match seconds_to(upper, target)? {
                Some(seconds) => Some(reference / seconds),
                None => {
                    unreached.push(unit.as_str().to_owned());
                    None
                }
            }
        })
        .collect();
    TimeToQuality {
        ratio: geometric_mean(&ratios).ok(),
        unreached,
    }
}
