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
    block_bootstrap_ci, geometric_mean, linear_fit, mann_kendall, median, minimum, mser_truncation,
    Interval, LinearFit,
};

use super::definition::{rule, Rule, RuleForce, ShapeRules, SpeedInstrument};
use super::leg::{Leg, RungLegs, Unit};
use super::outcome::{AxisResult, Pair};
use super::refusal::Refusal;
use super::verdict::{
    Bound, Direction, Judgement, Ratio, Settled, ShapeVerdict, SpeedVerdict, TimeToQuality,
};

/// How an edge's cost is bounded.
#[derive(Debug, Clone, Copy)]
pub enum SpeedBound<'a> {
    /// Paired edge: the lower bound of `lower ÷ upper` must exceed the bar.
    NonInferiority(&'a Rule),
    /// Exact edge: the upper bound of `upper ÷ lower` must stay under the
    /// budget.
    OverheadBudget(&'a Rule),
    /// Revision edge: the cost must lie inside the rung's own noise band.
    WithinNoiseBand(RuleForce),
}

impl SpeedBound<'_> {
    fn rule(self) -> (&'static str, RuleForce) {
        match self {
            Self::NonInferiority(r) => (rule::SPEED_NON_INFERIORITY, r.force),
            Self::OverheadBudget(r) => (rule::OVERHEAD_BUDGET, r.force),
            Self::WithinNoiseBand(force) => ("speed_within_noise_band", force),
        }
    }
}

/// The time-to-quality bar and the slack the target is set with: the
/// margin the outcome axis derived, when it derived one.
#[derive(Debug, Clone, Copy)]
pub struct TimeToQualityRule<'a> {
    pub bar: &'a Rule,
    pub slack: Option<f64>,
}

/// A leg's settled series and where it settled; `None` when it carries no
/// series; or why it cannot be used.
type LegSeries<'a> = Result<Option<(&'a [f64], Settled)>, Refusal>;

/// A leg's run, settled: its initial transient cut by MSER, and what is
/// left long enough and steady — or why it cannot be used. The cut chooses a
/// point and tests nothing; the trend test after it is the one decision, so
/// a run that drifts throughout, or settles and then slows, is refused by it.
fn series(leg: &Leg) -> LegSeries<'_> {
    let Some(run) = leg.measured.iter_wall_s.as_deref() else {
        return Ok(None);
    };
    let name = || leg.name.to_string();
    if run.iter().any(|t| !(t.is_finite() && *t > 0.0)) {
        return Err(Refusal::LegUnreadable {
            leg: name(),
            reason: "iter_wall_s has an entry that is not a positive finite duration".into(),
        });
    }
    let cut = mser_truncation(run, SpeedInstrument::TRUNCATION_BATCH)
        .map_err(|e| Refusal::statistics(name(), e))?;
    let series = &run[cut.at..];
    if series.len() < SpeedInstrument::MIN_SAMPLES {
        return Err(Refusal::TooFewSamples {
            leg: name(),
            got: series.len(),
            need: SpeedInstrument::MIN_SAMPLES,
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
    Ok(Some((
        series,
        Settled {
            leg: name(),
            transient: cut.at,
            at_limit: cut.at_limit,
            iterations: run.len(),
        },
    )))
}

/// One rung's settled series, read unit by unit.
struct RungSeries<'a> {
    /// Every repeat's settled series for each unit: `None` when any repeat
    /// carries no usable series.
    per_unit: Option<Vec<Vec<&'a [f64]>>>,
    settled: Vec<Settled>,
    refusals: Vec<Refusal>,
}

fn rung_series<'a>(rung: &'a RungLegs, units: &[Unit]) -> RungSeries<'a> {
    let read: Vec<Vec<LegSeries<'a>>> = units
        .iter()
        .map(|unit| rung.repeats(unit).map(series).collect())
        .collect();
    RungSeries {
        refusals: read
            .iter()
            .flatten()
            .filter_map(|leg| leg.as_ref().err().cloned())
            .collect(),
        settled: read
            .iter()
            .flatten()
            .filter_map(|leg| Some(leg.as_ref().ok()?.as_ref()?.1.clone()))
            .collect(),
        per_unit: read
            .into_iter()
            .map(|repeats| {
                repeats
                    .into_iter()
                    .map(|leg| Some(leg.ok()??.0))
                    .collect::<Option<Vec<_>>>()
            })
            .collect(),
    }
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
    force: RuleForce,
    rebuilt: Option<&RungLegs>,
) -> AxisResult<SpeedVerdict> {
    let mut result = speed(pair, SpeedBound::WithinNoiseBand(force), None, None);
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
        Ok(Some(measured)) => measured.ratio,
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
                j.force,
                Bound::Derived { value: band },
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

/// A pair's cost, and the settled series it was measured from.
pub struct Cost<'a> {
    pub ratio: Ratio,
    /// The wider of the two rungs' own noise bands.
    pub band: Option<f64>,
    lower: Vec<Vec<&'a [f64]>>,
    upper: Vec<Vec<&'a [f64]>>,
    settled: Vec<Settled>,
}

/// The cost of `upper` over `lower`, measured from their settled series.
pub fn measure_cost<'a>(pair: &Pair<'a>) -> Result<Option<Cost<'a>>, Vec<Refusal>> {
    let (lower, upper) = (
        rung_series(pair.lower, pair.units),
        rung_series(pair.upper, pair.units),
    );
    let refusals: Vec<Refusal> = lower.refusals.into_iter().chain(upper.refusals).collect();
    if !refusals.is_empty() {
        return Err(refusals);
    }
    let settled = lower.settled.into_iter().chain(upper.settled).collect();
    let (Some(lower), Some(upper)) = (lower.per_unit, upper.per_unit) else {
        return Ok(None);
    };
    if pair.units.is_empty() {
        return Ok(None);
    }
    let measured = ratio(&upper, &lower, pair.edge).and_then(|ratio| {
        let bands = [
            noise_band(&lower, pair.edge)?,
            noise_band(&upper, pair.edge)?,
        ];
        Ok(Cost {
            ratio,
            band: bands.into_iter().flatten().reduce(f64::max),
            lower,
            upper,
            settled,
        })
    });
    measured.map(Some).map_err(|refusal| vec![refusal])
}

/// The cost against its bound.
fn cost_judgement(bound: SpeedBound<'_>, cost: &Ratio, band: Option<f64>) -> Judgement {
    let (name, force) = bound.rule();
    match bound {
        SpeedBound::NonInferiority(spec) => Judgement::budgeted(
            name,
            spec,
            Some(1.0 / cost.interval.upper),
            |at_least, bar| at_least > bar,
            format!(
                "lower/upper time at least {:.4}, bar {}",
                1.0 / cost.interval.upper,
                Bound::of(spec)
            ),
        ),
        SpeedBound::OverheadBudget(spec) => Judgement::budgeted(
            name,
            spec,
            Some(cost.interval.upper),
            |at_most, budget| at_most < budget,
            format!(
                "upper/lower time at most {:.4}, budget {}",
                cost.interval.upper,
                Bound::of(spec)
            ),
        ),
        SpeedBound::WithinNoiseBand(_) => {
            let inside = band.map(|b| cost.of_medians.ln().abs() <= b.ln());
            let direction = match inside {
                Some(false) if cost.of_medians < 1.0 => Direction::Improvement,
                Some(false) => Direction::Degradation,
                _ => Direction::None,
            };
            Judgement::directed(
                name,
                force,
                band.map_or(Bound::None, |value| Bound::Derived { value }),
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
    }
}

pub fn speed(
    pair: &Pair<'_>,
    bound: SpeedBound<'_>,
    shape_rules: Option<&ShapeRules>,
    time_to_quality: Option<TimeToQualityRule<'_>>,
) -> AxisResult<SpeedVerdict> {
    let (name, force) = bound.rule();
    let measured = match measure_cost(pair) {
        Err(refusals) => return AxisResult::refused(refusals),
        Ok(None) => {
            return AxisResult {
                verdict: None,
                judgements: vec![Judgement::new(
                    name,
                    force,
                    Bound::None,
                    None,
                    "no leg carries iter_wall_s",
                )],
                refusals: vec![],
            }
        }
        Ok(Some(measured)) => measured,
    };
    let (cost, band) = (measured.ratio, measured.band);
    let mut judgements = vec![cost_judgement(bound, &cost, band)];
    let mut refusals = vec![];

    let shape_verdict = shape_rules.and_then(|rules| match shape(pair, &measured, rules) {
        Ok(Some((verdict, judged))) => {
            judgements.extend(judged);
            Some(verdict)
        }
        Ok(None) => {
            judgements.extend(
                [
                    (rule::FIXED_COST, &rules.fixed_work_equivalent),
                    (rule::PER_WORK_COST, &rules.per_work_ratio),
                ]
                .map(|(name, spec)| {
                    Judgement::budgeted(
                        name,
                        spec,
                        None,
                        |_, _| false,
                        "fewer than 3 sizes carry `work`",
                    )
                }),
            );
            None
        }
        Err(refusal) => {
            refusals.push(refusal);
            None
        }
    });

    let ttq = time_to_quality.map(|rule| match rule.slack {
        None => {
            judgements.push(Judgement::budgeted(
                rule::TIME_TO_QUALITY,
                rule.bar,
                None,
                |_, _| false,
                "no margin was derived on the outcome axis to set the target with",
            ));
            TimeToQuality {
                ratio: None,
                unreached: vec![],
            }
        }
        Some(slack) => {
            let verdict = time_to_target(pair, slack);
            judgements.push(Judgement::budgeted(
                rule::TIME_TO_QUALITY,
                rule.bar,
                verdict.ratio,
                |ratio, bar| ratio > bar && verdict.unreached.is_empty(),
                format!(
                    "lower/upper seconds to target {:?}, bar {}",
                    verdict.ratio,
                    Bound::of(rule.bar)
                ),
            ));
            verdict
        }
    });

    AxisResult {
        verdict: Some(SpeedVerdict {
            cost,
            settled: measured.settled,
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
/// settled iteration against the work it did.
fn fit(
    rung: &RungLegs,
    units: &[Unit],
    series: &[Vec<&[f64]>],
    rung_name: &str,
    rules: &ShapeRules,
) -> Result<Option<LinearFit>, Refusal> {
    let points: Vec<(f64, f64)> = units
        .iter()
        .zip(series)
        .filter_map(|(unit, repeats)| {
            let work = rung.primary(unit)?.measured.work?;
            Some((work, minimum(&repeats.concat()).ok()?))
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
    measured: &Cost<'_>,
    rules: &ShapeRules,
) -> Result<Option<(ShapeVerdict, Vec<Judgement>)>, Refusal> {
    let name = |rung: &RungLegs, side: &str| {
        rung.all()
            .next()
            .map_or_else(|| side.to_owned(), |leg| leg.name.rung.clone())
    };
    let lower = fit(
        pair.lower,
        pair.units,
        &measured.lower,
        &name(pair.lower, "lower"),
        rules,
    )?;
    let upper = fit(
        pair.upper,
        pair.units,
        &measured.upper,
        &name(pair.upper, "upper"),
        rules,
    )?;
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
        Judgement::budgeted(
            rule::FIXED_COST,
            &rules.fixed_work_equivalent,
            Some(verdict.fixed_work_equivalent),
            |v, budget| v <= budget,
            format!(
                "the layer's fixed cost is worth {:.1} units of work, budget {}",
                verdict.fixed_work_equivalent,
                Bound::of(&rules.fixed_work_equivalent)
            ),
        ),
        Judgement::budgeted(
            rule::PER_WORK_COST,
            &rules.per_work_ratio,
            Some(verdict.per_work_ratio),
            |v, budget| v <= budget,
            format!(
                "per-work cost x{:.4}, budget {}",
                verdict.per_work_ratio,
                Bound::of(&rules.per_work_ratio)
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
        .map(|point| Some((point.held_out_mean, point.run_wall_s_cumulative?)))
        .collect::<Option<_>>()?;
    (!timed.is_empty()).then(|| {
        timed
            .iter()
            .find(|(loss, _)| *loss <= target)
            .map(|(_, seconds)| *seconds)
    })
}

/// Training wall time until the held-out loss first comes within `slack` of
/// the lower rung's own lowest loss, per unit. A stack that steps faster and
/// converges slower loses here.
fn time_to_target(pair: &Pair<'_>, slack: f64) -> TimeToQuality {
    let mut unreached = vec![];
    let ratios: Vec<f64> = pair
        .units
        .iter()
        .filter_map(|unit| {
            let (lower, upper) = (pair.lower.primary(unit)?, pair.upper.primary(unit)?);
            let target = lower.held_out_minimum()?.1 + slack;
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
