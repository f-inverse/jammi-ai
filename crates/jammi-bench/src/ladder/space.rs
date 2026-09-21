//! The space axis: one instrument per quantity, the same for every rung.
//!
//! Peak host memory is the kernel's high-water mark for the process; peak
//! device memory is one whole-device sampler wrapped around the leg. A
//! framework's own allocator counters never enter a comparison. Where a
//! rung's memory must not grow with its input, the quantity judged is the
//! fitted slope over a size sweep, not any single peak.

use jammi_numerics::stats::{geometric_mean, linear_fit, LinearFit};

use super::definition::{Judged, SpaceRules};
use super::leg::{Leg, RungLegs, Unit};
use super::outcome::{AxisResult, Pair};
use super::refusal::Refusal;
use super::verdict::{Judgement, SpaceVerdict};

/// A rung's peak for one unit: the largest any repeat reached.
fn peak(rung: &RungLegs, unit: &Unit, quantity: fn(&Leg) -> Option<f64>) -> Option<f64> {
    rung.repeats(unit).filter_map(quantity).reduce(f64::max)
}

/// Geometric mean over units of `upper ÷ lower`; `None` unless every unit
/// measured the quantity on both rungs.
fn ratio(pair: &Pair<'_>, quantity: fn(&Leg) -> Option<f64>) -> Option<f64> {
    let ratios: Vec<f64> = pair
        .units
        .iter()
        .map(|unit| Some(peak(pair.upper, unit, quantity)? / peak(pair.lower, unit, quantity)?))
        .collect::<Option<_>>()?;
    geometric_mean(&ratios).ok()
}

fn host_slope(pair: &Pair<'_>) -> Result<Option<LinearFit>, Refusal> {
    let (work, peaks): (Vec<f64>, Vec<f64>) = pair
        .units
        .iter()
        .filter_map(|unit| {
            Some((
                pair.upper.primary(unit)?.measured.work?,
                peak(pair.upper, unit, Leg::peak_rss_bytes)?,
            ))
        })
        .unzip();
    if work.len() < 3 {
        return Ok(None);
    }
    linear_fit(&work, &peaks)
        .map(Some)
        .map_err(|e| Refusal::statistics(format!("edge {} host-memory slope", pair.edge), e))
}

pub fn space(
    pair: &Pair<'_>,
    rules: &SpaceRules,
    flat_host_memory: Option<Judged<f64>>,
) -> AxisResult<SpaceVerdict> {
    let host_ratio = ratio(pair, Leg::peak_rss_bytes);
    let device_ratio = ratio(pair, Leg::peak_vram_bytes);
    let bounded = |rule, judged: Judged<f64>, value: Option<f64>| {
        Judgement::new(
            rule,
            judged.gate,
            value.map(|v| v <= judged.bound),
            value.map_or_else(
                || "not measured on every leg".to_owned(),
                |v| format!("{v:.4} against {}", judged.bound),
            ),
        )
    };
    let mut judgements = vec![
        bounded("host_memory_ratio", rules.host_ratio, host_ratio),
        bounded("device_memory_ratio", rules.device_ratio, device_ratio),
    ];
    let mut refusals = vec![];
    let slope = flat_host_memory.and_then(|budget| {
        let fitted = host_slope(pair).unwrap_or_else(|refusal| {
            refusals.push(refusal);
            None
        });
        judgements.push(bounded(
            "host_memory_flat_in_work",
            budget,
            fitted.map(|f| f.slope),
        ));
        fitted
    });
    AxisResult {
        verdict: Some(SpaceVerdict {
            host_ratio,
            device_ratio,
            host_slope: slope,
        }),
        judgements,
        refusals,
    }
}
