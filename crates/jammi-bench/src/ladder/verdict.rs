//! What the comparator says: one verdict per edge on three axes, folded
//! into one status for the run.

use serde::Serialize;

use jammi_numerics::stats::{
    EquivalenceResult, GoodnessOfFit, Interval, LinearFit, SignTestResult,
};

use super::definition::{Gate, RowMetric, Workload};
use super::mutant::DoseLadder;
use super::refusal::{Refusal, ReportedRefusal};

/// The run's bottom line. Ordered by severity: folding verdicts takes the
/// worst.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Status {
    /// Every hard rule held.
    Green,
    /// A hard rule was broken in the *favourable* direction — the upper rung
    /// learned significantly better. An anomaly is investigated, not
    /// celebrated.
    RedForInvestigation,
    /// A hard rule failed.
    Red,
    /// The measurement cannot be trusted; there is no verdict to give.
    Invalid,
}

/// The rule a detected directional difference fails.
pub const DIRECTION_RULE: &str = "no_directional_difference";

/// One rule, applied.
#[derive(Debug, Clone, Serialize)]
pub struct Judgement {
    pub rule: &'static str,
    pub gate: Gate,
    /// `None`: the quantity was not measured.
    pub passed: Option<bool>,
    pub detail: String,
}

impl Judgement {
    pub fn new(
        rule: &'static str,
        gate: Gate,
        passed: Option<bool>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            rule,
            gate,
            passed,
            detail: detail.into(),
        }
    }

    fn fails_hard(&self) -> bool {
        self.gate == Gate::Hard && self.passed == Some(false)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    None,
    /// The upper rung's loss is significantly higher.
    Degradation,
    /// The upper rung's loss is significantly lower.
    Improvement,
}

#[derive(Debug, Clone, Serialize)]
pub struct UnitDifference {
    pub unit: String,
    pub lower: Option<f64>,
    pub upper: Option<f64>,
    /// `upper − lower`.
    pub d: Option<f64>,
    /// Whether this unit's legs cleared every premise and so count.
    pub clean: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct RepeatFloor {
    /// Largest `|r1 − r_k|` of the outcome over any `(rung, unit)`.
    pub max_delta: f64,
    /// Population standard deviation of the paired differences.
    pub spread: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ControlVerdict {
    pub units: Vec<String>,
    /// The operator declared the control absent rather than letting it go
    /// missing.
    pub waived: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OutcomeVerdict {
    SeededLoss {
        per_unit: Vec<UnitDifference>,
        clean_units: usize,
        sign_test: Option<SignTestResult>,
        /// The concordance count the sign test needs at this edge's level
        /// and unit count.
        critical_count: Option<usize>,
        mean_d: Option<f64>,
        direction: Direction,
        equivalence: Option<EquivalenceResult>,
        repeat_floor: RepeatFloor,
        control: Option<ControlVerdict>,
    },
    RowAgreement {
        metric: RowMetric,
        /// The least cosine, or the greatest relative error, a row may show.
        bound: f64,
        rows: usize,
        /// The row furthest from agreement.
        worst: f64,
        rows_beyond_bound: usize,
    },
    Law {
        alpha: f64,
        lower: GoodnessOfFit,
        upper: GoodnessOfFit,
    },
    Digest {
        units_equal: usize,
        units: usize,
    },
}

/// A ratio with both location estimates and the interval it is judged on.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Ratio {
    /// From each leg's fastest iteration: the undisturbed-run estimate.
    pub of_minima: f64,
    /// From each leg's median iteration: the estimate the interval belongs
    /// to.
    pub of_medians: f64,
    pub interval: Interval,
}

#[derive(Debug, Clone, Serialize)]
pub struct ShapeVerdict {
    pub lower: LinearFit,
    pub upper: LinearFit,
    /// `(upper.fixed − lower.fixed) ÷ lower.per_work`.
    pub fixed_work_equivalent: f64,
    /// `upper.per_work ÷ lower.per_work`.
    pub per_work_ratio: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TimeToQuality {
    /// Geometric mean over units of `lower ÷ upper` seconds to target.
    pub ratio: Option<f64>,
    /// Units whose upper rung never reached the target.
    pub unreached: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpeedVerdict {
    /// `upper ÷ lower` time: the layer's cost. Geometric mean over units.
    pub cost: Ratio,
    /// The half-width, as a ratio, of the same rung measured against itself.
    pub noise_band: Option<f64>,
    /// The cost lies inside the noise band, whatever its point value.
    pub indistinguishable_from_one: Option<bool>,
    pub shape: Option<ShapeVerdict>,
    pub time_to_quality: Option<TimeToQuality>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpaceVerdict {
    /// `upper ÷ lower` peak host memory; geometric mean over units.
    pub host_ratio: Option<f64>,
    /// `upper ÷ lower` peak device memory; geometric mean over units.
    pub device_ratio: Option<f64>,
    /// The upper rung's fitted host bytes per unit of work over the size
    /// sweep.
    pub host_slope: Option<LinearFit>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EdgeVerdict {
    pub edge: String,
    /// The one layer the upper rung adds: what this edge's cost is the cost
    /// of.
    pub layer: String,
    pub units: Vec<String>,
    pub outcome: Option<OutcomeVerdict>,
    pub speed: Option<SpeedVerdict>,
    pub space: Option<SpaceVerdict>,
    pub judgements: Vec<Judgement>,
    pub refusals: Vec<ReportedRefusal>,
    pub status: Status,
}

impl EdgeVerdict {
    /// Fold an edge's findings into its status. A refusal outranks every
    /// rule: a number that cannot be trusted has not failed or passed.
    pub fn conclude(
        edge: String,
        layer: &str,
        units: Vec<String>,
        axes: (
            Option<OutcomeVerdict>,
            Option<SpeedVerdict>,
            Option<SpaceVerdict>,
        ),
        judgements: Vec<Judgement>,
        mut refusals: Vec<Refusal>,
    ) -> Self {
        refusals.extend(
            judgements
                .iter()
                .filter(|j| j.gate == Gate::Hard && j.passed.is_none())
                .map(|j| Refusal::MeasurementMissing {
                    subject: edge.clone(),
                    measurement: j.rule,
                }),
        );
        let direction = match &axes.0 {
            Some(OutcomeVerdict::SeededLoss { direction, .. }) => *direction,
            _ => Direction::None,
        };
        let failed: Vec<&Judgement> = judgements.iter().filter(|j| j.fails_hard()).collect();
        let only_direction = failed.iter().all(|j| j.rule == DIRECTION_RULE);
        let status = if !refusals.is_empty() {
            Status::Invalid
        } else if failed.is_empty() {
            Status::Green
        } else if only_direction && direction == Direction::Improvement {
            Status::RedForInvestigation
        } else {
            Status::Red
        };
        Self {
            edge,
            layer: layer.to_owned(),
            units,
            outcome: axes.0,
            speed: axes.1,
            space: axes.2,
            judgements,
            refusals: refusals.into_iter().map(Into::into).collect(),
            status,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Telescoping {
    /// Product of the edge costs, bounds multiplied: wider than the joint
    /// interval, so a contradiction it reports is never an artifact of the
    /// bound.
    pub product: Interval,
    pub direct: Ratio,
}

#[derive(Debug, Clone, Serialize)]
pub struct LadderVerdict {
    pub workload: Workload,
    pub from: String,
    pub to: String,
    pub edges: Vec<EdgeVerdict>,
    pub telescoping: Option<Telescoping>,
    pub dose_ladder: Option<DoseLadder>,
    /// Refusals that belong to no single edge.
    pub refusals: Vec<ReportedRefusal>,
    /// Everything that made `status` other than `GREEN`, by name.
    pub causes: Vec<String>,
    pub status: Status,
}

impl LadderVerdict {
    pub fn exit_code(&self) -> std::process::ExitCode {
        match self.status {
            Status::Green => std::process::ExitCode::SUCCESS,
            _ => std::process::ExitCode::FAILURE,
        }
    }

    /// The verdict as a table a person reads.
    pub fn table(&self) -> String {
        let mut lines = vec![format!(
            "# {} ladder, {} -> {}",
            serde_plain(&self.workload),
            self.from,
            self.to
        )];
        for edge in &self.edges {
            lines.push(String::new());
            lines.extend(edge.table());
        }
        if let Some(t) = &self.telescoping {
            lines.push(String::new());
            lines.push(format!(
                "telescoping: product [{:.4}, {:.4}] vs direct {:.4} [{:.4}, {:.4}]",
                t.product.lower,
                t.product.upper,
                t.direct.of_medians,
                t.direct.interval.lower,
                t.direct.interval.upper
            ));
        }
        if let Some(doses) = &self.dose_ladder {
            lines.push(String::new());
            lines.extend(doses.table());
        }
        if !self.refusals.is_empty() {
            lines.push(String::new());
            lines.extend(
                self.refusals
                    .iter()
                    .map(|r| format!("REFUSED: {}", r.message)),
            );
        }
        lines.push(String::new());
        if !self.causes.is_empty() {
            lines.push(format!("causes: {}", self.causes.join("; ")));
        }
        lines.push(format!("status: {}", serde_plain(&self.status)));
        lines.join("\n")
    }
}

impl EdgeVerdict {
    pub fn table(&self) -> Vec<String> {
        let mut lines = vec![format!(
            "## {}  [{}]  — {}",
            self.edge,
            serde_plain(&self.status),
            self.layer
        )];
        match &self.outcome {
            Some(OutcomeVerdict::SeededLoss {
                per_unit,
                clean_units,
                sign_test,
                critical_count,
                mean_d,
                direction,
                equivalence,
                repeat_floor,
                control,
            }) => {
                lines.push(format!("{:<10}{:<14}{:<14}{:<14}{}", "unit", "lower", "upper", "d", "premise"));
                let cell = |v: Option<f64>| v.map_or_else(|| "n/a".to_owned(), |v| format!("{v:.6}"));
                lines.extend(per_unit.iter().map(|u| {
                    format!(
                        "{:<10}{:<14}{:<14}{:<14}{}",
                        u.unit,
                        cell(u.lower),
                        cell(u.upper),
                        cell(u.d),
                        if u.clean { "clean" } else { "VIOLATED" }
                    )
                }));
                match sign_test {
                    Some(s) => lines.push(format!(
                        "sign_test: n={} n_pos={} n_neg={} ties={} p_value={} critical_count={}",
                        s.n,
                        s.n_pos,
                        s.n_neg,
                        s.ties,
                        s.p_value,
                        critical_count.map_or_else(|| "none".to_owned(), |c| c.to_string())
                    )),
                    None => lines.push("sign_test: n/a".to_owned()),
                }
                lines.push(format!(
                    "decision: clean_units={clean_units} mean_d={} direction={}",
                    cell(*mean_d),
                    serde_plain(direction)
                ));
                if let Some(e) = equivalence {
                    lines.push(format!(
                        "equivalence: mean_d interval [{:.6}, {:.6}] vs ±{} -> {}",
                        e.interval.lower,
                        e.interval.upper,
                        e.delta,
                        if e.equivalent { "parity" } else { "parity not established" }
                    ));
                }
                lines.push(format!(
                    "repeat_floor: max_delta={} spread={}",
                    repeat_floor.max_delta, repeat_floor.spread
                ));
                if let Some(c) = control {
                    lines.push(format!("control: units={:?} waived={}", c.units, c.waived));
                }
            }
            Some(OutcomeVerdict::RowAgreement {
                metric,
                bound,
                rows,
                worst,
                rows_beyond_bound,
            }) => lines.push(format!(
                "row {}: worst {worst:.3e} over {rows} rows, bound {bound:.3e}, {rows_beyond_bound} beyond",
                serde_plain(metric)
            )),
            Some(OutcomeVerdict::Law { alpha, lower, upper }) => lines.extend([("lower", lower), ("upper", upper)].map(
                |(side, fit)| {
                    format!(
                        "law fit, {side}: statistic {:.3} on {} dof over {} observations, p = {:.4} (alpha {alpha})",
                        fit.statistic, fit.degrees_of_freedom, fit.observations, fit.p_value
                    )
                },
            )),
            Some(OutcomeVerdict::Digest { units_equal, units }) => {
                lines.push(format!("outcome digests equal on {units_equal}/{units} unit(s)"));
            }
            None => {}
        }
        if let Some(s) = &self.speed {
            lines.push(format!(
                "cost (upper/lower time): {:.4} by minima, {:.4} by medians [{:.4}, {:.4}]{}",
                s.cost.of_minima,
                s.cost.of_medians,
                s.cost.interval.lower,
                s.cost.interval.upper,
                match (s.noise_band, s.indistinguishable_from_one) {
                    (Some(band), Some(true)) =>
                        format!(" — inside the x{band:.4} noise band: indistinguishable from 1"),
                    (Some(band), _) => format!(" — noise band x{band:.4}"),
                    _ => String::new(),
                }
            ));
            if let Some(shape) = &s.shape {
                lines.push(format!(
                    "shape: fixed cost worth {:+.1} units of work, per-work cost x{:.4}",
                    shape.fixed_work_equivalent, shape.per_work_ratio
                ));
            }
            if let Some(ttq) = &s.time_to_quality {
                lines.push(format!(
                    "time_to_quality: lower/upper {:?}, unreached {:?}",
                    ttq.ratio, ttq.unreached
                ));
            }
        }
        if let Some(s) = &self.space {
            lines.push(format!(
                "space (upper/lower): host {:?} device {:?} host bytes per unit of work {:?}",
                s.host_ratio,
                s.device_ratio,
                s.host_slope.map(|f| f.slope)
            ));
        }
        lines.extend(self.judgements.iter().map(|j| {
            format!(
                "  [{}] {:<8} {} — {}",
                match j.passed {
                    Some(true) => "pass",
                    Some(false) => "FAIL",
                    None => "n/m ",
                },
                serde_plain(&j.gate),
                j.rule,
                j.detail
            )
        }));
        lines.extend(
            self.refusals
                .iter()
                .map(|r| format!("  REFUSED: {}", r.message)),
        );
        lines
    }
}

/// A unit enum's serialized name, for the table.
pub fn serde_plain<T: Serialize>(value: &T) -> String {
    match serde_json::to_value(value) {
        Ok(serde_json::Value::String(s)) => s,
        _ => String::from("?"),
    }
}
