//! What the comparator says: one verdict per edge on three axes, folded
//! into one status for the run.

use serde::Serialize;

use jammi_numerics::stats::{GoodnessOfFit, Interval, LinearFit, MarginTestResult, SignTestResult};

use super::definition::{Difference, JudgedPoint, RowMetric, Rule, RuleForce, Workload};
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

/// The outcome claim of a seeded edge: the upper rung's loss is no worse than
/// the lower rung's by more than the margin.
pub const NON_INFERIORITY_RULE: &str = "outcome_non_inferior";

/// Where the number a rule is judged against came from. A rule is never
/// judged against a number nobody measured or derived.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum Bound {
    /// The rule has no bound: it holds or breaks on its own terms.
    None,
    /// A significance level, fixed in the ladder's definition.
    Level { alpha: f64 },
    /// Derived in this run from the legs themselves.
    Derived { value: f64 },
    /// Read off a committed, measured artifact.
    Measured { value: f64, measured_from: String },
    /// No artifact has measured a bound for this rule yet.
    Unbudgeted,
}

impl Bound {
    /// The bound a rule of the ladder's definition carries.
    pub fn of(rule: &Rule) -> Self {
        rule.budget
            .as_ref()
            .map_or(Self::Unbudgeted, |b| Self::Measured {
                value: b.bound,
                measured_from: b.measured_from.clone(),
            })
    }
}

impl std::fmt::Display for Bound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => f.write_str("no bound"),
            Self::Level { alpha } => write!(f, "alpha {alpha}"),
            Self::Derived { value } => write!(f, "{value:.6} (derived)"),
            Self::Measured {
                value,
                measured_from,
            } => write!(f, "{value} (measured, {measured_from})"),
            Self::Unbudgeted => f.write_str("UNBUDGETED"),
        }
    }
}

/// What applying a rule came to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Held {
    Passed,
    Failed,
    /// The quantity was not measured, or the rule has no bound to judge it
    /// against.
    Unjudged,
    /// The quantity's measurement was refused; the refusal, which names why,
    /// is the verdict's own.
    Refused,
}

impl From<Option<bool>> for Held {
    fn from(passed: Option<bool>) -> Self {
        match passed {
            Some(true) => Self::Passed,
            Some(false) => Self::Failed,
            None => Self::Unjudged,
        }
    }
}

/// One rule, applied.
#[derive(Debug, Clone, Serialize)]
pub struct Judgement {
    pub rule: &'static str,
    pub force: RuleForce,
    pub bound: Bound,
    pub held: Held,
    /// Which way the upper rung moved when the rule failed: a failure in the
    /// favourable direction is investigated, never counted as a pass.
    pub direction: Direction,
    pub detail: String,
}

impl Judgement {
    /// A one-sided rule whose only failure is a degradation.
    pub fn new(
        rule: &'static str,
        force: RuleForce,
        bound: Bound,
        passed: Option<bool>,
        detail: impl Into<String>,
    ) -> Self {
        let held = Held::from(passed);
        let direction = match held {
            Held::Failed => Direction::Degradation,
            _ => Direction::None,
        };
        Self {
            rule,
            force,
            bound,
            held,
            direction,
            detail: detail.into(),
        }
    }

    /// A rule whose quantity's measurement was refused: judged neither way,
    /// and not unmeasured — the refusal the verdict carries is its account.
    pub fn refused(
        rule: &'static str,
        force: RuleForce,
        bound: Bound,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            held: Held::Refused,
            ..Self::new(rule, force, bound, None, detail)
        }
    }

    /// A rule that can fail in either direction.
    pub fn directed(
        rule: &'static str,
        force: RuleForce,
        bound: Bound,
        passed: Option<bool>,
        direction: Direction,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            direction,
            ..Self::new(rule, force, bound, passed, detail)
        }
    }

    /// A rule judged against its measured budget: unmeasured when the
    /// quantity is absent, unjudged when no artifact has measured a bound.
    pub fn budgeted(
        rule: &'static str,
        spec: &Rule,
        value: Option<f64>,
        holds: impl Fn(f64, f64) -> bool,
        detail: impl Into<String>,
    ) -> Self {
        let passed = spec
            .budget
            .as_ref()
            .zip(value)
            .map(|(budget, value)| holds(value, budget.bound));
        Self::new(rule, spec.force, Bound::of(spec), passed, detail)
    }

    fn fails_hard(&self) -> bool {
        self.force == RuleForce::Hard && self.held == Held::Failed
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    None,
    /// The upper rung is worse: a higher loss, a slower step, more memory.
    Degradation,
    /// The upper rung is better.
    Improvement,
}

#[derive(Debug, Clone, Serialize)]
pub struct UnitDifference {
    pub unit: String,
    /// The judged epoch: where the lower rung's held-out loss is lowest.
    pub epoch: Option<usize>,
    pub lower: Option<f64>,
    pub upper: Option<f64>,
    /// `upper − lower`.
    pub d: Option<f64>,
    /// The lower rung's untrained held-out loss less its judged loss: what
    /// it learned.
    pub reference_improvement: Option<f64>,
    /// Whether this unit's legs cleared every premise and so count.
    pub clean: bool,
}

/// The reference rung's learning effect, from which the margin is derived.
#[derive(Debug, Clone, Serialize)]
pub struct Assay {
    /// Mean over clean units of the reference's improvement from its
    /// untrained loss to the judged point.
    pub mean_improvement: f64,
    /// The `1 − 2α` bootstrap interval of that mean.
    pub interval: Interval,
    /// `M1`: the interval's lower bound, the effect the reference is shown
    /// to have.
    pub established_effect: f64,
    /// `δ`: the fraction of `M1` the upper rung must preserve.
    pub delta: f64,
    /// Whether an effect was established at all (`M1 > 0`).
    pub sensitive: bool,
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

/// What one tensor's pair of gradients is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorKind {
    /// Both gradients carry a direction: the pair is measured.
    Signal,
    /// Both gradients are zero: structurally so at a zero adapter factor,
    /// and no evidence either way.
    Vacuous,
    /// Exactly one side's gradient is zero.
    OneSidedZero,
    /// A side's gradient has a non-finite entry.
    NonFinite,
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorAgreement {
    pub unit: String,
    pub name: String,
    pub kind: TensorKind,
    pub cosine: Option<f64>,
    /// `‖upper − lower‖ ÷ ‖lower‖`.
    pub relative_error: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OutcomeVerdict {
    SeededLoss {
        judged_at: JudgedPoint,
        per_unit: Vec<UnitDifference>,
        clean_units: usize,
        sign_test: Option<SignTestResult>,
        /// The concordance count the sign test needs at this edge's level
        /// and unit count.
        critical_count: Option<usize>,
        mean_d: Option<f64>,
        direction: Direction,
        /// The reference's learning effect and the margin derived from it.
        assay: Option<Assay>,
        /// Both one-sided tests of the mean paired difference against the
        /// edge's margin: non-inferiority is the claim, equivalence is
        /// reported beside it.
        margin_test: Option<MarginTestResult>,
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
    GradientAgreement {
        tensors: Vec<TensorAgreement>,
        /// Over the tensors that carry a signal on both sides.
        mean_cosine: Option<f64>,
        worst_cosine: Option<f64>,
        worst_relative_error: Option<f64>,
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
    /// Where each leg's run settled: the iterations its initial transient
    /// took, which no number above reads.
    pub settled: Vec<Settled>,
    /// The half-width, as a ratio, of the same rung measured against itself.
    pub noise_band: Option<f64>,
    /// The cost lies inside the noise band, whatever its point value.
    pub indistinguishable_from_one: Option<bool>,
    /// On a revision edge: a second build of the lower side against the
    /// lower side, whose interval widens the noise band.
    pub aa_null: Option<Ratio>,
    pub shape: Option<ShapeVerdict>,
    pub time_to_quality: Option<TimeToQuality>,
}

/// Where one leg's run settled.
#[derive(Debug, Clone, Serialize)]
pub struct Settled {
    pub leg: String,
    /// The iterations cut as the run's initial transient.
    pub transient: usize,
    /// The cut fell at the limit of the run's first half: the transient was
    /// not seen to end, and the second half is what the stationarity test
    /// judged.
    pub at_limit: bool,
    /// The iterations the run filed.
    pub iterations: usize,
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
    /// What differs between the two rungs: what this edge's cost is the
    /// cost of.
    pub difference: Difference,
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
        difference: &Difference,
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
                .filter(|j| j.force == RuleForce::Hard && j.held == Held::Unjudged)
                .map(|j| match j.bound {
                    Bound::Unbudgeted => Refusal::Unbudgeted {
                        edge: edge.clone(),
                        rule: j.rule.to_owned(),
                    },
                    _ => Refusal::MeasurementMissing {
                        subject: edge.clone(),
                        measurement: j.rule,
                    },
                }),
        );
        let failed: Vec<&Judgement> = judgements.iter().filter(|j| j.fails_hard()).collect();
        let only_improvements = failed.iter().all(|j| j.direction == Direction::Improvement);
        let status = if !refusals.is_empty() {
            Status::Invalid
        } else if failed.is_empty() {
            Status::Green
        } else if only_improvements {
            Status::RedForInvestigation
        } else {
            Status::Red
        };
        Self {
            edge,
            difference: difference.clone(),
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

fn cell(v: Option<f64>) -> String {
    v.map_or_else(|| "n/a".to_owned(), |v| format!("{v:.6}"))
}

impl EdgeVerdict {
    pub fn table(&self) -> Vec<String> {
        let mut lines = vec![format!(
            "## {}  [{}]  — {}",
            self.edge,
            serde_plain(&self.status),
            self.difference.describe()
        )];
        match &self.outcome {
            Some(OutcomeVerdict::SeededLoss {
                judged_at,
                per_unit,
                clean_units,
                sign_test,
                critical_count,
                mean_d,
                direction,
                assay,
                margin_test,
                repeat_floor,
                control,
            }) => {
                lines.push(format!("judged at: {}", serde_plain(judged_at)));
                lines.push(format!(
                    "{:<10}{:<7}{:<14}{:<14}{:<14}{:<14}{}",
                    "unit", "epoch", "lower", "upper", "d", "learned", "premise"
                ));
                lines.extend(per_unit.iter().map(|u| {
                    format!(
                        "{:<10}{:<7}{:<14}{:<14}{:<14}{:<14}{}",
                        u.unit,
                        u.epoch.map_or_else(|| "n/a".to_owned(), |e| e.to_string()),
                        cell(u.lower),
                        cell(u.upper),
                        cell(u.d),
                        cell(u.reference_improvement),
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
                if let Some(a) = assay {
                    lines.push(format!(
                        "assay: reference improved by {:.6} [{:.6}, {:.6}]; M1={:.6} delta={:.6} -> {}",
                        a.mean_improvement,
                        a.interval.lower,
                        a.interval.upper,
                        a.established_effect,
                        a.delta,
                        if a.sensitive {
                            "effect established"
                        } else {
                            "NO EFFECT ESTABLISHED"
                        }
                    ));
                }
                if let Some(m) = margin_test {
                    lines.push(format!(
                        "margin: mean_d interval [{:.6}, {:.6}] vs ±{:.6} -> {}; {}",
                        m.interval.lower,
                        m.interval.upper,
                        m.delta,
                        if m.below_upper_margin {
                            "non-inferior"
                        } else {
                            "non-inferiority not established"
                        },
                        if m.equivalent() {
                            "equivalent"
                        } else if m.below_upper_margin {
                            "not equivalent: the upper rung may be better by more than the margin"
                        } else {
                            "not equivalent"
                        }
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
            Some(OutcomeVerdict::GradientAgreement {
                tensors,
                mean_cosine,
                worst_cosine,
                worst_relative_error,
            }) => {
                lines.push(format!(
                    "{:<10}{:<48}{:<16}{:<14}{}",
                    "unit", "tensor", "kind", "cosine", "relative_error"
                ));
                lines.extend(tensors.iter().map(|t| {
                    format!(
                        "{:<10}{:<48}{:<16}{:<14}{}",
                        t.unit,
                        t.name,
                        serde_plain(&t.kind),
                        t.cosine
                            .map_or_else(|| "n/a".to_owned(), |c| format!("{c:.9}")),
                        t.relative_error
                            .map_or_else(|| "n/a".to_owned(), |e| format!("{e:.3e}"))
                    )
                }));
                lines.push(format!(
                    "gradients: mean cosine {}, worst cosine {}, worst relative error {}",
                    mean_cosine.map_or_else(|| "n/a".to_owned(), |c| format!("{c:.9}")),
                    worst_cosine.map_or_else(|| "n/a".to_owned(), |c| format!("{c:.9}")),
                    worst_relative_error.map_or_else(|| "n/a".to_owned(), |e| format!("{e:.3e}"))
                ));
            }
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
                        format!(" — INDETERMINATE: inside the x{band:.4} noise band"),
                    (Some(band), _) => format!(" — outside the x{band:.4} noise band"),
                    _ => String::new(),
                }
            ));
            if let Some((least, most)) = s.settled.iter().map(|l| l.transient).fold(
                None,
                |range: Option<(usize, usize)>, t| {
                    Some(range.map_or((t, t), |(lo, hi)| (lo.min(t), hi.max(t))))
                },
            ) {
                let at_limit = s.settled.iter().filter(|l| l.at_limit).count();
                lines.push(format!(
                    "settled: every leg's initial transient cut by MSER, {least}–{most} iterations; {at_limit} of {} legs cut at the half-way limit",
                    s.settled.len()
                ));
            }
            if let Some(aa) = &s.aa_null {
                lines.push(format!(
                    "A/A null (rebuilt/base): {:.4} by medians [{:.4}, {:.4}]",
                    aa.of_medians, aa.interval.lower, aa.interval.upper
                ));
            }
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
                "  [{}] {:<8} {} against {} — {}",
                match j.held {
                    Held::Passed => "pass",
                    Held::Failed => "FAIL",
                    Held::Unjudged => "n/m ",
                    Held::Refused => "REF ",
                },
                serde_plain(&j.force),
                j.rule,
                j.bound,
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
