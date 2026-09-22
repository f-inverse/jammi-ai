//! Every reason the comparator declines to give a verdict, as one typed
//! error. A refusal is a statement about the *measurement* — the legs do not
//! describe one comparison, or a number cannot be trusted — never about
//! which side won.

use serde::Serialize;

use jammi_numerics::stats::Interval;

#[derive(Debug, Clone, thiserror::Error, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Refusal {
    #[error("unknown rung {rung:?}; this ladder's rungs are {known:?}")]
    UnknownRung { rung: String, known: Vec<String> },

    #[error("no edge lies between {from:?} and {to:?}: a span runs upward over at least one edge")]
    EmptySpan { from: String, to: String },

    #[error("leg file {file:?} is not named <rung>__<unit>__<take>.json: {reason}")]
    LegNameMalformed { file: String, reason: String },

    #[error("leg {leg}: unreadable — {reason}")]
    LegUnreadable { leg: String, reason: String },

    #[error("rung {rung:?} has no {take} leg for unit {unit:?}")]
    MissingLeg {
        rung: String,
        unit: String,
        take: String,
    },

    #[error(
        "leg {leg}: take {take:?} is neither a repeat (r<N>) nor a control this edge declares"
    )]
    UnknownTake { leg: String, take: String },

    #[error("identity field {field:?} is absent or null on {legs:?}: the legs cannot be shown to share a premise")]
    IdentityMissing { field: String, legs: Vec<String> },

    #[error("identity field {field:?} disagrees: {values:?}")]
    IdentityDisagreement {
        field: String,
        /// Each distinct value with the legs that reported it.
        values: Vec<(String, Vec<String>)>,
    },

    #[error("leg {leg}: premise {premise} violated — {reason}")]
    PremiseViolated {
        leg: String,
        premise: &'static str,
        reason: String,
    },

    #[error("{subject}: {measurement} is not measured, and a hard rule needs it")]
    MeasurementMissing {
        subject: String,
        measurement: &'static str,
    },

    #[error("leg {leg}: {got} timed iterations, need at least {need}")]
    TooFewSamples {
        leg: String,
        got: usize,
        need: usize,
    },

    #[error(
        "leg {leg}: the time series is not stationary (trend p = {p_value:.3e}, drift {relative_drift:+.2}% of the median over the series)"
    )]
    NonStationary {
        leg: String,
        p_value: f64,
        relative_drift: f64,
    },

    #[error("unit {unit:?}: outcome digests differ on an exact edge: {digests:?}")]
    DigestMismatch {
        unit: String,
        digests: Vec<(String, String)>,
    },

    #[error(
        "edge {edge}: the reference rung establishes no learning effect to preserve — its mean improvement from the untrained held-out loss to the judged point is {mean_improvement}, lower confidence bound {lower_bound}; no margin can be derived, so no-worse-than cannot be judged"
    )]
    AssayInsensitive {
        edge: String,
        mean_improvement: f64,
        lower_bound: f64,
    },

    #[error("edge {edge}: rule {rule} has no measured budget, and a hard rule cannot be judged against a number nobody measured")]
    Unbudgeted { edge: String, rule: String },

    #[error(
        "edge {edge}: {clean} premise-clean unit(s); the rule is stated for exactly {required}"
    )]
    WrongUnitCount {
        edge: String,
        clean: usize,
        required: usize,
    },

    #[error("rung {rung:?} unit {unit:?}: repeats differ by {delta}, more than the spread {spread} of the paired differences across units")]
    RepeatExceedsSpread {
        rung: String,
        unit: String,
        delta: f64,
        spread: f64,
    },

    #[error("edge {edge}: {found} control unit(s) on both rungs, {required} required")]
    ControlMissing {
        edge: String,
        found: usize,
        required: usize,
    },

    #[error("control leg {leg}: {reason}")]
    ControlInvalid { leg: String, reason: String },

    #[error("unit {unit:?}: {reason}")]
    VectorsMalformed { unit: String, reason: String },

    #[error("rung {rung:?}: time is not a line in work (relative residual {relative_residual:.3}, limit {limit}) — {reason}")]
    ShapeFitPoor {
        rung: String,
        relative_residual: f64,
        limit: f64,
        reason: String,
    },

    #[error(
        "the product of the edge ratios [{}, {}] and the directly measured end-to-end ratio [{}, {}] do not overlap: a layer interaction or a leg measured under different conditions",
        product.lower, product.upper, direct.lower, direct.upper
    )]
    TelescopingContradiction { product: Interval, direct: Interval },

    #[error("{context}: {reason}")]
    Statistics { context: String, reason: String },

    #[error("law file {file:?}: {reason}")]
    LawUnusable { file: String, reason: String },

    #[error("mutant column {label:?}: {reason}")]
    MutantColumnInvalid { label: String, reason: String },
}

impl Refusal {
    pub fn statistics(context: impl Into<String>, error: jammi_numerics::NumericsError) -> Self {
        Self::Statistics {
            context: context.into(),
            reason: error.to_string(),
        }
    }
}

/// A refusal as it appears in a verdict: its typed fields and the sentence a
/// reader sees.
#[derive(Debug, Clone, Serialize)]
pub struct ReportedRefusal {
    #[serde(flatten)]
    pub refusal: Refusal,
    pub message: String,
}

impl From<Refusal> for ReportedRefusal {
    fn from(refusal: Refusal) -> Self {
        let message = refusal.to_string();
        Self { refusal, message }
    }
}
