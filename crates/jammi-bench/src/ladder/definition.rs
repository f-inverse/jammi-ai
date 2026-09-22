//! The ladder as data: workloads, rungs in order, what differs between
//! adjacent rungs, and the rules of each edge.
//!
//! An edge is never constructed from two free rung names. A [`Ladder`] holds
//! a reference rung, then the rungs reached by a *cross-stack* edge, then the
//! rungs reached by an *exact* edge, each list as long as the workload needs;
//! [`Ladder::edges`] derives each edge from a rung and the one below it. Two
//! properties therefore hold for every ladder that can be written down: an
//! edge always joins adjacent rungs, and no cross-stack edge ever sits above
//! an exact one — equality composes upward without tolerance only if nothing
//! above it reintroduces a margin. The one edge made outside a ladder is a
//! *revision* edge, [`Edge::revision`]: a rung against itself, built from
//! another revision of the engine.
//!
//! Every budget with no measurement behind it is [`Gate::Evidence`]: reported
//! beside the verdict, never gating it. They are collected in [`budget`].

use serde::Serialize;

use jammi_numerics::stats::FitStatistic;

use crate::kernel_arm::{KernelArm, KernelFamily};
use crate::leg::Payload;
use crate::report::{EncodePayload, Nullable, TrainRunPayload, TrainStepPayload};

use super::premise::LegPremise;
use super::refusal::Refusal;

/// A function from committed inputs to an artifact.
///
/// A composite workload is cut at its committed intermediate artifact, so
/// each ladder compares one thing: graph-supervised fine-tuning is
/// [`Self::GraphSample`] then [`Self::TrainRun`], the second fed the pair
/// table the first committed. Sharing that table between stacks removes the
/// sampler's randomness from the training comparison instead of averaging
/// over it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Workload {
    /// One vector per key.
    Encode,
    /// One optimizer step over a fixed synthetic batch: the cost of the
    /// step, never learning.
    TrainStep,
    /// An adapter and a held-out loss trajectory, from a pair table.
    TrainRun,
    /// A pair table, from random walks over a graph.
    GraphSample,
    /// One propagated vector per node.
    Propagate,
    /// A context predictor's weights and a held-out loss trajectory.
    PredictorTrainRun,
}

const NON_NULL: Nullable = Nullable::NonNull;

/// What two `graph-sample` legs must agree on. The graph and the law derived
/// from it are swept together: each size of the sweep is its own graph.
const GRAPH_SAMPLE_IDENTITY_FIELDS: &[(&str, Nullable)] = &[
    ("seed", NON_NULL),
    ("graph_edges_sha256", NON_NULL),
    ("law_sha256", NON_NULL),
    ("node_count", NON_NULL),
    ("edge_count", NON_NULL),
    ("walk_length", NON_NULL),
    ("walks_per_node", NON_NULL),
    ("return_p", NON_NULL),
    ("in_out_q", NON_NULL),
];

const PROPAGATE_IDENTITY_FIELDS: &[(&str, Nullable)] = &[
    ("graph_edges_sha256", NON_NULL),
    ("features_sha256", NON_NULL),
    ("node_count", NON_NULL),
    ("edge_count", NON_NULL),
    ("dim", NON_NULL),
    ("hops", NON_NULL),
    (
        "alpha",
        Nullable::NullMeans("no teleport: plain K-hop smoothing"),
    ),
    ("weighting", NON_NULL),
    ("compute_precision", NON_NULL),
];

const PREDICTOR_TRAIN_RUN_IDENTITY_FIELDS: &[(&str, Nullable)] = &[
    ("seed", NON_NULL),
    ("architecture", NON_NULL),
    ("context_k", NON_NULL),
    ("feature_dim", NON_NULL),
    ("value_dim", NON_NULL),
    ("hidden_dim", NON_NULL),
    ("num_heads", NON_NULL),
    ("num_layers", NON_NULL),
    ("head_width", NON_NULL),
    ("initial_weights_sha256", NON_NULL),
    ("train_episodes_sha256", NON_NULL),
    ("heldout_episodes_sha256", NON_NULL),
    ("epochs", NON_NULL),
    ("batch", NON_NULL),
    ("lr", NON_NULL),
    ("weight_decay", NON_NULL),
    ("schedule", NON_NULL),
    ("compute_precision", NON_NULL),
];

impl Workload {
    /// The key a producer's leg block lives under — `tiers.<key>` in a
    /// `jammi-bench` report, `<key>` at the top level of any other
    /// producer's JSON.
    pub fn tier_key(self) -> &'static str {
        match self {
            Self::Encode => "encode_step",
            Self::TrainStep => "finetune_step",
            Self::TrainRun => "finetune_run",
            Self::GraphSample => "graph_sample",
            Self::Propagate => "propagate",
            Self::PredictorTrainRun => "predictor_train_run",
        }
    }

    /// What two legs must agree on to be comparable. Where a tier declares
    /// its identity, that declaration — the one its producer asserts on every
    /// emit — is the list; the other workloads' lists live here, and their
    /// producers take them from here.
    pub fn identity_fields(self) -> &'static [(&'static str, Nullable)] {
        match self {
            Self::Encode => EncodePayload::IDENTITY_FIELDS,
            Self::TrainStep => TrainStepPayload::IDENTITY_FIELDS,
            Self::TrainRun => TrainRunPayload::IDENTITY_FIELDS,
            Self::GraphSample => GRAPH_SAMPLE_IDENTITY_FIELDS,
            Self::Propagate => PROPAGATE_IDENTITY_FIELDS,
            Self::PredictorTrainRun => PREDICTOR_TRAIN_RUN_IDENTITY_FIELDS,
        }
    }

    /// The identity fields that name a unit — the axis a sweep varies — and
    /// so differ between units by construction while agreeing within one.
    pub fn swept_fields(self) -> &'static [&'static str] {
        match self {
            Self::Encode => &["batch", "row_lengths"],
            Self::TrainStep => &["batch", "seq", "row_lengths", "lora_dropout"],
            Self::TrainRun | Self::PredictorTrainRun => &["seed"],
            Self::GraphSample => &[
                "graph_edges_sha256",
                "law_sha256",
                "node_count",
                "edge_count",
            ],
            Self::Propagate => &[
                "graph_edges_sha256",
                "features_sha256",
                "node_count",
                "edge_count",
            ],
        }
    }

    pub fn ladder(self) -> Ladder {
        match self {
            Self::Encode => encode_ladder(),
            Self::TrainStep => train_step_ladder(),
            Self::TrainRun => train_run_ladder(),
            Self::GraphSample => graph_sample_ladder(),
            Self::Propagate => propagate_ladder(),
            Self::PredictorTrainRun => predictor_train_run_ladder(),
        }
    }

    /// The rules of a revision edge of any rung of this workload.
    pub fn revision_rules(self) -> RevisionRules {
        RevisionRules {
            within_noise_band: Gate::Hard,
            space: budget::SPACE,
        }
    }
}

/// Whether a rule's failure fails the run or is reported beside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Gate {
    Hard,
    Evidence,
}

/// A bound and how much it counts.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Judged<T> {
    pub bound: T,
    pub gate: Gate,
}

const fn evidence<T>(bound: T) -> Judged<T> {
    Judged {
        bound,
        gate: Gate::Evidence,
    }
}

/// Every bound with no measurement behind it, in one place. Each is
/// [`Gate::Evidence`] until a measured value replaces it: a gate with an
/// invented number is worse than no gate.
pub mod budget {
    use super::{evidence, Judged, SpaceRules};

    /// `upper ÷ lower` time an engine layer may add.
    pub const LAYER_OVERHEAD: Judged<f64> = evidence(1.10);
    /// `upper ÷ lower` peak memory, host and device, on any edge.
    pub const SPACE: SpaceRules = SpaceRules {
        host_ratio: evidence(1.10),
        device_ratio: evidence(1.10),
    };
    /// `lower ÷ upper` time a jammi rung must reach against its reference
    /// framework.
    pub const FRAMEWORK_SPEED_BAR: Judged<f64> = evidence(0.9);
    /// `lower ÷ upper` time the fused kernels must reach against the
    /// reference kernels.
    pub const KERNEL_SPEED_BAR: Judged<f64> = evidence(1.0);
    /// `upper ÷ lower` per-work cost of a layer over a size sweep.
    pub const PER_WORK_RATIO: f64 = 1.10;
    /// A layer's added fixed cost, in units of work of the rung below, for
    /// an in-process plan layer.
    pub const PLAN_FIXED_WORK: f64 = 64.0;
    /// The same, for placement on an executor over a plan.
    pub const PLACED_FIXED_WORK: f64 = 256.0;
    /// The same, for a graph layer measured in edges.
    pub const GRAPH_FIXED_WORK: f64 = 4096.0;
    /// The same, for placement over a graph plan.
    pub const GRAPH_PLACED_FIXED_WORK: f64 = 16384.0;
    /// Host bytes the streaming loader may keep per training row: an offset,
    /// never the row.
    pub const STREAMED_BYTES_PER_ROW: Judged<f64> = evidence(16.0);
    /// Host bytes the graph sampler may keep per edge beyond the adjacency.
    pub const SAMPLER_BYTES_PER_EDGE: Judged<f64> = evidence(256.0);
}

/// What differs between the two legs of an edge: the one thing the edge's
/// cost is the cost of.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Difference {
    /// The rung below runs in a reference framework; this rung is the
    /// engine's lowest stack of the workload.
    Framework { reference: &'static str },
    /// The same engine, with these fused-kernel families off below and on
    /// above.
    KernelArm { families_on: Vec<KernelFamily> },
    /// One engine layer, added above.
    Layer { name: &'static str },
    /// The same rung, built from another revision of the engine.
    Revision,
}

impl Difference {
    fn kernel_arm(below: &KernelArm, above: &KernelArm) -> Self {
        Self::KernelArm {
            families_on: below.off.difference(&above.off).copied().collect(),
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Framework { reference } => format!("the engine against {reference}"),
            Self::KernelArm { families_on } => {
                let names: Vec<String> = families_on
                    .iter()
                    .map(super::verdict::serde_plain)
                    .collect();
                format!("the fused kernels {}", names.join(", "))
            }
            Self::Layer { name } => (*name).to_owned(),
            Self::Revision => "another revision of the engine".to_owned(),
        }
    }
}

/// One implementation stack of a workload.
#[derive(Debug, Clone)]
pub struct Rung {
    pub name: String,
    /// Facts every leg of this rung must show about itself.
    pub premises: Vec<LegPremise>,
    /// Set when this rung's host memory must not grow with input size: the
    /// budget on the fitted slope, in bytes per row.
    pub flat_host_memory: Option<Judged<f64>>,
}

impl Rung {
    fn new(name: &str, premises: Vec<LegPremise>) -> Self {
        Self {
            name: name.to_owned(),
            premises,
            flat_host_memory: None,
        }
    }

    /// A jammi training rung on `arm`: besides `premises`, its legs must
    /// state the arm and prove it by their dispatch counters, and dispatch
    /// `must_run` fused.
    fn on_arm(
        name: &str,
        arm: KernelArm,
        must_run: &[KernelFamily],
        premises: Vec<LegPremise>,
    ) -> Self {
        let mut premises = premises;
        premises.push(LegPremise::Arm(arm.label()));
        premises.push(LegPremise::KernelArm(arm));
        premises.extend(must_run.iter().map(|f| LegPremise::Dispatched(*f)));
        Self::new(name, premises)
    }
}

/// Peak-memory budgets: the ratio `upper ÷ lower` of each instrument.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct SpaceRules {
    pub host_ratio: Judged<f64>,
    pub device_ratio: Judged<f64>,
}

/// Budgets on the two coefficients of `time = fixed + per_work · work`,
/// where a leg's `work` is the size its cost scales with: rows for `encode`,
/// edges × hops for `propagate`, edges for `graph-sample`.
///
/// The fixed budget is in units of work: the layer's added fixed cost divided
/// by the lower rung's per-work cost — how much work the layer's constant
/// overhead is worth. Both budgets are dimensionless, so neither depends on
/// the speed of the box that measured them.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct ShapeRules {
    pub fixed_work_equivalent: f64,
    pub per_work_ratio: f64,
    /// Refuse the fit when the residual exceeds this fraction of the mean
    /// fitted time: the two coefficients only mean anything on a line.
    pub max_relative_residual: f64,
    pub gate: Gate,
}

/// A run that must *fail* a premise, measured beside the runs that must pass
/// it — the proof that the premise can fail at all.
#[derive(Debug, Clone, Serialize)]
pub struct ControlRule {
    /// The take tag control legs are filed under.
    pub take: &'static str,
    /// The identity field the control overrides, and the value it must show.
    pub field: &'static str,
    pub value: f64,
    /// Units that must carry a control leg on both rungs.
    pub required_units: usize,
}

/// How far apart two stacks' rows may be, measured one of two ways against
/// one allowance: a relative perturbation of `√ε` at the compute dtype's
/// machine epsilon `ε`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RowMetric {
    /// `cos(a, b) >= 1 − ε / 2`: direction only, for vectors consumed by
    /// similarity.
    Cosine,
    /// `‖a − b‖ ÷ ‖b‖ <= √ε`: direction and magnitude, for vectors consumed
    /// as values.
    RelativeError,
}

/// How the outcome of an edge between two stacks that cannot be
/// byte-identical is judged.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CrossStackOutcome {
    /// Paired by seed: both stacks start from the same initial tensors and
    /// see the same rows in the same order, one held-out loss per seed on
    /// each side; `d = upper − lower`.
    SeededLoss {
        /// The number of premise-clean seeds the rule is stated for. Any
        /// other count is refused, never rescaled.
        seeds: usize,
        /// Two-sided level of the exact sign test.
        sign_alpha: f64,
        direction_gate: Gate,
        /// The margin, fixed before any leg runs. `None` is a margin nobody
        /// has fixed, and the comparison is refused.
        delta: Option<f64>,
        /// Level of each one-sided margin test.
        margin_alpha: f64,
        /// The claim: the upper rung's loss is no worse than the lower's by
        /// more than `delta`. Two-sided equivalence is reported beside it as
        /// evidence — an upper rung better by more than `delta` has not
        /// failed "as good as".
        non_inferiority_gate: Gate,
        control: Option<ControlRule>,
    },
    /// Paired by row: deterministic math on the same keyed input, one vector
    /// per row on each side.
    RowAgreement { metric: RowMetric, gate: Gate },
    /// Not pairable: the two stacks draw from different random streams, so
    /// neither their outputs nor any per-unit difference of them can be
    /// compared. Each rung's output is instead tested against the workload's
    /// analytic law, and both must fit it.
    Law {
        statistic: FitStatistic,
        /// Level of each rung's goodness-of-fit test, fixed here: a faithful
        /// sampler is refused with exactly this probability.
        alpha: f64,
        gate: Gate,
    },
    /// The artifact is a cost, not a result: a step over synthetic inputs
    /// has no outcome to compare.
    None,
}

/// Rules of an edge between two stacks — two frameworks, or two kernel sets.
#[derive(Debug, Clone, Serialize)]
pub struct CrossStackRules {
    pub outcome: CrossStackOutcome,
    /// Non-inferiority: the lower bound of `lower ÷ upper` time must exceed
    /// this bar.
    pub speed_bar: Judged<f64>,
    pub space: SpaceRules,
    pub shape: Option<ShapeRules>,
    /// Non-inferiority bar on `lower ÷ upper` time-to-quality.
    pub time_to_quality_bar: Option<Judged<f64>>,
}

/// Rules of an edge inside one deterministic engine: the outcome is digest
/// equality, and the speed rule is the layer's overhead budget.
#[derive(Debug, Clone, Serialize)]
pub struct ExactRules {
    /// The upper bound of `upper ÷ lower` time must stay under this budget.
    pub overhead_budget: Judged<f64>,
    pub space: SpaceRules,
    pub shape: Option<ShapeRules>,
}

/// Rules of a revision edge: two builds of one rung are expected to cost the
/// same, so the cost must lie inside the rung's own noise band. The band is
/// the wider of what a build's repeats measure against each other and what
/// two builds of the *same* revision measure against each other — the edge's
/// A/A null, run in the same session, because build-to-build variation is
/// not visible to repeats of one build.
#[derive(Debug, Clone, Serialize)]
pub struct RevisionRules {
    pub within_noise_band: Gate,
    pub space: SpaceRules,
}

#[derive(Debug, Clone, Copy)]
pub enum EdgeKind<'a> {
    CrossStack(&'a CrossStackRules),
    Exact(&'a ExactRules),
    Revision(&'a RevisionRules),
}

/// An adjacent pair of rungs and the rules between them. Only
/// [`Ladder::edges`] and [`Edge::revision`] make one from nothing;
/// [`Edge::with_upper`] re-aims an existing one.
#[derive(Debug, Clone, Copy)]
pub struct Edge<'a> {
    lower: &'a Rung,
    upper: &'a Rung,
    difference: &'a Difference,
    kind: EdgeKind<'a>,
}

/// The side tags a revision edge's legs are filed under: `<rung>@base` and
/// `<rung>@revised`, with `<rung>@rebuilt` — a second build of the base
/// revision — beside them as the edge's A/A null, measured in the same
/// session.
pub const REVISION_BASE: &str = "base";
pub const REVISION_REVISED: &str = "revised";
pub const REVISION_REBUILT: &str = "rebuilt";

const REVISION: Difference = Difference::Revision;

impl<'a> Edge<'a> {
    /// `rung` against itself, built from another revision.
    pub fn revision(rung: &'a Rung, rules: &'a RevisionRules) -> Self {
        Self {
            lower: rung,
            upper: rung,
            difference: &REVISION,
            kind: EdgeKind::Revision(rules),
        }
    }

    pub fn lower(&self) -> &'a Rung {
        self.lower
    }

    pub fn upper(&self) -> &'a Rung {
        self.upper
    }

    pub fn kind(&self) -> EdgeKind<'a> {
        self.kind
    }

    pub fn difference(&self) -> &'a Difference {
        self.difference
    }

    pub fn is_revision(&self) -> bool {
        matches!(self.kind, EdgeKind::Revision(_))
    }

    /// The rung name the lower legs are filed under.
    pub fn lower_name(&self) -> String {
        if self.is_revision() {
            format!("{}@{REVISION_BASE}", self.lower.name)
        } else {
            self.lower.name.clone()
        }
    }

    /// The rung name the upper legs are filed under.
    pub fn upper_name(&self) -> String {
        if self.is_revision() {
            format!("{}@{REVISION_REVISED}", self.upper.name)
        } else {
            self.upper.name.clone()
        }
    }

    /// The rung name a revision edge's A/A twin legs are filed under.
    pub fn rebuilt_name(&self) -> Option<String> {
        self.is_revision()
            .then(|| format!("{}@{REVISION_REBUILT}", self.lower.name))
    }

    pub fn name(&self) -> String {
        format!("{} -> {}", self.lower_name(), self.upper_name())
    }

    /// This edge with a stand-in for its upper rung — a deliberately broken
    /// build measured against the same lower legs under the same rules, to
    /// show the rules can fail.
    pub fn with_upper(&self, upper: &'a Rung) -> Self {
        Self { upper, ..*self }
    }
}

#[derive(Debug, Clone)]
pub struct Ladder {
    pub workload: Workload,
    reference: Rung,
    cross_stack: Vec<(Rung, Difference, CrossStackRules)>,
    exact: Vec<(Rung, Difference, ExactRules)>,
}

impl Ladder {
    pub fn rungs(&self) -> impl Iterator<Item = &Rung> {
        std::iter::once(&self.reference)
            .chain(self.cross_stack.iter().map(|(r, _, _)| r))
            .chain(self.exact.iter().map(|(r, _, _)| r))
    }

    pub fn rung(&self, name: &str) -> Option<&Rung> {
        self.rungs().find(|r| r.name == name)
    }

    /// Every edge, bottom to top.
    pub fn edges(&self) -> Vec<Edge<'_>> {
        let uppers = self
            .cross_stack
            .iter()
            .map(|(r, d, rules)| (r, d, EdgeKind::CrossStack(rules)))
            .chain(
                self.exact
                    .iter()
                    .map(|(r, d, rules)| (r, d, EdgeKind::Exact(rules))),
            );
        self.rungs()
            .zip(uppers)
            .map(|(lower, (upper, difference, kind))| Edge {
                lower,
                upper,
                difference,
                kind,
            })
            .collect()
    }

    /// The contiguous edges from rung `from` up to rung `to`; the whole
    /// ladder when both are `None`.
    pub fn span(&self, from: Option<&str>, to: Option<&str>) -> Result<Vec<Edge<'_>>, Refusal> {
        let names: Vec<&str> = self.rungs().map(|r| r.name.as_str()).collect();
        let index = |name: Option<&str>, default: usize| match name {
            None => Ok(default),
            Some(n) => names
                .iter()
                .position(|r| *r == n)
                .ok_or_else(|| Refusal::UnknownRung {
                    rung: n.to_owned(),
                    known: names.iter().map(|s| (*s).to_owned()).collect(),
                }),
        };
        let (lo, hi) = (index(from, 0)?, index(to, names.len() - 1)?);
        if lo >= hi {
            return Err(Refusal::EmptySpan {
                from: names[lo].to_owned(),
                to: names[hi].to_owned(),
            });
        }
        Ok(self.edges()[lo..hi].to_vec())
    }
}

/// How a leg's time series is read, the same for every edge.
pub struct SpeedInstrument;

impl SpeedInstrument {
    /// Fewest post-warmup iterations a leg may carry.
    pub const MIN_SAMPLES: usize = 16;
    /// A series is refused as non-stationary when its Mann-Kendall test
    /// rejects at this level *and* its Theil-Sen drift over the whole series
    /// exceeds [`Self::MAX_RELATIVE_DRIFT`] of its median. Significance alone
    /// would refuse any long series for a drift too small to matter; drift
    /// alone would refuse a short noisy one for a slope it cannot resolve.
    pub const TREND_ALPHA: f64 = 0.01;
    pub const MAX_RELATIVE_DRIFT: f64 = 0.02;
    pub const BOOTSTRAP_ITERATIONS: usize = 2000;
    /// Two-sided level of every ratio interval.
    pub const INTERVAL_ALPHA: f64 = 0.05;
    pub const BOOTSTRAP_SEED: u64 = 0x1add_e700;
}

/// Resamples behind a paired margin test's interval.
pub const MARGIN_BOOTSTRAP_ITERATIONS: usize = 10_000;

/// The held-out loss difference the train-run instrument has been shown to
/// resolve: the smallest mean shift a deliberately mutated arm was detected
/// at by the 12-seed sign test over its fixture.
const TRAIN_RUN_DELTA: f64 = 0.0434;

/// A seeded-loss outcome whose two claims — no directional difference, and
/// the upper rung no worse by more than `delta` — count as `gate`.
fn seeded_loss(delta: Option<f64>, control: Option<ControlRule>, gate: Gate) -> CrossStackOutcome {
    CrossStackOutcome::SeededLoss {
        seeds: 12,
        sign_alpha: 0.0064,
        direction_gate: gate,
        delta,
        margin_alpha: 0.05,
        non_inferiority_gate: gate,
        control,
    }
}

fn shape(fixed_work_equivalent: f64) -> Option<ShapeRules> {
    Some(ShapeRules {
        fixed_work_equivalent,
        per_work_ratio: budget::PER_WORK_RATIO,
        max_relative_residual: 0.10,
        gate: Gate::Evidence,
    })
}

/// An edge inside the engine: the layer's overhead budget, with its fixed
/// cost budgeted in units of work where a sweep measures it.
fn engine_layer(fixed_work_equivalent: Option<f64>) -> ExactRules {
    ExactRules {
        overhead_budget: budget::LAYER_OVERHEAD,
        space: budget::SPACE,
        shape: fixed_work_equivalent.and_then(shape),
    }
}

/// The edge from a reference framework: evidence on every axis. A reference
/// that moves with every wheel release is not a merge condition.
fn reference_edge(
    outcome: CrossStackOutcome,
    fixed_work_equivalent: Option<f64>,
    time_to_quality: bool,
) -> CrossStackRules {
    CrossStackRules {
        outcome,
        speed_bar: budget::FRAMEWORK_SPEED_BAR,
        space: budget::SPACE,
        shape: fixed_work_equivalent.and_then(shape),
        time_to_quality_bar: time_to_quality.then_some(budget::FRAMEWORK_SPEED_BAR),
    }
}

const TORCH: &str = "torch";
const PYTORCH: &str = "PyTorch";

/// The reference arm of the how-well decision: the two families the fused
/// kernels are judged against, both live on the checkpoints it runs on.
fn how_well_reference_arm() -> KernelArm {
    KernelArm::off([KernelFamily::FlashAttention, KernelFamily::AdamW])
}

fn train_run_ladder() -> Ladder {
    let learns = || {
        vec![
            LegPremise::ConstantSchedule,
            LegPremise::LearningHappened(super::premise::TrainDirection::Descent),
            LegPremise::TieFractionBelow(0.5),
            LegPremise::PaddedAdmission,
        ]
    };
    let fused = |name| {
        Rung::on_arm(
            name,
            KernelArm::fused(),
            &[KernelFamily::FlashAttention],
            learns(),
        )
    };
    let reference_arm = how_well_reference_arm();
    let mut streamed = fused("streamed");
    streamed.flat_host_memory = Some(budget::STREAMED_BYTES_PER_ROW);
    Ladder {
        workload: Workload::TrainRun,
        reference: Rung::new(TORCH, learns()),
        cross_stack: vec![
            (
                Rung::on_arm(
                    "resident-reference",
                    reference_arm.clone(),
                    &[KernelFamily::AttentionBlock],
                    learns(),
                ),
                Difference::Framework { reference: PYTORCH },
                reference_edge(
                    seeded_loss(Some(TRAIN_RUN_DELTA), None, Gate::Evidence),
                    None,
                    true,
                ),
            ),
            (
                fused("resident"),
                Difference::kernel_arm(&reference_arm, &KernelArm::fused()),
                CrossStackRules {
                    outcome: seeded_loss(
                        Some(TRAIN_RUN_DELTA),
                        Some(ControlRule {
                            take: "lr0",
                            field: "lr",
                            value: 0.0,
                            required_units: 2,
                        }),
                        Gate::Hard,
                    ),
                    speed_bar: budget::KERNEL_SPEED_BAR,
                    space: budget::SPACE,
                    shape: None,
                    time_to_quality_bar: Some(budget::KERNEL_SPEED_BAR),
                },
            ),
        ],
        exact: vec![
            (
                streamed,
                Difference::Layer {
                    name: "the job path: training-set table, streaming loader",
                },
                engine_layer(None),
            ),
            (
                fused("placed"),
                Difference::Layer {
                    name: "the same job as a gang on an executor",
                },
                engine_layer(None),
            ),
        ],
    }
}

/// One optimizer step over a synthetic batch, swept over shapes: the step's
/// cost against PyTorch's, and what every fused kernel together is worth.
fn train_step_ladder() -> Ladder {
    let step_edge = |speed_bar| CrossStackRules {
        outcome: CrossStackOutcome::None,
        speed_bar,
        space: budget::SPACE,
        shape: None,
        time_to_quality_bar: None,
    };
    let reference_arm = KernelArm::all_off();
    Ladder {
        workload: Workload::TrainStep,
        reference: Rung::new(TORCH, vec![]),
        cross_stack: vec![
            (
                Rung::on_arm("reference", reference_arm.clone(), &[], vec![]),
                Difference::Framework { reference: PYTORCH },
                step_edge(budget::FRAMEWORK_SPEED_BAR),
            ),
            (
                Rung::on_arm(
                    "fused",
                    KernelArm::fused(),
                    &[KernelFamily::FlashAttention],
                    vec![],
                ),
                Difference::kernel_arm(&reference_arm, &KernelArm::fused()),
                step_edge(budget::KERNEL_SPEED_BAR),
            ),
        ],
        exact: vec![],
    }
}

fn encode_ladder() -> Ladder {
    let row_cosine = CrossStackOutcome::RowAgreement {
        metric: RowMetric::Cosine,
        gate: Gate::Evidence,
    };
    let layer = |name| Difference::Layer { name };
    Ladder {
        workload: Workload::Encode,
        reference: Rung::new(TORCH, vec![]),
        cross_stack: vec![(
            Rung::new("direct", vec![]),
            Difference::Framework { reference: PYTORCH },
            reference_edge(row_cosine, Some(budget::PLAN_FIXED_WORK), false),
        )],
        exact: vec![
            (
                Rung::new("plan", vec![]),
                layer("a DataFusion plan, one partition"),
                engine_layer(Some(budget::PLAN_FIXED_WORK)),
            ),
            (
                Rung::new("plan-partitioned", vec![]),
                layer("the same plan, N partitions"),
                engine_layer(Some(budget::PLAN_FIXED_WORK)),
            ),
            (
                Rung::new("placed", vec![]),
                layer("the same plan on a Ballista executor"),
                engine_layer(Some(budget::PLACED_FIXED_WORK)),
            ),
        ],
    }
}

fn graph_sample_ladder() -> Ladder {
    let mut sampler = Rung::new("sampler", vec![]);
    sampler.flat_host_memory = Some(budget::SAMPLER_BYTES_PER_EDGE);
    Ladder {
        workload: Workload::GraphSample,
        reference: Rung::new(TORCH, vec![]),
        cross_stack: vec![(
            sampler,
            Difference::Framework {
                reference: "PyTorch Geometric's node2vec random-walk sampler",
            },
            reference_edge(
                CrossStackOutcome::Law {
                    statistic: FitStatistic::LikelihoodRatioG,
                    alpha: 0.001,
                    gate: Gate::Hard,
                },
                Some(budget::GRAPH_FIXED_WORK),
                false,
            ),
        )],
        exact: vec![],
    }
}

fn propagate_ladder() -> Ladder {
    let row_error = CrossStackOutcome::RowAgreement {
        metric: RowMetric::RelativeError,
        gate: Gate::Evidence,
    };
    let layer = |name| Difference::Layer { name };
    Ladder {
        workload: Workload::Propagate,
        reference: Rung::new(TORCH, vec![]),
        cross_stack: vec![
            (
                Rung::new("torch-geometric", vec![]),
                Difference::Framework {
                    reference: "exact propagation by sparse matrix product",
                },
                reference_edge(row_error.clone(), Some(budget::GRAPH_FIXED_WORK), false),
            ),
            (
                Rung::new("plan", vec![]),
                Difference::Framework {
                    reference: "PyTorch Geometric's propagation layer",
                },
                reference_edge(row_error, Some(budget::GRAPH_FIXED_WORK), false),
            ),
        ],
        exact: vec![
            (
                Rung::new("plan-partitioned", vec![]),
                layer("the same plan, N partitions"),
                engine_layer(Some(budget::GRAPH_FIXED_WORK)),
            ),
            (
                Rung::new("placed", vec![]),
                layer("the same plan on a Ballista executor"),
                engine_layer(Some(budget::GRAPH_PLACED_FIXED_WORK)),
            ),
        ],
    }
}

fn predictor_train_run_ladder() -> Ladder {
    let learns = vec![
        LegPremise::ConstantSchedule,
        LegPremise::LearningHappened(super::premise::TrainDirection::Descent),
    ];
    Ladder {
        workload: Workload::PredictorTrainRun,
        reference: Rung::new(TORCH, learns.clone()),
        cross_stack: vec![(
            Rung::new("in-process", learns),
            Difference::Framework { reference: PYTORCH },
            // No mutated build has yet shown what this instrument resolves,
            // so no margin is fixed and no margin claim can be made.
            reference_edge(seeded_loss(None, None, Gate::Evidence), None, true),
        )],
        exact: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::ValueEnum;

    fn ladders() -> impl Iterator<Item = Ladder> {
        Workload::value_variants().iter().map(|w| w.ladder())
    }

    #[test]
    fn every_edge_joins_adjacent_rungs_and_cross_stack_edges_come_first() {
        for ladder in ladders() {
            let names: Vec<&str> = ladder.rungs().map(|r| r.name.as_str()).collect();
            let edges = ladder.edges();
            assert_eq!(edges.len(), names.len() - 1);
            for (i, edge) in edges.iter().enumerate() {
                assert_eq!(
                    (edge.lower().name.as_str(), edge.upper().name.as_str()),
                    (names[i], names[i + 1])
                );
            }
            let first_exact = edges
                .iter()
                .position(|e| matches!(e.kind(), EdgeKind::Exact(_)))
                .unwrap_or(edges.len());
            assert!(edges[first_exact..]
                .iter()
                .all(|e| matches!(e.kind(), EdgeKind::Exact(_))));
        }
    }

    /// A cross-stack edge differs by a framework or a kernel arm, an exact
    /// edge by a layer; a revision edge is never in a ladder.
    #[test]
    fn each_edge_kind_carries_the_difference_it_can_judge() {
        for ladder in ladders() {
            for edge in ladder.edges() {
                match (edge.kind(), edge.difference()) {
                    (EdgeKind::CrossStack(_), Difference::Framework { .. })
                    | (EdgeKind::CrossStack(_), Difference::KernelArm { .. })
                    | (EdgeKind::Exact(_), Difference::Layer { .. }) => {}
                    (kind, difference) => panic!("{}: {kind:?} with {difference:?}", edge.name()),
                }
            }
        }
    }

    #[test]
    fn a_kernel_arm_difference_names_the_families_switched_on() {
        let ladder = Workload::TrainRun.ladder();
        let edges = ladder.edges();
        assert_eq!(
            *edges[1].difference(),
            Difference::KernelArm {
                families_on: vec![KernelFamily::FlashAttention, KernelFamily::AdamW],
            }
        );
        let step = Workload::TrainStep.ladder();
        let Difference::KernelArm { families_on } = step.edges()[1].difference() else {
            panic!("the kernel edge of train-step differs by a kernel arm");
        };
        assert_eq!(families_on.len(), KernelFamily::ALL.len());
    }

    #[test]
    fn a_revision_edge_files_its_legs_under_side_tags() {
        let ladder = Workload::Encode.ladder();
        let rules = Workload::Encode.revision_rules();
        let edge = Edge::revision(ladder.rung("direct").unwrap(), &rules);
        assert_eq!(edge.name(), "direct@base -> direct@revised");
        assert!(edge.is_revision());
        assert_eq!(*edge.difference(), Difference::Revision);
    }

    #[test]
    fn rung_names_are_unique_and_safe_in_a_leg_file_name() {
        for ladder in ladders() {
            let names: Vec<&str> = ladder.rungs().map(|r| r.name.as_str()).collect();
            let unique: std::collections::BTreeSet<&str> = names.iter().copied().collect();
            assert_eq!(unique.len(), names.len());
            assert!(names
                .iter()
                .all(|n| !n.contains("__") && !n.contains('@') && !n.starts_with("mutant-")));
        }
    }

    #[test]
    fn a_span_is_contiguous_and_refuses_unknown_or_empty_ranges() {
        let ladder = Workload::TrainRun.ladder();
        let span = ladder
            .span(Some("resident-reference"), Some("resident"))
            .unwrap();
        assert_eq!(span.len(), 1);
        assert_eq!(span[0].name(), "resident-reference -> resident");
        assert_eq!(ladder.span(None, None).unwrap().len(), 4);
        assert!(matches!(
            ladder.span(Some("nope"), None),
            Err(Refusal::UnknownRung { .. })
        ));
        assert!(matches!(
            ladder.span(Some("resident"), Some("torch")),
            Err(Refusal::EmptySpan { .. })
        ));
    }

    #[test]
    fn ladders_are_as_long_as_their_workload_needs() {
        let lengths: Vec<usize> = ladders().map(|l| l.rungs().count()).collect();
        assert_eq!(lengths, [5, 3, 5, 2, 5, 2]);
    }

    #[test]
    fn swept_fields_are_identity_fields_and_tier_keys_are_distinct() {
        let mut keys = std::collections::BTreeSet::new();
        for workload in Workload::value_variants() {
            assert!(keys.insert(workload.tier_key()));
            for swept in workload.swept_fields() {
                assert!(
                    workload.identity_fields().iter().any(|(f, _)| f == swept),
                    "{swept}"
                );
            }
        }
    }

    /// Only measured rules gate: the seeded outcome's margin (measured by the
    /// dose ladder), digest equality, the law, and a revision's own noise
    /// band. Every invented budget is evidence.
    #[test]
    fn only_rules_with_a_measurement_behind_them_are_hard() {
        for ladder in ladders() {
            for edge in ladder.edges() {
                match edge.kind() {
                    EdgeKind::CrossStack(rules) => {
                        assert_eq!(rules.speed_bar.gate, Gate::Evidence, "{}", edge.name());
                        assert_eq!(rules.space.host_ratio.gate, Gate::Evidence);
                        assert_eq!(rules.space.device_ratio.gate, Gate::Evidence);
                        assert!(rules.shape.is_none_or(|s| s.gate == Gate::Evidence));
                        assert!(rules
                            .time_to_quality_bar
                            .is_none_or(|b| b.gate == Gate::Evidence));
                    }
                    EdgeKind::Exact(rules) => {
                        assert_eq!(rules.overhead_budget.gate, Gate::Evidence);
                        assert!(rules.shape.is_none_or(|s| s.gate == Gate::Evidence));
                    }
                    EdgeKind::Revision(_) => unreachable!(),
                }
                assert!(edge
                    .upper()
                    .flat_host_memory
                    .is_none_or(|b| b.gate == Gate::Evidence));
            }
        }
    }
}
