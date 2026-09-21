//! The ladder as data: workloads, rungs in order, and the rules of each
//! edge.
//!
//! An edge is never constructed from two free rung names. A [`Ladder`] holds
//! a reference rung, then the rungs reached by a *cross-stack* edge, then the
//! rungs reached by an *exact* edge, each list as long as the workload needs;
//! [`Ladder::edges`] derives each edge from a rung and the one below it. Two
//! properties therefore hold for every ladder that can be written down: an
//! edge always joins adjacent rungs, and no cross-stack edge ever sits above
//! an exact one — equality composes upward without tolerance only if nothing
//! above it reintroduces a margin.

use serde::Serialize;

use jammi_numerics::stats::FitStatistic;

use crate::report::{EncodeStepTier, FinetuneRunTier, Nullable};

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
            Self::Encode => EncodeStepTier::IDENTITY_FIELDS,
            Self::TrainRun => FinetuneRunTier::IDENTITY_FIELDS,
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
            Self::TrainRun => train_run_ladder(),
            Self::GraphSample => graph_sample_ladder(),
            Self::Propagate => propagate_ladder(),
            Self::PredictorTrainRun => predictor_train_run_ladder(),
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

const fn hard<T>(bound: T) -> Judged<T> {
    Judged {
        bound,
        gate: Gate::Hard,
    }
}

const fn evidence<T>(bound: T) -> Judged<T> {
    Judged {
        bound,
        gate: Gate::Evidence,
    }
}

/// One implementation stack of a workload.
#[derive(Debug, Clone)]
pub struct Rung {
    pub name: String,
    /// The one layer this rung adds over the rung below it.
    pub layer: &'static str,
    /// Facts every leg of this rung must show about itself.
    pub premises: Vec<LegPremise>,
    /// Set when this rung's host memory must not grow with input size: the
    /// budget on the fitted slope, in bytes per row.
    pub flat_host_memory: Option<Judged<f64>>,
}

impl Rung {
    fn new(name: &str, layer: &'static str, premises: Vec<LegPremise>) -> Self {
        Self {
            name: name.to_owned(),
            layer,
            premises,
            flat_host_memory: None,
        }
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

#[derive(Debug, Clone, Copy)]
pub enum EdgeKind<'a> {
    CrossStack(&'a CrossStackRules),
    Exact(&'a ExactRules),
}

/// An adjacent pair of rungs and the rules between them. Only
/// [`Ladder::edges`] makes one from nothing; [`Edge::with_upper`] re-aims an
/// existing one.
#[derive(Debug, Clone, Copy)]
pub struct Edge<'a> {
    lower: &'a Rung,
    upper: &'a Rung,
    kind: EdgeKind<'a>,
}

impl<'a> Edge<'a> {
    pub fn lower(&self) -> &'a Rung {
        self.lower
    }

    pub fn upper(&self) -> &'a Rung {
        self.upper
    }

    pub fn kind(&self) -> EdgeKind<'a> {
        self.kind
    }

    pub fn name(&self) -> String {
        format!("{} -> {}", self.lower.name, self.upper.name)
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
    cross_stack: Vec<(Rung, CrossStackRules)>,
    exact: Vec<(Rung, ExactRules)>,
}

impl Ladder {
    pub fn rungs(&self) -> impl Iterator<Item = &Rung> {
        std::iter::once(&self.reference)
            .chain(self.cross_stack.iter().map(|(r, _)| r))
            .chain(self.exact.iter().map(|(r, _)| r))
    }

    /// Every edge, bottom to top.
    pub fn edges(&self) -> Vec<Edge<'_>> {
        let uppers = self
            .cross_stack
            .iter()
            .map(|(r, rules)| (r, EdgeKind::CrossStack(rules)))
            .chain(
                self.exact
                    .iter()
                    .map(|(r, rules)| (r, EdgeKind::Exact(rules))),
            );
        self.rungs()
            .zip(uppers)
            .map(|(lower, (upper, kind))| Edge { lower, upper, kind })
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

const SPACE_EVIDENCE: SpaceRules = SpaceRules {
    host_ratio: evidence(1.10),
    device_ratio: evidence(1.10),
};

const SPACE_HARD: SpaceRules = SpaceRules {
    host_ratio: hard(1.10),
    device_ratio: hard(1.10),
};

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

fn shape(fixed_work_equivalent: f64, gate: Gate) -> Option<ShapeRules> {
    Some(ShapeRules {
        fixed_work_equivalent,
        per_work_ratio: 1.10,
        max_relative_residual: 0.10,
        gate,
    })
}

/// An edge inside the engine: a 10% overhead budget, hard, with the layer's
/// fixed cost budgeted in units of work.
fn engine_layer(fixed_work_equivalent: Option<f64>) -> ExactRules {
    ExactRules {
        overhead_budget: hard(1.10),
        space: SPACE_HARD,
        shape: fixed_work_equivalent.and_then(|fixed| shape(fixed, Gate::Hard)),
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
        speed_bar: evidence(0.9),
        space: SPACE_EVIDENCE,
        shape: fixed_work_equivalent.and_then(|fixed| shape(fixed, Gate::Evidence)),
        time_to_quality_bar: time_to_quality.then_some(evidence(0.9)),
    }
}

fn train_run_ladder() -> Ladder {
    use LegPremise::*;
    let learns = || {
        vec![
            ConstantSchedule,
            LearningHappened(super::premise::TrainDirection::Descent),
            TieFractionBelow(0.5),
        ]
    };
    let with = |mut base: Vec<LegPremise>, extra: &[LegPremise]| {
        base.extend_from_slice(extra);
        base
    };
    let fused = || with(learns(), &[PaddedAdmission, Arm("fused"), FusedDispatch]);
    let mut streamed = Rung::new(
        "streamed",
        "the job path: training-set table, streaming loader",
        fused(),
    );
    // A streaming loader may keep an offset per row; it may not keep the row.
    streamed.flat_host_memory = Some(hard(16.0));
    Ladder {
        workload: Workload::TrainRun,
        reference: Rung::new("torch", "the PyTorch twin", learns()),
        cross_stack: vec![
            (
                Rung::new(
                    "resident-reference",
                    "candle and the trainer over in-memory rows, reference kernels",
                    with(
                        learns(),
                        &[PaddedAdmission, Arm("alloff"), ReferenceDispatch],
                    ),
                ),
                reference_edge(
                    seeded_loss(Some(TRAIN_RUN_DELTA), None, Gate::Evidence),
                    None,
                    true,
                ),
            ),
            (
                Rung::new("resident", "the fused kernels", fused()),
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
                    speed_bar: evidence(1.0),
                    space: SPACE_EVIDENCE,
                    shape: None,
                    time_to_quality_bar: Some(evidence(1.0)),
                },
            ),
        ],
        exact: vec![
            (streamed, engine_layer(None)),
            (
                Rung::new("placed", "the same job as a gang on an executor", fused()),
                engine_layer(None),
            ),
        ],
    }
}

fn encode_ladder() -> Ladder {
    let row_cosine = CrossStackOutcome::RowAgreement {
        metric: RowMetric::Cosine,
        gate: Gate::Evidence,
    };
    Ladder {
        workload: Workload::Encode,
        reference: Rung::new("torch", "the PyTorch twin", vec![]),
        cross_stack: vec![(
            Rung::new(
                "direct",
                "the loaded model called on the same texts, no plan",
                vec![],
            ),
            reference_edge(row_cosine, Some(64.0), false),
        )],
        exact: vec![
            (
                Rung::new("plan", "a DataFusion plan, one partition", vec![]),
                engine_layer(Some(64.0)),
            ),
            (
                Rung::new("plan-partitioned", "the same plan, N partitions", vec![]),
                engine_layer(Some(64.0)),
            ),
            (
                Rung::new("placed", "the same plan on a Ballista executor", vec![]),
                engine_layer(Some(256.0)),
            ),
        ],
    }
}

fn graph_sample_ladder() -> Ladder {
    let mut sampler = Rung::new(
        "sampler",
        "the engine's second-order random-walk pair sampler",
        vec![],
    );
    // Walks hold the adjacency, which is the input; nothing per edge beyond it.
    sampler.flat_host_memory = Some(evidence(256.0));
    Ladder {
        workload: Workload::GraphSample,
        reference: Rung::new(
            "torch",
            "PyTorch Geometric's node2vec random-walk sampler",
            vec![],
        ),
        cross_stack: vec![(
            sampler,
            reference_edge(
                CrossStackOutcome::Law {
                    statistic: FitStatistic::LikelihoodRatioG,
                    alpha: 0.001,
                    gate: Gate::Hard,
                },
                Some(4096.0),
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
    Ladder {
        workload: Workload::Propagate,
        reference: Rung::new(
            "torch",
            "exact propagation by sparse matrix product",
            vec![],
        ),
        cross_stack: vec![
            (
                Rung::new(
                    "torch-geometric",
                    "PyTorch Geometric's propagation layer: the practical bar",
                    vec![],
                ),
                reference_edge(row_error.clone(), Some(4096.0), false),
            ),
            (
                Rung::new("plan", "the engine's propagation, one partition", vec![]),
                reference_edge(row_error, Some(4096.0), false),
            ),
        ],
        exact: vec![
            (
                Rung::new("plan-partitioned", "the same plan, N partitions", vec![]),
                engine_layer(Some(4096.0)),
            ),
            (
                Rung::new("placed", "the same plan on a Ballista executor", vec![]),
                engine_layer(Some(16384.0)),
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
        reference: Rung::new("torch", "the PyTorch twin", learns.clone()),
        cross_stack: vec![(
            Rung::new("in-process", "candle and the predictor trainer", learns),
            // No mutated build has yet shown what this instrument resolves,
            // so no margin is fixed and no parity claim can be made.
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

    #[test]
    fn rung_names_are_unique_and_safe_in_a_leg_file_name() {
        for ladder in ladders() {
            let names: Vec<&str> = ladder.rungs().map(|r| r.name.as_str()).collect();
            let unique: std::collections::BTreeSet<&str> = names.iter().copied().collect();
            assert_eq!(unique.len(), names.len());
            assert!(names
                .iter()
                .all(|n| !n.contains("__") && !n.starts_with("mutant-")));
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
        assert_eq!(lengths, [5, 5, 2, 5, 2]);
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
}
