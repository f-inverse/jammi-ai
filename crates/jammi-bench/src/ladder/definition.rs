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
//! No bound in a ladder is invented. Every budget — a speed bar, an overhead,
//! a memory ratio, a fixed-cost work equivalent, a bytes-per-row slope — is a
//! *measured* value a committed artifact supplies through [`Budgets`]; a rule
//! whose budget no artifact has measured is reported unbudgeted, never
//! judged against a number nobody measured. The bounds that are not budgets
//! are derived from the run itself (a margin from the reference rung's own
//! learning effect, a noise band from a rung's repeats) or are pre-fixed
//! statistical levels.

use serde::{Deserialize, Serialize};

use jammi_numerics::stats::FitStatistic;

use crate::context_predictor::PredictorTrainRunPayload;
use crate::graph_sample::GraphSamplePayload;
use crate::kernel_arm::{KernelArm, KernelFamily};
use crate::leg::Payload;
use crate::propagate::PropagatePayload;
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
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

    /// What two legs must agree on to be comparable: the declaration the
    /// workload's producer asserts on every emit, and nothing else.
    pub fn identity_fields(self) -> &'static [(&'static str, Nullable)] {
        match self {
            Self::Encode => EncodePayload::IDENTITY_FIELDS,
            Self::TrainStep => TrainStepPayload::IDENTITY_FIELDS,
            Self::TrainRun => TrainRunPayload::IDENTITY_FIELDS,
            Self::GraphSample => GraphSamplePayload::IDENTITY_FIELDS,
            Self::Propagate => PropagatePayload::IDENTITY_FIELDS,
            Self::PredictorTrainRun => PredictorTrainRunPayload::IDENTITY_FIELDS,
        }
    }

    /// The identity fields that name a unit — the axis a sweep varies — and
    /// so differ between units by construction while agreeing within one.
    pub fn swept_fields(self) -> &'static [&'static str] {
        match self {
            Self::Encode => &["rows", "corpus_sha256", "token_lengths_sha256", "tokens"],
            Self::TrainStep => &["batch", "seq", "row_lengths", "lora_dropout"],
            Self::TrainRun => &["seed"],
            // The seed fixes the task split and the initial weights, so the
            // digests of both are the unit's, never the edge's.
            Self::PredictorTrainRun => &[
                "seed",
                "initial_weights_sha256",
                "train_episodes_sha256",
                "heldout_episodes_sha256",
            ],
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

    /// This workload's ladder, its rules bounded by `budgets`.
    pub fn ladder_with(self, budgets: &Budgets) -> Ladder {
        match self {
            Self::Encode => encode_ladder(budgets),
            Self::TrainStep => train_step_ladder(budgets),
            Self::TrainRun => train_run_ladder(budgets),
            Self::GraphSample => graph_sample_ladder(budgets),
            Self::Propagate => propagate_ladder(budgets),
            Self::PredictorTrainRun => predictor_train_run_ladder(budgets),
        }
    }

    /// The rules of a revision edge of `rung` of this workload.
    pub fn revision_rules(self, rung: &str, budgets: &Budgets) -> RevisionRules {
        let edge = EdgeRules::of(
            budgets,
            self,
            &format!("{rung}@{REVISION_BASE}"),
            &format!("{rung}@{REVISION_REVISED}"),
        );
        RevisionRules {
            within_noise_band: RuleForce::Hard,
            space: edge.space(),
        }
    }
}

/// Whether a rule's failure is a verdict — it fails the run — or evidence,
/// reported beside the verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleForce {
    Hard,
    Evidence,
}

/// A measured bound and the committed artifact that measured it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Budget {
    pub bound: f64,
    /// Repository path of the artifact the bound was read from.
    pub measured_from: String,
}

/// A rule's bound and how much its failure counts. `budget: None` is a rule
/// nobody has measured a bound for: it is reported unbudgeted.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Rule {
    pub force: RuleForce,
    pub budget: Option<Budget>,
}

/// One budget of the committed table: which rule of which edge of which
/// workload it bounds.
#[derive(Debug, Clone, Deserialize)]
pub struct BudgetEntry {
    pub workload: Workload,
    /// The edge as [`Edge::name`] prints it, or a rung name for a rung's own
    /// rule (`host_memory_flat_in_work`).
    pub edge: String,
    pub rule: String,
    #[serde(flatten)]
    pub budget: Budget,
}

/// The committed budgets: every measured bound any ladder judges against.
/// The table is `crates/jammi-bench/budgets.json`; each entry names the
/// artifact its bound was read from.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Budgets {
    pub budgets: Vec<BudgetEntry>,
}

const COMMITTED_BUDGETS: &str = include_str!("../../budgets.json");

impl Budgets {
    pub fn committed() -> Self {
        serde_json::from_str(COMMITTED_BUDGETS).expect("crates/jammi-bench/budgets.json parses")
    }

    fn lookup(&self, workload: Workload, edge: &str, rule: &str) -> Option<Budget> {
        self.budgets
            .iter()
            .find(|b| b.workload == workload && b.edge == edge && b.rule == rule)
            .map(|b| b.budget.clone())
    }

    /// The rule `rule` of edge `edge`, at `force`, with whatever bound the
    /// table measured for it.
    fn rule(&self, workload: Workload, edge: &str, rule: &str, force: RuleForce) -> Rule {
        Rule {
            force,
            budget: self.lookup(workload, edge, rule),
        }
    }
}

/// Rule names, as the verdict prints them and the budgets table keys them.
pub mod rule {
    pub const SPEED_NON_INFERIORITY: &str = "speed_non_inferiority";
    pub const OVERHEAD_BUDGET: &str = "overhead_budget";
    pub const HOST_MEMORY_RATIO: &str = "host_memory_ratio";
    pub const DEVICE_MEMORY_RATIO: &str = "device_memory_ratio";
    pub const HOST_MEMORY_FLAT_IN_WORK: &str = "host_memory_flat_in_work";
    pub const FIXED_COST: &str = "fixed_cost";
    pub const PER_WORK_COST: &str = "per_work_cost";
    pub const TIME_TO_QUALITY: &str = "time_to_quality";
    pub const GRADIENT_COSINE_FLOOR: &str = "gradient_cosine_floor";
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
    /// rule on the fitted slope, in bytes per row.
    pub flat_host_memory: Option<Rule>,
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
#[derive(Debug, Clone, Serialize)]
pub struct SpaceRules {
    pub host_ratio: Rule,
    pub device_ratio: Rule,
}

/// Budgets on the two coefficients of `time = fixed + per_work · work`,
/// where a leg's `work` is the size its cost scales with: rows for `encode`,
/// edges × hops for `propagate`, edges for `graph-sample`.
///
/// The fixed budget is in units of work: the layer's added fixed cost divided
/// by the lower rung's per-work cost — how much work the layer's constant
/// overhead is worth. Both budgets are dimensionless, so neither depends on
/// the speed of the box that measured them.
#[derive(Debug, Clone, Serialize)]
pub struct ShapeRules {
    pub fixed_work_equivalent: Rule,
    pub per_work_ratio: Rule,
    /// Refuse the fit when the residual exceeds this fraction of the mean
    /// fitted time: the two coefficients only mean anything on a line.
    pub max_relative_residual: f64,
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

/// The evaluation point a seeded edge is read at, fixed before any leg runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgedPoint {
    /// Per unit, the epoch at which the lower rung's held-out loss is lowest:
    /// where the reference learned the most, never the final epoch of a run
    /// that may already be overfitting.
    ReferenceMinimum,
}

/// The fraction of the reference rung's learning effect the upper rung must
/// preserve: the non-inferiority margin is `δ = PRESERVED_EFFECT_FRACTION ·
/// M1`, where `M1` is the lower confidence bound of the reference's mean
/// improvement from its untrained held-out loss to the judged point. The
/// construction — a margin `M2` fixed as a fraction of the active control's
/// established effect `M1`, so the test stack is shown to keep at least that
/// fraction of the effect — is the one in FDA, *Non-Inferiority Clinical
/// Trials to Establish Effectiveness* (2016), §III; one half is its worked
/// example and the fraction commonly chosen.
///
/// Assay sensitivity — the same guidance's requirement that the active
/// control's effect be shown in the trial at hand, not assumed — is the
/// premise `M1 > 0`: the reference's improvement, read from the legs of this
/// very session, has an interval that excludes zero. No separate multiple
/// of `δ` states it: `δ` is a fraction of `M1`, so "improved by at least
/// `δ / PRESERVED_EFFECT_FRACTION`" *is* "the established effect is
/// positive", and a reference that does not establish one leaves no margin
/// to derive.
pub const PRESERVED_EFFECT_FRACTION: f64 = 0.5;

/// The margin of a seeded edge, and how its two one-sided tests count.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Margin {
    /// Fraction of the reference's learning effect the upper rung keeps.
    pub preserved_effect_fraction: f64,
    /// Level of each one-sided test, and of the interval `M1` is read from.
    pub alpha: f64,
    /// The claim: the upper rung's loss at the judged point is no worse than
    /// the lower's by more than `δ`. Two-sided equivalence is reported beside
    /// it as evidence — an upper rung better by more than `δ` has not failed
    /// "as good as".
    pub non_inferiority_force: RuleForce,
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
        direction_force: RuleForce,
        /// Where along the run the two rungs are read.
        judged_at: JudgedPoint,
        /// The non-inferiority margin, derived from the reference rung's own
        /// learning effect.
        margin: Margin,
        control: Option<ControlRule>,
    },
    /// Paired by row: deterministic math on the same keyed input, one vector
    /// per row on each side.
    RowAgreement { metric: RowMetric, force: RuleForce },
    /// Not pairable: the two stacks draw from different random streams, so
    /// neither their outputs nor any per-unit difference of them can be
    /// compared. Each rung's output is instead tested against the workload's
    /// analytic law, and both must fit it.
    Law {
        statistic: FitStatistic,
        /// Level of each rung's goodness-of-fit test, fixed here: a faithful
        /// sampler is refused with exactly this probability.
        alpha: f64,
        force: RuleForce,
    },
    /// Paired by tensor at shared weights: one forward and backward on each
    /// stack from the same loaded adapter over the same batch, read off the
    /// edge's `take` legs. Per tensor, both gradients zero is vacuous (no
    /// evidence either way — `dL/dA` is structurally zero at a zero `B`);
    /// exactly one zero, or a non-finite entry, breaks the structure;
    /// otherwise the tensor's cosine must clear the floor.
    GradientAgreement {
        take: &'static str,
        cosine_floor: Rule,
        /// How a structural break — a one-sided zero, a non-finite gradient,
        /// a tensor one side lacks, weights that differ — counts.
        structure: RuleForce,
    },
    /// The artifact is a cost, not a result: a step over synthetic inputs
    /// has no outcome to compare.
    None,
}

/// The take gradient legs are filed under.
pub const GRADIENTS_TAKE: &str = "grads";

/// The identity fields a gradient take leg is free to differ on from the
/// measured repeats: one forward and backward at loaded weights has no
/// warmup, no measured steps and no clip. Dropout is not among them — it
/// names the unit, and a gradient leg runs with dropout off, so it is filed
/// under a unit whose dropout is zero.
pub const GRADIENTS_TAKE_FREE_FIELDS: &[&str] = &["warmup", "steps_measured", "max_grad_norm"];

/// Rules of an edge between two stacks — two frameworks, or two kernel sets.
#[derive(Debug, Clone, Serialize)]
pub struct CrossStackRules {
    pub outcome: CrossStackOutcome,
    /// Non-inferiority: the lower bound of `lower ÷ upper` time must exceed
    /// this bar.
    pub speed_bar: Rule,
    pub space: SpaceRules,
    pub shape: Option<ShapeRules>,
    /// Non-inferiority bar on `lower ÷ upper` time-to-quality.
    pub time_to_quality_bar: Option<Rule>,
}

/// Rules of an edge inside one deterministic engine: the outcome is digest
/// equality, and the speed rule is the layer's overhead budget.
#[derive(Debug, Clone, Serialize)]
pub struct ExactRules {
    /// The upper bound of `upper ÷ lower` time must stay under this budget.
    pub overhead_budget: Rule,
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
    pub within_noise_band: RuleForce,
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

/// A seeded-loss outcome whose two claims — no directional difference, and
/// the upper rung no worse than the reference by more than the margin —
/// count as `force`.
fn seeded_loss(control: Option<ControlRule>, force: RuleForce) -> CrossStackOutcome {
    CrossStackOutcome::SeededLoss {
        seeds: 12,
        sign_alpha: 0.0064,
        direction_force: force,
        judged_at: JudgedPoint::ReferenceMinimum,
        margin: Margin {
            preserved_effect_fraction: PRESERVED_EFFECT_FRACTION,
            alpha: 0.05,
            non_inferiority_force: force,
        },
        control,
    }
}

/// The rules of one edge, looked up in the committed budgets.
struct EdgeRules<'a> {
    budgets: &'a Budgets,
    workload: Workload,
    edge: String,
}

impl<'a> EdgeRules<'a> {
    fn of(budgets: &'a Budgets, workload: Workload, lower: &str, upper: &str) -> Self {
        Self {
            budgets,
            workload,
            edge: format!("{lower} -> {upper}"),
        }
    }

    fn rule(&self, name: &str, force: RuleForce) -> Rule {
        self.budgets.rule(self.workload, &self.edge, name, force)
    }

    fn space(&self) -> SpaceRules {
        SpaceRules {
            host_ratio: self.rule(rule::HOST_MEMORY_RATIO, RuleForce::Evidence),
            device_ratio: self.rule(rule::DEVICE_MEMORY_RATIO, RuleForce::Evidence),
        }
    }

    /// The size sweep's two coefficients, where a sweep measures them.
    fn shape(&self, swept: bool) -> Option<ShapeRules> {
        swept.then(|| ShapeRules {
            fixed_work_equivalent: self.rule(rule::FIXED_COST, RuleForce::Evidence),
            per_work_ratio: self.rule(rule::PER_WORK_COST, RuleForce::Evidence),
            max_relative_residual: 0.10,
        })
    }

    /// An edge inside the engine: the layer's overhead budget, with its fixed
    /// cost budgeted in units of work where a sweep measures it.
    fn engine_layer(&self, swept: bool) -> ExactRules {
        ExactRules {
            overhead_budget: self.rule(rule::OVERHEAD_BUDGET, RuleForce::Evidence),
            space: self.space(),
            shape: self.shape(swept),
        }
    }

    /// An edge between two stacks: evidence on every budgeted axis. A
    /// reference that moves with every wheel release is not a merge
    /// condition.
    fn cross_stack(
        &self,
        outcome: CrossStackOutcome,
        swept: bool,
        time_to_quality: bool,
    ) -> CrossStackRules {
        CrossStackRules {
            outcome,
            speed_bar: self.rule(rule::SPEED_NON_INFERIORITY, RuleForce::Evidence),
            space: self.space(),
            shape: self.shape(swept),
            time_to_quality_bar: time_to_quality
                .then(|| self.rule(rule::TIME_TO_QUALITY, RuleForce::Evidence)),
        }
    }

    /// A rung's own slope rule over a size sweep.
    fn flat_host_memory(budgets: &Budgets, workload: Workload, rung: &str) -> Rule {
        budgets.rule(
            workload,
            rung,
            rule::HOST_MEMORY_FLAT_IN_WORK,
            RuleForce::Evidence,
        )
    }
}

const TORCH: &str = "torch";
const PYTORCH: &str = "PyTorch";

/// The reference arm of the how-well decision: the two families the fused
/// kernels are judged against, both live on the checkpoints it runs on.
fn how_well_reference_arm() -> KernelArm {
    KernelArm::off([KernelFamily::FlashAttention, KernelFamily::AdamW])
}

fn train_run_ladder(budgets: &Budgets) -> Ladder {
    let w = Workload::TrainRun;
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
    streamed.flat_host_memory = Some(EdgeRules::flat_host_memory(budgets, w, "streamed"));
    Ladder {
        workload: w,
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
                EdgeRules::of(budgets, w, TORCH, "resident-reference").cross_stack(
                    seeded_loss(None, RuleForce::Evidence),
                    false,
                    true,
                ),
            ),
            (
                fused("resident"),
                Difference::kernel_arm(&reference_arm, &KernelArm::fused()),
                EdgeRules::of(budgets, w, "resident-reference", "resident").cross_stack(
                    seeded_loss(
                        Some(ControlRule {
                            take: "lr0",
                            field: "lr",
                            value: 0.0,
                            required_units: 2,
                        }),
                        RuleForce::Hard,
                    ),
                    false,
                    true,
                ),
            ),
        ],
        exact: vec![
            (
                streamed,
                Difference::Layer {
                    name: "the job path: training-set table, streaming loader",
                },
                EdgeRules::of(budgets, w, "resident", "streamed").engine_layer(false),
            ),
            (
                fused("placed"),
                Difference::Layer {
                    name: "the same job as a gang on an executor",
                },
                EdgeRules::of(budgets, w, "streamed", "placed").engine_layer(false),
            ),
            (
                fused("shape-d"),
                Difference::Layer {
                    name: "the deployed topology: the job through the query tier, on a compute process",
                },
                EdgeRules::of(budgets, w, "placed", "shape-d").engine_layer(false),
            ),
        ],
    }
}

/// One optimizer step over a synthetic batch, swept over shapes: the step's
/// cost against PyTorch's, and what every fused kernel together is worth.
/// The outcome of the framework edge is gradient agreement at shared
/// weights, read off the edge's `grads` take.
fn train_step_ladder(budgets: &Budgets) -> Ladder {
    let w = Workload::TrainStep;
    let reference_arm = KernelArm::all_off();
    let torch_edge = EdgeRules::of(budgets, w, TORCH, "reference");
    Ladder {
        workload: w,
        reference: Rung::new(TORCH, vec![]),
        cross_stack: vec![
            (
                Rung::on_arm("reference", reference_arm.clone(), &[], vec![]),
                Difference::Framework { reference: PYTORCH },
                torch_edge.cross_stack(
                    CrossStackOutcome::GradientAgreement {
                        take: GRADIENTS_TAKE,
                        cosine_floor: torch_edge
                            .rule(rule::GRADIENT_COSINE_FLOOR, RuleForce::Evidence),
                        structure: RuleForce::Hard,
                    },
                    false,
                    false,
                ),
            ),
            (
                Rung::on_arm(
                    "fused",
                    KernelArm::fused(),
                    &[KernelFamily::FlashAttention],
                    vec![],
                ),
                Difference::kernel_arm(&reference_arm, &KernelArm::fused()),
                EdgeRules::of(budgets, w, "reference", "fused").cross_stack(
                    CrossStackOutcome::None,
                    false,
                    false,
                ),
            ),
        ],
        exact: vec![],
    }
}

fn encode_ladder(budgets: &Budgets) -> Ladder {
    let w = Workload::Encode;
    let row_cosine = CrossStackOutcome::RowAgreement {
        metric: RowMetric::Cosine,
        force: RuleForce::Evidence,
    };
    let layer = |name| Difference::Layer { name };
    let exact = |lower, upper| EdgeRules::of(budgets, w, lower, upper).engine_layer(true);
    Ladder {
        workload: w,
        reference: Rung::new(TORCH, vec![]),
        cross_stack: vec![(
            Rung::new("direct", vec![]),
            Difference::Framework { reference: PYTORCH },
            EdgeRules::of(budgets, w, TORCH, "direct").cross_stack(row_cosine, true, false),
        )],
        exact: vec![
            (
                Rung::new("plan", vec![]),
                layer("a DataFusion plan, one partition"),
                exact("direct", "plan"),
            ),
            (
                Rung::new("plan-partitioned", vec![]),
                layer("the same plan, N partitions"),
                exact("plan", "plan-partitioned"),
            ),
            (
                Rung::new("placed", vec![]),
                layer("the same plan on a Ballista executor"),
                exact("plan-partitioned", "placed"),
            ),
            (
                Rung::new("shape-d", vec![]),
                layer(
                    "the deployed topology: the serve through the query tier, on a compute process",
                ),
                exact("placed", "shape-d"),
            ),
        ],
    }
}

fn graph_sample_ladder(budgets: &Budgets) -> Ladder {
    let w = Workload::GraphSample;
    let mut sampler = Rung::new("sampler", vec![]);
    sampler.flat_host_memory = Some(EdgeRules::flat_host_memory(budgets, w, "sampler"));
    Ladder {
        workload: w,
        reference: Rung::new(TORCH, vec![]),
        cross_stack: vec![(
            sampler,
            Difference::Framework {
                reference: "PyTorch Geometric's node2vec random-walk sampler",
            },
            EdgeRules::of(budgets, w, TORCH, "sampler").cross_stack(
                CrossStackOutcome::Law {
                    statistic: FitStatistic::LikelihoodRatioG,
                    alpha: 0.001,
                    force: RuleForce::Hard,
                },
                true,
                false,
            ),
        )],
        exact: vec![],
    }
}

fn propagate_ladder(budgets: &Budgets) -> Ladder {
    let w = Workload::Propagate;
    let row_error = || CrossStackOutcome::RowAgreement {
        metric: RowMetric::RelativeError,
        force: RuleForce::Evidence,
    };
    let layer = |name| Difference::Layer { name };
    Ladder {
        workload: w,
        reference: Rung::new(TORCH, vec![]),
        cross_stack: vec![
            (
                Rung::new("torch-geometric", vec![]),
                Difference::Framework {
                    reference: "exact propagation by sparse matrix product",
                },
                EdgeRules::of(budgets, w, TORCH, "torch-geometric").cross_stack(
                    row_error(),
                    true,
                    false,
                ),
            ),
            (
                Rung::new("plan", vec![]),
                Difference::Framework {
                    reference: "PyTorch Geometric's propagation layer",
                },
                EdgeRules::of(budgets, w, "torch-geometric", "plan").cross_stack(
                    row_error(),
                    true,
                    false,
                ),
            ),
        ],
        exact: vec![
            (
                Rung::new("plan-partitioned", vec![]),
                layer("the same plan, N partitions"),
                EdgeRules::of(budgets, w, "plan", "plan-partitioned").engine_layer(true),
            ),
            (
                Rung::new("placed", vec![]),
                layer("the same plan on a Ballista executor"),
                EdgeRules::of(budgets, w, "plan-partitioned", "placed").engine_layer(true),
            ),
        ],
    }
}

fn predictor_train_run_ladder(budgets: &Budgets) -> Ladder {
    let w = Workload::PredictorTrainRun;
    let learns = vec![
        LegPremise::ConstantSchedule,
        LegPremise::LearningHappened(super::premise::TrainDirection::Descent),
    ];
    Ladder {
        workload: w,
        reference: Rung::new(TORCH, learns.clone()),
        cross_stack: vec![(
            Rung::new("in-process", learns.clone()),
            Difference::Framework { reference: PYTORCH },
            EdgeRules::of(budgets, w, TORCH, "in-process").cross_stack(
                seeded_loss(None, RuleForce::Evidence),
                false,
                true,
            ),
        )],
        exact: vec![
            (
                Rung::new("placed", learns.clone()),
                Difference::Layer {
                    name: "the same training as a job, placed on an executor",
                },
                EdgeRules::of(budgets, w, "in-process", "placed").engine_layer(false),
            ),
            (
                Rung::new("shape-d", learns),
                Difference::Layer {
                    name: "the deployed topology: the job claimed by a compute process",
                },
                EdgeRules::of(budgets, w, "placed", "shape-d").engine_layer(false),
            ),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::ValueEnum;

    fn ladders() -> impl Iterator<Item = Ladder> {
        Workload::value_variants()
            .iter()
            .map(|w| w.ladder_with(&Budgets::committed()))
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
        let ladder = Workload::TrainRun.ladder_with(&Budgets::committed());
        let edges = ladder.edges();
        assert_eq!(
            *edges[1].difference(),
            Difference::KernelArm {
                families_on: vec![KernelFamily::FlashAttention, KernelFamily::AdamW],
            }
        );
        let step = Workload::TrainStep.ladder_with(&Budgets::committed());
        let Difference::KernelArm { families_on } = step.edges()[1].difference() else {
            panic!("the kernel edge of train-step differs by a kernel arm");
        };
        assert_eq!(families_on.len(), KernelFamily::ALL.len());
    }

    #[test]
    fn a_revision_edge_files_its_legs_under_side_tags() {
        let ladder = Workload::Encode.ladder_with(&Budgets::committed());
        let rules = Workload::Encode.revision_rules("direct", &Budgets::committed());
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
        let ladder = Workload::TrainRun.ladder_with(&Budgets::committed());
        let span = ladder
            .span(Some("resident-reference"), Some("resident"))
            .unwrap();
        assert_eq!(span.len(), 1);
        assert_eq!(span[0].name(), "resident-reference -> resident");
        assert_eq!(ladder.span(None, None).unwrap().len(), 5);
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
        assert_eq!(lengths, [6, 3, 6, 2, 5, 4]);
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

    /// No ladder judges against an invented number: every budgeted rule is
    /// evidence, and every bound it carries is one the committed table read
    /// off an artifact that exists.
    #[test]
    fn every_budgeted_rule_is_evidence_and_every_committed_budget_names_an_artifact() {
        let committed = Budgets::committed();
        for entry in &committed.budgets {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(&entry.budget.measured_from);
            assert!(
                path.is_file(),
                "{}: no such artifact",
                entry.budget.measured_from
            );
        }
        fn shape(s: &ShapeRules) -> [&Rule; 2] {
            [&s.fixed_work_equivalent, &s.per_work_ratio]
        }
        for ladder in ladders() {
            for edge in ladder.edges() {
                let rules: Vec<&Rule> = match edge.kind() {
                    EdgeKind::CrossStack(r) => {
                        [&r.speed_bar, &r.space.host_ratio, &r.space.device_ratio]
                            .into_iter()
                            .chain(r.time_to_quality_bar.iter())
                            .chain(r.shape.iter().flat_map(shape))
                            .collect()
                    }
                    EdgeKind::Exact(r) => [
                        &r.overhead_budget,
                        &r.space.host_ratio,
                        &r.space.device_ratio,
                    ]
                    .into_iter()
                    .chain(r.shape.iter().flat_map(shape))
                    .collect(),
                    EdgeKind::Revision(_) => unreachable!(),
                };
                for rule in rules
                    .into_iter()
                    .chain(edge.upper().flat_host_memory.iter())
                {
                    assert_eq!(rule.force, RuleForce::Evidence, "{}", edge.name());
                    if let Some(budget) = &rule.budget {
                        assert!(
                            committed.budgets.iter().any(|b| b.budget == *budget),
                            "{}: a bound from outside the committed table",
                            edge.name()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_budget_reaches_the_rule_it_names_and_no_other() {
        let budgets: Budgets = serde_json::from_str(
            r#"{"budgets": [{"workload": "encode", "edge": "direct -> plan",
                "rule": "overhead_budget", "bound": 1.07, "measured_from": "Cargo.toml"}]}"#,
        )
        .unwrap();
        let ladder = Workload::Encode.ladder_with(&budgets);
        let edges = ladder.edges();
        let EdgeKind::Exact(plan) = edges[1].kind() else {
            panic!("direct -> plan is an exact edge")
        };
        assert_eq!(
            plan.overhead_budget.budget.as_ref().map(|b| b.bound),
            Some(1.07)
        );
        let EdgeKind::Exact(partitioned) = edges[2].kind() else {
            panic!("plan -> plan-partitioned is an exact edge")
        };
        assert!(partitioned.overhead_budget.budget.is_none());
        assert!(plan.space.host_ratio.budget.is_none());
    }
}
