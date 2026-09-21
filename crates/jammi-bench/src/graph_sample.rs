//! The `graph-sample` workload's engine rung: the biased-walk graph sampler
//! (`jammi_ai::fine_tune::graph_sampler::GraphSampler`) run over a graph *file*,
//! emitting one leg per graph.
//!
//! A graph fine-tune is a composite: sample the graph into a pair table, then
//! train on that table. The pair table is its committed intermediate artifact,
//! and this module is the cut at it. The sampling half is compared here, on its
//! own terms — cost against graph size, and the walk's transition statistics
//! against node2vec's law. The training half is the ordinary fine-tune over the
//! pair table this module writes ([`PairRow`], in the row order and the row
//! shape the trainer reads), so both stacks train on byte-identical input and
//! the sampler's randomness is shared rather than averaged over.
//!
//! ## What a leg carries
//!
//! * the per-iteration wall-clock of one whole-graph `sample_into`, warm, each
//!   iteration at its own seed (`seed + i`) so the series is over the sampler's
//!   work, not one memoised walk set;
//! * the kernel's peak resident set for the process — one graph per process, so
//!   a sweep's points do not inherit each other's peak;
//! * the pair table sampled at `seed`, in the engine's `_ordinal` order, and its
//!   digest;
//! * on request, the walk's raw second-order transition counts — for every
//!   `(previous, current, next)` the number of times a walk stepped
//!   `current → next` having arrived from `previous` — beside the analytic
//!   transition probabilities of the same graph ([`node2vec_transition_law`]).
//!   The counts are the evidence, the law is the ground truth; judging one
//!   against the other is not this module's job.
//!
//! ## The graph file
//!
//! A directory holding `nodes.jsonl` (`{"id", "text"}`) and `edges.jsonl`
//! (`{"src", "dst"}`), each edge row one *directed* edge exactly as the sampler
//! walks it; an undirected graph lists both directions. It is the same pair of
//! files `fine_tune_graph` registers as its node and edge sources.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use jammi_ai::fine_tune::graph_sampler::{
    sort_into_graph_read_order, GraphEdge, GraphSampleConfig, GraphSampler, SampledPair, TextNode,
};

use crate::leg::{artifact_of, write_jsonl, Artifact, IterationSeries};
use crate::report::Measurement;

/// The node file of a graph directory.
pub const NODES_FILE: &str = "nodes.jsonl";
/// The edge file of a graph directory.
pub const EDGES_FILE: &str = "edges.jsonl";

/// One row of `nodes.jsonl`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeRow {
    /// The node id edge endpoints join to.
    pub id: String,
    /// The node's text.
    pub text: String,
}

/// One row of `edges.jsonl`: a directed edge `src → dst`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EdgeRow {
    /// The edge's source node id.
    pub src: String,
    /// The edge's destination node id.
    pub dst: String,
}

/// A graph read from its directory, with the artifacts naming the exact bytes.
#[derive(Debug)]
pub struct GraphFiles {
    /// The nodes, in file order.
    pub nodes: Vec<NodeRow>,
    /// The directed edges, in file order.
    pub edges: Vec<EdgeRow>,
    /// `nodes.jsonl` as read.
    pub nodes_file: Artifact,
    /// `edges.jsonl` as read.
    pub edges_file: Artifact,
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(
    path: &Path,
) -> Result<Vec<T>, Box<dyn std::error::Error>> {
    std::fs::read_to_string(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(i, line)| {
            serde_json::from_str(line)
                .map_err(|e| format!("{} line {}: {e}", path.display(), i + 1).into())
        })
        .collect()
}

impl GraphFiles {
    /// Read the graph directory `dir`.
    pub fn read(dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let (nodes_path, edges_path) = (dir.join(NODES_FILE), dir.join(EDGES_FILE));
        Ok(Self {
            nodes: read_jsonl(&nodes_path)?,
            edges: read_jsonl(&edges_path)?,
            nodes_file: artifact_of(&nodes_path)?,
            edges_file: artifact_of(&edges_path)?,
        })
    }

    /// Write a graph directory and return it as read back.
    pub fn write(
        dir: &Path,
        nodes: &[NodeRow],
        edges: &[EdgeRow],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        write_jsonl(dir, NODES_FILE, nodes)?;
        write_jsonl(dir, EDGES_FILE, edges)?;
        Self::read(dir)
    }

    /// Whether every edge has its reverse — the graph is undirected.
    pub fn is_symmetric(&self) -> bool {
        let set: BTreeSet<(&str, &str)> = self
            .edges
            .iter()
            .map(|e| (e.src.as_str(), e.dst.as_str()))
            .collect();
        set.iter().all(|(s, d)| set.contains(&(*d, *s)))
    }

    /// A sampler over this graph, its nodes and declared edges put into the
    /// engine's graph read order first — the order a `fine_tune_graph` job's own
    /// scans produce — so this sampler draws the rows that job trains on.
    pub fn sampler(
        &self,
        config: GraphSampleConfig,
    ) -> Result<GraphSampler, Box<dyn std::error::Error>> {
        let mut nodes: Vec<TextNode> = self
            .nodes
            .iter()
            .map(|n| TextNode::new(&n.id, &n.text))
            .collect();
        let mut edges: Vec<GraphEdge> = self
            .edges
            .iter()
            .map(|e| GraphEdge::declared(&e.src, &e.dst))
            .collect();
        sort_into_graph_read_order(&mut nodes, &mut edges);
        Ok(GraphSampler::build(nodes, edges, config)?)
    }
}

/// A walk state: the node the walk arrived from (`None` on a walk's first
/// step) and the node it stands on.
pub type WalkState = (Option<String>, String);

/// node2vec's exact transition law over a directed edge list: for every walk
/// state `(t, v)` the probability of each next node `x`,
///
/// ```text
/// π(x | t, v) ∝ α_pq(t, x) · w(v, x)
/// α_pq(t, x) = 1/p  if x = t
///              1    if x is adjacent to t
///              1/q  otherwise
/// ```
///
/// with `w(v, x)` the multiplicity of the edge `v → x` in the list (`1` on a
/// simple graph) and the first step, which has no `t`, drawn `∝ w(v, x)`.
/// "Adjacent to `t`" means an edge joins the two in either direction, which on
/// an undirected graph is the only reading. A node with no out-edge ends the
/// walk and has no row.
///
/// Every state a walk can reach is present: `(None, v)` for each `v` with an
/// out-edge, and `(t, v)` for each edge `t → v` whose `v` has an out-edge. Each
/// state's probabilities are in ascending `x` order and sum to one.
pub fn node2vec_transition_law(
    edges: &[EdgeRow],
    return_p: f64,
    in_out_q: f64,
) -> BTreeMap<WalkState, Vec<(String, f64)>> {
    let mut out: BTreeMap<&str, BTreeMap<&str, f64>> = BTreeMap::new();
    let mut adjacent: BTreeSet<(&str, &str)> = BTreeSet::new();
    for e in edges {
        *out.entry(&e.src).or_default().entry(&e.dst).or_insert(0.0) += 1.0;
        adjacent.insert((&e.src, &e.dst));
        adjacent.insert((&e.dst, &e.src));
    }
    let normalised = |weights: Vec<(&str, f64)>| -> Vec<(String, f64)> {
        let total: f64 = weights.iter().map(|(_, w)| w).sum();
        weights
            .into_iter()
            .map(|(x, w)| (x.to_string(), w / total))
            .collect()
    };

    let first_steps = out.iter().map(|(v, nexts)| {
        (
            (None, v.to_string()),
            normalised(nexts.iter().map(|(x, w)| (*x, *w)).collect()),
        )
    });
    let later_steps = out.iter().flat_map(|(t, currents)| {
        let (out, adjacent, normalised) = (&out, &adjacent, &normalised);
        currents.keys().filter_map(move |v| {
            let nexts = out.get(v)?;
            let weights = nexts
                .iter()
                .map(|(x, w)| {
                    let alpha = if x == t {
                        1.0 / return_p
                    } else if adjacent.contains(&(*t, *x)) {
                        1.0
                    } else {
                        1.0 / in_out_q
                    };
                    (*x, alpha * w)
                })
                .collect();
            Some(((Some(t.to_string()), v.to_string()), normalised(weights)))
        })
    });
    first_steps.chain(later_steps).collect()
}

/// The raw second-order transition counts of a set of walks: how often a walk
/// in state `(previous, current)` stepped to `next`.
#[derive(Debug, Default)]
pub struct TransitionCounts(BTreeMap<(WalkState, String), u64>);

impl TransitionCounts {
    /// Count every step of one walk (its visited node ids, start included).
    pub fn observe(&mut self, walk: &[String]) {
        for (i, step) in walk.windows(2).enumerate() {
            let previous = i.checked_sub(1).map(|j| walk[j].clone());
            *self
                .0
                .entry(((previous, step[0].clone()), step[1].clone()))
                .or_insert(0) += 1;
        }
    }

    /// The counts as `(previous, current, next, count)` rows, ascending.
    pub fn rows(&self) -> impl Iterator<Item = TransitionRow<u64>> + '_ {
        self.0
            .iter()
            .map(|(((prev, cur), next), count)| TransitionRow {
                prev: prev.clone(),
                cur: cur.clone(),
                next: next.clone(),
                value: *count,
            })
    }
}

/// One row of a transition file: a walk state, a next node, and either the
/// observed count or the law's probability for that step.
#[derive(Debug, Serialize)]
pub struct TransitionRow<V> {
    /// The node the walk arrived from; `null` on a walk's first step.
    pub prev: Option<String>,
    /// The node the walk stands on.
    pub cur: String,
    /// The node stepped to.
    pub next: String,
    /// The observed count, or the law's probability.
    pub value: V,
}

fn law_rows(
    law: &BTreeMap<WalkState, Vec<(String, f64)>>,
) -> impl Iterator<Item = TransitionRow<f64>> + '_ {
    law.iter().flat_map(|((prev, cur), nexts)| {
        nexts.iter().map(move |(next, probability)| TransitionRow {
            prev: prev.clone(),
            cur: cur.clone(),
            next: next.clone(),
            value: *probability,
        })
    })
}

/// One row of the pair table, in the engine's `_ordinal` order: what the
/// materialised training set of a graph fine-tune holds (anchor, positive, and
/// the first mined hard negative when the config mines any), with the node ids
/// beside the text. With a negative present this is the triplet row shape
/// `finetune-run --train-jsonl` reads.
#[derive(Debug, Serialize)]
pub struct PairRow {
    /// The row's position in the training set.
    #[serde(rename = "_ordinal")]
    pub ordinal: u64,
    /// Anchor node id.
    pub anchor_id: String,
    /// Anchor node text.
    pub anchor_text: String,
    /// Positive node id.
    pub positive_id: String,
    /// Positive node text.
    pub positive_text: String,
    /// The trained hard negative's node id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_id: Option<String>,
    /// The trained hard negative's text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_text: Option<String>,
}

impl PairRow {
    fn new(ordinal: u64, pair: SampledPair) -> Self {
        Self {
            ordinal,
            anchor_id: pair.anchor_id,
            anchor_text: pair.anchor,
            positive_id: pair.positive_id,
            positive_text: pair.positive,
            negative_id: pair.hard_negative_ids.into_iter().next(),
            negative_text: pair.hard_negatives.into_iter().next(),
        }
    }
}

/// The checksum of a pair table: FNV-1a over the rows' text in `_ordinal`
/// order, separator bytes between fields and rows so two layouts cannot
/// collide. Integers and strings only, so it is equal across machines.
pub fn pair_digest<'a>(rows: impl IntoIterator<Item = (&'a str, &'a str, &'a [String])>) -> String {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    let mut mix = |byte: u8| {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    };
    for (anchor, positive, negatives) in rows {
        anchor.bytes().for_each(&mut mix);
        mix(0xff);
        positive.bytes().for_each(&mut mix);
        mix(0xfe);
        for negative in negatives {
            negative.bytes().for_each(&mut mix);
            mix(0xfd);
        }
        mix(0x00);
    }
    format!("{hash:016x}")
}

/// What one `graph-sample` leg is asked to run.
#[derive(Debug, Clone)]
pub struct GraphSampleParams {
    /// The graph directory to sample.
    pub graph: PathBuf,
    /// Where the leg's artifacts are written.
    pub out: PathBuf,
    /// The sampler configuration; `seed` is the pair table's seed and the first
    /// timed iteration's.
    pub config: GraphSampleConfig,
    /// Untimed iterations before the series starts.
    pub warmup: usize,
    /// Timed iterations.
    pub iterations: usize,
    /// Whether to count the walks' transitions and emit the analytic law.
    pub transitions: bool,
}

/// What two `graph-sample` legs must agree on to be comparable.
#[derive(Debug, Serialize)]
pub struct GraphSampleIdentity {
    /// sha256 of `nodes.jsonl`.
    pub nodes_sha256: String,
    /// sha256 of `edges.jsonl`.
    pub edges_sha256: String,
    /// Node count.
    pub nodes: usize,
    /// Directed edge-row count — the size a sweep is fitted against.
    pub edges: usize,
    /// Whether every edge has its reverse.
    pub edge_set_symmetric: bool,
    /// Steps per walk.
    pub walk_length: usize,
    /// Walks started at each node.
    pub walks_per_node: usize,
    /// node2vec return parameter `p`.
    pub return_p: f64,
    /// node2vec in-out parameter `q`.
    pub in_out_q: f64,
    /// Hard negatives mined per pair.
    pub hard_negatives: usize,
    /// The negative pool's excluded radius; `null` when no negative is mined,
    /// where it affects nothing.
    pub exclude_hops: Option<usize>,
    /// The pair table's seed.
    pub seed: u64,
    /// Untimed iterations before the series.
    pub warmup: usize,
    /// Timed iterations.
    pub iterations: usize,
}

/// Recorded, never compared.
#[derive(Debug, Serialize)]
pub struct GraphSampleProvenance {
    /// The implementation that walked.
    pub walker: &'static str,
    /// The device the walk ran on.
    pub device: &'static str,
}

/// What a `graph-sample` leg measured.
#[derive(Debug, Serialize)]
pub struct GraphSampleMeasured {
    /// Seconds per whole-graph sample, warm-up excluded.
    pub iteration_s: Vec<f64>,
    /// The process's peak resident set (kernel high-water mark).
    pub peak_rss_bytes: Measurement,
    /// Peak device memory; the walk uses no device.
    pub peak_vram_bytes: Measurement,
    /// Walks per sample: `nodes · walks_per_node`.
    pub walks: usize,
    /// Rows in the pair table.
    pub sampled_pairs: usize,
    /// [`pair_digest`] of the pair table.
    pub pairs_digest: String,
    /// The pair table at `seed`, one [`PairRow`] per line.
    pub pairs: Artifact,
    /// The observed transition counts over every iteration's walks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transitions: Option<Artifact>,
    /// [`node2vec_transition_law`] of this graph at this `p`, `q`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_transitions: Option<Artifact>,
}

/// One `graph-sample` leg.
pub type GraphSampleLeg =
    crate::leg::Leg<GraphSampleIdentity, GraphSampleProvenance, GraphSampleMeasured>;

/// Sample `graph` once at `config` and write its pair table into `out`.
/// Returns the table's artifact, its digest and its row count. The table is what
/// a fine-tune trains on, so a config a fine-tune refuses (more than one hard
/// negative per pair) is refused here too.
pub fn write_pair_table(
    graph: &GraphFiles,
    config: GraphSampleConfig,
    out: &Path,
) -> Result<(Artifact, String, usize), Box<dyn std::error::Error>> {
    config.validate_for_training()?;
    let pairs = graph.sampler(config)?.sample()?;
    let digest = pair_digest(pairs.iter().map(|p| {
        (
            p.anchor.as_str(),
            p.positive.as_str(),
            p.hard_negatives.as_slice(),
        )
    }));
    let count = pairs.len();
    let rows = pairs
        .into_iter()
        .enumerate()
        .map(|(i, pair)| PairRow::new(i as u64, pair));
    Ok((write_jsonl(out, "pairs.jsonl", rows)?, digest, count))
}

/// Run one `graph-sample` leg.
pub fn run(params: &GraphSampleParams) -> Result<GraphSampleLeg, Box<dyn std::error::Error>> {
    let graph = GraphFiles::read(&params.graph)?;
    let config = params.config;
    let at_seed = |i: usize| GraphSampleConfig {
        seed: config.seed.wrapping_add(i as u64),
        ..config
    };

    let mut series = IterationSeries::new(params.warmup, params.iterations);
    for i in 0..series.total() {
        let sampler = graph.sampler(at_seed(i))?;
        let start = Instant::now();
        sampler.sample_into(|_| Ok(()))?;
        series.record(start.elapsed());
    }
    let peak_rss_bytes = crate::rss::peak_rss_bytes();

    let (pairs, pairs_digest, sampled_pairs) = write_pair_table(&graph, config, &params.out)?;

    let (transitions, expected_transitions) = if params.transitions {
        // The same walks the timed iterations drew: one observed pass per seed.
        let mut counts = TransitionCounts::default();
        for i in 0..series.total() {
            graph.sampler(at_seed(i))?.sample_observing_walks(
                |walk| {
                    counts.observe(walk);
                    Ok(())
                },
                |_| Ok(()),
            )?;
        }
        let law = node2vec_transition_law(&graph.edges, config.return_p, config.in_out_q);
        (
            Some(write_jsonl(
                &params.out,
                "transitions.jsonl",
                counts.rows(),
            )?),
            Some(write_jsonl(
                &params.out,
                "expected_transitions.jsonl",
                law_rows(&law),
            )?),
        )
    } else {
        (None, None)
    };

    Ok(crate::leg::Leg {
        workload: "graph-sample",
        rung: "jammi".to_string(),
        identity: GraphSampleIdentity {
            nodes_sha256: graph.nodes_file.sha256.clone(),
            edges_sha256: graph.edges_file.sha256.clone(),
            nodes: graph.nodes.len(),
            edges: graph.edges.len(),
            edge_set_symmetric: graph.is_symmetric(),
            walk_length: config.walk_length,
            walks_per_node: config.walks_per_node,
            return_p: config.return_p,
            in_out_q: config.in_out_q,
            hard_negatives: config.hard_negatives,
            exclude_hops: (config.hard_negatives > 0).then_some(config.exclude_hops),
            seed: config.seed,
            warmup: params.warmup,
            iterations: params.iterations,
        },
        provenance: GraphSampleProvenance {
            walker: "jammi_ai::fine_tune::graph_sampler::GraphSampler",
            device: "cpu",
        },
        measured: GraphSampleMeasured {
            iteration_s: series.into_seconds(),
            peak_rss_bytes,
            peak_vram_bytes: Measurement::not_yet_measured("bytes"),
            walks: graph.nodes.len() * config.walks_per_node,
            sampled_pairs,
            pairs_digest,
            pairs,
            transitions,
            expected_transitions,
        },
    })
}

/// `graph-sample`'s flags: the graphs to sample (one leg each) and the sampler
/// configuration every leg shares.
#[derive(Debug, Clone, clap::Args)]
pub struct GraphSampleArgs {
    /// A graph directory (`nodes.jsonl` + `edges.jsonl`). Repeat for a size
    /// sweep: each graph is then sampled in its own process.
    #[arg(long = "graph", required = true)]
    graphs: Vec<PathBuf>,
    /// Where artifacts are written, one subdirectory per graph directory name.
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 4)]
    walk_length: usize,
    #[arg(long, default_value_t = 4)]
    walks_per_node: usize,
    #[arg(long, default_value_t = 1.0)]
    return_p: f64,
    #[arg(long, default_value_t = 1.0)]
    in_out_q: f64,
    /// Hard negatives mined per pair; `0` times the walk alone.
    #[arg(long, default_value_t = 0)]
    hard_negatives: usize,
    #[arg(long, default_value_t = 1)]
    exclude_hops: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long, default_value_t = 10)]
    iterations: usize,
    /// Also count the walks' transitions and emit the analytic law. Meant for a
    /// small graph: the law has one row per `(edge, next node)`.
    #[arg(long)]
    transitions: bool,
}

impl GraphSampleArgs {
    fn params(&self, graph: &Path) -> Result<GraphSampleParams, Box<dyn std::error::Error>> {
        let name = graph
            .file_name()
            .ok_or_else(|| format!("--graph {} has no directory name", graph.display()))?;
        Ok(GraphSampleParams {
            graph: graph.to_path_buf(),
            out: self.out.join(name),
            config: GraphSampleConfig {
                walk_length: self.walk_length,
                walks_per_node: self.walks_per_node,
                return_p: self.return_p,
                in_out_q: self.in_out_q,
                hard_negatives: self.hard_negatives,
                exclude_hops: self.exclude_hops,
                min_negatives: 1,
                seed: self.seed,
            },
            warmup: self.warmup,
            iterations: self.iterations,
            transitions: self.transitions,
        })
    }

    /// The invocation of this binary that samples `graph` alone under these
    /// flags.
    fn child_args(&self, graph: &Path) -> Vec<std::ffi::OsString> {
        let mut args: Vec<std::ffi::OsString> = vec![
            "graph-sample".into(),
            "--graph".into(),
            graph.into(),
            "--out".into(),
            (&self.out).into(),
        ];
        let flags = [
            ("--walk-length", self.walk_length.to_string()),
            ("--walks-per-node", self.walks_per_node.to_string()),
            ("--return-p", self.return_p.to_string()),
            ("--in-out-q", self.in_out_q.to_string()),
            ("--hard-negatives", self.hard_negatives.to_string()),
            ("--exclude-hops", self.exclude_hops.to_string()),
            ("--seed", self.seed.to_string()),
            ("--warmup", self.warmup.to_string()),
            ("--iterations", self.iterations.to_string()),
        ];
        args.extend(
            flags
                .into_iter()
                .flat_map(|(flag, value)| [flag.into(), value.into()]),
        );
        if self.transitions {
            args.push("--transitions".into());
        }
        args
    }

    /// Run the subcommand: one leg per graph, printed as one report.
    pub async fn execute(&self) -> Result<(), Box<dyn std::error::Error>> {
        let first = self.params(&self.graphs[0])?;
        let legs = crate::leg::leg_per_point(&self.graphs, async move { run(&first) }, |graph| {
            self.child_args(graph)
        })
        .await?;
        Ok(crate::leg::LegReport::new("graph-sample", legs).emit()?)
    }
}

/// `graph-pairs`' flags: a graph and the sampler configuration of the
/// `fine_tune_graph` job whose training set is wanted.
#[derive(Debug, Clone, clap::Args)]
pub struct GraphPairsArgs {
    /// The graph directory (`nodes.jsonl` + `edges.jsonl`).
    #[arg(long)]
    graph: PathBuf,
    /// The directory `pairs.jsonl` is written into.
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 4)]
    walk_length: usize,
    #[arg(long, default_value_t = 2)]
    walks_per_node: usize,
    #[arg(long, default_value_t = 1.0)]
    return_p: f64,
    #[arg(long, default_value_t = 1.0)]
    in_out_q: f64,
    /// `1` writes triplet rows, the shape `finetune-run --train-jsonl` reads.
    #[arg(long, default_value_t = 1)]
    hard_negatives: usize,
    #[arg(long, default_value_t = 1)]
    exclude_hops: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

impl GraphPairsArgs {
    /// Write the pair table and print its artifact, digest and row count.
    pub fn execute(&self) -> Result<(), Box<dyn std::error::Error>> {
        let config = GraphSampleConfig {
            walk_length: self.walk_length,
            walks_per_node: self.walks_per_node,
            return_p: self.return_p,
            in_out_q: self.in_out_q,
            hard_negatives: self.hard_negatives,
            exclude_hops: self.exclude_hops,
            min_negatives: 1,
            seed: self.seed,
        };
        let (pairs, digest, rows) =
            write_pair_table(&GraphFiles::read(&self.graph)?, config, &self.out)?;
        println!(
            "{}",
            serde_json::json!({"pairs": pairs, "pairs_digest": digest, "rows": rows})
        );
        Ok(())
    }
}

/// `graph-fixture`'s flags: the committed synthetic graph's shape, at a chosen
/// size.
#[derive(Debug, Clone, clap::Args)]
pub struct GraphFixtureArgs {
    /// Nodes per community; the edge count scales with it. Defaults to the
    /// committed fixture's.
    #[arg(long)]
    nodes_per: Option<usize>,
    /// The graph directory to write.
    #[arg(long)]
    out: PathBuf,
}

impl GraphFixtureArgs {
    /// Write the graph and print its size and file digests.
    pub fn execute(&self) -> Result<(), Box<dyn std::error::Error>> {
        let committed = CommittedSample::load()?.graph;
        let graph = SyntheticGraph {
            nodes_per: self.nodes_per.unwrap_or(committed.nodes_per),
            ..committed
        }
        .write(&self.out)?;
        println!(
            "{}",
            serde_json::json!({
                "nodes": graph.nodes.len(),
                "edges": graph.edges.len(),
                "nodes_file": graph.nodes_file,
                "edges_file": graph.edges_file,
            })
        );
        Ok(())
    }
}

/// The shape of the synthetic multi-community graph: `communities` homophilous
/// clusters of `nodes_per` text-bearing nodes, each wired intra-densely by a
/// bounded circulant and joined to the next by sparse bridges — the multi-scale
/// structure node2vec walks are meant to capture. The wiring is a pure function
/// of the shape, so a shape names one graph on any box, and the edge count
/// scales with `nodes_per`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyntheticGraph {
    /// Number of communities.
    pub communities: usize,
    /// Nodes per community; the total node count is `communities · nodes_per`.
    pub nodes_per: usize,
    /// Each node wires to its next `intra_degree` community-mates cyclically, so
    /// the within-community subgraph is degree-regular but multi-hop, not a
    /// clique, and the edge set stays `O(nodes · intra_degree)`.
    pub intra_degree: usize,
    /// Every `bridge_stride`-th node also wires to the matching-index node of
    /// the next community (the structure a DFS-biased walk crosses and a
    /// BFS-biased walk does not).
    pub bridge_stride: usize,
}

impl SyntheticGraph {
    /// Write the graph as a graph directory: node ids `c{community}_{i}`, node
    /// text `document c{c} node {i}` — **distinct per node**, so a pair digest is
    /// sensitive to the exact node a walk picked and not merely its community,
    /// while the text still names the community. An undirected simple graph:
    /// each wired pair appears once per direction.
    pub fn write(&self, dir: &Path) -> Result<GraphFiles, Box<dyn std::error::Error>> {
        let id = |c: usize, i: usize| format!("c{c}_{i}");
        let nodes: Vec<NodeRow> = (0..self.communities)
            .flat_map(|c| (0..self.nodes_per).map(move |i| (c, i)))
            .map(|(c, i)| NodeRow {
                id: id(c, i),
                text: format!("document c{c} node {i}"),
            })
            .collect();

        let reach = self.intra_degree.min(self.nodes_per.saturating_sub(1));
        let bridged = self.bridge_stride > 0 && self.communities > 1;
        let wired = (0..self.communities)
            .flat_map(|c| (0..self.nodes_per).map(move |i| (c, i)))
            .flat_map(|(c, i)| {
                let ring = (1..=reach).map(move |off| (c, (i + off) % self.nodes_per));
                let bridge = (bridged && i % self.bridge_stride == 0)
                    .then_some(((c + 1) % self.communities, i));
                ring.chain(bridge).map(move |other| ((c, i), other))
            })
            .filter(|(a, b)| a != b);
        // A small community can wrap the same pair twice; the set keeps the
        // graph simple and the rows in one order on any box.
        let edges: BTreeSet<EdgeRow> = wired
            .flat_map(|((c, i), (d, j))| {
                [
                    EdgeRow {
                        src: id(c, i),
                        dst: id(d, j),
                    },
                    EdgeRow {
                        src: id(d, j),
                        dst: id(c, i),
                    },
                ]
            })
            .collect();
        GraphFiles::write(dir, &nodes, &edges.into_iter().collect::<Vec<_>>())
    }
}

/// The committed sampler fixture (`baselines/graph_sample.json`): the synthetic
/// graph's shape, the sampler knobs, and the digest of the pair table the
/// engine's sampler draws over them.
///
/// The sampler is seeded — an integer walk/negative stream, a sequential scalar
/// `f64` roulette, no floating-point reduction — and the digest folds node text
/// only, so the pair table is byte-identical *across machines* and the digest is
/// asserted for equality, unlike an `f32`-output digest. A change in the walk
/// bias, the negative mining, the adjacency construction or the graph read order
/// moves it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommittedSample {
    /// The synthetic graph.
    pub graph: SyntheticGraph,
    /// The sampler configuration.
    pub config: GraphSampleConfig,
    /// [`pair_digest`] of the pair table sampled over `graph` at `config`.
    pub digest: String,
}

impl CommittedSample {
    /// Load the committed fixture.
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("baselines")
            .join("graph_sample.json");
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// The observed next-node frequencies of a count set, per state — test support
    /// for reading counts against the law.
    fn frequencies(counts: &TransitionCounts) -> HashMap<WalkState, HashMap<String, f64>> {
        let mut totals: HashMap<WalkState, f64> = HashMap::new();
        for row in counts.rows() {
            *totals
                .entry((row.prev.clone(), row.cur.clone()))
                .or_default() += row.value as f64;
        }
        let mut out: HashMap<WalkState, HashMap<String, f64>> = HashMap::new();
        for row in counts.rows() {
            let state = (row.prev, row.cur);
            let total = totals[&state];
            out.entry(state)
                .or_default()
                .insert(row.next, row.value as f64 / total);
        }
        out
    }

    fn edge(src: &str, dst: &str) -> EdgeRow {
        EdgeRow {
            src: src.into(),
            dst: dst.into(),
        }
    }

    fn undirected(pairs: &[(&str, &str)]) -> Vec<EdgeRow> {
        pairs
            .iter()
            .flat_map(|(a, b)| [edge(a, b), edge(b, a)])
            .collect()
    }

    fn probability(
        law: &BTreeMap<WalkState, Vec<(String, f64)>>,
        state: WalkState,
        x: &str,
    ) -> f64 {
        law[&state]
            .iter()
            .find(|(next, _)| next == x)
            .map_or(0.0, |(_, p)| *p)
    }

    /// The law on a graph small enough to work by hand. A triangle `a-b-c` with
    /// a tail `c-d`: from `(a, c)` the walk may return to `a` (1/p), step to `b`
    /// (adjacent to `a`: 1), or out to `d` (not adjacent to `a`: 1/q).
    #[test]
    fn law_matches_the_hand_worked_triangle_with_a_tail() {
        let edges = undirected(&[("a", "b"), ("b", "c"), ("a", "c"), ("c", "d")]);
        let (p, q) = (4.0, 0.5);
        let law = node2vec_transition_law(&edges, p, q);

        let state = (Some("a".to_string()), "c".to_string());
        let total = 1.0 / p + 1.0 + 1.0 / q;
        assert!((probability(&law, state.clone(), "a") - (1.0 / p) / total).abs() < 1e-15);
        assert!((probability(&law, state.clone(), "b") - 1.0 / total).abs() < 1e-15);
        assert!((probability(&law, state, "d") - (1.0 / q) / total).abs() < 1e-15);

        // A walk's first step is uniform over the out-neighbours.
        let first = (None, "c".to_string());
        for x in ["a", "b", "d"] {
            assert!((probability(&law, first.clone(), x) - 1.0 / 3.0).abs() < 1e-15);
        }
        // The tail's only move is back.
        assert_eq!(
            law[&(Some("c".to_string()), "d".to_string())],
            vec![("c".to_string(), 1.0)]
        );
        // Every reachable state is present and normalised: one first-step state
        // per node, one later state per directed edge.
        assert_eq!(law.len(), 4 + edges.len());
        for nexts in law.values() {
            assert!((nexts.iter().map(|(_, p)| p).sum::<f64>() - 1.0).abs() < 1e-12);
        }
    }

    /// A repeated edge row is a heavier edge, and a dead end has no state.
    #[test]
    fn law_weights_by_multiplicity_and_stops_at_a_dead_end() {
        let edges = vec![edge("a", "b"), edge("a", "b"), edge("a", "c")];
        let law = node2vec_transition_law(&edges, 1.0, 1.0);
        let first = (None, "a".to_string());
        assert!((probability(&law, first.clone(), "b") - 2.0 / 3.0).abs() < 1e-15);
        assert!((probability(&law, first, "c") - 1.0 / 3.0).abs() < 1e-15);
        assert_eq!(law.len(), 1, "b and c have no out-edge, so no state");
    }

    /// The engine's walk follows the law: over many seeded walks of a graph
    /// that exercises all three `α` branches, every state's observed next-node
    /// frequencies sit within a few standard errors of the analytic
    /// probabilities, and a walk never takes a step the law gives zero.
    #[test]
    fn engine_walk_frequencies_follow_the_law() {
        let edges = undirected(&[
            ("a", "b"),
            ("b", "c"),
            ("a", "c"),
            ("c", "d"),
            ("d", "e"),
            ("e", "c"),
            ("e", "f"),
        ]);
        let nodes: Vec<NodeRow> = ["a", "b", "c", "d", "e", "f"]
            .iter()
            .map(|id| NodeRow {
                id: id.to_string(),
                text: format!("text {id}"),
            })
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let graph = GraphFiles::write(dir.path(), &nodes, &edges).unwrap();
        assert!(graph.is_symmetric());

        let (p, q) = (0.25, 4.0);
        let mut counts = TransitionCounts::default();
        for seed in 0..400 {
            let config = GraphSampleConfig {
                walk_length: 6,
                walks_per_node: 8,
                return_p: p,
                in_out_q: q,
                hard_negatives: 0,
                seed,
                ..GraphSampleConfig::default()
            };
            graph
                .sampler(config)
                .unwrap()
                .sample_observing_walks(
                    |walk| {
                        counts.observe(walk);
                        Ok(())
                    },
                    |_| Ok(()),
                )
                .unwrap();
        }

        let law = node2vec_transition_law(&graph.edges, p, q);
        let observed = frequencies(&counts);
        let mut state_totals: HashMap<WalkState, u64> = HashMap::new();
        for row in counts.rows() {
            *state_totals.entry((row.prev, row.cur)).or_default() += row.value;
        }
        assert_eq!(
            observed.len(),
            law.len(),
            "the walks reach exactly the law's states"
        );
        for (state, nexts) in &law {
            let n = state_totals[state] as f64;
            for (x, expected) in nexts {
                let seen = observed[state].get(x).copied().unwrap_or(0.0);
                let standard_error = (expected * (1.0 - expected) / n).sqrt();
                assert!(
                    (seen - expected).abs() <= 5.0 * standard_error + 1e-12,
                    "state {state:?} → {x}: observed {seen}, law {expected}, n {n}"
                );
            }
            for x in observed[state].keys() {
                assert!(
                    nexts.iter().any(|(next, _)| next == x),
                    "state {state:?} stepped to {x}, which the law gives zero"
                );
            }
        }
    }

    /// A leg over a written graph: the series has the asked length, the pair
    /// table is the engine's sample at the seed in `_ordinal` order, and the
    /// transition files are present exactly when asked for.
    #[test]
    fn leg_carries_series_pair_table_and_transitions() {
        let dir = tempfile::tempdir().unwrap();
        let graph = SyntheticGraph {
            nodes_per: 8,
            ..CommittedSample::load().unwrap().graph
        }
        .write(&dir.path().join("graph"))
        .unwrap();
        assert!(graph.is_symmetric());
        let config = GraphSampleConfig {
            hard_negatives: 1,
            seed: 5,
            ..GraphSampleConfig::default()
        };
        let params = GraphSampleParams {
            graph: dir.path().join("graph"),
            out: dir.path().join("out"),
            config,
            warmup: 1,
            iterations: 3,
            transitions: true,
        };
        let leg = run(&params).unwrap();
        assert_eq!(leg.measured.iteration_s.len(), 3);
        assert_eq!(leg.identity.edges, graph.edges.len());
        assert_eq!(leg.identity.exclude_hops, Some(config.exclude_hops));

        let expected = graph.sampler(config).unwrap().sample().unwrap();
        let table: Vec<serde_json::Value> =
            read_jsonl(Path::new(&leg.measured.pairs.path)).unwrap();
        assert_eq!(table.len(), expected.len());
        assert_eq!(leg.measured.sampled_pairs, expected.len());
        for (i, (row, pair)) in table.iter().zip(&expected).enumerate() {
            assert_eq!(row["_ordinal"], i as u64);
            assert_eq!(row["anchor_id"], pair.anchor_id);
            assert_eq!(row["anchor_text"], pair.anchor);
            assert_eq!(row["positive_text"], pair.positive);
            assert_eq!(row["negative_text"], pair.hard_negatives[0]);
        }
        assert!(leg.measured.transitions.is_some());
        assert!(leg.measured.expected_transitions.is_some());

        let quiet = run(&GraphSampleParams {
            transitions: false,
            config: GraphSampleConfig {
                hard_negatives: 0,
                ..config
            },
            ..params
        })
        .unwrap();
        assert!(quiet.measured.transitions.is_none());
        assert_eq!(quiet.identity.exclude_hops, None);
        let rows: Vec<serde_json::Value> =
            read_jsonl(Path::new(&quiet.measured.pairs.path)).unwrap();
        assert!(rows[0].get("negative_text").is_none());
    }

    /// A leg over the committed graph at `config`, written under a fresh tempdir.
    fn committed_leg(
        committed: &CommittedSample,
        config: GraphSampleConfig,
    ) -> (GraphSampleLeg, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        committed.graph.write(&dir.path().join("graph")).unwrap();
        let leg = run(&GraphSampleParams {
            graph: dir.path().join("graph"),
            out: dir.path().join("out"),
            config,
            warmup: 0,
            iterations: 1,
            transitions: false,
        })
        .unwrap();
        (leg, dir)
    }

    /// The committed fixture is a structured multi-community graph under a
    /// higher-order walk, with a trainable config and an FNV-width digest.
    #[test]
    fn committed_sample_is_well_formed() {
        let committed = CommittedSample::load().expect("baselines/graph_sample.json");
        assert!(committed.graph.communities >= 2);
        assert!(committed.graph.nodes_per >= 2);
        assert!(committed.config.walk_length >= 2, "a higher-order walk");
        committed.config.validate_for_training().unwrap();
        assert!(
            committed.config.hard_negatives > 0,
            "the negative mining is folded"
        );
        assert_eq!(committed.digest.len(), 16);
        assert!(committed.digest.chars().all(|c| c.is_ascii_hexdigit()));
    }

    /// The sampler is a pure function of its seed on any machine: a leg over the
    /// committed graph at the committed knobs reproduces the committed digest.
    #[test]
    fn leg_reproduces_the_committed_digest() {
        let committed = CommittedSample::load().unwrap();
        let (leg, _dir) = committed_leg(&committed, committed.config);
        assert_eq!(
            leg.measured.pairs_digest, committed.digest,
            "the sampled pair table drifted off the committed digest"
        );
    }

    /// The equality above has teeth: the same graph re-sampled at a different
    /// seed, or a shorter walk, draws a different table.
    #[test]
    fn perturbed_sample_changes_the_digest() {
        let committed = CommittedSample::load().unwrap();
        for (what, config) in [
            (
                "a different seed",
                GraphSampleConfig {
                    seed: committed.config.seed.wrapping_add(1),
                    ..committed.config
                },
            ),
            (
                "a shorter walk",
                GraphSampleConfig {
                    walk_length: committed.config.walk_length - 1,
                    ..committed.config
                },
            ),
        ] {
            assert_ne!(
                committed_leg(&committed, config).0.measured.pairs_digest,
                committed.digest,
                "{what} must change the digest"
            );
        }
    }

    /// The digest is over real graph-structured work: every positive is a node
    /// other than its anchor, and on the homophilous graph most positives stay in
    /// the anchor's community.
    #[test]
    fn committed_pairs_are_structured() {
        let committed = CommittedSample::load().unwrap();
        let (leg, _dir) = committed_leg(&committed, committed.config);
        let rows: Vec<serde_json::Value> = read_jsonl(Path::new(&leg.measured.pairs.path)).unwrap();
        assert!(!rows.is_empty());
        let community =
            |id: &serde_json::Value| id.as_str().unwrap().split('_').next().unwrap().to_string();
        let in_community = rows
            .iter()
            .inspect(|r| assert_ne!(r["anchor_id"], r["positive_id"]))
            .filter(|r| community(&r["anchor_id"]) == community(&r["positive_id"]))
            .count();
        let ratio = in_community as f64 / rows.len() as f64;
        assert!(
            ratio > 0.7,
            "positives should stay mostly in-community, got {ratio}"
        );
    }

    /// A pair table is what a fine-tune trains on, so a config that mines more
    /// negatives than a training row holds is refused, not truncated.
    #[test]
    fn pair_table_refuses_more_negatives_than_a_row_trains() {
        let committed = CommittedSample::load().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let graph = committed.graph.write(&dir.path().join("graph")).unwrap();
        let config = GraphSampleConfig {
            hard_negatives: 2,
            ..committed.config
        };
        assert!(write_pair_table(&graph, config, dir.path()).is_err());
    }
}
