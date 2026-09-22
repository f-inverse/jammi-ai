//! The `graph-sample` workload's engine rung, `sampler`: the biased-walk
//! graph sampler (`jammi_ai::fine_tune::graph_sampler::GraphSampler`) run over
//! a graph *directory*, one leg per graph.
//!
//! A graph fine-tune is a composite: sample the graph into a pair table, then
//! train on that table. The pair table is its committed intermediate artifact,
//! and this module is the cut at it. The sampling half is compared here, on its
//! own terms — cost against graph size, and the walk's transition statistics
//! against node2vec's law. The training half is the ordinary `train-run` over
//! the pair table this module writes ([`PairRow`], in the row order and the row
//! shape the trainer reads), so both stacks train on byte-identical input and
//! the sampler's randomness is shared rather than averaged over.
//!
//! ## What a leg carries
//!
//! * `iter_wall_s`: the wall-clock of one whole-graph `sample_into`, warm, each
//!   iteration at its own seed (`seed + i`) so the series is over the sampler's
//!   work, not one memoised walk set; `work` is the edge count, the sweep's
//!   axis;
//! * `peak_rss_bytes`: the kernel's high-water mark for the process — one graph
//!   per process, so a sweep's points do not inherit each other's peak;
//! * the pair table sampled at `seed`, in the engine's `_ordinal` order, and
//!   its digest as `outcome_digest`;
//! * `law_observed`: the walks' raw second-order transition counts — for every
//!   walk state `(previous, current)`, how often the walk stepped to each
//!   `next` — in the order of the law file, `<unit>.json`
//!   (`{"cells": [[probability, …], …]}`), which this rung writes from
//!   [`node2vec_transition_law`] and every rung's leg names by its sha256 as
//!   the identity field `law_sha256`. The counts are the evidence, the law is
//!   the ground truth; judging one against the other is the ladder's.
//!
//! ## The graph directory
//!
//! `nodes.jsonl` (`{"id", "text"}`) and `edges.jsonl` (`{"src", "dst"}`), each
//! edge row one *directed* edge exactly as the sampler walks it; an undirected
//! graph lists both directions. It is the same pair of files `fine_tune_graph`
//! registers as its node and edge sources. A graph's unit is `edges<N>`, its
//! directed row count.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use jammi_ai::fine_tune::graph_sampler::{
    sort_into_graph_read_order, GraphEdge, GraphSampleConfig, GraphSampler, SampledPair, TextNode,
};

use crate::capture::{
    artifact_of, cpu_provenance, file_leg, leg_report, leg_stem, legs_per_point, write_artifact,
    write_jsonl, Artifact, IterationSeries,
};
use crate::leg::{Facts, Leg, Measured, Measurement, Payload};
use crate::report::{Nullable, Tiers};

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

pub fn read_jsonl<T: for<'de> Deserialize<'de>>(
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

/// The law as the ladder reads it: one cell per walk state, the probabilities
/// of that state's next nodes in ascending order — [`node2vec_transition_law`]'s
/// own order, which is also the order of every leg's `law_observed`.
pub fn law_file(law: &BTreeMap<WalkState, Vec<(String, f64)>>) -> Vec<u8> {
    let cells: Vec<Vec<f64>> = law
        .values()
        .map(|nexts| nexts.iter().map(|(_, p)| *p).collect())
        .collect();
    serde_json::to_vec(&serde_json::json!({ "cells": cells })).expect("serialize the law")
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

    /// The counts in the law's order: one row per state, one count per next
    /// node the law gives that state, zero where the walk never stepped there.
    /// A step the law gives no cell is an error — a walk that left the law.
    pub fn observed_by(
        &self,
        law: &BTreeMap<WalkState, Vec<(String, f64)>>,
    ) -> Result<Vec<Vec<u64>>, Box<dyn std::error::Error>> {
        if let Some(((state, next), _)) = self.0.iter().find(|((state, next), _)| {
            !law.get(state)
                .is_some_and(|nexts| nexts.iter().any(|(x, _)| x == next))
        }) {
            return Err(
                format!("a walk stepped {state:?} → {next}, which the law gives no cell").into(),
            );
        }
        Ok(law
            .iter()
            .map(|(state, nexts)| {
                nexts
                    .iter()
                    .map(|(next, _)| {
                        self.0
                            .get(&(state.clone(), next.clone()))
                            .copied()
                            .unwrap_or(0)
                    })
                    .collect()
            })
            .collect())
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

/// The rung this producer serves.
pub const RUNG: &str = "sampler";
/// Where the law files go under the legs directory unless told otherwise.
pub const LAW_DIR: &str = "law";

/// The sampler run one `graph-sample` leg is asked to make.
#[derive(Debug, Clone)]
pub struct GraphSampleParams {
    /// The graph directory to sample.
    pub graph: PathBuf,
    /// Where the leg, its pair table and the law file are filed.
    pub legs_dir: PathBuf,
    /// Where the law file `<unit>.json` is written; `legs_dir/law` when
    /// `None`. Never the legs directory itself, whose every `.json` is a leg.
    pub law_dir: Option<PathBuf>,
    /// The sampler configuration; `seed` is the pair table's seed and the first
    /// timed iteration's.
    pub config: GraphSampleConfig,
    /// Untimed iterations before the series starts.
    pub warmup: usize,
    /// Timed iterations.
    pub iterations: usize,
    /// The measured repeat this leg is filed as.
    pub take: usize,
}

/// The `graph-sample` leg's payload: what two legs must agree on, and what
/// this rung records about its run.
#[derive(Debug, Clone, Serialize)]
pub struct GraphSamplePayload {
    /// The pair table's seed.
    pub seed: u64,
    /// sha256 of `edges.jsonl`.
    pub graph_edges_sha256: String,
    /// sha256 of the law file the counts are against.
    pub law_sha256: String,
    /// Node count.
    pub node_count: usize,
    /// Directed edge-row count — the unit of the sweep.
    pub edge_count: usize,
    /// Steps per walk.
    pub walk_length: usize,
    /// Walks started at each node.
    pub walks_per_node: usize,
    /// node2vec return parameter `p`.
    pub return_p: f64,
    /// node2vec in-out parameter `q`.
    pub in_out_q: f64,
    /// The rung this leg claims.
    pub rung: &'static str,
    /// The unit of the sweep, `edges<N>`.
    pub unit: String,
    /// The measured repeat.
    pub take: usize,
    /// sha256 of `nodes.jsonl`.
    pub graph_nodes_sha256: String,
    /// Whether every edge has its reverse.
    pub edge_set_symmetric: bool,
    /// Hard negatives mined per pair; the torch rung mines none.
    pub hard_negatives: usize,
    /// The negative pool's excluded radius; `null` when no negative is mined.
    pub exclude_hops: Option<usize>,
    /// Untimed iterations before the series.
    pub warmup: usize,
    /// Timed iterations.
    pub iters_measured: usize,
    /// Walks per sample: `node_count · walks_per_node`.
    pub walks: usize,
    /// Rows in the pair table.
    pub sampled_pairs: usize,
    /// The pair table at `seed`, one [`PairRow`] per line.
    pub pairs_file: Artifact,
    /// The implementation that walked.
    pub walker: &'static str,
}

impl Payload for GraphSamplePayload {
    const IDENTITY_FIELDS: &'static [(&'static str, Nullable)] = &[
        ("seed", Nullable::NonNull),
        ("graph_edges_sha256", Nullable::NonNull),
        ("law_sha256", Nullable::NonNull),
        ("node_count", Nullable::NonNull),
        ("edge_count", Nullable::NonNull),
        ("walk_length", Nullable::NonNull),
        ("walks_per_node", Nullable::NonNull),
        ("return_p", Nullable::NonNull),
        ("in_out_q", Nullable::NonNull),
    ];
}

/// Sample `graph` once at `config` and write its pair table as `name` into
/// `dir`. Returns the table's artifact, its digest and its row count. The
/// table is what a fine-tune trains on, so a config a fine-tune refuses (more
/// than one hard negative per pair) is refused here too.
pub fn write_pair_table(
    graph: &GraphFiles,
    config: GraphSampleConfig,
    dir: &Path,
    name: &str,
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
    Ok((write_jsonl(dir, name, rows)?, digest, count))
}

/// Run one `graph-sample` leg and file it: the leg as
/// `sampler__edges<N>__r<take>.json`, its pair table beside it, and the law
/// file `<unit>.json` in the law directory. Returns the leg and its file name.
pub fn run_leg(
    params: &GraphSampleParams,
) -> Result<(Leg<GraphSamplePayload>, String), Box<dyn std::error::Error>> {
    let graph = GraphFiles::read(&params.graph)?;
    let config = params.config;
    let at_seed = |i: usize| GraphSampleConfig {
        seed: config.seed.wrapping_add(i as u64),
        ..config
    };
    let unit = format!("edges{}", graph.edges.len());
    let stem = leg_stem(RUNG, &unit, params.take);

    let mut series = IterationSeries::new(params.warmup, params.iterations);
    for i in 0..series.total() {
        let sampler = graph.sampler(at_seed(i))?;
        let start = Instant::now();
        sampler.sample_into(|_| Ok(()))?;
        series.record(start.elapsed());
    }
    let peak_rss_bytes = crate::rss::peak_rss_measurement();

    // The same walks the timed iterations drew, observed: one pass per seed.
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
    let law_dir = params
        .law_dir
        .clone()
        .unwrap_or_else(|| params.legs_dir.join(LAW_DIR));
    let law_written = write_artifact(&law_dir, &format!("{unit}.json"), &law_file(&law))?;
    let law_observed = counts.observed_by(&law)?;

    let (pairs_file, pairs_digest, sampled_pairs) = write_pair_table(
        &graph,
        config,
        &params.legs_dir,
        &format!("{stem}.pairs.jsonl"),
    )?;

    let payload = GraphSamplePayload {
        seed: config.seed,
        graph_edges_sha256: graph.edges_file.sha256.clone(),
        law_sha256: law_written.sha256,
        node_count: graph.nodes.len(),
        edge_count: graph.edges.len(),
        walk_length: config.walk_length,
        walks_per_node: config.walks_per_node,
        return_p: config.return_p,
        in_out_q: config.in_out_q,
        rung: RUNG,
        unit,
        take: params.take,
        graph_nodes_sha256: graph.nodes_file.sha256.clone(),
        edge_set_symmetric: graph.is_symmetric(),
        hard_negatives: config.hard_negatives,
        exclude_hops: (config.hard_negatives > 0).then_some(config.exclude_hops),
        warmup: params.warmup,
        iters_measured: params.iterations,
        walks: graph.nodes.len() * config.walks_per_node,
        sampled_pairs,
        pairs_file,
        walker: "jammi_ai::fine_tune::graph_sampler::GraphSampler",
    };
    let measured = Measured {
        iter_wall_s: Some(series.into_seconds()),
        work: Some(graph.edges.len() as f64),
        peak_rss_bytes,
        peak_vram_bytes: Measurement::not_yet_measured("bytes"),
        outcome_digest: Some(pairs_digest),
        law_observed: Some(law_observed),
        ..Default::default()
    };
    let leg = Leg::new(payload, cpu_provenance(), measured, Facts::default());
    let report = leg_report("graph-sample", leg.clone(), |leg| Tiers {
        graph_sample: Some(leg),
        ..Default::default()
    });
    let file = file_leg(&params.legs_dir, &stem, &report)?;
    Ok((leg, file))
}

/// `graph-sample`'s flags: the graphs to sample (one leg each) and the sampler
/// configuration every leg shares.
#[derive(Debug, Clone, clap::Args)]
pub struct GraphSampleArgs {
    /// A graph directory (`nodes.jsonl` + `edges.jsonl`). Repeat for a size
    /// sweep: each graph is then sampled in its own process.
    #[arg(long = "graph", required = true)]
    graphs: Vec<PathBuf>,
    /// Where the legs are filed, `sampler__edges<N>__r<take>.json`, each with
    /// its pair table beside it.
    #[arg(long)]
    legs_dir: PathBuf,
    /// Where each unit's law file `edges<N>.json` is written; `<legs-dir>/law`
    /// by default — what `ladder graph-sample --law-dir` reads.
    #[arg(long)]
    law_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 4)]
    walk_length: usize,
    #[arg(long, default_value_t = 4)]
    walks_per_node: usize,
    #[arg(long, default_value_t = 1.0)]
    return_p: f64,
    #[arg(long, default_value_t = 1.0)]
    in_out_q: f64,
    /// Hard negatives mined per pair; `0` times the walk alone, which is what
    /// the torch rung does.
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
    /// Measured repeats of each graph, each in a process of its own.
    #[arg(long, default_value_t = 1)]
    takes: usize,
    /// The take a single graph's run is filed as.
    #[arg(long, default_value_t = 1)]
    take: usize,
}

impl GraphSampleArgs {
    fn params(&self, graph: &Path, take: usize) -> GraphSampleParams {
        GraphSampleParams {
            graph: graph.to_path_buf(),
            legs_dir: self.legs_dir.clone(),
            law_dir: self.law_dir.clone(),
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
            take,
        }
    }

    /// The invocation of this binary that samples `graph` alone at `take`.
    fn child_args(&self, graph: &Path, take: usize) -> Vec<std::ffi::OsString> {
        let mut args: Vec<std::ffi::OsString> = vec![
            "graph-sample".into(),
            "--graph".into(),
            graph.into(),
            "--legs-dir".into(),
            (&self.legs_dir).into(),
        ];
        if let Some(law_dir) = &self.law_dir {
            args.extend(["--law-dir".into(), law_dir.into()]);
        }
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
            ("--take", take.to_string()),
        ];
        args.extend(
            flags
                .into_iter()
                .flat_map(|(flag, value)| [flag.into(), value.into()]),
        );
        args
    }

    /// Run the subcommand: one leg per (graph, take), and print the file names.
    pub async fn execute(&self) -> Result<(), Box<dyn std::error::Error>> {
        let points: Vec<(&Path, usize)> = self
            .graphs
            .iter()
            .flat_map(|g| (1..=self.takes).map(move |t| (g.as_path(), t)))
            .collect();
        let first = self.params(points[0].0, if self.takes == 1 { self.take } else { 1 });
        let files = legs_per_point(
            &points,
            async move { run_leg(&first).map(|(_, file)| vec![file]) },
            |&(graph, take)| self.child_args(graph, take),
        )
        .await?;
        println!("{}", serde_json::to_string_pretty(&files)?);
        Ok(())
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
        let (pairs, digest, rows) = write_pair_table(
            &GraphFiles::read(&self.graph)?,
            config,
            &self.out,
            "pairs.jsonl",
        )?;
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

    use crate::ladder::definition::Workload;
    use crate::ladder::verdict::OutcomeVerdict;
    use crate::ladder::{run_ladder, Axis, LadderArgs};

    /// The observed next-node frequencies of a count set, per state.
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

        let first = (None, "c".to_string());
        for x in ["a", "b", "d"] {
            assert!((probability(&law, first.clone(), x) - 1.0 / 3.0).abs() < 1e-15);
        }
        assert_eq!(
            law[&(Some("c".to_string()), "d".to_string())],
            vec![("c".to_string(), 1.0)]
        );
        assert_eq!(law.len(), 4 + edges.len());
        for nexts in law.values() {
            assert!((nexts.iter().map(|(_, p)| p).sum::<f64>() - 1.0).abs() < 1e-12);
        }

        // The law file is the cells in this order, and the counts follow it.
        let file: serde_json::Value = serde_json::from_slice(&law_file(&law)).unwrap();
        let cells = file["cells"].as_array().unwrap();
        assert_eq!(cells.len(), law.len());
        assert_eq!(
            cells[0].as_array().unwrap().len(),
            law.values().next().unwrap().len()
        );
        let mut counts = TransitionCounts::default();
        counts.observe(&["a".into(), "c".into(), "d".into(), "c".into()]);
        let observed = counts.observed_by(&law).unwrap();
        assert_eq!(observed.len(), law.len());
        assert_eq!(observed.iter().flatten().sum::<u64>(), 3);
        counts.observe(&["a".into(), "d".into()]);
        assert!(
            counts.observed_by(&law).is_err(),
            "a step off the law is refused"
        );
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

    fn params(graph: &Path, legs_dir: &Path, config: GraphSampleConfig) -> GraphSampleParams {
        GraphSampleParams {
            graph: graph.to_path_buf(),
            legs_dir: legs_dir.to_path_buf(),
            law_dir: None,
            config,
            warmup: 1,
            iterations: 3,
            take: 1,
        }
    }

    /// A leg over a written graph: filed under the ladder's name, the series of
    /// the asked length, the pair table the engine's sample at the seed in
    /// `_ordinal` order, the law file written and named by its sha256, and the
    /// counts in the law's order.
    #[test]
    fn leg_is_filed_with_its_pair_table_law_and_counts() {
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
        let legs = dir.path().join("legs");
        let (leg, file) = run_leg(&params(&dir.path().join("graph"), &legs, config)).unwrap();
        let unit = format!("edges{}", graph.edges.len());
        assert_eq!(file, format!("sampler__{unit}__r1.json"));
        assert!(legs.join(&file).is_file());
        assert_eq!(leg.measured.iter_wall_s.as_ref().unwrap().len(), 3);
        assert_eq!(leg.measured.work, Some(graph.edges.len() as f64));
        assert_eq!(leg.payload.exclude_hops, Some(config.exclude_hops));

        let law_file = artifact_of(&legs.join(LAW_DIR).join(format!("{unit}.json"))).unwrap();
        assert_eq!(law_file.sha256, leg.payload.law_sha256);
        let law = node2vec_transition_law(&graph.edges, config.return_p, config.in_out_q);
        let observed = leg.measured.law_observed.as_ref().unwrap();
        assert_eq!(observed.len(), law.len());
        assert_eq!(
            observed.iter().flatten().sum::<u64>() as usize,
            4 * graph.nodes.len() * config.walks_per_node * config.walk_length,
            "every step of every iteration's walks is counted"
        );

        let expected = graph.sampler(config).unwrap().sample().unwrap();
        let table: Vec<serde_json::Value> =
            read_jsonl(Path::new(&leg.payload.pairs_file.path)).unwrap();
        assert_eq!(table.len(), expected.len());
        for (i, (row, pair)) in table.iter().zip(&expected).enumerate() {
            assert_eq!(row["_ordinal"], i as u64);
            assert_eq!(row["anchor_id"], pair.anchor_id);
            assert_eq!(row["positive_text"], pair.positive);
            assert_eq!(row["negative_text"], pair.hard_negatives[0]);
        }

        // The filed report reads back as the ladder reads a leg.
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(legs.join(&file)).unwrap()).unwrap();
        let block = &report["tiers"]["graph_sample"];
        assert_eq!(block["law_sha256"], leg.payload.law_sha256);
        assert_eq!(block["rung"], "sampler");
        assert!(block["law_observed"].is_array());
    }

    /// The ladder runs end to end over legs this rung filed and reaches its
    /// law verdict. The reference rung's legs are stand-ins — this rung's own
    /// legs filed under `torch` — so the run exercises the comparator's path
    /// over the leg shape, not a second sampler.
    #[test]
    fn the_ladder_reaches_a_law_verdict_over_filed_legs() {
        let dir = tempfile::tempdir().unwrap();
        let legs = dir.path().join("legs");
        let committed = CommittedSample::load().unwrap();
        for nodes_per in [6, 9] {
            let graph_dir = dir.path().join(format!("g{nodes_per}"));
            SyntheticGraph {
                nodes_per,
                ..committed.graph
            }
            .write(&graph_dir)
            .unwrap();
            // Enough walks that every cell's expected count clears the fit's
            // floor of five: a state is entered about five times per
            // iteration at this size, and the rarest next node of a state has
            // probability near 1/50 under this `p`, `q`.
            let config = GraphSampleConfig {
                walk_length: 5,
                walks_per_node: 32,
                return_p: 0.25,
                in_out_q: 4.0,
                hard_negatives: 0,
                seed: 3,
                ..GraphSampleConfig::default()
            };
            let mut p = params(&graph_dir, &legs, config);
            (p.warmup, p.iterations) = (0, 40);
            let (_, file) = run_leg(&p).unwrap();
            std::fs::copy(
                legs.join(&file),
                legs.join(file.replacen("sampler__", "torch__", 1)),
            )
            .unwrap();
        }
        let verdict = run_ladder(&LadderArgs {
            workload: Workload::GraphSample,
            legs_dir: legs.clone(),
            out: None,
            from: None,
            to: None,
            axes: vec![Axis::Outcome],
            waive_control: false,
            law_dir: Some(legs.join(LAW_DIR)),
            mutants: vec![],
            revision: None,
        })
        .unwrap();
        assert!(verdict.refusals.is_empty(), "{:?}", verdict.refusals);
        assert_eq!(verdict.edges.len(), 1);
        let edge = &verdict.edges[0];
        assert!(edge.refusals.is_empty(), "{:?}", edge.refusals);
        match &edge.outcome {
            Some(OutcomeVerdict::Law {
                alpha,
                lower,
                upper,
            }) => {
                assert!(
                    lower.p_value >= *alpha && upper.p_value >= *alpha,
                    "{lower:?} {upper:?}"
                );
            }
            other => panic!("expected a law verdict, got {other:?}"),
        }
    }

    /// A leg over the committed graph at `config`, filed under a fresh tempdir.
    fn committed_leg(
        committed: &CommittedSample,
        config: GraphSampleConfig,
    ) -> (Leg<GraphSamplePayload>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        committed.graph.write(&dir.path().join("graph")).unwrap();
        let mut p = params(&dir.path().join("graph"), &dir.path().join("legs"), config);
        (p.warmup, p.iterations) = (0, 1);
        let (leg, _) = run_leg(&p).unwrap();
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
            leg.measured.outcome_digest.as_deref(),
            Some(committed.digest.as_str()),
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
                committed_leg(&committed, config)
                    .0
                    .measured
                    .outcome_digest
                    .as_deref(),
                Some(committed.digest.as_str()),
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
        let rows: Vec<serde_json::Value> =
            read_jsonl(Path::new(&leg.payload.pairs_file.path)).unwrap();
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
        assert!(write_pair_table(&graph, config, dir.path(), "pairs.jsonl").is_err());
    }
}
