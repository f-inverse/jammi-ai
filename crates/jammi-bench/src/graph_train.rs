//! The CPU-hermetic graph fine-tune tier: the engine's biased-walk graph sampler
//! (`jammi_ai::fine_tune::graph_sampler::GraphSampler`) — the data path
//! `fine_tune_graph` threads through — measured for throughput and gated for
//! determinism over a committed synthetic graph.
//!
//! This is the graph-supervision peer of [`crate::train_scale`]. Where that tier
//! measures the engine's GradCache backward + AdamW step (the *optimisation* half
//! of a fine-tune), this one measures the engine's `GraphSampler` (the
//! graph-structured *data* half `fine_tune_graph` is built on): the node2vec
//! biased walks that draw positives and the structure-aware k-hop negative
//! mining. The downstream MNRL/GradCache optimisation a graph fine-tune then runs
//! over these pairs is the generic train path the training tier already gates —
//! the distinct, graph-specific engine work this tier exists to gate is the
//! sampler, so it is the sampler this tier drives.
//!
//! ## Two lanes — a same-box rate and a portable digest
//!
//! * **Throughput** — the `(anchor, positive, hard_negatives)` rows the sampler
//!   draws per second over the committed graph. A *rate* (a property of the box),
//!   gated against a committed same-box baseline by [`crate::rate_gate`], the
//!   training tier's discipline — never a portable floor.
//! * **The determinism digest** — `GraphSampler::sample` is seeded (a
//!   `SplitMix64` walk/negative stream), so its pair set is byte-stable across
//!   runs (the engine's `graph_spec_round_trip_resamples_identical_pairs`
//!   contract). The committed digest is a stable checksum of the sampled rows;
//!   the `cargo test` gate exercises BOTH directions — the real sampler
//!   reproducing the committed digest, and a perturbed sample (a different seed,
//!   a different walk length) producing a different digest — so the gate is
//!   non-vacuous, the propagation-digest idiom applied to the sampler.
//!
//! ## Why a committed *spec*, not a committed digest constant
//!
//! The synthetic graph is generated deterministically from the committed spec
//! (community count, nodes per community, intra/inter-community edge density,
//! seed) plus the sampler knobs, so the committed artifact is the *generation
//! spec* plus the digest the sampler produced over it — never a hand-written
//! digest. The gate regenerates the same graph, re-samples it through the real
//! engine, and asserts the digest matches; the rebuild path re-derives both the
//! digest and the same-box baseline rate byte-for-byte.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use jammi_ai::fine_tune::graph_sampler::{
    GraphEdge, GraphSampleConfig, GraphSampler, SampledPair, TextNode,
};

use crate::report::{DigestGate, GraphTrainTier, Measurement, RateVerdict};

/// The committed graph fine-tune spec: the synthetic-graph generation parameters
/// and the sampler knobs the gated throughput and digest are measured over, plus
/// the committed digest and the same-box baseline rate. The on-disk
/// `baselines/graph_train.json` the tier and its gate read.
///
/// Nothing here is a hand-written digest: `digest` is the checksum the engine's
/// real `GraphSampler::sample` produced over the graph this spec regenerates, and
/// `baseline_pairs_per_s` is the same-box rate that sample ran at when the spec
/// was cut.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTrainSpec {
    /// Number of homophilous communities the synthetic graph wires within. Each
    /// community is an intra-dense cluster; sparse bridges join them, so the graph
    /// has the multi-scale structure node2vec walks are meant to capture.
    pub communities: usize,
    /// Nodes per community. The total node count is `communities · nodes_per`.
    pub nodes_per: usize,
    /// Intra-community fan-out: each node wires to its next `intra_degree`
    /// community-mates cyclically, so the within-community subgraph is a bounded
    /// circulant (degree-regular but multi-hop), not a clique — the edge set stays
    /// `O(nodes · intra_degree)`.
    pub intra_degree: usize,
    /// Inter-community bridges: every `bridge_stride`-th node also wires to the
    /// matching-index node of the next community, so the communities are joined
    /// sparsely (the structure a DFS-biased walk crosses and a BFS-biased walk
    /// does not).
    pub bridge_stride: usize,
    /// The sampler walk length `L` (node2vec). `L > 1` is the higher-order
    /// community-capturing regime.
    pub walk_length: usize,
    /// Walks started per node — more walks, more pairs, lower-variance positives.
    pub walks_per_node: usize,
    /// node2vec return parameter `p`.
    pub return_p: f64,
    /// node2vec in-out parameter `q`.
    pub in_out_q: f64,
    /// Structure-aware hard negatives mined per `(anchor, positive)` pair.
    pub hard_negatives: usize,
    /// Hops of the anchor's neighbourhood excluded from its negative pool (the
    /// false-negative guard).
    pub exclude_hops: usize,
    /// The seeded walk/negative RNG seed — the determinism anchor.
    pub seed: u64,
    /// The committed digest: the checksum of the sampled pair set the engine
    /// produced over the generated graph when the spec was cut.
    pub digest: String,
    /// The committed same-box baseline throughput, sampled pairs/s. A *rate* is
    /// not portable — this is a same-box reference refreshed by the rebuilder.
    pub baseline_pairs_per_s: f64,
}

impl GraphTrainSpec {
    /// The crate-relative path to the committed spec.
    pub fn path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("baselines")
            .join("graph_train.json")
    }

    /// Load the committed spec from `baselines/graph_train.json`.
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(Self::path())?;
        Ok(serde_json::from_str(&json)?)
    }

    /// The sampler configuration this spec drives the real `GraphSampler` with.
    fn sample_config(&self) -> GraphSampleConfig {
        GraphSampleConfig {
            walk_length: self.walk_length,
            walks_per_node: self.walks_per_node,
            return_p: self.return_p,
            in_out_q: self.in_out_q,
            hard_negatives: self.hard_negatives,
            exclude_hops: self.exclude_hops,
            // A single negative is always drawable on a graph this size; the
            // collapse guard is not the property under test here.
            min_negatives: 1,
            seed: self.seed,
        }
    }
}

/// Build the committed synthetic graph: `communities` homophilous clusters of
/// `nodes_per` text-bearing nodes each, wired intra-densely by a bounded circulant
/// and joined by sparse inter-community bridges.
///
/// Node ids are `c{community}_{i}`; each node's text is **distinct per node**
/// (`document c{c} node {i}`) so the sampled-pair digest is sensitive to the exact
/// node selection a biased walk makes, not merely the community — a per-community
/// shared text would let a walk-bias regression that re-picks within a community
/// slip the digest. The text *names the community* so the structural in-community
/// concentration is still readable (the homophily property the walk amplifies).
/// The wiring is a pure function of the spec, so the same graph regenerates on any
/// box — the sampler's determinism then makes the sampled pair set byte-stable.
fn build_graph(spec: &GraphTrainSpec) -> (Vec<TextNode>, Vec<GraphEdge>) {
    let mut nodes = Vec::with_capacity(spec.communities * spec.nodes_per);
    for c in 0..spec.communities {
        for i in 0..spec.nodes_per {
            nodes.push(TextNode::new(
                format!("c{c}_{i}"),
                format!("document c{c} node {i}"),
            ));
        }
    }

    let mut edges = Vec::new();
    let id = |c: usize, i: usize| format!("c{c}_{i}");
    for c in 0..spec.communities {
        for i in 0..spec.nodes_per {
            // Intra-community circulant: wire to the next `intra_degree`
            // community-mates (both directions, so a walk can traverse).
            let reach = spec.intra_degree.min(spec.nodes_per.saturating_sub(1));
            for off in 1..=reach {
                let j = (i + off) % spec.nodes_per;
                edges.push(GraphEdge::declared(id(c, i), id(c, j)));
                edges.push(GraphEdge::declared(id(c, j), id(c, i)));
            }
            // Sparse inter-community bridge: every `bridge_stride`-th node joins
            // the matching node of the next community.
            if spec.bridge_stride > 0 && i % spec.bridge_stride == 0 && spec.communities > 1 {
                let next = (c + 1) % spec.communities;
                edges.push(GraphEdge::declared(id(c, i), id(next, i)));
                edges.push(GraphEdge::declared(id(next, i), id(c, i)));
            }
        }
    }
    (nodes, edges)
}

/// The stable checksum of a sampled pair set: an FNV-1a hash over the sampler's
/// `(anchor, positive, hard_negatives)` rows in emit order, rendered as a
/// fixed-width hex string.
///
/// Pure arithmetic over the sampled rows' text bytes, no crate — the sampler
/// analogue of the propagation digest's fold. Because `GraphSampler::sample` is
/// seeded and deterministic, this digest is a stable reference: any change to the
/// sampled rows (a different walk bias, a different negative pool, a different
/// emit order) flips it. Separator bytes between fields and between rows so two
/// different row layouts cannot collide.
fn digest(pairs: &[SampledPair]) -> String {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    let mut mix = |byte: u8| {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    };
    for pair in pairs {
        for b in pair.anchor.bytes() {
            mix(b);
        }
        mix(0xff);
        for b in pair.positive.bytes() {
            mix(b);
        }
        mix(0xfe);
        for neg in &pair.hard_negatives {
            for b in neg.bytes() {
                mix(b);
            }
            mix(0xfd);
        }
        // A row terminator so adjacent rows cannot run together.
        mix(0x00);
    }
    format!("{hash:016x}")
}

/// Sample the committed graph through the real engine `GraphSampler` at an
/// explicit seed and walk length, returning the sampled pairs. The seam the
/// gate-fails test re-samples the SAME graph at a perturbed knob through, to
/// observe the digest move; the gate path itself measures through [`measure`].
#[cfg(test)]
fn sample_graph(
    spec: &GraphTrainSpec,
    seed: u64,
    walk_length: usize,
) -> Result<Vec<SampledPair>, Box<dyn std::error::Error>> {
    let (nodes, edges) = build_graph(spec);
    let mut config = spec.sample_config();
    config.seed = seed;
    config.walk_length = walk_length;
    let sampler = GraphSampler::build(nodes, edges, config)?;
    Ok(sampler.sample()?)
}

/// Re-sample the committed graph through the real engine sampler at the committed
/// knobs and return the digest of the sampled pair set — the DIGEST-CLEARS seam
/// the teeth test asserts the committed value against.
#[cfg(test)]
fn sample_digest(spec: &GraphTrainSpec) -> Result<String, Box<dyn std::error::Error>> {
    Ok(digest(&sample_graph(spec, spec.seed, spec.walk_length)?))
}

/// One graph-sample measurement: the throughput, the sample wall-time, the
/// sampled-pair count and the graph it was drawn over, and the digest of the
/// sampled rows. The unit `measure` returns so the rate, the report facts, and
/// the digest travel together rather than as an opaque tuple.
struct SampleMeasurement {
    pairs_per_s: f64,
    wall_ms: f64,
    sampled_pairs: usize,
    nodes: usize,
    edges: usize,
    digest: String,
}

/// Measure the graph-sample throughput over the committed graph: sample once
/// through the real engine sampler and divide the sampled-pair count by the
/// sample wall-clock.
fn measure(spec: &GraphTrainSpec) -> Result<SampleMeasurement, Box<dyn std::error::Error>> {
    let (nodes, edges) = build_graph(spec);
    let node_count = nodes.len();
    let edge_count = edges.len();
    let mut config = spec.sample_config();
    config.seed = spec.seed;
    config.walk_length = spec.walk_length;
    let sampler = GraphSampler::build(nodes, edges, config)?;

    let start = Instant::now();
    let pairs = sampler.sample()?;
    let elapsed = start.elapsed();

    let wall_ms = elapsed.as_secs_f64() * 1_000.0;
    let pairs_per_s = if elapsed.as_secs_f64() > 0.0 {
        pairs.len() as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };
    Ok(SampleMeasurement {
        pairs_per_s,
        wall_ms,
        sampled_pairs: pairs.len(),
        nodes: node_count,
        edges: edge_count,
        digest: digest(&pairs),
    })
}

/// Run the graph fine-tune tier against the committed spec: sample the graph
/// through the real engine sampler, gate the sampled-pair digest for equality
/// against the committed digest, and gate the measured throughput against the
/// committed same-box baseline.
///
/// This is the path the `graph-train-scale` subcommand drives and the `cargo
/// test` gate asserts: the digest is the engine's own sampled-pair checksum gated
/// for byte-equality; the throughput is a same-box rate gated by the relative-drop
/// [`crate::rate_gate`].
pub fn run(spec: &GraphTrainSpec) -> Result<GraphTrainTier, Box<dyn std::error::Error>> {
    let m = measure(spec)?;

    let gate = crate::rate_gate::RateGate::evaluate(
        m.pairs_per_s,
        spec.baseline_pairs_per_s,
        crate::rate_gate::DEFAULT_REGRESSION_THRESHOLD,
    );

    Ok(GraphTrainTier {
        nodes: m.nodes,
        edges: m.edges,
        sampled_pairs: m.sampled_pairs,
        pairs_per_s: Measurement::measured(m.pairs_per_s, "pairs_per_s"),
        sample_wall_ms: Measurement::measured(m.wall_ms, "ms"),
        rate_gate: Some(RateVerdict {
            measured_pairs_per_s: gate.measured,
            baseline_pairs_per_s: gate.baseline,
            threshold: gate.threshold,
            floor_pairs_per_s: gate.floor,
            passed: gate.passed,
            detail: gate.detail(),
        }),
        digest: DigestGate {
            passed: m.digest == spec.digest,
            measured: m.digest,
            committed: spec.digest.clone(),
        },
    })
}

/// Whether both gates held — the verdict the subcommand maps to its exit code and
/// the `cargo test` gate asserts: the determinism digest matched the committed
/// value AND the throughput cleared the same-box floor.
pub fn gates_passed(tier: &GraphTrainTier) -> bool {
    tier.digest.passed && tier.rate_gate.as_ref().is_none_or(|v| v.passed)
}

/// The generation parameters a rebuild draws the committed spec from — every
/// field of [`GraphTrainSpec`] except the two gated values the rebuild derives
/// (the digest and the same-box baseline rate). Passed as one struct so the
/// rebuilder takes the graph/sampler shape as a unit, not a long argument list.
#[derive(Debug, Clone, Copy)]
pub struct GraphTrainParams {
    /// Homophilous communities the synthetic graph wires within.
    pub communities: usize,
    /// Nodes per community.
    pub nodes_per: usize,
    /// Intra-community circulant fan-out.
    pub intra_degree: usize,
    /// Inter-community bridge stride.
    pub bridge_stride: usize,
    /// node2vec walk length.
    pub walk_length: usize,
    /// Walks started per node.
    pub walks_per_node: usize,
    /// node2vec return parameter `p`.
    pub return_p: f64,
    /// node2vec in-out parameter `q`.
    pub in_out_q: f64,
    /// Structure-aware hard negatives per pair.
    pub hard_negatives: usize,
    /// k-hop false-negative exclusion radius.
    pub exclude_hops: usize,
    /// The seeded walk/negative RNG seed.
    pub seed: u64,
}

/// Re-derive the committed spec from a fresh sample: regenerate the graph, sample
/// it through the engine, and record the digest and the same-box throughput. The
/// off-box one-shot that writes `baselines/graph_train.json`; CI only ever loads
/// and re-samples it.
pub fn rebuild_spec(
    params: GraphTrainParams,
) -> Result<GraphTrainSpec, Box<dyn std::error::Error>> {
    // A spec with placeholder gated values, so `measure` can regenerate the same
    // graph; the real digest and baseline replace the placeholders below.
    let mut spec = GraphTrainSpec {
        communities: params.communities,
        nodes_per: params.nodes_per,
        intra_degree: params.intra_degree,
        bridge_stride: params.bridge_stride,
        walk_length: params.walk_length,
        walks_per_node: params.walks_per_node,
        return_p: params.return_p,
        in_out_q: params.in_out_q,
        hard_negatives: params.hard_negatives,
        exclude_hops: params.exclude_hops,
        seed: params.seed,
        digest: String::new(),
        baseline_pairs_per_s: 0.0,
    };
    let m = measure(&spec)?;
    spec.digest = m.digest;
    spec.baseline_pairs_per_s = m.pairs_per_s;
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The committed spec is well-formed: a structured multi-community graph, a
    /// higher-order walk, a non-empty digest of the FNV width, and a positive
    /// baseline rate.
    #[test]
    fn committed_spec_is_well_formed() {
        let spec = GraphTrainSpec::load().expect("baselines/graph_train.json must be present");
        assert!(
            spec.communities >= 2,
            "the graph needs multiple communities"
        );
        assert!(spec.nodes_per >= 2, "a community needs an intra-edge");
        assert!(spec.intra_degree >= 1);
        assert!(spec.walk_length >= 2, "a higher-order walk (L > 1)");
        assert!(spec.walks_per_node >= 1);
        assert!(spec.exclude_hops >= 1, "the false-negative guard");
        assert_eq!(spec.digest.len(), 16, "digest is a 64-bit FNV hex string");
        assert!(
            spec.digest.chars().all(|c| c.is_ascii_hexdigit()),
            "digest must be hex"
        );
        assert!(
            spec.baseline_pairs_per_s > 0.0,
            "committed baseline rate must be positive"
        );
    }

    /// The teeth, DIGEST-CLEARS direction: re-sampling the committed graph through
    /// the engine's real `GraphSampler` reproduces the committed digest. A
    /// regression in the walk bias, the negative mining, or the adjacency moves the
    /// rows and trips this.
    #[test]
    fn resample_matches_committed_digest() {
        let spec = GraphTrainSpec::load().expect("baselines/graph_train.json must be present");
        let tier = run(&spec).expect("graph-train tier runs over the spec");
        assert!(
            tier.digest.passed,
            "the re-sampled graph digest drifted off the committed one: measured {} vs committed {}",
            tier.digest.measured, tier.digest.committed
        );
        assert_eq!(tier.digest.measured, spec.digest);
    }

    /// The teeth, GATE-FAILS direction (RC1: an assertion must be able to fail).
    ///
    /// Perturbed samples — the SAME committed graph re-sampled through the SAME
    /// real engine sampler with a regressed knob — must each produce a different
    /// digest, proving the gate catches the sampler regressions it exists to catch:
    ///
    /// * a **different seed** — the walk/negative RNG stream, the determinism
    ///   anchor a re-seeding bug would break.
    /// * a **different walk length** — the node2vec walk depth, which changes the
    ///   reachable positives (a shallower or deeper community capture).
    ///
    /// The sampler at the committed knobs reproduces the committed digest — the
    /// contrast that gives each perturbation its teeth.
    #[test]
    fn perturbed_sample_changes_the_digest() {
        let spec = GraphTrainSpec::load().expect("baselines/graph_train.json must be present");

        let other_seed = sample_digest(&GraphTrainSpec {
            seed: spec.seed.wrapping_add(1),
            ..spec.clone()
        })
        .expect("re-seeded sample runs");
        assert_ne!(
            other_seed, spec.digest,
            "a different sampler seed must change the digest (else a re-seeding regression slips)"
        );

        let shorter_walk = digest(
            &sample_graph(&spec, spec.seed, spec.walk_length - 1)
                .expect("shorter-walk sample runs"),
        );
        assert_ne!(
            shorter_walk, spec.digest,
            "a shorter walk must change the digest (else a walk-depth regression slips)"
        );

        // The sampler at the committed knobs reproduces the committed digest.
        let correct = sample_digest(&spec).expect("correct sample runs");
        assert_eq!(
            correct, spec.digest,
            "the sampler at the committed knobs must reproduce the committed digest"
        );
    }

    /// The gate-fails direction at the harness level: a tampered committed digest
    /// fails [`gates_passed`], proving the verdict reacts to the committed value,
    /// not just to the re-sample.
    #[test]
    fn tampered_committed_digest_fails_the_gate() {
        let mut spec = GraphTrainSpec::load().expect("baselines/graph_train.json must be present");
        spec.digest = "deadbeefdeadbeef".to_string();
        let tier = run(&spec).expect("tier still runs");
        assert!(
            !gates_passed(&tier),
            "a tampered committed digest must trip the gate"
        );
    }

    /// The committed throughput baseline gates with teeth: a run *at* the baseline
    /// clears the gate and a run that regressed past the threshold *fails* it
    /// (RC1). Asserts the committed baseline is a well-formed, generously
    /// thresholded same-box reference without re-measuring the (noisy) rate in the
    /// hermetic test lane.
    #[test]
    fn committed_baseline_gates_with_teeth() {
        use crate::rate_gate::{RateGate, DEFAULT_REGRESSION_THRESHOLD};

        let spec = GraphTrainSpec::load().expect("baselines/graph_train.json must be present");
        let rate = spec.baseline_pairs_per_s;
        assert!(rate > 0.0, "committed baseline rate must be positive");

        // A run at the baseline clears its own gate.
        assert!(
            RateGate::evaluate(rate, rate, DEFAULT_REGRESSION_THRESHOLD).passed,
            "a run at the committed baseline must pass the gate"
        );
        // A run just below the derived floor fails — the gate has teeth.
        let floor = rate * (1.0 - DEFAULT_REGRESSION_THRESHOLD);
        assert!(
            !RateGate::evaluate(floor - 1.0, rate, DEFAULT_REGRESSION_THRESHOLD).passed,
            "a rate below the floor must fail the gate"
        );
        // A faster-than-baseline run passes.
        assert!(
            RateGate::evaluate(rate * 1.1, rate, DEFAULT_REGRESSION_THRESHOLD).passed,
            "a faster-than-baseline run must pass"
        );
    }

    /// The sampler does real graph-structured work: the committed graph samples a
    /// non-empty pair set, every anchor's positive is a distinct node, and (given
    /// the homophilous wiring) positives concentrate in-community — so the digest
    /// is over a meaningful biased-walk output, not a degenerate empty or
    /// identity sample.
    #[test]
    fn sampler_produces_meaningful_structured_pairs() {
        let spec = GraphTrainSpec::load().expect("baselines/graph_train.json must be present");
        let pairs = sample_graph(&spec, spec.seed, spec.walk_length).expect("sample runs");
        assert!(!pairs.is_empty(), "the committed graph must sample pairs");
        for p in &pairs {
            // Distinct per-node text, so a pair's positive text differs from its
            // anchor's — the digest resolves the exact node selection.
            assert_ne!(p.anchor, p.positive, "a pair's positive is a distinct node");
        }
        // The text names the community (`document c{c} node {i}`); a homophilous
        // graph keeps most positives in-community (the structural property a
        // biased walk amplifies).
        let in_community = pairs
            .iter()
            .filter(|p| community_of(&p.anchor) == community_of(&p.positive))
            .count();
        let ratio = in_community as f64 / pairs.len() as f64;
        assert!(
            ratio > 0.7,
            "biased-walk positives should stay mostly in-community on a homophilous graph, got {ratio}"
        );
    }

    /// The community token (`c{c}`) of a node's sampled-pair text
    /// (`document c{c} node {i}`), used by the structural-property check.
    fn community_of(text: &str) -> &str {
        text.split(' ').nth(1).unwrap_or("")
    }

    /// `rebuild_spec` is the inverse of the gate: the spec it derives, re-run
    /// through the gate, passes — the digest it writes is, by construction, the
    /// exact sample the gate re-computes. Guards the off-box rebuilder against
    /// drifting from the committed idiom.
    #[test]
    fn rebuild_spec_round_trips_through_the_gate() {
        let spec = GraphTrainSpec::load().expect("baselines/graph_train.json must be present");
        let rebuilt = rebuild_spec(GraphTrainParams {
            communities: spec.communities,
            nodes_per: spec.nodes_per,
            intra_degree: spec.intra_degree,
            bridge_stride: spec.bridge_stride,
            walk_length: spec.walk_length,
            walks_per_node: spec.walks_per_node,
            return_p: spec.return_p,
            in_out_q: spec.in_out_q,
            hard_negatives: spec.hard_negatives,
            exclude_hops: spec.exclude_hops,
            seed: spec.seed,
        })
        .expect("rebuild runs");
        assert_eq!(
            rebuilt.digest, spec.digest,
            "a rebuild over the same parameters must reproduce the committed digest"
        );
        let tier = run(&rebuilt).expect("tier runs over the rebuilt spec");
        assert!(
            tier.digest.passed,
            "a freshly rebuilt spec must pass its digest gate"
        );
    }
}
