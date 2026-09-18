//! S11 — graph-supervised fine-tune integration tests.
//!
//! Two layers:
//! 1. **Sampler / loader structural contracts** (hermetic, no model): the
//!    biased-walk positive sampler, the k-hop false-negative guard, and the
//!    load-bearing **circularity** distinction between declared and similarity
//!    edge supervision. These run in milliseconds on tiny synthetic graphs.
//! 2. **End-to-end session path** (`tiny_bert`): `fine_tune_graph` threads a
//!    real graph (node CSV + edge CSV) through the existing trainer to a
//!    completed job + saved adapter — proving a graph sample drives the
//!    MNRL/Triplet path with no new loss.
//!
//! ## Circularity — what is demonstrated vs documented
//!
//! The full R1 contract ("declared-edge supervision yields a *statistically
//! significant* held-out gain, while S9-similarity-only supervision yields a
//! *near-zero* gain") needs real training on a real golden set and a paired
//! significance test — too heavy for a bounded hermetic test. What is
//! demonstrated here, deterministically:
//! - the sampler tracks edge provenance and separates declared from similarity
//!   supervision ([`circularity_declared_vs_similarity_is_separated`]);
//! - on a synthetic homophilous graph, biased-walk positives raise the
//!   in-community pair rate over cross-community pairs
//!   ([`walk_positives_concentrate_in_community`]) — the structural property a
//!   declared-edge fine-tune then amplifies, and the property an S9-similarity
//!   graph cannot add (its edges were drawn by the base metric it would
//!   re-learn).
//!
//! ### Full R1 protocol (for the real eval, not run here)
//! 1. Build two supervision graphs over the same nodes: one from declared edges
//!    (hierarchy/crosswalk/citation/confirmed pairs), one from S9 k-NN edges.
//! 2. `fine_tune_graph` each into a model; hold out a golden relevance set.
//! 3. `eval_embeddings` base vs each fine-tune at k; paired bootstrap / t-test.
//! 4. Assert the declared-edge gain is significant and the similarity-edge gain
//!    is near-zero — the degenerate feedback loop, measured.

use std::sync::Arc;

use jammi_ai::fine_tune::data::{TrainingDataLoader, TrainingFormat};
use jammi_ai::fine_tune::graph_sampler::{
    EdgeProvenance, GraphEdge, GraphFineTuneSources, GraphSampleConfig, GraphSampler, TextNode,
};
use jammi_ai::session::InferenceSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use tempfile::TempDir;

use crate::common;

/// A homophilous two-community graph: `a*` nodes form one clique, `b*` another,
/// joined by one bridge edge. All edges `Declared` by default.
fn two_communities(provenance: EdgeProvenance) -> (Vec<TextNode>, Vec<GraphEdge>) {
    let a = ["a0", "a1", "a2", "a3"];
    let b = ["b0", "b1", "b2", "b3"];
    let nodes = a
        .iter()
        .chain(b.iter())
        .map(|id| TextNode::new(*id, format!("document about topic {id}")))
        .collect();

    let edge = |s: &str, d: &str| GraphEdge {
        src: s.to_string(),
        dst: d.to_string(),
        provenance,
    };
    let mut edges = Vec::new();
    for clique in [a, b] {
        for (i, s) in clique.iter().enumerate() {
            for (j, d) in clique.iter().enumerate() {
                if i != j {
                    edges.push(edge(s, d));
                }
            }
        }
    }
    edges.push(edge("a0", "b0"));
    edges.push(edge("b0", "a0"));
    (nodes, edges)
}

/// The load-bearing circularity contract, demonstrated structurally: the
/// sampler separates declared from similarity supervision, and a similarity-only
/// graph self-reports as carrying no declared signal (a weak bootstrap, never
/// the sole supervision).
#[test]
fn circularity_declared_vs_similarity_is_separated() {
    let cfg = GraphSampleConfig::default();

    let (nodes, edges) = two_communities(EdgeProvenance::Declared);
    let declared = GraphSampler::build(nodes, edges, cfg).unwrap();
    assert!(
        declared.has_declared_supervision(),
        "declared-edge graph must report declared supervision (genuine signal)"
    );

    let (nodes, edges) = two_communities(EdgeProvenance::Similarity);
    let similarity = GraphSampler::build(nodes, edges, cfg).unwrap();
    assert!(
        !similarity.has_declared_supervision(),
        "similarity-only graph must report NO declared supervision — training on \
         it largely re-learns the base metric (the degenerate feedback loop)"
    );

    // Both still produce a trainable dataset (similarity edges are a valid weak
    // bootstrap), so the distinction is provenance, not producibility.
    assert!(!declared.sample().unwrap().is_empty());
    assert!(!similarity.sample().unwrap().is_empty());
}

/// Biased-walk positives concentrate in-community on a homophilous graph — the
/// structural property graph-supervised fine-tune amplifies (neighbours pulled
/// together). Walk-based (L>1) reaches beyond the immediate neighbour, so this
/// is the higher-order node2vec property, not 1-hop.
#[test]
fn walk_positives_concentrate_in_community() {
    let (nodes, edges) = two_communities(EdgeProvenance::Declared);
    let cfg = GraphSampleConfig {
        walk_length: 4,
        walks_per_node: 8,
        hard_negatives: 0,
        seed: 7,
        ..GraphSampleConfig::default()
    };
    let sampler = GraphSampler::build(nodes, edges, cfg).unwrap();
    let pairs = sampler.sample().unwrap();

    let a_pairs: Vec<_> = pairs
        .iter()
        .filter(|p| p.anchor.contains("topic a"))
        .collect();
    assert!(!a_pairs.is_empty());
    let in_community = a_pairs
        .iter()
        .filter(|p| p.positive.contains("topic a"))
        .count();
    let ratio = in_community as f64 / a_pairs.len() as f64;
    assert!(
        ratio > 0.7,
        "walk positives for a-community anchors should stay mostly in-community, \
         got {ratio}"
    );
}

/// The false-negative guard: a sampled negative for an anchor is never inside
/// the anchor's excluded k-hop neighbourhood (a node there is likely a missing
/// edge — a true positive that would supply a false-negative gradient).
#[test]
fn negatives_respect_k_hop_exclusion() {
    let (nodes, edges) = two_communities(EdgeProvenance::Declared);
    let cfg = GraphSampleConfig {
        walk_length: 2,
        walks_per_node: 6,
        hard_negatives: 2,
        exclude_hops: 1,
        seed: 31,
        ..GraphSampleConfig::default()
    };
    let sampler = GraphSampler::build(nodes, edges, cfg).unwrap();
    let pairs = sampler.sample().unwrap();

    // a0's 1-hop neighbourhood: a0, a1, a2, a3 (clique) + b0 (bridge).
    let excluded = [
        "document about topic a0",
        "document about topic a1",
        "document about topic a2",
        "document about topic a3",
        "document about topic b0",
    ];
    for pair in pairs
        .iter()
        .filter(|p| p.anchor == "document about topic a0")
    {
        for neg in &pair.hard_negatives {
            assert!(
                !excluded.contains(&neg.as_str()),
                "negative {neg} is inside a0's excluded 1-hop neighbourhood"
            );
        }
    }
}

/// `TrainingDataLoader::from_graph` yields the right format/shape for the
/// existing trainer: a graph with mined hard negatives is a `Graph {
/// has_negatives: true }` whose in-batch view exposes explicit negatives (the
/// Triplet/MNRL path), while one without mining is `Graph { has_negatives: false
/// }` exposing none (the Pairs/MNRL path). No new loss is involved — the loader
/// is the only S11 change to the data path.
#[test]
fn from_graph_loader_threads_pairs_and_triplet_shapes() {
    // With hard negatives → Triplet shape.
    let (nodes, edges) = two_communities(EdgeProvenance::Declared);
    let cfg = GraphSampleConfig {
        hard_negatives: 1,
        seed: 1,
        ..GraphSampleConfig::default()
    };
    let sampler = GraphSampler::build(nodes, edges, cfg).unwrap();
    let loader = TrainingDataLoader::from_graph(&sampler).unwrap();
    assert!(matches!(loader.format(), TrainingFormat::Triplet));
    let (anchors, positives, negatives) = loader.in_batch_negative_texts().unwrap();
    assert_eq!(anchors.len(), positives.len());
    assert!(
        negatives.is_some(),
        "a mined-negative graph exposes explicit negatives (Triplet/MNRL path)"
    );

    // Without hard negatives → Pairs shape.
    let (nodes, edges) = two_communities(EdgeProvenance::Declared);
    let cfg = GraphSampleConfig {
        hard_negatives: 0,
        seed: 1,
        ..GraphSampleConfig::default()
    };
    let sampler = GraphSampler::build(nodes, edges, cfg).unwrap();
    let loader = TrainingDataLoader::from_graph(&sampler).unwrap();
    assert!(matches!(loader.format(), TrainingFormat::Pairs));
    let (_, _, negatives) = loader.in_batch_negative_texts().unwrap();
    assert!(
        negatives.is_none(),
        "a no-mining graph exposes no explicit negatives (Pairs/MNRL in-batch path)"
    );
}

/// A graph job is durable: the submitter persists a `TrainingSpec::GraphFineTune`
/// and a worker reconstructs the run from that JSON alone. The reconstruction
/// must be deterministic — two runs from the *same persisted spec* re-sample the
/// identical pairs (the seed lives in `sample_config`), or a re-claimed job after
/// a lost lease would train on different data than the first attempt. This is the
/// job-round-trip determinism contract: serialise the spec, deserialise it twice,
/// and assert the rebuilt sampler yields byte-identical pairs both times.
#[test]
fn graph_spec_round_trip_resamples_identical_pairs() {
    use jammi_ai::fine_tune::spec::TrainingSpec;

    let sources = GraphFineTuneSources {
        node_source: "nodes".into(),
        id_column: "id".into(),
        text_column: "text".into(),
        edge_source: "edges".into(),
        src_column: "src".into(),
        dst_column: "dst".into(),
        provenance: EdgeProvenance::Declared,
    };
    let sample_config = GraphSampleConfig {
        walk_length: 4,
        walks_per_node: 6,
        hard_negatives: 2,
        exclude_hops: 1,
        seed: 4242,
        ..GraphSampleConfig::default()
    };
    let spec = TrainingSpec::GraphFineTune {
        sources,
        sample_config,
        common: jammi_ai::fine_tune::spec::TrainingCommon {
            base_model: "local:tiny".into(),
            config: jammi_ai::fine_tune::FineTuneConfig::default(),
            world_size: jammi_ai::fine_tune::spec::DEFAULT_WORLD_SIZE,
        },
    };

    // Persist exactly as the submit path does, then reconstruct twice — the two
    // independent deserialisations stand in for two worker attempts at the job.
    let json = serde_json::to_string(&spec).unwrap();
    let resample = |json: &str| -> Vec<jammi_ai::fine_tune::graph_sampler::SampledPair> {
        let TrainingSpec::GraphFineTune {
            sources,
            sample_config,
            ..
        } = serde_json::from_str(json).unwrap()
        else {
            panic!("round-trip must yield a GraphFineTune spec");
        };
        // The worker reads the sources from SQL; here the node/edge content is
        // fixed by the fixture, so the sampler input is the same — the only
        // variable across attempts is the seeded sampler, which the spec carries.
        let _ = &sources;
        let (nodes, edges) = two_communities(EdgeProvenance::Declared);
        GraphSampler::build(nodes, edges, sample_config)
            .unwrap()
            .sample()
            .unwrap()
    };

    let first = resample(&json);
    let second = resample(&json);
    assert!(
        !first.is_empty(),
        "the round-tripped spec must sample a non-empty pair set"
    );
    assert_eq!(
        first, second,
        "two runs from the same persisted spec must re-sample identical pairs"
    );
}

/// The text-bearing precondition surfaces as a typed error end-to-end: an edge
/// endpoint with no node text is rejected at sampler build, never silently
/// dropped.
#[test]
fn dangling_endpoint_is_a_typed_error() {
    let nodes = vec![TextNode::new("a", "alpha"), TextNode::new("b", "beta")];
    let edges = vec![GraphEdge::declared("a", "missing")];
    let Err(err) = GraphSampler::build(nodes, edges, GraphSampleConfig::default()) else {
        panic!("dangling endpoint must be rejected");
    };
    assert!(format!("{err}").contains("text-bearing"));
}

/// GA1 (issue #538): the graph sampler's input is read with an explicit
/// order (`GRAPH_READ_ORDER_RULE_V1`), so two physical layouts of the
/// IDENTICAL node/edge set sample byte-identical pairs — never a function of
/// scan order. The model is deliberately bogus so the job fails fast right
/// after `materialize_graph_training_set` runs (sampling happens before model
/// load); the fingerprint hook has already fired by the time `job.wait()`
/// returns either way.
///
/// Mutation executed (not committed): removing the `ORDER BY` clauses from
/// `materialize_graph_training_set`'s node/edge queries makes this assertion fail
/// (the two fingerprints differ) — confirmed by hand before this test was
/// added, restoring the fix afterward.
#[cfg(feature = "test-hooks")]
#[tokio::test(flavor = "multi_thread")]
async fn graph_sample_is_a_function_of_the_set_not_the_scan_order() {
    use jammi_ai::fine_tune::worker::training_test_hooks::graph_sample_fingerprint_for;

    async fn sample_fingerprint(
        node_rows: &[(String, String)],
        edge_rows: &[(String, String)],
    ) -> String {
        let dir = TempDir::new().unwrap();
        let config = common::test_config(dir.path());
        let session = Arc::new(InferenceSession::new(config).await.unwrap());
        let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
            .expect("default worker intervals are valid");

        let node_url = write_csv(dir.path(), "nodes.csv", "id,text", node_rows);
        let edge_url = write_csv(dir.path(), "edges.csv", "src,dst", edge_rows);
        session
            .add_source(
                "nodes",
                SourceType::File,
                SourceConnection {
                    url: Some(node_url),
                    format: Some(FileFormat::Csv),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        session
            .add_source(
                "edges",
                SourceType::File,
                SourceConnection {
                    url: Some(edge_url),
                    format: Some(FileFormat::Csv),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let sources = GraphFineTuneSources {
            node_source: "nodes".into(),
            id_column: "id".into(),
            text_column: "text".into(),
            edge_source: "edges".into(),
            src_column: "src".into(),
            dst_column: "dst".into(),
            provenance: EdgeProvenance::Declared,
        };
        let sample = GraphSampleConfig {
            walk_length: 3,
            walks_per_node: 4,
            // GA1 tests read-order independence, not negative mining; 0
            // side-steps GA3's empty-negative-pool refusal on this small
            // fixture (a 5-node ring can legitimately exhaust an anchor's
            // candidate pool under exclude_hops=1).
            hard_negatives: 0,
            exclude_hops: 1,
            min_negatives: 1,
            seed: 4242,
            ..GraphSampleConfig::default()
        };
        // Deliberately unresolvable: the job fails at model load, well after
        // `materialize_graph_training_set` has already run and the hook fired.
        let job = session
            .fine_tune_graph(
                &sources,
                "local:/nonexistent/definitely-not-a-model",
                sample,
                Some(jammi_ai::fine_tune::FineTuneConfig::default()),
            )
            .await
            .unwrap();
        let _ = job.wait().await;
        graph_sample_fingerprint_for(&job.job_id)
            .expect("materialize_graph_training_set must have sampled and recorded a fingerprint")
    }

    let ids = ["n0000", "n0001", "n0002", "n0003", "n0004"];
    let node_rows_a: Vec<(String, String)> = ids
        .iter()
        .map(|id| (id.to_string(), format!("text of {id}")))
        .collect();
    // A different physical layout of the SAME node rows.
    let mut node_rows_b = node_rows_a.clone();
    node_rows_b.rotate_left(2);
    node_rows_b.swap(0, 3);

    let edge_pairs = [
        ("n0000", "n0001"),
        ("n0001", "n0002"),
        ("n0002", "n0003"),
        ("n0003", "n0004"),
        ("n0004", "n0000"),
        ("n0001", "n0004"),
        ("n0002", "n0000"),
    ];
    let edge_rows_a: Vec<(String, String)> = edge_pairs
        .iter()
        .map(|(s, d)| (s.to_string(), d.to_string()))
        .collect();
    // A different physical layout of the SAME edge rows (rotated, not sorted).
    let mut edge_rows_b = edge_rows_a.clone();
    edge_rows_b.rotate_left(3);

    let fp_a = sample_fingerprint(&node_rows_a, &edge_rows_a).await;
    let fp_b = sample_fingerprint(&node_rows_b, &edge_rows_b).await;
    assert_eq!(
        fp_a, fp_b,
        "two physical layouts of the identical node/edge set must sample \
         byte-identical pairs under GRAPH_READ_ORDER_RULE_V1"
    );
}

/// GA1: a node source with a duplicate id fails the job end to end with a
/// typed error naming the duplicate — never a silent last-write-wins sample.
#[tokio::test(flavor = "multi_thread")]
async fn fine_tune_graph_duplicate_node_id_fails() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");

    let node_rows: Vec<(String, String)> = vec![
        ("n0".to_string(), "text of n0".to_string()),
        ("n1".to_string(), "text of n1".to_string()),
        ("n0".to_string(), "DUPLICATE text of n0".to_string()),
    ];
    let node_url = write_csv(dir.path(), "nodes.csv", "id,text", &node_rows);
    let edge_rows: Vec<(String, String)> = vec![
        ("n0".to_string(), "n1".to_string()),
        ("n1".to_string(), "n0".to_string()),
    ];
    let edge_url = write_csv(dir.path(), "edges.csv", "src,dst", &edge_rows);

    session
        .add_source(
            "nodes",
            SourceType::File,
            SourceConnection {
                url: Some(node_url),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    session
        .add_source(
            "edges",
            SourceType::File,
            SourceConnection {
                url: Some(edge_url),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let model = "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap();
    let sources = GraphFineTuneSources {
        node_source: "nodes".into(),
        id_column: "id".into(),
        text_column: "text".into(),
        edge_source: "edges".into(),
        src_column: "src".into(),
        dst_column: "dst".into(),
        provenance: EdgeProvenance::Declared,
    };
    let job = session
        .fine_tune_graph(
            &sources,
            &model,
            GraphSampleConfig::default(),
            Some(jammi_ai::fine_tune::FineTuneConfig::default()),
        )
        .await
        .expect("submit persists the spec and returns a handle");

    let result = job.wait().await;
    assert!(
        result.is_err(),
        "a duplicate node id must drive the job to a typed failure"
    );
    let record = session.catalog().get_job(&job.job_id).await.unwrap();
    assert_eq!(record.status, "failed");
}

/// GA7 (issue #538): the sampler's resident adjacency + node-text bytes are
/// reserved against a NAMED `training_set_graph_sample` `MemoryConsumer`,
/// sized against a REAL measurement — never zero, never a placeholder.
/// Asserted via the `test-hooks` recorder rather than by forcing a real
/// `ResourcesExhausted` (which would need the graph large enough to also
/// perturb `materialize_graph_training_set`'s own node/edge `ORDER BY` scans —
/// GA1 — a separate, comparably-sized DataFusion-side sort competing for
/// the same bounded pool during the READ, before this reservation is even
/// attempted; entangling the two would make this test's failure ambiguous
/// about which mechanism actually tripped).
#[tokio::test(flavor = "multi_thread")]
async fn fine_tune_graph_reservation_is_sized_against_a_real_measurement() {
    use jammi_ai::fine_tune::worker::training_test_hooks::graph_sample_reservation_bytes_for;

    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");

    let ids = ["n0", "n1", "n2", "n3"];
    let texts = [
        "text of node zero somewhat long",
        "text of node one",
        "text of node two also fairly long indeed",
        "text of node three",
    ];
    let node_rows: Vec<(String, String)> = ids
        .iter()
        .zip(texts.iter())
        .map(|(id, text)| (id.to_string(), text.to_string()))
        .collect();
    // An independent lower bound computed HERE, from the same fixture, via
    // arithmetic that does not call `GraphSampler::resident_bytes` — id +
    // text bytes for every node, never the tautological "call the same
    // method and compare to itself".
    let node_text_bytes: usize =
        ids.iter().map(|s| s.len()).sum::<usize>() + texts.iter().map(|s| s.len()).sum::<usize>();
    let node_url = write_csv(dir.path(), "nodes.csv", "id,text", &node_rows);
    let edge_rows: Vec<(String, String)> = vec![
        ("n0".to_string(), "n1".to_string()),
        ("n1".to_string(), "n2".to_string()),
        ("n2".to_string(), "n3".to_string()),
        ("n3".to_string(), "n0".to_string()),
    ];
    let edge_url = write_csv(dir.path(), "edges.csv", "src,dst", &edge_rows);

    session
        .add_source(
            "nodes",
            SourceType::File,
            SourceConnection {
                url: Some(node_url),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    session
        .add_source(
            "edges",
            SourceType::File,
            SourceConnection {
                url: Some(edge_url),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let sources = GraphFineTuneSources {
        node_source: "nodes".into(),
        id_column: "id".into(),
        text_column: "text".into(),
        edge_source: "edges".into(),
        src_column: "src".into(),
        dst_column: "dst".into(),
        provenance: EdgeProvenance::Declared,
    };
    let sample = GraphSampleConfig {
        walk_length: 2,
        walks_per_node: 1,
        hard_negatives: 0,
        exclude_hops: 1,
        min_negatives: 1,
        seed: 1,
        ..GraphSampleConfig::default()
    };
    // Deliberately unresolvable model: the reservation and the fingerprint
    // hook both fire well before any model load would matter.
    let job = session
        .fine_tune_graph(
            &sources,
            "local:/nonexistent/definitely-not-a-model",
            sample,
            Some(jammi_ai::fine_tune::FineTuneConfig::default()),
        )
        .await
        .unwrap();
    let _ = job.wait().await;

    let reserved = graph_sample_reservation_bytes_for(&job.job_id)
        .expect("materialize_graph_training_set must have reserved and recorded its bytes");
    assert!(
        reserved >= node_text_bytes,
        "the reservation ({reserved} B) must be at least the node id+text bytes alone \
         ({node_text_bytes} B) — it also covers the (small, here) out_adj/undirected \
         adjacency on top"
    );
}

/// GA7's release-timing oracle (issue #538): the named
/// `training_set_graph_sample` reservation is released STRICTLY AFTER the
/// materialised table's write commits, never right after sampling and
/// before the write — a closing audit found the reservation dropped before
/// the batches it had reserved for were even built. A mutation that
/// restores the early `drop(reservation)` right after `sample_into`
/// returns (before the batch-build loop) turns this RED (`Some(false)`,
/// executed and confirmed, then reverted before committing).
#[tokio::test(flavor = "multi_thread")]
async fn fine_tune_graph_reservation_is_released_after_the_write_commits() {
    use jammi_ai::fine_tune::worker::training_test_hooks::graph_sample_reservation_released_after_write_for;

    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");

    let node_rows: Vec<(String, String)> = ["n0", "n1", "n2", "n3"]
        .iter()
        .map(|id| (id.to_string(), format!("text of node {id}")))
        .collect();
    let node_url = write_csv(dir.path(), "nodes.csv", "id,text", &node_rows);
    let edge_rows: Vec<(String, String)> = vec![
        ("n0".to_string(), "n1".to_string()),
        ("n1".to_string(), "n2".to_string()),
        ("n2".to_string(), "n3".to_string()),
        ("n3".to_string(), "n0".to_string()),
    ];
    let edge_url = write_csv(dir.path(), "edges.csv", "src,dst", &edge_rows);

    session
        .add_source(
            "nodes",
            SourceType::File,
            SourceConnection {
                url: Some(node_url),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    session
        .add_source(
            "edges",
            SourceType::File,
            SourceConnection {
                url: Some(edge_url),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let sources = GraphFineTuneSources {
        node_source: "nodes".into(),
        id_column: "id".into(),
        text_column: "text".into(),
        edge_source: "edges".into(),
        src_column: "src".into(),
        dst_column: "dst".into(),
        provenance: EdgeProvenance::Declared,
    };
    let sample = GraphSampleConfig {
        walk_length: 2,
        walks_per_node: 1,
        hard_negatives: 0,
        exclude_hops: 1,
        min_negatives: 1,
        seed: 1,
        ..GraphSampleConfig::default()
    };
    let job = session
        .fine_tune_graph(
            &sources,
            "local:/nonexistent/definitely-not-a-model",
            sample,
            Some(jammi_ai::fine_tune::FineTuneConfig::default()),
        )
        .await
        .unwrap();
    let _ = job.wait().await;

    assert_eq!(
        graph_sample_reservation_released_after_write_for(&job.job_id),
        Some(true),
        "the reservation must release AFTER the table's write commits, not before"
    );
}

/// GA5/GA2/GA9 (issue #538): a graph fine-tune's worker path actually
/// materialises a `TrainingSet`-kind table through the seam, carrying a
/// `GraphTrainingSet` descriptor with the sources/columns/format/sample
/// config the job ran with — never silently still sampling in memory with
/// no table at all (`origin/main`'s pre-#538 shape).
#[tokio::test(flavor = "multi_thread")]
async fn fine_tune_graph_materialises_a_graph_training_set_table() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");

    let node_rows: Vec<(String, String)> = ["a0", "a1", "a2", "b0", "b1", "b2"]
        .iter()
        .map(|id| (id.to_string(), format!("document about topic {id}")))
        .collect();
    let node_url = write_csv(dir.path(), "nodes.csv", "id,text", &node_rows);
    let edge_pairs = [
        ("a0", "a1"),
        ("a1", "a0"),
        ("a1", "a2"),
        ("a2", "a1"),
        ("a0", "a2"),
        ("a2", "a0"),
        ("b0", "b1"),
        ("b1", "b0"),
        ("b1", "b2"),
        ("b2", "b1"),
        ("b0", "b2"),
        ("b2", "b0"),
        ("a0", "b0"),
    ];
    let edge_rows: Vec<(String, String)> = edge_pairs
        .iter()
        .map(|(s, d)| (s.to_string(), d.to_string()))
        .collect();
    let edge_url = write_csv(dir.path(), "edges.csv", "src,dst", &edge_rows);

    session
        .add_source(
            "nodes",
            SourceType::File,
            SourceConnection {
                url: Some(node_url),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    session
        .add_source(
            "edges",
            SourceType::File,
            SourceConnection {
                url: Some(edge_url),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let sources = GraphFineTuneSources {
        node_source: "nodes".into(),
        id_column: "id".into(),
        text_column: "text".into(),
        edge_source: "edges".into(),
        src_column: "src".into(),
        dst_column: "dst".into(),
        provenance: EdgeProvenance::Declared,
    };
    let sample = GraphSampleConfig {
        walk_length: 3,
        walks_per_node: 2,
        hard_negatives: 1,
        exclude_hops: 1,
        min_negatives: 1,
        seed: 11,
        ..GraphSampleConfig::default()
    };
    let job = session
        .fine_tune_graph(
            &sources,
            "local:/nonexistent/definitely-not-a-model",
            sample,
            Some(jammi_ai::fine_tune::FineTuneConfig::default()),
        )
        .await
        .unwrap();
    let _ = job.wait().await;

    let ready = session
        .catalog()
        .list_result_tables_by_status(jammi_db::catalog::status::ResultTableStatus::Ready)
        .await
        .unwrap();
    let training_set = ready
        .iter()
        .find(|t| t.kind == jammi_db::catalog::result_repo::ResultTableKind::TrainingSet)
        .expect("the graph arm must materialise a TrainingSet-kind table");

    let descriptor = session
        .result_store()
        .producing_descriptor(training_set)
        .await
        .unwrap();
    match descriptor {
        jammi_db::store::manifest::ProducingDescriptor::GraphTrainingSet {
            node_source,
            edge_source,
            id_column,
            text_column,
            src_column,
            dst_column,
            format,
            sample,
            read_order_rule,
            ..
        } => {
            assert_eq!(node_source, "nodes");
            assert_eq!(edge_source, "edges");
            assert_eq!(id_column, "id");
            assert_eq!(text_column, "text");
            assert_eq!(src_column, "src");
            assert_eq!(dst_column, "dst");
            assert_eq!(
                format, "triplet",
                "hard_negatives=1 must record the triplet format"
            );
            assert_eq!(sample.hard_negatives, 1);
            assert_eq!(sample.seed, 11);
            assert_eq!(read_order_rule, jammi_db::store::GRAPH_READ_ORDER_RULE_V1);
        }
        other => panic!("expected a GraphTrainingSet descriptor, got {other:?}"),
    }
    assert!(training_set.row_count > 0);
}

/// GA6 (issue #538): two attempts of ONE `graph_fine_tune` job id (a stale
/// lease reclaimed to a second worker) never displace or clobber each
/// other's materialised `GraphTrainingSet` table — each attempt's own call
/// to `materialize_graph_training_set` re-samples and re-materialises
/// independently (the source anchors are `UnpinnedAtInstant`, so the second
/// attempt's `materialize_training_set` call never reuses the first's row —
/// `probe_ready_training_set` never matches an unpinned anchor), and the
/// per-attempt table name (`materialization.rs:1613`'s timestamp+random
/// suffix, unchanged since before #538) keeps the two tables distinct. The
/// same shape `fine_tune.rs::loser_prefix_is_never_the_committed_artifact`
/// proves for the tabular arm, generic over `JobWorker::run_claimed_job`'s
/// kind dispatch — this is that oracle's graph-arm instance.
#[tokio::test(flavor = "multi_thread")]
async fn two_attempts_of_one_graph_job_never_displace_each_others_table() {
    use jammi_ai::fine_tune::worker::JobWorker;
    use std::time::Duration;

    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());

    let node_rows: Vec<(String, String)> = ["a0", "a1", "a2", "b0", "b1", "b2"]
        .iter()
        .map(|id| (id.to_string(), format!("document about topic {id}")))
        .collect();
    let node_url = write_csv(dir.path(), "nodes.csv", "id,text", &node_rows);
    let edge_pairs = [
        ("a0", "a1"),
        ("a1", "a0"),
        ("a1", "a2"),
        ("a2", "a1"),
        ("a0", "a2"),
        ("a2", "a0"),
        ("b0", "b1"),
        ("b1", "b0"),
        ("b1", "b2"),
        ("b2", "b1"),
        ("b0", "b2"),
        ("b2", "b0"),
        ("a0", "b0"),
    ];
    let edge_rows: Vec<(String, String)> = edge_pairs
        .iter()
        .map(|(s, d)| (s.to_string(), d.to_string()))
        .collect();
    let edge_url = write_csv(dir.path(), "edges.csv", "src,dst", &edge_rows);

    session
        .add_source(
            "nodes",
            SourceType::File,
            SourceConnection {
                url: Some(node_url),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    session
        .add_source(
            "edges",
            SourceType::File,
            SourceConnection {
                url: Some(edge_url),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let model = "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap();
    let sources = GraphFineTuneSources {
        node_source: "nodes".into(),
        id_column: "id".into(),
        text_column: "text".into(),
        edge_source: "edges".into(),
        src_column: "src".into(),
        dst_column: "dst".into(),
        provenance: EdgeProvenance::Declared,
    };
    let sample = GraphSampleConfig {
        walk_length: 3,
        walks_per_node: 2,
        hard_negatives: 1,
        exclude_hops: 1,
        min_negatives: 1,
        seed: 11,
        ..GraphSampleConfig::default()
    };
    let train = jammi_ai::fine_tune::FineTuneConfig {
        epochs: 1,
        batch_size: 4,
        lora_rank: 4,
        warmup_steps: 0,
        validation_fraction: 0.0,
        early_stopping_metric: jammi_ai::fine_tune::EarlyStoppingMetric::TrainLoss,
        ..Default::default()
    };

    let job = session
        .fine_tune_graph(&sources, &model, sample, Some(train))
        .await
        .unwrap();

    let worker_a = JobWorker::new(&session).expect("default worker intervals are valid");
    let worker_b = JobWorker::new(&session).expect("default worker intervals are valid");

    // worker-a claims with an already-expired lease; worker-b reclaims and
    // re-claims under a long lease, so worker-b owns the job.
    let stale_claim = session
        .catalog()
        .claim_next(worker_a.worker_id(), &["graph_fine_tune"], Duration::ZERO)
        .await
        .unwrap()
        .expect("worker-a claims the queued job");
    let actioned = session
        .catalog()
        .reclaim_expired_jobs(Duration::from_secs(60), 5)
        .await
        .unwrap();
    assert_eq!(actioned, 1, "the expired lease is re-queued");
    let owned = session
        .catalog()
        .claim_next(
            worker_b.worker_id(),
            &["graph_fine_tune"],
            Duration::from_secs(3600),
        )
        .await
        .unwrap()
        .expect("worker-b re-claims the requeued job");

    // Both attempts run to completion; each independently samples and
    // materialises its OWN GraphTrainingSet table (no shared, name-collided
    // artifact) before either trains.
    worker_a.run_claimed_job(&session, stale_claim).await;
    let after_loser = session.catalog().get_job(&job.job_id).await.unwrap();
    assert_ne!(
        after_loser.status, "completed",
        "the loser's finalize CAS fails; the job is not completed by it"
    );

    worker_b.run_claimed_job(&session, owned).await;
    job.wait().await.unwrap();
    let done = session.catalog().get_job(&job.job_id).await.unwrap();
    assert_eq!(done.status, "completed", "the winner finalizes the job");

    // Both attempts' TrainingSet tables exist, `ready`, and are DISTINCT —
    // the loser's is an orphan (never referenced by any model row), never a
    // silently-clobbered or reused-by-name artifact.
    let ready = session
        .catalog()
        .list_result_tables_by_status(jammi_db::catalog::status::ResultTableStatus::Ready)
        .await
        .unwrap();
    let training_sets: Vec<_> = ready
        .iter()
        .filter(|t| t.kind == jammi_db::catalog::result_repo::ResultTableKind::TrainingSet)
        .collect();
    assert_eq!(
        training_sets.len(),
        2,
        "both attempts must have materialised their OWN table: {:?}",
        training_sets
            .iter()
            .map(|t| &t.table_name)
            .collect::<Vec<_>>()
    );
    assert_ne!(
        training_sets[0].table_name, training_sets[1].table_name,
        "the two attempts' tables must never share one name"
    );
}

/// GA9 (issue #538): recomputing a `GraphTrainingSet` table over UNMOVED
/// node/edge sources re-samples through the SAME shared core a fresh run
/// uses and writes a byte-identical artifact — the descriptor records every
/// sample determinant, so nothing about the replay can drift from the
/// original.
#[tokio::test(flavor = "multi_thread")]
async fn graph_training_set_recompute_is_byte_identical_over_unmoved_sources() {
    use jammi_ai::pipeline::recompute::Cascade;

    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");

    let node_rows: Vec<(String, String)> = ["a0", "a1", "a2", "b0", "b1", "b2"]
        .iter()
        .map(|id| (id.to_string(), format!("document about topic {id}")))
        .collect();
    let node_url = write_csv(dir.path(), "nodes.csv", "id,text", &node_rows);
    let edge_pairs = [
        ("a0", "a1"),
        ("a1", "a0"),
        ("a1", "a2"),
        ("a2", "a1"),
        ("a0", "a2"),
        ("a2", "a0"),
        ("b0", "b1"),
        ("b1", "b0"),
        ("b1", "b2"),
        ("b2", "b1"),
        ("b0", "b2"),
        ("b2", "b0"),
        ("a0", "b0"),
    ];
    let edge_rows: Vec<(String, String)> = edge_pairs
        .iter()
        .map(|(s, d)| (s.to_string(), d.to_string()))
        .collect();
    let edge_url = write_csv(dir.path(), "edges.csv", "src,dst", &edge_rows);

    session
        .add_source(
            "nodes",
            SourceType::File,
            SourceConnection {
                url: Some(node_url),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    session
        .add_source(
            "edges",
            SourceType::File,
            SourceConnection {
                url: Some(edge_url),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let sources = GraphFineTuneSources {
        node_source: "nodes".into(),
        id_column: "id".into(),
        text_column: "text".into(),
        edge_source: "edges".into(),
        src_column: "src".into(),
        dst_column: "dst".into(),
        provenance: EdgeProvenance::Declared,
    };
    let sample = GraphSampleConfig {
        walk_length: 3,
        walks_per_node: 2,
        hard_negatives: 1,
        exclude_hops: 1,
        min_negatives: 1,
        seed: 11,
        ..GraphSampleConfig::default()
    };
    let job = session
        .fine_tune_graph(
            &sources,
            "local:/nonexistent/definitely-not-a-model",
            sample,
            Some(jammi_ai::fine_tune::FineTuneConfig::default()),
        )
        .await
        .unwrap();
    let _ = job.wait().await;

    let ready = session
        .catalog()
        .list_result_tables_by_status(jammi_db::catalog::status::ResultTableStatus::Ready)
        .await
        .unwrap();
    let training_set = ready
        .iter()
        .find(|t| t.kind == jammi_db::catalog::result_repo::ResultTableKind::TrainingSet)
        .expect("the graph arm must materialise a TrainingSet-kind table");

    let before_url = jammi_db::storage::StorageUrl::parse(&training_set.parquet_path).unwrap();
    let before_digest = session
        .result_store()
        .read_materialization_manifest(&before_url)
        .await
        .unwrap()
        .expect("manifest sidecar present")
        .artifact
        .0;

    let report = jammi_ai::Session::new(Arc::clone(&session))
        .recompute(&training_set.table_name, Cascade::ReportOnly)
        .await
        .unwrap();
    assert_eq!(report.recomputed.len(), 1);
    let replay = &report.recomputed[0];
    assert_eq!(replay.original, training_set.table_name);
    assert_ne!(
        replay.recomputed, training_set.table_name,
        "the replay must write a NEW table, not hand back the one it replayed"
    );
    assert_eq!(
        replay.outcome,
        jammi_db::store::CacheOutcome::Computed,
        "an unpinned source anchor never matches the verb's reuse probe, so a replay always \
         recomputes"
    );

    let after_record = session
        .catalog()
        .get_result_table(&replay.recomputed)
        .await
        .unwrap()
        .expect("the replayed table exists");
    let after_url = jammi_db::storage::StorageUrl::parse(&after_record.parquet_path).unwrap();
    let after_digest = session
        .result_store()
        .read_materialization_manifest(&after_url)
        .await
        .unwrap()
        .expect("manifest sidecar present")
        .artifact
        .0;
    assert_eq!(
        before_digest, after_digest,
        "GA9: a replay over UNMOVED node/edge sources must be byte-identical to the original"
    );
}

// ─── End-to-end: fine_tune_graph drives the real trainer ────────────────────

/// Write a 2-column CSV to `dir/name` and return its `file://` URL.
fn write_csv(dir: &std::path::Path, name: &str, header: &str, rows: &[(String, String)]) -> String {
    let mut body = String::from(header);
    body.push('\n');
    for (a, b) in rows {
        body.push_str(a);
        body.push(',');
        body.push_str(b);
        body.push('\n');
    }
    let path = dir.join(name);
    std::fs::write(&path, body).unwrap();
    format!("file://{}", path.display())
}

/// `fine_tune_graph` reads a node source + a declared-edge source, samples the
/// graph, and trains a real (tiny_bert) model to a completed job with a saved
/// adapter — the integration proof that a graph sample threads through
/// the existing trainer with no new loss.
#[tokio::test(flavor = "multi_thread")]
async fn fine_tune_graph_end_to_end_completes() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    // `fine_tune_graph` submits a queued job; the worker re-reads the sources,
    // re-samples the graph from the seeded spec, and trains it.
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");

    // Node text: two small communities.
    let node_rows: Vec<(String, String)> = ["a0", "a1", "a2", "b0", "b1", "b2"]
        .iter()
        .map(|id| (id.to_string(), format!("document about topic {id}")))
        .collect();
    let node_url = write_csv(dir.path(), "nodes.csv", "id,text", &node_rows);

    // Declared edges: two triangles (clique-ish) plus a bridge. Directed both
    // ways so walks can traverse.
    let edge_pairs = [
        ("a0", "a1"),
        ("a1", "a0"),
        ("a1", "a2"),
        ("a2", "a1"),
        ("a0", "a2"),
        ("a2", "a0"),
        ("b0", "b1"),
        ("b1", "b0"),
        ("b1", "b2"),
        ("b2", "b1"),
        ("b0", "b2"),
        ("b2", "b0"),
        ("a0", "b0"),
    ];
    let edge_rows: Vec<(String, String)> = edge_pairs
        .iter()
        .map(|(s, d)| (s.to_string(), d.to_string()))
        .collect();
    let edge_url = write_csv(dir.path(), "edges.csv", "src,dst", &edge_rows);

    session
        .add_source(
            "nodes",
            SourceType::File,
            SourceConnection {
                url: Some(node_url),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    session
        .add_source(
            "edges",
            SourceType::File,
            SourceConnection {
                url: Some(edge_url),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let model = "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap();

    let sources = GraphFineTuneSources {
        node_source: "nodes".into(),
        id_column: "id".into(),
        text_column: "text".into(),
        edge_source: "edges".into(),
        src_column: "src".into(),
        dst_column: "dst".into(),
        provenance: EdgeProvenance::Declared,
    };
    let sample = GraphSampleConfig {
        walk_length: 3,
        walks_per_node: 2,
        hard_negatives: 1,
        exclude_hops: 1,
        min_negatives: 1,
        seed: 11,
        ..GraphSampleConfig::default()
    };
    let train = jammi_ai::fine_tune::FineTuneConfig {
        epochs: 1,
        batch_size: 4,
        lora_rank: 4,
        warmup_steps: 0,
        validation_fraction: 0.0,
        early_stopping_metric: jammi_ai::fine_tune::EarlyStoppingMetric::TrainLoss,
        embedding_loss: Some(
            jammi_ai::fine_tune::EmbeddingLoss::MultipleNegativesRanking { temperature: 20.0 },
        ),
        ..Default::default()
    };

    let job = session
        .fine_tune_graph(&sources, &model, sample, Some(train))
        .await
        .unwrap();
    assert!(job.model_id().starts_with("jammi:fine-tuned:"));

    job.wait().await.unwrap();

    let record = session.catalog().get_job(&job.job_id).await.unwrap();
    assert_eq!(
        record.status, "completed",
        "graph fine-tune job should complete, got {}",
        record.status
    );

    let ft = session
        .catalog()
        .get_model(job.model_id())
        .await
        .unwrap()
        .expect("graph fine-tune registered the model");
    let prefix_url =
        jammi_db::storage::StorageUrl::parse(ft.artifact_path.as_deref().unwrap()).unwrap();
    let local = session
        .artifact_store()
        .fetch_artifact(&prefix_url)
        .await
        .expect("published graph adapter fetches and verifies");
    let adapter = local.dir().join("adapter.safetensors");
    assert!(
        adapter.exists(),
        "graph fine-tune should publish an adapter, missing at {adapter:?}"
    );

    // This fixture's edge rows are NOT already in `(src, dst)` order, and
    // `materialize_graph_training_set` scans nodes/edges with an explicit
    // `ORDER BY` (see that method's doc), so the exact physical row the
    // sampler's first draw sees — and therefore the trained adapter's bytes
    // — depends on that scan order, never on the source file's own row
    // order (issue #538). This assertion pins the resulting bytes, so a
    // regression of the ordering fix (or an unrelated change to the
    // sampler/trainer) shows up here as a moved fingerprint.
    // The bytes are a function of the CPU architecture, not the operating
    // system (aarch64 Linux reproduces aarch64 macOS byte for byte), so the
    // pin carries one value per `target_arch`. A re-pin states the old and
    // new values and the run that produced the new one.
    let expected = if cfg!(target_arch = "x86_64") {
        "1184:d923987b7592d7bd"
    } else {
        "1184:7a8781df1cd80172"
    };
    assert_eq!(
        fingerprint(&std::fs::read(&adapter).unwrap()),
        expected,
        "the graph fine-tune adapter's bytes moved off the GA1 (issue #538) pinned value for \
         this exact fixture; a mismatch here means the read-order rule, the sampler, or the \
         trainer changed since GA1 was pinned"
    );
}

/// FNV-1a over a byte slice, as `{len}:{hash:016x}` — the same algorithm and
/// format `training_set::refactor_parity` uses, kept local here rather than
/// shared: this integration-test binary has no dev-dependency on `sha2`, and
/// `DefaultHasher` is explicitly not stable across toolchains, so neither can
/// back a byte fingerprint pinned in source.
fn fingerprint(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{}:{:016x}", bytes.len(), hash)
}

/// A node source with no edge source (isolated graph) is a failure end to end —
/// a graph with no structure carries no supervision. Submit no longer reads the
/// graph (it persists the spec), so the failure surfaces when the worker
/// re-samples: the job lands `failed` and `wait()` returns the typed error,
/// never a wedged job or a silent no-op.
#[tokio::test(flavor = "multi_thread")]
async fn fine_tune_graph_isolated_graph_fails() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");

    let node_rows: Vec<(String, String)> = ["n0", "n1", "n2"]
        .iter()
        .map(|id| (id.to_string(), format!("text {id}")))
        .collect();
    let node_url = write_csv(dir.path(), "nodes.csv", "id,text", &node_rows);
    // An edge file with a header but no rows → an isolated graph.
    let edge_url = write_csv(dir.path(), "edges.csv", "src,dst", &[]);

    session
        .add_source(
            "nodes",
            SourceType::File,
            SourceConnection {
                url: Some(node_url),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    session
        .add_source(
            "edges",
            SourceType::File,
            SourceConnection {
                url: Some(edge_url),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let model = "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap();
    let sources = GraphFineTuneSources {
        node_source: "nodes".into(),
        id_column: "id".into(),
        text_column: "text".into(),
        edge_source: "edges".into(),
        src_column: "src".into(),
        dst_column: "dst".into(),
        provenance: EdgeProvenance::Declared,
    };

    // Submit succeeds (it only persists the spec); the worker re-samples the
    // graph, the sampler refuses an edgeless graph, and the job lands `failed`.
    let job = session
        .fine_tune_graph(
            &sources,
            &model,
            GraphSampleConfig::default(),
            Some(jammi_ai::fine_tune::FineTuneConfig::default()),
        )
        .await
        .expect("submit persists the spec and returns a handle");

    let result = job.wait().await;
    assert!(
        result.is_err(),
        "an isolated graph (no edges) must drive the job to a typed failure"
    );
    let record = session.catalog().get_job(&job.job_id).await.unwrap();
    assert_eq!(record.status, "failed");
}
