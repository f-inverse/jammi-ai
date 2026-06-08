//! S11 — graph-supervised fine-tune integration tests.
//!
//! Two layers:
//! 1. **Sampler / loader structural contracts** (hermetic, no model): the
//!    biased-walk positive sampler, the k-hop false-negative guard, and the
//!    load-bearing **circularity** distinction between declared and similarity
//!    edge supervision. These run in milliseconds on tiny synthetic graphs.
//! 2. **End-to-end session path** (`tiny_bert`): `fine_tune_graph` threads a
//!    real graph (node CSV + edge CSV) through the existing trainer to a
//!    completed job + saved adapter — proving `TrainingFormat::Graph` drives the
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
    assert!(matches!(
        loader.format(),
        TrainingFormat::Graph {
            has_negatives: true
        }
    ));
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
    assert!(matches!(
        loader.format(),
        TrainingFormat::Graph {
            has_negatives: false
        }
    ));
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
/// adapter — the integration proof that `TrainingFormat::Graph` threads through
/// the existing trainer with no new loss.
#[tokio::test(flavor = "multi_thread")]
async fn fine_tune_graph_end_to_end_completes() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    // `fine_tune_graph` submits a queued job; the worker re-reads the sources,
    // re-samples the graph from the seeded spec, and trains it.
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session);

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

    let record = session
        .catalog()
        .get_training_job(&job.job_id)
        .await
        .unwrap();
    assert_eq!(
        record.status, "completed",
        "graph fine-tune job should complete, got {}",
        record.status
    );

    let adapter = dir
        .path()
        .join("models")
        .join(&job.job_id)
        .join("adapter.safetensors");
    assert!(
        adapter.exists(),
        "graph fine-tune should write an adapter, missing at {adapter:?}"
    );
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
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session);

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
    let record = session
        .catalog()
        .get_training_job(&job.job_id)
        .await
        .unwrap();
    assert_eq!(record.status, "failed");
}
