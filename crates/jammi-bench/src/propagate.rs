//! The `propagate` workload's engine rungs over the synthetic graph
//! `graph_legs` builds: `plan`, `plan-partitioned` and `placed`, each leg
//! filed with the unit's inputs the PyTorch rungs read.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;

use jammi_ai::pipeline::graph_neighbourhood::{EdgeDirection, EdgeSourceRef};
use jammi_ai::pipeline::graph_propagation::{
    PropagateRequest, PropagationOutput, PropagationWeighting,
};
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::result_repo::ResultTableRecord;

use crate::capture::{
    cpu_provenance, file_leg, leg_report, leg_stem, legs_per_point, vector_rows_digest,
    write_jsonl, write_vector_rows, IterationSeries,
};
use crate::graph_legs::{
    add_graph_sources, build_edges, build_features, build_nodes, materialize_features,
    read_sorted_vectors, EngineRung, GraphHost, GraphShape, GraphSources, DEFAULT_SHAPE,
    EDGES_FILE, INPUT_DIR, INPUT_MODEL_ID, X0_STEM,
};
use crate::leg::{Facts, Leg, Measured, Measurement, Payload, Provenance};
use crate::plane::{PlaneArgs, PlaneParams};
use crate::report::{Nullable, Tiers};

/// Build the propagation request over the registered synthetic graph, pinned to
/// the synthetic input embedding table (never a prior propagation's output) and
/// read undirected so the symmetric `Â` is meaningful.
async fn build_request(
    session: &Arc<InferenceSession>,
    sources: &GraphSources,
    hops: usize,
    alpha: f64,
) -> Result<PropagateRequest, Box<dyn std::error::Error>> {
    let source_table = session
        .catalog()
        .find_result_tables(&sources.nodes, None, Some(INPUT_MODEL_ID))
        .await?
        .into_iter()
        .next()
        .ok_or("the synthetic source embedding table is missing")?;
    Ok(PropagateRequest::new(
        sources.nodes.clone(),
        EdgeSourceRef::Registered {
            source_id: sources.edges.clone(),
            src_column: "src".into(),
            dst_column: "dst".into(),
            type_column: None,
            weight_column: None,
            as_of_column: None,
        },
    )
    .with_embedding_table(source_table.table_name)
    .with_direction(EdgeDirection::Undirected)
    .with_weighting(PropagationWeighting::DegreeNormalized)
    .with_output(PropagationOutput::Final)
    .with_hops(hops)
    .with_alpha(alpha))
}

/// One propagation of `request` on `host`: the table it committed.
async fn propagate(
    host: &mut GraphHost,
    request: &PropagateRequest,
) -> Result<ResultTableRecord, Box<dyn std::error::Error>> {
    match host {
        GraphHost::InProcess { session, .. } => {
            let (table, _) = session
                .propagate_embeddings(request, jammi_db::store::CachePolicy::Bypass)
                .await?;
            Ok(table)
        }
        #[cfg(feature = "plane")]
        GraphHost::Placed { .. } => {
            host.run_placed(
                jammi_ai::jobs::JobSpec::Propagate {
                    request: request.clone(),
                    cache: jammi_db::store::CachePolicy::Bypass,
                },
                "propagation",
            )
            .await
        }
    }
}

/// The propagation one `propagate` leg is asked to run: one graph size on one
/// rung.
#[derive(Debug, Clone)]
pub struct PropagateLegParams {
    /// The synthetic graph's shape.
    pub shape: GraphShape,
    /// Requested node count; spread evenly over the shape's classes.
    pub nodes: usize,
    /// The rung.
    pub rung: EngineRung,
    /// DataFusion `target_partitions` of the `plan-partitioned` rung — the
    /// one parameter separating it from `plan`.
    pub partitions: usize,
    /// Requested hop count `K`.
    pub hops: usize,
    /// Teleport probability `α`; `0` is SGC's `Âᴷ·X`.
    pub alpha: f64,
    /// Where the leg, its vectors and the unit's inputs are filed.
    pub legs_dir: PathBuf,
    /// Untimed iterations before the series.
    pub warmup: usize,
    /// Timed iterations.
    pub iterations: usize,
    /// The measured repeat this leg is filed as.
    pub take: usize,
    /// Where the `placed` rung runs.
    pub plane: PlaneParams,
}

/// The `propagate` leg's payload: what two legs must agree on, and what this
/// rung records about its run.
#[derive(Debug, Clone, Serialize)]
pub struct PropagatePayload {
    /// sha256 of the edge list every rung reads.
    pub graph_edges_sha256: String,
    /// sha256 of the input vectors file every rung reads.
    pub features_sha256: String,
    /// Node count.
    pub node_count: usize,
    /// Undirected edge count — the unit of the sweep.
    pub edge_count: usize,
    /// Vector width.
    pub dim: usize,
    /// Hops actually run — the request clamped to the engine's hop cap.
    pub hops: usize,
    /// Teleport probability `α`.
    pub alpha: f64,
    /// The operator: `D̃^{-1/2}(A+I)D̃^{-1/2}` over an undirected read.
    pub weighting: &'static str,
    /// The precision the vectors are held in.
    pub compute_precision: &'static str,
    /// The rung this leg claims.
    pub rung: &'static str,
    /// The unit of the sweep, `edges<N>`.
    pub unit: String,
    /// The measured repeat.
    pub take: usize,
    /// The session's `target_partitions`.
    pub target_partitions: usize,
    /// The hop count asked for, before the clamp.
    pub requested_hops: usize,
    /// The wiring rule's fan-out cap.
    pub fan_out: usize,
    /// Untimed iterations before the series.
    pub warmup: usize,
    /// Timed iterations.
    pub iters_measured: usize,
    /// The implementation that propagated.
    pub operator: &'static str,
}

impl Payload for PropagatePayload {
    const IDENTITY_FIELDS: &'static [(&'static str, Nullable)] = &[
        ("graph_edges_sha256", Nullable::NonNull),
        ("features_sha256", Nullable::NonNull),
        ("node_count", Nullable::NonNull),
        ("edge_count", Nullable::NonNull),
        ("dim", Nullable::NonNull),
        ("hops", Nullable::NonNull),
        (
            "alpha",
            Nullable::NullMeans("no teleport: plain K-hop smoothing"),
        ),
        ("weighting", Nullable::NonNull),
        ("compute_precision", Nullable::NonNull),
    ];
}

/// Run one `propagate` leg and file it: write the unit's inputs (the files the
/// PyTorch rungs read), stand the same graph up in a session at
/// `target_partitions`, propagate it warm, and persist the key-sorted output
/// beside the leg. Returns the leg and its file name.
pub async fn run_leg(
    params: &PropagateLegParams,
) -> Result<(Leg<PropagatePayload>, String), Box<dyn std::error::Error>> {
    let shape = params.shape;
    let per_class = (params.nodes / shape.n_classes).max(2);
    let nodes = build_nodes(shape.n_classes, per_class);
    let edges = build_edges(&nodes, shape.fan_out);
    let unit = format!("edges{}", edges.len());
    let rung = params.rung.as_str();
    let stem = leg_stem(rung, &unit, params.take);

    let input_dir = params.legs_dir.join(INPUT_DIR).join(&unit);
    let x0 = write_vector_rows(&input_dir, X0_STEM, &build_features(&nodes, shape.dim))?;
    let edges_file = write_jsonl(
        &input_dir,
        EDGES_FILE,
        edges
            .iter()
            .map(|(src, dst)| serde_json::json!({"src": src, "dst": dst})),
    )?;

    let dir = tempfile::tempdir()?;
    let mut host = GraphHost::stand_up(
        params.rung,
        params.partitions,
        &params.plane,
        "propagate",
        &format!("propagate-nodes{}-r{}", params.nodes, params.take),
        &["propagate"],
        dir.path(),
    )
    .await?;
    let sources = host.sources().clone();
    add_graph_sources(host.session(), dir.path(), &sources, &nodes, &edges).await?;
    materialize_features(host.session(), &sources, &nodes, shape.dim).await?;
    let request = build_request(host.session(), &sources, params.hops, params.alpha).await?;

    let mut series = IterationSeries::new(params.warmup, params.iterations);
    let mut last = None;
    for _ in 0..series.total() {
        let start = Instant::now();
        let table = propagate(&mut host, &request).await?;
        series.record(start.elapsed());
        last = Some(table);
    }
    let peak_rss_bytes = crate::rss::peak_rss_measurement();

    let table = last.ok_or("a propagate leg needs at least one iteration")?;
    let rows = read_sorted_vectors(host.session(), &table).await?;
    let ran_on = host.ran_on(&table.table_name).await?;
    let vectors = write_vector_rows(&params.legs_dir, &stem, &rows)?;
    let hops = request.effective_hops();

    let payload = PropagatePayload {
        graph_edges_sha256: edges_file.sha256,
        features_sha256: x0.file.sha256,
        node_count: nodes.len(),
        edge_count: edges.len(),
        dim: shape.dim,
        hops,
        alpha: params.alpha,
        weighting: "degree_normalized",
        compute_precision: "f32",
        rung,
        unit,
        take: params.take,
        target_partitions: params.rung.target_partitions(params.partitions),
        requested_hops: params.hops,
        fan_out: shape.fan_out,
        warmup: params.warmup,
        iters_measured: params.iterations,
        operator: "jammi_ai::session::InferenceSession::propagate_embeddings",
    };
    let measured = Measured {
        iter_wall_s: Some(series.into_seconds()),
        work: Some((edges.len() * hops) as f64),
        peak_rss_bytes,
        peak_vram_bytes: Measurement::not_yet_measured("bytes"),
        outcome_digest: Some(vector_rows_digest(&rows)),
        vectors_file: Some(format!("{stem}.vectors.f32")),
        vector_dim: Some(vectors.dim),
        ..Default::default()
    };
    let provenance = Provenance {
        ran_on,
        ..cpu_provenance()
    };
    let leg = Leg::new(payload, provenance, measured, Facts::default());
    let report = leg_report("propagate", leg.clone(), |leg| Tiers {
        propagate: Some(leg),
        ..Default::default()
    });
    let file = file_leg(&params.legs_dir, &stem, &report)?;
    Ok((leg, file))
}

/// `propagate`'s flags: the size sweep, the rungs, and the operator's
/// parameters. One leg per `(nodes, rung, take)` point.
#[derive(Debug, Clone, clap::Args)]
pub struct PropagateArgs {
    /// Node counts to sweep, comma-separated.
    #[arg(long, value_delimiter = ',', required = true)]
    nodes: Vec<usize>,
    /// `target_partitions` of the `plan-partitioned` rung.
    #[arg(long, default_value_t = 4)]
    partitions: usize,
    /// The rungs to run: `plan`, `plan-partitioned`, `placed`; the first two
    /// by default. `placed` needs `--features plane`, the plane's backends
    /// in the environment and `--server-bin`.
    #[arg(long = "rung", value_enum, value_delimiter = ',', default_values = ["plan", "plan-partitioned"])]
    rungs: Vec<EngineRung>,
    #[arg(long, default_value_t = jammi_ai::pipeline::graph_propagation::DEFAULT_PROPAGATE_HOPS)]
    hops: usize,
    #[arg(long, default_value_t = jammi_ai::pipeline::graph_propagation::DEFAULT_TELEPORT_ALPHA)]
    alpha: f64,
    /// Where the legs are filed, `<rung>__edges<N>__r<take>.json` with the
    /// vectors beside, and the inputs under `input/edges<N>/`.
    #[arg(long)]
    legs_dir: PathBuf,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    /// Timed iterations; the default is the comparator's minimum series, and a
    /// shorter run files legs the speed axis refuses by name.
    #[arg(long, default_value_t = crate::ladder::definition::SpeedInstrument::MIN_SAMPLES)]
    iterations: usize,
    /// Measured repeats of each point, each in a process of its own.
    #[arg(long, default_value_t = 1)]
    takes: usize,
    /// The take a single point's run is filed as.
    #[arg(long, default_value_t = 1)]
    take: usize,
    #[command(flatten)]
    plane: PlaneArgs,
}

impl PropagateArgs {
    fn params(&self, nodes: usize, rung: EngineRung, take: usize) -> PropagateLegParams {
        PropagateLegParams {
            shape: DEFAULT_SHAPE,
            nodes,
            rung,
            partitions: self.partitions,
            hops: self.hops,
            alpha: self.alpha,
            legs_dir: self.legs_dir.clone(),
            warmup: self.warmup,
            iterations: self.iterations,
            take,
            plane: self.plane.clone().into(),
        }
    }

    /// Run the subcommand: one leg per point, and print the file names.
    pub async fn execute(&self) -> Result<(), Box<dyn std::error::Error>> {
        let points: Vec<(usize, EngineRung, usize)> = self
            .nodes
            .iter()
            .flat_map(|&n| {
                self.rungs
                    .iter()
                    .flat_map(move |&r| (1..=self.takes).map(move |t| (n, r, t)))
            })
            .collect();
        let (n, rung, _) = *points
            .first()
            .ok_or("propagate needs at least one --nodes value")?;
        let first = self.params(n, rung, if points.len() == 1 { self.take } else { 1 });
        let files = legs_per_point(
            &points,
            async move { run_leg(&first).await.map(|(_, file)| vec![file]) },
            |&(nodes, rung, take)| {
                let flags = [
                    ("--nodes", nodes.to_string()),
                    ("--rung", rung.as_str().to_string()),
                    ("--partitions", self.partitions.to_string()),
                    ("--hops", self.hops.to_string()),
                    ("--alpha", self.alpha.to_string()),
                    ("--warmup", self.warmup.to_string()),
                    ("--iterations", self.iterations.to_string()),
                    ("--take", take.to_string()),
                ];
                [
                    "propagate".into(),
                    "--legs-dir".into(),
                    (&self.legs_dir).into(),
                ]
                .into_iter()
                .chain(
                    flags
                        .into_iter()
                        .flat_map(|(flag, value)| [flag.into(), value.into()]),
                )
                .chain(PlaneParams::from(self.plane.clone()).child_args())
                .collect()
            },
        )
        .await?;
        println!("{}", serde_json::to_string_pretty(&files)?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::capture::read_vector_rows;
    use crate::ladder::definition::Workload;
    use crate::ladder::verdict::OutcomeVerdict;
    use crate::ladder::{run_ladder, Axis, LadderArgs};

    const HOPS: usize = 2;
    const ALPHA: f64 = 0.1;

    /// One cold iteration of a 32-node fixture (4 classes × 8), filed under
    /// `legs_dir`.
    async fn leg(
        legs_dir: &std::path::Path,
        nodes: usize,
        hops: usize,
        alpha: f64,
        partitions: usize,
    ) -> (Leg<PropagatePayload>, String) {
        run_leg(&PropagateLegParams {
            shape: DEFAULT_SHAPE,
            nodes,
            rung: if partitions == 1 {
                EngineRung::Plan
            } else {
                EngineRung::PlanPartitioned
            },
            partitions,
            hops,
            alpha,
            legs_dir: legs_dir.to_path_buf(),
            warmup: 0,
            iterations: 1,
            take: 1,
            plane: PlaneParams::default(),
        })
        .await
        .expect("propagate leg runs")
    }

    /// The engine's real contract for an `f32` output: folding the same fixture
    /// twice on THIS box produces the byte-identical digest and byte-identical
    /// rows. True on any box by construction.
    #[tokio::test]
    async fn refold_is_deterministic_on_this_machine() {
        let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let (first, file) = leg(a.path(), 32, HOPS, ALPHA, 1).await;
        let (second, _) = leg(b.path(), 32, HOPS, ALPHA, 1).await;
        assert_eq!(
            first.measured.outcome_digest,
            second.measured.outcome_digest
        );
        assert_eq!(
            first.payload.features_sha256,
            second.payload.features_sha256
        );
        assert_eq!(file, format!("plan__{}__r1.json", first.payload.unit));
        let rows = read_vector_rows(
            &a.path()
                .join(format!("plan__{}__r1.vectors.f32", first.payload.unit)),
            DEFAULT_SHAPE.dim,
        )
        .unwrap();
        assert_eq!(rows.len(), first.payload.node_count);
        assert_eq!(
            vector_rows_digest(&rows),
            first.measured.outcome_digest.unwrap()
        );
    }

    /// The `plan` → `plan-partitioned` edge is exact: same identity, same bits.
    #[tokio::test]
    async fn digest_is_invariant_across_target_partitions() {
        let dir = tempfile::tempdir().unwrap();
        let (one, _) = leg(dir.path(), 32, HOPS, ALPHA, 1).await;
        let (four, _) = leg(dir.path(), 32, HOPS, ALPHA, 4).await;
        assert_eq!(
            (one.payload.rung, four.payload.rung),
            ("plan", "plan-partitioned")
        );
        assert_eq!(one.measured.outcome_digest, four.measured.outcome_digest);
    }

    /// The equality above has teeth: a regressed parameter moves the digest.
    #[tokio::test]
    async fn perturbed_propagation_changes_the_digest() {
        let dir = tempfile::tempdir().unwrap();
        let (baseline, _) = leg(&dir.path().join("base"), 32, HOPS, ALPHA, 1).await;
        for (what, hops, alpha) in [
            ("one hop fewer", HOPS - 1, ALPHA),
            ("one hop more", HOPS + 1, ALPHA),
            ("a different teleport α", HOPS, ALPHA + 0.4),
        ] {
            let (perturbed, _) =
                leg(&dir.path().join(what.replace(' ', "_")), 32, hops, alpha, 1).await;
            assert_ne!(
                perturbed.measured.outcome_digest, baseline.measured.outcome_digest,
                "{what}"
            );
        }
    }

    /// The propagated vectors are not the input.
    #[tokio::test]
    async fn propagation_is_not_the_identity() {
        let dir = tempfile::tempdir().unwrap();
        let (propagated, _) = leg(dir.path(), 32, HOPS, ALPHA, 1).await;
        let nodes = build_nodes(DEFAULT_SHAPE.n_classes, 8);
        let mut input = build_features(&nodes, DEFAULT_SHAPE.dim);
        input.sort_by(|a, b| a.0.cmp(&b.0));
        assert_ne!(
            Some(vector_rows_digest(&input)),
            propagated.measured.outcome_digest
        );
    }

    /// The hop count a leg is identified by is the depth actually run.
    #[tokio::test]
    async fn identity_records_the_hops_actually_run() {
        let dir = tempfile::tempdir().unwrap();
        let (capped, _) = leg(dir.path(), 32, 9, ALPHA, 1).await;
        assert_eq!(
            capped.payload.hops,
            jammi_ai::pipeline::graph_neighbourhood::DEFAULT_HOP_CAP
        );
        assert_eq!(capped.payload.requested_hops, 9);
    }

    /// The ladder runs end to end over filed legs: the exact edge between
    /// `plan` and `plan-partitioned` by digest, the cross-stack edges by row
    /// agreement. The reference rungs' legs are stand-ins — the `plan` legs
    /// filed under `torch` and `torch-geometric` — so the run exercises the
    /// comparator's path over the leg shape, not a second operator.
    #[tokio::test]
    async fn the_ladder_reaches_its_verdicts_over_filed_legs() {
        let dir = tempfile::tempdir().unwrap();
        let legs = dir.path().join("legs");
        for nodes in [32, 64] {
            let (_, plan) = leg(&legs, nodes, HOPS, ALPHA, 1).await;
            leg(&legs, nodes, HOPS, ALPHA, 4).await;
            for rung in ["torch", "torch-geometric"] {
                std::fs::copy(
                    legs.join(&plan),
                    legs.join(plan.replacen("plan__", &format!("{rung}__"), 1)),
                )
                .unwrap();
            }
        }
        let verdict = run_ladder(&LadderArgs {
            workload: Workload::Propagate,
            legs_dir: legs,
            out: None,
            from: None,
            to: Some("plan-partitioned".into()),
            axes: vec![Axis::Outcome],
            waive_control: false,
            law_dir: None,
            mutants: vec![],
            revision: None,
        })
        .unwrap();
        assert!(verdict.refusals.is_empty(), "{:?}", verdict.refusals);
        assert_eq!(verdict.edges.len(), 3);
        for edge in &verdict.edges {
            assert!(
                edge.refusals.is_empty(),
                "{}: {:?}",
                edge.edge,
                edge.refusals
            );
        }
        assert!(matches!(
            verdict.edges[0].outcome,
            Some(OutcomeVerdict::RowAgreement { .. })
        ));
        assert!(matches!(
            verdict.edges[1].outcome,
            Some(OutcomeVerdict::RowAgreement { .. })
        ));
        assert!(
            matches!(
                verdict.edges[2].outcome,
                Some(OutcomeVerdict::Digest { .. })
            ),
            "{:?}",
            verdict.edges[2].outcome
        );
    }
}
