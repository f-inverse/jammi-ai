//! The `structure` workload's engine rungs over the synthetic graph
//! `graph_legs` builds: `generate_structure_embeddings` — an embedding table
//! from the edge relation alone — at `plan`, `plan-partitioned` and
//! `placed`, each leg filed with the unit's inputs the PyTorch rung reads:
//! the edge list and the engine's own seed rows, so the reference starts from
//! the bytes the engine starts from and what remains between them is the
//! operator.

use std::path::PathBuf;
use std::time::Instant;

use serde::Serialize;

use jammi_ai::pipeline::graph_neighbourhood::EdgeSourceRef;
use jammi_ai::pipeline::graph_propagation::seed::SeedSpec;
use jammi_ai::pipeline::graph_structure::{
    StructureRequest, DEFAULT_STRUCTURE_BETA, DEFAULT_STRUCTURE_DIMENSIONS, DEFAULT_STRUCTURE_SEED,
    DEFAULT_STRUCTURE_SPARSITY, DEFAULT_STRUCTURE_WEIGHTS,
};
use jammi_db::catalog::result_repo::ResultTableRecord;

use crate::capture::{
    cpu_provenance, file_leg, leg_report, leg_stem, legs_per_point, vector_rows_digest,
    write_jsonl, write_vector_rows,
};
use crate::graph_legs::{
    add_graph_sources, augmented_degrees, build_edges, build_nodes, read_sorted_vectors,
    EngineRung, GraphHost, GraphShape, GraphSources, KeyedRows, DEFAULT_SHAPE, EDGES_FILE,
    INPUT_DIR, X0_STEM,
};
use crate::ladder::leg::Take;
use crate::leg::{Facts, Leg, Measured, Measurement, Payload, Provenance};
use crate::plane::{PlaneArgs, PlaneParams};
use crate::report::{Nullable, Tiers};

/// The encoding one `structure` leg is asked to run: one graph size on one
/// rung.
#[derive(Debug, Clone)]
pub struct StructureLegParams {
    /// The synthetic graph's shape; its `dim` is unused, the width is
    /// `dimensions`.
    pub shape: GraphShape,
    /// Requested node count; spread evenly over the shape's classes.
    pub nodes: usize,
    pub rung: EngineRung,
    /// DataFusion `target_partitions` of the `plan-partitioned` rung.
    pub partitions: usize,
    /// The embedding width `d`.
    pub dimensions: usize,
    /// The readout weight of each block `0..=K`.
    pub weights: Vec<f64>,
    /// The degree exponent `β`.
    pub beta: f64,
    /// The projection sparsity `s`.
    pub sparsity: f64,
    /// Keys every node's projection stream.
    pub seed: u64,
    /// Where the leg, its vectors and the unit's inputs are filed.
    pub legs_dir: PathBuf,
    /// Iterations timed, every one filed in run order.
    pub iterations: usize,
    /// The measured repeat this leg is filed as.
    pub take: usize,
    /// Where the `placed` rung runs.
    pub plane: PlaneParams,
}

/// The `structure` leg's payload: what two legs must agree on, and what this
/// rung records about its run.
#[derive(Debug, Clone, Serialize)]
pub struct StructurePayload {
    /// sha256 of the edge list every rung reads.
    pub graph_edges_sha256: String,
    /// sha256 of the seed rows file every rung starts from.
    pub seed_rows_sha256: String,
    /// Node count.
    pub node_count: usize,
    /// Undirected edge count — the unit of the sweep.
    pub edge_count: usize,
    /// The embedding width.
    pub dimensions: usize,
    /// The readout weight of each block `0..=K`; the depth is `len − 1`.
    pub weights: Vec<f64>,
    /// The degree exponent of the seed scale.
    pub beta: f64,
    /// The projection sparsity.
    pub sparsity: f64,
    /// The projection stream's seed.
    pub seed: u64,
    /// The walk operator: the lazy walk `D̃⁻¹(A + I)` over an undirected read.
    pub weighting: &'static str,
    pub direction: &'static str,
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
    /// The wiring rule's fan-out cap.
    pub fan_out: usize,
    /// Iterations timed and filed: the whole run, its transient for the
    /// ladder to cut.
    pub iters_measured: usize,
    /// The implementation that encoded.
    pub operator: &'static str,
}

impl Payload for StructurePayload {
    const IDENTITY_FIELDS: &'static [(&'static str, Nullable)] = &[
        ("graph_edges_sha256", Nullable::NonNull),
        ("seed_rows_sha256", Nullable::NonNull),
        ("node_count", Nullable::NonNull),
        ("edge_count", Nullable::NonNull),
        ("dimensions", Nullable::NonNull),
        ("weights", Nullable::NonNull),
        ("beta", Nullable::NonNull),
        ("sparsity", Nullable::NonNull),
        ("seed", Nullable::NonNull),
        ("weighting", Nullable::NonNull),
        ("direction", Nullable::NonNull),
        ("compute_precision", Nullable::NonNull),
    ];
}

/// The seed rows the engine starts every node from, in node order, as the
/// `f32` the vectors file holds — every entry is `0` or `±√s · d̃^β`, exact in
/// `f32` for the defaults.
fn seed_rows(
    params: &StructureLegParams,
    nodes: &[crate::graph_legs::Node],
    edges: &[(String, String)],
) -> Result<KeyedRows, Box<dyn std::error::Error>> {
    let spec = SeedSpec::new(params.seed, params.dimensions, params.sparsity, params.beta)?;
    let degrees = augmented_degrees(nodes, edges);
    Ok(nodes
        .iter()
        .map(|n| {
            let degree = degrees.get(&n.id).copied().unwrap_or(1);
            let row = spec
                .row(&n.id, degree)
                .into_iter()
                .map(|v| v as f32)
                .collect();
            (n.id.clone(), row)
        })
        .collect())
}

fn build_request(params: &StructureLegParams, sources: &GraphSources) -> StructureRequest {
    StructureRequest::new(
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
    .with_key_column("_row_id")
    .with_dimensions(params.dimensions)
    .with_weights(params.weights.clone())
    .with_beta(params.beta)
    .with_sparsity(params.sparsity)
    .with_seed(params.seed)
}

/// One encoding of `request` on `host`: the table it committed.
async fn encode(
    host: &mut GraphHost,
    request: &StructureRequest,
) -> Result<ResultTableRecord, Box<dyn std::error::Error>> {
    match host {
        GraphHost::InProcess { session, .. } => {
            let (table, _) = session
                .generate_structure_embeddings(request, jammi_db::store::CachePolicy::Bypass)
                .await?;
            Ok(table)
        }
        #[cfg(feature = "plane")]
        GraphHost::Placed { .. } => {
            host.run_placed(
                jammi_ai::jobs::JobSpec::GraphStructure {
                    request: request.clone(),
                    cache: jammi_db::store::CachePolicy::Bypass,
                },
                "structure encoding",
            )
            .await
        }
    }
}

/// Run one `structure` leg and file it: write the unit's inputs (the edge list
/// and the engine's seed rows, the files the PyTorch rung reads), stand the
/// same graph up in a session at `target_partitions`, encode it warm, and
/// persist the key-sorted output beside the leg. Returns the leg and its file
/// name.
pub async fn run_leg(
    params: &StructureLegParams,
) -> Result<(Leg<StructurePayload>, String), Box<dyn std::error::Error>> {
    let shape = params.shape;
    let per_class = (params.nodes / shape.n_classes).max(2);
    let nodes = build_nodes(shape.n_classes, per_class);
    let edges = build_edges(&nodes, shape.fan_out);
    let unit = format!("edges{}", edges.len());
    let rung = params.rung.as_str();
    let stem = leg_stem(rung, &unit, Take::Repeat(params.take as u32));

    let input_dir = params.legs_dir.join(INPUT_DIR).join(&unit);
    let x0 = write_vector_rows(&input_dir, X0_STEM, &seed_rows(params, &nodes, &edges)?)?;
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
        "structure",
        &format!("structure-nodes{}-r{}", params.nodes, params.take),
        &["graph_structure"],
        dir.path(),
    )
    .await?;
    let sources = host.sources().clone();
    add_graph_sources(host.session(), dir.path(), &sources, &nodes, &edges).await?;
    let request = build_request(params, &sources);
    let hops = request.hops()?;

    // Each iteration's wall in run order, and the last one's table.
    let mut iter_wall_s = Vec::with_capacity(params.iterations);
    let mut last = None;
    for _ in 0..params.iterations {
        let start = Instant::now();
        let table = encode(&mut host, &request).await?;
        iter_wall_s.push(start.elapsed().as_secs_f64());
        last = Some(table);
    }
    let peak_rss_bytes = crate::rss::peak_rss_measurement();

    let table = last.ok_or("a structure leg needs at least one iteration")?;
    let rows = read_sorted_vectors(host.session(), &table).await?;
    let ran_on = host.ran_on(&table.table_name).await?;
    let vectors = write_vector_rows(&params.legs_dir, &stem, &rows)?;

    let payload = StructurePayload {
        graph_edges_sha256: edges_file.sha256,
        seed_rows_sha256: x0.file.sha256,
        node_count: nodes.len(),
        edge_count: edges.len(),
        dimensions: params.dimensions,
        weights: params.weights.clone(),
        beta: params.beta,
        sparsity: params.sparsity,
        seed: params.seed,
        weighting: "uniform",
        direction: "undirected",
        compute_precision: "f32",
        rung,
        unit,
        take: params.take,
        target_partitions: params.rung.target_partitions(params.partitions),
        fan_out: shape.fan_out,
        iters_measured: params.iterations,
        operator: "jammi_ai::session::InferenceSession::generate_structure_embeddings",
    };
    let measured = Measured {
        iter_wall_s: Some(iter_wall_s),
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
    let report = leg_report("structure", leg.clone(), |leg| Tiers {
        structure: Some(leg),
        ..Default::default()
    });
    let file = file_leg(&params.legs_dir, &stem, &report)?;
    Ok((leg, file))
}

/// `structure`'s flags: the size sweep, the rungs, and the encoding's
/// parameters. One leg per `(nodes, rung, take)` point.
#[derive(Debug, Clone, clap::Args)]
pub struct StructureArgs {
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
    /// The embedding width.
    #[arg(long, default_value_t = DEFAULT_STRUCTURE_DIMENSIONS)]
    dimensions: usize,
    /// The readout weight of each block `0..=K`, comma-separated; the depth
    /// is one less than their count.
    #[arg(long, value_delimiter = ',', default_values_t = DEFAULT_STRUCTURE_WEIGHTS)]
    weights: Vec<f64>,
    #[arg(long, default_value_t = DEFAULT_STRUCTURE_BETA)]
    beta: f64,
    #[arg(long, default_value_t = DEFAULT_STRUCTURE_SPARSITY)]
    sparsity: f64,
    #[arg(long, default_value_t = DEFAULT_STRUCTURE_SEED)]
    seed: u64,
    /// Where the legs are filed, `<rung>__edges<N>__r<take>.json` with the
    /// vectors beside, and the inputs under `input/edges<N>/`.
    #[arg(long)]
    legs_dir: PathBuf,
    /// Iterations timed, every one filed; the default is the fewest the
    /// ladder settles, and a shorter run files legs it refuses by name.
    #[arg(long, default_value_t = crate::ladder::definition::SpeedInstrument::MIN_RUN)]
    iterations: usize,
    /// Measured repeats, each in a process of its own; the default is the
    /// fewest the ladder measures a rung against itself with.
    #[arg(long, default_value_t = crate::ladder::definition::SpeedInstrument::MIN_REPEATS)]
    takes: usize,
    /// The take a single point's run is filed as.
    #[arg(long, default_value_t = 1)]
    take: usize,
    #[command(flatten)]
    plane: PlaneArgs,
}

impl StructureArgs {
    fn params(&self, nodes: usize, rung: EngineRung, take: usize) -> StructureLegParams {
        StructureLegParams {
            shape: DEFAULT_SHAPE,
            nodes,
            rung,
            partitions: self.partitions,
            dimensions: self.dimensions,
            weights: self.weights.clone(),
            beta: self.beta,
            sparsity: self.sparsity,
            seed: self.seed,
            legs_dir: self.legs_dir.clone(),
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
            .ok_or("structure needs at least one --nodes value")?;
        let first = self.params(n, rung, if points.len() == 1 { self.take } else { 1 });
        let weights = self
            .weights
            .iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let files = legs_per_point(
            &points,
            async move { run_leg(&first).await.map(|(_, file)| vec![file]) },
            |&(nodes, rung, take)| {
                let flags = [
                    ("--nodes", nodes.to_string()),
                    ("--rung", rung.as_str().to_string()),
                    ("--partitions", self.partitions.to_string()),
                    ("--dimensions", self.dimensions.to_string()),
                    ("--weights", weights.clone()),
                    ("--beta", self.beta.to_string()),
                    ("--sparsity", self.sparsity.to_string()),
                    ("--seed", self.seed.to_string()),
                    ("--iterations", self.iterations.to_string()),
                    ("--take", take.to_string()),
                ];
                [
                    "structure".into(),
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

    const DIM: usize = 16;

    /// One cold iteration of a 32-node fixture (4 classes × 8), filed under
    /// `legs_dir`, at 16 lanes.
    async fn leg(
        legs_dir: &std::path::Path,
        nodes: usize,
        weights: &[f64],
        seed: u64,
        partitions: usize,
    ) -> (Leg<StructurePayload>, String) {
        run_leg(&StructureLegParams {
            shape: DEFAULT_SHAPE,
            nodes,
            rung: if partitions == 1 {
                EngineRung::Plan
            } else {
                EngineRung::PlanPartitioned
            },
            partitions,
            dimensions: DIM,
            weights: weights.to_vec(),
            beta: DEFAULT_STRUCTURE_BETA,
            sparsity: DEFAULT_STRUCTURE_SPARSITY,
            seed,
            legs_dir: legs_dir.to_path_buf(),
            iterations: 1,
            take: 1,
            plane: PlaneParams::default(),
        })
        .await
        .expect("structure leg runs")
    }

    /// Encoding the same graph twice on THIS box produces the byte-identical
    /// digest and byte-identical rows, and the seed rows the twin starts
    /// from are the same bytes both times.
    #[tokio::test]
    async fn re_encoding_is_deterministic_on_this_machine() {
        let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let (first, file) = leg(a.path(), 32, &DEFAULT_STRUCTURE_WEIGHTS, 0, 1).await;
        let (second, _) = leg(b.path(), 32, &DEFAULT_STRUCTURE_WEIGHTS, 0, 1).await;
        assert_eq!(
            first.measured.outcome_digest,
            second.measured.outcome_digest
        );
        assert_eq!(
            first.payload.seed_rows_sha256,
            second.payload.seed_rows_sha256
        );
        assert_eq!(file, format!("plan__{}__r1.json", first.payload.unit));
        let rows = read_vector_rows(
            &a.path()
                .join(format!("plan__{}__r1.vectors.f32", first.payload.unit)),
            DIM,
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
        let (one, _) = leg(dir.path(), 32, &DEFAULT_STRUCTURE_WEIGHTS, 0, 1).await;
        let (four, _) = leg(dir.path(), 32, &DEFAULT_STRUCTURE_WEIGHTS, 0, 4).await;
        assert_eq!(
            (one.payload.rung, four.payload.rung),
            ("plan", "plan-partitioned")
        );
        assert_eq!(one.measured.outcome_digest, four.measured.outcome_digest);
    }

    /// The equality above has teeth: another seed or another readout moves
    /// the digest, and the seed rows move with the seed alone.
    #[tokio::test]
    async fn a_perturbed_encoding_changes_the_digest() {
        let dir = tempfile::tempdir().unwrap();
        let (baseline, _) = leg(
            &dir.path().join("base"),
            32,
            &DEFAULT_STRUCTURE_WEIGHTS,
            0,
            1,
        )
        .await;
        let (reseeded, _) = leg(
            &dir.path().join("seed"),
            32,
            &DEFAULT_STRUCTURE_WEIGHTS,
            1,
            1,
        )
        .await;
        assert_ne!(
            reseeded.measured.outcome_digest,
            baseline.measured.outcome_digest
        );
        assert_ne!(
            reseeded.payload.seed_rows_sha256,
            baseline.payload.seed_rows_sha256
        );
        let (shallower, _) = leg(&dir.path().join("weights"), 32, &[0.0, 1.0, 1.0], 0, 1).await;
        assert_ne!(
            shallower.measured.outcome_digest,
            baseline.measured.outcome_digest
        );
        assert_eq!(
            shallower.payload.seed_rows_sha256,
            baseline.payload.seed_rows_sha256
        );
    }

    /// The encoding is not the seed: the walks moved every row.
    #[tokio::test]
    async fn the_encoding_is_not_the_seed() {
        let dir = tempfile::tempdir().unwrap();
        let (encoded, _) = leg(dir.path(), 32, &DEFAULT_STRUCTURE_WEIGHTS, 0, 1).await;
        let unit = &encoded.payload.unit;
        let seed = read_vector_rows(
            &dir.path()
                .join(INPUT_DIR)
                .join(unit)
                .join(format!("{X0_STEM}.vectors.f32")),
            DIM,
        )
        .unwrap();
        let mut seed = seed;
        seed.sort_by(|a, b| a.0.cmp(&b.0));
        assert_ne!(
            Some(vector_rows_digest(&seed)),
            encoded.measured.outcome_digest
        );
    }

    /// The ladder runs end to end over filed legs: the exact edge between
    /// `plan` and `plan-partitioned` by digest, the cross-stack edge by row
    /// agreement. The reference rung's legs are stand-ins — the `plan` legs
    /// filed under `torch` — so the run exercises the comparator's path over
    /// the leg shape, not a second operator.
    #[tokio::test]
    async fn the_ladder_reaches_its_verdicts_over_filed_legs() {
        let dir = tempfile::tempdir().unwrap();
        let legs = dir.path().join("legs");
        for nodes in [32, 64] {
            let (_, plan) = leg(&legs, nodes, &DEFAULT_STRUCTURE_WEIGHTS, 0, 1).await;
            leg(&legs, nodes, &DEFAULT_STRUCTURE_WEIGHTS, 0, 4).await;
            std::fs::copy(
                legs.join(&plan),
                legs.join(plan.replacen("plan__", "torch__", 1)),
            )
            .unwrap();
        }
        let verdict = run_ladder(&LadderArgs {
            workload: Workload::Structure,
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
        assert_eq!(verdict.edges.len(), 2);
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
        assert!(
            matches!(
                verdict.edges[1].outcome,
                Some(OutcomeVerdict::Digest { .. })
            ),
            "{:?}",
            verdict.edges[1].outcome
        );
    }
}
