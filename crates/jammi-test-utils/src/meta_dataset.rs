//! A synthetic episodic meta-dataset: tasks of rows whose outcome is a
//! per-task linear function of a small feature vector, as the two things a
//! context predictor trains from — a source table keyed `_row_id` carrying
//! each row's task and outcome, and an embedding result table holding the
//! row's features as its vector under the same key.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use jammi_db::session::QueryContext;
use jammi_db::source::{FileFormat, SourceConnection};
use jammi_db::store::ResultStore;
use parquet::arrow::ArrowWriter;

/// The width of every row's feature vector.
pub const FEATURE_DIM: usize = 4;

/// The model id the embedding table records as its producer.
pub const EMBEDDING_MODEL_ID: &str = "synthetic-embed";

/// splitmix64 — a deterministic generator so the dataset is reproducible
/// without a test-only rng dependency.
pub struct Rng(pub u64);

impl Rng {
    /// The next draw, in `[-1, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

/// One synthetic row: its key (`_row_id`, the same identity the embedding table
/// keys its vector by), the task it belongs to, its feature vector `x`, and its
/// outcome `y`.
pub struct Row {
    pub id: String,
    pub task: String,
    pub x: Vec<f32>,
    pub y: f64,
}

/// A linear-function meta-dataset: `n_tasks` tasks, each with a random
/// weight vector `w` and `rows_per_task` rows of `y = w · x`, in task-major
/// order.
pub fn linear_tasks(n_tasks: usize, rows_per_task: usize, seed: u64) -> Vec<Row> {
    let mut rng = Rng(seed);
    let mut rows = Vec::with_capacity(n_tasks * rows_per_task);
    for t in 0..n_tasks {
        let w: Vec<f32> = (0..FEATURE_DIM).map(|_| rng.next_f32()).collect();
        for r in 0..rows_per_task {
            let x: Vec<f32> = (0..FEATURE_DIM).map(|_| rng.next_f32()).collect();
            let y: f64 = x.iter().zip(&w).map(|(xi, wi)| (xi * wi) as f64).sum();
            rows.push(Row {
                id: format!("t{t}_r{r}"),
                task: format!("task_{t}"),
                x,
                y,
            });
        }
    }
    rows
}

/// Write `rows` as the source parquet under `dir` — `_row_id` (the key,
/// shared with the embedding table's identity), `task`, `y` — and return
/// the connection that registers it. The split predicate scopes a context
/// over this source on the embedding table's key column, so the key column
/// is a real source column: naming it `_row_id` shares one identity end to
/// end.
pub fn write_source(dir: &Path, rows: &[Row]) -> SourceConnection {
    let schema = Arc::new(Schema::new(vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("task", DataType::Utf8, false),
        Field::new("y", DataType::Float64, false),
    ]));
    let ids: Vec<&str> = rows.iter().map(|r| r.id.as_str()).collect();
    let tasks: Vec<&str> = rows.iter().map(|r| r.task.as_str()).collect();
    let ys: Vec<f64> = rows.iter().map(|r| r.y).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(ids)) as ArrayRef,
            Arc::new(StringArray::from(tasks)),
            Arc::new(Float64Array::from(ys)),
        ],
    )
    .unwrap();
    let path = dir.join("source.parquet");
    let file = std::fs::File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    SourceConnection {
        url: Some(format!("file://{}", path.to_str().unwrap())),
        format: Some(FileFormat::Parquet),
        ..Default::default()
    }
}

/// Materialize `source_id`'s embedding result table — each row's vector its
/// feature `x`, keyed `_row_id` — through the store's own embedding-table
/// writer, so it carries a real sidecar ANN index and every session over the
/// store's catalog and result root reads it the way production reads one.
pub async fn materialize_embeddings(
    store: &ResultStore,
    ctx: &QueryContext,
    source_id: &str,
    rows: &[Row],
) {
    let pairs: Vec<(String, Vec<f32>)> = rows.iter().map(|r| (r.id.clone(), r.x.clone())).collect();
    let (descriptor, env, inputs) =
        crate::synthetic_seed_contract(EMBEDDING_MODEL_ID, source_id, FEATURE_DIM);
    store
        .materialize_embedding_table(
            ctx,
            jammi_db::store::EmbeddingTableSpec {
                source_id,
                model_id: EMBEDDING_MODEL_ID,
                derived_from: None,
                dimensions: FEATURE_DIM,
                key_column: Some("_row_id"),
                text_columns: None,
            },
            &pairs,
            jammi_db::store::manifest::Materialization::new(&descriptor, &env, inputs),
            None,
        )
        .await
        .unwrap();
}
