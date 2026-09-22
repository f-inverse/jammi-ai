//! What a result table's physical row order costs its readers, measured on
//! the engine's own Parquet writer and DataFusion's own scan, printed under
//! `--nocapture`; asserted only for consistency.
//!
//! Two files of the embedding table schema hold the same rows: one clustered
//! by `_row_id`, one in a cost-like order (a fixed permutation of the keys).
//! Per file: a key lookup (`WHERE _row_id = ?`, the query-by-example read and
//! any keyed SQL over the table), a key join against a ten-key probe set, and
//! a full read ordered by `_row_id` (the ordered read-backs), each with the
//! scan's own row-group and byte metrics. Row counts come from
//! `JAMMI_ORDER_MEASURE_ROWS` (comma-separated), `65536` when unset.

use std::sync::Arc;
use std::time::Instant;

use arrow::array::{FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field};
use datafusion::physical_plan::metrics::MetricValue;
use datafusion::physical_plan::{collect, ExecutionPlan, ExecutionPlanVisitor};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use tempfile::TempDir;

use jammi_db::storage::{ObjectParquetWriter, StorageRegistry, StorageUrl};
use jammi_db::store::schema::embedding_table_schema;

const DIMS: usize = 32;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

/// Rows `keys[i]` with a vector derived from the key, in the embedding
/// table schema.
fn batch(keys: &[u64]) -> RecordBatch {
    let mut values = Vec::with_capacity(keys.len() * DIMS);
    for &k in keys {
        let mut rng = Lcg(k);
        values.extend((0..DIMS).map(|_| rng.next() as f32 / (1u64 << 31) as f32 - 0.5));
    }
    let vectors = FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        DIMS as i32,
        Arc::new(Float32Array::from(values)),
        None,
    );
    RecordBatch::try_new(
        embedding_table_schema(DIMS),
        vec![
            Arc::new(StringArray::from_iter_values(
                keys.iter().map(|k| format!("{k:08}")),
            )),
            Arc::new(StringArray::from(vec!["corpus"; keys.len()])),
            Arc::new(StringArray::from(vec!["model"; keys.len()])),
            Arc::new(vectors),
            Arc::new(StringArray::from_iter_values(
                keys.iter().map(|k| format!("{:064x}", k * 2654435761)),
            )),
        ],
    )
    .unwrap()
}

/// Write `keys` in the given order through the engine's own writer, 8192
/// rows per batch, and return the file's URL.
async fn write(dir: &std::path::Path, name: &str, keys: &[u64]) -> StorageUrl {
    let url = StorageUrl::parse(&format!("file://{}/{name}.parquet", dir.display())).unwrap();
    let handle = StorageRegistry::new().handle_for(&url, None).unwrap();
    let mut writer = ObjectParquetWriter::open(&handle, embedding_table_schema(DIMS))
        .await
        .unwrap();
    for run in keys.chunks(8192) {
        writer.write_batch(&batch(run)).await.unwrap();
    }
    writer.close().await.unwrap();
    url
}

/// The scan metrics summed over every `DataSourceExec` in `plan`.
#[derive(Default)]
struct Scan {
    row_groups_matched: usize,
    row_groups_pruned: usize,
    bytes_scanned: usize,
}

impl ExecutionPlanVisitor for Scan {
    type Error = ();
    fn pre_visit(&mut self, plan: &dyn ExecutionPlan) -> Result<bool, ()> {
        if let Some(metrics) = plan.metrics() {
            for metric in metrics.iter() {
                match metric.value() {
                    MetricValue::Count { name, count } if name == "bytes_scanned" => {
                        self.bytes_scanned += count.value();
                    }
                    MetricValue::PruningMetrics {
                        name,
                        pruning_metrics,
                    } if name == "row_groups_pruned_statistics" => {
                        self.row_groups_matched += pruning_metrics.matched();
                        self.row_groups_pruned += pruning_metrics.pruned();
                    }
                    _ => {}
                }
            }
        }
        Ok(true)
    }
}

async fn measure(ctx: &SessionContext, label: &str, sql: &str, repeat: usize) {
    let mut scan = Scan::default();
    let mut rows = 0;
    let start = Instant::now();
    for _ in 0..repeat {
        let plan = ctx
            .sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let out = collect(Arc::clone(&plan), ctx.task_ctx()).await.unwrap();
        rows += out.iter().map(|b| b.num_rows()).sum::<usize>();
        datafusion::physical_plan::accept(plan.as_ref(), &mut scan).unwrap();
    }
    let per = start.elapsed().as_secs_f64() * 1000.0 / repeat as f64;
    println!(
        "{label:<28} {:>8.2} {:>8} {:>10} {:>10} {:>12}",
        per,
        rows / repeat,
        scan.row_groups_matched / repeat,
        scan.row_groups_pruned / repeat,
        scan.bytes_scanned / repeat
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn key_lookup_join_and_ordered_read_by_physical_order() {
    let sizes: Vec<usize> = std::env::var("JAMMI_ORDER_MEASURE_ROWS")
        .ok()
        .map(|v| v.split(',').map(|n| n.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![65_536]);
    let dir = TempDir::new().unwrap();
    for n in sizes {
        let keyed: Vec<u64> = (0..n as u64).collect();
        // A cost-like order: the same keys, permuted by a fixed hash.
        let mut costed = keyed.clone();
        costed.sort_by_key(|k| Lcg(*k).next());
        let probe: Vec<u64> = (0..10).map(|i| (i * 7919 + 13) as u64 % n as u64).collect();
        for (order, keys) in [("key", &keyed), ("cost", &costed)] {
            let url = write(dir.path(), &format!("{order}-{n}"), keys).await;
            let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
            ctx.register_parquet("t", url.as_str(), ParquetReadOptions::default())
                .await
                .unwrap();
            let probe_list = probe
                .iter()
                .map(|k| format!("('{k:08}')"))
                .collect::<Vec<_>>()
                .join(",");
            ctx.sql(&format!("CREATE TABLE probe(k VARCHAR) AS VALUES {probe_list}"))
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
            println!("== physical order={order} rows={n} ==");
            println!(
                "{:<28} {:>8} {:>8} {:>10} {:>10} {:>12}",
                "read", "ms", "rows", "rg_matched", "rg_pruned", "bytes"
            );
            for &k in &probe[..3] {
                measure(
                    &ctx,
                    &format!("lookup _row_id='{k:08}'"),
                    &format!("SELECT vector FROM t WHERE _row_id = '{k:08}'"),
                    5,
                )
                .await;
            }
            measure(
                &ctx,
                "join 10 keys",
                "SELECT t._row_id, t.vector FROM t JOIN probe ON t._row_id = probe.k",
                3,
            )
            .await;
            measure(
                &ctx,
                "ordered full read",
                "SELECT _row_id, vector FROM t ORDER BY _row_id",
                1,
            )
            .await;
        }
    }
}
