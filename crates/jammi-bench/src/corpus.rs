//! Corpus materialization, registration, and read-back for the scale proofs.
//!
//! Two corpus *origins* feed the harness, both flowing through one read path:
//!
//! * **Synthetic** — a seeded LCG produces a deterministic `(row_id, vector)`
//!   corpus, written to a Parquet object through the *engine's own* writer
//!   ([`ObjectParquetWriter`]) so both the streamed search and the naive
//!   baseline read the exact production path. The generator streams its rows
//!   into fixed-size `RecordBatch`es and drops each after writing, so
//!   materializing a 4-million-row corpus never holds more than one batch
//!   resident — the harness's own footprint stays a small constant that the RSS
//!   delta cancels.
//! * **Committed Parquet** — a real-embedding corpus emitted once on the GPU box
//!   and committed as an artifact. [`load_vectors`] reads its `(_row_id,
//!   vector)` rows back through the same register / [`extend_with_fixed_size_list_f32`]
//!   path the synthetic loader uses, so a frozen sidecar can be built over the
//!   identical vectors the exact oracle scores.
//!
//! Both origins register under `jammi.{table_name}` and are read by the engine's
//! `exact_vector_search` — the read side never branches on origin.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use datafusion::prelude::SessionContext;
use datafusion::sql::TableReference;
use futures::TryStreamExt;

use jammi_db::storage::{JammiObjectStore, ObjectParquetWriter, StorageRegistry, StorageUrl};
use jammi_db::store::schema::embedding_table_schema;
use jammi_db::store::vectors::extend_with_fixed_size_list_f32;

/// A 64-bit LCG (Numerical-Recipes constants) mapped to `f32` in `[-1, 1)`.
///
/// The same constants the engine's own determinism test uses, so a corpus
/// generated here is bit-identical to one the in-engine test would build for
/// the same seed/shape — no entropy, no rng crate, fully reproducible.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // High 24 bits as a fraction, away from the low-bit LCG correlation.
        let frac = ((self.state >> 40) as f32) / ((1u64 << 24) as f32);
        frac * 2.0 - 1.0
    }
}

/// Generate a single query vector from the LCG so the search is over a
/// realistic (non-degenerate) query rather than a constant.
pub fn lcg_query(seed: u64, dim: usize) -> Vec<f32> {
    let mut lcg = Lcg::new(seed);
    (0..dim).map(|_| lcg.next_f32()).collect()
}

/// Build one `RecordBatch` of `count` rows starting at `start_row`, filling the
/// engine embedding schema. `_source_id`/`_model_id` are constant — they are
/// not read by the search, only the schema requires them present.
fn batch(start_row: usize, count: usize, dim: usize, lcg: &mut Lcg) -> RecordBatch {
    let schema = embedding_table_schema(dim);
    let mut row_ids: Vec<String> = Vec::with_capacity(count);
    let mut flat: Vec<f32> = Vec::with_capacity(count * dim);
    for r in 0..count {
        // Zero-padded so lexical `_row_id` order matches numeric order, which
        // keeps the deterministic tie-break order legible.
        row_ids.push(format!("row_{:09}", start_row + r));
        for _ in 0..dim {
            flat.push(lcg.next_f32());
        }
    }
    let id_refs: Vec<&str> = row_ids.iter().map(String::as_str).collect();
    let values = Arc::new(Float32Array::from(flat));
    let item = Arc::new(arrow::datatypes::Field::new(
        "item",
        arrow::datatypes::DataType::Float32,
        false,
    ));
    let vectors = FixedSizeListArray::try_new(item, dim as i32, values, None)
        .expect("fixed-size list build is total over a flat f32 buffer of count*dim");
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(id_refs)) as ArrayRef,
            Arc::new(StringArray::from(vec!["src"; count])),
            Arc::new(StringArray::from(vec!["model"; count])),
            Arc::new(vectors),
        ],
    )
    .expect("record batch matches the embedding schema by construction")
}

/// Generate `rows` LCG vectors of width `dim` and write them to a Parquet
/// object at `path`, in `batch_rows`-sized chunks, through the engine writer.
///
/// Returns the [`StorageUrl`] the object was written to. The corpus is never
/// held whole in memory: each chunk is generated, written, and dropped, so the
/// generator's peak footprint is one batch regardless of `rows`.
pub async fn materialize(
    path: &Path,
    rows: usize,
    dim: usize,
    seed: u64,
    batch_rows: usize,
) -> Result<StorageUrl, Box<dyn std::error::Error>> {
    let schema = embedding_table_schema(dim);
    let url = StorageUrl::parse(path.to_str().ok_or("corpus path is not valid UTF-8")?)?;
    let registry = StorageRegistry::new();
    let driver = registry.driver_for(&url, None)?;
    let handle = JammiObjectStore::new(driver, url.clone());
    let mut writer = ObjectParquetWriter::open(&handle, Arc::clone(&schema)).await?;

    let mut lcg = Lcg::new(seed);
    let mut written = 0;
    while written < rows {
        let count = batch_rows.min(rows - written);
        let rb = batch(written, count, dim, &mut lcg);
        writer.write_batch(&rb).await?;
        written += count;
    }
    writer.close().await?;
    Ok(url)
}

/// Parse a filesystem path into the [`StorageUrl`] a corpus was written to, so
/// a separate process can register a corpus the parent materialized.
pub fn storage_url(path: &Path) -> Result<StorageUrl, Box<dyn std::error::Error>> {
    Ok(StorageUrl::parse(
        path.to_str().ok_or("corpus path is not valid UTF-8")?,
    )?)
}

/// Register a materialized corpus under the `jammi.{table_name}` identifier
/// `exact_vector_search` resolves, returning a live context. Runs under the
/// engine's default schema settings (`schema_force_view_types` on), so the read
/// path is the production one.
pub async fn register(
    url: &StorageUrl,
    table_name: &str,
) -> Result<SessionContext, Box<dyn std::error::Error>> {
    let ctx = SessionContext::new();
    ctx.register_parquet(
        TableReference::bare(format!("jammi.{table_name}")),
        url.as_str(),
        datafusion::datasource::file_format::options::ParquetReadOptions::default(),
    )
    .await?;
    Ok(ctx)
}

/// Read the full `(_row_id, vector)` corpus back from a registered Parquet table.
///
/// Streams the same `SELECT _row_id, vector` scan `exact_vector_search` runs and
/// decodes each batch through [`extend_with_fixed_size_list_f32`] — the engine's
/// single sanctioned vector downcast — so a sidecar built from these rows is
/// built over exactly the vectors the exact oracle scores. The `_row_id` column
/// is cast to `Utf8` before downcast to cover the `Utf8View`/`LargeUtf8` families
/// the parquet reader can surface, matching the engine's own read.
///
/// Holds the whole corpus resident: this is the recall path's read-back, where
/// the sidecar and the oracle both need every vector — not the streamed
/// bounded-RSS path. Callers point it at the hermetic slice (or a deterministic
/// subset of a larger corpus), never the unbounded binding tier.
pub async fn load_vectors(
    ctx: &SessionContext,
    table_name: &str,
) -> Result<Vec<(String, Vec<f32>)>, Box<dyn std::error::Error>> {
    let df = ctx
        .sql(&format!(
            "SELECT _row_id, vector FROM \"jammi.{table_name}\""
        ))
        .await?;
    let mut stream = df.execute_stream().await?;

    let mut out: Vec<(String, Vec<f32>)> = Vec::new();
    let mut vectors: Vec<Vec<f32>> = Vec::new();
    while let Some(batch) = stream.try_next().await? {
        let row_ids_col = batch
            .column_by_name("_row_id")
            .ok_or("missing _row_id in corpus load")?;
        let row_ids_utf8 = cast(row_ids_col, &DataType::Utf8)?;
        let row_ids = row_ids_utf8
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("_row_id is not a Utf8-castable string type")?;
        vectors.clear();
        extend_with_fixed_size_list_f32(&batch, table_name, "vector", &mut vectors)?;
        // `extend_with_fixed_size_list_f32` appends exactly one Vec<f32> per row,
        // so the batch's vectors map 1:1 with `row_ids`.
        for (offset, vec) in vectors.drain(..).enumerate() {
            out.push((row_ids.value(offset).to_string(), vec));
        }
    }
    Ok(out)
}

/// The deterministic first-`n` projection of a corpus by sorted `_row_id`.
///
/// Sorts the `(row_id, vector)` rows by ascending `_row_id` and returns the
/// first `n`. `_row_id` is the table's unique primary key, so the sort is a
/// total order with no ties — the same `n` rows come back for the same corpus
/// on every box, independent of parquet row-group layout or scan order. This is
/// the provable-projection slice: a hermetic recall fixture is this subset of
/// the *same* committed vectors, so it cannot drift from the full corpus the way
/// an independently generated fixture could.
///
/// `n` larger than the corpus yields the whole corpus (sorted), never a panic.
pub fn sorted_row_id_subset(
    mut rows: Vec<(String, Vec<f32>)>,
    n: usize,
) -> Vec<(String, Vec<f32>)> {
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    rows.truncate(n);
    rows
}
