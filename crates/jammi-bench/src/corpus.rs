//! Synthetic corpus generation and on-disk materialization for the scale proof.
//!
//! A seeded LCG produces a deterministic `(row_id, vector)` corpus, written to a
//! Parquet object through the *engine's own* writer (`ObjectParquetWriter`) so
//! both the streamed search and the naive baseline read the exact production
//! path. The generator streams its rows into fixed-size `RecordBatch`es and
//! drops each after writing, so materializing a 4-million-row corpus never holds
//! more than one batch resident — the harness's own footprint stays a small
//! constant that the RSS delta cancels.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use datafusion::prelude::SessionContext;
use datafusion::sql::TableReference;

use jammi_db::storage::{JammiObjectStore, ObjectParquetWriter, StorageRegistry, StorageUrl};
use jammi_db::store::schema::embedding_table_schema;

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
