use arrow::array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray, UInt32Array,
};

use jammi_engine::catalog::Catalog;
use jammi_engine::error::{JammiError, Result};
use jammi_engine::index::sidecar::SidecarIndex;
use jammi_engine::index::VectorIndex;
use jammi_engine::store::writer::ParquetResultWriter;

/// Streams InferenceExec output to Parquet + optional ANN index,
/// filtering failed rows for embedding tables.
pub struct ResultSink<'a> {
    writer: ParquetResultWriter,
    index: Option<SidecarIndex>,
    catalog: &'a Catalog,
    table_name: String,
    is_embedding: bool,
    checkpoint_interval: usize,
    batch_num: usize,
}

impl<'a> ResultSink<'a> {
    /// Create a sink for embedding results (filters OK rows, feeds sidecar index).
    pub fn for_embeddings(
        writer: ParquetResultWriter,
        index: SidecarIndex,
        catalog: &'a Catalog,
        table_name: String,
        checkpoint_interval: usize,
    ) -> Self {
        Self {
            writer,
            index: Some(index),
            catalog,
            table_name,
            is_embedding: true,
            checkpoint_interval,
            batch_num: 0,
        }
    }

    /// Create a sink for inference results (writes all rows, no index).
    pub fn for_inference(
        writer: ParquetResultWriter,
        catalog: &'a Catalog,
        table_name: String,
    ) -> Self {
        Self {
            writer,
            index: None,
            catalog,
            table_name,
            is_embedding: false,
            checkpoint_interval: 0,
            batch_num: 0,
        }
    }

    /// Write a batch: filter OK rows (for embeddings), write to Parquet, feed index.
    pub fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        self.batch_num += 1;
        if self.is_embedding {
            let (ok_batch, row_ids, vectors) = filter_ok_and_extract_vectors(batch)?;
            if ok_batch.num_rows() > 0 {
                self.writer.write_batch(&ok_batch)?;
                if let Some(ref mut index) = self.index {
                    for (id, vec) in row_ids.iter().zip(vectors.iter()) {
                        index.add(id, vec)?;
                    }
                }
            }
        } else {
            self.writer.write_batch(batch)?;
        }
        if self.checkpoint_interval > 0 && self.batch_num % self.checkpoint_interval == 0 {
            self.catalog
                .set_checkpoint(&self.table_name, self.batch_num)?;
        }
        Ok(())
    }

    /// Close the writer and build the index (if embedding).
    pub fn finalize(self) -> Result<(usize, Option<SidecarIndex>)> {
        let row_count = self.writer.close()?;
        let index = match self.index {
            Some(mut idx) if idx.len() > 0 => {
                idx.build()?;
                Some(idx)
            }
            _ => None,
        };
        Ok((row_count, index))
    }
}

/// Filter a batch to only OK rows, transform to embedding schema,
/// and extract `_row_id` + `vector` columns.
///
/// Input schema: `_row_id, _source, _model, _status, _error, _latency_ms, vector`
/// Output schema: `_row_id, _source_id, _model_id, vector`
pub fn filter_ok_and_extract_vectors(
    batch: &RecordBatch,
) -> Result<(RecordBatch, Vec<String>, Vec<Vec<f32>>)> {
    let status = batch
        .column_by_name("_status")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| JammiError::Inference("Missing _status column".into()))?;

    // Build filter indices for OK rows
    let keep: Vec<u32> = (0..status.len())
        .filter(|&i| status.value(i) == "ok")
        .map(|i| i as u32)
        .collect();
    let indices = UInt32Array::from(keep);

    // Extract and filter the columns we need
    let row_ids_col = batch
        .column_by_name("_row_id")
        .ok_or_else(|| JammiError::Inference("Missing _row_id".into()))?;
    let source_col = batch
        .column_by_name("_source")
        .ok_or_else(|| JammiError::Inference("Missing _source".into()))?;
    let model_col = batch
        .column_by_name("_model")
        .ok_or_else(|| JammiError::Inference("Missing _model".into()))?;
    let vector_col = batch
        .column_by_name("vector")
        .ok_or_else(|| JammiError::Inference("Missing vector".into()))?;

    let filtered_row_ids = arrow::compute::take(row_ids_col.as_ref(), &indices, None)
        .map_err(|e| JammiError::Other(format!("Arrow take: {e}")))?;
    let filtered_source = arrow::compute::take(source_col.as_ref(), &indices, None)
        .map_err(|e| JammiError::Other(format!("Arrow take: {e}")))?;
    let filtered_model = arrow::compute::take(model_col.as_ref(), &indices, None)
        .map_err(|e| JammiError::Other(format!("Arrow take: {e}")))?;
    let filtered_vector = arrow::compute::take(vector_col.as_ref(), &indices, None)
        .map_err(|e| JammiError::Other(format!("Arrow take: {e}")))?;

    // Get dimensions from the vector column to build the embedding schema
    let dims = vector_col
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .map(|fsl| fsl.value_length() as usize)
        .unwrap_or(0);

    let embedding_schema = jammi_engine::store::schema::embedding_table_schema(dims);

    // Build the output RecordBatch with renamed columns (_source → _source_id, _model → _model_id)
    let ok_batch = RecordBatch::try_new(
        embedding_schema,
        vec![
            filtered_row_ids.clone(),
            filtered_source,
            filtered_model,
            filtered_vector,
        ],
    )
    .map_err(|e| JammiError::Other(format!("RecordBatch build: {e}")))?;

    // Extract row_ids and vectors for the sidecar index
    let row_ids: Vec<String> = filtered_row_ids
        .as_any()
        .downcast_ref::<StringArray>()
        .map(|a| (0..a.len()).map(|i| a.value(i).to_string()).collect())
        .unwrap_or_default();

    let vectors: Vec<Vec<f32>> = match ok_batch
        .column_by_name("vector")
        .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
    {
        Some(fsl) => (0..fsl.len())
            .map(|i| {
                let v = fsl.value(i);
                let a = v.as_any().downcast_ref::<Float32Array>().ok_or_else(|| {
                    JammiError::Inference("Expected Float32Array in vector column".into())
                })?;
                Ok((0..a.len()).map(|j| a.value(j)).collect())
            })
            .collect::<Result<Vec<Vec<f32>>>>()?,
        None => Vec::new(),
    };

    Ok((ok_batch, row_ids, vectors))
}
