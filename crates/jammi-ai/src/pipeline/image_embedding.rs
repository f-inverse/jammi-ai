//! Image embedding pipeline with optional rotation-invariant strategy.
//!
//! Generates image embeddings using a vision model (e.g., PatentCLIP),
//! optionally expanding each image into multiple rotation variants.

use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Schema};
use jammi_engine::catalog::result_repo::ResultTableRecord;
use jammi_engine::error::{JammiError, Result};
use jammi_engine::index::sidecar::SidecarIndex;
use jammi_engine::store::writer::ParquetResultWriter;
use jammi_engine::store::ResultStore;

use crate::inference::image_preprocess::rotate_image;
use crate::inference::{adapter::BackendOutput, arrow_to_images};
use crate::model::{ModelSource, ModelTask};
use crate::pipeline::result_sink::ResultSink;
use crate::session::InferenceSession;

/// Strategy for generating embeddings from images.
#[derive(Debug, Clone)]
pub enum EmbeddingStrategy {
    /// One embedding per image, no rotation.
    Single,
    /// Multiple embeddings per image, one per rotation angle (degrees).
    RotationInvariant { angles: Vec<u16> },
}

impl Default for EmbeddingStrategy {
    fn default() -> Self {
        Self::Single
    }
}

/// Orchestrates image embedding generation with optional rotation expansion.
pub struct ImageEmbeddingPipeline<'a> {
    session: &'a InferenceSession,
    result_store: &'a ResultStore,
    strategy: EmbeddingStrategy,
}

impl<'a> ImageEmbeddingPipeline<'a> {
    pub fn new(
        session: &'a InferenceSession,
        result_store: &'a ResultStore,
        strategy: EmbeddingStrategy,
    ) -> Self {
        Self {
            session,
            result_store,
            strategy,
        }
    }

    /// Run the full image embedding pipeline for a source.
    pub async fn run(
        &self,
        source_id: &str,
        model_id: &str,
        image_column: &str,
        key_column: &str,
    ) -> Result<ResultTableRecord> {
        let model_source = ModelSource::parse(model_id);

        // Pre-load model to get embedding dimensions
        let guard = self
            .session
            .model_cache()
            .get_or_load(&model_source, ModelTask::ImageEmbedding, None)
            .await?;
        let embedding_dim = guard
            .model
            .embedding_dim()
            .ok_or_else(|| JammiError::Inference("Model does not support embeddings".into()))?;
        drop(guard);

        // Create result table in catalog
        let canonical_model_id = model_source.to_string();
        let table_info = self.result_store.create_table(
            source_id,
            "embedding",
            &canonical_model_id,
            Some(embedding_dim as i32),
            Some(key_column),
            Some(image_column),
        )?;

        // Scan source for key + image column
        let table_name = self.session.find_table_name(source_id)?;
        let query = self.session.build_source_query(
            source_id,
            &table_name,
            key_column,
            &[image_column.to_string()],
        );

        let df = self
            .session
            .context()
            .sql(&query)
            .await
            .map_err(|e| JammiError::Inference(format!("Failed to scan source: {e}")))?;
        let batches = df
            .collect()
            .await
            .map_err(|e| JammiError::Inference(format!("Failed to collect source: {e}")))?;

        // Set up result sink
        let embedding_schema = jammi_engine::store::schema::embedding_table_schema(embedding_dim);
        let writer = ParquetResultWriter::new(&table_info.parquet_path, embedding_schema)?;
        let sidecar = SidecarIndex::new(embedding_dim)?;
        let checkpoint_interval = self.session.inner_config().embedding.checkpoint_interval;
        let mut sink = ResultSink::for_embeddings(
            writer,
            sidecar,
            self.session.catalog(),
            table_info.table_name.clone(),
            checkpoint_interval,
        );

        // Load model for inference
        let guard = self
            .session
            .model_cache()
            .get_or_load(&model_source, ModelTask::ImageEmbedding, None)
            .await?;

        let batch_size = self.session.inner_config().inference.batch_size.min(16);

        for batch in &batches {
            let keys = batch.column_by_name(key_column).ok_or_else(|| {
                JammiError::Inference(format!("Key column '{key_column}' not found"))
            })?;
            let image_col = batch.column_by_name(image_column).ok_or_else(|| {
                JammiError::Inference(format!("Image column '{image_column}' not found"))
            })?;

            let images = arrow_to_images(&[Arc::clone(image_col)])?;
            let num_rows = images.len();

            // Process in sub-batches
            for chunk_start in (0..num_rows).step_by(batch_size) {
                let chunk_end = (chunk_start + batch_size).min(num_rows);

                match &self.strategy {
                    EmbeddingStrategy::Single => {
                        self.process_chunk_single(
                            &guard.model,
                            keys,
                            &images[chunk_start..chunk_end],
                            chunk_start,
                            embedding_dim,
                            source_id,
                            &canonical_model_id,
                            &mut sink,
                        )?;
                    }
                    EmbeddingStrategy::RotationInvariant { angles } => {
                        self.process_chunk_rotated(
                            &guard.model,
                            keys,
                            &images[chunk_start..chunk_end],
                            chunk_start,
                            angles,
                            embedding_dim,
                            source_id,
                            &canonical_model_id,
                            &mut sink,
                        )?;
                    }
                }
            }
        }

        drop(guard);

        let (row_count, index) = sink.finalize()?;

        if let Some(ref idx) = index {
            if let Some(ref idx_path) = table_info.index_path {
                idx.save(idx_path)?;
            }
        }

        self.result_store
            .finalize(
                self.session.context(),
                &table_info.table_name,
                &table_info.parquet_path,
                row_count,
            )
            .await?;

        self.session
            .catalog()
            .get_result_table(&table_info.table_name)?
            .ok_or_else(|| {
                JammiError::Catalog(format!(
                    "Result table '{}' not found after finalization",
                    table_info.table_name
                ))
            })
    }

    /// Process a chunk of images without rotation (1:1 mapping).
    #[allow(clippy::too_many_arguments)]
    fn process_chunk_single(
        &self,
        model: &crate::model::LoadedModel,
        keys: &ArrayRef,
        images: &[Option<image::DynamicImage>],
        offset: usize,
        embedding_dim: usize,
        source_id: &str,
        model_id: &str,
        sink: &mut ResultSink<'_>,
    ) -> Result<()> {
        // Build a binary array from the valid images
        let mut row_ids = Vec::with_capacity(images.len());
        for i in 0..images.len() {
            let key = get_key_string(keys, offset + i)?;
            row_ids.push(key);
        }

        let image_arrays = self.images_to_binary_array(images)?;
        let output = model.forward(&[image_arrays], ModelTask::ImageEmbedding)?;

        let result_batch =
            self.build_result_batch(&row_ids, source_id, model_id, &output, embedding_dim)?;
        sink.write_batch(&result_batch)
    }

    /// Process a chunk with rotation expansion (N:1 per image).
    #[allow(clippy::too_many_arguments)]
    fn process_chunk_rotated(
        &self,
        model: &crate::model::LoadedModel,
        keys: &ArrayRef,
        images: &[Option<image::DynamicImage>],
        offset: usize,
        angles: &[u16],
        embedding_dim: usize,
        source_id: &str,
        model_id: &str,
        sink: &mut ResultSink<'_>,
    ) -> Result<()> {
        // For each angle, rotate all images and embed as a batch
        for &angle in angles {
            let mut row_ids = Vec::with_capacity(images.len());
            let mut rotated_images: Vec<Option<image::DynamicImage>> =
                Vec::with_capacity(images.len());

            for (i, img) in images.iter().enumerate() {
                let key = get_key_string(keys, offset + i)?;
                row_ids.push(format!("{key}_r{angle}"));
                rotated_images.push(img.as_ref().map(|im| rotate_image(im, angle)));
            }

            let image_arrays = self.images_to_binary_array(&rotated_images)?;
            let output = model.forward(&[image_arrays], ModelTask::ImageEmbedding)?;

            let result_batch =
                self.build_result_batch(&row_ids, source_id, model_id, &output, embedding_dim)?;
            sink.write_batch(&result_batch)?;
        }
        Ok(())
    }

    /// Convert a slice of optional images to an Arrow BinaryArray.
    /// Valid images are PNG-encoded; None values become null.
    fn images_to_binary_array(&self, images: &[Option<image::DynamicImage>]) -> Result<ArrayRef> {
        let mut builder = arrow::array::BinaryBuilder::new();
        for img in images {
            match img {
                Some(im) => {
                    let mut buf = Vec::new();
                    im.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
                        .map_err(|e| {
                            JammiError::Inference(format!("Failed to encode image to PNG: {e}"))
                        })?;
                    builder.append_value(&buf);
                }
                None => builder.append_null(),
            }
        }
        Ok(Arc::new(builder.finish()) as ArrayRef)
    }

    /// Build a result RecordBatch in the embedding output schema.
    fn build_result_batch(
        &self,
        row_ids: &[String],
        source_id: &str,
        model_id: &str,
        output: &BackendOutput,
        embedding_dim: usize,
    ) -> Result<RecordBatch> {
        use crate::inference::schema::build_prefix_columns;

        let row_count = row_ids.len();
        let keys_array = Arc::new(StringArray::from(row_ids.to_vec())) as ArrayRef;

        let prefix = build_prefix_columns(
            &keys_array,
            source_id,
            model_id,
            &output.row_status,
            &output.row_errors,
            0.0, // latency tracked elsewhere
            row_count,
        );

        let adapter = crate::inference::adapter::EmbeddingAdapter::new(embedding_dim);
        let task_columns =
            crate::inference::adapter::OutputAdapter::adapt(&adapter, output, row_count)?;

        let mut all_columns = prefix;
        all_columns.extend(task_columns);

        let schema = crate::inference::schema::build_output_schema(
            &ModelTask::ImageEmbedding,
            &Arc::new(Schema::empty()),
            "",
            Some(embedding_dim),
        )?;

        RecordBatch::try_new(schema, all_columns)
            .map_err(|e| JammiError::Inference(format!("Failed to build result batch: {e}")))
    }
}

/// Extract a string key from an Arrow array at the given index.
fn get_key_string(keys: &ArrayRef, i: usize) -> Result<String> {
    use arrow::array::Array;
    if keys.is_null(i) {
        return Ok(String::new());
    }
    match keys.data_type() {
        DataType::Utf8 => Ok(keys
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|a| a.value(i).to_string())
            .unwrap_or_default()),
        DataType::Int64 => Ok(keys
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .map(|a| a.value(i).to_string())
            .unwrap_or_default()),
        _ => Ok(arrow::compute::cast(keys, &DataType::Utf8)
            .ok()
            .and_then(|casted| {
                casted
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .map(|a| a.value(i).to_string())
            })
            .unwrap_or_else(|| format!("row_{i}"))),
    }
}
