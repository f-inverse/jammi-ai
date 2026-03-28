//! Rotation expansion for image embedding.
//!
//! Rotation is data preparation — it transforms a source by creating
//! multiple rotated copies of each image before embedding. The actual
//! embedding generation flows through the standard `EmbeddingPipeline`.

use std::sync::Arc;

use arrow::array::{
    Array, BinaryArray, BinaryViewArray, LargeBinaryArray, StringArray, StringViewArray,
};
use arrow::datatypes::DataType;
use jammi_engine::error::{JammiError, Result};
use jammi_engine::source::{FileFormat, SourceConnection, SourceType};

use crate::inference::image_preprocess::rotate_image;
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

/// Expand a source by rotating each image at multiple angles.
///
/// Creates a new Parquet source where each original row produces N rows
/// (one per angle). Row IDs are suffixed with `_r{angle}`. The new source
/// is registered and its name is returned.
pub async fn expand_with_rotations(
    session: &InferenceSession,
    source_id: &str,
    image_column: &str,
    key_column: &str,
    angles: &[u16],
) -> Result<String> {
    // Scan the original source
    let table_name = session.find_table_name(source_id)?;
    let query = session.build_source_query(
        source_id,
        &table_name,
        key_column,
        &[image_column.to_string()],
    );
    let batches = session.sql(&query).await?;

    // Expand: for each row × each angle, rotate and collect
    let mut expanded_keys: Vec<String> = Vec::new();
    let mut expanded_images: Vec<Vec<u8>> = Vec::new();

    for batch in &batches {
        let keys = extract_keys(batch, key_column)?;
        let images = extract_image_bytes(batch, image_column)?;

        for (key, image_bytes) in keys.iter().zip(&images) {
            let img = image::load_from_memory(image_bytes).map_err(|e| {
                JammiError::Inference(format!("Failed to decode image for key '{key}': {e}"))
            })?;

            for &angle in angles {
                let rotated = rotate_image(&img, angle);
                let mut buf = Vec::new();
                rotated
                    .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
                    .map_err(|e| {
                        JammiError::Inference(format!("Failed to encode rotated image: {e}"))
                    })?;
                expanded_keys.push(format!("{key}_r{angle}"));
                expanded_images.push(buf);
            }
        }
    }

    // Write expanded Parquet
    let expanded_name = format!("{source_id}__rotated");
    let parquet_path = session
        .inner_config()
        .artifact_dir
        .join(format!("{expanded_name}.parquet"));

    write_expanded_parquet(
        &parquet_path,
        key_column,
        image_column,
        &expanded_keys,
        &expanded_images,
    )?;

    // Register as a new source
    session
        .add_source(
            &expanded_name,
            SourceType::Local,
            SourceConnection {
                url: Some(format!("file://{}", parquet_path.display())),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await?;

    Ok(expanded_name)
}

/// Write the expanded key + image data to a Parquet file.
fn write_expanded_parquet(
    path: &std::path::Path,
    key_column: &str,
    image_column: &str,
    keys: &[String],
    images: &[Vec<u8>],
) -> Result<()> {
    use arrow::array::ArrayRef;
    use arrow::datatypes::{Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    let schema = Arc::new(Schema::new(vec![
        Field::new(key_column, DataType::Utf8, false),
        Field::new(image_column, DataType::Binary, false),
    ]));

    let key_array = Arc::new(StringArray::from(keys.to_vec())) as ArrayRef;
    let image_refs: Vec<&[u8]> = images.iter().map(|v| v.as_slice()).collect();
    let image_array = Arc::new(BinaryArray::from(image_refs)) as ArrayRef;

    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![key_array, image_array])
        .map_err(|e| JammiError::Inference(format!("Failed to create expanded batch: {e}")))?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props))
        .map_err(|e| JammiError::Inference(format!("Failed to create Parquet writer: {e}")))?;
    writer
        .write(&batch)
        .map_err(|e| JammiError::Inference(format!("Failed to write expanded batch: {e}")))?;
    writer
        .close()
        .map_err(|e| JammiError::Inference(format!("Failed to close Parquet writer: {e}")))?;

    Ok(())
}

/// Extract string keys from a RecordBatch column (handles Utf8, Utf8View).
fn extract_keys(batch: &arrow::record_batch::RecordBatch, column: &str) -> Result<Vec<String>> {
    let col = batch
        .column_by_name(column)
        .ok_or_else(|| JammiError::Inference(format!("Column '{column}' not found")))?;

    match col.data_type() {
        DataType::Utf8 => {
            let arr = col.as_any().downcast_ref::<StringArray>().unwrap();
            Ok((0..arr.len()).map(|i| arr.value(i).to_string()).collect())
        }
        DataType::Utf8View => {
            let arr = col.as_any().downcast_ref::<StringViewArray>().unwrap();
            Ok((0..arr.len()).map(|i| arr.value(i).to_string()).collect())
        }
        _ => {
            let casted = arrow::compute::cast(col, &DataType::Utf8)
                .map_err(|e| JammiError::Inference(format!("Failed to cast key column: {e}")))?;
            let arr = casted.as_any().downcast_ref::<StringArray>().unwrap();
            Ok((0..arr.len()).map(|i| arr.value(i).to_string()).collect())
        }
    }
}

/// Extract raw image bytes from a RecordBatch column (handles Binary, BinaryView, LargeBinary).
fn extract_image_bytes(
    batch: &arrow::record_batch::RecordBatch,
    column: &str,
) -> Result<Vec<Vec<u8>>> {
    let col = batch
        .column_by_name(column)
        .ok_or_else(|| JammiError::Inference(format!("Column '{column}' not found")))?;

    match col.data_type() {
        DataType::Binary => {
            let arr = col.as_any().downcast_ref::<BinaryArray>().unwrap();
            Ok((0..arr.len()).map(|i| arr.value(i).to_vec()).collect())
        }
        DataType::BinaryView => {
            let arr = col.as_any().downcast_ref::<BinaryViewArray>().unwrap();
            Ok((0..arr.len()).map(|i| arr.value(i).to_vec()).collect())
        }
        DataType::LargeBinary => {
            let arr = col.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
            Ok((0..arr.len()).map(|i| arr.value(i).to_vec()).collect())
        }
        dt => Err(JammiError::Inference(format!(
            "Column '{column}' has type {dt}, expected Binary"
        ))),
    }
}
