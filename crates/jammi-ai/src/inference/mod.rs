pub mod adapter;
pub mod audio_preprocess;
pub mod image_preprocess;
pub mod observer;
pub mod runner;
pub mod schema;

use arrow::array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, LargeBinaryArray, LargeStringArray, StringArray,
    StringViewArray,
};
use arrow::datatypes::DataType;
use image::DynamicImage;
use jammi_db::error::{JammiError, Result};

/// Extract text from Arrow string columns (handles Utf8, LargeUtf8, and Utf8View).
/// If multiple columns, concatenate with " " separator.
/// Null values produce empty strings (caller handles null tracking).
pub fn arrow_to_texts(columns: &[ArrayRef]) -> Result<Vec<String>> {
    if columns.is_empty() {
        return Err(JammiError::Inference("No content columns provided".into()));
    }
    let row_count = columns[0].len();
    let mut texts = Vec::with_capacity(row_count);

    for i in 0..row_count {
        let parts: Vec<&str> = columns
            .iter()
            .filter_map(|col| get_string_value(col, i))
            .collect();
        texts.push(parts.join(" "));
    }
    Ok(texts)
}

/// Extract a string value from any Arrow string-like array type at index `i`.
fn get_string_value(col: &ArrayRef, i: usize) -> Option<&str> {
    if col.is_null(i) {
        return None;
    }
    match col.data_type() {
        DataType::Utf8 => col
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|a| a.value(i)),
        DataType::LargeUtf8 => col
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .map(|a| a.value(i)),
        DataType::Utf8View => col
            .as_any()
            .downcast_ref::<StringViewArray>()
            .map(|a| a.value(i)),
        _ => None,
    }
}

/// Extract named columns from a RecordBatch as ArrayRefs.
pub fn extract_columns(
    batch: &arrow::record_batch::RecordBatch,
    column_names: &[String],
) -> Result<Vec<ArrayRef>> {
    column_names
        .iter()
        .map(|name| {
            batch
                .column_by_name(name)
                .map(std::sync::Arc::clone)
                .ok_or_else(|| {
                    JammiError::Inference(format!("Column '{name}' not found in input batch"))
                })
        })
        .collect()
}

/// Extract a single named column from a RecordBatch.
pub fn extract_column(
    batch: &arrow::record_batch::RecordBatch,
    column_name: &str,
) -> Result<ArrayRef> {
    batch
        .column_by_name(column_name)
        .map(std::sync::Arc::clone)
        .ok_or_else(|| {
            JammiError::Inference(format!("Column '{column_name}' not found in input batch"))
        })
}

/// Extract images from an Arrow column.
///
/// Supports two input modes:
/// - `Utf8` / `LargeUtf8`: values are file paths, loaded from disk.
/// - `Binary` / `LargeBinary`: values are image bytes, decoded in memory.
///
/// Null values produce `None` (caller tracks via `row_status`).
pub fn arrow_to_images(columns: &[ArrayRef]) -> Result<Vec<Option<DynamicImage>>> {
    if columns.is_empty() {
        return Err(JammiError::Inference("No image columns provided".into()));
    }
    // Use the first column only (image embedding expects a single column).
    let col = &columns[0];
    let row_count = col.len();
    let mut images = Vec::with_capacity(row_count);

    for i in 0..row_count {
        if col.is_null(i) {
            images.push(None);
            continue;
        }
        let img = match col.data_type() {
            DataType::Utf8 => {
                let path = col
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .map(|a| a.value(i))
                    .ok_or_else(|| {
                        JammiError::Inference(format!("Failed to read path at row {i}"))
                    })?;
                image::open(path).map_err(|e| {
                    JammiError::Inference(format!("Failed to load image '{path}': {e}"))
                })?
            }
            DataType::LargeUtf8 => {
                let path = col
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .map(|a| a.value(i))
                    .ok_or_else(|| {
                        JammiError::Inference(format!("Failed to read path at row {i}"))
                    })?;
                image::open(path).map_err(|e| {
                    JammiError::Inference(format!("Failed to load image '{path}': {e}"))
                })?
            }
            DataType::Binary => {
                let bytes = col
                    .as_any()
                    .downcast_ref::<BinaryArray>()
                    .map(|a| a.value(i))
                    .ok_or_else(|| {
                        JammiError::Inference(format!("Failed to read bytes at row {i}"))
                    })?;
                image::load_from_memory(bytes).map_err(|e| {
                    JammiError::Inference(format!("Failed to decode image at row {i}: {e}"))
                })?
            }
            DataType::LargeBinary => {
                let bytes = col
                    .as_any()
                    .downcast_ref::<LargeBinaryArray>()
                    .map(|a| a.value(i))
                    .ok_or_else(|| {
                        JammiError::Inference(format!("Failed to read bytes at row {i}"))
                    })?;
                image::load_from_memory(bytes).map_err(|e| {
                    JammiError::Inference(format!("Failed to decode image at row {i}: {e}"))
                })?
            }
            DataType::BinaryView => {
                let bytes = col
                    .as_any()
                    .downcast_ref::<BinaryViewArray>()
                    .map(|a| a.value(i))
                    .ok_or_else(|| {
                        JammiError::Inference(format!("Failed to read bytes at row {i}"))
                    })?;
                image::load_from_memory(bytes).map_err(|e| {
                    JammiError::Inference(format!("Failed to decode image at row {i}: {e}"))
                })?
            }
            dt => {
                return Err(JammiError::Inference(format!(
                    "Unsupported column type for image input: {dt}. \
                     Expected Utf8 (file paths) or Binary (image bytes)"
                )));
            }
        };
        images.push(Some(img));
    }

    Ok(images)
}

/// Extract and decode audio clips from an Arrow column.
///
/// Supports two input modes, mirroring [`arrow_to_images`]:
/// - `Utf8` / `LargeUtf8`: values are file paths, read and decoded from disk.
/// - `Binary` / `LargeBinary` / `BinaryView`: values are encoded audio bytes
///   (WAV/FLAC/MP3/Ogg), decoded in memory.
///
/// Each value is decoded to mono PCM via
/// [`audio_preprocess::decode_audio_bytes`]. Null values produce `None`
/// (caller tracks via `row_status`).
pub fn arrow_to_audio(columns: &[ArrayRef]) -> Result<Vec<Option<audio_preprocess::DecodedAudio>>> {
    if columns.is_empty() {
        return Err(JammiError::Inference("No audio columns provided".into()));
    }
    // Use the first column only (audio embedding expects a single column).
    let col = &columns[0];
    let row_count = col.len();
    let mut clips = Vec::with_capacity(row_count);

    for i in 0..row_count {
        if col.is_null(i) {
            clips.push(None);
            continue;
        }
        let bytes: std::borrow::Cow<[u8]> = match col.data_type() {
            DataType::Utf8 => {
                let path = col
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .map(|a| a.value(i))
                    .ok_or_else(|| {
                        JammiError::Inference(format!("Failed to read path at row {i}"))
                    })?;
                std::borrow::Cow::Owned(std::fs::read(path).map_err(|e| {
                    JammiError::Inference(format!("Failed to read audio file '{path}': {e}"))
                })?)
            }
            DataType::LargeUtf8 => {
                let path = col
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .map(|a| a.value(i))
                    .ok_or_else(|| {
                        JammiError::Inference(format!("Failed to read path at row {i}"))
                    })?;
                std::borrow::Cow::Owned(std::fs::read(path).map_err(|e| {
                    JammiError::Inference(format!("Failed to read audio file '{path}': {e}"))
                })?)
            }
            DataType::Binary => std::borrow::Cow::Borrowed(
                col.as_any()
                    .downcast_ref::<BinaryArray>()
                    .map(|a| a.value(i))
                    .ok_or_else(|| {
                        JammiError::Inference(format!("Failed to read bytes at row {i}"))
                    })?,
            ),
            DataType::LargeBinary => std::borrow::Cow::Borrowed(
                col.as_any()
                    .downcast_ref::<LargeBinaryArray>()
                    .map(|a| a.value(i))
                    .ok_or_else(|| {
                        JammiError::Inference(format!("Failed to read bytes at row {i}"))
                    })?,
            ),
            DataType::BinaryView => std::borrow::Cow::Borrowed(
                col.as_any()
                    .downcast_ref::<BinaryViewArray>()
                    .map(|a| a.value(i))
                    .ok_or_else(|| {
                        JammiError::Inference(format!("Failed to read bytes at row {i}"))
                    })?,
            ),
            dt => {
                return Err(JammiError::Inference(format!(
                    "Unsupported column type for audio input: {dt}. \
                     Expected Utf8 (file paths) or Binary (audio bytes)"
                )));
            }
        };
        let decoded = audio_preprocess::decode_audio_bytes(&bytes).map_err(|e| {
            JammiError::Inference(format!("Failed to decode audio at row {i}: {e}"))
        })?;
        clips.push(Some(decoded));
    }

    Ok(clips)
}

/// Slice a set of columns to a sub-range.
pub fn slice_columns(columns: &[ArrayRef], offset: usize, length: usize) -> Vec<ArrayRef> {
    columns
        .iter()
        .map(|col| col.slice(offset, length))
        .collect()
}
