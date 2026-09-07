pub mod adapter;
pub mod audio_preprocess;
pub mod image_preprocess;
pub mod observer;
pub mod runner;
pub mod schema;

use std::borrow::Cow;

use arrow::array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, LargeBinaryArray, LargeStringArray, StringArray,
    StringViewArray,
};
use arrow::datatypes::DataType;
use image::DynamicImage;
use jammi_db::error::{JammiError, Result};

/// Fold a batch of per-row `Result`s into either the ordered `Ok` values or
/// the FIRST (lowest-index) `Err`.
///
/// The parallel decode/preprocess stages (`audio_preprocess::decode_audio_batch`
/// / `image_preprocess::decode_image_batch` and the `par_chunks_mut` writers in
/// [`audio_preprocess::preprocess_clap_fusion`] /
/// [`image_preprocess::preprocess_image_batch`]) build their per-row results
/// with a rayon `collect()` on an INDEXED parallel iterator, which preserves
/// the original row order regardless of which thread finished which row
/// first. So a single forward scan for the first `Err` already IS the
/// lowest-index selection — no separate index bookkeeping is needed here.
pub(crate) fn lowest_index_result<T>(results: Vec<Result<T>>) -> Result<Vec<T>> {
    let mut out = Vec::with_capacity(results.len());
    for r in results {
        out.push(r?);
    }
    Ok(out)
}

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
///
/// Two stages: resolving each row to its raw bytes runs SEQUENTIALLY here (a
/// path-valued row's `std::fs::read` happens outside the parallel stage);
/// decoding those bytes to a [`DynamicImage`] runs in parallel across the
/// batch on rayon's global pool, through the SAME
/// [`image_preprocess::decode_image_batch`] helper the training path
/// (`fine_tune::trainer`'s `image_encoder_input`) calls — one decode loop
/// shared by both.
pub fn arrow_to_images(columns: &[ArrayRef]) -> Result<Vec<Option<DynamicImage>>> {
    if columns.is_empty() {
        return Err(JammiError::Inference("No image columns provided".into()));
    }
    // Use the first column only (image embedding expects a single column).
    let col = &columns[0];
    let row_count = col.len();

    // Stage 1 (sequential): resolve every row to its raw bytes. Path-valued
    // rows read the file from disk here; bytes-valued rows borrow straight
    // out of the Arrow buffer (no copy). Null rows carry no bytes.
    let mut byte_rows: Vec<Option<Cow<[u8]>>> = Vec::with_capacity(row_count);
    for i in 0..row_count {
        if col.is_null(i) {
            byte_rows.push(None);
            continue;
        }
        let bytes: Cow<[u8]> = match col.data_type() {
            DataType::Utf8 => {
                let path = col
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .map(|a| a.value(i))
                    .ok_or_else(|| {
                        JammiError::Inference(format!("Failed to read path at row {i}"))
                    })?;
                Cow::Owned(std::fs::read(path).map_err(|e| {
                    JammiError::Inference(format!("Failed to read image file '{path}': {e}"))
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
                Cow::Owned(std::fs::read(path).map_err(|e| {
                    JammiError::Inference(format!("Failed to read image file '{path}': {e}"))
                })?)
            }
            DataType::Binary => Cow::Borrowed(
                col.as_any()
                    .downcast_ref::<BinaryArray>()
                    .map(|a| a.value(i))
                    .ok_or_else(|| {
                        JammiError::Inference(format!("Failed to read bytes at row {i}"))
                    })?,
            ),
            DataType::LargeBinary => Cow::Borrowed(
                col.as_any()
                    .downcast_ref::<LargeBinaryArray>()
                    .map(|a| a.value(i))
                    .ok_or_else(|| {
                        JammiError::Inference(format!("Failed to read bytes at row {i}"))
                    })?,
            ),
            DataType::BinaryView => Cow::Borrowed(
                col.as_any()
                    .downcast_ref::<BinaryViewArray>()
                    .map(|a| a.value(i))
                    .ok_or_else(|| {
                        JammiError::Inference(format!("Failed to read bytes at row {i}"))
                    })?,
            ),
            dt => {
                return Err(JammiError::Inference(format!(
                    "Unsupported column type for image input: {dt}. \
                     Expected Utf8 (file paths) or Binary (image bytes)"
                )));
            }
        };
        byte_rows.push(Some(bytes));
    }

    // Stage 2 (parallel): decode every non-null row's bytes on rayon's global
    // pool, then re-thread the per-row null bookkeeping.
    let non_null: Vec<&Cow<[u8]>> = byte_rows.iter().flatten().collect();
    let decoded = image_preprocess::decode_image_batch(&non_null)?;
    let mut decoded_iter = decoded.into_iter();
    let mut images = Vec::with_capacity(row_count);
    for row in &byte_rows {
        images.push(match row {
            Some(_) => Some(decoded_iter.next().ok_or_else(|| {
                JammiError::Inference(
                    "internal error: decoded image count did not match non-null row count".into(),
                )
            })?),
            None => None,
        });
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
/// Two stages, mirroring [`arrow_to_images`]: resolving each row to its raw
/// bytes runs SEQUENTIALLY (a path-valued row's `std::fs::read` happens
/// outside the parallel stage); decoding those bytes to mono PCM runs in
/// parallel across the batch on rayon's global pool, through the SAME
/// [`audio_preprocess::decode_audio_batch`] helper the training path
/// (`fine_tune::trainer`'s `audio_encoder_input`) calls. Null values produce
/// `None` (caller tracks via `row_status`).
pub fn arrow_to_audio(columns: &[ArrayRef]) -> Result<Vec<Option<audio_preprocess::DecodedAudio>>> {
    if columns.is_empty() {
        return Err(JammiError::Inference("No audio columns provided".into()));
    }
    // Use the first column only (audio embedding expects a single column).
    let col = &columns[0];
    let row_count = col.len();

    // Stage 1 (sequential): resolve every row to its raw bytes. Path-valued
    // rows read the file from disk here; bytes-valued rows borrow straight
    // out of the Arrow buffer (no copy). Null rows carry no bytes.
    let mut byte_rows: Vec<Option<Cow<[u8]>>> = Vec::with_capacity(row_count);
    for i in 0..row_count {
        if col.is_null(i) {
            byte_rows.push(None);
            continue;
        }
        let bytes: Cow<[u8]> = match col.data_type() {
            DataType::Utf8 => {
                let path = col
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .map(|a| a.value(i))
                    .ok_or_else(|| {
                        JammiError::Inference(format!("Failed to read path at row {i}"))
                    })?;
                Cow::Owned(std::fs::read(path).map_err(|e| {
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
                Cow::Owned(std::fs::read(path).map_err(|e| {
                    JammiError::Inference(format!("Failed to read audio file '{path}': {e}"))
                })?)
            }
            DataType::Binary => Cow::Borrowed(
                col.as_any()
                    .downcast_ref::<BinaryArray>()
                    .map(|a| a.value(i))
                    .ok_or_else(|| {
                        JammiError::Inference(format!("Failed to read bytes at row {i}"))
                    })?,
            ),
            DataType::LargeBinary => Cow::Borrowed(
                col.as_any()
                    .downcast_ref::<LargeBinaryArray>()
                    .map(|a| a.value(i))
                    .ok_or_else(|| {
                        JammiError::Inference(format!("Failed to read bytes at row {i}"))
                    })?,
            ),
            DataType::BinaryView => Cow::Borrowed(
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
        byte_rows.push(Some(bytes));
    }

    // Stage 2 (parallel): decode every non-null row's bytes on rayon's global
    // pool, then re-thread the per-row null bookkeeping.
    let non_null: Vec<&Cow<[u8]>> = byte_rows.iter().flatten().collect();
    let decoded = audio_preprocess::decode_audio_batch(&non_null)?;
    let mut decoded_iter = decoded.into_iter();
    let mut clips = Vec::with_capacity(row_count);
    for row in &byte_rows {
        clips.push(match row {
            Some(_) => Some(decoded_iter.next().ok_or_else(|| {
                JammiError::Inference(
                    "internal error: decoded audio count did not match non-null row count".into(),
                )
            })?),
            None => None,
        });
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
