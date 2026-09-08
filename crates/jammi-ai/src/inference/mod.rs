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
///
/// Every column is COLUMN-level validated by `validate_text_column` before
/// any row is read — a column whose physical type this text task cannot
/// honestly read is a whole-call refusal naming the column's data type, never
/// a per-row `""` reading (see that function's doc for why: this mirrors
/// `fine_tune::worker::extract_string_column`'s policy exactly, closing the
/// SAME class of confident-wrong-number the trainer already refuses/casts
/// against).
pub fn arrow_to_texts(columns: &[ArrayRef]) -> Result<Vec<String>> {
    if columns.is_empty() {
        return Err(JammiError::Inference("No content columns provided".into()));
    }
    let validated: Vec<ArrayRef> = columns
        .iter()
        .map(validate_text_column)
        .collect::<Result<_>>()?;
    let row_count = validated[0].len();
    let mut texts = Vec::with_capacity(row_count);

    for i in 0..row_count {
        let parts: Vec<&str> = validated
            .iter()
            .filter_map(|col| get_string_value(col, i))
            .collect();
        texts.push(parts.join(" "));
    }
    Ok(texts)
}

/// Validate a content column's PHYSICAL Arrow type is honestly readable as
/// text for the caller's embedding/inference call — never per-row, never
/// silent — mirroring `fine_tune::worker::extract_string_column`'s policy
/// exactly (the trainer refuses/casts the identical input; this is the same
/// class reached a second home at serve time, per the training-path fix's
/// own doc: "one root cause can have a second home").
///
/// The string families (`Utf8`/`LargeUtf8`/`Utf8View`) pass through
/// unchanged. The binary families (`Binary`/`LargeBinary`/`BinaryView`/
/// `FixedSizeBinary`) are refused OUTRIGHT: `arrow::compute::cast`'s DEFAULT
/// `safe: true` option turns a value the target type cannot represent into
/// NULL rather than an error, and [`get_string_value`] reads a null slot as
/// absent (contributing nothing to the joined text) — so, uncaught, an
/// image/audio-bytes column submitted under a text task would cast
/// cell-by-cell into NULLs and every row would silently embed as the empty
/// string, with `row_status` reading all-ok: a confident wrong answer, not a
/// refusal. Every OTHER type is cast to `Utf8`; a cast that introduces a
/// null the source did not have is refused for the identical reason — the
/// empty string it would otherwise read is fabricated, not a reading of the
/// caller's data. A row that was ALREADY null keeps the documented `""`
/// reading (via `get_string_value`'s null check) — a pre-existing
/// null-handling contract this function does not disturb.
fn validate_text_column(col: &ArrayRef) -> Result<ArrayRef> {
    match col.data_type() {
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => Ok(std::sync::Arc::clone(col)),
        DataType::Binary
        | DataType::LargeBinary
        | DataType::BinaryView
        | DataType::FixedSizeBinary(_) => Err(JammiError::Inference(format!(
            "text embedding input column has type {dt}, which holds raw binary bytes, \
                 not text — refusing to silently embed the bytes as an empty string for \
                 every row",
            dt = col.data_type()
        ))),
        other => {
            let casted = arrow::compute::cast(col.as_ref(), &DataType::Utf8).map_err(|e| {
                JammiError::Inference(format!(
                    "text embedding input column has type {other}, which cannot be cast to \
                     text: {e}"
                ))
            })?;
            if (0..col.len()).any(|i| casted.is_null(i) && !col.is_null(i)) {
                return Err(JammiError::Inference(format!(
                    "text embedding input column has type {other}; casting it to text \
                     introduced a null value the source column did not have — refusing to \
                     silently drop data"
                )));
            }
            Ok(casted)
        }
    }
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
/// - `Utf8` / `LargeUtf8` / `Utf8View`: values are file paths, loaded from disk.
/// - `Binary` / `LargeBinary` / `BinaryView`: values are image bytes, decoded in memory.
///
/// Null values produce `None` (caller tracks via `row_status`); a row whose
/// bytes fail to DECODE produces `Some(Err(..))` rather than failing the
/// whole batch — the caller (`CandleBackend::forward_image_embedding`) marks
/// that one row's `_status`/`_error` and every other row's embedding still
/// computes, per `docs/guide/src/generate-image-embeddings.md`'s
/// "Error handling" table. A column-level problem (an unsupported Arrow
/// type, an unreadable path) is still a hard `Err` on the whole call — those
/// are not per-row concerns.
///
/// Two stages: resolving each row to its raw bytes runs SEQUENTIALLY here (a
/// path-valued row's `std::fs::read` happens outside the parallel stage);
/// decoding those bytes to a [`DynamicImage`] runs in parallel across the
/// batch on rayon's global pool, through
/// [`image_preprocess::decode_image_batch_per_row_indexed`] — the SAME
/// per-item decode body the training path (`fine_tune::trainer`'s
/// `image_encoder_input`) runs via `decode_image_batch` (which still hard-fails
/// on the lowest-index error — a training corpus with a corrupt item is a
/// refusal, not a per-row skip), called here with the ORIGINAL Arrow row of
/// each non-null item (this function compacts nulls out before decoding, so
/// the position in the compacted slice and the row a caller means by "row N"
/// diverge after the first null — the per-row call keeps every outcome
/// Arrow-row-numbered regardless of how many nulls precede it).
///
/// A path-valued row is decoded from its raw bytes (content sniffing, like
/// every other row), not re-opened by path — so a file whose bytes need an
/// extension hint to identify (rather than a magic-number match) fails here
/// where `image::open` would have succeeded; a decode failure on a
/// path-valued row still names its source path in the error, so the
/// distinction from a bytes-valued row's failure is not lost.
pub fn arrow_to_images(columns: &[ArrayRef]) -> Result<Vec<Option<Result<DynamicImage>>>> {
    if columns.is_empty() {
        return Err(JammiError::Inference("No image columns provided".into()));
    }
    // Use the first column only (image embedding expects a single column).
    let col = &columns[0];
    let row_count = col.len();

    // Stage 1 (sequential): resolve every non-null row to its raw bytes and
    // its Arrow row id. Path-valued rows read the file from disk here;
    // bytes-valued rows borrow straight out of the Arrow buffer (no copy).
    // `is_null` is the ONLY thing carried past decode for null bookkeeping
    // (advisory: a resident-bytes null mask, not the bytes themselves) — the
    // resolved bytes live only through Stage 2's decode call, then drop.
    let mut is_null: Vec<bool> = Vec::with_capacity(row_count);
    let mut row_ids: Vec<usize> = Vec::new();
    let mut byte_rows: Vec<Cow<[u8]>> = Vec::new();
    let mut source_paths: Vec<Option<String>> = Vec::new();
    for i in 0..row_count {
        if col.is_null(i) {
            is_null.push(true);
            continue;
        }
        is_null.push(false);
        let (bytes, path): (Cow<[u8]>, Option<String>) = match col.data_type() {
            DataType::Utf8 => {
                let path = col
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .map(|a| a.value(i))
                    .ok_or_else(|| {
                        JammiError::Inference(format!("Failed to read path at row {i}"))
                    })?;
                let bytes = std::fs::read(path).map_err(|e| {
                    JammiError::Inference(format!("Failed to read image file '{path}': {e}"))
                })?;
                (Cow::Owned(bytes), Some(path.to_string()))
            }
            DataType::LargeUtf8 => {
                let path = col
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .map(|a| a.value(i))
                    .ok_or_else(|| {
                        JammiError::Inference(format!("Failed to read path at row {i}"))
                    })?;
                let bytes = std::fs::read(path).map_err(|e| {
                    JammiError::Inference(format!("Failed to read image file '{path}': {e}"))
                })?;
                (Cow::Owned(bytes), Some(path.to_string()))
            }
            // `Utf8View` is the physical layout DataFusion's Parquet scan gives a plain
            // `Utf8` path column, so it takes the `Utf8` arm's contract exactly: the value
            // is a file path, an unreadable path fails the whole call, nulls are per-row.
            DataType::Utf8View => {
                let path = col
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .map(|a| a.value(i))
                    .ok_or_else(|| {
                        JammiError::Inference(format!("Failed to read path at row {i}"))
                    })?;
                let bytes = std::fs::read(path).map_err(|e| {
                    JammiError::Inference(format!("Failed to read image file '{path}': {e}"))
                })?;
                (Cow::Owned(bytes), Some(path.to_string()))
            }
            DataType::Binary => (
                Cow::Borrowed(
                    col.as_any()
                        .downcast_ref::<BinaryArray>()
                        .map(|a| a.value(i))
                        .ok_or_else(|| {
                            JammiError::Inference(format!("Failed to read bytes at row {i}"))
                        })?,
                ),
                None,
            ),
            DataType::LargeBinary => (
                Cow::Borrowed(
                    col.as_any()
                        .downcast_ref::<LargeBinaryArray>()
                        .map(|a| a.value(i))
                        .ok_or_else(|| {
                            JammiError::Inference(format!("Failed to read bytes at row {i}"))
                        })?,
                ),
                None,
            ),
            DataType::BinaryView => (
                Cow::Borrowed(
                    col.as_any()
                        .downcast_ref::<BinaryViewArray>()
                        .map(|a| a.value(i))
                        .ok_or_else(|| {
                            JammiError::Inference(format!("Failed to read bytes at row {i}"))
                        })?,
                ),
                None,
            ),
            dt => {
                return Err(JammiError::Inference(format!(
                    "Unsupported column type for image input: {dt}. \
                     Expected Utf8 (file paths) or Binary (image bytes)"
                )));
            }
        };
        row_ids.push(i);
        byte_rows.push(bytes);
        source_paths.push(path);
    }

    // Stage 2 (parallel): decode every non-null row's bytes on rayon's global
    // pool, Arrow-row-numbered via `row_ids`, keeping EVERY row's own outcome
    // (not collapsed to the lowest-index failure) — a path-valued row's
    // failure is re-attached to its source path. `byte_rows` (the resident
    // encoded bytes) is dropped as soon as decode returns, before the
    // re-thread loop below runs — only `is_null` and the decoded outcomes
    // survive it.
    let decoded: Vec<Result<DynamicImage>> =
        image_preprocess::decode_image_batch_per_row_indexed(&row_ids, &byte_rows)?
            .into_iter()
            .zip(row_ids.iter())
            .zip(source_paths.iter())
            .map(|((outcome, &row), path)| {
                outcome.map_err(|e| attach_source_path(e, "image", row, path.as_deref()))
            })
            .collect();
    drop(byte_rows);
    drop(source_paths);

    let mut decoded_iter = decoded.into_iter();
    let mut images = Vec::with_capacity(row_count);
    for &null in &is_null {
        images.push(if null {
            None
        } else {
            Some(decoded_iter.next().ok_or_else(|| {
                JammiError::Inference(
                    "internal error: decoded image count did not match non-null row count".into(),
                )
            })?)
        });
    }

    Ok(images)
}

/// A row's decode failure re-attaches its source path (for a path-valued
/// row) into the error text, in the ONE documented per-row decode-failure
/// shape (`docs/guide/src/generate-image-embeddings.md`'s "Error handling"
/// table: `"Failed to decode {kind} at row N: ..."`), shared by both media
/// types (`kind` is `"image"` or `"audio"`) and both the path-valued and
/// bytes-valued arms — a path, when present, is appended AFTER the
/// documented prefix and cause (`"... (path '...')"`) rather than spliced
/// into the middle of it, so a caller matching on the documented prefix
/// never has to skip a variable-length path segment first. This is NOT the
/// pre-unit path arm's shape (that was `"Failed to load image '{path}':
/// {e}"`, a wholly different message with no row number) — it is the shape
/// the per-row serving contract documents today, applied uniformly.
///
/// [`decode_image_batch_per_row_indexed`](image_preprocess::decode_image_batch_per_row_indexed)
/// and its audio peer only ever see raw bytes, never a path, so they cannot
/// produce this shape themselves; this function splices the path in
/// afterward by matching the delimiter-safe `"row {row}: "` marker against
/// their own, fully-controlled error format (never user data) and keeping
/// only the text AFTER that marker; a bytes-valued row (no path) or a match
/// miss returns the error unchanged.
fn attach_source_path(err: JammiError, kind: &str, row: usize, path: Option<&str>) -> JammiError {
    let Some(path) = path else {
        return err;
    };
    let msg = err.to_string();
    let marker = format!("row {row}: ");
    let cause = match msg.find(&marker) {
        Some(marker_at) => &msg[marker_at + marker.len()..],
        None => return err,
    };
    JammiError::Inference(format!(
        "Failed to decode {kind} at row {row}: {cause} (path '{path}')"
    ))
}

/// Extract and decode audio clips from an Arrow column.
///
/// Supports two input modes, mirroring [`arrow_to_images`]:
/// - `Utf8` / `LargeUtf8` / `Utf8View`: values are file paths, read and decoded from disk.
/// - `Binary` / `LargeBinary` / `BinaryView`: values are encoded audio bytes
///   (WAV/FLAC/MP3/Ogg), decoded in memory.
///
/// Two stages, mirroring [`arrow_to_images`]: resolving each row to its raw
/// bytes runs SEQUENTIALLY (a path-valued row's `std::fs::read` happens
/// outside the parallel stage); decoding those bytes to mono PCM runs in
/// parallel across the batch on rayon's global pool, through
/// [`audio_preprocess::decode_audio_batch_per_row_indexed`] — the SAME
/// per-item decode body the training path (`fine_tune::trainer`'s
/// `audio_encoder_input`) runs via `decode_audio_batch` (which still
/// hard-fails on the lowest-index error — a training corpus with a corrupt
/// item is a refusal, not a per-row skip), called here with the ORIGINAL
/// Arrow row of each non-null item (see [`arrow_to_images`]'s doc for why:
/// this function compacts nulls out before decoding, so the position in the
/// compacted slice and the row a caller means by "row N" diverge after the
/// first null). Null values produce `None`; a row whose bytes fail to decode
/// produces `Some(Err(..))` rather than failing the whole batch — mirroring
/// [`arrow_to_images`]'s per-row contract (caller tracks both via
/// `row_status`).
pub fn arrow_to_audio(
    columns: &[ArrayRef],
) -> Result<Vec<Option<Result<audio_preprocess::DecodedAudio>>>> {
    if columns.is_empty() {
        return Err(JammiError::Inference("No audio columns provided".into()));
    }
    // Use the first column only (audio embedding expects a single column).
    let col = &columns[0];
    let row_count = col.len();

    // Stage 1 (sequential): resolve every non-null row to its raw bytes and
    // its Arrow row id. Path-valued rows read the file from disk here;
    // bytes-valued rows borrow straight out of the Arrow buffer (no copy).
    // `is_null` is the ONLY thing carried past decode for null bookkeeping —
    // the resolved bytes live only through Stage 2's decode call, then drop.
    let mut is_null: Vec<bool> = Vec::with_capacity(row_count);
    let mut row_ids: Vec<usize> = Vec::new();
    let mut byte_rows: Vec<Cow<[u8]>> = Vec::new();
    let mut source_paths: Vec<Option<String>> = Vec::new();
    for i in 0..row_count {
        if col.is_null(i) {
            is_null.push(true);
            continue;
        }
        is_null.push(false);
        let (bytes, path): (Cow<[u8]>, Option<String>) = match col.data_type() {
            DataType::Utf8 => {
                let path = col
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .map(|a| a.value(i))
                    .ok_or_else(|| {
                        JammiError::Inference(format!("Failed to read path at row {i}"))
                    })?;
                let bytes = std::fs::read(path).map_err(|e| {
                    JammiError::Inference(format!("Failed to read audio file '{path}': {e}"))
                })?;
                (Cow::Owned(bytes), Some(path.to_string()))
            }
            DataType::LargeUtf8 => {
                let path = col
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .map(|a| a.value(i))
                    .ok_or_else(|| {
                        JammiError::Inference(format!("Failed to read path at row {i}"))
                    })?;
                let bytes = std::fs::read(path).map_err(|e| {
                    JammiError::Inference(format!("Failed to read audio file '{path}': {e}"))
                })?;
                (Cow::Owned(bytes), Some(path.to_string()))
            }
            // `Utf8View` is the physical layout DataFusion's Parquet scan gives a plain
            // `Utf8` path column, so it takes the `Utf8` arm's contract exactly: the value
            // is a file path, an unreadable path fails the whole call, nulls are per-row.
            DataType::Utf8View => {
                let path = col
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .map(|a| a.value(i))
                    .ok_or_else(|| {
                        JammiError::Inference(format!("Failed to read path at row {i}"))
                    })?;
                let bytes = std::fs::read(path).map_err(|e| {
                    JammiError::Inference(format!("Failed to read audio file '{path}': {e}"))
                })?;
                (Cow::Owned(bytes), Some(path.to_string()))
            }
            DataType::Binary => (
                Cow::Borrowed(
                    col.as_any()
                        .downcast_ref::<BinaryArray>()
                        .map(|a| a.value(i))
                        .ok_or_else(|| {
                            JammiError::Inference(format!("Failed to read bytes at row {i}"))
                        })?,
                ),
                None,
            ),
            DataType::LargeBinary => (
                Cow::Borrowed(
                    col.as_any()
                        .downcast_ref::<LargeBinaryArray>()
                        .map(|a| a.value(i))
                        .ok_or_else(|| {
                            JammiError::Inference(format!("Failed to read bytes at row {i}"))
                        })?,
                ),
                None,
            ),
            DataType::BinaryView => (
                Cow::Borrowed(
                    col.as_any()
                        .downcast_ref::<BinaryViewArray>()
                        .map(|a| a.value(i))
                        .ok_or_else(|| {
                            JammiError::Inference(format!("Failed to read bytes at row {i}"))
                        })?,
                ),
                None,
            ),
            dt => {
                return Err(JammiError::Inference(format!(
                    "Unsupported column type for audio input: {dt}. \
                     Expected Utf8 (file paths) or Binary (audio bytes)"
                )));
            }
        };
        row_ids.push(i);
        byte_rows.push(bytes);
        source_paths.push(path);
    }

    // Stage 2 (parallel): decode every non-null row's bytes on rayon's global
    // pool, Arrow-row-numbered via `row_ids`, keeping EVERY row's own outcome
    // (not collapsed to the lowest-index failure) — a path-valued row's
    // failure is re-attached to its source path, mirroring
    // [`arrow_to_images`]. `byte_rows` (the resident encoded bytes) is
    // dropped as soon as decode returns, before the re-thread loop below
    // runs — only `is_null` and the decoded outcomes survive it.
    let decoded: Vec<Result<audio_preprocess::DecodedAudio>> =
        audio_preprocess::decode_audio_batch_per_row_indexed(&row_ids, &byte_rows)?
            .into_iter()
            .zip(row_ids.iter())
            .zip(source_paths.iter())
            .map(|((outcome, &row), path)| {
                outcome.map_err(|e| attach_source_path(e, "audio", row, path.as_deref()))
            })
            .collect();
    drop(byte_rows);
    drop(source_paths);

    let mut decoded_iter = decoded.into_iter();
    let mut clips = Vec::with_capacity(row_count);
    for &null in &is_null {
        clips.push(if null {
            None
        } else {
            Some(decoded_iter.next().ok_or_else(|| {
                JammiError::Inference(
                    "internal error: decoded audio count did not match non-null row count".into(),
                )
            })?)
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
