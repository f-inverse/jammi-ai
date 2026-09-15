//! The ONE Arrow → training-row decoder (#500 U2c, M3): every column-level
//! extractor and format classifier that turns a training-set's `RecordBatch`
//! columns into the text/media/target values a [`TrainingDataLoader`]
//! or a [`super::stream::TrainingSetStream`] chunk carries.
//!
//! Factored OUT of `worker.rs` (where it lived as
//! `extract_string_column`/`extract_numeric_column`/`build_training_data_loader`
//! before this unit) so there is exactly ONE decoder both the EAGER loader
//! (`build_training_data_loader`, called once over the whole read-back) and
//! the STREAM (`append_selected_rows`, called per step over the rows the
//! current chunk needs) share — a chunk either produces from an identical
//! Arrow batch and row set can never disagree, because both paths call the
//! same `extract_string_column`/`extract_binary_column`/`extract_numeric_column`
//! primitives.
//!
//! # Two entry points, one set of column extractors
//!
//! - `build_training_data_loader`: the eager entry point. Reads EVERY row of
//!   EVERY batch into a fully in-memory [`TrainingDataLoader`] — what
//!   `materialize_and_read`'s collected `Vec<RecordBatch>` feeds today, and
//!   what a whole-set arm (mining, GradCache, classification — see below)
//!   still needs, since those arms require the complete row set before they
//!   can do anything (mining scores every candidate, GradCache treats the
//!   whole set as one in-batch-negative batch, classification's label
//!   vocabulary is a function of every row).
//! - `append_selected_rows`: the per-row-index entry point. Given a batch
//!   and the row indices *within that batch* the current step's chunk wants,
//!   it extends a `ChunkAccumulator` with ONLY those rows — a row this
//!   function is not asked to keep is never cloned into the accumulator, so a
//!   stream walking a batch that spans several ranks' worth of rows (world >
//!   1) allocates nothing for the rows another rank owns.
//!
//! # Classification is an ADDITIONAL exemption (not just K3/mining/GradCache)
//!
//! `ChunkAccumulator::new_for` refuses `DetectedFormat::Classification`:
//! assigning a label its integer class index needs the FULL label vocabulary
//! (`build_training_data_loader`'s `BTreeSet` pass over every row), which is
//! exactly the same "whole-dataset pass before any chunk can be built" shape
//! the regression K3 scaler already carries as a named, separate, unfiltered
//! pass (`super::target::TargetScaler`) — never streamed. Classification
//! therefore stays on the eager path alongside the stated mining/GradCache
//! exemptions; [`super::stream::TrainingSetStream::open`] refuses it at open,
//! not mid-stream.

use arrow::array::RecordBatch;
use jammi_db::error::{JammiError, Result};

use crate::model::ModelTask;

use super::data::{TextChunk, TrainingDataLoader, TrainingFormat};

/// Extract all string values from an Arrow column, or `None` when the column
/// is not honestly readable as text.
///
/// DataFusion 52+ returns Parquet string columns as `Utf8View` by default;
/// older versions returned `Utf8` or `LargeUtf8`. Dictionary-encoded variants
/// are also possible. Fast paths cover the three common types; the `cast`
/// fallback handles everything else.
///
/// # Two refusals the cast fallback cannot be trusted to make (family D)
///
/// `arrow::compute::cast`'s DEFAULT options are `safe: true`, which means a
/// value the target type cannot represent becomes NULL rather than an error.
/// Combined with `StringArray::value(i)` — which returns `""` for a null slot
/// rather than failing — the fallback silently turned unreadable cells into
/// empty strings:
///
/// - A **binary** column (an image or audio triplet submitted under a TEXT
///   task) cast cell-by-cell into NULLs, and every training row became the
///   empty string. The job then completed, published an adapter, and reported
///   success — a fine-tune of a text tower on nothing at all. Bytes are not
///   text: the binary families are refused OUTRIGHT here, so the caller gets
///   the typed, task-naming schema error its caller already raises.
/// - Any OTHER column whose cast introduces a null where the source had a
///   value is refused for the same reason — the empty string is a fabricated
///   input, not a reading of the caller's data.
///
/// A column that was ALREADY null keeps its historical `""` reading: that is a
/// pre-existing null-handling contract of the text path, not a value this
/// function invented.
pub(crate) fn extract_string_column(col: &dyn arrow::array::Array) -> Option<Vec<String>> {
    use arrow::array::{Array, LargeStringArray, StringArray, StringViewArray};
    use arrow::datatypes::DataType;

    if let Some(a) = col.as_any().downcast_ref::<StringViewArray>() {
        return Some((0..a.len()).map(|i| a.value(i).to_string()).collect());
    }
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        return Some((0..a.len()).map(|i| a.value(i).to_string()).collect());
    }
    if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
        return Some((0..a.len()).map(|i| a.value(i).to_string()).collect());
    }
    if matches!(
        col.data_type(),
        DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView
            | DataType::FixedSizeBinary(_)
    ) {
        return None;
    }
    let casted = arrow::compute::cast(col, &DataType::Utf8).ok()?;
    let a = casted.as_any().downcast_ref::<StringArray>()?;
    if (0..a.len()).any(|i| a.is_null(i) && !col.is_null(i)) {
        return None;
    }
    Some((0..a.len()).map(|i| a.value(i).to_string()).collect())
}

/// Extract a binary column into owned byte vectors, accepting the Arrow binary
/// families DataFusion produces for an audio-bytes column
/// (`Binary`/`LargeBinary`/`BinaryView`). Returns `None` for any other type so
/// the caller can surface a typed schema error.
pub(crate) fn extract_binary_column(col: &dyn arrow::array::Array) -> Option<Vec<Vec<u8>>> {
    use arrow::array::{Array, BinaryArray, BinaryViewArray, LargeBinaryArray};

    if let Some(a) = col.as_any().downcast_ref::<BinaryArray>() {
        return Some((0..a.len()).map(|i| a.value(i).to_vec()).collect());
    }
    if let Some(a) = col.as_any().downcast_ref::<LargeBinaryArray>() {
        return Some((0..a.len()).map(|i| a.value(i).to_vec()).collect());
    }
    if let Some(a) = col.as_any().downcast_ref::<BinaryViewArray>() {
        return Some((0..a.len()).map(|i| a.value(i).to_vec()).collect());
    }
    None
}

/// Why a numeric column could not be read into clean `f32` targets.
pub(crate) enum NumericColumnError {
    /// The column's Arrow type is not numeric (and the cast fallback failed).
    NotNumeric,
    /// A null target at the cited row index. Rejected rather than coerced to
    /// `0.0`, which would silently corrupt the scaler's μ/σ.
    Null(usize),
    /// A `NaN` target at the cited row index (float columns only). Rejected for
    /// the same reason as a null.
    Nan(usize),
}

/// Extract a numeric column into `Vec<f32>`, accepting the Arrow numeric
/// families DataFusion emits for a regression `target` column. Integer targets
/// (e.g. an `int64` year) are common, so the fast paths cover
/// `Int64`/`Int32`/`Float64`/`Float32`; the final `cast` fallback handles the
/// remaining numeric types (`UInt*`, `Int16`, `Decimal`, …) so a target's exact
/// Arrow width never decides whether the fine-tune is reachable.
///
/// **Null/NaN rejection is load-bearing.** `Array::value(i)` on a null slot
/// returns a zero default rather than erroring, which would silently corrupt
/// the scaler's μ/σ. A null or `NaN` target therefore returns a typed error
/// citing the row, never a coerced `0.0`.
pub(crate) fn extract_numeric_column(
    col: &dyn arrow::array::Array,
) -> std::result::Result<Vec<f32>, NumericColumnError> {
    use arrow::array::{Array, Float32Array, Float64Array, Int32Array, Int64Array};
    use arrow::datatypes::DataType;

    // A string/binary `target` is a schema mistake, not numeric data — reject it
    // as "not numeric" rather than letting the Float64 cast turn unparseable
    // strings into nulls (which would surface a misleading per-row null error).
    if matches!(
        col.data_type(),
        DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView
            | DataType::Boolean
            | DataType::Null
    ) {
        return Err(NumericColumnError::NotNumeric);
    }

    // Reject a null in any slot up front; `value(i)` would otherwise return a
    // garbage default for it.
    if let Some(i) = (0..col.len()).find(|&i| col.is_null(i)) {
        return Err(NumericColumnError::Null(i));
    }

    let floats: Vec<f32> = if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
        (0..a.len()).map(|i| a.value(i) as f32).collect()
    } else if let Some(a) = col.as_any().downcast_ref::<Int32Array>() {
        (0..a.len()).map(|i| a.value(i) as f32).collect()
    } else if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
        (0..a.len()).map(|i| a.value(i) as f32).collect()
    } else if let Some(a) = col.as_any().downcast_ref::<Float32Array>() {
        (0..a.len()).map(|i| a.value(i)).collect()
    } else {
        // Fallback: cast through Float64 for the remaining numeric families. A
        // cast failure means the column is not numeric.
        let casted = arrow::compute::cast(col, &DataType::Float64)
            .map_err(|_| NumericColumnError::NotNumeric)?;
        let a = casted
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or(NumericColumnError::NotNumeric)?;
        // The cast can introduce nulls (e.g. an unrepresentable value); reject
        // them with the same per-row contract.
        if let Some(i) = (0..a.len()).find(|&i| a.is_null(i)) {
            return Err(NumericColumnError::Null(i));
        }
        (0..a.len()).map(|i| a.value(i) as f32).collect()
    };

    // A NaN target (float columns only) would corrupt the scaler; reject it
    // citing the row, mirroring the null contract.
    if let Some(i) = floats.iter().position(|v| v.is_nan()) {
        return Err(NumericColumnError::Nan(i));
    }
    Ok(floats)
}

/// Build a [`TrainingDataLoader`] from query result batches.
///
/// `task` selects how `anchor`/`positive`/`negative` triplet columns are read:
/// an image or audio embedding task reads them as encoded MEDIA bytes; every
/// other task reads them as text. The column names are identical across
/// modalities (the triplet shape is the same) — only the cell decoding
/// differs, so the caller's chosen task is the discriminator, not a parallel
/// set of column names, and not a byte-header sniff (an encoded WAV and an
/// encoded PNG are both binary blobs).
/// The training format a projection's COLUMN NAMES and the job's task fix,
/// before a single row is read — everything the training-set producer must know
/// to name the table it is about to write.
///
/// The data-derived parameters of [`TrainingFormat`] (`Classification`'s
/// `num_classes`, `Ner`'s `num_labels`) are absent by construction: they are
/// counts of what the rows turned out to contain, which is not knowable at the
/// point the table is named, and which the canonical tag drops for exactly that
/// reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DetectedFormat {
    Contrastive,
    Pairs,
    Triplet,
    MediaTriplet,
    Classification,
    Regression,
}

impl DetectedFormat {
    /// The canonical tag this shape records. Every arm names the
    /// [`TrainingFormat`] variant the loader will build and reads the tag off
    /// [`TrainingFormat::format_tag`] — the mapping lives in exactly one place,
    /// so the tag a table is written under and the tag its loader reports can
    /// never be spelled differently.
    pub(crate) fn format_tag(self) -> &'static str {
        match self {
            Self::Contrastive => TrainingFormat::Contrastive.format_tag(),
            Self::Pairs => TrainingFormat::Pairs.format_tag(),
            Self::Triplet => TrainingFormat::Triplet.format_tag(),
            Self::MediaTriplet => TrainingFormat::MediaTriplet.format_tag(),
            // `num_classes` is a function of the rows and the tag discards it
            // (see `format_tag`), so every value names the same tag; the zero
            // is a neutral placeholder that never leaves this expression.
            Self::Classification => TrainingFormat::Classification { num_classes: 0 }.format_tag(),
            Self::Regression => TrainingFormat::Regression.format_tag(),
        }
    }
}

/// Detect the training format from the projected column names and the job's
/// task — the SINGLE classifier, shared by the producer (which needs the format
/// tag before it writes the table) and by [`build_training_data_loader`] (which
/// needs the shape to read it back). Two copies of these predicates would let a
/// table be written under one format and read under another.
///
/// The arm ORDER is part of the contract: media triplets are recognised before
/// text ones (the columns are identical; only `task` distinguishes an encoded
/// blob from a string), and regression is tested before classification and
/// gated on `task == Regression`, so a numeric outcome can never fall into the
/// classification path and be gathered as a class index.
pub(crate) fn detect_training_format(
    columns: &[String],
    task: ModelTask,
) -> Result<DetectedFormat> {
    let col_names: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();

    let has_contrastive = col_names.contains(&"text_a")
        && col_names.contains(&"text_b")
        && col_names.contains(&"score");
    let has_triplet = col_names.contains(&"anchor")
        && col_names.contains(&"positive")
        && col_names.contains(&"negative");
    // Pairs = anchor + positive with no negative column. In-batch negatives
    // (MultipleNegativesRanking) supply the contrast, so `negative` is absent.
    let has_pairs = col_names.contains(&"anchor")
        && col_names.contains(&"positive")
        && !col_names.contains(&"negative");
    let has_classification = col_names.contains(&"text") && col_names.contains(&"label");
    let has_regression = col_names.contains(&"text") && col_names.contains(&"target");

    if has_triplet && matches!(task, ModelTask::AudioEmbedding | ModelTask::ImageEmbedding) {
        Ok(DetectedFormat::MediaTriplet)
    } else if has_contrastive {
        Ok(DetectedFormat::Contrastive)
    } else if has_triplet {
        Ok(DetectedFormat::Triplet)
    } else if has_pairs {
        Ok(DetectedFormat::Pairs)
    } else if task == ModelTask::Regression {
        // A `task=regression` request with no usable `target` column is a typed
        // error here, never a fall-through to classification.
        if !has_regression {
            return Err(JammiError::FineTune(format!(
                "task=regression needs a string 'text' column and a numeric 'target' column, \
                 but the projected columns are {col_names:?}. (Classification's string 'label' \
                 is distinct: name the numeric outcome column 'target'.)"
            )));
        }
        Ok(DetectedFormat::Regression)
    } else if has_classification {
        Ok(DetectedFormat::Classification)
    } else {
        Err(JammiError::FineTune(format!(
            "Cannot detect training format from columns: {col_names:?}. \
             Expected contrastive (text_a, text_b, score), triplet (anchor, positive, negative), \
             pairs (anchor, positive), classification (text, label), or regression \
             (text, target) with task=regression. For image/audio triplets, use the \
             same (anchor, positive, negative) columns with binary cells and \
             task=image_embedding/audio_embedding."
        )))
    }
}

/// The training-set producer's read-back, converted into a
/// [`TrainingDataLoader`]: dispatches on [`detect_training_format`]'s
/// classification of the projected columns and the job's task, then decodes
/// every `RecordBatch` the eager read-back returned into the matching
/// `TrainingRow` shape.
pub(crate) fn build_training_data_loader(
    batches: &[RecordBatch],
    columns: &[String],
    task: ModelTask,
) -> Result<TrainingDataLoader> {
    let detected = detect_training_format(columns, task)?;

    // Exhaustive on every `DetectedFormat` arm — no `_`, so a seventh variant
    // is a compile error here rather than a silent fall-through to whatever
    // arm happened to be last.
    match detected {
        DetectedFormat::MediaTriplet => build_media_triplet_loader(batches, task),
        DetectedFormat::Contrastive => {
            let mut rows = Vec::new();
            for batch in batches {
                let a_col = batch
                    .column_by_name("text_a")
                    .ok_or_else(|| JammiError::FineTune("Missing column 'text_a'".into()))?;
                let b_col = batch
                    .column_by_name("text_b")
                    .ok_or_else(|| JammiError::FineTune("Missing column 'text_b'".into()))?;
                let s_col = batch
                    .column_by_name("score")
                    .ok_or_else(|| JammiError::FineTune("Missing column 'score'".into()))?;

                let a_vals = extract_string_column(a_col.as_ref()).ok_or_else(|| {
                    JammiError::FineTune("'text_a' is not a string column".into())
                })?;
                let b_vals = extract_string_column(b_col.as_ref()).ok_or_else(|| {
                    JammiError::FineTune("'text_b' is not a string column".into())
                })?;
                let s_arr = s_col
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .map(|arr| {
                        (0..arr.len())
                            .map(|i| arr.value(i) as f32)
                            .collect::<Vec<_>>()
                    })
                    .or_else(|| {
                        s_col
                            .as_any()
                            .downcast_ref::<arrow::array::Float32Array>()
                            .map(|arr| (0..arr.len()).map(|i| arr.value(i)).collect())
                    })
                    .ok_or_else(|| JammiError::FineTune("'score' is not a float column".into()))?;

                for (i, &score) in s_arr.iter().enumerate().take(batch.num_rows()) {
                    rows.push((a_vals[i].clone(), b_vals[i].clone(), score));
                }
            }
            Ok(TrainingDataLoader::from_contrastive(rows))
        }
        DetectedFormat::Triplet => {
            let mut rows = Vec::new();
            for batch in batches {
                let schema_info = || {
                    batch
                        .schema()
                        .fields()
                        .iter()
                        .map(|f| format!("{}:{}", f.name(), f.data_type()))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                let anchor_vals = batch
                    .column_by_name("anchor")
                    .and_then(|c| extract_string_column(c.as_ref()))
                    .ok_or_else(|| {
                        JammiError::FineTune(format!(
                        "Missing/invalid 'anchor' column: task {task} expects text columns; for \
                         image/audio triplets submit task=image_embedding/audio_embedding. \
                         Batch schema: [{}]",
                        schema_info()
                    ))
                    })?;
                let pos_vals = batch
                    .column_by_name("positive")
                    .and_then(|c| extract_string_column(c.as_ref()))
                    .ok_or_else(|| {
                        JammiError::FineTune(format!(
                        "Missing/invalid 'positive' column: task {task} expects text columns; for \
                         image/audio triplets submit task=image_embedding/audio_embedding. \
                         Batch schema: [{}]",
                        schema_info()
                    ))
                    })?;
                let neg_vals = batch
                    .column_by_name("negative")
                    .and_then(|c| extract_string_column(c.as_ref()))
                    .ok_or_else(|| {
                        JammiError::FineTune(format!(
                        "Missing/invalid 'negative' column: task {task} expects text columns; for \
                         image/audio triplets submit task=image_embedding/audio_embedding. \
                         Batch schema: [{}]",
                        schema_info()
                    ))
                    })?;

                for i in 0..batch.num_rows() {
                    rows.push((
                        anchor_vals[i].clone(),
                        pos_vals[i].clone(),
                        neg_vals[i].clone(),
                    ));
                }
            }
            Ok(TrainingDataLoader::from_triplets(rows))
        }
        DetectedFormat::Pairs => {
            let mut rows = Vec::new();
            for batch in batches {
                let schema_info = || {
                    batch
                        .schema()
                        .fields()
                        .iter()
                        .map(|f| format!("{}:{}", f.name(), f.data_type()))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                let anchor_vals = batch
                    .column_by_name("anchor")
                    .and_then(|c| extract_string_column(c.as_ref()))
                    .ok_or_else(|| {
                        JammiError::FineTune(format!(
                        "Missing/invalid 'anchor' column: task {task} expects text columns, and \
                         this source has the anchor/positive PAIR shape, which is read as text \
                         for every task. Image/audio training reads encoded media bytes only \
                         from the anchor/positive/negative TRIPLET shape under \
                         task=image_embedding/audio_embedding — add a 'negative' column. \
                         Batch schema: [{}]",
                        schema_info()
                    ))
                    })?;
                let pos_vals = batch
                    .column_by_name("positive")
                    .and_then(|c| extract_string_column(c.as_ref()))
                    .ok_or_else(|| {
                        JammiError::FineTune(format!(
                            "Missing/invalid 'positive' column: task {task} expects text columns, \
                         and this source has the anchor/positive PAIR shape, which is read as \
                         text for every task. Image/audio training reads encoded media bytes \
                         only from the anchor/positive/negative TRIPLET shape under \
                         task=image_embedding/audio_embedding — add a 'negative' column. \
                         Batch schema: [{}]",
                            schema_info()
                        ))
                    })?;
                for i in 0..batch.num_rows() {
                    rows.push((anchor_vals[i].clone(), pos_vals[i].clone()));
                }
            }
            Ok(TrainingDataLoader::from_pairs(rows))
        }
        DetectedFormat::Regression => {
            // Regression: a string `text` column and a numeric `target` column. The
            // target is read into `f32` (handling int64/float64/float32/… via
            // `extract_numeric_column`); nulls and NaNs are rejected citing the row
            // rather than coerced, since a coerced `0.0` would silently corrupt the
            // scaler's μ/σ.
            let mut rows = Vec::new();
            for batch in batches {
                let text_vals = batch
                    .column_by_name("text")
                    .and_then(|c| extract_string_column(c.as_ref()))
                    .ok_or_else(|| JammiError::FineTune("Missing/invalid 'text' column".into()))?;
                let target_col = batch
                    .column_by_name("target")
                    .ok_or_else(|| JammiError::FineTune("Missing 'target' column".into()))?;
                let target_vals = extract_numeric_column(target_col.as_ref()).map_err(|e| {
                    JammiError::FineTune(match e {
                        NumericColumnError::NotNumeric => format!(
                            "regression 'target' is not a numeric column (its Arrow type is {})",
                            target_col.data_type()
                        ),
                        NumericColumnError::Null(i) => format!(
                            "regression 'target' has a null at row {i}; a null target cannot be \
                         coerced (it would corrupt the scaler) — remove or fill the row"
                        ),
                        NumericColumnError::Nan(i) => format!(
                        "regression 'target' has a NaN at row {i}; a NaN target cannot be used \
                         (it would corrupt the scaler) — remove or fix the row"
                    ),
                    })
                })?;
                for i in 0..batch.num_rows() {
                    rows.push((text_vals[i].clone(), target_vals[i]));
                }
            }
            Ok(TrainingDataLoader::from_regression(rows))
        }
        DetectedFormat::Classification => {
            let mut label_set = std::collections::BTreeSet::new();
            let mut rows = Vec::new();
            for batch in batches {
                let text_vals = batch
                    .column_by_name("text")
                    .and_then(|c| extract_string_column(c.as_ref()))
                    .ok_or_else(|| JammiError::FineTune("Missing/invalid 'text' column".into()))?;
                let label_vals = batch
                    .column_by_name("label")
                    .and_then(|c| extract_string_column(c.as_ref()))
                    .ok_or_else(|| JammiError::FineTune("Missing/invalid 'label' column".into()))?;
                for i in 0..batch.num_rows() {
                    label_set.insert(label_vals[i].clone());
                    rows.push((text_vals[i].clone(), label_vals[i].clone()));
                }
            }
            let label_to_idx: std::collections::HashMap<String, u32> = label_set
                .iter()
                .enumerate()
                .map(|(i, l)| (l.clone(), i as u32))
                .collect();
            let num_classes = label_to_idx.len();
            let indexed_rows: Vec<(String, u32)> = rows
                .into_iter()
                .map(|(text, label)| {
                    let idx = label_to_idx[&label];
                    (text, idx)
                })
                .collect();
            Ok(TrainingDataLoader::from_classification(
                indexed_rows,
                num_classes,
            ))
        }
    }
}

/// Build a MEDIA-triplet loader: read `anchor`/`positive`/`negative` as
/// encoded binary columns (audio clips or images, per `task`). Shares the
/// triplet column shape with the text path; only the cell type differs
/// (binary blobs vs strings).
fn build_media_triplet_loader(
    batches: &[RecordBatch],
    task: ModelTask,
) -> Result<TrainingDataLoader> {
    let mut rows = Vec::new();
    for batch in batches {
        let schema_info = || {
            batch
                .schema()
                .fields()
                .iter()
                .map(|f| format!("{}:{}", f.name(), f.data_type()))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let anchor_vals = batch
            .column_by_name("anchor")
            .and_then(|c| extract_binary_column(c.as_ref()))
            .ok_or_else(|| {
                JammiError::FineTune(format!(
                    "Missing/invalid binary 'anchor' column for media triplets (task \
                     {task}). Batch schema: [{}]",
                    schema_info()
                ))
            })?;
        let pos_vals = batch
            .column_by_name("positive")
            .and_then(|c| extract_binary_column(c.as_ref()))
            .ok_or_else(|| {
                JammiError::FineTune(format!(
                    "Missing/invalid binary 'positive' column for media triplets (task \
                     {task}). Batch schema: [{}]",
                    schema_info()
                ))
            })?;
        let neg_vals = batch
            .column_by_name("negative")
            .and_then(|c| extract_binary_column(c.as_ref()))
            .ok_or_else(|| {
                JammiError::FineTune(format!(
                    "Missing/invalid binary 'negative' column for media triplets (task \
                     {task}). Batch schema: [{}]",
                    schema_info()
                ))
            })?;

        for i in 0..batch.num_rows() {
            rows.push((
                anchor_vals[i].clone(),
                pos_vals[i].clone(),
                neg_vals[i].clone(),
            ));
        }
    }
    Ok(TrainingDataLoader::from_media_triplets(rows))
}

// =========================================================================
// The per-row-index decode entry (new, U2c): the stream's chunk builder.
// =========================================================================

/// A partially-built [`TextChunk`], grown incrementally by
/// [`append_selected_rows`] across however many `RecordBatch`es the current
/// step's row range spans, then converted to the immutable [`TextChunk`] a
/// consumer receives. Mirrors [`TextChunk`]'s own shapes (minus
/// `Classification`/`Ner`, neither of which a stream ever builds — see the
/// module doc).
#[derive(Debug)]
pub(crate) enum ChunkAccumulator {
    Contrastive {
        texts_a: Vec<String>,
        texts_b: Vec<String>,
        scores: Vec<f32>,
    },
    Pairs {
        anchors: Vec<String>,
        positives: Vec<String>,
    },
    Triplet {
        anchors: Vec<String>,
        positives: Vec<String>,
        negatives: Vec<String>,
    },
    MediaTriplet {
        anchors: Vec<Vec<u8>>,
        positives: Vec<Vec<u8>>,
        negatives: Vec<Vec<u8>>,
    },
    Regression {
        texts: Vec<String>,
        targets: Vec<f32>,
    },
}

impl ChunkAccumulator {
    /// A fresh, empty accumulator shaped for `detected` — refuses
    /// [`DetectedFormat::Classification`] (see the module doc: it needs a
    /// whole-dataset label vocabulary, not a per-step one).
    pub(crate) fn new_for(detected: DetectedFormat) -> Result<Self> {
        match detected {
            DetectedFormat::Contrastive => Ok(Self::Contrastive {
                texts_a: Vec::new(),
                texts_b: Vec::new(),
                scores: Vec::new(),
            }),
            DetectedFormat::Pairs => Ok(Self::Pairs {
                anchors: Vec::new(),
                positives: Vec::new(),
            }),
            DetectedFormat::Triplet => Ok(Self::Triplet {
                anchors: Vec::new(),
                positives: Vec::new(),
                negatives: Vec::new(),
            }),
            DetectedFormat::MediaTriplet => Ok(Self::MediaTriplet {
                anchors: Vec::new(),
                positives: Vec::new(),
                negatives: Vec::new(),
            }),
            DetectedFormat::Regression => Ok(Self::Regression {
                texts: Vec::new(),
                targets: Vec::new(),
            }),
            DetectedFormat::Classification => Err(JammiError::FineTune(
                "classification cannot stream per-step: assigning a label its integer class \
                 index needs the FULL label vocabulary (every row), the same whole-dataset-pass \
                 shape the regression K3 scaler already carries as a named, separate, unfiltered \
                 pass — it stays on the eager `TrainingDataLoader` path, alongside the stated \
                 mining/GradCache exemptions"
                    .into(),
            )),
        }
    }

    /// Consume this accumulator into the immutable [`TextChunk`] a consumer
    /// receives — a one-to-one field rename, never a re-derivation.
    pub(crate) fn into_text_chunk(self) -> TextChunk {
        match self {
            Self::Contrastive {
                texts_a,
                texts_b,
                scores,
            } => TextChunk::Contrastive {
                texts_a,
                texts_b,
                scores,
            },
            Self::Pairs { anchors, positives } => TextChunk::Pairs { anchors, positives },
            Self::Triplet {
                anchors,
                positives,
                negatives,
            } => TextChunk::Triplet {
                anchors,
                positives,
                negatives,
            },
            Self::MediaTriplet {
                anchors,
                positives,
                negatives,
            } => TextChunk::MediaTriplet {
                anchors,
                positives,
                negatives,
            },
            Self::Regression { texts, targets } => TextChunk::Regression { texts, targets },
        }
    }
}

/// Append the rows at `indices` (row positions **within `batch`**, any order
/// the caller likes but always ascending in practice) into `acc` — the SAME
/// column extractors [`build_training_data_loader`] uses, applied to the
/// whole batch once (an Arrow array is columnar; there is no cheaper way to
/// read one cell than to have the typed array reference in hand) and then
/// cloned ONLY for `indices` — a row this function is not asked to keep by
/// its index is never cloned into `acc`, so a stream skipping another rank's
/// rows (world > 1) allocates nothing for them.
///
/// `acc` must have been built by [`ChunkAccumulator::new_for`] with the SAME
/// `detected` this call receives — this is an internal invariant of
/// [`super::stream::TrainingSetStream`]'s pump, not a caller-facing contract,
/// so a mismatch is an internal-error panic rather than a typed `Result`.
pub(crate) fn append_selected_rows(
    detected: DetectedFormat,
    task: ModelTask,
    batch: &RecordBatch,
    indices: &[usize],
    acc: &mut ChunkAccumulator,
) -> Result<()> {
    if indices.is_empty() {
        return Ok(());
    }
    let schema_info = || {
        batch
            .schema()
            .fields()
            .iter()
            .map(|f| format!("{}:{}", f.name(), f.data_type()))
            .collect::<Vec<_>>()
            .join(", ")
    };
    match (detected, acc) {
        (
            DetectedFormat::Contrastive,
            ChunkAccumulator::Contrastive {
                texts_a,
                texts_b,
                scores,
            },
        ) => {
            let a_col = batch
                .column_by_name("text_a")
                .ok_or_else(|| JammiError::FineTune("Missing column 'text_a'".into()))?;
            let b_col = batch
                .column_by_name("text_b")
                .ok_or_else(|| JammiError::FineTune("Missing column 'text_b'".into()))?;
            let s_col = batch
                .column_by_name("score")
                .ok_or_else(|| JammiError::FineTune("Missing column 'score'".into()))?;
            let a_vals = extract_string_column(a_col.as_ref())
                .ok_or_else(|| JammiError::FineTune("'text_a' is not a string column".into()))?;
            let b_vals = extract_string_column(b_col.as_ref())
                .ok_or_else(|| JammiError::FineTune("'text_b' is not a string column".into()))?;
            let s_vals = s_col
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .map(|arr| {
                    (0..arr.len())
                        .map(|i| arr.value(i) as f32)
                        .collect::<Vec<_>>()
                })
                .or_else(|| {
                    s_col
                        .as_any()
                        .downcast_ref::<arrow::array::Float32Array>()
                        .map(|arr| (0..arr.len()).map(|i| arr.value(i)).collect())
                })
                .ok_or_else(|| JammiError::FineTune("'score' is not a float column".into()))?;
            for &i in indices {
                texts_a.push(a_vals[i].clone());
                texts_b.push(b_vals[i].clone());
                scores.push(s_vals[i]);
            }
        }
        (DetectedFormat::Pairs, ChunkAccumulator::Pairs { anchors, positives }) => {
            let anchor_vals = batch
                .column_by_name("anchor")
                .and_then(|c| extract_string_column(c.as_ref()))
                .ok_or_else(|| {
                    JammiError::FineTune(format!(
                        "Missing/invalid 'anchor' column: task {task} expects text columns. \
                         Batch schema: [{}]",
                        schema_info()
                    ))
                })?;
            let pos_vals = batch
                .column_by_name("positive")
                .and_then(|c| extract_string_column(c.as_ref()))
                .ok_or_else(|| {
                    JammiError::FineTune(format!(
                        "Missing/invalid 'positive' column: task {task} expects text columns. \
                         Batch schema: [{}]",
                        schema_info()
                    ))
                })?;
            for &i in indices {
                anchors.push(anchor_vals[i].clone());
                positives.push(pos_vals[i].clone());
            }
        }
        (
            DetectedFormat::Triplet,
            ChunkAccumulator::Triplet {
                anchors,
                positives,
                negatives,
            },
        ) => {
            let anchor_vals = batch
                .column_by_name("anchor")
                .and_then(|c| extract_string_column(c.as_ref()))
                .ok_or_else(|| {
                    JammiError::FineTune(format!(
                        "Missing/invalid 'anchor' column: task {task} expects text columns. \
                         Batch schema: [{}]",
                        schema_info()
                    ))
                })?;
            let pos_vals = batch
                .column_by_name("positive")
                .and_then(|c| extract_string_column(c.as_ref()))
                .ok_or_else(|| {
                    JammiError::FineTune(format!(
                        "Missing/invalid 'positive' column: task {task} expects text columns. \
                         Batch schema: [{}]",
                        schema_info()
                    ))
                })?;
            let neg_vals = batch
                .column_by_name("negative")
                .and_then(|c| extract_string_column(c.as_ref()))
                .ok_or_else(|| {
                    JammiError::FineTune(format!(
                        "Missing/invalid 'negative' column: task {task} expects text columns. \
                         Batch schema: [{}]",
                        schema_info()
                    ))
                })?;
            for &i in indices {
                anchors.push(anchor_vals[i].clone());
                positives.push(pos_vals[i].clone());
                negatives.push(neg_vals[i].clone());
            }
        }
        (
            DetectedFormat::MediaTriplet,
            ChunkAccumulator::MediaTriplet {
                anchors,
                positives,
                negatives,
            },
        ) => {
            let anchor_vals = batch
                .column_by_name("anchor")
                .and_then(|c| extract_binary_column(c.as_ref()))
                .ok_or_else(|| {
                    JammiError::FineTune(format!(
                        "Missing/invalid binary 'anchor' column for media triplets (task \
                         {task}). Batch schema: [{}]",
                        schema_info()
                    ))
                })?;
            let pos_vals = batch
                .column_by_name("positive")
                .and_then(|c| extract_binary_column(c.as_ref()))
                .ok_or_else(|| {
                    JammiError::FineTune(format!(
                        "Missing/invalid binary 'positive' column for media triplets (task \
                         {task}). Batch schema: [{}]",
                        schema_info()
                    ))
                })?;
            let neg_vals = batch
                .column_by_name("negative")
                .and_then(|c| extract_binary_column(c.as_ref()))
                .ok_or_else(|| {
                    JammiError::FineTune(format!(
                        "Missing/invalid binary 'negative' column for media triplets (task \
                         {task}). Batch schema: [{}]",
                        schema_info()
                    ))
                })?;
            for &i in indices {
                anchors.push(anchor_vals[i].clone());
                positives.push(pos_vals[i].clone());
                negatives.push(neg_vals[i].clone());
            }
        }
        (DetectedFormat::Regression, ChunkAccumulator::Regression { texts, targets }) => {
            let text_vals = batch
                .column_by_name("text")
                .and_then(|c| extract_string_column(c.as_ref()))
                .ok_or_else(|| JammiError::FineTune("Missing/invalid 'text' column".into()))?;
            let target_col = batch
                .column_by_name("target")
                .ok_or_else(|| JammiError::FineTune("Missing 'target' column".into()))?;
            let target_vals = extract_numeric_column(target_col.as_ref()).map_err(|e| {
                JammiError::FineTune(match e {
                    NumericColumnError::NotNumeric => format!(
                        "regression 'target' is not a numeric column (its Arrow type is {})",
                        target_col.data_type()
                    ),
                    NumericColumnError::Null(i) => format!(
                        "regression 'target' has a null at row {i}; a null target cannot be \
                         coerced (it would corrupt the scaler) — remove or fill the row"
                    ),
                    NumericColumnError::Nan(i) => format!(
                        "regression 'target' has a NaN at row {i}; a NaN target cannot be used \
                         (it would corrupt the scaler) — remove or fix the row"
                    ),
                })
            })?;
            for &i in indices {
                texts.push(text_vals[i].clone());
                targets.push(target_vals[i]);
            }
        }
        (detected, acc) => {
            unreachable!(
                "append_selected_rows: ChunkAccumulator variant does not match detected format \
                 {detected:?} — an internal invariant of TrainingSetStream's pump, which always \
                 builds `acc` via `ChunkAccumulator::new_for(detected)` first: {acc:?}"
            );
        }
    }
    Ok(())
}

/// Read-only column-family/null/NaN check the [`super::stream::TrainingSetStream`]
/// load-time pre-pass drives: refuse a column this decoder cannot honestly
/// read BEFORE any chunk is produced, from a schema-only probe (a `LIMIT 0`
/// read never touches a data row) — the SAME family checks
/// [`extract_string_column`]/[`extract_numeric_column`] apply per-batch,
/// hoisted to a single load-time check over the SCHEMA rather than the rows.
pub(crate) fn check_schema_matches_format(
    schema: &arrow::datatypes::Schema,
    detected: DetectedFormat,
    task: ModelTask,
) -> Result<()> {
    let text_columns: &[&str] = match detected {
        DetectedFormat::Contrastive => &["text_a", "text_b"],
        DetectedFormat::Pairs => &["anchor", "positive"],
        DetectedFormat::Triplet => &["anchor", "positive", "negative"],
        DetectedFormat::MediaTriplet => &[],
        DetectedFormat::Regression => &["text"],
        DetectedFormat::Classification => &["text", "label"],
    };
    for name in text_columns {
        let field = schema
            .field_with_name(name)
            .map_err(|_| JammiError::FineTune(format!("Missing column '{name}'")))?;
        if matches!(
            field.data_type(),
            arrow::datatypes::DataType::Binary
                | arrow::datatypes::DataType::LargeBinary
                | arrow::datatypes::DataType::BinaryView
                | arrow::datatypes::DataType::FixedSizeBinary(_)
        ) {
            return Err(JammiError::FineTune(format!(
                "'{name}' is a binary column but task {task} expects text; for image/audio \
                 triplets submit task=image_embedding/audio_embedding"
            )));
        }
    }
    if matches!(detected, DetectedFormat::MediaTriplet) {
        for name in ["anchor", "positive", "negative"] {
            let field = schema
                .field_with_name(name)
                .map_err(|_| JammiError::FineTune(format!("Missing column '{name}'")))?;
            if !matches!(
                field.data_type(),
                arrow::datatypes::DataType::Binary
                    | arrow::datatypes::DataType::LargeBinary
                    | arrow::datatypes::DataType::BinaryView
            ) {
                return Err(JammiError::FineTune(format!(
                    "Missing/invalid binary '{name}' column for media triplets (task {task})"
                )));
            }
        }
    }
    if matches!(detected, DetectedFormat::Regression) {
        let field = schema
            .field_with_name("target")
            .map_err(|_| JammiError::FineTune("Missing 'target' column".into()))?;
        if matches!(
            field.data_type(),
            arrow::datatypes::DataType::Utf8
                | arrow::datatypes::DataType::LargeUtf8
                | arrow::datatypes::DataType::Utf8View
                | arrow::datatypes::DataType::Binary
                | arrow::datatypes::DataType::LargeBinary
                | arrow::datatypes::DataType::BinaryView
                | arrow::datatypes::DataType::Boolean
                | arrow::datatypes::DataType::Null
        ) {
            return Err(JammiError::FineTune(format!(
                "regression 'target' is not a numeric column (its Arrow type is {})",
                field.data_type()
            )));
        }
    }
    Ok(())
}

/// Every column [`check_schema_matches_format`]/the null-NaN aggregate pass
/// needs, for `detected` — used to render the load-time pre-pass's aggregate
/// SQL (`super::stream`'s `validate_window`).
pub(crate) fn numeric_target_column(detected: DetectedFormat) -> Option<&'static str> {
    match detected {
        DetectedFormat::Regression => Some("target"),
        _ => None,
    }
}
