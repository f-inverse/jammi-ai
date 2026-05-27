//! Golden dataset loading and schema validation.

use std::collections::HashMap;

use arrow::array::{Array, StringArray};
use arrow::datatypes::{DataType, Schema};
use jammi_db::error::{JammiError, Result};
use jammi_numerics::ner::Entity;
pub use jammi_numerics::retrieval::RelevanceJudgment;

/// The input for a retrieval query — either text or image bytes.
pub enum QueryInput {
    /// Text query to encode via `encode_query()`.
    Text(String),
    /// Image bytes (PNG/JPEG) to encode via `encode_image_query()`.
    Image(Vec<u8>),
}

/// A retrieval query with relevance judgments.
pub struct RetrievalQuery {
    pub query_id: String,
    pub input: QueryInput,
    pub judgments: Vec<RelevanceJudgment>,
}

/// A loaded retrieval golden dataset, grouped by query.
pub struct RetrievalGolden {
    pub queries: Vec<RetrievalQuery>,
}

/// A loaded classification golden dataset: id → label.
pub struct ClassificationGolden {
    pub labels: HashMap<String, String>,
}

/// A loaded NER golden dataset: row `id` → gold entity-span set.
///
/// One CSV/parquet row per entity span — multiple spans on the same `id`
/// are grouped into the same `Vec<Entity>` by [`load_ner_golden_from_batches`].
/// `text` is empty and `confidence` is `0.0` for gold rows; the metric's
/// strict equality on `(label, start, end)` ignores both fields.
#[derive(Debug)]
pub struct NerGolden {
    pub entities: HashMap<String, Vec<Entity>>,
}

/// Returns true if `actual` is compatible with `expected` for golden schema validation.
///
/// When Utf8 is expected, we accept any string-like type (Utf8View, LargeUtf8) and
/// any numeric type (Int32, Int64, Float64) because ID columns are commonly integers
/// that we'll cast to string during extraction.
fn is_compatible(actual: &DataType, expected: &DataType) -> bool {
    if actual == expected {
        return true;
    }
    if *expected == DataType::Utf8 {
        matches!(
            actual,
            DataType::Utf8View
                | DataType::LargeUtf8
                | DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float32
                | DataType::Float64
        )
    } else {
        false
    }
}

/// Validate that a schema contains a column with a compatible type.
///
/// For Utf8 expectations, also accepts numeric types (common for ID columns)
/// and string variants (Utf8View, LargeUtf8). Rejects truly incompatible types
/// (Boolean, Binary, List, etc.) with a clear error naming the column and type.
pub fn ensure_column(schema: &Schema, name: &str, expected: DataType) -> Result<()> {
    match schema.field_with_name(name) {
        Ok(field) => {
            if is_compatible(field.data_type(), &expected) {
                Ok(())
            } else {
                Err(JammiError::Eval(format!(
                    "Golden dataset column '{name}' has type {:?}, expected {:?}",
                    field.data_type(),
                    expected
                )))
            }
        }
        Err(_) => Err(JammiError::Eval(format!(
            "Golden dataset missing required column '{name}'"
        ))),
    }
}

/// Load a retrieval golden dataset from DataFusion query results.
///
/// Supports two modes:
/// - Text queries: golden source has `query_text` (Utf8) column
/// - Image queries: golden source has `query_image` (Binary) column
pub fn load_retrieval_golden_from_batches(
    batches: &[arrow::array::RecordBatch],
    has_grades: bool,
    is_image: bool,
) -> Result<RetrievalGolden> {
    let mut query_map: HashMap<String, (QueryInput, Vec<RelevanceJudgment>)> = HashMap::new();

    for batch in batches {
        let query_ids = extract_string_column(batch, "query_id")?;
        let relevant_ids = extract_string_column(batch, "relevant_id")?;

        let grades: Vec<i32> = if has_grades {
            batch
                .column_by_name("relevance_grade")
                .and_then(|col| col.as_any().downcast_ref::<arrow::array::Int32Array>())
                .map(|arr| (0..arr.len()).map(|i| arr.value(i)).collect())
                .unwrap_or_else(|| vec![1; batch.num_rows()])
        } else {
            vec![1; batch.num_rows()]
        };

        if is_image {
            let image_col = batch.column_by_name("query_image").ok_or_else(|| {
                JammiError::Eval("Column 'query_image' not found in batch".into())
            })?;
            let image_bytes = extract_binary_column(image_col)?;
            for i in 0..batch.num_rows() {
                let entry = query_map
                    .entry(query_ids[i].clone())
                    .or_insert_with(|| (QueryInput::Image(image_bytes[i].clone()), Vec::new()));
                entry.1.push(RelevanceJudgment {
                    doc_id: relevant_ids[i].clone(),
                    grade: grades[i],
                });
            }
        } else {
            let query_texts = extract_string_column(batch, "query_text")?;
            for i in 0..batch.num_rows() {
                let entry = query_map
                    .entry(query_ids[i].clone())
                    .or_insert_with(|| (QueryInput::Text(query_texts[i].clone()), Vec::new()));
                entry.1.push(RelevanceJudgment {
                    doc_id: relevant_ids[i].clone(),
                    grade: grades[i],
                });
            }
        }
    }

    let queries = query_map
        .into_iter()
        .map(|(query_id, (input, judgments))| RetrievalQuery {
            query_id,
            input,
            judgments,
        })
        .collect();

    Ok(RetrievalGolden { queries })
}

/// Extract binary data from an Arrow column (handles Binary, LargeBinary, BinaryView).
fn extract_binary_column(col: &dyn Array) -> Result<Vec<Vec<u8>>> {
    use arrow::array::{BinaryArray, BinaryViewArray, LargeBinaryArray};

    if let Some(arr) = col.as_any().downcast_ref::<BinaryArray>() {
        return Ok((0..arr.len()).map(|i| arr.value(i).to_vec()).collect());
    }
    if let Some(arr) = col.as_any().downcast_ref::<LargeBinaryArray>() {
        return Ok((0..arr.len()).map(|i| arr.value(i).to_vec()).collect());
    }
    if let Some(arr) = col.as_any().downcast_ref::<BinaryViewArray>() {
        return Ok((0..arr.len()).map(|i| arr.value(i).to_vec()).collect());
    }
    Err(JammiError::Eval(format!(
        "Column has type {:?}, expected Binary",
        col.data_type()
    )))
}

/// Load a classification golden dataset from DataFusion query results.
pub fn load_classification_golden_from_batches(
    batches: &[arrow::array::RecordBatch],
) -> Result<ClassificationGolden> {
    let mut labels = HashMap::new();

    for batch in batches {
        let ids = extract_string_column(batch, "id")?;
        let label_vals = extract_string_column(batch, "label")?;

        for (id, label) in ids.into_iter().zip(label_vals) {
            labels.insert(id, label);
        }
    }

    Ok(ClassificationGolden { labels })
}

/// Validate that a schema contains an integer-typed column.
///
/// NER gold-span offsets must be integer-coercible; accepts every signed and
/// unsigned integer width Arrow exposes so users producing offsets from
/// pandas (`int64`), DataFusion (`Int64`), or hand-built Arrow arrays
/// (`Int32`/`UInt32`) all land here without a custom cast step.
pub fn ensure_column_int64(schema: &Schema, name: &str) -> Result<()> {
    match schema.field_with_name(name) {
        Ok(field) => {
            if matches!(
                field.data_type(),
                DataType::Int8
                    | DataType::Int16
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::UInt8
                    | DataType::UInt16
                    | DataType::UInt32
                    | DataType::UInt64
            ) {
                Ok(())
            } else {
                Err(JammiError::Eval(format!(
                    "Golden dataset column '{name}' has type {:?}, expected an integer type",
                    field.data_type()
                )))
            }
        }
        Err(_) => Err(JammiError::Eval(format!(
            "Golden dataset missing required column '{name}'"
        ))),
    }
}

/// Load an NER golden dataset from DataFusion query results.
///
/// Input batches must carry four columns: `id` (Utf8 or numeric coerced via
/// [`extract_string_column`]), `label` (Utf8), `start` (any integer width),
/// and `end` (any integer width). Each row contributes one entity span to
/// `entities[id]`; multiple rows sharing an `id` accumulate into the same
/// span vector. `text` is left empty and `confidence` is `0.0`, which the
/// metric ignores via [`Entity`]'s custom equality.
pub fn load_ner_golden_from_batches(batches: &[arrow::array::RecordBatch]) -> Result<NerGolden> {
    let mut entities: HashMap<String, Vec<Entity>> = HashMap::new();

    for batch in batches {
        let ids = extract_string_column(batch, "id")?;
        let labels = extract_string_column(batch, "label")?;
        let starts = extract_usize_column(batch, "start")?;
        let ends = extract_usize_column(batch, "end")?;

        for (((id, label), start), end) in ids.into_iter().zip(labels).zip(starts).zip(ends) {
            entities.entry(id).or_default().push(Entity {
                label,
                start,
                end,
                text: String::new(),
                confidence: 0.0,
            });
        }
    }

    Ok(NerGolden { entities })
}

/// Extract an integer column as `usize`, rejecting negative values with a
/// schema-named error. NER spans index byte offsets (`Entity::start`,
/// `Entity::end`), which the metric kernel models as `usize`.
fn extract_usize_column(batch: &arrow::array::RecordBatch, column: &str) -> Result<Vec<usize>> {
    use arrow::array::{
        Int16Array, Int32Array, Int64Array, Int8Array, UInt16Array, UInt32Array, UInt64Array,
        UInt8Array,
    };

    let col = batch
        .column_by_name(column)
        .ok_or_else(|| JammiError::Eval(format!("Column '{column}' not found in batch")))?;

    macro_rules! signed_to_usize {
        ($arr:expr) => {{
            let arr = $arr;
            (0..arr.len())
                .map(|i| {
                    let v = arr.value(i);
                    if v < 0 {
                        Err(JammiError::Eval(format!(
                            "Golden dataset column '{column}' has negative value {v} at row {i}; \
                             span offsets must be non-negative"
                        )))
                    } else {
                        Ok(v as usize)
                    }
                })
                .collect::<Result<Vec<usize>>>()
        }};
    }

    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
        return signed_to_usize!(arr);
    }
    if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
        return signed_to_usize!(arr);
    }
    if let Some(arr) = col.as_any().downcast_ref::<Int16Array>() {
        return signed_to_usize!(arr);
    }
    if let Some(arr) = col.as_any().downcast_ref::<Int8Array>() {
        return signed_to_usize!(arr);
    }
    if let Some(arr) = col.as_any().downcast_ref::<UInt64Array>() {
        return Ok((0..arr.len()).map(|i| arr.value(i) as usize).collect());
    }
    if let Some(arr) = col.as_any().downcast_ref::<UInt32Array>() {
        return Ok((0..arr.len()).map(|i| arr.value(i) as usize).collect());
    }
    if let Some(arr) = col.as_any().downcast_ref::<UInt16Array>() {
        return Ok((0..arr.len()).map(|i| arr.value(i) as usize).collect());
    }
    if let Some(arr) = col.as_any().downcast_ref::<UInt8Array>() {
        return Ok((0..arr.len()).map(|i| arr.value(i) as usize).collect());
    }

    Err(JammiError::Eval(format!(
        "Column '{column}' has type {:?}, expected an integer type",
        col.data_type()
    )))
}

/// Extract a column as strings, handling Utf8, Utf8View, LargeUtf8, and numeric types.
///
/// Numeric columns (Int64, Int32, Float64, etc.) are converted to their string
/// representation. This is essential for ID columns that may be stored as integers.
pub(crate) fn extract_string_column(
    batch: &arrow::array::RecordBatch,
    column: &str,
) -> Result<Vec<String>> {
    let col = batch
        .column_by_name(column)
        .ok_or_else(|| JammiError::Eval(format!("Column '{column}' not found in batch")))?;

    // String types
    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        return Ok((0..arr.len()).map(|i| arr.value(i).to_string()).collect());
    }
    if let Some(arr) = col.as_any().downcast_ref::<arrow::array::StringViewArray>() {
        return Ok((0..arr.len()).map(|i| arr.value(i).to_string()).collect());
    }
    if let Some(arr) = col
        .as_any()
        .downcast_ref::<arrow::array::LargeStringArray>()
    {
        return Ok((0..arr.len()).map(|i| arr.value(i).to_string()).collect());
    }

    // Numeric types — convert to string (common for ID columns)
    if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Int64Array>() {
        return Ok((0..arr.len()).map(|i| arr.value(i).to_string()).collect());
    }
    if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Int32Array>() {
        return Ok((0..arr.len()).map(|i| arr.value(i).to_string()).collect());
    }
    if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Float64Array>() {
        return Ok((0..arr.len()).map(|i| arr.value(i).to_string()).collect());
    }

    Err(JammiError::Eval(format!(
        "Column '{column}' has unsupported type {:?} (expected string or numeric)",
        col.data_type()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, RecordBatch, StringArray};
    use arrow::datatypes::{Field, Schema as ArrowSchema};
    use std::sync::Arc;

    #[test]
    fn load_ner_golden_round_trips_three_row_fixture() {
        // Mirrors `cookbook/fixtures/tiny_ner_gold.csv` shape — one row per
        // entity span, multiple rows per id permitted.
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("label", DataType::Utf8, false),
            Field::new("start", DataType::Int64, false),
            Field::new("end", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["1", "1", "2"])),
                Arc::new(StringArray::from(vec!["PER", "ORG", "PER"])),
                Arc::new(Int64Array::from(vec![0, 12, 4])),
                Arc::new(Int64Array::from(vec![5, 18, 11])),
            ],
        )
        .expect("valid batch");

        let golden = load_ner_golden_from_batches(&[batch]).expect("loader succeeds");

        assert_eq!(golden.entities.len(), 2);

        let row_one = golden.entities.get("1").expect("row 1 present");
        assert_eq!(row_one.len(), 2);
        let labels: Vec<&str> = row_one.iter().map(|e| e.label.as_str()).collect();
        assert!(labels.contains(&"PER"));
        assert!(labels.contains(&"ORG"));

        let row_two = golden.entities.get("2").expect("row 2 present");
        assert_eq!(row_two.len(), 1);
        assert_eq!(row_two[0].label, "PER");
        assert_eq!(row_two[0].start, 4);
        assert_eq!(row_two[0].end, 11);
        // Gold rows leave `text` empty and `confidence` zero — the metric
        // ignores both fields via the `Entity` custom equality.
        assert!(row_two[0].text.is_empty());
        assert_eq!(row_two[0].confidence, 0.0);
    }

    #[test]
    fn load_ner_golden_rejects_negative_span_offsets() {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("label", DataType::Utf8, false),
            Field::new("start", DataType::Int64, false),
            Field::new("end", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["1"])),
                Arc::new(StringArray::from(vec!["PER"])),
                Arc::new(Int64Array::from(vec![-1])),
                Arc::new(Int64Array::from(vec![3])),
            ],
        )
        .expect("valid batch");

        let err =
            load_ner_golden_from_batches(&[batch]).expect_err("negative offset must be rejected");
        let msg = format!("{err}");
        assert!(msg.contains("negative"), "error message: {msg}");
        assert!(msg.contains("start"), "error message: {msg}");
    }
}
