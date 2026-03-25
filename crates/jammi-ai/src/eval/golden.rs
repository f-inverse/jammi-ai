//! Golden dataset loading and schema validation.

use std::collections::HashMap;

use arrow::array::{Array, StringArray};
use arrow::datatypes::{DataType, Schema};
use jammi_engine::error::{JammiError, Result};

/// A retrieval query with relevance judgments.
pub struct RetrievalQuery {
    pub query_id: String,
    pub query_text: String,
    pub judgments: Vec<RelevanceJudgment>,
}

/// A relevance judgment: a document ID and its relevance grade.
pub struct RelevanceJudgment {
    pub doc_id: String,
    /// 0 = not relevant, 1 = marginally, 2 = relevant, 3 = highly relevant.
    pub grade: i32,
}

/// A loaded retrieval golden dataset, grouped by query.
pub struct RetrievalGolden {
    pub queries: Vec<RetrievalQuery>,
}

/// A loaded classification golden dataset: id → label.
pub struct ClassificationGolden {
    pub labels: HashMap<String, String>,
}

/// A loaded summarization golden dataset: id → reference.
pub struct SummarizationGolden {
    pub references: HashMap<String, String>,
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
pub fn load_retrieval_golden_from_batches(
    batches: &[arrow::array::RecordBatch],
    has_grades: bool,
) -> Result<RetrievalGolden> {
    let mut query_map: HashMap<String, (String, Vec<RelevanceJudgment>)> = HashMap::new();

    for batch in batches {
        let query_ids = extract_string_column(batch, "query_id")?;
        let query_texts = extract_string_column(batch, "query_text")?;
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

        for i in 0..batch.num_rows() {
            let entry = query_map
                .entry(query_ids[i].clone())
                .or_insert_with(|| (query_texts[i].clone(), Vec::new()));
            entry.1.push(RelevanceJudgment {
                doc_id: relevant_ids[i].clone(),
                grade: grades[i],
            });
        }
    }

    let queries = query_map
        .into_iter()
        .map(|(query_id, (query_text, judgments))| RetrievalQuery {
            query_id,
            query_text,
            judgments,
        })
        .collect();

    Ok(RetrievalGolden { queries })
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

/// Load a summarization golden dataset from DataFusion query results.
pub fn load_summarization_golden_from_batches(
    batches: &[arrow::array::RecordBatch],
) -> Result<SummarizationGolden> {
    let mut references = HashMap::new();

    for batch in batches {
        let ids = extract_string_column(batch, "id")?;
        let refs = extract_string_column(batch, "reference")?;

        for (id, reference) in ids.into_iter().zip(refs) {
            references.insert(id, reference);
        }
    }

    Ok(SummarizationGolden { references })
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
