pub mod provenance;
pub mod schema;

use std::collections::HashMap;

use arrow::array::ArrayRef;

pub use schema::evidence_schema;

/// Primary key for a row in the evidence model.
///
/// Stable across operators: the same (source_id, row_id) pair from
/// different operators produces a single merged output row.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EvidenceRowId {
    pub source_id: String,
    pub row_id: String,
}

/// Provenance tracking for an evidence row.
///
/// Records *how* a row was found (`retrieved_by`) vs *what was added
/// after retrieval* (`annotated_by`). For join results, `contributing_rows`
/// tracks which source rows combined to produce this row.
#[derive(Debug, Clone)]
pub struct RowProvenance {
    pub retrieved_by: Vec<String>,
    pub annotated_by: Vec<String>,
    pub contributing_rows: Vec<EvidenceRowId>,
}

/// A result row with structured evidence from multiple operators.
///
/// This is the stable type that downstream phases (Evaluation, Caching)
/// depend on. The dynamic `List<Utf8>` columns in RecordBatch output map
/// directly to these fields.
#[derive(Debug)]
pub struct EvidenceRow {
    pub source_id: String,
    pub row_id: String,

    pub retrieved_by: Vec<String>,
    pub annotated_by: Vec<String>,

    /// Evidence fields from each channel (nullable).
    pub vector_similarity: Option<f32>,
    pub inference_model: Option<String>,
    pub inference_task: Option<String>,
    pub inference_confidence: Option<f32>,

    /// Original data columns (from source or inference output).
    pub data_columns: HashMap<String, ArrayRef>,
}
