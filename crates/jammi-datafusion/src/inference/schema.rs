use std::sync::Arc;

use crate::error::{Error, Result};
use arrow::array::{ArrayRef, Float32Array, StringArray};
use arrow::compute;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

use crate::inference::adapter;
use crate::task::ModelTask;

/// The name of the row-order column: assigned by
/// [`NumberedInputExec`](crate::inference::numbered::NumberedInputExec)
/// below every model, read by `InferenceExec` as a required input column, and
/// carried to the output as a prefix column.
pub const ORDINAL_COLUMN: &str = "_ordinal";

/// Common prefix columns on every inference output.
///
/// `_ordinal` is the row's position in the input's order — key order for a
/// keyed input, arrival order otherwise — 0-based and contiguous over the
/// whole input. `InferenceExec` reads it as a required input column and
/// carries it through unchanged, so it is one global sequence at every
/// partition count, and the plan above the model puts its output back in
/// that order.
///
/// It exists so a caller can read the result table back in the input's
/// order (`ORDER BY _row_id, _ordinal`) even when `_row_id` carries
/// duplicate or non-monotonic keys, and even when the underlying Parquet
/// scan reorders row groups on read. A consumer's own table need not keep
/// it — an embedding table keyed by `_row_id` alone reads back by key —
/// but every batch the operators emit carries it.
pub fn common_prefix_fields() -> Vec<Field> {
    vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new(ORDINAL_COLUMN, DataType::UInt64, false),
        Field::new("_source", DataType::Utf8, false),
        Field::new("_model", DataType::Utf8, false),
        Field::new("_status", DataType::Utf8, false),
        Field::new("_error", DataType::Utf8, true),
        Field::new("_latency_ms", DataType::Float32, false),
    ]
}

/// Build the full output schema: prefix + task-specific.
///
/// For embedding tasks, `embedding_dim` must be the model's actual hidden size.
/// For regression tasks, `regression_form` must be the served head's persisted
/// [`DistributionForm`](adapter::DistributionForm) so the planned schema matches
/// the runtime adapter's columns (a quantile head's schema is its level
/// columns, not the Gaussian default of `mean`/`std`). `None` form ⇒ Gaussian.
///
/// `passthrough` names input columns copied verbatim to the END of every
/// output batch (after the task columns), each keeping its input field
/// (type and nullability). The embedding pipeline passes `["_content_hash"]`
/// so the hash the source scan computed lands beside the vector; `infer`
/// passes nothing (its output schema is fixed). A name absent from the input
/// is a typed refusal here, at plan build, never a runtime column miss.
pub fn build_output_schema(
    task: &ModelTask,
    input_schema: &SchemaRef,
    _key_column: &str,
    embedding_dim: Option<usize>,
    regression_form: Option<&adapter::DistributionForm>,
    passthrough: &[String],
) -> Result<SchemaRef> {
    let mut fields = common_prefix_fields();
    let task_adapter = adapter::create_adapter_for_schema(*task, embedding_dim, regression_form);
    fields.extend(task_adapter.output_schema());
    for name in passthrough {
        let field = input_schema.field_with_name(name).map_err(|_| {
            Error::Inference(format!(
                "passthrough column '{name}' is not in the inference input schema"
            ))
        })?;
        fields.push(field.clone());
    }
    Ok(Arc::new(Schema::new(fields)))
}

/// Build common prefix arrays for an output batch.
///
/// `row_status` and `row_errors` are two independently-sized fields on
/// `BackendOutput`. Building `_status` off `row_status.len()` while `_error`
/// and every other prefix column is built off `row_count` would return
/// columns of mismatched length whenever a producer's two fields disagree
/// with the batch's row count — a shape bug that only
/// `RecordBatch::try_new`'s generic length-mismatch error would ever catch,
/// far from the producer that emitted the disagreeing lengths. This is the
/// ONE policy applied at every reader of a possibly-short `row_status` (and,
/// where read, `row_errors`): refuse by name the moment a field's length
/// disagrees with the row count, rather than defaulting a missing entry to
/// some policy-specific value. Every reader carries it independently —
/// this function; [`DistributionAdapter::adapt`](super::adapter::DistributionAdapter)
/// and [`EmbeddingAdapter::adapt`](super::adapter::EmbeddingAdapter), which
/// also check `row_errors`; and the `nullify_strings`/`nullify_floats`
/// helpers shared by `ClassificationAdapter` and `NerAdapter`, which check
/// `values`/`row_status` against the `row_count` each adapter passes them.
/// None of these readers depends on another reader running first: the
/// runner building the prefix columns before calling the task adapter is
/// call ordering, not a safety dependency.
///
/// `ordinals` is this batch's `_ordinal` column (one `UInt64` entry per row,
/// non-null), read off the input — see [`common_prefix_fields`]; this
/// function only places it and checks its length agrees with every other
/// prefix column.
#[allow(clippy::too_many_arguments)]
pub fn build_prefix_columns(
    keys: &ArrayRef,
    key_column: &str,
    source_id: &str,
    model_id: &str,
    row_status: &[bool],
    row_errors: &[String],
    latency_ms: f32,
    row_count: usize,
    ordinals: &ArrayRef,
) -> Result<Vec<ArrayRef>> {
    if row_status.len() != row_count {
        return Err(Error::Inference(format!(
            "build_prefix_columns: row_status has {} entries, expected one per row ({row_count})",
            row_status.len()
        )));
    }
    if row_errors.len() != row_count {
        return Err(Error::Inference(format!(
            "build_prefix_columns: row_errors has {} entries, expected one per row ({row_count})",
            row_errors.len()
        )));
    }
    if ordinals.len() != row_count {
        return Err(Error::Inference(format!(
            "build_prefix_columns: ordinals has {} entries, expected one per row ({row_count})",
            ordinals.len()
        )));
    }

    let status_strs: Vec<&str> = row_status
        .iter()
        .map(|&ok| if ok { "ok" } else { "error" })
        .collect();
    let status = StringArray::from(status_strs);

    // Lengths are now pinned equal to `row_count` by the checks above, so a
    // plain index (never `.get(i)`) is safe here.
    let errors: StringArray = row_errors
        .iter()
        .enumerate()
        .map(|(i, e)| {
            if row_status[i] {
                None
            } else {
                Some(e.as_str())
            }
        })
        .collect();

    // Cast keys to Utf8 if needed (key column may be Int64, etc.). A key
    // type that cannot render as Utf8 (a Struct/Map, whose Arrow cast
    // kernel has no Utf8 arm) is a typed, named refusal — never a silent
    // `unwrap_or_else(|_| Arc::clone(keys))` fallback, which would pass the
    // RAW non-Utf8 array through as `_row_id` and let a downstream schema-shape mismatch
    // misattribute the failure to whatever column happened to trip over the wrong type first (see
    // the unit oracle `build_prefix_columns_refuses_a_key_that_cannot_cast_to_utf8`
    // below, and the end-to-end oracle
    // `crates/jammi-ai/tests/it/partitioned_inference.rs`'s
    // `struct_key_through_annotate_is_a_typed_refusal_naming_the_key`).
    let row_ids: ArrayRef = if keys.data_type() == &DataType::Utf8 {
        Arc::clone(keys)
    } else {
        compute::cast(keys, &DataType::Utf8).map_err(|e| {
            Error::Inference(format!(
                "key column '{key_column}' (type {:?}) cannot be cast to Utf8: {e}",
                keys.data_type()
            ))
        })?
    };

    Ok(vec![
        row_ids,                                                               // _row_id
        Arc::clone(ordinals),                                                  // _ordinal
        Arc::new(StringArray::from(vec![source_id; row_count])) as ArrayRef,   // _source
        Arc::new(StringArray::from(vec![model_id; row_count])) as ArrayRef,    // _model
        Arc::new(status) as ArrayRef,                                          // _status
        Arc::new(errors) as ArrayRef,                                          // _error
        Arc::new(Float32Array::from(vec![latency_ms; row_count])) as ArrayRef, // _latency_ms
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, UInt64Array};

    /// `row_status` shorter than `row_count` (with `row_errors` matching
    /// `row_count`, as `BackendOutput`'s two independently-sized fields can
    /// disagree) must be a named refusal, never a `_status` column built off
    /// `row_status.len()` while `_error` and the rest of the prefix are built
    /// off `row_count` — that mismatch is a shape bug only
    /// `RecordBatch::try_new`'s generic error would ever catch, far from the
    /// producer that emitted the disagreeing length. Without the length
    /// checks (a `row_status.get(i)` default-to-"not ok" policy never
    /// refuses), the result is `Ok` with a `_status` column of length 1 next
    /// to `_error`/`_row_id` columns of length 3 — a mutually-inconsistent
    /// batch — instead of the `Err` asserted below.
    fn ordinals_for(row_count: usize, start: u64) -> ArrayRef {
        Arc::new((start..start + row_count as u64).collect::<UInt64Array>())
    }

    #[test]
    fn build_prefix_columns_refuses_a_row_status_shorter_than_row_count() {
        let keys: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let row_status = vec![true]; // one entry; row_count is 3
        let row_errors = vec![
            String::new(),
            "row 1 failed".to_string(),
            "row 2 failed".to_string(),
        ];
        let ords = ordinals_for(3, 0);
        let err = build_prefix_columns(
            &keys,
            "id",
            "src",
            "model",
            &row_status,
            &row_errors,
            1.0,
            3,
            &ords,
        )
        .expect_err("a row_status shorter than row_count must be a typed refusal");
        let msg = err.to_string();
        assert!(msg.contains("row_status"), "must name the field: {msg}");
        assert!(
            msg.contains('1') && msg.contains('3'),
            "must name both the got (1) and expected (3) lengths: {msg}"
        );
    }

    /// The peer of the above: `row_errors` shorter than `row_count` (with
    /// `row_status` matching `row_count`) must be refused the same way.
    /// Without the length checks, the result is `Ok` with an `_error` column
    /// built by iterating the short `row_errors` — shorter than the rest of
    /// the prefix — instead of the `Err` asserted below.
    #[test]
    fn build_prefix_columns_refuses_a_row_errors_shorter_than_row_count() {
        let keys: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let row_status = vec![true, false, true];
        let row_errors = vec!["row 1 failed".to_string()]; // one entry; row_count is 3
        let ords = ordinals_for(3, 0);
        let err = build_prefix_columns(
            &keys,
            "id",
            "src",
            "model",
            &row_status,
            &row_errors,
            1.0,
            3,
            &ords,
        )
        .expect_err("a row_errors shorter than row_count must be a typed refusal");
        let msg = err.to_string();
        assert!(msg.contains("row_errors"), "must name the field: {msg}");
        assert!(
            msg.contains('1') && msg.contains('3'),
            "must name both the got (1) and expected (3) lengths: {msg}"
        );
    }

    /// A `row_status`/`row_errors` pair that both agree with `row_count`
    /// must still return mutually consistent column lengths, and the
    /// `_error` column must still route through `row_status` per row (not
    /// treat "ok" rows as errored).
    #[test]
    fn build_prefix_columns_returns_consistent_lengths_when_fields_agree() {
        let keys: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let row_status = vec![true, false, true];
        let row_errors = vec![String::new(), "row 1 failed".to_string(), String::new()];
        let ords = ordinals_for(3, 0);
        let cols = build_prefix_columns(
            &keys,
            "id",
            "src",
            "model",
            &row_status,
            &row_errors,
            1.0,
            3,
            &ords,
        )
        .expect("mutually consistent lengths must not be refused");
        for (i, col) in cols.iter().enumerate() {
            assert_eq!(col.len(), 3, "column {i} must have one entry per row");
        }
        // cols: [_row_id, _ordinal, _source, _model, _status, _error, _latency_ms]
        let errors = cols[5].as_any().downcast_ref::<StringArray>().unwrap();
        assert!(errors.is_null(0), "row 0's own recorded status was ok");
        assert_eq!(errors.value(1), "row 1 failed");
        assert!(errors.is_null(2), "row 2's own recorded status was ok");
    }

    /// `build_prefix_columns` places the CALLER'S `ordinals` array verbatim
    /// at `_ordinal` — it never recomputes it.
    #[test]
    fn build_prefix_columns_places_the_given_ordinals_verbatim() {
        let keys: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let row_status = vec![true, true, true];
        let row_errors = vec![String::new(), String::new(), String::new()];
        let ords = ordinals_for(3, 7);
        let cols = build_prefix_columns(
            &keys,
            "id",
            "src",
            "model",
            &row_status,
            &row_errors,
            1.0,
            3,
            &ords,
        )
        .expect("mutually consistent lengths must not be refused");
        let ordinals = cols[1].as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(ordinals.values(), &[7u64, 8, 9]);
    }

    /// A key type the Arrow cast kernel cannot render as Utf8 (a Struct) is
    /// a typed `Error::Inference` naming the key COLUMN, its TYPE, and
    /// "cannot be cast to Utf8" — never a silent `unwrap_or_else(|_|
    /// Arc::clone(keys))` fallback, which would pass the raw Struct array
    /// through as `_row_id` (`Ok` with a Struct-typed `_row_id` column) and
    /// let `RecordBatch::try_new` fail later with a schema-shape error that
    /// names the wrong column.
    #[test]
    fn build_prefix_columns_refuses_a_key_that_cannot_cast_to_utf8() {
        use arrow::array::{Int32Array, StructArray};
        use arrow::datatypes::Fields;

        let fields: Fields = vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, false),
        ]
        .into();
        let keys: ArrayRef = Arc::new(StructArray::new(
            fields,
            vec![
                Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
                Arc::new(StringArray::from(vec!["x", "y"])) as ArrayRef,
            ],
            None,
        ));
        let row_status = vec![true, true];
        let row_errors = vec![String::new(), String::new()];
        let ords = ordinals_for(2, 0);
        let err = build_prefix_columns(
            &keys,
            "struct_key",
            "src",
            "model",
            &row_status,
            &row_errors,
            1.0,
            2,
            &ords,
        )
        .expect_err("a key type with no Utf8 cast kernel must be a typed refusal");
        let msg = err.to_string();
        assert!(
            msg.contains("struct_key"),
            "must name the key column: {msg}"
        );
        assert!(
            msg.contains("cannot be cast to Utf8"),
            "must state the failure: {msg}"
        );
    }

    /// An `ordinals` array shorter than `row_count` must be refused the same
    /// way as `row_status`/`row_errors` — the same class of shape bug.
    #[test]
    fn build_prefix_columns_refuses_ordinals_shorter_than_row_count() {
        let keys: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let row_status = vec![true, true, true];
        let row_errors = vec![String::new(), String::new(), String::new()];
        let ords = ordinals_for(1, 0); // one entry; row_count is 3
        let err = build_prefix_columns(
            &keys,
            "id",
            "src",
            "model",
            &row_status,
            &row_errors,
            1.0,
            3,
            &ords,
        )
        .expect_err("ordinals shorter than row_count must be a typed refusal");
        let msg = err.to_string();
        assert!(msg.contains("ordinals"), "must name the field: {msg}");
    }
}
