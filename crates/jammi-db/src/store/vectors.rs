//! Typed reads of `FixedSizeList<Float32>` vector columns from a result-table
//! Parquet object. Centralises the downcast-and-collect logic that both the
//! brute-force search path and downstream callers (e.g. resilience checks
//! that need the raw vectors) would otherwise duplicate.

use arrow::array::{Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use arrow::compute::cast;
use arrow_schema::DataType;

use crate::error::{JammiError, Result};
use crate::storage::{self, JammiObjectStore};

/// Materialise a `FixedSizeList<Float32>` column from one `RecordBatch` into
/// `Vec<f32>` rows, appending them to `out`.
///
/// Returns a typed [`JammiError::Schema`] when the column is missing, has the
/// wrong Arrow type, or has a non-`Float32` inner item. The `table` argument
/// is folded into the error so the caller does not need to wrap.
///
/// Hidden invariant: this helper is the only place in the engine that should
/// downcast a vector column to `FixedSizeListArray<Float32>`. The brute-force
/// ANN scan, the typed-read API, and the neighbor-graph node reader all call
/// through here rather than re-implementing the downcast.
pub fn extend_with_fixed_size_list_f32(
    batch: &RecordBatch,
    table: &str,
    column: &str,
    out: &mut Vec<Vec<f32>>,
) -> Result<()> {
    let col = batch
        .column_by_name(column)
        .ok_or_else(|| JammiError::Schema {
            table: table.to_string(),
            column: column.to_string(),
            expected: "FixedSizeList<Float32>".to_string(),
            actual: "missing".to_string(),
        })?;
    let list = col
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| JammiError::Schema {
            table: table.to_string(),
            column: column.to_string(),
            expected: "FixedSizeList<Float32>".to_string(),
            actual: format!("{:?}", col.data_type()),
        })?;
    if !matches!(list.value_type(), DataType::Float32) {
        return Err(JammiError::Schema {
            table: table.to_string(),
            column: column.to_string(),
            expected: "FixedSizeList<Float32>".to_string(),
            actual: format!("FixedSizeList<{:?}>", list.value_type()),
        });
    }
    let dim = list.value_length() as usize;
    for row in 0..list.len() {
        let v = list.value(row);
        let floats =
            v.as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| JammiError::Schema {
                    table: table.to_string(),
                    column: column.to_string(),
                    expected: "FixedSizeList<Float32>".to_string(),
                    actual: format!("FixedSizeList<{:?}>", v.data_type()),
                })?;
        let mut row_vec = Vec::with_capacity(dim);
        for i in 0..dim {
            row_vec.push(floats.value(i));
        }
        out.push(row_vec);
    }
    Ok(())
}

/// True when `dt` is a member of the Utf8 family this module normalises
/// through [`cast`] before the key-column downcast: `Utf8`, `LargeUtf8`,
/// `Utf8View`, or a `Dictionary` whose *value* type is itself a member of the
/// family (recursively, so a dictionary-of-dictionary-of-Utf8 still counts).
///
/// This is the validate-before-cast gate: [`arrow::compute::cast`] happily
/// stringifies far more than the Utf8 family — an `Int64`, `Float64`,
/// `Boolean`, or `Timestamp` key column all pass `cast(_, &DataType::Utf8)`
/// without error, silently turning a wrongly-typed key column into a
/// plausible-looking string instead of surfacing the typed
/// [`JammiError::Schema`] the caller's whole-table read contract promises.
/// Restricting the pre-cast domain to exactly the Utf8 family closes that
/// hole while still absorbing the wire-encoding variance (`Utf8` vs.
/// `Utf8View` vs. `LargeUtf8`, and dictionary-encoded variants of each) the
/// cast step exists for.
fn is_utf8_family(dt: &DataType) -> bool {
    match dt {
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => true,
        DataType::Dictionary(_, value) => is_utf8_family(value),
        _ => false,
    }
}

/// Materialise the paired `(key, vector)` rows of one `RecordBatch` — a string
/// `key_column` alongside a `FixedSizeList<Float32>` `vector_column` — appending
/// each `(String, Vec<f32>)` to `out` in row order.
///
/// The key column's *logical* string value is what matters here, not its wire
/// encoding: a table read straight off local Parquet surfaces `Utf8`, but the
/// same column read back through a Flight SQL round-trip (or any other path
/// through DataFusion's default `schema_force_view_types`) surfaces
/// `Utf8View` (and a wide table can surface `LargeUtf8`, or a dictionary-
/// encoded variant of any of the three). Rather than hard-require `Utf8` and
/// panic-by-proxy (a typed [`JammiError::Schema`]) on the others, the column's
/// `DataType` is validated against the Utf8 family (see `is_utf8_family`)
/// *before* casting to `Utf8` — mirroring
/// [`crate::index::exact::exact_vector_search`]'s `_row_id` handling — so a
/// single `StringArray` downcast covers every Utf8 family the column can
/// arrive as, while a non-Utf8-family column (e.g. `Int64`, `Float64`,
/// `Boolean`, `Timestamp`) never reaches the cast at all: `cast` would
/// otherwise silently stringify it instead of raising a typed error, since
/// `arrow::compute::cast` casts far more into `Utf8` than the Utf8 family
/// alone. Equal logical string values therefore extract identically
/// regardless of which admitted encoding produced them.
///
/// Returns a typed [`JammiError::Schema`] when either column is missing, the
/// key column's `DataType` is not a member of the Utf8 family, or the vector
/// column has the wrong Arrow type (a non-`Float32` vector), so a caller
/// reading precomputed vectors sees a typed signal rather than a downcast
/// panic (or, for a numeric/temporal/boolean key column, a silently
/// stringified success). The vector leg delegates to
/// [`extend_with_fixed_size_list_f32`] so the downcast rules stay defined in
/// exactly one place; this pairs each resulting vector with its key by
/// position.
pub fn extend_with_keyed_fixed_size_list_f32(
    batch: &RecordBatch,
    table: &str,
    key_column: &str,
    vector_column: &str,
    out: &mut Vec<(String, Vec<f32>)>,
) -> Result<()> {
    let key_col = batch
        .column_by_name(key_column)
        .ok_or_else(|| JammiError::Schema {
            table: table.to_string(),
            column: key_column.to_string(),
            expected: "Utf8".to_string(),
            actual: "missing".to_string(),
        })?;
    // Validate the DataType against the Utf8 family *before* casting:
    // `arrow::compute::cast` is cast-compatibility-permissive (it will
    // stringify an Int64, Float64, Boolean, or Timestamp column into a
    // plausible-looking `Utf8` array without error), so admitting anything
    // Arrow can cast to `Utf8` would silently widen this column's accepted
    // domain past "logically a string". Only `Utf8`, `LargeUtf8`,
    // `Utf8View`, and Utf8-family dictionaries carry the same logical string
    // value across encodings; everything else keeps the typed
    // `JammiError::Schema` this function's contract promises.
    if !is_utf8_family(key_col.data_type()) {
        return Err(JammiError::Schema {
            table: table.to_string(),
            column: key_column.to_string(),
            expected: "Utf8".to_string(),
            actual: format!("{:?}", key_col.data_type()),
        });
    }
    // Cast rather than hard-downcast: `Utf8`, `Utf8View`, and `LargeUtf8` all
    // carry the same logical string, so normalise the encoding here instead
    // of forcing every caller to know which one a given read path produces.
    // The DataType gate above has already ruled out every non-Utf8-family
    // input, so this cast only ever normalises within the admitted domain.
    let keys_utf8 = cast(key_col, &DataType::Utf8).map_err(|_| JammiError::Schema {
        table: table.to_string(),
        column: key_column.to_string(),
        expected: "Utf8".to_string(),
        actual: format!("{:?}", key_col.data_type()),
    })?;
    let keys = keys_utf8
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| JammiError::Schema {
            table: table.to_string(),
            column: key_column.to_string(),
            expected: "Utf8".to_string(),
            actual: format!("{:?}", key_col.data_type()),
        })?;

    let mut vectors = Vec::with_capacity(keys.len());
    extend_with_fixed_size_list_f32(batch, table, vector_column, &mut vectors)?;
    if vectors.len() != keys.len() {
        return Err(JammiError::Schema {
            table: table.to_string(),
            column: vector_column.to_string(),
            expected: format!("{} vectors (one per key)", keys.len()),
            actual: format!("{} vectors", vectors.len()),
        });
    }

    for (row, vector) in vectors.into_iter().enumerate() {
        out.push((keys.value(row).to_string(), vector));
    }
    Ok(())
}

/// Read every `(key, vector)` pair — a string (`Utf8`/`Utf8View`/`LargeUtf8`)
/// `key_column` and a `FixedSizeList<Float32>` `vector_column` — from the
/// Parquet object behind `handle`, in file order.
///
/// The read path behind importing precomputed vectors: streams batches through
/// the engine's `storage::reader` and delegates each to
/// [`extend_with_keyed_fixed_size_list_f32`]. Reads the whole object into
/// memory.
pub async fn read_keyed_vectors_f32(
    handle: &JammiObjectStore,
    table: &str,
    key_column: &str,
    vector_column: &str,
) -> Result<Vec<(String, Vec<f32>)>> {
    let batches = storage::reader::read_all_record_batches(handle).await?;
    let mut out = Vec::new();
    for batch in batches {
        extend_with_keyed_fixed_size_list_f32(&batch, table, key_column, vector_column, &mut out)?;
    }
    Ok(out)
}

/// Read every value of a `FixedSizeList<Float32>` column from the Parquet
/// object behind `handle`, returning one `Vec<f32>` per row.
///
/// Streams batches through the engine's `storage::reader` and delegates each
/// to [`extend_with_fixed_size_list_f32`].
pub(crate) async fn read_fixed_size_list_f32_column(
    handle: &JammiObjectStore,
    table: &str,
    column: &str,
) -> Result<Vec<Vec<f32>>> {
    let batches = storage::reader::read_all_record_batches(handle).await?;
    let mut out = Vec::new();
    for batch in batches {
        extend_with_fixed_size_list_f32(&batch, table, column, &mut out)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{ArrayRef, Int64Array, StringViewArray};
    use arrow_schema::{Field, Schema};

    use super::*;

    /// Build a `FixedSizeList<Float32>` of the given inner length from a flat
    /// `Vec<Vec<f32>>`, mirroring `read_vectors.rs`'s IT-suite fixture.
    fn fixed_size_list_from(rows: &[Vec<f32>], dim: i32) -> FixedSizeListArray {
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let values = Arc::new(Float32Array::from(flat));
        let field = Arc::new(Field::new("item", DataType::Float32, false));
        FixedSizeListArray::try_new(field, dim, values, None).unwrap()
    }

    fn rows() -> Vec<Vec<f32>> {
        vec![
            vec![0.1, 0.2, 0.3],
            vec![-1.0, 0.0, 1.0],
            vec![2.5, 3.5, -4.5],
        ]
    }

    fn keys() -> Vec<&'static str> {
        vec!["row-a", "row-b", "row-c"]
    }

    /// A `Utf8` key column extracts as the plain `(key, vector)` pairs — the
    /// case every existing caller (local Parquet reads) exercises.
    #[test]
    fn extracts_utf8_key_column() {
        let dim = 3_i32;
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    dim,
                ),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(keys())) as ArrayRef,
                Arc::new(fixed_size_list_from(&rows(), dim)),
            ],
        )
        .unwrap();

        let mut out = Vec::new();
        extend_with_keyed_fixed_size_list_f32(&batch, "t", "key", "vector", &mut out).unwrap();

        let expected: Vec<(String, Vec<f32>)> =
            keys().into_iter().map(str::to_string).zip(rows()).collect();
        assert_eq!(out, expected);
    }

    /// The same key values, materialised as `Utf8View` (`StringViewArray`) —
    /// the encoding a Flight SQL round-trip / DataFusion's default
    /// `schema_force_view_types` parquet reader surfaces — extract to the
    /// IDENTICAL `(key, vector)` pairs as the `Utf8` case above. Regression
    /// coverage for the K4 Utf8View helper flag: prior to this fix, this case
    /// hit `extend_with_keyed_fixed_size_list_f32`'s hard `Utf8` downcast and
    /// returned a `JammiError::Schema` instead of the logically-identical
    /// rows.
    #[test]
    fn extracts_utf8view_key_column_identically_to_utf8() {
        let dim = 3_i32;
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8View, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    dim,
                ),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringViewArray::from(keys())) as ArrayRef,
                Arc::new(fixed_size_list_from(&rows(), dim)),
            ],
        )
        .unwrap();

        let mut out = Vec::new();
        extend_with_keyed_fixed_size_list_f32(&batch, "t", "key", "vector", &mut out).unwrap();

        let expected: Vec<(String, Vec<f32>)> =
            keys().into_iter().map(str::to_string).zip(rows()).collect();
        assert_eq!(out, expected);
    }

    /// A key column that cannot be cast to a string at all (e.g. a
    /// `FixedSizeList<Float32>` — not `from_type.is_primitive()` under
    /// Arrow's cast-compatibility rules, unlike an integer or float column,
    /// which numeric-to-string casting *would* silently stringify) still
    /// surfaces the typed [`JammiError::Schema`] signal rather than a
    /// downcast panic — the cast-then-downcast normalisation widens which
    /// encodings succeed, it does not weaken the error path for genuinely
    /// wrong types.
    #[test]
    fn non_string_key_column_still_surfaces_typed_schema_error() {
        let dim = 3_i32;
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "key",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    dim,
                ),
                false,
            ),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    dim,
                ),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(fixed_size_list_from(&rows(), dim)) as ArrayRef,
                Arc::new(fixed_size_list_from(&rows(), dim)),
            ],
        )
        .unwrap();

        let mut out = Vec::new();
        let err = extend_with_keyed_fixed_size_list_f32(&batch, "t", "key", "vector", &mut out)
            .unwrap_err();
        match err {
            JammiError::Schema { table, column, .. } => {
                assert_eq!(table, "t");
                assert_eq!(column, "key");
            }
            other => panic!("expected JammiError::Schema, got {other:?}"),
        }
    }

    /// The hazard the previous `FixedSizeList` case (above) didn't actually
    /// exercise: `arrow::compute::cast` *does* stringify an `Int64` column
    /// into `Utf8` without error (unlike `FixedSizeList`, which is not
    /// `is_primitive()` under Arrow's cast-compatibility rules and always
    /// fails the cast). A cast-then-downcast implementation that skips the
    /// pre-cast `DataType` gate would silently turn an `Int64` key column
    /// into plausible-looking decimal strings (`"1"`, `"2"`, `"3"`) instead
    /// of raising the typed [`JammiError::Schema`] this function's contract
    /// promises for a wrongly-typed key column. Asserts the typed error, not
    /// a stringified success — the numeric-key oracle the prior
    /// `non_string_key_column_still_surfaces_typed_schema_error` test named
    /// as a hazard but did not cover.
    #[test]
    fn int64_key_column_is_rejected_not_stringified() {
        let dim = 3_i32;
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int64, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    dim,
                ),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![1_i64, 2, 3])) as ArrayRef,
                Arc::new(fixed_size_list_from(&rows(), dim)),
            ],
        )
        .unwrap();

        // Sanity check the premise: Arrow's cast *would* silently succeed
        // here (Int64 is_primitive() casts cleanly to Utf8), which is
        // exactly why the pre-cast DataType gate, not the cast's own
        // fallibility, is what has to reject this column.
        assert!(cast(batch.column_by_name("key").unwrap(), &DataType::Utf8).is_ok());

        let mut out = Vec::new();
        let err = extend_with_keyed_fixed_size_list_f32(&batch, "t", "key", "vector", &mut out)
            .unwrap_err();
        match err {
            JammiError::Schema {
                table,
                column,
                expected,
                actual,
            } => {
                assert_eq!(table, "t");
                assert_eq!(column, "key");
                assert_eq!(expected, "Utf8");
                assert_eq!(actual, "Int64");
            }
            other => panic!("expected JammiError::Schema, got {other:?}"),
        }
        assert!(
            out.is_empty(),
            "no rows should be extracted on a rejected key column"
        );
    }
}
