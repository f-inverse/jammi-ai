//! Typed reads of `FixedSizeList<Float32>` vector columns from a result-table
//! Parquet object. Centralises the downcast-and-collect logic that both the
//! brute-force search path and downstream callers (e.g. resilience checks
//! that need the raw vectors) would otherwise duplicate.
//!
//! **Whose fault a shape mismatch is depends on who owns the object being
//! read, and this module has no way to know that** — [`extend_with_fixed_size_list_f32`]
//! and [`extend_with_keyed_fixed_size_list_f32`] take a bare `RecordBatch`,
//! never a marker for "this is a caller-supplied file" vs. "this is an
//! engine-owned result table". So their error, [`VectorColumnError`], is
//! provenance-NEUTRAL: it names the shape defect but picks no
//! [`crate::error::JammiError`] variant itself. `?` converts it to
//! [`JammiError::IncompatibleFormat`] by default (an engine-owned artifact's
//! corruption is never the caller's fault) — correct for every reader of a
//! result table's OWN parquet (the brute-force scan, `read_vectors`,
//! `read_vector_by_key`, the neighbor-graph and propagation node readers,
//! embedding refresh). The ONE genuinely caller-supplied path
//! ([`read_keyed_vectors_f32`], behind `import_embeddings`, reading a file
//! the caller handed the engine) explicitly reclassifies via
//! [`VectorColumnError::into_caller_fault`] instead of `?`.

use arrow::array::{Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use arrow::compute::cast;
use arrow_schema::DataType;

use crate::error::{JammiError, Result};
use crate::storage::{self, JammiObjectStore};

/// A `FixedSizeList<Float32>` vector (or its paired key) column disagreed
/// with what a reader expected — provenance-neutral (see the module doc):
/// this type carries no opinion on whose fault the disagreement is, only
/// what disagreed and how. [`From<VectorColumnError> for JammiError`]
/// supplies the DEFAULT (engine-owned artifact); [`Self::into_caller_fault`]
/// is the explicit override for the one path that needs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorColumnError {
    /// The table (or, for an import, the object) the column was read from.
    pub table: String,
    /// The column name.
    pub column: String,
    /// What was expected.
    pub expected: String,
    /// What was found.
    pub actual: String,
}

impl std::fmt::Display for VectorColumnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}.{}: expected {}, found {}",
            self.table, self.column, self.expected, self.actual
        )
    }
}

impl std::error::Error for VectorColumnError {}

impl VectorColumnError {
    /// Reclassify as the CALLER's fault — reserved for a reader whose
    /// `RecordBatch` came directly from an object the caller supplied (an
    /// import file), where a shape mismatch is genuinely a bad input, never
    /// the engine's own corruption.
    pub fn into_caller_fault(self) -> JammiError {
        JammiError::Schema {
            table: self.table,
            column: self.column,
            expected: self.expected,
            actual: self.actual,
        }
    }
}

/// The DEFAULT conversion: an engine-owned artifact's own shape disagreeing
/// with what a reader expected is never the caller's fault, mirroring
/// [`jammi_numerics::query::ValidatedQuery::require_width`]'s default to the
/// engine class. Every call site that reads a RESULT TABLE's own stored
/// parquet (the overwhelming majority of this module's callers) gets this
/// for free through `?`; only [`read_keyed_vectors_f32`] overrides it.
impl From<VectorColumnError> for JammiError {
    fn from(e: VectorColumnError) -> Self {
        JammiError::IncompatibleFormat {
            artifact: format!("{}.{}", e.table, e.column),
            found: e.actual,
            supported: e.expected,
        }
    }
}

/// Materialise a `FixedSizeList<Float32>` column from one `RecordBatch` into
/// `Vec<f32>` rows, appending them to `out`.
///
/// Returns a typed [`VectorColumnError`] when the column is missing, has the
/// wrong Arrow type, or has a non-`Float32` inner item. The `table` argument
/// is folded into the error so the caller does not need to wrap. See the
/// module doc for why this is provenance-neutral rather than a
/// [`JammiError`] directly.
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
) -> std::result::Result<(), VectorColumnError> {
    let col = batch
        .column_by_name(column)
        .ok_or_else(|| VectorColumnError {
            table: table.to_string(),
            column: column.to_string(),
            expected: "FixedSizeList<Float32>".to_string(),
            actual: "missing".to_string(),
        })?;
    let list = col
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| VectorColumnError {
            table: table.to_string(),
            column: column.to_string(),
            expected: "FixedSizeList<Float32>".to_string(),
            actual: format!("{:?}", col.data_type()),
        })?;
    if !matches!(list.value_type(), DataType::Float32) {
        return Err(VectorColumnError {
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
                .ok_or_else(|| VectorColumnError {
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
/// Returns a typed [`VectorColumnError`] when either column is missing, the
/// key column's `DataType` is not a member of the Utf8 family, or the vector
/// column has the wrong Arrow type (a non-`Float32` vector), so a caller
/// reading precomputed vectors sees a typed signal rather than a downcast
/// panic (or, for a numeric/temporal/boolean key column, a silently
/// stringified success). The vector leg delegates to
/// [`extend_with_fixed_size_list_f32`] so the downcast rules stay defined in
/// exactly one place; this pairs each resulting vector with its key by
/// position. See the module doc for why this is provenance-neutral rather
/// than a [`JammiError`] directly.
pub fn extend_with_keyed_fixed_size_list_f32(
    batch: &RecordBatch,
    table: &str,
    key_column: &str,
    vector_column: &str,
    out: &mut Vec<(String, Vec<f32>)>,
) -> std::result::Result<(), VectorColumnError> {
    let key_col = batch
        .column_by_name(key_column)
        .ok_or_else(|| VectorColumnError {
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
    // value across encodings; everything else keeps the typed error this
    // function's contract promises.
    if !is_utf8_family(key_col.data_type()) {
        return Err(VectorColumnError {
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
    let keys_utf8 = cast(key_col, &DataType::Utf8).map_err(|_| VectorColumnError {
        table: table.to_string(),
        column: key_column.to_string(),
        expected: "Utf8".to_string(),
        actual: format!("{:?}", key_col.data_type()),
    })?;
    let keys = keys_utf8
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| VectorColumnError {
            table: table.to_string(),
            column: key_column.to_string(),
            expected: "Utf8".to_string(),
            actual: format!("{:?}", key_col.data_type()),
        })?;

    let mut vectors = Vec::with_capacity(keys.len());
    extend_with_fixed_size_list_f32(batch, table, vector_column, &mut vectors)?;
    if vectors.len() != keys.len() {
        return Err(VectorColumnError {
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
/// memory. This is the ONE call site in this module that reads an object the
/// CALLER supplied directly (the file behind `import_embeddings`), so — unlike
/// every other reader here, which reads back the engine's own stored parquet
/// — a shape mismatch really is the caller's fault: explicitly reclassified
/// via [`VectorColumnError::into_caller_fault`] rather than the `?`-conversion
/// default every other caller in this module gets.
pub async fn read_keyed_vectors_f32(
    handle: &JammiObjectStore,
    table: &str,
    key_column: &str,
    vector_column: &str,
) -> Result<Vec<(String, Vec<f32>)>> {
    let batches = storage::reader::read_all_record_batches(handle).await?;
    let mut out = Vec::new();
    for batch in batches {
        extend_with_keyed_fixed_size_list_f32(&batch, table, key_column, vector_column, &mut out)
            .map_err(VectorColumnError::into_caller_fault)?;
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
    /// surfaces the typed [`VectorColumnError`] signal rather than a
    /// downcast panic — the cast-then-downcast normalisation widens which
    /// encodings succeed, it does not weaken the error path for genuinely
    /// wrong types.
    #[test]
    fn non_string_key_column_still_surfaces_typed_error() {
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
        assert_eq!(err.table, "t");
        assert_eq!(err.column, "key");
    }

    /// The hazard the previous `FixedSizeList` case (above) didn't actually
    /// exercise: `arrow::compute::cast` *does* stringify an `Int64` column
    /// into `Utf8` without error (unlike `FixedSizeList`, which is not
    /// `is_primitive()` under Arrow's cast-compatibility rules and always
    /// fails the cast). A cast-then-downcast implementation that skips the
    /// pre-cast `DataType` gate would silently turn an `Int64` key column
    /// into plausible-looking decimal strings (`"1"`, `"2"`, `"3"`) instead
    /// of raising the typed [`VectorColumnError`] this function's contract
    /// promises for a wrongly-typed key column. Asserts the typed error, not
    /// a stringified success — the numeric-key oracle the prior
    /// `non_string_key_column_still_surfaces_typed_error` test named
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
        assert_eq!(err.table, "t");
        assert_eq!(err.column, "key");
        assert_eq!(err.expected, "Utf8");
        assert_eq!(err.actual, "Int64");
        assert!(
            out.is_empty(),
            "no rows should be extracted on a rejected key column"
        );
    }

    /// The provenance-neutral [`VectorColumnError`] converts to the ENGINE
    /// class by default through `?` (an engine-owned artifact's own
    /// corruption is never the caller's fault) — the same default
    /// `ValidatedQuery::require_width` uses.
    #[test]
    fn default_conversion_is_the_engine_class() {
        let e = VectorColumnError {
            table: "docs".into(),
            column: "vector".into(),
            expected: "FixedSizeList<Float32>".into(),
            actual: "Utf8".into(),
        };
        let engine: JammiError = e.into();
        assert!(
            matches!(
                &engine,
                JammiError::IncompatibleFormat { artifact, .. } if artifact == "docs.vector"
            ),
            "{engine:?}"
        );
    }

    /// [`VectorColumnError::into_caller_fault`] is the explicit override
    /// reserved for the one genuinely caller-supplied path (import).
    #[test]
    fn into_caller_fault_is_the_caller_class() {
        let e = VectorColumnError {
            table: "import".into(),
            column: "_row_id".into(),
            expected: "Utf8".into(),
            actual: "Int64".into(),
        };
        assert!(matches!(
            e.into_caller_fault(),
            JammiError::Schema { table, column, .. } if table == "import" && column == "_row_id"
        ));
    }
}
