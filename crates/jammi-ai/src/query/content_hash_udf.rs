//! `jammi_content_hash(col1, col2, …)` — the variadic scalar UDF every source
//! scan the embedding pipeline, `infer` and an incremental refresh build
//! carries as its `_content_hash` projection.
//!
//! The hash is computed over the RAW source columns rendered exactly as the
//! inference runner renders them ([`crate::inference::render_content_column`]:
//! string families pass through, binary families pass through as bytes, every
//! other type is cast to `Utf8` with the runner's null-introduction refusal),
//! then folded by [`jammi_db::store::content_hash::content_hash_columns`]. One
//! rendering by construction: a SQL-side `CAST` with a coarser rendering would
//! hash distinct values equal and let a refresh serve a STALE vector as
//! "unchanged". A rendering refusal is a typed [`jammi_db::error::JammiError`]
//! carried as `DataFusionError::External`, which the structural classifier
//! (`impl From<DataFusionError> for JammiError`) restores for every caller.

use std::sync::Arc;

use arrow::datatypes::DataType;
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use datafusion::prelude::SessionContext;
use jammi_db::store::content_hash::content_hash_columns;

/// The SQL name of the UDF.
pub const CONTENT_HASH_UDF_NAME: &str = "jammi_content_hash";

/// The UDF: variadic over any column types, returns a nullable `Utf8` hex
/// hash per row (never null itself — a null cell hashes as the `n` tag).
#[derive(Debug)]
pub struct ContentHashUdf {
    signature: Signature,
}

impl Default for ContentHashUdf {
    fn default() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl PartialEq for ContentHashUdf {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for ContentHashUdf {}

impl std::hash::Hash for ContentHashUdf {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        CONTENT_HASH_UDF_NAME.hash(state);
    }
}

impl ScalarUDFImpl for ContentHashUdf {
    fn name(&self) -> &str {
        CONTENT_HASH_UDF_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> DfResult<DataType> {
        if arg_types.is_empty() {
            return Err(DataFusionError::Plan(format!(
                "{CONTENT_HASH_UDF_NAME} needs at least one content column"
            )));
        }
        Ok(DataType::Utf8)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DfResult<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        let rendered = arrays
            .iter()
            .map(crate::inference::render_content_column)
            .collect::<jammi_db::error::Result<Vec<_>>>()
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let hashes =
            content_hash_columns(&rendered).map_err(|e| DataFusionError::External(Box::new(e)))?;
        Ok(ColumnarValue::Array(Arc::new(hashes)))
    }
}

/// Register the UDF on `ctx`. Idempotent (a re-registration replaces).
pub fn register_content_hash_udf(ctx: &SessionContext) {
    ctx.register_udf(ScalarUDF::new_from_impl(ContentHashUdf::default()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Int64Array, RecordBatch, StringArray};
    use arrow::datatypes::{Field, Schema};
    use jammi_db::store::content_hash::{content_hash_row, ContentValue};

    /// The SQL projection renders a non-string column with the runner's
    /// kernel (`Int64` → its decimal text), so the hash equals the pure fold
    /// over the rendered text — and a null cell is the `n` tag.
    #[tokio::test]
    async fn udf_hashes_the_runner_rendering() {
        let ctx = SessionContext::new();
        register_content_hash_udf(&ctx);
        let schema = Arc::new(Schema::new(vec![
            Field::new("n", DataType::Int64, true),
            Field::new("t", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![Some(7), None])),
                Arc::new(StringArray::from(vec![Some("a"), Some("b")])),
            ],
        )
        .unwrap();
        ctx.register_batch("src", batch).unwrap();
        let out = ctx
            .sql("SELECT jammi_content_hash(\"n\", \"t\") AS h FROM src")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let col = out[0].column(0);
        let h = col.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(
            h.value(0),
            content_hash_row(&[ContentValue::Str("7"), ContentValue::Str("a")]).to_hex()
        );
        assert_eq!(
            h.value(1),
            content_hash_row(&[ContentValue::Null, ContentValue::Str("b")]).to_hex()
        );
        assert_eq!(h.null_count(), 0);
    }
}
