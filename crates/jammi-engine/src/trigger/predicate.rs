//! Parsed-and-typed SQL `WHERE` predicates evaluated against `RecordBatch`es.
//!
//! The proto's `SubscribeRequest.predicate` is a SQL fragment; the engine
//! parses it once through DataFusion at subscribe time and the resulting
//! `PhysicalExpr` filters every batch the broker delivers. Empty input maps
//! to [`Predicate::match_all`] — the default that yields every row.

use std::sync::Arc;

use arrow::array::BooleanArray;
use arrow::compute::filter_record_batch;
use arrow::record_batch::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::common::{DFSchema, ScalarValue};
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::Expr;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::ColumnarValue;

use crate::trigger::error::TriggerError;

/// A parsed-and-type-checked predicate ready to evaluate against a batch.
/// `None` ≡ match every row (the proto's empty-string default).
#[derive(Clone)]
pub struct Predicate {
    physical: Option<Arc<dyn PhysicalExpr>>,
}

impl Predicate {
    /// The identity predicate — every row passes.
    pub fn match_all() -> Self {
        Self { physical: None }
    }

    /// Parse a SQL `WHERE`-clause fragment against `schema`. The DataFusion
    /// parser admits the full SQL dialect; this function walks the resulting
    /// `Expr` and rejects constructs outside the trigger-stream subset.
    pub fn from_sql(
        ctx: &SessionContext,
        schema: SchemaRef,
        sql: &str,
    ) -> Result<Self, TriggerError> {
        let trimmed = sql.trim();
        if trimmed.is_empty() {
            return Ok(Self::match_all());
        }
        let df_schema = DFSchema::try_from(schema.as_ref().clone())
            .map_err(|e| TriggerError::PredicateParse(e.to_string()))?;
        let logical = ctx
            .parse_sql_expr(trimmed, &df_schema)
            .map_err(|e| TriggerError::PredicateParse(e.to_string()))?;
        check_supported(&logical)?;
        let physical = ctx
            .create_physical_expr(logical, &df_schema)
            .map_err(|e| TriggerError::PredicateParse(e.to_string()))?;
        Ok(Self {
            physical: Some(physical),
        })
    }

    /// Apply the predicate to `batch`. Returns:
    /// * `Ok(None)` when zero rows match (the caller may skip delivery),
    /// * `Ok(Some(filtered))` when at least one row matches.
    pub fn evaluate(&self, batch: &RecordBatch) -> Result<Option<RecordBatch>, TriggerError> {
        let Some(physical) = &self.physical else {
            return Ok(Some(batch.clone()));
        };
        let columnar = physical
            .evaluate(batch)
            .map_err(|e| TriggerError::PredicateEval(e.to_string()))?;
        let mask = match columnar {
            ColumnarValue::Array(arr) => arr
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    TriggerError::PredicateEval("predicate did not produce Boolean array".into())
                })?
                .clone(),
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))) => {
                return Ok(Some(batch.clone()));
            }
            ColumnarValue::Scalar(ScalarValue::Boolean(_)) => return Ok(None),
            ColumnarValue::Scalar(other) => {
                return Err(TriggerError::PredicateEval(format!(
                    "predicate produced non-boolean scalar {other:?}"
                )));
            }
        };
        let filtered = filter_record_batch(batch, &mask)
            .map_err(|e| TriggerError::PredicateEval(e.to_string()))?;
        if filtered.num_rows() == 0 {
            Ok(None)
        } else {
            Ok(Some(filtered))
        }
    }
}

/// Whitelist of scalar functions admitted in subscribe predicates. Read-only
/// string helpers only — keeping the surface narrow until a real use case
/// pushes back.
const ALLOWED_FUNCTIONS: &[&str] = &["lower", "upper", "length", "starts_with", "ends_with"];

fn check_supported(expr: &Expr) -> Result<(), TriggerError> {
    use Expr::*;
    match expr {
        Column(_) | Literal(_, _) | Placeholder(_) => Ok(()),
        BinaryExpr(b) => {
            check_supported(&b.left)?;
            check_supported(&b.right)
        }
        Not(inner) | IsNotNull(inner) | IsNull(inner) | Negative(inner) | IsTrue(inner)
        | IsFalse(inner) | IsUnknown(inner) | IsNotTrue(inner) | IsNotFalse(inner)
        | IsNotUnknown(inner) => check_supported(inner),
        Between(b) => {
            check_supported(&b.expr)?;
            check_supported(&b.low)?;
            check_supported(&b.high)
        }
        Like(like) | SimilarTo(like) => {
            check_supported(&like.expr)?;
            check_supported(&like.pattern)
        }
        InList(in_list) => {
            check_supported(&in_list.expr)?;
            for item in &in_list.list {
                check_supported(item)?;
            }
            Ok(())
        }
        Cast(c) => check_supported(&c.expr),
        TryCast(c) => check_supported(&c.expr),
        ScalarFunction(call) => {
            let name = call.func.name().to_lowercase();
            if !ALLOWED_FUNCTIONS.contains(&name.as_str()) {
                return Err(TriggerError::PredicateUnsupported(format!(
                    "function `{name}` is not in the subscribe-predicate whitelist"
                )));
            }
            for arg in &call.args {
                check_supported(arg)?;
            }
            Ok(())
        }
        Alias(a) => check_supported(&a.expr),
        AggregateFunction(_) => Err(TriggerError::PredicateUnsupported(
            "aggregate functions are not allowed in subscribe predicates".into(),
        )),
        WindowFunction(_) => Err(TriggerError::PredicateUnsupported(
            "window functions are not allowed in subscribe predicates".into(),
        )),
        ScalarSubquery(_) | Exists(_) | InSubquery(_) => Err(TriggerError::PredicateUnsupported(
            "subqueries are not allowed in subscribe predicates".into(),
        )),
        Case(_) => Err(TriggerError::PredicateUnsupported(
            "CASE expressions are not allowed in subscribe predicates".into(),
        )),
        _ => Err(TriggerError::PredicateUnsupported(format!(
            "expression form is not allowed in subscribe predicates: {expr}"
        ))),
    }
}
