//! The one write every mutable-table statement is, and the `DataSink` that
//! runs it for `INSERT` / `REPLACE INTO`.
//!
//! Every statement — append, upsert, update, delete — is `replace_rows`:
//! remove the session-owned rows at some primary keys, then insert some rows,
//! in one transaction. Per DataFusion: *"This method will be called exactly
//! once during each DML statement. Thus prior to return, the sink should do
//! any commit or rollback required."* The sink wraps the entire write in one
//! `crate::catalog::backend::CatalogBackend::transaction` closure.

use std::fmt;
use std::sync::Arc;

use arrow::array::{
    Array, BinaryArray, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array,
    Int64Array, Int8Array, LargeBinaryArray, StringArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt16Array,
    UInt32Array, UInt64Array, UInt8Array,
};
use arrow::record_batch::RecordBatch;
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::common::DataFusionError;
use datafusion::datasource::sink::DataSink;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::execution::TaskContext;
use datafusion::logical_expr::dml::InsertOp;
use datafusion::physical_plan::DisplayAs;
use datafusion::physical_plan::DisplayFormatType;
use futures::StreamExt;

use crate::catalog::backend::{BackendError, SqlNullType, SqlValue, Transaction, TxOptions};

use super::definition::MutableTableDefinition;
use super::{owned_rows, MutableBackend};

/// How an incoming row relates to an existing row with the same primary key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyConflict {
    /// The statement fails (`INSERT`).
    Reject,
    /// The existing row is replaced (`REPLACE INTO`).
    Replace,
}

impl TryFrom<InsertOp> for KeyConflict {
    type Error = DataFusionError;

    fn try_from(op: InsertOp) -> Result<Self, Self::Error> {
        match op {
            InsertOp::Append => Ok(Self::Reject),
            InsertOp::Replace => Ok(Self::Replace),
            InsertOp::Overwrite => Err(DataFusionError::NotImplemented(
                "INSERT OVERWRITE is not supported on mutable tables; \
                 DELETE then INSERT, or REPLACE INTO by primary key"
                    .into(),
            )),
        }
    }
}

pub struct MutableTableSink {
    def: Arc<MutableTableDefinition>,
    backend: Arc<dyn MutableBackend>,
    tenant: crate::tenant_scope::TenantBinding,
    on_conflict: KeyConflict,
}

impl MutableTableSink {
    pub fn new(
        def: Arc<MutableTableDefinition>,
        backend: Arc<dyn MutableBackend>,
        tenant: crate::tenant_scope::TenantBinding,
        on_conflict: KeyConflict,
    ) -> Self {
        Self {
            def,
            backend,
            tenant,
            on_conflict,
        }
    }
}

impl fmt::Debug for MutableTableSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MutableTableSink")
            .field("table", &self.def.id.as_str())
            .finish()
    }
}

impl DisplayAs for MutableTableSink {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MutableTableSink(table={})", self.def.id)
    }
}

#[async_trait]
impl DataSink for MutableTableSink {
    fn schema(&self) -> &SchemaRef {
        &self.def.schema
    }

    async fn write_all(
        &self,
        mut data: SendableRecordBatchStream,
        _ctx: &Arc<TaskContext>,
    ) -> Result<u64, DataFusionError> {
        // Collect batches up-front so the transaction closure can be `Send`
        // without capturing the stream's non-Send internals.
        let mut batches: Vec<RecordBatch> = Vec::new();
        while let Some(b) = data.next().await {
            batches.push(b?);
        }

        let def = Arc::clone(&self.def);
        let backend_for_closure = Arc::clone(&self.backend);
        // Snapshot the tenant binding once per write_all call (not per row).
        // The DataSink contract is "exactly one write_all per DML statement"
        // so this is the natural unit of consistency.
        let session_tenant = self.tenant.current_tenant();
        let table_name = def.id.as_str().to_string();
        let on_conflict = self.on_conflict;
        let written = self
            .backend
            .catalog_backend()
            .transaction(TxOptions::default(), move |tx| {
                let backend = backend_for_closure;
                Box::pin(async move {
                    tx.set_tenant(session_tenant);
                    // Defence-in-depth: confirm the transaction sees the same
                    // tenant we're about to bind to every row. A future change
                    // that mutates the binding mid-flight would fail here
                    // instead of silently re-tenanting the rows.
                    tx.assert_tenant_matches(session_tenant, &table_name)?;
                    let mut total: u64 = 0;
                    for batch in batches {
                        let replaced = match on_conflict {
                            KeyConflict::Reject => None,
                            KeyConflict::Replace => Some(primary_key_of(&def, &batch)?),
                        };
                        total +=
                            replace_rows(tx, backend.as_ref(), &def, replaced.as_ref(), &batch)
                                .await?;
                        #[cfg(feature = "test-hooks")]
                        crate::store::mutable::test_hook::maybe_signal(total).await;
                    }
                    Ok(total)
                })
            })
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        Ok(written)
    }
}

/// Replace the session-owned rows at the primary keys in `keys` (when given)
/// with `rows`, inside `tx`: the one write every mutable-table statement is.
/// An append replaces nothing; a delete writes nothing; an update and an
/// upsert do both. Rows are stamped with the transaction's tenant, and only
/// that tenant's rows are removed, so a statement never moves a row across
/// tenants. Returns the number of rows written.
pub(crate) async fn replace_rows(
    tx: &mut Transaction<'_>,
    backend: &dyn MutableBackend,
    def: &MutableTableDefinition,
    keys: Option<&RecordBatch>,
    rows: &RecordBatch,
) -> Result<u64, BackendError> {
    if let Some(keys) = keys.filter(|k| k.num_rows() > 0) {
        let dml = backend.delete_keys_dml(def, keys.num_rows(), &owned_rows(tx.tenant()));
        let params = batch_to_params(keys, None).map_err(execution)?;
        // `batch_to_params` appends a tenant slot per row; a key tuple has none.
        let width = keys.num_columns();
        let params: Vec<_> = params
            .chunks(width + 1)
            .flat_map(|row| row[..width].iter().cloned())
            .collect();
        tx.execute(&dml, &params).await?;
    }
    insert_rows(tx, backend, def, rows).await
}

/// Append `rows` inside `tx`, stamped with the transaction's tenant. Returns
/// the number of rows written.
pub(crate) async fn insert_rows(
    tx: &mut Transaction<'_>,
    backend: &dyn MutableBackend,
    def: &MutableTableDefinition,
    rows: &RecordBatch,
) -> Result<u64, BackendError> {
    if rows.num_rows() == 0 {
        return Ok(0);
    }
    let schema = rows.schema();
    let cols: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    let dml = backend.insert_dml(def, &cols, rows.num_rows());
    let params = batch_to_params(rows, tx.tenant()).map_err(execution)?;
    tx.execute(&dml, &params).await
}

/// The primary-key columns of `rows`, in declared key order.
pub(crate) fn primary_key_of(
    def: &MutableTableDefinition,
    rows: &RecordBatch,
) -> Result<RecordBatch, BackendError> {
    let schema = rows.schema();
    let indices = def
        .primary_key
        .iter()
        .map(|c| schema.index_of(c))
        .collect::<Result<Vec<_>, _>>()
        .map_err(execution)?;
    rows.project(&indices).map_err(execution)
}

pub(crate) fn execution(e: impl ToString) -> BackendError {
    BackendError::Execution(e.to_string())
}

/// Translate every cell of a `RecordBatch` into the engine's [`SqlValue`]
/// taxonomy, in row-major order. The implicit `tenant_id` slot is bound to
/// the session-bound tenant: when `Some(t)`, every row carries the same
/// tenant string; when `None`, every row carries `NULL` (a globally-scoped
/// write, as in a single-tenant deployment).
pub(crate) fn batch_to_params(
    batch: &RecordBatch,
    tenant: Option<crate::tenant::TenantId>,
) -> Result<Vec<SqlValue<'static>>, &'static str> {
    let n_rows = batch.num_rows();
    let arrays: Vec<&dyn Array> = batch.columns().iter().map(|c| c.as_ref()).collect();
    let tenant_value = match tenant {
        Some(t) => SqlValue::TextOwned(t.to_string()),
        None => SqlValue::Null(SqlNullType::Text),
    };
    let mut out = Vec::with_capacity(n_rows * (arrays.len() + 1));
    for r in 0..n_rows {
        for (col_idx, arr) in arrays.iter().enumerate() {
            let value = extract_value(*arr, r, batch.schema().field(col_idx).data_type())?;
            out.push(value);
        }
        out.push(tenant_value.clone());
    }
    Ok(out)
}

fn extract_value(
    arr: &dyn Array,
    idx: usize,
    ty: &arrow_schema::DataType,
) -> Result<SqlValue<'static>, &'static str> {
    use arrow_schema::DataType::*;
    if arr.is_null(idx) {
        // A null cell must bind the same SqlNullType the non-null arm below
        // produces for this column's Arrow type, so Postgres sees one
        // consistent SQL type across null and non-null rows.
        let null_type = match ty {
            Boolean => SqlNullType::Bool,
            Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64 => SqlNullType::Int,
            Float32 | Float64 => SqlNullType::Float,
            Utf8 => SqlNullType::Text,
            Binary | LargeBinary => SqlNullType::Bytes,
            // A null timestamp mirrors the non-null arm's bind-kind so a
            // column binds one consistent SQL type across null and
            // non-null rows.
            Timestamp(_, _) => SqlNullType::Int,
            _ => return Err("unsupported arrow type for mutable-table insert"),
        };
        return Ok(SqlValue::Null(null_type));
    }
    match ty {
        Boolean => arr
            .as_any()
            .downcast_ref::<BooleanArray>()
            .map(|a| SqlValue::Bool(a.value(idx)))
            .ok_or("expected BooleanArray"),
        Int8 => arr
            .as_any()
            .downcast_ref::<Int8Array>()
            .map(|a| SqlValue::Int(a.value(idx) as i64))
            .ok_or("expected Int8Array"),
        Int16 => arr
            .as_any()
            .downcast_ref::<Int16Array>()
            .map(|a| SqlValue::Int(a.value(idx) as i64))
            .ok_or("expected Int16Array"),
        Int32 => arr
            .as_any()
            .downcast_ref::<Int32Array>()
            .map(|a| SqlValue::Int(a.value(idx) as i64))
            .ok_or("expected Int32Array"),
        Int64 => arr
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(|a| SqlValue::Int(a.value(idx)))
            .ok_or("expected Int64Array"),
        UInt8 => arr
            .as_any()
            .downcast_ref::<UInt8Array>()
            .map(|a| SqlValue::Int(a.value(idx) as i64))
            .ok_or("expected UInt8Array"),
        UInt16 => arr
            .as_any()
            .downcast_ref::<UInt16Array>()
            .map(|a| SqlValue::Int(a.value(idx) as i64))
            .ok_or("expected UInt16Array"),
        UInt32 => arr
            .as_any()
            .downcast_ref::<UInt32Array>()
            .map(|a| SqlValue::Int(a.value(idx) as i64))
            .ok_or("expected UInt32Array"),
        // BIGINT is the storage type on both backends, so a UInt64 value
        // above `i64::MAX` cannot round-trip; refused here at the publish
        // edge rather than silently wrapped or truncated at write time.
        UInt64 => {
            let a = arr
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or("expected UInt64Array")?;
            let v = a.value(idx);
            if v > i64::MAX as u64 {
                return Err("UInt64 value exceeds i64::MAX (BIGINT storage range)");
            }
            Ok(SqlValue::Int(v as i64))
        }
        Float32 => arr
            .as_any()
            .downcast_ref::<Float32Array>()
            .map(|a| SqlValue::Float(a.value(idx) as f64))
            .ok_or("expected Float32Array"),
        Float64 => arr
            .as_any()
            .downcast_ref::<Float64Array>()
            .map(|a| SqlValue::Float(a.value(idx)))
            .ok_or("expected Float64Array"),
        Utf8 => arr
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|a| SqlValue::TextOwned(a.value(idx).to_string()))
            .ok_or("expected StringArray"),
        Binary => arr
            .as_any()
            .downcast_ref::<BinaryArray>()
            .map(|a| SqlValue::BytesOwned(a.value(idx).to_vec()))
            .ok_or("expected BinaryArray"),
        LargeBinary => arr
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .map(|a| SqlValue::BytesOwned(a.value(idx).to_vec()))
            .ok_or("expected LargeBinaryArray"),
        // Timestamps are stored as their integer tick in the column's
        // declared `TimeUnit` — `INTEGER` on SQLite, `BIGINT` on Postgres.
        // DataFusion reconstructs the Arrow `Timestamp` array (and performs
        // all temporal operations) at read time, so the backend column
        // itself never needs to be a SQL timestamp type.
        Timestamp(arrow_schema::TimeUnit::Second, _) => arr
            .as_any()
            .downcast_ref::<TimestampSecondArray>()
            .map(|a| SqlValue::Int(a.value(idx)))
            .ok_or("expected TimestampSecondArray"),
        Timestamp(arrow_schema::TimeUnit::Millisecond, _) => arr
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .map(|a| SqlValue::Int(a.value(idx)))
            .ok_or("expected TimestampMillisecondArray"),
        Timestamp(arrow_schema::TimeUnit::Microsecond, _) => arr
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .map(|a| SqlValue::Int(a.value(idx)))
            .ok_or("expected TimestampMicrosecondArray"),
        Timestamp(arrow_schema::TimeUnit::Nanosecond, _) => arr
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .map(|a| SqlValue::Int(a.value(idx)))
            .ok_or("expected TimestampNanosecondArray"),
        _ => Err("unsupported arrow type for mutable-table insert"),
    }
}
