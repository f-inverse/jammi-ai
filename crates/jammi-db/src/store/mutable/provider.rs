//! `TableProvider` implementation for mutable companion tables.
//!
//! The provider supports `scan` (full-table reads) and `insert_into` (DataFusion DML through
//! [`MutableTableSink`]). Predicate pushdown, projection, and limit are translated to backend SQL
//! when straightforward; otherwise DataFusion's planner handles them above the scan node.

use std::fmt;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BinaryArray, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array,
    Int64Array, Int8Array, LargeBinaryArray, RecordBatch, StringArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt16Array,
    UInt32Array, UInt64Array, UInt8Array,
};
use arrow::compute::filter_record_batch;
use arrow_schema::{DataType, SchemaRef, TimeUnit};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::DFSchema;
use datafusion::datasource::sink::DataSinkExec;
use datafusion::datasource::{MemTable, TableType};
use datafusion::error::DataFusionError;
use datafusion::logical_expr::dml::InsertOp;
use datafusion::logical_expr::utils::conjunction;
use datafusion::physical_expr::{create_physical_expr, PhysicalExpr};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::Expr;

use crate::catalog::backend::{BackendError, Row, Transaction, TxOptions};

use super::definition::MutableTableDefinition;
use super::sink::{execution, primary_key_of, replace_rows, KeyConflict, MutableTableSink};
use super::{owned_rows, visible_rows, MutableBackend};

/// `TableProvider` for one mutable companion table.
pub struct MutableTableProvider {
    pub(crate) def: Arc<MutableTableDefinition>,
    pub(crate) backend: Arc<dyn MutableBackend>,
    pub(crate) tenant: crate::tenant_scope::TenantBinding,
}

impl fmt::Debug for MutableTableProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MutableTableProvider")
            .field("table", &self.def.id.as_str())
            .finish()
    }
}

impl MutableTableProvider {
    pub fn new(
        def: Arc<MutableTableDefinition>,
        backend: Arc<dyn MutableBackend>,
        tenant: crate::tenant_scope::TenantBinding,
    ) -> Self {
        Self {
            def,
            backend,
            tenant,
        }
    }

    /// Read all rows from the backend table into a single in-memory partition.
    /// This is the unsophisticated scan path — the entire table is read with
    /// the tenant-scope predicate pushed down to backend SQL, then DataFusion
    /// applies projection / additional filters above the scan node.
    async fn read_to_batch(&self, limit: Option<usize>) -> Result<RecordBatch, DataFusionError> {
        // Inject the tenant-scope predicate at the backend SQL layer so we
        // ship the correct row set off the SQLite/Postgres side, not just
        // the union (which DataFusion's AnalyzerRule would also filter, but
        // at the cost of materializing the full table first).
        //
        // Skip the predicate entirely when the caller is executing inside a
        // `JammiSession::with_admin_scope` closure: cross-tenant
        // administrative scans (server-startup recovery, audit reads) need
        // rows from every tenant. The analyzer-rule bypass on its own would
        // still leave the provider's SQL filter in place, so the provider
        // must consult the same marker.
        let visible = (!crate::tenant_scope::TenantBinding::is_admin_scope())
            .then(|| visible_rows(self.tenant.current_tenant()));
        let def = Arc::clone(&self.def);
        let backend = Arc::clone(&self.backend);
        self.backend
            .catalog_backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                move |tx| {
                    Box::pin(async move {
                        select_rows(tx, backend.as_ref(), &def, visible.as_deref(), limit).await
                    })
                },
            )
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))
    }

    /// Rewrite the session-owned rows `filters` selects, in one transaction:
    /// read them, then [`replace_rows`] them with `rewrite` applied — the
    /// rows' new values when `rewrite` is `Some`, nothing (a delete) when it
    /// is `None`. The read happens inside the write transaction, so the rows
    /// rewritten are exactly the rows the predicate matched. Returns the
    /// number of rows matched.
    async fn rewrite(
        &self,
        state: &dyn Session,
        filters: &[Expr],
        rewrite: Option<&[(String, Expr)]>,
    ) -> Result<u64, DataFusionError> {
        let schema = DFSchema::try_from(Arc::clone(&self.def.schema))?;
        let props = state.execution_props();
        let predicate = conjunction(filters.iter().cloned())
            .map(|e| create_physical_expr(&e, &schema, props))
            .transpose()?;
        let assignments = rewrite
            .map(|assignments| {
                assignments
                    .iter()
                    .map(|(column, e)| {
                        Ok((
                            self.def.schema.index_of(column)?,
                            create_physical_expr(e, &schema, props)?,
                        ))
                    })
                    .collect::<Result<Vec<_>, DataFusionError>>()
            })
            .transpose()?;

        let def = Arc::clone(&self.def);
        let backend = Arc::clone(&self.backend);
        let tenant = self.tenant.current_tenant();
        self.backend
            .catalog_backend()
            .transaction(TxOptions::default(), move |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.assert_tenant_matches(tenant, def.id.as_str())?;
                    let owned = owned_rows(tenant);
                    let rows = select_rows(tx, backend.as_ref(), &def, Some(&owned), None).await?;
                    let matched = matching(&rows, predicate.as_ref()).map_err(execution)?;
                    let keys = primary_key_of(&def, &matched)?;
                    let written = match &assignments {
                        Some(assignments) => assign(&matched, assignments).map_err(execution)?,
                        None => RecordBatch::new_empty(Arc::clone(&def.schema)),
                    };
                    replace_rows(tx, backend.as_ref(), &def, Some(&keys), &written).await?;
                    Ok(matched.num_rows() as u64)
                })
            })
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

#[async_trait]
impl TableProvider for MutableTableProvider {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.def.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        let batch = self.read_to_batch(limit).await?;
        let schema = batch.schema();
        let mem = MemTable::try_new(schema, vec![vec![batch]])?;
        mem.scan(state, projection, filters, limit).await
    }

    async fn insert_into(
        &self,
        _state: &dyn Session,
        input: Arc<dyn ExecutionPlan>,
        insert_op: InsertOp,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        let sink = Arc::new(MutableTableSink::new(
            Arc::clone(&self.def),
            Arc::clone(&self.backend),
            self.tenant.clone(),
            KeyConflict::try_from(insert_op)?,
        ));
        Ok(Arc::new(DataSinkExec::new(input, sink, None)))
    }

    async fn delete_from(
        &self,
        state: &dyn Session,
        filters: Vec<Expr>,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        let deleted = self.rewrite(state, &filters, None).await?;
        affected(state, deleted).await
    }

    async fn update(
        &self,
        state: &dyn Session,
        assignments: Vec<(String, Expr)>,
        filters: Vec<Expr>,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        let updated = self.rewrite(state, &filters, Some(&assignments)).await?;
        affected(state, updated).await
    }

    async fn truncate(
        &self,
        state: &dyn Session,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        self.delete_from(state, Vec::new()).await
    }
}

/// The single-row `count` plan a DML statement answers with — the shape
/// DataFusion's own `DataSinkExec` and `MemTable` DML return.
async fn affected(
    state: &dyn Session,
    rows: u64,
) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
    let batch = RecordBatch::try_from_iter_with_nullable(vec![(
        "count",
        Arc::new(UInt64Array::from(vec![rows])) as ArrayRef,
        false,
    )])?;
    MemTable::try_new(batch.schema(), vec![vec![batch]])?
        .scan(state, None, &[], None)
        .await
}

/// Every row of `def` that `predicate` selects, read inside `tx`.
async fn select_rows(
    tx: &mut Transaction<'_>,
    backend: &dyn MutableBackend,
    def: &MutableTableDefinition,
    predicate: Option<&str>,
    limit: Option<usize>,
) -> Result<RecordBatch, BackendError> {
    let columns: Vec<(String, DataType)> = def
        .schema
        .fields()
        .iter()
        .map(|f| (f.name().clone(), f.data_type().clone()))
        .collect();
    let names: Vec<&str> = columns.iter().map(|(name, _)| name.as_str()).collect();
    let sql = backend.scan_dml(def, &names, predicate, limit);
    let rows = tx.query(&sql, &[], |row| decode_row(row, &columns)).await?;
    // Transpose Vec<Row> → Vec<Column>
    let mut transposed: Vec<Vec<DecodedValue>> = (0..columns.len())
        .map(|_| Vec::with_capacity(rows.len()))
        .collect();
    for r in rows {
        for (i, v) in r.into_iter().enumerate() {
            transposed[i].push(v);
        }
    }
    let arrays = build_arrays(&columns, transposed).map_err(execution)?;
    RecordBatch::try_new(Arc::clone(&def.schema), arrays).map_err(execution)
}

/// The rows of `rows` that `predicate` selects; every row when there is no
/// predicate. A `NULL` verdict does not select (SQL three-valued logic), which
/// is how `filter_record_batch` reads a null mask slot.
fn matching(
    rows: &RecordBatch,
    predicate: Option<&Arc<dyn PhysicalExpr>>,
) -> Result<RecordBatch, DataFusionError> {
    let Some(predicate) = predicate else {
        return Ok(rows.clone());
    };
    let verdict = predicate.evaluate(rows)?.into_array(rows.num_rows())?;
    let verdict = verdict
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| {
            DataFusionError::Plan("a DML predicate must evaluate to a boolean".into())
        })?;
    Ok(filter_record_batch(rows, verdict)?)
}

/// `rows` with each assigned column replaced by its expression's value.
fn assign(
    rows: &RecordBatch,
    assignments: &[(usize, Arc<dyn PhysicalExpr>)],
) -> Result<RecordBatch, DataFusionError> {
    let mut columns = rows.columns().to_vec();
    for (index, expr) in assignments {
        columns[*index] = expr.evaluate(rows)?.into_array(rows.num_rows())?;
    }
    Ok(RecordBatch::try_new(rows.schema(), columns)?)
}

/// One column value read from a backend row, after **width-faithful**
/// type-aware extraction: the decode width matches the storage type the
/// mutable backends actually declare for each Arrow `DataType` (SMALLINT for
/// Int8/Int16, INTEGER for Int32/UInt8/UInt16, BIGINT for
/// Int64/UInt32/UInt64/Timestamp — see `store/mutable/postgres.rs::pg_type`)
/// rather than folding every integer width into `i64` and every float width
/// into `f64`. Postgres rejects an `i32`/`f64` bind against a narrower
/// column (SMALLINT/REAL), so the fold silently broke replay for 8 of the 13
/// topic-schema types this crate accepts (`topic_repo.rs`).
#[derive(Debug, Clone)]
pub(crate) enum DecodedValue {
    Null,
    Bool(bool),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Text(String),
    Bytes(Vec<u8>),
}

pub(crate) fn decode_row(
    row: &Row<'_>,
    columns: &[(String, DataType)],
) -> Result<Vec<DecodedValue>, crate::catalog::backend::BackendError> {
    columns
        .iter()
        .map(|(name, ty)| match ty {
            DataType::Boolean => Ok(row
                .try_get::<bool>(name)?
                .map(DecodedValue::Bool)
                .unwrap_or(DecodedValue::Null)),
            // SMALLINT/INT2 — a bind as `i32` is rejected by Postgres.
            DataType::Int8 | DataType::Int16 => Ok(row
                .try_get::<i16>(name)?
                .map(DecodedValue::Int16)
                .unwrap_or(DecodedValue::Null)),
            // INTEGER/INT4.
            DataType::Int32 | DataType::UInt8 | DataType::UInt16 => Ok(row
                .try_get::<i32>(name)?
                .map(DecodedValue::Int32)
                .unwrap_or(DecodedValue::Null)),
            // BIGINT/INT8. The mutable-table DDL stores a timestamp column as
            // its integer tick (see `sink.rs::extract_value`); the tick is
            // decoded as a plain i64 here and reassembled into the typed
            // Arrow Timestamp array in `build_arrays`, which knows the
            // column's `TimeUnit`.
            DataType::Int64 | DataType::UInt32 | DataType::UInt64 | DataType::Timestamp(_, _) => {
                Ok(row
                    .try_get::<i64>(name)?
                    .map(DecodedValue::Int64)
                    .unwrap_or(DecodedValue::Null))
            }
            // REAL/FLOAT4 — a bind as `f64` is rejected by Postgres.
            DataType::Float16 | DataType::Float32 => Ok(row
                .try_get::<f32>(name)?
                .map(DecodedValue::Float32)
                .unwrap_or(DecodedValue::Null)),
            DataType::Float64 => Ok(row
                .try_get::<f64>(name)?
                .map(DecodedValue::Float64)
                .unwrap_or(DecodedValue::Null)),
            DataType::Utf8 | DataType::LargeUtf8 => Ok(row
                .try_get::<String>(name)?
                .map(DecodedValue::Text)
                .unwrap_or(DecodedValue::Null)),
            DataType::Binary | DataType::LargeBinary => Ok(row
                .try_get::<Vec<u8>>(name)?
                .map(DecodedValue::Bytes)
                .unwrap_or(DecodedValue::Null)),
            other => Err(crate::catalog::backend::BackendError::Execution(format!(
                "mutable-table scan: column {name:?} has unsupported Arrow type {other:?}"
            ))),
        })
        .collect()
}

pub(crate) fn build_arrays(
    columns: &[(String, DataType)],
    rows_per_col: Vec<Vec<DecodedValue>>,
) -> Result<Vec<ArrayRef>, DataFusionError> {
    columns
        .iter()
        .zip(rows_per_col)
        .map(|((_, ty), values)| -> Result<ArrayRef, DataFusionError> {
            match ty {
                DataType::Boolean => {
                    let arr: BooleanArray = values
                        .into_iter()
                        .map(|v| match v {
                            DecodedValue::Bool(b) => Some(b),
                            _ => None,
                        })
                        .collect();
                    Ok(Arc::new(arr) as ArrayRef)
                }
                DataType::Int8 => {
                    let arr: Int8Array = values
                        .into_iter()
                        .map(|v| match v {
                            DecodedValue::Int16(i) => Some(i as i8),
                            _ => None,
                        })
                        .collect();
                    Ok(Arc::new(arr) as ArrayRef)
                }
                DataType::Int16 => {
                    let arr: Int16Array = values
                        .into_iter()
                        .map(|v| match v {
                            DecodedValue::Int16(i) => Some(i),
                            _ => None,
                        })
                        .collect();
                    Ok(Arc::new(arr) as ArrayRef)
                }
                DataType::Int32 => {
                    let arr: Int32Array = values
                        .into_iter()
                        .map(|v| match v {
                            DecodedValue::Int32(i) => Some(i),
                            _ => None,
                        })
                        .collect();
                    Ok(Arc::new(arr) as ArrayRef)
                }
                DataType::UInt8 => {
                    let arr: UInt8Array = values
                        .into_iter()
                        .map(|v| match v {
                            DecodedValue::Int32(i) => Some(i as u8),
                            _ => None,
                        })
                        .collect();
                    Ok(Arc::new(arr) as ArrayRef)
                }
                DataType::UInt16 => {
                    let arr: UInt16Array = values
                        .into_iter()
                        .map(|v| match v {
                            DecodedValue::Int32(i) => Some(i as u16),
                            _ => None,
                        })
                        .collect();
                    Ok(Arc::new(arr) as ArrayRef)
                }
                DataType::Int64 => {
                    let arr: Int64Array = values
                        .into_iter()
                        .map(|v| match v {
                            DecodedValue::Int64(i) => Some(i),
                            _ => None,
                        })
                        .collect();
                    Ok(Arc::new(arr) as ArrayRef)
                }
                DataType::UInt32 => {
                    let arr: UInt32Array = values
                        .into_iter()
                        .map(|v| match v {
                            DecodedValue::Int64(i) => Some(i as u32),
                            _ => None,
                        })
                        .collect();
                    Ok(Arc::new(arr) as ArrayRef)
                }
                DataType::UInt64 => {
                    let arr: UInt64Array = values
                        .into_iter()
                        .map(|v| match v {
                            DecodedValue::Int64(i) => Some(i as u64),
                            _ => None,
                        })
                        .collect();
                    Ok(Arc::new(arr) as ArrayRef)
                }
                DataType::Float32 => {
                    let arr: Float32Array = values
                        .into_iter()
                        .map(|v| match v {
                            DecodedValue::Float32(f) => Some(f),
                            _ => None,
                        })
                        .collect();
                    Ok(Arc::new(arr) as ArrayRef)
                }
                DataType::Float64 => {
                    let arr: Float64Array = values
                        .into_iter()
                        .map(|v| match v {
                            DecodedValue::Float64(f) => Some(f),
                            _ => None,
                        })
                        .collect();
                    Ok(Arc::new(arr) as ArrayRef)
                }
                DataType::Utf8 | DataType::LargeUtf8 => {
                    let arr: StringArray = values
                        .into_iter()
                        .map(|v| match v {
                            DecodedValue::Text(s) => Some(s),
                            _ => None,
                        })
                        .collect();
                    Ok(Arc::new(arr) as ArrayRef)
                }
                DataType::Binary => {
                    let owned: Vec<Option<Vec<u8>>> = values
                        .into_iter()
                        .map(|v| match v {
                            DecodedValue::Bytes(b) => Some(b),
                            _ => None,
                        })
                        .collect();
                    let arr: BinaryArray = owned
                        .iter()
                        .map(|o| o.as_deref())
                        .collect::<Vec<_>>()
                        .into();
                    Ok(Arc::new(arr) as ArrayRef)
                }
                DataType::LargeBinary => {
                    let owned: Vec<Option<Vec<u8>>> = values
                        .into_iter()
                        .map(|v| match v {
                            DecodedValue::Bytes(b) => Some(b),
                            _ => None,
                        })
                        .collect();
                    let arr: LargeBinaryArray = owned
                        .iter()
                        .map(|o| o.as_deref())
                        .collect::<Vec<_>>()
                        .into();
                    Ok(Arc::new(arr) as ArrayRef)
                }
                DataType::Timestamp(unit, tz) => {
                    let ticks: Vec<Option<i64>> = values
                        .into_iter()
                        .map(|v| match v {
                            DecodedValue::Int64(i) => Some(i),
                            _ => None,
                        })
                        .collect();
                    // The tz is carried through unchanged (not reinterpreted)
                    // so the reconstructed array's DataType exactly equals
                    // the schema's `Timestamp(unit, tz)`.
                    Ok(match unit {
                        TimeUnit::Second => {
                            let arr: TimestampSecondArray = ticks.into_iter().collect();
                            Arc::new(arr.with_timezone_opt(tz.clone())) as ArrayRef
                        }
                        TimeUnit::Millisecond => {
                            let arr: TimestampMillisecondArray = ticks.into_iter().collect();
                            Arc::new(arr.with_timezone_opt(tz.clone())) as ArrayRef
                        }
                        TimeUnit::Microsecond => {
                            let arr: TimestampMicrosecondArray = ticks.into_iter().collect();
                            Arc::new(arr.with_timezone_opt(tz.clone())) as ArrayRef
                        }
                        TimeUnit::Nanosecond => {
                            let arr: TimestampNanosecondArray = ticks.into_iter().collect();
                            Arc::new(arr.with_timezone_opt(tz.clone())) as ArrayRef
                        }
                    })
                }
                other => Err(DataFusionError::NotImplemented(format!(
                    "mutable-table scan cannot materialise Arrow type {other:?}"
                ))),
            }
        })
        .collect()
}
