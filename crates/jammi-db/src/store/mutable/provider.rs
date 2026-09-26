//! `TableProvider` implementation for mutable companion tables.
//!
//! The provider supports `scan` (full-table reads), `insert_into` (DataFusion DML through
//! [`MutableTableSink`]) and `truncate`. Predicate pushdown, projection, and limit are translated
//! to backend SQL when straightforward; otherwise DataFusion's planner handles them above the scan
//! node. An `UPDATE` / `DELETE` plans to [`super::rewrite::RowRewriteNode`], which hands the rows
//! the statement selected to `MutableTableProvider::rewrite_rows`.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use arrow::array::Array;
use arrow::array::{
    ArrayRef, BinaryArray, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array,
    Int64Array, Int8Array, LargeBinaryArray, RecordBatch, StringArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt16Array,
    UInt32Array, UInt64Array, UInt8Array,
};
use arrow::compute::{concat_batches, filter_record_batch, take_record_batch};
use arrow::row::{Row as ValueRow, RowConverter, Rows, SortField};
use arrow::util::display::{ArrayFormatter, FormatOptions};
use arrow_schema::{DataType, SchemaRef, TimeUnit};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::datasource::sink::DataSinkExec;
use datafusion::datasource::{MemTable, TableType};
use datafusion::error::DataFusionError;
use datafusion::logical_expr::dml::InsertOp;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::Expr;

use crate::catalog::backend::{BackendError, Row, SqlValue, Transaction, TxOptions};
use crate::error::JammiError;

use super::definition::{MutableTableDefinition, MutableTableError};
use super::sink::{
    execution, key_params, primary_key_of, replace_rows, row_chunks, KeyConflict, MutableTableSink,
};
use super::{keys_predicate, owned_rows, visible_rows, MutableBackend};

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
                        let columns = select_rows(
                            tx,
                            backend.as_ref(),
                            &def,
                            &[],
                            visible.as_deref(),
                            &[],
                            limit,
                        )
                        .await?;
                        RecordBatch::try_new(Arc::clone(&def.schema), columns).map_err(execution)
                    })
                },
            )
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))
    }

    /// Rewrite the rows an `UPDATE` / `DELETE` selected, in one serializable
    /// transaction: `old` is each selected row as the statement read it, and
    /// `new`, for an `UPDATE`, the row it becomes (row for row); a `DELETE`
    /// writes nothing in their place.
    ///
    /// The statement read its rows before this transaction opened, so the
    /// transaction first re-reads the rows at the selected keys: a row
    /// another writer changed or removed in between fails the statement
    /// with [`MutableTableError::WriteConflict`] before anything is written,
    /// rather than overwriting that writer's change. A selected row the
    /// session reads but does not own (a global row, for a tenant-bound
    /// session) is left in place. Returns the number of rows rewritten.
    pub(crate) async fn rewrite_rows(
        &self,
        old: RecordBatch,
        new: Option<RecordBatch>,
    ) -> Result<u64, JammiError> {
        let (old, new) = distinct_rows(&self.def, old, new)?;
        let def = Arc::clone(&self.def);
        let backend = Arc::clone(&self.backend);
        let tenant = self.tenant.current_tenant();
        let settled = self
            .backend
            .catalog_backend()
            .serializable(move |tx| {
                let (def, backend) = (Arc::clone(&def), Arc::clone(&backend));
                let (old, new) = (old.clone(), new.clone());
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.assert_tenant_matches(tenant, def.id.as_str())?;
                    let keys = primary_key_of(&def, &old)?;
                    let current = visible_at_keys(tx, backend.as_ref(), &def, &keys).await?;
                    let owned = match unchanged(&def, &old, &current).map_err(execution)? {
                        Unchanged::Owned(owned) => owned,
                        Unchanged::Conflict(rows) => return Ok(Settled::Conflict(rows)),
                    };
                    let keys = filter_record_batch(&keys, &owned).map_err(execution)?;
                    let written = match &new {
                        Some(new) => filter_record_batch(new, &owned).map_err(execution)?,
                        None => RecordBatch::new_empty(Arc::clone(&def.schema)),
                    };
                    replace_rows(tx, backend.as_ref(), &def, Some(&keys), &written).await?;
                    Ok(Settled::Rewrote(keys.num_rows() as u64))
                })
            })
            .await?;
        match settled {
            Settled::Rewrote(rows) => Ok(rows),
            Settled::Conflict(rows) => Err(MutableTableError::WriteConflict {
                table: self.def.id.clone(),
                rows,
            }
            .into()),
        }
    }
}

/// How a rewrite's transaction settled.
enum Settled {
    /// It rewrote this many rows.
    Rewrote(u64),
    /// This many selected rows had changed; it wrote nothing.
    Conflict(u64),
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

    async fn truncate(
        &self,
        state: &dyn Session,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        let def = Arc::clone(&self.def);
        let backend = Arc::clone(&self.backend);
        let tenant = self.tenant.current_tenant();
        let deleted = self
            .backend
            .catalog_backend()
            .transaction(TxOptions::default(), move |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.assert_tenant_matches(tenant, def.id.as_str())?;
                    tx.execute(&backend.delete_dml(&def, &owned_rows(tenant)), &[])
                        .await
                })
            })
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        affected(state, deleted).await
    }
}

/// The single-row `count` a DML statement answers with — the shape
/// DataFusion's own `DataSinkExec` and `MemTable` DML return.
pub(crate) fn count_batch(rows: u64) -> Result<RecordBatch, DataFusionError> {
    Ok(RecordBatch::try_from_iter_with_nullable(vec![(
        "count",
        Arc::new(UInt64Array::from(vec![rows])) as ArrayRef,
        false,
    )])?)
}

/// [`count_batch`] as a plan.
async fn affected(
    state: &dyn Session,
    rows: u64,
) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
    let batch = count_batch(rows)?;
    MemTable::try_new(batch.schema(), vec![vec![batch]])?
        .scan(state, None, &[], None)
        .await
}

/// Every row of `def` that `predicate` selects, `params` bound, read inside
/// `tx`: the table's columns, then each of `extra` (a column the table's
/// schema does not declare, such as the tenant slot).
async fn select_rows(
    tx: &mut Transaction<'_>,
    backend: &dyn MutableBackend,
    def: &MutableTableDefinition,
    extra: &[(String, DataType)],
    predicate: Option<&str>,
    params: &[SqlValue<'_>],
    limit: Option<usize>,
) -> Result<Vec<ArrayRef>, BackendError> {
    let columns: Vec<(String, DataType)> = def
        .schema
        .fields()
        .iter()
        .map(|f| (f.name().clone(), f.data_type().clone()))
        .chain(extra.iter().cloned())
        .collect();
    let names: Vec<&str> = columns.iter().map(|(name, _)| name.as_str()).collect();
    let sql = backend.scan_dml(def, &names, predicate, limit);
    let rows = tx
        .query(&sql, params, |row| decode_row(row, &columns))
        .await?;
    // Transpose Vec<Row> → Vec<Column>
    let mut transposed: Vec<Vec<DecodedValue>> = (0..columns.len())
        .map(|_| Vec::with_capacity(rows.len()))
        .collect();
    for r in rows {
        for (i, v) in r.into_iter().enumerate() {
            transposed[i].push(v);
        }
    }
    build_arrays(&columns, transposed).map_err(execution)
}

/// The rows of `def` at the primary keys in `keys` that the transaction's
/// tenant reads, each with whether it owns the row (a tenant reads the
/// global rows too, but owns only its own).
async fn visible_at_keys(
    tx: &mut Transaction<'_>,
    backend: &dyn MutableBackend,
    def: &MutableTableDefinition,
    keys: &RecordBatch,
) -> Result<(RecordBatch, BooleanArray), BackendError> {
    let tenant_slot = [("tenant_id".to_string(), DataType::Utf8)];
    let visible = visible_rows(tx.tenant());
    let tenant_bound = tx.tenant().is_some();
    let mut rows = Vec::new();
    let mut owned = Vec::new();
    for chunk in row_chunks(keys, keys.num_columns(), backend) {
        let predicate = format!("{} AND {visible}", keys_predicate(def, chunk.num_rows()));
        let mut columns = select_rows(
            tx,
            backend,
            def,
            &tenant_slot,
            Some(&predicate),
            &key_params(&chunk)?,
            None,
        )
        .await?;
        let row_tenant = columns.pop().expect("the tenant slot was selected");
        // Every visible row is the session's own or global; a tenant-bound
        // session owns the tenanted ones, an unbound session the global ones.
        owned.extend((0..row_tenant.len()).map(|i| Some(row_tenant.is_valid(i) == tenant_bound)));
        rows.push(RecordBatch::try_new(Arc::clone(&def.schema), columns).map_err(execution)?);
    }
    let rows = concat_batches(&def.schema, &rows).map_err(execution)?;
    Ok((rows, BooleanArray::from(owned)))
}

/// `old` and `new` with each selected row once. A join can select one row
/// several times; the copies must agree on the row's new value, or the
/// `UPDATE` is ambiguous.
fn distinct_rows(
    def: &MutableTableDefinition,
    old: RecordBatch,
    new: Option<RecordBatch>,
) -> Result<(RecordBatch, Option<RecordBatch>), JammiError> {
    let keys = primary_key_of(def, &old)?;
    let keys = row_values(&keys).map_err(execution)?;
    let values = row_values(new.as_ref().unwrap_or(&old)).map_err(execution)?;
    let mut first: HashMap<ValueRow<'_>, usize> = HashMap::with_capacity(keys.num_rows());
    let mut kept = Vec::with_capacity(keys.num_rows());
    for i in 0..keys.num_rows() {
        match first.entry(keys.row(i)) {
            Entry::Vacant(slot) => {
                slot.insert(i);
                kept.push(i as u64);
            }
            Entry::Occupied(seen) if values.row(*seen.get()) == values.row(i) => {}
            Entry::Occupied(_) => {
                return Err(MutableTableError::AmbiguousUpdate {
                    table: def.id.clone(),
                    key: key_display(&primary_key_of(def, &old)?.slice(i, 1)),
                }
                .into())
            }
        }
    }
    if kept.len() == keys.num_rows() {
        return Ok((old, new));
    }
    let kept = UInt64Array::from(kept);
    let take = |rows: &RecordBatch| take_record_batch(rows, &kept).map_err(execution);
    Ok((take(&old)?, new.as_ref().map(take).transpose()?))
}

/// Whether the rows the statement read (`old`) are the rows the transaction
/// reads now (`current`, with which of them the session owns).
enum Unchanged {
    /// All are; the mask marks, row for row of `old`, the ones the session
    /// owns.
    Owned(BooleanArray),
    /// This many had changed or gone.
    Conflict(u64),
}

fn unchanged(
    def: &MutableTableDefinition,
    old: &RecordBatch,
    (current, current_owned): &(RecordBatch, BooleanArray),
) -> Result<Unchanged, BackendError> {
    let current_keys = row_values(&primary_key_of(def, current)?).map_err(execution)?;
    let current_rows = row_values(current).map_err(execution)?;
    let now: HashMap<ValueRow<'_>, (ValueRow<'_>, bool)> = (0..current.num_rows())
        .map(|i| {
            let owned = current_owned.value(i);
            (current_keys.row(i), (current_rows.row(i), owned))
        })
        .collect();
    let old_keys = row_values(&primary_key_of(def, old)?).map_err(execution)?;
    let old_rows = row_values(old).map_err(execution)?;
    let seen: Vec<Option<bool>> = (0..old.num_rows())
        .map(|i| match now.get(&old_keys.row(i)) {
            Some((row, owned)) if *row == old_rows.row(i) => Some(*owned),
            _ => None,
        })
        .collect();
    let changed = seen.iter().filter(|s| s.is_none()).count() as u64;
    Ok(match changed {
        0 => Unchanged::Owned(seen.into_iter().collect()),
        changed => Unchanged::Conflict(changed),
    })
}

/// `rows` in the row format, where two rows compare equal exactly when every
/// value does (nulls equal nulls, as a key or a stored value does).
fn row_values(rows: &RecordBatch) -> Result<Rows, arrow::error::ArrowError> {
    let converter = RowConverter::new(
        rows.schema()
            .fields()
            .iter()
            .map(|f| SortField::new(f.data_type().clone()))
            .collect(),
    )?;
    converter.convert_columns(rows.columns())
}

/// A one-row key batch as `(v1, v2, …)`.
fn key_display(key: &RecordBatch) -> String {
    let values = key
        .columns()
        .iter()
        .map(|column| {
            ArrayFormatter::try_new(column.as_ref(), &FormatOptions::default())
                .map(|f| f.value(0).to_string())
                .unwrap_or_else(|e| e.to_string())
        })
        .collect::<Vec<_>>();
    format!("({})", values.join(", "))
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
