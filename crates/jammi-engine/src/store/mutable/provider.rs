//! `TableProvider` implementation for mutable companion tables.
//!
//! Phase 2 ships a minimum viable provider that supports `scan` (full-table
//! reads) and `insert_into` (DataFusion DML through [`MutableTableSink`]).
//! Predicate pushdown, projection, and limit are translated to backend SQL
//! when straightforward; otherwise DataFusion's planner handles them above
//! the scan node.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, Float32Array, Float64Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::datasource::sink::DataSinkExec;
use datafusion::datasource::{MemTable, TableType};
use datafusion::error::DataFusionError;
use datafusion::logical_expr::dml::InsertOp;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::Expr;

use crate::catalog::backend::{Row, TxOptions};

use super::definition::MutableTableDefinition;
use super::sink::MutableTableSink;
use super::MutableBackend;

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
        // Always read the full schema; let DataFusion apply column projection.
        let projected_cols: Vec<&str> = self
            .def
            .schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        let projected_schema: SchemaRef = Arc::clone(&self.def.schema);

        // Inject the tenant-scope predicate at the backend SQL layer so we
        // ship the correct row set off the SQLite/Postgres side, not just
        // the union (which DataFusion's AnalyzerRule would also filter, but
        // at the cost of materializing the full table first).
        let tenant_pred = match self.tenant.current_tenant() {
            Some(t) => Some(format!("(\"tenant_id\" = '{t}' OR \"tenant_id\" IS NULL)")),
            None => Some("\"tenant_id\" IS NULL".to_string()),
        };
        let sql = self
            .backend
            .scan_dml(&self.def, &projected_cols, tenant_pred.as_deref(), limit);

        // Build an owned copy of the SQL and column descriptors so the closure
        // can capture them with `'static` lifetimes.
        let owned_sql = sql.clone();
        let columns: Vec<(String, DataType)> = projected_schema
            .fields()
            .iter()
            .map(|f| (f.name().clone(), f.data_type().clone()))
            .collect();

        let rows_per_col: Vec<Vec<DecodedValue>> = self
            .backend
            .catalog_backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    let columns = columns.clone();
                    let owned_sql = owned_sql.clone();
                    Box::pin(async move {
                        let raw = tx
                            .query(&owned_sql, &[], |row| decode_row(row, &columns))
                            .await?;
                        // Transpose Vec<Row> → Vec<Column>
                        let mut transposed: Vec<Vec<DecodedValue>> = (0..columns.len())
                            .map(|_| Vec::with_capacity(raw.len()))
                            .collect();
                        for r in raw {
                            for (i, v) in r.into_iter().enumerate() {
                                transposed[i].push(v);
                            }
                        }
                        Ok(transposed)
                    })
                },
            )
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        let arrays = build_arrays(&columns, rows_per_col)?;
        RecordBatch::try_new(projected_schema, arrays)
            .map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

#[async_trait]
impl TableProvider for MutableTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

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
        if !matches!(insert_op, InsertOp::Append | InsertOp::Replace) {
            return Err(DataFusionError::NotImplemented(format!(
                "InsertOp {insert_op:?} not supported on mutable tables; \
                 use Append or Replace (see SPEC-02 §13 OQ#5)"
            )));
        }
        let sink = Arc::new(MutableTableSink::new(
            Arc::clone(&self.def),
            Arc::clone(&self.backend),
            self.tenant.clone(),
        ));
        Ok(Arc::new(DataSinkExec::new(input, sink, None)))
    }
}

/// One column value read from a backend row, after type-aware extraction.
#[derive(Debug, Clone)]
enum DecodedValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
}

fn decode_row(
    row: &Row<'_>,
    columns: &[(String, DataType)],
) -> Result<Vec<DecodedValue>, crate::catalog::backend::BackendError> {
    columns
        .iter()
        .map(|(name, ty)| match ty {
            DataType::Boolean => row
                .try_get::<bool>(name)?
                .map(DecodedValue::Bool)
                .map(Ok)
                .unwrap_or(Ok(DecodedValue::Null)),
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => row
                .try_get::<i64>(name)?
                .map(DecodedValue::Int)
                .map(Ok)
                .unwrap_or(Ok(DecodedValue::Null)),
            DataType::Float16 | DataType::Float32 | DataType::Float64 => row
                .try_get::<f64>(name)?
                .map(DecodedValue::Float)
                .map(Ok)
                .unwrap_or(Ok(DecodedValue::Null)),
            _ => row
                .try_get::<String>(name)?
                .map(DecodedValue::Text)
                .map(Ok)
                .unwrap_or(Ok(DecodedValue::Null)),
        })
        .collect()
}

fn build_arrays(
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
                DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64 => {
                    let arr: Int64Array = values
                        .into_iter()
                        .map(|v| match v {
                            DecodedValue::Int(i) => Some(i),
                            _ => None,
                        })
                        .collect();
                    Ok(Arc::new(arr) as ArrayRef)
                }
                DataType::Float32 => {
                    let arr: Float32Array = values
                        .into_iter()
                        .map(|v| match v {
                            DecodedValue::Float(f) => Some(f as f32),
                            _ => None,
                        })
                        .collect();
                    Ok(Arc::new(arr) as ArrayRef)
                }
                DataType::Float64 => {
                    let arr: Float64Array = values
                        .into_iter()
                        .map(|v| match v {
                            DecodedValue::Float(f) => Some(f),
                            _ => None,
                        })
                        .collect();
                    Ok(Arc::new(arr) as ArrayRef)
                }
                _ => {
                    let arr: StringArray = values
                        .into_iter()
                        .map(|v| match v {
                            DecodedValue::Text(s) => Some(s),
                            _ => None,
                        })
                        .collect();
                    Ok(Arc::new(arr) as ArrayRef)
                }
            }
        })
        .collect()
}
