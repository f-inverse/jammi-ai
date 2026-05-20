//! `DataSink` implementation for mutable companion tables.
//!
//! Per DataFusion: *"This method will be called exactly once during each DML
//! statement. Thus prior to return, the sink should do any commit or rollback
//! required."* We wrap the entire write in one
//! [`CatalogBackend::transaction`](crate::catalog::backend::CatalogBackend::transaction)
//! closure. Each [`RecordBatch`] is translated into a multi-row
//! `INSERT … VALUES (…), (…), …` statement built from the backend's
//! [`MutableBackend::insert_dml`](crate::store::mutable::MutableBackend::insert_dml)
//! renderer.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::array::{
    Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
};
use arrow::record_batch::RecordBatch;
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::common::DataFusionError;
use datafusion::datasource::sink::DataSink;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::DisplayAs;
use datafusion::physical_plan::DisplayFormatType;
use futures::StreamExt;

use crate::catalog::backend::{SqlValue, TxOptions};

use super::definition::MutableTableDefinition;
use super::MutableBackend;

pub struct MutableTableSink {
    def: Arc<MutableTableDefinition>,
    backend: Arc<dyn MutableBackend>,
}

impl MutableTableSink {
    pub fn new(def: Arc<MutableTableDefinition>, backend: Arc<dyn MutableBackend>) -> Self {
        Self { def, backend }
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
    fn as_any(&self) -> &dyn Any {
        self
    }

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
        let written = self
            .backend
            .catalog_backend()
            .transaction(TxOptions::default(), move |tx| {
                let backend = backend_for_closure;
                Box::pin(async move {
                    let mut total: u64 = 0;
                    for batch in batches {
                        if batch.num_rows() == 0 {
                            continue;
                        }
                        let schema = batch.schema();
                        let col_names: Vec<String> =
                            schema.fields().iter().map(|f| f.name().clone()).collect();
                        let cols: Vec<&str> = col_names.iter().map(String::as_str).collect();
                        let dml = backend.insert_dml(&def, &cols, batch.num_rows());
                        let params = batch_to_params(&batch).map_err(|e| {
                            crate::catalog::backend::BackendError::Execution(e.to_string())
                        })?;
                        let rows = tx.execute(&dml, &params).await?;
                        total += rows;
                    }
                    Ok(total)
                })
            })
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        Ok(written)
    }
}

/// Translate every cell of a `RecordBatch` into the engine's [`SqlValue`]
/// taxonomy, in row-major order. The implicit `tenant_id` slot is appended
/// per row as `NULL` (Phase 3 binds the session's tenant here).
fn batch_to_params(batch: &RecordBatch) -> Result<Vec<SqlValue<'static>>, &'static str> {
    let n_rows = batch.num_rows();
    let arrays: Vec<&dyn Array> = batch.columns().iter().map(|c| c.as_ref()).collect();
    let mut out = Vec::with_capacity(n_rows * (arrays.len() + 1));
    for r in 0..n_rows {
        for (col_idx, arr) in arrays.iter().enumerate() {
            let value = extract_value(*arr, r, batch.schema().field(col_idx).data_type())?;
            out.push(value);
        }
        // Implicit tenant_id placeholder for Phase 3.
        out.push(SqlValue::Null);
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
        return Ok(SqlValue::Null);
    }
    match ty {
        Boolean => arr
            .as_any()
            .downcast_ref::<BooleanArray>()
            .map(|a| SqlValue::Bool(a.value(idx)))
            .ok_or("expected BooleanArray"),
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
        _ => Err("unsupported arrow type for mutable-table insert"),
    }
}
