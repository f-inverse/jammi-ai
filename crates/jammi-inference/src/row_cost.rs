//! `RowCostExec` — every row's cost, appended as it streams.
//!
//! A streaming map over one input partition that appends [`COST_COLUMN`]: a
//! non-null `UInt32`, the row's length along the axis the model's forward
//! pads ([`BoundModel::row_costs`]) — its truncated token count for a text
//! task, one for a fixed-shape input. The numbered input
//! ([`crate::numbered`]) composes it privately: it orders rows by
//! cost and cuts forward chunks under a budget from it, and the cost never
//! leaves that node.
//!
//! Computing the cost is host work over the model's own tokenizer, so the
//! node binds the model through the process's runtime at its first poll, as
//! `InferenceExec` does. It maintains its input's order and
//! partitioning.

use std::fmt::{self, Formatter};
use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
};
use futures::{StreamExt, TryStreamExt};

use crate::columns::extract_columns;
use crate::runtime::{BoundModel, InferenceRuntime};
use crate::spec::InferenceSpec;

/// The row-cost column this node appends. Private to the numbered input.
pub const COST_COLUMN: &str = "_cost";

/// The cost-appending map. See the module doc.
pub struct RowCostExec {
    input: Arc<dyn ExecutionPlan>,
    spec: InferenceSpec,
    runtime: InferenceRuntime,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl fmt::Debug for RowCostExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("RowCostExec")
            .field("spec", &self.spec)
            .finish_non_exhaustive()
    }
}

impl RowCostExec {
    /// Append the cost of each row of `input` under `spec`'s model and task.
    /// Refuses an input that already carries [`COST_COLUMN`] or lacks one of
    /// `spec.content_columns`.
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        spec: InferenceSpec,
        runtime: InferenceRuntime,
    ) -> DfResult<Self> {
        let input_schema = input.schema();
        if input_schema.field_with_name(COST_COLUMN).is_ok() {
            return Err(DataFusionError::Plan(format!(
                "RowCostExec: input schema already has a '{COST_COLUMN}' column"
            )));
        }
        for column in &spec.content_columns {
            input_schema.field_with_name(column).map_err(|_| {
                DataFusionError::Plan(format!(
                    "RowCostExec: content column '{column}' is not in the input schema"
                ))
            })?;
        }
        let schema: SchemaRef = Arc::new(Schema::new(
            input_schema
                .fields()
                .iter()
                .map(|f| f.as_ref().clone())
                .chain(std::iter::once(Field::new(
                    COST_COLUMN,
                    DataType::UInt32,
                    false,
                )))
                .collect::<Vec<_>>(),
        ));
        // The input's ordering holds over the extended schema: its columns
        // keep their indexes, the cost is appended after them.
        let mut eq = EquivalenceProperties::new(Arc::clone(&schema));
        if let Some(ordering) = input.output_ordering() {
            eq.add_ordering(ordering.clone());
        }
        let properties = PlanProperties::new(
            eq,
            input.output_partitioning().clone(),
            input.pipeline_behavior(),
            input.boundedness(),
        );
        Ok(Self {
            input,
            spec,
            runtime,
            schema,
            properties: Arc::new(properties),
        })
    }
}

/// `batch` with the cost of each row appended.
fn with_costs(
    model: &dyn BoundModel,
    spec: &InferenceSpec,
    schema: &SchemaRef,
    batch: RecordBatch,
) -> DfResult<RecordBatch> {
    let content = extract_columns(&batch, &spec.content_columns).map_err(DataFusionError::from)?;
    let costs = tracing::debug_span!("input.cost", rows = batch.num_rows())
        .in_scope(|| model.row_costs(&content, spec.task))
        .map_err(DataFusionError::from)?;
    let costs: ArrayRef = Arc::new(UInt32Array::from(costs));
    let columns = batch
        .columns()
        .iter()
        .cloned()
        .chain(std::iter::once(costs))
        .collect();
    Ok(RecordBatch::try_new(Arc::clone(schema), columns)?)
}

impl DisplayAs for RowCostExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "RowCostExec: model={}, task={:?}, columns={:?}",
            self.spec.source, self.spec.task, self.spec.content_columns
        )
    }
}

impl ExecutionPlan for RowCostExec {
    fn name(&self) -> &str {
        "RowCostExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::try_new(
            Arc::clone(&children[0]),
            self.spec.clone(),
            self.runtime.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let schema = Arc::clone(&self.schema);
        let spec = self.spec.clone();
        let runtime = Arc::clone(&self.runtime.model);
        let costed = async move {
            let model = runtime
                .bind(&spec.source, spec.task)
                .await
                .map_err(DataFusionError::from)?;
            Ok::<_, DataFusionError>(input.map(move |batch| {
                batch.and_then(|b| with_costs(model.as_ref(), &spec, &schema, b))
            }))
        };
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            futures::stream::once(costed).try_flatten(),
        )))
    }
}
