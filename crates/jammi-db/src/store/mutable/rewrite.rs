//! `UPDATE` / `DELETE` on a mutable table: the rows the statement's own plan
//! selects, rewritten.
//!
//! DataFusion hands a table provider a DML statement's predicate only as
//! filters over the target's own columns, so a subquery, a join or a `LIMIT`
//! would be lost. Here the statement's input plan runs as a query instead —
//! whatever it joins, filters or limits — and yields each selected row as it
//! was read and, for an `UPDATE`, as it becomes. The provider then rewrites
//! exactly those rows (`MutableTableProvider::rewrite_rows`).

use std::any::Any;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};

use arrow::compute::concat_batches;
use arrow::record_batch::RecordBatch;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::common::{Column, DFSchema, DFSchemaRef, Result as DfResult, TableReference};
use datafusion::datasource::DefaultTableSource;
use datafusion::error::DataFusionError;
use datafusion::execution::context::SessionState;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::{
    DmlStatement, Expr, Extension, LogicalPlan, Projection, TableSource, UserDefinedLogicalNode,
    UserDefinedLogicalNodeCore, WriteOp,
};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    collect, DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use datafusion::physical_planner::{ExtensionPlanner, PhysicalPlanner};

use super::provider::{count_batch, MutableTableProvider};

/// What a rewrite does to the rows it selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd)]
pub enum Rewrite {
    /// Remove them.
    Delete,
    /// Replace each with its new value.
    Update,
}

/// The logical node an `UPDATE` / `DELETE` on a mutable table plans to.
///
/// Its input yields one row per selected row: the target's columns as read,
/// then — for an [`Rewrite::Update`] — the target's columns as they become.
/// It answers with DataFusion's DML shape, one `count` row.
pub struct RowRewriteNode {
    table: TableReference,
    target: Arc<MutableTableProvider>,
    rewrite: Rewrite,
    input: LogicalPlan,
}

impl RowRewriteNode {
    /// The rewrite `dml` is, when it is an `UPDATE` / `DELETE` whose target
    /// is a mutable table; otherwise `dml` back, unchanged.
    pub fn of(dml: DmlStatement) -> DfResult<std::result::Result<Self, DmlStatement>> {
        let rewrite = match dml.op {
            WriteOp::Delete => Rewrite::Delete,
            WriteOp::Update => Rewrite::Update,
            _ => return Ok(Err(dml)),
        };
        let Some(target) = mutable_target(dml.target.as_ref()) else {
            return Ok(Err(dml));
        };
        let input = Arc::unwrap_or_clone(dml.input);
        let input = match rewrite {
            Rewrite::Delete => input,
            Rewrite::Update => read_and_written(input, target.def.schema.fields().len())?,
        };
        Ok(Ok(Self {
            table: dml.table_name,
            target,
            rewrite,
            input,
        }))
    }

    /// The node as a plan.
    pub fn into_plan(self) -> LogicalPlan {
        LogicalPlan::Extension(Extension {
            node: Arc::new(self),
        })
    }
}

/// The mutable table behind `source`, when it is one.
fn mutable_target(source: &dyn TableSource) -> Option<Arc<MutableTableProvider>> {
    let provider = Arc::clone(
        &(source as &dyn Any)
            .downcast_ref::<DefaultTableSource>()?
            .table_provider,
    );
    (provider as Arc<dyn Any + Send + Sync>)
        .downcast::<MutableTableProvider>()
        .ok()
}

/// An `UPDATE`'s input — a projection of each target column's new value
/// over the rows it selects — widened to yield each row's `width` target
/// columns as read ahead of the new values. The SQL planner puts the target
/// relation leftmost (`UPDATE t … FROM u` joins `t` to `u`), so the target's
/// columns as read are the first `width` columns under the projection.
fn read_and_written(input: LogicalPlan, width: usize) -> DfResult<LogicalPlan> {
    let LogicalPlan::Projection(projection) = input else {
        return Err(DataFusionError::Internal(format!(
            "an UPDATE's input is a projection of the new values, got {}",
            input.display()
        )));
    };
    let selected = projection.input.schema();
    if selected.fields().len() < width || projection.expr.len() != width {
        return Err(DataFusionError::Internal(format!(
            "an UPDATE of a {width}-column table projects {} value(s) over {} column(s)",
            projection.expr.len(),
            selected.fields().len()
        )));
    }
    let read = (0..width).map(|i| {
        let (qualifier, field) = selected.qualified_field(i);
        Expr::Column(Column::from((qualifier, field))).alias_qualified(Some("read"), field.name())
    });
    let written = projection.expr.into_iter().map(|e| {
        let name = e.schema_name().to_string();
        e.unalias().alias_qualified(Some("written"), name)
    });
    Ok(LogicalPlan::Projection(Projection::try_new(
        read.chain(written).collect(),
        projection.input,
    )?))
}

/// The one-column `count` schema a DML statement answers with.
fn count_schema() -> &'static DFSchemaRef {
    static COUNT: OnceLock<DFSchemaRef> = OnceLock::new();
    COUNT.get_or_init(|| {
        let schema = Schema::new(vec![Field::new("count", DataType::UInt64, false)]);
        Arc::new(DFSchema::try_from(schema).expect("a one-field schema is valid"))
    })
}

impl fmt::Debug for RowRewriteNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RowRewriteNode")
            .field("table", &self.table)
            .field("rewrite", &self.rewrite)
            .finish_non_exhaustive()
    }
}

impl PartialEq for RowRewriteNode {
    fn eq(&self, other: &Self) -> bool {
        (&self.table, self.rewrite, &self.input) == (&other.table, other.rewrite, &other.input)
    }
}

impl Eq for RowRewriteNode {}

impl Hash for RowRewriteNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (&self.table, self.rewrite, &self.input).hash(state);
    }
}

impl PartialOrd for RowRewriteNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (&self.table, self.rewrite, &self.input).partial_cmp(&(
            &other.table,
            other.rewrite,
            &other.input,
        ))
    }
}

impl UserDefinedLogicalNodeCore for RowRewriteNode {
    fn name(&self) -> &str {
        "RowRewrite"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        count_schema()
    }

    fn expressions(&self) -> Vec<Expr> {
        Vec::new()
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RowRewrite: {:?} {}", self.rewrite, self.table)
    }

    fn with_exprs_and_inputs(
        &self,
        _exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> DfResult<Self> {
        let (Some(input), true) = (inputs.pop(), inputs.is_empty()) else {
            return Err(DataFusionError::Internal(
                "RowRewrite has exactly one input".into(),
            ));
        };
        Ok(Self {
            table: self.table.clone(),
            target: Arc::clone(&self.target),
            rewrite: self.rewrite,
            input,
        })
    }
}

/// Plans [`RowRewriteNode`] into [`RowRewriteExec`]; every other extension
/// node is left to the next planner.
#[derive(Debug, Default)]
pub struct RowRewritePlanner;

#[async_trait]
impl ExtensionPlanner for RowRewritePlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        _session_state: &SessionState,
    ) -> DfResult<Option<Arc<dyn ExecutionPlan>>> {
        let Some(node) = node.as_any().downcast_ref::<RowRewriteNode>() else {
            return Ok(None);
        };
        let [input] = physical_inputs else {
            return Err(DataFusionError::Internal(
                "RowRewrite has exactly one input".into(),
            ));
        };
        Ok(Some(Arc::new(RowRewriteExec::new(
            Arc::clone(&node.target),
            node.rewrite,
            Arc::clone(input),
        ))))
    }
}

/// The physical node a rewrite runs as: its input run to completion, then
/// the selected rows rewritten in one transaction. Emits the `count` row.
pub struct RowRewriteExec {
    target: Arc<MutableTableProvider>,
    rewrite: Rewrite,
    input: Arc<dyn ExecutionPlan>,
    properties: Arc<PlanProperties>,
}

impl fmt::Debug for RowRewriteExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RowRewriteExec")
            .field("table", &self.target.def.id.as_str())
            .field("rewrite", &self.rewrite)
            .finish_non_exhaustive()
    }
}

impl RowRewriteExec {
    fn new(
        target: Arc<MutableTableProvider>,
        rewrite: Rewrite,
        input: Arc<dyn ExecutionPlan>,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(count_schema().inner().clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self {
            target,
            rewrite,
            input,
            properties,
        }
    }

    async fn run(
        target: Arc<MutableTableProvider>,
        rewrite: Rewrite,
        input: Arc<dyn ExecutionPlan>,
        context: Arc<TaskContext>,
    ) -> DfResult<RecordBatch> {
        let schema = input.schema();
        let selected = concat_batches(&schema, &collect(input, context).await?)?;
        let width = target.def.schema.fields().len();
        let rows = |from: usize| {
            RecordBatch::try_new(
                Arc::clone(&target.def.schema),
                selected.columns()[from..from + width].to_vec(),
            )
        };
        let old = rows(0)?;
        let new = match rewrite {
            Rewrite::Delete => None,
            Rewrite::Update => Some(rows(width)?),
        };
        let rewritten = target
            .rewrite_rows(old, new)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        count_batch(rewritten)
    }
}

impl DisplayAs for RowRewriteExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "RowRewriteExec: {:?} {}",
            self.rewrite,
            self.target.def.id.as_str()
        )
    }
}

impl ExecutionPlan for RowRewriteExec {
    fn name(&self) -> &str {
        "RowRewriteExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let (Some(input), true) = (children.pop(), children.is_empty()) else {
            return Err(DataFusionError::Internal(
                "RowRewriteExec has exactly one child".into(),
            ));
        };
        Ok(Arc::new(Self::new(
            Arc::clone(&self.target),
            self.rewrite,
            input,
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "RowRewriteExec has one output partition, partition {partition} was asked for"
            )));
        }
        let schema: SchemaRef = self.schema();
        let run = Self::run(
            Arc::clone(&self.target),
            self.rewrite,
            Arc::clone(&self.input),
            context,
        );
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::once(run),
        )))
    }
}
