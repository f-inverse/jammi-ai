//! Federation that sends a database source only what it runs as the engine
//! does.
//!
//! `datafusion-federation` hands the largest sub-plan whose scans share one
//! remote to that remote as SQL. SQL that reads the same can mean something
//! else there, and a function the engine has — DataFusion's own or one the
//! session installed — may not exist there at all. [`RemoteFederation`]
//! therefore federates only the maximal sub-plans [`remote::federable`]
//! admits: ones that scan a remote, every node of which every remote they
//! scan runs with the engine's meaning (each remote's allowlist, which names
//! DataFusion's own implementations, so an installed function is never on
//! one). Any other node stays in the engine, and each of its inputs is
//! federated on its own.
//!
//! The sub-plan a remote runs is planned as a [`RemoteExec`]: a source of rows
//! to the engine, whose plan no engine optimizer reaches into. A filter or a
//! sort the engine runs above a remote's rows stays in the engine, whether
//! the rows come from a federated sub-plan or from a remote table's own scan;
//! only a limit reaches the remote.

use std::fmt;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::Statistics;
use datafusion::config::ConfigOptions;
use datafusion::error::Result;
use datafusion::execution::context::SessionState;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNode};
use datafusion::optimizer::optimizer::{OptimizerConfig, OptimizerRule};
use datafusion::physical_expr::{PhysicalExpr, PhysicalSortExpr};
use datafusion::physical_plan::filter_pushdown::{FilterDescription, FilterPushdownPhase};
use datafusion::physical_plan::sort_pushdown::SortOrderPushdownResult;
use datafusion::physical_plan::{
    displayable, DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
};
use datafusion::physical_planner::{ExtensionPlanner, PhysicalPlanner};
use datafusion_federation::{FederatedPlanner, FederationOptimizerRule};

pub(crate) mod remote;

/// The federation optimizer rule, applied only to the sub-plans
/// [`remote::federable`] admits.
#[derive(Debug, Default)]
pub(crate) struct RemoteFederation {
    federation: FederationOptimizerRule,
}

impl OptimizerRule for RemoteFederation {
    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        plan.transform_down(|node| {
            if !remote::federable(&node)? {
                return Ok(Transformed::no(node));
            }
            let federated = self.federation.rewrite(node, config)?;
            Ok(Transformed::new(
                federated.data,
                federated.transformed,
                TreeNodeRecursion::Jump,
            ))
        })
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "remote_federation"
    }
}

/// Federation's physical planner, each federated sub-plan it plans behind a
/// [`RemoteExec`].
pub(crate) struct RemotePlanner(FederatedPlanner);

impl fmt::Debug for RemotePlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RemotePlanner")
    }
}

impl RemotePlanner {
    pub(crate) fn new() -> Self {
        Self(FederatedPlanner::new())
    }
}

#[async_trait::async_trait]
impl ExtensionPlanner for RemotePlanner {
    async fn plan_extension(
        &self,
        planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let federated = self
            .0
            .plan_extension(
                planner,
                node,
                logical_inputs,
                physical_inputs,
                session_state,
            )
            .await?;
        Ok(federated.map(|plan| Arc::new(RemoteExec::new(plan)) as _))
    }
}

/// Rows a remote produces — a federated sub-plan's, or a remote table's own
/// scan — as a source to the engine.
///
/// The remote's plan is held, not exposed as a child: no engine optimizer
/// reaches into it. Physical filter pushdown would hand it a `FilterExec`'s
/// predicate and drop the `FilterExec`; sort enforcement would move a
/// `SortExec` beneath any node that keeps its input's order, onto a remote
/// scan whose own sort pushdown writes the sort into its statement. A
/// remote's plan writes what it takes in the remote's dialect and with the
/// remote's meaning — or, for `datafusion-federation`'s plan, takes a
/// predicate and applies it nowhere. Above this node both stay in the engine.
/// The filters a remote runs as the engine does reach it at planning, through
/// the remote's own provider. A limit is the remote's to apply — any `n` rows
/// are any `n` rows — and is the one thing this node passes down.
#[derive(Debug)]
pub(crate) struct RemoteExec {
    remote: Arc<dyn ExecutionPlan>,
}

impl RemoteExec {
    pub(crate) fn new(remote: Arc<dyn ExecutionPlan>) -> Self {
        Self { remote }
    }
}

/// The remote's plan on one line, outermost node first, down to the
/// statement the remote is sent.
impl DisplayAs for RemoteExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        let plan = displayable(self.remote.as_ref()).indent(false).to_string();
        let nodes = plan.lines().map(str::trim).collect::<Vec<_>>();
        write!(f, "RemoteExec: {}", nodes.join(" <- "))
    }
}

impl ExecutionPlan for RemoteExec {
    fn name(&self) -> &str {
        "RemoteExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.remote.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.remote.execute(partition, context)
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<Statistics>> {
        self.remote.partition_statistics(partition)
    }

    fn supports_limit_pushdown(&self) -> bool {
        self.remote.supports_limit_pushdown()
    }

    fn fetch(&self) -> Option<usize> {
        self.remote.fetch()
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        let remote = self.remote.with_fetch(limit)?;
        Some(Arc::new(Self::new(remote)))
    }

    /// No filter from above reaches the remote; written out, not left to the
    /// default, so a change to the default cannot quietly change it.
    fn gather_filters_for_pushdown(
        &self,
        _phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> Result<FilterDescription> {
        Ok(FilterDescription::all_unsupported(
            &parent_filters,
            &self.children(),
        ))
    }

    /// No sort from above reaches the remote; written out for the same reason.
    fn try_pushdown_sort(
        &self,
        _order: &[PhysicalSortExpr],
    ) -> Result<SortOrderPushdownResult<Arc<dyn ExecutionPlan>>> {
        Ok(SortOrderPushdownResult::Unsupported)
    }
}
