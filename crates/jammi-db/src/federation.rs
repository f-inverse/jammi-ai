//! Federation that keeps the engine's own functions in the engine.
//!
//! `datafusion-federation` hands the largest sub-plan whose scans share one
//! remote to that remote as SQL, whatever functions the sub-plan calls. A
//! function the session installed ([`crate::session::JammiSession::install_functions`])
//! exists only here, so a remote asked to run it fails. [`LocalFunctionFederation`]
//! federates only the maximal sub-plans that call none: a node that calls an
//! engine function — anywhere beneath it, subqueries included — stays in the
//! engine, and each of its inputs is federated on its own.
//!
//! A filter the engine runs above a federated sub-plan stays in the engine
//! too: [`FederatedScanPlanner`] plans each sub-plan behind a
//! [`FederatedScanExec`], which takes no filter from the plan above it.

//! The same holds for what a database source would read differently from the
//! engine ([`remote`]): a node a source's SQL would carry with another
//! meaning is not federated, and its inputs are federated on their own.

use std::collections::HashSet;
use std::fmt;
use std::sync::{Arc, PoisonError, RwLock};

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::Statistics;
use datafusion::config::ConfigOptions;
use datafusion::error::Result;
use datafusion::execution::context::SessionState;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNode};
use datafusion::optimizer::optimizer::{OptimizerConfig, OptimizerRule};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::execution_plan::CardinalityEffect;
use datafusion::physical_plan::filter_pushdown::{FilterDescription, FilterPushdownPhase};
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion::physical_planner::{ExtensionPlanner, PhysicalPlanner};
use datafusion_federation::{FederatedPlanner, FederationOptimizerRule};

#[cfg(any(feature = "postgres", feature = "mysql"))]
pub(crate) mod remote;

/// The names of the functions a session installed: what no remote can run.
#[derive(Debug, Default)]
pub(crate) struct EngineFunctions(RwLock<HashSet<String>>);

impl EngineFunctions {
    /// Record the function called `name`, or any of its `aliases`, as the
    /// engine's own.
    pub(crate) fn insert(&self, name: &str, aliases: &[String]) {
        self.0
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .extend(std::iter::once(name.to_string()).chain(aliases.iter().cloned()));
    }

    fn contains(&self, name: &str) -> bool {
        self.0
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .contains(name)
    }

    /// Whether `plan` calls one of these functions anywhere, subqueries
    /// included.
    fn called_by(&self, plan: &LogicalPlan) -> Result<bool> {
        let mut called = false;
        plan.apply_with_subqueries(|node| {
            for expr in node.expressions() {
                expr.apply(|e| {
                    called = function_name(e).is_some_and(|name| self.contains(name));
                    Ok(if called {
                        TreeNodeRecursion::Stop
                    } else {
                        TreeNodeRecursion::Continue
                    })
                })?;
                if called {
                    return Ok(TreeNodeRecursion::Stop);
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        Ok(called)
    }
}

/// The function `expr` calls at its root, by the name it is registered under.
fn function_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::ScalarFunction(f) => Some(f.func.name()),
        Expr::AggregateFunction(f) => Some(f.func.name()),
        Expr::WindowFunction(f) => Some(f.fun.name()),
        _ => None,
    }
}

/// Whether every database source `plan` scans runs it with the engine's
/// meaning ([`remote::runs`]); a build without database sources has none.
fn remotes_run(plan: &LogicalPlan) -> Result<bool> {
    #[cfg(any(feature = "postgres", feature = "mysql"))]
    return remote::runs(plan);
    #[cfg(not(any(feature = "postgres", feature = "mysql")))]
    {
        let _ = plan;
        Ok(true)
    }
}

/// The federation optimizer rule, applied only to sub-plans that call no
/// engine function and that their sources run with the engine's meaning.
#[derive(Debug)]
pub(crate) struct LocalFunctionFederation {
    functions: Arc<EngineFunctions>,
    federation: FederationOptimizerRule,
}

impl LocalFunctionFederation {
    pub(crate) fn new(functions: Arc<EngineFunctions>) -> Self {
        Self {
            functions,
            federation: FederationOptimizerRule::new(),
        }
    }
}

impl OptimizerRule for LocalFunctionFederation {
    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        plan.transform_down(|node| {
            if self.functions.called_by(&node)? || !remotes_run(&node)? {
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
        "local_function_federation"
    }
}

/// Federation's physical planner, each federated sub-plan it plans behind a
/// [`FederatedScanExec`].
pub(crate) struct FederatedScanPlanner(FederatedPlanner);

impl fmt::Debug for FederatedScanPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("FederatedScanPlanner")
    }
}

impl FederatedScanPlanner {
    pub(crate) fn new() -> Self {
        Self(FederatedPlanner::new())
    }
}

#[async_trait::async_trait]
impl ExtensionPlanner for FederatedScanPlanner {
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
        Ok(federated.map(|plan| Arc::new(FederatedScanExec::new(plan)) as _))
    }
}

/// A federated sub-plan, which takes no filter from the plan above it.
///
/// Physical filter pushdown offers a `FilterExec`'s predicate to the plan
/// beneath it and drops the `FilterExec` once that plan takes it.
/// `datafusion-federation`'s `VirtualExecutionPlan` takes every predicate
/// offered and hands it to its SQL executor, and the executor
/// `datafusion-table-providers` gives a database source ignores it: the
/// predicate is neither in the statement the database runs nor applied to the
/// rows it returns. A filter the engine runs above a federated sub-plan — one
/// on a database column in a query that also joins a file — would be lost
/// that way. Above this node it stays in its `FilterExec`: a plan takes part
/// in filter pushdown only by opting in, and this one does not. It is
/// otherwise its input, unchanged.
#[derive(Debug)]
pub(crate) struct FederatedScanExec {
    input: Arc<dyn ExecutionPlan>,
}

impl FederatedScanExec {
    fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        Self { input }
    }
}

impl DisplayAs for FederatedScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FederatedScanExec")
    }
}

impl ExecutionPlan for FederatedScanExec {
    fn name(&self) -> &str {
        "FederatedScanExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(children.swap_remove(0))))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.input.execute(partition, context)
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<Statistics>> {
        self.input.partition_statistics(partition)
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }

    /// No filter from above reaches the federated sub-plan (see the type's
    /// doc); written out, not left to the default, so a change to the
    /// default cannot quietly change it.
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::functions::expr_fn::{lower, upper};
    use datafusion::logical_expr::{lit, LogicalPlanBuilder};

    fn plan(expr: Expr) -> LogicalPlan {
        LogicalPlanBuilder::empty(true)
            .project([expr])
            .unwrap()
            .build()
            .unwrap()
    }

    #[test]
    fn a_plan_calls_an_installed_function_by_its_name_at_any_depth() {
        let nested = plan(lower(upper(lit("a"))));
        let functions = EngineFunctions::default();
        assert!(!functions.called_by(&nested).unwrap());
        functions.insert("upper", &[]);
        assert!(functions.called_by(&nested).unwrap());
        assert!(!functions.called_by(&plan(lower(lit("a")))).unwrap());
    }
}
