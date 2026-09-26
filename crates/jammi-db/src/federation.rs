//! Federation that keeps the engine's own functions in the engine.
//!
//! `datafusion-federation` hands the largest sub-plan whose scans share one
//! remote to that remote as SQL, whatever functions the sub-plan calls. A
//! function the session installed ([`crate::session::JammiSession::install_functions`])
//! exists only here, so a remote asked to run it fails. [`LocalFunctionFederation`]
//! federates only the maximal sub-plans that call none: a node that calls an
//! engine function — anywhere beneath it, subqueries included — stays in the
//! engine, and each of its inputs is federated on its own.

use std::collections::HashSet;
use std::sync::{Arc, PoisonError, RwLock};

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::error::Result;
use datafusion::logical_expr::{Expr, LogicalPlan};
use datafusion::optimizer::optimizer::{OptimizerConfig, OptimizerRule};
use datafusion_federation::FederationOptimizerRule;

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

/// The federation optimizer rule, applied only to sub-plans that call no
/// engine function.
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
            if self.functions.called_by(&node)? {
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
