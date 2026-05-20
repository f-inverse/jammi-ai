//! Predicate-injection analyzer rule for tenant-scoped queries.
//!
//! At plan-analysis time (before optimization), every [`TableScan`] whose
//! schema declares a `tenant_id` column — or whose source registration
//! declares a `tenant_column` override — is wrapped in a [`Filter`] that
//! emits `tenant_id = $current_tenant OR tenant_id IS NULL` when the
//! session is `Scoped(t)`, or `tenant_id IS NULL` when `Unscoped`.
//!
//! Choice of `AnalyzerRule` over `OptimizerRule`: DataFusion's own rustdoc
//! pins this — *"to modify the semantics of a LogicalPlan, one should
//! implement an AnalyzerRule instead"*. Tenant scope changes *which rows the
//! query returns*; that's semantics, not optimization.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::config::ConfigOptions;
use datafusion::error::Result as DfResult;
use datafusion::logical_expr::{BinaryExpr, Expr, Filter, LogicalPlan, Operator, TableScan};
use datafusion::optimizer::AnalyzerRule;
use datafusion::scalar::ScalarValue;

use crate::tenant::TenantContext;

/// Source-name → optional `tenant_column` lookup. Populated from the catalog
/// at session construction; the analyzer consults it when a `TableScan`'s
/// schema does *not* itself declare a `tenant_id` column (i.e., federated
/// external sources whose tenant discriminator has a different name).
#[derive(Debug, Default)]
pub struct SourceTenantColumns {
    inner: RwLock<HashMap<String, Option<String>>>,
}

impl SourceTenantColumns {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&self, source: &str, column: Option<String>) {
        self.inner
            .write()
            .expect("source tenant columns lock poisoned")
            .insert(source.to_string(), column);
    }

    /// Returns the tenant-discriminator column registered for `source_name`,
    /// if any. Used by the rewriter to translate scans on federated sources.
    pub fn tenant_column(&self, source_name: &str) -> Option<String> {
        self.inner
            .read()
            .expect("source tenant columns lock poisoned")
            .get(source_name)
            .cloned()
            .flatten()
    }
}

/// Per-session shared tenant state read by the analyzer rule on every plan
/// analysis. Sessions update this via [`crate::session::JammiSession::with_tenant`].
pub type TenantBinding = Arc<RwLock<TenantContext>>;

/// `AnalyzerRule` that wraps every tenant-aware `TableScan` in a `Filter`
/// emitting the session's tenant predicate.
#[derive(Debug)]
pub struct TenantScopeAnalyzerRule {
    binding: TenantBinding,
    source_columns: Arc<SourceTenantColumns>,
}

impl TenantScopeAnalyzerRule {
    pub fn new(binding: TenantBinding, source_columns: Arc<SourceTenantColumns>) -> Self {
        Self {
            binding,
            source_columns,
        }
    }

    fn current_context(&self) -> TenantContext {
        *self.binding.read().expect("tenant binding lock poisoned")
    }

    /// Find the tenant-discriminator column for a `TableScan`. Schema column
    /// `tenant_id` wins; otherwise consult the source-registration override.
    fn discover_tenant_column(&self, scan: &TableScan) -> Option<String> {
        if scan.source.schema().field_with_name("tenant_id").is_ok() {
            return Some("tenant_id".to_string());
        }
        // The TableScan's table reference is a multi-part name; try its
        // catalog name first, then the table name.
        let r = &scan.table_name;
        if let Some(col) = self.source_columns.tenant_column(r.table()) {
            return Some(col);
        }
        if let Some(catalog) = r.catalog() {
            if let Some(col) = self.source_columns.tenant_column(catalog) {
                return Some(col);
            }
        }
        None
    }

    fn build_predicate(&self, col_name: &str, ctx: TenantContext) -> Expr {
        let col_expr = Expr::Column(col_name.into());
        match ctx {
            TenantContext::Unscoped => Expr::IsNull(Box::new(col_expr)),
            TenantContext::Scoped(t) => {
                let eq = Expr::BinaryExpr(BinaryExpr {
                    left: Box::new(col_expr.clone()),
                    op: Operator::Eq,
                    right: Box::new(Expr::Literal(ScalarValue::Utf8(Some(t.to_string())), None)),
                });
                let is_null = Expr::IsNull(Box::new(col_expr));
                Expr::BinaryExpr(BinaryExpr {
                    left: Box::new(eq),
                    op: Operator::Or,
                    right: Box::new(is_null),
                })
            }
        }
    }

    fn rewrite_node(
        &self,
        node: LogicalPlan,
        ctx: TenantContext,
    ) -> DfResult<Transformed<LogicalPlan>> {
        if let LogicalPlan::TableScan(scan) = &node {
            if let Some(col) = self.discover_tenant_column(scan) {
                let predicate = self.build_predicate(&col, ctx);
                let filter = Filter::try_new(predicate, Arc::new(node))?;
                return Ok(Transformed::yes(LogicalPlan::Filter(filter)));
            }
        }
        Ok(Transformed::no(node))
    }
}

impl AnalyzerRule for TenantScopeAnalyzerRule {
    fn name(&self) -> &str {
        "tenant_scope_predicate_injection"
    }

    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> DfResult<LogicalPlan> {
        let ctx = self.current_context();
        plan.transform_up(|node| self.rewrite_node(node, ctx))
            .map(|t| t.data)
    }
}
