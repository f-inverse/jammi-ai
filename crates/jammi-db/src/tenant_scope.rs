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
//!
//! A query's scope is read once, where the query enters the engine, and
//! travels in its session configuration as a [`TenantConfig`]: the analyzer
//! reads it from the configuration it is handed, a scan from the state that
//! plans it, a sink from the task context that runs it. Nothing a query
//! does after it entered reads an ambient value, so a query never sees two
//! tenants, whichever task each part of it runs on.

use std::any::Any;
use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, RwLock};

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::config::{ConfigEntry, ConfigExtension, ConfigOptions, ExtensionOptions};
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::context::SessionState;
use datafusion::logical_expr::{BinaryExpr, Expr, Filter, LogicalPlan, Operator, TableScan};
use datafusion::optimizer::AnalyzerRule;
use datafusion::scalar::ScalarValue;

use crate::tenant::{TenantContext, TenantId};

tokio::task_local! {
    /// Task-local tenant override consulted by [`TenantBinding::current`].
    ///
    /// Set by [`crate::session::JammiSession::with_tenant_scoped`] for the
    /// duration of one closure (typically one gRPC request handler). When
    /// present, it shadows the session's shared [`TenantBinding::shared`]
    /// for the lifetime of the executing task — including across `.await`
    /// points and recursive calls inside the closure — without mutating
    /// any shared state. Concurrent tasks each see their own override.
    static CURRENT_TENANT_OVERRIDE: TenantContext;

    /// Task-local admin-scope marker. When present (set by
    /// [`crate::session::JammiSession::with_admin_scope`] for the duration
    /// of one closure), the [`TenantScopeAnalyzerRule`] skips predicate
    /// injection and the mutable-table read paths drop the tenant filter
    /// from the backend SQL — yielding rows across every tenant. Concurrent
    /// tasks each carry their own marker, so an admin scope on one task
    /// does not bleed into a tenant-scoped query running on another.
    ///
    /// Read by [`TenantBinding::is_admin_scope`]; not exposed otherwise.
    static ADMIN_SCOPE_ACTIVE: ();
}

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

/// The tenant a call into the engine runs under, where the call enters it.
///
/// The binding overlays two values:
///
/// 1. A **task-local override** (`CURRENT_TENANT_OVERRIDE`) installed for
///    the duration of one [`crate::session::JammiSession::with_tenant_scoped`]
///    or [`crate::session::JammiSession::with_request_scope`] closure: how a
///    server binds each request. Concurrent tasks each carry their own
///    override, with no shared-state write involved.
/// 2. The **session's bound tenant**, set by
///    [`crate::session::JammiSession::bind_tenant`]: one embedded
///    connection's tenant, the value every call it makes enters with. A
///    session shared between requests never binds one; each request scopes
///    itself instead.
///
/// [`Self::current`] returns the override when present, the bound tenant
/// otherwise. It is read where a call enters the engine: a catalog verb on
/// its own task, and a query once, when it is planned, into the
/// [`TenantConfig`] its execution reads.
///
/// `Clone` is cheap (an `Arc` clone); the binding is shared between the
/// session, its [`crate::catalog::Catalog`], and its query context.
#[derive(Debug, Clone)]
pub struct TenantBinding {
    shared: Arc<RwLock<TenantContext>>,
}

impl TenantBinding {
    /// A fresh binding whose bound tenant is [`TenantContext::Unscoped`].
    pub fn unscoped() -> Self {
        Self::bound_to(TenantContext::Unscoped)
    }

    /// A fresh binding whose bound tenant is `ctx`: a handle pinned to one
    /// tenant, sharing nothing with any other binding.
    pub fn bound_to(ctx: TenantContext) -> Self {
        Self {
            shared: Arc::new(RwLock::new(ctx)),
        }
    }

    /// The currently effective tenant context for this binding.
    ///
    /// Returns the task-local override installed by
    /// [`crate::session::JammiSession::with_tenant_scoped`] if the current
    /// task is executing inside such a scope; otherwise returns the
    /// session's sticky shared value.
    pub fn current(&self) -> TenantContext {
        CURRENT_TENANT_OVERRIDE
            .try_with(|c| *c)
            .unwrap_or_else(|_| *self.shared.read().expect("tenant binding lock poisoned"))
    }

    /// Convenience: the [`TenantId`] of [`Self::current`], if any.
    pub fn current_tenant(&self) -> Option<TenantId> {
        self.current().tenant()
    }

    /// Bind `ctx` as the session's tenant: what a call enters with when no
    /// task-local override is installed. Has no effect on a call already
    /// inside a scope, nor on a query already planned.
    pub(crate) fn bind(&self, ctx: TenantContext) {
        *self.shared.write().expect("tenant binding lock poisoned") = ctx;
    }

    /// Run `f` with `ctx` installed as the task-local override for the
    /// duration of the returned future: every binding's [`Self::current`]
    /// reads it on this task. Concurrent calls on different tasks each see
    /// their own `ctx` — no shared state is mutated, so there is no race
    /// window between binding and read.
    pub async fn within<F, T>(ctx: TenantContext, f: F) -> T
    where
        F: Future<Output = T>,
    {
        CURRENT_TENANT_OVERRIDE.scope(ctx, f).await
    }

    /// Run the synchronous `f` under `ctx`: one poll of a response stream a
    /// transport hands back after its handler returned, so the stream reads
    /// the scope its request entered under.
    pub fn within_sync<F, T>(ctx: TenantContext, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        CURRENT_TENANT_OVERRIDE.sync_scope(ctx, f)
    }

    /// Returns `true` when the current task is executing inside a
    /// [`crate::session::JammiSession::with_admin_scope`] closure.
    ///
    /// Consulted by the [`TenantScopeAnalyzerRule`] and by the mutable-table
    /// read path: when `true`, both drop the per-row tenant filter for the
    /// duration of plan analysis / scan, yielding rows across every tenant.
    /// The marker is task-local, so a concurrent task without admin scope
    /// continues to observe its normal tenant binding.
    pub fn is_admin_scope() -> bool {
        ADMIN_SCOPE_ACTIVE.try_with(|_| ()).is_ok()
    }

    /// Run `f` with the admin-bypass task-local installed for the duration
    /// of the returned future. Inside `f`, [`Self::is_admin_scope`] returns
    /// `true`; the analyzer rule and mutable-table read path skip tenant
    /// filtering.
    ///
    /// The closure-shaped surface — rather than a sticky toggle — exists so
    /// the bypass cannot be left enabled by a panicking caller and cannot
    /// be observed by a concurrent task on the same session.
    pub async fn admin_scope<F, T>(f: F) -> T
    where
        F: Future<Output = T>,
    {
        ADMIN_SCOPE_ACTIVE.scope((), f).await
    }
}

impl Default for TenantBinding {
    fn default() -> Self {
        Self::unscoped()
    }
}

/// The tenant scope a query was planned under, carried in its session
/// configuration from analysis to execution.
///
/// Set where the query enters the engine — a session's query context from
/// its [`TenantBinding`] as the call enters, a Flight SQL request's state
/// from the scope its resolver read — and read by everything the query does
/// after: the [`TenantScopeAnalyzerRule`] from the configuration it is
/// handed, a mutable table's scan from the state that plans it, its sink
/// from the task context that runs it. The scope belongs to the request,
/// so SQL cannot set it (`SET` refuses its keys) and lists none of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TenantConfig {
    context: TenantContext,
    every_tenant: bool,
}

impl TenantConfig {
    /// A query's scope under `context`: the tenant's rows and the global
    /// ones, or only the global ones when unscoped.
    pub fn new(context: TenantContext) -> Self {
        Self {
            context,
            every_tenant: false,
        }
    }

    /// The scope a call entering under `binding` plans with now: its
    /// current tenant, and every tenant's rows inside
    /// [`crate::session::JammiSession::with_admin_scope`].
    pub fn current(binding: &TenantBinding) -> Self {
        Self {
            context: binding.current(),
            every_tenant: TenantBinding::is_admin_scope(),
        }
    }

    /// The scope a query was planned under, from its configuration. A plan
    /// whose session carries none was planned outside the engine's query
    /// context; it is refused rather than read as any tenant.
    pub fn of(options: &ConfigOptions) -> DfResult<Self> {
        options.extensions.get::<Self>().copied().ok_or_else(|| {
            DataFusionError::Internal(
                "a plan reached tenant scoping with no tenant scope in its session \
                 configuration: it was planned outside the engine's query context"
                    .into(),
            )
        })
    }

    /// `state`, planning under this scope.
    pub fn scoping(self, mut state: SessionState) -> SessionState {
        state.config_mut().options_mut().extensions.insert(self);
        state
    }

    /// The tenant context rows are read and written under.
    pub fn context(&self) -> TenantContext {
        self.context
    }

    /// The bound tenant, if any.
    pub fn tenant(&self) -> Option<TenantId> {
        self.context.tenant()
    }

    /// Whether reads see every tenant's rows (an admin scope).
    pub fn every_tenant(&self) -> bool {
        self.every_tenant
    }
}

impl ExtensionOptions for TenantConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(*self)
    }

    fn set(&mut self, key: &str, _value: &str) -> DfResult<()> {
        Err(DataFusionError::Configuration(format!(
            "`{}.{key}` is the request's tenant scope, which SQL cannot set",
            Self::PREFIX
        )))
    }

    fn entries(&self) -> Vec<ConfigEntry> {
        Vec::new()
    }
}

impl ConfigExtension for TenantConfig {
    const PREFIX: &'static str = "jammi_tenant";
}

/// `AnalyzerRule` that wraps every tenant-aware `TableScan` in a `Filter`
/// emitting the session's tenant predicate.
#[derive(Debug)]
pub struct TenantScopeAnalyzerRule {
    source_columns: Arc<SourceTenantColumns>,
}

impl TenantScopeAnalyzerRule {
    pub fn new(source_columns: Arc<SourceTenantColumns>) -> Self {
        Self { source_columns }
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
        // Cast the column to `Utf8` so the comparison is well-typed when the
        // underlying column is `Utf8View` (Arrow 57's parquet reader default
        // under DataFusion 52). Without the cast, `Utf8View = Utf8` raises
        // `Invalid comparison operation: Utf8View == Utf8` at execution time
        // because the engine's Utf8 literal isn't coerced to Utf8View by the
        // analyzer-phase predicate injection. The cast is identity for Utf8
        // (mutable companion tables); coerces for Utf8View (federated parquet
        // sources); errors loudly if a non-string column ever shows up here
        // (better than silent miscompare).
        let col_expr = Expr::Cast(datafusion::logical_expr::Cast::new(
            Box::new(Expr::Column(col_name.into())),
            arrow_schema::DataType::Utf8,
        ));
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

    fn analyze(&self, plan: LogicalPlan, config: &ConfigOptions) -> DfResult<LogicalPlan> {
        let scope = TenantConfig::of(config)?;
        // An admin scope reads every tenant's rows: server-side
        // administrative scans (recovery enumerating work-in-progress rows
        // across tenants). The caller of `JammiSession::with_admin_scope` is
        // responsible for the boundary.
        if scope.every_tenant() {
            return Ok(plan);
        }
        plan.transform_up(|node| self.rewrite_node(node, scope.context()))
            .map(|t| t.data)
    }
}

#[cfg(test)]
mod tests {
    //! `TenantScopeAnalyzerRule` predicate-injection invariants.
    //!
    //! Each test builds a `LogicalPlan` via `LogicalPlanBuilder` against a
    //! `LogicalTableSource` carrying a `tenant_id` column, runs the rule's
    //! `analyze`, and walks the result to assert structural shape — no
    //! `to_string()`-based matching, no debug-format string compare.
    use super::*;
    use std::str::FromStr;
    use std::sync::Arc;

    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use datafusion::functions_aggregate::count::count;
    use datafusion::logical_expr::{
        col, JoinType, LogicalPlan, LogicalPlanBuilder, LogicalTableSource, SubqueryAlias,
    };
    use datafusion::optimizer::AnalyzerRule;

    use crate::tenant::TenantId;

    fn tenant_a() -> TenantId {
        TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap()
    }

    fn tenanted_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("region", DataType::Utf8, true),
            Field::new("tenant_id", DataType::Utf8, true),
        ]))
    }

    fn tenanted_source() -> Arc<LogicalTableSource> {
        Arc::new(LogicalTableSource::new(tenanted_schema()))
    }

    /// A query configuration planning under `config`.
    fn planned_under(config: TenantConfig) -> ConfigOptions {
        let mut options = ConfigOptions::default();
        options.extensions.insert(config);
        options
    }

    fn scoped(t: TenantId) -> ConfigOptions {
        planned_under(TenantConfig::new(TenantContext::Scoped(t)))
    }

    fn unscoped() -> ConfigOptions {
        planned_under(TenantConfig::new(TenantContext::Unscoped))
    }

    fn rule() -> TenantScopeAnalyzerRule {
        TenantScopeAnalyzerRule::new(Arc::new(SourceTenantColumns::new()))
    }

    /// Walk `plan` top-down and count `Filter` nodes whose direct child is a
    /// `TableScan` for `table`. Used to assert that JOIN inputs each get
    /// exactly one filter inserted between the scan and its consumer.
    fn count_tenant_filters_over(plan: &LogicalPlan, table: &str) -> usize {
        use datafusion::logical_expr::Filter;
        let mut count = 0;
        plan.apply(|node| {
            if let LogicalPlan::Filter(Filter {
                input, predicate, ..
            }) = node
            {
                if let LogicalPlan::TableScan(scan) = input.as_ref() {
                    if scan.table_name.table() == table && involves_tenant_id(predicate) {
                        count += 1;
                    }
                }
            }
            Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
        })
        .unwrap();
        count
    }

    fn involves_tenant_id(expr: &Expr) -> bool {
        match expr {
            Expr::Column(c) => c.name == "tenant_id",
            Expr::Cast(c) => involves_tenant_id(c.expr.as_ref()),
            Expr::IsNull(inner) => involves_tenant_id(inner),
            Expr::BinaryExpr(BinaryExpr { left, right, .. }) => {
                involves_tenant_id(left) || involves_tenant_id(right)
            }
            _ => false,
        }
    }

    /// Pattern-match the canonical Scoped predicate:
    /// `(col = '<tenant>') OR col IS NULL`.
    fn assert_or_eq_isnull(predicate: &Expr, col_name: &str, t: TenantId) {
        let outer = match predicate {
            Expr::BinaryExpr(b) => b,
            other => panic!("expected OR BinaryExpr, got {other:?}"),
        };
        assert!(
            matches!(outer.op, Operator::Or),
            "outer predicate must be OR, got {:?}",
            outer.op
        );
        let left = &outer.left;
        let right = &outer.right;
        // Left arm: CAST(col AS Utf8) = 'tenant'
        let eq = match left.as_ref() {
            Expr::BinaryExpr(b) => b,
            other => panic!("expected Eq BinaryExpr, got {other:?}"),
        };
        assert!(matches!(eq.op, Operator::Eq), "left arm op should be Eq");
        let lhs_cast = match eq.left.as_ref() {
            Expr::Cast(c) => c,
            other => panic!("expected Cast on left arm lhs, got {other:?}"),
        };
        let lhs_col = match lhs_cast.expr.as_ref() {
            Expr::Column(c) => c,
            other => panic!("expected column inside Cast, got {other:?}"),
        };
        assert_eq!(lhs_col.name, col_name);
        assert_eq!(lhs_cast.field.data_type(), &arrow_schema::DataType::Utf8);
        let rhs_lit = match eq.right.as_ref() {
            Expr::Literal(ScalarValue::Utf8(Some(s)), _) => s.clone(),
            other => panic!("expected Utf8 literal on left arm rhs, got {other:?}"),
        };
        assert_eq!(rhs_lit, t.to_string());
        // Right arm: CAST(col AS Utf8) IS NULL
        let is_null_inner = match right.as_ref() {
            Expr::IsNull(inner) => inner,
            other => panic!("expected IsNull on right arm, got {other:?}"),
        };
        let cast = match is_null_inner.as_ref() {
            Expr::Cast(c) => c,
            other => panic!("expected Cast inside IsNull, got {other:?}"),
        };
        let isnull_col = match cast.expr.as_ref() {
            Expr::Column(c) => c,
            other => panic!("expected column inside Cast(IsNull), got {other:?}"),
        };
        assert_eq!(isnull_col.name, col_name);
    }

    fn assert_isnull_only(predicate: &Expr, col_name: &str) {
        let inner = match predicate {
            Expr::IsNull(b) => b,
            other => panic!("expected IsNull predicate, got {other:?}"),
        };
        let cast = match inner.as_ref() {
            Expr::Cast(c) => c,
            other => panic!("expected Cast inside IsNull, got {other:?}"),
        };
        let col = match cast.expr.as_ref() {
            Expr::Column(c) => c,
            other => panic!("expected column inside Cast, got {other:?}"),
        };
        assert_eq!(col.name, col_name);
    }

    /// Walk `plan` and return the predicate of the first `Filter` whose
    /// direct child is a `TableScan`. Owned `Expr` clone so the caller can
    /// pattern-match without lifetime gymnastics.
    fn first_predicate_above_scan(plan: &LogicalPlan) -> Expr {
        fn walk(plan: &LogicalPlan, out: &mut Option<Expr>) {
            if out.is_some() {
                return;
            }
            if let LogicalPlan::Filter(f) = plan {
                if matches!(f.input.as_ref(), LogicalPlan::TableScan(_)) {
                    *out = Some(f.predicate.clone());
                    return;
                }
            }
            for child in plan.inputs() {
                walk(child, out);
                if out.is_some() {
                    return;
                }
            }
        }
        let mut out = None;
        walk(plan, &mut out);
        out.expect("expected a Filter directly above a TableScan")
    }

    // -- 1. Single scan, Scoped session → eq OR isnull --

    #[test]
    fn single_scan_scoped_emits_eq_or_isnull() {
        let plan = LogicalPlanBuilder::scan("sources", tenanted_source(), None)
            .unwrap()
            .project(vec![col("id")])
            .unwrap()
            .build()
            .unwrap();

        let out = rule()
            .analyze(plan, &scoped(tenant_a()))
            .unwrap();

        let predicate = first_predicate_above_scan(&out);
        assert_or_eq_isnull(&predicate, "tenant_id", tenant_a());
    }

    // -- 2. JOIN: both sides rewritten --

    #[test]
    fn join_rewrites_both_sides() {
        let right = LogicalPlanBuilder::scan("models", tenanted_source(), None)
            .unwrap()
            .build()
            .unwrap();
        let plan = LogicalPlanBuilder::scan("sources", tenanted_source(), None)
            .unwrap()
            .join_on(
                right,
                JoinType::Inner,
                vec![col("sources.id").eq(col("models.id"))],
            )
            .unwrap()
            .build()
            .unwrap();

        let out = rule()
            .analyze(plan, &scoped(tenant_a()))
            .unwrap();

        assert_eq!(
            count_tenant_filters_over(&out, "sources"),
            1,
            "JOIN left should get exactly one tenant filter"
        );
        assert_eq!(
            count_tenant_filters_over(&out, "models"),
            1,
            "JOIN right should get exactly one tenant filter"
        );
    }

    // -- 3. Subquery: inner scan rewritten via bottom-up traversal --

    #[test]
    fn subquery_inner_scan_rewritten_via_transform_up() {
        // Inner: SELECT id FROM sources
        let inner = LogicalPlanBuilder::scan("sources", tenanted_source(), None)
            .unwrap()
            .project(vec![col("id")])
            .unwrap()
            .build()
            .unwrap();
        let inner_alias =
            LogicalPlan::SubqueryAlias(SubqueryAlias::try_new(Arc::new(inner), "inner_q").unwrap());

        // Outer scan over models, then cross-join with inner alias.
        let outer = LogicalPlanBuilder::scan("models", tenanted_source(), None)
            .unwrap()
            .cross_join(inner_alias)
            .unwrap()
            .build()
            .unwrap();

        let out = rule()
            .analyze(outer, &scoped(tenant_a()))
            .unwrap();

        // Both `sources` (inside the subquery alias) and `models` (outer scan)
        // must have one tenant filter each.
        assert_eq!(count_tenant_filters_over(&out, "sources"), 1);
        assert_eq!(count_tenant_filters_over(&out, "models"), 1);
    }

    // -- 4. GROUP BY: scan rewritten; Aggregate node untouched --

    #[test]
    fn group_by_input_rewritten_aggregate_untouched() {
        let plan = LogicalPlanBuilder::scan("sources", tenanted_source(), None)
            .unwrap()
            .aggregate(vec![col("region")], vec![count(col("id"))])
            .unwrap()
            .build()
            .unwrap();

        let out = rule()
            .analyze(plan, &scoped(tenant_a()))
            .unwrap();

        // Top-level node is still an Aggregate.
        let agg = match &out {
            LogicalPlan::Aggregate(a) => a,
            other => panic!("expected Aggregate at top, got {other:?}"),
        };
        // Aggregate.input is a Filter…
        let filter = match agg.input.as_ref() {
            LogicalPlan::Filter(f) => f,
            other => panic!("expected Filter under Aggregate, got {other:?}"),
        };
        // …whose input is the original TableScan.
        assert!(matches!(filter.input.as_ref(), LogicalPlan::TableScan(_)));
        // Group expr and aggr expr are unchanged.
        assert_eq!(agg.group_expr.len(), 1);
        assert_eq!(agg.aggr_expr.len(), 1);
    }

    // -- 5. Unscoped session → IS NULL only --

    #[test]
    fn unscoped_session_rewrites_to_isnull_only() {
        let plan = LogicalPlanBuilder::scan("sources", tenanted_source(), None)
            .unwrap()
            .project(vec![col("id")])
            .unwrap()
            .build()
            .unwrap();

        let out = rule()
            .analyze(plan, &unscoped())
            .unwrap();

        let predicate = first_predicate_above_scan(&out);
        assert_isnull_only(&predicate, "tenant_id");
    }

    /// A plan whose session carries no scope was planned outside the engine's
    /// query context; the rule refuses it rather than read it as any tenant.
    #[test]
    fn a_plan_without_a_tenant_scope_is_refused() {
        let plan = LogicalPlanBuilder::scan("t", tenanted_source(), None)
            .unwrap()
            .build()
            .unwrap();
        let err = rule().analyze(plan, &ConfigOptions::default()).unwrap_err();
        assert!(
            err.to_string().contains("no tenant scope"),
            "unexpected error: {err}"
        );
    }

    /// An admin scope reads every tenant's rows: no predicate is injected.
    #[test]
    fn an_admin_scope_injects_no_predicate() {
        let plan = LogicalPlanBuilder::scan("t", tenanted_source(), None)
            .unwrap()
            .build()
            .unwrap();
        let admin = TenantConfig {
            context: TenantContext::Scoped(tenant_a()),
            every_tenant: true,
        };
        let out = rule().analyze(plan.clone(), &planned_under(admin)).unwrap();
        assert_eq!(out, plan);
    }

    /// The scope is the request's: SQL `SET` on any of its keys is refused,
    /// so a client cannot rebind its own query to another tenant.
    #[test]
    fn sql_cannot_set_the_tenant_scope() {
        let mut options = scoped(tenant_a());
        let err = options
            .set("jammi_tenant.context", "Unscoped")
            .unwrap_err();
        assert!(err.to_string().contains("SQL cannot set"), "{err}");
        assert_eq!(
            TenantConfig::of(&options).unwrap(),
            TenantConfig::new(TenantContext::Scoped(tenant_a()))
        );
        assert!(options.entries().iter().all(|e| !e.key.starts_with("jammi_tenant")));
    }
}
