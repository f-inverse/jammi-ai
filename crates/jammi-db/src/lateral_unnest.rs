//! A lateral `unnest` — `FROM t CROSS JOIN unnest(f(t.c)) AS u(x)` — planned
//! as the rows it means.
//!
//! DataFusion plans the lateral form as a join whose right side is a
//! correlated subquery: projections and unnests over one empty row, reading
//! the left side's columns as outer references. Its decorrelation does not
//! reach that shape, so the physical planner meets the outer reference and
//! refuses it. Run over the left side instead — the outer references read as
//! its own columns, which each projection carries through — the subquery's
//! projections and unnests produce exactly the joined rows: each left row
//! with each item its list holds.

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion::common::Result;
use datafusion::logical_expr::{
    Expr, JoinType, LogicalPlan, LogicalPlanBuilder, Projection, Unnest,
};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};

/// The rule that plans a lateral `unnest` over the join's left side.
#[derive(Debug, Default)]
pub(crate) struct LateralUnnest;

impl OptimizerRule for LateralUnnest {
    fn name(&self) -> &str {
        "lateral_unnest"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Join(join) = &plan else {
            return Ok(Transformed::no(plan));
        };
        if join.join_type != JoinType::Inner || !join.on.is_empty() || join.filter.is_some() {
            return Ok(Transformed::no(plan));
        }
        let Some(lateral) = Lateral::of(&join.right, &join.left) else {
            return Ok(Transformed::no(plan));
        };
        let over = lateral.over(Arc::clone(&join.left))?;
        // The rows the join produced, under the names and qualifiers it gave
        // them: the left side's, then the subquery's under its alias.
        let named = over
            .schema()
            .columns()
            .into_iter()
            .zip(join.schema.iter())
            .map(|(column, (qualifier, field))| {
                Expr::Column(column).alias_qualified(qualifier.cloned(), field.name())
            })
            .collect();
        Ok(Transformed::yes(LogicalPlan::Projection(
            Projection::try_new(named, Arc::new(over))?,
        )))
    }
}

/// One step of a lateral subquery, top first.
enum Step<'a> {
    Project(&'a Projection),
    Unnest(&'a Unnest),
}

/// A correlated subquery the rule places over the join's left side: an
/// optional alias over projections and unnests whose only table is one empty
/// row, every outer reference naming a column of the left side, and no column
/// of its own sharing a name with one there.
struct Lateral<'a> {
    steps: Vec<Step<'a>>,
}

impl<'a> Lateral<'a> {
    fn of(right: &'a LogicalPlan, left: &LogicalPlan) -> Option<Self> {
        let subquery = match right {
            LogicalPlan::SubqueryAlias(alias) => match alias.input.as_ref() {
                LogicalPlan::Subquery(subquery) => subquery,
                _ => return None,
            },
            LogicalPlan::Subquery(subquery) => subquery,
            _ => return None,
        };
        let left_schema = left.schema();
        let reads_left = subquery.outer_ref_columns.iter().all(|reference| {
            matches!(reference, Expr::OuterReferenceColumn(_, column) if left_schema.has_column(column))
        });
        if !reads_left {
            return None;
        }
        let mut steps = Vec::new();
        let mut node = subquery.subquery.as_ref();
        loop {
            let shares_a_name = node.schema().fields().iter().any(|field| {
                left_schema
                    .fields()
                    .iter()
                    .any(|l| l.name() == field.name())
            });
            if shares_a_name {
                return None;
            }
            node = match node {
                LogicalPlan::Projection(projection) => {
                    steps.push(Step::Project(projection));
                    projection.input.as_ref()
                }
                LogicalPlan::Unnest(unnest) => {
                    steps.push(Step::Unnest(unnest));
                    unnest.input.as_ref()
                }
                LogicalPlan::EmptyRelation(empty) if empty.produce_one_row => break,
                _ => return None,
            };
        }
        Some(Self { steps })
    }

    /// The subquery's steps run over `left`, bottom first: each projection
    /// carries `left`'s columns ahead of its own, its outer references read as
    /// those columns, and each unnest replicates them for every item.
    fn over(&self, left: Arc<LogicalPlan>) -> Result<LogicalPlan> {
        let carried: Vec<Expr> = left
            .schema()
            .columns()
            .into_iter()
            .map(Expr::Column)
            .collect();
        self.steps
            .iter()
            .rev()
            .try_fold(Arc::unwrap_or_clone(left), |input, step| match step {
                Step::Project(projection) => {
                    let own = projection
                        .expr
                        .iter()
                        .map(|expr| unreferenced(expr.clone()))
                        .collect::<Result<Vec<_>>>()?;
                    let exprs = carried.iter().cloned().chain(own).collect();
                    Ok(LogicalPlan::Projection(Projection::try_new(
                        exprs,
                        Arc::new(input),
                    )?))
                }
                Step::Unnest(unnest) => LogicalPlanBuilder::from(input)
                    .unnest_columns_with_options(
                        unnest.exec_columns.clone(),
                        unnest.options.clone(),
                    )?
                    .build(),
            })
    }
}

/// `expr` with every outer reference read as the column it names.
fn unreferenced(expr: Expr) -> Result<Expr> {
    expr.transform(|e| match e {
        Expr::OuterReferenceColumn(_, column) => Ok(Transformed::yes(Expr::Column(column))),
        other => Ok(Transformed::no(other)),
    })
    .data()
}

#[cfg(test)]
mod tests {
    use super::*;

    use datafusion::common::Column;
    use datafusion::prelude::SessionContext;

    /// The optimized plan of `sql` over `t(id, items)`, with the rule in
    /// DataFusion's list right after `scalar_subquery_to_join` when `rule`.
    async fn optimized(sql: &str, rule: bool) -> LogicalPlan {
        let ctx = SessionContext::new();
        ctx.sql("CREATE TABLE t (id INT, items VARCHAR) AS VALUES (1, 'a,b'), (2, 'c')")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let state = ctx.state();
        let mut rules = state.optimizer().rules.clone();
        let after = rules
            .iter()
            .position(|r| r.name() == "scalar_subquery_to_join")
            .unwrap()
            + 1;
        if rule {
            rules.insert(after, Arc::new(LateralUnnest));
        }
        let state = datafusion::execution::SessionStateBuilder::new_from_existing(state)
            .with_optimizer_rules(rules)
            .build();
        let plan = state.create_logical_plan(sql).await.unwrap();
        state.optimize(&plan).unwrap()
    }

    fn holds_subquery(plan: &LogicalPlan) -> bool {
        plan.exists(|node| Ok(matches!(node, LogicalPlan::Subquery(_))))
            .unwrap()
    }

    #[tokio::test]
    async fn a_lateral_unnest_reads_the_left_side_with_no_subquery_left() {
        let plan = optimized(
            "SELECT t.id, u.item FROM t CROSS JOIN unnest(string_to_array(t.items, ',')) AS u(item)",
            true,
        )
        .await;
        assert!(!holds_subquery(&plan), "{plan}");
    }

    #[tokio::test]
    async fn a_lateral_subquery_reading_a_table_stays_as_written() {
        let sql = "SELECT t.id, s.n FROM t CROSS JOIN LATERAL \
                   (SELECT count(*) AS n FROM t AS x WHERE x.id = t.id) AS s";
        assert_eq!(
            optimized(sql, true).await.to_string(),
            optimized(sql, false).await.to_string()
        );
    }

    #[test]
    fn an_outer_reference_reads_as_its_column() {
        let column = Column::new(Some("t"), "items");
        let field = Arc::new(datafusion::arrow::datatypes::Field::new(
            "items",
            datafusion::arrow::datatypes::DataType::Utf8,
            true,
        ));
        let expr = Expr::OuterReferenceColumn(field, column.clone());
        assert_eq!(unreferenced(expr).unwrap(), Expr::Column(column));
    }
}
