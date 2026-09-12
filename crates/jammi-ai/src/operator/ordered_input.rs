//! The deterministic total order every model-facing scan is fed in.
//!
//! One plan shape at the embed, `infer` and refresh sites:
//!
//! ```text
//! scan (n partitions) → CoalescePartitionsExec → KeyCheckExec → SortExec → InferenceExec
//! ```
//!
//! `InferenceExec` declares one output partition but forwards `execute(p)` to
//! its input, and the optimizer parallelises a projection (the
//! `jammi_content_hash` UDF) with a round-robin `RepartitionExec`, so without
//! the explicit `CoalescePartitionsExec` the partition-0 execution would see a
//! fraction of the rows. `SortExec` with `preserve_partitioning = false` sorts
//! only the partition it is asked to execute and merely *declares* a
//! single-partition requirement — satisfied by the optimizer in a planned
//! query, not in a hand-built plan — hence the coalesce below it.
//!
//! Sort keys, a TOTAL order: `(CAST(key AS Utf8) ASC NULLS LAST, _content_hash
//! ASC NULLS LAST)`. Rows tied on both keys have equal `_row_id` and equal
//! content, so they are mutually substitutable and the written bytes are
//! invariant under permuting them — which a partial key could not promise,
//! since the arrow sort is unstable and the coalesce interleave is
//! nondeterministic. The physical `CastExpr` is `safe: false`: a key that
//! cannot render fails loudly, never nulls. The global sort is blocking and
//! unspillable (no memory pool is wired), so `KeyCheckExec` below it refuses a
//! null key before the model is ever invoked.

use std::sync::Arc;

use arrow::compute::SortOptions;
use arrow::datatypes::DataType;
use datafusion::physical_expr::expressions::{col, CastExpr};
use datafusion::physical_expr::{LexOrdering, PhysicalExpr, PhysicalSortExpr};
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::ExecutionPlan;
use jammi_db::error::{JammiError, Result};
use jammi_db::store::schema::CONTENT_HASH_COLUMN;

use super::key_check_exec::KeyCheckExec;

/// `SortExec(D10 keys, KeyCheckExec(CoalescePartitionsExec(plan), key_column))`.
/// The input must carry `key_column` and a `_content_hash` column (every
/// source query built by `build_source_query` does).
pub fn ordered_input(
    plan: Arc<dyn ExecutionPlan>,
    key_column: &str,
) -> Result<Arc<dyn ExecutionPlan>> {
    let checked = key_checked(plan, key_column)?;
    let schema = checked.schema();
    let key = col(key_column, schema.as_ref())?;
    let key_utf8: Arc<dyn PhysicalExpr> = Arc::new(CastExpr::new(key, DataType::Utf8, None));
    let hash = col(CONTENT_HASH_COLUMN, schema.as_ref())?;
    let options = SortOptions {
        descending: false,
        nulls_first: false,
    };
    let ordering = LexOrdering::new([
        PhysicalSortExpr::new(key_utf8, options),
        PhysicalSortExpr::new(hash, options),
    ])
    .ok_or_else(|| JammiError::Inference("ordered_input: empty sort key".into()))?;
    Ok(Arc::new(SortExec::new(ordering, checked)))
}

/// `KeyCheckExec(CoalescePartitionsExec(plan), key_column)` alone — the shape
/// a classifying scan (no model below it) uses.
pub fn key_checked(
    plan: Arc<dyn ExecutionPlan>,
    key_column: &str,
) -> Result<Arc<dyn ExecutionPlan>> {
    let coalesced: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(plan));
    Ok(Arc::new(KeyCheckExec::try_new(coalesced, key_column)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, RecordBatch, StringArray};
    use arrow::datatypes::{Field, Schema};
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::physical_plan::common::collect;
    use datafusion::physical_plan::ExecutionPlanProperties;
    use datafusion::prelude::SessionContext;

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new(CONTENT_HASH_COLUMN, DataType::Utf8, true),
        ]))
    }

    fn batch(ids: Vec<Option<i64>>, hashes: Vec<&str>) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(hashes)),
            ],
        )
        .unwrap()
    }

    fn four_partition_source(parts: Vec<RecordBatch>) -> Arc<dyn ExecutionPlan> {
        let partitions: Vec<Vec<RecordBatch>> = parts.into_iter().map(|b| vec![b]).collect();
        MemorySourceConfig::try_new_exec(&partitions, schema(), None).unwrap()
    }

    /// The plan-builder unit: below the sort there is exactly one partition
    /// regardless of the input's, and `KeyCheckExec` keeps the defaults a
    /// fetch must never be pushed past.
    #[test]
    fn one_partition_below_the_sort_and_no_limit_pushdown() {
        let src = four_partition_source(vec![
            batch(vec![Some(1)], vec!["h"]),
            batch(vec![Some(2)], vec!["h"]),
            batch(vec![Some(3)], vec!["h"]),
            batch(vec![Some(4)], vec!["h"]),
        ]);
        assert_eq!(src.output_partitioning().partition_count(), 4);
        let plan = ordered_input(src, "id").unwrap();
        assert_eq!(plan.name(), "SortExec");
        let below = plan.children()[0];
        assert_eq!(below.name(), "KeyCheckExec");
        assert_eq!(below.output_partitioning().partition_count(), 1);
        assert!(!below.supports_limit_pushdown());
        assert!(Arc::clone(below).with_fetch(Some(1)).is_none());
        assert_eq!(plan.output_partitioning().partition_count(), 1);
    }

    /// Executing partition 0 sees every input partition, in the total order
    /// `(CAST(key AS Utf8), _content_hash)` — `"10"` before `"2"`.
    #[tokio::test]
    async fn partition_zero_carries_every_row_in_key_order() {
        let src = four_partition_source(vec![
            batch(vec![Some(2)], vec!["b"]),
            batch(vec![Some(10)], vec!["a"]),
            batch(vec![Some(1), Some(1)], vec!["z", "a"]),
            batch(vec![Some(3)], vec!["c"]),
        ]);
        let plan = ordered_input(src, "id").unwrap();
        let ctx = SessionContext::new();
        let out = collect(plan.execute(0, ctx.task_ctx()).unwrap())
            .await
            .unwrap();
        let mut ids = Vec::new();
        let mut hashes = Vec::new();
        for b in &out {
            let id = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let h = b.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            for i in 0..b.num_rows() {
                ids.push(id.value(i));
                hashes.push(h.value(i).to_string());
            }
        }
        assert_eq!(ids, vec![1, 1, 10, 2, 3]);
        assert_eq!(hashes, vec!["a", "z", "a", "b", "c"]);
    }

    /// A null key anywhere in the input is one typed refusal with the exact
    /// total, raised at end of input — the sort emits nothing before it.
    #[tokio::test]
    async fn null_keys_are_one_typed_refusal_with_the_exact_count() {
        let src = four_partition_source(vec![
            batch(vec![Some(1), None], vec!["a", "b"]),
            batch(vec![None, None, Some(5)], vec!["c", "d", "e"]),
        ]);
        let plan = ordered_input(src, "id").unwrap();
        let ctx = SessionContext::new();
        let err = collect(plan.execute(0, ctx.task_ctx()).unwrap())
            .await
            .expect_err("null keys must refuse");
        match JammiError::from(err) {
            JammiError::InvalidKey { column, null_count } => {
                assert_eq!(column, "id");
                assert_eq!(null_count, 3);
            }
            other => panic!("expected InvalidKey, got {other:?}"),
        }
    }
}
