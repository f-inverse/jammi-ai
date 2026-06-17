//! Scale-sanity for the as-of operator (exit-criterion #8): a large facts × spine
//! join over many groups completes within the sort-merge bound — O((n+m) log)
//! dominated by the per-side sort — not the `NestedLoopJoinExec` quadratic path
//! DataFusion falls back to for plain inequality joins.
//!
//! The test runs IN-MEMORY at TRUE SCALE (1,000,000 facts × 100,000 spine over
//! 10,000 groups), driving the real operator path: each side is registered as an
//! in-memory relation, planned, wrapped in the same `SortExec` the verb inserts,
//! and merged by `AsofJoinExec`. It bypasses Parquet IO and the catalog so the
//! wall-clock it asserts is the sort-merge work itself, not storage — the bound
//! the exit-criterion is about.
//!
//! The ceiling is deliberately generous and measured once (the same same-box-rate
//! discipline the W1 benches use): a quadratic regression (10^5 × 10^6 ≈ 10^11
//! row-pairs) would take minutes-to-hours and blow any sane ceiling, while the
//! sort-merge completes in low single-digit seconds on the reference box. The
//! assertion is a structural-complexity guard, not a micro-benchmark — a true
//! scale run, not a scaled-down proxy.

use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::physical_expr::{expressions::col, LexOrdering, PhysicalSortExpr};
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;
use jammi_ai::pipeline::asof::{
    AsofJoinExec, AsofJoinSpecBuilder, AsofKey, Boundary, MatchDirection, TieBreak,
};

fn spine_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("sym", DataType::Utf8, false),
        Field::new("t", DataType::Int64, false),
    ]))
}

fn facts_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("sym", DataType::Utf8, false),
        Field::new("qt", DataType::Int64, false),
        Field::new("val", DataType::Int64, false),
    ]))
}

/// Plan an in-memory relation and wrap it in a `SortExec` over (`sym`, time) —
/// the order the operator requires, the same wrapping the verb does.
async fn sorted_scan(
    ctx: &SessionContext,
    name: &str,
    batch: RecordBatch,
    time: &str,
) -> Arc<dyn ExecutionPlan> {
    ctx.register_batch(name, batch).unwrap();
    let plan = ctx
        .sql(&format!("SELECT * FROM {name}"))
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let schema = plan.schema();
    let options = SortOptions {
        descending: false,
        nulls_first: false,
    };
    let ordering = LexOrdering::new(vec![
        PhysicalSortExpr {
            expr: col("sym", schema.as_ref()).unwrap(),
            options,
        },
        PhysicalSortExpr {
            expr: col(time, schema.as_ref()).unwrap(),
            options,
        },
    ])
    .unwrap();
    Arc::new(SortExec::new(ordering, plan))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn scale_sanity_sort_merge_bound() {
    let groups = 10_000i64;
    let facts_n = 1_000_000usize; // 100 facts per group
    let spine_n = 100_000usize; // 10 spine rows per group

    // Facts: group g has quotes at 0,10,...,990 (100 per group), bid = qt.
    let mut f_sym = Vec::with_capacity(facts_n);
    let mut f_qt = Vec::with_capacity(facts_n);
    let mut f_val = Vec::with_capacity(facts_n);
    for g in 0..groups {
        for step in 0..(facts_n as i64 / groups) {
            f_sym.push(format!("g{g}"));
            f_qt.push(step * 10);
            f_val.push(step * 10);
        }
    }
    let facts = RecordBatch::try_new(
        facts_schema(),
        vec![
            Arc::new(StringArray::from(f_sym)),
            Arc::new(Int64Array::from(f_qt)),
            Arc::new(Int64Array::from(f_val)),
        ],
    )
    .unwrap();

    // Spine: 10 trades per group at varied exec times.
    let mut s_sym = Vec::with_capacity(spine_n);
    let mut s_t = Vec::with_capacity(spine_n);
    for i in 0..spine_n {
        let g = (i as i64) % groups;
        s_sym.push(format!("g{g}"));
        s_t.push(((i as i64) * 13) % 1010);
    }
    let spine = RecordBatch::try_new(
        spine_schema(),
        vec![
            Arc::new(StringArray::from(s_sym)),
            Arc::new(Int64Array::from(s_t)),
        ],
    )
    .unwrap();

    let ctx = SessionContext::new();
    let spine_plan = sorted_scan(&ctx, "spine", spine, "t").await;
    let facts_plan = sorted_scan(&ctx, "facts", facts, "qt").await;

    let spec = AsofJoinSpecBuilder::new(
        AsofKey {
            by: vec!["sym".into()],
            time: "t".into(),
        },
        AsofKey {
            by: vec!["sym".into()],
            time: "qt".into(),
        },
    )
    .direction(MatchDirection::Backward)
    .boundary(Boundary::Inclusive)
    .tie_break(TieBreak::Error)
    .project(vec!["val".into()])
    .build();

    let exec: Arc<dyn ExecutionPlan> =
        Arc::new(AsofJoinExec::try_new(spine_plan, facts_plan, spec).unwrap());

    let start = Instant::now();
    let stream = exec.execute(0, ctx.task_ctx()).unwrap();
    let batches = datafusion::physical_plan::common::collect(stream)
        .await
        .unwrap();
    let elapsed = start.elapsed();

    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, spine_n, "every spine row is preserved at scale");

    // Generous ceiling: the sort-merge of 1.1M total rows over 10k groups
    // completes in low single-digit seconds on the reference box; 60s leaves a
    // wide margin for a contended CI runner while still being orders of magnitude
    // below the quadratic NestedLoop path (which would not finish). A breach here
    // means the operator regressed off the sort-merge bound, not a slow box.
    assert!(
        elapsed.as_secs() < 60,
        "asof_join over {facts_n} facts × {spine_n} spine ({groups} groups) took {elapsed:?}; \
         expected the sort-merge bound (seconds), not the quadratic path"
    );
}
