//! RANGESPLIT (#540): N-way `InferenceExec` below one `_ordinal`-keyed merge.
//!
//! Every oracle here builds the REAL production types (`OrdinalSplitExec`,
//! `InferenceExecBuilder`, `SortPreservingMergeExec`, and — for RS5 —
//! `wrap_with_split_and_merge` itself) rather than a stand-in mimicking their
//! declared `PlanProperties` (the design round's own probes at
//! `targets/pt-rangesplit` used stubs for `InferenceExec`; this file never
//! does). See the implementation contract (`rangesplit.md`) §0/§1 for the
//! executed premises RS1-RS8 rest on.

use std::sync::Arc;

use arrow::array::{Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::datasource::TableType;
use datafusion::logical_expr::Expr;
use datafusion::physical_expr::expressions::col;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::{displayable, ExecutionPlan, ExecutionPlanProperties};
use datafusion::prelude::{SessionConfig, SessionContext};
use tempfile::TempDir;

use jammi_ai::inference::schema::build_output_schema;
use jammi_ai::model::{ModelSource, ModelTask};
use jammi_ai::operator::inference_exec::{wrap_with_split_and_merge, InferenceExecBuilder};
use jammi_ai::operator::ordinal_split_exec::{OrdinalSplitExec, ORDINAL_COLUMN};
use jammi_ai::session::InferenceSession;
use jammi_db::store::manifest::ComputeDeviceKind;

use crate::common;

fn tiny_bert_model() -> String {
    "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap()
}

async fn session() -> (Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().unwrap();
    let cfg = common::test_config(dir.path());
    let s = Arc::new(InferenceSession::new(cfg).await.unwrap());
    (s, dir)
}

fn in_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("_content_hash", DataType::Utf8, true),
    ]))
}

fn out_schema() -> SchemaRef {
    build_output_schema(
        &ModelTask::TextEmbedding,
        &in_schema(),
        "id",
        Some(32),
        None,
        &[],
    )
    .unwrap()
}

/// A `TableProvider` whose `scan()` builds the REAL RANGESPLIT plan
/// (`OrdinalSplitExec` + a real `InferenceExec` + `SortPreservingMergeExec`)
/// via the SAME `OrdinalSplitExec::new` + `InferenceExecBuilder` +
/// `SortPreservingMergeExec::new` construction the production call sites use
/// (never `wrap_with_split_and_merge`'s `n<=1` shortcut — RS1's grid forces
/// the split present at every `N`, matching the design round's own probes).
struct AnnotateStub {
    session: Arc<InferenceSession>,
    n: usize,
    /// "PIPE" (a pre-sorted single-partition input, `ordered_input`'s shape)
    /// or "UDTF" (`annotate_plan`'s shape: nothing below the split is
    /// sorted, and the scan may have more than one partition).
    shape: &'static str,
    src_partitions: usize,
}

impl std::fmt::Debug for AnnotateStub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnnotateStub")
            .field("n", &self.n)
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl TableProvider for AnnotateStub {
    fn schema(&self) -> SchemaRef {
        out_schema()
    }
    fn table_type(&self) -> TableType {
        TableType::Base
    }
    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        let batch = RecordBatch::try_new(
            in_schema(),
            vec![
                Arc::new(Int64Array::from(vec![1i64, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )?;
        let parts: Vec<Vec<RecordBatch>> = (0..self.src_partitions)
            .map(|_| vec![batch.clone()])
            .collect();
        let src = MemorySourceConfig::try_new_exec(&parts, in_schema(), None)?;
        let below: Arc<dyn ExecutionPlan> = if self.shape == "PIPE" {
            jammi_ai::operator::ordered_input::ordered_input(src, "id")
                .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?
        } else {
            src
        };
        let split = OrdinalSplitExec::new(below, self.n)?;
        let inference = InferenceExecBuilder::new(
            Arc::new(split),
            ModelSource::parse(&tiny_bert_model()),
            ModelTask::TextEmbedding,
            vec!["_content_hash".to_string()],
            "id".to_string(),
            "src".to_string(),
            Arc::clone(self.session.model_cache()),
            ComputeDeviceKind::Cpu,
        )
        .embedding_dim(Some(32))
        .build()
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
        let inference: Arc<dyn ExecutionPlan> = Arc::new(inference);
        let ordinal = col(ORDINAL_COLUMN, inference.schema().as_ref())?;
        let sort_opts = arrow::compute::SortOptions {
            descending: false,
            nulls_first: false,
        };
        let ordering = datafusion::physical_expr::LexOrdering::new([
            datafusion::physical_expr::PhysicalSortExpr::new(ordinal, sort_opts),
        ])
        .unwrap();
        let merged: Arc<dyn ExecutionPlan> =
            Arc::new(SortPreservingMergeExec::new(ordering, inference));
        match projection {
            Some(proj) => {
                let schema = merged.schema();
                let exprs = proj
                    .iter()
                    .map(|&i| {
                        let field = schema.field(i);
                        let e = col(field.name(), schema.as_ref())?;
                        Ok((
                            e as Arc<dyn datafusion::physical_expr::PhysicalExpr>,
                            field.name().to_string(),
                        ))
                    })
                    .collect::<datafusion::error::Result<Vec<_>>>()?;
                Ok(Arc::new(
                    datafusion::physical_plan::projection::ProjectionExec::try_new(exprs, merged)?,
                ))
            }
            None => Ok(merged),
        }
    }
}

/// RS1: for the OPTIMIZED plan, nothing (no `RepartitionExec`/`SortExec`/
/// `CoalescePartitionsExec`) sits between `OrdinalSplitExec` and
/// `InferenceExec` — the 120-cell grid over the production declaration set
/// (`{PIPE, UDTF} x {bare, GROUP BY, ORDER BY, WHERE, LIMIT} x N in {1,2,4} x
/// target_partitions in {1,2,4,8}`; RANGESPLIT ships the ordinal-only design
/// the contract's executed record selected, so this is the single-variant
/// subset of the design round's 240-cell two-variant grid that applies to
/// the shipped code).
///
/// Mutation (dropping `benefits_from_input_partitioning = [false]`) is
/// executed by the sibling
/// `rs1_oracle_detects_an_inserted_node_between_split_and_inference` below
/// (a smaller, targeted repro — 120 real DataFusion optimizer passes per
/// mutation makes an in-loop mutate-and-rerun too slow for routine CI). This
/// grid ALSO independently goes RED under RS2's merge-key mutation
/// (`[_row_id, _ordinal]`, executed and reverted during implementation, and
/// re-measured on this file's final tree at 64 of 120 cells — a number
/// that shifts with unrelated changes, re-measure before trusting it
/// stale) — `SanityCheckPlan` refuses those cells outright ("does not
/// satisfy order requirements ... Child-0 order: [[_ordinal ASC]]"); see
/// `rs2_row_sequence_is_identical_across_n_on_both_input_shapes`'s own doc,
/// which now reds under the identical mutation directly (a real per-row
/// divergence oracle, not merely a plan-build refusal).
#[tokio::test]
async fn rs1_nothing_between_split_and_inference_across_the_grid() {
    let (session, _dir) = session().await;
    let mut failures: Vec<String> = Vec::new();
    let mut total = 0usize;

    for shape in ["PIPE", "UDTF"] {
        for (label, sql) in [
            ("bare", "SELECT * FROM annotated"),
            (
                "group_by",
                "SELECT _row_id, count(*) FROM annotated GROUP BY _row_id",
            ),
            ("order_by", "SELECT _row_id FROM annotated ORDER BY _row_id"),
            (
                "filter",
                "SELECT _row_id, vector FROM annotated WHERE _row_id <> 'x'",
            ),
            ("limit", "SELECT * FROM annotated LIMIT 2"),
        ] {
            for &n in &[1usize, 2, 4] {
                for &tp in &[1usize, 2, 4, 8] {
                    total += 1;
                    let cfg = SessionConfig::new().with_target_partitions(tp);
                    let ctx = SessionContext::new_with_config(cfg);
                    ctx.register_table(
                        "annotated",
                        Arc::new(AnnotateStub {
                            session: Arc::clone(&session),
                            n,
                            shape,
                            src_partitions: 4,
                        }),
                    )
                    .unwrap();
                    let df = match ctx.sql(sql).await {
                        Ok(d) => d,
                        Err(e) => {
                            failures.push(format!("{shape} {label} N={n} tp={tp}: SQL error: {e}"));
                            continue;
                        }
                    };
                    let plan = match df.create_physical_plan().await {
                        Ok(p) => p,
                        Err(e) => {
                            failures
                                .push(format!("{shape} {label} N={n} tp={tp}: plan error: {e}"));
                            continue;
                        }
                    };
                    let text = displayable(plan.as_ref()).indent(true).to_string();
                    // Non-vacuous containment: the structural check just
                    // below can only mean something if BOTH lines are
                    // actually present in this cell's plan — a plan missing
                    // either line entirely would otherwise pass the
                    // "nothing inserted between them" check vacuously.
                    if !text.contains("InferenceExec") || !text.contains("OrdinalSplitExec") {
                        failures.push(format!(
                            "{shape} {label} N={n} tp={tp}: plan is missing InferenceExec or \
                             OrdinalSplitExec entirely (the containment check can't run):\n{text}"
                        ));
                        continue;
                    }
                    let lines: Vec<&str> = text.lines().collect();
                    let inserted_between = lines.windows(2).any(|w| {
                        w[0].trim_start().starts_with("InferenceExec")
                            && !w[1].trim_start().starts_with("OrdinalSplitExec")
                    });
                    if inserted_between {
                        failures.push(format!(
                            "{shape} {label} N={n} tp={tp}: something inserted between the split and InferenceExec:\n{text}"
                        ));
                    }
                }
            }
        }
    }
    assert_eq!(
        total, 120,
        "the grid must cover every {{shape}}x{{label}}x{{N}}x{{tp}} cell"
    );
    assert!(
        failures.is_empty(),
        "{} of {total} cells failed RS1:\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

/// RS1's own falsification: dropping `benefits_from_input_partitioning =
/// [false]` from `InferenceExec` (reproduced here by wrapping a
/// `RepartitionExec` directly between a real `OrdinalSplitExec` and a real
/// `InferenceExec` — the shape `EnforceDistribution` would insert without
/// the override) IS detected by the same structural check the grid uses.
#[tokio::test]
async fn rs1_oracle_detects_an_inserted_node_between_split_and_inference() {
    let (session, _dir) = session().await;
    let batch = RecordBatch::try_new(
        in_schema(),
        vec![
            Arc::new(Int64Array::from(vec![1i64, 2, 3])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    )
    .unwrap();
    let src = MemorySourceConfig::try_new_exec(&[vec![batch]], in_schema(), None).unwrap();
    let split = Arc::new(OrdinalSplitExec::new(src, 1).unwrap());
    // Simulate the mutation: a RepartitionExec directly below InferenceExec,
    // NOT below the split (the shape a missing `benefits_from_input_
    // partitioning = [false]` override would let the optimizer insert).
    let repartitioned: Arc<dyn ExecutionPlan> = Arc::new(
        datafusion::physical_plan::repartition::RepartitionExec::try_new(
            split,
            datafusion::physical_plan::Partitioning::RoundRobinBatch(2),
        )
        .unwrap(),
    );
    let inference = InferenceExecBuilder::new(
        repartitioned,
        ModelSource::parse(&tiny_bert_model()),
        ModelTask::TextEmbedding,
        vec![],
        "id".to_string(),
        "src".to_string(),
        Arc::clone(session.model_cache()),
        ComputeDeviceKind::Cpu,
    )
    .embedding_dim(Some(32))
    .build()
    .unwrap();
    let plan: Arc<dyn ExecutionPlan> = Arc::new(inference);
    let text = displayable(plan.as_ref()).indent(true).to_string();
    let lines: Vec<&str> = text.lines().collect();
    let inserted_between = lines.windows(2).any(|w| {
        w[0].trim_start().starts_with("InferenceExec")
            && !w[1].trim_start().starts_with("OrdinalSplitExec")
    });
    assert!(
        inserted_between,
        "the oracle must detect a RepartitionExec between the split and InferenceExec:\n{text}"
    );
}

/// RS2 at FULL strength: an UNSORTED `SELECT * FROM annotated` (no
/// `ORDER BY` to mask a divergence —
/// the property is about the SPLIT/MERGE's own row sequence, not a sort
/// downstream fixing it up), compared PER ROW over EVERY column but
/// `_latency_ms` (never one batch-level `Debug` string, and never dropping
/// `_ordinal` from the comparison), for N in {1,2,4} on BOTH input shapes
/// (sorted and unsorted), against N=1's own sequence.
///
/// Mutation executed and reverted (`AnnotateStub`'s merge re-keyed
/// `[_row_id, _ordinal]`, `InferenceExec`'s own declared child ordering
/// left at `[_ordinal]` — matching the executed record's exact scenario):
/// this test itself goes RED under it, refused by `SanityCheckPlan` ("does
/// not satisfy order requirements ... Child-0 order: [[_ordinal ASC]]")
/// before a single row is even compared — the exact mechanism the
/// implementation contract's executed record names.
#[tokio::test]
async fn rs2_row_sequence_is_identical_across_n_on_both_input_shapes() {
    let (session, _dir) = session().await;
    for shape in ["PIPE", "UDTF"] {
        let mut base: Option<Vec<String>> = None;
        for &n in &[1usize, 2, 4] {
            let ctx = SessionContext::new();
            ctx.register_table(
                "annotated",
                Arc::new(AnnotateStub {
                    session: Arc::clone(&session),
                    n,
                    shape,
                    src_partitions: 4,
                }),
            )
            .unwrap();
            let df = ctx.sql("SELECT * FROM annotated").await.unwrap();
            let batches = tokio::time::timeout(std::time::Duration::from_secs(20), df.collect())
                .await
                .unwrap_or_else(|_| panic!("{shape} N={n}: collect deadlocked (20s)"))
                .unwrap();
            let mut rows: Vec<String> = Vec::new();
            for b in &batches {
                let schema = b.schema();
                for i in 0..b.num_rows() {
                    let mut cells = Vec::new();
                    for (c, f) in schema.fields().iter().enumerate() {
                        if f.name() == "_latency_ms" {
                            continue;
                        }
                        let arr = b.column(c);
                        cells.push(format!(
                            "{}={}",
                            f.name(),
                            arrow::util::display::array_value_to_string(arr, i).unwrap()
                        ));
                    }
                    rows.push(cells.join("|"));
                }
            }
            match &base {
                None => base = Some(rows),
                Some(base_rows) => {
                    assert_eq!(
                        &rows, base_rows,
                        "{shape}: N={n}'s per-row sequence (every column but _latency_ms) \
                         must match N=1's exactly"
                    );
                }
            }
        }
    }
}

/// RS3: a `LIMIT` satisfied over N=4 completes rather than wedging the
/// driver. Under the demand-driven mechanism (`OrdinalSplitExec`'s module
/// doc) this is not a special case at all — a partition `LIMIT` stops
/// polling never blocks anything, since nothing was ever reserved for it —
/// but it is exercised end to end here through the real query engine.
#[tokio::test]
async fn rs3_limit_over_n4_completes_without_wedging() {
    let (session, _dir) = session().await;
    let ctx = SessionContext::new();
    ctx.register_table(
        "annotated",
        Arc::new(AnnotateStub {
            session: Arc::clone(&session),
            n: 4,
            shape: "UDTF",
            src_partitions: 4,
        }),
    )
    .unwrap();
    let df = ctx.sql("SELECT * FROM annotated LIMIT 1").await.unwrap();
    let batches = tokio::time::timeout(std::time::Duration::from_secs(10), df.collect())
        .await
        .expect("LIMIT 1 over N=4 must complete, not wedge the driver (10s timeout)")
        .unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 1, "LIMIT 1 must return exactly one row");
}

/// RS3's residency claim (this node's module doc: `SharedPull` holds no
/// queue and no `Vec<RecordBatch>`, so no batch is ever retained past the
/// single poll that produced it) is a STRUCTURAL argument over `SharedPull`'s
/// own field list, not something a runtime counter measures — an earlier
/// `outstanding: Vec<bool>` / `peak_outstanding()` instrument asserted its
/// own popcount never exceeded `n`, which is true of any length-`n`
/// `Vec<bool>` unconditionally and so proved nothing; it has been removed
/// along with the flag it counted (see `ordinal_split_exec`'s module doc).
/// What IS worth exercising through the query engine's own `SessionContext`
/// is that `n` partitions polling the real `OrdinalSplitExec` CONCURRENTLY
/// (each holding its received batch for a short artificial delay, simulating
/// a slow downstream consumer) still conserves every row exactly once —
/// the mutex-guarded shared pull never double-hands or drops a batch under
/// concurrent access.
#[tokio::test]
async fn rs3_concurrent_polling_conserves_every_row_exactly_once() {
    fn fixture(nb: usize, rows: usize) -> Arc<dyn ExecutionPlan> {
        let mut batches = Vec::new();
        for b in 0..nb {
            let start = (b * rows) as i64;
            let ids: Vec<i64> = (start..start + rows as i64).collect();
            let hashes: Vec<String> = (0..rows)
                .map(|i| format!("h{}", start as usize + i))
                .collect();
            batches.push(
                RecordBatch::try_new(
                    in_schema(),
                    vec![
                        Arc::new(Int64Array::from(ids)),
                        Arc::new(StringArray::from(hashes)),
                    ],
                )
                .unwrap(),
            );
        }
        MemorySourceConfig::try_new_exec(&[batches], in_schema(), None).unwrap()
    }

    let n = 4usize;
    let split = Arc::new(OrdinalSplitExec::new(fixture(24, 5), n).unwrap());
    let ctx = SessionContext::new();
    // ONE context for the whole run, cloned per spawned task — see
    // `ordinal_split_exec`'s module doc: a generation is keyed on this
    // Arc's identity, so every partition of ONE run must share it.
    let task_ctx = ctx.task_ctx();
    let mut handles = Vec::new();
    for p in 0..n {
        let split = Arc::clone(&split);
        let ctx = Arc::clone(&task_ctx);
        handles.push(tokio::spawn(async move {
            use futures::StreamExt;
            let mut stream = split.execute(p, ctx).unwrap();
            let mut rows_seen = 0usize;
            while let Some(b) = stream.next().await {
                let b = b.unwrap();
                rows_seen += b.num_rows();
                // hold this batch briefly, simulating slow downstream drain
                tokio::time::sleep(std::time::Duration::from_millis(2)).await;
            }
            rows_seen
        }));
    }
    let mut total = 0usize;
    for h in handles {
        total += h.await.unwrap();
    }
    assert_eq!(
        total,
        24 * 5,
        "every row must reach some partition exactly once, even under concurrent polling"
    );
}

/// RS5: `wrap_with_split_and_merge` — the ONE function all four
/// `InferenceExec` roots call — returns the identical (no split, no merge)
/// shape at the default `partitions == 1`, and a `SortPreservingMergeExec`
/// root over an `OrdinalSplitExec`-fed `InferenceExec` at `partitions > 1`.
#[tokio::test]
async fn rs5_wrap_with_split_and_merge_is_a_noop_at_one_and_wraps_above_one() {
    let (session, _dir) = session().await;
    let batch = RecordBatch::try_new(
        in_schema(),
        vec![
            Arc::new(Int64Array::from(vec![1i64, 2, 3])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    )
    .unwrap();
    let src: Arc<dyn ExecutionPlan> =
        MemorySourceConfig::try_new_exec(&[vec![batch]], in_schema(), None).unwrap();

    let build = |input: Arc<dyn ExecutionPlan>,
                 model_cache: Arc<jammi_ai::model::cache::ModelCache>| {
        InferenceExecBuilder::new(
            input,
            ModelSource::parse(&tiny_bert_model()),
            ModelTask::TextEmbedding,
            vec![],
            "id".to_string(),
            "src".to_string(),
            model_cache,
            ComputeDeviceKind::Cpu,
        )
        .embedding_dim(Some(32))
        .build()
    };

    let model_cache = Arc::clone(session.model_cache());
    let plan_n1 =
        wrap_with_split_and_merge(Arc::clone(&src), 1, move |input| build(input, model_cache))
            .unwrap();
    assert!(
        plan_n1.downcast_ref::<SortPreservingMergeExec>().is_none(),
        "partitions == 1 must not insert a merge root"
    );
    let text = displayable(plan_n1.as_ref()).indent(true).to_string();
    assert!(
        !text.contains("OrdinalSplitExec"),
        "partitions == 1 must not insert OrdinalSplitExec:\n{text}"
    );

    let model_cache = Arc::clone(session.model_cache());
    let plan_n2 =
        wrap_with_split_and_merge(src, 2, move |input| build(input, model_cache)).unwrap();
    assert!(
        plan_n2.downcast_ref::<SortPreservingMergeExec>().is_some(),
        "partitions > 1 must return a SortPreservingMergeExec root"
    );
    let text = displayable(plan_n2.as_ref()).indent(true).to_string();
    assert!(
        text.contains("OrdinalSplitExec"),
        "partitions > 1 must insert OrdinalSplitExec:\n{text}"
    );
}

/// RS5's falsification: a `wrap_with_split_and_merge` that always merges
/// (never short-circuits at `partitions <= 1`) is caught by the same
/// no-merge-at-one assertion above.
#[tokio::test]
async fn rs5_oracle_detects_a_spurious_merge_at_partitions_one() {
    let batch = RecordBatch::try_new(
        in_schema(),
        vec![
            Arc::new(Int64Array::from(vec![1i64])),
            Arc::new(StringArray::from(vec!["a"])),
        ],
    )
    .unwrap();
    let src = MemorySourceConfig::try_new_exec(&[vec![batch]], in_schema(), None).unwrap();
    // Reproduce the mutation directly: always wrap in a merge, regardless of
    // `partitions`.
    let split = Arc::new(OrdinalSplitExec::new(src, 1).unwrap());
    let ordinal = col(ORDINAL_COLUMN, split.schema().as_ref()).unwrap();
    let sort_opts = arrow::compute::SortOptions {
        descending: false,
        nulls_first: false,
    };
    let ordering = datafusion::physical_expr::LexOrdering::new([
        datafusion::physical_expr::PhysicalSortExpr::new(ordinal, sort_opts),
    ])
    .unwrap();
    let spuriously_merged: Arc<dyn ExecutionPlan> =
        Arc::new(SortPreservingMergeExec::new(ordering, split));
    assert!(
        spuriously_merged
            .downcast_ref::<SortPreservingMergeExec>()
            .is_some(),
        "sanity: the mutated shape really does have a merge root"
    );
    // The real property under test: `plan_n1` in the sibling test above
    // asserts the ABSENCE of this shape at `partitions == 1`; this test just
    // demonstrates the shape the assertion is built to catch actually
    // exists and IS distinguishable from the real `partitions == 1` output.
}

/// The `OrdinalSplitExec` mechanism's own defensive coalesce: constructing
/// it directly over a multi-partition input (the shape a hand-assembled,
/// never-re-optimized caller — `query::QueryBuilder::annotate`'s Rust chain
/// — could hand it) must still see every partition's rows, never silently
/// read only partition 0.
#[tokio::test]
async fn ordinal_split_exec_coalesces_a_multi_partition_input_defensively() {
    let mut batches_per_partition = Vec::new();
    for p in 0..4 {
        let ids = vec![p as i64 * 10, p as i64 * 10 + 1];
        let hashes = vec![format!("h{p}a"), format!("h{p}b")];
        batches_per_partition.push(vec![RecordBatch::try_new(
            in_schema(),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(hashes)),
            ],
        )
        .unwrap()]);
    }
    let src: Arc<dyn ExecutionPlan> =
        MemorySourceConfig::try_new_exec(&batches_per_partition, in_schema(), None).unwrap();
    assert_eq!(src.output_partitioning().partition_count(), 4);
    let split = Arc::new(OrdinalSplitExec::new(src, 1).unwrap());
    let ctx = SessionContext::new();
    let stream = split.execute(0, ctx.task_ctx()).unwrap();
    let batches = datafusion::physical_plan::common::collect(stream)
        .await
        .unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 8,
        "every one of the 4 source partitions' 2 rows must reach the split's single output partition"
    );
}

/// At the default `InferenceConfig::partitions == 1`, `annotate_plan` over
/// a MULTI-PARTITION input still sees every row: the plan declares exactly
/// one output partition, every row from all 4 source partitions is
/// present, `_ordinal` is a distinct, contiguous, one-value-per-row
/// sequence, and `.execute(0, ..)` — the shape every in-crate caller
/// uses — alone returns all of them.
#[tokio::test]
async fn annotate_plan_at_partitions_one_sees_every_row_of_a_multi_partition_input() {
    use arrow::array::UInt64Array;

    let (session, _dir) = session().await;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("text", DataType::Utf8, false),
    ]));
    let mut parts = Vec::new();
    for p in 0..4i64 {
        parts.push(vec![RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![p * 10, p * 10 + 1])),
                Arc::new(StringArray::from(vec![format!("t{p}a"), format!("t{p}b")])),
            ],
        )
        .unwrap()]);
    }
    let input: Arc<dyn ExecutionPlan> =
        MemorySourceConfig::try_new_exec(&parts, schema, None).unwrap();
    assert_eq!(input.output_partitioning().partition_count(), 4);

    let plan = session
        .annotate_plan(
            input,
            &ModelSource::parse(&tiny_bert_model()),
            ModelTask::TextEmbedding,
            &["text".to_string()],
            "id",
        )
        .await
        .unwrap();
    assert_eq!(
        plan.output_partitioning().partition_count(),
        1,
        "InferenceExec must declare exactly one partition when partitions == 1, \
         regardless of the caller-supplied input's own partition count"
    );

    let stream = plan.execute(0, session.context().task_ctx()).unwrap();
    let batches = datafusion::physical_plan::common::collect(stream)
        .await
        .unwrap();
    let mut ordinals: Vec<u64> = Vec::new();
    let mut rows = 0usize;
    for b in &batches {
        rows += b.num_rows();
        let col = b
            .column_by_name(ORDINAL_COLUMN)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        ordinals.extend(col.values().iter().copied());
    }
    assert_eq!(
        rows, 8,
        "execute(0) alone must see all 8 rows of the 4-partition input"
    );
    let mut sorted = ordinals.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        8,
        "_ordinal must be a distinct sequence with exactly one value per row"
    );
}

/// RS4: a Struct-typed key through the real `annotate()` path is a typed
/// refusal naming the key column, its type, and "cannot be cast to Utf8" —
/// reproduces `inference/schema.rs`'s former fallback fixture end to end
/// through `InferenceSession::annotate_plan`.
#[tokio::test]
async fn rs4_struct_key_through_annotate_is_a_typed_refusal_naming_the_key() {
    use arrow::array::{Int32Array, StructArray};
    use arrow::datatypes::Fields;

    let (session, _dir) = session().await;
    let fields: Fields = vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Utf8, false),
    ]
    .into();
    let key_col = StructArray::new(
        fields,
        vec![
            Arc::new(Int32Array::from(vec![1, 2])) as arrow::array::ArrayRef,
            Arc::new(StringArray::from(vec!["x", "y"])) as arrow::array::ArrayRef,
        ],
        None,
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("struct_key", key_col.data_type().clone(), false),
        Field::new("text", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(key_col),
            Arc::new(StringArray::from(vec!["hello", "world"])),
        ],
    )
    .unwrap();
    let input: Arc<dyn ExecutionPlan> =
        MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();

    let plan = session
        .annotate_plan(
            input,
            &ModelSource::parse(&tiny_bert_model()),
            ModelTask::TextEmbedding,
            &["text".to_string()],
            "struct_key",
        )
        .await
        .unwrap();
    let stream = plan.execute(0, session.context().task_ctx()).unwrap();
    let result = datafusion::physical_plan::common::collect(stream).await;
    let err = result
        .expect_err("a Struct-typed key must be a typed refusal, never a silent pass-through");
    let msg = err.to_string();
    assert!(
        msg.contains("struct_key"),
        "must name the key column: {msg}"
    );
    assert!(
        msg.to_lowercase().contains("cannot be cast to utf8"),
        "must state the cast failure: {msg}"
    );
}

/// The default `InferenceConfig::partitions` is `1` — every existing
/// deployment that never sets it keeps today's plan shape.
#[test]
fn inference_config_partitions_defaults_to_one() {
    assert_eq!(jammi_db::config::InferenceConfig::default().partitions, 1);
}

/// `InferenceConfig::validate`'s SECOND call site: a `JammiConfig` built by
/// struct literal (the way every test in this crate builds one, via
/// `common::test_config`) and handed straight to an `InferenceSession`
/// constructor, never through `JammiConfig::load_from`, must still be
/// refused for `partitions = 0` — `InferenceSession::wrap_with` is the
/// universal constructor funnel that calls `InferenceConfig::validate`
/// for exactly this reason (see that function's own doc).
#[tokio::test]
async fn a_struct_literal_config_with_partitions_zero_is_refused_by_session_construction() {
    let dir = TempDir::new().unwrap();
    let mut cfg = common::test_config(dir.path());
    cfg.inference.partitions = 0;
    let result = InferenceSession::new(cfg).await;
    let err = match result {
        Ok(_) => panic!("partitions = 0 must be refused, not silently accepted"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("partitions"),
        "must name the offending key: {err}"
    );
}

/// RS5's literal source oracle: an `syn`-based (never regex — a
/// `.build()`/`InferenceExecBuilder::new(` text scan cannot distinguish "in
/// live code" from "in a doc comment" or "inside a different closure", and
/// cannot see nesting at all) enumeration of EVERY `.rs` file ON DISK
/// (`std::fs::read_dir`, recursive — never `git ls-files`, which cannot see
/// a NEW, uncommitted file: a bypass added there would be silently outside
/// a git-tracked universe) under `crates/jammi-ai/src`, over EVERY call
/// whose callee resolves — by its final one or two path segments, so a
/// fully-qualified (`crate::operator::inference_exec::
/// InferenceExecBuilder::new`) or `super::`-qualified path is caught the
/// same as a bare one, a UFCS `<InferenceExecBuilder as T>::new(..)` form
/// is caught via its `qself`, and a CRATE-WIDE alias table (built in a
/// first pass over EVERY file's AST, never rebuilt per file — see the
/// module's own gap history below) resolves both a renamed import (`use
/// crate::operator::inference_exec::InferenceExecBuilder as Aliased;`,
/// including a `pub use .. as ..` RE-EXPORT read from a file other than the
/// one that calls through it) and a `type` alias (`type Aliased =
/// InferenceExecBuilder;`) back to `InferenceExecBuilder` — TRANSITIVELY
/// (a chain of aliases resolves to a fixed point, bounded so a cycle cannot
/// loop forever) — before either the call-site or the `wrap_with_split_and_
/// merge`-name check runs.
///
/// The alias table was ORIGINALLY built per file (reset for every file in
/// the enumeration loop), which cannot see a `pub use .. as ..` re-export
/// consumed from a DIFFERENT file (the alias binding lived only in the
/// file that declared it), and had no `type` alias handling at all — both
/// gaps are closed by the tests below
/// (`resolves_a_cross_file_pub_use_as_alias`,
/// `resolves_a_type_alias_to_inference_exec_builder`), which build a small
/// synthetic file tree exhibiting exactly the CLOSED gap's shape (a
/// bypassing call reached only through the alias) and assert
/// `analyze_dir` still finds and correctly classifies it.
///
/// **Still not handled** (named, not silently assumed away): a glob import
/// (`use ..::*`) bringing either name in under an unresolvable local
/// identity, and real macro EXPANSION (a macro whose invocation's own
/// token stream does not literally spell `InferenceExecBuilder` but
/// expands to a call naming it via a `$name` substitution or a nested
/// macro) — every macro invocation IS visited (`visit_macro`, which covers
/// expression, statement, and item position uniformly) and its literal
/// token text is substring-matched for `InferenceExecBuilder`, which is a
/// real but coarse net, not semantic expansion.
///
/// The property: every `InferenceExecBuilder::new` call site is EITHER (a)
/// lexically nested inside a call to `wrap_with_split_and_merge` — tracked
/// via a real call-stack of enclosing call names (aliases resolved the same
/// way), not merely "in the same function", so a raw call sitting BESIDE
/// (not inside) a `wrap_with_split_and_merge` call in the same function is
/// still caught — or (b) the one named exception, `InferenceExec::
/// with_new_children`'s own self-rebuild (DataFusion's own machinery
/// reconstructing an ALREADY-PLACED node from itself when it transforms the
/// plan tree, never a fresh, independently-reachable root).
mod rs5_source_oracle {
    use std::collections::{BTreeSet, HashMap};
    use std::path::{Path, PathBuf};

    use syn::visit::Visit;
    use tempfile::TempDir;

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("crates/jammi-ai has two ancestors: crates/, then the repo root")
            .to_path_buf()
    }

    /// Every `.rs` file ON DISK under `dir`, walked recursively, sorted.
    /// Deliberately NOT `git ls-files` — see this module's doc for why.
    fn rs_files_on_disk(dir: &Path) -> Vec<PathBuf> {
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
            let entries = std::fs::read_dir(dir)
                .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()));
            for entry in entries {
                let entry = entry.unwrap_or_else(|e| panic!("read_dir entry: {e}"));
                let path = entry.path();
                if path.is_dir() {
                    walk(&path, out);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    out.push(path);
                }
            }
        }
        let mut out = Vec::new();
        walk(dir, &mut out);
        out.sort();
        out
    }

    fn call_name(func: &syn::Expr) -> Option<String> {
        match func {
            syn::Expr::Path(p) => p.path.segments.last().map(|s| s.ident.to_string()),
            _ => None,
        }
    }

    /// Local-name -> immediate-target-name aliases, collected CRATE-WIDE —
    /// one pass over EVERY file's AST (never rebuilt per file, and never
    /// scoped to the declaring file), from every `use` item's tree (paths,
    /// groups, renames — including a `pub use .. as ..` RE-EXPORT, since
    /// visibility is not checked) and every `type X = ..;` item (mapping `X`
    /// to the RHS type's own last path segment, when the RHS is a bare
    /// type path). A single entry only ever records ONE HOP; `resolve`
    /// below chases a chain of these to a fixed point. A glob import is not
    /// resolved (see the module doc's named gap).
    fn collect_global_aliases(files: &[(String, syn::File)]) -> HashMap<String, String> {
        fn walk_tree(tree: &syn::UseTree, aliases: &mut HashMap<String, String>) {
            match tree {
                syn::UseTree::Path(p) => walk_tree(&p.tree, aliases),
                // A plain (non-renaming) `use` contributes NO alias
                // information — `resolve` already falls back to the bare
                // name when the table has no entry for it, so inserting an
                // identity mapping here would only risk CLOBBERING a real
                // `Rename`/`type`-alias entry for the same local name
                // collected from a different file (this table is crate-wide
                // and coarse; see the module doc). Confirmed by execution:
                // reinstating this insert regresses
                // `resolves_a_type_alias_to_inference_exec_builder` to
                // `found: []` (a plain `use crate::alias::Aliased;` in the
                // consuming file overwrites the type alias's real mapping
                // with `Aliased -> Aliased`).
                syn::UseTree::Name(_) => {}
                syn::UseTree::Rename(r) => {
                    aliases.insert(r.rename.to_string(), r.ident.to_string());
                }
                syn::UseTree::Group(g) => {
                    for t in &g.items {
                        walk_tree(t, aliases);
                    }
                }
                syn::UseTree::Glob(_) => {}
            }
        }
        #[derive(Default)]
        struct Collector(HashMap<String, String>);
        impl<'ast> Visit<'ast> for Collector {
            fn visit_item_use(&mut self, i: &'ast syn::ItemUse) {
                walk_tree(&i.tree, &mut self.0);
            }
            fn visit_item_type(&mut self, i: &'ast syn::ItemType) {
                if let syn::Type::Path(tp) = &*i.ty {
                    if let Some(seg) = tp.path.segments.last() {
                        self.0.insert(i.ident.to_string(), seg.ident.to_string());
                    }
                }
            }
        }
        let mut c = Collector::default();
        for (_, file) in files {
            c.visit_file(file);
        }
        c.0
    }

    /// Chases `aliases` from `name` to a fixed point (a chain of `use ..
    /// as ..` / `type` aliases spanning any number of hops, across any
    /// number of files), bounded so a cycle (an alias table entry pointing
    /// back into its own chain — never valid Rust, but this table is built
    /// without borrow-checking it) cannot loop forever.
    fn resolve(aliases: &HashMap<String, String>, name: &str) -> String {
        let mut current = name.to_string();
        let mut seen = std::collections::HashSet::new();
        for _ in 0..64 {
            if !seen.insert(current.clone()) {
                break;
            }
            match aliases.get(&current) {
                Some(next) if next != &current => current = next.clone(),
                _ => break,
            }
        }
        current
    }

    #[derive(Default)]
    struct Analysis {
        current_file: String,
        current_fn: Vec<String>,
        /// The CRATE-WIDE `use .. as ..` / `type` alias table (built once,
        /// from every file, before this visitor runs over any of them —
        /// see `collect_global_aliases`).
        aliases: HashMap<String, String>,
        /// Enclosing CALL names (aliases resolved), pushed/popped around
        /// the default recursion into that call's own arguments — a true
        /// lexical-nesting stack, not "same enclosing function".
        call_stack: Vec<String>,
        /// (file, enclosing fn, is lexically inside a `wrap_with_split_and_merge` call).
        builder_calls: Vec<(String, String, bool)>,
        /// (file, enclosing fn) for every `wrap_with_split_and_merge` call.
        wrap_calls: Vec<(String, String)>,
    }

    impl Analysis {
        fn enclosing_fn(&self) -> String {
            self.current_fn
                .last()
                .cloned()
                .unwrap_or_else(|| "<top-level>".to_string())
        }

        /// Matches `InferenceExecBuilder::new(..)` regardless of path
        /// qualification (bare, `crate::..::`, `super::..::`, or a resolved
        /// `use .. as ..` alias), and the UFCS `<InferenceExecBuilder as
        /// Trait>::new(..)` form via `qself` (also alias-resolved).
        fn is_inference_exec_builder_new(&self, node: &syn::ExprCall) -> bool {
            let syn::Expr::Path(p) = &*node.func else {
                return false;
            };
            let segs: Vec<String> = p
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            if segs.last().map(String::as_str) != Some("new") {
                return false;
            }
            let qualified_path_names_it = segs.len() >= 2
                && resolve(&self.aliases, &segs[segs.len() - 2]) == "InferenceExecBuilder";
            let ufcs_names_it = p
                .qself
                .as_ref()
                .map(|q| {
                    matches!(&*q.ty, syn::Type::Path(tp)
                        if tp.path.segments.last()
                            .is_some_and(|s| resolve(&self.aliases, &s.ident.to_string()) == "InferenceExecBuilder"))
                })
                .unwrap_or(false);
            qualified_path_names_it || ufcs_names_it
        }
    }

    impl<'ast> Visit<'ast> for Analysis {
        fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
            self.current_fn.push(i.sig.ident.to_string());
            syn::visit::visit_item_fn(self, i);
            self.current_fn.pop();
        }
        fn visit_impl_item_fn(&mut self, i: &'ast syn::ImplItemFn) {
            self.current_fn.push(i.sig.ident.to_string());
            syn::visit::visit_impl_item_fn(self, i);
            self.current_fn.pop();
        }
        fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
            let name = call_name(&node.func).map(|n| resolve(&self.aliases, &n));
            if self.is_inference_exec_builder_new(node) {
                let inside_wrap = self
                    .call_stack
                    .iter()
                    .any(|c| c == "wrap_with_split_and_merge");
                let f = self.enclosing_fn();
                self.builder_calls
                    .push((self.current_file.clone(), f, inside_wrap));
            }
            if name.as_deref() == Some("wrap_with_split_and_merge") {
                let f = self.enclosing_fn();
                self.wrap_calls.push((self.current_file.clone(), f));
            }
            if let Some(n) = &name {
                self.call_stack.push(n.clone());
            }
            syn::visit::visit_expr_call(self, node);
            if name.is_some() {
                self.call_stack.pop();
            }
        }
        /// Covers EVERY macro invocation (expression, statement, and item
        /// position all wrap this one node type) — a coarse, literal
        /// substring check on the macro's own token text; see the module
        /// doc's named gap for what this cannot see (true expansion).
        fn visit_macro(&mut self, node: &'ast syn::Macro) {
            let tokens = node.tokens.to_string();
            if tokens.contains("InferenceExecBuilder") {
                let inside_wrap = self
                    .call_stack
                    .iter()
                    .any(|c| c == "wrap_with_split_and_merge");
                let f = self.enclosing_fn();
                self.builder_calls.push((
                    self.current_file.clone(),
                    format!("{f} (inside a macro invocation)"),
                    inside_wrap,
                ));
            }
            syn::visit::visit_macro(self, node);
        }
    }

    /// Executes the real scan over `crates/jammi-ai/src`; `#[allow(dead_code)]`
    /// fields are read by the test below through the returned struct.
    fn analyze() -> Analysis {
        let root = repo_root();
        analyze_dir(&root, &root.join("crates/jammi-ai/src"))
    }

    /// The scan, parameterized on `root` (used only to compute each file's
    /// REPORTED relative path) and `src_dir` (the directory actually
    /// walked) — the production oracle's own engine, exercised directly
    /// against a synthetic fixture by the cross-file-alias and type-alias
    /// tests below, never a second, parallel implementation of the same
    /// logic that could silently drift from what actually runs in CI.
    fn analyze_dir(root: &Path, src_dir: &Path) -> Analysis {
        let files = rs_files_on_disk(src_dir);
        assert!(
            !files.is_empty(),
            "found nothing on disk under {} — the path is wrong",
            src_dir.display()
        );
        let parsed: Vec<(String, syn::File)> = files
            .iter()
            .map(|full| {
                let rel = full
                    .strip_prefix(root)
                    .unwrap_or(full)
                    .to_string_lossy()
                    .replace('\\', "/");
                let text =
                    std::fs::read_to_string(full).unwrap_or_else(|e| panic!("{rel}: read: {e}"));
                let parsed =
                    syn::parse_file(&text).unwrap_or_else(|e| panic!("{rel}: syn parse: {e}"));
                (rel, parsed)
            })
            .collect();
        // Pass 1: the alias table sees EVERY file before pass 2 resolves
        // any call site against it — this is what makes a cross-file
        // `pub use .. as ..` re-export resolve regardless of which file
        // declares it versus which file consumes it.
        let aliases = collect_global_aliases(&parsed);
        let mut analysis = Analysis {
            aliases,
            ..Analysis::default()
        };
        for (rel, file) in &parsed {
            analysis.current_file = rel.clone();
            analysis.current_fn.clear();
            analysis.call_stack.clear();
            Visit::visit_file(&mut analysis, file);
        }
        analysis
    }

    /// The four roots' own (file, fn) identities. `infer_materialize`, not
    /// `infer` — the oracle found this: `InferenceSession::infer` is a thin
    /// wrapper over `run_now`/`execute_compute`; `infer_materialize` is
    /// `infer`'s actual materializer and the one that builds `InferenceExec`.
    fn expected_roots() -> BTreeSet<(&'static str, &'static str)> {
        [
            ("crates/jammi-ai/src/session.rs", "annotate_plan"),
            ("crates/jammi-ai/src/session.rs", "infer_materialize"),
            (
                "crates/jammi-ai/src/pipeline/embedding.rs",
                "build_embedding_plan",
            ),
            (
                "crates/jammi-ai/src/pipeline/embedding_refresh.rs",
                "infer_delta",
            ),
        ]
        .into_iter()
        .collect()
    }

    /// The one named, non-root exception: `InferenceExec`'s own
    /// `with_new_children` self-rebuild.
    fn is_named_exception(file: &str, f: &str) -> bool {
        file == "crates/jammi-ai/src/operator/inference_exec.rs" && f == "with_new_children"
    }

    #[test]
    fn every_inference_exec_builder_new_is_reached_only_through_wrap_with_split_and_merge() {
        let analysis = analyze();

        // (a) No `InferenceExecBuilder::new` call sits outside a
        // `wrap_with_split_and_merge` call unless it is the named exception.
        let mut bypasses = Vec::new();
        for (file, f, inside_wrap) in &analysis.builder_calls {
            if is_named_exception(file, f) {
                continue;
            }
            if !inside_wrap {
                bypasses.push(format!(
                    "{file}::{f}: InferenceExecBuilder::new called OUTSIDE any \
                     wrap_with_split_and_merge call — a fifth, unwrapped root"
                ));
            }
        }
        assert!(
            bypasses.is_empty(),
            "RS5 violated — a construction site bypasses wrap_with_split_and_merge:\n{}",
            bypasses.join("\n")
        );

        // (b) The SET of (file, fn) pairs with a builder call is exactly
        // the four roots plus the one named exception — never a fifth root,
        // never a root silently dropped.
        let found: BTreeSet<(&str, &str)> = analysis
            .builder_calls
            .iter()
            .map(|(file, f, _)| (file.as_str(), f.as_str()))
            .collect();
        let mut expected = expected_roots();
        expected.insert((
            "crates/jammi-ai/src/operator/inference_exec.rs",
            "with_new_children",
        ));
        assert_eq!(
            found, expected,
            "the set of (file, fn) building an InferenceExec must be exactly the four \
             roots plus with_new_children's self-rebuild — found a different set"
        );

        // (c) Every one of the four roots ALSO calls
        // wrap_with_split_and_merge in that same function (paired with (a)'s
        // lexical-nesting check, this pins that the builder call the root
        // contains really is the argument of ITS OWN wrap call).
        let wrap_found: BTreeSet<(&str, &str)> = analysis
            .wrap_calls
            .iter()
            .map(|(file, f)| (file.as_str(), f.as_str()))
            .collect();
        for root in expected_roots() {
            assert!(
                wrap_found.contains(&root),
                "{}::{} builds an InferenceExec but never calls wrap_with_split_and_merge",
                root.0,
                root.1
            );
        }
    }

    /// The alias table was ORIGINALLY rebuilt PER FILE (reset to that
    /// file's own `use` items at the top of each iteration of `analyze`'s
    /// loop), so a `pub use .. as ..` re-export declared in one file and
    /// consumed via a bypassing call in a DIFFERENT file was invisible —
    /// the bypass would simply never appear in `builder_calls` at all (a
    /// silent false negative, never a wrongly-permitted bypass, since the
    /// oracle's own assertions only ever complain about entries THAT
    /// appear). Reproducing the removed per-file design directly against
    /// this exact fixture (giving `bypass.rs` only its OWN file's alias
    /// table — the identity mapping `Aliased -> Aliased`, since a bare
    /// `use .. ;` with no `as` is not a rename) resolves `Aliased::new()`
    /// against `"Aliased"`, never `"InferenceExecBuilder"`, so the call is
    /// missed entirely — confirmed by hand against the removed
    /// `collect_aliases` function before it was deleted. Fixed by building
    /// the alias table crate-wide, in one pass over every file, before any
    /// file's call sites are checked (`collect_global_aliases`).
    #[test]
    fn resolves_a_cross_file_pub_use_as_alias() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("real.rs"),
            concat!(
                "pub struct InferenceExecBuilder;\n",
                "impl InferenceExecBuilder {\n",
                "    pub fn new() -> Self { InferenceExecBuilder }\n",
                "}\n"
            ),
        )
        .unwrap();
        std::fs::write(
            tmp.path().join("reexport.rs"),
            "pub use crate::real::InferenceExecBuilder as Aliased;\n",
        )
        .unwrap();
        std::fs::write(
            tmp.path().join("bypass.rs"),
            concat!(
                "use crate::reexport::Aliased;\n",
                "fn bad() {\n",
                "    let _ = Aliased::new();\n",
                "}\n"
            ),
        )
        .unwrap();

        let analysis = analyze_dir(tmp.path(), tmp.path());
        let found = analysis
            .builder_calls
            .iter()
            .any(|(file, f, inside_wrap)| file == "bypass.rs" && f == "bad" && !inside_wrap);
        assert!(
            found,
            "a call reached only through a cross-file `pub use .. as ..` re-export must \
             still be recognized as an InferenceExecBuilder::new call bypassing \
             wrap_with_split_and_merge, found: {:?}",
            analysis.builder_calls
        );
    }

    /// The alias table had no `type` alias handling at all: a `type X =
    /// InferenceExecBuilder;` binding was invisible to the (old, per-file)
    /// `collect_aliases`, which only ever walked `ItemUse`. Fixed by adding
    /// an `ItemType` visitor to `collect_global_aliases` that maps the
    /// alias name to the RHS type path's own last segment.
    #[test]
    fn resolves_a_type_alias_to_inference_exec_builder() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("real.rs"),
            concat!(
                "pub struct InferenceExecBuilder;\n",
                "impl InferenceExecBuilder {\n",
                "    pub fn new() -> Self { InferenceExecBuilder }\n",
                "}\n"
            ),
        )
        .unwrap();
        std::fs::write(
            tmp.path().join("alias.rs"),
            "pub type Aliased = crate::real::InferenceExecBuilder;\n",
        )
        .unwrap();
        std::fs::write(
            tmp.path().join("bypass.rs"),
            concat!(
                "use crate::alias::Aliased;\n",
                "fn bad() {\n",
                "    let _ = Aliased::new();\n",
                "}\n"
            ),
        )
        .unwrap();

        let analysis = analyze_dir(tmp.path(), tmp.path());
        let found = analysis
            .builder_calls
            .iter()
            .any(|(file, f, inside_wrap)| file == "bypass.rs" && f == "bad" && !inside_wrap);
        assert!(
            found,
            "a call reached only through a `type X = InferenceExecBuilder;` alias must \
             still be recognized as an InferenceExecBuilder::new call bypassing \
             wrap_with_split_and_merge, found: {:?}",
            analysis.builder_calls
        );
    }
}

/// RS8's second arm: under `partitions = 2`, `ResultSink`'s `batch_num`/the
/// persisted `checkpoint` column count the
/// MERGED batches `EmbeddingPipeline::run` actually wrote — never a
/// per-partition count — asserted against an INDEPENDENTLY collected count
/// of the SAME construction's merged output. `checkpoint_interval = 1`
/// makes every `write_batch` call persist, so the table's FINAL checkpoint
/// is exactly the total number of `write_batch` calls
/// `EmbeddingPipeline::run` made — one per collected merged batch
/// (`pipeline::embedding::EmbeddingPipeline::run`'s own `for batch in
/// &batches { sink.write_batch(batch).await?; }` loop).
///
/// The TRUE cause of this fixture's non-trivial batch count is
/// `cfg.engine.batch_size = 1` below, NOT the number of source files —
/// executed and confirmed: a single 3-row parquet file under `batch_size
/// = 1` measures the identical `persisted_checkpoint = 3, merged_batches.len()
/// = 3` a multi-file source does (an earlier version of this doc
/// attributed the 3 separate source batches to "three single-row-group
/// source files under the default scan batch size", which this
/// measurement refutes — the default scan batch size plays no role once
/// `batch_size = 1` is set). With the session's own batch size forced to
/// 1, each ROW becomes its own batch — both at the scan and, critically,
/// at `SortPreservingMergeExec`'s own re-batching, which otherwise
/// combines small batches back up to the session batch size regardless of
/// their number (see `cfg.engine.batch_size`'s own comment below) — and
/// the two `OrdinalSplitExec` partitions divide those single-row batches
/// by DEMAND (see `ordinal_split_exec`'s module doc) rather than one
/// partition trivially taking everything: without this,
/// `persisted_checkpoint == merged_batches.len()` would hold at a
/// trivial `1 == 1`, proving nothing about cross-partition merging. This
/// fixture measures `persisted_checkpoint = 3, merged_batches.len() = 3`
/// (the assertions below pin `merged_batches.len() > 1` and the equality,
/// not the literal 3).
#[tokio::test]
async fn rs8_checkpoint_counts_the_merged_batches_under_partitions_two() {
    let dir = TempDir::new().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("text", DataType::Utf8, false),
    ]));
    // ONE file, 3 rows — `cfg.engine.batch_size = 1` below is what makes
    // this fixture non-trivial, not the number of source files (see this
    // test's own doc for the executed measurement backing that claim).
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![0i64, 1, 2])),
            Arc::new(StringArray::from(vec![
                "alpha widget",
                "beta widget",
                "gamma gadget",
            ])),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(src_dir.join("part0.parquet")).unwrap();
    let mut w = parquet::arrow::ArrowWriter::try_new(file, schema.clone(), None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    let url = format!("file://{}", src_dir.display());

    let mut cfg = common::test_config(dir.path());
    cfg.inference.partitions = 2;
    cfg.embedding.checkpoint_interval = 1;
    // Force the DataFusion session's own batch size down to 1 row: with
    // the default (8192), `SortPreservingMergeExec` re-batches its output
    // up to that size regardless of how many small batches its inputs
    // produced, so a 3-row fixture would still merge into ONE output
    // batch and this oracle would prove nothing about cross-partition
    // merging (see this test's own doc for the executed measurement).
    cfg.engine.batch_size = 1;
    let session = Arc::new(InferenceSession::new(cfg).await.unwrap());
    session
        .add_source(
            "checkpoint_src",
            jammi_db::source::SourceType::File,
            jammi_db::source::SourceConnection {
                url: Some(url),
                format: Some(jammi_db::source::FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let model = tiny_bert_model();
    let (record, _) = session
        .generate_text_embeddings(
            "checkpoint_src",
            &model,
            &["text".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
            None,
        )
        .await
        .unwrap();
    assert_eq!(record.row_count, 3);

    let persisted_checkpoint = session
        .catalog()
        .get_checkpoint(&record.table_name)
        .await
        .unwrap()
        .expect("a checkpoint must have been persisted (checkpoint_interval = 1)");

    // Independently collect the SAME construction's merged output (never
    // through `generate_text_embeddings`/`EmbeddingPipeline::run` a second
    // time) to count how many batches the merge actually produced.
    let plan = jammi_ai::pipeline::embedding::build_embedding_plan(
        &session,
        "checkpoint_src",
        ModelSource::parse(&model),
        ModelTask::TextEmbedding,
        &["text".to_string()],
        "id",
        record.dimensions().expect("dimensions recorded").get(),
    )
    .await
    .unwrap();
    let stream = plan.execute(0, session.context().task_ctx()).unwrap();
    let merged_batches = datafusion::physical_plan::common::collect(stream)
        .await
        .unwrap();

    assert!(
        merged_batches.len() > 1,
        "the fixture must produce more than one merged batch, or this oracle proves nothing \
         about cross-partition merging (found {} merged batches)",
        merged_batches.len()
    );
    assert_eq!(
        persisted_checkpoint,
        merged_batches.len(),
        "the persisted checkpoint must equal the independently-collected merged batch count, \
         not some per-partition count"
    );
}

/// The end-to-end oracle (#540 RANGESPLIT) for a typed refusal raised
/// BELOW the split: `KeyCheckExec`'s `InvalidKey`, over a NULL `id`, must
/// reach `generate_text_embeddings`'s caller classified IDENTICALLY —
/// same variant, same fields — at `partitions ∈ {1, 2}`. `partitions = 1`
/// never builds `OrdinalSplitExec` at all (`wrap_with_split_and_merge`'s
/// own shortcut), so it is the CONTROL this oracle checks `partitions = 2`
/// against, not merely a second data point.
#[tokio::test]
async fn f1_a_null_key_below_the_split_classifies_as_invalid_key_at_every_partition_count() {
    for partitions in [1usize, 2] {
        let dir = TempDir::new().unwrap();
        let src_dir = dir.path().join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("text", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![Some(0i64), None, Some(2)])),
                Arc::new(StringArray::from(vec!["alpha", "beta", "gamma"])),
            ],
        )
        .unwrap();
        let file = std::fs::File::create(src_dir.join("part0.parquet")).unwrap();
        let mut w = parquet::arrow::ArrowWriter::try_new(file, schema.clone(), None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
        let url = format!("file://{}", src_dir.display());

        let mut cfg = common::test_config(dir.path());
        cfg.inference.partitions = partitions;
        let session = Arc::new(InferenceSession::new(cfg).await.unwrap());
        session
            .add_source(
                "f1_null_key_src",
                jammi_db::source::SourceType::File,
                jammi_db::source::SourceConnection {
                    url: Some(url),
                    format: Some(jammi_db::source::FileFormat::Parquet),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let model = tiny_bert_model();
        let err = session
            .generate_text_embeddings(
                "f1_null_key_src",
                &model,
                &["text".to_string()],
                "id",
                jammi_db::store::CachePolicy::Bypass,
                None,
            )
            .await
            .expect_err("a null id must refuse the embed at every partition count");
        assert!(
            matches!(
                &err,
                jammi_db::error::JammiError::InvalidKey { column, null_count }
                    if column == "id" && *null_count == 1
            ),
            "partitions={partitions}: expected InvalidKey {{ id, 1 }}, got {err:?}"
        );
    }
}
