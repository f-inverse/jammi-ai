//! Partitioned inference: one plan shape at every fan-out, and the same rows
//! and bytes out of it.
//!
//! Every oracle here runs the REAL production planner
//! (`operator::inference_exec::plan_inference`) and the real nodes it builds,
//! never a stand-in mimicking their declared `PlanProperties`: a stub would
//! assert the stub's declarations, not the shipped ones.

use std::num::NonZeroUsize;
use std::sync::Arc;

use arrow::array::{Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::datasource::TableType;
use datafusion::logical_expr::Expr;
use datafusion::physical_expr::expressions::col;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::{
    displayable, ExecutionPlan, ExecutionPlanProperties, Partitioning,
};
use datafusion::prelude::{SessionConfig, SessionContext};
use tempfile::TempDir;

use jammi_ai::session::InferenceSession;
use jammi_datafusion::inference::chunk::CHUNK_COLUMN;
use jammi_datafusion::inference::schema::{build_output_schema, ORDINAL_COLUMN};
use jammi_datafusion::ComputeDeviceKind;
use jammi_datafusion::ModelSource;
use jammi_datafusion::ModelTask;
use jammi_datafusion::{plan_inference, InferenceExec, InferenceSpec};
use jammi_datafusion::{NumberedInputExec, RowOrder};
use jammi_numerics::ChunkBudget;

use crate::common;

/// Rows per forward chunk in these fixtures: small, so a few dozen rows span
/// several chunks and a fan-out genuinely spreads them.
const BATCH_SIZE: usize = 4;

/// Padded tokens per forward chunk: wide enough that the row cap alone cuts
/// these fixtures' chunks.
const BATCH_TOKENS: usize = 4096;

fn chunk_budget() -> ChunkBudget {
    ChunkBudget {
        rows: NonZeroUsize::new(BATCH_SIZE).unwrap(),
        tokens: NonZeroUsize::new(BATCH_TOKENS).unwrap(),
    }
}

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
        Field::new("text", DataType::Utf8, true),
        Field::new("_content_hash", DataType::Utf8, true),
    ]))
}

/// A passage whose length varies with `i`, so the rows of one forward pad to
/// different widths and a chunk holding different rows would embed
/// differently.
fn passage(i: i64) -> String {
    const WORDS: [&str; 8] = [
        "a method for",
        "coating",
        "semiconductor wafers",
        "with a thin film",
        "of oxide",
        "under vacuum",
        "at low temperature",
        "in a single pass",
    ];
    (0..=(i as usize * 5) % 11)
        .map(|w| WORDS[(w + i as usize) % WORDS.len()])
        .collect::<Vec<_>>()
        .join(" ")
}

fn rows(ids: &[i64]) -> RecordBatch {
    RecordBatch::try_new(
        in_schema(),
        vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(StringArray::from_iter_values(
                ids.iter().map(|&i| passage(i)),
            )),
            Arc::new(StringArray::from_iter_values(
                ids.iter().map(|&i| format!("hash-{i:04}")),
            )),
        ],
    )
    .unwrap()
}

/// The two inputs a model reads.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Shape {
    /// A source scan: four partitions of distinct, unsorted rows, read in the
    /// keyed total order.
    Keyed,
    /// An `annotate` input: unsorted, read in arrival order. Its four
    /// partitions each hold the SAME single batch, so whichever way the
    /// coalesce interleaves them the arrival sequence is one sequence, and
    /// any difference between two runs is the fan-out's doing.
    Arrival,
}

impl Shape {
    fn source(self) -> Arc<dyn ExecutionPlan> {
        let partitions: Vec<Vec<RecordBatch>> = match self {
            Shape::Keyed => (0..4i64)
                .map(|p| {
                    let ids: Vec<i64> = (0..9).map(|r| (r * 4 + p) * 7 % 36).collect();
                    vec![rows(&ids[..5]), rows(&ids[5..])]
                })
                .collect(),
            Shape::Arrival => (0..4)
                .map(|_| vec![rows(&[21, 3, 17, 8, 30, 2, 11, 26, 5])])
                .collect(),
        };
        MemorySourceConfig::try_new_exec(&partitions, in_schema(), None).unwrap()
    }

    fn order(self) -> RowOrder {
        match self {
            Shape::Keyed => RowOrder::Keyed {
                key_column: "id".into(),
                tie_breakers: vec![jammi_db::store::schema::CONTENT_HASH_COLUMN.to_string()],
            },
            Shape::Arrival => RowOrder::Arrival,
        }
    }
}

fn spec(partitions: usize) -> InferenceSpec {
    InferenceSpec {
        source: ModelSource::parse(&tiny_bert_model()),
        task: ModelTask::TextEmbedding,
        content_columns: vec!["text".to_string()],
        key_column: "id".to_string(),
        source_id: "src".to_string(),
        chunk: chunk_budget(),
        embedding_dim: Some(32),
        regression_form: None,
        passthrough: Vec::new(),
        device_kind: ComputeDeviceKind::Cpu,
        partitions: NonZeroUsize::new(partitions).unwrap(),
    }
}

fn planned(session: &InferenceSession, shape: Shape, partitions: usize) -> Arc<dyn ExecutionPlan> {
    plan_inference(
        shape.source(),
        shape.order(),
        spec(partitions),
        session.inference_runtime(),
    )
    .unwrap()
}

/// A table whose `scan()` is the production plan, so SQL over it takes that
/// plan back through DataFusion's optimizer.
struct Annotated {
    session: Arc<InferenceSession>,
    shape: Shape,
    partitions: usize,
}

impl std::fmt::Debug for Annotated {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Annotated")
            .field("shape", &self.shape)
            .field("partitions", &self.partitions)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl TableProvider for Annotated {
    fn schema(&self) -> SchemaRef {
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
        let plan = planned(&self.session, self.shape, self.partitions);
        let Some(projection) = projection else {
            return Ok(plan);
        };
        let schema = plan.schema();
        let exprs = projection
            .iter()
            .map(|&i| {
                let name = schema.field(i).name();
                Ok((
                    col(name, schema.as_ref())? as Arc<dyn datafusion::physical_expr::PhysicalExpr>,
                    name.to_string(),
                ))
            })
            .collect::<datafusion::error::Result<Vec<_>>>()?;
        Ok(Arc::new(
            datafusion::physical_plan::projection::ProjectionExec::try_new(exprs, plan)?,
        ))
    }
}

fn sql_context(
    session: &Arc<InferenceSession>,
    shape: Shape,
    partitions: usize,
    target_partitions: usize,
) -> SessionContext {
    let ctx = SessionContext::new_with_config(
        SessionConfig::new().with_target_partitions(target_partitions),
    );
    ctx.register_table(
        "annotated",
        Arc::new(Annotated {
            session: Arc::clone(session),
            shape,
            partitions,
        }),
    )
    .unwrap();
    ctx
}

/// Every row of `batches` as one string over every column but `_latency_ms`
/// (a wall-clock measurement) — the vector included, so two runs compare
/// equal only if they embedded to the same floats.
fn rendered_rows(batches: &[RecordBatch]) -> Vec<String> {
    batches
        .iter()
        .flat_map(|b| {
            let schema = b.schema();
            (0..b.num_rows())
                .map(|i| {
                    schema
                        .fields()
                        .iter()
                        .enumerate()
                        .filter(|(_, f)| f.name() != "_latency_ms")
                        .map(|(c, f)| {
                            let value = arrow::util::display::array_value_to_string(b.column(c), i)
                                .unwrap();
                            format!("{}={value}", f.name())
                        })
                        .collect::<Vec<_>>()
                        .join("|")
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// The first node named `name`, searching `plan` depth-first.
fn find<'a>(plan: &'a Arc<dyn ExecutionPlan>, name: &str) -> Option<&'a Arc<dyn ExecutionPlan>> {
    let mut stack = vec![plan];
    while let Some(node) = stack.pop() {
        if node.name() == name {
            return Some(node);
        }
        stack.extend(node.children());
    }
    None
}

/// The plan, as built: the one sort restores `_ordinal` order over the
/// model's chunk-ordered output; at a fan-out of one the exchange and the
/// coalesce above the model are absent (each is the identity over one
/// partition), at four they are the stock operators, the exchange keyed on
/// the chunk id.
#[tokio::test]
async fn the_planned_shape_at_one_and_at_four() {
    let (session, _dir) = session().await;
    let model = ModelSource::parse(&tiny_bert_model());
    let inference = |n: usize| {
        format!(
            "InferenceExec: model={model}, task=TextEmbedding, columns=[\"text\"], \
             batch_size={BATCH_SIZE}, batch_tokens={BATCH_TOKENS}, partitions={n}"
        )
    };

    let one = planned(&session, Shape::Keyed, 1);
    let text = displayable(one.as_ref()).indent(false).to_string();
    println!("N=1\n{text}");
    let lines: Vec<&str> = text.lines().map(str::trim_start).collect();
    let sort = "SortExec: expr=[_ordinal@1 ASC NULLS LAST], preserve_partitioning=[false]";
    assert_eq!(lines[0], sort);
    assert_eq!(lines[1], inference(1));
    let numbered = format!(
        "NumberedInputExec: order=key(id, _content_hash), batch_size={BATCH_SIZE}, batch_tokens={BATCH_TOKENS}"
    );
    assert_eq!(lines[2], numbered);
    assert_eq!(lines[3], "CoalescePartitionsExec");
    assert!(lines[4].starts_with("DataSourceExec"), "{text}");
    assert_eq!(lines.len(), 5, "{text}");

    let four = planned(&session, Shape::Keyed, 4);
    let text = displayable(four.as_ref()).indent(false).to_string();
    println!("N=4\n{text}");
    let lines: Vec<&str> = text.lines().map(str::trim_start).collect();
    assert_eq!(lines[0], sort);
    assert_eq!(lines[1], "CoalescePartitionsExec");
    assert_eq!(lines[2], inference(4));
    assert_eq!(
        lines[3],
        format!(
            "RepartitionExec: partitioning=Hash([{CHUNK_COLUMN}@4], 4), \
             input_partitions=1, maintains_sort_order=true"
        )
    );
    assert_eq!(lines[4], numbered);
    assert_eq!(lines[5], "CoalescePartitionsExec");
    assert!(lines[6].starts_with("DataSourceExec"), "{text}");
    assert_eq!(lines.len(), 7, "{text}");
    assert_eq!(four.output_partitioning().partition_count(), 1);
    assert_eq!(
        find(&four, "InferenceExec")
            .unwrap()
            .output_partitioning()
            .partition_count(),
        4
    );
}

/// Row-sequence identity at full strength: an UNSORTED `SELECT *` (no
/// `ORDER BY` to mask a divergence), compared per row over every column but
/// `_latency_ms` — ordinals and vectors included — for every fan-out, on both
/// input shapes, at several `target_partitions`, against the first cell.
#[tokio::test]
async fn row_sequence_is_identical_across_n_on_both_input_shapes() {
    let (session, _dir) = session().await;
    for shape in [Shape::Keyed, Shape::Arrival] {
        let mut base: Option<Vec<String>> = None;
        for n in [1usize, 2, 4] {
            for tp in [1usize, 2, 8] {
                let ctx = sql_context(&session, shape, n, tp);
                let df = ctx.sql("SELECT * FROM annotated").await.unwrap();
                let batches =
                    tokio::time::timeout(std::time::Duration::from_secs(60), df.collect())
                        .await
                        .unwrap_or_else(|_| panic!("{shape:?} N={n} tp={tp}: collect wedged"))
                        .unwrap();
                let rows = rendered_rows(&batches);
                assert_eq!(rows.len(), 36, "{shape:?} N={n} tp={tp}: every row once");
                match &base {
                    None => base = Some(rows),
                    Some(base) => assert_eq!(
                        &rows, base,
                        "{shape:?} N={n} tp={tp}: the row sequence must match N=1's exactly"
                    ),
                }
            }
        }
    }
}

/// DataFusion's optimizer, on its own: it strips every hand-placed exchange,
/// merge and coalesce and re-derives them from declared requirements. Across
/// `{shape} x {query} x N x target_partitions`, what it leaves is still the
/// planned shape — the numbered input over one partition, `InferenceExec`
/// directly over it or over a hash exchange on the chunk id whose only input
/// is the numbered input — but at ITS OWN width ([`optimizer_fan_out`]), not
/// the node's. `a_session_planned_query_keeps_the_nodes_own_fan_out` is the
/// same path with the session's `InferenceFanOut` rule registered.
#[tokio::test]
async fn the_optimizer_rederives_the_planned_shape_across_the_grid() {
    let (session, _dir) = session().await;
    let mut failures: Vec<String> = Vec::new();
    let mut cells = 0usize;
    for shape in [Shape::Keyed, Shape::Arrival] {
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
            for n in [1usize, 2, 4] {
                for tp in [1usize, 2, 4, 8] {
                    cells += 1;
                    let cell = format!("{shape:?} {label} N={n} tp={tp}");
                    let ctx = sql_context(&session, shape, n, tp);
                    let plan = ctx
                        .sql(sql)
                        .await
                        .unwrap()
                        .create_physical_plan()
                        .await
                        .unwrap_or_else(|e| panic!("{cell}: {e}"));
                    let text = displayable(plan.as_ref()).indent(true).to_string();
                    if let Err(why) = check_inference_shape(&plan, n, tp) {
                        failures.push(format!("{cell}: {why}\n{text}"));
                    }
                }
            }
        }
    }
    assert_eq!(cells, 120);
    assert!(
        failures.is_empty(),
        "{} of {cells} cells lost the planned shape:\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

/// The shape `the_optimizer_rederives_the_planned_shape_across_the_grid`
/// holds every optimized plan to.
fn check_inference_shape(
    plan: &Arc<dyn ExecutionPlan>,
    n: usize,
    target_partitions: usize,
) -> Result<(), String> {
    let inference = find(plan, "InferenceExec").ok_or("no InferenceExec")?;
    let below = inference.children()[0];
    let fan_out = optimizer_fan_out(n, target_partitions);
    let numbered = if fan_out == 1 {
        below
    } else {
        let exchange = below.downcast_ref::<RepartitionExec>().ok_or_else(|| {
            format!(
                "expected an exchange below InferenceExec, found {}",
                below.name()
            )
        })?;
        match exchange.partitioning() {
            Partitioning::Hash(exprs, count)
                if *count == fan_out
                    && exprs.len() == 1
                    && exprs[0].to_string() == format!("{CHUNK_COLUMN}@4") => {}
            other => return Err(format!("expected Hash([chunk], {fan_out}), found {other}")),
        }
        below.children()[0]
    };
    if inference.output_partitioning().partition_count() != fan_out {
        return Err(format!("InferenceExec does not run {fan_out} partitions"));
    }
    if numbered.downcast_ref::<NumberedInputExec>().is_none() {
        return Err(format!(
            "expected NumberedInputExec, found {}",
            numbered.name()
        ));
    }
    let input = numbered.children()[0];
    if input.output_partitioning().partition_count() != 1 {
        return Err("the numbered input reads more than one partition".into());
    }
    Ok(())
}

/// The width `EnforceDistribution` gives the exchange below a node declaring
/// a fan-out of `n`: it honours a fan-out of one exactly (the node requires a
/// single partition), and sizes every exchange it adds at `target_partitions`
/// — so none at all when that is 1.
fn optimizer_fan_out(n: usize, target_partitions: usize) -> usize {
    if n == 1 {
        1
    } else {
        target_partitions
    }
}

/// A 36-row single-file Parquet source of `(id, abstract)`: one file scans as
/// one partition, so arrival order is the file's order on every run.
fn write_passages(dir: &std::path::Path) -> String {
    let src_dir = dir.join("passages");
    std::fs::create_dir_all(&src_dir).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("abstract", DataType::Utf8, false),
    ]));
    let ids: Vec<i64> = (0..36).map(|i| i * 7 % 36).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(ids.clone())),
            Arc::new(StringArray::from_iter_values(
                ids.iter().map(|&i| passage(i)),
            )),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(src_dir.join("part0.parquet")).unwrap();
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    format!("file://{}", src_dir.display())
}

/// The SQL `annotate` table function, planned by a real session through
/// DataFusion's optimizer at `[inference] partitions = N` x `[engine]
/// execution_threads = T`: the node runs EXACTLY `N` partitions over a hash
/// exchange of that width at every `T` — including `T = 1`, where the
/// optimizer alone would have left it serial — and every cell returns the
/// identical row sequence.
#[tokio::test(flavor = "multi_thread")]
async fn a_session_planned_query_keeps_the_nodes_own_fan_out() {
    let mut base: Option<Vec<String>> = None;
    for n in [1usize, 2, 4] {
        for t in [1usize, 2, 8] {
            let cell = format!("partitions={n} execution_threads={t}");
            let dir = TempDir::new().unwrap();
            let mut cfg = common::test_config(dir.path());
            cfg.inference.partitions = n;
            cfg.inference.batch_size = BATCH_SIZE;
            cfg.engine.execution_threads =
                std::num::NonZeroUsize::new(t).expect("a positive thread count");
            let session = InferenceSession::open(cfg).await.unwrap();
            session
                .add_source(
                    "passages",
                    jammi_db::source::SourceType::File,
                    jammi_db::source::SourceConnection {
                        url: Some(write_passages(dir.path())),
                        format: Some(jammi_db::source::FileFormat::Parquet),
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            let sql = format!(
                "SELECT * FROM annotate('{}', 'text_embedding', \
                 'passages.public.passages', 'id', 'abstract')",
                tiny_bert_model()
            );
            let df = session.context().sql(&sql).await.unwrap();
            let plan = df.clone().create_physical_plan().await.unwrap();
            let text = displayable(plan.as_ref()).indent(true).to_string();
            let inference = find(&plan, "InferenceExec").expect("an InferenceExec");
            assert_eq!(
                inference.output_partitioning().partition_count(),
                n,
                "{cell}:\n{text}"
            );
            let below = inference.children()[0];
            if n > 1 {
                let exchange = below
                    .downcast_ref::<RepartitionExec>()
                    .unwrap_or_else(|| panic!("{cell}: no exchange:\n{text}"));
                assert!(
                    matches!(exchange.partitioning(), Partitioning::Hash(_, width) if *width == n),
                    "{cell}:\n{text}"
                );
                assert_eq!(below.children()[0].name(), "NumberedInputExec", "{text}");
            } else {
                assert_eq!(below.name(), "NumberedInputExec", "{cell}:\n{text}");
            }

            let rows = rendered_rows(&df.collect().await.unwrap());
            assert_eq!(rows.len(), 36, "{cell}");
            match &base {
                None => base = Some(rows),
                Some(base) => assert_eq!(&rows, base, "{cell}: the row sequence must not move"),
            }
        }
    }
}

/// A `LIMIT` satisfied over a fan-out of four completes: the merge stops
/// polling, the exchange and every partition's runner are dropped with it,
/// and nothing is left waiting on a partition nobody reads.
#[tokio::test]
async fn limit_over_n4_completes_without_wedging() {
    let (session, _dir) = session().await;
    let ctx = sql_context(&session, Shape::Arrival, 4, 8);
    let df = ctx.sql("SELECT * FROM annotated LIMIT 1").await.unwrap();
    let batches = tokio::time::timeout(std::time::Duration::from_secs(30), df.collect())
        .await
        .expect("LIMIT 1 over N=4 must complete, not wedge the driver")
        .unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 1, "LIMIT 1 must return exactly one row");
}

/// `InferenceExec` cannot be bound to an input that would let a chunk's rows
/// be forwarded apart: one with no `_ordinal`, or one of several partitions
/// not hashed on the chunk id.
#[tokio::test]
async fn an_unnumbered_or_unhashed_input_is_refused() {
    let (session, _dir) = session().await;
    let unnumbered = Shape::Arrival.source();
    let err = InferenceExec::bind(
        Arc::new(
            datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec::new(unnumbered),
        ),
        spec(1),
        session.inference_runtime(),
    )
    .expect_err("an input without _ordinal must refuse");
    assert!(err.to_string().contains(ORDINAL_COLUMN), "{err}");

    let numbered: Arc<dyn ExecutionPlan> = Arc::new(
        NumberedInputExec::try_new(
            Arc::new(
                datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec::new(
                    Shape::Arrival.source(),
                ),
            ),
            RowOrder::Arrival,
            spec(1),
            session.inference_runtime(),
        )
        .unwrap(),
    );
    let round_robin: Arc<dyn ExecutionPlan> =
        Arc::new(RepartitionExec::try_new(numbered, Partitioning::RoundRobinBatch(4)).unwrap());
    let err = InferenceExec::bind(round_robin, spec(4), session.inference_runtime())
        .expect_err("a round-robin input must refuse");
    assert!(err.to_string().contains("hash-partitioned"), "{err}");
}

/// `annotate_plan` over a MULTI-PARTITION input sees every row at every
/// fan-out: the plan has one output partition, `.execute(0, ..)` alone
/// returns every row of all four source partitions, and `_ordinal` runs
/// contiguously from 0 in the order the rows come out.
#[tokio::test]
async fn annotate_plan_sees_every_row_of_a_multi_partition_input() {
    use arrow::array::UInt64Array;

    for partitions in [1usize, 4] {
        let dir = TempDir::new().unwrap();
        let mut cfg = common::test_config(dir.path());
        cfg.inference.partitions = partitions;
        cfg.inference.batch_size = 2;
        let session = Arc::new(InferenceSession::new(cfg).await.unwrap());
        let input = Shape::Arrival.source();
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
        assert_eq!(plan.output_partitioning().partition_count(), 1);

        let stream = plan.execute(0, session.context().task_ctx()).unwrap();
        let batches = datafusion::physical_plan::common::collect(stream)
            .await
            .unwrap();
        let ordinals: Vec<u64> = batches
            .iter()
            .flat_map(|b| {
                b.column_by_name(ORDINAL_COLUMN)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        assert_eq!(
            ordinals,
            (0..36).collect::<Vec<u64>>(),
            "partitions={partitions}: every row once, in ordinal order"
        );
    }
}

/// The written artifact is byte-identical at every fan-out. A four-file
/// source (a multi-partition scan) of varied-length passages is embedded at
/// `[inference] partitions` 1, 2 and 4: the embedding table's artifact digest
/// — the SHA-256 of the Parquet bytes — is one value, its rows are in key
/// order, its ANN segments (64 rows each: 240 rows cut four ways) hold the
/// same row counts and answer every query bit-for-bit alike, and `infer`
/// returns one row sequence with bit-identical vectors.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_written_bytes_are_identical_at_every_fan_out() {
    use arrow::array::{FixedSizeListArray, Float32Array};
    use jammi_db::index::QuerySource;

    let dir = TempDir::new().unwrap();
    let src_dir = dir.path().join("corpus");
    std::fs::create_dir_all(&src_dir).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("text", DataType::Utf8, false),
    ]));
    for f in 0..4i64 {
        let ids: Vec<i64> = (0..60).map(|i| (f * 60 + i) * 13 % 240).collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(ids.clone())),
                Arc::new(StringArray::from_iter_values(
                    ids.iter().map(|&i| passage(i)),
                )),
            ],
        )
        .unwrap();
        let file = std::fs::File::create(src_dir.join(format!("part{f}.parquet"))).unwrap();
        let mut writer =
            parquet::arrow::ArrowWriter::try_new(file, Arc::clone(&schema), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
    let url = format!("file://{}", src_dir.display());
    let model = tiny_bert_model();

    let mut digests = Vec::new();
    let mut views: Vec<Vec<String>> = Vec::new();
    let mut layouts: Vec<Vec<i64>> = Vec::new();
    let mut hits: Vec<Vec<(String, u32)>> = Vec::new();
    for partitions in [1usize, 2, 4] {
        let artifact_dir = dir.path().join(format!("n{partitions}"));
        std::fs::create_dir_all(&artifact_dir).unwrap();
        let mut cfg = common::test_config(&artifact_dir);
        cfg.inference.partitions = partitions;
        cfg.embedding.index_segment_rows = std::num::NonZeroUsize::new(64).unwrap();
        cfg.engine.execution_threads =
            std::num::NonZeroUsize::new(4).expect("a positive thread count");
        let session = Arc::new(InferenceSession::new(cfg).await.unwrap());
        session
            .add_source(
                "corpus",
                jammi_db::source::SourceType::File,
                jammi_db::source::SourceConnection {
                    url: Some(url.clone()),
                    format: Some(jammi_db::source::FileFormat::Parquet),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let (record, _) = session
            .generate_text_embeddings(
                "corpus",
                &model,
                &["text".to_string()],
                "id",
                jammi_db::store::CachePolicy::Bypass,
                None,
            )
            .await
            .unwrap();
        assert_eq!(record.row_count, 240, "partitions={partitions}");
        let keys: Vec<String> = session
            .sql(&format!(
                "SELECT _row_id FROM \"jammi.{}\"",
                record.table_name
            ))
            .await
            .unwrap()
            .iter()
            .flat_map(|b| {
                arrow::compute::cast(b.column(0), &DataType::Utf8)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|k| k.unwrap().to_string())
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut sorted = keys.clone();
        sorted.sort_by_key(|k| k.parse::<i64>().unwrap());
        assert_eq!(
            keys, sorted,
            "partitions={partitions}: the table is in key order"
        );
        let segments = session
            .catalog()
            .list_index_segments(&record.table_name)
            .await
            .unwrap();
        layouts.push(segments.iter().map(|s| s.row_count as i64).collect());
        let store = session.result_store();
        let index = store
            .resolve_search_mode_local(&record)
            .await
            .unwrap()
            .expect("the table has an index");
        let pin = common::pin(&session, record.clone()).await;
        let mut answers = Vec::new();
        for key in ["0", "13", "117", "239"] {
            let vector = store
                .read_vector_by_key(session.context(), &pin, key)
                .await
                .unwrap();
            let width = vector.len();
            let query =
                jammi_db::index::validate_query(vector, width, QuerySource::Caller).unwrap();
            answers.extend(
                index
                    .search_final(&query, 10, 1)
                    .unwrap()
                    .into_iter()
                    .map(|(id, d)| (id, d.to_bits())),
            );
        }
        hits.push(answers);
        let manifest_url = jammi_db::storage::StorageUrl::parse(&record.parquet_path).unwrap();
        digests.push(
            session
                .result_store()
                .read_materialization_manifest(&manifest_url)
                .await
                .unwrap()
                .expect("manifest sidecar present")
                .artifact
                .0,
        );

        let (batches, _) = session
            .infer(
                "corpus",
                &ModelSource::parse(&model),
                ModelTask::TextEmbedding,
                &["text".to_string()],
                "id",
                jammi_db::store::CachePolicy::Bypass,
            )
            .await
            .unwrap();
        let view: Vec<String> = batches
            .iter()
            .flat_map(|b| {
                let ids = b
                    .column_by_name("_row_id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .clone();
                let vectors = b
                    .column_by_name("vector")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<FixedSizeListArray>()
                    .unwrap()
                    .clone();
                (0..b.num_rows())
                    .map(|i| {
                        let v = vectors.value(i);
                        let v = v.as_any().downcast_ref::<Float32Array>().unwrap();
                        let bits: Vec<u32> = v.values().iter().map(|x| x.to_bits()).collect();
                        format!("{}|{bits:?}", ids.value(i))
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(view.len(), 240, "partitions={partitions}");
        views.push(view);
    }
    assert!(
        digests.windows(2).all(|w| w[0] == w[1]),
        "the embedding artifact digest must not depend on the fan-out: {digests:?}"
    );
    assert!(
        views.windows(2).all(|w| w[0] == w[1]),
        "infer's row sequence and vector bits must not depend on the fan-out"
    );
    assert_eq!(
        layouts[0],
        vec![64, 64, 64, 48],
        "the segments are cut at the budget"
    );
    assert!(
        layouts.windows(2).all(|w| w[0] == w[1]),
        "the segment layout must not depend on the fan-out: {layouts:?}"
    );
    assert!(
        hits.windows(2).all(|w| w[0] == w[1]),
        "every search must answer alike at every fan-out"
    );
}

/// Forward admission belongs to the device, not to a plan node. Two plans,
/// each fanned out four ways, run side by side against ONE model cache whose
/// device admits one forward at a time: across all eight partitions of the
/// two `InferenceExec` instances, no two forwards are ever in flight together.
/// The same two plans against a device that admits many DO overlap, so the
/// oracle can see what it claims is absent.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn one_device_admits_forwards_across_two_inference_execs() {
    use jammi_ai::concurrency::GpuScheduler;
    use jammi_ai::model::backend::DeviceConfig;
    use jammi_ai::model::cache::ModelCache;
    use jammi_ai::model::resolver::ModelResolver;
    use jammi_datafusion::inference::runner::test_hooks::{
        peak_concurrent_forwards_for, reset_forward_concurrency_for,
    };
    use jammi_datafusion::InferenceRuntime;

    async fn peak_over_two_plans(device: GpuScheduler, source_id: &str) -> u64 {
        let dir = TempDir::new().unwrap();
        let catalog = Arc::new(jammi_db::catalog::Catalog::open(dir.path()).await.unwrap());
        let resolver = ModelResolver::new(
            catalog,
            common::test_artifact_store(),
            common::test_hub_source(),
        )
        .unwrap();
        let device_config = DeviceConfig {
            gpu_device: -1,
            devices: vec![-1],
            memory_fraction: 1.0,
            require_gpu: false,
            compute_precision: jammi_numerics::ComputePrecision::F32,
        };
        let runtime = InferenceRuntime {
            model: Arc::new(ModelCache::new(resolver, device_config, Arc::new(device))),
            observer: None,
        };
        reset_forward_concurrency_for(source_id);

        let plan = || {
            let ids: Vec<i64> = (0..120).collect();
            let input =
                MemorySourceConfig::try_new_exec(&[vec![rows(&ids)]], in_schema(), None).unwrap();
            plan_inference(
                input,
                RowOrder::Arrival,
                InferenceSpec {
                    source_id: source_id.to_string(),
                    ..spec(4)
                },
                runtime.clone(),
            )
            .unwrap()
        };
        let ctx = SessionContext::new();
        let (first, second) = tokio::join!(
            tokio::spawn(datafusion::physical_plan::collect(plan(), ctx.task_ctx())),
            tokio::spawn(datafusion::physical_plan::collect(plan(), ctx.task_ctx())),
        );
        for batches in [first, second] {
            let rows: usize = batches.unwrap().unwrap().iter().map(|b| b.num_rows()).sum();
            assert_eq!(rows, 120, "{source_id}: every row of each plan");
        }
        peak_concurrent_forwards_for(source_id)
    }

    assert_eq!(
        peak_over_two_plans(GpuScheduler::new(1 << 40, 0.0), "one-forward-device").await,
        1,
        "a device that admits one forward never runs two, whichever plan they belong to"
    );
    assert!(
        peak_over_two_plans(GpuScheduler::new_unlimited(), "many-forward-device").await > 1,
        "the same plans on a device that admits many must overlap, or the oracle sees nothing"
    );
}

/// A Struct-typed key through the real `annotate()` path is a typed
/// refusal naming the key column, its type, and "cannot be cast to Utf8",
/// end to end through `InferenceSession::annotate_plan`.
#[tokio::test]
async fn struct_key_through_annotate_is_a_typed_refusal_naming_the_key() {
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

/// The default `InferenceConfig::partitions` is `1`: a deployment that never
/// sets it runs the plan with no exchange and no merge.
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

/// Under `partitions = 2`, the sink's batch count/the persisted
/// `checkpoint` column count the MERGED batches `EmbeddingPipeline::run`
/// actually wrote — never a per-partition count — asserted against an
/// INDEPENDENTLY collected count of the SAME construction's merged output.
/// `checkpoint_interval = 1` makes every `write_batch` call persist, so the
/// table's FINAL checkpoint is exactly the number of `write_batch` calls
/// `EmbeddingPipeline::run` made: one per collected merged batch.
///
/// Two settings make the fixture non-trivial. `inference.batch_size = 1`
/// makes every row its own forward chunk, so the three rows genuinely divide
/// between the two partitions. `engine.batch_size = 1` keeps
/// `SortPreservingMergeExec` from re-batching its output back up to one
/// batch, which would make the equality hold at a trivial `1 == 1`. The
/// assertions pin `merged_batches.len() > 1` and the equality, not a literal
/// count.
#[tokio::test]
async fn checkpoint_counts_the_merged_batches_under_partitions_two() {
    let dir = TempDir::new().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("text", DataType::Utf8, false),
    ]));
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
    cfg.inference.batch_size = 1;
    cfg.embedding.checkpoint_interval = 1;
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

/// A typed refusal raised below the exchange — the numbered input's
/// `InvalidKey`, over a NULL `id` — reaches `generate_text_embeddings`'s
/// caller classified IDENTICALLY, same variant and same fields, at every
/// fan-out, and the model is never invoked. At a fan-out of one the error
/// travels owned; above one the stock exchange hands every partition a
/// shared copy, which the classifier sees through.
#[tokio::test]
async fn a_null_key_classifies_as_invalid_key_at_every_fan_out_before_any_forward() {
    use jammi_datafusion::inference::runner::test_hooks::{
        forward_calls_for, reset_forward_calls_for,
    };

    for partitions in [1usize, 2, 4] {
        let dir = TempDir::new().unwrap();
        let url = common::write_null_key_source(dir.path());

        let mut cfg = common::test_config(dir.path());
        cfg.inference.partitions = partitions;
        let session = Arc::new(InferenceSession::new(cfg).await.unwrap());
        let source = format!("null_key_src_{partitions}");
        reset_forward_calls_for(&source);
        session
            .add_source(
                &source,
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
                &source,
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
        assert_eq!(
            forward_calls_for(&source),
            0,
            "partitions={partitions}: the refusal precedes every forward"
        );
    }
}
