//! End-to-end tests for the `asof_join` verb: register two relations as
//! Parquet sources, run the verb through the session, and read the materialised
//! result table back via SQL.
//!
//! These exercise the whole lifecycle the merge tests do not — relation
//! resolution, the `SortExec` the verb inserts, the operator, and the
//! materialization funnel (the result table is a `ready` row carrying a manifest
//! with a `ProducingDescriptor::AsofJoin` and an anchor for both inputs). The
//! backward-inclusive correctness exit-criterion (#1) runs at the 1000-row × 50-
//! group scale the spec pins.

use std::sync::Arc;

use arrow::array::{Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use jammi_ai::pipeline::asof::{AsofJoinSpecBuilder, AsofKey, Boundary, MatchDirection, TieBreak};
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::result_repo::ResultTableKind;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

use crate::common;

/// Write `batch` to a Parquet file under `dir` and register it as a File source.
async fn register_parquet(
    session: &InferenceSession,
    dir: &TempDir,
    source_id: &str,
    batch: &RecordBatch,
) {
    let path = dir.path().join(format!("{source_id}.parquet"));
    let file = std::fs::File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
    session
        .add_source(
            source_id,
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", path.display())),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
}

fn spine_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("sym", DataType::Utf8, false),
        Field::new("exec_time", DataType::Int64, false),
        Field::new("trade_id", DataType::Int64, false),
    ]))
}

fn facts_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("sym", DataType::Utf8, false),
        Field::new("quote_time", DataType::Int64, false),
        Field::new("bid", DataType::Int64, false),
    ]))
}

async fn fresh_session() -> (Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    (session, dir)
}

/// The default backward-inclusive spec over `sym` keyed on the two time columns,
/// projecting the single `bid` fact column.
fn default_spec() -> jammi_ai::pipeline::asof::AsofJoinSpec {
    AsofJoinSpecBuilder::new(
        AsofKey {
            by: vec!["sym".into()],
            time: "exec_time".into(),
        },
        AsofKey {
            by: vec!["sym".into()],
            time: "quote_time".into(),
        },
    )
    .direction(MatchDirection::Backward)
    .boundary(Boundary::Inclusive)
    .tie_break(TieBreak::Error)
    .project(vec!["bid".into()])
    .build()
}

/// Read the materialised table's `(trade_id, bid)` pairs ordered by trade id —
/// `bid` is null where the spine row was unmatched.
async fn read_pairs(session: &InferenceSession, table: &str) -> Vec<(i64, Option<i64>)> {
    let batches = session
        .sql(&format!(
            "SELECT trade_id, bid FROM \"jammi.{table}\" ORDER BY trade_id"
        ))
        .await
        .unwrap();
    let batch = arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap();
    let tid = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let bid = batch
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    (0..batch.num_rows())
        .map(|i| (tid.value(i), (!bid.is_null(i)).then(|| bid.value(i))))
        .collect()
}

#[tokio::test]
async fn asof_join_writes_a_ready_result_table_with_manifest() {
    let (session, dir) = fresh_session().await;
    // group `a`: quotes at 10/20/30; trade at 25 → bid of the 20 quote.
    let spine = RecordBatch::try_new(
        spine_schema(),
        vec![
            Arc::new(StringArray::from(vec!["a", "a"])),
            Arc::new(Int64Array::from(vec![25, 5])),
            Arc::new(Int64Array::from(vec![1, 2])),
        ],
    )
    .unwrap();
    let facts = RecordBatch::try_new(
        facts_schema(),
        vec![
            Arc::new(StringArray::from(vec!["a", "a", "a"])),
            Arc::new(Int64Array::from(vec![10, 20, 30])),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
        ],
    )
    .unwrap();
    register_parquet(&session, &dir, "trades", &spine).await;
    register_parquet(&session, &dir, "quotes", &facts).await;

    let spec = default_spec();
    let record = session.asof_join("trades", "quotes", &spec).await.unwrap();

    assert_eq!(
        record.status, "ready",
        "the result table is published ready"
    );
    assert_eq!(record.kind, ResultTableKind::AsofJoin);
    assert_eq!(record.row_count, 2, "the spine is fully preserved");

    // trade 1 (exec 25) → bid 200; trade 2 (exec 5, before any quote) → null.
    let pairs = read_pairs(&session, &record.table_name).await;
    assert_eq!(pairs, vec![(1, Some(200)), (2, None)]);

    // The materialization manifest names the as-of producer and anchors BOTH
    // inputs — the verifiable identity every result table carries.
    let parquet_url = jammi_db::storage::StorageUrl::parse(&record.parquet_path).unwrap();
    let manifest = session
        .result_store()
        .read_materialization_manifest(&parquet_url)
        .await
        .unwrap()
        .expect("an asof_join result table carries a materialization manifest");
    assert_eq!(
        manifest.input_anchors.len(),
        2,
        "both the spine and facts relations are anchored"
    );
    let anchored: Vec<&str> = manifest
        .input_anchors
        .iter()
        .map(|a| a.source.as_str())
        .collect();
    assert!(anchored.contains(&"trades") && anchored.contains(&"quotes"));
}

#[tokio::test]
async fn exit_crit_1_backward_inclusive_correctness_at_scale() {
    // 1000 spine rows over 50 groups; each group has many facts. Assert every
    // matched fact is the maximal quote_time ≤ exec_time in-group, and every
    // spine row is preserved.
    let (session, dir) = fresh_session().await;
    let groups = 50i64;
    let spine_rows = 1000usize;

    // Facts: each group g has quotes at times 0,10,20,...,990 with bid = time.
    let mut f_sym = Vec::new();
    let mut f_qt = Vec::new();
    let mut f_bid = Vec::new();
    for g in 0..groups {
        for step in 0..100i64 {
            f_sym.push(format!("g{g}"));
            f_qt.push(step * 10);
            f_bid.push(step * 10);
        }
    }
    let facts = RecordBatch::try_new(
        facts_schema(),
        vec![
            Arc::new(StringArray::from(f_sym)),
            Arc::new(Int64Array::from(f_qt)),
            Arc::new(Int64Array::from(f_bid)),
        ],
    )
    .unwrap();

    // Spine: 1000 trades spread over the 50 groups at arbitrary exec_times.
    let mut s_sym = Vec::new();
    let mut s_exec = Vec::new();
    let mut s_tid = Vec::new();
    for i in 0..spine_rows {
        let g = (i as i64) % groups;
        // exec times that land between, on, and beyond fact instants.
        let exec = ((i as i64) * 7) % 1010;
        s_sym.push(format!("g{g}"));
        s_exec.push(exec);
        s_tid.push(i as i64);
    }
    let spine = RecordBatch::try_new(
        spine_schema(),
        vec![
            Arc::new(StringArray::from(s_sym.clone())),
            Arc::new(Int64Array::from(s_exec.clone())),
            Arc::new(Int64Array::from(s_tid.clone())),
        ],
    )
    .unwrap();

    register_parquet(&session, &dir, "trades", &spine).await;
    register_parquet(&session, &dir, "quotes", &facts).await;

    let record = session
        .asof_join("trades", "quotes", &default_spec())
        .await
        .unwrap();
    assert_eq!(record.row_count, spine_rows, "every spine row preserved");

    let pairs = read_pairs(&session, &record.table_name).await;
    assert_eq!(pairs.len(), spine_rows);
    // The oracle: the maximal fact instant ≤ exec, which (facts at multiples of
    // 10 from 0..=990) is `min(floor(exec/10)*10, 990)`, or null when exec < 0
    // (never here, so every row in 0..1010 has a match — exec ≥ 0 always finds
    // the 0 fact at least).
    for (tid, bid) in pairs {
        let exec = s_exec[tid as usize];
        let expected = (exec / 10 * 10).min(990);
        assert_eq!(
            bid,
            Some(expected),
            "trade {tid} (exec {exec}) must match the maximal quote_time ≤ exec"
        );
    }
}

#[tokio::test]
async fn ambiguous_duplicate_facts_fail_loud_through_the_verb() {
    let (session, dir) = fresh_session().await;
    let spine = RecordBatch::try_new(
        spine_schema(),
        vec![
            Arc::new(StringArray::from(vec!["a"])),
            Arc::new(Int64Array::from(vec![25])),
            Arc::new(Int64Array::from(vec![1])),
        ],
    )
    .unwrap();
    // Two quotes at the same instant 20, no tie-break column.
    let facts = RecordBatch::try_new(
        facts_schema(),
        vec![
            Arc::new(StringArray::from(vec!["a", "a"])),
            Arc::new(Int64Array::from(vec![20, 20])),
            Arc::new(Int64Array::from(vec![200, 201])),
        ],
    )
    .unwrap();
    register_parquet(&session, &dir, "trades", &spine).await;
    register_parquet(&session, &dir, "quotes", &facts).await;

    let err = session
        .asof_join("trades", "quotes", &default_spec())
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("ambiguous"),
        "a duplicate at the matched instant with TieBreak::Error must fail loud; got: {msg}"
    );
}
