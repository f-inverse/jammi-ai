//! #500 U2c, M3 — the per-rank streaming training-set loader's oracles
//! (P3–P7; P1's fixture-side plan-shape check; P6.i parity).
//!
//! Every oracle here builds through `common::multi_row_group_pairs` (the
//! ONE 70,000-row multi-row-group fixture builder, lifted from
//! `training_set.rs`) or `common::padded_regression_fixture` (P3's
//! residency oracle, which needs a table whose EAGER collected size
//! genuinely exceeds the 64 MiB `[engine] memory_limit` floor).
//!
//! Every test below that builds a 70,000-row fixture (or a padded one) is
//! `#[serial(training_set_stream)]` — the same idiom `acceleration_report.rs`
//! uses: several of these tests each carry a real wall-clock deadline (P4's
//! 60s liveness timeout), and running them concurrently under `cargo test`'s
//! default parallelism let CPU contention alone blow that budget on a loaded
//! CI runner (observed: `p4_liveness_over_every_held_chunk_at_every_
//! accepted_prefetch` timed out under the FULL `cargo test -p jammi-ai` run
//! but passed cleanly in isolation) — a false DEADLOCK finding from resource
//! contention, not from the mechanism P4 pins. Serializing this file's own
//! heavy tests against each other removes that confound without weakening
//! any assertion.

use std::sync::Arc;
use std::time::Duration;

use jammi_ai::fine_tune::data::TextChunk;
use jammi_ai::fine_tune::training_set::read_back_sql;
use jammi_ai::fine_tune::{partition, stream};
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::error::JammiError;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use serial_test::serial;
use tempfile::TempDir;

use crate::common;

/// A `(anchor, positive)` row reader over collected batches — a local copy
/// of `training_set.rs`'s own `rows_of` (module-private there), not shared,
/// so this file does not widen that module's visibility for one helper.
fn rows_of(batches: &[arrow::array::RecordBatch]) -> Vec<(String, String)> {
    use arrow::array::AsArray;
    fn string_column(batch: &arrow::array::RecordBatch, name: &str) -> Vec<String> {
        let column = batch.column_by_name(name).expect("column present");
        match column.data_type() {
            arrow::datatypes::DataType::Utf8View => column
                .as_string_view()
                .iter()
                .map(|v| v.unwrap_or_default().to_string())
                .collect(),
            arrow::datatypes::DataType::LargeUtf8 => column
                .as_string::<i64>()
                .iter()
                .map(|v| v.unwrap_or_default().to_string())
                .collect(),
            _ => column
                .as_string::<i32>()
                .iter()
                .map(|v| v.unwrap_or_default().to_string())
                .collect(),
        }
    }
    let mut out = Vec::new();
    for batch in batches {
        let anchor = string_column(batch, "anchor");
        let positive = string_column(batch, "positive");
        for i in 0..batch.num_rows() {
            out.push((anchor[i].clone(), positive[i].clone()));
        }
    }
    out
}

/// Drain a [`stream::TrainingSetStream`] to its terminal (empty) chunk on the
/// blocking pool — the shape production and every oracle here uses (B3: the
/// consumer must be on `spawn_blocking` from a `multi_thread` runtime).
/// Returns every NON-empty chunk's `(step, TextChunk)`, and the total row
/// count served.
async fn drain_ok(
    mut ts: stream::TrainingSetStream,
) -> Result<(Vec<(usize, TextChunk)>, usize), JammiError> {
    tokio::task::spawn_blocking(move || {
        let mut out = Vec::new();
        let mut served = 0usize;
        while let Some(chunk) = ts.next_chunk()? {
            let step = chunk.step();
            let n = chunk.chunk().row_count();
            served += n;
            let empty = n == 0;
            if !empty {
                out.push((step, chunk.into_chunk()));
            }
            if empty {
                break;
            }
        }
        Ok((out, served))
    })
    .await
    .expect("spawn_blocking join")
}

/// P1 (fixture side): the stream's OWN one-`target_partitions` derivation
/// (replicated here exactly as `TrainingSetStream::open` builds it) plans
/// the read-back with NO `SortExec` and NO `SortPreservingMergeExec` — the
/// merge term P3's inequality claims is zero by construction.
#[tokio::test(flavor = "multi_thread")]
#[serial(training_set_stream)]
async fn p1_the_loader_derived_state_plans_with_no_sort_and_no_merge() {
    let dir = TempDir::new().unwrap();
    let mut config = common::test_config(dir.path());
    config.engine.execution_threads = 4;
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    let fixture = common::multi_row_group_pairs(&session, dir.path(), true).await;

    let base_state = session.context().state();
    let one_partition_config = base_state.config().clone().with_target_partitions(1);
    let derived_state =
        datafusion::execution::session_state::SessionStateBuilder::new_from_existing(base_state)
            .with_config(one_partition_config)
            .build();
    let derived_ctx = datafusion::prelude::SessionContext::new_with_state(derived_state);

    let query = read_back_sql(&fixture.table, &fixture.columns);
    let batches = derived_ctx
        .sql(&format!("EXPLAIN {query}"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let plan = arrow::util::pretty::pretty_format_batches(&batches)
        .unwrap()
        .to_string();
    assert!(
        !plan.contains("SortExec"),
        "the loader-derived one-partition plan must carry no SortExec:\n{plan}"
    );
    assert!(
        !plan.contains("SortPreservingMergeExec"),
        "the loader-derived one-partition plan must carry no merge:\n{plan}"
    );
}

/// P6.i: the concatenation of a W=1 stream's chunks equals `read_back_sql`'s
/// collected rows in order, at `target_partitions` in `{1, 4}` — the
/// session's OWN configured partition count, to prove the stream's internal
/// `target_partitions = 1` derivation is robust regardless of it.
#[tokio::test(flavor = "multi_thread")]
#[serial(training_set_stream)]
async fn p6i_w1_stream_concatenation_matches_read_back_sql_at_various_partition_counts() {
    for execution_threads in [1usize, 4] {
        let dir = TempDir::new().unwrap();
        let mut config = common::test_config(dir.path());
        config.engine.execution_threads = execution_threads;
        let session = Arc::new(InferenceSession::new(config).await.unwrap());
        let fixture =
            common::multi_row_group_pairs(&session, dir.path(), execution_threads > 1).await;
        let table = fixture.table.clone();
        let columns = fixture.columns.clone();
        let total = fixture.written.len();

        let expected = rows_of(&session.sql(&read_back_sql(&table, &columns)).await.unwrap());

        let spec =
            partition::PartitionSpec::single_rank(13, partition::PartitionRule::BlockByGlobalBatch);
        let cfg = stream::StreamConfig::new(2).unwrap();
        let window = stream::RowWindow::new(0, total);
        let ts = stream::TrainingSetStream::open(
            &session,
            &table,
            &columns,
            ModelTask::TextEmbedding,
            window,
            stream::Slice::PerRank(spec),
            cfg,
        )
        .await
        .unwrap_or_else(|e| panic!("execution_threads={execution_threads}: open failed: {e}"));

        let (chunks, served) = drain_ok(ts)
            .await
            .unwrap_or_else(|e| panic!("execution_threads={execution_threads}: {e}"));
        assert_eq!(served, total);

        let mut anchors = Vec::new();
        let mut positives = Vec::new();
        for (_, chunk) in chunks {
            if let TextChunk::Pairs {
                anchors: a,
                positives: p,
            } = chunk
            {
                anchors.extend(a);
                positives.extend(p);
            } else {
                panic!("execution_threads={execution_threads}: expected a Pairs chunk");
            }
        }
        let observed: Vec<(String, String)> = anchors.into_iter().zip(positives).collect();
        assert_eq!(
            observed, expected,
            "execution_threads={execution_threads}: streamed concatenation must equal \
             read_back_sql's rows in order"
        );
    }
}

/// P4: for every accepted `prefetch` (the whole domain, `>= 1`), with `B =
/// 13` (does not divide 65,536, the writer's row-group size — the
/// regression pin for the excised arm's `prefetch = 2` deadlock), the stream
/// completes and serves exactly `window.len()` rows, under a wall-clock
/// timeout (120s, not the contract's literal 60s: a genuine deadlock hangs
/// FOREVER, so either bound catches it identically; the wider margin absorbs
/// CPU contention from the rest of a `cargo test -p jammi-ai` run without
/// weakening the property — `#[serial(training_set_stream)]` above already
/// removes contention from this file's OWN other heavy tests) — every
/// consuming test is `multi_thread` with the consumer on `spawn_blocking`
/// (B3).
#[tokio::test(flavor = "multi_thread")]
#[serial(training_set_stream)]
async fn p4_liveness_over_every_held_chunk_at_every_accepted_prefetch() {
    let dir = TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    let fixture = common::multi_row_group_pairs(&session, dir.path(), false).await;
    let table = fixture.table.clone();
    let columns = fixture.columns.clone();
    let total = fixture.written.len();

    for prefetch in [1usize, 2, 3, 4] {
        let cfg = stream::StreamConfig::new(prefetch).unwrap();
        let spec =
            partition::PartitionSpec::single_rank(13, partition::PartitionRule::BlockByGlobalBatch);
        let window = stream::RowWindow::new(0, total);
        let ts = stream::TrainingSetStream::open(
            &session,
            &table,
            &columns,
            ModelTask::TextEmbedding,
            window,
            stream::Slice::PerRank(spec),
            cfg,
        )
        .await
        .unwrap_or_else(|e| panic!("prefetch={prefetch}: open failed: {e}"));

        let drain = tokio::task::spawn_blocking(move || {
            let mut ts = ts;
            let mut served = 0usize;
            while let Some(chunk) = ts.next_chunk()? {
                let n = chunk.chunk().row_count();
                served += n;
                let empty = n == 0;
                // Hold the chunk across the ask for the next one — the
                // deadlock this pins (`tokio::sync::mpsc` async `send`,
                // never a sync channel a full buffer would park the runtime
                // on).
                std::thread::sleep(Duration::from_millis(5));
                drop(chunk);
                if empty {
                    break;
                }
            }
            Ok::<usize, JammiError>(served)
        });
        let served = tokio::time::timeout(Duration::from_secs(120), drain)
            .await
            .unwrap_or_else(|_| panic!("prefetch={prefetch}: deadlocked past the 120s timeout"))
            .expect("spawn_blocking join")
            .unwrap_or_else(|e| panic!("prefetch={prefetch}: {e}"));
        assert_eq!(
            served, total,
            "prefetch={prefetch}: must serve exactly window.len() rows"
        );
    }
}

/// P5: for W = 2, rank 0's and rank 1's streams over the SAME table are
/// independent objects; for every step, rank r's chunk equals the eager
/// `text_chunk_for_rank(spec_r, t)` exactly; the two row sets are disjoint
/// and their union is the window.
#[tokio::test(flavor = "multi_thread")]
#[serial(training_set_stream)]
async fn p5_two_rank_world_slices_match_eager_text_chunk_for_rank_exactly() {
    let dir = TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    let fixture = common::multi_row_group_pairs(&session, dir.path(), false).await;
    let table = fixture.table.clone();
    let columns = fixture.columns.clone();
    let total = fixture.written.len();

    let eager_rows = rows_of(&session.sql(&read_back_sql(&table, &columns)).await.unwrap());
    let eager_loader =
        jammi_ai::fine_tune::data::TrainingDataLoader::from_pairs(eager_rows.clone());

    let batch = 13usize;
    let world = 2usize;
    let window = stream::RowWindow::new(0, total);

    let mut per_rank_chunks: Vec<Vec<(usize, TextChunk)>> = Vec::new();
    for rank in 0..world {
        let spec = partition::PartitionSpec::for_rank(
            rank,
            world,
            batch,
            partition::PartitionRule::BlockByGlobalBatch,
        )
        .unwrap();
        let cfg = stream::StreamConfig::new(2).unwrap();
        let ts = stream::TrainingSetStream::open(
            &session,
            &table,
            &columns,
            ModelTask::TextEmbedding,
            window,
            stream::Slice::PerRank(spec),
            cfg,
        )
        .await
        .unwrap_or_else(|e| panic!("rank={rank}: open failed: {e}"));
        let (chunks, _served) = drain_ok(ts)
            .await
            .unwrap_or_else(|e| panic!("rank={rank}: {e}"));
        per_rank_chunks.push(chunks);
    }

    let mut seen_pairs: Vec<(String, String)> = Vec::new();
    for (rank, chunks) in per_rank_chunks.iter().enumerate() {
        let spec = partition::PartitionSpec::for_rank(
            rank,
            world,
            batch,
            partition::PartitionRule::BlockByGlobalBatch,
        )
        .unwrap();
        for (step, chunk) in chunks {
            let expected = eager_loader.text_chunk_for_rank(&spec, *step).unwrap();
            assert_eq!(
                *chunk, expected,
                "rank {rank} step {step}: the streamed chunk must equal the eager \
                 text_chunk_for_rank exactly"
            );
            if let TextChunk::Pairs { anchors, positives } = chunk {
                for (a, p) in anchors.iter().zip(positives.iter()) {
                    seen_pairs.push((a.clone(), p.clone()));
                }
            }
        }
    }

    let mut sorted_seen = seen_pairs.clone();
    sorted_seen.sort();
    let mut sorted_written = eager_rows.clone();
    sorted_written.sort();
    assert_eq!(
        sorted_seen.len(),
        sorted_written.len(),
        "the union of both ranks' rows must equal the window exactly (no duplicate, no drop)"
    );
    assert_eq!(sorted_seen, sorted_written);

    // Disjointness: no row observed by rank 0 is ALSO observed by rank 1 (a
    // shared cursor would duplicate rows across ranks).
    let rank0_rows: std::collections::BTreeSet<(String, String)> = per_rank_chunks[0]
        .iter()
        .flat_map(|(_, c)| match c {
            TextChunk::Pairs { anchors, positives } => anchors
                .iter()
                .cloned()
                .zip(positives.iter().cloned())
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect();
    for (_, chunk) in &per_rank_chunks[1] {
        if let TextChunk::Pairs { anchors, positives } = chunk {
            for (a, p) in anchors.iter().zip(positives.iter()) {
                assert!(
                    !rank0_rows.contains(&(a.clone(), p.clone())),
                    "rank 1 observed a row rank 0 already claimed: ({a}, {p})"
                );
            }
        }
    }
}

/// P7: `StreamConfig::new(0)` is refused — reachable directly from the
/// public path, no `open` call needed to reach it.
#[test]
fn p7_prefetch_zero_is_refused() {
    let err = stream::StreamConfig::new(0).unwrap_err();
    assert!(
        err.to_string().contains("prefetch"),
        "unexpected error: {err}"
    );
}

/// P7: a window whose `end` exceeds the table's actual row count refuses
/// ("streamed window ended after N of M rows"), never silently under-serving.
#[tokio::test(flavor = "multi_thread")]
#[serial(training_set_stream)]
async fn p7_early_end_when_window_exceeds_table_rows() {
    let dir = TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    let fixture = common::multi_row_group_pairs(&session, dir.path(), false).await;
    let table = fixture.table.clone();
    let columns = fixture.columns.clone();
    let total = fixture.written.len();

    let window = stream::RowWindow::new(0, total + 500);
    let spec =
        partition::PartitionSpec::single_rank(13, partition::PartitionRule::BlockByGlobalBatch);
    let cfg = stream::StreamConfig::new(2).unwrap();
    let ts = stream::TrainingSetStream::open(
        &session,
        &table,
        &columns,
        ModelTask::TextEmbedding,
        window,
        stream::Slice::PerRank(spec),
        cfg,
    )
    .await
    .unwrap();

    let result = drain_ok(ts).await;
    let err = result.expect_err(
        "a window past the table's actual row count must refuse, not silently under-serve",
    );
    assert!(
        err.to_string().contains("streamed window ended after"),
        "unexpected error: {err}"
    );
}

/// P7: classification is refused AT `open`, never mid-stream (§9 advisory —
/// classification's label vocabulary is a whole-dataset pass, the same
/// shape as the regression K3 scaler).
#[tokio::test(flavor = "multi_thread")]
#[serial(training_set_stream)]
async fn p7_classification_is_refused_at_open() {
    let dir = TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    let csv = dir.path().join("cls.csv");
    std::fs::write(&csv, "text,label\nhello,a\nworld,b\nfoo,a\n").unwrap();
    session
        .add_source(
            "cls",
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", csv.display())),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let columns = vec!["text".to_string(), "label".to_string()];
    let (table, _batches) = jammi_ai::fine_tune::training_set::materialize_projection(
        &session,
        "cls",
        &columns,
        ModelTask::Classification,
        "classification",
    )
    .await
    .unwrap();

    let window = stream::RowWindow::new(0, 3);
    let spec =
        partition::PartitionSpec::single_rank(2, partition::PartitionRule::BlockByGlobalBatch);
    let cfg = stream::StreamConfig::new(1).unwrap();
    let err = stream::TrainingSetStream::open(
        &session,
        &table,
        &columns,
        ModelTask::Classification,
        window,
        stream::Slice::PerRank(spec),
        cfg,
    )
    .await
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("classification cannot stream per-step"),
        "unexpected error: {err}"
    );
}

/// P3: the streamed read completes under a session pool sized at the 64 MiB
/// floor while the SAME table's eager collected-batch reservation (the exact
/// mechanism `training_set::materialize_and_read` applies —
/// `MemoryConsumer("training_set_eager")`, `try_grow` over
/// `RecordBatch::get_array_memory_size()`) fails `ResourcesExhausted` under
/// the SAME pool — RED at base (no pool existed to fail against before
/// M2/M3). Also exercises the stream's OWN pool exhaustion: a single
/// oversized chunk (the WHOLE 80 MiB-plus table in one step) is refused the
/// same typed way.
///
/// Two sessions, same on-disk catalog/storage root: session A (a normal,
/// large pool) WRITES the padded fixture — writing it under the 64 MiB floor
/// itself would fail from the producer's OWN sort/spill machinery, a
/// different mechanism than the one this oracle targets. Session A is
/// closed (releasing the SQLite exclusive lock) before session B, at the 64
/// MiB floor, opens the SAME directory and sees the already-materialised
/// table via `load_existing_tables`.
#[tokio::test(flavor = "multi_thread")]
#[serial(training_set_stream)]
async fn p3_streamed_read_completes_under_a_small_pool_while_eager_fails() {
    let dir = TempDir::new().unwrap();
    // ~80 MiB total (800 rows x ~100 KiB), comfortably past the 64 MiB
    // floor, while a single 8-row chunk is ~800 KiB — comfortably under it.
    let rows = 800usize;
    let pad_bytes = 100_000usize;

    let config_a = common::test_config(dir.path());
    let session_a = Arc::new(InferenceSession::new(config_a).await.unwrap());
    let (table, columns) =
        common::padded_regression_fixture(&session_a, dir.path(), rows, pad_bytes).await;
    session_a.close().await;

    let mut config_b = common::test_config(dir.path());
    config_b.engine.memory_limit = "64MB".to_string();
    let session_b = Arc::new(InferenceSession::new(config_b).await.unwrap());

    let window = stream::RowWindow::new(0, rows);
    let spec =
        partition::PartitionSpec::single_rank(8, partition::PartitionRule::BlockByGlobalBatch);
    let cfg = stream::StreamConfig::new(1).unwrap();
    let ts = stream::TrainingSetStream::open(
        &session_b,
        &table,
        &columns,
        ModelTask::Regression,
        window,
        stream::Slice::PerRank(spec),
        cfg,
    )
    .await
    .unwrap();

    let (_chunks, served) = tokio::time::timeout(Duration::from_secs(60), drain_ok(ts))
        .await
        .expect("the streamed read must not hang under the small pool")
        .expect("the streamed read must complete under the small pool");
    assert_eq!(served, rows);

    // The eager mechanism, replicated exactly
    // (`training_set::materialize_and_read`'s `reserve_eager_batches`): a
    // plain collected read (no operator reservation of its own — DataFusion
    // does not pool-account a caller's `Vec<RecordBatch>`) followed by the
    // SAME typed reservation check, which must now fail under the small
    // pool.
    let batches = session_b
        .sql(&read_back_sql(&table, &columns))
        .await
        .expect("the plain collected read itself must succeed (no operator needs the pool)");
    let total_bytes: usize = batches
        .iter()
        .map(arrow::array::RecordBatch::get_array_memory_size)
        .sum();
    let pool = session_b.memory_pool();
    let reservation = datafusion::execution::memory_pool::MemoryConsumer::new("training_set_eager")
        .register(&pool);
    let grow_err = reservation
        .try_grow(total_bytes)
        .expect_err("the eager collected size must exceed the 64 MiB floor");
    assert!(
        grow_err.to_string().contains("training_set_eager"),
        "the eager consumer's name must appear in the failure: {grow_err}"
    );

    // The stream's OWN pool exhaustion: one oversized (whole-table) chunk.
    let window_all = stream::RowWindow::new(0, rows);
    let spec_all =
        partition::PartitionSpec::single_rank(rows, partition::PartitionRule::BlockByGlobalBatch);
    let cfg_one = stream::StreamConfig::new(1).unwrap();
    let ts_one = stream::TrainingSetStream::open(
        &session_b,
        &table,
        &columns,
        ModelTask::Regression,
        window_all,
        stream::Slice::PerRank(spec_all),
        cfg_one,
    )
    .await
    .unwrap();
    let result = drain_ok(ts_one).await;
    match result {
        Err(JammiError::ResourcesExhausted { .. }) => {}
        other => panic!("expected the whole-table single chunk to exhaust the pool: {other:?}"),
    }
}
