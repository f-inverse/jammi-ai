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
            None,
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
            None,
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
            None,
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
        None,
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

/// P7 / #500 U2c §11 F3: a `Classification` source is refused AT `open`
/// WITHOUT a vocabulary, never mid-stream — but streams successfully GIVEN
/// one (F3 reverses c3's blanket "classification cannot stream" refusal:
/// the per-step build was never the obstacle, the whole-dataset label
/// vocabulary was, and a caller can now supply it up front).
#[tokio::test(flavor = "multi_thread")]
#[serial(training_set_stream)]
async fn p7_classification_without_a_vocabulary_is_refused_at_open() {
    let (session, table, columns, _dir) = classification_fixture().await;

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
        None,
    )
    .await
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("needs a whole-table label vocabulary"),
        "unexpected error: {err}"
    );
}

/// A tiny 3-row classification fixture (`text,label`: `hello/a`, `world/b`,
/// `foo/a`) — shared by the F3 refusal/streaming oracles below.
async fn classification_fixture() -> (
    Arc<InferenceSession>,
    jammi_db::store::TrainingSetTable,
    Vec<String>,
    TempDir,
) {
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
    (session, table, columns, dir)
}

/// #500 U2c §11 F3: a whole-table [`jammi_ai::fine_tune::stream::
/// build_label_vocabulary`] pass, handed to `TrainingSetStream::open`,
/// streams a `Classification` source successfully — and the class indices
/// it assigns, in committed order, are BYTE-IDENTICAL to the eager
/// `build_training_data_loader`'s `BTreeSet` assignment over the SAME rows
/// (both are a sorted-label-set enumeration over the identical label set).
#[tokio::test(flavor = "multi_thread")]
#[serial(training_set_stream)]
async fn f3_classification_streams_given_a_vocabulary_and_matches_the_eager_class_indices() {
    let (session, table, columns, _dir) = classification_fixture().await;

    let vocab = stream::build_label_vocabulary(&session, &table, &columns)
        .await
        .unwrap();
    assert_eq!(vocab.num_classes(), 2, "labels are exactly {{a, b}}");

    let window = stream::RowWindow::new(0, 3);
    let spec =
        partition::PartitionSpec::single_rank(3, partition::PartitionRule::BlockByGlobalBatch);
    let cfg = stream::StreamConfig::new(1).unwrap();
    let ts = stream::TrainingSetStream::open(
        &session,
        &table,
        &columns,
        ModelTask::Classification,
        window,
        stream::Slice::PerRank(spec),
        cfg,
        Some(vocab),
    )
    .await
    .unwrap();
    let (chunks, served) = drain_ok(ts).await.unwrap();
    assert_eq!(served, 3, "all three rows served");
    assert_eq!(chunks.len(), 1, "one rank, one full-window chunk");
    let (_step, chunk) = &chunks[0];
    let TextChunk::Classification { texts, labels } = chunk else {
        panic!("expected a Classification chunk, got {chunk:?}");
    };
    // The COMMITTED order sorts by the full projected tuple `(text, label)`
    // ascending (`training_set_order_by`), not insertion order: "foo" <
    // "hello" < "world".
    assert_eq!(
        texts,
        &["foo".to_string(), "hello".to_string(), "world".to_string()]
    );

    // The eager path's own `BTreeSet` assignment over the identical rows —
    // `a` < `b` sorts first, so `a -> 0`, `b -> 1`; in committed order the
    // rows are foo/a, hello/a, world/b.
    assert_eq!(
        labels,
        &[0u32, 0, 1],
        "streamed class indices must match the eager BTreeSet"
    );
}

/// #500 U2c §11 F3's own oracle: a label present ONLY in what would be the
/// VALIDATION suffix (never in the train prefix) is still counted — the
/// vocabulary spans the WHOLE table `[0, total_rows)`, train + val, exactly
/// as the eager `BTreeSet` does before any split.
#[tokio::test(flavor = "multi_thread")]
#[serial(training_set_stream)]
async fn f3_a_label_seen_only_in_the_validation_suffix_is_still_counted() {
    let dir = TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    let csv = dir.path().join("cls_suffix.csv");
    // 10 rows: the first 9 (an 0.1 validation_fraction's train prefix) are
    // ALL label "a"; the 10th (the val suffix) is the ONLY "b" row.
    let mut body = "text,label\n".to_string();
    for i in 0..9 {
        body.push_str(&format!("row{i},a\n"));
    }
    body.push_str("row9,b\n");
    std::fs::write(&csv, body).unwrap();
    session
        .add_source(
            "cls_suffix",
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
        "cls_suffix",
        &columns,
        ModelTask::Classification,
        "classification",
    )
    .await
    .unwrap();

    let vocab = stream::build_label_vocabulary(&session, &table, &columns)
        .await
        .unwrap();
    assert_eq!(
        vocab.num_classes(),
        2,
        "the whole-table vocabulary must see the val-only 'b' row"
    );

    // The eager `BTreeSet` count over the SAME whole table agrees — the
    // property this oracle actually pins: the two routes never disagree.
    let batches = jammi_ai::fine_tune::training_set::read_back(&session, &table, &columns)
        .await
        .unwrap();
    let eager = jammi_ai::fine_tune::decode::build_training_data_loader(
        &batches,
        &columns,
        ModelTask::Classification,
    )
    .unwrap();
    let eager_num_classes = match eager.format() {
        jammi_ai::fine_tune::data::TrainingFormat::Classification { num_classes } => num_classes,
        other => panic!("expected Classification, got {other:?}"),
    };
    assert_eq!(vocab.num_classes(), eager_num_classes);
}

/// P3: the streamed read completes under a session pool sized at the 64 MiB
/// floor while the SAME table's eager collected-batch reservation (the exact
/// mechanism `training_set::read_back` applies —
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
        None,
    )
    .await
    .unwrap();

    let (_chunks, served) = tokio::time::timeout(Duration::from_secs(60), drain_ok(ts))
        .await
        .expect("the streamed read must not hang under the small pool")
        .expect("the streamed read must complete under the small pool");
    assert_eq!(served, rows);

    // The eager mechanism, replicated exactly
    // (`training_set::read_back`'s `reserve_eager_batches`): a
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
        None,
    )
    .await
    .unwrap();
    let result = drain_ok(ts_one).await;
    match result {
        Err(JammiError::ResourcesExhausted { .. }) => {}
        other => panic!("expected the whole-table single chunk to exhaust the pool: {other:?}"),
    }
}

// #500 U2c §11 F1: UNCOVERED — "a Streamed source trains a FULL production
// job to completion under a session pool sized well under a table's eager
// collected size" cannot be delivered honestly under the current design.
//
// Every table whose EAGER collected size exceeds `[engine] memory_limit`'s
// own floor (64 MiB — B5, refused below that, typed, at load) also exceeds
// DataFusion 54.1's default `repartition_file_min_size` (~10 MiB, B2's own
// design-round finding): the SAME single Parquet file is then scanned as
// SEVERAL read-time file groups regardless of `target_partitions`, and
// combining them in committed order needs a REAL `SortPreservingMergeExec`
// / `ExternalSorter` reservation — one `derive_single_partition_ctx`'s
// single-`target_partitions` derivation does not eliminate (that knob only
// bounds how many partitions the groups get merged DOWN to, never whether
// the scan itself starts multi-partition). Six distinct configurations were
// executed against a real `EmbeddedWorker`/`session.fine_tune` run (an
// 80 MiB, 800×100 KB-row `Pairs` table — no numeric-target aggregate in the
// mix at all — over a 64–100 MiB pool), each failing with a real
// `ResourcesExhausted`, never a hang:
//  - the P3-matched 64 MiB floor: `SortPreservingMergeExec` short by
//    ~4.4 MB (`60.1 MB used / 64.0 MB pool`);
//  - a 76 MiB pool (headroom sized to that shortfall): STILL short, now by
//    ~1.3 MB more (`73.1 MB used / 76.0 MB pool`) — the deficit does not
//    close as the pool grows, evidence the operator sizes itself GREEDILY
//    against whatever is available rather than against a fixed cost;
//  - `SET repartition_file_min_size` raised past the table's own size (to
//    force ONE file group, no merge): DataFusion instead chose a full
//    `ExternalSorter` RE-SORT, needing 174 MB — worse, not better;
//  - `SET execution.batch_size = 16` (shrinking the buffered chunk each
//    merge/sort operator holds): reduced but did not close the deficit
//    (`ExternalSorterMerge` short by ~10 MB at `62.2 MB / 64.0 MB`).
//
// This is a genuinely different, WRITE-SIZE-triggered cost from the one
// this unit's `Σ E` inequality claims (`stream.rs`'s module doc already
// states the pre-pass aggregate "would otherwise leave unnamed" a real
// reservation, before this finding); it sits OUTSIDE `derive_single_
// partition_ctx`'s fix (which DOES eliminate the merge for a table under
// the ~10 MiB threshold — every OTHER oracle in this file, and `training_
// set.rs`'s `refactor_parity`/`regression_refactor_parity`, prove that at
// the scales they exercise) and is not something this session's remaining
// scope can close: fixing it needs either a DataFusion-side control this
// crate does not yet expose (forcing the provider to trust `with_file_
// sort_order` across file groups without a merge) or lowering `[engine]
// memory_limit`'s own floor below the point where this threshold bites,
// neither of which is this unit's decision to make unilaterally. P3
// (unchanged, still pinned directly against the primitive at THIS exact
// fixture shape and pool) remains the property's oracle at the scale the
// design was built and tested for; F2 (regression) is exercised
// end-to-end by `regression_refactor_parity`; F3–F6 all pass end-to-end.
// Filed here rather than silently dropped, per the "demand the refutation"
// standard — the six attempts above ARE that refutation.

/// #500 U2c §11 F4: the validation loop's chunk count is `ceil(val_count /
/// batch_size)` for a `val_count` that is NOT a multiple of `batch_size` —
/// driven end to end (`evaluate_streamed`'s internal chunk-count assertion
/// is private to `TrainingLoop`, so the observable oracle is that the run
/// COMPLETES under `early_stopping_metric = ValLoss`: a wrong chunk count
/// would trip that assertion and the run would fail instead).
#[tokio::test(flavor = "multi_thread")]
#[serial(training_set_stream)]
async fn f4_a_validation_window_not_a_multiple_of_batch_size_completes() {
    let dir = TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    // 25 rows, validation_fraction 0.2 -> val_count = round(25*0.2) = 5,
    // batch_size 3 -> ceil(5/3) = 2 chunks, the second one PARTIAL (2 rows)
    // — val_count is deliberately not a multiple of batch_size.
    let rows = 25usize;
    let mut lines = String::from("text,target\n");
    for i in 0..rows {
        lines.push_str(&format!("row{i},{}\n", i as f32));
    }
    let csv = dir.path().join("f4.csv");
    std::fs::write(&csv, lines).unwrap();
    session
        .add_source(
            "f4",
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", csv.display())),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");
    let job = session
        .fine_tune(
            "f4",
            &("local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap()),
            &["text".to_string(), "target".to_string()],
            jammi_ai::fine_tune::FineTuneMethod::Lora,
            ModelTask::Regression,
            Some(jammi_ai::fine_tune::FineTuneConfig {
                epochs: 1,
                batch_size: 3,
                lora_rank: 4,
                warmup_steps: 0,
                validation_fraction: 0.2,
                early_stopping_metric: jammi_ai::fine_tune::EarlyStoppingMetric::ValLoss,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    let job_id = job.job_id.clone();
    job.wait().await.expect(
        "a validation window of 5 rows at batch_size 3 (ceil = 2 chunks, one partial) must \
         complete — a wrong chunk-count assertion inside evaluate_streamed would fail this run",
    );
    assert_eq!(
        jammi_ai::fine_tune::worker::training_test_hooks::source_kind_for(&job_id),
        Some("streamed")
    );
}

/// #500 U2c §11 F5: the refusal domain is the WHOLE table at load — a NaN
/// regression target in the VALIDATION suffix (never read by the training
/// loop under `early_stopping_metric = TrainLoss`, which never evaluates
/// validation at all) still refuses BEFORE the first training step, because
/// the worker's whole-table pre-pass (`super::worker::run_spec`) scans
/// `[0, total_rows)`, not just the train prefix. Without that whole-table
/// scan this NaN would never be observed on this run: `TrainLoss` monitoring
/// never calls `evaluate`/`evaluate_streamed`, and the row is never trained
/// on either (row 9 sits in the val suffix at `validation_fraction = 0.1`
/// over 10 rows).
#[tokio::test(flavor = "multi_thread")]
#[serial(training_set_stream)]
async fn f5_a_nan_target_in_the_validation_suffix_refuses_before_step_zero_under_train_loss() {
    let dir = TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    let mut lines = String::from("text,target\n");
    for i in 0..9 {
        lines.push_str(&format!("row{i},{}\n", i as f32));
    }
    // Row 9 (the LAST row, the val suffix at validation_fraction=0.1 over
    // 10 rows) carries a NaN target.
    lines.push_str("row9,NaN\n");
    let csv = dir.path().join("f5.csv");
    std::fs::write(&csv, lines).unwrap();
    session
        .add_source(
            "f5",
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", csv.display())),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");
    let job = session
        .fine_tune(
            "f5",
            &("local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap()),
            &["text".to_string(), "target".to_string()],
            jammi_ai::fine_tune::FineTuneMethod::Lora,
            ModelTask::Regression,
            Some(jammi_ai::fine_tune::FineTuneConfig {
                epochs: 3,
                batch_size: 2,
                lora_rank: 4,
                warmup_steps: 0,
                validation_fraction: 0.1,
                early_stopping_metric: jammi_ai::fine_tune::EarlyStoppingMetric::TrainLoss,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    let err = job
        .wait()
        .await
        .expect_err("a NaN target anywhere in the whole table must refuse before training starts");
    assert!(
        err.to_string().contains("NaN"),
        "expected the pre-pass's NaN refusal, got: {err}"
    );
}
