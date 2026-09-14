//! The streaming training-set loader (#500 U2b): the per-epoch `RecordBatch`
//! stream, its prefetch-bounded residency, and the row order it reads back
//! in.
//!
//! Built directly against `ResultStore::materialize_training_set` (through
//! `training_set::materialize_projection`, which still eagerly reads the
//! table back for its OWN callers) so that setup's eager read never enters
//! what these tests measure — every assertion here drives
//! `TrainingDataLoader::from_training_set_stream` itself, which never asks
//! for that eager read.

use std::sync::Arc;

use arrow::array::Array;
use jammi_ai::fine_tune::data::{StreamConfig, TrainingDataLoader, TrainingFormat};
use jammi_ai::fine_tune::partition::{PartitionRule, PartitionSpec};
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use tempfile::TempDir;

use crate::common;

async fn session_with_pairs_source(dir: &TempDir) -> Arc<InferenceSession> {
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    session
        .add_source(
            "training",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("training_pairs.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    session
}

fn contrastive_columns() -> Vec<String> {
    vec!["text_a".into(), "text_b".into(), "score".into()]
}

/// Read a text column's values regardless of which Arrow string family
/// DataFusion resolved it as (`Utf8`/`Utf8View`/`LargeUtf8`) — the resolved
/// schema is `Utf8View` under the Parquet reader default (see
/// `training_set`'s own module doc), which a plain `StringArray` downcast
/// misses.
fn text_column_values(col: &dyn arrow::array::Array) -> Vec<String> {
    use arrow::array::{LargeStringArray, StringArray, StringViewArray};
    if let Some(a) = col.as_any().downcast_ref::<StringViewArray>() {
        return (0..a.len()).map(|i| a.value(i).to_string()).collect();
    }
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        return (0..a.len()).map(|i| a.value(i).to_string()).collect();
    }
    if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
        return (0..a.len()).map(|i| a.value(i).to_string()).collect();
    }
    panic!(
        "column is not a recognised text Arrow type: {}",
        col.data_type()
    );
}

/// Materialise the 30-row `training_pairs.csv` fixture as a committed
/// `TrainingSet` table and return the handle — never the eagerly read-back
/// `Vec<RecordBatch>` `materialize_projection` also returns (dropped here),
/// so nothing this module measures ever passes through it.
async fn materialize_contrastive_table(
    session: &InferenceSession,
) -> jammi_db::store::TrainingSetTable {
    let columns = contrastive_columns();
    let (table, _eagerly_read_back) = jammi_ai::fine_tune::training_set::materialize_projection(
        session,
        "training",
        &columns,
        ModelTask::TextEmbedding,
        "contrastive",
    )
    .await
    .unwrap();
    assert_eq!(table.record.row_count, 30, "fixture row count moved");
    table
}

/// Acceptance (a): on a fixture (30 rows) LARGER than `batch * prefetch`
/// (4 * 2 = 8), the resident-row high-water mark never exceeds that bound —
/// the counting seam lives in `ResidencyBound::high_water_mark`, exposed for
/// this assertion via `TrainingDataLoader::stream_residency_high_water_mark`.
///
/// RED at base: `TrainingDataLoader::from_training_set_stream` does not exist
/// on the base tree — `build_training_data_loader` there converts the WHOLE
/// eagerly-collected `Vec<RecordBatch>` into a `Vec<TrainingRow>` up front,
/// so the loader holds all 30 rows for the whole run, not 8.
#[tokio::test(flavor = "multi_thread")]
async fn residency_bound_holds_on_a_fixture_larger_than_batch_times_prefetch() {
    let dir = TempDir::new().unwrap();
    let session = session_with_pairs_source(&dir).await;
    let table = materialize_contrastive_table(&session).await;

    let cfg = StreamConfig {
        batch: 4,
        prefetch: 2,
    };
    let bound = cfg.batch * cfg.prefetch;
    let loader = TrainingDataLoader::from_training_set_stream(
        Arc::clone(&session),
        table,
        contrastive_columns(),
        ModelTask::TextEmbedding,
        TrainingFormat::Contrastive,
        cfg,
    )
    .await
    .unwrap();

    // Drive one full epoch, step by step, on a `spawn_blocking` thread —
    // production's own bridge (`worker.rs::train_fine_tune` wraps the WHOLE
    // training loop in exactly this).
    let spec = PartitionSpec {
        rank: 0,
        world: 1,
        batch: cfg.batch,
        rule: PartitionRule::BlockByGlobalBatch,
    };
    let high_water = tokio::task::spawn_blocking(move || {
        let mut step = 0usize;
        loop {
            let chunk = loader.text_chunk_for_rank(&spec, step).unwrap();
            if chunk.row_count() == 0 {
                break;
            }
            step += 1;
        }
        loader.stream_residency_high_water_mark()
    })
    .await
    .unwrap();

    let high_water =
        high_water.expect("a Stream loader that ran an epoch reports a high-water mark");
    assert!(
        high_water <= bound,
        "resident-row high-water mark {high_water} exceeded the configured bound {bound} \
         (batch={}, prefetch={})",
        cfg.batch,
        cfg.prefetch
    );
    assert!(high_water > 0, "the run must have actually held some rows");
}

// R-A for (a) — "make the bound one row too small and watch it fail" — is a
// UNIT test, not here: `ResidencyBound` is private to `jammi_ai::fine_tune::
// data`, so it is driven directly from `fine_tune::data::tests::
// residency_bound_semaphore_refuses_one_row_past_a_shrunk_bound` in that
// module's own `#[cfg(test)]` block.

/// M2's deadlock-prevention refusal (dying test): `StreamConfig::prefetch < 2`
/// is refused at loader construction, never silently accepted into a
/// configuration that would deadlock the first time the reader tried to get
/// one chunk ahead of the consumer.
#[tokio::test(flavor = "multi_thread")]
async fn prefetch_below_two_is_refused_at_construction() {
    let dir = TempDir::new().unwrap();
    let session = session_with_pairs_source(&dir).await;
    let table = materialize_contrastive_table(&session).await;

    let result = TrainingDataLoader::from_training_set_stream(
        Arc::clone(&session),
        table,
        contrastive_columns(),
        ModelTask::TextEmbedding,
        TrainingFormat::Contrastive,
        StreamConfig {
            batch: 4,
            prefetch: 1,
        },
    )
    .await;
    match result {
        Err(e) => assert!(e.to_string().contains("prefetch")),
        Ok(_) => panic!(
            "prefetch=1 must be refused at construction, not accepted into a deadlocking config"
        ),
    }
}

/// Acceptance (e): the streamed row order equals the committed
/// materialisation order, with no blocking sort — asserted at
/// `target_partitions ∈ {1, N}` for the AMBIENT session (the loader's own
/// scoped read always pins `target_partitions = 1` regardless).
///
/// Mechanism named: `open_row_range_stream` scans with NO `ORDER BY` at all
/// (so there is no `SortExec` to insert in the first place) and forces
/// `target_partitions = 1` on its own scoped session, so a single-partition
/// sequential scan visits the producer's already-sorted row groups in
/// exactly their committed order.
///
/// RED at base: no streaming reader exists to have an order property at all.
#[tokio::test(flavor = "multi_thread")]
async fn streamed_order_matches_committed_order_at_target_partitions_one_and_n() {
    for target_partitions in [1usize, 4usize] {
        let dir = TempDir::new().unwrap();
        let mut config = common::test_config(dir.path());
        config.engine.execution_threads = target_partitions;
        let session = Arc::new(InferenceSession::new(config).await.unwrap());
        session
            .add_source(
                "training",
                SourceType::File,
                SourceConnection {
                    url: Some(common::fixture_url("training_pairs.csv")),
                    format: Some(FileFormat::Csv),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let table = materialize_contrastive_table(&session).await;

        // The impossibility claim behind (e)'s mechanism, MEASURED rather than
        // assumed: the exact scoped-session, no-`ORDER BY` scan
        // `open_row_range_stream` builds contains no `SortExec` node at all —
        // walk the real physical plan's `Debug` output and assert the string
        // "SortExec" never appears, at both `target_partitions` values (the
        // ambient session's own config, which the loader's private scoped
        // session always overrides to 1 regardless).
        {
            use datafusion::execution::SessionState;
            use datafusion::physical_plan::ExecutionPlan;
            use datafusion::prelude::SessionContext;

            let mut state: SessionState = session.context().state();
            let scoped_config = state
                .config()
                .clone()
                .with_target_partitions(1)
                .with_batch_size(7);
            *state.config_mut() = scoped_config;
            let scoped_ctx = SessionContext::new_with_state(state);
            let sql = format!("SELECT * FROM {} LIMIT 30 OFFSET 0", table.sql_relation());
            let df = scoped_ctx.sql(&sql).await.unwrap();
            let physical: std::sync::Arc<dyn ExecutionPlan> =
                df.create_physical_plan().await.unwrap();
            let plan_debug = format!("{physical:?}");
            assert!(
                !plan_debug.contains("SortExec"),
                "target_partitions={target_partitions}: the streamed read must contain NO \
                 SortExec (no ORDER BY was ever asked for) — got plan: {plan_debug}"
            );
        }

        // The committed order, read back the WAY `training_set::read_back_sql`
        // defines it (full-tuple `ORDER BY`) — the oracle this test's
        // streamed order must match.
        let expected_sql = format!(
            "SELECT * FROM {} {}",
            table.sql_relation(),
            jammi_db::store::training_set_order_by(&contrastive_columns())
        );
        let expected_batches = session.sql(&expected_sql).await.unwrap();
        let expected: Vec<String> = expected_batches
            .iter()
            .flat_map(|b| {
                let col = b.column_by_name("text_a").unwrap();
                text_column_values(col.as_ref())
            })
            .collect();
        assert_eq!(expected.len(), 30);

        let cfg = StreamConfig {
            batch: 7,
            prefetch: 2,
        };
        let loader = TrainingDataLoader::from_training_set_stream(
            Arc::clone(&session),
            table,
            contrastive_columns(),
            ModelTask::TextEmbedding,
            TrainingFormat::Contrastive,
            cfg,
        )
        .await
        .unwrap();
        let spec = PartitionSpec {
            rank: 0,
            world: 1,
            batch: cfg.batch,
            rule: PartitionRule::BlockByGlobalBatch,
        };
        let streamed = tokio::task::spawn_blocking(move || {
            let mut texts_a = Vec::new();
            let mut step = 0usize;
            loop {
                let chunk = loader.text_chunk_for_rank(&spec, step).unwrap();
                match &*chunk {
                    jammi_ai::fine_tune::data::TextChunk::Contrastive { texts_a: a, .. } => {
                        if a.is_empty() {
                            break;
                        }
                        texts_a.extend(a.iter().cloned());
                    }
                    _ => panic!("expected a Contrastive chunk"),
                }
                step += 1;
            }
            texts_a
        })
        .await
        .unwrap();

        assert_eq!(
            streamed, expected,
            "target_partitions={target_partitions}: the streamed order must match the \
             committed materialisation order"
        );
    }
}

/// A 70 000-row `(anchor, positive)` pairs fixture — more than one 65 536-row
/// Parquet row group — committed as a training set and returned as its
/// handle. `anchor`/`positive` values are `a{i:05}`/`p{i:05}` in row order, so
/// the committed order is directly recoverable from a streamed anchor's own
/// text (`format!("a{i:05}", i)`), with no separate oracle query needed.
async fn materialize_multi_row_group_pairs_table(
    session: &InferenceSession,
    scratch_dir: &std::path::Path,
) -> jammi_db::store::TrainingSetTable {
    const ROWS: usize = 70_000;
    // A caller-supplied, per-test `TempDir` — NEVER the shared OS temp
    // directory keyed by process id: several of this file's `#[tokio::test]`
    // functions run concurrently as tasks in the SAME process, so a
    // process-id-keyed path collides between them (one test's fixture write
    // racing another's read/removal of the identically-named file).
    let path = scratch_dir.join("pairs70k.csv");
    let mut lines = String::from("anchor,positive\n");
    for i in 0..ROWS {
        lines.push_str(&format!("a{i:05},p{i:05}\n"));
    }
    std::fs::write(&path, lines).unwrap();
    session
        .add_source(
            "pairs70k",
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", path.display())),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let columns = vec!["anchor".to_string(), "positive".to_string()];
    let (table, _eager) = jammi_ai::fine_tune::training_set::materialize_projection(
        session,
        "pairs70k",
        &columns,
        ModelTask::TextEmbedding,
        "pairs",
    )
    .await
    .unwrap();
    assert_eq!(table.record.row_count, ROWS);
    table
}

/// M1 (CONTRACT-U2b-fix1.md): on a fixture LARGER than one Parquet row group
/// (70 000 rows > 65 536), at a batch size that does NOT divide the row-group
/// boundary (`100`; `65_536 % 100 == 36`), the streamed epoch takes exactly
/// `ceil(70000 / 100) = 700` steps and every non-final chunk is exactly 100
/// rows — never a short chunk at the row-group boundary.
///
/// RED at 6482ea99 (observed): 701 steps, with a 36-row short chunk at step
/// 655 (the tail of the first row group) — see this unit's report.
#[tokio::test(flavor = "multi_thread")]
async fn streamed_chunks_stay_batch_sized_across_a_row_group_boundary() {
    const ROWS: usize = 70_000;
    const BATCH: usize = 100;
    let dir = TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    let table = materialize_multi_row_group_pairs_table(&session, dir.path()).await;
    let columns = vec!["anchor".to_string(), "positive".to_string()];

    let loader = TrainingDataLoader::from_training_set_stream(
        Arc::clone(&session),
        table,
        columns,
        ModelTask::TextEmbedding,
        TrainingFormat::Pairs,
        StreamConfig {
            batch: BATCH,
            prefetch: 2,
        },
    )
    .await
    .unwrap();
    let expected_steps = ROWS.div_ceil(BATCH);
    let spec = PartitionSpec {
        rank: 0,
        world: 1,
        batch: BATCH,
        rule: PartitionRule::BlockByGlobalBatch,
    };
    let (sizes, order_ok) = tokio::task::spawn_blocking(move || {
        let mut sizes = Vec::with_capacity(expected_steps);
        let mut seen = Vec::with_capacity(ROWS);
        let mut step = 0usize;
        loop {
            let chunk = loader.text_chunk_for_rank(&spec, step).unwrap();
            if chunk.row_count() == 0 {
                break;
            }
            sizes.push(chunk.row_count());
            if let jammi_ai::fine_tune::data::TextChunk::Pairs { anchors, .. } = &*chunk {
                seen.extend(anchors.iter().cloned());
            } else {
                panic!("expected a Pairs chunk");
            }
            step += 1;
        }
        let order_ok = seen
            .iter()
            .enumerate()
            .all(|(i, a)| *a == format!("a{i:05}"));
        (sizes, order_ok)
    })
    .await
    .unwrap();

    assert_eq!(
        sizes.len(),
        expected_steps,
        "the streamed epoch must take exactly ceil(train_count / batch) steps"
    );
    let non_final_short: Vec<(usize, usize)> = sizes
        .iter()
        .enumerate()
        .filter(|(i, n)| **n != BATCH && *i + 1 != sizes.len())
        .map(|(i, n)| (i, *n))
        .collect();
    assert!(
        non_final_short.is_empty(),
        "every non-final chunk must be exactly {BATCH} rows; short chunks at {non_final_short:?}"
    );
    assert_eq!(
        *sizes.last().unwrap(),
        ROWS - BATCH * (expected_steps - 1),
        "the trailing chunk must hold exactly the remainder"
    );
    assert!(
        order_ok,
        "the streamed order must equal the committed order"
    );
}

/// M1's chunk-sequence identity: the `Stream` arm's `(chunk_index → rows)`
/// equals the eager `TrainingDataLoader::text_chunks(B)` reader's — same
/// sizes, same contents, row for row — on the SAME multi-row-group fixture, a
/// batch size (`997`; `65_536 % 997 == 728`) that does not divide the
/// row-group boundary either. The eager reader (`stream_drain_all` +
/// `text_chunks`) is a whole-table read, so it is unaffected by a row-group
/// boundary by construction — it is the oracle the per-step `Stream` arm
/// must match exactly.
#[tokio::test(flavor = "multi_thread")]
async fn streamed_chunk_sequence_matches_the_eager_reader_across_row_groups() {
    const BATCH: usize = 997;
    let dir = TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    let table = materialize_multi_row_group_pairs_table(&session, dir.path()).await;
    let columns = vec!["anchor".to_string(), "positive".to_string()];

    let eager_loader = TrainingDataLoader::from_training_set_stream(
        Arc::clone(&session),
        table.clone(),
        columns.clone(),
        ModelTask::TextEmbedding,
        TrainingFormat::Pairs,
        StreamConfig {
            batch: BATCH,
            prefetch: 2,
        },
    )
    .await
    .unwrap();
    let stream_loader = TrainingDataLoader::from_training_set_stream(
        Arc::clone(&session),
        table,
        columns,
        ModelTask::TextEmbedding,
        TrainingFormat::Pairs,
        StreamConfig {
            batch: BATCH,
            prefetch: 2,
        },
    )
    .await
    .unwrap();

    let (eager_chunks, streamed_chunks) = tokio::task::spawn_blocking(move || {
        // `text_chunks` is the whole-set exemption (`stream_drain_all`), so it
        // is a legitimate oracle independent of the per-step Stream arm this
        // test is checking.
        let eager: Vec<Vec<String>> = eager_loader
            .text_chunks(BATCH)
            .unwrap()
            .into_iter()
            .map(|c| match c {
                jammi_ai::fine_tune::data::TextChunk::Pairs { anchors, .. } => anchors,
                _ => panic!("expected a Pairs chunk"),
            })
            .collect();

        let spec = PartitionSpec {
            rank: 0,
            world: 1,
            batch: BATCH,
            rule: PartitionRule::BlockByGlobalBatch,
        };
        let mut streamed: Vec<Vec<String>> = Vec::new();
        let mut step = 0usize;
        loop {
            let chunk = stream_loader.text_chunk_for_rank(&spec, step).unwrap();
            if chunk.row_count() == 0 {
                break;
            }
            match &*chunk {
                jammi_ai::fine_tune::data::TextChunk::Pairs { anchors, .. } => {
                    streamed.push(anchors.clone())
                }
                _ => panic!("expected a Pairs chunk"),
            }
            step += 1;
        }
        (eager, streamed)
    })
    .await
    .unwrap();

    assert_eq!(
        streamed_chunks.len(),
        eager_chunks.len(),
        "the Stream arm must yield the same NUMBER of chunks as the eager reader"
    );
    for (i, (streamed, eager)) in streamed_chunks.iter().zip(eager_chunks.iter()).enumerate() {
        assert_eq!(
            streamed.len(),
            eager.len(),
            "chunk {i}: the Stream arm's chunk size must match the eager reader's"
        );
        assert_eq!(
            streamed, eager,
            "chunk {i}: the Stream arm's rows must match the eager reader's, row for row"
        );
    }
}

/// The executed refutation behind M1's ordering mechanism: `open_row_range_
/// stream`'s scoped `target_partitions = 1` override is load-bearing, not
/// incidental. Run the SAME `LIMIT ... OFFSET ...` scan `open_row_range_
/// stream` builds but on the AMBIENT session at `execution_threads = 4` (no
/// scoped override, no `ORDER BY`) against the SAME multi-row-group table,
/// and observe the row order NO LONGER matches the committed order — proving
/// the pin, not mere luck, is what keeps the real streamed reader correct.
#[tokio::test(flavor = "multi_thread")]
async fn removing_the_target_partitions_pin_breaks_order_on_a_multi_row_group_table() {
    let dir = TempDir::new().unwrap();
    let mut config = common::test_config(dir.path());
    config.engine.execution_threads = 4;
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    let table = materialize_multi_row_group_pairs_table(&session, dir.path()).await;
    session
        .sql("SET datafusion.optimizer.repartition_file_min_size = 1")
        .await
        .unwrap();

    // The ambient, UN-scoped scan: no `target_partitions` override, no
    // `ORDER BY` — exactly what `open_row_range_stream` would read if its
    // `with_target_partitions(1)` override were removed.
    let sql = format!(
        "SELECT * FROM {} LIMIT 70000 OFFSET 0",
        table.sql_relation()
    );
    let batches = session.sql(&sql).await.unwrap();
    let unscoped: Vec<String> = batches
        .iter()
        .flat_map(|b| text_column_values(b.column_by_name("anchor").unwrap().as_ref()))
        .collect();
    assert_eq!(unscoped.len(), 70_000);
    let in_order = unscoped
        .iter()
        .enumerate()
        .all(|(i, a)| *a == format!("a{i:05}"));
    assert!(
        !in_order,
        "the un-scoped, multi-partition read was expected to SCRAMBLE the committed order — \
         if it did not, the pin this mechanism relies on is not actually load-bearing here"
    );
}

/// M3 (CONTRACT-U2b-fix1.md): `build_classification_loader_eager` reads
/// through `training_set::read_back_range_sql` (`ORDER BY` re-applied), so
/// classification rows on a multi-row-group, multi-thread fixture arrive in
/// the committed order — never the ambient session's unordered
/// partition-by-partition scan the base `SELECT * ... LIMIT ... OFFSET ...`
/// (no `ORDER BY`) fell back to.
///
/// RED at 6482ea99 (observed): first divergence at index 0 — the base arm's
/// SQL carried no `ORDER BY` at all, so a 4-thread scan over more than one
/// row group came back interleaved from the first row.
#[tokio::test(flavor = "multi_thread")]
async fn classification_eager_fallback_preserves_committed_order_across_row_groups() {
    const ROWS: usize = 70_000;
    let dir = TempDir::new().unwrap();
    let mut lines = String::from("text,label\n");
    for i in 0..ROWS {
        let n = (i * 37) % ROWS;
        lines.push_str(&format!("t{n:05},l{}\n", n % 3));
    }
    let csv = dir.path().join("cls.csv");
    std::fs::write(&csv, lines).unwrap();

    let mut config = common::test_config(dir.path());
    config.engine.execution_threads = 4;
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
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
    session
        .sql("SET datafusion.optimizer.repartition_file_min_size = 1")
        .await
        .unwrap();

    let columns = vec!["text".to_string(), "label".to_string()];
    let (table, _eager) = jammi_ai::fine_tune::training_set::materialize_projection(
        &session,
        "cls",
        &columns,
        ModelTask::TextEmbedding,
        "classification",
    )
    .await
    .unwrap();
    assert_eq!(table.record.row_count, ROWS);

    // The committed order, exactly as `training_set::read_back_sql` defines
    // it — the oracle the classification arm's rows must match.
    let expected_sql = format!(
        "SELECT * FROM {} {}",
        table.sql_relation(),
        jammi_db::store::training_set_order_by(&columns)
    );
    let expected: Vec<String> = session
        .sql(&expected_sql)
        .await
        .unwrap()
        .iter()
        .flat_map(|b| text_column_values(b.column_by_name("text").unwrap().as_ref()))
        .collect();

    let loader = TrainingDataLoader::from_training_set_stream(
        Arc::clone(&session),
        table,
        columns,
        ModelTask::TextEmbedding,
        TrainingFormat::Classification { num_classes: 0 },
        StreamConfig {
            batch: 8,
            prefetch: 4,
        },
    )
    .await
    .unwrap();
    let chunks = loader.text_chunks(4096).unwrap();
    let mut actual = Vec::with_capacity(ROWS);
    for chunk in &chunks {
        match chunk {
            jammi_ai::fine_tune::data::TextChunk::Classification { texts, .. } => {
                actual.extend(texts.iter().cloned())
            }
            _ => panic!("expected a Classification chunk"),
        }
    }
    assert_eq!(actual.len(), ROWS, "row count");
    let first_divergence = expected.iter().zip(actual.iter()).position(|(e, a)| e != a);
    assert_eq!(
        first_divergence, None,
        "the classification loader's row order diverges from the committed order at {:?}",
        first_divergence
    );
}

/// Acceptance (d), the mechanism proof: hard-negative mining and GradCache
/// are W=1-only whole-set consumers that "stream the table in and hold what
/// they need" (`TrainingDataLoader::in_batch_negative_texts`'s `Stream` arm
/// — `stream_drain_all`, exempt from the residency bound). Both run through
/// the SAME `run_spec` → `from_source_stream` path
/// [`residency_bound_holds_on_a_fixture_larger_than_batch_times_prefetch`]
/// exercises for the ordinary per-step path, so this closes the ONE call
/// these two mechanisms make that the rest of this file's tests never drive:
/// a WHOLE-loader drain on a live `Stream` loader.
///
/// **What this does NOT claim.** This is a functional (job completes,
/// publishes an adapter) proof, not a byte-parity pin like (c)'s — no
/// hermetic fixture exercising mining/GradCache was pinned at U2b's base
/// (`4e27156a`) before this unit started, so a byte-identical-to-base claim
/// for this specific arm is UNCOVERED, not established; see this unit's
/// report.
#[tokio::test(flavor = "multi_thread")]
async fn hard_negative_mining_completes_through_the_streaming_loader_at_w1() {
    use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod, HardNegativeConfig};

    let dir = TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    session
        .add_source(
            "training",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("training_triplets.csv")),
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
            "training",
            &("local:".to_string() + &common::cookbook_fixture("tiny_bert").display().to_string()),
            &[
                "anchor".to_string(),
                "positive".to_string(),
                "negative".to_string(),
            ],
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            Some(FineTuneConfig {
                epochs: 1,
                batch_size: 4,
                lora_rank: 4,
                warmup_steps: 0,
                hard_negatives: HardNegativeConfig {
                    mine: true,
                    k: 1,
                    exclude_hops: 1,
                    refresh_every: 1,
                },
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    job.wait()
        .await
        .expect("a world=1 mining run over the streamed loader must complete");

    let models = session.catalog().list_models().await.unwrap();
    assert!(
        models
            .iter()
            .any(|m| m.model_id.starts_with("jammi:fine-tuned:")),
        "the mining run must publish a fine-tuned model like any other W=1 run"
    );
}

/// CONTRACT-U2b-fix1.md item 4: the GradCache UNCOVERED byte-parity claim
/// gets a functional, digest-pinned fixture of its own. GradCache
/// (`FineTuneConfig::cached = true`) is refused above `world_size = 1`
/// (`spec.rs`'s `world_size > 1 && common.config.cached` check), so a W=1 run
/// over the SAME streamed loader every other test in this file drives is the
/// only shape GradCache can ever take at this unit — this is that shape,
/// digest-pinned like `training_set::refactor_parity`'s adapter bytes (this
/// test's own FNV-1a `fingerprint`, for the same reason that test states:
/// `sha2` is not a dev-dependency and `DefaultHasher` is not toolchain-stable,
/// so neither can back a constant pinned in source).
///
/// Fingerprinted at THIS unit's own head with the recipe that runs here: a
/// `Pairs`-format projection (`anchor, positive` only) of the 15-row
/// `training_triplets.csv` fixture, one epoch, rank-4 LoRA, GradCache on,
/// MultipleNegativesRanking at its default temperature. Confirmed
/// byte-stable across two repeated runs before being pinned.
///
/// **What this pin does NOT cover.** Only the functional/digest claim for
/// THIS fixture is established. It is not a claim that GradCache is
/// byte-identical to any pre-U2b baseline — U2b's own base (`4e27156a`)
/// never routed GradCache through a hermetic, digest-pinned fixture at all
/// (the base-comparison UNCOVERED claim `hard_negative_mining_completes_
/// through_the_streaming_loader_at_w1`'s doc states stays UNCOVERED for that
/// reason), so there is no base fixture to diff against.
#[tokio::test(flavor = "multi_thread")]
async fn gradcache_completes_through_the_streaming_loader_at_w1_with_a_pinned_adapter_digest() {
    use std::collections::BTreeMap;

    use jammi_ai::fine_tune::{EmbeddingLoss, FineTuneConfig, FineTuneMethod};

    /// FNV-1a over a byte slice, as `{len}:{hash:016x}` — see `training_set::
    /// fingerprint`'s doc for why this (not `sha2`, not `DefaultHasher`).
    fn fingerprint(bytes: &[u8]) -> String {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for b in bytes {
            hash ^= u64::from(*b);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        format!("{}:{:016x}", bytes.len(), hash)
    }

    const GRADCACHE_ADAPTER_PRINTS: &[(&str, &str)] = &[
        ("adapter.safetensors", "1184:36a3ebd09680e266"),
        ("adapter_config.json", "143:1feeeb6239c3fd30"),
        ("checkpoint_1.safetensors", "1184:36a3ebd09680e266"),
        ("checkpoint_best.safetensors", "1184:36a3ebd09680e266"),
        ("manifest.json", "452:42481add2507ae14"),
    ];

    let dir = TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    session
        .add_source(
            "training",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("training_triplets.csv")),
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
            "training",
            &("local:".to_string() + &common::cookbook_fixture("tiny_bert").display().to_string()),
            &["anchor".to_string(), "positive".to_string()],
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            Some(FineTuneConfig {
                epochs: 1,
                batch_size: 4,
                lora_rank: 4,
                warmup_steps: 0,
                cached: true,
                embedding_loss: Some(EmbeddingLoss::MultipleNegativesRanking { temperature: 20.0 }),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    job.wait()
        .await
        .expect("a W=1 GradCache run over the streamed loader must complete");

    let models = session.catalog().list_models().await.unwrap();
    let ft = models
        .iter()
        .find(|m| m.model_id.starts_with("jammi:fine-tuned:"))
        .expect("the GradCache run registers its output model");
    let prefix =
        jammi_db::storage::StorageUrl::parse(ft.artifact_path.as_deref().unwrap()).unwrap();
    let local = session
        .artifact_store()
        .fetch_artifact(&prefix)
        .await
        .expect("the published GradCache adapter fetches and verifies");

    let mut prints = BTreeMap::new();
    for entry in std::fs::read_dir(local.dir()).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            let name = entry.file_name().to_string_lossy().into_owned();
            prints.insert(name, fingerprint(&std::fs::read(entry.path()).unwrap()));
        }
    }
    let expected: BTreeMap<String, String> = GRADCACHE_ADAPTER_PRINTS
        .iter()
        .map(|(n, p)| ((*n).to_string(), (*p).to_string()))
        .collect();
    assert_eq!(
        prints, expected,
        "the GradCache adapter bytes moved from the pinned fixture"
    );
}

/// The M1 oracle fold's trainer-level check: a REAL `TrainingLoop::run`, over
/// the job path (`session.fine_tune`), takes exactly `batches_per_epoch(train,
/// 1, batch)` optimizer steps on a table forced across several small row
/// groups by `test-hooks`' `JAMMI_TEST_ROW_GROUP_ROWS` override — never the
/// row-group-boundary-driven count a broken chunker would take.
///
/// 200 rows, a row-group size of 30 (7 groups: six of 30, one of 20) and a
/// batch of 13 (`30 % 13 == 4`, `20 % 13 == 7` — the row-group boundary does
/// not land on a batch boundary at either group size). `gradient_
/// accumulation_steps: 1` and `validation_fraction: 0.0` make `total_steps`
/// (the run's own reported optimizer-step count) exactly the number of
/// per-step chunks the epoch loop drove — `ceil(200 / 13) = 16`.
///
/// RED at 6482ea99 (observed, by direct construction rather than re-run: the
/// base tree dispatches one `TextChunk` per POLLED `RecordBatch`, and a
/// DataFusion scan never spans a row-group boundary within one polled batch —
/// within one 30-row group at `batch_size = 13` that is 13, 13, 4; six such
/// groups plus the trailing 20-row group's 13, 7 give `6*3 + 2 = 20` steps,
/// not 16).
#[tokio::test(flavor = "multi_thread")]
async fn trainer_realised_step_count_matches_the_global_batch_formula_across_row_groups() {
    const ROWS: usize = 200;
    const ROW_GROUP_ROWS: usize = 30;
    const BATCH: usize = 13;
    let expected_steps = jammi_ai::fine_tune::partition::batches_per_epoch(ROWS, 1, BATCH);
    assert_eq!(
        expected_steps, 16,
        "test setup: the arithmetic this test pins moved"
    );

    // `JAMMI_TEST_ROW_GROUP_ROWS` is process-global (db fold,
    // `jammi_db::storage::writer`); scope it to this test's own write with a
    // guard that always restores the prior value, even on panic — a
    // concurrently-running unrelated test only ever sees a SMALLER row
    // group, which changes no correctness property of its own assertions.
    struct RowGroupOverrideGuard(Option<String>);
    impl Drop for RowGroupOverrideGuard {
        fn drop(&mut self) {
            match &self.0 {
                Some(v) => std::env::set_var(jammi_db::storage::writer::ROW_GROUP_ROWS_ENV, v),
                None => std::env::remove_var(jammi_db::storage::writer::ROW_GROUP_ROWS_ENV),
            }
        }
    }
    let prior = std::env::var(jammi_db::storage::writer::ROW_GROUP_ROWS_ENV).ok();
    std::env::set_var(
        jammi_db::storage::writer::ROW_GROUP_ROWS_ENV,
        ROW_GROUP_ROWS.to_string(),
    );
    let _restore = RowGroupOverrideGuard(prior);

    let dir = TempDir::new().unwrap();
    let mut lines = String::from("anchor,positive\n");
    for i in 0..ROWS {
        lines.push_str(&format!("a{i:03},p{i:03}\n"));
    }
    let csv = dir.path().join("pairs200.csv");
    std::fs::write(&csv, lines).unwrap();

    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    session
        .add_source(
            "pairs200",
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
            "pairs200",
            &("local:".to_string() + &common::cookbook_fixture("tiny_bert").display().to_string()),
            &["anchor".to_string(), "positive".to_string()],
            jammi_ai::fine_tune::FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            Some(jammi_ai::fine_tune::FineTuneConfig {
                epochs: 1,
                batch_size: BATCH,
                lora_rank: 4,
                warmup_steps: 0,
                gradient_accumulation_steps: 1,
                validation_fraction: 0.0,
                early_stopping_metric: jammi_ai::fine_tune::EarlyStoppingMetric::TrainLoss,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    job.wait()
        .await
        .expect("the row-group-boundary-crossing run must complete");

    let record = session.catalog().get_job(&job.job_id).await.unwrap();
    let result_json = record
        .result
        .expect("a completed fine-tune job records its terminal JobResult");
    let job_result: jammi_ai::jobs::JobResult = serde_json::from_str(&result_json).unwrap();
    let metrics = match job_result {
        jammi_ai::jobs::JobResult::Model { metrics, .. } => {
            metrics.expect("a fine-tune run records metrics")
        }
        other => panic!("expected JobResult::Model, got {other:?}"),
    };
    let metrics: serde_json::Value = serde_json::from_str(&metrics).unwrap();
    let total_steps = metrics["total_steps"]
        .as_u64()
        .expect("metrics JSON must carry total_steps") as usize;
    assert_eq!(
        total_steps, expected_steps,
        "the realised optimizer-step count must equal ceil(train_count / batch), not the \
         row-group-boundary-driven count of a chunker that re-chunks at RecordBatch, not \
         partition-rule, boundaries"
    );
}

/// M2 (CONTRACT-U2b-fix1.md): with a consumer that holds each chunk for at
/// least 50 ms (an adversarial stand-in for the production loop's actual
/// model forward+backward, orders of magnitude slower than the reader), the
/// TRUE resident-row count — measured INDEPENDENTLY of `ResidencyBound`'s own
/// self-report via `TrainingDataLoader::stream_chunks_produced` (see that
/// method's doc for why the self-report alone cannot detect this bug: a
/// semaphore can never report past the bound it enforces, so it is
/// structurally blind to permits released too EARLY relative to reality) —
/// never exceeds `batch * prefetch`. Checked at `prefetch` in `{2, 4}`.
///
/// `true_resident = (chunks the reader has produced so far minus chunks THIS
/// consumer has already finished with) * batch`, sampled while still holding
/// the current chunk (after the sleep, before dropping it).
///
/// RED at 6482ea99 (observed, using this same instrumentation added at M2):
/// true high-water 12 vs a configured bound of 8 (batch=4, prefetch=2) —
/// `stream_next_chunk` dropped the permit the INSTANT it received a chunk off
/// the channel, before the caller ever touched it, so the reader could
/// reserve and decode a WHOLE EXTRA chunk's worth of rows beyond the bound
/// while the consumer still held the previous one.
#[tokio::test(flavor = "multi_thread")]
async fn slow_consumer_residency_stays_within_bound_at_prefetch_two_and_four() {
    for prefetch in [2usize, 4usize] {
        let dir = TempDir::new().unwrap();
        let session = session_with_pairs_source(&dir).await;
        let table = materialize_contrastive_table(&session).await;

        let cfg = StreamConfig { batch: 4, prefetch };
        let bound = cfg.batch * cfg.prefetch;
        let loader = TrainingDataLoader::from_training_set_stream(
            Arc::clone(&session),
            table,
            contrastive_columns(),
            ModelTask::TextEmbedding,
            TrainingFormat::Contrastive,
            cfg,
        )
        .await
        .unwrap();

        let spec = PartitionSpec {
            rank: 0,
            world: 1,
            batch: cfg.batch,
            rule: PartitionRule::BlockByGlobalBatch,
        };
        let true_high_water = tokio::task::spawn_blocking(move || {
            let mut step = 0usize;
            let mut true_high_water = 0usize;
            loop {
                let chunk = loader.text_chunk_for_rank(&spec, step).unwrap();
                if chunk.row_count() == 0 {
                    break;
                }
                // ADVERSARIAL: production's consumer runs a model
                // forward+backward per chunk — orders of magnitude slower
                // than the reader. Held for the sleep's whole duration
                // (`chunk` is not dropped until the end of this iteration),
                // exactly as the trainer's loop holds its own chunk across
                // `encode_chunk` + `compute_loss` + the optimizer step.
                std::thread::sleep(std::time::Duration::from_millis(50));
                let produced = loader
                    .stream_chunks_produced()
                    .expect("a live Stream loader reports its produced-chunk count");
                // `step` chunks are already fully finished (dropped) before
                // this one; this one (`chunk`, still held) and everything the
                // reader produced beyond it are simultaneously resident.
                let true_resident = produced.saturating_sub(step) * cfg.batch;
                true_high_water = true_high_water.max(true_resident);
                drop(chunk);
                step += 1;
            }
            true_high_water
        })
        .await
        .unwrap();

        assert!(
            true_high_water <= bound,
            "prefetch={prefetch}: TRUE resident-row high-water mark {true_high_water} exceeded \
             the configured bound {bound}"
        );
        assert!(
            true_high_water > 0,
            "prefetch={prefetch}: the run must have held some rows"
        );
    }
}

/// M2's liveness half of the "bounded AND live" property: holding chunk `k`
/// (never dropping it) while asking for chunk `k+1` must NOT hang. At
/// `prefetch = 2` the bound (`batch * 2`) covers exactly one held chunk plus
/// one more being assembled — `k+1`'s reservation fits without waiting on
/// `k`'s release. Bounded by a wall-clock timeout so a real deadlock in this
/// probe fails loudly (a hung `.await` with no timeout would instead hang the
/// whole test binary).
#[tokio::test(flavor = "multi_thread")]
async fn holding_chunk_k_while_asking_for_k_plus_1_does_not_hang() {
    for prefetch in [2usize, 4usize] {
        let dir = TempDir::new().unwrap();
        let session = session_with_pairs_source(&dir).await;
        let table = materialize_contrastive_table(&session).await;

        let cfg = StreamConfig { batch: 4, prefetch };
        let loader = TrainingDataLoader::from_training_set_stream(
            Arc::clone(&session),
            table,
            contrastive_columns(),
            ModelTask::TextEmbedding,
            TrainingFormat::Contrastive,
            cfg,
        )
        .await
        .unwrap();
        let spec = PartitionSpec {
            rank: 0,
            world: 1,
            batch: cfg.batch,
            rule: PartitionRule::BlockByGlobalBatch,
        };

        let probe = tokio::task::spawn_blocking(move || {
            let chunk_0 = loader.text_chunk_for_rank(&spec, 0).unwrap();
            assert!(
                chunk_0.row_count() > 0,
                "test setup: step 0 must be non-empty"
            );
            // `chunk_0` stays alive (NOT dropped) across this next call —
            // the adversarial hold.
            let chunk_1 = loader.text_chunk_for_rank(&spec, 1).unwrap();
            drop(chunk_0);
            drop(chunk_1);
        });
        let result = tokio::time::timeout(std::time::Duration::from_secs(10), probe).await;
        assert!(
            result.is_ok(),
            "prefetch={prefetch}: holding chunk k while asking for k+1 must not hang"
        );
        result.unwrap().unwrap();
    }
}
