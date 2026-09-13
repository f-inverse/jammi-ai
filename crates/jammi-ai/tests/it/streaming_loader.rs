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
                match &chunk {
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
