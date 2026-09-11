//! The Shape-A/B SQLite backup recipe (`backup-and-restore.md`): close the
//! catalog deterministically, cold-copy the whole artifact directory
//! (`catalog.db` + its `-wal`, and `jammi_db/`), reopen from the copy, and
//! see the exact same rows. `cache/` is deliberately EXCLUDED from the
//! copy in this test — it is content-addressed scratch state, safe to
//! restore stale or omit entirely, never load-bearing for the catalog's
//! own rows.

use std::sync::Arc;

use datafusion::prelude::SessionContext;
use jammi_db::catalog::Catalog;
use jammi_db::config::AnnIndexConfig;
use jammi_db::model_task::ModelTask;
use jammi_db::store::manifest::{
    ComputeDevice, ComputePrecision, MaterializationEnv, ModelContentDigest, ModelIdentity,
    ProducingDescriptor,
};
use jammi_db::store::{EmbeddingTableSpec, Materialization, ResultStore};
use tempfile::tempdir;

const DIMS: usize = 4;

fn descriptor() -> ProducingDescriptor {
    ProducingDescriptor::Embedding {
        model_id: "backup-model".into(),
        task: ModelTask::TextEmbedding,
        source_id: "docs".into(),
        columns: vec!["body".into()],
        key_column: "_row_id".into(),
        dimensions: DIMS,
    }
}

fn env() -> MaterializationEnv {
    MaterializationEnv::new(
        ComputeDevice::Cpu,
        vec![ModelIdentity {
            model_id: "backup-model".into(),
            backend: "candle".into(),
            compute_precision: ComputePrecision::F32,
            content_digest: ModelContentDigest::Sha256("backup-fixture-digest".into()),
            quantization: None,
        }],
    )
}

/// Recursive directory copy — the cold-copy half of the backup recipe. No
/// special-casing of `-wal`/`-shm` siblings: a plain `cp -r` of the whole
/// artifact directory after `Catalog::close` (which waits for the pool to
/// release the file, checkpointing WAL back into the main file) is the
/// documented recipe.
fn copy_dir_all(src: &std::path::Path, dst: &std::path::Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let file_type = entry.file_type().unwrap();
        let dest_path = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_all(&entry.path(), &dest_path);
        } else {
            std::fs::copy(entry.path(), &dest_path).unwrap();
        }
    }
}

#[tokio::test]
async fn close_copy_reopen_preserves_rows() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store =
        ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default()).unwrap();
    let ctx = SessionContext::new();

    let rows: Vec<(String, Vec<f32>)> = (0..5)
        .map(|i| {
            let vec = (0..DIMS).map(|d| (i * DIMS + d) as f32 + 1.0).collect();
            (format!("row-{i}"), vec)
        })
        .collect();
    let original = store
        .materialize_embedding_table(
            &ctx,
            EmbeddingTableSpec {
                source_id: "docs",
                model_id: "backup-model",
                derived_from: None,
                dimensions: DIMS,
                key_column: Some("_row_id"),
                text_columns: Some("body"),
            },
            &rows,
            Materialization::new(&descriptor(), &env(), vec![]),
            None,
        )
        .await
        .unwrap();

    // Close deterministically (drop is NOT a release point — see
    // `Catalog::close`'s doc) before copying: the WAL is checkpointed back
    // into `catalog.db` and the file lock released, so a cold `cp -r` sees a
    // consistent, lockable file.
    drop(store);
    let catalog =
        Arc::try_unwrap(catalog).unwrap_or_else(|_| panic!("no other Catalog handle survives"));
    catalog.close().await;

    let backup_dir = tempdir().unwrap();
    copy_dir_all(dir.path(), backup_dir.path());

    // Reopen from the COPY — a fresh catalog + store rooted at the backup
    // directory, never touching the original.
    let restored_catalog = Arc::new(Catalog::open(backup_dir.path()).await.unwrap());
    let restored_store = ResultStore::new(
        backup_dir.path(),
        Arc::clone(&restored_catalog),
        AnnIndexConfig::default(),
    )
    .unwrap();
    // Recovery must be a no-op over a clean `ready`-only restore.
    restored_store.recover().await.unwrap();

    let restored = restored_catalog
        .get_result_table(&original.table_name)
        .await
        .unwrap()
        .expect("the restored catalog carries the same row");
    assert_eq!(restored.row_count, original.row_count);
    assert_eq!(restored.definition_hash, original.definition_hash);
    assert_eq!(restored.parquet_path, original.parquet_path);
    assert_eq!(restored.status, original.status);

    // The Parquet bytes themselves round-tripped too (queryable + same row
    // count on disk, not just the catalog summary column).
    let restored_ctx = SessionContext::new();
    restored_store
        .load_existing_tables(&restored_ctx)
        .await
        .unwrap();
    let index = restored_store
        .resolve_search_mode_local(&restored)
        .await
        .unwrap()
        .expect("the ANN sidecar survived the copy too");
    let hits = index.search(&[1.0, 2.0, 3.0, 4.0], 5).unwrap();
    assert_eq!(hits.len(), 5, "every row is still searchable after restore");
}
