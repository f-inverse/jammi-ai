//! End-to-end migration tests. Asserts via the `applied_migrations`
//! ledger and direct sqlx queries against the on-disk catalog.
//!
//! The two `concurrent_migrate_on_fresh_*_is_safe` tests race two independent
//! backends' `migrate()` on one FRESH catalog: on SQLite the backend's
//! `BEGIN IMMEDIATE` serialises them; the Postgres arm is the oracle for the
//! advisory lock `catalog::migrations::run` takes.

use std::collections::BTreeSet;
use std::sync::Arc;

use jammi_db::catalog::backend::{BackendError, BackendImpl, CatalogBackend, TxOptions};
use jammi_db::catalog::backend_sqlite::SqliteBackend;
use jammi_db::catalog::Catalog;
use tempfile::tempdir;
use tokio::sync::Barrier;

/// Every migration name, in ledger order. Mirrors `catalog::migrations::MIGRATIONS`
/// (append-only) -- a new migration is added here
/// in the same change.
const EXPECTED_MIGRATION_NAMES: &[&str] = &[
    "001_core_tables",
    "002_result_tables",
    "003_eval_columns",
    "004_drop_embedding_sets",
    "005_tenant_scope",
    "006_channel_columns",
    "007_mutable_tables",
    "008_mutable_order_column",
    "009_topics",
    "010_rename_source_type_local_to_file",
    "011_eval_per_query",
    "012_topics_tenant_unique",
    "013_result_table_kind",
    "014_bm25_channel",
    "015_fine_tune_job_queue",
    "016_rename_training_jobs",
    "017_model_artifact_path_column",
    "018_eval_runs_model_id_nullable",
    "019_normalize_model_status",
    "020_channel_tenant_scope",
    "021_materialization_contract",
    "022_definition_hash_index",
    "023_storage_precision",
    "024_claim_policy",
    "025_index_segments",
    "026_acceleration_report",
    "027_result_table_lease",
    "028_topics_next_offset",
    "029_jobs_instances_workers",
    "030_jobs_idempotency_key",
    "031_jobs_releases_workers_state",
    "032_result_table_versions",
    "033_model_materialization",
    "034_jobs_training_set_identity",
    "035_instances_peer_addr_result_root",
    "036_instances_result_root_identity",
    "037_jobs_assembly_failures_next_after",
    "038_compute_cluster_state",
    "039_canonical_stamps",
];

async fn open_sqlite_backend(path: &std::path::Path) -> std::sync::Arc<SqliteBackend> {
    SqliteBackend::open(path)
        .await
        .expect("open sqlite backend")
}

#[tokio::test]
async fn migration_005_adds_tenant_id_to_every_table() {
    let dir = tempdir().unwrap();
    let _catalog = Catalog::open(dir.path()).await.unwrap();

    let backend = open_sqlite_backend(&dir.path().join("catalog.db")).await;
    let backend = BackendImpl::Sqlite(backend);
    // `training_jobs` carried a `tenant_id` column too (migration 005), but
    // migration 029 later drops the table entirely — its post-005 shape is
    // not observable after a full open, so it is not in this list; see
    // `migration_029_creates_jobs_instances_workers_and_drops_training_jobs`
    // for the current-schema equivalent (`jobs` also carries `tenant_id`).
    for table in [
        "sources",
        "models",
        "jobs",
        "eval_runs",
        "result_tables",
        "evidence_channels",
    ] {
        let sql = format!("SELECT name FROM pragma_table_info('{table}')");
        let columns = backend
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    let sql = sql.clone();
                    Box::pin(async move {
                        tx.query::<_, String>(&sql, &[], |row| row.get("name"))
                            .await
                    })
                },
            )
            .await
            .unwrap();
        assert!(
            columns.iter().any(|c| c == "tenant_id"),
            "table '{table}' must have a tenant_id column after migration 005; \
             got columns: {columns:?}"
        );
    }
}

#[tokio::test]
async fn migration_005_creates_tenant_index_per_table() {
    let dir = tempdir().unwrap();
    let _catalog = Catalog::open(dir.path()).await.unwrap();

    let backend = BackendImpl::Sqlite(open_sqlite_backend(&dir.path().join("catalog.db")).await);
    // `jobs` carries `tenant_id` (migration 029) but no dedicated tenant
    // index of its own — `idx_jobs_claim`/`idx_jobs_lease` already cover its
    // read paths — so it is not in this list (unlike its `training_jobs`
    // ancestor, migration 005, which is gone after migration 029 anyway).
    for (table, idx) in [
        ("sources", "idx_sources_tenant"),
        ("models", "idx_models_tenant"),
        ("eval_runs", "idx_eval_runs_tenant"),
        ("result_tables", "idx_result_tables_tenant"),
        ("evidence_channels", "idx_evidence_channels_tenant"),
    ] {
        let exists = backend
            .transaction(TxOptions { read_only: true, ..Default::default() }, |tx| {
                Box::pin(async move {
                    let rows: Vec<i64> = tx
                        .query(
                            "SELECT 1 AS one FROM sqlite_master WHERE type='index' AND name=$1 AND tbl_name=$2",
                            &[
                                jammi_db::catalog::backend::SqlValue::TextOwned(idx.into()),
                                jammi_db::catalog::backend::SqlValue::TextOwned(table.into()),
                            ],
                            |row| row.get::<i64>("one"),
                        )
                        .await?;
                    Ok(!rows.is_empty())
                })
            })
            .await
            .unwrap();
        assert!(
            exists,
            "index '{idx}' on '{table}' must exist after migration 005"
        );
    }
}

#[tokio::test]
async fn migration_005_back_fills_existing_rows_to_null() {
    let dir = tempdir().unwrap();
    let _catalog = Catalog::open(dir.path()).await.unwrap();
    let backend = BackendImpl::Sqlite(open_sqlite_backend(&dir.path().join("catalog.db")).await);

    let nulls = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    let rows: Vec<i64> = tx
                        .query(
                            "SELECT COUNT(*) AS c FROM evidence_channels WHERE tenant_id IS NULL",
                            &[],
                            |row| row.get::<i64>("c"),
                        )
                        .await?;
                    Ok(rows.first().copied().unwrap_or(0))
                })
            },
        )
        .await
        .unwrap();
    assert_eq!(
        nulls, 3,
        "the three seeded evidence_channels rows (vector, inference, bm25) must have tenant_id NULL"
    );
}

#[tokio::test]
async fn migrations_are_idempotent_across_reopens() {
    let dir = tempdir().unwrap();
    let c1 = Catalog::open(dir.path()).await.unwrap();
    drop(c1);
    let _c2 = Catalog::open(dir.path()).await.unwrap();
}

#[tokio::test]
async fn applied_migrations_ledger_records_all_migrations() {
    let dir = tempdir().unwrap();
    let _catalog = Catalog::open(dir.path()).await.unwrap();
    let backend = BackendImpl::Sqlite(open_sqlite_backend(&dir.path().join("catalog.db")).await);

    let names = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query::<_, String>(
                        "SELECT name FROM applied_migrations ORDER BY name",
                        &[],
                        |row| row.get("name"),
                    )
                    .await
                })
            },
        )
        .await
        .unwrap();
    assert_eq!(names, EXPECTED_MIGRATION_NAMES);
}

/// Migration 019 normalizes a stray `models.status = 'available'` (the frozen
/// migration-001 DDL default, which the typed `ModelStatus` enum never names) to
/// `'registered'`. The test inserts a legacy `'available'` row, runs the exact
/// migration SQL the runner executes for 019, and verifies the rewrite — while a
/// canonical `'registered'` row is left untouched so re-running is a no-op.
#[tokio::test]
async fn migration_019_normalizes_available_status_to_registered() {
    use jammi_db::catalog::backend::SqlValue;

    let dir = tempdir().unwrap();
    let _catalog = Catalog::open(dir.path()).await.unwrap();
    let backend = BackendImpl::Sqlite(open_sqlite_backend(&dir.path().join("catalog.db")).await);

    // Seed one legacy 'available' row and one canonical 'registered' row.
    backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                for (id, status) in [("legacy::1", "available"), ("modern::1", "registered")] {
                    tx.execute(
                        "INSERT INTO models (model_id, name, model_type, task, version, status, created_at, updated_at) \
                         VALUES ($1, $2, 'embedding', 'text-embedding', 1, $3, '2026-01-01T00:00:00.000000Z', '2026-01-01T00:00:00.000000Z')",
                        &[
                            SqlValue::TextOwned(id.into()),
                            SqlValue::TextOwned(id.into()),
                            SqlValue::TextOwned(status.into()),
                        ],
                    )
                    .await?;
                }
                Ok(())
            })
        })
        .await
        .unwrap();

    // Manually invoke the same SQL the migration runner executes for 019 — the
    // runner records 019 as applied on the first open, before the legacy row
    // existed, so we apply it directly (mirroring the migration-010 test).
    backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE models SET status = 'registered' WHERE status = 'available'",
                    &[],
                )
                .await?;
                Ok(())
            })
        })
        .await
        .unwrap();

    let rows = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query::<_, (String, String)>(
                        "SELECT model_id, status FROM models ORDER BY model_id",
                        &[],
                        |row| Ok((row.get("model_id")?, row.get("status")?)),
                    )
                    .await
                })
            },
        )
        .await
        .unwrap();
    assert_eq!(
        rows,
        vec![
            ("legacy::1".to_string(), "registered".to_string()),
            ("modern::1".to_string(), "registered".to_string()),
        ],
        "migration 019 rewrites 'available' to 'registered' and leaves 'registered' untouched"
    );
}

/// Migration 020 tenant-qualifies the evidence-channel identity. Asserts that
/// `evidence_channel_columns` gained a `tenant_id` column, that the seed columns
/// survived the table rebuild with `tenant_id IS NULL`, that the new per-tenant
/// `UNIQUE (tenant_id, channel_name)` constraint on `evidence_channels` rejects
/// a same-tenant duplicate while admitting the same name under a different
/// tenant, and that the partial unique index
/// `idx_evidence_channels_global_name` exists and atomically rejects a duplicate
/// *global* (`tenant_id IS NULL`) channel name at the DB level. The concurrent
/// Postgres race is closed by that DB-level index, so a serial duplicate insert
/// is sufficient to pin the constraint here — no flaky concurrency test needed.
#[tokio::test]
async fn migration_020_tenant_qualifies_channels() {
    let dir = tempdir().unwrap();
    let _catalog = Catalog::open(dir.path()).await.unwrap();
    let backend = BackendImpl::Sqlite(open_sqlite_backend(&dir.path().join("catalog.db")).await);

    // `evidence_channel_columns` now carries `tenant_id`.
    let cols = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query::<_, String>(
                        "SELECT name FROM pragma_table_info('evidence_channel_columns')",
                        &[],
                        |row| row.get("name"),
                    )
                    .await
                })
            },
        )
        .await
        .unwrap();
    assert!(
        cols.iter().any(|c| c == "tenant_id"),
        "evidence_channel_columns must have a tenant_id column after migration 020; got {cols:?}"
    );

    // The seed columns survived the rebuild and are global (tenant_id IS NULL).
    let null_seed_count = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query::<_, i64>(
                        "SELECT COUNT(*) AS n FROM evidence_channel_columns \
                         WHERE tenant_id IS NULL AND channel_name = 'vector'",
                        &[],
                        |row| row.get("n"),
                    )
                    .await
                })
            },
        )
        .await
        .unwrap();
    assert_eq!(
        null_seed_count.first().copied(),
        Some(1),
        "the 'vector' seed column must survive as a global (tenant_id IS NULL) row"
    );

    // The per-tenant UNIQUE constraint admits the same channel name under two
    // different tenants.
    let two_tenants = backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "INSERT INTO evidence_channels (tenant_id, channel_name, priority) \
                     VALUES ('a', 'dup', 1)",
                    &[],
                )
                .await?;
                tx.execute(
                    "INSERT INTO evidence_channels (tenant_id, channel_name, priority) \
                     VALUES ('b', 'dup', 1)",
                    &[],
                )
                .await?;
                Ok(())
            })
        })
        .await;
    assert!(
        two_tenants.is_ok(),
        "two tenants must be able to register the same channel name: {two_tenants:?}"
    );

    // ...but rejects a same-tenant duplicate.
    let dup = backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "INSERT INTO evidence_channels (tenant_id, channel_name, priority) \
                     VALUES ('a', 'dup', 2)",
                    &[],
                )
                .await?;
                Ok(())
            })
        })
        .await;
    assert!(
        matches!(
            dup,
            Err(jammi_db::catalog::backend::BackendError::Constraint { .. })
        ),
        "a same-tenant duplicate channel name must violate UNIQUE (tenant_id, channel_name); \
         got {dup:?}"
    );

    // The partial unique index that enforces global-channel-name uniqueness
    // exists in the post-migration schema.
    let global_index = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query::<_, String>(
                        "SELECT name FROM sqlite_master \
                         WHERE type = 'index' AND name = 'idx_evidence_channels_global_name'",
                        &[],
                        |row| row.get("name"),
                    )
                    .await
                })
            },
        )
        .await
        .unwrap();
    assert_eq!(
        global_index.first().map(String::as_str),
        Some("idx_evidence_channels_global_name"),
        "migration 020 must create the partial unique index on the global namespace"
    );

    // Two global (tenant_id IS NULL) inserts of the same channel name: the first
    // commits, the second is rejected by the partial unique index. This closes
    // the global-namespace race the composite UNIQUE (tenant_id, channel_name)
    // cannot — NULLs are distinct in that constraint on both backends. A serial
    // pair is enough to pin the constraint+error path; the concurrent Postgres
    // race is closed by the same DB-level index, so no concurrency test is run.
    let first_global = backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "INSERT INTO evidence_channels (tenant_id, channel_name, priority) \
                     VALUES (NULL, 'global_dup', 1)",
                    &[],
                )
                .await?;
                Ok(())
            })
        })
        .await;
    assert!(
        first_global.is_ok(),
        "the first global channel insert must succeed: {first_global:?}"
    );
    let second_global = backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "INSERT INTO evidence_channels (tenant_id, channel_name, priority) \
                     VALUES (NULL, 'global_dup', 2)",
                    &[],
                )
                .await?;
                Ok(())
            })
        })
        .await;
    assert!(
        matches!(
            second_global,
            Err(jammi_db::catalog::backend::BackendError::Constraint { .. })
        ),
        "a duplicate global channel name must violate the partial unique index; \
         got {second_global:?}"
    );
}

/// Migration 012 rebuilds `topics` so name uniqueness is scoped per tenant
/// (`UNIQUE(name, tenant_id)`) instead of the global `UNIQUE(name)` migration
/// 009 created. After the migration, two different tenants must be able to
/// hold the same topic name; inserting a duplicate `(name, tenant_id)` pair is
/// still rejected. Exercised directly against the rebuilt table.
#[tokio::test]
async fn migration_012_scopes_topic_name_uniqueness_per_tenant() {
    use jammi_db::catalog::backend::SqlValue;

    let dir = tempdir().unwrap();
    let _catalog = Catalog::open(dir.path()).await.unwrap();
    let backend = BackendImpl::Sqlite(open_sqlite_backend(&dir.path().join("catalog.db")).await);

    // Two backing mutable-table rows so the `backing_table` FK is satisfied for
    // the two topic rows we insert (one per tenant, same topic name).
    backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                for id in ["topic_back_a", "topic_back_b"] {
                    tx.execute(
                        "INSERT INTO mutable_tables (id, schema_json, primary_key, backend_kind) \
                         VALUES ($1, '{}', '[]', 'sqlite')",
                        &[SqlValue::TextOwned(id.into())],
                    )
                    .await?;
                }
                Ok(())
            })
        })
        .await
        .unwrap();

    let tenant_a = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a";
    let tenant_b = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8eff";

    // Two tenants registering the same topic name must both succeed.
    backend
        .transaction(TxOptions::default(), |tx| {
            let (a, b) = (tenant_a.to_string(), tenant_b.to_string());
            Box::pin(async move {
                tx.execute(
                    "INSERT INTO topics (topic_id, name, schema_json, tenant_id, backing_table) \
                     VALUES ('t-a', 'jammi.audit.search.v1', '{}', $1, 'topic_back_a')",
                    &[SqlValue::TextOwned(a)],
                )
                .await?;
                tx.execute(
                    "INSERT INTO topics (topic_id, name, schema_json, tenant_id, backing_table) \
                     VALUES ('t-b', 'jammi.audit.search.v1', '{}', $1, 'topic_back_b')",
                    &[SqlValue::TextOwned(b)],
                )
                .await?;
                Ok(())
            })
        })
        .await
        .expect("two tenants may hold the same topic name after migration 012");

    // A duplicate (name, tenant_id) pair is still rejected.
    let dup = backend
        .transaction(TxOptions::default(), |tx| {
            let a = tenant_a.to_string();
            Box::pin(async move {
                // Reuse a fresh backing table so only the (name, tenant) unique
                // — not the backing_table unique — can be the failure cause.
                tx.execute(
                    "INSERT INTO mutable_tables (id, schema_json, primary_key, backend_kind) \
                     VALUES ('topic_back_dup', '{}', '[]', 'sqlite')",
                    &[],
                )
                .await?;
                tx.execute(
                    "INSERT INTO topics (topic_id, name, schema_json, tenant_id, backing_table) \
                     VALUES ('t-dup', 'jammi.audit.search.v1', '{}', $1, 'topic_back_dup')",
                    &[SqlValue::TextOwned(a)],
                )
                .await?;
                Ok(())
            })
        })
        .await;
    assert!(
        dup.is_err(),
        "a duplicate (name, tenant_id) topic row must still be rejected"
    );
}

/// Migration 010 rewrites pre-upgrade `source_type = '"local"'` rows to
/// `'"file"'` so the deserialiser (which has no `#[serde(alias = "local")]`)
/// can read them. The test inserts a legacy row, runs the migration SQL,
/// and verifies the rewrite — exercising the exact statement the runner
/// executes on a real upgrade.
#[tokio::test]
async fn migration_010_rewrites_legacy_local_rows_to_file() {
    use jammi_db::catalog::backend::SqlValue;

    let dir = tempdir().unwrap();
    let _catalog = Catalog::open(dir.path()).await.unwrap();
    let backend = BackendImpl::Sqlite(open_sqlite_backend(&dir.path().join("catalog.db")).await);

    // Insert two rows: one with the legacy "local" encoding (what the
    // pre-rename catalog writes), one with the new "file" encoding (the
    // post-rename canonical form). The migration must rewrite only the
    // first; the second is left untouched so re-running is a no-op.
    backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "INSERT INTO sources (source_id, name, source_type, uri, options) \
                     VALUES ($1, $2, $3, $4, $5)",
                    &[
                        SqlValue::TextOwned("legacy".into()),
                        SqlValue::TextOwned("legacy".into()),
                        SqlValue::TextOwned("\"local\"".into()),
                        SqlValue::TextOwned("file:///legacy.parquet".into()),
                        SqlValue::TextOwned("{}".into()),
                    ],
                )
                .await?;
                tx.execute(
                    "INSERT INTO sources (source_id, name, source_type, uri, options) \
                     VALUES ($1, $2, $3, $4, $5)",
                    &[
                        SqlValue::TextOwned("modern".into()),
                        SqlValue::TextOwned("modern".into()),
                        SqlValue::TextOwned("\"file\"".into()),
                        SqlValue::TextOwned("file:///modern.parquet".into()),
                        SqlValue::TextOwned("{}".into()),
                    ],
                )
                .await?;
                Ok(())
            })
        })
        .await
        .unwrap();

    // Manually invoke the same SQL the migration runner executes for 010.
    // We can't lean on Catalog::open here because the runner records 010
    // as applied on the first open before any legacy row was inserted.
    backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE sources SET source_type = '\"file\"' WHERE source_type = '\"local\"'",
                    &[],
                )
                .await?;
                Ok(())
            })
        })
        .await
        .unwrap();

    let rows = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query::<_, (String, String)>(
                        "SELECT source_id, source_type FROM sources ORDER BY source_id",
                        &[],
                        |row| Ok((row.get("source_id")?, row.get("source_type")?)),
                    )
                    .await
                })
            },
        )
        .await
        .unwrap();
    assert_eq!(
        rows,
        vec![
            ("legacy".to_string(), "\"file\"".to_string()),
            ("modern".to_string(), "\"file\"".to_string()),
        ]
    );
}

/// Migration 021 adds the materialization-contract summary columns
/// (`definition_hash`, `input_anchors_json`) to `result_tables`. Both are
/// nullable so a pre-contract row carries NULL and verifies as an honest
/// `MissingManifest`. Asserted on a fresh SQLite DB via `pragma_table_info`.
#[tokio::test]
async fn migration_021_adds_materialization_summary_columns() {
    let dir = tempdir().unwrap();
    let _catalog = Catalog::open(dir.path()).await.unwrap();
    let backend = BackendImpl::Sqlite(open_sqlite_backend(&dir.path().join("catalog.db")).await);

    let columns = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query::<_, String>(
                        "SELECT name FROM pragma_table_info('result_tables')",
                        &[],
                        |row| row.get("name"),
                    )
                    .await
                })
            },
        )
        .await
        .unwrap();
    for expected in ["definition_hash", "input_anchors_json"] {
        assert!(
            columns.iter().any(|c| c == expected),
            "result_tables must have '{expected}' after migration 021; got {columns:?}"
        );
    }
}

/// Migration 023 adds the sidecar-index `storage_precision` / `oversample`
/// columns to `result_tables`. Both are nullable so a pre-migration row
/// carries NULL and is read back as the honest defaults (`F32`, the config
/// oversample default) rather than a fabricated precision.
#[tokio::test]
async fn migration_023_adds_storage_precision_and_oversample_columns() {
    let dir = tempdir().unwrap();
    let _catalog = Catalog::open(dir.path()).await.unwrap();
    let backend = BackendImpl::Sqlite(open_sqlite_backend(&dir.path().join("catalog.db")).await);

    let columns = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query::<_, String>(
                        "SELECT name FROM pragma_table_info('result_tables')",
                        &[],
                        |row| row.get("name"),
                    )
                    .await
                })
            },
        )
        .await
        .unwrap();
    for expected in ["storage_precision", "oversample"] {
        assert!(
            columns.iter().any(|c| c == expected),
            "result_tables must have '{expected}' after migration 023; got {columns:?}"
        );
    }
}

/// Migration 029 creates `jobs`/`instances`/`workers` and drops
/// `training_jobs` (and its predecessor name, `fine_tune_jobs`) entirely — no
/// shim, no compatibility view. `idx_jobs_claim`, `idx_jobs_lease`, and
/// `idx_instances_seen` all exist on a fresh, fully-migrated catalog.
#[tokio::test]
async fn migration_029_creates_jobs_instances_workers_and_drops_training_jobs() {
    use jammi_db::catalog::backend::SqlValue;

    let dir = tempdir().unwrap();
    let _catalog = Catalog::open(dir.path()).await.unwrap();
    let backend = BackendImpl::Sqlite(open_sqlite_backend(&dir.path().join("catalog.db")).await);

    let object_exists = |kind: &'static str, name: &'static str| {
        let backend = &backend;
        async move {
            backend
                .transaction(
                    TxOptions {
                        read_only: true,
                        ..Default::default()
                    },
                    |tx| {
                        Box::pin(async move {
                            let rows: Vec<i64> = tx
                                .query(
                                    "SELECT 1 AS one FROM sqlite_master WHERE type=$1 AND name=$2",
                                    &[
                                        SqlValue::TextOwned(kind.into()),
                                        SqlValue::TextOwned(name.into()),
                                    ],
                                    |row| row.get::<i64>("one"),
                                )
                                .await?;
                            Ok(!rows.is_empty())
                        })
                    },
                )
                .await
                .unwrap()
        }
    };

    for gone in ["training_jobs", "fine_tune_jobs"] {
        assert!(
            !object_exists("table", gone).await,
            "'{gone}' must be gone after migration 029"
        );
    }
    for table in ["jobs", "instances", "workers"] {
        assert!(
            object_exists("table", table).await,
            "'{table}' must exist after migration 029"
        );
    }
    for idx in ["idx_jobs_claim", "idx_jobs_lease", "idx_instances_seen"] {
        assert!(
            object_exists("index", idx).await,
            "index '{idx}' must exist after migration 029"
        );
    }
}

/// Migration 029 copies every pre-existing `training_jobs` row into `jobs`
/// with `execution = 'queued'`, then drops `training_jobs`. Exercised by
/// manufacturing the exact pre-029 state on a fully-migrated catalog (drop the
/// 029-created tables, recreate `training_jobs` in its final pre-029 shape,
/// seed one row, clear the ledger's `029_jobs_instances_workers` row and
/// every later row that alters the dropped tables) and reopening — the
/// reopen re-runs the REAL migration 029 DDL (never a test-duplicated copy
/// of it), then 030, 031, and 034 (032 and 033 never touch `jobs`), against
/// that manufactured state.
#[tokio::test]
async fn migration_029_copies_training_jobs_rows_into_jobs_as_queued() {
    use jammi_db::catalog::backend::SqlValue;

    let dir = tempdir().unwrap();
    {
        let _catalog = Catalog::open(dir.path()).await.unwrap();
    }
    let backend = BackendImpl::Sqlite(open_sqlite_backend(&dir.path().join("catalog.db")).await);

    backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute("DROP TABLE workers", &[]).await?;
                tx.execute("DROP TABLE instances", &[]).await?;
                tx.execute("DROP TABLE jobs", &[]).await?;
                // Every later migration that ALTERs the dropped `jobs` /
                // `instances` / `workers` tables must replay too (030's
                // idempotency key, 031's `releases` / `workers.state`, 034's
                // `training_set_ref`/`training_set_location`, 035's
                // `instances.peer_addr`/`result_root`, 037's
                // `jobs.assembly_failures`/`next_assembly_after` -- 032 and
                // 033 never touch `jobs`/`instances`/`workers`, so they are
                // left applied), or the reopen would rebuild a 029-shaped
                // table the current `SELECT_COLS` cannot read — the
                // manufactured state is pre-029, so the ledger must say so
                // for everything from 029 onwards that alters
                // `jobs`/`instances`/`workers`. 038 is deliberately left
                // applied despite also `ALTER TABLE workers ADD COLUMN
                // devices`: it bundles that ALTER with two `CREATE TABLE`s
                // (`compute_executors`, `compute_jobs`) this test never
                // drops, and those are non-idempotent — replaying 038 would
                // fail on "table already exists". The reopened `workers`
                // table below is missing `devices` as a result, which this
                // test's `jobs`-only assertions never observe. `039` DOES
                // join this list (replay-idempotence): its `jobs`/
                // `instances` triggers were dropped along with the tables
                // above (`DROP TABLE` drops a table's own triggers in
                // SQLite), so it must replay to reinstall them, or the
                // reopened `jobs`/`instances` would carry NO schema-edge
                // stamp domain at all. Replaying it is safe for the tables
                // this test does NOT drop (`result_tables`, `models`,
                // `compute_executors`, `applied_migrations`): their triggers
                // survive intact, `CREATE TRIGGER IF NOT EXISTS` makes
                // reinstalling them a no-op, and the data-rewrite `UPDATE`s
                // are naturally idempotent (their `WHERE` excludes
                // already-canonical rows).
                tx.execute(
                    "DELETE FROM applied_migrations WHERE name IN ( \
                       '029_jobs_instances_workers', '030_jobs_idempotency_key', \
                       '031_jobs_releases_workers_state', '034_jobs_training_set_identity', \
                       '035_instances_peer_addr_result_root', \
                       '036_instances_result_root_identity', \
                       '037_jobs_assembly_failures_next_after', \
                       '039_canonical_stamps')",
                    &[],
                )
                .await?;
                // The FK target `jobs.model_ref` (migration 029's DDL)
                // requires — the copy's `base_model_id -> model_ref` value
                // must resolve.
                tx.execute(
                    "INSERT INTO models (model_id, name, model_type, task, created_at, updated_at) \
                     VALUES ('base::1', 'base', 'embedding', 'text-embedding', \
                             '2026-01-01T00:00:00.000000Z', '2026-01-01T00:00:00.000000Z')",
                    &[],
                )
                .await?;
                tx.execute(
                    "CREATE TABLE training_jobs ( \
                         job_id TEXT PRIMARY KEY, \
                         base_model_id TEXT NOT NULL, \
                         output_model_id TEXT, \
                         training_source TEXT NOT NULL, \
                         loss_type TEXT NOT NULL, \
                         hyperparams TEXT NOT NULL, \
                         status TEXT NOT NULL DEFAULT 'queued', \
                         metrics TEXT, \
                         created_at TEXT NOT NULL DEFAULT (CAST(CURRENT_TIMESTAMP AS TEXT)), \
                         updated_at TEXT NOT NULL DEFAULT (CAST(CURRENT_TIMESTAMP AS TEXT)), \
                         tenant_id TEXT, \
                         kind TEXT NOT NULL DEFAULT 'fine_tune', \
                         claimed_by TEXT, \
                         lease_expires_at TEXT, \
                         attempts INTEGER NOT NULL DEFAULT 0, \
                         training_spec TEXT, \
                         priority INTEGER NOT NULL DEFAULT 0, \
                         claimable BOOLEAN NOT NULL DEFAULT TRUE, \
                         acceleration_report TEXT \
                     )",
                    &[],
                )
                .await?;
                tx.execute(
                    "INSERT INTO training_jobs \
                     (job_id, base_model_id, output_model_id, training_source, loss_type, \
                      hyperparams, status, kind, training_spec, priority, claimable, \
                      acceleration_report, created_at, updated_at) \
                     VALUES ('legacy-1', 'base::1', 'tuned::1', 'src', 'cosine', '{}', \
                             'queued', 'fine_tune', '{\"lr\":1}', 3, TRUE, \
                             '{\"state\":\"pending\"}', '2024-01-01T00:00:00.000000Z', \
                             '2024-01-01T00:00:00.000000Z')",
                    &[],
                )
                .await?;
                Ok(())
            })
        })
        .await
        .unwrap();

    // The reopen re-runs migration 029 for real (the ledger does not name
    // it) against the manufactured pre-029 state above.
    let reopened = Catalog::open(dir.path()).await.unwrap();
    let job = reopened.get_job("legacy-1").await.unwrap();
    assert_eq!(
        job.execution, "queued",
        "every copied row is execution='queued'"
    );
    assert_eq!(job.status, "queued");
    assert_eq!(job.kind, "fine_tune");
    assert_eq!(job.spec, "{\"lr\":1}");
    assert_eq!(job.model_ref.as_deref(), Some("base::1"));
    assert_eq!(job.output_model_id.as_deref(), Some("tuned::1"));
    assert_eq!(job.model_source, None);
    assert_eq!(job.priority, 3);
    assert!(job.claimable);
    assert_eq!(job.partial_result, None);
    assert_eq!(
        job.acceleration_report.as_deref(),
        Some("{\"state\":\"pending\"}")
    );

    let raw_backend =
        BackendImpl::Sqlite(open_sqlite_backend(&dir.path().join("catalog.db")).await);
    let table_exists = raw_backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    let rows: Vec<i64> = tx
                        .query(
                            "SELECT 1 AS one FROM sqlite_master WHERE type='table' AND name=$1",
                            &[SqlValue::TextOwned("training_jobs".into())],
                            |row| row.get::<i64>("one"),
                        )
                        .await?;
                    Ok(!rows.is_empty())
                })
            },
        )
        .await
        .unwrap();
    assert!(
        !table_exists,
        "training_jobs must be dropped by migration 029"
    );

    // 035 is on the ledger's DELETE list
    // above (it ALTERs `instances`, created fresh by 029's replayed DDL), so
    // an omission from that list would leave the reopened `instances`
    // missing both columns here -- a failure, not a silent pass.
    let instance_columns: Vec<String> = raw_backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query(
                        "SELECT name FROM pragma_table_info('instances') \
                         WHERE name IN ('peer_addr', 'result_root', 'result_root_identity')",
                        &[],
                        |row| row.get::<String>("name"),
                    )
                    .await
                })
            },
        )
        .await
        .unwrap();
    for col in ["peer_addr", "result_root", "result_root_identity"] {
        assert!(
            instance_columns.iter().any(|c| c == col),
            "reopened instances must carry '{col}' (migration 035 replayed): {instance_columns:?}"
        );
    }

    // 037 is on the ledger's DELETE list
    // above (it ALTERs `jobs`, dropped and recreated fresh by 029's
    // replayed DDL), so an omission from that list would leave the
    // reopened `jobs` missing both columns here -- a failure, not a silent pass.
    let job_columns: Vec<String> = raw_backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query(
                        "SELECT name FROM pragma_table_info('jobs') \
                         WHERE name IN ('assembly_failures', 'next_assembly_after')",
                        &[],
                        |row| row.get::<String>("name"),
                    )
                    .await
                })
            },
        )
        .await
        .unwrap();
    for col in ["assembly_failures", "next_assembly_after"] {
        assert!(
            job_columns.iter().any(|c| c == col),
            "reopened jobs must carry '{col}' (migration 037 replayed): {job_columns:?}"
        );
    }
}

/// Names in the `applied_migrations` ledger, sorted, read through `backend`.
async fn ledger_names<B: CatalogBackend + ?Sized>(backend: &B) -> Vec<String> {
    backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query::<_, String>(
                        "SELECT name FROM applied_migrations ORDER BY name",
                        &[],
                        |row| row.get("name"),
                    )
                    .await
                })
            },
        )
        .await
        .expect("read applied_migrations ledger")
}

/// The post-race ledger oracle: exactly `MIGRATIONS.len()` rows, no name
/// twice, and the set is the full migration list -- so two no-op callers on an
/// already-migrated catalog cannot pass vacuously (the freshness assertion in
/// each test is the other half of that control).
fn assert_ledger_complete(names: &[String]) {
    assert_eq!(
        names.len(),
        EXPECTED_MIGRATION_NAMES.len(),
        "applied_migrations must hold exactly one row per migration; got {names:?}"
    );
    let unique: BTreeSet<&str> = names.iter().map(String::as_str).collect();
    assert_eq!(
        unique.len(),
        names.len(),
        "a migration name appears twice in applied_migrations: {names:?}"
    );
    assert_eq!(
        names, EXPECTED_MIGRATION_NAMES,
        "applied_migrations must name every migration exactly once"
    );
}

/// Release two backends' `migrate()` from one barrier and return BOTH results
/// -- a caller asserts on each; nothing here swallows an `Err`.
async fn race_migrate<B: CatalogBackend + 'static>(
    a: Arc<B>,
    b: Arc<B>,
) -> (Result<(), BackendError>, Result<(), BackendError>) {
    let barrier = Arc::new(Barrier::new(2));
    let spawn = |backend: Arc<B>, barrier: Arc<Barrier>| {
        tokio::spawn(async move {
            barrier.wait().await;
            backend.migrate().await
        })
    };
    let task_a = spawn(a, Arc::clone(&barrier));
    let task_b = spawn(b, barrier);
    (
        task_a.await.expect("migrate task A panicked"),
        task_b.await.expect("migrate task B panicked"),
    )
}

/// Two `SqliteBackend`
/// pools in one process race `migrate()` on one fresh `catalog.db`. SQLite's
/// backend opens every write transaction
/// `BEGIN IMMEDIATE` under a 5 s `busy_timeout`, so the second runner waits for
/// the first's commit and then reads a complete ledger. Same assertions as the
/// Postgres arm so the two stay comparable.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_migrate_on_fresh_sqlite_is_safe() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("catalog.db");
    let a = open_sqlite_backend(&path).await;
    let b = open_sqlite_backend(&path).await;

    // Freshness: `open` does not migrate, so the ledger table must be absent.
    let ledger_tables = a
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query::<_, String>(
                        "SELECT name FROM sqlite_master \
                         WHERE type = 'table' AND name = 'applied_migrations'",
                        &[],
                        |row| row.get("name"),
                    )
                    .await
                })
            },
        )
        .await
        .expect("query sqlite_master");
    assert!(
        ledger_tables.is_empty(),
        "the catalog must be fresh before the race; found {ledger_tables:?}"
    );

    let (result_a, result_b) = race_migrate(Arc::clone(&a), Arc::clone(&b)).await;
    assert!(
        result_a.is_ok() && result_b.is_ok(),
        "both concurrent migrate() calls must return Ok on SQLite; \
         A = {result_a:?}, B = {result_b:?}"
    );

    let names = ledger_names(a.as_ref()).await;
    assert_ledger_complete(&names);

    a.close().await;
    b.close().await;
}

/// Env var that arms the migration runner's ledger-read rendezvous; the
/// runner's hook reads the same name (`test_hook::MIGRATION_LEDGER_BARRIER_ENV`).
#[cfg(all(feature = "live-postgres-tests", feature = "test-hooks"))]
use jammi_db::store::mutable::test_hook::MIGRATION_LEDGER_BARRIER_ENV as LEDGER_BARRIER_ENV;
/// Without `test-hooks` the hook does not exist; the variable is set anyway so
/// the test body is one shape, and the race is then whatever the barrier
/// release and two pools produce.
#[cfg(all(feature = "live-postgres-tests", not(feature = "test-hooks")))]
const LEDGER_BARRIER_ENV: &str = "JAMMI_TEST_MIGRATION_LEDGER_BARRIER";

/// Arms the rendezvous for the lifetime of the Postgres test and disarms it
/// on every exit path (the hook is one-shot as well; belt and braces so the
/// SQLite arm in the same binary can never be parked).
#[cfg(feature = "live-postgres-tests")]
struct LedgerBarrierArmed;

#[cfg(feature = "live-postgres-tests")]
impl LedgerBarrierArmed {
    /// Arms `parties` Postgres runners; the selector keeps a SQLite sibling in
    /// the same binary (the arm above) out of the rendezvous.
    fn arm(parties: usize) -> Self {
        std::env::set_var(LEDGER_BARRIER_ENV, format!("postgres:{parties}"));
        Self
    }
}

#[cfg(feature = "live-postgres-tests")]
impl Drop for LedgerBarrierArmed {
    fn drop(&mut self) {
        std::env::remove_var(LEDGER_BARRIER_ENV);
    }
}

/// `url` with its database path replaced by `db`.
#[cfg(feature = "live-postgres-tests")]
fn with_database(url: &str, db: &str) -> String {
    let mut parsed = url::Url::parse(url).expect("JAMMI_TEST_PG_URL parses as a URL");
    parsed.set_path(&format!("/{db}"));
    parsed.to_string()
}

/// Two independent `PostgresBackend`s (separate pools) race `migrate()` on a
/// FRESH database created for this test alone. Both must return `Ok` and the
/// ledger must name every migration exactly once. Without the advisory lock
/// in `catalog::migrations::run` the loser fails with SQLSTATE 42P07
/// (`relation already exists`) or 23505 on the ledger PK.
///
/// With `feature = "test-hooks"` the runner's ledger-read rendezvous is armed
/// so both runners are held between the ledger read and the first DDL for as
/// long as the lock lets them both get there (it lets only one; the other
/// blocks on the lock and passes through afterwards).
#[cfg(feature = "live-postgres-tests")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_migrate_on_fresh_postgres_is_safe() {
    use jammi_db::catalog::backend_postgres::PostgresBackend;

    const TEST_NAME: &str = "concurrent_migrate_on_fresh_postgres_is_safe";
    let admin_url = jammi_test_utils::postgres_url();

    let admin = sqlx::PgPool::connect(&admin_url)
        .await
        .expect("connect to JAMMI_TEST_PG_URL");
    let db_name = format!("jammi_migrate_race_{}", uuid::Uuid::new_v4().simple());
    sqlx::query(&format!("CREATE DATABASE \"{db_name}\""))
        .execute(&admin)
        .await
        .expect("CREATE DATABASE for the fresh-catalog race");
    let fresh_url = with_database(&admin_url, &db_name);

    // The body runs on its own task so the database is dropped whether it
    // passes or panics; the panic is re-raised afterwards, payload intact.
    let outcome = tokio::spawn(async move {
        let _armed = LedgerBarrierArmed::arm(2);

        let a = PostgresBackend::open_with_options(&fresh_url, 4, None)
            .await
            .expect("open PostgresBackend A");
        let b = PostgresBackend::open_with_options(&fresh_url, 4, None)
            .await
            .expect("open PostgresBackend B");

        // Freshness: no ledger table yet.
        let ledger_tables = a
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query::<_, i64>(
                            "SELECT count(*) AS n FROM information_schema.tables \
                             WHERE table_schema = 'public' AND table_name = 'applied_migrations'",
                            &[],
                            |row| row.get("n"),
                        )
                        .await
                    })
                },
            )
            .await
            .expect("query information_schema.tables");
        assert_eq!(
            ledger_tables,
            vec![0],
            "the database must be fresh before the race (no applied_migrations table)"
        );

        let (result_a, result_b) = race_migrate(Arc::clone(&a), Arc::clone(&b)).await;
        assert!(
            result_a.is_ok() && result_b.is_ok(),
            "both concurrent migrate() calls must return Ok on a fresh Postgres; \
             A = {result_a:?}, B = {result_b:?}"
        );

        let names = ledger_names(a.as_ref()).await;
        assert_ledger_complete(&names);

        a.close().await;
        b.close().await;
    })
    .await;

    sqlx::query(&format!("DROP DATABASE \"{db_name}\" WITH (FORCE)"))
        .execute(&admin)
        .await
        .expect("DROP DATABASE after the race");
    admin.close().await;

    if let Err(join) = outcome {
        match join.try_into_panic() {
            Ok(payload) => std::panic::resume_unwind(payload),
            Err(join) => panic!("{TEST_NAME}: body task failed without panicking: {join}"),
        }
    }
}

/// Migration 027 adds the writer-lease columns to `result_tables`:
/// `writer_id` and `lease_expires_at`, both nullable so a row born before the
/// migration reads back as "no writer, no lease" — the absent-lease state
/// recovery reconciles exactly as it always did — plus the
/// `(status, lease_expires_at)` index recovery's expired-lease scan uses.
#[tokio::test]
async fn migration_027_adds_result_table_lease_columns_nullable() {
    let dir = tempfile::tempdir().unwrap();
    let backend = open_sqlite_backend(&dir.path().join("catalog.db")).await;
    backend.migrate().await.unwrap();

    let columns: Vec<String> = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query(
                        "SELECT name, \"notnull\" FROM pragma_table_info('result_tables')",
                        &[],
                        |row| {
                            let name: String = row.get("name")?;
                            let notnull: i32 = row.get("notnull")?;
                            Ok(format!("{name}:{notnull}"))
                        },
                    )
                    .await
                })
            },
        )
        .await
        .unwrap();
    assert!(
        columns.iter().any(|c| c == "writer_id:0"),
        "result_tables must have a nullable 'writer_id' after migration 027; got {columns:?}"
    );
    assert!(
        columns.iter().any(|c| c == "lease_expires_at:0"),
        "result_tables must have a nullable 'lease_expires_at' after migration 027; got {columns:?}"
    );

    let indexes: Vec<String> = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query(
                        "SELECT name FROM sqlite_master WHERE type = 'index' \
                         AND tbl_name = 'result_tables'",
                        &[],
                        |row| row.get("name"),
                    )
                    .await
                })
            },
        )
        .await
        .unwrap();
    assert!(
        indexes.iter().any(|i| i == "idx_result_tables_lease"),
        "migration 027 must create idx_result_tables_lease; got {indexes:?}"
    );
}

/// Migration `031_jobs_releases_workers_state` is present and
/// ordered AFTER `030_jobs_idempotency_key` (relative position, never
/// `.last()`, so a renumber keeps this green), and
/// a fresh catalog carries what it adds: `jobs.releases`, `workers.state`,
/// and the gauge index `idx_jobs_kind_status`.
#[tokio::test]
async fn migration_031_is_ordered_after_030_and_adds_releases_and_workers_state() {
    let position = |name: &str| {
        EXPECTED_MIGRATION_NAMES
            .iter()
            .position(|m| *m == name)
            .unwrap_or_else(|| panic!("{name} missing from EXPECTED_MIGRATION_NAMES"))
    };
    assert!(
        position("031_jobs_releases_workers_state") > position("030_jobs_idempotency_key"),
        "031_jobs_releases_workers_state must follow 030"
    );

    let dir = tempdir().unwrap();
    let _catalog = Catalog::open(dir.path()).await.unwrap();
    let backend = BackendImpl::Sqlite(open_sqlite_backend(&dir.path().join("catalog.db")).await);

    let applied = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query::<_, String>(
                        "SELECT name FROM applied_migrations ORDER BY name",
                        &[],
                        |row| row.get("name"),
                    )
                    .await
                })
            },
        )
        .await
        .unwrap();
    let ledger_position = |name: &str| {
        applied
            .iter()
            .position(|m| m == name)
            .unwrap_or_else(|| panic!("{name} missing from the applied ledger: {applied:?}"))
    };
    assert!(
        ledger_position("031_jobs_releases_workers_state")
            > ledger_position("030_jobs_idempotency_key")
    );

    for (table, column) in [("jobs", "releases"), ("workers", "state")] {
        let sql = format!("SELECT name FROM pragma_table_info('{table}')");
        let columns = backend
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    let sql = sql.clone();
                    Box::pin(async move {
                        tx.query::<_, String>(&sql, &[], |row| row.get("name"))
                            .await
                    })
                },
            )
            .await
            .unwrap();
        assert!(
            columns.iter().any(|c| c == column),
            "{table}.{column} must exist after migration 031; got {columns:?}"
        );
    }
    let index_present = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query::<_, i64>(
                        "SELECT 1 AS one FROM sqlite_master WHERE type='index' \
                         AND name='idx_jobs_kind_status' AND tbl_name='jobs'",
                        &[],
                        |row| row.get("one"),
                    )
                    .await
                })
            },
        )
        .await
        .unwrap();
    assert_eq!(
        index_present.len(),
        1,
        "idx_jobs_kind_status must exist on jobs"
    );
}

/// Migration 032 creates `result_table_versions` (with its lease index),
/// adds the nullable `current_version` and the `NOT NULL DEFAULT 0`
/// `next_version` to `result_tables`, and the nullable `version` stamp to
/// `index_segments` — on both backends.
#[test_case::test_case(jammi_db::catalog::backend::BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(jammi_db::catalog::backend::BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn migration_032_creates_result_table_versions(
    kind: jammi_db::catalog::backend::BackendKind,
) {
    use jammi_db::catalog::backend::BackendKind;
    let dir = tempdir().unwrap();
    let backend = match kind {
        BackendKind::Sqlite => {
            BackendImpl::Sqlite(open_sqlite_backend(&dir.path().join("catalog.db")).await)
        }
        BackendKind::Postgres => {
            let url = jammi_test_utils::postgres_url();
            BackendImpl::Postgres(
                jammi_db::catalog::backend_postgres::PostgresBackend::open_with_options(
                    &url, 4, None,
                )
                .await
                .unwrap(),
            )
        }
    };
    backend.migrate().await.unwrap();

    // The table exists and holds no row for a table this test never created
    // (the Postgres lane shares one database with every other test, so a
    // global emptiness assertion would be a test-ordering fact, not a
    // migration fact).
    let probe = format!("mig032_probe_{}", jammi_test_utils::unique_suffix());
    let count: i64 = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                let probe = probe.clone();
                Box::pin(async move {
                    tx.query_opt(
                        "SELECT count(*) AS c FROM result_table_versions WHERE table_name = $1",
                        &[jammi_db::catalog::backend::SqlValue::TextOwned(probe)],
                        |row| row.get::<i64>("c"),
                    )
                    .await
                    .map(|c| c.unwrap_or(-1))
                })
            },
        )
        .await
        .unwrap();
    assert_eq!(
        count, 0,
        "result_table_versions must exist (and be queryable)"
    );

    // The three ADD COLUMNs: nullable current_version, NOT NULL DEFAULT 0
    // next_version, nullable index_segments.version — read through the
    // backend's own catalog so both dialects answer the same question.
    let columns: Vec<(String, String, bool)> = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    match kind {
                        BackendKind::Sqlite => {
                            let mut out = Vec::new();
                            for table in ["result_tables", "index_segments"] {
                                let rows = tx
                                    .query(
                                        &format!(
                                            "SELECT name, \"notnull\" FROM pragma_table_info('{table}')"
                                        ),
                                        &[],
                                        |row| {
                                            let name: String = row.get("name")?;
                                            let notnull: i32 = row.get("notnull")?;
                                            Ok((table.to_string(), name, notnull == 1))
                                        },
                                    )
                                    .await?;
                                out.extend(rows);
                            }
                            Ok(out)
                        }
                        BackendKind::Postgres => {
                            tx.query(
                                "SELECT table_name, column_name, is_nullable \
                                 FROM information_schema.columns \
                                 WHERE table_name IN ('result_tables', 'index_segments')",
                                &[],
                                |row| {
                                    let table: String = row.get("table_name")?;
                                    let name: String = row.get("column_name")?;
                                    let nullable: String = row.get("is_nullable")?;
                                    Ok((table, name, nullable == "NO"))
                                },
                            )
                            .await
                        }
                    }
                })
            },
        )
        .await
        .unwrap();
    let has = |t: &str, c: &str, notnull: bool| {
        columns
            .iter()
            .any(|(tt, cc, nn)| tt == t && cc == c && *nn == notnull)
    };
    assert!(
        has("result_tables", "current_version", false),
        "result_tables.current_version must be nullable; got {columns:?}"
    );
    assert!(
        has("result_tables", "next_version", true),
        "result_tables.next_version must be NOT NULL; got {columns:?}"
    );
    assert!(
        has("index_segments", "version", false),
        "index_segments.version must be nullable; got {columns:?}"
    );

    // A pre-existing row reads `next_version = 0` through the default.
    let dflt: i64 = backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                let name = format!("mig032_{}", jammi_test_utils::unique_suffix());
                tx.execute(
                    "INSERT INTO result_tables (table_name, source_id, model_id, task, \
                     parquet_path, created_at) VALUES ($1, 's', 'm', 'text_embedding', 'p', '2026-01-01T00:00:00.000000Z')",
                    &[jammi_db::catalog::backend::SqlValue::TextOwned(
                        name.clone(),
                    )],
                )
                .await?;
                let v = tx
                    .query_opt(
                        "SELECT next_version FROM result_tables WHERE table_name = $1",
                        &[jammi_db::catalog::backend::SqlValue::TextOwned(
                            name.clone(),
                        )],
                        |row| row.get::<i32>("next_version"),
                    )
                    .await?
                    .unwrap_or(-1);
                tx.execute(
                    "DELETE FROM result_tables WHERE table_name = $1",
                    &[jammi_db::catalog::backend::SqlValue::TextOwned(name)],
                )
                .await?;
                Ok(i64::from(v))
            })
        })
        .await
        .unwrap();
    assert_eq!(dflt, 0, "next_version defaults to 0");
}

/// Migration `033_model_materialization` is present and ordered
/// AFTER `032_result_table_versions` (relative position, never
/// `.last()`, so a renumber on a later merge keeps this green), adds
/// the two NULLABLE `models` columns (`definition_hash`,
/// `input_anchors_json`), the `idx_models_definition_hash` cache-lookup
/// index, and the `idx_models_artifact_path` reference-guard index — on
/// both backends. `manifest_path` is deliberately absent: the sidecar path
/// stays derived from the artifact prefix rather than recorded.
#[test_case::test_case(jammi_db::catalog::backend::BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(jammi_db::catalog::backend::BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn migration_033_is_ordered_after_032_and_adds_model_materialization_columns(
    kind: jammi_db::catalog::backend::BackendKind,
) {
    use jammi_db::catalog::backend::BackendKind;

    let position = |name: &str| {
        EXPECTED_MIGRATION_NAMES
            .iter()
            .position(|m| *m == name)
            .unwrap_or_else(|| panic!("{name} missing from EXPECTED_MIGRATION_NAMES"))
    };
    assert!(
        position("033_model_materialization") > position("032_result_table_versions"),
        "the model_materialization migration must follow 032"
    );

    let dir = tempdir().unwrap();
    let backend = match kind {
        BackendKind::Sqlite => {
            BackendImpl::Sqlite(open_sqlite_backend(&dir.path().join("catalog.db")).await)
        }
        BackendKind::Postgres => {
            let url = jammi_test_utils::postgres_url();
            BackendImpl::Postgres(
                jammi_db::catalog::backend_postgres::PostgresBackend::open_with_options(
                    &url, 4, None,
                )
                .await
                .unwrap(),
            )
        }
    };
    backend.migrate().await.unwrap();

    let applied = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query::<_, String>(
                        "SELECT name FROM applied_migrations ORDER BY name",
                        &[],
                        |row| row.get("name"),
                    )
                    .await
                })
            },
        )
        .await
        .unwrap();
    let ledger_position = |name: &str| {
        applied
            .iter()
            .position(|m| m == name)
            .unwrap_or_else(|| panic!("{name} missing from the applied ledger: {applied:?}"))
    };
    assert!(
        ledger_position("033_model_materialization") > ledger_position("032_result_table_versions")
    );

    let columns: Vec<(String, bool)> = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    match kind {
                        BackendKind::Sqlite => {
                            tx.query(
                                "SELECT name, \"notnull\" FROM pragma_table_info('models')",
                                &[],
                                |row| {
                                    let name: String = row.get("name")?;
                                    let notnull: i32 = row.get("notnull")?;
                                    Ok((name, notnull == 1))
                                },
                            )
                            .await
                        }
                        BackendKind::Postgres => {
                            tx.query(
                                "SELECT column_name, is_nullable \
                                 FROM information_schema.columns WHERE table_name = 'models'",
                                &[],
                                |row| {
                                    let name: String = row.get("column_name")?;
                                    let nullable: String = row.get("is_nullable")?;
                                    Ok((name, nullable == "NO"))
                                },
                            )
                            .await
                        }
                    }
                })
            },
        )
        .await
        .unwrap();
    for expected in ["definition_hash", "input_anchors_json"] {
        assert!(
            columns.iter().any(|(c, notnull)| c == expected && !notnull),
            "models.{expected} must exist and be nullable after migration 033; got {columns:?}"
        );
    }
    assert!(
        columns.iter().all(|(c, _)| c != "manifest_path"),
        "manifest_path must never exist on models (the sidecar path is derived from the \
         artifact prefix, never recorded); got {columns:?}"
    );

    for index_name in ["idx_models_definition_hash", "idx_models_artifact_path"] {
        let index_present = backend
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        match kind {
                            BackendKind::Sqlite => {
                                tx.query::<_, i64>(
                                    &format!(
                                        "SELECT 1 AS one FROM sqlite_master WHERE type='index' \
                                         AND name='{index_name}' AND tbl_name='models'"
                                    ),
                                    &[],
                                    |row| row.get("one"),
                                )
                                .await
                            }
                            BackendKind::Postgres => {
                                tx.query::<_, i64>(
                                    &format!(
                                        "SELECT 1::bigint AS one FROM pg_indexes \
                                         WHERE tablename = 'models' AND indexname = '{index_name}'"
                                    ),
                                    &[],
                                    |row| row.get("one"),
                                )
                                .await
                            }
                        }
                    })
                },
            )
            .await
            .unwrap();
        assert_eq!(index_present.len(), 1, "{index_name} must exist on models");
    }

    // A pre-migration-shaped row (NULL definition_hash) is never matched by
    // an equality probe -- the SQL-level half of "NULL never matches" the
    // model_repo unit test exercises through `probe_model_by_definition`.
    let name = format!("mig033_{}", jammi_test_utils::unique_suffix());
    let pk = name.clone();
    backend
        .transaction(TxOptions::default(), |tx| {
            let pk = pk.clone();
            Box::pin(async move {
                tx.execute(
                    "INSERT INTO models (model_id, name, model_type, task, version, created_at, updated_at) \
                     VALUES ($1, $1, 'lora', 'text_embedding', 1, \
                             '2026-01-01T00:00:00.000000Z', '2026-01-01T00:00:00.000000Z')",
                    &[jammi_db::catalog::backend::SqlValue::TextOwned(pk)],
                )
                .await
            })
        })
        .await
        .unwrap();
    let matches: i64 = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                let name = name.clone();
                Box::pin(async move {
                    tx.query_opt(
                        "SELECT count(*) AS c FROM models WHERE model_id = $1 \
                         AND definition_hash = $2",
                        &[
                            jammi_db::catalog::backend::SqlValue::TextOwned(name),
                            jammi_db::catalog::backend::SqlValue::Text("anything"),
                        ],
                        |row| row.get::<i64>("c"),
                    )
                    .await
                    .map(|c| c.unwrap_or(-1))
                })
            },
        )
        .await
        .unwrap();
    assert_eq!(
        matches, 0,
        "a NULL definition_hash must never equality-match a probed hash"
    );
}

/// Migration `034_jobs_training_set_identity`
/// is present, ordered AFTER `033_model_materialization` (relative
/// position, never `.last()`), adds `jobs.training_set_ref` /
/// `jobs.training_set_location` as nullable `TEXT` columns, and pins the
/// pair's stop rule ("never one column without the other in the same
/// statement") at the SCHEMA edge: a raw single-column write is
/// refused by the `CHECK` constraint itself, on both backends — never left
/// to "the only writer is the CAS" as the sole guarantee.
#[test_case::test_case(jammi_db::catalog::backend::BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(jammi_db::catalog::backend::BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn migration_034_is_ordered_after_033_and_pins_the_pair_at_the_schema_edge(
    kind: jammi_db::catalog::backend::BackendKind,
) {
    use jammi_db::catalog::backend::{BackendKind, SqlValue};

    let position = |name: &str| {
        EXPECTED_MIGRATION_NAMES
            .iter()
            .position(|m| *m == name)
            .unwrap_or_else(|| panic!("{name} missing from EXPECTED_MIGRATION_NAMES"))
    };
    assert!(
        position("034_jobs_training_set_identity") > position("033_model_materialization"),
        "the training-set identity migration must follow 033"
    );

    let dir = tempdir().unwrap();
    let backend = match kind {
        BackendKind::Sqlite => {
            BackendImpl::Sqlite(open_sqlite_backend(&dir.path().join("catalog.db")).await)
        }
        BackendKind::Postgres => {
            let url = jammi_test_utils::postgres_url();
            BackendImpl::Postgres(
                jammi_db::catalog::backend_postgres::PostgresBackend::open_with_options(
                    &url, 4, None,
                )
                .await
                .unwrap(),
            )
        }
    };
    backend.migrate().await.unwrap();

    // Both columns exist and are nullable, on both dialects.
    let columns: Vec<(String, bool)> = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    match kind {
                        BackendKind::Sqlite => {
                            tx.query(
                                "SELECT name, \"notnull\" FROM pragma_table_info('jobs') \
                                 WHERE name IN ('training_set_ref', 'training_set_location')",
                                &[],
                                |row| {
                                    let name: String = row.get("name")?;
                                    let notnull: i32 = row.get("notnull")?;
                                    Ok((name, notnull == 1))
                                },
                            )
                            .await
                        }
                        BackendKind::Postgres => {
                            tx.query(
                                "SELECT column_name, is_nullable FROM information_schema.columns \
                                 WHERE table_name = 'jobs' \
                                   AND column_name IN ('training_set_ref', 'training_set_location')",
                                &[],
                                |row| {
                                    let name: String = row.get("column_name")?;
                                    let nullable: String = row.get("is_nullable")?;
                                    Ok((name, nullable == "NO"))
                                },
                            )
                            .await
                        }
                    }
                })
            },
        )
        .await
        .unwrap();
    for col in ["training_set_ref", "training_set_location"] {
        assert!(
            columns.iter().any(|(c, notnull)| c == col && !notnull),
            "jobs.{col} must be a nullable TEXT column; got {columns:?}"
        );
    }

    let job_id = format!("mig033_{}", jammi_test_utils::unique_suffix());
    let insert_bare = format!(
        "INSERT INTO jobs (job_id, kind, execution, spec, created_at, updated_at) \
         VALUES ('{job_id}', 'k', 'queued', '{{}}', '2026-01-01T00:00:00.000000Z', '2026-01-01T00:00:00.000000Z')"
    );

    // Both NULL succeeds — the default, every existing and every `world_size
    // == 1` row.
    backend
        .transaction(TxOptions::default(), |tx| {
            let sql = insert_bare.clone();
            Box::pin(async move { tx.execute(&sql, &[]).await })
        })
        .await
        .expect("both columns NULL must be a valid row");

    // Both SET together succeeds.
    let job_id2 = format!("{job_id}_paired");
    backend
        .transaction(TxOptions::default(), |tx| {
            let job_id2 = job_id2.clone();
            Box::pin(async move {
                tx.execute(
                    "INSERT INTO jobs (job_id, kind, execution, spec, created_at, updated_at, \
                         training_set_ref, training_set_location) \
                     VALUES ($1, 'k', 'queued', '{}', '2026-01-01T00:00:00.000000Z', '2026-01-01T00:00:00.000000Z', 'digest-x', 'table-y')",
                    &[SqlValue::TextOwned(job_id2)],
                )
                .await
            })
        })
        .await
        .expect("both columns set together must be a valid row");

    // A raw write of ONE column, leaving the other NULL, is refused by the
    // schema itself — the mutation this test proves: drop the `CHECK` clause
    // from migration 033 and this exact statement stops erroring.
    let one_col_err = backend
        .transaction(TxOptions::default(), |tx| {
            let job_id = job_id.clone();
            Box::pin(async move {
                tx.execute(
                    "UPDATE jobs SET training_set_ref = $1 WHERE job_id = $2",
                    &[
                        SqlValue::TextOwned("digest-only".to_string()),
                        SqlValue::TextOwned(job_id),
                    ],
                )
                .await
            })
        })
        .await;
    assert!(
        one_col_err.is_err(),
        "setting training_set_ref alone (training_set_location left NULL) must be refused \
         by the schema CHECK constraint, never silently accepted"
    );

    // ... and the symmetric case: the OTHER column alone.
    let job_id3 = format!("{job_id}_other_alone");
    backend
        .transaction(TxOptions::default(), |tx| {
            let sql = insert_bare.replace(&job_id, &job_id3);
            Box::pin(async move { tx.execute(&sql, &[]).await })
        })
        .await
        .expect("fixture row for the symmetric case");
    let other_col_err = backend
        .transaction(TxOptions::default(), |tx| {
            let job_id3 = job_id3.clone();
            Box::pin(async move {
                tx.execute(
                    "UPDATE jobs SET training_set_location = $1 WHERE job_id = $2",
                    &[
                        SqlValue::TextOwned("location-only".to_string()),
                        SqlValue::TextOwned(job_id3),
                    ],
                )
                .await
            })
        })
        .await;
    assert!(
        other_col_err.is_err(),
        "setting training_set_location alone (training_set_ref left NULL) must be refused \
         by the schema CHECK constraint, never silently accepted"
    );
}

/// Migration `035_instances_peer_addr_result_root` is present,
/// ordered AFTER `034_jobs_training_set_identity` (relative position,
/// never `.last()`), and adds `instances.peer_addr` / `instances.result_root`
/// as nullable `TEXT` columns, on both backends.
#[test_case::test_case(jammi_db::catalog::backend::BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(jammi_db::catalog::backend::BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn migration_035_is_ordered_after_034_and_adds_instances_peer_addr_result_root(
    kind: jammi_db::catalog::backend::BackendKind,
) {
    use jammi_db::catalog::backend::{BackendKind, SqlValue};

    let position = |name: &str| {
        EXPECTED_MIGRATION_NAMES
            .iter()
            .position(|m| *m == name)
            .unwrap_or_else(|| panic!("{name} missing from EXPECTED_MIGRATION_NAMES"))
    };
    assert!(
        position("035_instances_peer_addr_result_root")
            > position("034_jobs_training_set_identity"),
        "the instances peer_addr/result_root migration must follow 034"
    );

    let dir = tempdir().unwrap();
    let backend = match kind {
        BackendKind::Sqlite => {
            BackendImpl::Sqlite(open_sqlite_backend(&dir.path().join("catalog.db")).await)
        }
        BackendKind::Postgres => {
            let url = jammi_test_utils::postgres_url();
            BackendImpl::Postgres(
                jammi_db::catalog::backend_postgres::PostgresBackend::open_with_options(
                    &url, 4, None,
                )
                .await
                .unwrap(),
            )
        }
    };
    backend.migrate().await.unwrap();

    // Both columns exist and are nullable, on both dialects -- the SAME
    // two-dialect nullability probe `migration_034_...`'s own oracle uses,
    // repeated for `instances`.
    let columns: Vec<(String, bool)> = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    match kind {
                        BackendKind::Sqlite => {
                            tx.query(
                                "SELECT name, \"notnull\" FROM pragma_table_info('instances') \
                                 WHERE name IN ('peer_addr', 'result_root', 'result_root_identity')",
                                &[],
                                |row| {
                                    let name: String = row.get("name")?;
                                    let notnull: i32 = row.get("notnull")?;
                                    Ok((name, notnull == 1))
                                },
                            )
                            .await
                        }
                        BackendKind::Postgres => {
                            tx.query(
                                "SELECT column_name, is_nullable FROM information_schema.columns \
                                 WHERE table_name = 'instances' \
                                   AND column_name IN ('peer_addr', 'result_root', 'result_root_identity')",
                                &[],
                                |row| {
                                    let name: String = row.get("column_name")?;
                                    let nullable: String = row.get("is_nullable")?;
                                    Ok((name, nullable == "NO"))
                                },
                            )
                            .await
                        }
                    }
                })
            },
        )
        .await
        .unwrap();
    for col in ["peer_addr", "result_root"] {
        assert!(
            columns.iter().any(|(c, notnull)| c == col && !notnull),
            "instances.{col} must be a nullable TEXT column; got {columns:?}"
        );
    }

    // A row with both columns NULL (every pre-existing/library-process row)
    // and a row with both set (a gang member) are both valid inserts -- no
    // paired CHECK, unlike migration 034's pair.
    let id_null = format!("mig035_null_{}", jammi_test_utils::unique_suffix());
    let id_set = format!("mig035_set_{}", jammi_test_utils::unique_suffix());
    backend
        .transaction(TxOptions::default(), |tx| {
            let id_null = id_null.clone();
            Box::pin(async move {
                tx.execute(
                    "INSERT INTO instances (instance_id, started_at, last_seen_at) \
                     VALUES ($1, '2026-01-01T00:00:00.000000Z', '2026-01-01T00:00:00.000000Z')",
                    &[SqlValue::TextOwned(id_null)],
                )
                .await
            })
        })
        .await
        .expect("both NULL must be a valid row");
    backend
        .transaction(TxOptions::default(), |tx| {
            let id_set = id_set.clone();
            Box::pin(async move {
                tx.execute(
                    "INSERT INTO instances \
                         (instance_id, started_at, last_seen_at, peer_addr, result_root) \
                     VALUES ($1, '2026-01-01T00:00:00.000000Z', '2026-01-01T00:00:00.000000Z', '10.0.0.1:9000', 'file:///data/jammi_db')",
                    &[SqlValue::TextOwned(id_set)],
                )
                .await
            })
        })
        .await
        .expect("both set must be a valid row");
}

/// Migration `036_instances_result_root_identity` is
/// present, ordered AFTER `035_instances_peer_addr_result_root` (relative
/// position, never `.last()`), and adds
/// `instances.result_root_identity` as a nullable `TEXT` column, on both
/// backends — the column `Catalog::list_gang_members` compares.
#[test_case::test_case(jammi_db::catalog::backend::BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(jammi_db::catalog::backend::BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn migration_036_is_ordered_after_035_and_adds_instances_result_root_identity(
    kind: jammi_db::catalog::backend::BackendKind,
) {
    use jammi_db::catalog::backend::BackendKind;
    let position = |name: &str| {
        EXPECTED_MIGRATION_NAMES
            .iter()
            .position(|m| *m == name)
            .unwrap_or_else(|| panic!("{name} missing from EXPECTED_MIGRATION_NAMES"))
    };
    assert!(
        position("036_instances_result_root_identity")
            > position("035_instances_peer_addr_result_root"),
        "the result_root_identity migration must follow 035"
    );
    let dir = tempdir().unwrap();
    let backend = match kind {
        BackendKind::Sqlite => {
            BackendImpl::Sqlite(open_sqlite_backend(&dir.path().join("catalog.db")).await)
        }
        BackendKind::Postgres => {
            let url = jammi_test_utils::postgres_url();
            BackendImpl::Postgres(
                jammi_db::catalog::backend_postgres::PostgresBackend::open_with_options(
                    &url, 4, None,
                )
                .await
                .unwrap(),
            )
        }
    };
    backend.migrate().await.unwrap();
    let columns: Vec<(String, bool)> = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    match kind {
                        BackendKind::Sqlite => {
                            tx.query(
                                "SELECT name, \"notnull\" FROM pragma_table_info('instances') \
                                 WHERE name = 'result_root_identity'",
                                &[],
                                |row| {
                                    let name: String = row.get("name")?;
                                    let notnull: i32 = row.get("notnull")?;
                                    Ok((name, notnull == 1))
                                },
                            )
                            .await
                        }
                        BackendKind::Postgres => {
                            tx.query(
                                "SELECT column_name, is_nullable FROM information_schema.columns \
                                 WHERE table_name = 'instances' \
                                   AND column_name = 'result_root_identity'",
                                &[],
                                |row| {
                                    let name: String = row.get("column_name")?;
                                    let nullable: String = row.get("is_nullable")?;
                                    Ok((name, nullable == "NO"))
                                },
                            )
                            .await
                        }
                    }
                })
            },
        )
        .await
        .unwrap();
    assert!(
        columns
            .iter()
            .any(|(c, notnull)| c == "result_root_identity" && !notnull),
        "instances.result_root_identity must be a nullable TEXT column; got {columns:?}"
    );
}

/// Migration `037_jobs_assembly_failures_next_after` is present,
/// ordered AFTER `036_instances_result_root_identity` (relative
/// position, never `.last()`), and adds `jobs.assembly_failures`
/// (`NOT NULL DEFAULT 0`) and `jobs.next_assembly_after` (nullable `TEXT`,
/// the SAME representation `jobs.lease_expires_at` uses) on both backends.
#[test_case::test_case(jammi_db::catalog::backend::BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(jammi_db::catalog::backend::BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn migration_037_is_ordered_after_036_and_adds_assembly_failures_next_after(
    kind: jammi_db::catalog::backend::BackendKind,
) {
    use jammi_db::catalog::backend::BackendKind;
    let position = |name: &str| {
        EXPECTED_MIGRATION_NAMES
            .iter()
            .position(|m| *m == name)
            .unwrap_or_else(|| panic!("{name} missing from EXPECTED_MIGRATION_NAMES"))
    };
    assert!(
        position("037_jobs_assembly_failures_next_after")
            > position("036_instances_result_root_identity"),
        "the assembly cooldown/counter migration must follow 036"
    );
    let dir = tempdir().unwrap();
    let backend = match kind {
        BackendKind::Sqlite => {
            BackendImpl::Sqlite(open_sqlite_backend(&dir.path().join("catalog.db")).await)
        }
        BackendKind::Postgres => {
            let url = jammi_test_utils::postgres_url();
            BackendImpl::Postgres(
                jammi_db::catalog::backend_postgres::PostgresBackend::open_with_options(
                    &url, 4, None,
                )
                .await
                .unwrap(),
            )
        }
    };
    backend.migrate().await.unwrap();
    let columns: Vec<(String, bool)> = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    match kind {
                        BackendKind::Sqlite => {
                            tx.query(
                                "SELECT name, \"notnull\" FROM pragma_table_info('jobs') \
                                 WHERE name IN ('assembly_failures', 'next_assembly_after')",
                                &[],
                                |row| {
                                    let name: String = row.get("name")?;
                                    let notnull: i32 = row.get("notnull")?;
                                    Ok((name, notnull == 1))
                                },
                            )
                            .await
                        }
                        BackendKind::Postgres => {
                            tx.query(
                                "SELECT column_name, is_nullable FROM information_schema.columns \
                                 WHERE table_name = 'jobs' \
                                   AND column_name IN ('assembly_failures', 'next_assembly_after')",
                                &[],
                                |row| {
                                    let name: String = row.get("column_name")?;
                                    let nullable: String = row.get("is_nullable")?;
                                    Ok((name, nullable == "NO"))
                                },
                            )
                            .await
                        }
                    }
                })
            },
        )
        .await
        .unwrap();
    assert!(
        columns
            .iter()
            .any(|(c, notnull)| c == "assembly_failures" && *notnull),
        "jobs.assembly_failures must be a NOT NULL column; got {columns:?}"
    );
    assert!(
        columns
            .iter()
            .any(|(c, notnull)| c == "next_assembly_after" && !notnull),
        "jobs.next_assembly_after must be a nullable TEXT column; got {columns:?}"
    );

    // The default is 0/NULL for a freshly migrated (i.e. pre-existing) row
    // -- inserting with neither column named must not fail and must read
    // back the documented defaults.
    let job_id = format!("mig037_default_{}", jammi_test_utils::unique_suffix());
    backend
        .transaction(TxOptions::default(), |tx| {
            let job_id = job_id.clone();
            Box::pin(async move {
                tx.execute(
                    "INSERT INTO jobs \
                     (job_id, kind, status, execution, spec, created_at, updated_at) \
                     VALUES ($1, 'fine_tune', 'queued', 'queued', '{}', '2026-01-01T00:00:00.000000Z', '2026-01-01T00:00:00.000000Z')",
                    &[jammi_db::catalog::backend::SqlValue::TextOwned(job_id)],
                )
                .await
            })
        })
        .await
        .expect("a row naming neither column must be a valid insert");
    let (failures, next_after): (i32, Option<String>) = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                let job_id = job_id.clone();
                Box::pin(async move {
                    tx.query_opt(
                        "SELECT assembly_failures, next_assembly_after FROM jobs \
                         WHERE job_id = $1",
                        &[jammi_db::catalog::backend::SqlValue::TextOwned(job_id)],
                        |row| {
                            Ok((
                                row.get::<i32>("assembly_failures")?,
                                row.try_get::<String>("next_assembly_after")?,
                            ))
                        },
                    )
                    .await
                })
            },
        )
        .await
        .unwrap()
        .expect("the inserted row must be readable");
    assert_eq!(failures, 0, "assembly_failures defaults to 0");
    assert_eq!(next_after, None, "next_assembly_after defaults to NULL");
}

/// Migration `038_compute_cluster_state` is present, ordered
/// AFTER BOTH `035_instances_peer_addr_result_root` (its
/// `compute_executors.instance_id` join target) and
/// `037_jobs_assembly_failures_next_after` (relative position, never
/// `.last()`), creates `compute_executors`/`compute_jobs`, and adds
/// `workers.devices` (`NOT NULL DEFAULT '[]'`) on both backends.
#[test_case::test_case(jammi_db::catalog::backend::BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(jammi_db::catalog::backend::BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn migration_038_is_ordered_after_035_and_037_and_creates_compute_tables(
    kind: jammi_db::catalog::backend::BackendKind,
) {
    use jammi_db::catalog::backend::{BackendKind, SqlValue};

    let position = |name: &str| {
        EXPECTED_MIGRATION_NAMES
            .iter()
            .position(|m| *m == name)
            .unwrap_or_else(|| panic!("{name} missing from EXPECTED_MIGRATION_NAMES"))
    };
    assert!(
        position("038_compute_cluster_state") > position("035_instances_peer_addr_result_root"),
        "the compute-cluster-state migration must follow 035 (its instance_id join target)"
    );
    assert!(
        position("038_compute_cluster_state") > position("037_jobs_assembly_failures_next_after"),
        "the compute-cluster-state migration must follow 037"
    );

    let dir = tempdir().unwrap();
    let backend = match kind {
        BackendKind::Sqlite => {
            BackendImpl::Sqlite(open_sqlite_backend(&dir.path().join("catalog.db")).await)
        }
        BackendKind::Postgres => {
            let url = jammi_test_utils::postgres_url();
            BackendImpl::Postgres(
                jammi_db::catalog::backend_postgres::PostgresBackend::open_with_options(
                    &url, 4, None,
                )
                .await
                .unwrap(),
            )
        }
    };
    backend.migrate().await.unwrap();

    // Both new tables exist.
    for table in ["compute_executors", "compute_jobs"] {
        let exists: bool = backend
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        match kind {
                            BackendKind::Sqlite => {
                                let rows: Vec<i64> = tx
                                    .query(
                                        "SELECT 1 AS one FROM sqlite_master \
                                         WHERE type='table' AND name=$1",
                                        &[SqlValue::TextOwned(table.to_string())],
                                        |row| row.get::<i64>("one"),
                                    )
                                    .await?;
                                Ok(!rows.is_empty())
                            }
                            BackendKind::Postgres => {
                                let rows: Vec<i64> = tx
                                    .query(
                                        "SELECT 1::bigint AS one FROM information_schema.tables \
                                         WHERE table_schema = 'public' AND table_name = $1",
                                        &[SqlValue::TextOwned(table.to_string())],
                                        |row| row.get::<i64>("one"),
                                    )
                                    .await?;
                                Ok(!rows.is_empty())
                            }
                        }
                    })
                },
            )
            .await
            .unwrap();
        assert!(exists, "'{table}' must exist after migration 038");
    }

    // `compute_executors.devices` is a NOT NULL column -- the placement
    // policy's own device authority, distinct
    // from `workers.devices` below.
    let columns: Vec<(String, bool)> = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    match kind {
                        BackendKind::Sqlite => {
                            tx.query(
                                "SELECT name, \"notnull\" FROM pragma_table_info('compute_executors') \
                                 WHERE name = 'devices'",
                                &[],
                                |row| {
                                    let name: String = row.get("name")?;
                                    let notnull: i32 = row.get("notnull")?;
                                    Ok((name, notnull == 1))
                                },
                            )
                            .await
                        }
                        BackendKind::Postgres => {
                            tx.query(
                                "SELECT column_name, is_nullable FROM information_schema.columns \
                                 WHERE table_name = 'compute_executors' AND column_name = 'devices'",
                                &[],
                                |row| {
                                    let name: String = row.get("column_name")?;
                                    let nullable: String = row.get("is_nullable")?;
                                    Ok((name, nullable == "NO"))
                                },
                            )
                            .await
                        }
                    }
                })
            },
        )
        .await
        .unwrap();
    assert!(
        columns
            .iter()
            .any(|(c, notnull)| c == "devices" && *notnull),
        "compute_executors.devices must be a NOT NULL column; got {columns:?}"
    );

    // The default is `'[]'` for a freshly registered executor row --
    // inserting with `devices` unnamed must not fail and must read back
    // the documented default.
    let executor_id = format!("mig038_executor_{}", jammi_test_utils::unique_suffix());
    backend
        .transaction(TxOptions::default(), |tx| {
            let executor_id = executor_id.clone();
            Box::pin(async move {
                tx.execute(
                    "INSERT INTO compute_executors \
                     (executor_id, instance_id, host, port, grpc_port, task_slots, \
                      available_slots, status, heartbeat_at, metadata) \
                     VALUES ($1, 'inst-1', 'localhost', 50051, 50052, 1, 1, 'live', '2026-01-01T00:00:00.000000Z', '{}')",
                    &[SqlValue::TextOwned(executor_id.clone())],
                )
                .await
            })
        })
        .await
        .expect("a compute_executors row naming no `devices` must be a valid insert");
    let devices: String = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                let executor_id = executor_id.clone();
                Box::pin(async move {
                    tx.query_opt(
                        "SELECT devices FROM compute_executors WHERE executor_id = $1",
                        &[SqlValue::TextOwned(executor_id)],
                        |row| row.get::<String>("devices"),
                    )
                    .await
                })
            },
        )
        .await
        .unwrap()
        .expect("the inserted row must be readable");
    assert_eq!(
        devices, "[]",
        "compute_executors.devices defaults to the empty JSON array"
    );

    // `workers.devices` is a NOT NULL column.
    let columns: Vec<(String, bool)> = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    match kind {
                        BackendKind::Sqlite => {
                            tx.query(
                                "SELECT name, \"notnull\" FROM pragma_table_info('workers') \
                                 WHERE name = 'devices'",
                                &[],
                                |row| {
                                    let name: String = row.get("name")?;
                                    let notnull: i32 = row.get("notnull")?;
                                    Ok((name, notnull == 1))
                                },
                            )
                            .await
                        }
                        BackendKind::Postgres => {
                            tx.query(
                                "SELECT column_name, is_nullable FROM information_schema.columns \
                                 WHERE table_name = 'workers' AND column_name = 'devices'",
                                &[],
                                |row| {
                                    let name: String = row.get("column_name")?;
                                    let nullable: String = row.get("is_nullable")?;
                                    Ok((name, nullable == "NO"))
                                },
                            )
                            .await
                        }
                    }
                })
            },
        )
        .await
        .unwrap();
    assert!(
        columns
            .iter()
            .any(|(c, notnull)| c == "devices" && *notnull),
        "workers.devices must be a NOT NULL column; got {columns:?}"
    );

    // The default is `'[]'` for a freshly migrated worker row -- inserting
    // with neither `devices` nor `state` named must not fail and must read
    // back the documented default.
    let instance_id = format!("mig038_default_{}", jammi_test_utils::unique_suffix());
    backend
        .transaction(TxOptions::default(), |tx| {
            let instance_id = instance_id.clone();
            Box::pin(async move {
                tx.execute(
                    "INSERT INTO instances (instance_id, started_at, last_seen_at) \
                     VALUES ($1, '2026-01-01T00:00:00.000000Z', '2026-01-01T00:00:00.000000Z')",
                    &[SqlValue::TextOwned(instance_id.clone())],
                )
                .await?;
                tx.execute(
                    "INSERT INTO workers (instance_id, kinds) VALUES ($1, 'all')",
                    &[SqlValue::TextOwned(instance_id)],
                )
                .await
            })
        })
        .await
        .expect("a worker row naming neither `devices` nor `state` must be a valid insert");
    let devices: String = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                let instance_id = instance_id.clone();
                Box::pin(async move {
                    tx.query_opt(
                        "SELECT devices FROM workers WHERE instance_id = $1",
                        &[SqlValue::TextOwned(instance_id)],
                        |row| row.get::<String>("devices"),
                    )
                    .await
                })
            },
        )
        .await
        .unwrap()
        .expect("the inserted row must be readable");
    assert_eq!(
        devices, "[]",
        "workers.devices defaults to the empty JSON array"
    );
}

/// The 13 `(table, column)` pairs `catalog::lease`'s schema-edge domain
/// (migration `039_canonical_stamps`) enforces — the universe gate this
/// oracle enumerates against (a later rebuild-dance migration cannot
/// silently drop enforcement).
const CANONICAL_STAMP_COLUMNS: &[(&str, &str)] = &[
    ("jobs", "lease_expires_at"),
    ("jobs", "next_assembly_after"),
    ("jobs", "updated_at"),
    ("jobs", "created_at"),
    ("instances", "last_seen_at"),
    ("instances", "started_at"),
    ("result_tables", "lease_expires_at"),
    ("result_tables", "created_at"),
    ("result_table_versions", "lease_expires_at"),
    ("compute_executors", "heartbeat_at"),
    ("models", "created_at"),
    ("models", "updated_at"),
    ("applied_migrations", "applied_at"),
];

/// `stale_before_clause` compares a stamp column lexically, which is
/// chronological only for a column the schema holds to the canonical shape.
/// Every column it can be asked about is therefore one this migration
/// enforces — the set the test below proves is installed exactly.
#[test]
fn every_column_stale_before_clause_accepts_is_schema_enforced() {
    use jammi_db::catalog::lease::CanonicalStampColumn;
    use strum::VariantArray;
    for column in CanonicalStampColumn::VARIANTS {
        assert!(
            CANONICAL_STAMP_COLUMNS.contains(&column.table_and_column()),
            "{column:?} is compared lexically by stale_before_clause but \
             {:?} is not in the canonical-stamps enforcement set",
            column.table_and_column()
        );
    }
}

/// Migration `039_canonical_stamps` is
/// ordered after `038_compute_cluster_state` (relative position, never
/// `.last()` — it names `compute_executors`, which `038` creates), and the
/// enforcement set it installs — on SQLite, a `BEFORE INSERT` AND a
/// `BEFORE UPDATE OF <col>` trigger per [`CANONICAL_STAMP_COLUMNS`] entry
/// (26 triggers); on Postgres, one `sdchk__<table>__<column>` `CHECK`
/// constraint per entry (13 constraints) — equals exactly that set on a
/// freshly migrated catalog. A future migration that rebuilds one of these
/// tables (SQLite's create-new/copy/drop/rename dance, migration 012's own
/// shape) without reinstalling its two triggers would silently drop
/// enforcement for that column; this oracle catches it by enumerating
/// `sqlite_master`/`pg_constraint` directly rather than trusting that the
/// DDL which installed them once still applies.
#[test_case::test_case(jammi_db::catalog::backend::BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(jammi_db::catalog::backend::BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn migration_039_is_ordered_after_038_and_the_enforcement_set_is_exact(
    kind: jammi_db::catalog::backend::BackendKind,
) {
    use jammi_db::catalog::backend::{BackendKind, SqlValue};

    let position = |name: &str| {
        EXPECTED_MIGRATION_NAMES
            .iter()
            .position(|m| *m == name)
            .unwrap_or_else(|| panic!("{name} missing from EXPECTED_MIGRATION_NAMES"))
    };
    assert!(
        position("039_canonical_stamps") > position("038_compute_cluster_state"),
        "039_canonical_stamps must follow 038 (it enforces compute_executors.heartbeat_at, \
         which 038 creates)"
    );

    let dir = tempdir().unwrap();
    let backend = match kind {
        BackendKind::Sqlite => {
            BackendImpl::Sqlite(open_sqlite_backend(&dir.path().join("catalog.db")).await)
        }
        BackendKind::Postgres => {
            let url = jammi_test_utils::postgres_url();
            BackendImpl::Postgres(
                jammi_db::catalog::backend_postgres::PostgresBackend::open_with_options(
                    &url, 4, None,
                )
                .await
                .unwrap(),
            )
        }
    };
    backend.migrate().await.unwrap();

    match kind {
        BackendKind::Sqlite => {
            let expected: std::collections::BTreeSet<String> = CANONICAL_STAMP_COLUMNS
                .iter()
                .flat_map(|(t, c)| {
                    [
                        format!("trg_{t}_{c}_canonical_ins"),
                        format!("trg_{t}_{c}_canonical_upd"),
                    ]
                })
                .collect();
            assert_eq!(expected.len(), CANONICAL_STAMP_COLUMNS.len() * 2);
            let actual: std::collections::BTreeSet<String> = backend
                .transaction(
                    TxOptions {
                        read_only: true,
                        ..Default::default()
                    },
                    |tx| {
                        Box::pin(async move {
                            tx.query(
                                "SELECT name FROM sqlite_master WHERE type = 'trigger' \
                                 AND name LIKE 'trg_%_canonical_%'",
                                &[],
                                |row| row.get::<String>("name"),
                            )
                            .await
                        })
                    },
                )
                .await
                .unwrap()
                .into_iter()
                .collect();
            assert_eq!(
                actual, expected,
                "the SQLite trigger set must equal CANONICAL_STAMP_COLUMNS exactly"
            );
        }
        BackendKind::Postgres => {
            let expected: std::collections::BTreeSet<String> = CANONICAL_STAMP_COLUMNS
                .iter()
                .map(|(t, c)| format!("sdchk__{t}__{c}"))
                .collect();
            let actual: std::collections::BTreeSet<String> = backend
                .transaction(
                    TxOptions {
                        read_only: true,
                        ..Default::default()
                    },
                    |tx| {
                        Box::pin(async move {
                            tx.query(
                                "SELECT conname FROM pg_constraint \
                                 WHERE contype = 'c' AND conname LIKE $1",
                                &[SqlValue::TextOwned("sdchk\\_\\_%".to_string())],
                                |row| row.get::<String>("conname"),
                            )
                            .await
                        })
                    },
                )
                .await
                .unwrap()
                .into_iter()
                .collect();
            assert_eq!(
                actual, expected,
                "the Postgres sdchk__ constraint set must equal CANONICAL_STAMP_COLUMNS exactly"
            );
        }
    }
}

/// The rewrite's fail-closed behaviour is an OUTCOME property,
/// not a mechanism — a refused re-application of `039_canonical_stamps`
/// must leave the ledger at 038, the offending row's own value untouched,
/// AND install NEITHER the trigger nor the `CHECK` for the column it could
/// not normalise (a `RAISE(ABORT)`/cast fault rolls back only its own
/// statement; fail-closed holds because the migration RUNNER's one
/// transaction is never committed once an `Err` propagates — the state
/// this oracle asserts, not the rollback mechanism itself).
///
/// SQLite only, deliberately: 039 is ONE migration entry, tracked by ONE
/// ledger name, so re-applying it (clearing its ledger row) re-runs its
/// FULL 26-trigger/13-rewrite statement list, not just the one column this
/// test manufactures an offending value for. On a private tempdir catalog
/// (this test's own, touched by nothing else) that is harmless — every
/// other column's trigger/rewrite is a no-op against already-canonical
/// data. On Postgres the lane shares one `jammi_test` database: re-applying
/// 039 there re-attempts `ADD CONSTRAINT` for the 12 OTHER columns this test
/// never touched, which already exist from the real, once-only migration
/// every other test in this suite depends on, and fails with `already
/// exists` — corrupting the shared migration ledger. A migration that is
/// monolithic across many columns cannot safely be partially rewound
/// against a database other concurrent test runs share; SQLite's per-test
/// isolation is the only backend this fixture shape is safe on. The property
/// itself — a failed migration's transaction never commits — is
/// backend-symmetric and is exercised by EVERY OTHER migration's own tests on
/// both backends (the runner's transaction wrapping is not 039-specific
/// code).
#[tokio::test]
async fn migration_039_on_an_unclassifiable_value_fails_closed() {
    use jammi_db::catalog::backend::SqlValue;

    let dir = tempdir().unwrap();
    let backend = BackendImpl::Sqlite(open_sqlite_backend(&dir.path().join("catalog.db")).await);
    // A real, fully-migrated catalog first (039 included) -- the baseline
    // every assertion below diffs against.
    backend.migrate().await.unwrap();

    let job_id = "mig039-fail-closed".to_string();
    let offending = "unclassifiable-garbage-value";
    backend
        .transaction(TxOptions::default(), |tx| {
            let job_id = job_id.clone();
            Box::pin(async move {
                // A minimal, valid `jobs` row so the UPDATE below has
                // something to hit.
                tx.execute(
                    "INSERT INTO jobs (job_id, kind, execution, spec, created_at, updated_at) \
                     VALUES ($1, 'k', 'queued', '{}', '2026-01-01T00:00:00.000000Z', \
                             '2026-01-01T00:00:00.000000Z')",
                    &[SqlValue::TextOwned(job_id.clone())],
                )
                .await?;
                tx.execute("DROP TRIGGER trg_jobs_lease_expires_at_canonical_ins", &[])
                    .await?;
                tx.execute("DROP TRIGGER trg_jobs_lease_expires_at_canonical_upd", &[])
                    .await?;
                tx.execute(
                    "UPDATE jobs SET lease_expires_at = $1 WHERE job_id = $2",
                    &[
                        SqlValue::Text(offending),
                        SqlValue::TextOwned(job_id.clone()),
                    ],
                )
                .await?;
                tx.execute(
                    "DELETE FROM applied_migrations WHERE name = '039_canonical_stamps'",
                    &[],
                )
                .await
            })
        })
        .await
        .unwrap();

    // The re-application must fail.
    let err = backend.migrate().await.expect_err(
        "039 must refuse to normalise a value none of its three recognised shapes match",
    );
    assert!(
        matches!(
            err,
            jammi_db::catalog::backend::BackendError::DomainViolation { .. }
        ),
        "the refusal must be the typed domain-violation class, got {err:?}"
    );

    // Outcome, not mechanism: the ledger stays at 038 for this migration,
    let ledger: Vec<String> = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query(
                        "SELECT name FROM applied_migrations WHERE name = '039_canonical_stamps'",
                        &[],
                        |row| row.get::<String>("name"),
                    )
                    .await
                })
            },
        )
        .await
        .unwrap();
    assert!(
        ledger.is_empty(),
        "039_canonical_stamps must NOT be recorded as applied after a failed re-application"
    );

    // the row's own value is untouched (never rewritten, never nulled),
    let stored: String = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                let job_id = job_id.clone();
                Box::pin(async move {
                    tx.query_opt(
                        "SELECT lease_expires_at FROM jobs WHERE job_id = $1",
                        &[SqlValue::TextOwned(job_id)],
                        |row| row.get::<String>("lease_expires_at"),
                    )
                    .await
                })
            },
        )
        .await
        .unwrap()
        .expect("the row must still exist");
    assert_eq!(
        stored, offending,
        "the offending row's value must be intact after the refused migration"
    );

    // and the enforcement this exact failed attempt would have installed
    // does NOT exist (the dropped trigger for THIS column stays dropped --
    // a refused migration installs nothing, not even partially).
    let enforcement_exists: bool = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    let rows: Vec<i64> = tx
                        .query(
                            "SELECT 1 AS one FROM sqlite_master WHERE type = 'trigger' \
                             AND name = 'trg_jobs_lease_expires_at_canonical_upd'",
                            &[],
                            |row| row.get::<i64>("one"),
                        )
                        .await?;
                    Ok(!rows.is_empty())
                })
            },
        )
        .await
        .unwrap();
    assert!(
        !enforcement_exists,
        "a refused migration must install neither the trigger nor the CHECK it was attempting"
    );
}

/// `catalog::lease::pg_canonical_stamp`'s GUC-independence: `to_char` with an explicit picture must
/// render the SAME text regardless of the session's `DateStyle`/`TimeZone`, unlike a bare `::text`
/// cast. `SET LOCAL` (transaction-scoped, reverted automatically at commit or rollback) rather than
/// `SET`, since this runs on a POOLED connection another test could reuse afterward.
/// Self-contained: no table at all, a plain `SELECT` of a literal expression, never touching the
/// shared schema.
#[cfg(feature = "live-postgres-tests")]
#[tokio::test]
async fn pg_canonical_stamp_is_independent_of_session_datestyle_and_timezone() {
    use jammi_db::catalog::lease::pg_canonical_stamp;

    let url = jammi_test_utils::postgres_url();
    let backend = BackendImpl::Postgres(
        jammi_db::catalog::backend_postgres::PostgresBackend::open_with_options(&url, 2, None)
            .await
            .unwrap(),
    );

    let default_render: String = backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                let sql = format!(
                    "SELECT {} AS s",
                    pg_canonical_stamp("'2026-01-01T00:00:00.123456Z'::timestamptz")
                );
                tx.query_opt(&sql, &[], |row| row.get::<String>("s")).await
            })
        })
        .await
        .unwrap()
        .expect("one row");

    let altered_render: String = backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute("SET LOCAL datestyle = 'SQL, MDY'", &[]).await?;
                tx.execute("SET LOCAL timezone = 'Asia/Kolkata'", &[])
                    .await?;
                let sql = format!(
                    "SELECT {} AS s",
                    pg_canonical_stamp("'2026-01-01T00:00:00.123456Z'::timestamptz")
                );
                tx.query_opt(&sql, &[], |row| row.get::<String>("s")).await
            })
        })
        .await
        .unwrap()
        .expect("one row");

    assert_eq!(
        default_render, "2026-01-01T00:00:00.123456Z",
        "the default-session rendering must already be canonical"
    );
    assert_eq!(
        default_render, altered_render,
        "pg_canonical_stamp's rendering must not depend on DateStyle/TimeZone"
    );

    // Control: a bare `::text` cast (what the pre-039 writer used) DOES
    // depend on these GUCs -- proving the oracle's altered-session arm
    // actually changed something observable, not merely a no-op SET.
    let bare_cast_default: String = backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.query_opt(
                    "SELECT ('2026-01-01T00:00:00.123456Z'::timestamptz)::text AS s",
                    &[],
                    |row| row.get::<String>("s"),
                )
                .await
            })
        })
        .await
        .unwrap()
        .expect("one row");
    let bare_cast_altered: String = backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute("SET LOCAL datestyle = 'SQL, MDY'", &[]).await?;
                tx.query_opt(
                    "SELECT ('2026-01-01T00:00:00.123456Z'::timestamptz)::text AS s",
                    &[],
                    |row| row.get::<String>("s"),
                )
                .await
            })
        })
        .await
        .unwrap()
        .expect("one row");
    assert_ne!(
        bare_cast_default, bare_cast_altered,
        "control: a bare ::text cast DOES vary with DateStyle -- confirms the SET actually took \
         effect, so pg_canonical_stamp's invariance above is not a vacuous pass"
    );
}

/// A ledger-level source oracle. `jobs` may become an FK PARENT in this
/// schema, and `PRAGMA foreign_keys` is ON for the pool
/// migrations run on (`backend_sqlite.rs`'s `.foreign_keys(true)`) — so a
/// migration that rebuilds `jobs` with this codebase's own canonical
/// SQLite create-new/copy/DROP-old/RENAME idiom (schema.rs uses it
/// for `topics`, `eval_runs`, `evidence_channels`, `training_jobs`) would
/// silently CASCADE-delete every row of any table that FK-references
/// `jobs(job_id) ON DELETE CASCADE` the moment the old `jobs` table is
/// dropped, unless that migration explicitly disables FK enforcement (or
/// otherwise preserves the referencing rows) for the swap. No migration in
/// `MIGRATIONS` rebuilds `jobs` (every existing rebuild targets a different
/// table); this oracle pins that state so a later migration cannot
/// introduce the hazard unnoticed. Scoped by
/// TABLE NAME, not substring: `training_jobs` (migration 029's own DROP) is
/// a DIFFERENT table and must not false-positive.
#[test]
fn no_migration_text_rebuilds_the_jobs_table_without_a_stated_preserving_idiom() {
    let source = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/catalog/schema.rs"
    ))
    .expect("read schema.rs");

    let forbidden = jobs_table_rebuild_hits(&source);
    assert!(
        forbidden.is_empty(),
        "a migration text rebuilds (or renames away) the `jobs` table without a stated \
         preserving idiom -- this can silently drop every FK-referencing row \
         (job_dependencies/job_ancestors) once `jobs` is a FK parent under \
         PRAGMA foreign_keys = ON: {forbidden:?}. Either this migration explicitly disables \
         FK enforcement for the swap (documented in its own rustdoc, matching \
         the FK rule migration 039's rustdoc in schema.rs states) or the change belongs elsewhere."
    );
}

/// Every line in `source` matching a `jobs`-table rebuild/rename shape,
/// scoped to the EXACT table name (never a substring like `training_jobs`).
fn jobs_table_rebuild_hits(source: &str) -> Vec<(usize, String)> {
    let mut hits = Vec::new();
    for (idx, line) in source.lines().enumerate() {
        let has_drop_jobs = contains_word_pair(line, "DROP TABLE", "jobs");
        let has_rename_to_jobs = contains_word_pair(line, "RENAME TO", "jobs");
        let has_alter_jobs_rename = line.contains("ALTER TABLE jobs") && line.contains("RENAME");
        if has_drop_jobs || has_rename_to_jobs || has_alter_jobs_rename {
            hits.push((idx + 1, line.trim().to_string()));
        }
    }
    hits
}

/// True when `line` contains `prefix` immediately followed by whitespace and
/// EXACTLY the identifier `word` (not a longer identifier it is a suffix of
/// — `training_jobs` must not match `word = "jobs"`).
fn contains_word_pair(line: &str, prefix: &str, word: &str) -> bool {
    let Some(pos) = line.find(prefix) else {
        return false;
    };
    let rest = line[pos + prefix.len()..].trim_start();
    let ident_end = rest
        .find(|c: char| !(c.is_alphanumeric() || c == '_'))
        .unwrap_or(rest.len());
    &rest[..ident_end] == word
}

#[cfg(test)]
mod jobs_table_rebuild_hits_self_tests {
    use super::{contains_word_pair, jobs_table_rebuild_hits};

    #[test]
    fn a_bare_drop_table_jobs_is_a_hit() {
        assert_eq!(
            jobs_table_rebuild_hits("DROP TABLE jobs;\n"),
            vec![(1, "DROP TABLE jobs;".to_string())]
        );
    }

    #[test]
    fn a_rename_to_jobs_is_a_hit() {
        assert_eq!(
            jobs_table_rebuild_hits("ALTER TABLE jobs_new RENAME TO jobs;\n"),
            vec![(1, "ALTER TABLE jobs_new RENAME TO jobs;".to_string())]
        );
    }

    /// The negative control naming the exact false-positive this scoping
    /// exists to avoid: `training_jobs` is a DIFFERENT table.
    #[test]
    fn drop_table_training_jobs_is_not_a_hit() {
        assert!(jobs_table_rebuild_hits("DROP TABLE training_jobs;\n").is_empty());
    }

    #[test]
    fn rename_to_training_jobs_is_not_a_hit() {
        assert!(
            jobs_table_rebuild_hits("ALTER TABLE fine_tune_jobs RENAME TO training_jobs;\n")
                .is_empty()
        );
    }

    #[test]
    fn contains_word_pair_rejects_a_longer_identifier() {
        assert!(!contains_word_pair(
            "DROP TABLE training_jobs;",
            "DROP TABLE",
            "jobs"
        ));
        assert!(contains_word_pair("DROP TABLE jobs;", "DROP TABLE", "jobs"));
    }
}
