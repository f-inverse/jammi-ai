//! End-to-end migration tests. Asserts via the new `applied_migrations`
//! ledger and direct sqlx queries against the on-disk catalog.
//!
//! The two `concurrent_migrate_on_fresh_*_is_safe` tests race two independent
//! backends' `migrate()` on one FRESH catalog (escape-ledger row
//! `esc-093-postgres-migrations-race-without-cross-process-lock`, issue #479):
//! the SQLite arm is a regression guard that was green before the fix (the
//! backend's `BEGIN IMMEDIATE` already serialises it); the Postgres arm is the
//! RED-then-GREEN oracle for the advisory lock `catalog::migrations::run` takes.

use std::collections::BTreeSet;
use std::sync::Arc;

use jammi_db::catalog::backend::{BackendError, BackendImpl, CatalogBackend, TxOptions};
use jammi_db::catalog::backend_sqlite::SqliteBackend;
use jammi_db::catalog::Catalog;
use tempfile::tempdir;
use tokio::sync::Barrier;

/// Every migration name, in ledger order. Mirrors `catalog::migrations::MIGRATIONS`
/// (K5: append-only, currently ending at 027) -- a new migration is added here
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
    for table in [
        "sources",
        "models",
        "training_jobs",
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
    for (table, idx) in [
        ("sources", "idx_sources_tenant"),
        ("models", "idx_models_tenant"),
        ("training_jobs", "idx_training_jobs_tenant"),
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
                        "INSERT INTO models (model_id, name, model_type, task, version, status) \
                         VALUES ($1, $2, 'embedding', 'text-embedding', 1, $3)",
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

/// Migration 015 adds the lease-based job-queue columns and the
/// `(status, lease_expires_at)` claim index; migration 016 renames the table to
/// `training_jobs` and the index to `idx_training_jobs_claim`. Asserted against
/// the post-016 names via `pragma_table_info` and `sqlite_master`.
#[tokio::test]
async fn migration_015_adds_job_queue_columns_and_claim_index() {
    use jammi_db::catalog::backend::SqlValue;

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
                        "SELECT name FROM pragma_table_info('training_jobs')",
                        &[],
                        |row| row.get("name"),
                    )
                    .await
                })
            },
        )
        .await
        .unwrap();
    for expected in [
        "kind",
        "claimed_by",
        "lease_expires_at",
        "attempts",
        "training_spec",
    ] {
        assert!(
            columns.iter().any(|c| c == expected),
            "training_jobs must have '{expected}' after migrations 015+016; got {columns:?}"
        );
    }

    let index_exists = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    let rows: Vec<i64> = tx
                        .query(
                            "SELECT 1 AS one FROM sqlite_master \
                             WHERE type='index' AND name=$1 AND tbl_name='training_jobs'",
                            &[SqlValue::TextOwned("idx_training_jobs_claim".into())],
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
        index_exists,
        "idx_training_jobs_claim must exist after migrations 015+016"
    );
}

/// Migration 016 renames the job table `fine_tune_jobs → training_jobs` and its
/// three indexes (`idx_fine_tune_jobs_{status,tenant,claim} →
/// idx_training_jobs_{status,tenant,claim}`). After a full open the renamed
/// table and indexes exist and the old names are gone. Asserted via
/// `sqlite_master`.
#[tokio::test]
async fn migration_016_renames_job_table_and_indexes() {
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
                                    "SELECT 1 AS one FROM sqlite_master \
                                     WHERE type=$1 AND name=$2",
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

    assert!(
        object_exists("table", "training_jobs").await,
        "training_jobs table must exist after migration 016"
    );
    assert!(
        !object_exists("table", "fine_tune_jobs").await,
        "fine_tune_jobs table must be gone after migration 016"
    );
    for renamed in [
        "idx_training_jobs_status",
        "idx_training_jobs_tenant",
        "idx_training_jobs_claim",
    ] {
        assert!(
            object_exists("index", renamed).await,
            "index '{renamed}' must exist after migration 016"
        );
    }
    for old in [
        "idx_fine_tune_jobs_status",
        "idx_fine_tune_jobs_tenant",
        "idx_fine_tune_jobs_claim",
    ] {
        assert!(
            !object_exists("index", old).await,
            "old index '{old}' must be gone after migration 016"
        );
    }
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

/// Migration 024 adds the claim-policy columns (`priority`, `claimable`) and
/// the `idx_training_jobs_claim_policy` index to `training_jobs`, on both a
/// fresh catalog and a catalog migrated up from a pre-024 file — the older
/// reclaim index (`idx_training_jobs_claim`, migration 016) survives either
/// way, and a row born under either schema backfills to the same defaults
/// (`priority = 0`, `claimable = TRUE`).
#[tokio::test]
async fn migration_024_adds_claim_policy_columns_and_index() {
    use jammi_db::catalog::backend::SqlValue;
    use jammi_db::catalog::model_repo::RegisterModelParams;
    use jammi_db::catalog::training_repo::CreateTrainingJobParams;
    use jammi_db::model_task::ModelTask;

    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).await.unwrap();
    let backend = BackendImpl::Sqlite(open_sqlite_backend(&dir.path().join("catalog.db")).await);

    let table_columns = || {
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
                            tx.query::<_, String>(
                                "SELECT name FROM pragma_table_info('training_jobs')",
                                &[],
                                |row| row.get("name"),
                            )
                            .await
                        })
                    },
                )
                .await
                .unwrap()
        }
    };
    let index_exists = |name: &'static str| {
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
                                    "SELECT 1 AS one FROM sqlite_master \
                                     WHERE type='index' AND name=$1 AND tbl_name='training_jobs'",
                                    &[SqlValue::TextOwned(name.into())],
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
    let job_claim_policy = |job_id: &'static str| {
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
                            tx.query::<_, (i64, bool)>(
                                "SELECT priority, claimable FROM training_jobs WHERE job_id = $1",
                                &[SqlValue::TextOwned(job_id.into())],
                                |row| Ok((row.get("priority")?, row.get("claimable")?)),
                            )
                            .await
                        })
                    },
                )
                .await
                .unwrap()
                .into_iter()
                .next()
                .unwrap_or_else(|| panic!("job '{job_id}' must exist"))
        }
    };

    // --- Fresh catalog: the columns and the new index exist; the older
    // reclaim index survives alongside the new one. ---
    let columns = table_columns().await;
    for expected in ["priority", "claimable"] {
        assert!(
            columns.iter().any(|c| c == expected),
            "training_jobs must have '{expected}' after migration 024; got {columns:?}"
        );
    }
    assert!(
        index_exists("idx_training_jobs_claim_policy").await,
        "idx_training_jobs_claim_policy must exist after migration 024"
    );
    assert!(
        index_exists("idx_training_jobs_claim").await,
        "idx_training_jobs_claim (the reclaim index, migration 016) must survive migration 024"
    );

    // A freshly enqueued row is born at the defaults.
    catalog
        .register_model(RegisterModelParams {
            model_id: "shape-base",
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: None,
            config_json: None,
        })
        .await
        .unwrap();
    catalog
        .create_training_job(CreateTrainingJobParams {
            job_id: "fresh",
            base_model_id: "shape-base::1",
            training_source: "src.csv",
            loss_type: "contrastive",
            hyperparams: "{}",
            kind: "fine_tune",
            training_spec: "{}",
        })
        .await
        .unwrap();
    let (priority, claimable) = job_claim_policy("fresh").await;
    assert_eq!(priority, 0, "a freshly enqueued job defaults to priority 0");
    assert!(
        claimable,
        "a freshly enqueued job defaults to claimable = TRUE"
    );

    // --- Pre-024-migrated catalog: roll migration 024 back by hand (drop its
    // index and columns, remove its ledger entry) to reproduce the schema a
    // pre-024 catalog file would carry, insert a row under that schema, then
    // reopen — the runner re-applies 024 on top of the existing row and
    // backfills it to the same defaults, exactly as a real upgrade would. ---
    backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute("DROP INDEX idx_training_jobs_claim_policy", &[])
                    .await?;
                tx.execute("ALTER TABLE training_jobs DROP COLUMN claimable", &[])
                    .await?;
                tx.execute("ALTER TABLE training_jobs DROP COLUMN priority", &[])
                    .await?;
                tx.execute(
                    "DELETE FROM applied_migrations WHERE name = '024_claim_policy'",
                    &[],
                )
                .await?;
                Ok(())
            })
        })
        .await
        .unwrap();

    catalog
        .create_training_job(CreateTrainingJobParams {
            job_id: "pre-024",
            base_model_id: "shape-base::1",
            training_source: "src.csv",
            loss_type: "contrastive",
            hyperparams: "{}",
            kind: "fine_tune",
            training_spec: "{}",
        })
        .await
        .unwrap();

    drop(catalog);
    let _reopened = Catalog::open(dir.path()).await.unwrap();

    let columns_after = table_columns().await;
    for expected in ["priority", "claimable"] {
        assert!(
            columns_after.iter().any(|c| c == expected),
            "training_jobs must regain '{expected}' after re-applying migration 024; \
             got {columns_after:?}"
        );
    }
    assert!(
        index_exists("idx_training_jobs_claim_policy").await,
        "idx_training_jobs_claim_policy must exist again after re-applying migration 024"
    );
    assert!(
        index_exists("idx_training_jobs_claim").await,
        "idx_training_jobs_claim must still exist after re-applying migration 024"
    );

    let (priority, claimable) = job_claim_policy("pre-024").await;
    assert_eq!(
        priority, 0,
        "a row born before migration 024 backfills to priority 0"
    );
    assert!(
        claimable,
        "a row born before migration 024 backfills to claimable = TRUE"
    );
}

/// Migration 026 adds the `acceleration_report` column to `training_jobs`
/// (esc-075). Asserted append-only-append (the column exists on a fresh open,
/// migrations remain idempotent across a reopen) and the two catalog-owned
/// states of its producer-owned-payload contract: a row that predates the
/// migration reads back `NULL` — "unknown" — never a fabricated `pending`,
/// while a row created after the migration through
/// [`Catalog::create_training_job`] carries the explicit
/// `{"state":"pending"}` marker from `INSERT` onward. Every other payload
/// shape (e.g. `{"state":"determined", ...}`) is the producer's to define and
/// is exercised in `fine_tune_queue.rs`, not here.
#[tokio::test]
async fn migration_026_adds_acceleration_report_column_with_tristate_backfill() {
    use jammi_db::catalog::model_repo::RegisterModelParams;
    use jammi_db::catalog::training_repo::CreateTrainingJobParams;
    use jammi_db::model_task::ModelTask;

    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).await.unwrap();
    let backend = BackendImpl::Sqlite(open_sqlite_backend(&dir.path().join("catalog.db")).await);

    // --- Fresh catalog: the column exists. ---
    let columns = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query::<_, String>(
                        "SELECT name FROM pragma_table_info('training_jobs')",
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
        columns.iter().any(|c| c == "acceleration_report"),
        "training_jobs must have 'acceleration_report' after migration 026; got {columns:?}"
    );

    // A freshly-submitted job carries the explicit pending marker, never NULL.
    catalog
        .register_model(RegisterModelParams {
            model_id: "acc-base",
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: None,
            config_json: None,
        })
        .await
        .unwrap();
    catalog
        .create_training_job(CreateTrainingJobParams {
            job_id: "acc-fresh",
            base_model_id: "acc-base::1",
            training_source: "src.csv",
            loss_type: "contrastive",
            hyperparams: "{}",
            kind: "fine_tune",
            training_spec: "{}",
        })
        .await
        .unwrap();
    let fresh = catalog.get_training_job("acc-fresh").await.unwrap();
    assert_eq!(
        fresh.acceleration_report.as_deref(),
        Some(r#"{"state":"pending"}"#),
        "a freshly submitted job must carry the explicit pending marker"
    );

    // --- Pre-026-migrated catalog: roll migration 026 back by hand to
    // reproduce a pre-migration schema, insert a row under that schema (no
    // acceleration_report column at all, so the eventual backfilled value is
    // SQL NULL, not an empty string or fabricated pending marker), then
    // reopen — the runner re-applies 026 and the legacy row reads back NULL
    // ("unknown"), never "pending" or any other fabricated state. ---
    backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "ALTER TABLE training_jobs DROP COLUMN acceleration_report",
                    &[],
                )
                .await?;
                tx.execute(
                    "DELETE FROM applied_migrations WHERE name = '026_acceleration_report'",
                    &[],
                )
                .await?;
                Ok(())
            })
        })
        .await
        .unwrap();
    backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "INSERT INTO training_jobs \
                     (job_id, base_model_id, training_source, loss_type, hyperparams, status, kind, \
                      training_spec) \
                     VALUES ('acc-legacy', 'acc-base::1', 'src.csv', 'contrastive', '{}', 'queued', \
                              'fine_tune', '{}')",
                    &[],
                )
                .await
            })
        })
        .await
        .unwrap();

    drop(catalog);
    let reopened = Catalog::open(dir.path()).await.unwrap();

    // Idempotent re-open: migrations run again with no error, and the column
    // is back.
    let columns_after = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query::<_, String>(
                        "SELECT name FROM pragma_table_info('training_jobs')",
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
        columns_after.iter().any(|c| c == "acceleration_report"),
        "training_jobs must regain 'acceleration_report' after re-applying migration 026"
    );

    let legacy = reopened.get_training_job("acc-legacy").await.unwrap();
    assert_eq!(
        legacy.acceleration_report, None,
        "a row born before migration 026 backfills to NULL ('unknown'), never a \
         fabricated pending or determined state"
    );

    // A genuinely idempotent re-open (no manual rollback this time — the
    // column and the `applied_migrations` row are both already in place):
    // running the migration set again must be a no-op, leaving the
    // backfilled legacy row's NULL exactly as it was.
    drop(reopened);
    let reopened_again = Catalog::open(dir.path()).await.unwrap();
    let legacy_after_second_reopen = reopened_again.get_training_job("acc-legacy").await.unwrap();
    assert_eq!(
        legacy_after_second_reopen.acceleration_report, None,
        "a second, genuinely idempotent reopen must not disturb the backfilled NULL"
    );
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

/// Regression guard (S8 of the deploy-shapes plan), GREEN before the esc-093
/// change: two `SqliteBackend` pools in one process race `migrate()` on one
/// fresh `catalog.db`. SQLite's backend opens every write transaction
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

/// Require-gate (KO-7) mirroring `recovery.rs`: an unset `JAMMI_TEST_PG_URL`
/// silently skips the Postgres arm by default, but a lane that sets
/// `JAMMI_REQUIRE_PG` must run it, so the skip becomes a loud failure there.
#[cfg(feature = "live-postgres-tests")]
fn require_live_pg(test_name: &str) {
    if std::env::var_os("JAMMI_REQUIRE_PG").is_some() {
        panic!(
            "{test_name}: JAMMI_REQUIRE_PG is set but JAMMI_TEST_PG_URL is unset -- this lane \
             must run the real Postgres arm, not skip it"
        );
    }
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

/// esc-093 oracle: two independent `PostgresBackend`s (separate pools) race
/// `migrate()` on a FRESH database created for this test alone. Both must
/// return `Ok` and the ledger must name every migration exactly once. Before
/// the advisory lock in `catalog::migrations::run` the loser failed with
/// SQLSTATE 42P07 (`relation already exists`) or 23505 on the ledger PK.
///
/// With `feature = "test-hooks"` the runner's ledger-read rendezvous is armed
/// so both runners are held between the ledger read and the first DDL for as
/// long as the lock lets them both get there (in the fixed world it lets only
/// one; the other blocks on the lock and passes through afterwards).
#[cfg(feature = "live-postgres-tests")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_migrate_on_fresh_postgres_is_safe() {
    use jammi_db::catalog::backend_postgres::PostgresBackend;
    use jammi_test_utils::pg_url_for_tests;

    const TEST_NAME: &str = "concurrent_migrate_on_fresh_postgres_is_safe";
    let Some(admin_url) = pg_url_for_tests() else {
        require_live_pg(TEST_NAME);
        return;
    };

    let admin = sqlx::PgPool::connect(&admin_url)
        .await
        .expect("connect to JAMMI_TEST_PG_URL");
    let db_name = format!("jammi_esc093_{}", uuid::Uuid::new_v4().simple());
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

/// Migration 027 adds the writer-lease columns to `result_tables` (esc-094):
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
