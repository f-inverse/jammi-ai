//! End-to-end migration tests. Asserts via the new `applied_migrations`
//! ledger and direct sqlx queries against the on-disk catalog.

use jammi_db::catalog::backend::{BackendImpl, TxOptions};
use jammi_db::catalog::backend_sqlite::SqliteBackend;
use jammi_db::catalog::Catalog;
use tempfile::tempdir;

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
    assert_eq!(
        names,
        vec![
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
        ]
    );
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
