/// SQL DDL for the initial catalog schema: sources, models, embeddings, fine-tune jobs, evals.
pub(super) const MIGRATION_001_CORE_TABLES: &str = r#"
CREATE TABLE sources (
    source_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    source_type TEXT NOT NULL,
    uri         TEXT NOT NULL,
    schema_json TEXT,
    options     TEXT,
    created_at  TEXT NOT NULL DEFAULT (CAST(CURRENT_TIMESTAMP AS TEXT)),
    updated_at  TEXT NOT NULL DEFAULT (CAST(CURRENT_TIMESTAMP AS TEXT))
);

CREATE TABLE embedding_sets (
    set_id      TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL REFERENCES sources(source_id),
    model_id    TEXT NOT NULL,
    text_column TEXT NOT NULL,
    table_name  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    num_rows    INTEGER,
    dimensions  INTEGER,
    created_at  TEXT NOT NULL DEFAULT (CAST(CURRENT_TIMESTAMP AS TEXT)),
    updated_at  TEXT NOT NULL DEFAULT (CAST(CURRENT_TIMESTAMP AS TEXT))
);
CREATE INDEX idx_embedding_sets_source ON embedding_sets(source_id);
CREATE INDEX idx_embedding_sets_model  ON embedding_sets(model_id);
CREATE INDEX idx_embedding_sets_status ON embedding_sets(status);

CREATE TABLE models (
    model_id    TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    model_type  TEXT NOT NULL,
    task        TEXT NOT NULL,
    backend     TEXT,
    version     INTEGER NOT NULL DEFAULT 1,
    source      TEXT,
    dimensions  INTEGER,
    status      TEXT NOT NULL DEFAULT 'available',
    metadata    TEXT,
    created_at  TEXT NOT NULL DEFAULT (CAST(CURRENT_TIMESTAMP AS TEXT)),
    updated_at  TEXT NOT NULL DEFAULT (CAST(CURRENT_TIMESTAMP AS TEXT))
);
CREATE INDEX idx_models_type ON models(model_type);
CREATE INDEX idx_models_task ON models(task);

CREATE TABLE fine_tune_jobs (
    job_id          TEXT PRIMARY KEY,
    base_model_id   TEXT NOT NULL REFERENCES models(model_id),
    output_model_id TEXT,
    training_source TEXT NOT NULL,
    loss_type       TEXT NOT NULL,
    hyperparams     TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    metrics         TEXT,
    created_at      TEXT NOT NULL DEFAULT (CAST(CURRENT_TIMESTAMP AS TEXT)),
    updated_at      TEXT NOT NULL DEFAULT (CAST(CURRENT_TIMESTAMP AS TEXT))
);
CREATE INDEX idx_fine_tune_jobs_status ON fine_tune_jobs(status);

CREATE TABLE eval_runs (
    run_id      TEXT PRIMARY KEY,
    model_id    TEXT NOT NULL REFERENCES models(model_id),
    eval_type   TEXT NOT NULL,
    source_id   TEXT,
    metrics     TEXT NOT NULL,
    config      TEXT,
    created_at  TEXT NOT NULL DEFAULT (CAST(CURRENT_TIMESTAMP AS TEXT))
);
CREATE INDEX idx_eval_runs_model   ON eval_runs(model_id);
CREATE INDEX idx_eval_runs_type    ON eval_runs(eval_type);
CREATE INDEX idx_eval_runs_created ON eval_runs(created_at);

CREATE TABLE evidence_channels (
    channel_name    TEXT PRIMARY KEY,
    schema_json     TEXT NOT NULL,
    priority        INTEGER NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (CAST(CURRENT_TIMESTAMP AS TEXT))
);
INSERT INTO evidence_channels (channel_name, schema_json, priority) VALUES
    ('vector',    '{"similarity": "Float32"}', 1),
    ('inference', '{"inference_model": "Utf8", "inference_task": "Utf8", "inference_confidence": "Float32"}', 2);
"#;

/// Result tables: Parquet-backed embedding and inference outputs with sidecar ANN indexes.
pub(super) const MIGRATION_002_RESULT_TABLES: &str = r#"
CREATE TABLE result_tables (
    table_name      TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    task            TEXT NOT NULL,
    parquet_path    TEXT NOT NULL,
    index_path      TEXT,
    dimensions      INTEGER,
    distance_metric TEXT DEFAULT 'cosine',
    row_count       INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'building',
    key_column      TEXT,
    text_columns    TEXT,
    checkpoint      INTEGER,
    created_at      TEXT NOT NULL DEFAULT (CAST(CURRENT_TIMESTAMP AS TEXT)),
    completed_at    TEXT
);
CREATE INDEX idx_result_tables_source ON result_tables(source_id);
CREATE INDEX idx_result_tables_task ON result_tables(task);
CREATE INDEX idx_result_tables_status ON result_tables(status);
"#;

/// Phase 08: add golden_source, k, and status columns to eval_runs.
pub(super) const MIGRATION_003_EVAL_COLUMNS: &str = r#"
ALTER TABLE eval_runs ADD COLUMN golden_source TEXT;
ALTER TABLE eval_runs ADD COLUMN k INTEGER;
ALTER TABLE eval_runs ADD COLUMN status TEXT NOT NULL DEFAULT 'completed';
"#;

/// Drop unused embedding_sets table. Defined in MIGRATION_001 but never
/// referenced by any Rust code — no repo, no types, no callers.
pub(super) const MIGRATION_004_DROP_EMBEDDING_SETS: &str = r#"
DROP TABLE IF EXISTS embedding_sets;
"#;

/// Add a nullable `tenant_id` column to every catalog table, plus a B-tree
/// index per table. The column stores the canonical hyphenated lowercase
/// `Uuid::Display` form (SQLite has no native UUID type; `TEXT` is the
/// convention). Existing rows back-fill to NULL.
pub(super) const MIGRATION_005_TENANT_SCOPE: &str = r#"
ALTER TABLE sources           ADD COLUMN tenant_id TEXT;
ALTER TABLE models            ADD COLUMN tenant_id TEXT;
ALTER TABLE fine_tune_jobs    ADD COLUMN tenant_id TEXT;
ALTER TABLE eval_runs         ADD COLUMN tenant_id TEXT;
ALTER TABLE result_tables     ADD COLUMN tenant_id TEXT;
ALTER TABLE evidence_channels ADD COLUMN tenant_id TEXT;

CREATE INDEX idx_sources_tenant           ON sources(tenant_id);
CREATE INDEX idx_models_tenant            ON models(tenant_id);
CREATE INDEX idx_fine_tune_jobs_tenant    ON fine_tune_jobs(tenant_id);
CREATE INDEX idx_eval_runs_tenant         ON eval_runs(tenant_id);
CREATE INDEX idx_result_tables_tenant     ON result_tables(tenant_id);
CREATE INDEX idx_evidence_channels_tenant ON evidence_channels(tenant_id);
"#;

/// Normalise the JSON-blob `evidence_channels.schema_json` column into a
/// child `evidence_channel_columns` table. After this migration each
/// declared column is a row keyed by `(channel_name, column_name)`,
/// making the append-only invariant a database constraint rather than
/// a parser check.
pub(super) const MIGRATION_006_CHANNEL_COLUMNS: &str = r#"
CREATE TABLE evidence_channel_columns (
    channel_name    TEXT NOT NULL REFERENCES evidence_channels(channel_name),
    column_name     TEXT NOT NULL,
    column_type     TEXT NOT NULL,
    ordinal         INTEGER NOT NULL,
    declared_at     TEXT NOT NULL DEFAULT (CAST(CURRENT_TIMESTAMP AS TEXT)),
    PRIMARY KEY (channel_name, column_name)
);
CREATE UNIQUE INDEX idx_channel_cols_ordinal
    ON evidence_channel_columns(channel_name, ordinal);

INSERT INTO evidence_channel_columns(channel_name, column_name, column_type, ordinal) VALUES
    ('vector',    'similarity',            'Float32', 0),
    ('inference', 'inference_model',       'Utf8',    0),
    ('inference', 'inference_task',        'Utf8',    1),
    ('inference', 'inference_confidence',  'Float32', 2);

ALTER TABLE evidence_channels DROP COLUMN schema_json;
"#;

/// Migration 007 — mutable companion tables registry.
///
/// Adds two catalog tables that record user-declared mutable tables:
///   * `mutable_tables` — one row per registered table, carrying the Arrow
///     schema JSON, primary-key column list, optional tenant scope, free-form
///     user metadata, and a backend identifier (`'sqlite'` | `'postgres'`).
///   * `mutable_table_indexes` — secondary indexes per registered table.
///
/// The `tenant_id` column on `mutable_tables` is defined by migration 005;
/// Phase 2 stores `NULL` for every row it writes. Phase 3 wires the
/// session-attribute layer that populates it.
pub(super) const MIGRATION_007_MUTABLE_TABLES: &str = r#"
CREATE TABLE mutable_tables (
    id              TEXT PRIMARY KEY,
    schema_json     TEXT NOT NULL,
    primary_key     TEXT NOT NULL,
    tenant_id       TEXT,
    user_metadata   TEXT NOT NULL DEFAULT '{}',
    backend_kind    TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (CAST(CURRENT_TIMESTAMP AS TEXT)),
    updated_at      TEXT NOT NULL DEFAULT (CAST(CURRENT_TIMESTAMP AS TEXT))
);

CREATE INDEX idx_mutable_tables_tenant ON mutable_tables(tenant_id);

CREATE TABLE mutable_table_indexes (
    table_id        TEXT NOT NULL REFERENCES mutable_tables(id) ON DELETE CASCADE,
    index_name      TEXT NOT NULL,
    columns         TEXT NOT NULL,
    -- BIGINT (8-byte) so the column decodes as `i64` on both SQLite (stores
    -- INTEGER as variable-width up to 8 bytes) and Postgres (where `INTEGER`
    -- is INT4 / i32, incompatible with the engine's i64 read shape).
    is_unique       BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (table_id, index_name)
);
"#;

/// Migration 008 — `order_column` on `mutable_tables`.
///
/// `MutableTableDefinition` already carries an optional `order_column` field
/// validated at build time, but Phase 2's `mutable_tables` catalog row did
/// not persist it; every reload via `get_mutable_table` returned
/// `order_column: None`. Phase 4's trigger-stream replay path consumes
/// `order_column` via `MutableTableRegistry::scan_after`, so we round-trip
/// it now.
pub(super) const MIGRATION_008_MUTABLE_ORDER_COLUMN: &str = r#"
ALTER TABLE mutable_tables ADD COLUMN order_column TEXT;
"#;

/// Migration 010 — rename `source_type = 'local'` to `'file'`.
///
/// The `SourceType` enum on the Rust side has its `Local` variant
/// renamed `File` so the engine's file-shaped source driver can target
/// any `StorageUrl` (local disk, S3, GCS, Azure). The serde rename is
/// not back-compatible — there is no `#[serde(alias = "local")]` — so
/// every existing row whose JSON-encoded `source_type` column reads
/// `"local"` must be rewritten to `"file"` before the next catalog
/// read.
///
/// The column stores the JSON-encoded enum tag rather than the bare
/// snake-case string — `'"local"'` is what `serde_json::to_string` emits
/// for `SourceType::Local`. The UPDATE matches that exact spelling.
pub(super) const MIGRATION_010_RENAME_SOURCE_TYPE_LOCAL_TO_FILE: &str = r#"
UPDATE sources SET source_type = '"file"' WHERE source_type = '"local"';
"#;

/// Migration 009 — trigger-stream `topics` catalog table.
///
/// One row per registered topic. The Arrow schema is persisted as JSON
/// (matching the convention used by `mutable_repo`) — `BLOB` / `BYTEA`
/// would force dialect-aware DDL whereas `TEXT` decodes identically on
/// both backends. `backing_table` references the Phase-2 mutable table
/// that persists the event log; `ON DELETE RESTRICT` keeps the topic and
/// its backing table aligned. Tenant scope follows ADR-00 — nullable.
pub(super) const MIGRATION_009_TOPICS: &str = r#"
CREATE TABLE topics (
    topic_id          TEXT PRIMARY KEY,
    name              TEXT NOT NULL UNIQUE,
    schema_json       TEXT NOT NULL,
    tenant_id         TEXT,
    broker_metadata   TEXT NOT NULL DEFAULT '{}',
    backing_table     TEXT NOT NULL UNIQUE REFERENCES mutable_tables(id) ON DELETE RESTRICT,
    created_at        TEXT NOT NULL DEFAULT (CAST(CURRENT_TIMESTAMP AS TEXT))
);

CREATE INDEX idx_topics_tenant ON topics(tenant_id);
CREATE INDEX idx_topics_name ON topics(name);
"#;

/// Migration 011 — per-query eval persistence (spec J9).
///
/// Companion to `eval_runs`: one row per (eval_run_id, query_id), carrying the
/// per-query metric vector (Recall@{1,3,5,10}, MRR, nDCG, distance) as JSON and
/// an opaque `cohorts` JSON object (`'{}'` when none supplied). The aggregate
/// `eval_runs.metrics` path is untouched; this table is purely additive so
/// downstream consumers (J7 enterprise cohort aggregation) can re-aggregate
/// stored per-query arrays instead of re-running the eval.
///
/// The `_jammi_` name prefix marks the table substrate-owned (same reserved
/// convention as the J2 audit table): users may read it but the substrate owns
/// writes. `tenant_id` follows the catalog convention (migration 005) —
/// nullable `TEXT` holding the canonical hyphenated `Uuid::Display` form — and
/// reads are tenant-filtered exactly like `eval_runs`.
pub(super) const MIGRATION_011_EVAL_PER_QUERY: &str = r#"
CREATE TABLE _jammi_eval_per_query (
    eval_run_id TEXT NOT NULL,
    query_id    TEXT NOT NULL,
    cohorts     TEXT NOT NULL DEFAULT '{}',
    metrics     TEXT NOT NULL,
    tenant_id   TEXT,
    PRIMARY KEY (eval_run_id, query_id)
);
CREATE INDEX idx_eval_per_query_run    ON _jammi_eval_per_query(eval_run_id);
CREATE INDEX idx_eval_per_query_tenant ON _jammi_eval_per_query(tenant_id);
"#;

/// Migration 012 — scope topic-name uniqueness per tenant.
///
/// Migration 009 created `topics` with a global `name TEXT NOT NULL UNIQUE`.
/// That is wrong for the substrate's trigger-stream model: per-tenant topics
/// (`tenant: Some(_)`) are the norm, and two tenants must be able to hold the
/// same logical topic name (e.g. each tenant's own `jammi.audit.search.v1`).
/// Under the global unique, the first tenant to register a topic claims the
/// name process-wide and every other tenant's first registration fails with
/// `UNIQUE constraint failed: topics.name` — the J2 per-query audit log
/// crashes for the second tenant onward.
///
/// SQLite cannot drop or alter a column-level `UNIQUE` constraint in place, so
/// this migration rebuilds `topics` via the canonical
/// create-new / copy / drop / rename dance, replacing the global unique on
/// `name` with a composite `UNIQUE(name, tenant_id)` (mirroring the enterprise
/// `UNIQUE(tenant_id, name)` convention). The rebuilt table carries identical
/// columns, defaults, and the same FK / `ON DELETE RESTRICT` on
/// `backing_table`; only the uniqueness rule changes.
///
/// Per-tenant rows with the same name now coexist. SQLite treats NULLs as
/// distinct in UNIQUE constraints, so global (`tenant_id IS NULL`) topics with
/// the same name are *not* deduplicated by this constraint — but the only
/// globally-registered topics today are user-declared via the CLI/session/
/// Python surfaces, and the substrate-owned topics (audit, session lifecycle)
/// are all tenant-pinned, so this matches existing behaviour. The
/// `idx_topics_tenant` and `idx_topics_name` secondary indexes are recreated.
///
/// `PRAGMA foreign_keys` is OFF inside the migration transaction (sqlx opens
/// SQLite connections without it), so the temporary FK-less window during the
/// table swap does not trip referential checks; the `backing_table` FK is
/// restored on the rebuilt table.
pub(super) const MIGRATION_012_TOPICS_TENANT_UNIQUE: &str = r#"
CREATE TABLE topics_new (
    topic_id          TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    schema_json       TEXT NOT NULL,
    tenant_id         TEXT,
    broker_metadata   TEXT NOT NULL DEFAULT '{}',
    backing_table     TEXT NOT NULL UNIQUE REFERENCES mutable_tables(id) ON DELETE RESTRICT,
    created_at        TEXT NOT NULL DEFAULT (CAST(CURRENT_TIMESTAMP AS TEXT)),
    UNIQUE(name, tenant_id)
);

INSERT INTO topics_new (topic_id, name, schema_json, tenant_id, broker_metadata, backing_table, created_at)
    SELECT topic_id, name, schema_json, tenant_id, broker_metadata, backing_table, created_at FROM topics;

DROP TABLE topics;

ALTER TABLE topics_new RENAME TO topics;

CREATE INDEX idx_topics_tenant ON topics(tenant_id);
CREATE INDEX idx_topics_name ON topics(name);
"#;
