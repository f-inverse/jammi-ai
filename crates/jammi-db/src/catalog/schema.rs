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
///
/// `created_at` is always app-supplied via `lease::canonical_stamp_now` — no
/// SQL `DEFAULT`, so a forgotten bind fails loudly at `NOT NULL` instead of
/// silently persisting an unsortable/oldest row.
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
    created_at      TEXT NOT NULL,
    completed_at    TEXT
);
CREATE INDEX idx_result_tables_source ON result_tables(source_id);
CREATE INDEX idx_result_tables_task ON result_tables(task);
CREATE INDEX idx_result_tables_status ON result_tables(status);
"#;

/// Migration 003 — add golden_source, k, and status columns to eval_runs.
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
/// The `tenant_id` column on `mutable_tables` is defined by migration 005 and
/// populated from the session's bound tenant.
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
/// Persists `MutableTableDefinition`'s optional `order_column` so a reload via
/// `get_mutable_table` round-trips it; the trigger-stream replay path consumes
/// it via `MutableTableRegistry::scan_after`.
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
/// both backends. `backing_table` references the mutable table
/// that persists the event log; `ON DELETE RESTRICT` keeps the topic and
/// its backing table aligned. Tenant scope follows the engine's
/// tenant-identifier discipline — nullable (see
/// `docs/guide/src/philosophy.md#the-one-rule-everything-else-follows-from`).
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

/// Migration 013 — derived result tables (`kind` + `derived_from`).
///
/// A result table is either a *model output* (an embedding or inference table
/// produced by running a model over a source) or a *derivation of* another
/// result table. The similarity-graph edge table is the first derivation kind:
/// it is computed from an existing embedding table, carries no model, and has
/// no sidecar index.
///
/// `kind` is a discriminator orthogonal to `result_tables.task`: a `'model'`
/// row is resolved as an embedding/inference output, while a non-`'model'` row
/// is excluded from embedding-table resolution even though its `task` column
/// still names the source embedding's task. This keeps the `ModelTask` enum a
/// pristine catalogue of genuine model tasks — the kind, not a fake `ModelTask`
/// variant, is what marks a row non-resolvable as an embedding source.
///
/// `derived_from` references the source result table the derivation was
/// computed from (the embedding table for an edge table); it is `NULL` for
/// `'model'` rows. Existing rows back-fill to `kind = 'model'`, `derived_from
/// = NULL` via the column defaults.
pub(super) const MIGRATION_013_RESULT_TABLE_KIND: &str = r#"
ALTER TABLE result_tables ADD COLUMN kind TEXT NOT NULL DEFAULT 'model';
ALTER TABLE result_tables ADD COLUMN derived_from TEXT REFERENCES result_tables(table_name);
CREATE INDEX idx_result_tables_kind ON result_tables(kind);
"#;

/// Migration 011 — per-query eval persistence.
///
/// Companion to `eval_runs`: one row per (eval_run_id, query_id), carrying the
/// per-query metric vector (Recall@{1,3,5,10}, MRR, nDCG, distance) as JSON and
/// an opaque `cohorts` JSON object (`'{}'` when none supplied). The aggregate
/// `eval_runs.metrics` path is untouched; this table is purely additive so
/// downstream consumers can re-aggregate the stored per-query arrays by cohort
/// instead of re-running the eval.
///
/// The `_jammi_` name prefix marks the table substrate-owned (same reserved
/// convention as the audit table): users may read it but the substrate owns
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
/// `UNIQUE constraint failed: topics.name` — the per-tenant audit log
/// fails for the second tenant onward.
///
/// SQLite cannot drop or alter a column-level `UNIQUE` constraint in place, so
/// this migration rebuilds `topics` via the canonical
/// create-new / copy / drop / rename dance, replacing the global unique on
/// `name` with a composite `UNIQUE(name, tenant_id)`. The rebuilt table carries
/// identical columns, defaults, and the same FK / `ON DELETE RESTRICT` on
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
/// `PRAGMA foreign_keys` is ON for every connection this crate opens
/// (`backend_sqlite.rs`'s `SqliteConnectOptions::foreign_keys(true)`),
/// migration transaction included, so the rebuild's DROP of `topics`
/// mid-transaction fires every `ON DELETE`/`ON UPDATE` action any OTHER
/// table's FK declares against it — the `backing_table` FK this migration
/// itself restores on the rebuilt table is unaffected (that reference points
/// FROM `topics` TO `mutable_tables`, never the other way), but a future
/// rebuild of a table something else references by FK must account for
/// cascading actions firing during the swap, not assume a quiet FK-less
/// window (the false premise a `v1` design for a different migration was
/// refuted on).
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

/// Migration 014 — seed the `bm25` lexical-retrieval evidence channel.
///
/// The lexical (tantivy/BM25) sidecar contributes its rank and score on this
/// channel, the lexical peer of the `vector` channel's `similarity`. It shares
/// `inference`'s priority slot order only incidentally; what matters is that it
/// sorts after `vector` (priority 1) so a fused result's dense column precedes
/// its lexical columns. The channel carries the raw BM25 score plus the 0-based
/// lexical rank RRF fuses on — both caller-supplied, exactly as
/// `vector.similarity` is.
pub(super) const MIGRATION_014_BM25_CHANNEL: &str = r#"
INSERT INTO evidence_channels (channel_name, priority) VALUES ('bm25', 3);

INSERT INTO evidence_channel_columns(channel_name, column_name, column_type, ordinal) VALUES
    ('bm25', 'bm25_score', 'Float32', 0),
    ('bm25', 'bm25_rank',  'Int64',   1);
"#;

/// Migration 015 — lease-based job-queue columns on `fine_tune_jobs`.
///
/// Turns the table into a durable work queue a worker can poll. A queued job
/// is claimed by setting `status = 'running'`, stamping the claiming worker in
/// `claimed_by`, and writing a `lease_expires_at` deadline; the worker renews
/// the lease by heartbeating, and an expired lease lets the row be re-queued
/// (or failed once `attempts` is exhausted).
///
///   * `kind` — the training-job kind. Discriminates which trainer drives the
///     row (`'fine_tune'` for the contrastive-adapter path; future kinds add
///     their own values). Existing rows back-fill to `'fine_tune'` via the
///     column default.
///   * `claimed_by` — id of the worker holding the lease; `NULL` while the job
///     is queued or otherwise unclaimed.
///   * `lease_expires_at` — the lease deadline. An engine-clock UTC timestamp
///     stored in the canonical `%Y-%m-%dT%H:%M:%S%.6fZ` form (the same lexical
///     shape the repo writes), so that `lease_expires_at < $now` is a correct
///     text comparison on both SQLite and Postgres with no dialect-specific
///     interval arithmetic. `NULL` when the job is not leased.
///   * `attempts` — how many times the job has been claimed; incremented on
///     each claim and bounding reclaim retries.
///   * `training_spec` — reserved for a self-contained job specification a
///     worker can execute without re-deriving it from session state. Nullable;
///     no writer populates it yet.
///
/// `idx_fine_tune_jobs_claim` on `(status, lease_expires_at)` serves both the
/// oldest-queued claim scan and the expired-lease reclaim scan.
pub(super) const MIGRATION_015_FINE_TUNE_JOB_QUEUE: &str = r#"
ALTER TABLE fine_tune_jobs ADD COLUMN kind TEXT NOT NULL DEFAULT 'fine_tune';
ALTER TABLE fine_tune_jobs ADD COLUMN claimed_by TEXT;
ALTER TABLE fine_tune_jobs ADD COLUMN lease_expires_at TEXT;
ALTER TABLE fine_tune_jobs ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE fine_tune_jobs ADD COLUMN training_spec TEXT;
CREATE INDEX idx_fine_tune_jobs_claim ON fine_tune_jobs(status, lease_expires_at);
"#;

/// Migration 016 — rename the job table to `training_jobs`.
///
/// The job machinery carries more than one training kind (the `kind`
/// discriminator added in 015), so the table's name is generalised from
/// `fine_tune_jobs` to `training_jobs`. A behaviour-preserving rename: the
/// column set, constraints, and row contents are unchanged.
///
/// `ALTER TABLE … RENAME TO …` is portable across SQLite and Postgres. The
/// three indexes are renamed by dropping and recreating them against the new
/// table name — `DROP INDEX` / `CREATE INDEX` are portable, whereas SQLite has
/// no `ALTER INDEX … RENAME`. After the table rename the indexes still exist
/// under their old names (both backends carry indexes across a table rename),
/// so each is dropped before being recreated with its new name.
pub(super) const MIGRATION_016_RENAME_TRAINING_JOBS: &str = r#"
ALTER TABLE fine_tune_jobs RENAME TO training_jobs;
DROP INDEX idx_fine_tune_jobs_status;
DROP INDEX idx_fine_tune_jobs_tenant;
DROP INDEX idx_fine_tune_jobs_claim;
CREATE INDEX idx_training_jobs_status ON training_jobs(status);
CREATE INDEX idx_training_jobs_tenant ON training_jobs(tenant_id);
CREATE INDEX idx_training_jobs_claim ON training_jobs(status, lease_expires_at);
"#;

/// Migration 017 — a dedicated `artifact_path` column on `models`, outside the
/// descriptive `metadata` JSON blob (`base_model_id`, `config_json`).
/// [`MIGRATION_041_MODELS_ARTIFACT_REFERENCE`] renames it `external_location`.
pub(super) const MIGRATION_017_MODEL_ARTIFACT_PATH_COLUMN: &str = r#"
ALTER TABLE models ADD COLUMN artifact_path TEXT;
"#;

/// Migration 018 — make `eval_runs.model_id` nullable.
///
/// An eval run is only sometimes model-scoped. Embedding and inference evals
/// score a registered model's output, so they carry the catalog PK of that
/// model and the FK to `models(model_id)` is theirs to satisfy. A calibration
/// eval is different in kind: it scores a held-out predictive *distribution*
/// supplied in the golden source — `(mean, sd, outcome)` or `(draws, outcome)`
/// — and never loads or names a model. Forcing it to put *something* in a
/// `NOT NULL REFERENCES models(model_id)` column made it write a synthetic id
/// (the calibration shape, e.g. `gaussian::1`) for which no `models` row
/// exists, so the insert failed the foreign key at runtime — calibration eval
/// could not record a single run.
///
/// The fix is to let `model_id` be `NULL`: a model-scoped run records its
/// model's PK and the FK validates it; a calibration run records `NULL` and is
/// simply not model-scoped. The FK is *kept* — when a value is present it must
/// reference a real model — only the `NOT NULL` is dropped.
///
/// SQLite cannot drop a column-level `NOT NULL` in place, so this rebuilds
/// `eval_runs` via the canonical create-new / copy / drop / rename dance (the
/// same shape migration 012 uses for `topics`), preserving every other column,
/// default, and the `model_id` FK. The secondary indexes
/// (`model`, `type`, `created`, `tenant`) are recreated. `PRAGMA foreign_keys`
/// is ON for every connection this crate opens, migration transaction
/// included (`backend_sqlite.rs`'s `SqliteConnectOptions::foreign_keys(true)`)
/// — nothing else's FK references `eval_runs`, so this particular rebuild's
/// DROP fires no cascading action, but that is a fact about `eval_runs`'
/// referents, not a quiet FK-less window during the swap.
pub(super) const MIGRATION_018_EVAL_RUNS_MODEL_ID_NULLABLE: &str = r#"
CREATE TABLE eval_runs_new (
    run_id        TEXT PRIMARY KEY,
    model_id      TEXT REFERENCES models(model_id),
    eval_type     TEXT NOT NULL,
    source_id     TEXT,
    metrics       TEXT NOT NULL,
    config        TEXT,
    created_at    TEXT NOT NULL DEFAULT (CAST(CURRENT_TIMESTAMP AS TEXT)),
    golden_source TEXT,
    k             INTEGER,
    status        TEXT NOT NULL DEFAULT 'completed',
    tenant_id     TEXT
);

INSERT INTO eval_runs_new (run_id, model_id, eval_type, source_id, metrics, config, created_at, golden_source, k, status, tenant_id)
    SELECT run_id, model_id, eval_type, source_id, metrics, config, created_at, golden_source, k, status, tenant_id FROM eval_runs;

DROP TABLE eval_runs;

ALTER TABLE eval_runs_new RENAME TO eval_runs;

CREATE INDEX idx_eval_runs_model   ON eval_runs(model_id);
CREATE INDEX idx_eval_runs_type    ON eval_runs(eval_type);
CREATE INDEX idx_eval_runs_created ON eval_runs(created_at);
CREATE INDEX idx_eval_runs_tenant  ON eval_runs(tenant_id);
"#;

/// Migration 019 — normalize the `models.status` value set onto the canonical
/// [`ModelStatus`](super::status::ModelStatus) variants.
///
/// Migration 001 created `models` with `status TEXT NOT NULL DEFAULT
/// 'available'`, but `register_model` always writes an explicit `'registered'`
/// and the typed `ModelStatus` enum carries no `'available'` variant — so the
/// DDL default was a string the type system never names. A row could only carry
/// `'available'` if it were inserted by some path that omitted the column and
/// fell through to the default; no such path exists today, so this is defensive
/// rather than corrective. It rewrites any stray `'available'` to `'registered'`
/// (the canonical "registered, not loaded" state) so the on-disk value set is
/// total over the enum without introducing an `'available'` alias variant.
pub(super) const MIGRATION_019_NORMALIZE_MODEL_STATUS: &str = r#"
UPDATE models SET status = 'registered' WHERE status = 'available';
"#;

/// Migration 020 — tenant-qualify the evidence-channel identity.
///
/// Migration 005 added a `tenant_id` column to `evidence_channels` but no code
/// ever wrote or read it, and the channel name stayed a *global* `TEXT PRIMARY
/// KEY` (migration 001). `evidence_channel_columns` (migration 006) keyed on
/// `(channel_name, column_name)` and FK-referenced `evidence_channels(channel_name)`
/// — also tenant-blind. So two tenants registering a channel of the same name
/// collided on one global PK slot: tenant B's `register("X")` hit tenant A's row
/// (a cross-tenant collision, the same class as the one `models` had), and
/// `list()` returned every tenant's channels regardless of the bound tenant — a
/// cross-tenant read leak. The gRPC handlers already wrapped these calls in a
/// tenant `scoped(...)`, so the scope was a lie the repo ignored.
///
/// This migration makes the channel name unique PER TENANT by reshaping the
/// identity of both tables to carry `tenant_id`:
///   * `evidence_channels` gains a `UNIQUE (tenant_id, channel_name)` constraint
///     in place of the global `channel_name` PK.
///   * `evidence_channel_columns` carries `tenant_id`, gains a
///     `UNIQUE (tenant_id, channel_name, column_name)` constraint in place of
///     its old `(channel_name, column_name)` PK, and its FK becomes a composite
///     `REFERENCES evidence_channels(tenant_id, channel_name)` so the parent
///     link holds *within* a tenant. The per-channel ordinal-uniqueness index is
///     rebuilt as `(tenant_id, channel_name, ordinal)`.
///
/// `tenant_id` deliberately stays OUT of any PRIMARY KEY and is expressed as a
/// UNIQUE constraint instead — exactly as migration 012 did for `topics`.
/// Postgres (the CI-authoritative backend, which runs these same migration
/// constants) makes every PRIMARY-KEY column implicitly `NOT NULL`, so a
/// composite `PRIMARY KEY (tenant_id, …)` would reject the global
/// (`tenant_id IS NULL`) seed channels on insert. A UNIQUE constraint admits
/// NULLs (treated as distinct) on both backends, and Postgres accepts it as the
/// composite FK target the child table references.
///
/// SQLite cannot alter a PK or a column-level FK in place, so each table is
/// rebuilt via the canonical create-new / copy / drop / rename dance (the same
/// shape migrations 012 and 018 use). Every pre-existing row's `tenant_id` is
/// already `NULL` (no writer ever set it), so the copy carries that forward —
/// the seed channels (`vector`, `inference`, `bm25`) and any rows present stay
/// in the global (`tenant_id IS NULL`) namespace, preserving existing
/// default-channel behaviour. The child rows are parked in a temporary table
/// while the parent is swapped, then copied back with an explicit
/// `tenant_id = NULL` (the old child table had no `tenant_id` column); the child
/// is dropped *before* the parent and recreated *after* it, so the rebuild is
/// FK-safe on both backends regardless of FK-enforcement state.
///
/// Note on NULL semantics: both backends treat NULLs as DISTINCT in a UNIQUE
/// constraint, so the `UNIQUE (tenant_id, channel_name)` constraint alone does
/// not reject a duplicate *global* (`tenant_id IS NULL`) channel. A PARTIAL
/// UNIQUE INDEX on `channel_name WHERE tenant_id IS NULL` closes that gap: it
/// makes the database enforce global-channel-name uniqueness atomically, so two
/// concurrent unbound registrations of the same name cannot both commit.
/// Together the two constraints enforce per-namespace uniqueness with no
/// app-level race — the composite UNIQUE covers the non-NULL tenants, the
/// partial index covers the global namespace. `CREATE UNIQUE INDEX … WHERE …`
/// is accepted with identical syntax by SQLite (≥3.8.0) and Postgres, and runs
/// from this shared migration constant on both backends. The repo's `register`
/// keeps its explicit tenant-scoped existence check for a friendly
/// "already exists" error; the partial index (global) and the composite UNIQUE
/// (tenant-scoped) are the authoritative DB-level backstops, and a duplicate
/// surfaces through the same `BackendError::Constraint` → "already exists" path
/// in either namespace. Greenfield: no production rows to preserve beyond the
/// seeds.
pub(super) const MIGRATION_020_CHANNEL_TENANT_SCOPE: &str = r#"
CREATE TABLE evidence_channel_columns_old (
    channel_name    TEXT NOT NULL,
    column_name     TEXT NOT NULL,
    column_type     TEXT NOT NULL,
    ordinal         INTEGER NOT NULL,
    declared_at     TEXT NOT NULL
);

INSERT INTO evidence_channel_columns_old (channel_name, column_name, column_type, ordinal, declared_at)
    SELECT channel_name, column_name, column_type, ordinal, declared_at FROM evidence_channel_columns;

DROP TABLE evidence_channel_columns;

CREATE TABLE evidence_channels_new (
    tenant_id       TEXT,
    channel_name    TEXT NOT NULL,
    priority        INTEGER NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (CAST(CURRENT_TIMESTAMP AS TEXT)),
    UNIQUE (tenant_id, channel_name)
);

INSERT INTO evidence_channels_new (tenant_id, channel_name, priority, created_at)
    SELECT tenant_id, channel_name, priority, created_at FROM evidence_channels;

DROP TABLE evidence_channels;

ALTER TABLE evidence_channels_new RENAME TO evidence_channels;

CREATE INDEX idx_evidence_channels_tenant ON evidence_channels(tenant_id);

CREATE UNIQUE INDEX idx_evidence_channels_global_name
    ON evidence_channels(channel_name) WHERE tenant_id IS NULL;

CREATE TABLE evidence_channel_columns (
    tenant_id       TEXT,
    channel_name    TEXT NOT NULL,
    column_name     TEXT NOT NULL,
    column_type     TEXT NOT NULL,
    ordinal         INTEGER NOT NULL,
    declared_at     TEXT NOT NULL DEFAULT (CAST(CURRENT_TIMESTAMP AS TEXT)),
    UNIQUE (tenant_id, channel_name, column_name),
    FOREIGN KEY (tenant_id, channel_name)
        REFERENCES evidence_channels(tenant_id, channel_name)
);
CREATE UNIQUE INDEX idx_channel_cols_ordinal
    ON evidence_channel_columns(tenant_id, channel_name, ordinal);

INSERT INTO evidence_channel_columns(tenant_id, channel_name, column_name, column_type, ordinal, declared_at)
    SELECT NULL, channel_name, column_name, column_type, ordinal, declared_at FROM evidence_channel_columns_old;

DROP TABLE evidence_channel_columns_old;
"#;

/// Migration 021 — the materialization contract's catalog summary columns.
///
/// Every result table carries a `.materialization.json` sidecar attesting *how*
/// it was produced (a `definition_hash` over the producing descriptor + the
/// output-affecting environment) and the as-of `input_anchors` of every input
/// it read. Those two values are mirrored onto `result_tables` as an indexable
/// summary so `verify_materialization` and provenance queries need not open
/// every sidecar; the sidecar remains the full attestation.
///
/// Both columns are nullable: a row created before this migration (a
/// pre-contract table) carries `NULL` here and verifies as a truthful
/// `MissingManifest`, never a fabricated match. A row created after it always
/// carries both — they are written in the same transaction that flips
/// `building -> ready` (see `Catalog::promote_result_table_with_manifest`), so a
/// `ready` post-contract row with `NULL` summary columns is a torn write that
/// recovery reconciles, never a silent gap.
///
/// `input_anchors_json` is the single source of truth for input provenance: the
/// older `derived_from` column (migration 013) is an FK-lineage convenience that
/// is now a *view over* the anchors (a derivation's `ResultDigest` anchor names
/// the same source table), not a second, independently-maintained provenance
/// record.
pub(super) const MIGRATION_021_MATERIALIZATION_CONTRACT: &str = r#"
ALTER TABLE result_tables ADD COLUMN definition_hash    TEXT;
ALTER TABLE result_tables ADD COLUMN input_anchors_json TEXT;
"#;

/// Migration 022 — the cache-lookup index over `definition_hash`.
///
/// The sensing layer's cache lookup answers "is there already a `ready` table
/// produced by *this* definition?" — `WHERE definition_hash = $1 AND status =
/// 'ready'`. That predicate is the hot path of a recompute decision (asked once
/// per producer invocation), so `definition_hash` carries its own index; the
/// `status` arm is the existing low-cardinality filter the planner already
/// handles. Many rows can share a `definition_hash` (the same definition over
/// different input anchors, or re-emissions of the same inputs), so this is a
/// non-unique index — the *exact* `(definition_hash, input_anchors)` match is a
/// Rust post-filter over the decoded `input_anchors_json` of the candidate rows
/// the index narrows to.
///
/// `input_anchors_json` is deliberately **not** indexed: an anchor set is a
/// structured value matched for set equality in Rust, not by a SQL predicate, so
/// an index over its opaque JSON text would never be probed.
pub(super) const MIGRATION_022_DEFINITION_HASH_INDEX: &str = r#"
CREATE INDEX idx_result_tables_definition_hash ON result_tables(definition_hash);
"#;

/// Migration 023 — the sidecar-index storage-precision + rescore-oversample
/// columns.
///
/// `storage_precision` is the [`StoragePrecision`](crate::config::StoragePrecision)
/// (`"f32"` / `"f16"` / `"int8"`) an embedding table's ANN sidecar index was
/// built at — stamped once, at `create_table`, from the deployment's
/// [`AnnIndexConfig::storage_precision`](crate::config::AnnIndexConfig::storage_precision)
/// default, and read back verbatim by every later build/load of that table's
/// index (crash-recovery rebuild included) so a deployment-wide config change
/// never silently rebuilds an *existing* table's index at a different
/// precision than its catalog row promises. `oversample` is the matching
/// per-table default for the quantized-index retrieve→rescore multiplier
/// (`k * oversample` candidates retrieved before rescoring to `k` exact
/// results); a search request may still override it for that one call.
///
/// Both columns are nullable: a row created before this migration carries
/// `NULL` and is read back as the honest defaults (`F32`, the config
/// oversample default) — never a fabricated precision. A row created after
/// this migration always carries both, written in the same `INSERT` as every
/// other identity column.
pub(super) const MIGRATION_023_STORAGE_PRECISION: &str = r#"
ALTER TABLE result_tables ADD COLUMN storage_precision TEXT;
ALTER TABLE result_tables ADD COLUMN oversample        INTEGER;
"#;

/// Migration 024 — the claim-policy columns on `training_jobs`.
///
/// `priority` and `claimable` turn the job claim's scheduling policy into
/// catalog data instead of a fixed rule: `priority` breaks ties before
/// `created_at` (higher claims first), and `claimable` is a hold flag that
/// data-excludes a row from the claim without deleting it or introducing a
/// new status. Both default such that an existing or freshly enqueued row —
/// nothing writes a non-default value today — sits at `priority = 0,
/// claimable = TRUE`, where the claim ordering degenerates to plain
/// oldest-first FIFO: the defaults are chosen to preserve today's behavior
/// exactly, not merely approximately.
///
/// `claimable` is declared `BOOLEAN`, a conscious departure from this
/// table's earlier integer-flag columns: SQLite accepts the `BOOLEAN` type
/// name (numeric affinity) and the `TRUE` literal, and Postgres has a native
/// boolean, so the same DDL is portable and the column reads back as an
/// actual boolean on both backends — `DEFAULT 1` would not be, since Postgres
/// rejects an implicit integer-to-boolean default.
///
/// Two indexes now serve two distinct predicates over `training_jobs`:
/// `idx_training_jobs_claim (status, lease_expires_at)` (added in migration
/// 016) serves the expired-lease reclaim scan, and the new
/// `idx_training_jobs_claim_policy (status, claimable, priority DESC,
/// created_at)` serves the claim's own predicate and ordering. Neither
/// replaces the other.
pub(super) const MIGRATION_024_CLAIM_POLICY: &str = r#"
ALTER TABLE training_jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 0;
ALTER TABLE training_jobs ADD COLUMN claimable BOOLEAN NOT NULL DEFAULT TRUE;
CREATE INDEX idx_training_jobs_claim_policy
    ON training_jobs(status, claimable, priority DESC, created_at);
"#;

/// Migration 025 — the ANN index-segment set.
///
/// A table's ANN index is a *set* of immutable segments, one row per segment,
/// rather than the single sidecar bundle the dropped `result_tables.index_path`
/// column named. Each segment is a self-contained
/// [`SidecarIndex`](crate::index::sidecar::SidecarIndex) bundle over a disjoint
/// row subset; the reader merges them. Appending a batch of new rows writes a
/// new segment and inserts one row here, leaving every existing segment
/// untouched — the row-set grows without rebuilding the graph.
///
/// `segment_id` is a per-table sequence starting at `0` (the first segment a
/// fresh embedding table writes). The composite primary key
/// `(table_name, segment_id)` both enforces that a table never has two segments
/// at the same id — the collision an allocator's read-then-insert must retry
/// against under concurrent appends — and serves the table-scoped ordered
/// lookup its leading column covers, so no secondary index is added. The FK to
/// `result_tables` with `ON DELETE CASCADE` makes a table's segment rows a
/// dependent part of the table row: deleting the table reaps its segment set in
/// the same statement, and the file-level sidecar cleanup a caller runs
/// alongside enumerates this set before the cascade removes it.
///
/// `index_path` is the per-segment sidecar-bundle base URL (no extension; the
/// layout helpers append `.usearch` / `.rowmap` / `.manifest.json` / …).
/// `row_count` records the segment's own contribution, and `tenant_id` carries
/// the owning tenant stamped from the appending session (inherited from the
/// parent table), `NULL` for a GLOBAL table.
///
/// The prior single-index era is dropped, not migrated: `result_tables` loses
/// its `index_path` column in the same migration, so a pre-existing embedding
/// table's index is rebuilt from its Parquet on next recovery into a segment
/// set rather than read from the old column.
pub(super) const MIGRATION_025_INDEX_SEGMENTS: &str = r#"
CREATE TABLE index_segments (
    table_name TEXT NOT NULL REFERENCES result_tables(table_name) ON DELETE CASCADE,
    segment_id INTEGER NOT NULL,
    index_path TEXT NOT NULL,
    row_count  INTEGER NOT NULL DEFAULT 0,
    tenant_id  TEXT,
    created_at TEXT NOT NULL DEFAULT (CAST(CURRENT_TIMESTAMP AS TEXT)),
    PRIMARY KEY (table_name, segment_id)
);
ALTER TABLE result_tables DROP COLUMN index_path;
"#;

/// Migration 026 — the per-job acceleration-report column on `training_jobs`
/// (a compute precision that silently runs the unaccelerated eager
/// composition has no caller-visible, per-job signal). `acceleration_report`
/// carries an **opaque, self-describing JSON payload whose vocabulary the
/// payload's producer owns** — mirroring this table's own `training_spec`
/// and `metrics` columns' schema-at-the-producer deferral — not a closed
/// enum enumerated here. The catalog itself guarantees only two things about
/// this column, both mechanical rather than semantic:
///
///   - **absent / SQL `NULL`** — a legacy row written before this migration,
///     or (should it ever occur) a row this code never touched. Read back as
///     "unknown", never fabricated as any producer state.
///   - **`{"state":"pending"}`** — stamped by
///     [`Catalog::create_training_job`] at submission time: the job exists and
///     is queued/running, but no claimant has yet recorded a determination.
///     That sentence stops being true the instant the row goes terminal, so
///     each of the three terminal writes retires this exact marker to its own
///     `{"state":"undetermined","reason":…}` inside the SAME UPDATE that flips
///     the status, and the reclaim requeue arm resets it. Those four are the
///     only payloads the catalog itself writes;
///     [`crate::catalog::training_repo::TrainingJobRecord::acceleration_report`]
///     states the whole lifecycle in one place and is the sole place it is
///     stated.
///
/// Every other payload — most commonly `{"state":"determined", ...}`,
/// written by [`Catalog::record_acceleration_report`] once the claiming
/// worker resolves `(device, compiled capabilities, admission predicates)`
/// for this attempt, but not limited to it (a non-fine-tune job kind or a
/// pre-device-resolution failure path may record a different `"state"`) — is
/// the producer's to define and evolve; the catalog stores it byte-for-byte
/// and never inspects, validates, or enumerates its shape. A `"state"` key is
/// the convention every producer uses as its discriminant, not a contract
/// this migration or column enforces.
///
/// The column is nullable so a pre-migration row backfills to the honest
/// "unknown" state rather than a fabricated `pending`; every row created
/// after this migration always carries the explicit `pending` marker from
/// `INSERT` onward, so "no report yet" and "row pre-dates this feature" are
/// never confused for a row born under the new contract.
pub(super) const MIGRATION_026_ACCELERATION_REPORT: &str = r#"
ALTER TABLE training_jobs ADD COLUMN acceleration_report TEXT;
"#;

/// Migration 027 — the writer lease on a `building` result table.
///
/// A result table is published in two steps — bytes first, then a single
/// catalog row flip `building -> ready` — and until this migration nothing on
/// the row said *who* was producing it or *whether they were still alive*.
/// Startup recovery therefore reaped every `building` row it saw, including
/// one a live writer in another process was seconds from finishing, and the
/// writer's unguarded promote then flipped the reaped row to `ready` over
/// deleted bytes. The two columns make the row lease-owned, on the same
/// primitive `training_jobs` already uses ([`crate::catalog::lease`]):
///
///   * `writer_id` — the `ResultStore` instance that created the row
///     (`writer-{uuid}`); every transition on a `building` row is a
///     compare-and-set that names it. Promote and fail leave it in place as
///     history (they clear only the lease), so a later reader can still tell
///     which writer produced or abandoned the table.
///   * `lease_expires_at` — the writer's lease deadline, renewed by its
///     heartbeat; the same engine-clock `%Y-%m-%dT%H:%M:%S%.6fZ` text form as
///     `training_jobs.lease_expires_at`, so `lease_expires_at < $now` is a
///     correct comparison on both backends. `NULL` once the row is terminal —
///     and `NULL` on a `building` row created before this migration, which
///     recovery reads as "absent lease" and reconciles exactly as before.
///
/// Two `ALTER TABLE` statements rather than one: SQLite accepts a single
/// `ADD COLUMN` per statement. `idx_result_tables_lease` on
/// `(status, lease_expires_at)` serves recovery's expired-lease scan.
pub(super) const MIGRATION_027_RESULT_TABLE_LEASE: &str = r#"
ALTER TABLE result_tables ADD COLUMN writer_id TEXT;
ALTER TABLE result_tables ADD COLUMN lease_expires_at TEXT;
CREATE INDEX idx_result_tables_lease ON result_tables(status, lease_expires_at);
"#;

/// Migration 028 — `topics.next_offset`: the cross-process monotone offset
/// counter for the trigger-stream publish path.
///
/// `NULL` means "unseeded": a topic registered before this migration (or a
/// freshly-registered one) has never had an offset assigned through this
/// column. The publish-time locking statement
/// (`Publisher::publish_scoped`) seeds it from `COALESCE(MAX("_offset"),
/// -1) + 1` on the topic's own backing table the first time it is read,
/// inside the same row-locked `UPDATE` that bumps it — so the seed-then-bump
/// race is closed by the UPDATE's own row lock rather than an unlocked
/// read-then-write window. A fresh topic (empty backing table) seeds to `0`.
pub(super) const MIGRATION_028_TOPICS_NEXT_OFFSET: &str = r#"
ALTER TABLE topics ADD COLUMN next_offset BIGINT;
"#;

/// Migration 029 — the kind-agnostic `jobs` table, replacing `training_jobs`,
/// plus the process-liveness `instances` table and the claim-loop-membership
/// `workers` table.
///
/// `training_jobs` named exactly one kind of durable work; `jobs` generalises
/// the same claim/lease/reclaim machinery
/// ([`crate::catalog::jobs_repo`]) to every kind-agnostic unit of work a
/// worker can claim — training AND the compute verbs that opt into the queue.
/// Every column `training_jobs` carried survives under the same name except
/// `training_spec`, which becomes `spec` (still an opaque, producer-owned
/// tagged JSON payload — the rename only drops the "training" qualifier now
/// that the column serves every job kind), and `base_model_id`, which becomes
/// `model_ref` (unchanged meaning: the PK-keyed base model a training kind
/// fine-tunes from).
///
///   * `execution` — `'queued'` (claimed by the poll loop, `claim_next`) or
///     `'inline'` (claimed once, by id, in the submitting call itself —
///     `claim_by_id`; never selected by the poll loop's
///     `WHERE execution = 'queued'` predicate). Every legacy `training_jobs`
///     row copied below is `'queued'` — training was always poll-claimed.
///   * `spec` — the self-contained, producer-owned tagged JSON specification
///     a worker reconstructs the run from on a fresh process. `NOT NULL`
///     (every job, of every kind, carries one); a legacy row with no
///     `training_spec` backfills to `'{}'` rather than leaving a `NULL` a
///     `NOT NULL` column can never actually hold.
///   * `partial_result` — the name of the `result_tables` row this attempt is
///     (or already has) materialising into, written inside the SAME
///     transaction as that row's own INSERT
///     ([`crate::catalog::Catalog::create_result_table`]'s job-CAS) so a
///     reclaimed later attempt can find and adopt a predecessor's in-flight or
///     finished table instead of starting over. `NULL` until a producer that
///     stages through a result table sets it — a training kind, which
///     publishes a model artifact instead, leaves it `NULL` for the row's
///     whole lifetime.
///   * `result` — the tagged JSON terminal payload a successful attempt
///     writes at finish. Distinct from `partial_result` (an intermediate
///     dedupe key, a table name) and from the legacy `metrics` column (free
///     text); `NULL` until the job finishes.
///   * `error` — the terminal failure message, `NULL` until the job fails.
///     Replaces `training_jobs.metrics`' overloaded free-text blob (which
///     carried run-start metrics, the error message, AND `started_at`/
///     `completed_at`, disambiguated only by JSON-shape sniffing in
///     [`crate::catalog::jobs_repo`]'s row parser) with one column per
///     concern; a legacy row's `metrics` blob is not carried forward by this
///     migration — reclaim/fail bookkeeping starts fresh under the new
///     columns, `metrics`' history stays queryable only via
///     `applied_migrations`-gated raw SQL against a pre-migration backup.
///   * `progress_rows_done` / `progress_rows_total` / `progress_phase` — the
///     one checkpoint-progress vocabulary every job kind reports through,
///     `NULL` until the first checkpoint.
///   * `cancel_requested` — set by `CancelJob`; the executor observes it at a
///     checkpoint boundary. `NOT NULL DEFAULT FALSE`, never `NULL` — the
///     three-state need (never/requested/observed) is carried by this flag
///     plus the terminal `status`, not by a nullable tri-state column.
///   * `model_ref` — the PK-keyed base model a training kind fine-tunes from
///     (`training_jobs.base_model_id`, renamed; same `REFERENCES
///     models(model_id)`, now `ON DELETE SET NULL`). `NULL` for every compute
///     kind, which has no base model. The `ON DELETE SET NULL` arm exists
///     because [`crate::catalog::model_repo`]'s referential scan lets a
///     `delete_model` proceed past a `jobs.model_ref` edge once every
///     referencing row is terminal and past `[jobs] retention_days` — the
///     scan's OWN age predicate, not the database FK, decides whether the
///     DELETE is allowed to run at all; once it IS allowed to run, the FK
///     action is what keeps the surviving (necessarily exempted, by the same
///     scan) job rows from pointing at a row that has been deleted, without
///     the DELETE itself needing to touch `jobs`.
///   * `output_model_id` — the NAME-keyed model a training kind registers on
///     finish (`training_jobs.output_model_id`, unchanged; still FK-free —
///     the name is minted by the finalize CAS, not resolved against an
///     existing row).
///   * `model_source` — the NAME-keyed, FK-free `ModelSource` string a
///     compute verb resolves its model against (the exact vocabulary
///     `result_tables.model_id` already carries — no new resolution rule).
///     `NULL` for a training kind, which has no such source.
///
/// `priority` / `claimable` (migration 024's temporary operator hold) and
/// `acceleration_report` (migration 026) carry over unchanged in both name
/// and contract.
///
/// Indexes: `idx_jobs_claim(status, execution, claimable, priority DESC,
/// created_at)` is the claim predicate's own index (replacing
/// `idx_training_jobs_claim_policy`; the poll loop's `WHERE status = 'queued'
/// AND execution = 'queued' AND claimable ORDER BY priority DESC,
/// created_at`, one composite covering both the filter and the order, is
/// exactly what `idx_training_jobs_claim` and `idx_training_jobs_claim_policy`
/// jointly served before). `idx_jobs_lease(status, lease_expires_at)` serves
/// the expired-lease reclaim scan (replacing `idx_training_jobs_claim`,
/// migration 016's name for the same predicate over the renamed table).
///
/// `instances` is the process-liveness table: every process upserts its row
/// at construction and heartbeats `last_seen_at`; `instance_id` is a UUID
/// minted per process, `label` the (non-unique) `JAMMI_WORKER_ID` value.
/// `idx_instances_seen(last_seen_at)` serves the staleness scan an inline-job
/// reclaim and a construction sweep both run. `workers` is the
/// claim-loop-membership table: exactly the processes actually running the
/// claim loop get a row (`instance_id REFERENCES instances(instance_id) ON
/// DELETE CASCADE` — a worker row is dependent bookkeeping on its instance
/// row, never independently meaningful once the instance is gone), naming the
/// `kinds` it claims.
///
/// The copy below carries every `training_jobs` row forward as a `'queued'`
/// job (`kind` already discriminates `'fine_tune'` from any other training
/// kind the table ever held); `training_jobs` is then dropped — no shim, no
/// compatibility view. `REFERENCE_EDGES`
/// ([`crate::catalog::model_repo`]) and the artifact-reconcile read
/// ([`crate::store::reconcile`]) are updated in the SAME change to read
/// `jobs`, so no window exists where a live reader still expects
/// `training_jobs`.
pub(super) const MIGRATION_029_JOBS_INSTANCES_WORKERS: &str = r#"
CREATE TABLE jobs (
    job_id               TEXT PRIMARY KEY,
    kind                 TEXT NOT NULL,
    tenant_id            TEXT,
    status                TEXT NOT NULL DEFAULT 'queued',
    execution            TEXT NOT NULL CHECK (execution IN ('queued', 'inline')),
    spec                 TEXT NOT NULL,
    partial_result       TEXT,
    result                TEXT,
    error                 TEXT,
    progress_rows_done    BIGINT,
    progress_rows_total   BIGINT,
    progress_phase        TEXT,
    cancel_requested      BOOLEAN NOT NULL DEFAULT FALSE,
    model_ref             TEXT REFERENCES models(model_id) ON DELETE SET NULL,
    output_model_id       TEXT,
    model_source          TEXT,
    claimed_by            TEXT,
    attempts              INTEGER NOT NULL DEFAULT 0,
    lease_expires_at      TEXT,
    priority              INTEGER NOT NULL DEFAULT 0,
    claimable             BOOLEAN NOT NULL DEFAULT TRUE,
    acceleration_report   TEXT,
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL
);
CREATE INDEX idx_jobs_claim ON jobs(status, execution, claimable, priority DESC, created_at);
CREATE INDEX idx_jobs_lease ON jobs(status, lease_expires_at);

CREATE TABLE instances (
    instance_id  TEXT PRIMARY KEY,
    label        TEXT,
    host         TEXT,
    started_at   TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);
CREATE INDEX idx_instances_seen ON instances(last_seen_at);

CREATE TABLE workers (
    instance_id TEXT PRIMARY KEY REFERENCES instances(instance_id) ON DELETE CASCADE,
    kinds       TEXT NOT NULL
);

INSERT INTO jobs (
    job_id, kind, tenant_id, status, execution, spec, partial_result, result, error,
    progress_rows_done, progress_rows_total, progress_phase, cancel_requested,
    model_ref, output_model_id, model_source, claimed_by, attempts, lease_expires_at,
    priority, claimable, acceleration_report, created_at, updated_at
)
SELECT
    job_id, kind, tenant_id, status, 'queued', COALESCE(training_spec, '{}'), NULL, NULL, NULL,
    NULL, NULL, NULL, FALSE,
    base_model_id, output_model_id, NULL, claimed_by, attempts, lease_expires_at,
    priority, claimable, acceleration_report, created_at, updated_at
FROM training_jobs;

DROP TABLE training_jobs;
"#;

/// Migration 030 — durable per-tenant `SubmitJob` dedupe key:
/// `jobs.idempotency_key` is nullable (unset for every pre-existing caller
/// and every non-deduped `Catalog::submit_job` call, which never dedupes) and
/// `idx_jobs_tenant_idempotency_key` is a PARTIAL unique index — it indexes
/// only rows whose key is set, so a `NULL` key never collides with another
/// `NULL` key (unlimited un-keyed submissions stay legal). The index key is
/// `COALESCE(tenant_id, '')`, not the bare column: a bare `(tenant_id, key)`
/// unique index would let two GLOBAL (`tenant_id IS NULL`) submissions reuse
/// the same key freely, because SQL's default NULL-is-distinct-from-NULL rule
/// makes an ordinary unique index a no-op across NULL-tenant rows on both
/// Postgres and SQLite — the `COALESCE` folds every un-scoped tenant onto the
/// same `''` bucket so the SAME-tenant dedupe guarantee holds in single-tenant
/// deployments too, the common case. `Catalog::submit_job_deduped`'s `INSERT
/// ... ON CONFLICT (COALESCE(tenant_id, ''), idempotency_key) WHERE
/// idempotency_key IS NOT NULL DO NOTHING` targets this exact index.
pub(super) const MIGRATION_030_JOBS_IDEMPOTENCY_KEY: &str = r#"
ALTER TABLE jobs ADD COLUMN idempotency_key TEXT;
CREATE UNIQUE INDEX idx_jobs_tenant_idempotency_key
    ON jobs (COALESCE(tenant_id, ''), idempotency_key)
    WHERE idempotency_key IS NOT NULL;
"#;

/// Migration 031 — lease RELEASE bookkeeping and the worker's
/// lifecycle state.
///
///   * `jobs.releases` — how many times a claimant handed this job's lease
///     back on purpose (`Catalog::release_job_lease` /
///     `release_jobs_claimed_by`: a two-mode shutdown's RELEASE arm, never
///     an expiry). The reclaim cap compares `attempts - releases` against
///     the attempts limit, so a rollout storm of releases never burns the
///     cap a genuine crash does; `attempts` still bumps on every claim.
///   * `idx_jobs_kind_status(status, execution, kind)` — the index the
///     gauge sampler's `GROUP BY kind, status` over `execution = 'queued'`
///     rows reads (`Catalog::count_jobs_by_kind_status`) so a metrics tick
///     is an index-only aggregate, not a heap scan; `idx_jobs_claim` lacks
///     `kind`.
///   * `workers.state` — `warming` (the loop task's row exists but the
///     process is not yet warm / its worker gate is closed), `claiming`
///     (the claim loop is live), `draining` (a DRAIN is in progress). The
///     CHECK pins the vocabulary at the SQL edge. Every pre-existing
///     row is a live claimant, hence the default.
pub(super) const MIGRATION_031_JOBS_RELEASES_WORKERS_STATE: &str = r#"
ALTER TABLE jobs ADD COLUMN releases INTEGER NOT NULL DEFAULT 0;
CREATE INDEX idx_jobs_kind_status ON jobs(status, execution, kind);
ALTER TABLE workers ADD COLUMN state TEXT NOT NULL DEFAULT 'claiming'
    CHECK (state IN ('warming', 'claiming', 'draining'));
"#;

/// Migration 032 — versioned result tables: `result_table_versions`, the
/// `current_version` / `next_version` columns on `result_tables`, and the
/// producing-version stamp on `index_segments`.
///
/// One logical table keeps its `result_tables` row (the identity every
/// predicate renders); its refreshed states live in `result_table_versions`,
/// one row per version, immutable once `ready`. `result_tables.current_version`
/// is the published version (`NULL` = never refreshed = today's table with
/// zero behaviour change) and `next_version` the monotonic allocator: a
/// number is allocated exactly once by `UPDATE ... SET next_version =
/// next_version + 1` under the row lock, never reused — a failed or crashed
/// version keeps its number and expiry never decrements it. A version row is
/// lease-owned while `building` (`writer_id` / `lease_expires_at`, the same
/// text form as migration 027) and terminal as `ready` / `failed`;
/// `tenant_id` is inherited from the parent row in the same transaction;
/// `created_at` is app-supplied. `index_segments.version` stamps the
/// producing version on every segment appended by a refresh (`NULL` = a base
/// segment, read as the base version through a manifest).
///
/// One `ADD COLUMN` per statement (SQLite); the composite index on
/// `(status, lease_expires_at)` serves recovery's expired-lease scan exactly
/// as `idx_result_tables_lease` does for tables.
pub(super) const MIGRATION_032_RESULT_TABLE_VERSIONS: &str = r#"
CREATE TABLE result_table_versions (
    table_name       TEXT NOT NULL REFERENCES result_tables(table_name) ON DELETE CASCADE,
    version          INTEGER NOT NULL,
    parent_version   INTEGER,
    status           TEXT NOT NULL DEFAULT 'building',
    manifest_path    TEXT NOT NULL,
    identity         TEXT,
    live_rows        INTEGER,
    masked_rows      INTEGER,
    writer_id        TEXT,
    lease_expires_at TEXT,
    tenant_id        TEXT,
    created_at       TEXT NOT NULL,
    completed_at     TEXT,
    PRIMARY KEY (table_name, version)
);
CREATE INDEX idx_result_table_versions_lease ON result_table_versions(status, lease_expires_at);
ALTER TABLE result_tables ADD COLUMN current_version INTEGER;
ALTER TABLE result_tables ADD COLUMN next_version INTEGER NOT NULL DEFAULT 0;
ALTER TABLE index_segments ADD COLUMN version INTEGER;
"#;

/// Migration 033 — a materialization summary (`definition_hash`,
/// `input_anchors_json`) on `models`, with an index on it and one on
/// `models.artifact_path`.
///
/// Both columns and both indexes are retired by
/// [`MIGRATION_041_MODELS_ARTIFACT_REFERENCE`]: the summary is a property of
/// an artifact's bytes and lives on `model_artifacts`.
pub(super) const MIGRATION_033_MODEL_MATERIALIZATION: &str = r#"
ALTER TABLE models ADD COLUMN definition_hash TEXT;
ALTER TABLE models ADD COLUMN input_anchors_json TEXT;
CREATE INDEX idx_models_definition_hash ON models(definition_hash);
CREATE INDEX idx_models_artifact_path ON models(artifact_path);
"#;

/// Migration 034: the gang's training-set identity pair on `jobs` — the
/// `ArtifactDigest` of the coordinator's materialized `TrainingSet`
/// (`training_set_ref`) and the
/// `result_tables` NAME it materialized under (`training_set_location`),
/// job-scoped (never attempt-scoped), written/consulted only for
/// `world_size > 1`. Both columns start `NULL` on every existing and new
/// row; a `world_size == 1` job never touches them, so this migration
/// changes zero observable behaviour for every row it does not itself write.
///
/// **The pair is one fact, not two independently nullable columns (the
/// write-once CAS's own stop rule).** A `CHECK` constraint pins this at the
/// schema edge — preferred over "the only writer is the CAS" (Greenfield: a schema
/// constraint that makes the wrong shape UNREPRESENTABLE beats a mechanism
/// that merely avoids constructing it) — because both backends support a
/// same-table-column `CHECK` referenced from an `ALTER TABLE ADD COLUMN`
/// statement: SQLite (bundled `libsqlite3-sys` 0.30, SQLite ≥ 3.31) lifted
/// its historical "no other columns" restriction on an added column's
/// `CHECK` expression years ago, and Postgres has never had that
/// restriction. The constraint fires on every `UPDATE` that would leave the
/// pair split, not just on `INSERT` — a raw single-column write (never a
/// call site this program makes; the CAS is the only writer) is
/// refused by the database itself, not merely by convention.
pub(super) const MIGRATION_034_JOBS_TRAINING_SET_IDENTITY: &str = r#"
ALTER TABLE jobs ADD COLUMN training_set_ref TEXT;
ALTER TABLE jobs ADD COLUMN training_set_location TEXT
    CHECK ((training_set_ref IS NULL) = (training_set_location IS NULL));
"#;

/// Migration 035 — the gang-membership carrier on `instances` — `peer_addr` (the `host:port`
/// this process's peer/gang listener is reachable at) and `result_root` (the
/// VERBATIM configured result-table root, `JammiConfig::resolved_result_root`,
/// carried for display — the membership predicate does NOT consult it; it
/// compares root identity, migration 036).
///
/// Both columns are NULLABLE, with a shared meaning: `NULL` = "this process
/// never joins a gang" — every library/CLI process, and every server process
/// that never sets `[server] peer_advertise`. There is no paired `CHECK`
/// (unlike migration 034's `training_set_ref`/`training_set_location`): a row
/// with `peer_addr` set and `result_root` NULL is representable; the
/// pairing is a writer convention, not a schema constraint.
/// `Catalog::list_gang_members` never compares THIS column — it compares
/// `result_root_identity` (migration 036), the root's identity across
/// spellings; this verbatim column is carried for humans.
///
/// `Catalog::upsert_instance`/`Catalog::reregister_instance` are the only
/// writers (through `InstanceRegistration`, `catalog::instance`); every
/// pre-existing row (and every row written by a process with no
/// `peer_advertise`) carries both columns `NULL`, so this migration changes
/// zero observable behaviour for any row it does not itself write.
pub(super) const MIGRATION_035_INSTANCES_PEER_ADDR_RESULT_ROOT: &str = r#"
ALTER TABLE instances ADD COLUMN peer_addr TEXT;
ALTER TABLE instances ADD COLUMN result_root TEXT;
"#;

/// Migration 036 — `instances.result_root_identity` — the identity of the
/// member's result root ACROSS SPELLINGS (`catalog::instance::RootIdentity`,
/// derived once by the owning process from the verbatim `result_root` at
/// registration). This is the column `Catalog::list_gang_members` compares
/// (a byte-exact `=` against the caller's own identity); `result_root`
/// stays the verbatim configured spelling. Nullable: every pre-existing row,
/// and every row written by a process with no `[server] peer_advertise`,
/// carries NULL — and a NULL identity never matches, so such a row is never
/// a gang member. `Catalog::upsert_instance`/`Catalog::reregister_instance`
/// are the only writers, through `InstanceRegistration`.
pub(super) const MIGRATION_036_INSTANCES_RESULT_ROOT_IDENTITY: &str = r#"
ALTER TABLE instances ADD COLUMN result_root_identity TEXT;
"#;

/// Migration 037 — the assembly cooldown/counter on `jobs` — `assembly_failures` (the
/// running count of COUNTED assembly refusals this job has accumulated,
/// never reset except by a success) and `next_assembly_after` (when this
/// job's next assembly attempt may run at the earliest; `NULL` = no
/// cooldown pending), in the SAME representation
/// [`super::lease::LEASE_TS_FORMAT`]-family lease columns already use on
/// this table (`lease_expires_at`, migration 029): nullable `TEXT`, read and
/// written through the SAME lease-module helpers those columns use — never
/// a second clock source or a new stored representation.
/// [`super::jobs_repo::Catalog::claim_next`]'s cooldown conjunct in its
/// CANDIDATE subselect reuses [`super::lease::lease_expired_clause`]
/// VERBATIM against this column (`next_assembly_after IS NULL OR
/// next_assembly_after` has passed, on the backend's own clock — no bound
/// application timestamp on Postgres), and
/// [`super::jobs_repo::Catalog::record_assembly_outcome`] reuses
/// [`super::lease::lease_deadline_expr`] to stamp a fresh deadline.
/// `assembly_failures` starts at `0` for every existing and new row (an
/// unconfigured/pre-migration job has never failed assembly);
/// `next_assembly_after` starts `NULL` (no cooldown), so the cooldown
/// conjunct admits every pre-existing row — zero observable behaviour change for any row this
/// migration does not itself write.
///
/// [`super::jobs_repo::Catalog::record_assembly_outcome`] is the only
/// writer: it applies the exhaustive
/// [`super::jobs_repo::AssemblyOutcome`] rule (counted reasons bump
/// `assembly_failures`; every non-proceeding, in-assembly reason stamps a
/// fresh `next_assembly_after` via bounded exponential backoff on the new
/// count; a success resets both columns).
pub(super) const MIGRATION_037_JOBS_ASSEMBLY_FAILURES_NEXT_AFTER: &str = r#"
ALTER TABLE jobs ADD COLUMN assembly_failures INTEGER NOT NULL DEFAULT 0;
ALTER TABLE jobs ADD COLUMN next_assembly_after TEXT;
"#;

/// Migration 038 — `compute_cluster_state` — the catalog-backed cluster state a Ballista
/// scheduler role reads/writes through `catalog::compute_repo`, and
/// `workers.devices` — a per-worker device MIRROR, informational only, for
/// `ListWorkers`. DISTRIBUTOR-NEUTRAL: no `ballista` in any
/// identifier here, so the tables carry no distributor vocabulary into the
/// engine's own catalog.
///
/// * `compute_executors` — one row per registered compute executor:
///   `executor_id` (PK, the distributor's own executor identity — opaque to
///   this crate), `instance_id` (FK-shaped, not enforced — carried for
///   display/correlation only; see `devices` below for why it is NOT the
///   placement join key), `host`/`port`/`grpc_port` (the executor's two
///   listeners), `task_slots` (total capacity) and `available_slots`
///   (capacity not currently bound — `available_slots <= task_slots`
///   always, enforced by
///   `compute_repo::Catalog::adjust_compute_slots`/`bind_compute_slots`,
///   never by a schema `CHECK`, since a batch adjustment's intermediate
///   per-row state during its one transaction is not itself required to
///   satisfy the bound, only the committed result), `status` (the
///   executor's lifecycle state, `status::ComputeExecutorStatus`, which
///   the heartbeat write only ever moves forward), `heartbeat_at` (last
///   liveness signal, `TEXT` in the same lease-timestamp family other
///   catalog clocks use), `metadata` (free-form `TEXT`, e.g. the
///   distributor's own JSON executor description — never parsed by this
///   crate), and **`devices`** — the JSON `[{kind, ordinal}]`
///   (`instance.rs::DeviceFact`) THIS EXECUTOR itself registers with,
///   written by `compute_repo::Catalog::upsert_compute_executor` from
///   `ComputeExecutorRecord.devices`. This is the executor's OWN
///   registration fact and the placement join's ONLY authority —
///   `catalog::compute_repo::Catalog::list_compute_executor_devices` reads
///   THIS column directly, never `workers.devices` and never a join on
///   `instance_id`: an executor process and a `[worker]` process are
///   different roles that may run in different containers with different
///   device visibility, so the executor's own device claim, not another
///   table's, is what a placement decision must trust. `NOT NULL DEFAULT
///   '[]'` so an executor registered before this column existed (or one
///   that never names a device) reads back an empty list, never `NULL`.
/// * `compute_jobs` — one row per submitted compute job: `job_id` (PK,
///   opaque), `owner`, `status`, `queued_at`, `updated_at`. The execution
///   GRAPH itself has no serialisation in Ballista 54.1, so
///   it is deliberately NOT a column here — a scheduler restart keeps this
///   row's status but never revives the in-flight graph; jammi's own
///   reclaim re-runs the job, never Ballista's.
/// * `workers.devices` — the JSON `[{kind, ordinal}]` device inventory
///   `catalog::jobs_repo::Catalog::upsert_worker` writes from
///   `WorkerFacts.devices`; `NOT NULL DEFAULT '[]'` so every pre-existing
///   row (and every row a caller that still passes `&[]` writes) reads back
///   an empty device list rather than `NULL` — the same "additive column,
///   zero behaviour change for a row this migration does not itself write"
///   shape as 034's/037's own `ADD COLUMN ... DEFAULT`. This column is a
///   `ListWorkers` mirror ONLY, read back verbatim on
///   `jammi.v1.job.WorkerSummary.devices` (field 8, an additive field on
///   the frozen RPC surface — `crates/jammi-wire/proto/jammi/v1/job.proto`)
///   — `compute_executors.devices` above is the placement policy's sole
///   authority, never this one, because a `[worker]` row and a
///   compute-executor row describe potentially different processes.
///
/// Ordered after BOTH `035_instances_peer_addr_result_root` (the `instances`/
/// `workers` membership columns) and `037_jobs_assembly_failures_next_after`
/// (the `jobs` assembly columns) — asserted by
/// `tests/it/migrations.rs::migration_038_is_ordered_after_035_and_037_and_creates_compute_tables`
/// on both backends by relative position, never `.last()`.
pub(super) const MIGRATION_038_COMPUTE_CLUSTER_STATE: &str = r#"
CREATE TABLE compute_executors (
    executor_id     TEXT PRIMARY KEY,
    instance_id     TEXT NOT NULL,
    host            TEXT NOT NULL,
    port            INTEGER NOT NULL,
    grpc_port       INTEGER NOT NULL,
    task_slots      INTEGER NOT NULL,
    available_slots INTEGER NOT NULL,
    status          TEXT NOT NULL,
    heartbeat_at    TEXT NOT NULL,
    metadata        TEXT NOT NULL,
    devices         TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE compute_jobs (
    job_id     TEXT PRIMARY KEY,
    owner      TEXT NOT NULL,
    status     TEXT NOT NULL,
    queued_at  TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
ALTER TABLE workers ADD COLUMN devices TEXT NOT NULL DEFAULT '[]';
"#;

/// Migration 039 — the ONE canonical catalog stamp, enforced at the schema
/// edge on both backends.
///
/// Rows written before this migration can carry any of four shapes: SQLite
/// leases (`lease.rs`'s `LEASE_TS_FORMAT`, six fraction digits), Postgres
/// leases (the database's own `timestamptz`-cast-to-text rendering,
/// DateStyle/TimeZone-dependent), app-clock stamps with nine fraction digits
/// (on EITHER backend), and the `applied_migrations` ledger / `models`
/// (`CAST(CURRENT_TIMESTAMP AS TEXT)`, a backend-native rendering distinct
/// from all three). A reader cannot tolerate that mix (SQLite's lexical stamp
/// comparisons are only correct when every row shares one shape; a
/// Postgres-side `col::timestamptz` cast faults the whole statement on one
/// unreadable row). Every writer produces the one shape
/// (`catalog::lease::canonical_stamp_now` / `pg_canonical_stamp`), and this
/// migration (a) normalises every EXISTING value to that one shape and (b)
/// enforces the domain going forward so another shape can never appear.
///
/// The universe (every TEXT column a reader compares, `catalog::lease`'s
/// docs enumerate the readers): `jobs.{lease_expires_at,
/// next_assembly_after, updated_at, created_at}`, `instances.{last_seen_at,
/// started_at}`, `result_tables.{lease_expires_at, created_at}`,
/// `result_table_versions.lease_expires_at`, `compute_executors.
/// heartbeat_at`, `models.{created_at, updated_at}`, `applied_migrations.
/// applied_at`. `models.updated_at` and `applied_migrations.applied_at` join
/// the domain even though no SQL predicate compares them TODAY — one shape
/// everywhere, not "one shape everywhere a predicate happens to read it".
/// A domain column's legacy schema `DEFAULT (CAST(CURRENT_TIMESTAMP AS
/// TEXT))` is NOT a writer: its space-separated shape is refused at the
/// edge like any other non-canonical value, so a raw `INSERT` that omits
/// the column fails loudly (`<table>.<column>: not a canonical stamp`) and
/// every writer — the crate's own and any test fixture — stamps explicitly.
/// `compute_jobs.{queued_at, updated_at}`'s `queued_at` is a decimal epoch
/// counter (`jammi-ballista/src/cluster.rs`), not this shape, at all — out
/// of the universe entirely, a different domain. Every remaining `*_at`
/// column (`sources`, `eval_runs`, `evidence_channels`,
/// `evidence_channel_columns`, `mutable_tables`, `topics`, `index_segments`,
/// `result_table_versions.created_at`/`completed_at`,
/// `result_tables.completed_at`, `compute_jobs.updated_at`) keeps its
/// schema `DEFAULT`/hand-rolled writer shape: no reader compares it (an
/// `ORDER BY` or a `catalog::lease` helper), so its shape cannot corrupt an
/// ordering or fault a sweep — verified by grep over every SQL string this
/// crate authors, not assumed. The `fine_tune_jobs`/`training_jobs` lineage
/// (migrations 001-016) is dead — migration 029 drops the table entirely
/// after copying its rows into `jobs`, so there is no live column to
/// enforce; the copied VALUES persist forward into `jobs.created_at`/
/// `updated_at`, which this migration's `jobs` rewrite already covers.
///
/// SQLite (`MIGRATION_039_CANONICAL_STAMPS_SQLITE`): a `BEFORE INSERT` and a
/// `BEFORE UPDATE OF <col>` trigger per column, installed FIRST — SQLite has
/// no `ALTER TABLE ADD CONSTRAINT` and this crate does not rebuild tables to add one (`PRAGMA
/// foreign_keys` is ON for every connection this crate opens, so a rebuild's DROP fires cascading
/// FK actions against whatever else references the table — see migration 012's/018's docs). The
/// triggers check SHAPE ONLY (`GLOB` over a digit-class pattern — SQLite has no calendar parser, so
/// a month of `13` passes; `catalog::lease::LeaseFact::Undecodable` stays reachable for exactly
/// this reason, see its docs). Then ONE `UPDATE … SET c = CASE … END` per column rewrites by shape:
/// a nine-digit ISO fraction truncates to six; a space-separated, no-offset value (SQLite's own
/// historical
/// `CAST(CURRENT_TIMESTAMP AS TEXT)`, inherited by `jobs`/`applied_migrations`/
/// `models` — SQLite never receives Postgres's own WITH-offset rendering,
/// a separate installation) becomes `T`-separated with `.000000Z`; an
/// already-canonical value is excluded by the `WHERE` and left untouched;
/// anything else falls to the `CASE`'s `ELSE` arm — an identity assignment
/// that still fires the `BEFORE UPDATE OF <col>` trigger (SQLite fires an
/// UPDATE trigger because the column is named in `SET`, regardless of
/// whether the value actually changes), which then refuses it by name —
/// fail-closed, the ledger left at 038 (the transaction the migration runner
/// wraps every migration and the ledger read in never reaches the
/// `INSERT INTO applied_migrations` this migration's own entry needs), the
/// row intact (`RAISE(ABORT, …)` undoes only the failing statement, and the
/// runner's transaction is never committed once an `Err` propagates —
/// `CatalogBackend::transaction`'s sqlx `Transaction` rolls back on drop
/// without a commit).
///
/// Postgres (`MIGRATION_039_CANONICAL_STAMPS_POSTGRES`): the rewrite runs
/// FIRST (there is no pre-installed enforcement to install ahead of it) —
/// `UPDATE … SET c = to_char((…)::timestamptz AT TIME ZONE 'UTC', …)` per
/// column, where the `CASE` inside truncates a nine-digit ISO fraction to
/// six BEFORE the cast (`left(c, 26) || 'Z'`): `::timestamptz` ROUNDS a
/// cast's fractional seconds to Postgres's native microsecond resolution
/// rather than truncating, so an untruncated nine-digit value ending
/// `.999999600Z` would silently roll into the NEXT second — a different
/// instant than SQLite's own truncation-based rewrite produces for the
/// identical seed. Every other shape (Postgres's own with-offset rendering,
/// already at microsecond precision with an explicit, unambiguous zone) is
/// cast directly with no truncation. A value the cast cannot parse or whose
/// calendar fields are out of range faults the statement immediately
/// (SQLSTATE `22007`/`22008`) — fail-closed by the backend itself, the same
/// transactional guarantee as the SQLite arm, `BackendError::DomainViolation`
/// naming neither table nor column for this specific fault class (`backend.
/// rs`'s `classify` docs state the asymmetry). THEN `ALTER TABLE … ADD
/// CONSTRAINT sdchk__<table>__<column> CHECK (…)` per column, validating
/// every now-rewritten row: shape (the same regex the SQLite trigger's GLOB
/// expresses) AND `::timestamptz` cast-validity, via a `CASE … END`
/// expression rather than a bare `AND` — Postgres does not guarantee an
/// `AND`'s operand evaluation order, so a bare `c ~ '…' AND c::timestamptz
/// IS NOT NULL` could attempt the cast on shape-invalid text first and
/// fault on the WRONG SQLSTATE for a shape refusal. The constraint name is
/// `sdchk__<table>__<column>` (double-underscore separated: a table or
/// column name may itself carry a single underscore —
/// `result_table_versions`, `lease_expires_at` — which a single-underscore
/// split could not place unambiguously) so `classify`'s domain-violation
/// arm can recover which column refused a write without Postgres's own
/// protocol naming one directly (`backend.rs::parse_domain_violation_
/// constraint_name`).
pub(super) const MIGRATION_039_CANONICAL_STAMPS_SQLITE: &str = r#"
-- Install the schema-edge domain triggers FIRST, then rewrite:
-- the rewrite's own UPDATEs are validated by these same triggers, so a
-- legacy shape this migration cannot classify (the CASE's ELSE arm, an
-- identity assignment) is refused by the UPDATE itself -- fail-closed,
-- named by column, the ledger left at 038, the row intact.
CREATE TRIGGER IF NOT EXISTS trg_jobs_lease_expires_at_canonical_ins
BEFORE INSERT ON jobs
WHEN NEW.lease_expires_at IS NOT NULL AND NEW.lease_expires_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'jobs.lease_expires_at: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_jobs_lease_expires_at_canonical_upd
BEFORE UPDATE OF lease_expires_at ON jobs
WHEN NEW.lease_expires_at IS NOT NULL AND NEW.lease_expires_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'jobs.lease_expires_at: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_jobs_next_assembly_after_canonical_ins
BEFORE INSERT ON jobs
WHEN NEW.next_assembly_after IS NOT NULL AND NEW.next_assembly_after NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'jobs.next_assembly_after: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_jobs_next_assembly_after_canonical_upd
BEFORE UPDATE OF next_assembly_after ON jobs
WHEN NEW.next_assembly_after IS NOT NULL AND NEW.next_assembly_after NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'jobs.next_assembly_after: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_jobs_updated_at_canonical_ins
BEFORE INSERT ON jobs
WHEN NEW.updated_at IS NOT NULL AND NEW.updated_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'jobs.updated_at: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_jobs_updated_at_canonical_upd
BEFORE UPDATE OF updated_at ON jobs
WHEN NEW.updated_at IS NOT NULL AND NEW.updated_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'jobs.updated_at: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_jobs_created_at_canonical_ins
BEFORE INSERT ON jobs
WHEN NEW.created_at IS NOT NULL AND NEW.created_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'jobs.created_at: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_jobs_created_at_canonical_upd
BEFORE UPDATE OF created_at ON jobs
WHEN NEW.created_at IS NOT NULL AND NEW.created_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'jobs.created_at: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_instances_last_seen_at_canonical_ins
BEFORE INSERT ON instances
WHEN NEW.last_seen_at IS NOT NULL AND NEW.last_seen_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'instances.last_seen_at: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_instances_last_seen_at_canonical_upd
BEFORE UPDATE OF last_seen_at ON instances
WHEN NEW.last_seen_at IS NOT NULL AND NEW.last_seen_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'instances.last_seen_at: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_instances_started_at_canonical_ins
BEFORE INSERT ON instances
WHEN NEW.started_at IS NOT NULL AND NEW.started_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'instances.started_at: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_instances_started_at_canonical_upd
BEFORE UPDATE OF started_at ON instances
WHEN NEW.started_at IS NOT NULL AND NEW.started_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'instances.started_at: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_result_tables_lease_expires_at_canonical_ins
BEFORE INSERT ON result_tables
WHEN NEW.lease_expires_at IS NOT NULL AND NEW.lease_expires_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'result_tables.lease_expires_at: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_result_tables_lease_expires_at_canonical_upd
BEFORE UPDATE OF lease_expires_at ON result_tables
WHEN NEW.lease_expires_at IS NOT NULL AND NEW.lease_expires_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'result_tables.lease_expires_at: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_result_tables_created_at_canonical_ins
BEFORE INSERT ON result_tables
WHEN NEW.created_at IS NOT NULL AND NEW.created_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'result_tables.created_at: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_result_tables_created_at_canonical_upd
BEFORE UPDATE OF created_at ON result_tables
WHEN NEW.created_at IS NOT NULL AND NEW.created_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'result_tables.created_at: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_result_table_versions_lease_expires_at_canonical_ins
BEFORE INSERT ON result_table_versions
WHEN NEW.lease_expires_at IS NOT NULL AND NEW.lease_expires_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'result_table_versions.lease_expires_at: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_result_table_versions_lease_expires_at_canonical_upd
BEFORE UPDATE OF lease_expires_at ON result_table_versions
WHEN NEW.lease_expires_at IS NOT NULL AND NEW.lease_expires_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'result_table_versions.lease_expires_at: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_compute_executors_heartbeat_at_canonical_ins
BEFORE INSERT ON compute_executors
WHEN NEW.heartbeat_at IS NOT NULL AND NEW.heartbeat_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'compute_executors.heartbeat_at: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_compute_executors_heartbeat_at_canonical_upd
BEFORE UPDATE OF heartbeat_at ON compute_executors
WHEN NEW.heartbeat_at IS NOT NULL AND NEW.heartbeat_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'compute_executors.heartbeat_at: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_models_created_at_canonical_ins
BEFORE INSERT ON models
WHEN NEW.created_at IS NOT NULL AND NEW.created_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'models.created_at: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_models_created_at_canonical_upd
BEFORE UPDATE OF created_at ON models
WHEN NEW.created_at IS NOT NULL AND NEW.created_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'models.created_at: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_models_updated_at_canonical_ins
BEFORE INSERT ON models
WHEN NEW.updated_at IS NOT NULL AND NEW.updated_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'models.updated_at: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_models_updated_at_canonical_upd
BEFORE UPDATE OF updated_at ON models
WHEN NEW.updated_at IS NOT NULL AND NEW.updated_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'models.updated_at: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_applied_migrations_applied_at_canonical_ins
BEFORE INSERT ON applied_migrations
WHEN NEW.applied_at IS NOT NULL AND NEW.applied_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'applied_migrations.applied_at: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_applied_migrations_applied_at_canonical_upd
BEFORE UPDATE OF applied_at ON applied_migrations
WHEN NEW.applied_at IS NOT NULL AND NEW.applied_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'applied_migrations.applied_at: not a canonical stamp');
END;
-- Data rewrite: nine-digit ISO -> truncate to six; space-separated (no
-- offset -- SQLite never receives Postgres's own with-offset rendering,
-- a separate installation) -> 'T' + '.000000Z'; already canonical ->
-- excluded by the WHERE, untouched; anything else -> the CASE's identity
-- ELSE arm, refused by the trigger above.
UPDATE jobs SET lease_expires_at = CASE
    WHEN lease_expires_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]Z' THEN substr(lease_expires_at,1,19) || substr(lease_expires_at,20,7) || 'Z'
    WHEN lease_expires_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]' THEN substr(lease_expires_at,1,10) || 'T' || substr(lease_expires_at,12,8) || '.000000Z'
    ELSE lease_expires_at
END
WHERE lease_expires_at IS NOT NULL AND lease_expires_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z';
UPDATE jobs SET next_assembly_after = CASE
    WHEN next_assembly_after GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]Z' THEN substr(next_assembly_after,1,19) || substr(next_assembly_after,20,7) || 'Z'
    WHEN next_assembly_after GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]' THEN substr(next_assembly_after,1,10) || 'T' || substr(next_assembly_after,12,8) || '.000000Z'
    ELSE next_assembly_after
END
WHERE next_assembly_after IS NOT NULL AND next_assembly_after NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z';
UPDATE jobs SET updated_at = CASE
    WHEN updated_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]Z' THEN substr(updated_at,1,19) || substr(updated_at,20,7) || 'Z'
    WHEN updated_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]' THEN substr(updated_at,1,10) || 'T' || substr(updated_at,12,8) || '.000000Z'
    ELSE updated_at
END
WHERE updated_at IS NOT NULL AND updated_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z';
UPDATE jobs SET created_at = CASE
    WHEN created_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]Z' THEN substr(created_at,1,19) || substr(created_at,20,7) || 'Z'
    WHEN created_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]' THEN substr(created_at,1,10) || 'T' || substr(created_at,12,8) || '.000000Z'
    ELSE created_at
END
WHERE created_at IS NOT NULL AND created_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z';
UPDATE instances SET last_seen_at = CASE
    WHEN last_seen_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]Z' THEN substr(last_seen_at,1,19) || substr(last_seen_at,20,7) || 'Z'
    WHEN last_seen_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]' THEN substr(last_seen_at,1,10) || 'T' || substr(last_seen_at,12,8) || '.000000Z'
    ELSE last_seen_at
END
WHERE last_seen_at IS NOT NULL AND last_seen_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z';
UPDATE instances SET started_at = CASE
    WHEN started_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]Z' THEN substr(started_at,1,19) || substr(started_at,20,7) || 'Z'
    WHEN started_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]' THEN substr(started_at,1,10) || 'T' || substr(started_at,12,8) || '.000000Z'
    ELSE started_at
END
WHERE started_at IS NOT NULL AND started_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z';
UPDATE result_tables SET lease_expires_at = CASE
    WHEN lease_expires_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]Z' THEN substr(lease_expires_at,1,19) || substr(lease_expires_at,20,7) || 'Z'
    WHEN lease_expires_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]' THEN substr(lease_expires_at,1,10) || 'T' || substr(lease_expires_at,12,8) || '.000000Z'
    ELSE lease_expires_at
END
WHERE lease_expires_at IS NOT NULL AND lease_expires_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z';
UPDATE result_tables SET created_at = CASE
    WHEN created_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]Z' THEN substr(created_at,1,19) || substr(created_at,20,7) || 'Z'
    WHEN created_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]' THEN substr(created_at,1,10) || 'T' || substr(created_at,12,8) || '.000000Z'
    ELSE created_at
END
WHERE created_at IS NOT NULL AND created_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z';
UPDATE result_table_versions SET lease_expires_at = CASE
    WHEN lease_expires_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]Z' THEN substr(lease_expires_at,1,19) || substr(lease_expires_at,20,7) || 'Z'
    WHEN lease_expires_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]' THEN substr(lease_expires_at,1,10) || 'T' || substr(lease_expires_at,12,8) || '.000000Z'
    ELSE lease_expires_at
END
WHERE lease_expires_at IS NOT NULL AND lease_expires_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z';
UPDATE compute_executors SET heartbeat_at = CASE
    WHEN heartbeat_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]Z' THEN substr(heartbeat_at,1,19) || substr(heartbeat_at,20,7) || 'Z'
    WHEN heartbeat_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]' THEN substr(heartbeat_at,1,10) || 'T' || substr(heartbeat_at,12,8) || '.000000Z'
    ELSE heartbeat_at
END
WHERE heartbeat_at IS NOT NULL AND heartbeat_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z';
UPDATE models SET created_at = CASE
    WHEN created_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]Z' THEN substr(created_at,1,19) || substr(created_at,20,7) || 'Z'
    WHEN created_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]' THEN substr(created_at,1,10) || 'T' || substr(created_at,12,8) || '.000000Z'
    ELSE created_at
END
WHERE created_at IS NOT NULL AND created_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z';
UPDATE models SET updated_at = CASE
    WHEN updated_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]Z' THEN substr(updated_at,1,19) || substr(updated_at,20,7) || 'Z'
    WHEN updated_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]' THEN substr(updated_at,1,10) || 'T' || substr(updated_at,12,8) || '.000000Z'
    ELSE updated_at
END
WHERE updated_at IS NOT NULL AND updated_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z';
UPDATE applied_migrations SET applied_at = CASE
    WHEN applied_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]Z' THEN substr(applied_at,1,19) || substr(applied_at,20,7) || 'Z'
    WHEN applied_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]' THEN substr(applied_at,1,10) || 'T' || substr(applied_at,12,8) || '.000000Z'
    ELSE applied_at
END
WHERE applied_at IS NOT NULL AND applied_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z';
"#;

/// The Postgres arm of migration 039 — see
/// [`MIGRATION_039_CANONICAL_STAMPS_SQLITE`]'s docs for the shared design;
/// this constant's own doc block states only what differs.
pub(super) const MIGRATION_039_CANONICAL_STAMPS_POSTGRES: &str = r#"
-- Rewrite FIRST: a value this UPDATE cannot cast faults the statement
-- (fail-closed by the backend itself, the value named in its own error) --
-- there is no pre-installed enforcement to install ahead of it the way
-- SQLite's triggers are. A nine-digit ISO fraction is TRUNCATED to six
-- digits BEFORE the cast (never left to `::timestamptz`'s own rounding,
-- which can carry the value into the next second): PostgreSQL rounds a
-- cast's fractional seconds to its native microsecond resolution rather
-- than truncating, so a value like `...:59.999999600Z` would silently
-- become `...:00.000000Z` of the NEXT second one row apart from the SAME
-- instant SQLite's own six-digit truncation preserves as `...:59.999999Z`
-- -- explicit text truncation keeps both backends' migration converging on
-- the identical canonical value for the identical nine-digit seed.
-- Postgres's own with-offset rendering (`lease_expires_at`,
-- `next_assembly_after`, `applied_at`, `models.created_at`/`updated_at`'s
-- pre-039 `CAST(CURRENT_TIMESTAMP AS TEXT)`/`now()+interval` shapes) is
-- ALREADY at microsecond precision with an explicit offset, so it casts
-- directly with no truncation step and no ambiguity about which zone it
-- names.
UPDATE jobs SET lease_expires_at = to_char(
    (CASE
        WHEN lease_expires_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}Z$' THEN (left(lease_expires_at, 26) || 'Z')
        ELSE lease_expires_at
    END)::timestamptz AT TIME ZONE 'UTC',
    'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
)
WHERE lease_expires_at IS NOT NULL AND lease_expires_at !~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$';
UPDATE jobs SET next_assembly_after = to_char(
    (CASE
        WHEN next_assembly_after ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}Z$' THEN (left(next_assembly_after, 26) || 'Z')
        ELSE next_assembly_after
    END)::timestamptz AT TIME ZONE 'UTC',
    'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
)
WHERE next_assembly_after IS NOT NULL AND next_assembly_after !~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$';
UPDATE jobs SET updated_at = to_char(
    (CASE
        WHEN updated_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}Z$' THEN (left(updated_at, 26) || 'Z')
        ELSE updated_at
    END)::timestamptz AT TIME ZONE 'UTC',
    'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
)
WHERE updated_at IS NOT NULL AND updated_at !~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$';
UPDATE jobs SET created_at = to_char(
    (CASE
        WHEN created_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}Z$' THEN (left(created_at, 26) || 'Z')
        ELSE created_at
    END)::timestamptz AT TIME ZONE 'UTC',
    'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
)
WHERE created_at IS NOT NULL AND created_at !~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$';
UPDATE instances SET last_seen_at = to_char(
    (CASE
        WHEN last_seen_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}Z$' THEN (left(last_seen_at, 26) || 'Z')
        ELSE last_seen_at
    END)::timestamptz AT TIME ZONE 'UTC',
    'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
)
WHERE last_seen_at IS NOT NULL AND last_seen_at !~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$';
UPDATE instances SET started_at = to_char(
    (CASE
        WHEN started_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}Z$' THEN (left(started_at, 26) || 'Z')
        ELSE started_at
    END)::timestamptz AT TIME ZONE 'UTC',
    'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
)
WHERE started_at IS NOT NULL AND started_at !~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$';
UPDATE result_tables SET lease_expires_at = to_char(
    (CASE
        WHEN lease_expires_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}Z$' THEN (left(lease_expires_at, 26) || 'Z')
        ELSE lease_expires_at
    END)::timestamptz AT TIME ZONE 'UTC',
    'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
)
WHERE lease_expires_at IS NOT NULL AND lease_expires_at !~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$';
UPDATE result_tables SET created_at = to_char(
    (CASE
        WHEN created_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}Z$' THEN (left(created_at, 26) || 'Z')
        ELSE created_at
    END)::timestamptz AT TIME ZONE 'UTC',
    'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
)
WHERE created_at IS NOT NULL AND created_at !~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$';
UPDATE result_table_versions SET lease_expires_at = to_char(
    (CASE
        WHEN lease_expires_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}Z$' THEN (left(lease_expires_at, 26) || 'Z')
        ELSE lease_expires_at
    END)::timestamptz AT TIME ZONE 'UTC',
    'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
)
WHERE lease_expires_at IS NOT NULL AND lease_expires_at !~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$';
UPDATE compute_executors SET heartbeat_at = to_char(
    (CASE
        WHEN heartbeat_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}Z$' THEN (left(heartbeat_at, 26) || 'Z')
        ELSE heartbeat_at
    END)::timestamptz AT TIME ZONE 'UTC',
    'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
)
WHERE heartbeat_at IS NOT NULL AND heartbeat_at !~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$';
UPDATE models SET created_at = to_char(
    (CASE
        WHEN created_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}Z$' THEN (left(created_at, 26) || 'Z')
        ELSE created_at
    END)::timestamptz AT TIME ZONE 'UTC',
    'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
)
WHERE created_at IS NOT NULL AND created_at !~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$';
UPDATE models SET updated_at = to_char(
    (CASE
        WHEN updated_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}Z$' THEN (left(updated_at, 26) || 'Z')
        ELSE updated_at
    END)::timestamptz AT TIME ZONE 'UTC',
    'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
)
WHERE updated_at IS NOT NULL AND updated_at !~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$';
UPDATE applied_migrations SET applied_at = to_char(
    (CASE
        WHEN applied_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}Z$' THEN (left(applied_at, 26) || 'Z')
        ELSE applied_at
    END)::timestamptz AT TIME ZONE 'UTC',
    'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
)
WHERE applied_at IS NOT NULL AND applied_at !~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$';
-- The domain, enforced going forward: shape-before-cast (a CASE, not
-- a bare AND -- Postgres does not guarantee AND's operand evaluation order,
-- so a bare `c ~ '...' AND c::timestamptz IS NOT NULL` could attempt the
-- cast on shape-invalid text first).
ALTER TABLE jobs ADD CONSTRAINT sdchk__jobs__lease_expires_at CHECK (
    lease_expires_at IS NULL OR (CASE WHEN lease_expires_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$' THEN lease_expires_at::timestamptz IS NOT NULL ELSE false END)
);
ALTER TABLE jobs ADD CONSTRAINT sdchk__jobs__next_assembly_after CHECK (
    next_assembly_after IS NULL OR (CASE WHEN next_assembly_after ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$' THEN next_assembly_after::timestamptz IS NOT NULL ELSE false END)
);
ALTER TABLE jobs ADD CONSTRAINT sdchk__jobs__updated_at CHECK (
    updated_at IS NULL OR (CASE WHEN updated_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$' THEN updated_at::timestamptz IS NOT NULL ELSE false END)
);
ALTER TABLE jobs ADD CONSTRAINT sdchk__jobs__created_at CHECK (
    created_at IS NULL OR (CASE WHEN created_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$' THEN created_at::timestamptz IS NOT NULL ELSE false END)
);
ALTER TABLE instances ADD CONSTRAINT sdchk__instances__last_seen_at CHECK (
    last_seen_at IS NULL OR (CASE WHEN last_seen_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$' THEN last_seen_at::timestamptz IS NOT NULL ELSE false END)
);
ALTER TABLE instances ADD CONSTRAINT sdchk__instances__started_at CHECK (
    started_at IS NULL OR (CASE WHEN started_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$' THEN started_at::timestamptz IS NOT NULL ELSE false END)
);
ALTER TABLE result_tables ADD CONSTRAINT sdchk__result_tables__lease_expires_at CHECK (
    lease_expires_at IS NULL OR (CASE WHEN lease_expires_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$' THEN lease_expires_at::timestamptz IS NOT NULL ELSE false END)
);
ALTER TABLE result_tables ADD CONSTRAINT sdchk__result_tables__created_at CHECK (
    created_at IS NULL OR (CASE WHEN created_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$' THEN created_at::timestamptz IS NOT NULL ELSE false END)
);
ALTER TABLE result_table_versions ADD CONSTRAINT sdchk__result_table_versions__lease_expires_at CHECK (
    lease_expires_at IS NULL OR (CASE WHEN lease_expires_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$' THEN lease_expires_at::timestamptz IS NOT NULL ELSE false END)
);
ALTER TABLE compute_executors ADD CONSTRAINT sdchk__compute_executors__heartbeat_at CHECK (
    heartbeat_at IS NULL OR (CASE WHEN heartbeat_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$' THEN heartbeat_at::timestamptz IS NOT NULL ELSE false END)
);
ALTER TABLE models ADD CONSTRAINT sdchk__models__created_at CHECK (
    created_at IS NULL OR (CASE WHEN created_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$' THEN created_at::timestamptz IS NOT NULL ELSE false END)
);
ALTER TABLE models ADD CONSTRAINT sdchk__models__updated_at CHECK (
    updated_at IS NULL OR (CASE WHEN updated_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$' THEN updated_at::timestamptz IS NOT NULL ELSE false END)
);
ALTER TABLE applied_migrations ADD CONSTRAINT sdchk__applied_migrations__applied_at CHECK (
    applied_at IS NULL OR (CASE WHEN applied_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$' THEN applied_at::timestamptz IS NOT NULL ELSE false END)
);
"#;

/// Migration 040 — the model artifact as a catalog entity.
///
/// `model_artifacts` is the peer of `result_tables` for bytes under
/// `models/`: one row per bundle, keyed by the bundle's prefix (a full
/// [`crate::storage::StorageUrl`] string). The row is written `staged`
/// BEFORE the bundle's first byte, flips to `published` inside the finalize
/// transaction that attaches the first `models` row to it, and flips to
/// `reclaiming` only through the compare-and-set that licenses the byte
/// delete ([`crate::catalog::artifact_repo`]). `definition_hash` and
/// `input_anchors_json` are properties of the BYTES, so they live here, not
/// on a `models` row that merely names them. `staging_job_id` /
/// `staging_attempt` identify the writer while the row is `staged`: an
/// attempt-scoped bundle carries both, a job-scoped one (the durable resume
/// checkpoint, shared across a job's attempts) carries a `NULL` attempt.
/// Neither is a foreign key — a `jobs` row is retention-swept on its own
/// clock, and the identity is read only while the artifact is `staged`.
///
/// `models.artifact_prefix` is the ONE reference edge to an artifact: a
/// FOREIGN KEY to `model_artifacts(prefix)` with `ON DELETE RESTRICT`, so "is
/// this artifact referenced" is `EXISTS (SELECT 1 FROM models WHERE
/// artifact_prefix = $1)` and nothing else, and an artifact row cannot be
/// retired while any `models` row, in any tenant, still names it.
///
/// `created_at` joins the canonical-stamp domain (it is the grace clock a
/// reconcile pass ages an unreferenced artifact against), enforced at the
/// schema edge exactly like every other compared `*_at` column — a trigger
/// pair on SQLite, a `CHECK` on Postgres — which is the only reason this
/// migration's text differs per backend.
pub(super) const MIGRATION_040_MODEL_ARTIFACTS_SQLITE: &str = r#"
CREATE TABLE model_artifacts (
    prefix              TEXT PRIMARY KEY,
    tenant_id           TEXT,
    state               TEXT NOT NULL,
    definition_hash     TEXT,
    input_anchors_json  TEXT,
    staging_job_id      TEXT,
    staging_attempt     BIGINT,
    created_at          TEXT NOT NULL
);
CREATE INDEX idx_model_artifacts_definition ON model_artifacts(definition_hash);
CREATE INDEX idx_model_artifacts_staging ON model_artifacts(staging_job_id, staging_attempt);
ALTER TABLE models ADD COLUMN artifact_prefix TEXT
    REFERENCES model_artifacts(prefix) ON DELETE RESTRICT;
CREATE INDEX idx_models_artifact_prefix ON models(artifact_prefix);
CREATE TRIGGER IF NOT EXISTS trg_model_artifacts_created_at_canonical_ins
BEFORE INSERT ON model_artifacts
WHEN NEW.created_at IS NOT NULL AND NEW.created_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'model_artifacts.created_at: not a canonical stamp');
END;
CREATE TRIGGER IF NOT EXISTS trg_model_artifacts_created_at_canonical_upd
BEFORE UPDATE OF created_at ON model_artifacts
WHEN NEW.created_at IS NOT NULL AND NEW.created_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'model_artifacts.created_at: not a canonical stamp');
END;
"#;

/// The Postgres text of [`MIGRATION_040_MODEL_ARTIFACTS_SQLITE`].
pub(super) const MIGRATION_040_MODEL_ARTIFACTS_POSTGRES: &str = r#"
CREATE TABLE model_artifacts (
    prefix              TEXT PRIMARY KEY,
    tenant_id           TEXT,
    state               TEXT NOT NULL,
    definition_hash     TEXT,
    input_anchors_json  TEXT,
    staging_job_id      TEXT,
    staging_attempt     BIGINT,
    created_at          TEXT NOT NULL
);
CREATE INDEX idx_model_artifacts_definition ON model_artifacts(definition_hash);
CREATE INDEX idx_model_artifacts_staging ON model_artifacts(staging_job_id, staging_attempt);
ALTER TABLE models ADD COLUMN artifact_prefix TEXT
    REFERENCES model_artifacts(prefix) ON DELETE RESTRICT;
CREATE INDEX idx_models_artifact_prefix ON models(artifact_prefix);
ALTER TABLE model_artifacts ADD CONSTRAINT sdchk__model_artifacts__created_at CHECK (
    created_at IS NULL OR (CASE WHEN created_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$' THEN created_at::timestamptz IS NOT NULL ELSE false END)
);
"#;

/// Migration 041 — a `models` row names its bytes one of two typed ways.
///
/// `models.artifact_path` carried two unrelated things: the prefix of an
/// engine-produced bundle under `models/`, and the local directory of a
/// directly-registered base model. They split into `artifact_prefix` (the
/// foreign key to `model_artifacts`, migration 040) and `external_location`
/// (the renamed column), and a row never carries both.
///
/// **Backfill rule.** A row is engine-produced exactly when `model_type IN
/// ('fine-tuned', 'context-predictor')` — the two types a training job
/// registers (a retained epoch checkpoint is a `fine-tuned` row). Each
/// distinct non-`NULL` `artifact_path` among those rows becomes ONE
/// `published` `model_artifacts` row: `prefix` the path, `tenant_id` the
/// `MIN` over the rows naming it, `definition_hash` / `input_anchors_json`
/// the `MAX` over them (every row naming one prefix carries the same
/// summary or none), `created_at` the earliest, and no staging identity. A
/// prefix that already has an artifact row keeps it. Those rows then
/// reference it through `artifact_prefix` and lose their
/// `external_location`. Every other row's path is a base-model directory and
/// stays in `external_location`.
///
/// The materialization summary is a property of the bytes and lives on the
/// artifact row, so `models.definition_hash` / `models.input_anchors_json`
/// and the two migration-033 indexes are dropped.
pub(super) const MIGRATION_041_MODELS_ARTIFACT_REFERENCE: &str = r#"
INSERT INTO model_artifacts
    (prefix, tenant_id, state, definition_hash, input_anchors_json, created_at)
SELECT artifact_path, MIN(tenant_id), 'published', MAX(definition_hash),
       MAX(input_anchors_json), MIN(created_at)
FROM models
WHERE artifact_path IS NOT NULL
  AND model_type IN ('fine-tuned', 'context-predictor')
GROUP BY artifact_path
ON CONFLICT(prefix) DO NOTHING;
UPDATE models SET artifact_prefix = artifact_path
WHERE artifact_path IS NOT NULL
  AND model_type IN ('fine-tuned', 'context-predictor');
DROP INDEX idx_models_artifact_path;
DROP INDEX idx_models_definition_hash;
ALTER TABLE models RENAME COLUMN artifact_path TO external_location;
UPDATE models SET external_location = NULL WHERE artifact_prefix IS NOT NULL;
ALTER TABLE models DROP COLUMN definition_hash;
ALTER TABLE models DROP COLUMN input_anchors_json;
"#;

/// Migration 042 — a `result_tables` row can be built to replace another, and
/// a row's dependents follow its name.
///
/// `replaces` names the `ready` row a `building` row supersedes when it is
/// promoted: `CREATE OR REPLACE TABLE <name> AS` builds the new artifact
/// under a name of its own with `replaces = '<name>'`, and the promote
/// compare-and-set removes the old row and moves the new one onto the name
/// in one transaction, so a reader of the name resolves the old table or the
/// new one and never none. `NULL` for every row published under its own
/// name. Recovery never promotes a row that replaces another — a replacement
/// nobody is driving is reaped, the table it was to replace left as it is.
///
/// Moving a row onto a name is an update of the primary key, and a row's
/// segment (`index_segments`) and version (`result_table_versions`) rows
/// reference it by that key: their foreign keys gain `ON UPDATE CASCADE`,
/// beside the `ON DELETE CASCADE` they carry, so the dependents of a renamed
/// row follow it in the same statement. Postgres swaps the constraints in
/// place under the names it gave them (`<table>_<column>_fkey`); SQLite
/// cannot alter a constraint, so both child tables are rebuilt by the
/// create-new / copy / drop / rename dance of migration 012 — nothing
/// references either child by foreign key, so the drop cascades into
/// nothing — with their index and the migration-039 stamp triggers
/// recreated on the rebuilt table (a trigger is dropped with its table).
pub(super) const MIGRATION_042_RESULT_TABLE_REPLACEMENT_SQLITE: &str = r#"
ALTER TABLE result_tables ADD COLUMN replaces TEXT;
CREATE TABLE index_segments_new (
    table_name TEXT NOT NULL REFERENCES result_tables(table_name) ON DELETE CASCADE ON UPDATE CASCADE,
    segment_id INTEGER NOT NULL,
    index_path TEXT NOT NULL,
    row_count  INTEGER NOT NULL DEFAULT 0,
    tenant_id  TEXT,
    created_at TEXT NOT NULL DEFAULT (CAST(CURRENT_TIMESTAMP AS TEXT)),
    version    INTEGER,
    PRIMARY KEY (table_name, segment_id)
);
INSERT INTO index_segments_new (table_name, segment_id, index_path, row_count, tenant_id, created_at, version)
    SELECT table_name, segment_id, index_path, row_count, tenant_id, created_at, version FROM index_segments;
DROP TABLE index_segments;
ALTER TABLE index_segments_new RENAME TO index_segments;
CREATE TABLE result_table_versions_new (
    table_name       TEXT NOT NULL REFERENCES result_tables(table_name) ON DELETE CASCADE ON UPDATE CASCADE,
    version          INTEGER NOT NULL,
    parent_version   INTEGER,
    status           TEXT NOT NULL DEFAULT 'building',
    manifest_path    TEXT NOT NULL,
    identity         TEXT,
    live_rows        INTEGER,
    masked_rows      INTEGER,
    writer_id        TEXT,
    lease_expires_at TEXT,
    tenant_id        TEXT,
    created_at       TEXT NOT NULL,
    completed_at     TEXT,
    PRIMARY KEY (table_name, version)
);
INSERT INTO result_table_versions_new (table_name, version, parent_version, status, manifest_path, identity, live_rows, masked_rows, writer_id, lease_expires_at, tenant_id, created_at, completed_at)
    SELECT table_name, version, parent_version, status, manifest_path, identity, live_rows, masked_rows, writer_id, lease_expires_at, tenant_id, created_at, completed_at FROM result_table_versions;
DROP TABLE result_table_versions;
ALTER TABLE result_table_versions_new RENAME TO result_table_versions;
CREATE INDEX idx_result_table_versions_lease ON result_table_versions(status, lease_expires_at);
CREATE TRIGGER trg_result_table_versions_lease_expires_at_canonical_ins
BEFORE INSERT ON result_table_versions
WHEN NEW.lease_expires_at IS NOT NULL AND NEW.lease_expires_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'result_table_versions.lease_expires_at: not a canonical stamp');
END;
CREATE TRIGGER trg_result_table_versions_lease_expires_at_canonical_upd
BEFORE UPDATE OF lease_expires_at ON result_table_versions
WHEN NEW.lease_expires_at IS NOT NULL AND NEW.lease_expires_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
BEGIN
    SELECT RAISE(ABORT, 'result_table_versions.lease_expires_at: not a canonical stamp');
END;
"#;

/// The Postgres arm of migration 042 — see
/// [`MIGRATION_042_RESULT_TABLE_REPLACEMENT_SQLITE`].
pub(super) const MIGRATION_042_RESULT_TABLE_REPLACEMENT_POSTGRES: &str = r#"
ALTER TABLE result_tables ADD COLUMN replaces TEXT;
ALTER TABLE index_segments DROP CONSTRAINT index_segments_table_name_fkey;
ALTER TABLE index_segments ADD CONSTRAINT index_segments_table_name_fkey
    FOREIGN KEY (table_name) REFERENCES result_tables(table_name) ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE result_table_versions DROP CONSTRAINT result_table_versions_table_name_fkey;
ALTER TABLE result_table_versions ADD CONSTRAINT result_table_versions_table_name_fkey
    FOREIGN KEY (table_name) REFERENCES result_tables(table_name) ON DELETE CASCADE ON UPDATE CASCADE;
"#;

/// Migration 043: every `models` row names the backend that runs it.
///
/// Every engine writer (registration, a job's finalize) records one, and the
/// catalog's typed read refuses a row without one, so the schema says so
/// too. The VALUE set stays the typed read's (`ModelBackendKind`), as
/// `models.task`'s does — the schema holds presence only. Postgres alters
/// the column in place; SQLite cannot add a constraint to a column, so it
/// refuses a missing backend with the trigger pair migration 039 uses for
/// canonical stamps.
pub(super) const MIGRATION_043_MODELS_BACKEND_REQUIRED_SQLITE: &str = r#"
CREATE TRIGGER trg_models_backend_required_ins
BEFORE INSERT ON models
WHEN NEW.backend IS NULL
BEGIN
    SELECT RAISE(ABORT, 'models.backend: every model names the backend that runs it');
END;
CREATE TRIGGER trg_models_backend_required_upd
BEFORE UPDATE OF backend ON models
WHEN NEW.backend IS NULL
BEGIN
    SELECT RAISE(ABORT, 'models.backend: every model names the backend that runs it');
END;
"#;

/// The Postgres arm of migration 043 — see
/// [`MIGRATION_043_MODELS_BACKEND_REQUIRED_SQLITE`].
pub(super) const MIGRATION_043_MODELS_BACKEND_REQUIRED_POSTGRES: &str = r#"
ALTER TABLE models ALTER COLUMN backend SET NOT NULL;
"#;
