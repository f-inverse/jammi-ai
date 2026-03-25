/// SQL DDL for the initial catalog schema: sources, models, embeddings, fine-tune jobs, evals.
pub(super) const MIGRATION_001_CORE_TABLES: &str = r#"
CREATE TABLE sources (
    source_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    source_type TEXT NOT NULL,
    uri         TEXT NOT NULL,
    schema_json TEXT,
    options     TEXT,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
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
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
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
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
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
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX idx_fine_tune_jobs_status ON fine_tune_jobs(status);

CREATE TABLE eval_runs (
    run_id      TEXT PRIMARY KEY,
    model_id    TEXT NOT NULL REFERENCES models(model_id),
    eval_type   TEXT NOT NULL,
    source_id   TEXT,
    metrics     TEXT NOT NULL,
    config      TEXT,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX idx_eval_runs_model   ON eval_runs(model_id);
CREATE INDEX idx_eval_runs_type    ON eval_runs(eval_type);
CREATE INDEX idx_eval_runs_created ON eval_runs(created_at);

CREATE TABLE evidence_channels (
    channel_name    TEXT PRIMARY KEY,
    schema_json     TEXT NOT NULL,
    priority        INTEGER NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
INSERT INTO evidence_channels VALUES
    ('vector',    '{"similarity": "Float32"}', 1, strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    ('inference', '{"inference_model": "Utf8", "inference_task": "Utf8", "inference_confidence": "Float32"}', 2, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));
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
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
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
