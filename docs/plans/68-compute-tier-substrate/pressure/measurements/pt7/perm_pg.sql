DROP SCHEMA IF EXISTS pt7 CASCADE;
CREATE SCHEMA pt7;
SET search_path = pt7, public;
CREATE TABLE jobs(
  job_id TEXT PRIMARY KEY, kind TEXT NOT NULL DEFAULT 'fine_tune', status TEXT NOT NULL,
  execution TEXT NOT NULL DEFAULT 'queued', claimable BOOLEAN NOT NULL DEFAULT TRUE,
  priority INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL DEFAULT '2026-01-01T00:00:00Z',
  updated_at TEXT NOT NULL DEFAULT '2026-01-01T00:00:00Z',
  tenant_id TEXT, spec TEXT NOT NULL DEFAULT '{"a":1}', error TEXT,
  attempts INTEGER NOT NULL DEFAULT 0, claimed_by TEXT, lease_expires_at TEXT,
  cancel_requested BOOLEAN NOT NULL DEFAULT FALSE, acceleration_report TEXT,
  idempotency_key TEXT, parent_id TEXT, has_deps BOOLEAN NOT NULL DEFAULT FALSE);
CREATE TABLE job_dependencies(job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
  depends_on_job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
  PRIMARY KEY(job_id, depends_on_job_id), CHECK (job_id <> depends_on_job_id));
CREATE INDEX idx_jobs_claim ON jobs(status, execution, claimable, priority DESC, created_at);
CREATE INDEX idx_jobs_lease ON jobs(lease_expires_at);
CREATE INDEX idx_jobs_parent ON jobs(parent_id);
CREATE INDEX idx_job_dependencies_reverse ON job_dependencies(depends_on_job_id);
INSERT INTO jobs(job_id,status) SELECT 'f'||to_char(i,'FM0000000'),'failed' FROM generate_series(1,50000) i;
INSERT INTO jobs(job_id,status,has_deps) SELECT 'x'||to_char(i,'FM0000000'),'failed',TRUE FROM generate_series(1,50000) i;
INSERT INTO job_dependencies SELECT 'x'||to_char(i,'FM0000000'),'f'||to_char(i,'FM0000000') FROM generate_series(1,50000) i;
INSERT INTO jobs(job_id,status) SELECT 'c'||to_char(i,'FM0000000'),'completed' FROM generate_series(1,4000) i;
INSERT INTO jobs(job_id,status,has_deps) SELECT 'g'||to_char(i,'FM0000000'),'queued',TRUE FROM generate_series(1,4000) i;
INSERT INTO job_dependencies SELECT 'g'||to_char(i,'FM0000000'),'c'||to_char(i,'FM0000000') FROM generate_series(1,4000) i;
INSERT INTO jobs(job_id,status) SELECT 'q'||to_char(i,'FM0000000'),'queued' FROM generate_series(1,5000) i;
