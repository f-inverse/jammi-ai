BEGIN;
CREATE TEMP TABLE jobs(job_id TEXT PRIMARY KEY, status TEXT NOT NULL, error TEXT, acceleration_report TEXT, updated_at TEXT NOT NULL DEFAULT '') ON COMMIT DROP;
PREPARE push_untyped AS
WITH v(id,msg) AS (VALUES ($1,$2),($3,$4),($5,$6)) UPDATE jobs SET status = 'failed', error = v.msg, acceleration_report = CASE WHEN acceleration_report = $7 THEN $8 ELSE acceleration_report END, updated_at = $9 FROM v WHERE jobs.job_id = v.id AND jobs.status = 'queued' RETURNING jobs.job_id;
SELECT 'inferred_types', parameter_types FROM pg_prepared_statements WHERE name='push_untyped';
ROLLBACK;
