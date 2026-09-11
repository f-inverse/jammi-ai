\set ON_ERROR_STOP on
DROP SCHEMA IF EXISTS ptA CASCADE; CREATE SCHEMA ptA;
SET search_path = ptA, public;
CREATE TABLE jobs(job_id TEXT PRIMARY KEY, status TEXT NOT NULL, execution TEXT NOT NULL DEFAULT 'queued',
  claimable BOOLEAN NOT NULL DEFAULT TRUE, priority INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT '2026-01-01T00:00:00Z', spec TEXT NOT NULL DEFAULT '{"a":1}',
  has_deps BOOLEAN NOT NULL DEFAULT FALSE);
CREATE TABLE job_dependencies(job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
  depends_on_job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
  PRIMARY KEY(job_id, depends_on_job_id), CHECK (job_id <> depends_on_job_id));
CREATE INDEX idx_jobs_claim ON jobs(status, execution, claimable, priority DESC, created_at);
CREATE INDEX idx_job_dependencies_reverse ON job_dependencies(depends_on_job_id);
INSERT INTO jobs(job_id,status) SELECT 'f'||to_char(i,'FM0000000'),'failed' FROM generate_series(1,50000) i;
INSERT INTO jobs(job_id,status) SELECT 'c'||to_char(i,'FM0000000'),'completed' FROM generate_series(1,4000) i;
INSERT INTO jobs(job_id,status,has_deps) SELECT 'g'||to_char(i,'FM0000000'),'queued',TRUE FROM generate_series(1,4000) i;
INSERT INTO job_dependencies SELECT 'g'||to_char(i,'FM0000000'),'c'||to_char(i,'FM0000000') FROM generate_series(1,4000) i;
VACUUM (ANALYZE) jobs;
VACUUM (ANALYZE) job_dependencies;
SELECT count(*) AS total_jobs FROM jobs;
PREPARE a3(int) AS
SELECT j.job_id,
       (SELECT MIN(p.job_id) FROM job_dependencies d JOIN jobs p ON p.job_id = d.depends_on_job_id
         WHERE d.job_id = j.job_id AND p.status IN ('failed','cancelled')) AS dep_id,
       (SELECT q.status FROM jobs q WHERE q.job_id =
          (SELECT MIN(p.job_id) FROM job_dependencies d JOIN jobs p ON p.job_id = d.depends_on_job_id
            WHERE d.job_id = j.job_id AND p.status IN ('failed','cancelled'))) AS dep_status
  FROM jobs j
 WHERE j.status = 'queued' AND j.has_deps
   AND EXISTS (SELECT 1 FROM job_dependencies d JOIN jobs p ON p.job_id = d.depends_on_job_id
                WHERE d.job_id = j.job_id AND p.status IN ('failed','cancelled'))
 LIMIT $1;
\echo '===== 58k catalog, execution 1 (custom plan) ====='
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) EXECUTE a3(500);
EXECUTE a3(500); EXECUTE a3(500); EXECUTE a3(500); EXECUTE a3(500);
\echo '===== 58k catalog, execution 6 ====='
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) EXECUTE a3(500);
\echo '===== 58k catalog, execution 7 ====='
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) EXECUTE a3(500);
\echo '===== 58k catalog, LITERAL 500 (what an EXPLAIN-by-hand gate measures) ====='
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF)
SELECT j.job_id,
       (SELECT MIN(p.job_id) FROM job_dependencies d JOIN jobs p ON p.job_id = d.depends_on_job_id
         WHERE d.job_id = j.job_id AND p.status IN ('failed','cancelled')) AS dep_id,
       (SELECT q.status FROM jobs q WHERE q.job_id =
          (SELECT MIN(p.job_id) FROM job_dependencies d JOIN jobs p ON p.job_id = d.depends_on_job_id
            WHERE d.job_id = j.job_id AND p.status IN ('failed','cancelled'))) AS dep_status
  FROM jobs j
 WHERE j.status = 'queued' AND j.has_deps
   AND EXISTS (SELECT 1 FROM job_dependencies d JOIN jobs p ON p.job_id = d.depends_on_job_id
                WHERE d.job_id = j.job_id AND p.status IN ('failed','cancelled'))
 LIMIT 500;
