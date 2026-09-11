BEGIN;
CREATE TEMP TABLE jobs(
  job_id TEXT PRIMARY KEY, kind TEXT NOT NULL DEFAULT 'k', status TEXT NOT NULL,
  execution TEXT NOT NULL DEFAULT 'queued', claimable BOOLEAN NOT NULL DEFAULT TRUE,
  priority INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL DEFAULT '2026-01-01',
  has_deps BOOLEAN NOT NULL DEFAULT FALSE) ON COMMIT DROP;
CREATE TEMP TABLE job_dependencies(job_id TEXT NOT NULL, depends_on_job_id TEXT NOT NULL,
  PRIMARY KEY(job_id, depends_on_job_id)) ON COMMIT DROP;
CREATE INDEX idx_jobs_claim ON jobs(status, execution, claimable, priority DESC, created_at);
CREATE INDEX idx_job_dependencies_reverse ON job_dependencies(depends_on_job_id);

-- 190k completed
INSERT INTO jobs(job_id,status) SELECT 'done'||to_char(i,'FM0000000'),'completed' FROM generate_series(1,190000) i;
-- 5k queued, no deps
INSERT INTO jobs(job_id,status) SELECT 'q'||to_char(i,'FM0000000'),'queued' FROM generate_series(1,5000) i;
-- 4000 queued with deps on a completed job (non-qualifying)
INSERT INTO jobs(job_id,status,has_deps) SELECT 'g'||to_char(i,'FM0000000'),'queued',TRUE FROM generate_series(1,4000) i;
INSERT INTO job_dependencies SELECT 'g'||to_char(i,'FM0000000'),'done'||to_char(i,'FM0000000') FROM generate_series(1,4000) i;
-- 1000 failed dependencies + 600 qualifying dependants
INSERT INTO jobs(job_id,status) SELECT 'f'||to_char(i,'FM0000000'),'failed' FROM generate_series(1,600) i;
INSERT INTO jobs(job_id,status,has_deps) SELECT 'w'||to_char(i,'FM0000000'),'queued',TRUE FROM generate_series(1,600) i;
INSERT INTO job_dependencies SELECT 'w'||to_char(i,'FM0000000'),'f'||to_char(i,'FM0000000') FROM generate_series(1,600) i;
ANALYZE jobs; ANALYZE job_dependencies;
\echo === bulk sweep plan (scope NULL, LIMIT 500) ===
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF)
SELECT j.job_id,
       (SELECT p.job_id FROM job_dependencies d JOIN jobs p ON p.job_id = d.depends_on_job_id
         WHERE d.job_id = j.job_id AND p.status IN ('failed','cancelled') ORDER BY p.job_id LIMIT 1) AS dep_id,
       (SELECT p.status FROM job_dependencies d JOIN jobs p ON p.job_id = d.depends_on_job_id
         WHERE d.job_id = j.job_id AND p.status IN ('failed','cancelled') ORDER BY p.job_id LIMIT 1) AS dep_status
  FROM jobs j
 WHERE j.status = 'queued' AND j.has_deps
   AND EXISTS (SELECT 1 FROM job_dependencies d JOIN jobs p ON p.job_id = d.depends_on_job_id
                WHERE d.job_id = j.job_id AND p.status IN ('failed','cancelled'))
 ORDER BY j.job_id LIMIT 500;
\echo === scoped (job_id = one qualifying row, LIMIT 1) ===
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF)
SELECT j.job_id,
       (SELECT p.job_id FROM job_dependencies d JOIN jobs p ON p.job_id = d.depends_on_job_id
         WHERE d.job_id = j.job_id AND p.status IN ('failed','cancelled') ORDER BY p.job_id LIMIT 1) AS dep_id,
       (SELECT p.status FROM job_dependencies d JOIN jobs p ON p.job_id = d.depends_on_job_id
         WHERE d.job_id = j.job_id AND p.status IN ('failed','cancelled') ORDER BY p.job_id LIMIT 1) AS dep_status
  FROM jobs j
 WHERE j.status = 'queued' AND j.has_deps
   AND EXISTS (SELECT 1 FROM job_dependencies d JOIN jobs p ON p.job_id = d.depends_on_job_id
                WHERE d.job_id = j.job_id AND p.status IN ('failed','cancelled'))
   AND j.job_id = 'w0000123'
 ORDER BY j.job_id LIMIT 1;
\echo === no-work case (delete the failed deps) ===
UPDATE jobs SET status='completed' WHERE status='failed';
ANALYZE jobs;
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF)
SELECT j.job_id,
       (SELECT p.job_id FROM job_dependencies d JOIN jobs p ON p.job_id = d.depends_on_job_id
         WHERE d.job_id = j.job_id AND p.status IN ('failed','cancelled') ORDER BY p.job_id LIMIT 1) AS dep_id
  FROM jobs j
 WHERE j.status = 'queued' AND j.has_deps
   AND EXISTS (SELECT 1 FROM job_dependencies d JOIN jobs p ON p.job_id = d.depends_on_job_id
                WHERE d.job_id = j.job_id AND p.status IN ('failed','cancelled'))
 ORDER BY j.job_id LIMIT 500;
ROLLBACK;
