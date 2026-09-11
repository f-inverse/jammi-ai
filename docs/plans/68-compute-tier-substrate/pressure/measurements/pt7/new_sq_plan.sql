.print "=== sqlite version ==="
SELECT sqlite_version();
PRAGMA foreign_keys=ON;
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
CREATE INDEX idx_jobs_parent ON jobs(parent_id);
CREATE INDEX idx_job_dependencies_reverse ON job_dependencies(depends_on_job_id);
WITH RECURSIVE s(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM s WHERE i<50000)
INSERT INTO jobs(job_id,status) SELECT 'f'||printf('%07d',i),'failed' FROM s;
WITH RECURSIVE s(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM s WHERE i<50000)
INSERT INTO jobs(job_id,status,has_deps) SELECT 'x'||printf('%07d',i),'failed',1 FROM s;
WITH RECURSIVE s(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM s WHERE i<50000)
INSERT INTO job_dependencies SELECT 'x'||printf('%07d',i),'f'||printf('%07d',i) FROM s;
WITH RECURSIVE s(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM s WHERE i<4000)
INSERT INTO jobs(job_id,status) SELECT 'c'||printf('%07d',i),'completed' FROM s;
WITH RECURSIVE s(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM s WHERE i<4000)
INSERT INTO jobs(job_id,status,has_deps) SELECT 'g'||printf('%07d',i),'queued',1 FROM s;
WITH RECURSIVE s(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM s WHERE i<4000)
INSERT INTO job_dependencies SELECT 'g'||printf('%07d',i),'c'||printf('%07d',i) FROM s;
WITH RECURSIVE s(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM s WHERE i<5000)
INSERT INTO jobs(job_id,status) SELECT 'q'||printf('%07d',i),'queued' FROM s;

.print "===== NEW TEXT / NO ANALYZE ====="
EXPLAIN QUERY PLAN
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

ANALYZE;
.print "===== NEW TEXT / AFTER ANALYZE ====="
EXPLAIN QUERY PLAN
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

.print "===== SCOPED (job_id = ?, LIMIT 1), AFTER ANALYZE ====="
EXPLAIN QUERY PLAN
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
   AND j.job_id = 'g0000123'
 LIMIT 1;
.print "===== timing new text, after analyze ====="
.timer on
SELECT count(*) FROM (
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
 LIMIT 500);
.timer off
