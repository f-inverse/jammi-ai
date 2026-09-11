BEGIN;
CREATE TEMP TABLE jobs(job_id TEXT PRIMARY KEY, status TEXT NOT NULL, has_deps BOOLEAN NOT NULL DEFAULT FALSE) ON COMMIT DROP;
CREATE TEMP TABLE job_dependencies(job_id TEXT NOT NULL, depends_on_job_id TEXT NOT NULL, PRIMARY KEY(job_id,depends_on_job_id)) ON COMMIT DROP;
INSERT INTO jobs(job_id,status,has_deps) VALUES
 ('r0001','failed',FALSE),
 ('b0001','queued',TRUE),('b0002','queued',TRUE),
 ('c0001','queued',TRUE),('c0002','queued',TRUE),('c0003','queued',TRUE),('c0004','queued',TRUE),
 ('m0001','queued',TRUE),('m0002','queued',TRUE),
 ('k0001','completed',FALSE),('x0001','queued',TRUE),('y0001','queued',TRUE),('z0001','queued',FALSE);
INSERT INTO job_dependencies(job_id,depends_on_job_id) VALUES
 ('b0001','r0001'),('b0002','r0001'),
 ('c0001','b0001'),('c0002','b0001'),('c0003','b0002'),('c0004','b0002'),
 ('m0002','b0002'),('m0002','b0001'),
 ('m0001','b0001'),('m0001','b0002'),
 ('x0001','k0001'),
 ('y0001','b0001'),('y0001','k0001');
\echo '=== PASS 1 ==='
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
UPDATE jobs SET status='failed' WHERE job_id IN ('b0001','b0002');
\echo '=== PASS 2 ==='
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
\echo '=== PASS 2 variant: b0001 cancelled, b0002 failed (mixed) ==='
UPDATE jobs SET status='cancelled' WHERE job_id='b0001';
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
ROLLBACK;
