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
\echo === PASS 1 ===
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
\echo --- retire winners ---
UPDATE jobs SET status='failed' WHERE job_id IN (SELECT j.job_id FROM jobs j WHERE j.status='queued' AND j.has_deps AND EXISTS (SELECT 1 FROM job_dependencies d JOIN jobs p ON p.job_id=d.depends_on_job_id WHERE d.job_id=j.job_id AND p.status IN ('failed','cancelled')) ORDER BY j.job_id LIMIT 500);
\echo === PASS 2 ===
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
\echo --- retire winners ---
UPDATE jobs SET status='failed' WHERE job_id IN (SELECT j.job_id FROM jobs j WHERE j.status='queued' AND j.has_deps AND EXISTS (SELECT 1 FROM job_dependencies d JOIN jobs p ON p.job_id=d.depends_on_job_id WHERE d.job_id=j.job_id AND p.status IN ('failed','cancelled')) ORDER BY j.job_id LIMIT 500);
\echo === PASS 3 ===
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
\echo --- retire winners ---
UPDATE jobs SET status='failed' WHERE job_id IN (SELECT j.job_id FROM jobs j WHERE j.status='queued' AND j.has_deps AND EXISTS (SELECT 1 FROM job_dependencies d JOIN jobs p ON p.job_id=d.depends_on_job_id WHERE d.job_id=j.job_id AND p.status IN ('failed','cancelled')) ORDER BY j.job_id LIMIT 500);
ROLLBACK;
