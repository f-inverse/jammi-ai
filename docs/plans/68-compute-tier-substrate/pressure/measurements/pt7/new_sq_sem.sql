SELECT sqlite_version();
CREATE TABLE jobs(job_id TEXT PRIMARY KEY, status TEXT NOT NULL, has_deps INT NOT NULL DEFAULT 0);
CREATE TABLE job_dependencies(job_id TEXT NOT NULL, depends_on_job_id TEXT NOT NULL, PRIMARY KEY(job_id,depends_on_job_id));
CREATE INDEX idx_jobs_claim ON jobs(status, has_deps);
CREATE INDEX idx_job_dependencies_reverse ON job_dependencies(depends_on_job_id);
INSERT INTO jobs(job_id,status,has_deps) VALUES
 ('r0001','failed',0),
 ('b0001','queued',1),('b0002','queued',1),
 ('c0001','queued',1),('c0002','queued',1),('c0003','queued',1),('c0004','queued',1),
 ('m0001','queued',1),('m0002','queued',1),
 ('k0001','completed',0),('x0001','queued',1),('y0001','queued',1),('z0001','queued',0);
INSERT INTO job_dependencies(job_id,depends_on_job_id) VALUES
 ('b0001','r0001'),('b0002','r0001'),
 ('c0001','b0001'),('c0002','b0001'),('c0003','b0002'),('c0004','b0002'),
 ('m0002','b0002'),('m0002','b0001'),
 ('m0001','b0001'),('m0001','b0002'),
 ('x0001','k0001'),
 ('y0001','b0001'),('y0001','k0001');
.mode column
.headers on
.print "=== PASS 1 (new text, no outer ORDER BY, LIMIT 500) ==="
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
.print "=== PASS 2 ==="
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
.print "=== PASS 2 mixed-status check: cancel b0001 instead ==="
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
