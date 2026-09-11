SET search_path = pt7, public;
INSERT INTO jobs(job_id,status) SELECT 'h'||to_char(i,'FM0000000'),'failed' FROM generate_series(1,250000) i;
VACUUM (ANALYZE) jobs;
VACUUM (ANALYZE) job_dependencies;
SELECT count(*) AS total_jobs FROM jobs;
\echo '===== 300k catalog, LITERAL limit (custom plan) ====='
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
PREPARE arm3b(int) AS
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
EXECUTE arm3b(500); EXECUTE arm3b(500); EXECUTE arm3b(500); EXECUTE arm3b(500); EXECUTE arm3b(500);
\echo '===== 300k catalog, BOUND limit, 6th execution ====='
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) EXECUTE arm3b(500);
\echo '===== 300k catalog, BOUND limit, 7th execution ====='
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) EXECUTE arm3b(500);
