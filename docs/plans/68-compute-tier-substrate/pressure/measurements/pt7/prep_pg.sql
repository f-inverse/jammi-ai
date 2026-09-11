SET search_path = pt7, public;
PREPARE arm3(int) AS
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
EXECUTE arm3(500); EXECUTE arm3(500); EXECUTE arm3(500); EXECUTE arm3(500); EXECUTE arm3(500);
\echo '===== 6th execution: generic-plan candidate ====='
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF, GENERIC_PLAN OFF) EXECUTE arm3(500);
\echo '===== 7th ====='
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) EXECUTE arm3(500);
\echo '===== 8th ====='
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) EXECUTE arm3(500);
