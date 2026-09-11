SET search_path = pt7, public;
PREPARE arm3s(text) AS
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
   AND j.job_id = $1
 LIMIT 1;
EXECUTE arm3s('g0000123'); EXECUTE arm3s('g0000123'); EXECUTE arm3s('g0000123'); EXECUTE arm3s('g0000123'); EXECUTE arm3s('g0000123');
\echo '===== scoped, 6th execution (generic-plan regime) ====='
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) EXECUTE arm3s('g0000123');
\echo '===== scoped, 7th ====='
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) EXECUTE arm3s('g0000123');
