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
 LIMIT ?;
