CREATE TABLE jobs(job_id TEXT PRIMARY KEY, status TEXT NOT NULL, has_deps INT NOT NULL DEFAULT 0);
CREATE TABLE job_dependencies(job_id TEXT NOT NULL, depends_on_job_id TEXT NOT NULL, PRIMARY KEY(job_id,depends_on_job_id));
-- perfect binary in-tree: node n (1..2^15-1), children 2n, 2n+1; each child DEPENDS ON its parent.
WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n WHERE i < 32767)
INSERT INTO jobs(job_id,status,has_deps) SELECT 'j'||i,'queued', CASE WHEN i>1 THEN 1 ELSE 0 END FROM n;
INSERT INTO job_dependencies(job_id,depends_on_job_id)
  SELECT job_id, 'j'||(CAST(substr(job_id,2) AS INTEGER)/2) FROM jobs WHERE CAST(substr(job_id,2) AS INTEGER) > 1;
-- per-dependency fan-out actually present (the MAX_DEPENDANTS_PER_JOB metric):
SELECT 'max_dependants_per_job', MAX(c) FROM (SELECT COUNT(*) c FROM job_dependencies GROUP BY depends_on_job_id);
-- hop sizes of the push starting from the root j1 going terminal:
WITH RECURSIVE hop(job_id, depth) AS (
  SELECT 'j1', 0
  UNION ALL
  SELECT d.job_id, h.depth+1 FROM job_dependencies d JOIN hop h ON d.depends_on_job_id = h.job_id WHERE h.depth < 64
)
SELECT 'hop', depth, COUNT(*) FROM hop WHERE depth>0 GROUP BY depth ORDER BY depth;
SELECT 'total_pushed_rows_in_one_txn', COUNT(*)-1 FROM (
 WITH RECURSIVE hop(job_id, depth) AS (
  SELECT 'j1', 0 UNION ALL
  SELECT d.job_id, h.depth+1 FROM job_dependencies d JOIN hop h ON d.depends_on_job_id = h.job_id WHERE h.depth < 64)
 SELECT job_id FROM hop);
