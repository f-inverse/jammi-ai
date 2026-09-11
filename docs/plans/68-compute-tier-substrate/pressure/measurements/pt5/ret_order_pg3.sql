BEGIN;
CREATE TEMP TABLE jobs(job_id TEXT PRIMARY KEY, status TEXT NOT NULL, error TEXT, updated_at TEXT NOT NULL) ON COMMIT DROP;
INSERT INTO jobs SELECT 'd'||to_char(i,'FM0000'),'queued',NULL,'s' FROM generate_series(0,7) i;
PREPARE r(text,text,text,text,text,text,text,text,text,text,text,text,text,text,text,text,text) AS
WITH v(id, msg) AS (VALUES ($1, $2), ($3, $4), ($5, $6), ($7, $8), ($9, $10), ($11, $12), ($13, $14), ($15, $16))
UPDATE jobs SET status='cancelled', error=v.msg, updated_at=$17 FROM v WHERE jobs.job_id=v.id AND jobs.status='queued' RETURNING jobs.job_id;
\echo === RETURNING with DESCENDING v input ===
EXECUTE r('d0007','m7','d0006','m6','d0005','m5','d0004','m4','d0003','m3','d0002','m2','d0001','m1','d0000','m0','NOW');
ROLLBACK;
