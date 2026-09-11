BEGIN;
CREATE TEMP TABLE jobs(job_id TEXT PRIMARY KEY,status TEXT NOT NULL,error TEXT,cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,acceleration_report TEXT,updated_at TEXT NOT NULL) ON COMMIT DROP;
INSERT INTO jobs VALUES ('d0','queued',NULL,FALSE,'{"state":"pending"}','s');
WITH v(id,msg) AS (VALUES ($1,$2)) UPDATE jobs SET status='failed', error=v.msg, acceleration_report = CASE WHEN acceleration_report = $3 THEN $4 ELSE acceleration_report END, updated_at=$5 FROM v WHERE jobs.job_id=v.id AND jobs.status='queued' RETURNING jobs.job_id \bind 'd0' 'msg' '{"state":"pending"}' 'MARK' 'NOW' \g
SELECT * FROM jobs;
ROLLBACK;
