CREATE TABLE jobs(job_id TEXT PRIMARY KEY, status TEXT NOT NULL, error TEXT, updated_at TEXT NOT NULL DEFAULT '');
INSERT INTO jobs(job_id,status) VALUES ('a','queued'),('b','queued');
WITH v(id,msg) AS (VALUES ('a','from-depZ'),('a','from-depA'),('b','from-depB'))
UPDATE jobs SET status='failed', error=v.msg, updated_at='T' FROM v
WHERE jobs.job_id=v.id AND jobs.status='queued' RETURNING jobs.job_id, jobs.error;
SELECT 'final', job_id, error FROM jobs ORDER BY job_id;
