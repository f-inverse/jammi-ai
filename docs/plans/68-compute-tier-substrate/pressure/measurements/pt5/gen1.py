import io
PEND = '{"state":"pending"}'
DF   = '{"state":"undetermined","reason":"dependency_failed"}'
CAN  = '{"state":"undetermined","reason":"cancelled"}'
NOW  = '2026-09-10T00:00:00Z'

def stmt(k, variant):
    vals = ", ".join("($%d, $%d)" % (2*i+1, 2*i+2) for i in range(k))
    extra = "cancel_requested = TRUE,\n       " if variant=="cancelled" else ""
    status = 'cancelled' if variant=='cancelled' else 'failed'
    return ("WITH v(id, msg) AS (VALUES %s)\n"
            "UPDATE jobs\n"
            "   SET status = '%s', error = v.msg, %s"
            "acceleration_report = CASE WHEN acceleration_report = $%d THEN $%d ELSE acceleration_report END,\n"
            "       updated_at = $%d\n"
            "  FROM v\n"
            " WHERE jobs.job_id = v.id AND jobs.status = 'queued'\n"
            "RETURNING jobs.job_id" % (vals, status, extra, 2*k+1, 2*k+2, 2*k+3))

def rows(k):
    # k pairs; for k>=2 make the SECOND id a 'running' decoy (must not be returned)
    out=[]
    for i in range(k):
        out.append(("d%04d"%i, "dependency `a` failed for job `d%04d`"%i))
    return out

def pg(k, variant, f):
    r = rows(k)
    marker = CAN if variant=='cancelled' else DF
    f.write("BEGIN;\n")
    f.write("CREATE TEMP TABLE jobs(job_id TEXT PRIMARY KEY, status TEXT NOT NULL, error TEXT, cancel_requested BOOLEAN NOT NULL DEFAULT FALSE, acceleration_report TEXT, updated_at TEXT NOT NULL) ON COMMIT DROP;\n")
    ins=[]
    for i,(jid,_) in enumerate(r):
        st = 'running' if (k>1 and i==1) else 'queued'
        ar = {0:PEND,1:None,2:'{"state":"determined"}'}.get(i%3, PEND)
        arl = 'NULL' if ar is None else "'%s'"%ar
        ins.append("('%s','%s',%s,'seed')"%(jid,st,arl))
    f.write("INSERT INTO jobs(job_id,status,acceleration_report,updated_at) VALUES\n"+",\n".join(ins)+";\n")
    types = ",".join(["text"]*(2*k+3))
    f.write("PREPARE r(%s) AS\n%s;\n"%(types, stmt(k,variant)))
    args = []
    for jid,msg in r:
        args += [jid, msg]
    args += [PEND, marker, NOW]
    f.write("\\echo === RETURNING k=%d %s ===\n"%(k,variant))
    f.write("EXECUTE r(%s);\n"%(",".join("'%s'"%a.replace("'","''") for a in args)))
    f.write("\\echo === state ===\n")
    f.write("SELECT status, cancel_requested, coalesce(acceleration_report,'<NULL>') ar, updated_at, count(*) FROM jobs GROUP BY 1,2,3,4 ORDER BY 1,3;\n")
    f.write("SELECT 'updated_at_nulls', count(*) FROM jobs WHERE updated_at IS NULL;\n")
    f.write("ROLLBACK;\n")

def sq(k, variant, f):
    r = rows(k)
    marker = CAN if variant=='cancelled' else DF
    f.write("DROP TABLE IF EXISTS jobs;\n")
    f.write("CREATE TABLE jobs(job_id TEXT PRIMARY KEY, status TEXT NOT NULL, error TEXT, cancel_requested BOOLEAN NOT NULL DEFAULT 0, acceleration_report TEXT, updated_at TEXT NOT NULL);\n")
    ins=[]
    for i,(jid,_) in enumerate(r):
        st = 'running' if (k>1 and i==1) else 'queued'
        ar = {0:PEND,1:None,2:'{"state":"determined"}'}.get(i%3, PEND)
        arl = 'NULL' if ar is None else "'%s'"%ar
        ins.append("('%s','%s',%s,'seed')"%(jid,st,arl))
    f.write("INSERT INTO jobs(job_id,status,acceleration_report,updated_at) VALUES\n"+",\n".join(ins)+";\n")
    args=[]
    for jid,msg in r: args += [jid,msg]
    args += [PEND, marker, NOW]
    for i,a in enumerate(args):
        f.write(".parameter set $%d '%s'\n"%(i+1, a.replace("'","''")))
    f.write(".print === RETURNING k=%d %s ===\n"%(k,variant))
    f.write(stmt(k,variant)+";\n")
    f.write(".print === state ===\n")
    f.write("SELECT status, cancel_requested, coalesce(acceleration_report,'<NULL>'), updated_at, count(*) FROM jobs GROUP BY 1,2,3,4 ORDER BY 1,3;\n")
    f.write("SELECT 'updated_at_nulls', count(*) FROM jobs WHERE updated_at IS NULL;\n")
    f.write(".parameter clear\n")

import sys
base="/private/tmp/claude-501/-Users-vijaychakilam-git-f-inverse-jammi-ai/6f1f73dd-6a38-4987-87e9-680f3abf3579/scratchpad/pt5/"
with open(base+"p1_pg.sql","w") as f:
    for k in (1,500):
        for v in ("failed","cancelled"):
            pg(k,v,f)
with open(base+"p1_sq.sql","w") as f:
    f.write(".param init\n")
    for k in (1,500):
        for v in ("failed","cancelled"):
            sq(k,v,f)
print("ok")
