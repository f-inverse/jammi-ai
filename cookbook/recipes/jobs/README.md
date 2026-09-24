# Jobs

Submit long-running work, watch it, cancel it, and pick it up from another
process.

**When to use this pattern.** Training and compute verbs run as rows of the
engine's durable job queue. You need to see a job's state, stop one, run
submission and execution in different processes, or clean up finished jobs.

## What `example.py` does

1. Opens the engine with `[worker] enabled = false`, so submitted jobs stay
   queued
2. Submits three fine-tunes; reads `job.kind`, `job.status()`,
   `job.progress()`
3. Cancels one through its handle (`job.cancel()`) and one by id
   (`cancel_job`); `wait()` on a cancelled job raises `JobCancelled`
4. `list_jobs` shows the queue; `list_workers` is empty — nothing claims
5. Reopens the same catalog with a worker: `list_workers` shows it,
   `job(id)` attaches to the job left queued, and `wait()` returns once it
   has run — then `metrics()`, `acceleration_report()`, `output_model_id`
6. `prune_jobs` deletes finished jobs past `[jobs] retention_days`

## Run it

```bash
python cookbook/recipes/jobs/example.py
```

Seconds on CPU.
