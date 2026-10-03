"""Jobs: submit work, watch it, cancel it, pick it up from another process.

Run with `python cookbook/recipes/jobs/example.py`, or a step at a time as a
notebook: each `# %%` cell is one step. Seconds on a CPU.
"""

# %% [markdown]
# Every long-running verb (a fine-tune, a context-predictor training, a
# compute verb run as a job) is a row in the engine's durable job queue. A job
# outlives the connection that submitted it, and any process with a worker
# enabled claims it. Two configurations show both sides: one process submits
# with no worker, another runs the worker.

# %%
import tempfile
from pathlib import Path

import jammi
from jammi.errors import JobCancelled
from jammi_cookbook import fixtures

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

home = Path(tempfile.mkdtemp())
engine = f"file://{home}/engine"
no_worker = home / "submit-only.toml"
no_worker.write_text("[worker]\nenabled = false\n")
with_worker = home / "worker.toml"
with_worker.write_text("[jobs]\nretention_days = 0\n")


def submit(db):
    return db.fine_tune(
        source="training",
        base_model=BASE_MODEL,
        columns=["text_a", "text_b", "score"],
        method="lora",
        task="text_embedding",
        lora_rank=4,
        epochs=1,
    )


# %% [markdown]
# ## Submit work no one claims
#
# With `[worker] enabled = false`, this process only submits: its jobs stay
# queued. A handle reads a job's `kind`, `status()` and `progress()`.

# %%
db = jammi.connect(engine, config=str(no_worker))
db.add_source("training", url=str(fixtures.path("tiny_pairs.csv")), format="csv")

by_handle, by_id, kept = submit(db), submit(db), submit(db)
print(f"{by_handle.job_id}: kind={by_handle.kind} status={by_handle.status()}")
print(f"progress: {by_handle.progress()}")
assert by_handle.status() == "queued"

# %% [markdown]
# ## Cancel
#
# A job cancels through its handle or by id. Cancelling a finished job does
# nothing, and `wait()` on a cancelled job raises `JobCancelled`.

# %%
assert by_handle.cancel()
assert db.cancel_job(by_id.job_id)
assert not db.cancel_job(by_id.job_id), "a terminal job cancels no further"
try:
    by_handle.wait()
    raise AssertionError("a cancelled job must not complete")
except JobCancelled:
    print(f"{by_handle.job_id}: cancelled")

# %% [markdown]
# ## The queue and who claims from it
#
# `list_jobs` is the queue; `list_workers` lists the processes running the
# claim loop, and here there are none.

# %%
for row in db.list_jobs():
    print(f"  {row['job_id']}  {row['kind']:<10} {row['status']}")
assert db.list_workers() == [], "no process runs the claim loop"
db.close()

# %% [markdown]
# ## Pick the job up from another process
#
# Opened with a worker, the engine claims the job left queued. `job(id)`
# attaches to it by id, and `wait()` returns once the worker has run it; the
# finished job reports its metrics, the acceleration it ran with, and the
# model it produced.

# %%
db = jammi.connect(engine, config=str(with_worker))
for worker in db.list_workers():
    print(f"worker {worker['label'] or worker['instance_id']}: {worker['state']}")

job = db.job(kept.job_id)
job.wait()
print(f"{job.job_id}: {job.status()} → {job.output_model_id}")
print(f"metrics: {job.metrics()}")
print(f"acceleration: {job.acceleration_report()}")
assert job.status() == "completed"

# %% [markdown]
# ## Clean up finished jobs
#
# `prune_jobs` deletes finished jobs older than `[jobs] retention_days`, zero
# here. The two cancelled jobs were already swept when this process opened the
# catalog; `prune_jobs` sweeps the one that finished since.

# %%
pruned = db.prune_jobs()
print(f"pruned {pruned} finished job(s); {len(db.list_jobs())} left")
assert pruned == 1 and db.list_jobs() == []

# %%
db.close()
