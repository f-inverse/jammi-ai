"""Jobs: submit work, watch it, cancel it, pick it up from another process.

Every long-running verb (a fine-tune, a context-predictor training, a compute
verb run as a job) is a row in the engine's durable job queue. A job outlives
the connection that submitted it; any process with a worker enabled claims it.

1. Submit with no worker running (`[worker] enabled = false`): jobs stay queued
2. `job.kind` / `job.status()` / `job.progress()` — read a job's state
3. `job.cancel()` and `db.cancel_job(id)` — cancel through the handle, or by id
4. `db.list_jobs()` / `db.list_workers()` — the queue, and who claims from it
5. Reopen with a worker: `db.job(id)` attaches to the job left queued, and
   `wait()` returns once the worker has run it — `metrics()`,
   `acceleration_report()`, `output_model_id`
6. `db.prune_jobs()` — delete finished jobs older than `[jobs] retention_days`

Uses the `tiny_bert` fixture and 30 text pairs; the one job that runs trains
for one epoch, seconds on CPU.

Run with `python cookbook/recipes/jobs/example.py`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jammi
from jammi.errors import JobCancelled
from jammi_cookbook import fixtures

PAIRS_PATH = fixtures.path("tiny_pairs.csv")
BASE_MODEL = fixtures.model("tiny_bert")


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


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        engine = f"file://{tmp}/engine"
        no_worker = Path(tmp) / "submit-only.toml"
        no_worker.write_text("[worker]\nenabled = false\n")
        with_worker = Path(tmp) / "worker.toml"
        with_worker.write_text("[jobs]\nretention_days = 0\n")

        # 1-4. A process that submits but never claims.
        with jammi.connect(engine, config=str(no_worker)) as db:
            db.add_source("training", url=str(PAIRS_PATH), format="csv")
            by_handle, by_id, kept = submit(db), submit(db), submit(db)
            print(f"{by_handle.job_id}: kind={by_handle.kind} status={by_handle.status()}")
            print(f"progress: {by_handle.progress()}")
            assert by_handle.status() == "queued"

            assert by_handle.cancel()
            assert db.cancel_job(by_id.job_id)
            assert not db.cancel_job(by_id.job_id), "a terminal job cancels no further"
            try:
                by_handle.wait()
                raise AssertionError("a cancelled job must not complete")
            except JobCancelled:
                print(f"{by_handle.job_id}: cancelled")

            for row in db.list_jobs():
                print(f"  {row['job_id']}  {row['kind']:<10} {row['status']}")
            assert db.list_workers() == [], "no process runs the claim loop"

        # 5. A process with a worker: the job left queued is claimed and run.
        with jammi.connect(engine, config=str(with_worker)) as db:
            for worker in db.list_workers():
                print(f"worker {worker['label'] or worker['instance_id']}: {worker['state']}")
            job = db.job(kept.job_id)
            job.wait()
            print(f"{job.job_id}: {job.status()} → {job.output_model_id}")
            print(f"metrics: {job.metrics()}")
            print(f"acceleration: {job.acceleration_report()}")
            assert job.status() == "completed"

            # 6. Every job is finished and past a zero-day retention. The two
            #    cancelled ones were swept when this process opened the catalog;
            #    `prune_jobs` sweeps the one that finished since.
            pruned = db.prune_jobs()
            print(f"pruned {pruned} finished job(s); {len(db.list_jobs())} left")
            assert pruned == 1 and db.list_jobs() == []
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
