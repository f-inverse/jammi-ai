"""Predict a row's outcome from the rows most like it — an in-context predictor.

A context predictor is trained across many small tasks at once. For a target
row it retrieves the target's nearest neighbours (by embedding) from the same
task, reads their outcomes, and predicts the target's outcome as a
distribution. It never trains on the task it is asked about: the task is
learned from the context, in one forward pass.

1. `db.import_embeddings` — bring your own vectors: here, each row's features
2. `db.train_context_predictor` — meta-train across tasks (a job)
3. `db.predict_with_context_predictor` — a predictive distribution for one row,
   with the context rows that informed it

The data is synthetic: 16 tasks of 24 rows. Each task has its own level —
every row's outcome is the task's level plus a little noise — and its rows'
4-dimensional features cluster around a task centroid, so a row's nearest
neighbours are rows of its own task. A predictor that learned "read the level
off the context" predicts rows of a task it has never seen, where the best any
context-free model can do is the overall mean. Runs in seconds on CPU.

Run with `python cookbook/recipes/context_predictor/example.py`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jammi
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TASKS = 16
ROWS_PER_TASK = 24
DIM = 4


def meta_dataset(seed: int = 0) -> tuple[pa.Table, pa.Table]:
    """The source rows `(_row_id, task, y)` and their feature vectors
    `(_row_id, vector)`, keyed by one identity."""
    rng = np.random.default_rng(seed)
    ids, tasks, ys, xs = [], [], [], []
    for t in range(TASKS):
        level = rng.uniform(-3, 3)
        centroid = rng.uniform(-1, 1, DIM)
        for r in range(ROWS_PER_TASK):
            ids.append(f"t{t:02d}_r{r:02d}")
            tasks.append(f"task_{t:02d}")
            ys.append(float(level + rng.normal(0, 0.2)))
            xs.append((centroid + rng.normal(0, 0.05, DIM)).astype(np.float32))
    source = pa.table({"_row_id": ids, "task": tasks, "y": ys})
    vectors = pa.table(
        {
            "_row_id": ids,
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(np.concatenate(xs), type=pa.float32()), DIM
            ),
        }
    )
    return source, vectors


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp, jammi.connect(f"file://{tmp}/engine") as db:
        source, vectors = meta_dataset()
        pq.write_table(source, Path(tmp) / "rows.parquet")
        pq.write_table(vectors, Path(tmp) / "features.parquet")
        db.add_source("rows", url=str(Path(tmp) / "rows.parquet"), format="parquet")

        # 1. The features ARE the embedding: imported, not computed.
        db.import_embeddings(
            source="rows",
            model="synthetic-features",
            vectors_url=str(Path(tmp) / "features.parquet"),
            key="_row_id",
            dimensions=DIM,
        )

        # 2. Meta-train. Tasks (not rows) are split into train and held-out
        #    test, so the test tasks are unseen at training time.
        job = db.train_context_predictor(
            "rows",
            key_column="_row_id",
            task_column="task",
            value_column="y",
            context_k=8,
            hidden_dim=32,
            num_heads=2,
            epochs=60,
            test_task_fraction=0.25,
            seed=0,
        )
        job.wait()
        model_id = job.output_model_id
        print(f"trained {model_id}; final loss {job.metrics()['final_loss']:.3f}")

        # 3. Predict rows of the last task — held out, never trained on — and
        #    compare with the best context-free guess, the overall mean.
        truth = {r["_row_id"]: r["y"] for r in source.to_pylist()}
        overall = float(np.mean(list(truth.values())))
        errors, baseline = [], []
        for r in range(6):
            key = f"t{TASKS - 1:02d}_r{r:02d}"
            prediction = db.predict_with_context_predictor(
                model_id, source="rows", target_key=key
            )
            mean, std = prediction["mean"], prediction["std"]
            print(
                f"{key}: y={truth[key]:+.2f}  predicted {mean:+.2f} ± {std:.2f}  "
                f"(context: {len(prediction['context_ref'])} rows, via {prediction['source']})"
            )
            errors.append(abs(mean - truth[key]))
            baseline.append(abs(overall - truth[key]))
        print(
            f"mean absolute error on the held-out task: {np.mean(errors):.3f} "
            f"(overall mean as the guess: {np.mean(baseline):.3f})"
        )
        assert np.mean(errors) < np.mean(baseline), "the context beats no context"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
