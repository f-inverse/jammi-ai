"""The model catalog: see what an engine serves and has trained, and clean it up.

1. `db.get_server_info` — the engine's version, features and storage backends
2. `db.describe_source` — one registered source and the result tables built on it
3. `db.preload_model` — load a model into the cache before the first request
4. `db.list_models` / `db.describe_model` — the models the catalog holds
5. `db.delete_model` — refused while a job still references the model, a no-op
   for an absent one with `if_exists=True`, and a real delete once the job that
   references it is past `[jobs] retention_days` and `db.prune_jobs` has
   deleted it

Training is what puts a model in the catalog, so the recipe trains one: a
one-epoch LoRA over 30 text pairs with the `tiny_bert` fixture, seconds on CPU.

Run with `python cookbook/recipes/model_catalog/example.py`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jammi
from jammi.errors import BackendError
from jammi_cookbook import fixtures

PAIRS_PATH = fixtures.path("tiny_pairs.csv")
BASE_MODEL = fixtures.model("tiny_bert")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        engine = f"file://{tmp}/engine"
        with jammi.connect(engine) as db:
            # 1. What this engine is.
            info = db.get_server_info()
            print(f"engine {info['version']}; storage backends: {info['storage_backends']}")

            # 2. A source, and the result tables built on it.
            db.add_source("training", url=str(PAIRS_PATH), format="csv")
            source = db.describe_source("training")
            print(f"source {source['source_id']}: {source['status']}")
            assert db.describe_source("no_such_source") is None

            # 3. Preloading pays a model's load before the first request needs it.
            db.preload_model(BASE_MODEL)

            # 4. Train: the base model is registered at submission, the
            #    fine-tuned one on completion.
            job = db.fine_tune(
                source="training",
                base_model=BASE_MODEL,
                columns=["text_a", "text_b", "score"],
                method="lora",
                task="text_embedding",
                lora_rank=4,
                epochs=1,
            )
            job.wait()
            tuned = job.output_model_id
            for model in db.list_models():
                print(f"model {model['model_id'][-40:]}: {model['task']} ({model['status']})")
            described = db.describe_model(tuned)
            print(f"describe_model: {described}")
            assert described is not None and described["model_id"] == tuned

            # 5a. The finished job that trained it still references the model:
            #     a job keeps its models for `[jobs] retention_days` (30 by
            #     default) after it ends.
            try:
                db.delete_model(tuned)
                raise AssertionError("a referenced model must not delete")
            except BackendError as refused:
                print(f"delete while referenced: refused ({refused})")

            # 5b. An absent model: a no-op with `if_exists`.
            db.delete_model("jammi:fine-tuned:absent", if_exists=True)

        # 5c. Retention is a deployment setting. Reopen the same catalog with a
        #     zero-day retention: the finished job is past it — the engine
        #     sweeps such jobs when it opens, and `prune_jobs` sweeps on
        #     demand — so nothing references the model any more.
        config = Path(tmp) / "jammi.toml"
        config.write_text("[jobs]\nretention_days = 0\n")
        with jammi.connect(engine, config=str(config)) as db:
            db.prune_jobs()
            assert db.list_jobs() == [], "the finished job is gone"
            db.delete_model(tuned)
            assert db.describe_model(tuned) is None
            print(f"deleted {tuned}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
