# The model catalog

See what an engine serves and has trained, and clean up models nothing
needs any more.

**When to use this pattern.** You run fine-tunes and want to list, inspect
and delete the resulting models — and to know why a delete is refused.

## What `example.py` does

1. `get_server_info` — the engine's version and storage backends
2. `describe_source` — a registered source, and `None` for an absent one
3. `preload_model` — pay a model's load before the first request
4. Trains a one-epoch LoRA (training is what registers models), then
   `list_models` / `describe_model`
5. `delete_model` — refused while the job that trained the model still
   references it (a finished job keeps its models for
   `[jobs] retention_days`); a no-op for an absent model with
   `if_exists=True`; and, after reopening with `retention_days = 0` and
   `prune_jobs`, a real delete

## Run it

```bash
python cookbook/recipes/model_catalog/example.py
```

Seconds on CPU.
