# Fine-tune an encoder

Run a LoRA fine-tune on top of an existing text encoder, poll the job to
completion, and use the resulting checkpoint to encode a query.

**When to use this pattern.** Your domain (legal contracts, medical
abstracts, patent claims, internal product docs) doesn't match the
distribution the base encoder was trained on, and you have a few
hundred to a few thousand labelled or contrastive pairs. LoRA gets you
~80% of the lift of a full fine-tune at a fraction of the cost; the
resulting adapter is small enough to ship as an attachment to the base
model rather than a re-distributed full checkpoint.

## What `example.py` does

1. Connects to a temporary artifact dir
2. Registers `tiny_pairs.csv` (30 contrastive pairs) as `training`
3. Calls `db.fine_tune(...)` with the local `tiny_bert` base, a small
   LoRA rank, and one epoch — kept fast for CI
4. Waits for terminal status via `job.wait()`
5. Asserts the resulting `model_id` starts with `jammi:fine-tuned:`
6. Encodes a query through the fine-tuned model to confirm it loads

## API surface exercised

- `Database.fine_tune(*, source, base_model, columns, method, task=..., ...)`
- `FineTuneJob.wait()`
- `FineTuneJob.job_id`, `FineTuneJob.model_id`
- `Database.encode_text_query(model_id, text)`

The full keyword list on `fine_tune` covers LoRA rank/alpha/dropout,
learning rate, epochs, batch size, max sequence length, validation
fraction, early-stopping patience/metric, warmup, gradient accumulation,
backbone dtype, weight decay, and gradient clipping — the recipe uses
the defaults for everything except rank and epochs.

## Performance note

This recipe is excluded from the per-PR smoke matrix because even at one
epoch it runs ~30 seconds on CPU. The nightly cron with
`JAMMI_COOKBOOK_SLOW=1` includes it. Override the gate locally:

```bash
JAMMI_COOKBOOK_SLOW=1 python tests/cookbook_smoke.py
```

## Run it

```bash
python cookbook/recipes/fine_tune/example.py
```

Exits 0 on success, prints `job_id`, `model_id`, and `fine_tune: OK`.
