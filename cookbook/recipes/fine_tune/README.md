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

## API surface exercised

- `Session.fine_tune(*, source, base_model, columns, method, task=..., ...)`
- `Job.wait()`
- `Job.job_id`, `Job.output_model_id`
- `Session.encode_query(*, model, query, modality="text")`

The full keyword list on `fine_tune` covers LoRA rank/alpha/dropout,
learning rate, epochs, batch size, max sequence length, validation
fraction, early-stopping patience/metric, warmup, gradient accumulation,
backbone dtype, weight decay, and gradient clipping — the recipe uses
the defaults for everything except rank and epochs.

## Run it

```bash
python cookbook/recipes/fine_tune/example.py
```

It prints each job's id and the fine-tuned model's id; seconds on a CPU.
