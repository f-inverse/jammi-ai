# Evaluate inference (classification)

Run a classifier over a registered source and score its predictions
against gold labels.

**When to use this pattern.** You have a labelled holdout set and you
want a single number — accuracy, macro F1, per-class F1 — to compare
two classifiers, or to track drift over time on the same classifier.

## What `example.py` does

1. Connects to a temporary artifact dir
2. Registers the tiny corpus as `corpus` (parquet)
3. Registers `tiny_labels.csv` as `golden` (csv) — `(id, label)` rows
4. Runs `db.eval_inference` with the local
   `tiny_modernbert_classifier` fixture against the `content` column
5. Prints the returned aggregate `accuracy`, macro `f1`, per-class
   metrics, and the count of per-record predictions
6. Asserts every reported rate is in `[0.0, 1.0]`

## API surface exercised

- `Database.eval_inference(*, model, source, columns, task, golden_source, label_column)`

The returned dict carries `aggregate` (tagged by `"task"` — currently
`"classification"`) with `accuracy`, `f1`, and `per_class`, plus
`per_record` (one entry per aligned `{record_id, predicted, gold}`).

The `task` argument is the string form of the inference task —
`"classification"` here. NER is recognized but not yet supported via this
entrypoint (see the runner's `EvalTask::Ner` branch); for token-level
eval, call the `jammi-numerics` NER kernels directly.

## Golden source shape

`eval_inference` requires a registered source with these columns:

| column            | type | example   |
|-------------------|------|-----------|
| `id`              | utf8 | `"1"`     |
| `<label_column>`  | utf8 | `physics` |

`label_column` is the kwarg you pass at call time — `label` in this
recipe. Every `id` in the golden source must resolve to a row in the
input source; rows without a gold label are silently dropped from the
metric.

## Run it

```bash
python cookbook/recipes/eval_inference/example.py
```

Exits 0 on success, prints the metrics dict + `eval_inference: OK`.
