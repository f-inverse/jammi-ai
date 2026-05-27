# Evaluate inference (NER)

Run a token-classification model over a registered source and score its
predicted entity spans against gold spans.

**When to use this pattern.** You have a labelled NER holdout set (one
gold span per row) and you want strict entity-level precision, recall,
and F1 — both overall and per entity type — to compare two NER models
or to track regressions on the same one.

## What `example.py` does

1. Connects to a temporary artifact dir
2. Registers `tiny_ner_corpus.parquet` as `corpus` (parquet)
3. Registers `tiny_ner_gold.csv` as `golden` (csv) — one row per gold
   entity span: `(id, label, start, end)`
4. Runs `db.eval_inference` with the local `tiny_modernbert_ner`
   fixture against the `text` column, `task="ner"`
5. Prints the returned aggregate `precision`, `recall`, `f1`, the
   per-type breakdown, and the count of per-record predictions
6. Asserts every reported rate is in `[0.0, 1.0]`

## API surface exercised

- `Database.eval_inference(*, model, source, columns, task, golden_source, label_column)`

The returned dict carries `aggregate` (tagged by `"task"` — `"ner"` for
this recipe) with `precision`, `recall`, `f1`, and `per_type` (one
breakdown per entity type the model emitted or the gold set carried),
plus `per_record` (one entry per aligned `{record_id, predicted, gold}`
where `predicted` and `gold` are entity-span lists, each tagged
`"task": "ner"`).

The `task` argument is the string form of the inference task — `"ner"`
here. For classification, see `../eval_inference/`.

## Golden source shape

`eval_inference` with `task="ner"` requires a registered source with
these columns — one row per entity span (multiple spans on the same
`id` accumulate into one per-row gold set):

| column           | type | example      |
|------------------|------|--------------|
| `id`             | utf8 | `"1"`        |
| `<label_column>` | utf8 | `PER`        |
| `start`          | i64  | `0`          |
| `end`            | i64  | `13`         |

`label_column` is the kwarg you pass at call time — `label` in this
recipe. `start` is inclusive, `end` is exclusive, both byte offsets
into the source row's text column. The label set must match the
shipped model's `id2label` minus the `B-`/`I-` prefixes —
`tiny_modernbert_ner` knows `PER` and `ORG` only.

Rows in the source without a matching gold `id` are silently dropped
from the metric (same alignment rule the classification recipe uses).

## Run it

```bash
python cookbook/recipes/eval_inference_ner/example.py
```

Exits 0 on success, prints the metrics dict + `eval_inference (ner): OK`.
