# In-context prediction

Train a predictor across many small tasks, then predict a row of a task it
has never seen — from the task's own rows, retrieved as context.

**When to use this pattern.** You have many related groups (stores,
patients, products) with few rows each, and want a calibrated prediction
for a new group without training a model per group.

## What `example.py` does

1. Builds a synthetic dataset: 16 tasks of 24 rows, each task with its own
   level; a row's 4-dimensional features cluster by task
2. `import_embeddings` — the features become the embedding table (bring
   your own vectors)
3. `train_context_predictor` — meta-train across tasks; a quarter of the
   tasks are held out
4. `predict_with_context_predictor` — a mean and a spread for rows of a
   held-out task, with the context rows that informed each prediction —
   compared with the best context-free guess, the overall mean

## Run it

```bash
python cookbook/recipes/context_predictor/example.py
```

Seconds on CPU. The book's *Predict* chapter runs the same verbs on a
citation graph, with conformal calibration on top.
