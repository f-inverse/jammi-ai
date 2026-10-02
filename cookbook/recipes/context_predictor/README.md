# In-context prediction

Train a predictor across many small tasks, then predict a row of a task it
has never seen — from the task's own rows, retrieved as context.

**When to use this pattern.** You have many related groups (stores,
patients, products) with few rows each, and want a calibrated prediction
for a new group without training a model per group.

## Run it

```bash
python cookbook/recipes/context_predictor/example.py
```

Seconds on CPU. The book's *Predict* chapter runs the same verbs on a
citation graph, with conformal calibration on top.
