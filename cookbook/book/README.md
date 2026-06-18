# The Theory↔Computation Cookbook

A **runnable book** that bridges *applied theory* — Stanković et al.'s *Data
Analytics on Graphs* (Foundations & Trends in ML, 2020) and the modern GNN canon
— with *software and computation*: the [`jammi_ai`](https://pypi.org/project/jammi_ai/)
engine. The signature move of every recipe: **one Jammi recipe = one equation in
the monograph = one line in the GNN canon**, executed and measured.

It reads as **4 tiers × 3 rails**, echoing AWS Neptune's Database / Analytics /
ML split and adding a fourth tier Neptune structurally lacks — calibrated,
provenance-stamped, context-conditioned prediction:

| Tier | Recipe | Theory |
|---|---|---|
| 01 Construct | `build_neighbor_graph` | topology from data (Part I) |
| 02 Analyze | `propagate_embeddings` | graph signal processing = SGC/APPNP (Part II) |
| 03 Learn | `fine_tune_graph` | representation learning on graphs (Part III) |
| 04 Predict & Quantify | `train_context_predictor` + conformal | context-conditioned posterior + honest coverage |

Rails woven through every tier: **provenance**, **tenancy**, **measurement**.

## Repository layout

```
jammi_cookbook/   the shared lib: composes jammi_ai, enforces contracts + rails
chapters/         the book (Quarto .qmd with executable Python cells)
artifacts/        the committed golden-sample cache (small subset only)
data/ids/         committed seeded subset _row_id lists
scripts/          the API-reference guard + the no-deferral grep
.github/workflows the --execute CI harness (PR-gating) + the opt-in full-scale run
```

## Develop

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[book,dev]"          # jammi_ai is pinned to ==0.31.0 (CPU embed wheel)
python scripts/check_api_reference.py # confirm the API reference matches the wheel
pytest                                # lib unit tests
quarto render                         # build + execute the book (reads the committed cache)
```

The book's spine is **`connect(target)` parity**: a recipe is written once and the
only thing that changes is the target — `connect("file://…")` for the embedded CPU
engine the chapters and CI run on, `connect("grpc://…")` for the GPU
`jammi-server` the keystone slice uses to produce the cache.

The heavy work (embedding, fine-tune, context-predictor train) runs **once** in
the keystone slice, on the GPU server (`scripts/build_arxiv_cache.py --target
grpc://…`), and is committed as the small-subset cache; CI and every later chapter
*read* that cache on CPU and assert measured verdicts against frozen golden metrics
within a tolerance. See `CLAUDE.md` for the engineering standards and
`EXECUTION-STATUS.md` for build state.

## License

Apache-2.0 (see `LICENSE`). Depends on, and is published separately from, the
`jammi_ai` engine. Dataset attributions are in `NOTICE` and the loader docstrings.
