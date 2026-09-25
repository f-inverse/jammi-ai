# The Theory↔Computation Cookbook

A **runnable book** that bridges *applied theory* — Stanković et al.'s *Data
Analytics on Graphs* (Foundations & Trends in ML, 2020) and the modern GNN canon
— with *software and computation*: the [`jammi`](https://pypi.org/project/jammi-ai/)
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
jammi_cookbook/   the shared lib: datasets, encoders, frozen goldens, rails
  goldens/        the frozen measurements, one file per dataset and scale
chapters/         the book (Quarto .qmd with executable Python cells)
scripts/          the API-reference, citation and no-deferral guards
tests/            the lib's unit tests
```

The book lives in the engine monorepo at `cookbook/book/`; the engine CI builds
the HEAD wheels and renders the chapters a diff can move against them (see
`.github/workflows/cookbook-book.yml` at the repo root).

## Two scales, one code path

Every chapter runs its capability live and checks what it measured against a
frozen golden. `JAMMI_COOKBOOK_SCALE` picks what it runs over:

* `small` (the default) — the committed fixtures and tiny fixture encoders, on
  the CPU, in seconds per chapter. What CI renders.
* `full` — the published datasets (fetched once, checksum-gated, into
  `~/.cache/jammi-cookbook`) and real encoders, on a GPU.

The chapter code is identical at both; only the data, the encoders and the
goldens differ. A golden is never typed in: `JAMMI_COOKBOOK_FREEZE=1` records
what a run measured, and re-freezing after a deliberate change is running the
chapter once at that scale and reviewing the diff.

## Develop

From the repo root, build the HEAD engine wheel, then install the book against it:

```bash
# HEAD embed engine (CPU): install the base client (import `jammi`), generate
# stubs, then build + install the compiled native wheel from packaging/native.
pip install -e 'clients/python[dev]' && make -C clients/python generate
(cd packaging/native && maturin build --release --out ../../target/wheels)
pip install --force-reinstall --no-deps target/wheels/jammi_ai_native-*.whl

pip install -e 'cookbook/book[book,dev]'   # the jammi-ai client is unpinned; the HEAD engine above wins
cd cookbook/book
python scripts/check_api_reference.py      # confirm the API reference matches the wheel
pytest                                     # lib unit tests
quarto render                              # run every chapter at `small` scale
JAMMI_COOKBOOK_SCALE=full quarto render chapters/03-learn/learn.qmd   # one chapter at `full`, on a GPU
```

The chapters that start a `jammi-server` of their own need the binary on `PATH`
(`cargo build --release -p jammi-server`). The book's spine is **`connect(target)`
parity**: a recipe is written once and the only thing that changes is the target —
`connect("file://…")` for the embedded engine, `connect("grpc://…")` for a
`jammi-server`.

## License

Apache-2.0 (see `LICENSE`). Depends on, and is published separately from, the
`jammi` engine. Dataset attributions are in `NOTICE` and the loader docstrings.
