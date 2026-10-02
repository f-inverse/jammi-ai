# The Jammi Cookbook

Learn every Jammi capability by running it. The book is a learning path —
start here, search, models and inference, evaluation, fine-tuning, graphs and
prediction, data that changes, where data lives, running Jammi — and ends in a
case study that puts the pieces together for graph machine learning on
ogbn-arxiv. Every chapter is code that runs and checks what it measured; every
recipe under `cookbook/recipes/` is a chapter too. The rendered book:
<https://f-inverse.github.io/jammi-ai/cookbook/>.

## Repository layout

```
jammi_cookbook/   the shared lib: datasets, encoders, frozen goldens, rails
  goldens/        the frozen measurements, one file per dataset and scale
chapters/         the book (Quarto .qmd with executable Python cells); its order lives
                  in _quarto.yml
  recipes/        the recipes as chapters, generated from cookbook/recipes and
                  cookbook/quickstart by scripts/build_notebooks.py
scripts/          the API-reference, citation and no-deferral guards
tests/            the lib's unit tests
```

The book lives in the engine monorepo at `cookbook/book/`; the engine CI renders
the chapters a change can move against the wheels, server and CLI that same run
built (the book jobs in `.github/workflows/ci.yml` at the repo root), and the
whole book nightly (`.github/workflows/cookbook-render.yml`).

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
JAMMI_COOKBOOK_SCALE=full quarto render chapters/learn/learn.qmd   # one chapter at `full`, on a GPU
```

The chapters that start a `jammi-server` of their own need the binary on `PATH`
(`cargo build --release -p jammi-server`). The book's spine is **`connect(target)`
parity**: a recipe is written once and the only thing that changes is the target —
`connect("file://…")` for the embedded engine, `connect("grpc://…")` for a
`jammi-server`.

## License

Apache-2.0 (see `LICENSE`). Depends on, and is published separately from, the
`jammi` engine. Dataset attributions are in `NOTICE` and the loader docstrings.
