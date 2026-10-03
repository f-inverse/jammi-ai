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
jammi_cookbook/   the shared lib: datasets, encoders, claims, rails, data fixtures
chapters/         the book (Quarto .qmd with executable Python cells); its order lives
                  in _quarto.yml
  recipes/        the recipes as chapters, generated from cookbook/recipes and
                  cookbook/quickstart by scripts/build_notebooks.py
scripts/          the API-reference, citation and no-deferral guards
tests/            the lib's unit tests
```

The book lives in the engine monorepo at `cookbook/book/`; the engine CI renders
the pages a change can move against the wheels, server and CLI that same run
built (the book jobs in `.github/workflows/ci.yml` at the repo root). When that is
every page — a release's tree always is — the run also assembles the book, and the
release publishes it with the guide (`.github/workflows/pages.yml`).

## Two scales, one code path

Every chapter runs its capability live and ends each finding in the claim it
makes: `claim(statement, holds, evidence)` prints the statement beside the run's
own numbers and raises, naming it, when it does not hold. A claim is a relation
the capability guarantees — an ordering, a bound, an equality the engine defines —
so it holds on any host and GPU; a chapter never checks a number one machine
measured once. `JAMMI_COOKBOOK_SCALE` picks what a chapter runs over:

* `small` (the default) — the committed samples of the datasets and a compact
  sentence encoder (`all-MiniLM-L6-v2`), in seconds to minutes per chapter on a
  CPU. What CI renders.
* `full` — the published datasets (fetched once, checksum-gated, into
  `~/.cache/jammi-cookbook`) and a larger text encoder (`gte-modernbert-base`),
  on a GPU.

Both scales run real, pretrained models from the Hugging Face Hub, downloaded
on first use; the chapter code is identical at both, and only the data and the
text encoder differ. A chapter runs on the GPU when the reader has one. A
claim only the published data can show — a finding the small samples are too few to bear —
is guarded by `SCALE is Scale.FULL`, and every other claim holds at both scales.

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
