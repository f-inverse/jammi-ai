# Image search

Run image-to-image semantic search over a corpus with an OpenCLIP-format
vision model, then measure retrieval quality.

**When to use this pattern.** You have a corpus of images (figures, drawings,
photos) and want to find the ones most similar to a query image — and a number
that tells you how good the retrieval is. This is the image counterpart of the
text `eval_embeddings` recipe.

## Flow

1. **Load** a small image corpus (inline image bytes in a Parquet source)
2. **Generate** L2-normalized vision embeddings over the image column
3. **Search** the index with an encoded image query (cosine ANN)
4. **Eval** retrieval quality (Recall@K / MRR) against a held-out golden set

## Model

The example uses **PatentCLIP** as the reference model — it is the federal
patent-figure-search use case driving this recipe:

```bash
JAMMI_IMAGE_MODEL=patentclip/PatentCLIP_Vit_B \
    python cookbook/recipes/image_search/example.py
```

`patentclip/PatentCLIP_Vit_B` is pulled from the Hugging Face Hub on first use
and produces **512-dim** L2-normalized embeddings. Any OpenCLIP-format model
works the same way — OpenAI CLIP, LAION `CLIP-ViT-B-32-*`, EVA-CLIP, etc. — the
encoder is auto-detected from the model's `open_clip_config.json`.

By **default** (no env var) the recipe runs against the hermetic
`cookbook/fixtures/tiny_open_clip` fixture so it runs offline in CI in under a
few seconds. That fixture has random weights, so its retrieval numbers are
meaningless — it exercises the full pipeline, not model quality. Use PatentCLIP
(or any real model) for real numbers.

## What `example.py` does

1. Connects to a temporary artifact dir
2. Reads the 20 committed 224×224 PNGs under
   `cookbook/fixtures/tiny_image_corpus/` into a Parquet `corpus` source
   (`image_id`, `image` bytes)
3. `db.generate_image_embeddings(source="corpus", model=MODEL, image_column="image", key="image_id")`
4. `db.encode_image_query(MODEL, png_bytes)` → `db.search("corpus", query=vec, k=5).run()`
5. Builds the image-query golden source from `tiny_image_golden.json` and calls
   `db.eval_embeddings(source="corpus", golden_source="golden.public.golden", k=5)`
6. Prints the aggregate Recall@K / precision@K / MRR / nDCG and the per-query
   records. It **reports** the metrics; it does **not** assert a quality bar.

## Stepwise scripts

`example.py` runs all four phases in one process (this is the version wired into
`tests/cookbook_smoke.py`). The numbered scripts decompose the same flow and
share a persistent workdir, so run them in order:

```bash
python cookbook/recipes/image_search/01-load-corpus.py
python cookbook/recipes/image_search/02-generate-embeddings.py
python cookbook/recipes/image_search/03-search.py
python cookbook/recipes/image_search/04-eval.py
```

## API surface exercised

- `Database.generate_image_embeddings(*, source, model, image_column, key)`
- `Database.encode_image_query(model_id, image_bytes)` → `list[float]`
- `Database.search(source, *, query, k)` → `SearchBuilder` → `.run()`
- `Database.eval_embeddings(*, source, golden_source, model=None, k=10)`

## Input schema

| column     | type            | notes                                   |
|------------|-----------------|-----------------------------------------|
| `image_id` | utf8            | per-row key                             |
| `image`    | binary          | raw PNG/JPEG/TIFF bytes (decoded by the encoder) |

Preprocessing (pad-to-square, no center crop, normalization, L2-normalized
output) is handled inside the encoder per the model's `preprocess_cfg`.

## Golden source shape (image mode)

`eval_embeddings` switches to image-query mode when the golden source carries a
`query_image` (binary) column instead of `query_text`:

| column        | type   | example                          |
|---------------|--------|----------------------------------|
| `query_id`    | utf8   | `q_circle`                       |
| `query_image` | binary | raw PNG bytes of the query image |
| `relevant_id` | utf8   | `img_circle_0` (matches `image_id`) |

## Fixtures

- `cookbook/fixtures/tiny_image_corpus/` — 20 synthetic 224×224 PNGs in 5 shape
  families (circle / triangle / square / hexagon / grating), 4 per family, plus
  a held-out query image per family under `queries/`. Rendered programmatically
  by `cookbook/fixtures/generate.py` — **no real patent imagery** (licensing).
- `cookbook/fixtures/tiny_image_golden.json` — per-query → expected corpus IDs
  (same shape family).
- `cookbook/fixtures/tiny_open_clip/` — tiny offline OpenCLIP fixture used as
  the default CI model.

## Run it

```bash
python cookbook/recipes/image_search/example.py
```

Exits 0 on success, prints the top-K and the metrics dict + `image_search: OK`.
