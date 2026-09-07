# Image search

Run image-to-image semantic search over a corpus with an OpenCLIP-format
vision model, measure retrieval quality, and adapt the vision tower itself on
caller-supplied image triplets.

**When to use this pattern.** You have a corpus of images (figures, drawings,
photos) and want to find the ones most similar to a query image — and a number
that tells you how good the retrieval is. This is the image counterpart of the
text `eval_embeddings` recipe.

## Flow

1. **Load** a small image corpus (inline image bytes in a Parquet source)
2. **Generate** L2-normalized vision embeddings over the image column
3. **Search** the index with an encoded image query (cosine ANN)
4. **Eval** retrieval quality (Recall@K / MRR) against a held-out golden set
5. **Fine-tune** LoRA adapters inside the vision tower on image triplets
   (adapted ≠ base), then watch a wrong selector get **refused**

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
3. `db.generate_embeddings(source="corpus", model=MODEL, columns=["image"], key="image_id", modality="image")`
4. `db.encode_query(model=MODEL, query=png_bytes, modality="image")` → `db.search("corpus", query=vec, k=5)` (returns a `pyarrow.Table`)
5. Builds the image-query golden source from `tiny_image_golden.json` and calls
   `db.eval_embeddings(source="corpus", golden_source="golden.public.golden", k=5)`
6. Prints the aggregate Recall@K / precision@K / MRR / nDCG and the per-query
   records. It **reports** the metrics; it does **not** assert a quality bar.
7. Builds synthetic `(anchor, positive, negative)` image triplets from the
   corpus (positive = same shape family, negative = a different family) and
   calls
   `db.fine_tune(source="triplets", base_model=MODEL, columns=["anchor","positive","negative"], method="lora", task="image_embedding", target_modules=["in_proj","c_fc"], ...)`.
   A **non-empty** `target_modules` puts LoRA **inside the vision tower** —
   `in_proj` is the transformer block's fused-QKV projection, `c_fc` the MLP's
   first linear — so the tower's own representation moves. (The audio recipe
   runs both modes side by side: an **empty** list instead trains a projection
   head on a *frozen* tower — cheaper, less capacity.) It then re-encodes the
   **same** query image through the adapted model and asserts the **embedding
   vector** changed (max elementwise `|Δ| > 1e-4` versus the base encoding).
8. Submits one more job with `target_modules=["q_proj"]` — a real selector on
   plenty of decoder checkpoints and on **nothing** in an OpenCLIP tower — and
   asserts `job.wait()` raises `jammi.errors.TrainingError` whose message
   echoes `q_proj` *and* names this tower's real sites (`in_proj`, …). It
   prints the message.

### What each leg proves, and the honesty rule

- **The tower leg** proves the adapter is trained *and applied when the model
  is served*: an adapter that trained but was silently dropped at serve time
  leaves the two query vectors bit-identical, and that is what the `|Δ|` check
  catches. It asserts **change, not improvement** — the default fixture has
  **random weights**, so the *direction* of the change carries no information.
  The vector check is also the deterministic one: asserting a top-k metric
  moved is flaky, because on this tiny eval set the rankings rarely flip even
  when the vectors do.
  What the tower leg canNOT check from here is the saved adapter's *kind* —
  that it is an encoder-adapters bundle carrying the vision tower's id. That
  is engine-internal and is pinned by the engine's own integration tests; the
  client surface (`describe_model`) reports only the model's id, backend, task
  and status, so the recipe asserts the task and leans on the `|Δ|` check for
  the rest.
- **The refusal leg** proves a selector that matches no site fails the *job*
  rather than publishing an adapter that changes nothing — and that the message
  is actionable, carrying this architecture's own site vocabulary.
- The **independently-known improvement number** — tuned retrieval quality
  beating the base by a measured margin — is not this recipe's to claim. It
  belongs to the real-checkpoint chapter, which reads a committed cache
  produced on a GPU. A recipe running a random-weight fixture on a laptop can
  honestly prove mechanism; it cannot prove quality.

The pairing semantics (what a "positive" *means*) are the caller's training
data, not the trainer's: the trainer only minimizes the contrastive triplet
loss over whatever images you pair.

## Stepwise scripts

`example.py` runs every phase in one process (this is the version wired into
`tests/cookbook_smoke.py`). The numbered scripts decompose the search-and-eval
flow and share a persistent workdir, so run them in order:

```bash
python cookbook/recipes/image_search/01-load-corpus.py
python cookbook/recipes/image_search/02-generate-embeddings.py
python cookbook/recipes/image_search/03-search.py
python cookbook/recipes/image_search/04-eval.py
```

## API surface exercised

- `Database.generate_embeddings(*, source, model, columns, key, modality="image")`
- `Database.encode_query(*, model, query, modality="image")` → `list[float]`
- `Database.search(source, *, query, k, filter=None, select=None)` → `pyarrow.Table`
- `Database.eval_embeddings(*, source, golden_source, model=None, k=10)`
- `Database.fine_tune(*, source, base_model, columns, method, task="image_embedding", target_modules=[...], ...)` → `TrainingJob`
- `Database.describe_model(model_id)` → `dict | None`

### Image triplet schema (fine-tune input)

| column     | type   | notes                                   |
|------------|--------|-----------------------------------------|
| `anchor`   | binary | encoded image                           |
| `positive` | binary | an image the caller deems related       |
| `negative` | binary | an image the caller deems unrelated     |

Same column shape as text and audio triplets — `task="image_embedding"` is what
tells the loader to read the three columns as encoded images rather than text.

### Vision-tower LoRA sites

`target_modules` names sites on **this** architecture. The OpenCLIP towers
offer `in_proj` (fused QKV), `out_proj`, `c_fc` and `c_proj`; `all-linear`
selects every one. A selector matches a site name exactly or as a suffix of it.
A list matching **nothing** fails the job with a message that echoes what you
submitted and lists the tower's real names.

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
