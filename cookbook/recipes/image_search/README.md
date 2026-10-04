# Image search

Run image-to-image semantic search over a corpus with an OpenCLIP-format
vision model, measure retrieval quality, and adapt the vision tower itself on
caller-supplied image triplets.

**When to use this pattern.** You have a corpus of images (figures, drawings,
photos) and want to find the ones most similar to a query image — and a number
that tells you how good the retrieval is. This is the image counterpart of the
text `eval_embeddings` recipe.

## Model

The recipe runs LAION's CLIP ViT-B/32
(`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`, an OpenCLIP checkpoint with 512-dim
L2-normalized embeddings), downloaded from the Hugging Face Hub on first use
(about 600 MB) and cached. Any OpenCLIP-format checkpoint works the same way —
OpenAI CLIP, other LAION sizes, EVA-CLIP, or one tuned on a domain's imagery,
such as `patentclip/PatentCLIP_Vit_B` for technical drawings — the encoder is
read from the checkpoint's `open_clip_config.json`. Change `MODEL` in the
program to try one.

## API surface exercised

- `Session.generate_embeddings(*, source, model, columns, key, modality="image")`
- `Session.encode_query(*, model, query, modality="image")` → `list[float]`
- `Session.search(source, *, query, k, filter=None, select=None)` → `pyarrow.Table`
- `Session.eval_embeddings(*, source, golden_source, model=None, k=10)`
- `Session.fine_tune(*, source, base_model, columns, method, task="image_embedding", target_modules=[...], ...)` → `TrainingJob`
- `Session.describe_model(model_id)` → `dict | None`

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
  by `cookbook/fixtures/generate.py` — **no real-world imagery** (licensing).
- `cookbook/fixtures/tiny_image_golden.json` — per-query → expected corpus IDs
  (same shape family).

## Run it

```bash
python cookbook/recipes/image_search/example.py
```

It prints the top five, the retrieval metrics, how far the adapted tower moved
the query's vector, and the refusal's message.
