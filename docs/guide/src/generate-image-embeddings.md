# Generate Image Embeddings

Generate vector embeddings from images using an OpenCLIP-compatible vision model. Results are persisted to Parquet with sidecar ANN indexes, identical to text embeddings — the same `search()`, evaluation, and SQL tools work on both.

The OpenCLIP family is cross-modal: the vision tower and the text tower in the same checkpoint produce embeddings in a shared latent space, so a text query encoded with the same model can search image embeddings directly. See [Search Text Against Images (Cross-Modal)](./cross-modal-search.md) for the full text-to-image recipe.

## Basic usage

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# async fn ex(session: &InferenceSession) -> jammi_db::error::Result<()> {
let record = session.generate_image_embeddings(
    "figures",
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "image",       // column containing image data
    "figure_id",   // key column
).await?;

println!("Embedded {} images, {} dimensions", record.row_count, record.dimensions.unwrap());
# Ok(()) }
```

### Python

```python
db.generate_image_embeddings(
    source="figures",
    model="laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    image_column="image",
    key="figure_id",
)
```

## Image column format

The image column can be either:

- **Binary** — inline image bytes (PNG, JPEG, TIFF) stored directly in Parquet
- **Utf8** — file paths pointing to images on disk

## Image preprocessing

Each image is automatically preprocessed before embedding:

1. **Pad to square** — white canvas, image centered (preserves aspect ratio)
2. **Resize** — bicubic interpolation to the model's input size (224x224 for CLIP)
3. **Normalize** — per-channel normalization using constants from the model's config

Preprocessing parameters (mean, std, image size) are model-driven — parsed from the model's config file, not hardcoded.

## Encode a single image

To embed one image without persistence (e.g., for a query):

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# async fn ex(session: &InferenceSession) -> Result<(), Box<dyn std::error::Error>> {
let image_bytes = std::fs::read("query.png")?;
let vector = session
    .encode_image_query("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", &image_bytes)
    .await?;
// vector: Vec<f32>, L2-normalized, dimensionality = model's embed_dim
# Ok(()) }
```

### Python

```python
with open("query.png", "rb") as f:
    image_bytes = f.read()

vector = db.encode_image_query("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", image_bytes)
```

## Supported models

**OpenCLIP-compatible models** with safetensors weights. The repo must carry:

- `open_clip_config.json` with `model_cfg.vision_cfg` (and `model_cfg.text_cfg` if you want cross-modal text queries)
- `open_clip_model.safetensors` with OpenCLIP weight key naming (`visual.*` for vision, root-level for text)
- Either a `tokenizer.json` or the OpenCLIP-native `bpe_simple_vocab_16e6.txt.gz` (only required for text-side queries)

The architecture (ViT width, layers, heads, patch size, pooling strategy), the shared latent dimensionality (`embed_dim`), and the preprocessing config (mean, std, image size) are detected automatically from the config — no per-model code path.

## Output schema

Same as text embeddings:

| Column | Type | Description |
|--------|------|-------------|
| `_row_id` | Utf8 | Key value |
| `_source_id` | Utf8 | Source identifier |
| `_model_id` | Utf8 | Model identifier |
| `vector` | FixedSizeList(Float32, N) | L2-normalized embedding vector (N = `embed_dim`) |

## Search

Image embeddings work with the same `search()` API as text embeddings:

```python
vector = db.encode_image_query("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", query_bytes)
results = db.search("figures", query=vector, k=10).run()
```

All SearchBuilder operations (join, filter, sort, limit, annotate) compose identically.

## Error handling

| Condition | `_status` | `_error` |
|-----------|-----------|----------|
| Valid image | `"ok"` | null |
| Null image | `"error"` | `"Null or missing image input"` |
| Corrupt image | `"error"` | `"Failed to decode image at row N: ..."` |
