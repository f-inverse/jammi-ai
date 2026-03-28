# Generate Image Embeddings

Generate vector embeddings from images using a vision model. Results are persisted to Parquet with sidecar ANN indexes, identical to text embeddings — the same search, evaluation, and SQL tools work on both.

## Basic usage

### Rust

```rust
let record = session.generate_image_embeddings(
    "figures",
    "patentclip/PatentCLIP_Vit_B",
    "image",       // column containing image data
    "figure_id",   // key column
).await?;

println!("Embedded {} images, {} dimensions", record.row_count, record.dimensions.unwrap());
```

### Python

```python
db.generate_image_embeddings(
    source="figures",
    model="patentclip/PatentCLIP_Vit_B",
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

```rust
let image_bytes = std::fs::read("query.png")?;
let vector = session.encode_image_query("patentclip/PatentCLIP_Vit_B", &image_bytes).await?;
// vector: Vec<f32>, 512 dimensions, L2-normalized
```

### Python

```python
with open("query.png", "rb") as f:
    image_bytes = f.read()

vector = db.encode_image_query("patentclip/PatentCLIP_Vit_B", image_bytes)
```

## Supported models

**OpenCLIP-compatible vision models** with safetensors weights on HuggingFace Hub:

- `patentclip/PatentCLIP_Vit_B` — CLIP ViT-B fine-tuned on 500K+ design patents (512-dim)

The model must have:
- `open_clip_config.json` with `model_cfg.vision_cfg`
- `open_clip_model.safetensors` with OpenCLIP weight key naming

The architecture (ViT width, layers, heads, patch size, pooling strategy) and preprocessing config (mean, std) are detected automatically from the config.

## Output schema

Same as text embeddings:

| Column | Type | Description |
|--------|------|-------------|
| `_row_id` | Utf8 | Key value |
| `_source_id` | Utf8 | Source identifier |
| `_model_id` | Utf8 | Model identifier |
| `vector` | FixedSizeList(Float32, N) | L2-normalized embedding vector |

## Search

Image embeddings work with the same `search()` API as text embeddings:

```python
vector = db.encode_image_query("patentclip/PatentCLIP_Vit_B", query_bytes)
results = db.search("figures", query=vector, k=10).run()
```

All SearchBuilder operations (join, filter, sort, limit, annotate) compose identically.

## Error handling

| Condition | `_status` | `_error` |
|-----------|-----------|----------|
| Valid image | `"ok"` | null |
| Null image | `"error"` | `"Null or missing image input"` |
| Corrupt image | `"error"` | `"Failed to decode image at row N: ..."` |
