# Generate Image Embeddings

Generate vector embeddings from images using a vision model. Results are persisted to Parquet with sidecar ANN indexes, identical to text embeddings — the same search, evaluation, and SQL tools work on both.

## Basic usage

### Rust

```rust
use jammi_ai::pipeline::image_embedding::EmbeddingStrategy;

let record = session.generate_image_embeddings(
    "figures",
    "patentclip/PatentCLIP_Vit_B",
    "image",       // column containing image data
    "figure_id",   // key column
    EmbeddingStrategy::Single,
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

Example source registration with inline images:

```python
# Parquet with binary image column
db.add_source("figures", path="figures.parquet", format="parquet")
```

Example source registration with file paths:

```python
# CSV with image_path column pointing to files on disk
db.add_source("drawings", path="drawings.csv", format="csv")
db.generate_image_embeddings(
    source="drawings",
    model="patentclip/PatentCLIP_Vit_B",
    image_column="image_path",
    key="id",
)
```

## Image preprocessing

Each image is automatically preprocessed before embedding:

1. **Pad to square** — white canvas, image centered (preserves aspect ratio)
2. **Resize** — bicubic interpolation to the model's input size (224x224 for CLIP)
3. **Normalize** — CLIP normalization constants applied per channel

No manual preprocessing is needed.

## Rotation-invariant strategy

For images that may appear at arbitrary orientations (e.g., patent drawings, architectural plans), use the rotation-invariant strategy. Each image is embedded at multiple rotation angles, producing one row per angle:

### Rust

```rust
let record = session.generate_image_embeddings(
    "figures",
    "patentclip/PatentCLIP_Vit_B",
    "image",
    "figure_id",
    EmbeddingStrategy::RotationInvariant {
        angles: vec![0, 90, 180, 270],
    },
).await?;

// 4 rotations x N images = 4N embedding rows
```

### Python

```python
db.generate_image_embeddings(
    source="figures",
    model="patentclip/PatentCLIP_Vit_B",
    image_column="image",
    key="figure_id",
    rotation_angles=[0, 90, 180, 270],
)
```

Row IDs encode the rotation angle: `{key}_r{angle}` (e.g., `fig_001_r90`).

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

The architecture (ViT width, layers, heads, patch size, pooling strategy) is detected automatically from the config.

## Output schema

Same as text embeddings:

| Column | Type | Description |
|--------|------|-------------|
| `_row_id` | Utf8 | Key value (or `{key}_r{angle}` for rotation strategy) |
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
