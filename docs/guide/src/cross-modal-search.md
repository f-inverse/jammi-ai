# Search Text Against Images (Cross-Modal)

OpenCLIP-family models carry both a vision tower and a text tower in the same checkpoint, with both towers projecting into a shared latent space. That means a text query embedded with the text tower lives in the same vector space as image embeddings produced by the vision tower — vector search against an image corpus accepts a text query directly, no separate text encoder, no projection bridge.

This recipe shows the full path: index images with the vision tower, embed a text query with the text tower, run `search()`.

## 1. Index the image corpus with the vision tower

### Rust

```rust,no_run
# extern crate jammi_engine;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# async fn ex(session: &InferenceSession) -> jammi_engine::error::Result<()> {
session.generate_image_embeddings(
    "figures",
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "image",       // column containing image data
    "figure_id",   // key column
).await?;
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

## 2. Embed a text query with the same model's text tower

`encode_text_query` dispatches to the OpenCLIP text tower when the model ID resolves to an OpenCLIP checkpoint. The output vector dimensionality matches `embed_dim` — the same dim the image embeddings carry.

### Rust

```rust,no_run
# extern crate jammi_engine;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# async fn ex(session: &InferenceSession) -> jammi_engine::error::Result<()> {
let query_vec = session
    .encode_text_query(
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        "a red circle on a white background",
    )
    .await?;
// query_vec: Vec<f32>, L2-normalized, same length as the image embedding vector
# Ok(()) }
```

### Python

```python
query_vec = db.encode_text_query(
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "a red circle on a white background",
)
```

## 3. Search image embeddings with the text vector

### Rust

```rust,no_run
# extern crate jammi_engine;
# extern crate jammi_ai;
# extern crate tokio;
# use std::sync::Arc;
# use jammi_ai::session::InferenceSession;
# async fn ex(session: Arc<InferenceSession>, query_vec: Vec<f32>) -> jammi_engine::error::Result<()> {
let results = session.search("figures", query_vec, 10).await?.run().await?;
# Ok(()) }
```

### Python

```python
results = db.search("figures", query=query_vec, k=10).run()
```

The hydrated results carry your source's columns (`figure_id`, plus any joined / annotated columns) alongside the `similarity` score. All `SearchBuilder` operations — `join`, `filter`, `sort`, `limit`, `annotate` — compose identically to text-against-text search.

## Why this works

Both towers in an OpenCLIP checkpoint emit vectors of size `embed_dim` (the shared latent dimensionality declared at the top of `open_clip_config.json`). The vision tower applies a `visual.proj` matrix after pooling its patch tokens; the text tower applies a `text_projection` matrix after pooling at the `<|endoftext|>` token. The two projections are jointly trained so the cosine similarity between a text vector and an image vector reflects semantic alignment.

If you embed text and images with separate models (e.g. a BERT encoder + a vision model that wasn't jointly trained with it), the resulting vectors don't share a latent space and the similarities are meaningless. Cross-modal search only works when both modalities are projected by the same CLIP-style joint training.

## Model requirements

Same as [Generate Image Embeddings](./generate-image-embeddings.md), plus:

- `open_clip_config.json` must contain a populated `model_cfg.text_cfg` (with `width`, `layers`, and either `heads` or a `width` that is a multiple of 64).
- The safetensors checkpoint must contain the text-tower keys: `token_embedding.weight`, `positional_embedding`, `transformer.resblocks.*`, `ln_final.*`, and `text_projection`.
- A tokenizer must be available — either an HF-converted `tokenizer.json` or the OpenCLIP-native `bpe_simple_vocab_16e6.txt.gz`.
