# Cross-modal search

Search an image corpus with a **text** query: a CLIP-family checkpoint's text
tower encodes the query, and its vision tower built the image index, both
projecting into one shared space.

**When to use this pattern.** You have images (figures, drawings, photos) and
want to find them by describing them in words, with no captions to index.

## Flow

1. **Index** the images with the vision tower (`generate_embeddings(...,
   modality="image")`)
2. **Encode** a text query with the same model's text tower (`encode_query`)
3. **Search** the image index with that vector (`search`)

The recipe checks that the search's top-k is exactly the cosine ranking in the
shared space: every image encoded through the vision tower and scored against
the text vector, ids and scores equal.

## Model

The default is the hermetic `tiny_open_clip` fixture (random weights), so the
recipe runs offline; it checks the mechanics and prints the ranking without
judging it. For the semantics, point it at a trained checkpoint:

```bash
JAMMI_CROSS_MODAL_MODEL=laion/CLIP-ViT-B-32-laion2B-s34B-b79K \
  python cookbook/recipes/cross_modal_search/example.py
```

## Run

```bash
python cookbook/recipes/cross_modal_search/example.py
```

Guide: [Search Text Against Images (Cross-Modal)](../../../docs/guide/src/cross-modal-search.md).
