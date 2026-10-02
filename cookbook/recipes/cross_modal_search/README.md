# Cross-modal search

Search an image corpus with a **text** query: a CLIP-family checkpoint's text
tower encodes the query, and its vision tower built the image index, both
projecting into one shared space.

**When to use this pattern.** You have images (figures, drawings, photos) and
want to find them by describing them in words, with no captions to index.

## Model

The default is the hermetic `tiny_open_clip` fixture (random weights), so the
recipe runs offline; it checks the mechanics and prints the ranking without
judging it. For the semantics, point it at a trained checkpoint:

```bash
JAMMI_CROSS_MODAL_MODEL=laion/CLIP-ViT-B-32-laion2B-s34B-b79K \
  python cookbook/recipes/cross_modal_search/example.py
```

Guide: [Search Text Against Images (Cross-Modal)](../../../docs/guide/src/cross-modal-search.md).

## Run it

```bash
python cookbook/recipes/cross_modal_search/example.py
```
