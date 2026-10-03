# Cross-modal search

Search an image corpus with a **text** query: a CLIP-family checkpoint's text
tower encodes the query, and its vision tower built the image index, both
projecting into one shared space.

**When to use this pattern.** You have images (figures, drawings, photos) and
want to find them by describing them in words, with no captions to index.

## Model

The recipe runs LAION's CLIP ViT-B/32
(`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`) from the Hugging Face Hub, downloaded
on first use (about 600 MB). Its text tower encodes the words and its vision
tower the images, so the program asks for each shape family by name and finds
its drawings, though the model never saw these drawings in training.

Guide: [Search Text Against Images (Cross-Modal)](../../../docs/guide/src/cross-modal-search.md).

## Run it

```bash
python cookbook/recipes/cross_modal_search/example.py
```
