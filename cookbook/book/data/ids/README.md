# `data/ids/` — committed subset `_row_id` lists

The source of truth for *which* rows the book runs on. A seed does not reproduce
the same node selection across library versions, so the loaders (K2) write the
selected `_row_id` list here — one id per line, `<dataset>.txt` — and commit it.
`jammi_cookbook.determinism.committed_ids(dataset)` reads it; the seed is
recorded only for provenance.

Expected files (added by K2):

- `arxiv.txt` — the connected ogbn-arxiv subgraph's paper ids.
- `air.txt` — the Air Routes airport subset (or the full small graph).
