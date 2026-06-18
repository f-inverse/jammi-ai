# `artifacts/` — the committed golden-sample cache (K0 layer 2)

The keystone slice (KV-arxiv) runs the heavy pipeline **once** and commits its
small-subset outputs here, under `artifacts/<dataset>/`:

- the artifact files themselves (`embeddings.parquet`, `neighbor_graph.parquet`,
  `propagated.parquet`, model-id records, the calibration/test split), and
- `golden_metrics.json` — the frozen metrics, each `{ "value": …, "tol": … }`.

Every later chapter **reads** this cache (`jammi_cookbook.contracts.load_artifact`)
and asserts its measured verdict against `golden_metrics.json`
(`jammi_cookbook.contracts.assert_close`). Chapters never recompute upstream —
re-execution drifts numbers and breaks CI tolerances with no provenance.

Only the small committed subset lives here. The full datasets are fetched on
demand (checksum-gated) by the loaders and are never committed.
