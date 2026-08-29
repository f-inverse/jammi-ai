# finetune_heldout — the how-well (unit 63, CONTRACT H3) committed fixture

Committed held-out split of the arxiv same-subject supervised pair set the
`08-finetune-methods` chapter mines (`cookbook/book/scripts/
build_finetune_cache.py::mine_supervision`), over the SAME committed
4000-paper ogbn-arxiv subset the chapter uses
(`cookbook/book/data/ids/arxiv.txt`) and the chapter's own frozen base recall
(`base_recall_at_10 = 0.538`, `cookbook/book/artifacts/finetune/
golden_metrics.json`).

This exists so C16's held-out-loss gate (`docs/plans/63-how-well/
CONTRACT.md` §Frame, §H3) has a `heldout_ids_sha256` that hashes checkout
content: the chapter's own arxiv pairs are mined from a network download at
emit time and are never committed by the chapter itself
(`build_finetune_cache.py:505-507` deletes the intermediate supervised
parquets at the end of every emit).

## What is committed here

| file | content |
| --- | --- |
| `derive_heldout_fixture.py` | the deterministic, scripted derivation (no hand-curation) — a byte-for-byte port of the chapter's `mine_supervision` mining loop, minus the `db`/parquet side effects |
| `heldout_ids.txt` | the explicit committed id list — `anchor_id<TAB>positive_id<TAB>negative_id`, one held-out pair per line, in mining order. **This is the file `heldout_ids_sha256` hashes.** |
| `heldout_pairs.jsonl` | the FULL pair content (ids + title+abstract-derived text, clipped to 1500 chars, exactly as the chapter clips) for the 128 held-out pairs — one JSON object per line |
| `train_ids_sha256.json` | for the remaining 1372 train-side pairs: ids + a per-pair SHA-256 over their text (see "Why train text isn't committed" below) |
| `manifest.json` | `dataset_sha256`, `heldout_ids_sha256`, provenance (source URLs + pinned checksums, snapshot info), the seed, and the batch-size decision |

Total committed fixture size: ~740 KB (text pairs — no model weights).

## Reproducibility

```
python derive_heldout_fixture.py            # regenerate (network download; checksum-gated)
python derive_heldout_fixture.py --check     # verify committed content == a fresh re-derivation
```

`--check` re-downloads the pinned, checksum-gated ogbn-arxiv sources (the same
ones `cookbook/book/jammi_cookbook/datasets.py` uses — a tampered or reissued
source fails the checksum gate, never silently reshapes this fixture),
re-derives the full 1500-pair set, and diffs every committed file against the
fresh derivation byte-for-byte. Confirmed green at fixture authoring time.

## Provenance

- **Source**: Open Graph Benchmark `ogbn-arxiv` — `arxiv.zip` (graph/labels/
  split, pinned SHA-256 `49f85c80…3193fe276`) and `titleabs.tsv.gz` (title +
  abstract text, pinned SHA-256 `7bce99ab…09ae0bc8ed7`), both fetched from
  `snap.stanford.edu`, checksum-gated by `jammi_cookbook.datasets._download`
  (see `manifest.json.provenance` for the full URLs).
- **License**: **ODC-BY 1.0** (Open Graph Benchmark). This is a metadata
  fact recorded here for the fixture's own provenance; it does NOT append the
  NOTICE entry — that attribution edit is owned by docs-ci (CONTRACT H3:
  "ODC-BY 1.0 attribution appended to NOTICE (docs-ci)"), not by this agent
  or this PR.
- **Subset**: the committed 4000-id `cookbook/book/data/ids/arxiv.txt` list
  (unchanged by this fixture — read, never regenerated).
- **Mining**: `mine_supervision`'s algorithm exactly — same-subject grouping,
  `sorted(papers_rows, key=paper_id)` traversal order, `np.random.default_rng
  (SEED=0)` draws in the same per-anchor order (positive pool draw, negative
  subject draw, negative-within-subject draw), `_text()`'s `title + ". " +
  abstract`, clipped to 1500 chars. 1500 pairs total — matches
  `build_finetune_cache.py`'s `N_PAIRS = 1500`.
- **Snapshot**: derived 2026-08-28 against the pinned source checksums above
  (the OGB arxiv release is static — `arxiv.zip` and `titleabs.tsv.gz` have
  been unchanged since 2020, per their pinned digests).

## Split rule

The **last 128 pairs** in the deterministic mining order become the held-out
set; the leading 1372 remain train-side. Recorded in `manifest.json.
split_rule`. This is an id-based split — not `validation_fraction` rounding
(CONTRACT H3 explicitly forbids validation_fraction rounding here, since MNRL
per-example loss is batch-coupled and the batch partition must be identity).

## Batch-size pre-registration flag (needs lead confirmation)

**128 was chosen because it is a multiple of BOTH candidate batch sizes found
in the repo — flagging both for lead confirmation, per the task's
pre-registration requirement:**

1. **`BATCH = 32`** — `cookbook/book/scripts/build_finetune_cache.py:77`
   and `build_finetune_regression_cache.py:69`, the `batch_size` explicitly
   passed to every `db.fine_tune(...)` call in the SAME chapter this fixture's
   pairs are mined from (i.e. the "chapter config" that produced the frozen
   `0.538` base and the pinned `methods.json` recall rows this fixture's
   provenance traces back to).
2. **`batch_size = 8`** — `crates/jammi-wire/src/fine_tune.rs:412`,
   `FineTuneConfig::default().batch_size` — the engine's own unset-default
   for `db.fine_tune(...)` (also asserted at `crates/jammi-wire/src/
   training.rs:440`).

The repo's *cost-measurement* fine-tune bench (`crates/jammi-bench/src/
finetune_step.rs`, driven by `ci/scripts/perf/finetune_ab.sh` /
`ci/scripts/perf/stacked_sweep.sh`) is a synthetic-triplet throughput
sweep across MANY batch sizes (b1, b8, b16, …) — it measures cost, never
learning (`finetune-step: … synthetic uniform token ids, so it measures
*cost*, never learning`, `docs/maintainer/fine-tune-performance-guide.
md:259`), so it has no single canonical batch_size to cite for a REAL
learning run; it is not a third candidate.

**This agent's pick, pending lead confirmation: candidate (1),
`BATCH = 32`**, because CONTRACT H3 directs deriving "the arxiv pair set the
chapter uses" specifically, and the chapter's own real (non-synthetic)
`db.fine_tune(...)` calls that actually train on this exact pair set always
pass `batch_size=32` explicitly (never relying on the engine's `8` default).
The planned H4 `finetune_run_ab.sh` A/B protocol (`PLAN.md` (c)) drives the
real `TrainingLoopBuilder` over this same fixture with an MNRL held-out loss,
the same shape of run — so `32` is the more directly analogous precedent.
**Because 128 is a multiple of 8 too, this pick does not need to be acted on
before lead confirmation** — the fixture's held-out count is correct either
way; only the *reported* "N held-out = 4 batches of 32" vs "16 batches of 8"
framing in H4's future pre-registration text depends on the answer.

## Why train text isn't committed (repo-size discipline)

CONTRACT H3 item 5: keep the fixture "sane for a committed repo artifact
(text pairs, not model weights)". Committing all 1500 pairs' full text would
add another ~5 MB text dump duplicating a subset of a still-larger public
corpus already available (checksum-gated) via `jammi_cookbook.datasets`.
Instead:

- The **held-out** pairs (the only ones any C16 held-out-loss run reads) are
  committed in full — `heldout_ids_sha256` hashes checkout content per
  CONTRACT H3.
- The **train-side** pairs are committed as ids + a per-pair SHA-256 digest
  of their text. `manifest.json.dataset_sha256` is a Merkle-style hash over
  ALL 1500 per-pair hashes (held-out pairs hashed from their own committed
  text; train pairs hashed from the committed `train_ids_sha256.json`
  digests) — so the FULL 1500-pair set's identity is checkable from checkout
  content alone, without re-downloading or re-committing the train text.
