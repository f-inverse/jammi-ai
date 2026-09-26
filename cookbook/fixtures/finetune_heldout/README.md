# finetune_heldout — a committed held-out split of arxiv supervised pairs

1500 same-subject supervised `(anchor, positive, negative)` pairs mined from a
4000-paper ogbn-arxiv subset, with the last 128 held out. The fine-tune run
bench (`crates/jammi-bench/src/finetune_run.rs`, driven by
`ci/scripts/perf/finetune_run_ab.sh`) trains on the train side and scores a
held-out loss on the held-out side; `heldout_ids_sha256` hashes checkout
content, so a run's held-out set is identified by what is committed, never by
a network re-fetch.

## What is committed here

| file | content |
| --- | --- |
| `arxiv_subset_ids.txt` | the 4000 ogbn-arxiv paper ids the pairs are mined from |
| `heldout_ids.txt` | `anchor_id<TAB>positive_id<TAB>negative_id`, one held-out pair per line, in mining order. **This is the file `heldout_ids_sha256` hashes.** |
| `heldout_pairs.jsonl` | the held-out pairs' ids and text (title + ". " + abstract, clipped to 1500 chars), one JSON object per line |
| `train_pairs.jsonl` | the 1372 train-side pairs, in the same shape |
| `train_ids_sha256.json` | the train-side pairs' ids and a per-pair SHA-256 over their text |
| `manifest.json` | `dataset_sha256`, `heldout_ids_sha256`, the seed, the split rule, the batch-size candidates, and provenance (source URLs and pinned checksums) |
| `NOTICE` | ODC-BY 1.0 attribution for the ogbn-arxiv text redistributed here |

## Verification

`ci/scripts/perf/check_heldout_fixture_integrity.py` is hermetic: it proves
the committed files agree with each other — `heldout_pairs.jsonl`'s ids equal
`heldout_ids.txt` exactly and in order, `heldout_ids_sha256` and the
Merkle-style `dataset_sha256` (over all 1500 per-pair hashes) recompute from
checkout content, and `arxiv_subset_ids.txt` matches its recorded digest.

## Provenance

- **Source**: Open Graph Benchmark `ogbn-arxiv` — `arxiv.zip` (graph, labels,
  split) and `titleabs.tsv.gz` (title and abstract text) from
  `snap.stanford.edu`, pinned by SHA-256 in `manifest.json.provenance`. The
  release has been static since 2020.
- **License**: ODC-BY 1.0 — see `NOTICE`.
- **Mining**: papers are traversed in sorted `paper_id` order; for each anchor
  a seeded `np.random.default_rng(0)` draws a positive of the same subject, a
  negative subject, then a negative within it. The pair text is
  `title + ". " + abstract`, clipped to 1500 chars.

## Split rule

The last 128 pairs in mining order are held out; the leading 1372 are
train-side. The split is by id, not by `validation_fraction` rounding: MNRL's
per-example loss is batch-coupled, so the batch partition must be an identity.
128 is a multiple of both batch sizes in play — 32, which the bench's
learning runs pass, and 8, the engine's `FineTuneConfig` default — so the
held-out set is a whole number of batches either way.
