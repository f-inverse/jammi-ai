#!/usr/bin/env python3
"""Derive the committed how-well held-out fixture (63-how-well H3).

Re-derives the SAME 1500 (anchor, positive, negative) same-subject supervised
pairs the arxiv fine-tune chapter mines (``cookbook/book/scripts/
build_finetune_cache.py::mine_supervision``), over the SAME committed 4000-paper
ogbn-arxiv subset (``cookbook/book/data/ids/arxiv.txt``), then carves off a
held-out tail sized to an explicit multiple of the fine-tune protocol's
``batch_size`` (see README.md — flagged for lead pre-registration) and commits:

* ``heldout_ids.txt`` — the explicit id list (anchor_id, positive_id,
  negative_id per line, tab-separated) for the held-out pairs. This is the file
  ``heldout_ids_sha256`` hashes — checkout content, never a network re-fetch.
* ``heldout_pairs.jsonl`` — the FULL pair content (ids + title+abstract-derived
  text) for the held-out pairs only, one JSON object per line, in mining order.
* ``train_ids_sha256.json`` — for the remaining (train-side) pairs: ids +
  a per-pair SHA-256 over their (unmounted) text, so the full 1500-pair
  identity is checkable against committed content without committing every
  train pair's text (keeps the repo artifact text-pairs-sized, not a full
  corpus dump — CONTRACT H3 item 5).
* ``manifest.json`` — dataset_sha256 (a Merkle-style hash over all 1500
  per-pair hashes, itself reconstructable from committed content only),
  heldout_ids_sha256, provenance (source URLs + pinned checksums, snapshot
  date), the seed, and the batch_size decision.

The mining algorithm below is a byte-for-byte port of ``mine_supervision`` in
``build_finetune_cache.py`` (subject grouping, RNG draw order, text clipping)
minus the ``db.add_source``/parquet side effects — this script produces the
same 1500 pairs an actual chapter emit run mines, from the same committed
subset, the same pinned checksum-gated downloads, and the same seed.

Usage::

    python derive_heldout_fixture.py            # (re)generate the committed fixture
    python derive_heldout_fixture.py --check     # verify committed content still
                                                  # matches a fresh re-derivation
                                                  # (exit 1 on any divergence)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent
_BOOK_ROOT = FIXTURE_DIR.parent.parent / "book"
sys.path.insert(0, str(_BOOK_ROOT))

import numpy as np  # noqa: E402

from jammi_cookbook import datasets, determinism  # noqa: E402

# --------------------------------------------------------------------------- #
# Pinned identity — mirrors build_finetune_cache.py exactly (cited in README).
# --------------------------------------------------------------------------- #
SUBSET = 4000
N_PAIRS = 1500
TEXT_CLIP = 1500

# Held-out sizing (CONTRACT H3 item 2 / PLAN.md v2 delta 2): the held-out set
# must be an explicit multiple of the fine-tune protocol's batch_size. TWO
# candidates exist in the repo (see README.md "Batch-size pre-registration
# flag" — this is a FLAGGED, not yet lead-confirmed, choice):
#   (A) BATCH = 32 — cookbook/book/scripts/build_finetune_cache.py:77 and
#       build_finetune_regression_cache.py:69, the batch_size explicitly
#       passed to every db.fine_tune(...) call in the chapter that mines this
#       exact pair set (the "chapter config").
#   (B) batch_size = 8 — crates/jammi-wire/src/fine_tune.rs:412,
#       FineTuneConfig::default().batch_size (the engine's own unset-default,
#       the "bench/engine config").
# 128 is a multiple of BOTH candidates (128 = 4*32 = 16*8), so sizing the
# held-out set to 128 is correct under either resolution of the flagged
# question -- no rework needed once the lead confirms which the A/B protocol
# (PLAN.md (c) / the future finetune_run_ab.sh) pins.
N_HELDOUT = 128
BATCH_SIZE_CANDIDATES = {"chapter_config_build_finetune_cache_py": 32,
                          "engine_default_FineTuneConfig": 8}
assert all(N_HELDOUT % b == 0 for b in BATCH_SIZE_CANDIDATES.values())

SPLIT_RULE = (
    "last N_HELDOUT pairs (by the deterministic mining order — sorted "
    "paper_id traversal, seeded rng draws) become held-out; the leading "
    "N_PAIRS - N_HELDOUT pairs remain train-side."
)


def _text(row: dict) -> str:
    return ((row["title"] or "") + ". " + (row["abstract"] or ""))[:TEXT_CLIP]


def _load_committed_papers() -> list[dict]:
    """papers_rows for the committed 4000-id arxiv subset (matches datasets.load_ogbn_arxiv)."""
    raw = datasets._load_arxiv_raw()
    text = datasets._load_titleabs()
    committed = determinism.committed_ids("arxiv")
    if len(committed) != SUBSET:
        raise RuntimeError(
            f"committed arxiv id list has {len(committed)} ids, expected {SUBSET} — "
            "the chapter's committed subset changed; re-derive with care.")
    pid_to_node = {raw.node2pid[i]: i for i in range(raw.num_nodes)}
    keep_pids = [int(p) for p in committed]
    rows = []
    for pid in keep_pids:
        node = pid_to_node[pid]
        title, abstract = text[pid]
        rows.append({
            "paper_id": str(pid),
            "title": title,
            "abstract": abstract,
            "subject": raw.label_names[raw.labels[node]],
            "year": raw.years[node],
        })
    return rows


def mine_pairs(papers_rows: list[dict]) -> list[dict]:
    """Byte-for-byte port of build_finetune_cache.py::mine_supervision's mining loop.

    Returns N_PAIRS dicts, each carrying anchor/positive/negative paper_id AND
    text, in deterministic mining order (matches the chapter's ft_pairs /
    ft_triplets row order exactly).
    """
    by_subject: dict[str, list[dict]] = {}
    for r in papers_rows:
        by_subject.setdefault(r["subject"], []).append(r)
    subjects = sorted(by_subject)
    rng = np.random.default_rng(determinism.SEED)
    ordered = sorted(papers_rows, key=lambda r: r["paper_id"])

    pairs: list[dict] = []
    for a in ordered:
        if len(pairs) >= N_PAIRS:
            break
        pool = [r for r in by_subject[a["subject"]] if r["paper_id"] != a["paper_id"]]
        if not pool:
            continue
        p = pool[int(rng.integers(len(pool)))]
        other = [s for s in subjects if s != a["subject"]]
        ns = other[int(rng.integers(len(other)))]
        n = by_subject[ns][int(rng.integers(len(by_subject[ns])))]
        pairs.append({
            "anchor_id": a["paper_id"], "anchor_text": _text(a),
            "positive_id": p["paper_id"], "positive_text": _text(p),
            "negative_id": n["paper_id"], "negative_text": _text(n),
        })
    if len(pairs) != N_PAIRS:
        raise RuntimeError(f"mined {len(pairs)} pairs, expected {N_PAIRS}")
    return pairs


def _pair_sha256(pair: dict) -> str:
    payload = "\x00".join([
        pair["anchor_id"], pair["anchor_text"],
        pair["positive_id"], pair["positive_text"],
        pair["negative_id"], pair["negative_text"],
    ]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive() -> dict:
    """Full re-derivation, returning the in-memory fixture (not yet written)."""
    papers_rows = _load_committed_papers()
    pairs = mine_pairs(papers_rows)
    train_pairs = pairs[: N_PAIRS - N_HELDOUT]
    heldout_pairs = pairs[N_PAIRS - N_HELDOUT :]

    per_pair_hashes = [_pair_sha256(p) for p in pairs]
    dataset_sha256 = hashlib.sha256("\n".join(per_pair_hashes).encode("utf-8")).hexdigest()

    return {
        "papers_rows_n": len(papers_rows),
        "pairs": pairs,
        "train_pairs": train_pairs,
        "heldout_pairs": heldout_pairs,
        "dataset_sha256": dataset_sha256,
    }


def _write_heldout_ids(heldout_pairs: list[dict]) -> Path:
    path = FIXTURE_DIR / "heldout_ids.txt"
    lines = [f"{p['anchor_id']}\t{p['positive_id']}\t{p['negative_id']}" for p in heldout_pairs]
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_heldout_pairs(heldout_pairs: list[dict]) -> None:
    path = FIXTURE_DIR / "heldout_pairs.jsonl"
    with path.open("w") as f:
        for p in heldout_pairs:
            f.write(json.dumps(p, sort_keys=True) + "\n")


def _write_train_hashes(train_pairs: list[dict]) -> None:
    rows = [
        {
            "anchor_id": p["anchor_id"], "positive_id": p["positive_id"],
            "negative_id": p["negative_id"], "pair_sha256": _pair_sha256(p),
        }
        for p in train_pairs
    ]
    (FIXTURE_DIR / "train_ids_sha256.json").write_text(
        json.dumps(rows, indent=2))


def _write_manifest(derived: dict, heldout_ids_sha256: str) -> None:
    manifest = {
        "unit": "63-how-well",
        "contract_item": "H3",
        "seed": determinism.SEED,
        "subset_n": SUBSET,
        "n_pairs": N_PAIRS,
        "n_heldout": N_HELDOUT,
        "n_train": N_PAIRS - N_HELDOUT,
        "split_rule": SPLIT_RULE,
        "dataset_sha256": derived["dataset_sha256"],
        "heldout_ids_sha256": heldout_ids_sha256,
        "batch_size_candidates": BATCH_SIZE_CANDIDATES,
        "batch_size_flagged_for_lead_confirmation": True,
        "n_heldout_is_multiple_of_all_candidates": True,
        "provenance": {
            "committed_subset_ids_file": "cookbook/book/data/ids/arxiv.txt",
            "source_zip": {
                "url": datasets._ARXIV_ZIP[0],
                "sha256": datasets._ARXIV_ZIP[1],
            },
            "source_titleabs": {
                "url": datasets._ARXIV_TITLEABS[0],
                "sha256": datasets._ARXIV_TITLEABS[1],
            },
            "license": datasets.LICENSES["ogbn_arxiv"],
            "mining_algorithm": (
                "cookbook/book/scripts/build_finetune_cache.py::mine_supervision "
                "(ported without db/parquet side effects, see mine_pairs() above)"),
        },
    }
    (FIXTURE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def generate() -> None:
    derived = derive()
    heldout_path = _write_heldout_ids(derived["heldout_pairs"])
    _write_heldout_pairs(derived["heldout_pairs"])
    _write_train_hashes(derived["train_pairs"])
    heldout_ids_sha256 = _file_sha256(heldout_path)
    _write_manifest(derived, heldout_ids_sha256)
    print(f"wrote fixture: {FIXTURE_DIR}")
    print(f"  dataset_sha256:     {derived['dataset_sha256']}")
    print(f"  heldout_ids_sha256: {heldout_ids_sha256}")
    print(f"  n_heldout={N_HELDOUT}  n_train={N_PAIRS - N_HELDOUT}")


def check() -> int:
    """Re-derive fresh and diff against committed content. Returns exit code."""
    derived = derive()
    ok = True

    committed_heldout_ids = (FIXTURE_DIR / "heldout_ids.txt").read_text()
    fresh_heldout_ids = "\n".join(
        f"{p['anchor_id']}\t{p['positive_id']}\t{p['negative_id']}"
        for p in derived["heldout_pairs"]) + "\n"
    if committed_heldout_ids != fresh_heldout_ids:
        print("MISMATCH: heldout_ids.txt diverges from a fresh re-derivation", file=sys.stderr)
        ok = False

    committed_manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text())
    if committed_manifest["dataset_sha256"] != derived["dataset_sha256"]:
        print(
            f"MISMATCH: dataset_sha256 committed={committed_manifest['dataset_sha256']} "
            f"fresh={derived['dataset_sha256']}", file=sys.stderr)
        ok = False

    committed_pairs = [json.loads(line) for line in
                       (FIXTURE_DIR / "heldout_pairs.jsonl").read_text().splitlines()]
    if committed_pairs != derived["heldout_pairs"]:
        print("MISMATCH: heldout_pairs.jsonl diverges from a fresh re-derivation",
              file=sys.stderr)
        ok = False

    committed_train = json.loads((FIXTURE_DIR / "train_ids_sha256.json").read_text())
    fresh_train = [
        {"anchor_id": p["anchor_id"], "positive_id": p["positive_id"],
         "negative_id": p["negative_id"], "pair_sha256": _pair_sha256(p)}
        for p in derived["train_pairs"]
    ]
    if committed_train != fresh_train:
        print("MISMATCH: train_ids_sha256.json diverges from a fresh re-derivation",
              file=sys.stderr)
        ok = False

    print("check: OK — committed fixture matches a fresh re-derivation" if ok
          else "check: FAILED — see MISMATCH lines above", file=sys.stderr)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                     help="verify the committed fixture against a fresh re-derivation")
    args = ap.parse_args()
    if args.check:
        return check()
    generate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
