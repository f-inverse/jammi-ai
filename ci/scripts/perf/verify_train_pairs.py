#!/usr/bin/env python3
"""Byte-verify a re-derived ``train_pairs.jsonl`` against the committed
``cookbook/fixtures/finetune_heldout/train_ids_sha256.json`` (unit 63,
CONTRACT amendment 2026-08-28b PRE-RUN provisioning step).

``train_pairs.jsonl`` is never committed (repo-size discipline — see
``cookbook/fixtures/finetune_heldout/README.md`` "Why train text isn't
committed"); the only committed identity for the 1372 TRAIN-side pairs is a
per-pair SHA-256 in ``train_ids_sha256.json``. Whenever a producer
re-derives (or an operator hand-places) a ``train_pairs.jsonl`` before a real
``ci/scripts/perf/finetune_run_ab.sh`` leg, this module is the SINGLE
reviewable unit that decides whether that file is trustworthy: same pair
identity set (no missing id, no extra id, no duplicate), same per-pair
content (each pair's SHA-256, computed the exact same way
``cookbook/book/scripts/derive_heldout_fixture.py::_pair_sha256`` computed it
when the committed hashes were written), and the exact committed count
(1372) on both sides.

This is deliberately its own file (not inlined into
``finetune_run_ab.sh`` or ``derive_heldout_fixture.py``): a producer's
pre-run provisioning step and a fixture-derivation script are both
callers of this ONE verification, never two independent re-implementations
of the same byte-check that could silently drift apart.

Run: ``python3 ci/scripts/perf/verify_train_pairs.py``
     (defaults to the committed fixture paths under
     ``cookbook/fixtures/finetune_heldout/``)
Self-test: ``python3 ci/scripts/perf/verify_train_pairs.py --self-test``
     (a synthetic 2-pair fixture, GREEN on committed-shaped hashes, RED on
     a flipped byte / a missing id / an extra pair / a wrong count)
Hermetic: reads only the two files named on the command line (or the
committed fixture defaults); no network, no build, no GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / "cookbook" / "fixtures" / "finetune_heldout"
DEFAULT_PAIRS = FIXTURE_DIR / "train_pairs.jsonl"
DEFAULT_HASHES = FIXTURE_DIR / "train_ids_sha256.json"

# The committed train-side pair count (CONTRACT H3: N_PAIRS=1500,
# N_HELDOUT=128, 1500-128=1372 -- cookbook/book/scripts/
# derive_heldout_fixture.py's own N_PAIRS/N_HELDOUT constants). A self-test
# fixture overrides this via --expected-count to exercise the same logic on
# a tiny synthetic pair set without needing 1372 real rows.
EXPECTED_TRAIN_COUNT = 1372


def _pair_sha256(pair: dict) -> str:
    """MUST match cookbook/book/scripts/derive_heldout_fixture.py::_pair_sha256
    exactly -- this is the one hash definition both the producer (writing
    train_ids_sha256.json) and this verifier (checking a re-derived
    train_pairs.jsonl against it) share."""
    payload = "\x00".join([
        pair["anchor_id"], pair["anchor_text"],
        pair["positive_id"], pair["positive_text"],
        pair["negative_id"], pair["negative_text"],
    ]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _key(row: dict) -> tuple[str, str, str]:
    return (row["anchor_id"], row["positive_id"], row["negative_id"])


def _load_hashes(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _load_pairs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify(pairs_path: Path, hashes_path: Path, expected_count: int = EXPECTED_TRAIN_COUNT) -> list[str]:
    """Returns a list of findings (empty == verified). The FIRST finding is
    always the most actionable single fact a caller should surface loudly
    (the first mismatching pair id, in committed order, when one exists) --
    findings that describe the shape of the whole file (extras/duplicates/
    count) are appended after every per-pair finding, never before, so a
    single flipped byte in pair #1 is never buried under an aggregate count
    line.
    """
    if not hashes_path.exists():
        return [f"{hashes_path} does not exist -- cannot verify without the committed hashes"]
    expected_rows = _load_hashes(hashes_path)

    if not pairs_path.exists():
        return [f"{pairs_path} does not exist -- provisioning did not produce train pairs"]
    actual_pairs = _load_pairs(pairs_path)

    actual_by_key: dict[tuple, dict] = {}
    dup_actual: list[tuple] = []
    for pair in actual_pairs:
        k = _key(pair)
        if k in actual_by_key:
            dup_actual.append(k)
        actual_by_key[k] = pair

    expected_keys_ordered = [_key(r) for r in expected_rows]
    expected_key_set = set(expected_keys_ordered)
    dup_expected = [k for k in expected_key_set
                     if expected_keys_ordered.count(k) > 1]

    per_pair: list[str] = []
    for row in expected_rows:
        k = _key(row)
        actual = actual_by_key.get(k)
        if actual is None:
            per_pair.append(
                f"missing pair id {k}: present in {hashes_path.name}, absent from "
                f"{pairs_path.name}")
            continue
        actual_sha = _pair_sha256(actual)
        if actual_sha != row["pair_sha256"]:
            per_pair.append(
                f"pair id {k} sha256 mismatch: expected {row['pair_sha256']}, "
                f"got {actual_sha}")

    structural: list[str] = []
    extra_keys = [k for k in actual_by_key if k not in expected_key_set]
    if extra_keys:
        structural.append(
            f"{pairs_path.name} contains pair id(s) not present in {hashes_path.name}: "
            f"{extra_keys}")
    if dup_actual:
        structural.append(f"{pairs_path.name} contains duplicate pair id(s): {dup_actual}")
    if dup_expected:
        structural.append(f"{hashes_path.name} contains duplicate pair id(s): {dup_expected}")
    if len(expected_rows) != expected_count:
        structural.append(
            f"{hashes_path.name} has {len(expected_rows)} rows, expected exactly "
            f"{expected_count}")
    if len(actual_pairs) != expected_count:
        structural.append(
            f"{pairs_path.name} has {len(actual_pairs)} pairs, expected exactly "
            f"{expected_count}")

    return per_pair + structural


def _fixture(tmp: Path, pairs: list[dict], name: str = "train_pairs.jsonl") -> tuple[Path, Path]:
    hashes = [
        {"anchor_id": p["anchor_id"], "positive_id": p["positive_id"],
         "negative_id": p["negative_id"], "pair_sha256": _pair_sha256(p)}
        for p in pairs
    ]
    hashes_path = tmp / "train_ids_sha256.json"
    hashes_path.write_text(json.dumps(hashes, indent=2))
    pairs_path = tmp / name
    with pairs_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, sort_keys=True) + "\n")
    return pairs_path, hashes_path


_SYNTH_PAIRS = [
    {"anchor_id": "a1", "anchor_text": "Anchor one text.",
     "positive_id": "p1", "positive_text": "Positive one text.",
     "negative_id": "n1", "negative_text": "Negative one text."},
    {"anchor_id": "a2", "anchor_text": "Anchor two text.",
     "positive_id": "p2", "positive_text": "Positive two text.",
     "negative_id": "n2", "negative_text": "Negative two text."},
]


def self_test() -> int:
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)

        # GREEN: synthetic 2-pair fixture, committed-shaped hashes.
        green_dir = tmp / "green"
        green_dir.mkdir()
        pairs_path, hashes_path = _fixture(green_dir, _SYNTH_PAIRS)
        findings = verify(pairs_path, hashes_path, expected_count=2)
        if findings:
            failures.append(f"GREEN case unexpectedly RED: {findings}")

        # (a) RED: one flipped byte in a pair's text (pair a1's anchor text).
        a_dir = tmp / "a_flipped_byte"
        a_dir.mkdir()
        flipped = [dict(_SYNTH_PAIRS[0]), dict(_SYNTH_PAIRS[1])]
        flipped[0]["anchor_text"] = "Bnchor one text."  # A -> B, one byte
        _, hashes_path_a = _fixture(a_dir, _SYNTH_PAIRS)  # hashes from the ORIGINAL text
        pairs_path_a = a_dir / "train_pairs.jsonl"
        with pairs_path_a.open("w") as f:
            for p in flipped:
                f.write(json.dumps(p, sort_keys=True) + "\n")
        findings_a = verify(pairs_path_a, hashes_path_a, expected_count=2)
        if not findings_a or "sha256 mismatch" not in findings_a[0]:
            failures.append(
                f"(a) flipped-byte case expected a leading sha256-mismatch finding, got {findings_a}")

        # (b) RED: a missing id (train_pairs.jsonl drops pair a2 entirely,
        # count still checked separately below via (d)).
        b_dir = tmp / "b_missing_id"
        b_dir.mkdir()
        _, hashes_path_b = _fixture(b_dir, _SYNTH_PAIRS)
        pairs_path_b = b_dir / "train_pairs.jsonl"
        with pairs_path_b.open("w") as f:
            f.write(json.dumps(_SYNTH_PAIRS[0], sort_keys=True) + "\n")
        findings_b = verify(pairs_path_b, hashes_path_b, expected_count=2)
        if not any("missing pair id" in x for x in findings_b):
            failures.append(f"(b) missing-id case expected a 'missing pair id' finding, got {findings_b}")

        # (c) RED: an extra pair (train_pairs.jsonl has both committed pairs
        # PLUS a third id never present in train_ids_sha256.json).
        c_dir = tmp / "c_extra_pair"
        c_dir.mkdir()
        _, hashes_path_c = _fixture(c_dir, _SYNTH_PAIRS)
        extra_pair = {"anchor_id": "a3", "anchor_text": "Anchor three text.",
                      "positive_id": "p3", "positive_text": "Positive three text.",
                      "negative_id": "n3", "negative_text": "Negative three text."}
        pairs_path_c = c_dir / "train_pairs.jsonl"
        with pairs_path_c.open("w") as f:
            for p in _SYNTH_PAIRS + [extra_pair]:
                f.write(json.dumps(p, sort_keys=True) + "\n")
        findings_c = verify(pairs_path_c, hashes_path_c, expected_count=2)
        if not any("not present in" in x for x in findings_c):
            failures.append(f"(c) extra-pair case expected a 'not present in' finding, got {findings_c}")

        # (d) RED: wrong count (hashes file legitimately has 2 rows, but the
        # caller asserts the committed constant 1372 -- a truncated/corrupted
        # committed hashes file must fail even if internally self-consistent).
        d_dir = tmp / "d_wrong_count"
        d_dir.mkdir()
        pairs_path_d, hashes_path_d = _fixture(d_dir, _SYNTH_PAIRS)
        findings_d = verify(pairs_path_d, hashes_path_d, expected_count=1372)
        if not any("expected exactly 1372" in x for x in findings_d):
            failures.append(f"(d) wrong-count case expected an 'expected exactly 1372' finding, got {findings_d}")

        # GREEN control: missing pairs_path entirely is a clean, single finding.
        missing_dir = tmp / "missing"
        missing_dir.mkdir()
        _, hashes_path_missing = _fixture(missing_dir, _SYNTH_PAIRS)
        findings_missing = verify(missing_dir / "nope.jsonl", hashes_path_missing, expected_count=2)
        if len(findings_missing) != 1 or "does not exist" not in findings_missing[0]:
            failures.append(f"missing-file case expected exactly one 'does not exist' finding, got {findings_missing}")

    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        print("verify-train-pairs self-test: FAIL", file=sys.stderr)
        return 1
    print(
        "verify-train-pairs self-test: OK -- GREEN on a synthetic 2-pair fixture matching "
        "committed-shaped hashes; RED on (a) one flipped byte, (b) a missing id, (c) an "
        "extra pair, and (d) a wrong count."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" in argv:
        return self_test()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS,
                     help=f"path to the re-derived train_pairs.jsonl (default: {DEFAULT_PAIRS})")
    ap.add_argument("--hashes", type=Path, default=DEFAULT_HASHES,
                     help=f"path to the committed train_ids_sha256.json (default: {DEFAULT_HASHES})")
    ap.add_argument("--expected-count", type=int, default=EXPECTED_TRAIN_COUNT,
                     help=f"exact expected pair count on both sides (default: {EXPECTED_TRAIN_COUNT})")
    ap.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    findings = verify(args.pairs, args.hashes, args.expected_count)
    if findings:
        print(
            f"::error::verify-train-pairs: REFUSING -- first mismatch: {findings[0]}",
            file=sys.stderr,
        )
        for f in findings[1:]:
            print(f"  - {f}", file=sys.stderr)
        print(f"\nverify-train-pairs: {len(findings)} finding(s).", file=sys.stderr)
        return 1

    print(
        f"verify-train-pairs: OK -- {args.expected_count} train pairs in {args.pairs} byte-"
        f"verified against {args.hashes} (no missing/extra/duplicate ids, all per-pair "
        "sha256 match)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
