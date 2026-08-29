#!/usr/bin/env python3
"""Hermetic self-consistency guard for the committed how-well held-out
fixture (unit 63, CONTRACT H3, `cookbook/fixtures/finetune_heldout/`) --
audit finding 3 advisory (e).

Everything `cookbook/book/scripts/derive_heldout_fixture.py::check()` proves
is network-backed (it re-downloads the pinned ogbn-arxiv sources and
re-derives the whole 1500-pair mining run from scratch). This guard proves a
DIFFERENT, strictly weaker property that needs none of that: the fixture
FILES ALREADY COMMITTED here agree with EACH OTHER. No network, no re-mining
-- every check below is a pure function of checkout content, so it can run
on every PR (unlike `--check`, which nobody wants gating merge on a
best-effort external download).

Checks:

1. **Id-order equality** -- `heldout_pairs.jsonl`'s (anchor_id, positive_id,
   negative_id) triples, in line order, must equal `heldout_ids.txt`'s
   tab-separated triples, in line order, EXACTLY (same count, same ids, same
   order -- not just the same set). `heldout_ids_sha256` hashes
   `heldout_ids.txt`; a consumer that reads `heldout_pairs.jsonl` for text
   and trusts its ids to line up with that hash needs this equality to
   actually hold.
2. **`heldout_ids_sha256` recomputation** -- `manifest.json`'s
   `heldout_ids_sha256` must equal a fresh SHA-256 over `heldout_ids.txt`'s
   own committed bytes (the exact definition
   `derive_heldout_fixture.py::_file_sha256`/`generate()` uses).
3. **`dataset_sha256` recomputation** -- `manifest.json`'s `dataset_sha256`
   must equal a fresh SHA-256 over `"\n".join(per_pair_hashes)` where
   `per_pair_hashes` is the TRAIN pairs' already-committed `pair_sha256`
   values (`train_ids_sha256.json`, in committed row order) followed by the
   HELDOUT pairs' freshly-recomputed `_pair_sha256` values
   (`heldout_pairs.jsonl`, in committed line order) -- the exact
   `derive()`/`_write_manifest()` Merkle-style definition, entirely
   reconstructable from committed content (the train side reuses its
   already-committed hash rather than needing the train TEXT, which this
   fixture deliberately never commits).
4. **`NOTICE` exists** and is non-empty (the ODC-BY 1.0 attribution
   discharge the fixture's own README names as living in this directory).
5. **`arxiv_subset_ids.txt` pointer** -- IF `manifest.json` records a
   SHA-256 for the vendored subset ids file anywhere (a
   `arxiv_subset_ids_sha256` key at the top level or nested under
   `provenance`), it must match a fresh SHA-256 of the committed
   `arxiv_subset_ids.txt` bytes. The committed `manifest.json` records this
   key under `provenance.arxiv_subset_ids_sha256` (unit-63 audit advisory
   (c) -- previously it recorded only the file's PATH, as a provenance
   citation, with no content check at all), so this check is now ACTIVE on
   the real fixture, not merely a no-op standing by for a future manifest --
   a producer that vendors a stale/tampered subset-ids file now fails this
   gate loudly rather than silently.

Run: `python3 ci/scripts/perf/check_heldout_fixture_integrity.py`
Self-test: `python3 ci/scripts/perf/check_heldout_fixture_integrity.py --self-test`
Hermetic: reads only files under `cookbook/fixtures/finetune_heldout/`
(or a throwaway self-test fixture directory) -- no network, no build, no GPU.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / "cookbook" / "fixtures" / "finetune_heldout"

# The exact per-pair hash definition
# cookbook/book/scripts/derive_heldout_fixture.py::_pair_sha256 uses --
# duplicated here (not imported) because that script lives on the BOOK side
# (one-way rule, ci/scripts/check_cookbook_one_way.sh) and is network-backed
# besides; this guard's whole point is running WITHOUT either dependency.
# ci/scripts/perf/verify_train_pairs.py duplicates the SAME definition for
# the SAME reason -- both cite this docstring as the shared source of truth.


def _pair_sha256(pair: dict) -> str:
    payload = "\x00".join([
        pair["anchor_id"], pair["anchor_text"],
        pair["positive_id"], pair["positive_text"],
        pair["negative_id"], pair["negative_text"],
    ]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_heldout_ids(path: Path) -> list[tuple[str, str, str]]:
    ids: list[tuple[str, str, str]] = []
    for line in path.read_text().splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"{path}: line {line!r} does not split into exactly 3 tab-separated fields")
        ids.append((parts[0], parts[1], parts[2]))
    return ids


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _find_arxiv_subset_sha256(manifest: dict) -> str | None:
    """Look for a recorded SHA-256 pointer for the vendored subset ids file
    -- top-level `arxiv_subset_ids_sha256`, or nested under `provenance`.
    Returns None if no such key exists anywhere (check 5 becomes a no-op)."""
    if "arxiv_subset_ids_sha256" in manifest:
        return manifest["arxiv_subset_ids_sha256"]
    provenance = manifest.get("provenance", {})
    if isinstance(provenance, dict) and "arxiv_subset_ids_sha256" in provenance:
        return provenance["arxiv_subset_ids_sha256"]
    return None


def run_gate(fixture_dir: Path) -> list[str]:
    findings: list[str] = []

    heldout_ids_path = fixture_dir / "heldout_ids.txt"
    heldout_pairs_path = fixture_dir / "heldout_pairs.jsonl"
    train_hashes_path = fixture_dir / "train_ids_sha256.json"
    manifest_path = fixture_dir / "manifest.json"
    notice_path = fixture_dir / "NOTICE"
    subset_ids_path = fixture_dir / "arxiv_subset_ids.txt"

    required = [heldout_ids_path, heldout_pairs_path, train_hashes_path, manifest_path, notice_path]
    missing = [p for p in required if not p.exists()]
    if missing:
        return [f"missing required committed fixture file: {p}" for p in missing]

    manifest = json.loads(manifest_path.read_text())

    # --- (1) id-order equality: heldout_pairs.jsonl vs heldout_ids.txt ---
    ids_from_txt = _load_heldout_ids(heldout_ids_path)
    heldout_pairs = _load_jsonl(heldout_pairs_path)
    ids_from_jsonl = [(p["anchor_id"], p["positive_id"], p["negative_id"]) for p in heldout_pairs]
    if ids_from_txt != ids_from_jsonl:
        if len(ids_from_txt) != len(ids_from_jsonl):
            findings.append(
                f"heldout_ids.txt has {len(ids_from_txt)} rows but heldout_pairs.jsonl has "
                f"{len(ids_from_jsonl)} -- must be the exact same count")
        else:
            first_diff = next(
                i for i, (a, b) in enumerate(zip(ids_from_txt, ids_from_jsonl)) if a != b)
            findings.append(
                f"heldout_ids.txt and heldout_pairs.jsonl diverge at line {first_diff + 1}: "
                f"{ids_from_txt[first_diff]} != {ids_from_jsonl[first_diff]} -- ids must match "
                "exactly, in order")

    # --- (2) heldout_ids_sha256 recomputation ---
    fresh_heldout_ids_sha256 = _file_sha256(heldout_ids_path)
    committed_heldout_ids_sha256 = manifest.get("heldout_ids_sha256")
    if committed_heldout_ids_sha256 != fresh_heldout_ids_sha256:
        findings.append(
            f"manifest.json heldout_ids_sha256={committed_heldout_ids_sha256!r} does not match "
            f"a fresh SHA-256 of heldout_ids.txt ({fresh_heldout_ids_sha256})")

    # --- (3) dataset_sha256 recomputation (Merkle-style, train hashes reused
    # as-committed, heldout hashes freshly recomputed from committed text) ---
    train_rows = json.loads(train_hashes_path.read_text())
    train_hashes = [row["pair_sha256"] for row in train_rows]
    heldout_hashes = [_pair_sha256(p) for p in heldout_pairs]
    fresh_dataset_sha256 = hashlib.sha256(
        "\n".join(train_hashes + heldout_hashes).encode("utf-8")).hexdigest()
    committed_dataset_sha256 = manifest.get("dataset_sha256")
    if committed_dataset_sha256 != fresh_dataset_sha256:
        findings.append(
            f"manifest.json dataset_sha256={committed_dataset_sha256!r} does not match a fresh "
            f"recomputation over train_ids_sha256.json's pair_sha256 values + heldout_pairs.jsonl's "
            f"freshly-hashed pairs ({fresh_dataset_sha256})")

    # --- (4) NOTICE exists and is non-empty ---
    if notice_path.stat().st_size == 0:
        findings.append(f"{notice_path} exists but is empty -- ODC-BY 1.0 attribution must be discharged here")

    # --- (5) arxiv_subset_ids.txt pointer, IF manifest records one ---
    recorded_subset_sha = _find_arxiv_subset_sha256(manifest)
    if recorded_subset_sha is not None:
        if not subset_ids_path.exists():
            findings.append(
                f"manifest.json records arxiv_subset_ids_sha256={recorded_subset_sha!r} but "
                f"{subset_ids_path} does not exist")
        else:
            fresh_subset_sha = _file_sha256(subset_ids_path)
            if fresh_subset_sha != recorded_subset_sha:
                findings.append(
                    f"manifest.json's recorded arxiv_subset_ids_sha256={recorded_subset_sha!r} does "
                    f"not match a fresh SHA-256 of arxiv_subset_ids.txt ({fresh_subset_sha})")

    return findings


def _write_fixture(
    dir_: Path,
    *,
    heldout_pairs: list[dict],
    train_rows: list[dict],
    manifest_overrides: dict | None = None,
    notice_text: str = "ODC-BY 1.0 attribution.\n",
    write_notice: bool = True,
    write_subset_ids: bool = False,
    subset_ids_text: str = "1\n2\n3\n",
) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    ids_lines = [f"{p['anchor_id']}\t{p['positive_id']}\t{p['negative_id']}" for p in heldout_pairs]
    (dir_ / "heldout_ids.txt").write_text("\n".join(ids_lines) + "\n")
    with (dir_ / "heldout_pairs.jsonl").open("w") as f:
        for p in heldout_pairs:
            f.write(json.dumps(p, sort_keys=True) + "\n")
    (dir_ / "train_ids_sha256.json").write_text(json.dumps(train_rows, indent=2))
    if write_notice:
        (dir_ / "NOTICE").write_text(notice_text)
    if write_subset_ids:
        (dir_ / "arxiv_subset_ids.txt").write_text(subset_ids_text)

    fresh_heldout_ids_sha256 = _file_sha256(dir_ / "heldout_ids.txt")
    train_hashes = [row["pair_sha256"] for row in train_rows]
    heldout_hashes = [_pair_sha256(p) for p in heldout_pairs]
    fresh_dataset_sha256 = hashlib.sha256(
        "\n".join(train_hashes + heldout_hashes).encode("utf-8")).hexdigest()
    manifest = {
        "heldout_ids_sha256": fresh_heldout_ids_sha256,
        "dataset_sha256": fresh_dataset_sha256,
    }
    if manifest_overrides:
        manifest.update(manifest_overrides)
    (dir_ / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return dir_


_SYNTH_HELDOUT = [
    {"anchor_id": "h1", "anchor_text": "Held anchor one.",
     "positive_id": "hp1", "positive_text": "Held positive one.",
     "negative_id": "hn1", "negative_text": "Held negative one."},
    {"anchor_id": "h2", "anchor_text": "Held anchor two.",
     "positive_id": "hp2", "positive_text": "Held positive two.",
     "negative_id": "hn2", "negative_text": "Held negative two."},
]
_SYNTH_TRAIN_ROWS = [
    {"anchor_id": "t1", "positive_id": "tp1", "negative_id": "tn1", "pair_sha256": "a" * 64},
]


def self_test() -> int:
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)

        # GREEN: a fully self-consistent synthetic fixture.
        green = _write_fixture(tmp / "green", heldout_pairs=_SYNTH_HELDOUT, train_rows=_SYNTH_TRAIN_ROWS)
        findings = run_gate(green)
        if findings:
            failures.append(f"GREEN case unexpectedly RED: {findings}")

        # RED (1): heldout_ids.txt id order/content diverges from heldout_pairs.jsonl.
        red1 = _write_fixture(tmp / "red_id_order", heldout_pairs=_SYNTH_HELDOUT, train_rows=_SYNTH_TRAIN_ROWS)
        (red1 / "heldout_ids.txt").write_text("h2\thp2\thn2\nh1\thp1\thn1\n")  # swapped order
        # heldout_ids_sha256 in manifest now stale too, but we want to isolate
        # the id-order finding -- recompute it to match the (wrong-order) file
        # so only check (1) fires, not check (2) as well.
        manifest = json.loads((red1 / "manifest.json").read_text())
        manifest["heldout_ids_sha256"] = _file_sha256(red1 / "heldout_ids.txt")
        (red1 / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        findings1 = run_gate(red1)
        if not any("diverge at line" in f for f in findings1):
            failures.append(f"RED(1) id-order case expected a 'diverge at line' finding, got {findings1}")

        # RED (2): manifest.json's heldout_ids_sha256 does not match a fresh hash.
        red2 = _write_fixture(
            tmp / "red_heldout_sha", heldout_pairs=_SYNTH_HELDOUT, train_rows=_SYNTH_TRAIN_ROWS,
            manifest_overrides={"heldout_ids_sha256": "0" * 64})
        findings2 = run_gate(red2)
        if not any("heldout_ids_sha256" in f and "does not match" in f for f in findings2):
            failures.append(f"RED(2) heldout_ids_sha256 case expected a mismatch finding, got {findings2}")

        # RED (3): manifest.json's dataset_sha256 does not match a fresh recomputation.
        red3 = _write_fixture(
            tmp / "red_dataset_sha", heldout_pairs=_SYNTH_HELDOUT, train_rows=_SYNTH_TRAIN_ROWS,
            manifest_overrides={"dataset_sha256": "0" * 64})
        findings3 = run_gate(red3)
        if not any("dataset_sha256" in f and "does not match" in f for f in findings3):
            failures.append(f"RED(3) dataset_sha256 case expected a mismatch finding, got {findings3}")

        # RED (4): NOTICE missing entirely.
        red4 = _write_fixture(
            tmp / "red_no_notice", heldout_pairs=_SYNTH_HELDOUT, train_rows=_SYNTH_TRAIN_ROWS,
            write_notice=False)
        findings4 = run_gate(red4)
        if not any("missing required committed fixture file" in f and "NOTICE" in f for f in findings4):
            failures.append(f"RED(4) missing-NOTICE case expected a missing-file finding, got {findings4}")

        # RED (4b): NOTICE present but empty.
        red4b = _write_fixture(
            tmp / "red_empty_notice", heldout_pairs=_SYNTH_HELDOUT, train_rows=_SYNTH_TRAIN_ROWS,
            notice_text="")
        findings4b = run_gate(red4b)
        if not any("empty" in f for f in findings4b):
            failures.append(f"RED(4b) empty-NOTICE case expected an 'empty' finding, got {findings4b}")

        # RED (5): manifest records an arxiv_subset_ids_sha256 pointer that
        # does not match the committed file.
        red5 = _write_fixture(
            tmp / "red_subset_sha", heldout_pairs=_SYNTH_HELDOUT, train_rows=_SYNTH_TRAIN_ROWS,
            write_subset_ids=True, manifest_overrides={"arxiv_subset_ids_sha256": "0" * 64})
        findings5 = run_gate(red5)
        if not any("arxiv_subset_ids_sha256" in f and "does not match" in f for f in findings5):
            failures.append(f"RED(5) subset-sha case expected a mismatch finding, got {findings5}")

        # GREEN control (5a): manifest records NO arxiv_subset_ids_sha256 key
        # at all (a legacy manifest shape, predating unit-63 audit advisory
        # (c)) -- must be a no-op, never a finding.
        green5 = _write_fixture(
            tmp / "green_no_subset_pointer", heldout_pairs=_SYNTH_HELDOUT, train_rows=_SYNTH_TRAIN_ROWS)
        findings_green5 = run_gate(green5)
        if findings_green5:
            failures.append(f"GREEN(5a) no-pointer-recorded case unexpectedly RED: {findings_green5}")

        # GREEN control (5b): manifest records a MATCHING arxiv_subset_ids_sha256
        # -- today's real committed manifest.json's own shape (unit-63 audit
        # advisory (c)) -- must clear the check, never a false finding.
        green5b = _write_fixture(
            tmp / "green_matching_subset_pointer", heldout_pairs=_SYNTH_HELDOUT, train_rows=_SYNTH_TRAIN_ROWS,
            write_subset_ids=True)
        manifest_5b = json.loads((green5b / "manifest.json").read_text())
        manifest_5b["arxiv_subset_ids_sha256"] = _file_sha256(green5b / "arxiv_subset_ids.txt")
        (green5b / "manifest.json").write_text(json.dumps(manifest_5b, indent=2, sort_keys=True))
        findings_green5b = run_gate(green5b)
        if findings_green5b:
            failures.append(f"GREEN(5b) matching-pointer case unexpectedly RED: {findings_green5b}")

    # End-to-end: the REAL committed fixture must be clean today.
    real_findings = run_gate(FIXTURE_DIR)
    if real_findings:
        failures.append(f"self-test FAILED: real committed fixture is not clean: {real_findings}")

    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        print("check-heldout-fixture-integrity self-test: FAIL", file=sys.stderr)
        return 1
    print(
        "check-heldout-fixture-integrity self-test: OK -- id-order equality, "
        "heldout_ids_sha256 recomputation, dataset_sha256 recomputation, NOTICE presence, and "
        "the arxiv_subset_ids_sha256 pointer (now recorded under manifest.json's own "
        "provenance, unit-63 audit advisory (c)) all bite on throwaway fixtures; the real "
        "committed fixture is clean."
    )
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()

    findings = run_gate(FIXTURE_DIR)
    if findings:
        print("check-heldout-fixture-integrity: FAIL", file=sys.stderr)
        for msg in findings:
            print(f"  - {msg}", file=sys.stderr)
        print(f"\ncheck-heldout-fixture-integrity: {len(findings)} finding(s).", file=sys.stderr)
        return 1
    print(
        "check-heldout-fixture-integrity: PASS -- cookbook/fixtures/finetune_heldout/ is "
        "self-consistent (heldout_pairs.jsonl ids == heldout_ids.txt exactly and in order, "
        "manifest.json's heldout_ids_sha256/dataset_sha256 both recompute clean, NOTICE exists)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
