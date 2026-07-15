#!/usr/bin/env python3
"""Emit the segmented-ANN-index cache — CPU, embedded-only, hermetic.

The engine↔cookbook validator for the segmented-ANN capability
(`crates/jammi-db/src/index/segment.rs` + `ResultStore::append_segment`,
`crates/jammi-db/src/store/mod.rs`): a table's
ANN index is a *set* of immutable segments rather than one sidecar. Appending a
batch of new rows writes a new segment beside the existing ones and leaves them
byte-for-byte untouched; a search fans across every segment and merges under one
total order that is byte-identical to the single-index path at `N = 1`.

This cache measures TWO properties, both live against real engine artifacts —
never a Python reimplementation of the merge:

* **`N = 1` byte-identity (the SDK-reachable half).** A single
  `generate_embeddings` pass over the engine's public `patents` fixture (embedded
  with the deterministic `tiny_modernbert` encoder) writes exactly one segment.
  This script reads that fact back off the embedded engine's own catalog
  (`<artifact_dir>/catalog.db`, a real SQLite file the running wheel wrote — not a
  reimplementation of the schema) and confirms `segment_id = 0` with the
  documented `{table}__seg0.*` sidecar naming. It then runs `db.search(...)`
  LIVE for a spread of `(query row, k)` pairs and records the hits; the chapter
  independently recomputes the exact brute-force cosine ranking over the SAME
  vectors (read back from the table's own Parquet) and asserts it against the
  recorded hits — "a freshly-built single-segment table's search is the exact
  top-k" is what "byte-identical to the pre-existing single-index behavior"
  cashes out to (the single-index path is also single-stage exact cosine for an
  `F32` table).

* **Incremental append leaves segment 0 untouched (the write-side primitive this
  release ships).** There is no public verb yet that appends a SECOND segment
  onto an ALREADY-READY table: every embedding-producing pipeline
  (`generate_embeddings`, `import_embeddings`, graph propagation, context-set
  materialization) calls `ResultStore::create_table` fresh, so calling
  `generate_embeddings` again always lands a brand-new table, never a second
  segment on this one (grep-verified against
  `crates/jammi-ai/src/pipeline/embedding.rs`, `pipeline/import.rs`, and
  `crates/jammi-db/src/store/mod.rs`). `ResultStore::append_segment` is a
  `jammi-db` crate-level Rust primitive with no SDK/wire binding today — a
  genuine, honest engine-surface finding (the "teach the honest negative"
  discipline this book applies everywhere), not a gap in this script.

  So this half is measured at the layer where it is genuinely live and public
  today: this script shells out to the ALREADY-SHIPPED, unmodified
  `crates/jammi-db` integration-test binary (`cargo test -p jammi-db --test it
  segment::`), the same compiled artifact CI runs, and records its live verdict
  (exit code + per-test pass/fail) — the same boolean-verdict-from-a-live-run
  pattern `lifecycle.qmd`'s `every_catalog_model_is_referenced` already uses. It
  is never a fabricated number, and never a Python reimplementation of
  `SegmentedIndex`'s merge.

Names no consumer: a generic 20-row `patents` fixture and the engine's own
public `tiny_modernbert` encoder, both already used by the engine's own
`storage_precision` integration tests.

Usage::

    python scripts/build_segmented_ann_cache.py --fixtures-root /path/to/jammi-ai

`--fixtures-root` is the engine checkout carrying `tests/fixtures/patents.parquet`
and `tests/fixtures/tiny_modernbert` (defaults to `JAMMI_FIXTURES_ROOT`) — it is
also the checkout the `cargo test` half runs against, so it must be a working
tree with `cargo` on `PATH`. This is an emit-only script: it imports a build of
the engine wheel + client and shells out to `cargo test`; PR CI never runs it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import tempfile
from pathlib import Path

import jammi
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import jammi_cookbook  # noqa: F401  # applies the determinism env on import

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts" / "segmented_ann"

# The `cargo test` target the append/no-rebuild half runs — every test under the
# `segment::` module of the `jammi-db` it-suite. `append_does_not_rebuild_prior_segments`
# is the headline proof (segment 0's raw `.usearch` bytes are read before/after
# the append and asserted byte-identical, and both segments' rows are searchable
# through the merge); the other three cover the catalog migration, the
# quantized two-segment merge, and concurrent-append id allocation.
_CARGO_TEST_FILTER = "segment::"

# k values swept per query row for the N=1 exact-search property.
_KS = (1, 3, 5)
# A spread of query row ids (not every row — 6 of the fixture's 20 rows is
# already redundant coverage for an exact-match property; more rows would not
# change the verdict, only the runtime).
_QUERY_IDS = ("1", "4", "7", "10", "14", "19")


def _cosine_brute_force(
    vectors: dict[str, np.ndarray], query_id: str, k: int
) -> list[str]:
    """The independent oracle: exact brute-force cosine top-`k` over `vectors`,
    ordered `(distance ASC, row_id ASC)` — the same tie-break `SegmentedIndex`
    uses, so a tie-free corpus (this one; the fixture rows are all distinct real
    sentences) makes the comparison exact, not merely approximate."""
    q = vectors[query_id]
    q = q / np.linalg.norm(q)
    scored = []
    for row_id, v in vectors.items():
        vn = v / np.linalg.norm(v)
        distance = 1.0 - float(np.dot(q, vn))
        scored.append((distance, row_id))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [row_id for _, row_id in scored[:k]]


def _run_n1_property(db, fixtures_root: Path) -> dict:
    """Build a single embedding table over `patents` and measure the N=1
    exact-search property live. Returns the recorded matrix (never asserted
    here — the chapter reads this back and asserts it against the frozen
    golden)."""
    patents_url = f"file://{fixtures_root / 'tests' / 'fixtures' / 'patents.parquet'}"
    tiny_modernbert = f"local:{fixtures_root / 'tests' / 'fixtures' / 'tiny_modernbert'}"

    db.add_source("patents", url=patents_url, format="parquet")
    table_name = db.generate_embeddings(
        source="patents",
        model=tiny_modernbert,
        columns=["abstract"],
        key="id",
        modality="text",
    )

    # Ground truth: read the table's own Parquet vectors back — independent of
    # the ANN sidecar entirely.
    ground_truth = db.sql(f'SELECT _row_id, vector FROM "jammi.{table_name}"').to_pylist()
    vectors = {row["_row_id"]: np.asarray(row["vector"], dtype=np.float32) for row in ground_truth}

    # Live search, one call per (query, k) — the SDK-reachable half.
    live_hits: dict[str, dict[str, list[str]]] = {}
    for query_id in _QUERY_IDS:
        query_vec = (vectors[query_id] / np.linalg.norm(vectors[query_id])).tolist()
        live_hits[query_id] = {}
        for k in _KS:
            hits = db.search("patents", query=query_vec, k=k)
            live_hits[query_id][str(k)] = hits.column("_row_id").to_pylist()

    # Independent recomputation, live, right now — never transcribed.
    n_trials = 0
    n_matches = 0
    mismatches = []
    for query_id in _QUERY_IDS:
        for k in _KS:
            expected = _cosine_brute_force(vectors, query_id, k)
            got = live_hits[query_id][str(k)]
            n_trials += 1
            if got == expected:
                n_matches += 1
            else:
                mismatches.append({"query": query_id, "k": k, "expected": expected, "got": got})

    return {
        "table_name": table_name,
        "vectors": vectors,
        "live_hits": live_hits,
        "n1_search_trials": n_trials,
        "n1_search_matches": n_matches,
        "n1_search_mismatches": mismatches,
    }


def _read_segment_catalog(artifact_dir: Path, table_name: str) -> dict:
    """Read `index_segments` for `table_name` straight off the embedded engine's
    own SQLite catalog file — a real artifact the running wheel wrote, read
    read-only, never reimplemented or guessed."""
    con = sqlite3.connect(str(artifact_dir / "catalog.db"))
    try:
        rows = con.execute(
            "SELECT segment_id, index_path, row_count FROM index_segments "
            "WHERE table_name = ? ORDER BY segment_id",
            (table_name,),
        ).fetchall()
    finally:
        con.close()
    segments = [{"segment_id": r[0], "index_path": r[1], "row_count": r[2]} for r in rows]
    return {"segments": segments, "segment_count": len(segments)}


def _seg0_sidecar_files_present(index_path: str) -> dict:
    """Whether the three sidecar files an F32 (no-rescore) segment carries exist
    on disk, keyed off the URL the catalog itself recorded (a `file://` path in
    this embedded-only emit)."""
    base = Path(index_path.removeprefix("file://"))
    return {
        "usearch": base.with_suffix(".usearch").exists(),
        "rowmap": base.with_suffix(".rowmap").exists(),
        "manifest_json": base.with_suffix(".manifest.json").exists(),
    }


def _run_append_suite(fixtures_root: Path) -> dict:
    """Shell out to the already-shipped `jammi-db` it-suite's `segment::` tests
    — the live oracle for the append/no-rebuild property, since no SDK verb
    exercises `ResultStore::append_segment` yet. Never modifies the test; just
    runs the exact compiled artifact CI runs."""
    cmd = [
        "cargo", "test", "-p", "jammi-db", "--test", "it", "--",
        _CARGO_TEST_FILTER, "--test-threads=1",
    ]
    proc = subprocess.run(
        cmd, cwd=fixtures_root, capture_output=True, text=True, timeout=600,
    )
    stdout = proc.stdout
    m = re.search(
        r"test result: (ok|FAILED)\. (\d+) passed; (\d+) failed;", stdout
    )
    if m is None:
        raise RuntimeError(
            f"could not parse 'cargo test' summary line from output:\n{stdout}\n{proc.stderr}"
        )
    passed, failed = int(m.group(2)), int(m.group(3))
    test_lines = [
        line.strip() for line in stdout.splitlines()
        if line.strip().startswith("test segment::") and " ... " in line
    ]
    return {
        "command": " ".join(cmd),
        "exit_code": proc.returncode,
        "passed": passed,
        "failed": failed,
        "tests": test_lines,
    }


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _write_checksums() -> None:
    sums = {
        p.name: _checksum(p)
        for p in sorted(ARTIFACTS.glob("*"))
        if p.is_file() and p.name != "checksums.json"
    }
    (ARTIFACTS / "checksums.json").write_text(json.dumps(sums, indent=2, sort_keys=True))


def _fixtures_root(arg: str | None) -> Path:
    root = arg or os.environ.get("JAMMI_FIXTURES_ROOT")
    if not root:
        raise SystemExit(
            "pass --fixtures-root (or set JAMMI_FIXTURES_ROOT) to the engine "
            "checkout carrying tests/fixtures/patents.parquet + tests/fixtures/tiny_modernbert"
        )
    p = Path(root).resolve()
    if not (p / "tests" / "fixtures" / "patents.parquet").exists():
        raise SystemExit(f"--fixtures-root {p} has no tests/fixtures/patents.parquet")
    if not (p / "tests" / "fixtures" / "tiny_modernbert" / "config.json").exists():
        raise SystemExit(f"--fixtures-root {p} has no tests/fixtures/tiny_modernbert")
    return p


def emit(fixtures_root: Path) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="jammi_segmented_ann_") as artifact_dir:
        db = jammi.connect(f"file://{artifact_dir}")
        print("== embedded engine: N=1 segmented-index property ==", flush=True)
        n1 = _run_n1_property(db, fixtures_root)
        catalog = _read_segment_catalog(Path(artifact_dir), n1["table_name"])
        if catalog["segment_count"] != 1:
            raise RuntimeError(
                f"expected exactly 1 segment for a freshly-built table, "
                f"found {catalog['segment_count']}: {catalog['segments']}"
            )
        seg0 = catalog["segments"][0]
        naming_ok = seg0["index_path"].endswith(f"{n1['table_name']}__seg0.idx")
        sidecar = _seg0_sidecar_files_present(seg0["index_path"])

    print("== jammi-db it-suite: append/no-rebuild property (live) ==", flush=True)
    append_suite = _run_append_suite(fixtures_root)

    # --- commit the ground-truth vectors + live hits (parquet) --------------- #
    # The engine embedding-table parquet shape: `_row_id` a string key, `vector`
    # a FixedSizeList<Float32, dim> — matching what the corpus reader / exact
    # oracle everywhere else in this book expect (see `build_scale_cache.py`'s
    # `_write_vectors`), not a variable-width list of doubles.
    row_ids = list(n1["vectors"].keys())
    dim = int(next(iter(n1["vectors"].values())).shape[0])
    vec_type = pa.list_(pa.field("item", pa.float32(), nullable=False), dim)
    pq.write_table(
        pa.table({
            "_row_id": pa.array(row_ids, type=pa.string()),
            "vector": pa.array(
                (n1["vectors"][rid] for rid in row_ids), type=vec_type
            ),
        }),
        ARTIFACTS / "corpus_vectors.parquet",
    )
    (ARTIFACTS / "live_hits.json").write_text(
        json.dumps(n1["live_hits"], indent=2, sort_keys=True)
    )
    (ARTIFACTS / "catalog_segment0.json").write_text(json.dumps({
        "segment_count": catalog["segment_count"],
        "segment_id": seg0["segment_id"],
        "row_count": seg0["row_count"],
        "naming_matches_seg0_convention": naming_ok,
        "sidecar_files_present": sidecar,
    }, indent=2, sort_keys=True))
    (ARTIFACTS / "append_suite.json").write_text(
        json.dumps(append_suite, indent=2, sort_keys=True)
    )

    golden = {
        "n1.segment_count": {"value": float(catalog["segment_count"]), "tol": 0.0},
        "n1.naming_matches_seg0_convention": {
            "value": 1.0 if naming_ok else 0.0, "tol": 0.0
        },
        "n1.sidecar_files_present": {
            "value": 1.0 if all(sidecar.values()) else 0.0, "tol": 0.0
        },
        "n1.search_matches_bruteforce_fraction": {
            "value": n1["n1_search_matches"] / n1["n1_search_trials"], "tol": 0.0
        },
        "append.exit_code": {"value": float(append_suite["exit_code"]), "tol": 0.0},
        "append.tests_passed": {"value": float(append_suite["passed"]), "tol": 0.0},
        "append.tests_failed": {"value": float(append_suite["failed"]), "tol": 0.0},
    }
    (ARTIFACTS / "golden_metrics.json").write_text(
        json.dumps(golden, indent=2, sort_keys=True)
    )

    record = {
        "purpose": (
            "The engine↔cookbook validator for the segmented-ANN capability "
            "(crates/jammi-db/src/index/segment.rs + ResultStore::append_segment): "
            "a table's ANN index as a set "
            "of immutable segments, searched through a merge that is byte-identical "
            "to the single-index path at N=1. Measures the SDK-reachable N=1 exact-"
            "search property live against the wheel, and the write-side append/"
            "no-rebuild property live against the already-shipped jammi-db it-suite "
            "(the layer it is public at today — no SDK/wire verb appends a second "
            "segment onto an already-ready table today)."
        ),
        "n1": {
            # `table_name` itself is not recorded: it embeds the sanitized
            # absolute `--fixtures-root` path (the model id is path-keyed), so
            # it is a machine-local string with no portable meaning across
            # emits. The property that matters — the `{table}__seg0.idx`
            # naming convention held — is already captured below.
            "segment_count": catalog["segment_count"],
            "segment_id": seg0["segment_id"],
            "row_count": seg0["row_count"],
            "naming_matches_seg0_convention": naming_ok,
            "sidecar_files_present": sidecar,
            "search_trials": n1["n1_search_trials"],
            "search_matches_bruteforce": n1["n1_search_matches"],
            "mismatches": n1["n1_search_mismatches"],
        },
        "append_no_rebuild": {
            "finding": (
                "No public verb appends a second segment onto an already-ready "
                "table: generate_embeddings, import_embeddings, graph propagation, "
                "and context-set materialization all call ResultStore::create_table "
                "fresh (grep-verified). ResultStore::append_segment is a jammi-db "
                "crate-level Rust primitive with no SDK/wire binding today, "
                "measured here at the layer it is live and public: the already-"
                "shipped jammi-db it-suite, re-run live, never modified."
            ),
            "measured_via": append_suite["command"],
            "exit_code": append_suite["exit_code"],
            "tests_passed": append_suite["passed"],
            "tests_failed": append_suite["failed"],
            "tests": append_suite["tests"],
            "headline_test": "segment::append_does_not_rebuild_prior_segments",
            "headline_test_asserts": [
                "segment 0's raw .usearch bytes are byte-identical before and after "
                "appending segment 1 (read off disk both times)",
                "the appended segment gets a fresh, distinct segment_id",
                "both segments' rows become searchable through the merged index "
                "(search_final finds each segment's own nearest row)",
            ],
        },
    }
    (ARTIFACTS / "record.json").write_text(json.dumps(record, indent=2, sort_keys=True))

    _write_checksums()

    print("\n=== segmented ANN index, measured ===", flush=True)
    print(f"  N=1: segment_count={catalog['segment_count']} segment_id={seg0['segment_id']} "
          f"naming_ok={naming_ok} sidecar_present={all(sidecar.values())}", flush=True)
    print(f"  N=1 search-vs-bruteforce: {n1['n1_search_matches']}/{n1['n1_search_trials']} "
          f"trials match", flush=True)
    print(f"  append it-suite: exit={append_suite['exit_code']} "
          f"passed={append_suite['passed']} failed={append_suite['failed']}", flush=True)
    print("\nemitted cache:", flush=True)
    for f in sorted(ARTIFACTS.glob("*")):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size} bytes)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures-root", default=None,
                    help="engine checkout with tests/fixtures/patents.parquet + "
                         "tests/fixtures/tiny_modernbert (or set JAMMI_FIXTURES_ROOT)")
    args = ap.parse_args()
    emit(_fixtures_root(args.fixtures_root))


if __name__ == "__main__":
    main()
