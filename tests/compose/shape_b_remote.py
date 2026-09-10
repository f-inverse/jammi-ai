"""Compose smoke: drive `deploy/docker-compose.yml` (Shape B) from a REMOTE
Python client over the wire, and prove a restart does not lose an existing
result.

Run by `.github/workflows/compose-smoke.yml` AFTER `docker compose -f
deploy/docker-compose.yml -f deploy/docker-compose.ci.yml up --wait` has
brought the stack up healthy. Deliberately outside `tests/uat/` — ci.yml's
`test-python` job globs `tests/uat/shape_b_*.py` / `tests/uat/shape_c_*.py`
on every PR against the EMBEDDED engine; this script needs a running remote
server + Postgres + JetStream, which only this workflow provisions.

Connects to the published gRPC/Flight SQL ports (8081), asserts the running
deployment's `get_server_info().broker` is actually `"jet_stream"` (a
runtime oracle — the compile-time `features` list alone would still pass
with the JetStream URL deleted from the compose file), registers the
bundled `patents.parquet` fixture (bind-mounted by the CI compose override
at `/fixtures/patents.parquet`), embeds one column with the bundled
`tiny_bert` fixture, searches for the stored vector's own nearest neighbor
(an exact self-hit), restarts the server container, and repeats the same
search — asserting the segment bundle on the Postgres-backed volume survives
a restart untouched (Shape B durability), not merely that the server
answers again. The workflow (`.github/workflows/compose-smoke.yml`) queries
the Postgres container directly after this script exits, as the matching
runtime oracle for the catalog side.

Every assertion prints what it compared before raising, so a CI failure log
shows the mismatch without a re-run.

Usage: `python3 tests/compose/shape_b_remote.py [--url grpc://127.0.0.1:8081]
[--health-url http://127.0.0.1:8080] [--dry-run]`. `--dry-run` parses
arguments, prints the plan (target, fixture paths, the SQL literals this
script issues), and exits 0 without connecting — the offline check this
script's own argument parsing and SQL text get when Docker is unavailable.
Exits 0 on success.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request

SOURCE_URL = "file:///fixtures/patents.parquet"
MODEL = "local:/fixtures/tiny_bert"
EXPECTED_ROW0_KEY = "1"
SCORE_TOLERANCE = 1e-6
COMPOSE_FILES = [
    "deploy/docker-compose.yml",
    "deploy/docker-compose.ci.yml",
]
COMPOSE_SERVICE = "jammi-server"


def _compose_cmd(*args: str) -> list[str]:
    cmd = ["docker", "compose"]
    for f in COMPOSE_FILES:
        cmd += ["-f", f]
    cmd += list(args)
    return cmd


def wait_for_ready(health_url: str, timeout_secs: float) -> None:
    """Poll `<health_url>/readyz` until it returns HTTP 200 or `timeout_secs`
    elapses, then raise. This mirrors `jammi-server probe`'s own success
    criterion without shelling out to the binary (the runner host has no
    `jammi-server` on PATH — the binary lives only inside the container)."""
    deadline = time.monotonic() + timeout_secs
    url = f"{health_url}/readyz"
    last_error: str = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310
                if resp.status == 200:
                    return
                last_error = f"HTTP {resp.status}"
        except (urllib.error.URLError, OSError) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise AssertionError(
        f"{url} did not return 200 within {timeout_secs}s (last: {last_error})"
    )


def restart_and_wait(health_url: str, timeout_secs: float) -> None:
    print(f"=== docker compose restart {COMPOSE_SERVICE} ===")
    subprocess.run(
        _compose_cmd("restart", COMPOSE_SERVICE),
        check=True,
    )
    wait_for_ready(health_url, timeout_secs)
    print(f"{COMPOSE_SERVICE} healthy after restart")


def run(target: str, health_url: str) -> int:
    import jammi  # deferred: --dry-run must not require the package installed

    db = jammi.connect(target)

    info = db.get_server_info()

    # Compile-time capability check only: this build was compiled with the
    # jetstream-broker feature. It does NOT prove the RUNNING deployment is
    # actually using JetStream as its broker -- deleting the broker URL from
    # the compose file would still leave this assertion passing.
    features = info["features"]
    assert "jetstream-broker" in features, (
        f"expected 'jetstream-broker' in get_server_info().features, got {features}"
    )

    # Runtime oracle: the broker this session is ACTUALLY running, per
    # `BrokerKind::as_str` (crates/jammi-db/src/trigger/broker.rs). Unlike
    # `features` above, this would fail if the compose file's
    # `JAMMI_BROKER__JET_STREAM__URL` were deleted (the server would fall
    # back to the in-memory broker and report `"in_memory"` here instead).
    broker = info["broker"]
    assert broker == "jet_stream", (
        f"expected get_server_info().broker == 'jet_stream', got {broker!r} -- "
        "the running deployment is not actually backed by the compose file's "
        "JetStream service"
    )

    db.add_source("patents", url=SOURCE_URL, format="parquet")

    n_table = db.sql("SELECT count(*) FROM patents.public.patents")
    n = n_table.column(0)[0].as_py()
    print(f"source row count N = {n}")

    table = db.generate_embeddings(
        source="patents",
        model=MODEL,
        columns=["abstract"],
        key="id",
    )
    print(f"generate_embeddings -> table = {table!r}")

    result_ident = f'"jammi.{table}"'
    result_count_table = db.sql(f"SELECT count(*) FROM {result_ident}")
    result_count = result_count_table.column(0)[0].as_py()
    assert result_count == n, (
        f"expected {result_ident} row count == N ({n}), got {result_count}"
    )

    vec_table = db.sql(
        f"SELECT vector FROM {result_ident} WHERE _row_id = '{EXPECTED_ROW0_KEY}'"
    )
    vec = vec_table.column("vector")[0].as_py()
    assert vec is not None, f"expected a stored vector for _row_id = '{EXPECTED_ROW0_KEY}'"

    def do_search():
        return db.search(source="patents", query=vec, k=5)

    hits = do_search()
    hit_keys = hits.column("key").to_pylist()
    hit_scores = hits.column("score").to_pylist()
    print("top-5 hits:")
    for key, score in zip(hit_keys, hit_scores):
        print(f"  key={key!r} score={score}")

    assert len(hits) == 5, f"expected 5 hits, got {len(hits)}: keys={hit_keys}"
    assert hit_keys[0] == EXPECTED_ROW0_KEY, (
        f"expected rank-1 key == {EXPECTED_ROW0_KEY!r}, got {hit_keys[0]!r} (all keys: {hit_keys})"
    )
    assert hit_scores[0] >= 1.0 - SCORE_TOLERANCE, (
        f"expected rank-1 score >= {1.0 - SCORE_TOLERANCE}, got {hit_scores[0]}"
    )

    segments_before = db.list_index_segments(table)
    print(f"list_index_segments({table!r}) before restart: {segments_before}")

    restart_and_wait(health_url, timeout_secs=60)

    hits_after = do_search()
    hit_keys_after = hits_after.column("key").to_pylist()
    hit_scores_after = hits_after.column("score").to_pylist()
    assert hit_keys_after == hit_keys, (
        f"expected identical keys after restart, before={hit_keys} after={hit_keys_after}"
    )
    for before, after in zip(hit_scores, hit_scores_after):
        assert abs(before - after) <= SCORE_TOLERANCE, (
            f"expected scores within {SCORE_TOLERANCE} after restart, "
            f"before={hit_scores} after={hit_scores_after}"
        )

    segments_after = db.list_index_segments(table)
    assert segments_after == segments_before, (
        f"expected list_index_segments unchanged after restart, "
        f"before={segments_before} after={segments_after}"
    )

    print("shape_b_remote: OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="grpc://127.0.0.1:8081")
    parser.add_argument("--health-url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan (target, fixtures, SQL literals) and exit 0 without connecting.",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("shape_b_remote --dry-run")
        print(f"  target       = {args.url}")
        print(f"  health url   = {args.health_url}")
        print(f"  source url   = {SOURCE_URL}")
        print(f"  model        = {MODEL}")
        print("  assert get_server_info().broker == \"jet_stream\"")
        print("  add_source(\"patents\", url=SOURCE_URL, format=\"parquet\")")
        print("  SELECT count(*) FROM patents.public.patents")
        print(
            "  generate_embeddings(source=\"patents\", model=MODEL, "
            "columns=[\"abstract\"], key=\"id\")"
        )
        print('  SELECT count(*) FROM "jammi.{table}"')
        print('  SELECT vector FROM "jammi.{table}" WHERE _row_id = \'1\'')
        print("  search(source=\"patents\", query=vec, k=5)")
        print(f"  docker compose restart {COMPOSE_SERVICE}")
        return 0

    return run(args.url, args.health_url)


if __name__ == "__main__":
    sys.exit(main())
