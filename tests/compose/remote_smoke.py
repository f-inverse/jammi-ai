"""Shared remote-smoke oracle: fixture constants, the `/readyz` poller, the
end-to-end `run()` body, and two named after-restart callbacks.

This module carries the entire behavioral core that used to live in
`shape_b_remote.py` (the Compose driver). `shape_b_remote.py` and
`shape_c_kube_remote.py` (the Kubernetes driver) are now thin: each supplies
its own `restart` strategy (how to actually bounce the server process) and
its own `after_restart` callback (what property must still hold once the
server answers again), and both call `run()` here.

`run(target, health_url, *, restart, after_restart)`:
  1. connects, asserts the compile-time `jetstream-broker` feature AND the
     RUNTIME `get_server_info().broker == "jet_stream"` (the latter is the
     real oracle: deleting the broker URL from a deployment's config would
     still leave the former passing);
  2. registers the bundled `patents.parquet` fixture, embeds one column with
     the bundled `tiny_bert` fixture, searches for the stored vector's own
     nearest neighbor (an exact self-hit);
  3. calls `restart()` — the injected strategy that actually bounces the
     server (`docker compose restart` for Compose, `kubectl rollout
     restart` for Kubernetes);
  4. calls `after_restart(ctx)` — the injected property that must still
     hold once the server is back up. `durable_after_restart` (Compose, a
     Postgres-backed volume) re-asserts the exact same search result and an
     unchanged index-segment list — durability across the restart.
     `shared_catalog_after_restart` (Kubernetes, an emptyDir — NO durability
     claim, and the result table itself is NOT queried: see the callback's
     own docstring) instead asserts the NEW pod can see what the OLD pod
     wrote to the shared catalog/broker: the registered source is still
     visible, the sources count is unchanged, and the broker is still
     `jet_stream`.

Every assertion prints what it compared before raising, so a CI failure log
shows the mismatch without a re-run.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

SOURCE_URL = "file:///fixtures/patents.parquet"
MODEL = "local:/fixtures/tiny_bert"
EXPECTED_ROW0_KEY = "1"
SCORE_TOLERANCE = 1e-6


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
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
                last_error = f"HTTP {resp.status}"
        except (urllib.error.URLError, OSError) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise AssertionError(
        f"{url} did not return 200 within {timeout_secs}s (last: {last_error})"
    )


@dataclass
class Ctx:
    """What an `after_restart` callback needs: the still-open client, the
    result table `run()` just built, the same search closure it already
    ran once (so an after-restart callback can re-run the identical
    query), the hits from that first run (to diff against), the index
    segments observed before the restart, the source fixture's row count
    `N`, and the catalog's `list_sources()` snapshot taken before the
    restart (so a callback can assert the sources count is unchanged
    without ever touching the result table)."""

    db: Any
    table: str
    do_search: Callable[[], Any]
    hit_keys: list[Any]
    hit_scores: list[float]
    segments_before: Any
    n: int
    sources_before: Any


def durable_after_restart(ctx: Ctx) -> None:
    """Shape B (Compose): the segment bundle lives on a Postgres-backed
    volume, so the exact same search must still answer identically, and the
    index segment list must be unchanged."""
    hits_after = ctx.do_search()
    hit_keys_after = hits_after.column("key").to_pylist()
    hit_scores_after = hits_after.column("score").to_pylist()
    assert hit_keys_after == ctx.hit_keys, (
        f"expected identical keys after restart, before={ctx.hit_keys} after={hit_keys_after}"
    )
    for before, after in zip(ctx.hit_scores, hit_scores_after):
        assert abs(before - after) <= SCORE_TOLERANCE, (
            f"expected scores within {SCORE_TOLERANCE} after restart, "
            f"before={ctx.hit_scores} after={hit_scores_after}"
        )

    segments_after = ctx.db.list_index_segments(ctx.table)
    assert segments_after == ctx.segments_before, (
        f"expected list_index_segments unchanged after restart, "
        f"before={ctx.segments_before} after={segments_after}"
    )


def shared_catalog_after_restart(ctx: Ctx) -> None:
    """Shape C (Kubernetes): the volume is an emptyDir — NO durability
    claim, for the CATALOG or the RESULT TABLE. The ci overlay's
    `deploy/kubernetes/overlays/ci/jammi.toml` carries no `[storage]`
    block, so a result table's Parquet segment lands under `artifact_dir`
    on the emptyDir; `load_existing_tables` (`crates/jammi-db/src/store/
    mod.rs`) only registers a `ready` row whose Parquet still `exists` at
    its `parquet_path`, and a rollout-restart's new pod starts with that
    path empty. Querying the result table here would therefore assert a
    property this deployment shape deliberately does not have --
    deliberately NOT done. What must hold instead is that a NEW pod,
    reconnecting to the SAME catalog and broker, sees what the OLD pod
    wrote to the catalog: the registered source is visible, the sources
    count is unchanged, and the broker is still `jet_stream`."""
    described = ctx.db.describe_source("patents")
    assert described is not None, (
        "expected describe_source('patents') to be visible to the pod after "
        f"restart, got {described!r} -- the new pod is not sharing the old "
        "pod's catalog"
    )

    sources_after = ctx.db.list_sources()
    assert len(sources_after) == len(ctx.sources_before), (
        f"expected list_sources() count unchanged after restart, "
        f"before={len(ctx.sources_before)} after={len(sources_after)}"
    )

    info = ctx.db.get_server_info()
    broker = info["broker"]
    assert broker == "jet_stream", (
        f"expected get_server_info().broker == 'jet_stream' after restart, got {broker!r} -- "
        "the new pod is not backed by the same JetStream broker"
    )


def run(
    target: str,
    health_url: str,
    *,
    restart: Callable[[], None],
    after_restart: Callable[[Ctx], None],
) -> int:
    import jammi  # deferred: --dry-run must not require the package installed

    db = jammi.connect(target)

    info = db.get_server_info()

    # Compile-time capability check only: this build was compiled with the
    # jetstream-broker feature. It does NOT prove the RUNNING deployment is
    # actually using JetStream as its broker -- deleting the broker URL from
    # the deployment's config would still leave this assertion passing.
    features = info["features"]
    assert "jetstream-broker" in features, (
        f"expected 'jetstream-broker' in get_server_info().features, got {features}"
    )

    # Runtime oracle: the broker this session is ACTUALLY running, per
    # `BrokerKind::as_str` (crates/jammi-db/src/trigger/broker.rs). Unlike
    # `features` above, this would fail if the deployment's
    # `JAMMI_BROKER__JET_STREAM__URL` were deleted (the server would fall
    # back to the in-memory broker and report `"in_memory"` here instead).
    broker = info["broker"]
    assert broker == "jet_stream", (
        f"expected get_server_info().broker == 'jet_stream', got {broker!r} -- "
        "the running deployment is not actually backed by the JetStream service"
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

    sources_before = db.list_sources()
    print(f"list_sources() before restart: {sources_before}")

    restart()

    after_restart(
        Ctx(
            db=db,
            table=table,
            do_search=do_search,
            hit_keys=hit_keys,
            hit_scores=hit_scores,
            segments_before=segments_before,
            n=n,
            sources_before=sources_before,
        )
    )

    print("remote_smoke: OK")
    return 0
