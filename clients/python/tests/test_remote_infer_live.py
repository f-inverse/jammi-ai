"""Live remote round-trip for the bulk inference verb.

Stands up a real CPU `jammi-server` and drives `infer` through the pure-Python
`RemoteDatabase`, asserting the remote result equals the embedded engine's:
the same registered source (the `patents.parquet` fixture) run through the same
deterministic local fixture model (`tiny_modernbert`) produces the same output
table on both transports. The per-row `_latency_ms` provenance column is
wall-clock (a fact about each run, not the data), so the parity assertion pins
its presence and type but not its values.

A zero-batch result is pinned explicitly: inference over an empty source yields
no batches, which the embedded binding surfaces as a schema-less empty
`pyarrow.Table` and the wire carries as an empty `ArrowBatch` — decoded to the
same schema-less empty table, so the transports agree on the degenerate shape.

Gated, not hermetic: the test needs a built server binary, so it is skipped
unless `JAMMI_SERVER_BIN` points at a `jammi-server` executable. The embedded
engine (`jammi_ai`) must also be importable (the parity peer).
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

jammi_ai = pytest.importorskip("jammi_ai")
import jammi_client  # noqa: E402

SERVER_BIN = os.environ.get("JAMMI_SERVER_BIN")

pytestmark = pytest.mark.skipif(
    not SERVER_BIN or not os.path.exists(SERVER_BIN),
    reason="JAMMI_SERVER_BIN not set to a built jammi-server binary",
)

# The repo's shared generic fixtures: the smallest source + deterministic local
# model the embedded inference tests already run (`patents.parquet` through
# `tiny_modernbert`, a 32-dim ModernBERT with committed weights).
FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures"
PATENTS_URL = f"file://{FIXTURES / 'patents.parquet'}"
TINY_MODERNBERT = f"local:{FIXTURES / 'tiny_modernbert'}"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    """A real `jammi-server` (CPU, all tiers) on a free port; torn down at module
    exit. Yields the `grpc://127.0.0.1:<port>` endpoint."""
    artifact_dir = tmp_path_factory.mktemp("jammi-srv")
    flight_port = _free_port()
    health_port = _free_port()
    env = dict(os.environ)
    env["JAMMI_ARTIFACT_DIR"] = str(artifact_dir)
    env["JAMMI_SERVER__FLIGHT_LISTEN"] = f"127.0.0.1:{flight_port}"
    env["JAMMI_SERVER__HEALTH_LISTEN"] = f"127.0.0.1:{health_port}"
    env["JAMMI_SERVER__SERVICES"] = "all"

    proc = subprocess.Popen(
        [SERVER_BIN],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Poll readiness via a trivial RemoteDatabase handshake.
    endpoint = f"grpc://127.0.0.1:{flight_port}"
    deadline = time.time() + 30
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            raise RuntimeError(f"jammi-server exited early:\n{out}")
        try:
            db = jammi_client.connect(endpoint)
            db.get_server_info()
            db.close()
            ready = True
            break
        except Exception:
            time.sleep(0.25)
    if not ready:
        proc.terminate()
        raise RuntimeError("jammi-server did not become ready within 30s")

    yield endpoint

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_infer_round_trip_matches_embedded(live_server, tmp_path):
    """`infer` over the same source with the same deterministic fixture model
    returns the same table on both transports: same column names and types, and
    value-equal rows for every column except the wall-clock `_latency_ms`."""
    remote = jammi_client.connect(live_server)
    embedded = jammi_ai.connect(f"file://{tmp_path}")
    try:
        results = {}
        for name, db in (("remote", remote), ("embedded", embedded)):
            db.add_source("patents", url=PATENTS_URL, format="parquet")
            results[name] = db.infer(
                source="patents",
                model=TINY_MODERNBERT,
                columns=["abstract"],
                task="text_embedding",
                key="id",
            )

        remote_table, embedded_table = results["remote"], results["embedded"]
        assert remote_table.schema == embedded_table.schema
        assert remote_table.num_rows == embedded_table.num_rows == 20

        # `_latency_ms` is per-run wall clock — its presence and type are pinned
        # by the schema assertion above; its values are facts about each run.
        deterministic = [
            name for name in remote_table.column_names if name != "_latency_ms"
        ]
        assert (
            remote_table.select(deterministic).to_pydict()
            == embedded_table.select(deterministic).to_pydict()
        )

        # Every row came back ok through the deterministic fixture model.
        assert set(remote_table.column("_status").to_pylist()) == {"ok"}
    finally:
        # The embedded engine releases its resources on drop (RAII); only the
        # remote client holds a gRPC channel that needs an explicit close.
        remote.close()


def test_infer_empty_source_is_a_schema_less_empty_table(live_server, tmp_path):
    """Inference over an empty source yields zero batches; both transports
    surface that as a schema-less empty `pyarrow.Table` (no columns, no rows).
    Embedded: the binding returns `pa.table({})` for an empty batch list. Remote:
    the server has no schema to encode, sends an empty `ArrowBatch`, and the
    client decodes it to the same `pa.table({})`."""
    empty_path = tmp_path / "empty.parquet"
    pq.write_table(
        pa.table(
            {
                "id": pa.array([], type=pa.string()),
                "abstract": pa.array([], type=pa.string()),
            }
        ),
        empty_path,
    )

    remote = jammi_client.connect(live_server)
    embedded = jammi_ai.connect(f"file://{tmp_path / 'engine'}")
    try:
        results = {}
        for name, db in (("remote", remote), ("embedded", embedded)):
            db.add_source("empty", url=f"file://{empty_path}", format="parquet")
            results[name] = db.infer(
                source="empty",
                model=TINY_MODERNBERT,
                columns=["abstract"],
                task="text_embedding",
                key="id",
            )

        for name, table in results.items():
            assert table.num_rows == 0, name
            assert table.num_columns == 0, name
        assert results["remote"].schema == results["embedded"].schema
    finally:
        # The embedded engine releases its resources on drop (RAII); only the
        # remote client holds a gRPC channel that needs an explicit close.
        remote.close()
