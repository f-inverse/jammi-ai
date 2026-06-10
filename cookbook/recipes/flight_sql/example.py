"""Run a `SELECT` against a `jammi-server` subprocess over Arrow Flight SQL.

Run with `python cookbook/recipes/flight_sql/example.py`. Exits 0 on
success. Excluded from the default smoke matrix because it shells out to
the `jammi-server` binary; gated behind `JAMMI_COOKBOOK_SLOW=1` in
`tests/cookbook_smoke.py` and the nightly CI cron.

Requires the OSS server binary at `target/release/jammi-server` (or whatever
`JAMMI_BIN` points to). Build it with `cargo build --release -p jammi-server`.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import pyarrow.flight as flight

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_BIN = REPO_ROOT / "target" / "release" / "jammi-server"

HEALTH_URL = "http://127.0.0.1:8080/healthz"
FLIGHT_URL = "grpc://127.0.0.1:8081"

# FlightSQL carries a SQL statement as a `CommandStatementQuery` protobuf
# (field 1 = the query string) packed into a `google.protobuf.Any`. The
# server (datafusion-flight-sql-server) decodes the FlightDescriptor command
# as that Any, so a bare SQL string is rejected as an invalid command. We
# encode it directly to keep the recipe dependency-free (no FlightSQL client
# package); these are the two short length-delimited protobuf messages the
# wire format requires.
_FLIGHTSQL_STATEMENT_TYPE_URL = (
    b"type.googleapis.com/arrow.flight.protocol.sql.CommandStatementQuery"
)


def _protobuf_varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _protobuf_field(field_no: int, payload: bytes) -> bytes:
    """Encode one length-delimited (wire type 2) protobuf field."""
    tag = (field_no << 3) | 2
    return _protobuf_varint(tag) + _protobuf_varint(len(payload)) + payload


def flightsql_statement_command(sql: str) -> bytes:
    """Encode `Any { CommandStatementQuery { query = sql } }` for FlightSQL."""
    command = _protobuf_field(1, sql.encode("utf-8"))  # CommandStatementQuery.query
    return _protobuf_field(1, _FLIGHTSQL_STATEMENT_TYPE_URL) + _protobuf_field(2, command)

READY_TIMEOUT_S = 10
POLL_INTERVAL_S = 0.1


def wait_for_health(url: str, timeout: float) -> None:
    """Poll the health endpoint until it returns 200 or the budget runs out."""
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001 — broad catch is intentional here
            last_err = exc
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(
        f"server health endpoint {url} did not respond within {timeout}s "
        f"(last error: {last_err})"
    )


def ports_free(*ports: int) -> bool:
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                return False
    return True


def main() -> int:
    binary = Path(os.environ.get("JAMMI_BIN", DEFAULT_BIN))
    if not binary.exists():
        print(
            f"jammi-server binary not found at {binary}. Build it with "
            "`cargo build --release -p jammi-server` or set JAMMI_BIN.",
            file=sys.stderr,
        )
        return 1
    if not ports_free(8080, 8081):
        print(
            "ports 8080 / 8081 already in use — close the conflicting "
            "process before running this recipe.",
            file=sys.stderr,
        )
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        env = os.environ.copy()
        env["JAMMI_ARTIFACT_DIR"] = tmp

        proc = subprocess.Popen(
            [str(binary)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            wait_for_health(HEALTH_URL, READY_TIMEOUT_S)

            # Flight SQL is a two-step protocol: resolve the command into a
            # FlightInfo (returns one or more endpoints with tickets), then
            # `do_get` each ticket. The FlightDescriptor.for_command bytes
            # carry the SQL statement.
            client = flight.FlightClient(FLIGHT_URL)
            descriptor = flight.FlightDescriptor.for_command(
                flightsql_statement_command("SELECT 1 AS one")
            )
            info = client.get_flight_info(descriptor)
            reader = client.do_get(info.endpoints[0].ticket)
            table = reader.read_all()

            assert table.num_rows == 1, f"expected 1 row, got {table.num_rows}"
            assert table.column("one").to_pylist() == [1]
            print(table.to_pydict())
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    print("flight_sql: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
