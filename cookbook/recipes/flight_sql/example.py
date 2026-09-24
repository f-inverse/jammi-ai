"""Run a `SELECT` against a `jammi-server` subprocess over Arrow Flight SQL.

Run with `python cookbook/recipes/flight_sql/example.py`. Exits 0 on
success.

Requires a `jammi-server` binary on PATH: `pip install jammi-server`, or
`cargo build --release -p jammi-server` with `target/release` on PATH.
"""

from __future__ import annotations

import tempfile

import pyarrow.flight as flight
from jammi.testing import LiveServer

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

def main() -> int:
    # A real server over a scratch artifact directory, on kernel-assigned
    # ports; `jammi-server` comes from PATH (the `jammi-server` wheel, or a
    # `cargo build --release -p jammi-server` on PATH).
    with tempfile.TemporaryDirectory() as tmp, LiveServer(tmp) as server:
        # Flight SQL is a two-step protocol: resolve the command into a
        # FlightInfo (one or more endpoints with tickets), then `do_get` each
        # ticket. The FlightDescriptor.for_command bytes carry the SQL.
        client = flight.FlightClient(server.endpoint)
        descriptor = flight.FlightDescriptor.for_command(
            flightsql_statement_command("SELECT 1 AS one")
        )
        info = client.get_flight_info(descriptor)
        table = client.do_get(info.endpoints[0].ticket).read_all()
        client.close()

        assert table.num_rows == 1, f"expected 1 row, got {table.num_rows}"
        assert table.column("one").to_pylist() == [1]
        print(table.to_pydict())

    print("flight_sql: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
