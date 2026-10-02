"""Query a `jammi-server` over Arrow Flight SQL, with nothing but pyarrow.

Run with `python cookbook/recipes/flight_sql/example.py`, or a step at a time
as a notebook: each `# %%` cell is one step. Needs a `jammi-server` on PATH
(`pip install jammi-server`).
"""

# %% [markdown]
# ## Encode a Flight SQL statement
#
# Flight SQL carries a statement as a `CommandStatementQuery` protobuf (field 1
# holds the query text) packed into a `google.protobuf.Any`. The server decodes
# a FlightDescriptor's command as that `Any`, so it refuses a bare SQL string.
# The two length-delimited messages are short enough to encode by hand, which
# keeps this program free of any Flight SQL client package.

# %%
import tempfile

import pyarrow.flight as flight
from jammi.testing import LiveServer

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
    """Encode `Any { CommandStatementQuery { query = sql } }` for Flight SQL."""
    command = _protobuf_field(1, sql.encode("utf-8"))  # CommandStatementQuery.query
    return _protobuf_field(1, _FLIGHTSQL_STATEMENT_TYPE_URL) + _protobuf_field(2, command)


# %% [markdown]
# ## Run the query against a server
#
# `LiveServer` starts a real `jammi-server` over a scratch artifact directory,
# on ports the kernel assigns, and waits until it answers a handshake; the
# server stops when the block ends. Flight SQL is two calls: `get_flight_info`
# resolves the statement into endpoints, each holding a ticket, and `do_get`
# streams a ticket's rows as Arrow record batches.

# %%
with tempfile.TemporaryDirectory() as tmp, LiveServer(tmp) as server:
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
