"""A statement that fails with a long message reaches a default client as its error.

A failure's message rides the gRPC trailers, which a default client caps at
8 KiB of metadata. A long message — a cast of a 20,000-character string, whose
error quotes the string — used to overflow them, so the client raised a
transport error ("received metadata size exceeds soft limit") and the real
error was lost. The server now caps what the trailers carry and logs the full
text, so the client raises the engine's error, whose message says where the
rest is. The embedded engine, which has no trailers, keeps the whole message.

Selected by the `live_server` and `embedded` markers: it needs a built
`jammi-server` (`JAMMI_SERVER_BIN`) and the in-process engine.
"""

from __future__ import annotations

import pytest

import jammi
from jammi.errors import JammiError

pytestmark = [pytest.mark.live_server, pytest.mark.embedded]

LONG = "x" * 20_000
STATEMENT = f"SELECT CAST('{LONG}' AS INT) AS n"


def test_a_long_failure_reaches_a_default_remote_client_as_its_error(live_server):
    db = jammi.connect(live_server)
    try:
        with pytest.raises(JammiError) as raised:
            db.sql(STATEMENT)
    finally:
        db.close()
    message = str(raised.value)
    assert "metadata size" not in message, message[:300]
    assert "server log" in message, message[-300:]
    assert len(message) < 8 * 1024


def test_a_long_failure_keeps_its_whole_message_in_process(tmp_path):
    db = jammi.connect(f"file://{tmp_path}")
    try:
        with pytest.raises(JammiError) as raised:
            db.sql(STATEMENT)
    finally:
        db.close()
    assert LONG in str(raised.value)
