"""A database source's URL never writes its password, on both transports.

The engine keeps a source's URL in its catalog, so a password written into the URL
would be written into the catalog (#756). A URL names its password with a
``${JAMMI_SOURCE_<NAME>}`` placeholder, read from the environment of the process that
connects; one that writes the password, or names any other variable, is refused as
``InvalidArgument`` before anything connects or is stored, and the refusal never repeats it.

Selected by the `live_server` and `embedded` markers: it needs a built
`jammi-server` (`JAMMI_SERVER_BIN`) and the in-process engine.
"""

from __future__ import annotations

import pytest

import jammi
from jammi.errors import InvalidArgument

pytestmark = [pytest.mark.live_server, pytest.mark.embedded]

# Nothing listens on port 9: an engine that tried to connect would fail there.
WRITTEN = [
    "postgresql://ada:hunter2@127.0.0.1:9/sales",
    "postgresql://ada@127.0.0.1:9/sales?password=hunter2",
    "mysql://ada:hunter2@127.0.0.1:9/sales",
]
# A placeholder names a `JAMMI_SOURCE_` variable: the environment is the
# server's, so naming any other would send the server's own secret, as the
# password, to a host the client chose.
OUTSIDE = [
    "postgresql://ada:${AWS_SECRET_ACCESS_KEY}@127.0.0.1:9/sales",
    "mysql://ada:${JAMMI_AUDIT_MASTER_KEY}@127.0.0.1:9/sales",
]


@pytest.mark.parametrize("transport", ["embedded", "remote"])
def test_a_url_writing_its_password_is_refused(transport, tmp_path, live_server):
    if transport == "embedded":
        db = jammi.connect(f"file://{tmp_path / 'engine'}")
    else:
        db = jammi.connect(live_server)
    try:
        for n, url in enumerate(WRITTEN + OUTSIDE):
            with pytest.raises(InvalidArgument) as refused:
                db.add_source(f"db_{transport}_{n}", url=url)
            message = str(refused.value)
            assert "${" in message and "JAMMI_SOURCE_" in message, message
            assert "hunter2" not in message, message
        assert not [s for s in db.list_sources() if s["source_id"].startswith(f"db_{transport}")]
    finally:
        db.close()
