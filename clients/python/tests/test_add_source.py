"""`RemoteDatabase.add_source`'s source kinds and `format=` vocabulary — hermetic,
no server.

A `postgres://` / `postgresql://` / `mysql://` URL is a database source, sent
with its wire `SourceKind` and no format, as the engine's
`SourceDefinition::from_url` (which the embedded arm calls) decides.

`_assembly._FILE_FORMAT` is a HAND-MAINTAINED mirror of the engine's
`jammi_db::source::FileFormat::from_str` (the embedded backend calls that
`FromStr` directly, unmediated by any Python dict — see
`jammi.EmbeddedBackend.add_source`). A token accepted by the engine but
missing from this dict makes the remote arm reject a format the embedded arm
accepts — a cross-surface-parity break (e.g. for `"jsonl"`/`"ndjson"`).
These tests pin
the dict itself (so a future new engine token is caught here, not only by a
slower native-build conformance test) and the two failure/success shapes of
`add_source`'s client-side validation seam — which runs and raises (or
resolves the format) before any channel I/O, so no server is contacted.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import jammi
from jammi._assembly import _FILE_FORMAT
from jammi._generated.jammi.v1 import catalog_pb2


def test_file_format_vocabulary_covers_every_wire_token_including_both_jsonl_spellings():
    """Every non-`UNSPECIFIED` wire `FileFormat` value is reachable through at
    least one `_FILE_FORMAT` key, and both accepted spellings of the
    line-delimited-JSON token ("jsonl" and "ndjson") resolve to the SAME wire
    value, `FILE_FORMAT_JSONL` — mirroring the engine's `FileFormat::from_str`,
    which accepts both for its one `JsonLines` variant."""
    wire_values = {
        v.number
        for v in catalog_pb2.FileFormat.DESCRIPTOR.values
        if v.name != "FILE_FORMAT_UNSPECIFIED"
    }
    mirrored_values = set(_FILE_FORMAT.values())
    missing = wire_values - mirrored_values
    assert not missing, (
        f"_FILE_FORMAT is missing a mirror for wire FileFormat value(s) "
        f"{missing} — a new wire value exists but no Python spelling reaches it"
    )

    assert _FILE_FORMAT["jsonl"] == catalog_pb2.FileFormat.FILE_FORMAT_JSONL
    assert _FILE_FORMAT["ndjson"] == catalog_pb2.FileFormat.FILE_FORMAT_JSONL
    assert _FILE_FORMAT["jsonl"] == _FILE_FORMAT["ndjson"]


def test_add_source_rejects_an_unknown_format_before_any_channel_i_o():
    """An unrecognised `format=` is a client-side `InvalidArgument`, raised
    before `RemoteDatabase._call` ever runs — asserted by patching `_call` to
    fail the test if reached."""
    remote = jammi.connect("grpc://127.0.0.1:8081")
    try:
        with patch.object(remote, "_call") as mock_call:
            with pytest.raises(jammi.InvalidArgument):
                remote.add_source("s", url="/tmp/x.parquet", format="bogus")
        mock_call.assert_not_called()
    finally:
        remote.close()


@pytest.mark.parametrize("token", ["jsonl", "ndjson"])
def test_add_source_accepts_jsonl_and_ndjson_and_sends_the_jsonl_wire_value(token):
    """Both jsonl spellings pass client-side validation (no `InvalidArgument`)
    and encode onto the wire as `FILE_FORMAT_JSONL` — `_call` is mocked so the
    assertion is on the constructed request, never a real RPC."""
    remote = jammi.connect("grpc://127.0.0.1:8081")
    try:
        with patch.object(remote, "_call", return_value=None) as mock_call:
            remote.add_source("s", url="/tmp/x.jsonl", format=token)
        mock_call.assert_called_once()
        sent_request = mock_call.call_args[0][1]
        assert (
            sent_request.connection.format == catalog_pb2.FileFormat.FILE_FORMAT_JSONL
        )
    finally:
        remote.close()


def _sent(url, **kwargs):
    """The `AddSourceRequest` `add_source(url=url, **kwargs)` would send."""
    remote = jammi.connect("grpc://127.0.0.1:8081")
    try:
        with patch.object(remote, "_call", return_value=None) as mock_call:
            remote.add_source("s", url=url, **kwargs)
        mock_call.assert_called_once()
        return mock_call.call_args[0][1]
    finally:
        remote.close()


@pytest.mark.parametrize(
    "url, kind",
    [
        ("postgres://reader:pw@db.internal:5432/sales", catalog_pb2.SourceKind.SOURCE_KIND_POSTGRES),
        ("postgresql://postgres:@/postgres?host=/tmp/pg", catalog_pb2.SourceKind.SOURCE_KIND_POSTGRES),
        ("mysql://reader:pw@127.0.0.1:3306/registry", catalog_pb2.SourceKind.SOURCE_KIND_MYSQL),
    ],
)
def test_a_database_url_is_sent_as_its_database_kind_with_the_url_verbatim(url, kind):
    sent = _sent(url, tenant_column="org")
    assert sent.source_kind == kind
    assert sent.connection.url == url
    assert sent.connection.format == catalog_pb2.FileFormat.FILE_FORMAT_UNSPECIFIED
    assert sent.connection.tenant_column == "org"


def test_a_file_url_is_sent_as_a_file_in_its_format():
    sent = _sent("/tmp/x.csv", format="csv")
    assert sent.source_kind == catalog_pb2.SourceKind.SOURCE_KIND_FILE
    assert sent.connection.url == "file:///tmp/x.csv"
    assert sent.connection.format == catalog_pb2.FileFormat.FILE_FORMAT_CSV


@pytest.mark.parametrize(
    "url, format",
    [("postgres://u@h/d", "parquet"), ("mysql://u@h/d", "csv"), ("/tmp/x.parquet", None)],
)
def test_a_database_with_a_format_or_a_file_without_one_is_refused_before_any_i_o(url, format):
    remote = jammi.connect("grpc://127.0.0.1:8081")
    try:
        with patch.object(remote, "_call") as mock_call:
            with pytest.raises(jammi.InvalidArgument):
                remote.add_source("s", url=url, format=format)
        mock_call.assert_not_called()
    finally:
        remote.close()
