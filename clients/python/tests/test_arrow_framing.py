"""Hermetic checks on the publish/subscribe Arrow-IPC framing helpers.

These run without a server: the encode/decode helpers are pure functions, and
the wire contract they must honour — `decode_publish_batch` accepts *exactly one*
RecordBatch message — is a property of the byte stream alone.
"""

import pyarrow as pa

from jammi_client._database import _arrow_batch_to_table, _table_to_arrow_batch


def _batch_count(body: bytes) -> int:
    """Number of RecordBatch messages in a self-describing IPC stream."""
    reader = pa.ipc.open_stream(body)
    return sum(1 for _ in reader)


def test_multi_chunk_table_publishes_as_one_batch():
    """A multi-chunk `pyarrow.Table` must encode to a *single* RecordBatch — the
    publish wire (`decode_publish_batch`) rejects any other count, and the embedded
    `publish_topic` collapses chunks via `concat_batches`, so the client must match.
    """
    chunk_a = pa.table({"id": [1, 2], "v": [0.5, 1.5]})
    chunk_b = pa.table({"id": [3, 4], "v": [2.5, 3.5]})
    multi = pa.concat_tables([chunk_a, chunk_b])
    # Sanity: this really is the >1-chunk shape that exposed the divergence.
    assert multi.column(0).num_chunks == 2

    encoded = _table_to_arrow_batch(multi)
    body = bytes(encoded.data_header) + bytes(encoded.data_body)
    assert _batch_count(body) == 1

    # And the single batch still carries every row, in order.
    assert _arrow_batch_to_table(encoded).combine_chunks() == multi.combine_chunks()


def test_single_chunk_table_round_trips():
    """The common single-chunk case stays a one-batch round-trip."""
    table = pa.table({"id": [10, 11, 12], "v": [1.0, 2.0, 3.0]})
    assert table.column(0).num_chunks == 1

    encoded = _table_to_arrow_batch(table)
    body = bytes(encoded.data_header) + bytes(encoded.data_body)
    assert _batch_count(body) == 1
    assert _arrow_batch_to_table(encoded) == table
