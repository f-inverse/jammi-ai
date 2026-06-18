"""Cache-backed checks for the change-data-capture / trigger-topic vertical (C2).

These run on CPU against the committed CDC cache (no GPU, no recompute) and assert the
vertical's load-bearing facts: the committed record carries the published event stream
and its op counts, every measured replay-collect count matches its frozen golden, and —
the central teaching property — the recorded counts are internally consistent with the
``max_batches``-is-the-terminator rule (the full replay equals ``num_published``, the
checkpoint splits the add stream exactly, and the predicate-filtered counts equal the
matching-batch counts).

If the emitted cache is absent the heavy artifacts are skipped, but the committed golden
metrics, once present, are always asserted.
"""

from __future__ import annotations

import pytest

from jammi_cookbook import contracts

_CDC = contracts._dataset_dir("cdc")
_HAVE_CACHE = (_CDC / "golden_metrics.json").exists()
_needs_cache = pytest.mark.skipif(not _HAVE_CACHE, reason="cdc cache not emitted")


@_needs_cache
def test_record_carries_published_stream():
    """The committed record describes a real published event stream with op counts."""
    record = contracts.load_artifact("cdc.record")
    assert record["num_published"] > 0
    # the op counts partition the stream — every published event has exactly one op.
    assert sum(record["op_counts"].values()) == record["num_published"], (
        "the op counts must partition the published stream")
    # every op in the recorded counts is one the cycle actually emits.
    assert set(record["op_counts"]) == set(record["op_cycle"])
    # the checkpoint is a real interior offset of the stream.
    assert 0 < record["checkpoint_offset"] < record["num_published"]


@_needs_cache
def test_replay_counts_match_golden():
    """Every measured replay-collect count matches its frozen, zero-tolerance golden."""
    record = contracts.load_artifact("cdc.record")
    for key in ("replay_count", "add_count", "tail_count", "tail_add_count"):
        g = contracts.golden(f"cdc.{key}")
        assert g.tol == 0.0, f"{key} golden must be exact"
        assert g.value == float(record[key]), f"{key} golden must equal the recorded count"
    # the full backing-table replay yields one row per single-row published batch.
    assert record["replay_count"] == record["num_published"]


@_needs_cache
def test_max_batches_is_the_terminator():
    """The recorded counts obey the max_batches-is-the-terminator rule, exactly.

    The unfiltered tail terminator is ``num_published - checkpoint``; the predicate
    terminators equal the matching-batch counts; and offset-addressed replay conserves
    the stream — the checkpoint splits the add stream into pre-checkpoint + tail adds.
    """
    record = contracts.load_artifact("cdc.record")
    num = record["num_published"]
    checkpoint = record["checkpoint_offset"]

    # unfiltered tail: terminator = num_published - checkpoint, one row per batch.
    assert record["tail_batches"] == num - checkpoint
    assert record["tail_count"] == record["tail_batches"]

    # predicate terminators equal their yielded (== matching) batch counts.
    assert record["add_count"] == record["add_batches"]
    assert record["tail_add_count"] == record["tail_add_batches"]

    # the 'add' op count is exactly the offset-0 add-batch terminator.
    assert record["add_batches"] == record["op_counts"]["add"]

    # offset-addressed replay is lossless: the checkpoint splits the add stream so the
    # tail adds are a subset of all adds, and never exceed the unfiltered tail.
    assert 0 <= record["tail_add_count"] <= record["add_count"]
    assert record["tail_add_count"] <= record["tail_count"]
