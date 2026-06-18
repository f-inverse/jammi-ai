#!/usr/bin/env python3
"""Emit the change-data-capture / trigger-topic cache (C2) — CPU, no GPU.

A trigger topic is the engine's offset-addressed, append-only **commit log**: you
register it with a row schema, ``publish`` discrete batches onto it (each landing at
the next 0-based offset), and a consumer ``subscribe``s by *replaying* the backing
table from a chosen offset — a Kafka-style publish → offset-addressed replay model
[@kreps2011kafka], the streaming complement of the append-only result log
[@kleppmann2017ddia]. This vertical builds a deterministic stream of **record-change
events** — each an ``op`` (``add`` / ``remove`` / ``update``) on a key, keyed by a
real committed ``arxiv`` ``paper_id`` for realism — publishes them as discrete
batches, and records the measured replay-collect counts the chapter reproduces live
against this golden.

The central, load-bearing property this vertical teaches (and the build itself
relies on): **``subscribe_collect``'s ``max_batches`` is the terminator, not a
ceiling-with-a-timeout.** ``subscribe_collect`` replays the backing table from
``from_offset`` and then tails the live broker; the in-memory broker tail blocks
forever waiting for the next publish. So the call returns synchronously **iff**
``max_batches`` equals exactly the number of *yielded* batches available from
``from_offset``:

* with no predicate, every published batch from the offset is yielded, so
  ``max_batches = num_published - from_offset``;
* with a predicate, a published batch whose rows all filter out is **dropped**
  (not yielded as an empty batch), so ``max_batches`` is the number of published
  batches at offset ``>= from_offset`` that contain **at least one** matching row.

Every ``subscribe_collect`` here — and in the chapter — pins ``max_batches`` to that
exact, programmatically-derived count. Setting it larger hangs the call uninterruptibly.

The events are fully deterministic and reproducible: the ``op`` cycle and the keys are
derived from the committed ``arxiv.papers`` cache in a fixed order, so a re-run is
byte-identical. The chapter re-derives the same events, publishes them onto a live
in-memory topic, and re-runs the bounded replay-collects, asserting against the goldens
emitted here.

Usage::

    python scripts/build_cdc_cache.py            # embedded CPU engine
    python scripts/build_cdc_cache.py --target file:///tmp/cdc
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import jammi_ai
import pyarrow as pa

import jammi_cookbook  # noqa: F401  # applies the determinism env on import
from jammi_cookbook import contracts

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts" / "cdc"

# The topic's row schema — one record-change event per row/batch. Generic CDC
# vocabulary: an operation, the key it acts on, and a thin payload + sequence.
TOPIC_SCHEMA = pa.schema(
    [
        ("op", pa.string()),
        ("key", pa.string()),
        ("payload", pa.string()),
        ("seq", pa.int64()),
    ]
)

# The deterministic op cycle. A fixed, repeating pattern over a real key order makes
# the stream reproducible and the predicate-matching batch count derivable by hand.
OP_CYCLE = ("add", "update", "remove")

# How many discrete change events (one published batch each) the stream carries.
NUM_EVENTS = 60

# The CDC "resume from a checkpoint" offset the chapter replays the tail from.
CHECKPOINT_OFFSET = 24

# The predicate the subscriber filters the replay by — the canonical CDC selective
# subscribe ("give me only the inserts").
PREDICATE = "op = 'add'"


def _events() -> list[dict]:
    """The deterministic change-event stream, keyed by committed ``arxiv`` paper ids.

    One event per row, in a fixed order: the i-th event's ``op`` cycles through
    ``OP_CYCLE`` and its ``key`` is the i-th committed paper id (wrapping if the
    stream is longer than the subset). Fully reproducible — same cache, same stream.
    """
    paper_ids = [p["paper_id"] for p in contracts.load_artifact("arxiv.papers").to_pylist()]
    events = []
    for i in range(NUM_EVENTS):
        op = OP_CYCLE[i % len(OP_CYCLE)]
        key = paper_ids[i % len(paper_ids)]
        events.append({"op": op, "key": key, "payload": f"rev{i:03d}", "seq": i})
    return events


def _batch(event: dict) -> pa.Table:
    """One change event as a single-row Arrow batch (one publish = one offset)."""
    return pa.table(
        {
            "op": [event["op"]],
            "key": [event["key"]],
            "payload": [event["payload"]],
            "seq": [event["seq"]],
        },
        schema=TOPIC_SCHEMA,
    )


def _max_batches_for_predicate(events: list[dict], predicate_op: str, from_offset: int) -> int:
    """The exact terminator for a predicate replay-collect from ``from_offset``.

    A single-row batch is yielded iff its row matches the predicate, so the count of
    yielded batches is the number of events at offset ``>= from_offset`` whose ``op``
    equals ``predicate_op``. Pinning ``max_batches`` to this returns synchronously;
    a larger value hangs on the live broker tail.
    """
    return sum(1 for e in events[from_offset:] if e["op"] == predicate_op)


def emit(db, _work: Path) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    events = _events()
    num_published = len(events)

    # --- Register the topic -------------------------------------------------- #
    topic_id = db.register_topic("changes", schema=TOPIC_SCHEMA)
    listed = db.list_topics()
    print(f"registered topic id={topic_id!r}; list_topics -> {listed}", flush=True)
    assert "changes" in listed, "the registered topic must appear in list_topics"

    # --- Publish N discrete batches, capturing the 0-based offsets ----------- #
    offsets = [db.publish_topic("changes", batch=_batch(e)) for e in events]
    assert offsets == list(range(num_published)), (
        "each publish_topic lands one batch at the next 0-based offset")
    print(f"published {num_published} change events at offsets "
          f"{offsets[0]}..{offsets[-1]}", flush=True)

    # --- The full backing-table replay (no predicate) ------------------------ #
    # max_batches = num_published - 0: every published batch is yielded, so the call
    # returns synchronously. This is the backing-table replay count.
    replay_all = db.subscribe_collect("changes", from_offset=0, max_batches=num_published)
    replay_count = replay_all.num_rows
    assert replay_count == num_published, (
        "the full replay yields one row per single-row published batch")
    print(f"full replay (from_offset=0, max_batches={num_published}) -> "
          f"{replay_count} rows", flush=True)

    # --- The predicate-filtered selective subscribe ('only the adds') -------- #
    add_batches = _max_batches_for_predicate(events, "add", 0)
    add_collected = db.subscribe_collect(
        "changes", predicate=PREDICATE, from_offset=0, max_batches=add_batches)
    add_count = add_collected.num_rows
    # The collected set must equal the predicate-filtered published set, exactly.
    published_adds = [e["seq"] for e in events if e["op"] == "add"]
    assert add_collected.column("seq").to_pylist() == published_adds, (
        "the predicate-collected events must equal the published adds, in order")
    assert add_count == add_batches == len(published_adds)
    print(f"selective subscribe ({PREDICATE!r}, from_offset=0, "
          f"max_batches={add_batches}) -> {add_count} rows", flush=True)

    # --- The 'resume from a checkpoint' tail replay -------------------------- #
    # max_batches = num_published - CHECKPOINT_OFFSET: every batch from the checkpoint
    # on is yielded. The collected tail must equal the published tail slice exactly.
    tail_batches = num_published - CHECKPOINT_OFFSET
    tail = db.subscribe_collect(
        "changes", from_offset=CHECKPOINT_OFFSET, max_batches=tail_batches)
    tail_count = tail.num_rows
    published_tail = [e["seq"] for e in events[CHECKPOINT_OFFSET:]]
    assert tail.column("seq").to_pylist() == published_tail, (
        "the checkpoint-resumed tail must equal the published tail slice, in order")
    assert tail_count == tail_batches == published_tail[-1] - CHECKPOINT_OFFSET + 1
    print(f"resume tail (from_offset={CHECKPOINT_OFFSET}, max_batches={tail_batches}) "
          f"-> {tail_count} rows", flush=True)

    # --- The predicate AND checkpoint combined ------------------------------- #
    tail_add_batches = _max_batches_for_predicate(events, "add", CHECKPOINT_OFFSET)
    tail_adds = db.subscribe_collect(
        "changes", predicate=PREDICATE, from_offset=CHECKPOINT_OFFSET,
        max_batches=tail_add_batches)
    tail_add_count = tail_adds.num_rows
    published_tail_adds = [
        e["seq"] for e in events[CHECKPOINT_OFFSET:] if e["op"] == "add"]
    assert tail_adds.column("seq").to_pylist() == published_tail_adds
    assert tail_add_count == tail_add_batches == len(published_tail_adds)
    print(f"resume + filter ({PREDICATE!r}, from_offset={CHECKPOINT_OFFSET}, "
          f"max_batches={tail_add_batches}) -> {tail_add_count} rows", flush=True)

    op_counts = {op: sum(1 for e in events if e["op"] == op) for op in OP_CYCLE}

    record = {
        "topic": "changes",
        "key_source": "arxiv.papers (paper_id)",
        "num_published": num_published,
        "op_cycle": list(OP_CYCLE),
        "op_counts": op_counts,
        "predicate": PREDICATE,
        "checkpoint_offset": CHECKPOINT_OFFSET,
        "replay_count": int(replay_count),
        "add_count": int(add_count),
        "add_batches": int(add_batches),
        "tail_count": int(tail_count),
        "tail_batches": int(tail_batches),
        "tail_add_count": int(tail_add_count),
        "tail_add_batches": int(tail_add_batches),
        "note": (
            "A change-data-capture stream over a trigger topic: N discrete record-change "
            "events (op add/update/remove on a key, keyed by a committed arxiv paper_id) "
            "published as single-row batches at offsets 0..N-1, then drained by bounded "
            "offset-addressed replay-collect. The load-bearing property: max_batches is "
            "the TERMINATOR, not a ceiling — subscribe_collect replays the backing table "
            "from from_offset then tails the live broker, and the in-memory broker tail "
            "blocks forever, so the call returns synchronously IFF max_batches equals the "
            "exact number of YIELDED batches from from_offset. With no predicate that is "
            "num_published - from_offset; with a predicate it is the count of published "
            "batches at offset >= from_offset with at least one matching row (empty-"
            "filtered batches are dropped, not yielded). Every collect here pins "
            "max_batches to that programmatically-derived count. The measured verdicts are "
            "the full backing-table replay count, the predicate-filtered add count, the "
            "checkpoint-resumed tail count, and the resume+filter count — each reproduced "
            "live by the chapter against this golden."
        ),
    }
    (ARTIFACTS / "cdc.json").write_text(json.dumps(record, indent=2))

    metrics = {
        "replay_count": {"value": float(replay_count), "tol": 0.0},
        "add_count": {"value": float(add_count), "tol": 0.0},
        "tail_count": {"value": float(tail_count), "tol": 0.0},
        "tail_add_count": {"value": float(tail_add_count), "tol": 0.0},
    }
    (ARTIFACTS / "golden_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True))

    _write_checksums()
    print("\n=== change-data-capture, measured ===", flush=True)
    print(f"  num_published={num_published}  replay_count={replay_count}  "
          f"add_count={add_count}  tail_count={tail_count}  "
          f"tail_add_count={tail_add_count}", flush=True)
    print("\nemitted cache:", flush=True)
    for f in sorted(ARTIFACTS.glob("*")):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size} bytes)", flush=True)


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _write_checksums() -> None:
    sums = {p.name: _checksum(p) for p in sorted(ARTIFACTS.glob("*"))
            if p.is_file() and p.name != "checksums.json"}
    (ARTIFACTS / "checksums.json").write_text(json.dumps(sums, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None,
                    help="connect() target — file:// for the embedded CPU engine "
                         "(a fresh temp catalog is used if omitted).")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as catalog, tempfile.TemporaryDirectory() as work:
        db = jammi_ai.connect(args.target or f"file://{catalog}")
        emit(db, Path(work))


if __name__ == "__main__":
    main()
