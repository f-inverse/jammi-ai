"""Publish to a Jammi topic and read it back, on the in-process broker.

Run with `python cookbook/recipes/trigger_streams/example.py`, or a step at a
time as a notebook: each `# %%` cell is one step.
"""

# %%
import tempfile

import pyarrow as pa

import jammi

db = jammi.connect(f"file://{tempfile.mkdtemp()}")

# %% [markdown]
# ## Register a topic
#
# A topic carries Arrow record batches of one schema. Registering it adds it
# to the catalog, where `list_topics` finds it.

# %%
schema = pa.schema(
    [
        pa.field("event_id", pa.int64(), nullable=False),
        pa.field("payload", pa.string(), nullable=False),
    ]
)
topic_id = db.register_topic("events.demo", schema=schema)
assert topic_id, "register_topic returned empty id"

topics = db.list_topics()
assert "events.demo" in topics, f"events.demo missing from {topics}"
print(topics)

# %% [markdown]
# ## Publish a batch
#
# The broker assigns each topic's batches sequential offsets, from 0 for a
# fresh topic, and `publish_topic` returns the one it assigned.

# %%
batch = pa.table(
    {
        "event_id": pa.array([1, 2, 3], type=pa.int64()),
        "payload": pa.array(["alpha", "beta", "gamma"], type=pa.string()),
    },
    schema=schema,
)
offset = db.publish_topic("events.demo", batch=batch)
assert offset == 0, f"expected offset 0, got {offset}"

# %% [markdown]
# ## Read it back from an offset
#
# Every published batch is also stored in the topic's backing table, so a
# subscriber can replay from any offset. `subscribe_collect` drains that table
# from `from_offset` and returns what it read.

# %%
collected = db.subscribe_collect("events.demo", from_offset=0)
assert collected.column("event_id").to_pylist() == [1, 2, 3]
assert collected.column("payload").to_pylist() == ["alpha", "beta", "gamma"]
print(collected.to_pydict())

# %% [markdown]
# ## Drop the topic
#
# `drop_topic` removes it from the catalog. With `if_exists=True`, a topic
# that is already gone is not an error; without it, the error names the topic.

# %%
db.drop_topic("events.demo")
assert "events.demo" not in db.list_topics(), "events.demo persisted after drop"

db.drop_topic("events.demo", if_exists=True)

try:
    db.drop_topic("never.registered")
except (ValueError, RuntimeError) as missing:
    assert "never.registered" in str(missing), f"drop-missing error lost topic name: {missing}"
    print(missing)
else:
    raise AssertionError("drop-missing must raise")

# %%
db.close()
