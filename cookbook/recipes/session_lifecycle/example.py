"""Ephemeral session storage: tables deleted when the session ends, with proof.

Run with `python cookbook/recipes/session_lifecycle/example.py`, or a step at
a time as a notebook: each `# %%` cell is one step.
"""

# %% [markdown]
# An ephemeral session is a tenant-scoped storage context whose tables are
# deleted when the session ends: on `close()`, on leaving its `with` block, or
# when the timeout scanner force-closes it. Every transition is published to
# the `jammi.audit.session_lifecycle.v1` topic, so an audit-log aggregator can
# prove the deletion happened.

# %%
import hashlib
import json
import tempfile

import pyarrow as pa

import jammi

TENANT = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a"

db = jammi.connect(f"file://{tempfile.mkdtemp()}")
db.set_tenant(TENANT)

# %% [markdown]
# ## What outlives the request
#
# Only lineage is durable: the hashes of the uploaded images go in an ordinary
# mutable table. The images themselves are the throwaway working set.

# %%
db.create_mutable_table(
    "query_lineage",
    schema=pa.schema([pa.field("image_hash", pa.string(), nullable=False)]),
    primary_key=["image_hash"],
)

uploads = {"img-1": b"...image one bytes...", "img-2": b"...image two bytes..."}
hashes = {iid: "sha256:" + hashlib.sha256(data).hexdigest() for iid, data in uploads.items()}

# %% [markdown]
# ## Work in an ephemeral session
#
# Tables created in the session belong to it. `ephem.sql` replaces `{table}`
# with the tenant-scoped reference to the named ephemeral table. The lineage
# is written to the persistent table before the session closes, while the
# working data still exists; leaving the block closes the session, which drops
# its tables and publishes a `closed` event.

# %%
images_schema = pa.schema(
    [
        pa.field("image_id", pa.string(), nullable=False),
        pa.field("image_hash", pa.string(), nullable=False),
    ]
)

with db.ephemeral_session(timeout_seconds=3600) as ephem:
    ephem.create_ephemeral_table("query_images", schema=images_schema, primary_key=["image_id"])
    batch = pa.table(
        {"image_id": list(hashes), "image_hash": list(hashes.values())}, schema=images_schema
    )
    inserted = ephem.insert("query_images", batch=batch)
    assert inserted == 2, "two rows stored in the ephemeral table"
    assert ephem.count_rows("query_images") == 2

    stored = ephem.sql("query_images", "SELECT image_hash FROM {table}")
    for h in stored.column("image_hash").to_pylist():
        db.sql(f"INSERT INTO mutable.public.query_lineage (image_hash) VALUES ('{h}')")
    print("ephemeral rows during session:", ephem.count_rows("query_images"))

# %% [markdown]
# ## After the session
#
# The persistent lineage survives, referring to the hashes, never to the
# deleted working data.

# %%
lineage = db.sql("SELECT image_hash FROM mutable.public.query_lineage")
assert lineage.num_rows == 2, "hash lineage persists after session close"
print("persistent lineage rows after close:", lineage.num_rows)

# %% [markdown]
# ## The proof of deletion
#
# The lifecycle topic carries an `opened` and a `closed` event for the
# session; the `closed` one reports how many rows were deleted. Lifecycle
# events (`opened`, `closed`, `timed_out`, `partial_deletion_failure`) carry
# the session id, the tenant, the table count and the deleted-row count.

# %%
events = db.subscribe_collect("jammi.audit.session_lifecycle.v1", from_offset=0)
records = [json.loads(r) for r in events.column("record").to_pylist()]
kinds = [r["event"] for r in records]
assert "opened" in kinds, "opened event published"
assert "closed" in kinds, "closed event published"
closed = next(r for r in records if r["event"] == "closed")
assert closed["deleted_row_count"] == 2, "closed event reports deleted rows"
print("lifecycle events:", kinds)

# %%
db.close()
