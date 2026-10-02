"""Sign, store, verify, query and stream a per-query search audit record.

Run with `python cookbook/recipes/search_audit/example.py`, or a step at a
time as a notebook: each `# %%` cell is one step.
"""

# %% [markdown]
# ## The audit master key
#
# The engine signs every audit record with a key it derives per tenant from
# `JAMMI_AUDIT_MASTER_KEY`, and refuses to sign without one. This program sets
# a fixed key when none is set; in production the key comes from your secret
# manager, never from code:
#
# ```bash
# export JAMMI_AUDIT_MASTER_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
# ```

# %%
import os
import tempfile
import uuid

import jammi

if not os.environ.get("JAMMI_AUDIT_MASTER_KEY"):
    os.environ["JAMMI_AUDIT_MASTER_KEY"] = "00" * 32

TENANT = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a"

db = jammi.connect(f"file://{tempfile.mkdtemp()}")
db.set_tenant(TENANT)

# %% [markdown]
# ## Record a search
#
# A `PerQueryAudit` says what was queried, with which model, and what came
# back. `query_lineage` holds hashes and ids, never raw payloads: it is capped
# at 8 KiB. `audit.log` adds the session's tenant, signs the record, stores it
# and publishes it.

# %%
query_id = str(uuid.uuid4())
record = jammi.PerQueryAudit(
    query_id=query_id,
    model_id="openai/clip-vit-base-patch32",
    model_version="3aa649a",
    query_lineage={
        "image_hashes": ["sha256:9f86d0...", "sha256:2c26b4..."],
        "user_id": "12345",
        "crop": [120, 80, 480, 360],
    },
    top_k_result_ids=["doc-0017", "doc-0042"],
    retrieval_scores=[0.92, 0.88],
)
db.audit.log([record])

# %% [markdown]
# ## Fetch it back and verify it
#
# `fetch_by_query_id` returns the typed record, and `verify` re-derives the
# tenant's key and checks the signature, raising when the record was altered.
# `fetch_recent` lists the newest records.

# %%
fetched = db.audit.fetch_by_query_id(query_id)
assert fetched is not None, "record should be retrievable"
assert fetched.tenant_id == TENANT
assert fetched.signature, "record must carry a signature"
fetched.verify()
print(f"verified audit record for query {fetched.query_id}")

assert len(db.audit.fetch_recent(limit=10)) == 1

# %% [markdown]
# ## Query it with SQL
#
# The records live in the reserved `_jammi_search_audit` table. SQL reads it
# freely, under the same tenant scope; nothing writes it except `audit.log`,
# which is what keeps every row signed.

# %%
table = db.sql('SELECT model_id, model_version FROM mutable.public."_jammi_search_audit"')
assert table.num_rows == 1, "SQL view returns the tenant's audit rows"
print("model_id via SQL:", table.column("model_id")[0].as_py())

# %% [markdown]
# ## Stream it
#
# Every logged record is also published on the `jammi.audit.search.v1` topic,
# for an alerting, analytics or warehouse subscriber. `from_offset=0` replays
# the topic's backing table and returns once it has caught up.

# %%
delivered = db.subscribe_collect("jammi.audit.search.v1", from_offset=0)
assert delivered.num_rows >= 1, "subscriber receives the audit payload"
print("audit topic delivered", delivered.num_rows, "row(s)")

# %%
db.close()
