# Ephemeral session storage

A session-scoped storage context whose tables are auto-deleted when the session
ends — on explicit `close()`, on context-manager exit, or when the 60-second
timeout scanner force-closes a session past its deadline. Every transition
publishes to the `jammi.audit.session_lifecycle.v1` trigger topic, giving an
audit-log aggregator durable proof that the data was deleted.

Run it:

```bash
python cookbook/recipes/session_lifecycle/example.py
```

## When to use it

Use an ephemeral session for sensitive transient data that must not outlive the
request that produced it: uploaded images, derived embeddings, draft model
inputs. The session is always tenant-scoped — tenant A can never see tenant B's
ephemeral tables.

## When NOT to use it

Do not store long-lived data in an ephemeral session. The audit record, the
persistent corpus, and anything compliance needs to read later belong in
ordinary mutable tables. The pattern is: keep the *throwaway working set*
(raw bytes, embeddings) in the ephemeral session, and write only durable
*lineage* (hashes, ids, scores) to a persistent table — before you close the
session, while the working data still exists.

## API

```python
with db.ephemeral_session(timeout_seconds=3600) as ephem:
    ephem.create_ephemeral_table("imgs", schema=schema, primary_key=["image_id"])
    ephem.insert("imgs", batch=table)
    rows = ephem.sql("imgs", "SELECT image_hash FROM {table}")
# close() runs on exit: tables dropped, `closed` event published
```

`{table}` in a `sql` query is replaced by the tenant-scoped reference to the
named ephemeral table. The context manager is the recommended path; `Drop` is
best-effort. Lifecycle events (`opened`, `closed`, `timed_out`,
`partial_deletion_failure`) carry the session id, tenant, table count, and
deleted-row count.
