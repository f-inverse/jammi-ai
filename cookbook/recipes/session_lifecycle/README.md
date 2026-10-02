# Ephemeral session storage

A session-scoped storage context whose tables are auto-deleted when the session
ends — on explicit `close()`, on context-manager exit, or when the 60-second
timeout scanner force-closes a session past its deadline. Every transition
publishes to the `jammi.audit.session_lifecycle.v1` trigger topic, giving an
audit-log aggregator durable proof that the data was deleted.

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

## Run it

```bash
python cookbook/recipes/session_lifecycle/example.py
```

It prints the rows held during the session, the lineage that outlives it, and
the lifecycle events that prove the deletion.
