# Per-query search audit

Record a tamper-evident audit row for every search: *what was queried, with
what model, what came back, and when.* The substrate signs each record, stores
it tenant-scoped, and publishes it to a trigger topic — so you do not hand-roll
an audit schema, a signature scheme, and a stream integration in every project.

This is the primitive every audited-ML deployment (financial, healthcare,
federal, legal) needs to answer "show me exactly what this model returned for
this query, and prove the record hasn't been altered."

## What this recipe shows

- Build a `PerQueryAudit` record (query id, model id/version, query lineage,
  top-K result ids, retrieval scores).
- `db.audit.log([...])` — the substrate injects `tenant_id`, signs the record
  with a per-tenant HMAC-SHA256 key, stores it, and publishes it.
- `db.audit.fetch_by_query_id(...)` / `db.audit.fetch_recent(...)` — typed reads,
  tenant-scoped.
- `record.verify()` — re-derive the key and check the signature.
- Plain SQL over `mutable.public."_jammi_search_audit"` — same tenant scope.
- `db.subscribe_collect("jammi.audit.search.v1", ...)` — every logged record is
  also delivered on a trigger topic for alerting / analytics / warehouse sinks.

## Run it

The audit master key is required — the substrate refuses to sign without it:

```bash
export JAMMI_AUDIT_MASTER_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
python cookbook/recipes/search_audit/example.py
```

The key derives a distinct signing secret per tenant via HKDF-SHA256 and is
deterministic across restarts, so signatures written today verify after a
redeploy. Source it from your secret manager — never hard-code it.

## Key points

- **Lineage is capped.** `query_lineage` JSON may not exceed 8 KiB (override with
  `JAMMI_AUDIT_MAX_LINEAGE_BYTES`). Store image hashes and row IDs, not raw
  payloads — compliance posture is structural, not advisory.
- **`top_k_result_ids` and `retrieval_scores` must be the same length.** This is
  checked when you construct the record.
- **The table is reserved.** `_jammi_search_audit` is created implicitly on the
  first log; you cannot create or directly `INSERT` into it (that would bypass
  signing). Read it freely via SQL.
- **Tenant isolation is automatic.** A record logged under tenant A is invisible
  to tenant B, through both the typed API and raw SQL.
