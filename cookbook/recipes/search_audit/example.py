"""Per-query search audit — log, fetch, verify, query, stream.

Records a tamper-evident audit row for a search invocation: what was queried,
with what model, what came back, and when. The substrate signs each record
(HMAC-SHA256, per-tenant key), stores it tenant-scoped in the reserved
`_jammi_search_audit` table, and publishes it to the `jammi.audit.search.v1`
trigger topic.

The audit master key is required. Generate one with:

    export JAMMI_AUDIT_MASTER_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")

Run from the repo root:  python cookbook/recipes/search_audit/example.py
"""

from __future__ import annotations

import os
import tempfile
import uuid

import jammi_ai

TENANT = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a"


def main() -> None:
    if not os.environ.get("JAMMI_AUDIT_MASTER_KEY"):
        # The substrate refuses to sign without a master key. For this runnable
        # demo we set a fixed key if the operator has not provided one; in
        # production this comes from your secret manager, never hard-coded.
        os.environ["JAMMI_AUDIT_MASTER_KEY"] = "00" * 32

    with tempfile.TemporaryDirectory() as tmp:
        db = jammi_ai.connect(artifact_dir=tmp)
        db.with_tenant(TENANT)

        # 1. Build an audit record for a (hypothetical) search. query_lineage
        #    holds hashes / IDs — never raw payloads (there is a size cap).
        query_id = str(uuid.uuid4())
        record = jammi_ai.PerQueryAudit(
            query_id=query_id,
            model_id="patentclip/PatentCLIP_Vit_B",
            model_version="3aa649a",
            query_lineage={
                "image_hashes": ["sha256:9f86d0...", "sha256:2c26b4..."],
                "examiner_id": "12345",
                "lasso_bbox": [120, 80, 480, 360],
            },
            top_k_result_ids=["app-35528617", "app-35528702"],
            retrieval_scores=[0.92, 0.88],
        )

        # 2. Log it. The substrate injects tenant_id, signs, stores, publishes.
        db.audit.log([record])

        # 3. Fetch it back as a typed record and verify the signature.
        fetched = db.audit.fetch_by_query_id(query_id)
        assert fetched is not None, "record should be retrievable"
        assert fetched.tenant_id == TENANT
        assert fetched.signature, "record must carry a signature"
        fetched.verify()  # raises if tampered or key mismatch
        print(f"verified audit record for query {fetched.query_id}")

        # 4. The most-recent view.
        recent = db.audit.fetch_recent(limit=10)
        assert len(recent) == 1

        # 5. It is also queryable as plain SQL (tenant scope auto-applied).
        table = db.sql(
            'SELECT model_id, model_version FROM mutable.public."_jammi_search_audit"'
        )
        assert table.num_rows == 1, "SQL view returns the tenant's audit rows"
        print("model_id via SQL:", table.column("model_id")[0].as_py())

        # 6. Each logged record lands on the audit trigger topic. A subscriber
        #    (alerting, analytics, warehouse sink) reads the JSON payloads.
        #    from_offset=0 replays the durable backing table and returns
        #    promptly once caught up, rather than blocking on the live tail.
        delivered = db.subscribe_collect(
            "jammi.audit.search.v1", from_offset=0, max_batches=1
        )
        assert delivered.num_rows >= 1, "subscriber receives the audit payload"
        print("audit topic delivered", delivered.num_rows, "row(s)")

        print("search_audit recipe OK")


if __name__ == "__main__":
    main()
