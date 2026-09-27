"""A source's tenant column scopes every verb that reads it, on both transports.

One shared file carries its tenant discriminator under `workspace`, not
`tenant_id`. Registered once, globally, with `tenant_column="workspace"`, it
serves each tenant exactly its own rows and the rows with no tenant — through
`sql`, through the embedding table `generate_embeddings` builds, and through
`search` over that table. The embedded engine and a real `jammi-server` give
the same answers, and both refuse a declaration naming a column the source
lacks, or competing with the source's own `tenant_id` column, as
`InvalidArgument`.

Selected by the `live_server` and `embedded` markers: it needs a built
`jammi-server` (`JAMMI_SERVER_BIN`) and the in-process engine as the parity peer.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import jammi
from jammi.errors import InvalidArgument

pytestmark = [pytest.mark.live_server, pytest.mark.embedded]

REPO = Path(__file__).resolve().parents[3]
TINY_BERT = f"local:{REPO / 'cookbook' / 'fixtures' / 'tiny_bert'}"

TENANT_A = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a"
TENANT_B = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9b"
A_IDS = [f"a{i}" for i in range(6)]
B_IDS = [f"b{i}" for i in range(4)]
SHARED_IDS = ["s0"]


def _write_notes(path: Path, *, with_tenant_id: bool = False) -> None:
    ids = A_IDS + B_IDS + SHARED_IDS
    columns = {
        "id": ids,
        "title": [f"note {i} about graph search and protein folding" for i in ids],
        "workspace": [TENANT_A] * len(A_IDS) + [TENANT_B] * len(B_IDS) + [None],
    }
    if with_tenant_id:
        columns["tenant_id"] = columns["workspace"]
    pq.write_table(pa.table(columns), path)


def _observe(db, url: str) -> dict:
    """What each tenant reads of the shared source, through each verb."""
    db.add_source("notes", url=url, format="parquet", tenant_column="workspace")
    seen = {}
    for tenant in (TENANT_A, TENANT_B):
        with db.tenant_scope(tenant):
            rows = db.sql("SELECT id FROM notes.public.notes").column("id").to_pylist()
            table = db.generate_embeddings(
                source="notes", model=TINY_BERT, columns=["title"], key="id"
            )
            embedded_ids = (
                db.sql(f'SELECT _row_id FROM "jammi.{table}"').column("_row_id").to_pylist()
            )
            hits = db.search("notes", row_key=sorted(rows)[0], k=50, select=["id"])
            seen[tenant] = {
                "sql": sorted(rows),
                "embedded": sorted(embedded_ids),
                "search": sorted(hits.column("id").to_pylist()),
            }
    return seen


def _refusals(db, missing_url: str, competing_url: str) -> list[str]:
    """The class each misdeclaration raises."""
    raised = []
    for name, url in (("missing", missing_url), ("competing", competing_url)):
        with pytest.raises(InvalidArgument) as refused:
            db.add_source(name, url=url, format="parquet", tenant_column="team")
        raised.append(type(refused.value).__name__)
    return raised


def test_a_tenant_column_scopes_every_verb_on_both_transports(tmp_path, live_server_on):
    notes = tmp_path / "notes.parquet"
    _write_notes(notes)
    missing = tmp_path / "missing.parquet"
    _write_notes(missing)
    competing = tmp_path / "competing.parquet"
    _write_notes(competing, with_tenant_id=True)
    pq.write_table(
        pq.read_table(competing).append_column("team", pa.array(["x"] * 11)), competing
    )

    embedded_dir = tmp_path / "embedded"
    embedded_dir.mkdir()
    db = jammi.connect(f"file://{embedded_dir}")
    try:
        embedded = _observe(db, f"file://{notes}")
        embedded_refusals = _refusals(db, f"file://{missing}", f"file://{competing}")
    finally:
        db.close()

    served_dir = tmp_path / "served"
    served_dir.mkdir()
    with live_server_on(served_dir) as endpoint:
        remote = jammi.connect(endpoint)
        try:
            served = _observe(remote, f"file://{notes}")
            served_refusals = _refusals(remote, f"file://{missing}", f"file://{competing}")
        finally:
            remote.close()

    expected = {
        TENANT_A: sorted(A_IDS + SHARED_IDS),
        TENANT_B: sorted(B_IDS + SHARED_IDS),
    }
    for tenant, own in expected.items():
        for verb in ("sql", "embedded", "search"):
            assert embedded[tenant][verb] == own, (tenant, verb, embedded[tenant][verb])
    assert served == embedded
    assert embedded_refusals == served_refusals == ["InvalidArgument", "InvalidArgument"]
