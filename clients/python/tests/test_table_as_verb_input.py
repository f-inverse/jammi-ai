"""A table made with SQL is a verb's input, on both transports (#762).

A verb reads a registered source, or, under a name no source is registered by,
a table the engine holds, named as SQL reads it: `jammi.<name>` for a result
table. So a `CREATE TABLE … AS` table is indexed and searched without its rows
leaving the engine.

Selected by the `live_server` and `embedded` markers: it needs a built
`jammi-server` (`JAMMI_SERVER_BIN`) and the in-process engine.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

import jammi

pytestmark = [pytest.mark.live_server, pytest.mark.embedded]

REPO = Path(__file__).resolve().parents[3]
PATENTS_URL = f"file://{REPO / 'tests' / 'fixtures' / 'patents.parquet'}"


@pytest.mark.parametrize("transport", ["embedded", "remote"])
def test_a_table_made_by_sql_is_indexed_and_searched(transport, tmp_path, live_server):
    if transport == "embedded":
        db = jammi.connect(f"file://{tmp_path / 'engine'}")
    else:
        db = jammi.connect(live_server)
    suffix = uuid.uuid4().hex[:8]
    source, table = f"patents_{suffix}", f"recent_{suffix}"
    try:
        db.add_source(source, url=PATENTS_URL, format="parquet")
        db.sql(
            f"CREATE TABLE {table} AS SELECT id, title, abstract "
            f"FROM {source}.public.patents WHERE year >= 2022"
        )
        recent = set(db.sql(f'SELECT id FROM "jammi.{table}"').column("id").to_pylist())
        db.build_lexical_index(f"jammi.{table}", columns=["title", "abstract"], key="id")
        found = db.lexical_search(f"jammi.{table}", text="quantum", k=20)
        ids = found.column("id").to_pylist()
        assert ids and set(ids) <= recent, (ids, recent)
    finally:
        db.close()
