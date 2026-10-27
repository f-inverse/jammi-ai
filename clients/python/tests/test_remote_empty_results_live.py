"""A result with no rows keeps its columns, on both transports.

`sql`, `search` and `lexical_search` return a `pyarrow.Table` whose schema is
the result's — its column names and types — whether or not any row matched:
a filter that matches nothing, `LIMIT 0`, a `GROUP BY` or `ORDER BY` over no
rows, and a search with no hits each answer zero rows of the same columns the
matching query answers. The embedded engine and a real `jammi-server` give the
same schemas.

Selected by the `live_server` and `embedded` markers: it needs a built
`jammi-server` (`JAMMI_SERVER_BIN`) and the in-process engine as the parity peer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import jammi

pytestmark = [pytest.mark.live_server, pytest.mark.embedded]

REPO = Path(__file__).resolve().parents[3]
PATENTS_URL = f"file://{REPO / 'tests' / 'fixtures' / 'patents.parquet'}"
TINY_BERT = f"local:{REPO / 'tests' / 'fixtures' / 'tiny_bert'}"

# Each empty query beside the query whose columns it must keep.
SQL = {
    "where_false": (
        "SELECT id, title, year FROM patents.public.patents WHERE 1 = 0",
        "SELECT id, title, year FROM patents.public.patents",
    ),
    "limit_0": (
        "SELECT id, title FROM patents.public.patents LIMIT 0",
        "SELECT id, title FROM patents.public.patents",
    ),
    "no_match": (
        "SELECT id, year FROM patents.public.patents WHERE year > 3000",
        "SELECT id, year FROM patents.public.patents",
    ),
    "group_by": (
        "SELECT year, COUNT(*) AS n FROM patents.public.patents WHERE year > 3000 GROUP BY year",
        "SELECT year, COUNT(*) AS n FROM patents.public.patents GROUP BY year",
    ),
    "order_by": (
        "SELECT id, year FROM patents.public.patents WHERE year > 3000 ORDER BY year",
        "SELECT id, year FROM patents.public.patents ORDER BY year",
    ),
}


def _schema(table) -> list[tuple[str, str]]:
    return [(field.name, str(field.type)) for field in table.schema]


def _observe(db) -> dict:
    """Each empty result's row count and schema, beside its matching result's schema."""
    db.add_source("patents", url=PATENTS_URL, format="parquet")
    db.generate_embeddings(source="patents", model=TINY_BERT, columns=["title"], key="id")
    db.build_lexical_index("patents", columns=["title", "abstract"], key="id")
    some_key = str(db.sql("SELECT id FROM patents.public.patents LIMIT 1").column("id")[0].as_py())

    seen = {}
    for name, (empty, full) in SQL.items():
        result = db.sql(empty)
        seen[name] = (result.num_rows, _schema(result), _schema(db.sql(full)))
    searches = {
        "search": lambda flt: db.search(
            "patents", row_key=some_key, k=5, filter=flt, select=["id", "year"]
        ),
        "lexical_search": lambda flt: db.lexical_search(
            "patents", text="quantum", k=5, filter=flt, select=["id", "year"]
        ),
    }
    for name, run in searches.items():
        result = run("year > 3000")
        seen[name] = (result.num_rows, _schema(result), _schema(run(None)))
    seen["lexical_search_no_term"] = (
        lambda r: (r.num_rows, _schema(r), _schema(searches["lexical_search"](None)))
    )(db.lexical_search("patents", text="zzqxj", k=5, select=["id", "year"]))
    return seen


def test_an_empty_result_keeps_its_schema_on_both_transports(tmp_path, live_server_on):
    embedded_dir = tmp_path / "embedded"
    embedded_dir.mkdir()
    db = jammi.connect(f"file://{embedded_dir}")
    try:
        embedded = _observe(db)
    finally:
        db.close()

    served_dir = tmp_path / "served"
    served_dir.mkdir()
    with live_server_on(served_dir) as endpoint:
        remote = jammi.connect(endpoint)
        try:
            served = _observe(remote)
        finally:
            remote.close()

    for name, (rows, schema, matching) in embedded.items():
        assert rows == 0, (name, rows)
        assert schema == matching, (name, schema, matching)
    assert served == embedded
