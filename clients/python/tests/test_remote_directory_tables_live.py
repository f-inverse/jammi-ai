"""A directory registers as one table, on both transports.

A directory's files are the parts of one table, so a directory whose files do
not agree on that table's columns is refused when it is registered, naming two
of the files, rather than read as the union of every file's columns with NULLs
wherever a file lacks one (#755). A Parquet partition that adds a column is
still a part of the table; CSV files, whose columns are read by position, must
share one header.

Selected by the `live_server` and `embedded` markers: it needs a built
`jammi-server` (`JAMMI_SERVER_BIN`) and the in-process engine.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import jammi
from jammi.errors import InvalidArgument

pytestmark = [pytest.mark.live_server, pytest.mark.embedded]


def _folder(root, name):
    folder = root / name
    folder.mkdir()
    return folder


@pytest.mark.parametrize("transport", ["embedded", "remote"])
def test_a_directory_registers_as_one_table(transport, tmp_path, live_server):
    mixed = _folder(tmp_path, "mixed")
    pq.write_table(
        pa.table({"order_id": [1, 2, 3], "amount": [10.0, 20.0, 30.0]}),
        mixed / "orders.parquet",
    )
    pq.write_table(
        pa.table({"customer_id": [7, 8], "city": ["Lisbon", "Porto"]}),
        mixed / "customers.parquet",
    )
    people = _folder(tmp_path, "people")
    (people / "a.csv").write_text("first,last\nAda,Lovelace\n")
    (people / "b.csv").write_text("last,first\nHopper,Grace\n")
    grown = _folder(tmp_path, "grown")
    pq.write_table(pa.table({"id": [1], "amount": [1.0]}), grown / "part-0.parquet")
    pq.write_table(
        pa.table({"id": [2], "amount": [2.0], "tax": [0.2]}), grown / "part-1.parquet"
    )

    if transport == "embedded":
        db = jammi.connect(f"file://{tmp_path / 'engine'}")
    else:
        db = jammi.connect(live_server)
    try:
        for folder, fmt, files in [
            (mixed, "parquet", ["orders.parquet", "customers.parquet"]),
            (people, "csv", ["a.csv", "b.csv"]),
        ]:
            with pytest.raises(InvalidArgument) as refused:
                db.add_source(f"{folder.name}_{transport}", url=f"file://{folder}", format=fmt)
            for file in files:
                assert file in str(refused.value), str(refused.value)

        db.add_source(f"grown_{transport}", url=f"file://{grown}", format="parquet")
        rows = db.sql(
            f"SELECT id, amount, tax FROM grown_{transport}.public.grown ORDER BY id"
        ).to_pylist()
        assert rows == [
            {"id": 1, "amount": 1.0, "tax": None},
            {"id": 2, "amount": 2.0, "tax": 0.2},
        ]
    finally:
        db.close()
