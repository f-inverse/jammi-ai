"""Create, insert into, query and drop a Jammi mutable companion table.

Run with `python cookbook/recipes/mutable_tables/example.py`, or a step at a
time as a notebook: each `# %%` cell is one step.
"""

# %%
import tempfile

import pyarrow as pa

import jammi

db = jammi.connect(f"file://{tempfile.mkdtemp()}")

# %% [markdown]
# ## Create the table
#
# A mutable table takes an Arrow schema and a primary key. Once created, the
# catalog resolves it as `mutable.public.notes` for SQL reads and DML.

# %%
schema = pa.schema(
    [
        pa.field("note_id", pa.int64(), nullable=False),
        pa.field("body", pa.string(), nullable=False),
    ]
)
table_id = db.create_mutable_table("notes", schema=schema, primary_key=["note_id"])
assert table_id == "notes", f"expected 'notes', got {table_id}"

# %% [markdown]
# ## Write and read it with SQL
#
# Rows go in through SQL's `INSERT`, and come back through `SELECT` like any
# other table's.

# %%
db.sql("INSERT INTO mutable.public.notes (note_id, body) VALUES (1, 'one')")
db.sql("INSERT INTO mutable.public.notes (note_id, body) VALUES (2, 'two')")
db.sql("INSERT INTO mutable.public.notes (note_id, body) VALUES (3, 'three')")

count = db.sql("SELECT COUNT(*) AS n FROM mutable.public.notes").column("n").to_pylist()[0]
assert count == 3, f"expected 3 rows, got {count}"

bodies = db.sql("SELECT body FROM mutable.public.notes ORDER BY note_id").column("body").to_pylist()
assert bodies == ["one", "two", "three"], f"unexpected rows {bodies}"
print(bodies)

# %% [markdown]
# ## Drop it
#
# After `drop_mutable_table`, SQL no longer resolves the name. With
# `if_exists=True`, dropping a table that is already gone does not raise.

# %%
db.drop_mutable_table("notes")
try:
    db.sql("SELECT COUNT(*) FROM mutable.public.notes")
except RuntimeError as dropped:
    print(f"after the drop: {dropped}")
else:
    raise AssertionError("post-drop SELECT must raise")

db.drop_mutable_table("notes", if_exists=True)

# %%
db.close()
