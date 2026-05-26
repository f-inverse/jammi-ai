# Mutable companion tables

End-to-end create / insert / select / drop on a Jammi mutable table —
the OSS primitive for state that needs to live alongside read-only result
tables.

**When to use this pattern.** You need a writable table that sits in the
same SQL catalog as your registered sources and embedding tables — for
caching enriched rows, holding cursor state, recording user feedback, or
any "small table I want to UPDATE / DELETE / INSERT from SQL" workload —
without standing up an external Postgres.

## What `example.py` does

1. Connects to a temporary artifact dir
2. Creates a `notes` mutable table with an `int64` primary key + `utf8`
   body column
3. Inserts three rows through DataFusion DML (`INSERT INTO ...`)
4. Verifies count and ordering via `SELECT`
5. Drops the table, then asserts a `SELECT` after the drop raises
6. Demonstrates the idempotent `drop_mutable_table(..., if_exists=True)`

## API surface exercised

- `Database.create_mutable_table(name, *, schema, primary_key, ...)`
- `Database.sql("INSERT INTO mutable.public.<name> ...")`
- `Database.sql("SELECT ... FROM mutable.public.<name>")`
- `Database.drop_mutable_table(name, *, if_exists=False)`

The DataFusion namespace for mutable tables is always
`mutable.public.<name>` — distinct from registered sources, which live
under `<source>.public.<source>`.

## Run it

```bash
python cookbook/recipes/mutable_tables/example.py
```

Exits 0 on success, prints `mutable_tables: OK` on the last line.
