# Mutable companion tables

End-to-end create / insert / select / drop on a Jammi mutable table —
the OSS primitive for state that needs to live alongside read-only result
tables.

**When to use this pattern.** You need a writable table that sits in the
same SQL catalog as your registered sources and embedding tables — for
caching enriched rows, holding cursor state, recording user feedback, or
any "small table I want to UPDATE / DELETE / INSERT from SQL" workload —
without standing up an external Postgres.

## API surface exercised

- `Session.create_mutable_table(name, *, schema, primary_key, ...)`
- `Session.sql("INSERT INTO mutable.public.<name> ...")`
- `Session.sql("SELECT ... FROM mutable.public.<name>")`
- `Session.drop_mutable_table(name, *, if_exists=False)`

The DataFusion namespace for mutable tables is always
`mutable.public.<name>` — distinct from registered sources, which live
under `<source>.public.<table>`, the table named after the file.

## Run it

```bash
python cookbook/recipes/mutable_tables/example.py
```

It prints the three rows it wrote, then the error SQL raises once the table is gone.
