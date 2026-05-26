# 3. Register a source

```python
db.add_source(
    "corpus",
    url="cookbook/fixtures/tiny_corpus.parquet",
    format="parquet",
)
```

`add_source` registers a file with the DataFusion catalog so SQL queries
and embedding jobs can reference it by name. The `url` argument accepts:

- a local filesystem path (parsed into `file://...` automatically)
- `s3://bucket/key`, `gs://bucket/key`, `azure://container/blob` for cloud
  object storage (when the wheel was built with the relevant feature)

`format` is one of `parquet`, `csv`, or `json`.

## What you can do once registered

```python
# SQL — returns a pyarrow.Table
table = db.sql("SELECT id, title FROM corpus.public.corpus LIMIT 3")
print(table.to_pandas())
```

The fully-qualified name follows DataFusion's `<catalog>.<schema>.<table>`
shape; for OSS Jammi the catalog and schema are always
`<name>.public.<name>`.

## Tiny corpus shape

The quickstart fixture is 20 rows wide with these columns:

| column        | type   | example                                                 |
|---------------|--------|---------------------------------------------------------|
| `id`          | int64  | `1`                                                     |
| `title`       | utf8   | `Quantum error correction in superconducting qubits`    |
| `content`     | utf8   | `We present a novel approach to quantum error...`       |
| `year`        | int64  | `2021`                                                  |
| `category`    | utf8   | `physics`                                               |
| `assignee_id` | int64  | `101`                                                   |

Next: [build embeddings and search](./04_vector_search.md).
