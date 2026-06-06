# Compound Retrieval and Inference over Flight SQL

`search` is the bounded primitive — nearest-neighbor top-k, returning a table
directly. **Compound query** — joining sources, filtering, and running a model
over a relation — is open, caller-shaped composition, so it rides **SQL**. The
same SQL runs in-process on the embedded engine and over the **Flight SQL** lane
against a remote engine; the `annotate(...)` table function makes model inference
available inside that SQL on both.

This is what lets a remote caller do **search → join → annotate** in one
round-trip, with the model running inside the engine — no per-row RPC, no bespoke
compound-search verb.

## The `annotate` table function

```text
annotate(model, task, relation, key_column, content_column [, content_column…])
```

It runs `model` (a `local:<path>`, an HF repo id, or a fine-tuned id) for `task`
over the named `relation`'s `content_column`(s), and returns the inference output:
the prefix `_row_id` / `_source` / `_model` / `_status` / `_error` /
`_latency_ms` — with `_row_id` carried from `key_column` — followed by the task's
columns (e.g. a `vector` FixedSizeList for an embedding task). Join it back to the
source on `_row_id` to place inference columns alongside source columns.

## Remote: `jammi-client` over Flight SQL

```python
import jammi_client

db = jammi_client.connect("grpc://engine.internal:8081")

# Compound retrieval + inference in one Flight SQL round-trip:
table = db.sql("""
    SELECT p.title, a.vector
    FROM annotate('sentence-transformers/all-MiniLM-L6-v2', 'text_embedding',
                  'patents.public.patents', 'id', 'abstract') AS a
    JOIN patents.public.patents AS p ON a._row_id = arrow_cast(p.id, 'Utf8')
    WHERE p.year >= 2020
""")
# table is a pyarrow.Table
```

`db.sql` carries the connection's tenant scope (the same `jammi-session-id` the
typed gRPC verbs use), so SQL reads observe the same tenant as `db.search`.

## Embedded: the same SQL, in-process

The embed wheel runs the identical SQL against its in-process DataFusion engine —
the `annotate` function is registered on the same context:

```python
import jammi_ai

db = jammi_ai.connect("file:///var/lib/jammi")
table = db.sql("""
    SELECT a._row_id, a.vector
    FROM annotate('local:/models/all-MiniLM-L6-v2', 'text_embedding',
                  'patents.public.patents', 'id', 'abstract') AS a
""")
```

Productionising from the embed wheel to the remote client is the M2 one-line import
swap (`import jammi_ai` → `import jammi_client`); the `sql` call is unchanged.

## In-process Rust: the fluent builder

In Rust, `session.search(vec, k)` returns a `QueryBuilder` that composes the same
operations as a fluent chain (the `annotate` node it builds is the very plan node
the SQL table function builds):

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use std::sync::Arc;
# use jammi_ai::session::InferenceSession;
use jammi_db::ModelTask;
# async fn ex(session: &Arc<InferenceSession>, query: Vec<f32>) -> jammi_db::error::Result<()> {
let results = session.search("patents", query, 10).await?
    .annotate("local:/models/all-MiniLM-L6-v2", ModelTask::TextEmbedding, &["abstract".into()]).await?
    .run().await?;
# Ok(()) }
```

## Notes

- A `WHERE` over the annotated output runs **above** the inference node — the
  table function declares inference non-pushdown, since a model runs row-wise and
  a predicate can't push below it. Filter the *source* (inside the relation a join
  scans) when you want to shrink the input the model sees.
- The output schema is fixed at planning time; the embedding dimension is read by
  loading the model, which is then warm for execution.
- Classification and NER ride the same prefix + task-column shape; pass their
  `task` string (`'classification'`, `'ner'`) and the content column.
