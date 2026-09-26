# Cloud object storage from the embedded engine

Read a source straight out of an S3 bucket and write every result table the
session produces — its Parquet and its ANN index segments — back into the
bucket, from `jammi.connect("file://…")`. The embedded engine speaks the same
`s3://` / `gs://` / `azure://` / `r2://` URLs a `jammi-server` does.

**When to use this pattern.** Your data already lives in object storage, or
the results must outlive the machine that computed them — a notebook runtime,
a batch job on a spot instance — without standing up a server.

## What `example.py` does

1. Serves S3 locally from `moto` (the cookbook's `cloud` extra) and uploads
   the tiny corpus into a bucket
2. Runs one program — register the corpus, embed it with the fixture
   `tiny_bert`, search it — against local disk: the golden
3. Runs the same program with the source at
   `s3://jammi-cookbook/sources/tiny_corpus.parquet` and a `[storage]`
   `result_root` of `s3://jammi-cookbook/results`
4. Asserts the S3 run returns the golden's rows with the golden's scores, and
   that the result table's Parquet and index segments are in the bucket

## The configuration

```toml
[storage]
result_root = "s3://jammi-cookbook/results"

[storage.cloud.s3]
region = "us-east-1"
endpoint = "http://127.0.0.1:<port>"  # a local S3-compatible server only
allow_http = true                     # a local S3-compatible server only
```

Pass it as `jammi.connect("file:///var/lib/jammi", config="jammi.toml")`. The
keys come from the environment (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`),
or from the SDK's credential chain against real S3. See
[Store Sources and Results in Cloud Object Storage](../../../docs/guide/src/cloud-storage.md).

## API surface exercised

- `jammi.connect("file://…", config=…)` with a `[storage]` section
- `Database.add_source(name, url="s3://…", format="parquet")`
- `Database.generate_embeddings(...)`, `encode_query(...)`, `search(...)`

## Run it

```bash
pip install -e 'cookbook/book[cloud]'
python cookbook/recipes/cloud_storage/example.py
```

Exits 0 on success.
