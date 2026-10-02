"""Read a source from S3 and write result tables to S3, from the embedded engine.

Run with `python cookbook/recipes/cloud_storage/example.py`, or a step at a
time as a notebook: each `# %%` cell is one step. Needs the cookbook's `cloud`
extra (`pip install "jammi-cookbook[cloud]"`), which serves S3 locally.
"""

# %% [markdown]
# The embedded engine speaks the same storage URLs a server does. A source
# registered at an `s3://` URL is read through the S3 driver, and a
# `[storage] result_root` on `s3://` puts every result table the session
# writes, its Parquet and its ANN index segments, in the bucket instead of on
# local disk.
#
# This program serves S3 locally from `moto`, an S3-compatible server, and
# runs one search twice: over local disk, then with both the corpus and the
# results in a bucket. The local run is the reference the bucket run must
# match.

# %%
import logging
import os
import socket
import tempfile
from pathlib import Path

import boto3
from moto.server import ThreadedMotoServer

import jammi
from jammi_cookbook import fixtures

os.environ.setdefault("JAMMI_GPU__DEVICE", "-1")
os.environ.setdefault("JAMMI_ENGINE__BATCH_SIZE", "8")

CORPUS = fixtures.path("tiny_corpus.parquet")
MODEL = fixtures.model("tiny_bert")
BUCKET = "jammi-cookbook"
QUERY = "how does quantum computing work?"

# %% [markdown]
# ## Serve S3 locally and fill a bucket
#
# The S3 driver reads its keys from the environment, as it does against real
# S3. Against real S3, the SDK's credential chain (environment variables, an
# instance profile, …) supplies them.

# %%
with socket.socket() as probe:
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
endpoint = f"http://127.0.0.1:{port}"

os.environ.update(
    AWS_ACCESS_KEY_ID="cookbook", AWS_SECRET_ACCESS_KEY="cookbook", AWS_REGION="us-east-1"
)
# The local S3 server logs every request; this program's own output is the story.
logging.getLogger("werkzeug").setLevel(logging.ERROR)
server = ThreadedMotoServer(ip_address="127.0.0.1", port=port)
server.start()

s3 = boto3.client("s3", endpoint_url=endpoint, region_name="us-east-1")
s3.create_bucket(Bucket=BUCKET)
s3.upload_file(str(CORPUS), BUCKET, "sources/tiny_corpus.parquet")

# %% [markdown]
# ## One program, wherever its data lives
#
# Register the corpus at a URL, embed it, search it, and return the ranked
# `(_row_id, similarity)` pairs.

# %%
def search_rows(target_dir: str, *, corpus_url: str, config: Path | None) -> list[tuple]:
    with jammi.connect(f"file://{target_dir}", config=str(config) if config else None) as db:
        db.add_source("corpus", url=corpus_url, format="parquet")
        db.generate_embeddings(
            source="corpus", model=MODEL, columns=["content"], key="id", modality="text"
        )
        query = db.encode_query(model=MODEL, query=QUERY)
        hits = db.search("corpus", query=query, k=5).to_pylist()
        return [(row["_row_id"], round(row["similarity"], 6)) for row in hits]


golden = search_rows(tempfile.mkdtemp(), corpus_url=str(CORPUS), config=None)
print(f"top-5 over local disk: {golden}")

# %% [markdown]
# ## Run it against the bucket
#
# Where results go, and how to reach the store, is deployment configuration:
# the same `[storage]` section a server reads. Against real S3, drop the
# `endpoint` and `allow_http` lines.

# %%
cloud = Path(tempfile.mkdtemp())
config = cloud / "jammi.toml"
config.write_text(
    f'[storage]\nresult_root = "s3://{BUCKET}/results"\n\n'
    f'[storage.cloud.s3]\nregion = "us-east-1"\nendpoint = "{endpoint}"\n'
    "allow_http = true\n"
)
from_s3 = search_rows(
    str(cloud), corpus_url=f"s3://{BUCKET}/sources/tiny_corpus.parquet", config=config
)
print(f"top-5 over s3://{BUCKET}: {from_s3}")
assert from_s3 == golden, f"the S3 run ranked differently: {from_s3} vs {golden}"

# %% [markdown]
# ## The results are in the bucket
#
# The result table's Parquet and its index segments were written under the
# bucket's `results/` prefix.

# %%
objects = [o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET)["Contents"]]
server.stop()

results = sorted(key for key in objects if key.startswith("results/"))
for key in results:
    print(f"s3://{BUCKET}/{key}")
assert any(key.endswith(".parquet") for key in results), results
assert any(key.endswith(".usearch") for key in results), results
