"""Read a source from S3 and write result tables to S3, from the embedded engine.

The embedded engine speaks the same storage URLs a server does. A source
registered at an `s3://` URL is read through the S3 driver, and a `[storage]
result_root` on `s3://` puts every result table the session writes — its
Parquet and its ANN index segments — in the bucket, not on local disk.

This recipe serves S3 locally from `moto` (an S3-compatible server, installed
by the cookbook's `cloud` extra) and runs one program twice: once with the
corpus and the result root on local disk, then once with both in a bucket. The
local run is the golden: the S3 run must return the same rows with the same
scores, and its result table's objects must be in the bucket. Against real
S3, drop the endpoint and `allow_http` lines and let the SDK's credential
chain (env vars, instance profile, …) supply the keys.

Run with `python cookbook/recipes/cloud_storage/example.py`.
"""

from __future__ import annotations

import logging
import os
import socket
import tempfile
from pathlib import Path

os.environ.setdefault("JAMMI_GPU__DEVICE", "-1")
os.environ.setdefault("JAMMI_ENGINE__BATCH_SIZE", "8")

import boto3
from moto.server import ThreadedMotoServer

import jammi
from jammi_cookbook import fixtures

CORPUS = fixtures.path("tiny_corpus.parquet")
MODEL = fixtures.model("tiny_bert")
BUCKET = "jammi-cookbook"
QUERY = "how does quantum computing work?"


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def search_rows(target_dir: str, *, corpus_url: str, config: Path | None) -> list[tuple]:
    """Register the corpus at `corpus_url`, embed it, search it: the ranked
    `(_row_id, similarity)` pairs."""
    with jammi.connect(f"file://{target_dir}", config=str(config) if config else None) as db:
        db.add_source("corpus", url=corpus_url, format="parquet")
        db.generate_embeddings(
            source="corpus", model=MODEL, columns=["content"], key="id", modality="text"
        )
        query = db.encode_query(model=MODEL, query=QUERY)
        hits = db.search("corpus", query=query, k=5).to_pylist()
        return [(row["_row_id"], round(row["similarity"], 6)) for row in hits]


def main() -> int:
    port = free_port()
    endpoint = f"http://127.0.0.1:{port}"
    # The S3 driver reads its keys from the environment, as against real S3.
    os.environ.update(
        AWS_ACCESS_KEY_ID="cookbook", AWS_SECRET_ACCESS_KEY="cookbook", AWS_REGION="us-east-1"
    )
    # The local S3 server logs every request; the recipe's own output is the story.
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=port)
    server.start()
    try:
        s3 = boto3.client("s3", endpoint_url=endpoint, region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        s3.upload_file(str(CORPUS), BUCKET, "sources/tiny_corpus.parquet")

        with tempfile.TemporaryDirectory() as local, tempfile.TemporaryDirectory() as cloud:
            golden = search_rows(local, corpus_url=str(CORPUS), config=None)

            # The result root and the S3 endpoint are deployment configuration:
            # the same `[storage]` section a server reads.
            config = Path(cloud) / "jammi.toml"
            config.write_text(
                f'[storage]\nresult_root = "s3://{BUCKET}/results"\n\n'
                f'[storage.cloud.s3]\nregion = "us-east-1"\nendpoint = "{endpoint}"\n'
                "allow_http = true\n"
            )
            from_s3 = search_rows(
                cloud, corpus_url=f"s3://{BUCKET}/sources/tiny_corpus.parquet", config=config
            )

        objects = [o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET)["Contents"]]
    finally:
        server.stop()

    results = [key for key in objects if key.startswith("results/")]
    print(f"top-5 over local disk: {golden}")
    print(f"top-5 over s3://{BUCKET}: {from_s3}")
    print(f"result-table objects in the bucket: {len(results)}")
    for key in sorted(results):
        print(f"  s3://{BUCKET}/{key}")

    assert from_s3 == golden, f"the S3 run ranked differently: {from_s3} vs {golden}"
    assert any(key.endswith(".parquet") for key in results), results
    assert any(key.endswith(".usearch") for key in results), results
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
