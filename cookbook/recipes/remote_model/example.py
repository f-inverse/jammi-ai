"""Embed and search with a model served at a remote endpoint.

A model the engine does not run itself — a hosted embeddings API, or an
inference server on another machine — is declared once in the deployment's
configuration and referenced as `remote:<name>`, in every verb a local model
is used in:

1. declare the endpoint under `[models.remote.<name>]` in `jammi.toml`
   (URL, the model name to ask for, its output width, a revision pin, and
   headers — credentials inline or `{ file = "…" }`)
2. `db.generate_embeddings(..., model="remote:<name>")` — the plan sends the
   rows to the endpoint in chunks, at most `max_in_flight` requests at once
3. `db.encode_query(model="remote:<name>", ...)` + `db.search(...)`

The endpoint here is a stand-in served by this script, speaking the
OpenAI-compatible embeddings protocol: a hashed bag-of-words embedder, so
the recipe runs with no network and no key. Point `url` at a hosted API (and
put its key in `headers`) for production.

Run from the repo root:  python cookbook/recipes/remote_model/example.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("JAMMI_GPU__DEVICE", "-1")

import jammi
from jammi_cookbook import fixtures

CORPUS_PATH = fixtures.path("tiny_corpus.parquet")
DIMS = 32
TOKEN = "local-demo-token"


def bag_of_words(text: str) -> list[float]:
    """A unit vector with one bucket per hashed word: texts sharing words
    point the same way."""
    vector = [0.0] * DIMS
    for word in text.lower().split():
        bucket = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") % DIMS
        vector[bucket] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class EmbeddingsEndpoint(BaseHTTPRequestHandler):
    """`POST /v1/embeddings`, OpenAI-compatible, behind a bearer token."""

    def do_POST(self) -> None:  # noqa: N802 — the handler name http.server calls
        if self.headers.get("Authorization") != f"Bearer {TOKEN}":
            self.send_error(401, "invalid token")
            return
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body = json.dumps(
            {
                "object": "list",
                "model": request["model"],
                "data": [
                    {"object": "embedding", "index": i, "embedding": bag_of_words(text)}
                    for i, text in enumerate(request["input"])
                ],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), EmbeddingsEndpoint)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            # 1. Declare the remote model. The credential never reaches the
            #    catalog or a result table: what a table records as its model
            #    run is the URL, the model name, the width and the revision.
            config = Path(tmp) / "jammi.toml"
            config.write_text(
                f"""
[models.remote.bag-of-words]
protocol = "openai_embeddings"
url = "http://127.0.0.1:{server.server_port}/v1/embeddings"
model = "bag-of-words-32"
dimensions = {DIMS}
revision = "demo-1"
headers = {{ Authorization = "Bearer {TOKEN}" }}
"""
            )
            model = "remote:bag-of-words"
            with jammi.connect(f"file://{tmp}/data", config=str(config)) as db:
                db.add_source("corpus", url=str(CORPUS_PATH), format="parquet")

                # 2. Embed the corpus through the endpoint.
                db.generate_embeddings(
                    source="corpus",
                    model=model,
                    columns=["content"],
                    key="id",
                    modality="text",
                )

                # 3. The query goes to the same endpoint, so it lands in the
                #    same space as the table.
                query = db.encode_query(model=model, query="quantum computing error correction")
                rows = db.search("corpus", query=query, k=3).to_pylist()
                if not rows:
                    raise RuntimeError("the search returned no rows")

                print("id        similarity  title")
                for row in rows:
                    print(f"{row['_row_id']:<8}  {row['similarity']:>9.4f}  {row['title']}")
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
