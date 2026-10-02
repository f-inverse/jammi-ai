"""Embed and search with a model served at a remote endpoint.

Run with `python cookbook/recipes/remote_model/example.py`, or a step at a
time as a notebook: each `# %%` cell is one step.
"""

# %% [markdown]
# A model the engine does not run itself — a hosted embeddings API, or an
# inference server on another machine — is declared once in the deployment's
# configuration and named `remote:<name>` in every verb a local model is used
# in.

# %%
import hashlib
import json
import math
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jammi
from jammi_cookbook import fixtures

os.environ.setdefault("JAMMI_GPU__DEVICE", "-1")

DIMS = 32
TOKEN = "local-demo-token"

# %% [markdown]
# ## A stand-in endpoint
#
# This program serves its own endpoint, speaking the OpenAI-compatible
# embeddings protocol behind a bearer token: a hashed bag-of-words embedder,
# where texts that share words point the same way. It runs with no network and
# no key; in production, the declaration below points at a hosted API instead.

# %%
def bag_of_words(text: str) -> list[float]:
    """A unit vector with one bucket per hashed word."""
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


server = ThreadingHTTPServer(("127.0.0.1", 0), EmbeddingsEndpoint)
threading.Thread(target=server.serve_forever, daemon=True).start()

# %% [markdown]
# ## Declare the model
#
# A `[models.remote.<name>]` section names the endpoint's URL and protocol,
# the model to ask it for, its output width, a revision pin, and the headers
# to send — credentials inline or `{ file = "…" }`. The credential never
# reaches the catalog or a result table: a table records its model run as the
# URL, the model name, the width and the revision.

# %%
home = Path(tempfile.mkdtemp())
config = home / "jammi.toml"
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
MODEL = "remote:bag-of-words"

db = jammi.connect(f"file://{home}/data", config=str(config))
db.add_source("corpus", url=str(fixtures.path("tiny_corpus.parquet")), format="parquet")

# %% [markdown]
# ## Embed through the endpoint, and search
#
# `generate_embeddings` sends the rows to the endpoint in chunks, at most
# `max_in_flight` requests at a time. The query goes to the same endpoint, so
# it lands in the table's space.

# %%
db.generate_embeddings(source="corpus", model=MODEL, columns=["content"], key="id", modality="text")

query = db.encode_query(model=MODEL, query="quantum computing error correction")
rows = db.search("corpus", query=query, k=3).to_pylist()
assert rows, "the search returned no rows"
for row in rows:
    print(f"{row['_row_id']:<8}  {row['similarity']:>9.4f}  {row['title']}")

# %%
db.close()
server.shutdown()
