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
import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jammi
from jammi_cookbook import fixtures

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
DIMS = 384
TOKEN = "local-demo-token"

# %% [markdown]
# ## An endpoint to call
#
# This program serves its own endpoint, speaking the OpenAI-compatible
# embeddings protocol behind a bearer token, and backed by a real sentence
# encoder (here, a Jammi session of its own running `all-MiniLM-L6-v2`). It
# needs no account and no key; in production, the declaration below points at a
# hosted API or an inference server instead, and nothing else changes.

# %%
encoder = jammi.connect(f"file://{tempfile.mkdtemp()}")
encoder_lock = threading.Lock()


class EmbeddingsEndpoint(BaseHTTPRequestHandler):
    """`POST /v1/embeddings`, OpenAI-compatible, behind a bearer token."""

    def do_POST(self) -> None:  # noqa: N802 — the handler name http.server calls
        if self.headers.get("Authorization") != f"Bearer {TOKEN}":
            self.send_error(401, "invalid token")
            return
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with encoder_lock:
            vectors = [encoder.encode_query(model=ENCODER, query=text) for text in request["input"]]
        body = json.dumps(
            {
                "object": "list",
                "model": request["model"],
                "data": [
                    {"object": "embedding", "index": i, "embedding": vector}
                    for i, vector in enumerate(vectors)
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
[models.remote.sentence-encoder]
protocol = "openai_embeddings"
url = "http://127.0.0.1:{server.server_port}/v1/embeddings"
model = "all-MiniLM-L6-v2"
dimensions = {DIMS}
revision = "demo-1"
headers = {{ Authorization = "Bearer {TOKEN}" }}
"""
)
MODEL = "remote:sentence-encoder"

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
for row in rows:
    print(f"{row['_row_id']:<8}  {row['similarity']:>9.4f}  {row['category']:<8} {row['title']}")
assert rows[0]["category"] == "physics", "the endpoint's encoder places the query among the quantum papers"

# %%
db.close()
server.shutdown()
encoder.close()
