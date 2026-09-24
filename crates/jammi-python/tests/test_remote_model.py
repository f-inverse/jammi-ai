"""A model served at a remote endpoint, through the public Python front door.

The endpoint is served by a thread of this same process, so the test also
proves every engine call releases the GIL: a verb that held it while waiting
on the endpoint would wait on a thread that can never run.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import jammi

DIMS = 4
TOKEN = "test-token"


def vector_of(text: str) -> list[float]:
    """Length, then the first three code points: distinct per text."""
    values = [float(len(text))] + [float(ord(c)) for c in text[: DIMS - 1]]
    return values + [0.0] * (DIMS - len(values))


class Endpoint(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802
        if self.headers.get("Authorization") != f"Bearer {TOKEN}":
            self.send_error(401, "invalid token")
            return
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Endpoint.requests.append(request)
        body = json.dumps(
            {
                "data": [
                    {"index": i, "embedding": vector_of(text)}
                    for i, text in enumerate(request["input"])
                ]
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture()
def endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    Endpoint.requests = []
    yield f"http://127.0.0.1:{server.server_port}/v1/embeddings"
    server.shutdown()


def test_a_remote_model_embeds_and_encodes_through_the_python_front_door(
    tmp_path: Path, endpoint: str
) -> None:
    config = tmp_path / "jammi.toml"
    config.write_text(
        f"""
[gpu]
device = -1

[models.remote.encoder]
protocol = "openai_embeddings"
url = "{endpoint}"
model = "encoder-small"
dimensions = {DIMS}
revision = "r1"
headers = {{ Authorization = "Bearer {TOKEN}" }}
timeout_secs = 10
"""
    )
    texts = ["alpha", "beta", "gamma"]
    corpus = tmp_path / "corpus.parquet"
    pq.write_table(pa.table({"id": [0, 1, 2], "text": texts}), corpus)

    with jammi.connect(f"file://{tmp_path / 'data'}", config=str(config)) as db:
        db.add_source("corpus", url=str(corpus), format="parquet")
        db.generate_embeddings(
            source="corpus",
            model="remote:encoder",
            columns=["text"],
            key="id",
            modality="text",
        )
        sent = sorted(t for request in Endpoint.requests for t in request["input"])
        assert sent == sorted(texts)
        assert {r["model"] for r in Endpoint.requests} == {"encoder-small"}

        query = db.encode_query(model="remote:encoder", query="gamma")
        assert list(query) == pytest.approx(vector_of("gamma"))

        rows = db.search("corpus", query=query, k=1).to_pylist()
        assert rows[0]["_row_id"] == "2"
