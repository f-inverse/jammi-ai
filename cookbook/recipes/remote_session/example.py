"""One program, two transports: the same session over an embedded engine and a server.

`jammi.connect(target)` picks the transport from the target alone —
`file://…` runs the engine in this process, `grpc://…` talks to a
`jammi-server`. Both return a `Session` with the same verbs, so the program
below runs unchanged against either. A handful of features exist on one
transport only; `db.supports(capability)` says which, and calling one the
backend lacks raises `NotSupportedOnBackend`.

1. Start a server (`jammi.testing.LiveServer`) over its own artifact directory
2. Run one function — register, embed, search — on an embedded session and on
   a remote one, and compare the answers
3. `db.supports(...)` on both, and the remote-only `db.session_id`
4. `jammi.parse_target` — how a target string picks its transport
5. The session journal: `jammi.observe` reports every session opened and
   closed, `jammi.open_sessions()` / `jammi.open_session_labels()` list the
   ones open now, and `jammi.describe_sessions` names them — how a program
   (or a notebook) finds a session it forgot to close

Needs a `jammi-server` binary on PATH: `pip install jammi-server`, or
`cargo build --release -p jammi-server` with `target/release` on PATH.

Run with `python cookbook/recipes/remote_session/example.py`.
"""

from __future__ import annotations

import tempfile

import jammi
from jammi import Capability
from jammi.errors import NotSupportedOnBackend
from jammi.testing import LiveServer
from jammi_cookbook import fixtures

CORPUS_URL = str(fixtures.path("tiny_corpus.parquet"))
MODEL = fixtures.model("tiny_bert")


def nearest(db) -> list[int]:
    """The same program for either transport: the five patents nearest a query."""
    db.add_source("corpus", url=CORPUS_URL, format="parquet")
    db.generate_embeddings(source="corpus", model=MODEL, columns=["content"], key="id")
    query = db.encode_query(model=MODEL, query="quantum error correction")
    hits = db.search("corpus", query=query, k=5).to_pylist()
    return [int(h["_row_id"]) for h in hits]


def main() -> int:
    events: list[str] = []
    unsubscribe = jammi.observe(
        lambda handle, label: events.append(f"opened {label}"),
        lambda handle, label: events.append(f"closed {label}"),
    )
    with tempfile.TemporaryDirectory() as local_dir, tempfile.TemporaryDirectory() as srv_dir:
        for target in (f"file://{local_dir}", "grpc://127.0.0.1:8081"):
            print(f"{target} → {jammi.parse_target(target)}")

        with jammi.connect(f"file://{local_dir}") as embedded:
            local_hits = nearest(embedded)
            print(f"embedded: {local_hits}")
            print(f"embedded supports audit: {embedded.supports(Capability.AUDIT)}")
            try:
                embedded.session_id
                raise AssertionError("the embedded engine has no connection id")
            except NotSupportedOnBackend as absent:
                print(f"embedded session_id: {absent}")

        with LiveServer(srv_dir) as server, jammi.connect(server.endpoint) as remote:
            open_now = dict(jammi.open_session_labels())
            print(f"open sessions: {jammi.describe_sessions(open_now)}")
            assert len(jammi.open_sessions()) == 1, "only the remote session is open"
            remote_hits = nearest(remote)
            print(f"remote ({server.endpoint}): {remote_hits}")
            print(f"remote supports audit: {remote.supports(Capability.AUDIT)}")
            print(f"remote session_id: {remote.session_id}")
            assert remote.supports(Capability.SESSION_ID)

        assert local_hits == remote_hits, "one engine, two transports, one answer"
    unsubscribe()
    print("journal:", "; ".join(events))
    assert not jammi.open_session_labels(), "every session was closed"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
