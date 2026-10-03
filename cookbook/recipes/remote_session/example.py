"""One program, two transports: the same session over an embedded engine and a server.

Run with `python cookbook/recipes/remote_session/example.py`, or a step at a
time as a notebook: each `# %%` cell is one step. Needs a `jammi-server` on
PATH (`pip install jammi-server`).
"""

# %% [markdown]
# ## Watch sessions open and close
#
# `jammi.observe` calls back on every session the process opens and closes,
# and returns the function that stops it. A program, or a notebook, uses it to
# find a session it forgot to close.

# %%
import tempfile

import jammi
from jammi import Capability
from jammi.errors import NotSupportedOnBackend
from jammi.testing import LiveServer
from jammi_cookbook import fixtures

CORPUS_URL = str(fixtures.path("tiny_corpus.parquet"))
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

events: list[str] = []
unsubscribe = jammi.observe(
    lambda handle, label: events.append(f"opened {label}"),
    lambda handle, label: events.append(f"closed {label}"),
)

# %% [markdown]
# ## The target picks the transport
#
# `jammi.connect(target)` reads the transport from the target alone:
# `file://…` runs the engine in this process, `grpc://…` talks to a
# `jammi-server`. `jammi.parse_target` shows the reading.

# %%
local_dir, server_dir = tempfile.mkdtemp(), tempfile.mkdtemp()
for target in (f"file://{local_dir}", "grpc://127.0.0.1:8081"):
    print(f"{target} → {jammi.parse_target(target)}")

# %% [markdown]
# ## One program
#
# Both transports return a `Session` with the same methods, so one function
# runs unchanged against either: register the corpus, embed it, and return the
# five rows nearest a query.

# %%
def nearest(db) -> list[int]:
    db.add_source("corpus", url=CORPUS_URL, format="parquet")
    db.generate_embeddings(source="corpus", model=MODEL, columns=["content"], key="id")
    query = db.encode_query(model=MODEL, query="quantum error correction")
    hits = db.search("corpus", query=query, k=5).to_pylist()
    return [int(h["_row_id"]) for h in hits]


# %% [markdown]
# ## Run it embedded
#
# A few features exist on one transport only. `db.supports(capability)` says
# which, and using one the backend lacks raises `NotSupportedOnBackend`: an
# embedded engine has no connection, so it has no `session_id`.

# %%
with jammi.connect(f"file://{local_dir}") as embedded:
    local_hits = nearest(embedded)
    print(f"embedded: {local_hits}")
    print(f"embedded supports audit: {embedded.supports(Capability.AUDIT)}")
    try:
        embedded.session_id
        raise AssertionError("the embedded engine has no connection id")
    except NotSupportedOnBackend as absent:
        print(f"embedded session_id: {absent}")

# %% [markdown]
# ## Run it against a server
#
# `LiveServer` starts a real `jammi-server` over its own artifact directory and
# stops it when the block ends. While the remote session is open,
# `jammi.open_session_labels()` lists it, and `jammi.describe_sessions` names
# it.

# %%
with LiveServer(server_dir) as server, jammi.connect(server.endpoint) as remote:
    open_now = dict(jammi.open_session_labels())
    print(f"open sessions: {jammi.describe_sessions(open_now)}")
    assert len(jammi.open_sessions()) == 1, "only the remote session is open"
    remote_hits = nearest(remote)
    print(f"remote ({server.endpoint}): {remote_hits}")
    print(f"remote supports audit: {remote.supports(Capability.AUDIT)}")
    print(f"remote session_id: {remote.session_id}")
    assert remote.supports(Capability.SESSION_ID)

# %% [markdown]
# ## One engine, one answer
#
# Both transports ran the same engine over the same data, so they return the
# same rows. The journal saw both sessions open and close.

# %%
assert local_hits == remote_hits, "one engine, two transports, one answer"

unsubscribe()
print("journal:", "; ".join(events))
assert not jammi.open_session_labels(), "every session was closed"
