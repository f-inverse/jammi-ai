"""Operate a server from the command line: status, sources, models, jobs, tables, topics, channels.

Run with `python cookbook/recipes/cli/example.py`, or a step at a time as a
notebook: each `# %%` cell is one step. Needs a `jammi-server` on PATH
(`pip install jammi-server`); the `jammi` CLI is fetched for this platform when
it is not already installed.
"""

# %% [markdown]
# ## A server to operate
#
# In production the server runs elsewhere and the CLI points at it with
# `--target`. Here the recipe starts one of its own, on a free local port.

# %%
import contextlib
import importlib.metadata
import json
import subprocess
import tempfile
from pathlib import Path

import jammi
from jammi.testing import LiveServer
from jammi_cookbook import fixtures, programs

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
work = Path(tempfile.mkdtemp(prefix="jammi_cli_"))

stack = contextlib.ExitStack()
server = stack.enter_context(LiveServer(work / "server"))
jammi_cli = programs.cli(importlib.metadata.version("jammi-ai"))


def cli(*args: str) -> str:
    """Run `jammi --target <server> <args…>` and return what it printed."""
    return subprocess.run([jammi_cli, "--target", server.endpoint, *args],
                          capture_output=True, text=True, check=True).stdout


print(f"server at {server.endpoint}")

# %% [markdown]
# ## Is it up, and what can it do?
#
# `jammi status` reports the server's version, the features it was built with,
# the storage backends it can read and write, and the services it mounts.
# Reaching it at all is the health check.

# %%
print(cli("status"))

# %% [markdown]
# ## Sources, embeddings and search
#
# `sources add` registers a file the server can read (a local path here,
# `s3://…`, `gs://…` or `azure://…` in production) and `sources list` shows
# what is registered. `embed` runs an encoder over a source's columns on the
# server; `search` ranks the rows nearest one row's vector and prints JSON
# lines.

# %%
print(cli("sources", "add", "corpus", "--url", fixtures.url("tiny_corpus.parquet"),
          "--format", "parquet"))
print(cli("sources", "list"))
print(cli("embed", "corpus", "--model", MODEL, "--columns", "content", "--key", "id"))
hits = [json.loads(line) for line in
        cli("search", "corpus", "--row-key", "1", "-k", "3", "--select", "id,title").splitlines()]
for hit in hits:
    print(hit)
assert hits[0]["id"] == 1, "a row is its own nearest neighbour"

# %% [markdown]
# ## Models
#
# Embedding the source registered its encoder; `models list` shows every model
# the server knows and `models describe` one of them.

# %%
print(cli("models", "list"))
print(cli("models", "describe", MODEL))

# %% [markdown]
# ## Jobs and workers
#
# A program submits jobs through a client; the CLI is the operator's side of
# them: `jobs list` and `jobs status` read them, `jobs cancel` stops one and
# `jobs prune` deletes finished rows past the retention window. Here a client
# submits two fine-tunes, and the CLI cancels the second. `workers list` shows
# the engine processes claiming jobs.

# %%
with jammi.connect(server.endpoint) as db:
    db.add_source("pairs", url=fixtures.url("tiny_pairs.csv"), format="csv")
    first = db.fine_tune(source="pairs", base_model=MODEL, columns=["text_a", "text_b", "score"],
                         method="lora", task="text_embedding", lora_rank=4, epochs=1)
    second = db.fine_tune(source="pairs", base_model=MODEL, columns=["text_a", "text_b", "score"],
                          method="lora", task="text_embedding", lora_rank=4, epochs=1)
    print(cli("jobs", "cancel", second.job_id))
    first.wait()
    tuned = first.output_model_id

print(cli("jobs", "list"))
print(cli("jobs", "status", first.job_id))
print(cli("jobs", "prune"))
print(cli("workers", "list"))

# %% [markdown]
# ## Mutable tables
#
# A mutable table is a keyed table a program updates in place, beside the
# immutable result tables. `mutable create` takes its schema as a JSON file and
# its primary key; `list` and `drop` do what they say.

# %%
schema = work / "features.json"
schema.write_text(json.dumps([
    {"name": "entity_id", "type": "Utf8", "nullable": False},
    {"name": "score", "type": "Float64", "nullable": True},
]))
print(cli("mutable", "create", "--name", "features", "--schema", str(schema),
          "--primary-key", "entity_id"))
print(cli("mutable", "list"))
print(cli("mutable", "drop", "features"))

# %% [markdown]
# ## Trigger topics
#
# A topic carries events a subscriber filters with SQL. `trigger register`
# declares one with its schema inline, as `name:type` pairs.

# %%
print(cli("trigger", "register", "--name", "cdc.orders",
          "--schema", "op:string,order_id:string,amount:float:nullable"))
print(cli("trigger", "list"))
print(cli("trigger", "drop", "--name", "cdc.orders"))

# %% [markdown]
# ## Provenance channels
#
# A channel names a kind of evidence attached to result rows (who scored a row,
# which model labelled it) and the columns it carries. Channels are
# append-only: `channels add-column` extends one, and redeclaring a column is
# refused.

# %%
print(cli("channels", "register", "--name", "reviewed_by", "--priority", "10",
          "--column", "reviewer:Utf8"))
print(cli("channels", "add-column", "reviewed_by", "--column", "reviewed_at:Int64"))
print(cli("channels", "list"))

# %% [markdown]
# ## Reconcile storage
#
# `reconcile` cross-checks the catalog against the object store: result files
# no row points at, and rows whose files are gone. Without `--apply` it only
# reports.

# %%
print(cli("reconcile"))

# %% [markdown]
# ## Delete a model
#
# `models delete` hard-deletes a model row, and refuses while anything still
# references it — here, the job that produced the fine-tuned model.

# %%
refused = subprocess.run([jammi_cli, "--target", server.endpoint, "models", "delete", tuned],
                         capture_output=True, text=True)
print(f"exit {refused.returncode}: {refused.stderr.strip()}")
assert refused.returncode != 0, "a referenced model is not deleted"

# %%
stack.close()
