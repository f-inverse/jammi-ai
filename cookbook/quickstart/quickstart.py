"""Jammi in five minutes: connect, register a source, embed it, search it.

Run with `python cookbook/quickstart/quickstart.py`, or a step at a time as a
notebook: each `# %%` cell is one step.
"""

# %% [markdown]
# ## 1. Install
#
# ```bash
# pip install "jammi-ai[embedded]"
# ```
#
# `jammi-ai` is the client (`import jammi`). The `[embedded]` extra adds the
# engine itself, `jammi-ai-native`, so the client can run it in your process;
# without it the client only reaches a remote `jammi-server`, and a `file://`
# target raises `NoEmbeddedEngineError`. On an NVIDIA GPU of compute capability
# 8.0 or newer (A100, L4, RTX 30-series and later), install the CUDA engine in
# place of the CPU one: `pip install jammi-ai jammi-ai-native-cu12`. It carries
# its CUDA libraries as pip dependencies; the host needs only the NVIDIA driver.
#
# Jammi runs on Python 3.9 or newer, on Linux (x86_64 or aarch64, glibc 2.28+)
# and macOS (Apple Silicon or Intel); the CUDA engine is Linux x86_64. Windows
# is not supported: the storage layer uses POSIX memory mapping.
#
# Both imports succeed once the engine is installed. `jammi_cookbook` holds the
# small datasets and models the cookbook runs on.

# %%
import os
import tempfile

import jammi
import jammi_native  # noqa: F401 -- the engine the [embedded] extra installed
from jammi_cookbook import fixtures

# %% [markdown]
# ## 2. Connect
#
# `jammi.connect(target)` is the one front door. A `file://` target runs the
# engine in this process, with its catalog and every result table kept under
# that directory; a `grpc://host:8081` or `https://host` target opens a session
# against a remote `jammi-server` instead, with the same methods.
#
# Device and batch size are engine configuration, read from the environment
# (or a `JAMMI_CONFIG` TOML file) when the engine starts, so they apply the same
# way in your process or behind a server. The engine takes GPU 0 when there is
# one; `JAMMI_GPU__DEVICE=-1` forces the CPU and `1` picks another device.
# `JAMMI_ENGINE__BATCH_SIZE` sets how many rows an encoder runs at once, here
# small for a 20-row corpus.

# %%
os.environ.setdefault("JAMMI_ENGINE__BATCH_SIZE", "8")

db = jammi.connect(f"file://{tempfile.mkdtemp()}")

# %% [markdown]
# ## 3. Register a source
#
# `add_source` registers a file so SQL and embedding jobs can name it. The
# `url` is a local path, or `s3://bucket/key`, `gs://bucket/key` or
# `azure://container/blob` for object storage; `format` is `parquet`, `csv` or
# `json`. This corpus is 20 short paper abstracts with an `id`, a `title`, the
# `content`, a `year` and a `category`.

# %%
db.add_source("corpus", url=str(fixtures.path("tiny_corpus.parquet")), format="parquet")

# %% [markdown]
# A registered file is a table in SQL, named `<source>.public.<table>`, where
# the table is the file's name without its extension: `corpus.public.tiny_corpus`
# here. Every query returns a `pyarrow.Table`.

# %%
for row in db.sql("SELECT id, title, year FROM corpus.public.tiny_corpus LIMIT 3").to_pylist():
    print(row)

# %% [markdown]
# ## 4. Embed the corpus and search it
#
# `generate_embeddings` runs an encoder over every row of `corpus`, writes the
# vectors and the `key` column to a Parquet result table, and builds an ANN
# index beside it. The job is checkpointed: interrupted and run again, it picks
# up where it left off.
#
# `model` is a Hugging Face Hub id, downloaded on first use and cached, or a
# local directory (`local:/path`) holding the same files. The quickstart uses
# `sentence-transformers/all-MiniLM-L6-v2`, a small sentence encoder (384
# dimensions, about 90 MB) that embeds this corpus in seconds on a CPU.

# %%
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

db.generate_embeddings(
    source="corpus",
    model=MODEL,
    columns=["content"],
    key="id",
    modality="text",
)

# %% [markdown]
# The query is encoded by the model that built the index, so its vector has
# the index's dimension. `search` returns the `k` nearest rows as a
# `pyarrow.Table`: the source's columns, `_row_id`, and `similarity` (cosine,
# 1.0 for identical). `filter="year > 2020"` ranks only the rows a predicate
# selects, and `select=[...]` picks the columns; joining sources, or running a
# model over the results, is SQL through `db.sql(...)`.

# %%
query = db.encode_query(model=MODEL, query="how does quantum computing work?")
results = db.search("corpus", query=query, k=3)

for row in results.to_pylist():
    print(f"{row['_row_id']:<8} {row['similarity']:>9.4f}  {row['category']:<8} {row['title']}")

# %% [markdown]
# The three nearest abstracts are physics papers on quantum computing. The
# encoder places text by meaning, so a question lands next to the abstracts
# that answer it rather than the ones that merely share a word.

# %%
assert results.num_rows == 3
assert set(results.column("category").to_pylist()) == {"physics"}, results.to_pylist()

# %% [markdown]
# An embedded engine holds its catalog until the session closes. A session is
# also a context manager: `with jammi.connect(...) as db:` closes it when the
# block ends.

# %%
db.close()
