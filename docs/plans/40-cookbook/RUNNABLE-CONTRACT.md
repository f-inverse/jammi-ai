# The runnable-cookbook contract (#653–#658)

**The contract.** A new jammi user learns every jammi capability by running
the cookbook — its recipes and its book chapters — in a fresh Google Colab, on
its GPU when one is present (sm_80+) and on its CPU otherwise.

This file is the decision log for the program that made the contract true:
every fork met, how it was resolved (CLAUDE.md → the philosophy guide → the
code → outside sources), and every cut.

## Decisions

### D1 — `describe_table` returns the manifest itself, not a projection (#654)

The verb reads a table's recorded materialization. The recorded thing is the
`MaterializationManifest`; a second struct projecting some of its fields would
be a second schema to keep in step with the first (CLAUDE.md: DRY, the right
abstraction). `describe_table` returns the manifest, `leaves` included.

### D2 — the wire carries the manifest as its own canonical JSON (#654)

`DescribeTableResponse { string manifest_json }`. The manifest's canonical
serialization is the sidecar's own, versioned by `manifest_version` and read
back through the engine's strict reader (`MaterializationManifest::
from_json_bytes`), so the Rust peer decodes a typed manifest through ONE
schema, and the Python remote's `json.loads` is byte-for-byte the embedded
binding's serde dict. Mirroring the ~15-variant `ProducingDescriptor` and the
model-run union as proto messages would be a second, drifting schema.
Precedent: `JobStatusResponse.metrics_json` / `acceleration_report_json`.

### D3 — a table with no manifest is a typed refusal (#654)

New `JammiError::MissingManifest { table }` (wire tag 47, `NOT_FOUND`, Python
`jammi.errors.MissingManifest(BackendError)`). `verify_materialization` /
`staleness` keep their `missing_manifest` *verdicts* — they answer a question
about the table; `describe_table` has nothing to answer with.

### D4 — the Rust remote peer carries the whole provenance family (#654)

`jammi_admin::CatalogClient` carried none of `verify_materialization`,
`staleness`, `derives_from`; adding only `describe_table` would leave the Rust
peer a partial one. All four ship, over the wire conversions that already
existed.

### D5 — the coverage gate is a repo guard over parsed protocols (#655)

`ci/scripts/check_cookbook_coverage.py` replaces the book's
`check_chapter_coverage.py`. Shipped = the `Session` + `JobHandle` protocol
members, the `Capability` values (the one-sided members) and the functions
`jammi.__all__` exports — 83 surfaces, parsed with `ast`. Lanes:
DirectCell (an executed chapter cell; `eval: false` cells never count),
WrapperLane, Recipe (the recipe must be registered in `tests/cookbook_smoke.py`).
No CacheLane, no Deferred. It runs in the guard job on every change, so an
engine PR that adds a verb fails there.

### D6 — every recipe runs on every PR (#655)

The "slow" tier was stale (`fine_tune` runs in seconds) and the server recipes
were nightly-only, so a remote-surface break reached main. `cookbook.yml` now
builds the CPU server once and runs every recipe plus the client's live-server
suite on every change — the repo's "select, never skip" rule.

### D7 — recipes resolve fixtures through the package (#655, #658)

`jammi_cookbook.fixtures` (`path` / `url` / `model`) over
`jammi_cookbook/_fixtures`, a link to `cookbook/fixtures` in a checkout and the
bytes themselves in a built wheel. No recipe reads `REPO_ROOT`.

### D8 — what running the recipes found in the engine

- **`search` did not agree across transports.** The wire `SearchResponse` was
  a lossy `{key, score, stringified columns}`; the embedded engine returns the
  typed, hydrated rows. The wire now carries the rows as one Arrow IPC stream
  (`ArrowBatch`, as `Infer` does) through one encode/decode pair
  (`jammi_wire::result_rows_{to,from}_proto`) for both verbs; `SearchHit` is
  gone. A live parity test compares the two tables whole.
- **`eval_*` over a derived table loaded the producer's tag as a model.**
  A propagated table's `model_id` is `graph_propagate`; the eval runner asked
  the Hub for it (401). The query encoder is now read off the recorded
  descriptors: an embedding's model, a context set's encoder, a propagation's
  input's encoder (walked iteratively down the acyclic lineage); structure
  embeddings and imported vectors are a typed `Eval` refusal naming the
  producer.
- **A CPU-only build "requested" GPU 0 on every model load.** `[gpu] device`
  defaulted to 0 whatever the build, so every load in a CPU wheel warned. The
  default is now a build property (`GpuConfig::DEFAULT_DEVICE`: the
  accelerator when `jammi-ai`'s `cuda`/`metal` compiles one in, the CPU
  otherwise), and `device` is an `Option` resolved by `primary()` — unset, the
  primary is the first of `devices`, so `devices = [0, 1]` alone means what it
  says. An explicit `device = 0` on a CPU build still warns.

### D9 — one pipeline per vertical, parameterized by scale (#656, #657)

CLAUDE.md: "two things that are the same thing at a different scale are one
thing". Each dataset vertical is one pipeline in `jammi_cookbook`, run by the
chapter at a `Scale`: `small` (the committed fixtures, the tiny fixture
encoders, CPU — what CI and a CPU Colab run) and `full` (the published
datasets and real encoders, on a GPU — the optional "at scale" run). The
chapter code is identical at both scales; each scale has its own frozen
goldens. The committed caches and the `scripts/build_*_cache.py` scripts that
produced them are the full-scale pipelines' former, cache-writing form: they
are folded into the verticals and deleted.

### D10 — the committed cache is retired, not kept beside the pipelines

With every chapter running live at both scales, the GPU-emitted cache
(`cookbook/book/artifacts/`, its LFS routing, `jammi_cookbook.cache`, the emit
scripts, and the tests that asserted the cache's contents) had no reader. It
is deleted outright — no "reference" copy. The render selector
(`ci/scripts/select_render_chapters.py`) lost its CACHE_READ bucket: a chapter
is LIVE, LIVE_NEEDS_SERVER or STATIC, every live chapter renders on an engine,
book-library or fixture diff, and a golden diff renders the chapters that
check that dataset. `tests/test_goldens.py` fails on a golden no chapter checks
any more, so a retired measurement cannot linger as a number that looks
verified.

The held-out fine-tune fixture (`cookbook/fixtures/finetune_heldout/`) was
produced by a script that mined through the retired loader, and the bench
regenerated its train text over the network through that same script. The
train text (4.9 MB, ODC-BY, the same terms as the committed held-out text) is
now committed and hash-verified by the fixture guard; the producer, the bench's
provisioning step and its pod venv are deleted.

### D11 — graph fine-tune edges are an engine source, not a staged table (#663)

`fine_tune_graph` took only an edge *table* the caller had materialised. The
cookbook needed to train on a registered citation source and on an
engine-built neighbour graph. `GraphFineTuneSources.edges` is a oneof —
`edge_graph_table` (a result table the engine built) or `edge_source`
(`source_id`, `src_column`, `dst_column`) — with no default arm: a request
naming neither is refused. `build_neighbor_graph`'s input is named
`embedding_table` on every surface, since that is what it is.

### D12 — BM25 is an engine verb, indexed in memory per input anchor (#664)

The retrieval chapter scored BM25 in Python. Lexical retrieval passes the
discipline test (a feature store, a support-ticket search, a code search all
reach for it), so it is `build_lexical_index` + `lexical_search` on every
surface. `build_lexical_index` materialises a `lexical` result table
(`_row_id`, the concatenated text) through the ordinary producer path, so
recompute, staleness and tenancy apply unchanged; the tantivy index is built in
memory from that table on first search and cached by the table's input anchor.
No on-disk index sidecar: the table is the durable artifact, the index a
derived cache of it. Results ride the same `ranked` channel machinery as
`search` (`bm25_score`, `bm25_rank`).

### D13 — a Matryoshka prefix is a serving width, recorded on the table (#668)

`generate_embeddings(dimensions=…)` and `encode_query(dimensions=…)` serve a
truncated, L2-renormalised prefix. The width is part of the embedding request
(`EmbeddingRequest.dimensions`), recorded in the producing descriptor, and
replayed by refresh and recompute, so a table is always extended at the width
it was built at; the eval runner encodes queries at the table's width.

### D14 — a scoped caller's tenant reaches the model bind (#666)

A tenant-scoped session embedding with its own fine-tuned model fell back to
the Hub, because the inference runner bound models outside the caller's
tenant scope. `TenantScopedModels` captures the binding and runs every model
load inside it. An unknown `jammi:fine-tuned:` id is `ModelNotFound`, never a
Hub lookup.

### D15 — the regression head needed a budget, not a bias (#669)

The pinball-loss chapter's coverage band failed at `small`. A trainable output
bias was built and measured, then reverted: it did not move coverage and it
halved the separation between quantile levels. The cause was the optimiser
budget — coverage 0.525 at `lr 1e-2` and 0.925 at `5e-2`, bias or not — so the
`small` run gets `learning_rate 5e-2` and the chapter says so.

### D16 — only a file's last catalog pool waits for its `-wal`

Every embedded `close()` took two seconds and logged a warning. The session's
lease keeper holds its own pool on `catalog.db` and closes first; its close
waited for the `-wal` to disappear, which cannot happen while the session's
pool is live. The `-wal` is evidence about the file, so the backend now counts
open pools per canonical path and only the close that releases the last one
waits. Closing takes tens of milliseconds, and the warning is left for the one
case it names: a close-time checkpoint SQLite declined.

### D17 — an embedded engine logs at `warn` unless asked

The Python host installed an INFO filter of its own and ignored `[logging]`,
so every chapter printed epoch and sink progress lines. `[logging] level` is
optional; each host supplies its default — `info` for `jammi-server`, `warn`
for the embedded engine — through one formatter
(`jammi_ai::telemetry::fmt_layer`), and `RUST_LOG` overrides both. ANSI colour
is used only on a terminal. The unused `jammi_db::init_tracing` is deleted.

## Cuts

- **T4 / sm_75.** The CUDA wheel targets sm_80+; a T4 Colab runtime runs the
  CPU wheel.
