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

### D18 — what running at `full` scale found in the engine

Each was found by a chapter failing on an L4 at `full` and fixed at its cause:

- **Hard-negative mining ran out of GPU memory** encoding every positive in one
  batch. It encodes them in the training batch size.
- **A mutable-table write wider than one statement failed** with "too many SQL
  variables". Each backend declares its bind-parameter ceiling (SQLite 32,766,
  Postgres 65,535) and writes and key deletes are chunked under it.
- **An OpenCLIP checkpoint resolved a config and weights from different
  conventions.** Resolution reads the repo listing first and picks the config
  and weights as a pair.
- **A bare `ModernBertModel` checkpoint did not load.** It now loads the way a
  bare `BertModel` already did.
- **The context-predictor bench's committed baseline named a serving table**
  that exists only in the session that trained it. The committed baseline is
  the model's trained shape, without that key; the determinism tests register
  it naming their own session's table.

### D19 — the `full` text encoder is embedding-trained, and full fine-tunes run in bf16

`answerdotai/ModernBERT-base` is a masked-LM checkpoint, not an embedding model:
raw same-subject precision@10 0.358, and propagation did nothing (−0.003).
`Alibaba-NLP/gte-modernbert-base` gives 0.513 raw and 0.551 propagated, and every
`full` finding was re-measured on it. Several chapter claims made on the old
encoder did not survive and were rewritten (D21–D23). Full-scale fine-tunes
train at `encoders.training_dtype(scale)` — bf16 on the sm_80+ GPU they need,
f32 on the small scale's CPU — which needed `fine_tune_graph` to take every
shared training knob.

### D20 — the dataset cache is written by atomic rename

Two sessions sharing the cache (two chapters or notebooks at once) re-cache the
same table, and the in-place write truncated the parquet under the other
session's scan ("Corrupt footer"). Downloads and cached tables now land through
one temp-file-and-rename helper.

### D21 — the tier-04 predictor's budget is one it descends

At the default learning rate (5e-3) the year predictor never descended its
objective at `full` scale: a step is one whole subject task, and the year is only
weakly predictable from a paper's neighbours (a kNN-mean regressor over the
propagated table reaches RMSE 1.00 against the mean's 1.04). The held-out CRPS
oscillated above its starting value, and two pods on different NVIDIA drivers
agreed for fifteen epochs and then diverged to interval coverages of 0.961 and
0.892. At 1e-4 it descends smoothly to its floor by epoch 30; the pods agree to
three decimals. The chapters' conformal findings are the ones measured on that
predictor: the year interval under-covers (0.689) and a density-ratio reweight
lifts it only to 0.702; the subject set holds nominal (0.908), its score
orthogonal to the shift.

### D22 — weighted conformal holds the test row's own mass

The book's weighted split-conformal quantile normalized over the calibration rows
alone, while its marginal pass used the finite-sample ⌈(n+1)(1−α)⌉ rule — the two
passes differed in more than their weights, and the weighted one was
anti-conservative. `shift.quantiles` is one routine (Tibshirani et al. 2019):
each test row's own weight sits at +∞, and uniform weights are the marginal
quantile. The conformal chapter's synthetic shift drew 90% of its pool without
replacement — barely a bias, and not the likelihood ratio its weights assumed —
so each paper is now kept independently with a known probability, the weight is
its exact inverse, and coverage is reported as the mean over 200 draws, the
expectation the guarantee is stated in.

### D23 — a finding the data does not support is rewritten, not re-tuned

Where a `full` measurement refuted a chapter's claim, the chapter now says what
was measured: rank fusion is judged by a paired per-query difference and is a
wash; the regression fine-tunes land on the predict-the-mean baseline and their
Gaussian heads under-cover out of sample; the declared-edge fine-tune lifts
precision about as much as propagation, not more. No budget or seed was searched
to make an old claim pass.

### D24 — the cookbook installs from its release tag, not from PyPI

Phase 3 first published the book's library and fixtures as a `jammi-cookbook`
PyPI package. Nothing installs cookbook helpers outside a notebook, so the
package bought a second PyPI project, a trusted publisher and a release workflow
for no reader. A notebook now installs the library from the tag its release was
built at (`jammi-cookbook @ git+…@py-vX#subdirectory=cookbook/book`): the same
pin, and the same wheel contents, since pip builds it from the book's own
`pyproject.toml`. The cost is the setup cell cloning the tagged tree (≈120 MB)
where the wheel was 2.6 MB. `pypi-cookbook.yml` is deleted.

## Cuts

- **T4 / sm_75.** The CUDA wheel targets sm_80+; a T4 Colab runtime runs the
  CPU wheel.
