# API Stability

Jammi exposes a deliberate, frozen public surface. This page is the operator's
reference for **what is stable**, **what semver promise covers it**, and **how
the freeze is enforced** — not as prose anyone can let drift, but as a CI guard
that reds the moment a stable surface changes shape.

The single principle: **a stable surface does not change under you without a
major.** A verb is not renamed, an rpc is not dropped, a wire package is not
removed, and a persisted-format version is not reinterpreted, across any release
that does not bump the major version. The surfaces below are the ones that
promise holds for; everything not listed here is internal and may move.

## The frozen stable surfaces

Three surfaces are frozen. Each is **machine-checked** against a committed
baseline, so the freeze is enforceable rather than aspirational (see [Enforcement](#enforcement-the-freeze-guard)).

### 1. The verb set — the call surface

The public verb vocabulary a caller invokes — identical name-for-name and
signature-for-signature across the embedded (`jammi_ai.Database`) and remote
(`jammi_client.RemoteDatabase`) transports. It is pinned, set-by-set, in
`crates/jammi-python/tests/test_conformance.py`; those sets **are** the frozen
verb list:

| Verb set (conformance constant) | Verbs |
|---|---|
| `_REMOTE_VERBS` | `add_source`, `generate_embeddings`, `encode_query`, `search`, `sql`, `list_sources`, `describe_source`, `set_tenant`, `tenant_scope`, `tenant`, `get_server_info` |
| `_TRAINING_VERBS` | `fine_tune`, `fine_tune_graph`, `train_context_predictor`, `predict_with_context_predictor` |
| `_INFERENCE_VERBS` | `infer` |
| `_PIPELINE_VERBS` | `build_neighbor_graph`, `propagate_embeddings`, `asof_join`, `assemble_context`, `recompute`, `verify_materialization`, `staleness`, `derives_from` |
| `_EVAL_VERBS` | `eval_embeddings`, `eval_per_query`, `eval_inference`, `eval_compare`, `eval_calibration` |
| `_CHANNEL_VERBS` | `register_channel`, `add_channel_columns`, `list_channels` |
| `_NUMERIC_VERBS` | `conformalize`, `conformalize_interval`, `conformalize_cqr`, `rrf_fuse` |
| `_MUTABLE_TOPIC_VERBS` | `create_mutable_table`, `drop_mutable_table`, `list_mutable_tables`, `register_topic`, `drop_topic`, `list_topics`, `publish_topic`, `subscribe_collect` |
| `_LIFECYCLE_VERBS` | `list_models`, `describe_model`, `delete_model` |
| `_SEARCH_VERBS` | `search` (pinned separately for the `embedding_table=` selector) |

The conformance suite is the *enforced annotation*: removing or renaming a verb,
or changing its signature on either transport, reds the suite. Jammi does not
carry a per-`pub`-item `#[stable]` rustdoc attribute — Rust has no such
attribute, and history-bearing version markers in rustdoc are explicitly
disallowed — so the conformance sets carry the freeze that a `#[stable]` pass
would carry elsewhere.

### 2. The wire contract — `package jammi.v1.*`

The gRPC/Flight SQL wire surface is the nine `jammi.v1.*` proto packages:

| Package | Surface |
|---|---|
| `jammi.v1.audit` | provenance / audit log rpcs |
| `jammi.v1.catalog` | sources, models, channels, tenant, server-info, mutable tables, topics |
| `jammi.v1.embedding` | embedding generation, query encode, search |
| `jammi.v1.error` | the typed wire-error message (no rpcs) |
| `jammi.v1.eval` | the evaluation rpcs |
| `jammi.v1.inference` | bulk inference + predict |
| `jammi.v1.pipeline` | graph / context / as-of / recompute / materialization rpcs |
| `jammi.v1.training` | training submit + status |
| `jammi.v1.trigger` | topic publish + subscribe |

The contract is the full set of `(Service, Method)` rpc paths these packages
serve — decoded from the compiled `FILE_DESCRIPTOR_SET`, the authoritative
machine-readable description of what the binary actually serves, not a
hand-maintained list. The `v1` in the package path is the wire-stability stamp:
a breaking change to a message or an rpc shape requires a `jammi.v2.*` package,
not an in-place edit of `v1`.

### 3. The persisted-format versions

The on-disk format-of-record versions, each a writer-stamped, reader-checked
version with reject-newer (or strict) semantics — the full contract is on the
[Format Stability](./format-stability.md) page:

| Format | Stamp | Current version |
|---|---|---|
| Materialization manifest (`.materialization.json`) | `MANIFEST_VERSION` | `3` |
| ANN row map (`.rowmap`) | `ROWMAP_VERSION` | `1` |
| ANN sidecar manifest (`.manifest.json`) | `ANN_MANIFEST_VERSION` | `1` |
| Catalog schema | append-only migration ledger | through `022` |

The catalog migration ledger is **append-only**: a migration is never edited or
removed once shipped, only a new numbered migration is appended. The other three
stamps follow the reject-newer idiom — a newer stamp than this build knows is a
typed rejection, never a silent misparse.

## The semver commitment

This release is the **terminal 0.x engineering bar**: the three surfaces above
are frozen, and a breaking change to any of them — a renamed/removed verb, a
dropped rpc, a removed `jammi.v1.*` package, an incompatible reinterpretation of
a persisted-format version — does not ship without a major version bump. New
*additive* surface (a new verb, a new rpc, a new appended migration) may land in
a minor; it does not break a caller written against the frozen set, because it
only grows the surface.

Concretely:

- A new verb is **added** to a conformance set in the same PR that adds the verb,
  on both transports — additive, minor-compatible.
- A removed or renamed verb is a **breaking** change — major only.
- A new rpc is a new `(Service, Method)` path appended to the wire baseline —
  additive. A removed/renamed rpc, or a removed `jammi.v1.*` package, is
  **breaking** — major only, or a `jammi.v2.*` package for a message-shape break.
- A persisted-format version is bumped only when the layout changes; the
  reject-newer guard then makes an old reader fail loud rather than misparse, and
  the recovery is to re-emit (see [Format Stability](./format-stability.md)).

## Experimental surfaces

There are **none**. Every public verb in the conformance sets, every `jammi.v1.*`
rpc, and every persisted-format version above is frozen — none is marked
provisional or experimental, and none ships behind an "unstable" flag. A surface
that is not yet ready to freeze does not appear on the public client at all; it
stays internal until it is ready to enter the frozen set. The freeze is total
across the published surface, which is what the terminal-0.x bar requires.

## Enforcement: the freeze-guard

The freeze is a **CI guard**, not a promise in prose. Two checks run on every PR:

- **The wire contract + manifest version** are pinned in a Rust integration test
  (`crates/jammi-server/tests/it`, the `api_freeze` module). It decodes
  `FILE_DESCRIPTOR_SET` into the live `(Service, Method)` rpc set and the live
  `jammi.v1.*` package set, and asserts they **equal** a committed frozen
  baseline; it also asserts `MANIFEST_VERSION` equals its frozen value. The test
  derives the live surface from the compiled descriptor — the same source the
  server actually serves — so a divergence between the served surface and the
  baseline cannot hide.
- **The verb set** is pinned in the conformance suite
  (`crates/jammi-python/tests/test_conformance.py`), which asserts every verb in
  every set is callable with an identical signature across both transports.

Removing or renaming a stable rpc reds the Rust guard: the live `(Service,
Method)` set decoded from the descriptor no longer equals the committed baseline,
and the assertion fails naming the rpc that disappeared (or the one that
appeared without a baseline update). Removing or renaming a verb reds the
conformance suite the same way. The freeze has teeth because the baseline is a
committed artifact a change must explicitly and visibly edit — and editing it to
drop a stable surface is exactly the breaking change the semver commitment
forbids outside a major.
