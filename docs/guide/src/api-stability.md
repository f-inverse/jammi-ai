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
signature-for-signature across the embedded (`jammi.EmbeddedBackend`) and remote
(`jammi.RemoteDatabase`) transports. It is pinned, set-by-set, in
`crates/jammi-python/tests/test_conformance.py`; those sets **are** the frozen
verb list:

| Verb set (conformance constant) | Verbs |
|---|---|
| `_REMOTE_VERBS` | `add_source`, `generate_embeddings`, `encode_query`, `search`, `sql`, `list_sources`, `describe_source`, `set_tenant`, `tenant_scope`, `tenant`, `get_server_info` |
| `_TRAINING_VERBS` | `fine_tune`, `fine_tune_graph`, `train_context_predictor`, `predict_with_context_predictor`, `training_job`, `list_training_jobs` |
| `_INFERENCE_VERBS` | `infer` |
| `_PIPELINE_VERBS` | `build_neighbor_graph`, `propagate_embeddings`, `asof_join`, `assemble_context`, `recompute`, `verify_materialization`, `staleness`, `derives_from` |
| `_EVAL_VERBS` | `eval_embeddings`, `eval_per_query`, `eval_inference`, `eval_compare`, `eval_calibration` |
| `_CHANNEL_VERBS` | `register_channel`, `add_channel_columns`, `list_channels` |
| `_NUMERIC_VERBS` | `conformalize`, `conformalize_interval`, `conformalize_cqr`, `rrf_fuse` |
| `_MUTABLE_TOPIC_VERBS` | `create_mutable_table`, `drop_mutable_table`, `list_mutable_tables`, `register_topic`, `drop_topic`, `list_topics`, `publish_topic`, `subscribe_collect` |
| `_LIFECYCLE_VERBS` | `list_models`, `describe_model`, `delete_model` |
| `_SEGMENT_VERBS` | `list_index_segments` |
| `_SEARCH_VERBS` | `search` (pinned separately for the `embedding_table=` selector) |

The conformance suite is the *enforced annotation*: removing or renaming a verb,
or changing its signature on either transport, reds the suite. Jammi does not
carry a per-`pub`-item `#[stable]` rustdoc attribute — Rust has no such
attribute, and history-bearing version markers in rustdoc are explicitly
disallowed — so the conformance sets carry the freeze that a `#[stable]` pass
would carry elsewhere.

### 2. The wire contract — `package jammi.v1.*`

The gRPC/Flight SQL wire surface is the twelve `jammi.v1.*` proto packages (ten
served on the public listener; one, `jammi.v1.peer`, served **only on the
internal `[server] peer_bind` listener** — the public listener answers
`UNIMPLEMENTED` for its rpcs; `jammi.v1.lifecycle` is a **contract-only**
surface — defined in the wire descriptor so the candle-free client can call a
platform server that implements it, but answered by no OSS handler):

| Package | Surface |
|---|---|
| `jammi.v1.audit` | provenance / audit log rpcs |
| `jammi.v1.catalog` | sources, models, channels, tenant, server-info, mutable tables, topics |
| `jammi.v1.embedding` | embedding generation, query encode, search |
| `jammi.v1.error` | the typed wire-error message (no rpcs) |
| `jammi.v1.eval` | the evaluation rpcs |
| `jammi.v1.inference` | bulk inference + predict |
| `jammi.v1.job` | the durable job queue: submit / status / wait / list / cancel / list-workers / prune (`JobService`) |
| `jammi.v1.lifecycle` | license apply / bootstrap / status / login — **contract-only**, answered by a platform server (the OSS engine returns `UNIMPLEMENTED`) |
| `jammi.v1.peer` | the engine-internal segment-search seam between replicas (`PeerService.SegmentSearch` / `ExactRescore`) — served **only on `peer_bind`**, never on the public listener; deliberately tenant-free (the coordinator enforces tenant scope; see [Security Posture](./security.md#the-peer-listener-i-peer)) |
| `jammi.v1.pipeline` | graph / context / as-of / recompute / materialization rpcs |
| `jammi.v1.training` | the training spec message vocabulary `JobService.SubmitJob`'s oneof carries (`FineTuneSpec`/`GraphFineTuneSpec`/`ContextPredictorSpec`/`FineTuneConfig`/…) — no rpcs of its own since `TrainingService` folded into `JobService` |
| `jammi.v1.trigger` | topic publish + subscribe |

The contract is the full set of `(Service, Method)` rpc paths these packages
*define* — decoded from the compiled `FILE_DESCRIPTOR_SET`, the authoritative
machine-readable description of the frozen wire surface, which may exceed what a
given build mounts (`jammi.v1.lifecycle` is defined here yet served by no OSS
handler), not a hand-maintained list. The `v1` in the package path is the wire-stability stamp:
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
| ANN sidecar manifest (`.manifest.json`) | `ANN_MANIFEST_VERSION` | `3` |
| Catalog schema | append-only migration ledger | see `crates/jammi-db/src/catalog/migrations.rs` |

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

**Pre-1.0 amendment (#485).** Until the 1.0 release, the `jammi.v1` wire
contract may still change — including a genuinely breaking rpc rename or
removal — when BOTH of the following hold in the same PR: the frozen baseline
(`crates/jammi-server/tests/it/api_freeze_baseline.txt`) is updated to match
the new live surface, and the CHANGELOG carries an explicit **Breaking**
entry describing the change and its migration. The freeze-guard test still
enforces that the live surface and the committed baseline agree exactly — the
amendment relaxes only WHICH edits to the baseline are allowed pre-1.0 (an
announced breaking edit, not a silent one), never the mechanism that catches
an *unannounced* divergence. The following release then bumps the **minor**
version rather than the major — the terminal-0.x window is itself the
"stabilizing" period the eventual 1.0 major bump closes. `TrainingService`
folding into `JobService` (`StartTraining`→`SubmitJob`,
`TrainingStatus`→`JobStatus`, `ListTrainingJobs`→`ListJobs`, plus new
`WaitJob`/`CancelJob`/`ListWorkers`/`PruneJobs`) is the first change to ship
under this amendment.

## Experimental surfaces

There are **none**. Every public verb in the conformance sets, every `jammi.v1.*`
rpc, and every persisted-format version above is frozen — none is marked
provisional or experimental, and none ships behind an "unstable" flag. A surface
that is not yet ready to freeze does not appear on the public client at all; it
stays internal until it is ready to enter the frozen set. The freeze is total
across the published surface, which is what the terminal-0.x bar requires.

## Published-crate Rust APIs

Eleven workspace crates lack `publish = false` and are therefore published
Rust crates: `jammi-admin`, `jammi-ai`, `jammi-cli`, `jammi-client`,
`jammi-db`, `jammi-encoders`, `jammi-kernels`, `jammi-lora`, `jammi-numerics`,
`jammi-server`, and `jammi-wire`. Their public items (types, functions, trait
signatures, struct field visibility) are a real compile-time surface for any
consumer outside this workspace, distinct from the three CI-enforced surfaces
above. This surface carries **no CI freeze guard** — there is no descriptor to
decode or conformance set to pin a bare Rust signature against — so a breaking
change here is caught only by review, and is recorded as a **BREAKING** entry
in the CHANGELOG the same way every other breaking change in this workspace
is, naming the item, what changed, and what the caller does instead.

**Deriving this section.** Do not hand-maintain this list from memory — a
prior round of this same section missed at least seven breaking changes,
including a `pub fn` removed with no entry anywhere, and its own scope
sentence named 2 of these 11 crates. Enumerate every public item whose
signature, visibility, or existence changed across the range instead. **The
range is a set of commits, not a contiguous `<base>..<head>` span**: this
unit's commits are interleaved on the branch with other units' commits (a
plain `git diff <base> <head>` between the oldest and newest of this unit's
own commits also picks up whatever any OTHER unit changed in between — that
is how a sibling unit's own unannounced removal was mistaken for this
section's gap in a prior round). Derive the exact commit set from the
commit-message tag every round of this unit's own history carries, and diff
the UNION of those commits, never a span:

```
# Every commit belonging to THIS unit, oldest first. NOT a literal
# '#482 DIST' substring match: this unit's own history also carries the tag
# as 'DIST-1' and 'DELTA/DIST' (a hyphen or a slash immediately after
# 'DIST', never a space) — a plain `--grep='#482 DIST'` silently drops those
# and under-ranges the set (measured, DIST round 8: it returns 3 commits and
# misses 3 more, including the one that introduced the sites a round-8 fix
# corrected). `-E --grep='#482.*DIST'` matches all three spellings because
# it does not require a space between the issue number and the tag:
commits=$(git log --oneline --reverse -E --grep='#482.*DIST' | cut -d' ' -f1)
for c in $commits; do
  git diff --name-only "$c"^.."$c" | grep '^crates/.*/src/.*\.rs$'
  # for each changed file, diff its `pub fn|struct|enum|trait|type|const|
  # static|use` items between "$c"^ and "$c" — a struct/enum/trait's full
  # brace-balanced body, a fn/const/static/type/use's signature up to its
  # body or `;` — and treat a normalized-text change as added-old +
  # added-new (catches a removal with no replacement, not just a same-line
  # diff hunk).
done
# Cross-check the crate list above against every `crates/*/Cargo.toml`
# lacking `publish = false`.
```

Nine commits are current state as of this release (the range this section
has covered started as three, was five, was six, was seven, was eight as of
DELTA round 6's own fold, and is nine as of DIST round 7's own fold — state
the true count rather than repeating a stale one). The corrected recipe
above resolves to SIX commits (`3a696a65`, `cbd427b4`, `320b73ee`,
`f37cb743`, `b1665c14`, `a2dfcb1f`) — three more than the three the old,
narrower grep found — but the three added (`320b73ee`'s re-apply, `cbd427b4`'s
formatting-only fmt, `3a696a65`'s masked-load reuse with no new `pub` item)
introduce no public-surface change beyond what the bullets below already
list; checked by diffing each for an added/removed/changed `pub` item
against the crate list above, not assumed.

- **The vector-search API takes a validated query type, not a bare slice.**
  `jammi_numerics::query::ValidatedQuery` is the only type
  `jammi_numerics::distance::{cosine_distance, cosine_similarity}`,
  `jammi_db::index::VectorIndex::search`,
  `jammi_db::index::segment::{search_unit, rescore}`,
  `jammi_db::index::segment::SegmentedIndex::{search, search_final}`,
  `jammi_db::index::exact::exact_vector_search`,
  `jammi_db::index::placed::PlacedIndex::search_final_placed`,
  `jammi_db::store::ResultStore::{search_vectors, search_vectors_local}`,
  `jammi_ai::operator::ann_search_exec::AnnSearchExec::new` (and its
  `query_vector` field), and `jammi_ai::pipeline::neighbor_graph::Node`'s
  `vector` field accept for a query vector.
  `jammi_db::index::segment::verify_query_width` (a free `pub fn`) is
  **removed** with no replacement — its check is now
  `ValidatedQuery::require_width`/`require_authority_width`, methods on the
  type itself, not a function a caller could import.
  Construct a `ValidatedQuery` with `jammi_db::index::validate_query(values,
  expected_width, source)` (re-exported from `jammi_numerics::query`, along
  with the new `jammi_numerics::query::QueryValidationError` error type),
  where `source` is a `jammi_db::index::QuerySource::{Caller, Stored {
  table }}`. Its inherent methods are `as_slice`, `into_inner`, `source`, and
  the two width checks below.
  `exact_vector_search` also gained a `catalog_dimensions: Option<usize>`
  parameter (a cross-check against the scan's own width; `None` when there is
  none on record). `jammi_db::index::peer::{SegmentSearchRequest,
  ExactRescoreRequest}`'s `query` field is a `ValidatedQuery`, not a
  `Vec<f32>`; `PeerFailureReason` gained the `CallerFault` variant,
  `PeerError` gained a `message: String` field, and `PEER_FAILURE_LABELS` is
  `[&str; 10]`.
- **`jammi_db::catalog::result_repo::ResultTableRecord::dimensions` is a
  method, not a field.** It returns `Option<std::num::NonZeroUsize>` — a
  non-positive stored value and an absent one are both `None`.
  `dimensions_raw() -> Option<i32>` returns the stored column verbatim, for a
  caller that must round-trip it unfiltered. A caller that built a
  `ResultTableRecord` field-by-field from outside this crate now goes through
  `ResultTableRecord::from_wire_projection`, the crate's sole cross-crate
  constructor.
- **`jammi_wire::peer::{phase_from_proto, precision_from_proto}` return a
  `ProtoEnumDecode<T>`, not an `Option<T>`.** The wire's explicit "not set"
  (`ProtoEnumDecode::Unspecified`) and a raw value this build's generated
  `enum` has no variant for (`ProtoEnumDecode::Unknown`) are no longer
  collapsed into one `None` — a caller that only needs "did this decode"
  calls the new `.known() -> Option<T>` to get the old behaviour back.
- **Whose-fault a downstream width check assigns no longer depends on the
  query's own provenance.** `ValidatedQuery::require_width` now takes an
  `artifact: impl Into<String>` and returns a new error variant,
  `QueryValidationError::ArtifactMismatch { artifact, expected, actual }`,
  which carries no `QuerySource` at all — a mismatch it finds is always
  attributed to the named artifact, never the caller, because by the time a
  query reaches any consumer an entry has already checked it once against an
  authority it had in hand. THREE call sites still attribute by the query's
  own provenance (the placement entry's all-remote shape, the placement
  entry's all-local shape against the set's own first segment, and
  `exact_vector_search`'s no-catalog-width fallback — round 8 closed the
  all-local gap, where the same deferral the other two already performed
  had never run), each checking a query with no width in hand against the
  only authority available to that call; all three use the new
  `ValidatedQuery::require_authority_width` instead, which keeps the OLD
  `require_width` behaviour under a name that says why it is different.
  `QuerySource::source()` on `QueryValidationError` now returns
  `Option<&QuerySource>` (`None` for `ArtifactMismatch`) rather than
  `&QuerySource` unconditionally. (DIST round 6/7 shipped a third
  `QuerySource` variant, `Artifact { name }`, to carry this same
  information; round 8 removed it in favor of the dedicated
  `ArtifactMismatch` error variant above, so `QuerySource` is back to
  exactly two variants, `Caller` and `Stored` — the states a query's own
  provenance can actually be. A round-6/7 caller matching on three
  `QuerySource` variants needs the arm removed, not added.) `JammiError::Schema`
  constructions in `jammi_db::index::exact::exact_vector_search`,
  `jammi_db::index::placed::PlacedIndex::search_mixed`,
  `jammi_db::store::vectors::{extend_with_fixed_size_list_f32,
  extend_with_keyed_fixed_size_list_f32}`, and `jammi_db::store::deletes::
  DeletionMask::read` that priced an ENGINE-owned artifact's own corruption
  as the caller's fault are now `JammiError::IncompatibleFormat`; the two
  `store::vectors` functions changed their error type to the new,
  provenance-neutral `jammi_db::store::vectors::VectorColumnError` (`?`
  converts it to `IncompatibleFormat` by default; `.into_caller_fault()` is
  the explicit override the one caller-supplied read path, behind
  `import_embeddings`, now uses).
- **`jammi_db::store::ResultStore::result_digest_anchor` is removed with no
  replacement.** It resolved a result table's current version and then
  discarded the resolution, returning a bare `InputAnchor` a caller could
  not get the matching content back from without a second, independent
  resolve — a version publish landing between the two could straddle.
  Call `ResultStore::pin_current_version(record).await?.input_anchor()`
  instead; the versioned arm already delegated to exactly that internally,
  so the returned value is unchanged. (This item predates the compute-tier
  substrate epic — it shipped in an earlier release — so its removal here
  is a genuine breaking change to an already-released surface, not internal
  churn within this epic's own unreleased history; the two DELTA-round
  functions that went `pub` → private/`pub(crate)` entirely within this
  epic's own unreleased commits, `current_version_provider` and
  `current_version_identity`, never shipped as `pub` in any release and so
  carry no such entry.)

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
