# SPEC-03 — Incremental recompute (sensing + action over the materialization contract)

> Part of the [60 — temporal correctness](./README.md) plan group. Builds directly on [`SPEC-02-materialization-contract.md`](./SPEC-02-materialization-contract.md) (the `ProducingDescriptor` / `InputAnchor` / `DefinitionHash` every result table carries) — it reads and re-runs the contract, it does not extend the contract's shape. Research rules: [`../cp9-substrate-primitives/RESEARCH-METHODOLOGY.md`](../cp9-substrate-primitives/RESEARCH-METHODOLOGY.md).
>
> **Status:** Sensing layer **implemented** (W-61a, migration 022). Cache foundation (`CachePolicy` / `CacheOutcome` / `ResultStore::probe_cache` / `JammiError::NotRecomputable`) **implemented** in `jammi-db`. **Producer memoization (§4) implemented (W-61b-cache):** all five result-table producers carry a `CachePolicy` parameter (default `Bypass`) embed==remote, probe at the top of the verb under `Use`, and return a `CacheOutcome`. The `recompute` verb (§5) remains specified here and follows in a sibling workstream.

## 1. Goal

SPEC-02 lets a reader *verify* that a materialised artifact is the output of definition D over input-state S. SPEC-03 answers the next three questions a recompute decision asks, and then provides the two **bounded, explicit** actions a consumer takes on the answers — without the engine ever becoming the loop that decides *when* to act.

The split is the load-bearing design choice, and it is the discipline-test line (§7):

- **Sensing (read-only, W-61a).** Three pure reads over the recorded contract: *Is this artifact still fresh?* (`staleness`), *Has this exact definition-over-inputs already been materialised?* (`lookup_cached`), *What derives from this table?* (`derives_from` / `derives_from_closure`). The sensing layer **reports**; it never mutates and never re-runs a producer.
- **Action (W-61b).** Two bounded operators the consumer invokes **explicitly**: opt-in memoization on the producers (`CachePolicy::Use` → reuse an existing exact materialisation instead of recomputing), and `recompute(table, cascade)` (re-invoke the recorded producer over the inputs' *current* state, optionally sweeping the bounded downstream DAG once). The engine ships the *mechanism* — one sweep on one explicit request. It ships **no** scheduler, no staleness-monitor-triggers-recompute loop, no cache eviction. Re-running the mechanism on a schedule or a sensor→actuator loop is the consumer's composition (a governing platform), built on a published Jammi version.

The engine's own guarantees motivate this. A materialised feature table whose source advanced is silently stale; a downstream conformal calibration built on it inherits the staleness with no signal. The sensing layer makes the staleness *visible*; the action layer makes correcting it a *bounded, observable* operation — never a hidden background process the engine runs on the consumer's behalf.

## 2. Implementation shape

Single capability, landed atomically across the workspace per [`../../CLAUDE.md` — *Atomic across the workspace*](../../CLAUDE.md).

1. **Sensing core** (`crates/jammi-db/src/store/freshness.rs`, W-61a) — `Staleness` / `StaleReason` / `CurrentAnchor` / `DerivesFromEdge`; `ResultStore::staleness` / `lookup_cached` / `current_anchor` / `derives_from`; the stack-safe `derives_from_closure`. Migration `022` indexes `result_tables.definition_hash` for the cache-lookup candidate query. `JammiError::DependencyCycle` for a corrupt (cyclic) lineage.
2. **Action core** (`store/freshness.rs`, W-61b) — `CachePolicy { Use, Bypass }` (default `Bypass`), `CacheOutcome { Computed, Reused { table } }`, `ResultStore::probe_cache` (the `lookup_cached` sensor + an extant-artifact re-confirmation), `JammiError::NotRecomputable { table }`.
3. **Producer probes** (`crates/jammi-ai/src/pipeline/*`, `session.rs`) — each of the five result-table producers gains a `CachePolicy` parameter and a cache probe at the **top** of the verb, before the expensive compute. The producer returns `CacheOutcome` so reuse is observable.
4. **Recompute verb** (`crates/jammi-ai/src/pipeline/recompute.rs`) — `recompute(table, cascade) -> RecomputeReport`, dispatching on the recorded `ProducingDescriptor` to reconstruct the producing verb call over the inputs' current state, optionally sweeping the bounded downstream DAG once.
5. **Surfaces** — `recompute` verb embed == remote (`Session` + gRPC `CatalogService` + `RemoteDatabase` + PyO3 + `_PIPELINE_VERBS`); `CachePolicy` as a parameter on the producer verbs' embed == remote signatures.
6. **Tests** — `tests/it/freshness.rs` (the `probe_cache` + diamond oracle), the producer-cache + recompute integration tests, the tenant-isolation oracle case for the `recompute` RPC, and the cache-hit SLO bench in `jammi-bench`.

## 3. Sensing layer (as built — W-61a)

### 3.1 The three sensors

```rust
// crates/jammi-db/src/store/freshness.rs
impl ResultStore {
    pub async fn lookup_cached(&self, definition: &DefinitionHash, inputs: &[InputAnchor])
        -> Result<Option<String>>;
    pub async fn staleness(&self, table: &ResultTableRecord, current_definition: &DefinitionHash)
        -> Result<Staleness>;
    pub async fn current_anchor(&self, anchor: &InputAnchor) -> Result<CurrentAnchor>;
    pub async fn derives_from(&self, source: &str) -> Result<Vec<DerivesFromEdge>>;
    pub async fn derives_from_closure(&self, source: &str) -> Result<Vec<DerivesFromEdge>>;
}
```

`Staleness` is `Fresh | Stale { reasons } | Undecidable { unpinned, decided_reasons } | MissingManifest`, ordered by confidence. `Fresh` is the only verdict that asserts reuse is safe; every other arm is a reason the reader must decide for itself. The engine ships the sensor, never the policy.

### 3.2 Honest scoping of what resolves *now* (the Q3 correction)

Freshness is only as confident as the inputs are reproducibly identifiable. Of the four `AnchorKind`s, only two have a live current-state surface this engine can read today:

- **`ResultDigest`** — the input is an immutable result table; its *current* anchor is its current artifact digest, read from the input's own manifest. A recomputed parent gets a new digest, so a child anchored on the old one is detected stale by the same per-input comparison — recursion falls out with no special case.
- **`UnpinnedAtInstant`** — an external source with no version surface, anchored only by a read instant. An instant is not a reproducible id, so such an input can never be confidently `Fresh` (it contributes to `Undecidable`) and **never yields a cache hit**.

`MutableVersion` and `SourceVersion` are structurally unreachable in a recorded anchor today and have **no current-resolution surface** (the `mutable_tables` catalog has no monotonic version column to re-read; an external source's as-of column is resolved at scan time, not stored). Rather than fabricate a read against a surface that does not exist, `current_anchor` resolves both to `CurrentAnchor::Undecidable` and documents it. When a producer first anchors a downstream table on a mutable/source version *and* the catalog grows the surface to re-resolve it, these arms gain a live resolution — the comparison shape is already in place.

### 3.3 Lineage and the stack-safe transitive walk

`derives_from(source)` returns the one-hop reverse-dependency edges — every `ready` table whose recorded `input_anchors` name `source`. The lineage is a *view over* `input_anchors_json` (the single source of truth), not a second edge store: an edge exists iff some `ready` table's manifest records `input` as a source.

`derives_from_closure(source)` walks that relation transitively with an **explicit work-stack** and two sets — `on_path` (the active root→node descent path) and `expanded` (nodes whose subtree is fully walked) — never recursion, so an arbitrarily deep chain can never blow the Rust call stack. The two-set distinction is what separates a **diamond** (two paths to the same descendant — walked once, both real edges reported) from a **cycle** (a node re-entered while still on the active path — a corruption surfaced as the typed `JammiError::DependencyCycle`). A materialization lineage is a DAG by construction (a producer anchors its inputs before its output exists), so a cycle is corruption of the recorded anchors, not a caller condition.

## 4. Action layer — cache (opt-in memoization)

### 4.1 The dial and the outcome

```rust
// crates/jammi-db/src/store/freshness.rs
pub enum CachePolicy { Use, Bypass }            // default Bypass
pub enum CacheOutcome { Computed, Reused { table: String } }
```

`CachePolicy` is **opt-in** (default `Bypass`): a producer must never silently hand back a table the caller did not just compute. Surprise reuse is the "honest, not silent" sin — a caller that wanted a fresh run and got a cached one, with no signal, cannot tell the difference. Reuse is therefore both explicitly requested (`Use`) *and* explicitly reported (the producer returns a `CacheOutcome`), never inferred.

### 4.2 The probe placement: top-of-producer, before the expensive compute

The cache probe runs at the **top** of each producer, *before* it scans a row or runs a model — this is where the SLO win is. Under `CachePolicy::Use` the producer:

1. builds its `ProducingDescriptor` and resolves its `InputAnchor`s (both knowable before compute);
2. folds them into the `DefinitionHash` via the same `MaterializationManifest::compute` path the funnel uses, so the probe key is byte-identical to what the funnel would record;
3. calls `ResultStore::probe_cache(definition, inputs)`;
4. on a sound hit, returns `(cached_record, CacheOutcome::Reused { table })` and **skips the compute entirely** — no `create_table`, no Parquet write, so there is no `building` orphan to reap.

`probe_cache` is `lookup_cached` (the exact `(definition_hash, input_anchors)` match) **plus an extant-artifact re-confirmation**: a `ready` catalog row whose Parquet bytes were reaped (a torn write that committed `ready` before durability on a power loss; a half-deleted table) is *not* a sound reuse — the probe falls through to a miss rather than short-circuit to an unreadable table.

### 4.3 Which producers are honestly cacheable

A cache hit requires every input anchor to be reproducibly identifiable. `lookup_cached` short-circuits any requested set containing an `UnpinnedAtInstant` anchor to a miss. Therefore:

| Producer | Sole/leading input anchor | Cacheable? |
|----------|---------------------------|------------|
| Embedding pipeline | `UnpinnedAtInstant` (raw source, no version surface) | **No** — honestly off until sources expose a version surface |
| Inference | `UnpinnedAtInstant` (raw source) | **No** — honestly off |
| Context set | `UnpinnedAtInstant` (raw source) | **No** — honestly off |
| Neighbor-graph | `ResultDigest` (source embedding table) | **Yes** |
| Graph propagation | `ResultDigest` ×2 (embedding table + edge relation) | **Yes** |

The three unpinned producers still gain the `CachePolicy` parameter (for embed==remote parity and so the surface is uniform), but the probe correctly resolves to a miss every time — the cache is honestly *off* there, not silently broken. The two derived producers, anchored on immutable `ResultDigest`s, are genuinely cacheable: the same build over the same parent yields the same output, so reuse is sound.

**As built (W-61b-cache).** The probe key is computed from the *same* fold the funnel records: a producer builds its `ProducingDescriptor` + `MaterializationEnv` + `InputAnchor`s at the top of the verb, derives the `DefinitionHash` via the new public `MaterializationManifest::definition_of(descriptor, env)` (the exact fold `MaterializationManifest::compute` runs at finalize), and calls `ResultStore::probe_cache_record(&def_hash, &inputs)` — `probe_cache` returning the reusable `ResultTableRecord` so a hit hands the record straight back. On `Some(record)` the producer returns `(record, CacheOutcome::Reused { table })` and skips the compute; on `None`/`Bypass` it computes and returns `(record, CacheOutcome::Computed)`. The probe runs **before any `create_table` / Parquet write**, so a hit leaves no `building` orphan — the reap-safe in-funnel re-probe (§4.4) is therefore unnecessary and not built.

**Wire representation.** `CachePolicy` is a proto enum (`jammi.v1.inference.CachePolicy`, `UNSPECIFIED → Bypass`, out-of-range rejected loudly via `cache_policy_from_proto`) carried as a `cache` field on the four producer requests that cross the wire (`InferRequest`, `GenerateEmbeddingsRequest`, `BuildNeighborGraphRequest`, `PropagateEmbeddingsRequest`); `CacheOutcome` is a proto enum carried on `ResultTable` (`cache_outcome`) and `InferResponse`, so reuse is observable over gRPC. The `_PIPELINE_VERBS` conformance guard is unchanged: `CachePolicy` is a *parameter* on existing verbs, not a new verb, so the Python `Database`/`RemoteDatabase` gain a matching `cache` keyword on all four wire verbs (name-for-name parity), no new `_PIPELINE_VERBS` entry. **The ContextSet durable producer (`materialize_context`) is embed-only** — it is not on the gRPC/Python surface (only `assemble_context`, the inline-vector RPC, is), so its `CachePolicy` parameter appears on the Rust embed signature alone.

### 4.4 The reap-safe in-funnel short-circuit (secondary)

The top-of-producer probe is the real compute-saver and creates no orphan. A secondary, belt-and-braces re-probe *inside* `finalize_with_manifest` guards the TOCTOU window: if another writer materialised the same key between the top probe and the finalize, the funnel detects the hit, reaps the redundant just-written `building` bytes via the **same machinery** `recover()` uses (a manifest-less `building` orphan → the existing reaper drives it to `failed`/deleted), and returns the existing table — **no new crash window**. The funnel path only ever saves the digest-read/sidecar-write; the top probe saves the whole compute.

## 5. Action layer — `recompute`

### 5.1 The verb

```rust
// crates/jammi-ai/src/pipeline/recompute.rs
pub enum Cascade { ReportOnly, Downstream }     // default ReportOnly
pub struct RecomputedTable { pub original: String, pub recomputed: String, pub outcome: CacheOutcome }
pub struct RecomputeReport { pub recomputed: Vec<RecomputedTable>, pub downstream_stale: Vec<String> }

impl Session {
    pub async fn recompute(&self, table: &str, cascade: Cascade) -> Result<RecomputeReport>;
}
```

`recompute` dispatches on the table's recorded `ProducingDescriptor` (which is **complete** as of PR-A — every output-affecting determinant is recorded), reconstructs the producing verb call from its recorded **typed** parameters over the inputs' **current** state, and runs it through the unmodified `finalize_with_manifest` (a fresh manifest with fresh anchors; the recompute itself is cacheable). Because the descriptor is faithful, the replay is byte-identical when the inputs have not moved (proven by the non-default-param test, §6).

A pre-contract table (`definition_hash IS NULL`, no descriptor) → typed `JammiError::NotRecomputable(table)` — a loud refusal, never a guessed re-run from the table's columns.

### 5.2 Per-variant dispatch

| `ProducingDescriptor` variant | Reconstructed call |
|-------------------------------|--------------------|
| `Embedding` | `EmbeddingPipeline::run(source_id, model_id, columns, key_column)` from the recorded fields |
| `Inference` | `InferenceSession::infer(source_id, model_id, task, content_columns, key_column)` |
| `NeighborGraph` | `build_neighbor_graph(source_table's source_id, Some(source_table), BuildNeighborGraph { k, min_similarity, mutual, self_exclude, exact, exact_max_rows })` |
| `GraphPropagation` | `propagate_embeddings(PropagateRequest { source_id, embedding_table, edge_source, direction, hops, alpha, weighting, output })` rebuilt from the descriptor + the recorded `edge_source` input anchor |
| `ContextSet` | the `assemble_context`→`materialize_context` **pair** (see §5.3) |
| `AsofJoin` | `asof_join` rebuilt from the recorded join knobs over the spine/facts current state |

### 5.3 ContextSet: re-invoking the assemble→materialize pair

The real ContextSet producer is the `assemble_context`→`materialize_context` **pair** — `materialize_context` is a *sink* that receives pre-pooled rows, so the determinant is the `assemble_context` *recipe* (the `ContextRequest`), not the sink. Recompute reconstructs the `ContextRequest` from the recorded `ContextSet` descriptor (encoder, source, `candidate_source`, `value_columns`, `aggregator`, `exclude_self`, `split`, `dimensions`), re-invokes `assemble_context` to re-pool every target's context over the source's *current* rows, then routes the pooled rows back through `materialize_context`. This is exactly why PR-A recorded the full recipe rather than one target's `query`/`exclude_key` (those vary per target — they are the inputs the recipe runs *over*, and become the output's row keys, not the recipe's definition).

### 5.4 Cascade: ReportOnly vs Downstream — the bounded sweep

- **`ReportOnly`** (default) — recompute the **named** table only; **report** the downstream-stale set (via `derives_from_closure`), recompute none of it. The consumer decides what to do with the report.
- **`Downstream`** — **one** bounded topological sweep on this single explicit request: recompute the named table, then every transitive dependent in topological order (a parent is recomputed before the children that anchor on its now-new digest, so each child senses the advance and replays over fresh inputs). **No poll, no re-check, no second pass after the sweep finishes** — this is the *last* engine surface. Re-running the sweep on a schedule or a monitor is platform, not engine (§7).

The sweep reuses `derives_from_closure`'s **stack-safe** explicit-work-stack walk for both the topological ordering and the cycle guard: a cycle in the DAG → the typed `JammiError::DependencyCycle` (a well-founded DAG; a cycle is corruption). The diamond case (two parents → one shared child) is handled by the `expanded`-set: the shared descendant is collected and recomputed **once**, after both its parents, never twice and never mis-flagged as a cycle.

### 5.5 Typed errors

`recompute` surfaces the existing `JammiError` arms the contract layer already uses: `NotRecomputable { table }` (pre-contract, no descriptor), `DependencyCycle { table }` (cyclic lineage in the Downstream sweep), the manifest-class arms (folded via `manifest_to_jammi`), and `Storage`. A separate `FreshnessError` enum is deliberately *not* introduced: `DependencyCycle` already lives on `JammiError` (where `derives_from_closure` raises it), and a parallel enum would duplicate it and force lossy conversions at every boundary — the right abstraction is one engine error type, extended with the one new arm (`NotRecomputable`).

## 6. Tests (non-vacuous)

- **Cache.** Exact hit → `Reused` and wall-clock orders under the cold producer (the SLO bench, `jammi-bench`); `Bypass` always recomputes; a **one-bit param change → `Computed`, not `Reused`** (proves the probe keys on the full descriptor); an unpinned-anchored producer never hits; a hit whose bytes were reaped falls through to a miss (`probe_cache_misses_when_the_artifact_was_reaped`).
- **Recompute.** Recompute a table built with **non-default** producer params → **byte-identical** output (a default-params test is vacuous — it passes even where the descriptor is lossy; the non-default test is what proves the complete descriptor carries enough to replay faithfully); a stale table → `Fresh` after recompute; recomputing a `Fresh` table is byte-identical; `ReportOnly` reports-but-does-not-recompute-downstream vs `Downstream` clears them; a pre-contract table → `NotRecomputable`; a cache-hit-reaped orphan reconciles identically to a torn write (no new crash window — exercised via the `test-hooks` crash point).
- **Lineage.** The **diamond** (re-converging DAG) test: two parents → one child, the shared descendant collected once, both real edges reported, not mis-flagged as a cycle (`derives_from_closure_collects_a_diamond_descendant_once`).
- **RPC.** The tenant-isolation oracle gains a `recompute` case: tenant B cannot recompute tenant A's table (resolved through the tenant-filtered `get_result_table`); `every_rpc_is_covered` confirms coverage.

## 7. Discipline test — the engine/platform boundary

Run the discipline test on every surface ([`../../CLAUDE.md` — *Engine, not platform*](../../CLAUDE.md)): *would a user who has never heard of any particular consumer reach for this on its own?*

**Engine (mechanism), all in this spec:**

- `CachePolicy::Use` — opt-in memoization of an exact recomputation. A feature store, an attribution chain, and a personal-search index all want "don't redo this exact derivation"; none needs naming.
- `recompute(ReportOnly)` — re-run one named derivation; report what is now downstream-stale.
- `recompute(Downstream)` — one bounded topological sweep on one explicit request. The *last* engine surface.

**Platform (NOT here), the consumer's composition:**

- A scheduled / cron `recompute`.
- A staleness-monitor that *triggers* recompute (a sensor→actuator loop).
- Cache eviction / TTL / size policy.

The line is sharp: **one bounded sweep on one explicit request is engine; re-running it on a schedule or a monitor is platform.** The engine ships the actuator, never the control loop that pulls it. Names no consumer anywhere — in code, config, docs, tests, fixtures, or scripts.

## 8. References

Same evidence-graded methodology as the sibling specs; the design rests on the in-repo contract (SPEC-02) rather than external behaviour claims, so the load-bearing citations are internal:

- The complete `ProducingDescriptor` (every output-affecting determinant recorded) — [`SPEC-02-materialization-contract.md`](./SPEC-02-materialization-contract.md) and the PR-A amendment closing the lossy-hash defect.
- The crash-consistent `building → ready` funnel and the reaper the in-funnel short-circuit reuses — `crates/jammi-db/src/store/mod.rs` (`finalize_with_manifest`, `recover`, `reconcile_ready_manifests`).
