# SPEC-01 — As-of temporal join

> Part of the [60 — temporal correctness](./README.md) plan group. Independent of [`SPEC-02-materialization-contract.md`](./SPEC-02-materialization-contract.md) except at the shared result-store write path (see [`README.md` — Concurrent-session strategy](./README.md)). Research rules: [`../cp9-substrate-primitives/RESEARCH-METHODOLOGY.md`](../cp9-substrate-primitives/RESEARCH-METHODOLOGY.md).
>
> **Status:** Proposed (draft).

## 1. Goal

An *as-of temporal join* matches each row of a left ("spine") relation to **at most one** row of a right ("facts") relation — the fact that was most recently valid *at or before* the spine row's as-of instant, within the same entity group. It is the relational primitive for **point-in-time-correct** data assembly: building a labelled set in which every feature reflects only what was knowable as of the label's timestamp, with no leakage of future information. The engine exposes it as one verb, `asof_join`, that reads two registered relations and materialises a point-in-time-correct result table; the verb is a thin wrapper over a reusable `AsofJoinExec` physical operator (a sort-merge join, partitioned by the equality keys, that the engine's other temporal paths can reuse). The primitive carries only what every time-aware consumer needs — an equality-key grouping, a temporal ordering key, a match direction, boundary inclusivity, an optional look-back tolerance, and a deterministic tie-break — and **no** domain vocabulary: not "feature," not "label," not "entity," not "store." A backtester matching trades to prevailing quotes, a clinical fabric matching observations to the lab value in effect at the time, and an ML pipeline assembling a leakage-free training set all reach for the same operator (§10).

This is a correctness primitive the engine owes itself independent of any consumer: the engine's conformal coverage guarantee assumes exchangeability of calibration and deployment scores, and future-information leakage breaks that assumption and inflates apparent coverage[^cpbe]. The engine's only existing temporal-pin surface is narrow and bespoke: `EdgeGather::as_of` / `EdgeSourceRef::as_of_column` (`crates/jammi-ai/src/pipeline/graph_neighbourhood.rs`) emit a backward-inclusive predicate as a *string-cast* `WHERE` clause (`arrow_cast(col,'Utf8') <= 'asof'`) scoped to edge loads — a lexicographic comparison, correct only for ISO-8601-shaped strings. §6.3 routes that pin through this spec's typed comparison so the engine has *one* as-of semantics, not a string filter in one path and a typed operator in another.

## 2. Concurrent-session strategy

Single-capability spec; smaller than a cp9 phase. One sequential scaffold, then two parallel subagents, then integration.

### 2.A — Sequential scaffolding (~45 min, main session)

Files every subsequent subagent reads but does not modify:

1. `crates/jammi-ai/src/pipeline/asof/mod.rs` — pub-uses + module declarations.
2. `crates/jammi-ai/src/pipeline/asof/spec.rs` — `AsofJoinSpec`, `AsofKey`, `MatchDirection`, `Boundary`, `Tolerance`, `TieBreak`, `AsofError` (the frozen contract, §3).
3. `crates/jammi-ai/src/pipeline/asof/exec.rs` — `AsofJoinExec` struct + `ExecutionPlan` impl with method bodies `todo!()`.
4. `crates/jammi-ai/src/pipeline/asof/merge.rs` — `fn merge_partition(...)` signature only (the per-group sort-merge core).

Phase A ends when `cargo check -p jammi-ai` passes against the stubs.

### 2.B — Two parallel subagents (~3 h wall-clock)

Dispatch in a single message:

| Subagent | Files written | Responsibility | Independent test |
|---|---|---|---|
| **Operator subagent** | `pipeline/asof/exec.rs`, `pipeline/asof/merge.rs` | `AsofJoinExec`: child plans, repartition-by-equality-keys, per-partition sort, the single-pointer merge in `merge.rs` | `crates/jammi-ai/tests/it/asof_merge.rs` (pure-`RecordBatch` cases, no catalog) |
| **Verb subagent** | `pipeline/asof/verb.rs`, session integration in `session.rs` | `asof_join` verb: resolve relations, build `AsofJoinSpec`, plan + run the operator, write the result table | `crates/jammi-ai/tests/it/asof_verb.rs` |

**Coordination contract** (frozen in Phase A, both read it identically): `merge_partition(left: &SortedPartition, right: &SortedPartition, spec: &AsofJoinSpec) -> Result<RecordBatch, AsofError>` and the `AsofJoinExec::try_new(left, right, spec)` signature in §3.

### 2.C — Sequential integration (~60 min, main session)

1. **DEFERRED** (tracked follow-up — see §6.3): route `EdgeGather::as_of` through a *shared, exported* `Boundary`/comparison helper. Two facts make this a real follow-up, not a one-line call-through: (a) **no open-core consumer reaches the pin** — `gather.as_of` is set to `Some` only by an internal replay (`recompute.rs` carrying an already-recorded value) and a manifest test fixture; no verb or public surface sets it, so the string-cast `WHERE` at `graph_neighbourhood.rs` is unreachable with a non-empty pin in open-core. (b) **the typed comparator is not yet a shared helper** — `eligible_at_or_before` / `temporal_i128` are *private free fns* in `pipeline/asof/merge.rs`, not an exported comparator the edge-gather path can call; §6.3 therefore requires *extracting and exporting* a shared comparator first, not merely pointing the edge predicate at an existing one. So the work carries its own rigor unit (pressure-test + audit + the lexicographic-vs-typed disagreement test of exit-criterion #6) rather than riding the new-module work.
2. Add `asof_join` to the embedded `Database` and the remote `RemoteDatabase`, then to `_PIPELINE_VERBS` in `crates/jammi-python/tests/test_conformance.py` (§7).
3. `cargo clippy --all-features -- -D warnings`, `cargo fmt --check`. No `#[allow(...)]`.

## 3. Public API surface (exhaustive)

Every signature is the frozen contract. Comments explain hidden invariants only.

### 3.1 `AsofKey` and join descriptor (`pipeline/asof/spec.rs`)

```rust
use arrow_schema::SchemaRef;

/// One side's column roles for the as-of match. The temporal key must be a
/// totally-ordered Arrow type (any `Timestamp(..)`, `Date32/64`, or signed/
/// unsigned integer); the engine rejects float temporal keys — NaN has no
/// total order, so "most recent at or before" would be undefined.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsofKey {
    /// Equality ("by") columns that partition the match into independent
    /// groups — e.g. an entity id, an instrument symbol, a subject id.
    /// May be empty: an empty `by` matches across the whole relation
    /// (a single global group), which is occasionally what a user wants
    /// (one global calendar of facts). Never silently defaulted.
    pub by: Vec<String>,
    /// The temporal ordering column. Required; exactly one.
    pub time: String,
}
```

### 3.2 `AsofJoinSpec` (`pipeline/asof/spec.rs`)

```rust
/// The frozen descriptor an `asof_join` lowers to. Construct via
/// `AsofJoinSpecBuilder` (>3 parameters → builder, per CLAUDE.md
/// *Type-driven design*).
#[derive(Debug, Clone)]
pub struct AsofJoinSpec {
    pub left: AsofKey,
    pub right: AsofKey,
    /// Which side of the spine instant the matched fact may fall on (§5.1).
    pub direction: MatchDirection,
    /// Whether a fact whose time exactly equals the spine instant matches (§5.2).
    pub boundary: Boundary,
    /// Optional maximum look-back/forward distance; a candidate outside it is
    /// treated as no-match (the spine row is preserved, fact columns null) (§5.3).
    pub tolerance: Option<Tolerance>,
    /// How coincident candidate facts are disambiguated into one match (§5.4).
    pub tie_break: TieBreak,
    /// Right-side columns to attach to the output. Empty = all non-key columns.
    pub project: Vec<String>,
}

/// Backward = most recent fact at/before the spine instant (the default and
/// the only leakage-safe choice for past-label assembly). Forward = first fact
/// at/after. Nearest = smallest absolute distance (ties resolved toward the
/// past). `Nearest` is rejected when the temporal key is non-numeric, matching
/// the constraint Polars documents for string keys[^polars].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchDirection { Backward, Forward, Nearest }

/// Inclusive = a fact whose time equals the spine instant is eligible
/// (`<=` / `>=`); Exclusive = strict (`<` / `>`). Mirrors pandas
/// `allow_exact_matches`[^pandas] and DuckDB's `>=` vs `>` interval rule[^duckdb-blog].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary { Inclusive, Exclusive }

/// Look-back/forward limit. `Duration` for temporal keys (microseconds),
/// `Steps` for integer keys. A candidate beyond it is no-match — the
/// engine's analog of a feature-store TTL, which Feast defines as measured
/// "relative to each timestamp within the entity dataframe", never relative
/// to wall-clock now[^feast-pit].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tolerance { Duration(i64), Steps(i64) }
```

### 3.3 Tie-break — the determinism contract (`pipeline/asof/spec.rs`)

```rust
/// When more than one right row shares the matched temporal value within a
/// group, the match is ambiguous unless disambiguated. Silent
/// non-determinism here is a known footgun (ClickHouse returns
/// query-range-dependent rows for duplicate asof timestamps[^ch-9906]); this
/// engine refuses it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TieBreak {
    /// Break ties by a secondary column, newest value winning — the
    /// transaction-/created-time column. This is the standard second
    /// timestamp: event time bounds the join, transaction time disambiguates
    /// late-arriving facts (bitemporal valid-time vs transaction-time[^bitemporal];
    /// Feast dedups coincident event timestamps "by the created timestamp,
    /// with newer values taking precedence"[^feast-retrieval]).
    ByColumnDesc(String),
    /// No secondary column. If a true duplicate remains (same `by`, same
    /// `time`), the verb fails with `AsofError::AmbiguousMatch` rather than
    /// pick non-deterministically. Correct-or-loud, never quiet-and-wrong.
    Error,
}
```

### 3.4 `AsofError` (`pipeline/asof/spec.rs`)

```rust
#[derive(Debug, thiserror::Error)]
pub enum AsofError {
    #[error("temporal key `{column}` has type {found}, which is not totally ordered; expected a timestamp, date, or integer")]
    UnorderedTimeKey { column: String, found: String },
    #[error("equality key `{column}` not found in {side} schema")]
    MissingByKey { column: String, side: &'static str },
    #[error("left and right temporal keys differ in type: {left} vs {right}")]
    TimeKeyTypeMismatch { left: String, right: String },
    #[error("`Nearest` direction requires a numeric temporal key; `{column}` is {found}")]
    NearestRequiresNumeric { column: String, found: String },
    #[error("ambiguous match: group has duplicate facts at the matched instant and no tie-break column was given")]
    AmbiguousMatch,
    #[error(transparent)]
    DataFusion(#[from] datafusion::error::DataFusionError),
}
```

### 3.5 `AsofJoinExec` (`pipeline/asof/exec.rs`)

```rust
use std::sync::Arc;
use datafusion::physical_plan::ExecutionPlan;

/// Sort-merge as-of join. Repartitions both children by the hash of the
/// equality keys (so each partition holds one or more whole `by`-groups),
/// requires each child sorted by (`by`..., `time`), and merges with a single
/// advancing pointer per group — the canonical algorithm every production
/// engine uses, where "at most one match" lets the merge stop at the first
/// hit[^duckdb-blog].
#[derive(Debug)]
pub struct AsofJoinExec { /* children, spec, cached plan properties */ }

impl AsofJoinExec {
    pub fn try_new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        spec: AsofJoinSpec,
    ) -> Result<Self, AsofError>;
}
// impl ExecutionPlan: required_input_distribution = HashPartitioned(by-keys)
// on both sides; required_input_ordering = (by..., time); maintains_input_order
// = false; output is left rows ⟕ matched right projection (left always preserved).
```

The operator declares its partitioning/ordering requirements and lets the DataFusion planner insert the `RepartitionExec` + `SortExec` it needs — no bespoke shuffle, reusing the same infrastructure the rest of the engine already plans against[^df-execplan].

### 3.6 Verb surface (`session.rs`, mirrored on `RemoteDatabase`)

```rust
/// Assemble a point-in-time-correct table: for each row of `spine`, attach the
/// `facts` row that was valid as-of the spine row's temporal key, within each
/// equality group. Writes a result table (carrying the SPEC-02 manifest) and
/// returns its handle. Left rows are always preserved; unmatched fact columns
/// are null.
pub async fn asof_join(
    &self,
    spine: RelationRef,
    facts: RelationRef,
    spec: AsofJoinSpec,
) -> Result<ResultTableHandle, JammiError>;
```

`RelationRef` is the engine's existing relation reference (a registered source/result table, optionally with a projection) — the same type the pipeline verbs (`build_neighbor_graph`, `propagate_embeddings`) already accept. No new relation concept is introduced.

## 4. Catalog schema changes

None. `asof_join` writes through the existing result-table path; its only persisted metadata is the SPEC-02 manifest, which owns migration `021`. If SPEC-01 lands before SPEC-02, the verb writes a result table with the current manifest and SPEC-02's migration backfills the contract fields — but the [`README.md` ordering](./README.md) pins SPEC-02 first to avoid that retrofit.

## 5. Semantics — the pinned decisions

These are the choices every engine gets subtly different; the engine fixes them once, loudly.

### 5.1 Direction

`Backward` (default): the matched fact's time is ≤ (Inclusive) or < (Exclusive) the spine instant — "most recent at or before." This is the only direction safe for assembling a set keyed on past labels, because any forward match imports a fact from after the label, i.e. leakage[^feast-pit]. `Forward` and `Nearest` exist for symmetric/non-leakage use cases (e.g. "the next maintenance event after each reading") and are never the default. `Nearest` resolves equidistant candidates toward the past (backward wins) and requires a numeric key (§3.4).

### 5.2 Boundary

`Inclusive` (default, `<=`): a fact stamped exactly at the spine instant matches. `Exclusive` (`<`): it does not. This is the single most error-prone as-of decision — DuckDB encodes it as the difference between the half-open intervals `[Tn, Tn+1)` and `(Tn, Tn+1]`[^duckdb-blog]; pandas exposes it as `allow_exact_matches`[^pandas]. Pin it on the spec, never infer it. Default `Inclusive` matches the "fact known *as of* T includes facts recorded at T" reading.

### 5.3 Tolerance

`None` (default): the search looks back arbitrarily far. `Some(Duration|Steps)`: a candidate farther than the limit is discarded and the spine row goes unmatched (fact columns null) — never matched to a stale fact. Semantics mirror a feature-store TTL/look-back window measured *relative to each spine instant*, not wall-clock now[^feast-pit] (the cross-engine `tolerance` knob in Polars[^polars] and pandas[^pandas]).

### 5.4 Ties, nulls, and preservation

- **Ties.** Multiple facts at the matched instant within a group are resolved by `TieBreak` (§3.3): a transaction-time column (newest wins) or a loud `AmbiguousMatch` error. No silent pick.
- **Nulls.** A null temporal key cannot be ordered, so such rows are excluded from matching: a null-time **spine** row is preserved with null fact columns; a null-time **fact** row is never a candidate. A null **equality** key never matches another null (SQL `NULL ≠ NULL`).
- **Left preservation.** The spine is always fully preserved (a left outer as-of). Dropping unmatched spine rows would silently shrink a labelled set and bias downstream metrics; if a consumer wants inner semantics they filter on a non-null fact column themselves.
- **Sortedness.** The operator never assumes pre-sorted input — it declares the ordering requirement and the planner satisfies it. A hand-rolled merge over unsorted input is the classic as-of bug and is structurally impossible here.

## 6. Operator design & wiring

### 6.1 Plan shape

`asof_join` builds a `LogicalPlan` whose extension node lowers (via an `ExtensionPlanner`) to `AsofJoinExec(left, right, spec)`. The planner sees the operator's `required_input_distribution` (hash-partitioned on the equality keys) and `required_input_ordering` ((by..., time)) and inserts `RepartitionExec` + `SortExec` as needed[^df-execplan]. Output schema = the spine schema followed by the projected fact columns, all fact columns nullable (left-outer).

### 6.2 The merge core (`merge.rs`)

Per group (one hash partition may hold several groups; the merge resets at each `by` boundary): two cursors over the sorted runs. For `Backward`/`Inclusive`, advance the fact cursor while `fact.time <= spine.time`, remembering the last eligible fact; emit it (or null) when the spine cursor advances. `Exclusive` uses `<`. `Forward` mirrors. `Nearest` keeps the straddling pair and picks by absolute distance. Because there is at most one match, the cursor never backtracks — O(n+m) per group after the sort. This is the DuckDB/kdb/ClickHouse sort-merge shape[^duckdb-blog][^kdb-aj].

### 6.3 Unifying the edge-gather pin — DEFERRED to a tracked follow-up

> **Status: DEFERRED — NOT shipped in this spec's delivery.** The `AsofJoinExec` operator and its merge core shipped, but the edge-gather unification did not. The deferral is the correct call, and the accurate blocker is two-fold:
>
> 1. **No open-core consumer reaches the pin.** `EdgeGather::as_of` is set to `Some` only by an internal replay path — `recompute.rs` carries an already-recorded value back onto a rebuilt `EdgeGather` — and a manifest test fixture. No verb, no PyO3 binding, no remote surface sets it. So the string-cast `WHERE` predicate at `graph_neighbourhood.rs` is **unreachable with a non-empty pin in open-core**: the `(Some(col), Some(asof))` match arm never fires from a consumer-driven call. The latent lexicographic bug is real but currently undriveable, which removes the urgency that would justify riding it on the new-module work.
> 2. **The typed comparator is not yet a shared, exported helper.** What §6.3 below calls "SPEC-01's typed comparison helper" is, as shipped, the **private free fns** `eligible_at_or_before` and `temporal_i128` in `pipeline/asof/merge.rs` — not `pub`, not exported, reachable only inside the merge core. Routing the edge-gather pin through "the shared comparator" therefore is **not a call-through to an existing exported helper**; it first requires *extracting and exporting* a shared comparator (lifting `eligible_at_or_before` + the `temporal_i128` normalization out of `merge.rs` into a shared, `pub` surface both the merge and the edge-gather predicate call), then rewriting the edge predicate against it. That extraction is a distinct change with its own failure modes, so it owns its own rigor unit (pressure-test → implement → independent audit → CI), including the lexicographic-vs-typed disagreement test of exit-criterion #6, rather than riding the new-module work.
>
> The string-cast pin remains in place until that follow-up lands. The paragraph below describes the *intended* end state, not the current one.

`EdgeGather::as_of` + `EdgeSourceRef::as_of_column` (in `graph_neighbourhood.rs`) today implement a backward-inclusive pin as a hand-built SQL string: `WHERE arrow_cast("col",'Utf8') <= 'asof'`. That is a *lexicographic* comparison — it happens to work for ISO-8601 timestamps but is silently wrong for any temporal key whose string order differs from its value order (e.g. unpadded integers, or `Date32` rendered without zero-padding). The intended end state is a single, type-correct source of "at or before" semantics — a shared `Boundary`/comparison helper *extracted and exported* from the now-private `merge.rs` free fns (above) — that both the relational join and the edge-gather predicate call, so the engine has one definition of "as of," and the more-correct one. This is the *right-abstraction* fix from [`../../CLAUDE.md` — *Clean, functional style*](../../CLAUDE.md) ("two things that are the same thing at a different scale are one thing"): the edge pin and the relational join are the same comparison at different scales.

A second instance of the same "records-but-ignores, no consumer reaches it" pattern lives in `propagate_embeddings`: its producing descriptor faithfully records the edge source's `as_of_column` in the `EdgeSourceRef::Registered` binding (via `to_binding`), but its `edge_scan_sql` (in `graph_propagation.rs`) destructures that binding with `..` and emits a scan with **no** as-of `WHERE` clause — the recorded pin is silently ignored on the propagate edge read. This is a low-severity records-but-ignores inconsistency with the *same* defer trigger: no open-core consumer reaches a non-empty pin, so nothing drives the discrepancy today. It is acknowledged here, not fixed; the extraction-and-export follow-up above is the natural place to give both edge-read paths the one shared, type-correct comparison.

## 7. Verb conventions (embed == remote)

Following the established path (`crates/jammi-python/tests/test_conformance.py` pins name-for-name + signature-for-signature parity across transports):

1. Rust impl: `Database::asof_join` (`crates/jammi-ai/src/session.rs`) over the operator.
2. PyO3 binding: `jammi_ai.Database.asof_join` (`crates/jammi-python/src/lib.rs`).
3. gRPC handler + `RemoteDatabase.asof_join` stub (`clients/python/jammi_client/_database.py`) — submits through the existing pipeline service.
4. Add `"asof_join"` to `_PIPELINE_VERBS` in `test_conformance.py`; the guard then asserts both surfaces callable with identical signatures.

## 8. Module layout

```
crates/jammi-ai/src/pipeline/asof/
  mod.rs       +new  pub-uses, module decls
  spec.rs      +new  AsofJoinSpec, AsofKey, MatchDirection, Boundary, Tolerance, TieBreak, AsofError, builder
  exec.rs      +new  AsofJoinExec + ExecutionPlan impl
  merge.rs     +new  merge_partition — the per-group sort-merge core
  verb.rs      +new  asof_join verb body
crates/jammi-ai/src/pipeline/graph_neighbourhood.rs   DEFERRED route as_of through the shared comparison (§6.3 — tracked follow-up, NOT in this delivery)
crates/jammi-ai/src/session.rs                         CHANGED  expose asof_join
crates/jammi-python/src/lib.rs                         CHANGED  PyO3 binding
clients/python/jammi_client/_database.py               CHANGED  remote stub
crates/jammi-python/tests/test_conformance.py          CHANGED  _PIPELINE_VERBS += asof_join
crates/jammi-ai/tests/it/asof_merge.rs                 +new
crates/jammi-ai/tests/it/asof_verb.rs                  +new
docs/guide/src/asof-join.md                            +new  cookbook recipe (§9)
```

## 9. Cookbook recipe (mandatory exit-criteria bullet)

`docs/guide/src/asof-join.md` — "Point-in-time joins: matching facts to the instant they were known." Outline: the leakage problem stated plainly (a forward join imports the future); the `asof_join` call on two registered relations; the four pinned knobs (direction, boundary, tolerance, tie-break) each shown changing the result; a closing note that the same verb assembles a leakage-free labelled set, a trade-vs-quote panel, and a clinical observation table — no domain words in the API. `SUMMARY.md` updated; `mdbook build` clean; every code sample compiles under `mdbook test`. (The runnable, golden-number-asserted demonstration lives in the cookbook repo, not here.)

## 10. Discipline-test example — a backtester, not a feature store

**Hypothetical user: *Meridian Quant*** — a solo systematic trader backtesting a strategy. Nothing to do with AccuRisk's audit-native coding, Lace's IaC bundles, or any ML feature store. Meridian has a `trades` relation (one row per fill, with `symbol` and `exec_time`) and a `quotes` relation (tick-level bid/ask, with `symbol` and `quote_time`). To compute realistic slippage, each trade must be matched to the **prevailing quote at or before the fill** — never a later quote, which would be lookahead bias (the backtester's name for leakage).

```rust
let spec = AsofJoinSpecBuilder::new(
        AsofKey { by: vec!["symbol".into()], time: "exec_time".into() },   // spine
        AsofKey { by: vec!["symbol".into()], time: "quote_time".into() },  // facts
    )
    .direction(MatchDirection::Backward)
    .boundary(Boundary::Inclusive)
    .tolerance(Some(Tolerance::Duration(5_000_000)))  // ignore quotes >5s stale
    .tie_break(TieBreak::ByColumnDesc("seq_no".into()))
    .project(vec!["bid".into(), "ask".into()])
    .build()?;

let table = db.asof_join(trades.into(), quotes.into(), spec).await?;
```

This passes the discipline test. None of *"trade," "quote," "slippage," "lookahead," "feature," "label,"* or *"store"* leaks into the engine; the engine provides the as-of relational primitive and the determinism contract, and Meridian builds backtest slippage on top — exactly as an ML pipeline builds a leakage-free training set or a clinical fabric builds an as-of observation table on the identical verb. A Jammi user who has never heard of any of those — or of Meridian — gets the same primitive.

## 11. Exit criteria

1. **Backward-inclusive correctness.** A spine of 1000 rows over 50 groups and a facts relation with multiple facts per group; assert every matched fact is the maximal `time ≤ spine.time` in-group, and every spine row is preserved. `tests/it/asof_verb.rs`.
2. **Boundary distinguishes coincident facts.** Identical inputs under `Inclusive` vs `Exclusive` differ exactly on rows with a fact stamped at the spine instant. `tests/it/asof_merge.rs`.
3. **Tolerance suppresses stale matches.** A fact just outside the tolerance yields a null match; just inside yields a hit. Both `Duration` and `Steps`.
4. **Ambiguity is loud.** Duplicate facts at the matched instant with `TieBreak::Error` returns `AsofError::AmbiguousMatch`; with `ByColumnDesc` returns the newest. No test observes a non-deterministic pick.
5. **Null + preservation.** Null-time spine rows preserved with null facts; null-time facts never matched; spine never shrinks.
6. **Edge-gather unified — DEFERRED (tracked follow-up, see §6.3).** Not an exit criterion of this delivery. The follow-up first **extracts and exports** a shared comparator from the private `merge.rs` free fns (`eligible_at_or_before` / `temporal_i128`), then routes `EdgeGather::as_of` through it. When that lands, a `graph_neighbourhood` test must exercise `EdgeGather::as_of` passing through the shared, exported typed comparison (not the string cast) and returning rows at-or-before the pin, including a case where lexicographic and typed order disagree (proving the old string-cast was wrong and the unified path is right). The same follow-up also gives `propagate_embeddings`'s `edge_scan_sql` the as-of clause it currently records-but-ignores (§6.3). Until then the string-cast pin stands, the propagate path ignores its recorded pin, and this criterion is intentionally not met — both are undriveable from open-core today (no consumer sets a non-empty pin).
7. **Embed == remote.** `asof_join` callable with identical signature on `Database` and `RemoteDatabase`; conformance guard green.
8. **Scale sanity.** 1M-row facts × 100k-row spine over 10k groups completes within the operator's sort-merge bound (O((n+m) log) dominated by the sort), not the `NestedLoopJoinExec` quadratic path DataFusion falls back to for plain inequality joins[^df-8393]. Asserted as a wall-clock ceiling in a `#[ignore]`-free bench-style integration test.
9. **No band-aids.** Zero net new `#[allow(...)]`, `let _ =`, `// TODO`, `#[ignore]`. `cargo clippy --all-features -- -D warnings` and `cargo fmt --check` pass.
10. **Recipe.** §9 chapter renders and its samples compile.

## 12. Engineering-principles audit

| Principle | How this spec satisfies it |
|---|---|
| *Clean, functional style — right abstraction* | One `AsofJoinExec`; the edge-gather pin (§6.3) is recognised as "the same thing at a different scale" and is sequenced to fold into a shared, exported comparison (the deferred extraction of the private `merge.rs` comparator) rather than kept as a second special case. The merge is an iterator over sorted runs, not bespoke per-direction loops — `MatchDirection` parameterises one cursor. |
| *Stack-safe* | The merge is iterative (two cursors), not recursive; bounded by partition size. |
| *Clear boundaries* | Four concerns, four types: `AsofJoinSpec` (the *what*), `AsofJoinExec` (the *plan contract*), `merge_partition` (the *algorithm*), the verb (the *lifecycle*). The operator depends on DataFusion's `ExecutionPlan` boundary; it does not reach into the planner's internals. |
| *DRY* | One merge core serves all directions; the comparison that drives the verb is the same one the edge-gather pin will call once it is extracted and exported (deferred, §6.3) — the spec's design avoids two definitions of "as of," even though the unification of the second caller is sequenced as a follow-up. |
| *No backwards compatibility* | New module; the reserved `as_of` field is *implemented*, not shimmed; no compatibility path. |
| *Type-driven design* | `MatchDirection`/`Boundary`/`Tolerance`/`TieBreak` are enums, never strings; `AsofKey` makes "which column is temporal" unmistakable; builder for the >3-param spec; float temporal keys are unrepresentable as valid (rejected at build, §3.4). |
| *No band-aids* | Duplicate-timestamp non-determinism — the easy band-aid every engine ships — is refused: `TieBreak::Error` makes ambiguity a typed failure. The bespoke string-cast edge pin is *sequenced* to be replaced by the shared typed comparison (deferred, §6.3); it is honestly recorded as a known latent issue, undriveable from open-core today, not papered over as already fixed. |
| *Engine, not platform* | No "feature/label/entity/store" vocabulary; the §10 third-tenant is a backtester. The verb produces a generic result table; what a consumer assembles from it is theirs. |
| *Atomic across the workspace* | §8 ships engine + PyO3 + remote client + conformance test in one PR. |
| *Docs reflect current state* | The recipe (§9) describes the verb as it is; no journey markers. |

## 13. Open questions

1. **SQL-surface `ASOF JOIN` syntax.** This spec exposes as-of as a verb, not as a SQL keyword, because (a) DataFusion has no native `ASOF JOIN` and its SQL parser (`sqlparser-rs`) does not recognise the keyword[^df-318], so a SQL surface would require a parser fork, and (b) the verb is the idiomatic Jammi surface for pipeline operations and is directly conformance-pinnable. If a consumer demonstrates a genuine need to compose as-of *inside arbitrary SQL* (a subquery, a CTE), a follow-up may register the operator behind a SQL extension — downstream-driven, not speculative ([`../PLAN-META.md` — Interface Discipline](../PLAN-META.md)).
2. **Bitemporal range (`VERSIONS BETWEEN`).** SQL:2011 also defines period-overlap and version-range queries[^sql2011]; this spec ships only the point lookup ("as of one instant"), the 99% case. A range variant is deferred until forced.
3. **Multi-fact (`AsofGather`).** Some consumers want the *k* most recent facts, not the single latest (a short history window per spine row). That is a distinct operator; not in this spec.
4. **Float temporal keys.** Rejected (§3.4). If a consumer presents an ordered-float use case with a defined NaN policy, revisit — but the default refusal is the correct, loud choice.

## 14. References

[^cpbe]: Barber, Candès, Ramdas & Tibshirani, "Conformal prediction beyond exchangeability," *Annals of Statistics* 51(2), 2023: validity "fundamentally depends on … exchangeability of the data," and "if the data distribution drifts over time, then the data points are no longer exchangeable." Leaking future information into the calibration set is a form of non-exchangeability that inflates apparent coverage. — https://arxiv.org/abs/2202.13415 (PDF: https://projecteuclid.org/journals/annals-of-statistics/volume-51/issue-2/Conformal-prediction-beyond-exchangeability/10.1214/23-AOS2276.pdf) (fetched 2026-06-16).

[^duckdb-blog]: DuckDB, "ASOF Joins: Fuzzy Temporal Lookups" (2023-09-15): an ASOF join "matches each left-side row with at most one right-side row" — the most recent value at/before the probe; the engine "can stop searching when it finds the first match because there is at most one match"; boundary intervals: `>=` → `[Tn, Tn+1)`, `>` → `(Tn, Tn+1]` ("Non-strict inequalities include the table timestamp; strict inequalities exclude it"); implementation "hash partitions and sorts" each side then "merge joins the sorted values within each hash partition." — https://duckdb.org/2023/09/15/asof-joins-fuzzy-temporal-lookups (fetched 2026-06-16).

[^pandas]: pandas `merge_asof` reference: a left join that matches "the nearest matches from the right"; `direction` ∈ {`backward` (default), `forward`, `nearest`}; `allow_exact_matches` (default `True`) → inclusive `<=`/`>=`, `False` → strict `<`/`>`; `tolerance` discards matches beyond the limit (→ NaN); `on` must be sorted ascending. — https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge_asof.html (fetched 2026-06-16).

[^polars]: Polars `DataFrame.join_asof`: matches the nearest key; both frames must be sorted by the asof key; `strategy` ∈ {`backward` (default), `forward`, `nearest`}; `by` joins within equality groups; `tolerance` accepts numeric or duration strings; **string keys are not supported for the `nearest` strategy**. — https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join_asof.html (fetched 2026-06-16).

[^kdb-aj]: kdb+/q `aj` reference: `aj[c; t1; t2]` where the last symbol in `c` is the temporal key and the preceding are equality keys; for each `t1` row it takes "the last value … (most recent time)" in `t2` matching the equality keys with time ≤ target; `aj0` returns the t2 time, `aj` the t1 boundary time. — https://code.kx.com/q/ref/aj/ (fetched 2026-06-16).

[^ch-9906]: ClickHouse issue #9906 — ASOF JOIN can return query-range-dependent (non-deterministic) rows when multiple right rows share the matched timestamp; cited as the motivation for this spec's loud `AmbiguousMatch`/explicit tie-break rather than a silent pick. — https://github.com/ClickHouse/ClickHouse/issues/9906 (fetched 2026-06-16).

[^feast-pit]: Feast, "Point-in-time joins": features are retrieved "with a timestamp at or before the entity row's `event_timestamp`"; "Feast will scan backward in time from the entity dataframe timestamp up to a maximum of the TTL … relative to each timestamp within the entity dataframe. TTL is not relative to the current point in time." — https://docs.feast.dev/getting-started/concepts/point-in-time-joins (fetched 2026-06-16).

[^feast-retrieval]: Feast, "Feature retrieval": `event_timestamp` "acts as the upper bound (inclusive) for which feature values are allowed to be retrieved for each entity row"; ties: "If multiple entity keys are found with the same event timestamp, then they are deduplicated by the created timestamp, with newer values taking precedence." — https://docs.feast.dev/getting-started/concepts/feature-retrieval (fetched 2026-06-16).

[^bitemporal]: Temporal database foundations — *valid time* = "the time period during [which] … a fact is true in the real world"; *transaction time* = "the time at which a fact was recorded in the database"; *bitemporal* combines both. SQL:2011 application-time period tables (valid time) and system-versioned tables (transaction time) standardise these. — https://en.wikipedia.org/wiki/Temporal_database (fetched 2026-06-16).

[^sql2011]: SQL:2011 (ISO/IEC 9075:2011) added application-time period tables (`PERIOD FOR`), system-versioned tables (`PERIOD FOR SYSTEM_TIME` + `WITH SYSTEM VERSIONING`), and temporal queries (`AS OF SYSTEM TIME`, `VERSIONS BETWEEN SYSTEM TIME … AND …`). — https://en.wikipedia.org/wiki/SQL:2011 (fetched 2026-06-16).

[^df-318]: Apache DataFusion issue #318, "ASOF join support / Specialize Range Joins" — open since 2021-05-11, no merged implementation as of mid-2026; confirms DataFusion has no native ASOF JOIN and that the SQL surface does not parse the keyword. — https://github.com/apache/datafusion/issues/318 (fetched 2026-06-16).

[^df-8393]: Apache DataFusion issue #8393, "Range/inequality joins are slow" — current range/inequality joins execute via `NestedLoopJoinExec` (Cartesian + filter), benchmarked ~8.3 s vs DuckDB ~1 s, with the recommendation to adopt an ASOF/IEJoin sort-merge operator. Motivates this spec's dedicated `AsofJoinExec` rather than leaning on a plain inequality join. — https://github.com/apache/datafusion/issues/8393 (fetched 2026-06-16).

[^df-execplan]: DataFusion `ExecutionPlan` — an operator declares `required_input_distribution` and `required_input_ordering`, and the planner inserts `RepartitionExec`/`SortExec` to satisfy them; this is how a custom physical operator reuses the engine's shuffle and sort infrastructure rather than hand-rolling it. Workspace pins `datafusion = "52.3"` (`/Users/vijaychakilam/git/f-inverse/jammi-ai/Cargo.toml`). — https://docs.rs/datafusion/52.3.0/datafusion/physical_plan/trait.ExecutionPlan.html (fetched 2026-06-16).
