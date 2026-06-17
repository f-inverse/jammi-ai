# 60 — Temporal correctness & the materialization contract

> Plan group. Two engine primitives that complete the substrate's temporal story: a point-in-time **as-of temporal join** ([`SPEC-01-asof-join.md`](./SPEC-01-asof-join.md)) and a **materialization contract** that lets a downstream reader verify *what definition produced a materialized table, and as-of what input state* ([`SPEC-02-materialization-contract.md`](./SPEC-02-materialization-contract.md)).
>
> **Status:** Proposed (draft). Targets a post-M1 minor line (≥ 0.31.x). Independent of the M2 1.0-engineering-bar batch; either can land first.
>
> Research rules: this group follows the same evidence-graded methodology as [`../cp9-substrate-primitives/RESEARCH-METHODOLOGY.md`](../cp9-substrate-primitives/RESEARCH-METHODOLOGY.md). Every external-behaviour claim carries a verbatim-quoted, URL-cited reference (see each spec's §References).

## Why these two, why now

The cp9 substrate ([`../cp9-substrate-primitives/README.md`](../cp9-substrate-primitives/README.md)) gave the engine *current-state* temporal machinery — mutable companion tables, a trigger stream, predicate-filtered replay. What it does **not** give is the other half of any time-aware workload: reconstructing the value of a fact *as it was known at a past instant T*, and proving that a materialized artifact still corresponds to the definition and input-state that produced it. These two gaps are independent of one another and each clears the discipline test on its own (§ each spec's discipline-test section). They are grouped only because they share a motivation: the engine's own guarantees become unsound without them.

**The as-of join is a correctness primitive the engine already owes itself.** The engine ships conformal prediction sets with a proven coverage guarantee. That guarantee rests on *exchangeability* of the calibration and deployment scores; future-information leakage breaks exchangeability and silently inflates apparent coverage (Barber, Candès, Ramdas & Tibshirani 2023 — see [`SPEC-01-asof-join.md` §References][^cpbe]). Any workflow that assembles a labelled training or calibration set from time-stamped facts — without an as-of join — is one careless `JOIN` away from leaking the future into the past. That is true for a quant backtester, a clinical-trial fabric, an ad-attribution chain, or an ML feature store alike; none of them needs to be named for the primitive to be justified. The `EdgeGather::as_of` / `EdgeSourceRef::as_of_column` pin in `crates/jammi-ai/src/pipeline/graph_neighbourhood.rs` is today the *only* temporal-pin surface in the engine — and it is a narrow one: a backward-inclusive predicate emitted as a string-cast `WHERE` clause (`arrow_cast(col,'Utf8') <= 'asof'`), scoped to edge loads, comparing lexicographically rather than by type. SPEC-01 generalises the concept into a first-class, type-correct relational primitive and routes the edge-gather pin through the same comparison so the engine has one definition of "as of," not a bespoke string filter in one path.

**The materialization contract is a seam, not an implementation.** When a materialized table crosses a boundary — exported, copied into a low-latency serving tier, handed to another process — the receiver today gets bytes and a path (`ResultTableInfo { table_name, parquet_url, index_url }`) and nothing that lets it assert *"this is the output of definition D over input-state S."* Without that assertion, any consumer that re-derives the downstream read by hand reintroduces drift one layer out, where the engine's conformance tests cannot see it. SPEC-02 makes the materialized artifact carry a verifiable identity — a content hash of the producing logical plan plus the immutable as-of anchors of its inputs — so a reader can check the match at read time. The engine ships the *contract and the verify primitive*; what a consumer does with the verdict (refuse to serve, alarm, fall back) is the consumer's policy.

Neither spec adds a serving tier, a KV store, RBAC, or any feature-store, governance, or domain vocabulary to the engine. Those remain — per [`../../PHILOSOPHY.md`](../../PHILOSOPHY.md) and [`../../CLAUDE.md` — *Engine, not platform*](../../CLAUDE.md) — the consumer's composition, built on a published Jammi version.

## The two specs

| Spec | Primitive | Primary surface | New verbs (embed == remote) | Migration |
|------|-----------|-----------------|------------------------------|-----------|
| [SPEC-01](./SPEC-01-asof-join.md) | As-of temporal join (sort-merge `AsofJoinExec` physical operator) | `asof_join` verb → materialized result table | `asof_join` | none (reads sources, writes a result table via the existing path) |
| [SPEC-02](./SPEC-02-materialization-contract.md) | Materialization contract (definition hash + input as-of anchors) | new `.materialization.json` sidecar + `verify_materialization` verb | `verify_materialization` | 021 (adds `definition_hash`, `input_anchors` to `result_tables`) |

## Concurrent-session strategy

The two specs touch disjoint subsystems and may be built in parallel worktrees:

- **SPEC-01** lives under `crates/jammi-ai/src/pipeline/` (a new `asof.rs` operator + verb) and reuses the result-store write path. It touches `graph_neighbourhood.rs` only to route the reserved `EdgeGather::as_of` through the shared comparison helper.
- **SPEC-02** lives under `crates/jammi-db/src/store/` (manifest extension), `crates/jammi-db/src/catalog/` (migration `021`), and `crates/jammi-ai/src/` (the `verify_materialization` verb).

The only shared touch-point is the result-store materialization path: SPEC-01 produces a table; SPEC-02 stamps every produced table with its manifest. To avoid a merge race, **SPEC-02 lands first or they land in one combined PR** — a table SPEC-01 materialises must already carry the SPEC-02 manifest, never a manifest retrofit. If built in parallel worktrees, the integrating session rebases SPEC-01 onto SPEC-02's materialization signature before the conformance run. Each remains atomic across the workspace per [`../../CLAUDE.md` — *Atomic across the workspace*](../../CLAUDE.md).

## Relationship to the cookbook

These primitives are the engine half of an engine↔cookbook evolution. The OSS cookbook ([`../40-cookbook/`](../40-cookbook/)) already demonstrates the *current-state* feature surface (the mutable-companion-table chapter); once SPEC-01 lands, that chapter's sibling can demonstrate **leakage-free training-set generation** — the hard half — asserting zero train/serve skew against golden numbers. The cookbook chapter is the forcing function that proves the primitive composes; it is authored in the cookbook repo, never in the engine. The discipline test is unchanged by that downstream use: the engine's job is the primitive, not the recipe.
