# Point-in-time joins: matching facts to the instant they were known

An *as-of join* matches each row of a **spine** relation to the at-most-one row
of a **facts** relation that was valid *as of* the spine row's instant, within
the same group. It is the relational primitive for point-in-time correctness.

The problem it solves is leakage. If you assemble a table by joining each spine
row to *any* fact in its group, you import facts stamped *after* the spine
instant — information that was not yet known. A forward join imports the future.
The as-of join takes only the fact valid at or before each instant, so every
attached value reflects what was knowable then, and nothing later.

The engine exposes this as one verb, `asof_join`, over two registered relations.
It carries only what every time-aware caller needs — an equality grouping, a
temporal ordering key, a match direction, boundary inclusivity, an optional
look-back tolerance, and a deterministic tie-break — and writes a result table
that carries the same materialization manifest every other producer does.

## The call

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use std::sync::Arc;
# use jammi_ai::session::InferenceSession;
# use jammi_db::config::JammiConfig;
use jammi_ai::pipeline::asof::{
    AsofJoinSpecBuilder, AsofKey, Boundary, MatchDirection, TieBreak, Tolerance,
};
# async fn ex(config: JammiConfig) -> jammi_db::error::Result<()> {
# let session = Arc::new(InferenceSession::new(config).await?);

// `events` is the spine (its `t` column is the as-of instant); `facts` carries
// the values that were valid over time (its `vt` column is their validity time).
// `key` groups the match so a fact only matches an event in the same group.
let spec = AsofJoinSpecBuilder::new(
        AsofKey { by: vec!["key".into()], time: "t".into() },  // spine
        AsofKey { by: vec!["key".into()], time: "vt".into() }, // facts
    )
    .direction(MatchDirection::Backward)  // most recent at or before
    .boundary(Boundary::Inclusive)        // a fact stamped exactly at t matches
    .tolerance(Some(Tolerance::Duration(5_000_000))) // ignore facts >5s stale
    .tie_break(TieBreak::ByColumnDesc("seq".into())) // newest seq wins a tie
    .project(vec!["value".into()])        // attach this fact column
    .build();

let table = session.asof_join("events", "facts", &spec).await?;

// The result is an ordinary relation: every spine row, with the matched fact's
// `value` attached (null where nothing matched within the rules). Read it via SQL.
let _rows = session
    .sql(&format!(
        "SELECT t, value FROM \"{name}\".public.\"{name}\" ORDER BY t",
        name = table.table_name,
    ))
    .await?;
# Ok(())
# }
```

The spine is always fully preserved — an unmatched spine row keeps its columns
and carries nulls for the fact columns. Dropping unmatched rows would silently
shrink the result; a caller who wants inner semantics filters on a non-null fact
column themselves.

## The four pinned knobs

Each knob is the choice every engine gets subtly different. The engine pins them
once, on the spec, never inferred — and each one changes the result.

### Direction

`MatchDirection::Backward` (the default) takes the most recent fact **at or
before** the instant — the only leakage-safe choice when the spine instant is a
point you must not see past. `Forward` takes the first fact at or after (e.g.
"the next scheduled event after each reading"). `Nearest` takes the smallest
absolute distance, resolving equidistant candidates toward the past, and
requires a numeric temporal key.

### Boundary

`Boundary::Inclusive` (the default, `<=`) lets a fact stamped exactly at the
instant match; `Boundary::Exclusive` (`<`) excludes it. Over identical inputs,
the two differ exactly on the rows that have a fact coincident with the instant.

### Tolerance

`None` (the default) looks back arbitrarily far. `Some(Tolerance::Duration(µs))`
for a temporal key, or `Some(Tolerance::Steps(n))` for an integer key, discards
a candidate farther than the limit — the spine row goes unmatched rather than
matching a stale fact. The limit is measured relative to each spine instant.

### Tie-break

When two facts share the matched instant within a group, the match is ambiguous.
`TieBreak::ByColumnDesc("seq")` disambiguates by a secondary column, the maximal
value winning (the transaction-time column). `TieBreak::Error` makes a true
duplicate a loud `AsofError::AmbiguousMatch` rather than a silent,
non-deterministic pick. With a tie-break in force the output is bit-reproducible.

## The temporal key must be totally ordered

The temporal key on each side must be a totally-ordered Arrow type — any
`Timestamp(..)`, `Date32`/`Date64`, or a signed/unsigned integer. A float key is
rejected: NaN has no total order, so "most recent at or before" would be
undefined. The two sides' temporal keys must share a type, and a null temporal
value is never ordered — a null-time spine row is preserved with null facts, and
a null-time fact is never a candidate.

## One verb, many shapes

The same `asof_join` assembles a leakage-free labelled set keyed on past
instants, matches each transaction to the value in effect when it occurred, and
pairs a measurement with the reading valid at the time it was taken. The engine
provides the as-of relational primitive and the determinism contract; what a
caller assembles on top of it is theirs.
