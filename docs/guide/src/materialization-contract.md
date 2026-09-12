# The Materialization Contract: Verifiable Result-Table Identity

> **Measured companion:** for the long-form, executed-and-measured Python treatment, see [The Cookbook → Incremental Recompute](https://f-inverse.github.io/jammi-ai/cookbook/chapters/20-recompute/recompute.html).

Every result table Jammi publishes carries a *verifiable identity*: a sidecar
attestation that lets a later reader assert **"this artifact is the output of
definition D over input-state S"** — without trusting a name, a path, or an
out-of-band convention. This recipe is the operator's view of that contract:
what the attestation contains, how it is written, how to check a table against
it, and how recovery reconciles it after a crash. Everything below describes the
system as it ships today.

## What a materialization manifest is

A result table is published as an immutable Parquet object (plus, for embedding
tables, an ANN-index sidecar bundle). Alongside it the engine writes a separate
`.materialization.json` sidecar — for **every** result table, not only embedding
tables — carrying an in-toto-shaped attestation that binds three things to the
artifact's content digest:

- a **definition hash** of *how* the table was produced;
- the **as-of anchors** of every input the producer read; and
- the **producing-run identity and instant**.

The on-disk shape is `MaterializationManifest`:

| Field | Meaning |
|-------|---------|
| `artifact` | The in-toto *subject* — SHA-256 over the Parquet object's bytes. The thing a verifier matches by digest. |
| `definition_hash` | SHA-256 of *how* the table was produced — the descriptor plus the environment (see below). |
| `input_anchors` | The immutable state pointer of each input, in producer order. |
| `produced_by` | The producing-run id — provenance, never the reproducibility anchor. |
| `produced_at` | The producing instant, RFC3339 — provenance, never the anchor. |
| `engine_version` | The engine semantic version that produced the artifact. |
| `manifest_version` | The manifest format version, so a future format change is a typed error rather than a silent misparse. |

The artifact digest deliberately covers the Parquet **data**, never the ANN
index sidecar: the index is a derived accelerator reconstructible from the data,
so a verdict attests the data-of-record, not the search structure. The two
sidecars never collide — `.manifest.json` describes the ANN accelerator;
`.materialization.json` attests the Parquet data.

## The two halves of identity: descriptor and environment

The `definition_hash` is *not* a hash of a logical plan. Result-table producers
in this engine are hand-built physical pipelines, so there is no single plan to
canonicalise. Instead the hash folds two typed, deterministically-serialisable
values:

- **`ProducingDescriptor`** — *how* the table was computed: the verb plus its
  typed parameters. Each producer fills in exactly one variant from its own
  parameters — `Inference`, `Embedding`, `NeighborGraph`, `GraphPropagation`, or
  `ContextSet`. A stable, sorted-key JSON encoding yields canonical bytes.
- **`MaterializationEnv`** — the output-affecting *environment* that is not part
  of the description itself: the engine semantic version, the **compute device**
  (`Cpu` / `Cuda { ordinal }` / `Metal { ordinal }`), and the identity + backend
  kind of every model the producer invoked.

The device is part of the environment for a concrete reason: a model produces
different float outputs on CPU versus an accelerator while carrying the same
model identity, so a hash that omitted the device would yield a false "match"
when only the device changed. The two halves are length-prefixed and
domain-separated before hashing, so a descriptor field can never alias an
environment field. Two runs of the same producer, with the same parameters, over
the same inputs, in the same environment, hash identically; any output-affecting
change to any of the three changes the hash.

The hash folds every *declared* parameter of the descriptor and the
environment — it does not fold the producing host's CPU microarchitecture.
Two CPU hosts of a different ISA (or even two same-ISA hosts with different
runtime vector-dispatch decisions) can produce the same `definition_hash` for
tables whose last float bits differ, because a CPU float reduction is a
same-box guarantee only (measured directly: `bits_snapshot.rs` pins CPU-float
bit output per box, not per `(target_arch, target_os)`). A `definition_hash`
match therefore asserts *which definition produced the table*, not that its
bytes are reproducible byte-for-byte on a different CPU host — identity across
hosts is the catalog row plus the `definition_hash`, never the raw bytes. This
CPU-variant gap in the hash is a known, open limitation, not a defect this
contract claims to close.

The input anchors are recorded but are deliberately **not** part of the
definition hash: the definition is *how* a table is produced, the anchors are
*over what*. A consumer that wants a combined "code + data" identity composes the
two itself.

## `BuildingTable::finish` is the sole building→ready transition

There is no manifest-free finalize. `ResultStore::create_table` returns a
lease-owned `BuildingTable` handle, and its `finish` is the single
`building → ready` path every producer goes through, so no table reaches
`ready` without an attestation. It performs the steps in a crash-safe order:

1. renew the writer's lease by compare-and-set — a writer whose lease was
   claimed by recovery learns it here, before it attests bytes it no longer
   owns;
2. `ResultStore::write_attestation`: read the Parquet bytes and compute the
   artifact digest, compute the manifest from the descriptor, environment, and
   resolved inputs, and write the `.materialization.json` sidecar;
3. flip the catalog row `building → ready` by a compare-and-set naming the
   writer (recording the `definition_hash` and the input anchors as summary
   columns, and clearing the lease);
4. register the table in DataFusion under the row's own catalog owner.

Because the bytes and the sidecar are durable *before* the status flip, a crash
in the window leaves a `building` row — never a queryable `ready` table missing
its manifest — and the row's lease then expires, so the next recovery sweep
promotes it from that very sidecar or reaps it. Every producer that materialises
an embedding table — graph propagation and context-set pooling — routes through
this funnel too: `ResultStore::materialize_embedding_table` writes the table and
then calls `finish` with the producer's `Materialization` (descriptor,
environment, and resolved input anchors).

## How to verify a table

`verify_materialization` is the read-only verb that recomputes a `ready` table's
artifact digest and checks it — and, optionally, an expected definition hash —
against the table's manifest. It returns a `MatchVerdict`; it never *acts* on one.
What a reader does with a mismatch (refuse, alarm, fall back) is the reader's
policy, not the engine's.

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use std::sync::Arc;
# use jammi_ai::session::InferenceSession;
# use jammi_db::store::manifest::{DefinitionHash, MatchVerdict};
# async fn ex(session: &Arc<InferenceSession>, table: &str) -> jammi_db::error::Result<()> {
let record = session
    .catalog()
    .get_result_table(table)
    .await?
    .expect("result table exists");

// No expectation: just assert the bytes still match the attestation.
let verdict = session
    .result_store()
    .verify_materialization(&record, None)
    .await?;

match verdict {
    MatchVerdict::Match => { /* artifact is the attested output */ }
    MatchVerdict::MatchWithUnpinnedInputs { unpinned } => {
        // Verified, but at least one input was anchored only to a read
        // instant, so reproducibility cannot be fully asserted. Honest,
        // not silent — downgrade confidence accordingly.
        let _ = unpinned;
    }
    MatchVerdict::Mismatch { expected, found } => {
        // The served artifact is not the output of the expected definition.
        let _ = (expected, found);
    }
    MatchVerdict::MissingManifest => {
        // No sidecar — a pre-contract table. A truthful unknown, never a
        // fabricated match.
    }
}

// Pin an expected definition hash to assert *which* definition produced it.
let expected = DefinitionHash("…".into());
let _ = session
    .result_store()
    .verify_materialization(&record, Some(&expected))
    .await?;
# Ok(()) }
```

### Python

`verify_materialization` takes the table name and an optional expected definition
hash, and returns the verdict as a dict tagged by `verdict`:

```python
verdict = db.verify_materialization("results__text_embedding__…")

if verdict["verdict"] == "match":
    pass  # artifact is the attested output
elif verdict["verdict"] == "match_with_unpinned_inputs":
    unpinned = verdict["unpinned"]      # sources anchored only to an instant
elif verdict["verdict"] == "mismatch":
    expected, found = verdict["expected"], verdict["found"]
elif verdict["verdict"] == "missing_manifest":
    pass  # pre-contract table — a truthful unknown

# Pin the definition you expect produced the table:
db.verify_materialization("results__…", expected_definition="<hex definition hash>")
```

## Reading the verdict

| Verdict | Meaning |
|---------|---------|
| `Match` | The recomputed artifact digest equals the manifest's, and (if supplied) the expected definition hash equals the manifest's. The artifact is the output of the expected definition. |
| `MatchWithUnpinnedInputs { unpinned }` | The artifact verifies, but at least one input was anchored only to a read instant (an external source with no version surface), so reproducibility cannot be fully asserted. The named sources are honest about not being reproducibly pinned. |
| `Mismatch { expected, found }` | The digest or the definition hash differs — the served artifact is not the output of the expected definition. Both sides are returned for the caller. |
| `MissingManifest` | No manifest sidecar exists — a table created before the contract landed. A truthful "unknown", never a fabricated match. |

An anchor is "pinned" when it points at an immutable id: a result table's content
digest, a mutable companion table's monotonic version, or an external source's
as-of/version value (an Iceberg snapshot id, a Delta version, an LSN, a
watermark). It is "unpinned" only when the source exposes no version surface, in
which case the anchor is the read instant and the verdict says so.

## How recovery reconciles manifest sidecars

`ResultStore::recover()` runs at startup and restores the crash-consistency
invariant of the catalog↔storage boundary across **every** tenant (it runs
under an admin scope — see
[Catalog Backend and Trigger Broker → Multi-writer safety](./catalog-and-broker.md#multi-writer-safety)
for the lease primitive this section assumes). Its domain is deliberately
narrow: it enumerates **only** `building` rows whose lease is absent or
expired (`Catalog::list_expired_building_tables`). A `building` row under a
**live** lease belongs to a writer that is still producing it — recovery does
not even look at its Parquet or manifest state, on any tenant, from any
session. For each expired-lease row it finds, recovery first *claims* it by a
one-row compare-and-set naming the recoverer the row's new owner (the
recoverer becomes a writer, heartbeating a fresh lease of its own) — only
after that claim succeeds does it inspect bytes or delete anything, so a
writer that renews mid-sweep, or a peer recoverer that claims first, is never
raced. For the materialization contract specifically, the claimed row's
Parquet and manifest state decide the terminal CAS:

- **A claimed row with valid Parquet but no manifest is reaped.** The write
  was torn in the window between the Parquet landing and the manifest being
  written — before the `building → ready` flip. The contract forbids promoting
  a table without an attestation, and the producing descriptor cannot be
  reconstructed after the fact, so the row is driven to `failed` by CAS and
  its bytes are deleted only after that CAS affects exactly one row — never
  promoted manifest-less.
- **A claimed row that *does* have its manifest is promoted**, backfilling the
  summary columns (`definition_hash`, input anchors, the *true* Parquet
  footer row count — never the count the writer intended before it crashed)
  the live path records, and rebuilding the ANN sidecar from the Parquet for
  an embedding table so it self-heals even if its segment set never landed.
- **A `ready` table whose manifest has since vanished is reaped.** This is
  `reconcile_ready_manifests`, which `recover()` also runs (across every
  post-contract `ready` row — one whose catalog `definition_hash` is set, so
  it was promoted under the contract): its `.materialization.json` being
  absent is a corruption, since the attestation a verifier would read is
  gone. Such a row is driven to `failed` by a status-guarded CAS and its
  bytes reaped, rather than left queryable with a silently missing manifest.

Every deletion in both reaping arms above follows its own one-row CAS — the row's *new*
owner (the recoverer, having just claimed it) deletes what it now owns; a
failed delete is logged and left for a later `jammi reconcile` pass, never
silently swallowed. This is two of the engine's three deletion arms for a
building row's bytes; the third is a writer's own `abort()` after its own
CAS. See [Operability](./operability.md#failure-mode-matrix) for the full
crash-mid-publish failure-mode entry, and
[Backup and Restore](./backup-and-restore.md) for what a catalog/storage
restore that lands *outside* this lease-driven window (an object whose row
was never written at all, or vice versa) requires — `jammi reconcile`, not
`recover()`.

Pre-contract tables report honestly rather than being penalised. A row created
before migration 021 carries `definition_hash IS NULL` in the catalog and
legitimately has no sidecar; recovery leaves it untouched, and
`verify_materialization` returns `MissingManifest` for it — a truthful unknown.
This is the distinction the contract draws: a bug (post-contract, no sidecar) is
reaped; a legitimate historical table is preserved.

## Versioned tables: the identity chain

An embedding table that has been refreshed (see
[Refresh an Embedding Table Incrementally](./incremental-refresh.md)) carries a
chain of versions beside its base manifest. The base version's identity **is**
the `.materialization.json` artifact digest — publishing the base changes no
anchor. Every later version `N` records, in `{table}__v{N}.version.json`, its
parent, the definition hash, the delta descriptor (`EmbeddingDelta` or
`EmbeddingCompaction`, which name the parent version and its identity), the
digest of every fragment it serves and the digest of its deletion mask; its
identity is a domain-separated SHA-256 over exactly those inputs. Counts, the
ANN segments and `produced_by`/`produced_at` are outputs, not inputs.

`verify_materialization` on a versioned table runs the base check unchanged,
then recomputes every fragment digest and the mask digest from the bytes,
recomputes the chain identity from the parent's recorded identity, and compares
it with both the version manifest and the catalog row. A `Mismatch` names the
artifact that diverged. The version's input anchors are the source at the
instant of the refresh (unpinned), so the verdict is
`MatchWithUnpinnedInputs` naming the source, exactly as for the base embed.

`staleness` and every `derives_from` anchor use the current version's identity,
so a refresh that changed content advances dependents to
`Stale { InputAdvanced }` and a `no_change` refresh leaves them `Fresh`.
`recompute` of a versioned table starts a new chain in a new table.

## Why this identity matters

The materialization manifest gives every result table a content-addressed,
verifiable identity that is independent of its name or path. That identity is the
nucleus a future freshness-and-caching layer builds on: once a table's output is
bound to the definition that produced it and the as-of state of its inputs, an
incremental-recompute layer can decide whether a cached artifact is still valid
by comparing definition hashes and input anchors — rather than re-running the
producer blind. The contract ships that identity and the verify primitive today;
it ships **no** policy. What a reader does with a verdict, and when a downstream
layer chooses to recompute, are decisions left to the consumer.
