# Design Philosophy

Jammi is an **engine of generic primitives, not a substrate-platform.** This page states
what that means, where the line falls, and how the engine is meant to be consumed and
deployed. It is principle-level on purpose — it pins no versions, index types, or other
details that move, so it stays true as the implementation evolves.

## The one rule everything else follows from

Jammi names no consumer. Not in code, config, docs, tests, fixtures, or scripts.
**References point one way only: a consumer may depend on Jammi; Jammi depends on no
consumer.** A consumer's name anywhere in the engine repo is a bug.

The engine is **edition-agnostic**: it makes no mention of a commercial, managed, paid, or
"enterprise" version — not even generically, and not in comments, docs, config, or specs. It
does not acknowledge that any such edition exists. Engine behavior is justified on its own
terms, never by what a downstream layer needs.

A real consumer need is a fine forcing function for the roadmap — but the thing that lands
in the engine is the generic primitive the need pointed at, with the name filed off. Two
unrelated consumers independently reaching for the same primitive is the strongest evidence
it is right; being able to justify a primitive only by naming one consumer is the strongest
evidence it is wrong.

## The discipline test

Before any capability enters the engine:

> Would a user who has never heard of any particular consumer reach for this on its own?

Justify it against unrelated, hypothetical consumers — a feature store, an ad-attribution
chain, a clinical-trial data fabric, a personal-knowledge search tool. If it survives only
with a real name attached, it is domain pull masquerading as a primitive, and it belongs in
that consumer's own repo, built on a published Jammi version.

## Where the line falls

| Stays in Jammi (engine primitives) | Lives in the consumer's repo (composition) |
|---|---|
| DataFusion SQL surface | Domain tables (the consumer's own entities) |
| Catalog primitives — typed status enums, append-only migrations | Domain status enums and their lifecycle meaning |
| Storage primitives — Parquet result-tables, sidecar ANN, mutable companion tables | Domain audit / recapture / gating semantics |
| Source registration and federation | Domain interop adapters (foreign format → a registered source) |
| AI primitives — embeddings, inference, search, fine-tune, eval | Domain agents / reactors and what they decide |
| Data-driven provenance channels | Domain audit columns (what "signed-by", "owned-by" mean) |
| Trigger stream — Arrow batches, SQL-predicate filters | Domain event taxonomies and their semantics |
| Tenant session scope | Domain ownership models — registries, ownership lanes, publish/install/bind |
| Server surfaces (Flight SQL, gRPC) | Domain operation contracts (typed verbs, signed transitions) |

Left column: every cell is something a user with no knowledge of any specific consumer would
still want. Right column: every cell is a composition a consumer builds in its own repo out
of the left column. The substrate-platform shape — append-only typed substrate + pluggable
reactors + read-back loops at decision moments — recurs across consumers, but it recurs in
*their* domain layers, not here. If Jammi codified that shape, the next consumer, in a domain
we have not met, would have to bend to fit or fork.

## Leak-guards

Domain pull leaks in through "almost-generic" primitives that quietly assume a consumer's
semantics. Three guards, all the same shape — **the primitive transports / persists / merges;
the semantics live above it:**

- **The trigger stream knows nothing about the payload.** A topic is a name; a message is an
  Arrow batch; a subscription is a SQL predicate over batch columns. No typed-event taxonomy,
  no required headers (actor, timestamp, signature), no ordering guarantee beyond per-topic
  FIFO. Adding a "typed event" with mandatory headers is where it would break.
- **Mutable tables expose CRUD through DML, nothing more.** No built-in transition log, no
  automatic versioning, no lifecycle-column convention. A consumer that wants
  append-only-with-history builds it from two table registrations. Adding a `LifecycleTable`
  wrapper is where it would break.
- **Provenance channels merge declared columns at query time.** The engine never writes to a
  provenance column. What a channel column *means* — signed-by, retrieved-by, scored-by,
  attributed-to — is the caller's vocabulary. Adding channel-writing helpers
  (`record_actor()`, `sign_with()`) is where it would break.

## How embeddings are consumed: `search`

There is one consumption verb. The embedding producers differ only in the encoder; the moment
vectors land in a result-table plus its sidecar index, consumption is identical:
`search(source, query, k)` returns top-k ids and scores — ANN over the sidecar index, with an
exact scan as the fallback when no index is present.

`search` is the curated path because it is where the engine adds value: the ANN index, the
exact-scan fallback, and the evidence/provenance attached to results. The raw vector itself is
a column in a SQL-addressable result-table, so it is reachable through the generic SQL surface
(`SELECT <vector_column> FROM <result_table> …`) — the same as selecting any other column. That
generic read path is legitimate, and making it ergonomic and documented is ordinary engine work;
it is *not* a violation of anything here. It is the right home for the rare genuine need —
exporting embeddings, a custom-metric re-rank, debugging.

What the engine does **not** add is a *dedicated vector-retrieval verb* on the embedding/search
API. A consumer should not pull vectors back to reconstruct a ranking the engine already
computes, or to compare across models (re-encode for that). A bespoke `get_vector` verb with no
caller is speculative domain-convenience: it competes with `search` as "how you consume
embeddings" and owes a per-id contract across every storage backend. So the line is precise —
the capability is the SQL surface, not a verb; the verb is what fails the discipline test.

## How it deploys: one binary, pluggable backends

The same engine binary serves every topology. Differences are **configuration** (which backend
driver) and **process count** (1 vs N) — never a topology-specific code path or a server-only
feature the library cannot do.

Four canonical shapes — points on a configuration surface, not tiers to graduate through:

- **A — Single-process embedded.** Library mode, SQLite catalog, local Parquet, in-memory
  trigger stream, in-process model cache. Notebooks, CLI, single-machine, laptop dev.
- **B — Single-tenant server.** One process, Flight SQL + gRPC trigger stream exposed, optional
  Postgres catalog for HA, local disk + object-store backup. Physical-isolation requirements,
  on-prem.
- **C — Multi-tenant server.** One process (or stateless fleet), Postgres catalog, shared object
  store, shared trigger broker; tenant scope filters every catalog query. SaaS hosting many
  tenants — e.g. an edge-function deployment with the engine as a container sidecar reached over
  gRPC, or a managed multi-tenant service.
- **D — Disaggregated.** Catalog process ↔ stateless query workers ↔ GPU-resident inference
  workers ↔ trigger broker, each scaling independently; the same binary in every role. Very high
  scale, specialized GPU pools, split compliance posture.

The five pluggable backends — the entire deployment-knob surface:

| Backend | Embedded default | Production driver(s) |
|---|---|---|
| Catalog | SQLite | Postgres |
| Result-table storage | Local filesystem | S3 / GCS / R2 / Azure Blob (via the `object_store` crate) |
| Mutable companion tables | SQLite | Postgres |
| Trigger broker | In-memory | Kafka / NATS / Redis Streams / a cloud queue |
| Model artifact source | Local cache + HF Hub | Mirror, private registry, object-store-backed store |

Everything else — load balancing, ingress, TLS, secrets, IAM, observability stack,
orchestration, autoscaling — is the consumer's runtime, not the engine's.

Three properties this preserves, and that a consumer evaluating Jammi should be able to test:

- **The library is never less capable than the server.** Anything the server does, the library
  does in-process. No feature gated to "clustered mode."
- **The default deployment fits on a laptop.** SQLite + local filesystem + in-memory trigger
  stream + HF Hub cache is a complete deployment, no cloud service required.
- **Production is a configuration change, not a fork.** Moving from Shape A to Shape C swaps
  backend drivers; the engine code, schema, catalog discipline, and trigger-stream contract are
  unchanged.

## Positioning

With these primitives the engine is, precisely: *an embeddable AI engine — federated SQL,
durable result-tables with ANN indexes, mutable companion tables, embeddings / inference /
search / fine-tune / eval, evidence provenance on every row, and a trigger-stream event
surface.* That is a general-utility data-and-AI engine. It is deliberately **not** a
substrate-platform engine — because the substrate-platform shape belongs to the consumers that
*are* that shape, and the next consumer may not be.
