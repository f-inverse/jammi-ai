# 68 — Compute-tier substrate

Deployment shapes, the distributed data plane, incremental embedding, job cancellation, and
compute-tier operability: the substrate a multi-process deployment of the engine stands on.
Each unit has its own design document under [`units/`](./units/); multi-GPU and multi-node
training is [plan 67](../67-distributed-training/README.md).

## Units

| Unit | Document | What it is |
|---|---|---|
| Kubernetes | [`K-KUBERNETES.md`](./units/K-KUBERNETES.md) | A kustomize base plus overlays. The base is the engine default; `shape-d` adds a scheduler `Deployment` and a compute `StatefulSet` (headless service, `nvidia.com/gpu` on compute only, 600 s grace). Nothing else is sized. Validated by kubeconform, a kind smoke of the `ci` overlay, and a rendered-manifest check that every config mount carries the file its server reads. |
| Distributed data plane | [`DIST-DATA-PLANE.md`](./units/DIST-DATA-PLANE.md) | Batch inference is the jobs fleet. Online retrieval beyond one node is an in-house `PeerService` on `[server] peer_bind`: two-phase per precision, coordinator-only tenancy, a bounded failure ladder ending in a typed `Unavailable`, and rendezvous placement over catalog membership. Distributed SQL through `datafusion-distributed` waits on the DataFusion 55 line ([#613](https://github.com/f-inverse/jammi-ai/issues/613)). |
| Incremental embedding | [`DELTA-INCREMENTAL-EMBEDDING.md`](./units/DELTA-INCREMENTAL-EMBEDDING.md) | Versioned result tables: a monotonic version, a base published in one transaction with identity equal to the artifact digest, a per-version deletion horizon, a masked provider, and a content hash computed inside the UDF. Null keys are a typed `InvalidKey` raised before any model call; DataFusion errors are classified structurally. |
| Job dependencies | [`GRAPH-JOB-DEPENDENCIES.md`](./units/GRAPH-JOB-DEPENDENCIES.md) | The record of a decision: the engine does not gate one job on another — orchestration is the consumer's runtime — and what it provides instead: `cancelled` as a terminal status, cancel-before-claim at no attempt cost, idempotent submit, and one terminality predicate. |
| Operability | [`OPS-COMPUTE-TIER-OPERABILITY.md`](./units/OPS-COMPUTE-TIER-OPERABILITY.md) | DRAIN (SIGTERM) and RELEASE (SIGINT), PostgreSQL's mapping; `jammi-server release`; never aborting while a claim transaction can be in flight; two lease classes with a `lease_present` guard on every renewal; warm-before-ready; gauges sampled off-scrape; `/healthz` as liveness. |

## Facts every unit relies on

Each of these refuted a design at least once, and holds on both catalog backends.

- `claim_next` is a client-driven `BEGIN`/`UPDATE`/`COMMIT`. "The task is aborted, so no claim
  can land" is false: a flushed `COMMIT` lands after the future is dropped. Hence the rule
  never to abort while a claim can be in flight.
- `RETURNING` order is unspecified on both backends (measured inverted on PostgreSQL 16 and
  SQLite 3.46); every consumer sorts in Rust.
- sqlx-sqlite binds an out-of-range placeholder as `NULL` silently, so bind indices are
  computed from the actual chunk length.
- A bound `LIMIT` flips PostgreSQL to a generic plan from the sixth execution of a persistent
  statement; every `LIMIT` in the catalog is interpolated.
- `[worker] idle_poll_secs` defaults to 1 s.
- A store's `writer_id` is minted once per store and shared by every materialization on a
  session; nothing keyed on it alone distinguishes a loop-claimed table from an inline one.
- `instances.peer_addr` does double duty: a replica that sets `peer_advertise` to be
  gang-reachable also joins the retrieval ring, which is every row with a peer address.
- A claim that something "never happens in CI" is checked against an artifact that would
  exist if it did.

## Migrations

Migrations are append-only and unreserved: a change takes the next free number and updates
both pin sites — the list in `crates/jammi-db/src/catalog/migrations.rs` and
`EXPECTED_MIGRATION_NAMES` in `crates/jammi-db/tests/it/migrations.rs`.
