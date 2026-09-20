# Distributed data plane — substrate decision and the peer-search mechanism

How jammi goes beyond one node for data work: which substrate serves batch inference, online
retrieval, and distributed SQL; why Ballista and `datafusion-distributed` are not the retrieval data
plane; and the design of the in-house mechanism — a per-segment `PeerService` on the internal
`[server] peer_bind` listener, a per-precision two-phase merge, a bounded failure ladder, and
catalog-row membership with rendezvous placement.

Code citations are symbols in the current tree (`crates/<crate>/src/<path>.rs::Item`). Third-party
facts name the version they were read at.

---

## 1. Decisions

**D1 — Batch inference needs nothing beyond the jobs fleet (S1).** Principle: topology is
configuration — one binary serves every deployment shape — and the actuator rule (D5). `embedding`
is a durable job kind (`crates/jammi-ai/src/jobs.rs::ComputeSpec::Embedding`); a worker claims the
highest-priority claimable queued job of its `[worker] kinds` with `FOR UPDATE SKIP LOCKED` on
Postgres (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::claim_next`), and
`generate_embeddings` submits that spec. GPU placement of inference jobs is the deployer's node pool
plus `[worker] kinds` (`crates/jammi-db/src/config/mod.rs::WorkerConfig`). No query-plane substrate
is needed for batch inference. The embedding job is whole-table: `ComputeSpec::Embedding` has no
row-range selector, and a `building` table has one `writer_id` lease
(`crates/jammi-db/src/store/building.rs::BuildingTable`). Growing a table incrementally is
version-owned segment append (`crates/jammi-db/src/store/mod.rs::ResultStore::append_segment_for_version`,
see `DELTA-INCREMENTAL-EMBEDDING.md`). A fan-in "publish when every shard is terminal" job is job
orchestration; the engine has no job dependencies, and orchestration belongs to the consumer's
runtime.

**D2 — Ballista (S2) is not the retrieval data plane; it is the compute/gang plane.** Principle:
topology is configuration, and the actuator rule (D5). What is true of Ballista 54.1.0:

- *No accelerator resource dimension, no affinity.* `ExecutorSpecification` is `{ task_slots }` and
  the `ExecutorResource` oneof has the single arm `TaskSlots`; `TaskDistribution` is
  `Bias | RoundRobin`. A GPU stage lands on an arbitrary executor, and a retrieval stage cannot be
  pinned to the executor that holds a segment.
- *Shuffle is a local `work_dir`.* `ShuffleWriterExec` writes to the executor's work directory. The
  executor's `override_execution_engine` receives each stage's plan and can rewrite
  `ShuffleReaderExec` nodes and wrap the writer — the seam an object-store shuffle would be
  installed at — but a cross-executor object-store read is unproven.
- *Cluster state IS pluggable.* The `ClusterStorage` config enum has only `Memory`, but
  `ClusterState` and `JobState` are public traits and `BallistaCluster::new(Arc<dyn ClusterState>,
  Arc<dyn JobState>)` accepts any implementation. jammi implements both against its catalog
  (`crates/jammi-ballista/src/cluster.rs::CatalogClusterState`, `CatalogJobState`).
- *`expire_dead_executors` is executor-membership liveness*, the same class as jammi's own lease
  reclaim and instance pruning. With task and stage retries off it never makes consumer work
  runnable, so it is not the control loop D5 forbids.
- *Latency shape.* A 50 ms status poll plus disk shuffle: batch only, never the online hot path.

The data-plane decision rests on the first, second and last points: online retrieval needs
rescoring where the segment lives and a one-round-trip hot path, and Ballista offers neither
placement affinity nor a shuffle that fits jammi's storage backends. Not grounds: "Ballista submits
logical plans only" (`execute_physical_plan` submits a physical plan at 54.1.0), "Ballista needs two
images" (one binary can host both roles; jammi does exactly that), and "cluster state is
memory-only" (above).

Ballista IS jammi's compute-plane dependency: `crates/jammi-ballista` encodes jammi's physical
operators across the scheduler/executor boundary (`codec::JammiCodec`), adapts per-stage execution,
hosts the scheduler, executor and client roles from `[ballista]` configuration, and is what a placed
training gang runs on and what a client-role process's result-table materializations run on: a
`CREATE TABLE … AS`, an embedding, inference, refresh, as-of join or training-set build roots in
`jammi_db::store::ResultTableSinkExec` and submits the whole plan — compute and write — when a live
executor can hold it; the sink writes the table's bytes on the executor under the row's lease
(`SinkLease`: taken from the submitter's writer id by transfer CAS, handed back on success) and one
summary batch streams back to the process that finishes the catalog row. Retrieval does not:
`search` and every inline read serve their rows from the process that received them, for the
reasons above. The
workspace pins `ballista-core`/`-scheduler`/`-executor` `54.1` beside `datafusion = "54.1"`
(`Cargo.toml::[workspace.dependencies]`), and their transitive `arrow-flight`, `datafusion-proto`,
`object_store`, `prost` and `tonic` lines match the workspace's. What Ballista would need to be a
data plane: (1) accelerator-aware placement — carried by jammi's own `DistributionPolicy` over the
catalog's device inventory; (2) pluggable cluster state — met by the public traits above; (3) an
object-store shuffle — not built, with the `ExecutionEngine` rewrite as the seam it would use.
Each is an extension at a seam Ballista exposes; a fork or an upstream request is never an option.

**D3 — Beyond-one-node ONLINE retrieval is S4: in-house scatter-gather over the existing gRPC
tower, on a separate internal listener.** Principle: topology is configuration; "the library is
never less capable than the server" (`docs/guide/src/philosophy.md#how-it-deploys-one-binary-pluggable-backends`);
remote equals embedded — both transports return the same result. The segment is already the
distribution unit and the merge is already a total order over `(distance, row_id, segment_id)`
(`crates/jammi-db/src/index/segment.rs::merge`); each segment is independently loadable; exact
rescoring dispatches to the owning segment's raw-f32 companion
(`crates/jammi-db/src/index/sidecar.rs::SidecarIndex::get_exact`), so rescoring runs where the
segment lives; `AnnSearchExec` is a single-partition leaf built directly
(`crates/jammi-ai/src/operator/ann_search_exec.rs::AnnSearchExec`). The transport is a unary gRPC
service (`PeerService`, D7), not Flight `DoGet`: a custom ticket would have to be served from
`do_get_fallback` inside `datafusion-flight-sql-server`'s own service implementation (0.4.18), which
jammi mounts but does not own; Flight SQL sits on the public, tenant-bound listener, where a
tenant-free peer path must not be (D7); and a unary call carries the per-RPC deadline directly.

**D4 — Distributed SQL / joins over large result tables is S3 `datafusion-distributed`, deferred
behind the gates in §9.** What the Ballista plane carries today is the result-table sink above
(D2) — the write rides the plane; retrieval does not — decided per statement by
`jammi_db::compute_plane::StatementClass`; a `SELECT` served inline stays single-process. It fits "topology is configuration": a library, no scheduler process, a
worker is a Tonic service spawnable inside an existing process. Read at 2026-09-10: 4.0.0 pins
DataFusion 55; four majors in four months (1.0.0 2026-04-16 → 4.0.0 2026-08-20); not part of Apache
DataFusion. jammi's operators hold process-local handles (`Arc<ModelCache>` in `InferenceExec`,
`Arc<ResultStore>` + `SessionContext` in `AnnSearchExec`) that never serialize; a distributed plan
must rebuild them against the receiving process's session, which is what `JammiCodec` does for the
Ballista plane. The per-segment RPC (D7) is shaped so it can later be wrapped as an S3 leaf.

**D5 — The actuator rule: the engine ships the actuator; it never ships the control loop that pulls
it** (`crates/jammi-ai/src/pipeline/recompute.rs`, module docs). Forbidden: any loop that makes
consumer-visible work runnable or re-runnable on a schedule of the engine's own — sweepers that flip
rows toward runnable, engine-side fan-out, scheduled re-runs. Allowed: a bounded sweep on one
explicit request (`Cascade::Downstream`); claim policy expressed as catalog data evaluated inside
the claim query (`priority`, `claimable`; `Catalog::claim_next`); and liveness bookkeeping — lease
reclaim returns an abandoned claim to the queue it was already in (bounded by `max_attempts`) or
fails it, and instance pruning retires rows (`Catalog::reclaim_expired_jobs`,
`Catalog::prune_instances`) — which recovers work the consumer already submitted and never
originates or schedules work. Placement and membership are read on the query path (D9), so the
peer-search mechanism adds no loop.

**D6 — Two-phase per-precision protocol; width = `over_fetch(k·oversample, N)`.** Goal: byte
identity with a single node at N=1, recall no worse than a single node at N>1, and the rescore
multiplier paid only where it fixes a real defect. `over_fetch(m, n)` returns `m` at `n ≤ 1`
(`segment.rs::over_fetch`) — the N=1 byte-identity branch — and
`ceil(m × DEFAULT_SEGMENT_OVERFETCH_FACTOR)` otherwise. F16/Int8 approximate distances ARE
cross-segment comparable: the sidecar hands the raw f32 vector to usearch on add and search, and
usearch's i8 cast scales each vector by its own magnitude (usearch 2.25.1
`include/usearch/index_plugins.hpp`, `cast_to_i8_gt`; f16 is a per-component cast) — a per-vector
scale that cosine ignores. Binary alone fits a per-segment corpus threshold τ at build (default
`ThresholdKind::Median`, `sidecar.rs`), so its per-segment Hamming distances are not on one scale;
truncating a cross-segment merge on raw Hamming keeps the wrong segment's row. Hence
`SegmentSearchPhase::Approximate` (F16/Int8: owners return approximate candidates; the coordinator
merges, truncates to `candidate_k`, then issues ONE `ExactRescore` per owner — two round trips,
exactly `candidate_k` exact reads) versus `SegmentSearchPhase::Final` (Binary: every segment
rescores all its `width` hits before the merge and the merge runs on final distance — one round
trip, `N·width` exact reads, paid only here; F32: one `Final` phase, no rescore). The Binary
per-segment rescore lives in `SegmentedIndex::search_final` itself, so a single node with two Binary
segments gets the same fix.

**D7 — The peer RPC is `jammi.v1.peer.PeerService` on a separate internal listener
`[server] peer_bind`; the coordinator alone enforces tenant scope; the owner is tenant-free with a
segment-belongs-to-table check.** Principles: tenant scope is one generic predicate applied at the
input edge; the single-binder rule — `TenantResolverLayer` is the single binder for every engine
gRPC service (`crates/jammi-server/src/tenant_resolver_layer.rs`, module docs), so a service that
binds no tenant must not sit on that chain; transport authentication is the consumer's runtime.

- `peer_bind` is the third listener, the same config class as `health_listen`/`flight_listen`:
  `Option<String>`, default unset = not mounted = single node. `ServerConfig::validate` parses it
  and applies the three-way collision check (`addresses_collide`; `:0` never collides).
  `OssServer::bind` binds it and builds its routes OUTSIDE `assemble_grpc_chain`: never added to the
  public `Routes`, never wrapped by `TenantResolverLayer`, never advertised by `GetServerInfo`. The
  public listener answers `UNIMPLEMENTED` for its paths (tonic 0.14.5's `Routes` fallback).
- The request carries `(table_name, segment_ids, storage_precision, query, width, phase)` and no
  tenant. The coordinator resolves the table through its own tenant-scoped catalog before any
  fan-out (`Catalog::resolve_embedding_table` → `Catalog::get_result_table`'s
  `(tenant_id = $2 OR tenant_id IS NULL)` predicate, `crates/jammi-db/src/catalog/result_repo.rs`).
  **No admin scope on the peer path:** `with_admin_scope` is not used anywhere between `Search` and
  the fan-out.
- The owner verifies that every requested `segment_id` belongs to the named table via
  `Catalog::list_index_segments` and refuses the whole request otherwise. That listing is
  deliberately not tenant-filtered; its doc comment states the caller obligation (the table name was
  already resolved through a tenant-scoped read) and names the peer owner as the one caller that
  inherits that obligation from the coordinator.
- The RPCs are part of the frozen wire surface (`crates/jammi-server/tests/it/api_freeze_baseline.txt`).
  The tenant-isolation oracle carries them in their own bucket, `PEER_LISTENER_ALLOWLIST`
  (`crates/jammi-server/tests/it/tenant_isolation_oracle.rs`): every other exempt entry is
  control-plane or handler-less, while these are data-plane, handler-bearing and deliberately
  tenant-free, so the exemption's premise — the public listener does not serve them — is asserted
  in the same file (`peer_service_is_unimplemented_on_the_public_listener`).
- The owner role is not a service-tier token: a replica is a segment owner iff `peer_bind` is set.
  The same listener also carries `jammi.v1.gang.GangService` for multi-host training gangs (see
  `docs/plans/67-distributed-training/`), under the same trust statement.
- Written invariant **I-PEER** (§7).

**D8 — Bounded failure ladder; marginal-load admission; typed `UNAVAILABLE`; readiness unchanged;
one failure counter.** Principle: a peer outage is visible, never masked by a silent full scan of a
larger-than-memory index; invalid input is a typed refusal at the edge. On an all-local table any
segment load failure falls the WHOLE table back to exact search
(`ResultStore::resolve_search_mode_local`) — a durable-fault policy that stays only for the
all-local case. Per remote segment: per-RPC deadline → one retry at the next rendezvous candidate →
a local load admitted by `[server] peer_local_load_bytes` (§5.5) → `JammiError::Unavailable` naming
the segment, mapped to gRPC `UNAVAILABLE` with a typed wire detail. Readiness stays the catalog ping
(`crates/jammi-server/src/runtime.rs::CatalogPingProbe`): a peer outage must not eject the
coordinator from the load balancer. `jammi_peer_search_failures_total{reason}` is the observable.

**D9 — Placement is derived, never declared; membership is catalog rows behind one knob,
`[server] peer_advertise`; N>1 placement requires a shared, replica-readable result root.**
Principle: jammi's pluggable backends are locally detectable; a replica's identity in shared state
is not, so it is honestly a knob, not a backend. Segment ids are allocated at write time by max+1
with retry, invisible to any static manifest, so placement is DERIVED — a rendezvous hash of
`(instance_id, table_name, segment_id)` over the live ring — never declared in configuration. Every
session, library and CLI included, upserts an `instances` row, and `instances.host` is a label, so
membership cannot key on row presence: a row is a member only when it carries `peer_addr`, written
only when `peer_advertise` is set. With a local unshared `artifact_dir` the ring is one node; a
remote segment bundle is fetched through the content-addressed cache
(`crates/jammi-db/src/storage/index_cache.rs::SegmentIndexCache::load_segment`) only from a root
every replica can read. DNS-based membership and a static peer list are both rejected (§5.8).

**D10 — Online inference never touches the substrate; placement applies to the ONLINE retrieval
leaf only; batch consumers force-local.** `encode_query` runs in-process on the query tier;
`InferenceExec` and `AnnSearchExec` are single-partition leaves. Batch builders — the
neighbor-graph pipeline (`NeighborGraphPipeline::resolve_strategy`, which holds one
`SegmentedIndex` inside `IndexAssisted` across the whole build and calls the sync `search_final`)
and the eval runner (`EvalRunner::eval_embeddings`, a per-query loop) — use the force-local entries
(§5.2), which ignore placement and load every segment locally; any replica can, through the
content-addressed cache over the shared root. A batch build over a placed table is not an online
path and never fans out per node (D5); it holds the whole table's segment set resident on the
building replica. The context-set single-shot retrieval (`InferenceSession::ann_candidates`, one
search per request) uses the placed path.

**D11 — Structural sync/async split.** `SegmentedIndex` stays the all-local SYNC type and
`search_final` stays sync. `ResultStore::resolve_search_mode` returns an opaque `PlacedIndex` whose
ONLY search entry is `async fn search_final_placed`; its constructors are `pub(crate)`. There is no
flag or runtime guard: a remote source cannot reach a sync path by construction, because no
`SegmentedIndex` can contain one. With zero remote sources `search_final_placed` literally calls
`SegmentedIndex::search_final`.

**D12 — Remote equals all-local equals brute force, proven in one process and across processes.**
The in-process oracle is two engine instances in ONE test process over one SQLite catalog file and
one shared local root — in-process multi-pool SQLite is supported, cross-process is refused
(`crates/jammi-db/src/catalog/backend_sqlite.rs`, module docs). It proves transport + merge + ladder,
not object-store fetch, process isolation or network partition; those are proven by the
multi-process lane (real `jammi-server` workers over shared Postgres and an object store, a real
rendezvous ring, a real SIGKILL). §6 names the tests.

**D13 — Rescore-at-owner and duplicated row ids.** Segments are row-disjoint by the append
invariant; the merge dedups by row id keeping the nearest. On a refreshed table the version's
deletion mask decides which copy of a re-embedded key surfaces
(`SegmentedIndex::new_masked`, `DELTA-INCREMENTAL-EMBEDDING.md`). The coordinator additionally
refuses a peer answer that repeats a row id across units (§5.3).

**D14 — Ray / Ray Serve: recorded, not proposed.** Python-side collectives and serving are a second
runtime beside the one binary.

---

## 2. The candidates

| | Candidate | Shape |
|---|---|---|
| S1 | The jobs fleet | Durable catalog jobs claimed by `[worker]` processes; no query-plane component |
| S2 | Apache Ballista | Scheduler + executors; stage plans, shuffle exchange |
| S3 | `datafusion-distributed` | A library: stage plans over gRPC between processes embedding the crate; no scheduler |
| S4 | In-house `PeerService` | One unary RPC per segment owner per phase on `[server] peer_bind`; no plan shipping |

---

## 3. Per-candidate facts

Third-party rows read at 2026-09-10 unless noted.

| Fact | S2 Ballista | S3 datafusion-distributed | S4 in-house |
|---|---|---|---|
| Release / DataFusion tracked | 54.1.0 (2026-08-09) → DF 54 / arrow 58.3; no DF-55 release on crates.io | 4.0.0 (2026-08-20) → DF 55 / arrow-flight 59; 1.0.0 (2026-04-16) … 4.0.0 in four months | n/a — the workspace is on DF 54.1 / arrow 58.3 |
| Plan submission | logical, substrait, or physical (`execute_physical_plan` → `Query::PhysicalPlan`) | stage plans over gRPC; `with_distributed_user_codec` on both sides | per-segment RPC, no plan shipping |
| Custom `ExecutionPlan` | codec overrides on scheduler and executor | codec on both sides; leaf variants keep schema/partition count | n/a — the leaf stays local; only the segment unit crosses |
| Scheduler process / loops | scheduler + executors; `expire_dead_executors` liveness loop + stage dispatcher | none | none |
| Exchange | shuffle files on local `work_dir` | worker-to-worker (Flight; 4.0.0 framing not established) | one unary RPC per owner per phase |
| Resources / affinity | `ExecutorSpecification { task_slots }`; `TaskDistribution = Bias \| RoundRobin` — no accelerator, no affinity | `RouteTaskHandler` affinity | rendezvous hash over the live ring |
| Deployment | one binary can run both roles; the upstream Kubernetes doc provisions a PVC for shuffle | any process embedding the crate | same binary, `peer_bind` set |
| Latency | 50 ms status poll + disk shuffle: batch only | not established | 1 RTT (F32, Binary) or 2 RTT (F16/Int8) + slowest owner |
| Tenant on every executor | not established | hooks exist, unverified | coordinator-side scoped resolve before fan-out + owner table-membership check (D7) |
| Maturity / adopter | Apache; the one known production adopter rides forks of ballista, datafusion, federation and arrow-rs | created 2025-06-19; 135 stars, 89 open issues; production users not established | jammi-owned |

---

## 4. Decision matrix

Criteria: one binary / no engine scheduler loop / DataFusion coupling / custom plan support /
rescoring locality / per-partition tenant guarantee / off the online hot path / distributed SQL &
joins / cost. Scored for the retrieval data plane.

| | one binary | no loop | DF coupling | custom plan | rescore local | tenant | off hot path | dist. SQL | cost |
|---|---|---|---|---|---|---|---|---|---|
| S2 | ✓ | ~ (liveness only) | ~ (tracks one DF major; four release trains) | ~ | ✗ (no affinity) | ✗ | ✗ | ✓ | high |
| S3 | ✓ | ✓ | ✗ (DF 55; 4 majors / 4 months) | ~ | ~ | ~ | ~ | ✓ | medium now / high churn |
| S4 | ✓ | ✓ | ✓ (none) | ✓ (n/a) | ✓ | ✓ (D7) | ✓ | ✗ (retrieval only) | low–medium |

Outcome: batch inference = S1 (D1); online beyond-one-node retrieval = S4 (D3); distributed
SQL/joins = S3 deferred (D4); S2 is the compute/gang plane, not the data plane (D2).

---

## 5. Design — the S4 mechanism

### 5.1 Vocabulary and types (`crates/jammi-db/src/index/`)

`peer.rs` — the vocabulary a coordinator and a segment owner exchange; nothing in it speaks a wire
protocol:

- `PeerAddr` — the address a coordinator dials an owner at (`host:port`, plaintext gRPC; transport
  encryption is the runtime's). Defined once in `crates/jammi-db/src/catalog/instance.rs` and
  re-exported, because the peer listener and the gang listener are the same address. Sealed: built
  only by `PeerAddr::parse`.
- `SegmentSearchPhase { Approximate, Final }` — `for_precision`: `F32 | Binary → Final`,
  `F16 | Int8 → Approximate`.
- `SegmentUnit { segment_id, hits: Vec<(String, f32)> }` — `(row_id, distance)`, never vectors:
  `search` stays the one way embeddings are consumed.
- `SegmentSearchRequest { table_name, segment_ids, storage_precision, query: ValidatedQuery, width,
  phase }` and `ExactRescoreRequest { table_name, storage_precision, query, row_ids_by_segment }`.
  The query is validated (finite) before it crosses the seam and re-validated at the owner's edge.
- `PeerError { segment, owner, reason, message }`; `PeerFailureReason { Deadline, Unreachable,
  Refused, Torn, Transport, Malformed, CallerFault }`.
- `trait PeerTransport` (`segment_search`, `exact_rescore`, each taking a deadline). The trait lives
  in jammi-db, which depends on no other jammi crate and has no `tonic`; the tonic implementation
  `GrpcPeerTransport` (one lazily-connected channel per owner) lives in
  `crates/jammi-wire/src/peer.rs` and is handed to the store by jammi-ai's store builder. `NoPeers`
  (every call `Unreachable`) is the default.
- `trait SegmentPlacement { async fn plan(&self, table, segments: &[SegmentId]) ->
  Result<Vec<Vec<PeerAddr>>> }` — one call for the WHOLE segment set of one table, so every segment
  of a query sees one ring snapshot. Exactly one entry per requested segment (the store checks the
  arity and refuses a mismatch); an empty entry means "local", a non-empty one is the
  rendezvous-ordered candidate list (first = owner, second = the one retry). `Err` on a catalog read
  failure — never a silent per-segment fallback to local, which would exact-scan a multi-node
  table's segment behind its owner's back. Implementations: `AllLocal` (default),
  `StaticPlacement(BTreeMap<(String, SegmentId), Vec<PeerAddr>>)` for an embedder that knows its
  topology, and `RendezvousPlacement` (§5.8).
- `PeerFailureCounters` — one `AtomicU64` per label in `PEER_FAILURE_LABELS`
  (`deadline, unreachable, refused, torn, transport, malformed, caller_fault, retry_ok, local_load,
  unavailable`), exposed by `ResultStore::peer_failures`.
- `PEER_RPC_DEADLINE = 2 s`; the local-load rung gets twice that.

`segment.rs` — three pure, sync kernels that carry no transport and run identically at an owner and
at a coordinator:

- `search_unit(segment, index, query, width, phase, exact)` — HNSW at `width` (capped at the
  segment's row count, never padded); for `Final` on a rescoring precision, rescore every hit.
- `merge(units, m)` — the total order and dedup, keeping the winning segment id.
- `rescore(segment, candidates, exact, query)` — exact cosine over named candidates; a missing exact
  vector is a hard error, never a silent drop.

The exact-vector lookup is a closure, so an exact-read count is observable without instrumenting
`SidecarIndex`. `SegmentedIndex::search_final` is expressed over these kernels. Both surfaces apply
one admissibility predicate to every distance (`crates/jammi-db/src/index/mod.rs::distance_is_admissible`,
finiteness): a distance is the merge's sort key, and one `-NaN` would take the whole top-`k`.
Locally a violation is a typed error naming the segment; from a peer it is `Malformed`.

`placed.rs`:

- `enum SegmentSource { Local(SegmentId, SidecarIndex), Remote { segment_id, owners, row_count,
  index_url } }` — a remote source carries its bundle URL and catalog row count because nothing is
  loaded for it unless the ladder's local-load rung admits it.
- `PlacedIndex` — opaque; holds `Placed::AllLocal(Arc<SegmentedIndex>)` or
  `Placed::Mixed { local, remote }`, the table name and precision, the transport, the segment cache
  and ANN config (for the local-load rung), the admission budget, the table's dimensions
  (`Option<NonZeroUsize>`, converted once from the catalog row) and the counters. Local sources'
  precision uniformity is asserted at construction; a remote source's is asserted by the owner's
  strict load.
- `len()` = loaded rows for local segments + catalog `row_count` for remote ones.
- `search_final_placed(query, k, oversample)` — the only search entry (§5.3).

### 5.2 Two resolve entries and the consumer routing table

`ResultStore` carries `placement`, `peer_transport`, `peer_local_load_bytes` and `peer_failures`
(builders `with_placement`, `with_peer_transport`, `with_peer_local_load_bytes`; accessors
`segment_cache`, `peer_failures`).

- `resolve_search_mode_local(table) -> Option<Arc<SegmentedIndex>>` — the force-local entry. No
  segments → `None`; any segment load failure → `warn!` + `None` (whole-table exact fallback, never
  a subset: dropping a failed segment would make its rows silently unsearchable). It is
  version-aware: a refreshed table resolves its current manifest and deletion mask. A ready table's
  loaded set is cached across calls.
- `resolve_search_mode(table) -> Option<PlacedIndex>` — the online entry. Lists segments; none →
  `None`; calls `placement.plan` ONCE for the whole set. If every owner list is empty it wraps
  exactly what `resolve_search_mode_local` loads (`PlacedIndex::from_local`), preserving the
  whole-table exact fallback and the mask. Otherwise it builds `Mixed`: local segments are loaded,
  remote ones recorded with owners and not loaded; a local load failure in `Mixed` is
  `JammiError::Unavailable` — a multi-node table is never exact-scanned silently. The `Mixed` path
  reads the flat segment list and is not version-aware: multi-node placement of a refreshed,
  versioned table is outside this design (§8).
- `search_vectors` — placed: `resolve_search_mode` → `search_final_placed`, `None` → exact.
- `search_vectors_local` — force-local twin over `resolve_search_mode_local` + sync `search_final`.

| Consumer | Site | Entry | Why |
|---|---|---|---|
| Online `Search` leaf | `AnnSearchExec::execute` | `resolve_search_mode` → `search_final_placed` | the one online retrieval leaf (D10) |
| Context-set retrieval | `InferenceSession::ann_candidates` | `search_vectors` (placed) | one search per request; online |
| Neighbor-graph build | `NeighborGraphPipeline::resolve_strategy` | `resolve_search_mode_local` | batch; holds one `SegmentedIndex` across the build (D10) |
| Eval runner | `EvalRunner::eval_embeddings` | `search_vectors_local` | batch per-query loop (D10) |
| Recall bench | `crates/jammi-bench/src/recall.rs` | its own `SegmentedIndex` | sync, all-local |

Precondition: N>1 placement requires `storage.result_root` (or a shared local `artifact_dir`) to be
a root every replica can read. Membership admits only replicas whose root identity matches (§5.8);
that is necessary, not sufficient — whether the location is actually readable by every replica is
the deployer's obligation, and an owner that cannot load a bundle is a ladder failure, not a wrong
answer.

### 5.3 Width and the per-precision protocol (`PlacedIndex::search_mixed`)

The query's width is checked against one authority exactly once, before any local search or
fan-out: the catalog's recorded `dimensions` when present, else the first local segment's width. An
all-remote set with no recorded width has no authority and is refused as an engine-side fault
(`IncompatibleFormat` naming `{table}.dimensions`) rather than fanned out unguarded. A caller's
wrong-width query is therefore refused at the coordinator and never reaches an owner.

`N` = local + remote segment count; `candidate_k = max(k, k·oversample)` for rescoring precisions
and `k` for F32; `width = over_fetch(candidate_k, N)`. Remote segments sharing one owner list ride
ONE request per phase, in parallel across owner groups (`futures::future::join_all`); local sources
run `search_unit` in-process.

- **F32 — one `Final` phase.** Each source returns its top-`width` exact cosine hits;
  `merge(units, k)`. 1 RTT.
- **F16 / Int8 — `Approximate` then `ExactRescore`.** Each source returns its top-`width`
  approximate hits; `merge(units, candidate_k)` truncates on approximate distance; survivors are
  grouped by winning segment — local ones (including segments the ladder loaded locally) through
  `rescore`, remote ones in ONE `ExactRescore` per owner group; the coordinator sorts by
  `(distance, row_id)` and truncates to `k`. 2 RTT; exactly `candidate_k` exact reads — a single
  node's count.
- **Binary — one `Final` phase.** Each source rescores all its `width` hits where the segment lives
  and returns exact distances; `merge(units, k)` on final distance. 1 RTT; `N·width` exact reads.
- **No remote source** never reaches this code: `Placed::AllLocal` calls `search_final`, which is
  byte-identical to a lone `SidecarIndex` at N=1 for every precision.

Every peer answer is reconciled against the request it answers, at the coordinator, so every
`PeerTransport` implementation is covered (`reconcile_units`, `reconcile_rescore`): exactly one
unit per requested segment, none unrequested or repeated, none wider than `width`, every distance
finite, no row id twice across the whole answer (segments are row-disjoint, so a repeat is knowably
wrong — and costly: the merge would attribute the row to a segment that does not hold it, and the
rescore phase would walk the whole ladder against a healthy bundle); a rescore answer names exactly
the requested rows, each once. A non-conforming answer is a `Malformed` failure of that rung — never
a result, never a panic.

### 5.4 The peer RPC

**Proto** (`crates/jammi-wire/proto/jammi/v1/peer.proto`, `package jammi.v1.peer`):

```
service PeerService {
  rpc SegmentSearch(SegmentSearchRequest) returns (SegmentSearchResponse);
  rpc ExactRescore(ExactRescoreRequest) returns (ExactRescoreResponse);
}
```

Unary, not streaming: a response is at most `N_owner·width` `(row_id, f32)` pairs, and a unary call
carries the per-RPC deadline as the gRPC deadline directly. Messages mirror §5.1 field for field.
`StoragePrecision` and `SegmentSearchPhase` each have an `UNSPECIFIED = 0`; the owner distinguishes
"not set" (request malformation) from "a non-zero value this build does not know" (a newer
coordinator — rolling-upgrade skew).

**Listener** (`crates/jammi-server/src/runtime.rs::OssServer::bind`,
`BoundServer::serve_with_shutdown`). The third `TcpListener` is bound with the other two, so a `:0`
request reports its real port (`BoundServer::peer_addr`). It is served by its own tonic server over
a `TcpListenerStream`, carrying only `MetricsLayer` — so `jammi_grpc_requests_total` and
`jammi_peer_requests_total{rpc}` count peer calls — with no tenant layer, no gRPC-web framing and no
`[server.limits]` stack; `[server.limits] max_message_bytes` still bounds its inbound decode. It
stays up through a drain, like the health side-channel, and is stopped last on its own signal: a
coordinator's fan-out to a draining owner is never cut early.

**Owner handler** (`crates/jammi-server/src/grpc/peer.rs::PeerServer`). (1) List the table's
segments with no tenant filter; (2) verify the request; (3) load each segment through the
content-addressed cache at the requested precision — the strict manifest check refuses a bundle
stamped otherwise; (4) run `search_unit` (or `rescore`) per segment; (5) return units. The owner
reads no `result_tables` row and binds no tenant (I-PEER). It loads per RPC through the cache (a
`file://` bundle loads in place; a remote bundle is fetched once per process).

Refusals split on whose fault they are, because the coordinator treats the two classes differently:

| Owner status | Meaning | Coordinator classification |
|---|---|---|
| `INVALID_ARGUMENT` | the REQUEST is malformed: empty or duplicated segment ids, a duplicated rescore row id, a non-finite query component, a `width` that does not fit, an `UNSPECIFIED` enum | `CallerFault` — TERMINAL: no retry, no local load, no `Unavailable`; surfaced as an internal error naming owner and segment, because the coordinator built the request |
| `FAILED_PRECONDITION` / `NOT_FOUND` | a disagreement about the owner's OWN data: a segment id absent from its list (its read can race an append or purge), a bundle stamped at another precision, a width drifted from the coordinator's authority, a rescore row it does not index, an enum value from a newer coordinator | `Refused` — ladders |
| `DATA_LOSS` | a torn bundle (a candidate with no exact vector) | `Torn` — ladders |
| `DEADLINE_EXCEEDED` / `UNAVAILABLE` / other | | `Deadline` / `Unreachable` / `Transport` — ladder |

(`crates/jammi-wire/src/peer.rs::classify_status`.)

**Coordinator path.** `InferenceSession::search` → tenant-scoped `resolve_embedding_table` →
`AnnSearchExec` → `resolve_search_mode` → `search_final_placed`. The scoped resolve precedes any
placement lookup, so a coordinator bound to tenant B cannot name tenant A's table: it fails
not-found before any fan-out.

### 5.5 Failure ladder, marginal-load admission, error surface

Per owner group, per phase (`PlacedIndex::call_with_retry`, `PlacedIndex::load_locally`); every rung
emits a `warn!` naming table, segment, owner and reason, and increments its counter:

1. Call `owners[0]` under `PEER_RPC_DEADLINE`.
2. On failure, call `owners[1]` once if present — the next rendezvous candidate; any owner can serve
   any segment of a shared root, so this holds for `ExactRescore` too. Success counts `retry_ok`.
3. On failure, load each of the group's segments locally under `2 × PEER_RPC_DEADLINE`, **iff**
   admitted: the table records its dimensions, and `loaded_this_query + estimate(seg) ≤ budget`
   (unset budget = unbounded). `estimate(seg) = row_count × (vector_bytes + 32 + 64)` with
   `vector_bytes = 4d (F32) | 2d (F16) | d (Int8) | ceil(d/8) (Binary)`, 32 the assumed mean row-id
   bytes and 64 the assumed per-node graph link overhead (`placed.rs::local_load_estimate`). Success
   counts `local_load`, and the loaded segment serves the rescore phase too.
4. Otherwise `JammiError::Unavailable { resource: "segment {table}/{id}", reason }` → gRPC
   `UNAVAILABLE`, wire detail `UnavailableError` (`crates/jammi-wire/proto/jammi/v1/error.proto`),
   counted `unavailable`.

A `CallerFault` exits the ladder at once (§5.4).

**`[server] peer_local_load_bytes: Option<u64>`** — plain integer bytes; `Some(0)` is refused by
`ServerConfig::validate`. It lives in `[server]`, its reader is `ResultStore`, and a library
embedder sets it through the same config. It is MARGINAL-LOAD ADMISSION per query: the maximum
estimated bytes ONE query may load locally for segments it does not own when their owners are
unreachable. It is NOT a memory cap: the segment cache never evicts, earlier queries' loads are
invisible to the check, distinct remote bundles accumulate on disk, and concurrent queries admit
independently, so peak heap is concurrency × budget. The estimate is a LOWER bound for quantized
precisions: the raw-f32 companion is excluded because it is a file read by `pread`, never resident
(including it would over-estimate Int8 by roughly 3.7×), while usearch's level-0 links and the
row-id `HashMap` are unmodelled. Guidance: set the budget to at most half the memory one query's
fallback loads may take. The neighbor-graph build never reaches this rung (it is force-local), so
the per-query framing holds for every site the ladder serves.

### 5.6 Observability

`PeerFailureCounters` is read at scrape by a collector registered additively
(`crates/jammi-server/src/routes/health.rs::MetricsRegistry::install_peer_failures`) as
`jammi_peer_search_failures_total{reason}`. `RendezvousPlacement` exposes
`jammi_placement_ring_empty_total` through the `SegmentPlacement::ring_empty_metrics` hook
(`install_ring_empty`); it is registered only when the placement returns `Some`, so a default
deployment's `/metrics` is unchanged. Owners count served calls as
`jammi_peer_requests_total{rpc}`. Readiness is untouched. The operator-facing statement is the
"segment owner is unreachable" row of `docs/guide/src/operability.md#failure-mode-matrix`.

### 5.7 Latency and cost

No remote source: a single node's path and exact-read count. With remote sources: F32 and Binary =
one parallel round trip bounded by the slowest owner plus the merge; F16/Int8 = two. Worst case per
phase = 2 × `PEER_RPC_DEADLINE` (owner + retry) + 2 × `PEER_RPC_DEADLINE` (local load) = 8 s, so
16 s for F16/Int8; owner groups fail in parallel, so failed groups do not add sequentially. The ring
read adds one untransacted catalog statement per placed search (§5.8). `AnnCache` holds merged
results only and is unaffected; online inference is unaffected (D10).

### 5.8 Membership

Membership is rows in the shared catalog, written by each replica about itself and read on the
query path. There is no gossip, no DNS lookup, no static peer list and no background loop.

**Why catalog rows.** Every replica of a deployment already shares the catalog and already
heartbeats an `instances` row under the deployment's lease window, so liveness costs nothing new and
has the same clock and margin as every other leased row family. A static peer list in configuration
cannot express liveness, goes stale on every reschedule, and would be a second membership mechanism
beside the one a training gang needs; DNS conflates "resolvable" with "serving this root". One
mechanism serves both the retrieval ring and gang assembly.

**Joining.** `[server] peer_advertise` is the address other replicas dial this process's `peer_bind`
listener at (`peer_bind` is commonly `0.0.0.0:PORT`, unusable as a dial target). The whole
eligibility check is one choke point, `MembershipConfig::validate`
(`crates/jammi-db/src/catalog/instance.rs`), reached by both `JammiConfig::load_from` and
`InstanceRegistration::from_config`, so a struct-literal config cannot skip it: `peer_advertise`
must parse as a `PeerAddr` and requires `peer_bind` (a typed error naming both keys);
`[server] placement = "rendezvous"` requires `peer_advertise`. `ServerConfig::validate` is not the
home: it cannot see `artifact_dir`, which the result root depends on. `InstanceRegistration` is the
one value every writer of the `instances` row builds (`Catalog::upsert_instance`,
`Catalog::reregister_instance`); every session constructor funnels through it. A process that never
sets `peer_advertise` — a library, the CLI, a plain server — writes `peer_addr` and the root columns
`NULL` and is never a member. `instances.host` stays a label.

**Root identity.** A member row carries the configured result root verbatim
(`instances.result_root`, for humans) and its `RootIdentity` (`instances.result_root_identity`, for
the predicate) — migrations `035_instances_peer_addr_result_root` and
`036_instances_result_root_identity`. The identity is the root *across spellings*, derived once, at
registration, by the process that owns the root (`MemberRoot::resolved`): parsed by the same
`StorageUrl` parser the store roots itself through (so `gcs://`/`gs://` and `abfss://`/`azure://`
fold by the one alias table), an object-store key normalised by the same `object_store` path parser,
the endpoint or account the store would dial included (two buckets of one name behind two endpoints
are two locations), a local root created and canonicalised on the owner's filesystem, and
`memory://` refused as unshareable. The store still roots at the verbatim string; the identity is
used only for equality. Two replicas are members of each other iff their identities are equal; a
`NULL` identity never matches. Equality is necessary for shared storage, never sufficient.

**The live-with-my-root predicate.** One SQL fragment, `live_with_root_clause`: `peer_addr IS NOT
NULL AND result_root_identity = (<mine>) AND NOT stale`, where staleness is `last_seen_at` older
than `instance_liveness_margin` = 2 × the lease window
(`crates/jammi-db/src/catalog/lease.rs::instance_liveness_margin`, rendered sargably by
`stale_before_clause`). Twice the lease, so one missed heartbeat does not drop a member; rows are
pruned only at 3 × the lease, strictly beyond the margin. Two callers share the fragment so "live
with my root" has one definition:

- `Catalog::list_ring_members` — the retrieval ring. No `workers` join and no job-kind vocabulary
  (placement is generic over what an owner does); the caller's OWN row is a member; the caller's
  identity is named by a self-referencing subquery, so its root identity has one source of truth.
  One statement, issued untransacted against the pool.
- `Catalog::list_gang_members` — gang assembly: additionally joins `workers`, requires a `claiming`
  worker whose `kinds` contains the wanted kind as a whole comma-split token, and excludes the
  caller. `Catalog::peer_addr_of` resolves one member by id under the same freshness margin.

A consequence of the shared mechanism: a replica that sets `peer_advertise` to be gang-reachable is
also an owner candidate in every same-root coordinator's retrieval ring. Scoping ring membership by
capability is not part of this design.

**Rendezvous placement** (`crates/jammi-db/src/index/peer.rs::RendezvousPlacement`, selected by
`[server] placement = "rendezvous"`; the default `"local"` is `AllLocal`;
`InferenceSession::open_with_placement` overrides both with an explicit `SegmentPlacement`). For
each segment every ring member, self included, is scored by highest-random-weight hashing: the
big-endian `u64` of the first 8 bytes of `domain_hash(PLACEMENT_HASH_DOMAIN, [instance_id, table,
segment_id])`, each field length-prefixed so no two inputs collide across a shifted boundary, under
a domain tag (`jammi.placement.v1`) distinct from every other hash the crate computes. Members are
sorted by score descending, ties broken by `instance_id` byte order — a pure function of the ring's
content, never of SQL row order. If the top-ranked member is this process the segment is local;
otherwise the candidates are `[first, second]` (the second may be this process, reached through its
own listener).

Why rendezvous hashing: segment ids are allocated at write time, so ownership must be computable by
every replica from `(ring, table, segment_id)` alone with no assignment table to write, lease or
repair; and a membership change must move only the segments the departed or arrived member wins —
HRW's minimal-disruption property — so one replica restarting does not reshuffle every owner's
working set. The second-ranked member is, by the same property, the member that becomes owner if
the first leaves, which is why it is the retry target.

The ring is read fresh on every `plan` call — once per placed search, above the per-segment loop —
never cached and never refreshed by a loop (D5). An empty ring, or a ring that does not contain the
caller's own row (a construction race, or a pruned self row), yields an all-local plan and counts
`jammi_placement_ring_empty_total`: that case changes behaviour and would otherwise be
indistinguishable from a healthy one-node ring. A row excluded for staleness, a missing address or
a foreign root is not counted by reason — distinguishing them would cost a second statement per
search for a fact only debugging uses. A ring of one maps every segment to self — the single-node
path.

Measured cost of the ring read on Postgres with a warmed connection: ~3.9–5.0 ms at 101 `instances`
rows (51 candidates) — the one-round-trip floor — and ~10.2–10.9 ms at 10,101 rows (5,051
candidates), of which ~3.3 ms is execution (a sequential scan; `idx_instances_seen` covers only the
liveness conjunct and the schema has no index on `result_root_identity`) and the rest is transfer
and decode proportional to the candidate count. The larger shape is a stress fixture; a real ring is
bounded by one deployment's live replica count. Skipping the transaction wrapper matters: on
Postgres it would add `BEGIN`, two `SET TRANSACTION` statements and `COMMIT` around every search.

---

## 6. Properties the tests hold

- **Binary merges on final distance.** A two-segment Binary fixture whose raw-Hamming merge keeps
  the wrong segment's row returns the brute-force answer —
  `segment.rs::tests::two_segment_binary_search_final_rescores_per_segment_before_the_merge`.
- **Exact-read counts per precision** (F16/Int8 exactly `candidate_k` at N=1 and N=2; Binary
  `N·width`; F32 zero) — `segment.rs::tests::exact_read_count_per_precision`.
- **N=1 byte identity on both entries** —
  `segment.rs::tests::n1_search_final_is_byte_identical_to_the_lone_sidecar_at_every_precision`,
  `placed.rs::tests::all_local_placed_search_is_byte_identical_to_segmented_search_final`.
- **Peer answers are reconciled** — `placed.rs::tests`
  (`non_conforming_search_units_are_a_typed_ladder_failure`,
  `duplicate_row_across_units_is_a_typed_ladder_failure`,
  `non_finite_distances_from_a_peer_are_a_typed_ladder_failure`,
  `non_conforming_rescore_rows_are_a_typed_ladder_failure`).
- **Listener and owner** — `crates/jammi-server/tests/it/peer_service.rs`:
  `peer_bind_unset_means_no_third_listener`;
  `segment_search_over_peer_bind_equals_in_process_search_unit`;
  `exact_rescore_over_peer_bind_equals_in_process_rescore`; `owner_refuses_non_conforming_requests`
  and `owner_refuses_non_conforming_requests_with_invalid_argument` (the two refusal classes).
- **The public listener does not serve the peer surface; the wire surface is frozen and covered** —
  `tenant_isolation_oracle.rs::peer_service_is_unimplemented_on_the_public_listener`,
  `every_rpc_is_covered`, `allowlist_and_cases_partition_the_wire_surface`; `api_freeze.rs`.
- **Placed search end to end, two instances in one process**
  (`crates/jammi-server/tests/it/peer_placement.rs`):
  `placed_search_over_two_instances_equals_all_local_and_brute_force`;
  `ladder_retries_then_loads_locally_or_refuses_unavailable` (retry, budget refusal, no recorded
  dimensions, admitted local load, readiness 200 throughout);
  `coordinator_under_another_tenant_fails_before_fan_out`;
  `force_local_entries_ignore_placement_while_placed_entries_refuse`;
  `a_caller_width_fault_is_refused_before_any_fan_out`;
  `all_remote_placement_refuses_a_caller_fault_before_any_fan_out`;
  `an_owner_caller_fault_is_terminal_and_classified_from_a_real_status`.
- **Ring predicate and placement** — `crates/jammi-db/tests/it/rendezvous_ring.rs` on SQLite and
  Postgres (`self_appears_as_a_candidate`, `a_stale_row_is_excluded`,
  `a_root_identity_mismatched_row_is_excluded`, `a_non_advertising_process_is_not_a_member`,
  `plan_falls_back_to_all_local_and_counts_when_self_is_absent_from_the_ring`,
  `ring_read_cost_is_measured_at_100_and_10k_instance_rows`); `peer.rs::rendezvous_tests`
  (`rank_is_independent_of_ring_encounter_order`, `minimal_disruption_on_membership_change`,
  `rendezvous_score_matches_a_hand_rolled_sha256_fold`); `crates/jammi-db/tests/it/arity_guard.rs`;
  `crates/jammi-server/tests/it/rendezvous_metrics.rs`.
- **Across real processes** —
  `crates/jammi-ai/tests/distributed/placed_search.rs::rendezvous_placed_search_over_real_worker_processes`
  (feature `live-distributed-tests`).
- **Remote equals embedded is unchanged under `AllLocal`** —
  `crates/jammi-server/tests/it/grpc_remote_session.rs::remote_round_trips_embeddings_and_search_like_local`.

---

## 7. Invariants and how each is preserved

- **Engine, not platform.** The vocabulary is table / segment / query / peer / owner; no consumer is
  named.
- **Embeddings are consumed through `search`.** The peer RPC is an engine-internal seam returning
  ids and distances, never vectors; `ExactRescore` returns distances computed at the owner.
- **Topology is configuration; one binary.** `peer_bind`, `peer_advertise` and `placement` are
  configuration. `SegmentedIndex`, `PlacedIndex`, placement, the transport trait and the ladder live
  in jammi-db; jammi-wire owns the client; jammi-server only mounts the handler. A library process
  is a full coordinator with `StaticPlacement` and `GrpcPeerTransport`.
- **Tenant isolation.** Tenant scope is the same generic predicate, applied once at the
  coordinator's resolve; the owner path is documented tenant-free and gated by I-PEER.
- **I-PEER.** Every client of `peer_bind` is a jammi coordinator; the owner trusts the channel and
  enforces only segment-belongs-to-table. Binding `peer_bind` on a routable interface without
  network policy or mTLS exposes cross-tenant reads; the default is unset. Stated for operators in
  `docs/guide/src/security.md#the-peer-listener-i-peer`, `docs/guide/src/deploy-server.md` and
  beside the knob in `docs/guide/src/configuration.md`.
- **Typed refusal at the edge.** The owner validates ids, precision, width and finiteness at its
  input edge; the coordinator refuses to exact-scan a multi-node table; configuration refuses
  listener collisions, `peer_local_load_bytes = 0`, `peer_advertise` without `peer_bind`, and
  `placement = "rendezvous"` without `peer_advertise`.
- **Remote equals embedded.** §6's byte-identity and end-to-end properties.
- **Append-only migrations; additive wire surface.** Membership adds columns by numbered migrations;
  the peer package extends the frozen `jammi.v1` surface additively.
- **Actuator rule (D5).** No background loop: placement and membership are read on the query path;
  the ladder is bounded per request; batch builders never fan out.

---

## 8. Not part of this design

Sharded embedding jobs with a fan-in publish (D1); S3 adoption and any DataFusion upgrade for its
sake (§9); an object-store shuffle for Ballista (D2); multi-node placement of a refreshed, versioned
table (§5.2); capability-scoped ring membership (§5.8); compaction, re-quantization and an mmap
`view()` of a segment; an owner-side resident segment set across RPCs; retrieval-heavy SQL and graph
builds beyond one node; eval beyond one node (force-local by D10); the multi-seed recall bench that
guards `DEFAULT_SEGMENT_OVERFETCH_FACTOR`; TLS, mTLS and network policy (the runtime's); Ray.

---

## 9. Gates for distributed SQL (S3)

S3 is admissible when:

- (a) jammi is on the same DataFusion major as `datafusion-distributed`, with every
  DataFusion-facing dependency released at that major. The workspace moves as one line:
  `deny.toml` bans multiple versions of `datafusion`, `datafusion-federation`,
  `datafusion-flight-sql-server` and `datafusion-table-providers` and fences the line at DF 54 /
  arrow 58 (oracle: `crates/jammi-db/tests/it/datafusion_version.rs`). What holds it at 54:
  `ballista-*` 54.1 (DataFusion ^54, no DF-55 release) and `datafusion-table-providers` 0.13.1
  (DataFusion ^54). `datafusion-flight-sql-server` is not the blocker — 0.4.19 is on DataFusion
  55.1 and no longer depends on `datafusion-federation`; `datafusion-federation` 0.5.6 is on 55.
- (b) two consecutive `datafusion-distributed` majors at least 60 days apart.
- (c) a spike proving tenant re-injection on the worker side. Handle re-resolution (`ModelCache`,
  `ResultStore`, `SessionContext` rebuilt against the receiving session) has a working precedent in
  `crates/jammi-ballista/src/codec.rs::JammiCodec`.

A DataFusion upgrade is not scheduled for S3's sake; revisit when (a) and (b) hold.
`datafusion-federation` is pinned exactly (`=0.5.5`, the last 0.5.x on DataFusion 54) because 0.5.6
is semver-compatible with a caret requirement yet pulls DataFusion 55, so a bare `cargo update`
would drag a second DataFusion into the graph.

---

## 10. References

Read 2026-09-10 unless noted.

- Ballista: crates.io `ballista` (max 54.1.0); apache/datafusion-ballista `main` `Cargo.toml`
  (DF 55 / arrow 59.2, unreleased) and tag 54.1.0 (DF 54 / arrow 58.3). At 54.1.0:
  `ballista.proto` (`ExecuteQueryParams` oneof including `physical_plan`; `ExecutorResource`),
  `scheduler/src/cluster/mod.rs` (`ClusterStorage`, `ClusterState`, `JobState`,
  `BallistaCluster::new`), `core/src/execution_plans/shuffle_writer.rs` (`work_dir`),
  `scheduler/src/config.rs` (`TaskDistribution`), `scheduler/src/scheduler_server/mod.rs`
  (`expire_dead_executors`), `executor/src/executor_process.rs` (`override_execution_engine`),
  `core/src/execution_plans/distributed_query.rs` and
  `client/tests/physical_plan_submission.rs`; the architecture, `standalone`, Kubernetes (PVC
  shuffle) and configuration docs; the Spice AI blog post "Apache Ballista at Spice AI" and
  spiceai/spiceai trunk `Cargo.toml` (forked dependency lines).
- `datafusion-distributed`: crates.io versions (1.0.0 2026-04-16 → 4.0.0 2026-08-20); `main`
  `Cargo.toml` (DF 55, arrow-flight 59); docs (no scheduler binary; `with_distributed_user_codec`;
  `RouteTaskHandler`; `WorkerResolver`); GitHub API (created 2025-06-19; 135 stars; 89 open issues).
- crates.io `datafusion` versions (≈ two months per major); `datafusion-flight-sql-server` 0.4.18
  (DataFusion 54, federation 0.5.5; `src/service.rs` implements `do_get_fallback`) and 0.4.19
  (DataFusion 55.1); `datafusion-federation` 0.5.5 (DataFusion 54) and 0.5.6 (DataFusion 55);
  `datafusion-table-providers` 0.13.1 (DataFusion 54).
- usearch 2.25.1 `include/usearch/index_plugins.hpp` (`cast_to_i8_gt` scales by the vector's own
  magnitude); tonic 0.14.5 `src/service/router.rs` (`Routes` fallback = `Status::unimplemented`).
