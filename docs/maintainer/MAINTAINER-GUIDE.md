# Jammi — Maintainer Guide

This is the maintainer reference for the Jammi workspace: a Rust workspace
(13 crates) that turns sources (files, Postgres, MySQL) into embedded/searchable
result tables, runs model inference and LoRA fine-tuning, and exposes a SQL
surface (DataFusion + Flight SQL) plus a typed gRPC API. The same verb vocabulary
(`search`, `infer`, `sql`, `fine_tune`, …) is served two ways from one
definition: an embedded in-process engine and a remote gRPC client, with the
transport chosen by configuration, not by a separate code path. That
"transport-as-config" property is the load-bearing architectural idea.

Anchors in this guide are `path/file` plus the enclosing symbol. The cookbook is
the engine's executable acceptance suite; chapter paths are cited where a feature
has an authoritative consumer spec.

This document is unpublished: it lives under `docs/` but outside
`docs/guide/src/`, so mdBook does not render it to the published guide. Roadmap,
tech-debt, and first-PR material live separately under
`docs/plans/52-maintainer-roadmap/`.

---

## 0. Orientation

**What Jammi is.** Jammi is an embeddable and serveable AI retrieval/inference
engine. The lay of the land: 13 crates (`Cargo.toml`, the `[workspace] members`
list). The decisive seam is the **candle split**: a candle-free client substrate
(`jammi-wire`, `jammi-admin`, `jammi-client`, and the `jammi` CLI) that speaks
only the wire and pulls no ML stack, versus the embedded engine (`jammi-ai` + its
default `local` feature) that compiles candle / hf-hub / tokenizers / symphonia.
Below the engine sit the leaf crates `jammi-numerics` (pure math), `jammi-db`
(catalog/storage/SQL/index), `jammi-lora` (LoRA primitives), and `jammi-encoders`
(candle transformers). Above it sits `jammi-ballista` — the Ballista compute
plane, publishable and lockstep with the rest of the workspace, no cargo
feature — which depends on `jammi-ai`/`jammi-db`/`jammi-wire` and which
`jammi-server` depends on unconditionally; above THAT sits `jammi-server`
(serves the wire over the engine) and `jammi-python` (a local-only PyO3
cdylib whose remote arm is the bundled pure-Python client).

**Engine, not platform.** The governing house rule (`CLAUDE.md`): Jammi names no
consumer anywhere — code, config, docs, tests, fixtures. References point one way
(a consumer may depend on Jammi; Jammi depends on no consumer). This is enforced
in CI by `ci/scripts/check_dep_direction.py` [§6, §2 contract 2.4]. The second
house rule is **atomic-across-the-workspace**: a behavior change ships across
every affected crate in one PR, split by capability never by crate, with no
back-compat shims [§5].

**Where the source-of-truth docs live.** The philosophy text is inlined in
`CLAUDE.md` ("Engine, not platform") and `docs/guide/src/philosophy.md`; the
implementation roadmap is `docs/plans/50-open-core-hardening-roadmap/ROADMAP.md`.
`docs/plans/` is gitignored, so a fresh clone may lack plan docs.

---

## 1. System topology & dependency graph

### 1.1 The crate graph

Two complementary views. The **auto-generated block** immediately below is the *compiler/build-link*
graph, refreshed from `build-graph` and freshness-gated in CI (`ci/scripts/gen_dep_dag.py` +
`.github/workflows/dep-dag.yml`). It reflects what the compiler links, so it **includes
`[dev-dependencies]`** (test/bench linkage) — that is why `jammi-server` shows `jammi-client`/
`jammi-admin` and several crates show `jammi-test-utils`. The hand-maintained **production dependency
DAG** that follows it lists only normal `[dependencies]`. The two differ *exactly* by the dev/test
edges, by design — not a discrepancy.

<!-- BEGIN GENERATED: dep-dag -->
```
jammi-admin -> jammi-db, jammi-wire
jammi-ai -> jammi-ai, jammi-db, jammi-encoders, jammi-kernels, jammi-lora, jammi-numerics, jammi-test-resources, jammi-test-utils, jammi-wire
jammi-ballista -> jammi-ai, jammi-db, jammi-test-utils, jammi-wire
jammi-bench -> jammi-ai, jammi-db, jammi-encoders, jammi-kernels, jammi-lora, jammi-numerics, jammi-test-resources
jammi-cli -> jammi-admin, jammi-db
jammi-client -> jammi-admin, jammi-db, jammi-wire
jammi-db -> jammi-numerics, jammi-test-resources, jammi-test-utils
jammi-encoders -> jammi-kernels, jammi-lora, jammi-numerics, jammi-test-resources
jammi-kernels -> jammi-test-resources
jammi-lora -> jammi-kernels, jammi-numerics, jammi-test-resources
jammi-numerics
jammi-python -> jammi-ai, jammi-db
jammi-server -> jammi-admin, jammi-ai, jammi-ballista, jammi-client, jammi-db, jammi-numerics, jammi-test-resources, jammi-test-utils, jammi-wire
jammi-test-resources
jammi-test-utils -> jammi-db, jammi-test-resources
jammi-wire -> jammi-db, jammi-lora, jammi-numerics
probed-ops-index -> jammi-kernels
symbol-index
```
<!-- END GENERATED: dep-dag -->

Production dependency edges — each crate's normal `[dependencies]`, dev/test excluded (cf. the
build-link graph above):

```
jammi-numerics   (pure math; no internal deps)
      ▲
jammi-db ────────────────► jammi-numerics
   ▲   ▲
jammi-kernels   (candle-core/candle-nn only; NO internal jammi-* deps — leaf; CUDA feature-gated)
      ▲
jammi-lora ──► jammi-numerics, jammi-kernels(opt, features=["candle"])   [candle OPTIONAL, default-features=false at root]
   ▲
jammi-wire ──► jammi-db, jammi-numerics, jammi-lora(no candle)   [CANDLE-FREE substrate]
   ▲   ▲
jammi-admin ──► jammi-wire, jammi-db                              [control-plane client, candle-free]
   ▲   ▲
jammi-client ──► jammi-wire, jammi-admin, jammi-db               [data-plane client, candle-free]
   │
   │  (jammi-cli ──► jammi-admin, jammi-db  — control-plane only, NO jammi-ai)
   │
jammi-ai ──► jammi-db, jammi-numerics, jammi-lora, jammi-wire,   [EMBEDDED ENGINE]
             jammi-encoders(opt, `local`), jammi-kernels(opt, `cuda`),
             candle(opt, `local`)
   ▲
jammi-ballista ──► jammi-ai, jammi-db, jammi-wire        [BALLISTA COMPUTE PLANE — codec, roles, client]
   ▲
jammi-server ──► jammi-wire, jammi-ai, jammi-ballista, jammi-db, jammi-numerics  [serves the wire over the engine]
   │
jammi-python ──► jammi-ai, jammi-db, jammi-lora                  [LOCAL-ONLY PyO3 cdylib]

jammi-encoders ──► jammi-numerics, jammi-kernels, jammi-lora(features=["candle"]), candle
```

The publish topological order (the canonical DAG statement,
`.github/workflows/crates.yml`, the publish-order list) is:
`jammi-numerics → jammi-db → jammi-kernels → jammi-lora → jammi-encoders →
jammi-wire → jammi-admin → jammi-client → jammi-ai → jammi-ballista → jammi-server → jammi-cli`.
`jammi-kernels` sits before `jammi-lora`, not after: `jammi-lora`'s default
feature set (`default = ["candle"]`, `crates/jammi-lora/Cargo.toml`) enables
the optional `jammi-kernels` dependency, so `cargo publish -p jammi-lora`
(no explicit feature flags in the publish step) needs `jammi-kernels` already
resolvable on crates.io.

Workspace membership (`Cargo.toml`, `[workspace] members`): 15 members;
`default-members` excludes `jammi-python` (PyO3 cdylib, built by maturin) and
`jammi-test-utils`. `jammi-bench` *is* a default member.

### 1.2 Why the load-bearing edges exist (the seams)

- **`jammi-wire` depends on `jammi-db` + `jammi-lora` (candle-free).** Both sides
  of the wire live in one crate so the `From`/`TryFrom` proto↔domain conversions
  satisfy the **orphan rule** without newtype wrappers: prost types are local
  (generated by `build.rs`), domain types are local (`jammi-db`). `jammi-lora` is
  pulled only for its candle-free config vocabulary (`BackboneDtype`,
  `LoraInitMode` on `FineTuneConfig`), pinned `default-features = false`
  (`crates/jammi-wire/Cargo.toml`, the `jammi-lora` dependency entry). See
  `crates/jammi-wire/src/lib.rs` (the crate-level module doc).
- **`jammi-client` composes `jammi-admin`** over the *same* `SessionTransport`
  (`crates/jammi-client/src/lib.rs`, `DataClient::over`) so a tenant bound by a
  control verb is observed by every data verb on the same session id [the
  single-session invariant, §5].
- **`jammi-cli` does NOT depend on `jammi-ai`** — strict control-plane over
  `jammi-admin`; the candle stack never reaches the `jammi` binary
  (`crates/jammi-cli/src/main.rs`, the crate imports). CI enforces this [§6].
- **`jammi-server` depends on `jammi-ai` (engine) AND `jammi-wire`**: it mounts
  service impls over the shared engine.
- **`jammi-ballista` depends on `jammi-ai`/`jammi-db`/`jammi-wire`, never the
  reverse.** Two seams `jammi-ai`'s `HostAdmission` exposes
  (`PlacedGangSubmitter`/`PlacedGangRunner`, `crates/jammi-ai/src/fine_tune/
  worker.rs`) are INSTALLED by `jammi-ballista`'s roles, never called from
  `jammi-ai`'s own dependency graph — the same shape `MemberDialer` already
  uses [§2.8a]. `jammi-server` depends on `jammi-ballista` unconditionally
  (no cargo feature): roles are `[ballista]` config, decided at runtime
  [§2.8f].
- **`jammi-python` depends on `jammi-ai`, `jammi-db`, `jammi-lora`** — no
  client-substrate crate. Local-only; its remote arm is the bundled pure-Python
  `jammi` (`crates/jammi-python/src/lib.rs`, the module setup), so the
  embed wheel links no gRPC transport.
- **`jammi-numerics` depends on nothing internal** — the leaf. Downstream depends
  on it, never the reverse (`crates/jammi-numerics/src/lib.rs`, the crate root).

### 1.3 The dual-surface map (one vocabulary, two transports)

```
        jammi_ai::Session  (local verb vocabulary, over Arc<InferenceSession>)
                       |
   +-------------------+---------------------------------------------+
   | embedded                  | server-side                 | remote (mirror)
   v                           v                             v
 jammi-python PyDatabase   jammi-server grpc/* handlers   jammi-client DataClient
 (file:// local arm)       (Flight SQL + typed gRPC)      jammi-admin CatalogClient
                                                          jammi-cli, pure-py jammi
```

Server handlers, the embedded Python `Database`, and the embedded SDK **all drive
the identical `Session` surface over the identical engine** — one definition, not
parallel reimplementations.

### 1.4 AuditService — the signed per-query audit surface (and the complete mounted-service set)

The audit chain proto → wire → engine verb → server handler → engine primitive is
fully wired, with a remote mirror on `DataClient`. It is mounted unconditionally
whenever the server carries an engine (Core tier), not behind a tier flag.

**The contracts.**
- **Proto** — `crates/jammi-wire/proto/jammi/v1/audit.proto`, `service
  AuditService` with three RPCs: `AuditLog(AuditLogRequest) -> Empty`,
  `AuditFetchByQueryId(...) -> AuditFetchByQueryIdResponse`,
  `AuditFetchRecent(...) -> AuditFetchRecentResponse`. The payload
  `message PerQueryAudit` mirrors the Rust record field-for-field (`query_id`,
  `tenant_id`, `model_id`, `model_version`, `query_lineage` JSON-string,
  `top_k_result_ids`, parallel `retrieval_scores`, `executed_at_micros`,
  `signature`). The timestamp field is named `executed_at_micros` (`int64`) on the
  wire but `executed_at` (`DateTime<Utc>`) on the Rust record
  (`crates/jammi-db/src/audit/record.rs`, the `PerQueryAudit` struct);
  `PerQueryAudit::executed_at_micros` (`crates/jammi-db/src/audit/record.rs`) does
  the micros conversion on encode. **Write-side contract** (proto comment,
  enforced in code): the caller leaves `tenant_id` and `signature` empty/ignored
  on write — the engine stamps the session tenant, computes the HMAC, and
  timestamps. Both are populated on every fetched record.
- **Engine verbs** — `crates/jammi-ai/src/local_session.rs`: `Session::audit_log`,
  `Session::audit_fetch_by_query_id`, `Session::audit_fetch_recent`. Each
  delegates to `self.engine.audit().<method>` — the `AuditHandle`
  (`crates/jammi-ai/src/session.rs`, the `InferenceSession::audit` accessor). These
  are the transport-agnostic surface both the server handler and the remote client
  drive.
- **DB primitive** — `crates/jammi-db/src/audit/`: `PerQueryAudit`
  (`record.rs`), `PerQueryAudit::new` (`record.rs`, enforces the
  `top_k_result_ids.len() == retrieval_scores.len()` invariant via
  `AuditError::LengthMismatch`), `AuditHandle` with `log`/`fetch_by_query_id`/
  `fetch_recent` (`audit/mod.rs`), `AuditError` (`audit/error.rs`). HMAC-SHA256
  over a canonical, key-sorted, whitespace-free serialization (`record.rs`,
  canonical-bytes doc) keyed by `JAMMI_AUDIT_MASTER_KEY` (32 bytes hex;
  `AuditError::MasterKey`).

**Data-flow / call-chain (write path).**
`crates/jammi-server/src/runtime.rs` (`serve_grpc_chain`) mounts
`AuditServiceServer::with_interceptor(AuditServer::new(session), interceptor)`
→ `AuditServer::audit_log` (`crates/jammi-server/src/grpc/audit.rs`) reads tenant
from the request extension via `session_tenant_traced`
(`crates/jammi-server/src/grpc/wire.rs`, the tracing-span-recording wrapper over
`session_tenant`), decodes each proto record through `record_from_proto`
(`crates/jammi-server/src/grpc/audit.rs`) (which calls `PerQueryAudit::new` so the
length invariant is enforced at the boundary and
`tenant_id`/`signature`/`executed_at` are dropped), then runs the verb inside
`scoped(&session, tenant, || session.audit_log(records))`
(`crates/jammi-server/src/grpc/wire.rs`, the per-task-local tenant scope — not
sticky `bind_tenant`) → `Session::audit_log`
(`crates/jammi-ai/src/local_session.rs`) → `AuditHandle::log`
(`crates/jammi-db/src/audit/mod.rs`) → sign + persist + publish to the audit
topic. Errors map through `map_audit_error`
(`crates/jammi-server/src/grpc/audit.rs`), which both picks a gRPC `Code` and
attaches a faithful `jammi_wire::attach_audit_detail` detail so a remote caller
reconstructs the exact `AuditError` variant. Fetch paths are symmetric, with
`parse_query_id` (`crates/jammi-wire/src/audit.rs`) decoding the UUID string.

**Remote mirror.** `crates/jammi-client/src/lib.rs` exposes `DataClient::audit_log`
/ `DataClient::audit_fetch_by_query_id` / `DataClient::audit_fetch_recent` over
`AuditServiceClient`, reconstructing `AuditError` from status detail. So the audit
verbs satisfy the dual-surface contract [§2.1]: identical owned shapes embedded and
remote. Audit is the only engine-service surface with a `DataClient` mirror; the
typed Pipeline/Catalog verbs have no Rust-client mirror — see the verb table.

**Invariants.**
- **Bound tenant required.** The primitive rejects an unscoped call with
  `AuditError::NoTenantBinding`, surfaced as gRPC `FailedPrecondition`
  (`map_audit_error`). "Bind first" [§5] applies here too.
- **Length agreement** between `top_k_result_ids` and `retrieval_scores` is
  enforced once, in `PerQueryAudit::new` (`crates/jammi-db/src/audit/record.rs`) —
  both the receive-side decode and the constructor share it.
- **Signature is engine-owned**, never caller-supplied; `SignatureMismatch` maps
  to gRPC `DataLoss`.
- **`executed_at` storage form is epoch microseconds**
  (`crates/jammi-db/src/audit/record.rs`, `PerQueryAudit::executed_at_micros`),
  matching the trigger backing table's `Int64`-micros convention so both backends
  round-trip identically.

**Extension note.** A new audit field is a four-site atomic change [§5]: the proto
`PerQueryAudit`, the Rust `PerQueryAudit` + canonical-bytes serialization (or the
HMAC silently changes meaning), the two proto↔domain conversion arms — the
receive-side decode `record_from_proto`
(`crates/jammi-server/src/grpc/audit.rs`) and the encode
(`crates/jammi-wire/src/audit.rs`); these live in different crates (server vs
wire), and the inverse read-side decode (`crates/jammi-wire/src/audit.rs`) must
move in lockstep too — and the client encode in `crates/jammi-client/src/lib.rs`.
Touching the canonical byte order invalidates every previously signed record.

**The complete mounted gRPC service set.** Source of truth:
`crates/jammi-server/src/runtime.rs` (`serve_grpc_chain`), with the live `mounted`
vector built alongside each `add_service`.

| Service | Mount condition |
| --- | --- |
| Flight SQL | always |
| `CatalogService` | always |
| `TriggerService` | iff `trigger` handles supplied (Event tier) |
| `EmbeddingService` | when `engine.is_some()` (Core) |
| `InferenceService` | when `engine.is_some()` (Core) |
| `PipelineService` | when `engine.is_some()` (Core) |
| `AuditService` | when `engine.is_some()` (Core) |
| `EvalService` | engine + `ServiceTier::Eval` |
| `JobService` | when `engine.is_some()` (Core — durable job submission/status/wait; the embedded worker is spawned beside it iff `[worker] enabled`) |

The `mounted` `Vec` itself is only a `tracing::info!` log line
(`crates/jammi-server/src/runtime.rs`), not the wire advertisement. The handshake
advertises the *tier tokens*, not the service list: `TierSet::as_wire`
(`crates/jammi-server/src/tiers.rs`, [§2.8]) returns the mounted tiers as sorted
wire tokens (`core`/`event`/`eval`) for `ServerInfo.services`. The
**invariant: advertised (tiers) == mounted (services)** is the caller's
responsibility (`serve_grpc_chain` doc,
`crates/jammi-server/src/runtime.rs`), since the tier set is resolved separately
from the per-service `add_service` calls. `AuditService`, `EmbeddingService`,
`InferenceService`, and `PipelineService` are *Core* (no tier flag) — present on
any engine-bearing server.

**The complete mounted RPC (verb) set.** Source of truth: the `service` blocks
under `crates/jammi-wire/proto/jammi/v1/*.proto`, cross-checked against the server
handler `impl` in `crates/jammi-server/src/grpc/*.rs`. The cookbook's API guard
(`cookbook/book/scripts/check_api_reference.py`) pins the caller-facing embedded
surface (48 surfaces: 46 verbs in `REQUIRED` plus the two `MODULE_FUNCTIONS`). The
guard tracks the Python wheel signature, so its list is a superset of the wire
RPCs (it also covers module functions `open_local`/`connect`, the pure-Python
`rrf_fuse`/`tenant_scope`, and conformal verbs); it is the right cross-check for
*consumer drift*, not a 1:1 mirror of mounted RPCs.

| Service | RPC (proto) | Server handler |
| --- | --- | --- |
| `CatalogService` | `SetTenant`/`GetTenant`/`ClearTenant`/`GetServerInfo` | `grpc/catalog.rs` |
| `CatalogService` | `AddSource`/`RemoveSource`/`ListSources`/`DescribeSource` | `grpc/catalog.rs` |
| `CatalogService` | `ListModels`/`DescribeModel`/`DeleteModel` | `grpc/catalog.rs` |
| `CatalogService` | `RegisterChannel`/`AddChannelColumns`/`ListChannels` | `grpc/catalog.rs` |
| `CatalogService` | `VerifyMaterialization` | `grpc/catalog.rs` (`CatalogService::verify_materialization`) |
| `CatalogService` | `Staleness`/`DerivesFrom`/`ListIndexSegments` | `grpc/catalog.rs` |
| `CatalogService` | `CreateMutableTable`/`DropMutableTable`/`ListMutableTables` | `grpc/catalog.rs` |
| `CatalogService` | `RegisterTopic`/`DropTopic`/`ListTopics` | `grpc/catalog.rs` |
| `CatalogService` | `Reconcile` | `grpc/catalog.rs` (`CatalogService::reconcile`; `all = true` gated by `AdminAuthorizer` [§2.8]) |
| `EmbeddingService` | `GenerateEmbeddings`/`EncodeQuery`/`Search` | `grpc/embedding.rs` |
| `InferenceService` | `Infer`/`Predict` | `grpc/inference.rs` |
| `PipelineService` | `BuildNeighborGraph`/`PropagateEmbeddings`/`AssembleContext` | `grpc/pipeline.rs` |
| `PipelineService` | `AsofJoin` | `grpc/pipeline.rs` (`PipelineService::asof_join`) |
| `PipelineService` | `Recompute` | `grpc/pipeline.rs` (`PipelineService::recompute`) |
| `AuditService` | `AuditLog`/`AuditFetchByQueryId`/`AuditFetchRecent` | `grpc/audit.rs` |
| `EvalService` | `EvalEmbeddings`/`EvalPerQuery`/`EvalInference`/`EvalCompare`/`EvalCalibration` | `grpc/eval.rs` |
| `JobService` | `SubmitJob`/`JobStatus`/`WaitJob`/`ListJobs`/`CancelJob`/`ListWorkers`/`PruneJobs` | `grpc/job.rs` |
| `TriggerService` | `Publish`/`Subscribe` (server-stream) | `grpc/trigger.rs` |

The point-in-time / materialization-contract surface (`VerifyMaterialization` on
CatalogService; `AsofJoin` and `Recompute` on PipelineService) is measured by
cookbook chapters `cookbook/book/chapters/19-point-in-time/point-in-time.qmd` and
`cookbook/book/chapters/20-recompute/` (the authoritative consumer spec:
`staleness` · `derives_from` · `verify_materialization` · `recompute(table,
cascade)`, all observed by table-name identity). `staleness`/`derives_from` live on
CatalogService.

> **`probe_cache` is NOT a mounted gRPC verb.** It is an internal `jammi-db`
> cache-freshness method — `Store::probe_cache`
> (`crates/jammi-db/src/store/freshness.rs`), with its record-returning sibling
> `Store::probe_cache_record` (`crates/jammi-db/src/store/freshness.rs`) — the
> variant the pipeline producers actually call
> (`crates/jammi-ai/src/pipeline/embedding.rs`, `EmbeddingPipeline::run`;
> `crates/jammi-ai/src/pipeline/neighbor_graph.rs`, `NeighborGraphPipeline::run`)
> under `cache="use"`. There is no `ProbeCache` RPC, proto message, or server
> handler. It surfaces to callers only as the `cache` kwarg on
> `generate_embeddings`/`build_neighbor_graph`/`propagate_embeddings`, never as a
> verb.

**Live/dormant status (the typed engine surfaces).**
- `AsofJoin` / `Recompute` / `VerifyMaterialization` — **LIVE**: each has a proto
  RPC, a server handler that delegates into the engine
  (`crates/jammi-server/src/grpc/pipeline.rs`, `PipelineService::asof_join` /
  `PipelineService::recompute` → `InferenceSession::asof_join`
  (`crates/jammi-ai/src/session.rs`) and `Session::recompute`
  (`crates/jammi-ai/src/local_session.rs`);
  `crates/jammi-server/src/grpc/catalog.rs`,
  `CatalogService::verify_materialization` → `Session::verify_materialization`
  (`crates/jammi-ai/src/local_session.rs`)), and a cookbook chapter exercising
  them.
- `rrf_fuse` / lexical retrieval — **DORMANT on the wire.** No
  `RrfFuse`/`LexicalSearch` RPC, proto message, or server handler anywhere under
  `crates/jammi-wire/proto/` or `crates/jammi-server/src/grpc/`. `rrf_fuse` exists
  only as a pure-Python convenience (`crates/jammi-python/src/database.rs`,
  `Database::rrf_fuse` → `jammi_ai::query::rrf_fuse`) and is in the API guard as a
  caller surface — it never crosses a transport. (Detailed retrieval-reality
  analysis: §2.4a.)
- conformal / uncertainty — **DORMANT / caller-driven.** Not present in
  `crates/jammi-server/src/grpc/inference.rs` (`Infer`/`Predict` only); the
  `conformalize*` verbs are guarded as embedded surfaces but have no mounted RPC.
  (Where conformal wraps the served predictor and why it stays dormant: §2.4c /
  §3.9.)

---

### 1.5 The engine↔cookbook loop — the in-monorepo contract suite & staleness oracle

The cookbook is a Quarto book inside this monorepo at `cookbook/book/`, wired into
CI by `.github/workflows/cookbook-book.yml`. It is the discipline loop that makes
the cookbook the engine's executable acceptance suite: a feature is not done
without **chapter + API-guard bump + golden-metric hold**.

**The one-way edge (a guarded invariant, not a convention).** `cookbook/book/`
consumes the engine; the engine never references `cookbook/book/`. This is
enforced by a dedicated CI gate, `cookbook-one-way` /
`ci/scripts/check_cookbook_one_way.sh` (`.github/workflows/ci.yml`, the
`cookbook-one-way` job): engine crates may still depend on the engine-owned
`cookbook/fixtures/`, but no crate may reference `cookbook/book/`. The book's own
`pyproject.toml` states it "never vendors or edits engine source"
(`cookbook/book/pyproject.toml`).

**The five coupling artifacts.**

1. **The API-surface guard — `cookbook/book/scripts/check_api_reference.py`.** This
   is the staleness oracle. It introspects the *live installed* `jammi` wheel
   and asserts every verb the chapters call still exists with the kwargs the
   recipes pass. The mechanism (worth reading; it mirrors the transport-parity
   collapse this guide documents in §2.1/§4.1): it opens an embedded engine
   (`jammi.connect("file://…")`, in `main`) and resolves each verb as a *bound
   method on the live instance* (the `_signature` helper) rather than introspecting
   the wrapper class — because the thin `Database` wrapper holds the native
   `_NativeDatabase` by composition and forwards un-migrated verbs through
   `__getattr__` (documented in `_signature`, corroborated by the engine at
   `crates/jammi-python/src/database.rs`). The contract is the `REQUIRED` dict (47
   verbs) plus `MODULE_FUNCTIONS = ["connect"]`; the gate prints
   "`API reference matches installed jammi (N surfaces checked)`" where
   `N = len(REQUIRED) + len(MODULE_FUNCTIONS) = 48`. This guide should be validated
   against this list — it is the authoritative enumeration of the consumer-facing
   Python verb surface (e.g. `asof_join`, `verify_materialization`, `staleness`,
   `derives_from`, `recompute`, `conformalize{,_interval,_cqr}`, the
   eval/channel/topic/mutable-table verbs). If a pin moves and a signature drifts,
   CI fails *here, loudly*, rather than a chapter calling a stale kwarg at execute
   time.

2. **The golden-metric hold — `cookbook/book/tests/test_closed_loop.py`.** Pins the
   construct→propagate→learn recall chain to frozen goldens (the values live in
   `cookbook/book/artifacts/<dataset>/golden_metrics.json` under
   `tier01/02/03.recall_at_10`, tol 0.03). It also re-derives the conformal
   verdicts *live* from committed per-row outputs (`db.conformalize(... score="aps")`
   and `db.conformalize_interval`) and asserts marginal coverage and interval
   coverage — the "honest under-coverage" lesson (`cov < 1 - alpha`). The
   structural contract: propagation denoises (`prop > base`) and the declared-edge
   fine-tune beats base (`ft > base`, the circularity contract). Goldens are read
   through `jammi_cookbook.contracts.golden(...)`.

3. **The chapters — `cookbook/book/chapters/`.** A 20-chapter Quarto book, one
   chapter per engine capability, plus the `api-reference.qmd`, `datasets.qmd`, and
   `recipes-quickstart.qmd` support pages. Each chapter is the **authoritative
   consumer spec** for the feature it exercises — when extending a feature that has
   a chapter, read it under `cookbook/book/chapters/` first. Chapter↔engine
   examples: `01-construct` (neighbor graph),
   `03-learn`/`08-finetune-methods`/`15-finetune-regression` (fine-tune),
   `06-closed-loop` (the recall+conformal spine above), `08-conformal`/`09-calibration`
   (conformal verbs), `10-retrieval`/`14-scale` (dense ANN + recall),
   `19-point-in-time`/`20-recompute` (`asof_join` + materialization/recompute tier).

4. **The grounded reference — `cookbook/book/chapters/api-reference.qmd`** renders
   `jammi_cookbook/_api_reference.md` (the single source of truth) and is the page
   `check_api_reference.py` guards.

5. **The session-lifecycle rail.** Property: every engine session opened under
   `cookbook/**` is `close()`d, and an embedded one is closed before the directory
   it lives in is removed — an in-process catalog held open past the `rmtree`
   races its own background writes and fails `OSError: [Errno 39] Directory not
   empty` (Linux) / `Errno 66` (macOS). One mechanism judges it everywhere: the
   client's own session registry (`clients/python/jammi/_sessions.py`), read
   through `jammi.SessionWindow` — "what did this window see open and not close,
   and what did it see close after its directory was gone" — independent of what
   name the code bound `connect`'s result to, and of whether the object was ever
   collected.
   - **The pytest lane** — `cookbook/book/tests/conftest.py` holds one window per
     test (`_no_leaked_sessions`, which fails the test *by label*) and one over
     the whole run (import-time and module-scoped sessions).
     `test_session_lifecycle_guard.py` is its non-vacuity control, run via
     `pytester` against the real committed `conftest.py`.
   - **The non-pytest lanes** (the recipes and quickstart `tests/cookbook_smoke.py`
     runs, the book's check scripts, and every `quarto render`) run under
     `python -m jammi.session_journal -- <command>`. With
     `JAMMI_SESSION_JOURNAL` set, every process that imports `jammi` appends each
     open and close to its own file as it happens, so the runner judges a leak in
     a grandchild or in a Jupyter kernel the renderer killed — processes whose
     exit status the lane never reads. `clients/python/tests/test_session_journal.py`
     proves each process shape. A cookbook script that no lane executes is not
     covered: add it to `RECIPES` in `tests/cookbook_smoke.py`.

**How the loop closes in CI (the atomicity guarantee).** The book is tested
against the engine commit it ships beside. The PR gate
(`.github/workflows/cookbook-book.yml`, the PR job) builds the HEAD embed wheel
with `maturin`, force-reinstalls it (`--force-reinstall --no-deps`) over the
unpinned `jammi-ai` dependency, then runs `check_api_reference.py`, the shared-lib
pytest suite including `test_closed_loop.py`, the no-deferral grep, and the
citation check. The nightly `render` job additionally runs `quarto render` over the
committed cache and a release-recipe leg that re-installs the last published PyPI
wheel and re-runs the gate, preserving the published-artifact signal HEAD-testing
alone would lose. So a feature and its proof land atomically in one PR.

**Maintainer implication.** A new or changed verb is not done until: (a) it has
(or updates) a chapter under `cookbook/book/chapters/`; (b) its entry is
added/updated in `REQUIRED` in `check_api_reference.py`; and (c) the relevant
golden in `cookbook/book/artifacts/.../golden_metrics.json` either holds or is
re-pinned with a justification. Cross-reference §4.1 (the wire-verb playbook) — a
new typed verb's Python leg is exactly what this guard checks; adding the verb
without touching `REQUIRED` leaves the surface unproven, and adding a recipe that
calls it without bumping the wheel fails the gate.

**Note on the pin.** The chapters are not pinned to an exact `jammi-ai==X.Y.Z`
PyPI release: `jammi-ai` is consumed **at HEAD**, declared unpinned in
`cookbook/book/pyproject.toml` (the `jammi-ai` dependency) and force-installed from
the maturin HEAD build in CI. The only exact pin is `usearch==2.25.1`
(`cookbook/book/pyproject.toml`), because the serialized ANN graph format is
backend-version-dependent.

**Retrieval reality check.** The cookbook coupling does not resurrect the dormant
retrieval surfaces:
- `rrf_fuse` is **caller-driven, not live in any served path**: the engine impl
  (`crates/jammi-ai/src/query/rrf.rs`, `rrf_fuse`) is re-exported
  (`crates/jammi-ai/src/query/mod.rs`) and exposed to Python
  (`crates/jammi-python/src/database.rs`, `Database::rrf_fuse`) — and the API guard
  lists it (`cookbook/book/scripts/check_api_reference.py`, the `"rrf_fuse"` entry)
  — but no engine-internal search path calls it; the only non-test callers are the
  Python binding and a bench comment (`crates/jammi-bench/src/main.rs`). A consumer
  must fuse explicitly; `search` returns dense ANN only.
- The `LexicalIndex` family is **DORMANT**:
  `crates/jammi-ai/src/index/lexical.rs` is re-exported
  (`crates/jammi-ai/src/index/mod.rs`) but has no caller in the served `search`
  path.
- Conformal is **DORMANT in serving / caller-driven as a primitive**:
  `conformalize{,_interval,_cqr}` exist on the Python surface
  (`crates/jammi-python/src/database.rs`, the `conformalize*` methods) and the
  guard (`cookbook/book/scripts/check_api_reference.py`, the `conformalize*`
  entries), and the closed-loop test drives them on committed outputs — but there
  is no `conformal` reference in `crates/jammi-ai/src/local_session.rs` or
  `crates/jammi-server/src/grpc/`, i.e. the served `InferenceService.Predict` path
  does not wrap predictions in conformal sets (consistent with §3.9 and §2.4c's
  DORMANT finding). The cookbook proves these as post-hoc, caller-invoked verbs,
  not engine-served behavior.

---

## 2. Core abstractions & contracts

Every trait/enum/base surface a maintainer extends, with anchors and invariants.

### 2.1 The front door & session surfaces

- **`Jammi::open(target) -> Result<Session>`** — `crates/jammi-ai/src/jammi.rs`
  (`Jammi::open`). Pure constructor: `Target::Local(config)` →
  `InferenceSession::open(config)` → `Session::with_configured_worker(engine)`
  (`with_configured_worker` (`crates/jammi-ai/src/local_session.rs`)) — the
  worker is spawned only when the loaded config's `[worker] enabled` is
  `true` (default `true`); **not** the unconditional `with_embedded_worker`
  form. This is the SAME key the server's chain assembly and the Python embedded
  arm read before deciding whether THEIR process claims —
  `worker.enabled` (`crates/jammi-server/src/runtime.rs`) and
  `worker.enabled` (`crates/jammi-python/src/database.rs`) — so a wire
  deployment and an in-process one answer "does THIS process claim?"
  identically rather than by three private conventions. `Target`
  is **Local-only** (`crates/jammi-ai/src/jammi.rs`, the `Target` enum); remote is
  reached through `jammi-client`'s `DataClient`, *not* this front door [§7].
- **`Session`** — `crates/jammi-ai/src/local_session.rs` (the `Session` struct). A
  thin `Arc<InferenceSession>` wrapper (re-exported as `crate::Session`,
  `crates/jammi-ai/src/lib.rs`). Three constructors with a load-bearing distinction:
  - `Session::with_configured_worker(engine) -> Result<Self>`
    (`with_configured_worker` (`crates/jammi-ai/src/local_session.rs`)):
    the **front-door** form (`Jammi::open` threads to this one, not to
    `with_embedded_worker`). Reads `WorkerConfig::enabled` (default
    `true`) off `engine`'s loaded config: `true` spawns the worker
    (`Some(worker)`, RAII; stops on drop) by calling into
    `with_embedded_worker` below; `false` spawns nothing and the session
    carries `None`, same as `Session::new`. Must run inside a tokio runtime
    when a worker is spawned. Returns `JammiError::Config` if `[worker]`
    timing violates worker invariants.
  - `Session::with_embedded_worker(engine) -> Result<Self>`
    (`with_embedded_worker` (`crates/jammi-ai/src/local_session.rs`)):
    the **explicit, spawn-regardless** form — carries `Some(worker)`
    **unconditionally**, whatever `[worker] enabled` says. For a caller
    that owns the claim decision itself out of band (test harnesses that must
    have a claimant); the front door does not call this directly. Same
    runtime/`Config`-error contract as `with_configured_worker`.
  - `Session::new(engine) -> Self` (`crates/jammi-ai/src/local_session.rs`):
    carries `None` — used by **every per-request wrapper** (gRPC handlers, Python
    `Database`'s internal session), so a worker is **not** spawned per call.
  - **Invariant:** whether a front-door session's worker exists at all is one
    configuration key, not a code-path choice; a per-request wrapper never
    owns a worker regardless, so spawning one per request would still
    multiply training claimants [§5].
- **`DataClient` / `CatalogClient`** — the remote mirror of `Session`'s data and
  control planes (`crates/jammi-client/src/lib.rs`, the `DataClient` struct;
  `crates/jammi-admin/src/lib.rs`, the `CatalogClient` struct). **Contract for
  interchangeability:** every method takes the same owned request shapes and returns
  the same owned terminal results; the remote side must reconstruct the *exact*
  `JammiError`/`TriggerError` variant from the server's structured detail, never a
  lossy gRPC-code guess (`crates/jammi-client/src/lib.rs`, the crate-level
  error-fidelity doc). (Known dtype-fidelity exception in remote search, [§5].)

### 2.2 The wire substrate (`jammi-wire`)

- **`SessionTransport`** — `crates/jammi-wire/src/transport.rs` (the
  `SessionTransport` struct). One gRPC `Channel` + one minted **v4 UUID session
  id**. `SessionTransport::service(make)` builds every per-service stub by *cloning*
  channel+header, so **all services share one connection and one session id**. The
  id rides every request in the `jammi-session-id` header (`SESSION_HEADER`) via the
  `SessionHeader` interceptor (`crates/jammi-wire/src/transport.rs`,
  `SessionHeader`). `SessionChannel = InterceptedService<Channel, SessionHeader>`
  (`crates/jammi-wire/src/transport.rs`) is the type every generated
  `with_interceptor` stub takes.
- **Request vocabulary** — `crates/jammi-wire/src/request.rs`: `Modality`,
  `QueryInput`, `SearchQuery`, `SearchRequest`, `FineTuneJobId`. **Owned,
  serialisable, hold no engine state.** Both the embedded `Session` and the remote
  `DataClient` build verbs from these; gRPC converters map them on/off the wire.
  Re-exported at `jammi_ai::*` (`crates/jammi-ai/src/local_session.rs`,
  `crates/jammi-ai/src/lib.rs`).
- **IPC framing** — `crates/jammi-wire/src/lib.rs`: `encode_ipc_stream` /
  `decode_ipc_schema` / `decode_ipc_stream`. Arrow batches cross the wire as one
  self-describing IPC stream in `ArrowBatch.data_body` (`data_header` stays empty).
  **Zero-row round-trip is a contract:** `encode_ipc_stream(schema, &[])` must decode
  to `Vec::new()` (`crates/jammi-wire/src/lib.rs`, `decode_ipc_stream`).
- **`ModelTask` wire mapping** — `model_task_from_proto`
  (`crates/jammi-wire/src/lib.rs`) rejects the unspecified/unknown variant with
  `invalid_argument`; `model_task_to_proto` (`crates/jammi-wire/src/lib.rs`) is
  total. A new `ModelTask` must be added to both arms.
- **The `fine_tune` config vocabulary** lives here too
  (`crates/jammi-wire/src/fine_tune.rs`): `FineTuneConfig`, `EmbeddingLoss`,
  `RegressionLoss`, `ClassificationLoss`, `LrSchedule`, `FineTuneMethod`,
  `HardNegativeConfig` — re-exported at `jammi_ai::fine_tune::*` so a client builds a
  training request without candle.
- **`jammi.ballista.v1` is a SEPARATE package, not part of the frozen
  `jammi.v1.*` surface** [§1.3] — it crosses a Ballista scheduler/executor
  boundary inside one cluster's own processes, never a client/server wire a
  foreign consumer decodes, so `jammi-ballista` (the crate that speaks
  Ballista's wire) owns its shape outright, compiled by its own `build.rs`
  [§2.8f].

### 2.3 Storage, catalog & SQL (`jammi-db`)

- **`CatalogBackend`** — `crates/jammi-db/src/catalog/backend.rs` (the
  `CatalogBackend` trait). The backend-agnostic transactional surface
  (`transaction`/`migrate`/`ping`/`backend_kind`). **Closure-passing transactions**:
  commit on `Ok`, rollback on `Err`; the `&mut Transaction` cannot escape the
  closure. **Not dyn-compatible** (generic method) → backends live behind the
  `BackendImpl` enum (`crates/jammi-db/src/catalog/backend.rs`), not `Arc<dyn …>`.
  Parameter type `SqlValue<'v>` (`crates/jammi-db/src/catalog/backend.rs`), read
  trait `FromSqlValue` (`crates/jammi-db/src/catalog/backend.rs`). **Invariant:
  never string-interpolate data into SQL — bind via `SqlValue`** (identifiers are
  the exception, [§2.3 ident]). Error taxonomy `BackendError` + `classify`
  (`crates/jammi-db/src/catalog/backend.rs`). **Tenant write-guard:** `set_tenant`
  then `assert_tenant_matches(row_tenant, table)` before every tenant-aware
  INSERT/UPDATE (`crates/jammi-db/src/catalog/backend.rs`; e.g.
  `crates/jammi-db/src/catalog/result_repo.rs`, the result-repo write path).
- **`SqliteBackend`** — `crates/jammi-db/src/catalog/backend_sqlite.rs`: WAL, 5s busy
  timeout, pool 8. Write tx → `BEGIN IMMEDIATE`, read tx → `BEGIN DEFERRED`:
  **`TxOptions.read_only` is load-bearing.** BEGIN runs on a detached spawn so a
  cancelled caller can't leak a connection mid-transaction.
- **Migration runner** — `crates/jammi-db/src/catalog/migrations.rs`. `MIGRATIONS:
  &[(name, SQL)]`, **append-only**: never rename/reorder; run all-in-one-transaction.
  Trust the array order, not the `schema.rs` constant order. On Postgres, `run`
  takes a transaction-scoped advisory lock (`SELECT pg_advisory_xact_lock($1)`,
  keyed by `JAMMI_MIGRATION_LOCK_KEY`) as its FIRST statement, before it reads
  the `applied_migrations` ledger or runs any DDL — without it, two
  fresh replicas booting together both see an empty ledger and one loses with
  SQLSTATE `42P07`/`23505`; the lock is released on commit or
  rollback (PgBouncer transaction-pooling safe). SQLite needs no equivalent —
  its `BEGIN IMMEDIATE` write transaction already serialises the one process
  that may hold the file. Migration `027_result_table_lease`
  adds `result_tables.writer_id` / `lease_expires_at` + `idx_result_tables_lease`
  for the lease module below.
- **Migration `038_compute_cluster_state`** (`schema.rs`; ordered after BOTH
  `035_instances_peer_addr_result_root` and
  `037_jobs_assembly_failures_next_after`, asserted on both backends by
  `tests/it/migrations.rs::migration_038_is_ordered_after_035_and_037_and_creates_compute_tables`)
  — the catalog-backed cluster state `jammi-ballista`'s scheduler role reads/
  writes [§2.8f]: distributor-neutral, no `ballista` in any
  identifier. `compute_executors` (`executor_id` PK, `instance_id`,
  `host`/`port`/`grpc_port`, `task_slots`/`available_slots`, `status`,
  `heartbeat_at`, `metadata`, and **`devices`** — JSON `[{kind, ordinal}]`,
  the executor's OWN registration fact and the placement join's ONLY
  authority: `Catalog::list_compute_executor_devices` reads THIS column
  directly, never `workers.devices` and never a join on `instance_id`, since
  an executor process and a `[worker]` process are different roles that may
  see different device sets); `compute_jobs` (`job_id` PK, `owner`,
  `status`, `queued_at`, `updated_at` — ownership/status only, since the
  execution GRAPH itself has no serialisation in Ballista 54.1); `ALTER
  TABLE workers ADD COLUMN devices` — a `ListWorkers` MIRROR only, surfaced
  as `WorkerSummary.devices` (field 8, `repeated DeviceFact {kind,
  ordinal}`, additive to the frozen RPC surface), never the placement
  join's authority.
  `compute_repo.rs` (generic CRUD, no distributor vocabulary):
  `upsert_compute_executor`, `list_compute_executors`,
  `record_compute_heartbeat`, `remove_compute_executor`,
  `adjust_compute_slots`/`bind_compute_slots` (the placement policy's slot
  CAS), `put_compute_job`/`get_compute_job`/`list_compute_jobs`/
  `delete_compute_job`, `list_compute_executor_devices`.
- **Migration `039_canonical_stamps`** (`schema.rs`; `catalog::lease`'s
  `CANONICAL_STAMP` — `YYYY-MM-DDTHH:MM:SS.ffffffZ`, UTC, exactly six
  fraction digits) — the ONE catalog stamp shape, enforced at the schema
  edge on both backends for `jobs.{lease_expires_at, next_assembly_after,
  updated_at, created_at}`, `instances.{last_seen_at, started_at}`,
  `result_tables.{lease_expires_at, created_at}`,
  `result_table_versions.lease_expires_at`, `compute_executors.heartbeat_at`,
  `models.{created_at, updated_at}`, `applied_migrations.applied_at`.
  SQLite: a `BEFORE INSERT`/`BEFORE UPDATE OF <col>` trigger per column
  (shape only — SQLite has no calendar parser); Postgres: `CHECK` per column
  (shape AND `::timestamptz` cast validity), named
  `sdchk__<table>__<column>` so `catalog::backend::classify` can recover
  which column refused a write. Existing rows are normalised first (a
  nine-digit ISO fraction truncates to six; Postgres's own pre-039
  `timestamptz`-cast-to-text/`CAST(CURRENT_TIMESTAMP AS TEXT)` rendering
  casts directly).
  `catalog::lease::canonical_stamp_now`/`pg_canonical_stamp` are the ONE
  writer on either backend, MigrationSql::PerBackend (this is the first
  migration whose SQLite/Postgres DDL text cannot be unified — a trigger
  body has no Postgres equivalent syntax).
- **Typed status enums** — `crates/jammi-db/src/catalog/status.rs`:
  `ResultTableStatus`, `JobStatus`, `EvalRunStatus`, `ModelStatus`. Each
  impls `Display`+`FromStr`. **Contract: the DB value set is total over the enum**
  (round-trip test in `status.rs`). `ResultTableKind`
  (Model / NeighborGraph / AsofJoin / TrainingSet,
  `crates/jammi-db/src/catalog/result_repo.rs`) is a *separate* discriminator
  from `ModelTask`; `ResultTableKind::ALL` is the one set its string codec's
  round-trip oracle ranges over. A `TrainingSet` table is the immutable,
  canonically ordered row set a training run reads from
  (`ResultStore::materialize_training_set`); like `AsofJoin` it is data of
  record rather than a search structure — no ANN sidecar, and excluded from
  embedding-table resolution even though its `task` column names a genuine
  model task (the task the rows train, not one this table is the output of).
- **The lease module** — `crates/jammi-db/src/catalog/lease.rs`: the ONE lease
  primitive a claimed `jobs` row (training AND compute kinds share this one
  table, migration 029) and a `building` `result_tables` row both share —
  `LEASE_TS_FORMAT` (a fixed-width UTC format whose lexicographic
  order matches chronological order, so `lease_expires_at < $now` needs no
  dialect-specific interval arithmetic), `canonical_stamp_now()`/`lease_deadline(lease)`,
  `LeaseIntervals { lease,
  heartbeat }` (only buildable through `config::LeaseConfig::intervals()` or
  `Default`, enforcing `heartbeat * 2 < lease` and both non-zero at
  construction), `lease_expired_clause(col, bind) -> "(col IS NULL OR col <
  $bind)"` — the one SQL fragment every expiry-scoped enumeration and CAS
  shares. Config: `[lease] duration_secs = 30, heartbeat_secs = 10`
  (`config::LeaseConfig`, `#[serde(deny_unknown_fields)]`); `[worker]` carries
  `enabled`/`kinds`/`idle_poll_secs` and refuses (no alias) the former
  `lease_duration_secs`/`heartbeat_interval_secs` keys. `[distributed]
  max_world_size = 1` (`config::DistributedConfig`) loads
  INDEPENDENTLY of `[worker]`'s own per-host rank count — the widest `Peer`
  gang any coordinator on this deployment may admit ACROSS FLEET MEMBERS,
  refused at load when `0` (never cross-checked against `[worker]` by
  anything in this crate; the per-job `world_size` submit-time check against
  it is `jammi-ai`'s `RankAdmission`, §2.8d).
- **Lease-owned building result tables** — `ResultStore` mints
  `writer_id = "writer-{uuid}"` per instance; `create_table` stamps it plus a
  lease on the row and returns a `BuildingTable` handle (`table_name()`,
  `parquet_url()`, `writer_id()`, `is_live()`, `set_checkpoint`,
  `append_segment`, `finish`, `abort`, `abandon`) whose background heartbeat
  renews the lease. Every transition on a `building` row is a compare-and-set
  through ONE predicate builder, `catalog::result_repo::ResultTableCas { table,
  tenant_arm: Admin | Strict(tenant), owner: Writer(id) | ExpiredLease(now) }`
  — `renew_lease`, `set_checkpoint`, `fail_building_table` (the only
  `building → failed`), `promote_result_table_with_manifest`,
  `claim_expired_building_table`, `insert_index_segment`,
  `delete_index_segments`. A zero-row CAS match classifies status-first into
  exactly one `JammiError` variant — `RowGone` (row absent), `TenantMismatch`
  (non-admin binding, row's tenant differs), `CasFailed{status}` (status is not
  `building` — e.g. recovery already promoted it), `LeaseLost` (`status ==
  'building' && writer_id != ours`) — and only `LeaseLost` licenses the WRITER
  to delete its own bytes. **Exactly three deletion arms, ever:** a writer's
  own `abort()` after its one-row CAS; recovery's reaper after its one-row
  expiry CAS (having first *claimed* a promotable row, becoming its writer,
  before it rebuilds and promotes — `claim_expired_building_table`); and
  `reconcile` (below). `BuildingTable::finish` = renew the lease by CAS →
  `ResultStore::write_attestation` (digest + manifest + sidecar) → promote CAS
  → `register_table`; on `CasFailed{status: ready}` (recovery promoted the
  writer's own bytes after its lease expired) `finish` still calls
  `register_table` and returns the catalog's record. `recover()`
  (`crates/jammi-db/src/store/mod.rs`, run at session construction under
  `TenantBinding::admin_scope` — the one named implicit-admin pass) enumerates
  ONLY rows whose lease is absent or expired
  (`Catalog::list_expired_building_tables`) — a live-lease row is left alone,
  from any tenant, from any session — and reconciles each to `ready` or
  `failed` per the materialization-contract rules [§2.4d]; it deletes
  expired-lease bytes across EVERY tenant even from a tenant-bound session.
- **The layout module** — `crates/jammi-db/src/store/layout.rs`: the
  tenant-prefixed key scheme every result-table and sidecar object lives
  under. `TenantSegment::of(Option<&TenantId>) -> String` (`_global` or the
  tenant's canonical hyphenated lowercase UUID) and
  `TenantSegment::parse(&str) -> Option<Option<TenantId>>` are EXACT inverses
  on every value `of` can produce — `parse` additionally rejects every
  non-canonical UUID spelling (braced, `urn:uuid:`, unhyphenated "simple")
  `of` would never emit, so a raw listed key in one of those forms is
  `unattributed`, never silently coerced. `result_table_url(root, seg, table)
  -> {root}/{seg}/{table}.parquet`; `segment_url`/`sidecar_url` derive a
  table's siblings from ITS OWN `parquet_path`, never the store's current
  root, so a table created under yesterday's root still resolves correctly if
  the root configuration later moves. Artifacts: `models/{seg}/{job_id}/…`
  (`ArtifactStore`, rooted at `{result_store_root}/models`); job ids are
  canonical `Uuid::new_v4().to_string()`. `ResultStore::with_root(root,
  registry, catalog, ann, local_cache_dir)` — `local_cache_dir` is the PARENT
  of the two local caches the store derives, `{local_cache_dir}/index` and
  `{local_cache_dir}/artifact` (relocated OUT of the result root; the default
  embedded `ResultStore::new(artifact_dir, …)` roots them at
  `{artifact_dir}/cache/{index,artifact}` instead of inside `jammi_db/`) —
  see `docs/guide/src/cloud-storage.md`.
- **Reconcile** — `crates/jammi-db/src/store/reconcile.rs`
  (`ResultStore::reconcile`/`reconcile_all`): the ONE place this engine
  performs an object-store `LIST` (`JammiObjectStore::list`, `pub(crate)`,
  used only here — the never-LIST-on-the-hot-path rule stands everywhere
  else). Lists FIRST, reads rows SECOND, so the row set is guaranteed a
  superset of every object's true referencer at listing time (a table
  materialising concurrently with a pass is always visible by the time rows
  are read, since the row is written before any byte the check looks for).
  **Allowlist** (`attribute(rel) -> Attribution`): first segment parses via
  `TenantSegment::parse` → a result-table key; first segment `models`, second
  parses via `TenantSegment::parse`, third is a canonical v4 UUID → an
  artifact key; anything else → `Unattributed` — reported, **never deleted at
  any grace**. **Row → object** (completeness, a live `exists()` per object):
  `required_row_objects_present(table)` checks Parquet present + (the
  `.materialization.json` sidecar present iff `definition_hash IS NOT NULL`) +
  per-segment `required_sidecar_extensions(kind, precision, row_count)`
  (`crates/jammi-db/src/storage/sidecar_layout.rs`); missing → `ready →
  failed` CAS, reported. **Object → row** (attribution):
  `referenced_result_keys(ready, live_building)` unions the full
  `sidecar_extensions(kind)` superset (referenced-if-present, deliberately
  more generous than the required-side check, since this side must never
  delete a legitimately-present object) of every `ready` row and every
  live-lease `building` row's CURRENT `index_segments` rows (segments are
  referenced by rows, never by filename pattern — a `{base}__segN.*` object
  with no row is an orphan candidate); a `running` job's checkpoints and every
  `models.artifact_path`-named prefix (via `ArtifactStore::expected_objects`)
  are referenced too; a present-but-unreadable `models.artifact_path` manifest
  is reported `damaged`, never orphaned, and never aborts the whole pass;
  else an orphan candidate, aged against `grace` (`apply=true` requires
  `grace >=` the configured lease duration — a typed refusal otherwise)
  before deletion, `pending` if younger. A tenant-scoped pass filters its own
  `ready`-row enumeration to its own tenant BEFORE either the dry-run report
  or the apply CAS, so it never reports (dry-run) or acts on (apply) a GLOBAL
  row it cannot touch — only `reconcile_all` ever does. The expired-building
  pre-pass runs BEFORE the object listing (not after, unlike every other row
  read here) so a key it deletes can never be double-counted by this same
  pass's own orphan accounting. `ReconcileReport { scope, applied,
  rows_failed, rows_failed_count, orphans, orphan_count, pending,
  pending_count, unattributed, unattributed_count, damaged, damaged_count,
  referenced, referenced_count, truncated, bytes_reclaimed }`, every list
  sorted and capped at `REPORT_LIST_CAP` (10,000 entries; `truncated` says
  whether any list hit the cap, `*_count` is always the true total).
  `referenced` names a `models/`-namespaced object this pass's reap-site
  consult of `ResultStore::prefix_is_referenced` found still referenced by
  some live `models` row in some tenant scope — reported instead of
  reclaimed, at any grace or `apply`, and never carrying row ids, model
  names, or tenant ids, only the object key. This is the ONE gate every
  `models/` byte-delete this pass performs runs through right before
  deleting: it asks whether some live `models` row, in any tenant, names
  the object's exact key or its immediate containing directory as
  `artifact_path` — one indexed COUNT lookup per candidate object, never a
  walk of ancestors further up. `reconcile` runs under the
  store's own binding (tenant-bound → its `{seg}/`; unbound → `_global`
  only); `reconcile_all` wraps the WHOLE pass in
  `TenantBinding::admin_scope` and covers every tenant. Wire: `CatalogService.
  Reconcile` [§1.4 RPC table]; CLI: `jammi reconcile [--apply] [--grace-secs
  N] [--all]` (`crates/jammi-cli/src/commands/reconcile.rs`); Python:
  `Database.reconcile(apply, grace_secs, all)`
  (`crates/jammi-python/src/database.rs`, embedded `all=True` = `reconcile_all`
  — there is no wire-level `AdminAuthorizer` gate to consult in-process).
- **`ResultStore`** — `crates/jammi-db/src/store/mod.rs` (the `ResultStore`
  struct): result-table storage coordinator (`root: StorageUrl`, `StorageRegistry`,
  `Arc<Catalog>`, `AnnIndexConfig`). Key methods: `ResultStore::create_table`,
  `finalize`, `recover`, `materialize_embedding_table`, `resolve_search_mode`,
  `reconcile`/`reconcile_all` (see the reconcile bullet above).
- **`SidecarKind` / `sidecar_extensions`** —
  `crates/jammi-db/src/storage/sidecar_layout.rs`: the single registry the writer,
  reader, and cleanup all consult. Ann →
  `["usearch","rowmap","manifest.json","rawf32","threshold"]` (`rawf32` present
  only for a quantized-precision graph; `threshold` present only for a `Binary`
  graph — its per-dimension threshold τ companion); Lexical → `["tantivy"]`;
  None → `[]`.
- **SQL identifier quoting** — `crates/jammi-db/src/sql/ident.rs`: `quote_ident`,
  `quote_relation`, `source_relation`. **Invariant: every identifier interpolated
  into a generated SQL string goes through here** (an unquoted hyphen parses as
  minus).
- **Source types** — `crates/jammi-db/src/source/mod.rs`: `SourceType { File,
  Postgres, Mysql }`; `FileFormat { Parquet, Csv, Json, JsonLines, Avro }` (Avro
  declared but unsupported, `crates/jammi-db/src/source/file_format.rs`;
  `JsonLines` parses `"jsonl"`/`"ndjson"`, otherwise shares `Json`'s
  line-delimited reader). With no explicit `file_extension` override,
  `create_listing_table` tries `.jsonl` first and falls back to `.ndjson`
  only when `.jsonl` has zero matches; for a source added through
  [`JammiSession::add_source`], the winning extension is RESOLVED ONCE AT
  REGISTRATION and PINNED into the persisted `SourceConnection` (the same
  persist-so-`reload_sources`-replays-it pattern `tenant_column` uses) so a
  later directory change can never silently flip which files a reload
  serves — `reload_sources` itself never backfills the pin, so this
  guarantee covers only a source `add_source` registered under this fix,
  not a row written some other way. `SourceConnection`
  (`crates/jammi-db/src/source/mod.rs`) JSON-serializes into `sources.options`, so
  new fields round-trip automatically.
- **`MutableBackend`** — `crates/jammi-db/src/store/mutable/mod.rs` (the
  `MutableBackend` trait): a **pure DDL/DML renderer** trait (no I/O); execution
  flows through `catalog_backend()`. Impls
  `crates/jammi-db/src/store/mutable/sqlite.rs`,
  `crates/jammi-db/src/store/mutable/postgres.rs`. Always emits the implicit
  `tenant_id TEXT` column.

### 2.4 Vector index & search (`jammi-db` + `jammi-ai`)

- **`VectorIndex`** — `crates/jammi-db/src/index/mod.rs` (the `VectorIndex` trait).
  `add`/`build`/`search`/`save`/`len`/`is_empty`. Invariants:
  - **Keyed by `_row_id` (string), never an internal integer.**
  - **`search` returns `(row_id, cosine_distance)` ascending** — *distance*, not
    similarity; the `1.0 - dist` flip happens in `AnnSearchExec`
    (`crates/jammi-ai/src/operator/ann_search_exec.rs`).
  - `build()` after all `add()`s (a no-op marker for USearch,
    `crates/jammi-db/src/index/sidecar.rs`, `SidecarIndex::build`).
  - `Send + Sync` (shared into the async DataFusion plan).
  - **Only `SidecarIndex` impls it** (`crates/jammi-db/src/index/sidecar.rs`,
    `impl VectorIndex for SidecarIndex`); the exact path is a free async fn, not a
    trait impl [§5].
- **`SidecarIndex`** — `crates/jammi-db/src/index/sidecar.rs` (the `SidecarIndex`
  struct): USearch HNSW + Jammi-owned rowmap (`ROWMAP_VERSION=1`) + JSON manifest
  (`ANN_MANIFEST_VERSION=3`). Metric hardcoded `Cos`; quantization is
  `StoragePrecision`-driven (`F32`/`F16`/`Int8`/`Binary`,
  `crates/jammi-db/src/config/mod.rs`), passed as an explicit `precision` argument to
  `SidecarIndex::new`/`load` — never read off `self.ann` internally, so a
  rebuild/load always uses the caller's resolved precision (the catalog row's
  persisted value), not today's deployment default. `SidecarIndex::index_options`
  is the **sole place USearch field names appear**. A quantized (`F16`/`Int8`)
  build also accumulates exact `f32` vectors in `SidecarIndex::add` and flushes
  them to the `.rawf32` rescore companion on `save` (`RawVectorCompanion`,
  mmap'd read-only on `load`); `get_exact` reads it (falling back to `get` at
  `F32`, whose own USearch vectors are already exact). `Binary` is USearch's
  `B1` scalar kind: each dimension is packed to a single sign bit,
  `sign(v − τ)` against a per-dimension threshold τ fit from the corpus (a
  bounded, deterministic sample) by `ThresholdKind::Mean` or `::Median`
  (default `Median` — an exactly-balanced 50/50 bit split measured a higher
  recall@10 than `Mean` on an anisotropic corpus); τ is persisted as the
  `.threshold` companion (one `f32` per dimension) and the fitting
  `ThresholdKind` as the manifest's `binary_threshold_kind` field, required
  whenever `scalar_kind` is `Binary`. `load` strict-compares the manifest's
  `scalar_kind` against the caller's expected precision — any mismatch is
  `IncompatibleFormat`, mirroring the `backend_version` strict-compare (no
  reject-newer ordering; a config drift since the table was built must never
  silently reopen the wrong-precision graph). The precisions, kept in parity
  with `StoragePrecision` (`crates/jammi-db/src/config/mod.rs`) by
  `ci/scripts/check_doc_parity.py`:

  <!-- BEGIN STORAGE-PRECISION-VARIANTS -->
  - `F32` — full-precision index vectors; exact, single-stage search (the default).
  - `F16` — 16-bit half-precision index vectors; quantized, rescored via the `.rawf32` companion.
  - `Int8` — 8-bit signed-integer index vectors (USearch `I8`, linear per-vector affine quantization); quantized, rescored.
  - `Binary` — 1-bit sign-quantized index vectors (USearch `B1`), searched by Hamming distance; quantized, rescored.
  <!-- END STORAGE-PRECISION-VARIANTS -->
- **`SegmentedIndex`** — `crates/jammi-db/src/index/segment.rs` (the
  `SegmentedIndex` struct): a table's ANN index as a **set of segments**, one
  immutable `SidecarIndex` per disjoint row subset (catalog table
  `index_segments`, migration 025 — `result_tables.index_path` is dropped). It
  does **not** implement `VectorIndex`; it owns the merge across segments and
  exposes two entry points. `search(query, m)` is the raw candidate primitive:
  each segment is searched at `over_fetch(m, n)` (`m` for a lone segment,
  `ceil(m * DEFAULT_SEGMENT_OVERFETCH_FACTOR)` = `2.0×` otherwise), concatenated,
  ordered `(distance, row_id, segment_id)`, deduped by row id keeping the
  nearest, truncated to `m`. `search_final(query, k, oversample)` is the **single
  final-results entry** every consumer routes through — `AnnSearchExec::execute`,
  `ResultStore::search_vectors`, and the neighbor-graph `IndexAssisted` driver:
  for an `F32` set it is `search(query, k)` (already exact-comparable), for a
  quantized/`Binary` set it retrieves `k * oversample` candidates and
  exact-rescores them via `get_exact` (dispatched to the owning segment) into one
  cross-segment comparable order. `N = 1` is the same operator at scale 1
  (byte-identical to a lone `SidecarIndex` on a tie-free corpus). Uniform
  precision is enforced (a drifted-precision segment fails `SidecarIndex::load`'s
  strict `scalar_kind` check and never reaches the constructor). `resolve_search_mode`
  builds it by loading every segment through the content-addressed
  `SegmentIndexCache` (`crates/jammi-db/src/storage/index_cache.rs`, keyed on the
  segment manifest bytes); **any** segment load failure falls the whole table
  back to exact search, never a `SegmentedIndex` over the surviving subset.
- **`AnnIndexConfig`** — `crates/jammi-db/src/config/mod.rs` (the `AnnIndexConfig`
  struct): `connectivity` (HNSW M, build-time), `build_expansion`
  (ef_construction, build-time), `search_expansion` (ef_search, query-time,
  mutable) — **`0` = backend default** for these three. `storage_precision`
  (default `F32`) and `oversample` (default `4`, clamped to `>= 1` via
  `effective_oversample`) are the deployment-wide defaults a newly-created
  embedding table's catalog row is stamped with (migration 023,
  `result_tables.storage_precision`/`oversample`, both nullable — `NULL` reads
  back as the honest default for a pre-migration row).
- **Retrieve→rescore** (`SegmentedIndex::search_final`,
  `crates/jammi-db/src/index/segment.rs`): the single rescore implementation,
  folded into the segment merge (there is no free `retrieve_then_rescore` fn).
  When the resolved index's `storage_precision().needs_rescore()`, it retrieves
  `k * oversample` merged candidates, reads each candidate's exact vector via
  `SegmentedIndex::get_exact` (dispatched to the owning segment's `.rawf32`
  companion), recomputes cosine distance, and re-sorts (`total_cmp` + `row_id`
  tie-break) down to `k` — a corpus-independent order comparable across every
  segment. `oversample` resolves via `AnnIndexConfig::resolve_oversample`
  (per-request `SearchRequest::oversample`, wire field 8, over the table's
  stamped default, over the deployment default). An `F32` table skips the rescore
  — the merge is already exact.
  `ProducingDescriptor::NeighborGraph::index_storage_precision`
  (`crates/jammi-db/src/store/manifest.rs`) folds the source table's precision
  into the materialization identity when the index-assisted driver ran
  (`None` when `exact = true`, which never touches the index).
- **`AnnSearchExec`** (`crates/jammi-ai/src/operator/ann_search_exec.rs`) and
  **`QueryBuilder`** (`crates/jammi-ai/src/query/builder.rs`) are the DataFusion
  overlay: `AnnSearchExec` is the leaf that hits the index; `QueryBuilder` seeds the
  plan, hydrates back to source rows, and composes
  filter/select/join/sort/limit/annotate.

### 2.4a Retrieval reality: live dense ANN, latent lexical/RRF, and the typed evidence contract

Trace the call graph before assuming hybrid retrieval is live. The lexical/BM25 +
RRF "hybrid retriever" exists as *primitives* but is **not wired into `search()`** —
no retrieval path references it.

- **`search()` is dense-ANN-only** — the §3.2 path (`Session::search`,
  `crates/jammi-ai/src/local_session.rs` → `InferenceSession::search`,
  `crates/jammi-ai/src/session.rs`). There is no lexical leg in it; neither
  `LexicalIndex` nor `rrf_fuse` is referenced from any retrieval path (`rrf_fuse` is
  re-exported from `crates/jammi-ai/src/query/mod.rs`, `LexicalIndex` from
  `crates/jammi-ai/src/index/mod.rs`; nothing more).
- **`rrf_fuse` (reciprocal-rank fusion)** — `crates/jammi-ai/src/query/rrf.rs`
  (`rrf_fuse`); `DEFAULT_K_RRF=60`. Rank-based (scale-free), N-ary, deterministic
  (ties break ascending by `_row_id`). Its only non-test caller in the entire
  workspace is the Python binding `Database::rrf_fuse`
  (`crates/jammi-python/src/database.rs`, calling `jammi_ai::query::rrf_fuse`). It
  is a **caller-driven client utility**: you supply the ranked lists (e.g. two
  `search()` calls, or your own BM25 list); the engine never assembles a hybrid for
  you.
- **`LexicalIndex` (BM25 over tantivy)** — `crates/jammi-ai/src/index/lexical.rs`
  (the `LexicalIndex` struct), persisted as the `SidecarKind::Lexical` sibling
  (`crates/jammi-ai/src/index/mod.rs`). Fully built, tested, and exported — but has
  **zero live callers** (Rust, pipeline, or Python): every non-test reference is the
  re-export or `lexical.rs` itself. Even its typed error variant
  `JammiError::Lexical` (`crates/jammi-db/src/error.rs`) is raised **only inside
  `lexical.rs`**. It is a **dormant primitive**: wiring it into a retrieval path is
  a feature, not a config flag.
- **The wired "hybrid" is a different concept.** `ContextSource::Hybrid { ann_k,
  edges, merge: HybridMerge::Union }`
  (`crates/jammi-ai/src/pipeline/context_set.rs`, the `ContextSource` enum and
  `HybridMerge` enum; proto-decoded at
  `crates/jammi-ai/src/wire/pipeline.rs`, gated on `req.hybrid`) fuses **dense ANN ∪
  a declared graph-edge walk** (neighbor-graph) — *not* lexical/RRF. It lives in the
  context-assembly pipeline (`PipelineService.AssembleContext`), not in `search()`.
  Full call-graph in §2.4c.

**The evidence/provenance layer (this part IS hot-path).**
- Every `QueryBuilder::run()` (`crates/jammi-ai/src/query/builder.rs`) ends by
  calling **`merge_channels`** (`crates/jammi-ai/src/evidence/merger.rs`). Output
  schema = source columns + **`retrieved_by`** + **`annotated_by`** (both
  `List<Utf8>` of channel ids, declared **non-null**:
  `crates/jammi-ai/src/evidence/merger.rs`) + each participating channel's declared
  columns, sorted by `(priority, ordinal)`.
- **The engine never writes provenance itself**
  (`crates/jammi-ai/src/evidence/merger.rs`, the module doc): callers pass
  `ChannelContribution`s (`crates/jammi-ai/src/evidence/channel.rs`); the dense path
  contributes the `vector` channel's `similarity`, `.annotate()` adds the
  `inference` channel (assembled in `crates/jammi-ai/src/query/builder.rs`,
  `QueryBuilder::annotate`). Unsupplied channels become **all-null arrays of the
  declared dtype** (`new_null_array`, `crates/jammi-ai/src/evidence/merger.rs`).

**The typed channel-error taxonomy.** There is no `JammiError::EvidenceChannel`
variant — channel errors are split into **two typed variants** with distinct
severities and distinct gRPC mappings:

- **`JammiError::ChannelAssembly(String)`** (`crates/jammi-db/src/error.rs`) —
  engine-internal data-shape contract violations raised by `merge_channels` on
  engine-derived inputs. The five validation return sites
  (`crates/jammi-ai/src/evidence/merger.rs`): contribution/batch length mismatch,
  unregistered participating channel, non-participating contribution,
  declared-column-count disagreement, and dtype disagreement. Because these are
  engine invariants, not caller conditions, they map to gRPC **`Internal`**
  (`crates/jammi-server/src/grpc/wire.rs`, `map_engine_error`).
- **`JammiError::ChannelCatalog(#[from] ChannelCatalogError)`**
  (`crates/jammi-db/src/error.rs`) — the *caller-facing* registry conditions
  (`register_channel` / `add_channel_columns`). The `ChannelCatalogError` enum
  (`crates/jammi-db/src/catalog/channel_repo.rs`) has six variants, mapped in
  `map_engine_error` (`crates/jammi-server/src/grpc/wire.rs`, the channel arm):
  - `AlreadyExists` and `ColumnAlreadyDeclared` (same-type redeclare) → **`AlreadyExists`**
  - `NotRegistered` → **`NotFound`**
  - `ColumnConflict` (different-type redeclare) → **`FailedPrecondition`**
  - `InvalidId` and `InvalidColumnType` → **`InvalidArgument`**

  Consumer-facing spec & cross-transport parity contract: cookbook chapter
  `cookbook/book/chapters/17-channels-taxonomy/channels-taxonomy.qmd`. It measures
  each `(failure mode → gRPC code)` cell against a frozen golden on the live
  `grpc://` transport, asserts the same normalized error *class* on the embedded
  engine, and asserts **no typed failure collapses to `INTERNAL`/`UNKNOWN`**
  (`INTERNAL` is the documented residual for a genuine fault). One nuance: an
  invalid dtype *string* is rejected **client-side** (a `ValueError`, never reaches
  the wire), so the honest `INVALID_ARGUMENT` wire cell is an **empty channel id**
  the server rejects.

- **Conformal & uncertainty** (`crates/jammi-ai/src/evidence/conformal.rs`,
  `crates/jammi-ai/src/evidence/uncertainty.rs`,
  `crates/jammi-ai/src/predict/conformal.rs`, split-conformal) are **not in the
  `search()` path** — and the `ConformalContextPredictor` wrap
  (`crates/jammi-ai/src/pipeline/context_predictor.rs`) is itself **DORMANT**: its
  only callers are the module itself plus
  `crates/jammi-ai/tests/it/context_predictor.rs` (no runtime caller anywhere in
  the workspace). The only runtime-reachable callers of `ConformalModel` are the
  jammi-python embed utilities `conformalize` / `conformalize_interval` /
  `conformalize_cqr` (`crates/jammi-python/src/database.rs`, the `conformalize*`
  methods) — caller-driven. `InferenceService.Predict` returns a bare distribution
  + source + `context_ref` (`crates/jammi-server/src/grpc/inference.rs`,
  `InferenceServer::predict`), **no coverage interval**. Full call-graph in §2.4c
  and §3.9.

> **Invariant (belongs in §5):** any code that builds result batches directly
> instead of going through `QueryBuilder::run` drops `retrieved_by`/`annotated_by`
> and breaks the result-schema contract every consumer expects. Append provenance
> via `merge_channels`, never by hand.

### 2.4b Trigger stream, evidence channels & mutable companion tables (the three substrate primitives)

Three substrate primitives hang off `Session`
(`crates/jammi-ai/src/local_session.rs`) and forward to the `InferenceSession`
(`crates/jammi-ai/src/session.rs`) → `JammiSession`
(`crates/jammi-db/src/session.rs`) chain. All three are wired end-to-end (engine
verb + server gRPC handler + Python binding). The shared house rule ("one binary
serves every topology via pluggable backends", `CLAUDE.md`) shows up concretely
here: the trigger broker is the pluggable one; the catalog/mutable backend rides the
`BackendImpl` enum [§2.3].

#### Evidence channels — the registry behind §2.4a's provenance

- **`ChannelId`** — `crates/jammi-db/src/evidence_channel.rs` (the `ChannelId`
  type). Validated ASCII slug `[a-z][a-z0-9_]{0,63}`; serde round-trips through
  `String`, so an invalid slug is rejected at deserialize. Re-exported
  `jammi_db::ChannelId`.
- **`ChannelSpec { id, priority: i32, columns: Vec<ChannelColumn> }`** —
  `crates/jammi-db/src/catalog/channel_repo.rs` (the `ChannelSpec` struct).
  `ChannelColumn { name, data_type }`; `ChannelColumnType` is the **closed**
  Arrow-type set `{Float32,Float64,Int32,Int64,Utf8,Boolean}`, PascalCase token
  shared with the catalog and the Python API (`ChannelColumnType::as_str` /
  `from_sql_str`).
- **`ChannelRepo`** (`crates/jammi-db/src/catalog/channel_repo.rs`, the
  `ChannelRepo` struct, constructed via `Catalog::channels`,
  `crates/jammi-db/src/catalog/mod.rs`) is the catalog surface over the
  `evidence_channels` + `evidence_channel_columns` tables. Verbs:
  - `ChannelRepo::register(&ChannelSpec)` — atomic insert of parent + ordered
    columns; duplicate → `JammiError::EvidenceChannel("…already exists")`. **The
    catalog enforces uniqueness two ways:** composite `UNIQUE (tenant_id,
    channel_name)` for tenant-scoped, *plus* a partial unique index on
    `channel_name WHERE tenant_id IS NULL` for global (because both backends treat
    NULLs as distinct in a UNIQUE constraint;
    `crates/jammi-db/src/catalog/channel_repo.rs`; migration backstop in
    `crates/jammi-db/src/catalog/schema.rs`).
  - `ChannelRepo::add_columns(&ChannelId, &[ChannelColumn])` — **append-only**;
    ordinal continues from `max_ordinal+1`; redeclaring an existing column (same or
    differing type) is rejected.
  - `ChannelRepo::get` / `ChannelRepo::list` — **own-shadows-global precedence**: a
    tenant resolves its own channel of a name, else falls back to the global
    (`tenant_id IS NULL`) channel; never another tenant's. `list` orders by
    `(priority, channel_name)`. This is what lets every tenant see the seed channels
    while owning a private channel of the same name.
  - `ChannelRepo::merged_schema(&[ChannelId])` — Arrow schema of the participating
    channels' columns in `(priority, ordinal)` order. **DORMANT in `src/`** — only
    test callers
    (`crates/jammi-db/tests/it/channels.rs`,
    `crates/jammi-ai/tests/it/channel_contract.rs`, plus in-module unit tests); no
    non-test caller.
- **Seed channels** (global, migrated): `vector` (priority 1, column `similarity:
  Float32`) and `inference` (priority 2, columns
  `inference_model`/`inference_task`/`inference_confidence`) seeded in the first
  migration (`crates/jammi-db/src/catalog/schema.rs`); `bm25` (priority 3, columns
  `bm25_score: Float32` + `bm25_rank: Int64`) seeded in a later migration
  (`crates/jammi-db/src/catalog/schema.rs`).
  - **Wiring caveat (mirrors §2.4a):** the `bm25` *channel* is seeded, but the BM25
    `LexicalIndex` that would contribute to it is **DORMANT** — zero live callers
    [§2.4a]. The seed channel exists so a caller-driven hybrid (Python `rrf_fuse`)
    can label its own contributions, not because the engine populates `bm25` on
    `search()`.
- **How channels feed provenance [§2.4a].** The hot path is `QueryBuilder::run`
  (`crates/jammi-ai/src/query/builder.rs`): it strips each batch's declared channel
  columns into `ChannelContribution`s
  (`crates/jammi-ai/src/evidence/channel.rs`) via
  `QueryBuilder::extract_channel_contributions`, then calls **`merge_channels`**
  (`crates/jammi-ai/src/evidence/merger.rs`), which resolves specs from the catalog,
  sorts by priority, and appends `retrieved_by` + `annotated_by` (`List<Utf8>`,
  non-null) plus each channel's columns; unsupplied channels become all-null arrays
  of the declared dtype. **LIVE** — `merge_channels`'s only non-test caller is
  `QueryBuilder::run`. The validator rejects shape/dtype/row/duplicate mismatches
  with the typed channel errors [§2.4a].
  - **`register_channel`/`add_channel_columns`/`list_channels`** are thin `Session`
    forwarders (`crates/jammi-ai/src/local_session.rs`) → `catalog().channels()`.
    **CALLER-DRIVEN**: the engine never auto-registers a custom channel; a caller
    declares one, then supplies its columns in the source batch so
    `QueryBuilder::run` picks them up.

#### Trigger stream — pub/sub over Arrow batches with a pluggable broker

- **Module** `crates/jammi-db/src/trigger/mod.rs`. Core types: `TopicDefinition`
  (`crates/jammi-db/src/trigger/topic.rs`), `Offset`
  (`crates/jammi-db/src/trigger/offset.rs`), `Predicate` (SQL filter,
  `crates/jammi-db/src/trigger/predicate.rs`), `DeliveredBatch`
  (`crates/jammi-db/src/trigger/subscription.rs`), `TriggerError`
  (`crates/jammi-db/src/trigger/error.rs`).
- **The broker IS a pluggable backend, but it is transport-only** —
  `TriggerBroker` trait (`crates/jammi-db/src/trigger/broker.rs`,
  `Send + Sync + 'static`, `#[async_trait]`):
  `register_topic`/`drop_topic`/`publish`/`subscribe`/`list_consumers`/
  `driver_kind`. **Contract: a driver MUST NOT persist**
  (`crates/jammi-db/src/trigger/broker.rs`, the trait doc); the engine's
  mutable backing table is the authoritative log, and the broker never sees
  `tenant_id` (tenant scope is enforced upstream by catalog lookup +
  predicate injection). A driver's `subscribe` returns a driver-level
  `LiveStream` (`crates/jammi-db/src/trigger/subscription.rs`, item
  `Result<LiveEvent, TriggerError>`): `LiveEvent::Batch(DeliveredBatch)` when
  the driver carries the published bytes itself, or `LiveEvent::Wake` — carrying
  no payload — for a driver that carries no bytes at all. There is no error
  variant for "history not retained" or "receiver lagged": every driver
  routes both into `Wake`, and the engine's subscribe seam self-heals by
  replaying the backing table. Three impls behind `Arc<dyn TriggerBroker>`:
  `InMemoryBroker` (`crates/jammi-db/src/trigger/in_memory.rs`,
  `BrokerKind::InMemory`, the **default**, yields `Batch`), `PostgresBroker`
  (`crates/jammi-db/src/trigger/postgres.rs`, `BrokerKind::Postgres`, yields
  only `Wake` over `LISTEN`/`NOTIFY` — **no cargo feature**, `sqlx`'s
  `postgres` feature is unconditional in the workspace), and `JetStreamBroker`
  (`crates/jammi-db/src/trigger/jetstream.rs`, `BrokerKind::JetStream`, yields
  `Batch`) — the latter is gated behind the **`jetstream-broker` cargo
  feature** (`crates/jammi-db/Cargo.toml`; re-exported only under that cfg,
  `crates/jammi-db/src/trigger/mod.rs`; `jammi-server` re-exposes it as
  `jetstream-broker`, `crates/jammi-server/Cargo.toml`). Selection is
  **config-driven**, `build_broker_from_config`
  (`crates/jammi-db/src/session.rs`): `BrokerConfig::InMemory` →
  `InMemoryBroker::new()`; `BrokerConfig::Postgres{…}` → connect (`url`
  defaults from `catalog.postgres.url` when the catalog is Postgres, else a
  typed `JammiError::Config` naming both keys); `BrokerConfig::JetStream{…}` →
  connect; choosing JetStream **without** the feature returns a typed
  `JammiError::Config`, not a panic (the
  `#[cfg(not(feature = "jetstream-broker"))]` `build_jetstream_broker`,
  `crates/jammi-db/src/session.rs`).
- **The engine-facing type never changes with the driver.** `Subscriber`
  (`crates/jammi-db/src/trigger/subscriber.rs`) resolves every driver's
  `LiveStream` into the transport-neutral `Subscription`
  (`crates/jammi-db/src/trigger/subscription.rs`, item
  `Result<DeliveredBatch, TriggerError>`) — a driver `Batch` is fanned out
  when contiguous, and a gap or a `Wake` triggers a replay of the backing
  table, so a caller two layers up never sees which driver is underneath.
  The mediator is `TopicTail` (`crates/jammi-db/src/trigger/tail.rs`): one per
  `(topic_id, tenant)` this process has ever served a subscriber for, holding
  the SINGLE driver-level `LiveStream` for that scope and fanning out to
  every subscriber of that pair through one `tokio::sync::broadcast` — `N`
  subscribers of the same topic/tenant cost one driver subscription and one
  replay per wake, not `N`. The tail keeps a cursor (the last engine
  `_offset` delivered or replayed, tenant-blind); a driver `Batch` is fanned
  out directly only when it is exactly `cursor + 1` — any gap or regression
  drops it and triggers a replay instead — which is what makes multi-replica
  publish correct for every driver, not only Postgres (post-commit fan-out
  across replicas is unordered). A tail's own driver subscription is always
  `(Predicate::match_all(), from_offset = None)`; per-subscriber predicate
  and tenant filtering happen in-process, never in the driver.
- **Publish/subscribe contract + error type.** Error type is `TriggerError`
  (`crates/jammi-db/src/trigger/error.rs`) — a `thiserror` enum: `TopicNotFound`,
  `SchemaConflict`, `UnsupportedSchemaType`, `BatchSchemaMismatch`,
  `PublishTenantMismatch`, `PredicateParse`/`Eval`/`Unsupported`,
  `BackingTable(#[from] MutableTableError)`, `Backend(#[from] BackendError)`,
  `Driver`, `Catalog`. Note the **`#[from] MutableTableError`** edge: the trigger
  log *is* a mutable companion table, so the two primitives share an error path.
  There is no `OffsetEvicted` variant: a driver that cannot start at or before a
  requested offset, or a receiver that lagged, yields `LiveEvent::Wake` instead
  of failing the stream.
- **Session verbs** (`crates/jammi-ai/src/local_session.rs`): all return
  `Result<…, TriggerError>`:
  - `Session::register_topic` — registers with **both** the broker driver *and* the
    catalog (`topic_repo`), in that order, because a `publish` resolves the topic
    against the broker; a catalog-only registration would make publish fail with
    `TopicNotFound`. `register_topic` is the engine's own responsibility for
    schema-conflict detection (`TopicRepo::register_topic`); a driver MAY treat
    its own `register_topic`/`drop_topic` as a no-op (Postgres does — there is
    nothing for a wake-up transport to own per topic).
  - `Session::publish(topic, batch) -> Offset` →
    `Publisher::publish_scoped(topic, tenant, batch)`
    (`crates/jammi-db/src/trigger/publisher.rs`). Tenant comes from the session, not
    the caller. Offset assignment is transactional — one locking
    `UPDATE topics SET next_offset = … RETURNING` inside the same transaction
    that inserts the augmented batch — so two engine replicas publishing
    against the same Postgres catalog never collide on the backing table's
    `(_offset, _row_idx)` composite key; an empty batch is rejected rather than
    burning a gap-free offset for zero rows.
  - `Session::subscribe(topic, predicate, from_offset, replay_only)` → a
    transport-neutral `Stream<Item = Result<DeliveredBatch, TriggerError>>`.
    `replay_only=true` selects the **finite-drain primitive**
    `Subscriber::replay_only_scoped` (`crates/jammi-db/src/trigger/subscriber.rs`) —
    yields the retained window and *terminates*; `false` selects
    `Subscriber::subscribe_scoped` — open-ended live tail. Same return shape; the
    bounded `jammi trigger subscribe --no-follow` just sees the stream end.
  - `Session::list_topics` / `Session::drop_topic` — catalog `topic_repo` is the
    system of record; `drop_topic` drops the catalog row first, then best-effort
    broker drop (failure logged, not reverted).
- **Server wiring — TWO gRPC services.** The control verbs
  (`register_topic`/`drop_topic`/`list_topics`) live on **`CatalogService`**
  (`crates/jammi-server/src/grpc/catalog.rs`) and mount with the core tier. The
  **data plane** (`publish`/`subscribe`) is a *separate* `TriggerService`
  (`crates/jammi-server/src/grpc/trigger.rs`, `TriggerService::publish` /
  `TriggerService::subscribe`) mounted **only as the Event tier**
  (`crates/jammi-server/src/runtime.rs`) — driven by the caller supplying trigger
  handles. `TriggerService.publish` calls `publisher.publish_scoped` directly;
  errors map via `map_trigger_error`. **Wiring status: LIVE**, but the data-plane
  service is **optional/tiered** — a serve build without the event tier exposes topic
  DDL but not publish/subscribe.
- **Python binding** (`crates/jammi-python/src/database.rs`):
  `Database::register_topic` (registers broker *and* topic_repo, mirroring the
  session), `Database::publish_topic`, `Database::subscribe_collect`,
  `Database::list_topics`, `Database::drop_topic`. **LIVE.**

#### Mutable companion tables — vs result tables

- **What they are** (`crates/jammi-db/src/store/mutable/mod.rs`, the module doc):
  catalog-registered relations living in the **same backend DB as the catalog**
  (SQLite default, Postgres in shared deployments), supporting transactional
  `INSERT`/`UPDATE`/`DELETE` through DataFusion DML and federating in one query plan.
- **How they differ from result tables [§2.3 `ResultStore`].** Result tables are
  **immutable Parquet** + sidecar ANN/lexical index, written once via
  `ResultStore::create_table`/`finalize` (`crates/jammi-db/src/store/mod.rs`) and
  read through `search()`. Mutable tables are **row-mutable rows in the catalog
  backend**, queried as `mutable.public.<id>` (`crates/jammi-db/src/session.rs`,
  the mutable-schema registration) — no Parquet, no sidecar, no ANN. Result tables
  answer "what did embedding/inference produce"; mutable tables are a read/write
  companion store (and the authoritative trigger-stream event log, hence
  `TriggerError::BackingTable`).
- **`MutableBackend`** — `crates/jammi-db/src/store/mutable/mod.rs` (the
  `MutableBackend` trait): a **pure DDL/DML renderer** trait (no I/O); execution
  flows through `catalog_backend()` → `BackendImpl`. Impls
  `crates/jammi-db/src/store/mutable/sqlite.rs`,
  `crates/jammi-db/src/store/mutable/postgres.rs`. **Invariant: always emits an
  implicit `tenant_id TEXT` column**; the definition builder *rejects* a
  user-declared `tenant_id` column
  (`crates/jammi-db/src/store/mutable/definition.rs`).
- **Verbs.** `Session::create_mutable_table(def) -> MutableTableId`
  (`crates/jammi-ai/src/local_session.rs` → `crates/jammi-ai/src/session.rs` →
  `crates/jammi-db/src/session.rs`, `JammiSession::create_mutable_table`): rejects
  reserved `_jammi_*` names — those are substrate-owned (e.g. audit, trigger log)
  created via `JammiSession::register_mutable_table_unchecked`. After register it
  adds a `TableProvider` to the `mutable.public` schema.
  `Session::drop_mutable_table`, `Session::list_mutable_tables`
  (`crates/jammi-ai/src/local_session.rs`, registry introspection — not a SQL
  query). Persisted tables are reloaded on startup across **all** tenants
  (DataFusion name resolution ignores session scope; per-row tenant filtering happens
  at query time, `crates/jammi-db/src/session.rs`).
- **Server + Python.** gRPC `CatalogService`: `create_mutable_table`,
  `drop_mutable_table`, `list_mutable_tables`
  (`crates/jammi-server/src/grpc/catalog.rs`). Python: `Database::create_mutable_table`,
  `Database::drop_mutable_table`, `Database::list_mutable_tables`
  (`crates/jammi-python/src/database.rs`). **LIVE.**

#### Extension notes

- **Add an evidence channel.** Either runtime — call
  `session.register_channel(&ChannelSpec{ id, priority, columns })` (slug, priority,
  ordered columns of the closed `ChannelColumnType` set), then feed its columns into
  the source batch so `QueryBuilder::run` extracts and merges them. For a *seed*
  (global) channel, add an append-only migration that inserts into
  `evidence_channels` + `evidence_channel_columns` (pattern: the `bm25` seed
  migration, `crates/jammi-db/src/catalog/schema.rs`) and bump the `MIGRATIONS`
  array [§2.3]. To add a new column *type*, extend `ChannelColumnType` and **both**
  `to_arrow`/`from_arrow` and `as_str`/`from_sql_str`
  (`crates/jammi-db/src/catalog/channel_repo.rs`) — the from-arms are total over the
  enum.
- **Add a broker backend.** Implement `TriggerBroker` (transport-only, never
  persist): `register_topic`/`drop_topic` may be no-ops (the engine owns
  schema-conflict detection), `publish` returns the engine-assigned offset
  without inspecting the tenant tag, and `subscribe` returns a `LiveStream`
  (`crates/jammi-db/src/trigger/subscription.rs`) whose item is
  `Result<LiveEvent, TriggerError>` — yield `LiveEvent::Batch(DeliveredBatch)`
  when the driver carries the published bytes itself, or `LiveEvent::Wake`
  (no payload) when it does not; a driver that cannot start at or before the
  requested offset, or whose receiver lagged, yields `Wake` — there is no
  error variant for either case. `PostgresBroker`
  (`crates/jammi-db/src/trigger/postgres.rs`) is the reference for a
  bytes-free driver: it carries no data of its own, because the topic's own
  mutable backing table is already the durable, authoritative log every
  driver replays through — a driver that persisted its own copy would be a
  second, unreconciled log, which the trait forbids. Whatever a driver
  yields, the engine-facing type never changes: `Subscriber`'s `TopicTail`
  (`crates/jammi-db/src/trigger/tail.rs`) holds the ONE driver-level
  `LiveStream` per `(topic, tenant)` and resolves it into the transport-
  neutral `Subscription` (item `DeliveredBatch`) every caller — embedded or
  remote — actually observes, so a new driver never touches
  `Subscriber`/`Publisher`/the wire encoders. Add a `BrokerKind` variant
  (`crates/jammi-db/src/trigger/broker.rs`) and a `BrokerConfig` arm, wire it
  in `build_broker_from_config` (`crates/jammi-db/src/session.rs`); gate a
  new dependency behind a cargo feature like `jetstream-broker` and return
  `JammiError::Config` when selected without it — **only when the dependency
  is genuinely optional**: `PostgresBroker` needs no feature at all, because
  `sqlx`'s `postgres` feature is already unconditional in the workspace (the
  catalog backend depends on it), so gating it would be a feature flag
  guarding nothing. No `search()` or result-table code changes — the broker
  plugs in at the session-construction seam.
- **Add a topic / mutable table at runtime** is data, not code: `register_topic` /
  `create_mutable_table`. No migration needed (the backing table is created by the
  renderer DDL).

### 2.4c PipelineService: neighbor graph, propagation, context assembly, as-of join & recompute (the wired hybrid + the dormant conformal)

This is where the **wired "hybrid"** lives — `ANN ∪ declared-edge walk`, not
lexical/RRF [§2.4a]. It is also where the conformal/uncertainty wrap *is defined* —
but trace before you trust: the wrap is built and tested, never exposed (DORMANT —
no non-test caller). The served predict path returns a bare
distribution. The service also carries `AsofJoin` (point-in-time temporal join) and
`Recompute` (re-invoke a result table's recorded producer over current inputs) —
both LIVE and both backed by their own cookbook chapters.

**The service is a thin adapter.** `PipelineService` mounts unconditionally inside
the always-on core tier whenever an engine is present
(`crates/jammi-server/src/runtime.rs`, in the `if let Some(session) = engine` block,
*not* gated by a `ServiceTier`). Five verbs, each a one-call adapter over
`InferenceSession` inside the request's tenant `scoped`
(`crates/jammi-server/src/grpc/pipeline.rs`); the service reimplements no
graph/retrieval/aggregation/join logic. All five are **LIVE**: gRPC + the Python
binding (`crates/jammi-python/src/database.rs`, the pipeline methods) both reach
them.

- **`BuildNeighborGraph`** → `InferenceSession::build_neighbor_graph`
  (`crates/jammi-ai/src/session.rs`) → `NeighborGraphPipeline::run`
  (`crates/jammi-ai/src/pipeline/neighbor_graph.rs`). Materialises the **kNN graph
  of an embedding table** (each row's `k` nearest *within the same table*) as a
  catalogued edge table — `(src, dst, rank, similarity)`, `similarity = 1.0 -
  cosine_distance`. Returns the table handle (shared `ResultTable` proto); the client
  reads edges via SQL. The edge table carries **no model, no sidecar, no evidence
  channel** — `model_id` is a fixed marker `"neighbor_graph"`
  (`NEIGHBOR_GRAPH_MODEL_ID`,
  `crates/jammi-ai/src/pipeline/neighbor_graph.rs`) and `similarity` is a plain
  `Float32` column. The verb is memoizable — `run` does a top-of-producer cache probe
  under `CachePolicy::Use`, reusing an exact prior materialisation keyed on the
  descriptor + the source table's digest; `CachePolicy::Bypass` (default) always
  rebuilds. The returned `CacheOutcome` reports which path ran.
- **`PropagateEmbeddings`** → `InferenceSession::propagate_embeddings`
  (`crates/jammi-ai/src/pipeline/graph_propagation.rs`, an `impl InferenceSession`
  block in the pipeline module, **not** session.rs). Iterates `X⁽⁰⁾` over a declared
  graph and materialises a new searchable embedding table with a sidecar,
  `derived_from` the source. Returns the table handle + `CacheOutcome`.
- **`AssembleContext`** → `InferenceSession::assemble_context`
  (`crates/jammi-ai/src/pipeline/context_set.rs`, also an `impl InferenceSession` in
  the pipeline module). Assembles + pools a target's context set and returns the
  result **inline**: the pooled `context_vector` as `repeated float` (bit-exact for
  `Vec<f32>`), hydrated value rows as one Arrow IPC stream
  (`crates/jammi-server/src/grpc/pipeline.rs`,
  `crates/jammi-ai/src/wire/pipeline.rs`).
- **`AsofJoin`** → `InferenceSession::asof_join` (`crates/jammi-ai/src/session.rs`) →
  `pipeline::asof::verb::run` (`crates/jammi-ai/src/pipeline/asof/verb.rs`).
  Point-in-time temporal join: for each `spine` row, attach the at-most-one `facts`
  row valid as-of its temporal key within each equality group; returns a materialised
  `ResultTable` (`crates/jammi-server/src/grpc/pipeline.rs`). Detail below.
- **`Recompute`** → resolve the named table through the tenant-scoped catalog, then
  `InferenceSession::recompute` (`crates/jammi-ai/src/pipeline/recompute.rs`);
  returns a `RecomputeReport` proto (`crates/jammi-server/src/grpc/pipeline.rs`).
  Detail below.

#### BuildNeighborGraph — data flow & the two drivers

`NeighborGraphPipeline::run` (`crates/jammi-ai/src/pipeline/neighbor_graph.rs`):
reject `k==0` → `resolve_embedding_table` (tenant-scoped) → optional cache probe →
`read_nodes` (`SELECT _row_id, vector`) → `build_edges` → `write_edge_table`. The
driver is chosen by `resolve_strategy` behind the `NeighborGraphStrategy` trait
(`crates/jammi-ai/src/pipeline/neighbor_graph.rs`):
- **`IndexAssisted`** (default when a sidecar index exists): queries the HNSW index
  `k+1` and drops the self-hit. **Approximate** (HNSW recall < 100%) and
  **non-deterministic** across runs — `is_exact()==false`.
- **`Exact`**: brute-force cosine over every pair via
  `jammi_numerics::distance::cosine_distance` → deterministic & complete, gated by
  `exact_max_rows` ceiling (default `DEFAULT_EXACT_MAX_ROWS = 50_000`). An `exact`
  build over a bigger table is **refused**, never silently downgraded.

Post-filters in `build_edges`: `self_exclude` (default on; off prepends the
self-edge at rank 0/similarity 1.0), `min_similarity` floor, `mutual` reciprocity
filter (`keep_mutual`). Ranks are 1-based. **Invariant:** the cosine→similarity map
is valid *only* because every Jammi index is cosine (`SidecarIndex` hardcodes `Cos`
[§2.4]); a future non-cosine metric must remap or refuse.

#### PropagateEmbeddings — data flow

`InferenceSession::propagate_embeddings`
(`crates/jammi-ai/src/pipeline/graph_propagation.rs`): `resolve_embedding_table` →
require non-null `dimensions` → `load_initial_features` (`X⁽⁰⁾`) → dispatch on
`PropagationWeighting`:
- `Uniform` / `DegreeNormalized` → `propagate_normalized` iterating `α·X⁽⁰⁾ +
  (1−α)·Â·X` for `effective_hops`, with `RandomWalk` (mean) or `Symmetric`
  normalisation; the default is `DegreeNormalized`.
- `EdgeSimilarity` → `propagate_edge_similarity`: per-node weighted mean `Σ(w·x)/Σw`
  over the bounded weighted-edge rows.

`assemble_output` keeps the last block (`Final`) or concatenates all blocks
(`JumpingKnowledge`, output width `dimensions·(effective_hops+1)`), then
`materialize_embedding_table` writes the new table `derived_from` the source. The
edge load runs through the **generic SQL surface** so the tenant-scope analyzer
scopes the scan — a cross-tenant endpoint is filtered before it reaches the
adjacency. An edge set above the ceiling is a typed error.

**Aggregation-operator caveat.** `propagate_embeddings` deliberately folds the
per-hop neighbour aggregation **in Rust** over a fixed `(group, neighbour)` order
(`aggregate_neighbours`) and explicitly does *not* route per-hop aggregation through
the streaming SQL UDAF, because the UDAF cannot promise byte-identical results across
partitionings (`crates/jammi-ai/src/pipeline/graph_propagation.rs`, the
aggregation-determinism doc). Only **`AssembleContext`'s** pool actually calls the
UDAF (next section). Two folds of the *same reduction*, two implementations chosen
for two determinism contracts — note this when you "unify the aggregator."

#### AssembleContext — Ann vs Edges vs Hybrid union

`InferenceSession::assemble_context`
(`crates/jammi-ai/src/pipeline/context_set.rs`) is the NP context-set encoder: `C =
search(query, k) ⋈ value_columns`, pooled permutation-invariantly. The **only** step
that varies by source is candidate gathering (`gather_candidates`); everything after
is one shared tail (exclude-self → split → cap → pool → hydrate):

- **`ContextSource::Ann { k }`**
  (`crates/jammi-ai/src/pipeline/context_set.rs`) → `ann_candidates`:
  `search_vectors(query, k+1)` (over-fetch by one under self-exclusion), keys in
  descending-similarity order.
- **`ContextSource::Edges(EdgeGather)`** → `gather_edge_candidates`
  (`crates/jammi-ai/src/pipeline/graph_neighbourhood.rs`), a bounded target-anchored
  declared-edge walk.
- **`ContextSource::Hybrid { ann_k, edges, merge: HybridMerge::Union }`** → run
  **both** arms, then merge: **ANN keys first (similarity order), then the
  declared-edge keys not already present** (dedup via `HashSet`), pooled once.
  `HybridMerge` is an enum (not a bool) so per-edge-type channels can be added
  without a breaking reshape; v1 ships only `Union`. Proto decode: absent `edges` ⇒
  `Ann`; present + `hybrid` flag ⇒ `Hybrid{Union}`; present alone ⇒ `Edges`
  (`crates/jammi-ai/src/wire/pipeline.rs`).

Shared tail invariants: `exclude_self` (default true) drops same-key neighbours — the
**leakage guard**; `split` predicate scopes to a train split, applied *after* gather
because the predicate is over source columns the ANN index doesn't carry; only ANN is
count-capped (`max_keys`) — an edge/hybrid set keeps every gathered member. Pooling
(`pool_context_vectors`) is the **one place AssembleContext touches the UDAF**:
resolves `vector_{mean,sum,max}` by name from the `FunctionRegistry` over a bound
IN-list of keys. Empty key set ⇒ `context_vector: None` (a degenerate set is not
silently averaged); `context_size` is carried **separately** from the vector so the
count signal never corrupts the pooled representation. The recorded `source` fact
(`ContextSourceKind`) is descriptive, **not** an exchangeability judgment —
governance decides coverage.

#### AsofJoin — point-in-time temporal join

`InferenceSession::asof_join` (`crates/jammi-ai/src/session.rs`, tenant-scoped) →
`asof::verb::run` (`crates/jammi-ai/src/pipeline/asof/verb.rs`): resolve
`spine`/`facts` to physical scans (`scan_relation`) → `SortExec` each by `(by…,
time[, tie-break])` ascending (`sort_for_merge`) → plan `AsofJoinExec`
(`crates/jammi-ai/src/pipeline/asof/exec.rs`) → single-pointer sort-merge
(`crates/jammi-ai/src/pipeline/asof/merge.rs`) → `BuildingTable::finish` with a
typed `ProducingDescriptor::AsofJoin` + an input anchor for **both** relations
(`crates/jammi-ai/src/pipeline/asof/verb.rs`). Left rows are always preserved;
unmatched fact columns are null. The result `model_id` is a sentinel `"asof-join"`
(no model runs, `ASOF_JOIN_MODEL_ID`, `crates/jammi-ai/src/pipeline/asof/verb.rs`).

The frozen contract is `AsofJoinSpec` (`crates/jammi-ai/src/pipeline/asof/spec.rs`),
built via `AsofJoinSpecBuilder` with **four pinned knobs**, each an enum (no
stringly-typed flags), defaulting to the leakage-safe choice:
- **`MatchDirection`** (`crates/jammi-ai/src/pipeline/asof/spec.rs`): `Backward`
  (default — most recent fact at/before the instant, the only leakage-safe choice for
  past-keyed assembly) / `Forward` / `Nearest` (`Nearest` requires a numeric temporal
  key).
- **`Boundary`**: `Inclusive` (default, `<=`/`>=`) / `Exclusive`.
- **`Tolerance`**: optional `Duration(µs)` or `Steps(i64)` look-back/forward; a
  candidate past the limit is no-match.
- **`TieBreak`**: `ByColumnDesc(col)` (max value wins — the transaction-time column)
  or `Error` (default — a true duplicate at the matched instant fails loudly with
  `AsofError::AmbiguousMatch`, never a non-deterministic pick).

Schema-dependent validation (`AsofJoinSpec::validate_against`,
`crates/jammi-ai/src/pipeline/asof/spec.rs`) runs when the operator binds: every `by`
column exists; the temporal key is a totally-ordered Arrow type (**floats rejected** —
NaN has no total order, `is_totally_ordered`); the two temporal keys share a type;
`Nearest` only over a numeric key (`is_numeric`). Errors are the typed `AsofError`
set. **Consumer spec:** cookbook
`cookbook/book/chapters/19-point-in-time/point-in-time.qmd` (`db.asof_join`,
backward/inclusive/`by` entity/`tolerance` look-back/deterministic tie-break) — the
authoritative usage. No conformal/evidence channel attaches here; it is a pure
temporal join.

#### Recompute — the action half of the materialization contract

`InferenceSession::recompute` (`crates/jammi-ai/src/pipeline/recompute.rs`,
tenant-scoped via the catalog resolution at the gRPC seam): read the named table's
recorded `ProducingDescriptor` (`recompute_one`/`replay_descriptor`), reconstruct the
producing verb call from its typed parameters, and replay it through the **unmodified
`BuildingTable::finish` funnel** with `CachePolicy::Bypass` (a recompute that reused
a cache would be a no-op). Byte-identical when inputs haven't moved, **on the
producing host** (the descriptor records every output-affecting determinant,
but not the CPU microarchitecture that ran the fold); on a different host the
replay is value-equivalent up to float-ULP rounding, and the `definition_hash`
+ catalog row — not the raw bytes — is what a recompute-vs-original comparison
is over. A pre-contract table with no descriptor
is the typed `JammiError::NotRecomputable` — a loud refusal, never a re-run guessed
from columns.

`Cascade` (`crates/jammi-ai/src/pipeline/recompute.rs`) selects one of two
**bounded** actions — the engine ships the actuator, never the control loop:
- **`Cascade::ReportOnly`** (default) — recompute the **named** table only and
  *report* the transitive downstream-stale set (`derives_from_closure`), recompute
  none of it.
- **`Cascade::Downstream`** — **one** bounded topological sweep on this single
  request: recompute the named table, then every transitive dependent in dependency
  order (parent's new digest lands before its child recomputes,
  `recompute_downstream_sweep`). Stack-safe iterative Kahn sort over the closure
  (`topological_recompute_order`); a diamond recomputes the shared child once; a cycle
  in the recorded lineage is `JammiError::DependencyCycle`.

`replay_descriptor` dispatches per `ProducingDescriptor` variant. The engine-owned
producers each rebuild their producer call from recorded typed params, all with
`CachePolicy::Bypass`; the one exception is `External`, a consumer-materialized producer
the engine does not own, so it carries no replay arm and returns
`JammiError::NotRecomputable` rather than a re-run guessed from columns (see the canonical
variant list in §2.4e, *producing_descriptor and NotRecomputable*). The intricate case
among the replayable producers is `ContextSet`: its real producer is the
`assemble_context`→`materialize_context` **pair**, so `recompute_context_set`
(`crates/jammi-ai/src/pipeline/recompute.rs`) re-pools every target's context over the
source's *current* rows under the recorded recipe (`context_recipe_from_manifest`),
skipping now-degenerate (empty) targets rather than fabricating a zero vector. The
output is a `RecomputeReport`: the tables re-produced (in recompute order) + the
transitive downstream-stale set. **Consumer spec:** cookbook
`cookbook/book/chapters/20-recompute/recompute.qmd` (`recompute(table, cascade=…)`
alongside `staleness` / `derives_from` / opt-in `cache="use"`) — the authoritative
usage, measured hermetically on CPU.

#### Where conformal / uncertainty wrap the served predictor — and why it stays dormant

The served context predictor lives in
`crates/jammi-ai/src/pipeline/context_predictor.rs`.
`predict_with_context_predictor_provenanced` reads the target vector, calls
`assemble_context` for the live context using the served source mapped via
`ContextServeSource::to_context_source` (which itself can yield `Hybrid{Union}`), runs
the in-context forward, and returns a `PredictedDistribution` + the `source` fact.
**This is LIVE** — gRPC `InferenceService.Predict`
(`crates/jammi-server/src/grpc/inference.rs`, `InferenceServer::predict`) and Python
`predict_with_context_predictor` (`crates/jammi-python/src/database.rs`) both call it,
and the served Hybrid arm is reachable via the serve-source reconstruction.

The conformal wrap is `ConformalContextPredictor`
(`crates/jammi-ai/src/pipeline/context_predictor.rs`) = a calibrated `ConformalModel`
(split-conformal) + the served `DistributionForm`.
`ConformalContextPredictor::interval(prediction, group)` turns one served
distribution into a coverage-guaranteed `[lower, upper]`: a Gaussian head wraps with
**AbsoluteResidual** centred on the predictive mean; a quantile head wraps with
**CQR** over its served quantile bounds. It is constructed by
`InferenceSession::calibrate_context_predictor_conformal` on a **held-out**
calibration set, with `ConformalLevers` (`Marginal`/`Mondrian`/`Weighted`) — the
engine *applies* the chosen lever, never *chooses* one.

**Wiring status — DORMANT.** `ConformalContextPredictor`,
`calibrate_context_predictor_conformal`, and `.interval(...)` have **no non-test
caller in the entire workspace** — the only callers are
`crates/jammi-ai/tests/it/context_predictor.rs`. There is **no gRPC verb
and no Python binding** for the served conformal wrap; `InferenceService.Predict`
returns the bare `distribution` + source tag + `context_ref`, not an interval. The
wrap is **defined in** the pipeline module but is a **dormant primitive** — exposing
it on a surface is a feature, not a config flag.

**Two distractor surfaces that are NOT the served conformal wrap — do not conflate:**
1. **The cookbook `08-conformal` chapter**
   (`cookbook/book/chapters/08-conformal/conformal.qmd`) exercises the Python
   `db.conformalize` / `db.conformalize_interval` / `db.conformalize_cqr` family
   (`crates/jammi-python/src/database.rs`). Those bind the **standalone
   `jammi_ai::predict::ConformalModel`** directly over caller-supplied class scores /
   `(pred_mean, pred_std)` arrays — the chapter calls these explicitly "client-local
   conformal numerics" and consumes a predictor's *already-computed* per-row
   mean/std, conformalizing client-side. It does **not** touch
   `ConformalContextPredictor`; it is not evidence the served wrap is wired.
2. The eval-runner `eval_calibration` (`crates/jammi-ai/src/eval/runner.rs`,
   `EvalRunner::eval_calibration`) is a *different* concept — strictly-proper-score
   calibration evaluation, not the served conformal wrap.

#### Extension — adding a context source, aggregator, or pipeline verb

- **New aggregator:** add a `SetAggregator` variant
  (`crates/jammi-ai/src/pipeline/context_set.rs`) + its `udaf_name()` arm; the UDAF
  must already be registered [§2.4a/§4.8]. Mirror it in `set_aggregator_from_proto`
  (`crates/jammi-ai/src/wire/pipeline.rs`) and the proto enum. One reduction, one
  registration site.
- **New context source / merge:** add a `ContextSource` variant
  (`crates/jammi-ai/src/pipeline/context_set.rs`) + its `kind()`/`max_keys()` arms, a
  `gather_candidates` arm, and (for hybrid behaviour) a `HybridMerge` variant. Keep
  the shared tail untouched — exclude-self/split/cap/pool/hydrate stays one path so
  every source is leakage-scoped identically. Wire decode is
  `assemble_context_request_from_proto` (`crates/jammi-ai/src/wire/pipeline.rs`) and
  the `ContextServeSource` serve mapping
  (`crates/jammi-ai/src/pipeline/context_predictor.rs`) for the predictor path.
- **Recompute a new producer:** an engine-owned materialising verb must add a
  `ProducingDescriptor` variant *and* a matching `replay_descriptor` arm
  (`crates/jammi-ai/src/pipeline/recompute.rs`) that replays it with `CachePolicy::Bypass`.
  The one deliberate exception is `External` — a consumer-materialized producer the engine
  does not own: it carries a `ProducingDescriptor::External` variant but **no** replay arm,
  returning `JammiError::NotRecomputable` by design (not recomputable, never a re-run guessed
  from columns).

> **Invariant (belongs in §5):** the `source` fact (`ContextSourceKind`) is
> descriptive, never an exchangeability claim. Graph-assembled (`Edges`/`Hybrid`)
> context can break the i.i.d. assumption marginal conformal needs; the honest repair
> is a caller-supplied `Mondrian`/`Weighted` lever, and *choosing* it is governance's
> call, not a serving output (`crates/jammi-ai/src/pipeline/context_predictor.rs`).

### 2.4d asof_join & the verifiable-materialization contract (point-in-time correctness — LIVE end-to-end)

This is the engine's point-in-time-correctness primitive, and it is LIVE on every
surface. Unlike the dormant conformal wrap of §2.4c, `asof_join` reaches code from
gRPC, the Python binding, *and* the recompute replay path; and it is the first compute
verb that writes through the **materialization contract** funnel
(`BuildingTable::finish`) — every result table carries a verifiable
`.materialization.json` attestation, and `asof_join` is the worked example the
cookbook documents (`cookbook/book/chapters/19-point-in-time/point-in-time.qmd`). The
two ship together: `asof_join` produces a leakage-free table, `verify_materialization`
attests it.

**Module shape** (`crates/jammi-ai/src/pipeline/asof/`, four concerns behind four
types, `crates/jammi-ai/src/pipeline/asof/mod.rs`): `spec` is the *what* (the frozen
`AsofJoinSpec` + four pinned knobs + typed errors), `exec` is the *plan contract*
(`AsofJoinExec` physical operator), `merge` is the *algorithm* (single-pointer
sort-merge core), `verb` is the *lifecycle* (resolve → plan → run → write → attest).
The operator/merge are reusable; the verb is the public surface.

#### The four pinned knobs — invalid-state-unrepresentable (`spec.rs`)

`AsofJoinSpec` (`crates/jammi-ai/src/pipeline/asof/spec.rs`) carries two `AsofKey`s
(each = `by: Vec<String>` equality columns + exactly one `time` column) plus four enum
knobs — never stringly-typed flags. Built through `AsofJoinSpecBuilder` (the >3-param
builder rule); every knob defaults to the leakage-safe choice (`Backward` /
`Inclusive` / no tolerance / loud `Error` / project-all):

- **`MatchDirection`**: `Backward` (default; most recent fact at/before — the *only*
  leakage-safe choice for past-keyed assembly), `Forward` (first fact at/after — leaks
  the future by construction), `Nearest` (smallest absolute distance, equidistant ties
  resolve toward the past; **requires a numeric temporal key**).
- **`Boundary`**: `Inclusive` (default, `<=`/`>=`) vs `Exclusive` (strict `<`/`>`).
  The single most error-prone as-of decision; pinned, never inferred.
- **`Tolerance`**: optional look-back/forward limit, `Duration(i64)` µs for temporal
  keys or `Steps(i64)` for integer keys; measured **relative to each spine instant,
  never wall-clock now**. A candidate beyond the limit → no-match (spine row
  preserved, fact columns null).
- **`TieBreak`**: `ByColumnDesc(String)` (a secondary column, maximal value wins —
  disambiguates late-arriving facts coincident on event time) or `Error` (no secondary
  column → a true duplicate at the matched instant fails **loudly** with
  `AsofError::AmbiguousMatch`, never a non-deterministic pick).

**Validation is split build-time vs bind-time.** `AsofJoinSpecBuilder::build` is pure
— assembles the descriptor without a schema. All schema-dependent invariants are
enforced by `AsofJoinSpec::validate_against` when the operator binds to its inputs:
every `by` column exists on its side, each temporal key is a totally-ordered type
(`is_totally_ordered` — every timestamp/date/integer width; **floats rejected because
NaN has no total order**), the two temporal keys share a type, and `Nearest` is used
only over a numeric key (`is_numeric`). Typed error set: `AsofError` =
`UnorderedTimeKey` / `MissingByKey` / `TimeKeyTypeMismatch` / `NearestRequiresNumeric`
/ `AmbiguousMatch` / transparent `DataFusion`.

#### AsofJoinExec — the physical operator (`exec.rs`)

A hand-built `ExecutionPlan` in the engine's existing operator idiom
(`InferenceExec`/`AnnSearchExec`), **not** a logical node behind an
`ExtensionPlanner` — the engine plans no `LogicalPlan` for its compute verbs.
`AsofJoinExec::try_new` (`crates/jammi-ai/src/pipeline/asof/exec.rs`) validates the
spec against both child schemas, resolves the right-projection to indices
(`resolve_projection`: empty `project` → every non-`by`, non-`time` right column),
derives the left-outer output schema, and caches `PlanProperties` (single
`UnknownPartitioning(1)`, `EmissionType::Final`, `Bounded`).

**The operator declares its semantic requirements truthfully** so any planner that
ever drives it inserts the right shuffle/sort and `EXPLAIN` shows the contract:
`AsofJoinExec::required_input_distribution` — both children hash-partitioned on their
`by` keys (empty `by` ⇒ `SinglePartition`, one global group);
`AsofJoinExec::required_input_ordering` — each child ascending by (`by`..., `time`),
and the *right* side additionally by the tie-break column when `ByColumnDesc` (the
tie-break lives only on the facts side, `ordering_requirement`). Ascending on the
tie-break is correct because the merge takes the **last** eligible fact, so the
maximal tie-break value lands last.

`AsofJoinExec::execute`: collect+concat each side into one sorted run
(`collect_concat` — both sides are bounded scans, so the full collect is the natural
shape), `SortedPartition::resolve` each, then `merge_partition`. "At most one match"
is what lets the single-pointer merge never backtrack.

#### The merge core — O(n+m) per group, no per-type branches (`merge.rs`)

`merge_partition` (`crates/jammi-ai/src/pipeline/asof/merge.rs`) emits exactly one
output row per left row (left row + matched fact's projected columns, or nulls when
unmatched — a **left-outer** as-of). Everything reduces to two scalar comparisons
defined once:

- **`by`-tuple equality** via Arrow's row encoding (`GroupKeys`): one `RowConverter`
  shared across both sides so encodings compare; an explicit null mask enforces SQL
  `NULL ≠ NULL` at group boundaries (a null-key row is its own singleton matching
  nothing). With **no** `by` columns it is the `Global` single-group variant —
  row-encoding skipped entirely (a 0-field converter has no `.row(i)`).
- **temporal "at or before"** via `temporal_i128`: every timestamp/date/integer width
  widened losslessly into `i128`, one comparison domain. A null instant is `None` — a
  null-time *left* row is preserved with a null match; a null-time *right* row is never
  a candidate.

`MatchDirection` parameterises one cursor: `merge_directional` (backward = remember the
last eligible `<=`/`<`; forward mirrors it = first `>=`/`>`) and `merge_nearest`
(single forward scan tracking the backward + forward candidate, `nearest_of` picks the
smaller `|distance|`, equidistant → past wins). Tolerance is applied by `within_limit`.
Ambiguity is detected loudly by `detect_ambiguous` (a run of two facts sharing the
matched instant under `TieBreak::Error`). Output assembled by `assemble_output`: left
columns ride through unchanged; each projected right column gathered by one vectorised
`take` per column (null index ⇒ null output), never a per-row copy.

#### The verb — lifecycle + the materialization-contract write (`verb.rs`)

`asof::verb::run(session, spine, facts, spec)`
(`crates/jammi-ai/src/pipeline/asof/verb.rs`) owns resolve → plan → run → write →
attest:

1. **Resolve** both relations through the session's tenant-scoped SQL path
   (`scan_relation`; `SELECT * FROM <relation>` so a caller cannot point either side
   at another tenant's relation).
2. **Plan**: wrap each scan in a `SortExec` ordering it for the merge (`sort_for_merge`
   — ascending by `by`..., `time`[, tie-break]; the verb satisfies the operator's
   declared ordering requirement), then `AsofJoinExec::try_new`.
3. **Run**: execute partition 0, collect batches.
4. **Write**: `create_table` with `ResultTableKind::AsofJoin`
   (`crates/jammi-db/src/catalog/result_repo.rs` — `"asof_join"`); `model_id` is a
   fixed sentinel `"asof-join"` (`crates/jammi-ai/src/pipeline/asof/verb.rs`, mirroring
   the neighbor-graph sentinel — the join invokes no model but the column is NOT NULL).
   `derived_from` is `None`: the inputs are registered *sources*, not result tables, so
   FK-lineage rides the manifest's input anchors instead.
5. **Attest** through `BuildingTable::finish`
   (`crates/jammi-ai/src/pipeline/asof/verb.rs` → `crates/jammi-db/src/store/building.rs`) —
   the single materialization funnel. The contract: a typed
   `ProducingDescriptor::AsofJoin` (`descriptor_for`, mapping the AI-crate enums to the
   transport-neutral manifest mirrors), a `MaterializationEnv` with an **empty model
   set** (the join runs no model), and a read-time `InputAnchor::unpinned_at_instant`
   for **both** relations. **Honesty invariant:** a registered source exposes no
   as-of/version surface in open-core, so each input is recorded `UnpinnedAtInstant`
   (the read instant only) rather than a fabricated pin — which is exactly why the
   cookbook's clean as-of training set verifies as `MatchWithUnpinnedInputs`, not
   `Match`.

**Call graph — LIVE:** `InferenceSession::asof_join` (`crates/jammi-ai/src/session.rs`,
tenant-scoped wrapper over `verb::run`) is reached by gRPC `PipelineService.asof_join`
(`crates/jammi-server/src/grpc/pipeline.rs`), the Python binding `_asof_join_proto`
(`crates/jammi-python/src/database.rs`, decoding `wire::asof_join_from_bytes`,
`crates/jammi-ai/src/wire/pipeline.rs`), **and** the recompute replay path
(`crates/jammi-ai/src/pipeline/recompute.rs`, which rebuilds the spec from the recorded
`ProducingDescriptor::AsofJoin` and re-invokes the join). The cookbook's
`db.asof_join(...)` takes lowercase `direction`/`boundary` strings and bare source ids,
reading the output as `"jammi.<table>"`.

#### The materialization contract (`jammi-db` `store/manifest.rs`) — the verifiable identity

A separate `.materialization.json` sidecar is written for **every** result table (not
only embedding tables — distinct from the ANN bundle's `.manifest.json`, which
describes the search accelerator; `crates/jammi-db/src/store/manifest.rs`). It binds
three things to the artifact's content digest:

1. **Definition hash** (`DefinitionHash`, `crates/jammi-db/src/store/manifest.rs`) —
   SHA-256 over the canonical `ProducingDescriptor` bytes **and** the
   `MaterializationEnv` (engine version, invoked-model identities, input backend kinds,
   **and the compute device** — a model on CPU vs CUDA yields different floats under the
   same model identity, so the device is part of the hashed world). Computed by
   `definition_hash`, length-prefixed + domain-separated so a descriptor field can never
   alias an env field; JSON canonicalised with sorted object keys but array order
   preserved (`canonicalize_json`).
2. **Immutable input anchors** (`InputAnchor`,
   `crates/jammi-db/src/store/manifest.rs`) — per-input state pointers by `AnchorKind`:
   `ResultDigest` (a result-table input's content digest), `MutableVersion` (a companion
   table's monotonic version), `SourceVersion` (an external source's pinned as-of
   column), or `UnpinnedAtInstant` (an external source with no version surface — the
   read instant only; the manifest records this input is **not** reproducibly pinned, so
   a verifier downgrades confidence honestly). Anchors are **not** part of the definition
   hash — the definition is *how*, the anchors are *over what*.
3. **Producing-run identity + instant** (`produced_by` / `produced_at`, provenance only,
   never the anchor).

Beside the digest — never in place of it — the manifest carries a **keyed leaf
inventory** (`leaves: Vec<LeafDigest>`, `LeafKey`, `crates/jammi-db/src/store/manifest.rs`): for a result table one leaf per Parquet row group, keyed by its
index and byte range as the footer locates it (`parquet_leaves`, the SHA-256 of exactly
that range); for a model bundle one leaf per file keyed by NAME (the bundle manifest's own
sha256, so adding a file changes no existing leaf and `combined_hash` stays the bundle's
content address). The inventory is what a peer verifies ONE partition against without
reading the rest. It is additive: `artifact` stays the whole-object digest, the in-toto
subject, the root of the version-identity chain, and what a verifier holding the bytes
recomputes; bytes outside every row group (footer, page index, bloom filters) belong to
no leaf and are the whole-object digest's to catch.

`MANIFEST_VERSION = 3` (`crates/jammi-db/src/store/manifest.rs`); a version mismatch or a
serde-shape mismatch is a typed `ManifestError`, never a silently-trusted stale hash
(`Manifest::from_json_bytes`). One shape rejection is named on its own: an object at the
current version with no `leaves` — a sidecar written before the inventory existed —
is `ManifestError::PreLeavesSidecar`, and `ResultStore::read_materialization_manifest`
reads exactly that as ABSENT (the pre-contract case every reader already handles: a
verify says `MissingManifest`, an anchor recomputes from the bytes, a cache probe misses
and re-materialises); a newer version or a corrupt body stays the error it is, so an
older binary never re-materialises over a newer engine's table. The descriptor is kept
**verbatim** in the manifest (not just the opaque hash) precisely so a reader can
*replay* it — that is what the recompute path reads.

#### verify_materialization — the read-only verb, four verdicts

`ResultStore::verify_materialization` (`crates/jammi-db/src/store/mod.rs`): re-reads the
Parquet bytes, recomputes `ArtifactDigest::of_bytes`, compares to the manifest's,
optionally compares the manifest's `definition_hash` to a caller-supplied expectation,
and returns a `MatchVerdict`. **It never acts on a verdict** — refuse / alarm / fall back
is the consumer's policy. The verdict attests the Parquet **data**, never the ANN index.
`MatchVerdict` (`crates/jammi-db/src/store/manifest.rs`):

- **`Match`** — recomputed digest equals the manifest's (and definition hash matches if
  supplied). A clean `Match` requires every input reproducibly pinned (the chapter's
  `match` case is a neighbor graph over an embeddings result table, sole input
  `ResultDigest`-anchored).
- **`Mismatch { expected, found }`** — digest or definition hash diverged (a stale copy,
  or a changed producing query). Carries both sides.
- **`MatchWithUnpinnedInputs { unpinned }`** — verifies, but ≥1 input was
  `UnpinnedAtInstant` (e.g. a registered file source), so reproducibility cannot be
  *fully* asserted; it names the unpinned inputs (`unpinned_inputs`). **This is the
  honest verdict for the as-of training set itself** — its inputs are registered file
  sources.
- **`MissingManifest`** — no sidecar (a pre-contract table, or a pre-`leaves` sidecar).
  A truthful unknown, never a fabricated match — distinct from a post-contract table
  that *should* carry one (a torn write recovery reconciles).

#### verify_partitions — the per-partition verb

`ResultStore::verify_partitions` (`crates/jammi-db/src/store/mod.rs`) recomputes every
leaf of the inventory from the bytes and the footer and compares each to the recorded
one by key, returning a `PartitionVerdict` (`crates/jammi-db/src/store/manifest.rs`):
`Match`; `Mismatch { key, expected, found }` naming the FIRST row group whose bytes are
not the attested ones; `InventoryDiffers { expected, found }` when the footer's row-group
set is not the recorded one; `MissingManifest` as above. It attests the parts, never the
whole — a footer-only mutation changes no leaf and is `verify_materialization`'s to
report. Read-only, like its sibling: it never acts on a verdict.

**Call graph — LIVE:** `Session::verify_materialization`
(`crates/jammi-ai/src/local_session.rs`) → gRPC
`CatalogService.verify_materialization` (`crates/jammi-server/src/grpc/catalog.rs`,
verdict mapped to the proto oneof via `match_verdict_to_proto`,
`crates/jammi-wire/src/catalog.rs`) and Python `Database::verify_materialization`
(`crates/jammi-python/src/database.rs`). The chapter freezes the full four-verdict matrix
as committed golden contracts.

> **Invariant (belongs in §5):** the `definition_hash` is not bit-reproducible when a
> producer's inputs include non-bit-reproducible content (e.g. CPU embeddings) — the
> chapter records it as provenance and asserts the *round-trip* fact instead (a table
> verifies `Match` against its **own** recorded hash within the run). The verdict
> *mechanics* are pinned; the per-run content digest is not.

#### Extension — adding a knob or a materialized producer

- **New as-of knob / direction:** add the `spec.rs` enum variant + its
  `validate_against` rule, the merge arm (`merge_partition`'s dispatch,
  `crates/jammi-ai/src/pipeline/asof/merge.rs`), the manifest mirror enum
  (`AsofDirection`/`AsofBoundary`/`AsofTolerance`,
  `crates/jammi-db/src/store/manifest.rs`) + both mappings (`descriptor_for`,
  `crates/jammi-ai/src/pipeline/asof/verb.rs`; the recompute reverse-mappers,
  `crates/jammi-ai/src/pipeline/recompute.rs`), and the wire decode
  (`crates/jammi-ai/src/wire/pipeline.rs`). The knob must move the definition hash —
  `asof_join_each_knob_moves_the_hash` (`crates/jammi-db/src/store/manifest.rs`) is the
  guard test.
- **New materialized producer (any verb):** write through `BuildingTable::finish` with a
  new `ProducingDescriptor` variant (`crates/jammi-db/src/store/manifest.rs`) carrying
  every output-affecting parameter, and honest `InputAnchor`s for every input read. Bump
  `MANIFEST_VERSION` (`crates/jammi-db/src/store/manifest.rs`) if the descriptor's
  determinant set changes. Do **not** hand-write a `.materialization.json` — the funnel is
  the one place it is produced.

### 2.4e Incremental recompute & opt-in caching — the sensing layer, the recompute actuator, and the memoization dial (LIVE end-to-end)

This is the engine's incremental-recompute primitive, LIVE on every surface — but it
ships only the *mechanism*, never the *control loop*. It stands directly on the
materialization contract [§2.4d]: every `ready` result table records a verifiable
`(definition_hash, input_anchors, ProducingDescriptor)` in its `.materialization.json`
sidecar, and this layer reads that record three ways. The split is deliberate and stated
in the source: a **sensing** half *reports* (read-only staleness, lineage, cache-lookup;
`crates/jammi-db/src/store/freshness.rs`) and an **action** half *acts* (`recompute`
replays the producer; `crates/jammi-ai/src/pipeline/recompute.rs`). The engine ships the
actuator (`recompute`) and the sensor (`staleness`/`derives_from`); it deliberately ships
**no** scheduler, no staleness-monitor that triggers recompute, no cache TTL/eviction —
wiring a sensor→actuator control loop is the consumer's composition on a published version
(`crates/jammi-ai/src/pipeline/recompute.rs`; cookbook
`cookbook/book/chapters/20-recompute/recompute.qmd`). The cookbook is the authoritative
consumer spec.

#### The cache primitives — opt-in, observable, never silent (`store/freshness.rs`)

Every result-table producer carries one shared **opt-in memoization dial** and returns one
**observable outcome** — the enums live in `crates/jammi-db/src/store/freshness.rs` (not a
wire type), so they are engine semantics, not transport:

- **`CachePolicy`** (`crates/jammi-db/src/store/freshness.rs`): `Use` (probe the cache,
  short-circuit on an exact hit) vs `Bypass` (always recompute). **The default is
  `Bypass`** — a producer must never silently hand back a table the caller did not just
  compute (the "honest, not silent" rule). Reuse is therefore both explicitly *requested*
  (`Use`) and explicitly *reported*.
- **`CacheOutcome`** (`crates/jammi-db/src/store/freshness.rs`): `Computed` (the compute
  ran) vs `Reused { table }` (an exact hit short-circuited). Returned alongside every
  producer's record so reuse is **observable on the wire**, never inferred.

The hit test is two functions:
- **`lookup_cached`** (`crates/jammi-db/src/store/freshness.rs`) — the read-only
  **sensor**: narrow by the indexed predicate `definition_hash = $1 AND status='ready'`
  (the `idx_result_tables_definition_hash` index,
  `crates/jammi-db/src/catalog/schema.rs`), then an exact Rust set-equality post-filter
  over each candidate's decoded `input_anchors_json` (`anchor_sets_equal` —
  order-insensitive, since an anchor set is structured, not a SQL scalar). **An
  `UnpinnedAtInstant` anchor in the requested set short-circuits to a miss**: an instant is
  not a reproducible id, so equal instants don't prove equal inputs — a "hit" would be
  fabricated reuse.
- **`probe_cache` / `probe_cache_record`** (`crates/jammi-db/src/store/freshness.rs`) —
  the **action-layer** probe a producer runs at the top of its verb: `lookup_cached`
  *plus* an **extant-artifact check** (re-confirm the Parquet bytes the cached `ready` row
  points at still exist on disk). The difference is torn-write safety: a `ready` catalog
  row whose bytes were reaped (a commit-before-durability power loss, a half-deleted table)
  must not be handed back — fall through to a recompute instead of short-circuiting to an
  unreadable table. `probe_cache_record` returns the full `ResultTableRecord` so a producer
  hands the reused record straight back with no second catalog read.

#### Which producers are cacheable — and "embed/infer is honestly off" (the unpinned-anchor truth)

Five producers carry the `CachePolicy` parameter and probe at the top of their verb;
**`asof_join` does not** (it has no cache dial and always recomputes —
`crates/jammi-ai/src/pipeline/recompute.rs`). Of the five, the **kind of input anchor**
decides whether a hit is even *possible*:

- **`infer`** (`crates/jammi-ai/src/session.rs`, `InferenceSession::infer`) and
  **`EmbeddingPipeline::run`** (`crates/jammi-ai/src/pipeline/embedding.rs`) anchor their
  sole input — a *raw source* — as `UnpinnedAtInstant`, because an open-core source has no
  version surface. So a `Use` request is **honestly always a miss** (`probe_cache`
  short-circuits any unpinned anchor) and the outcome is always `Computed`. **The probe
  still runs** so the surface is uniform and the off-ness is provable, not hidden. This is
  the "embed==remote memoization" point: the embedding/inference cache is *defined and
  wired* but *honestly inert* until sources expose a version surface — the path is correct
  the moment a versioned source makes it cacheable.
- **`build_neighbor_graph`** (`crates/jammi-ai/src/pipeline/neighbor_graph.rs`),
  **`propagate_embeddings`** (`crates/jammi-ai/src/pipeline/graph_propagation.rs`), and
  **`materialize_context`** (`crates/jammi-ai/src/pipeline/context_set.rs`) anchor on
  **immutable result tables** (`AnchorKind::ResultDigest`), so they are **genuinely
  cacheable** — `cache="use"` over an unchanged parent returns the *same* table name. The
  probe keys on the **full** `ProducingDescriptor`, not one knob: a `k=4` build or `hops=2`
  propagation misses against a `k=3`/`hops=1` cache. **Cookbook gotcha:**
  `propagate_embeddings` must pin `embedding_table=emb` explicitly to be cacheable — left
  unpinned it re-resolves "latest ready embedding for the source", a fresh-named table, so
  its `ResultDigest` anchor differs and the probe misses.

The wire mirror is one shared module: `crates/jammi-ai/src/wire/cache.rs` decodes
`pb::CachePolicy`→`CachePolicy` (`UNSPECIFIED`→`Bypass` default; out-of-range → loud
`invalid_argument`) and encodes `CacheOutcome`→`pb::CacheOutcome`. The proto defines the
enum **once** in `jammi.v1.inference`
(`crates/jammi-wire/proto/jammi/v1/inference.proto`) and the producer RPCs carry it as a
field (e.g. `crates/jammi-wire/proto/jammi/v1/pipeline.proto`). `SubmitJobRequest.cache`
(tag 10, `crates/jammi-wire/proto/jammi/v1/job.proto`) imports the same enum rather than
declaring a second wire vocabulary for the identical concept;
`crates/jammi-ai/src/wire/training.rs`'s `lora_common_from_proto` decodes it and returns it
alongside `TrainingCommon` (never folded into that type — only the `FineTune` decode arm
threads it onto `TrainingSpec::FineTune.cache`). Model-level cache reuse is not
supported for `TrainingSpec::FineTune`: `Use` is refused, typed, by `admit_training_spec`
— the one admission every durable submit edge for a training spec applies before a `jobs`
row is written (`InferenceSession::submit_fine_tune_spec_deduped`, `InferenceSession::enqueue`,
and `train_context_predictor_deduped`)
rather than probed against a recorded materialization, a different mechanism from the
*result-table* `probe_cache_record` path the producers above use.

**`cache = Use` is refused on both fine-tune kinds.** `cache` lives on
`TrainingSpec::FineTune` itself; `TrainingSpec::GraphFineTune` carries no `cache` field at
all, so a `GraphFineTune` job's `FineTuneRun::materialization_source` is unconditionally
`None` (`crates/jammi-ai/src/fine_tune/worker.rs`: the graph arm carries no materialization
to probe or record) and `lora_common_from_proto` refuses `cache = USE` for `GraphFineTune`
with a typed `InvalidArgument` at decode — the one place that can still see both the kind
and the requested value, mirroring the `ContextPredictor` `world_size` refusal in the same
module. On the column-source kind, `cache = Use` is refused later, at submit, for the
reason above. `Bypass`/unset is
unaffected on either kind: every fine-tune
job always trains. A stray `cache` key found
under `graph_fine_tune` in a persisted `jobs.spec` row is silently dropped at deserialize
rather than refused, since the type has nowhere to decode it onto.
The Python client still carries `cache=` as a kwarg on both `fine_tune` and
`fine_tune_graph` (beside `world_size`, on both transports); both are refused, not
silently dropped.

**A catalog row may be deleted at any time; bytes are reclaimed only when
unreferenced.** `ResultStore::prefix_is_referenced` is an admin-scoped (whole-catalog)
scan of `models.artifact_path` (the exact key or its immediate parent), and
`ResultStore::delete_unreferenced_prefix` consults it before every `models/`-prefix
byte-delete: this pass's reap, the worker's own abandon path, the worker's
epoch-checkpoint sweep (`JobWorker::gc_epoch_checkpoints_by_index`, which consults it on
each index's own exact checkpoint prefix before ever deleting), and the trainer's mid-run
retention prune (`delete_epoch_checkpoint_guarded`) — refusing typed
(`StorageError::Referenced { prefix, count }`) while any live `models` row, in any tenant,
still names the prefix. The one stated exemption is `{job}/_resume`: a sibling of every
attempt-level path, so no row's `artifact_path` can ever equal it or its immediate parent
— proven by an executed test, not asserted, in `crates/jammi-db/tests/it/reconcile.rs`'s
`a_resume_checkpoint_prefix_is_never_referenced_even_under_the_containment_aware_predicate`.
`reconcile`'s attribution set is
built from the same admin-scoped scan (never the tenant-scoped `list_models`), so a
tenant-bound reconcile pass can never reap a prefix a peer tenant's row still serves; a
prefix this pass's ordinary orphan check would otherwise reclaim, but which that same
admin-scoped consult still finds referenced, is reported in `ReconcileReport.referenced`
(and counted in `referenced_count`) instead of deleted.

**The recorded device identity.** `MaterializationEnv.device`
(`crates/jammi-db/src/store/manifest.rs`) folds `ComputeDevice::Cuda { ordinal }` /
`Metal { ordinal }` / `Cpu` into every recorded materialization — a device **ordinal**,
not a host or machine identity, so two invocations reporting the same ordinal are
numerically interchangeable, including across two different physical hosts that both
happen to enumerate a GPU at ordinal `0`. This field is recorded for every FineTune run
but is not consulted by any reuse decision (`cache = Use` on `FineTune` is refused
before training runs); should a future
reuse mechanism read it, treating same-ordinal
invocations as interchangeable is sound where accelerators are interchangeable, not a
guarantee across heterogeneous hardware, which would need a real per-host or per-device
identity folded into `ComputeDevice`.

#### The staleness/lineage sensing model (`store/freshness.rs`)

Three read-only sensors, all LIVE via the `Session` surface
(`crates/jammi-ai/src/local_session.rs`) and the Python binding
(`crates/jammi-python/src/database.rs`):

- **`staleness(table, current_definition)`** (`crates/jammi-db/src/store/freshness.rs`) →
  a **`Staleness`** verdict, variants ordered by confidence: `Fresh` (recorded hash ==
  current *and* every input anchor unchanged — reuse is safe); `Stale { reasons }` (every
  changed determinant confidently resolved); `Undecidable { unpinned, decided_reasons }`
  (an input has no reproducible current anchor — an honest "I don't fully know", with the
  confidently-decided reasons still reported); `MissingManifest` (a pre-contract table,
  `definition_hash IS NULL`). Reasons are typed `StaleReason`: `DefinitionChanged` /
  `InputAdvanced` / `InputVanished`.
- **`current_anchor(anchor)`** (`crates/jammi-db/src/store/freshness.rs`) resolves one
  recorded input to its *live* state-pointer. **Only `ResultDigest` has a live resolution
  surface today** — it reads the parent's current artifact digest from the parent's own
  manifest (recomputing from bytes for a pre-contract parent). `UnpinnedAtInstant`,
  `MutableVersion`, and `SourceVersion` all resolve to `CurrentAnchor::Undecidable`: the
  latter two are **structurally unreachable in a recorded anchor today** and have *no*
  current-resolution surface, so the layer documents the honest gap rather than fabricating
  a read. **Recursion falls out with no special case:** a recomputed parent gets a new
  digest, so a child anchored on the old one is detected stale by the same per-input
  comparison.
  - **Nuance — the `result_digest` input-drift arm is live in the engine but only
    described in the cookbook.** `InputAdvanced` (`crates/jammi-db/src/store/freshness.rs`)
    is fully wired and compared (in `staleness`). The cookbook *cuts the demonstration* of
    it: there is no hermetic Python verb to re-anchor an *existing* child onto a
    moved-digest parent (re-anchoring only happens when a producer runs and writes a *new*
    child), so fabricating the drift would require reaching past the public surface. The
    arm is engine-real; only its measured demo is cut. Do not read the cookbook callout as
    "the engine lacks this arm."
- **`derives_from(source)`** (`crates/jammi-db/src/store/freshness.rs`) → the one-hop
  reverse-dependency edges (`DerivesFromEdge`): every `ready` table whose recorded
  `input_anchors` name `source`. The lineage is a **view over `input_anchors_json`** — the
  single source of truth, *not* a second edge store. Candidate narrowing is a SQL `LIKE
  '%"source":"<name>"%'` over-approximation refined by an exact Rust decode-and-match. The
  transitive closure is **`derives_from_closure`** (`crates/jammi-db/src/store/freshness.rs`):
  a stack-safe iterative DFS with an explicit frame stack and an `on_path`/`expanded` pair
  that distinguishes a DAG diamond (walked once) from a true cycle (the typed
  `JammiError::DependencyCycle`) — never recursion.

#### producing_descriptor and NotRecomputable (the replay key)

A verifier reads the opaque hash; a **recomputer** reads the descriptor.
**`producing_descriptor(table)`** (`crates/jammi-db/src/store/freshness.rs`) reads the
`.materialization.json` sidecar and returns its `ProducingDescriptor` (the verbatim verb +
typed params, persisted **not** merely hashed away). A pre-contract table (no manifest) is
the typed **`JammiError::NotRecomputable { table }`** — a loud refusal, never a re-run
guessed from columns. `ProducingDescriptor` (`crates/jammi-db/src/store/manifest.rs`) is the
`#[serde(tag="producer")]` enum with one variant per producer, each carrying every
output-affecting determinant (float knobs stored by IEEE-754 bit pattern so the descriptor
stays `Eq`/`Hash`, e.g. `min_similarity_bits`, `alpha_bits`). The canonical variant set —
bound to the code enum and to `recompute.rs` by `ci/scripts/check_doc_parity.py`, which fails
CI if the guide and the code diverge:

<!-- BEGIN PRODUCING-DESCRIPTOR-VARIANTS -->
- `Inference` — a model run over a source's content columns, keyed by `key_column`.
- `Embedding` — a model embedding over a source's columns.
- `NeighborGraph` — a k-NN edge relation derived from an embedding table.
- `GraphPropagation` — K hops of feature propagation over a neighbor graph.
- `ContextSet` — per-target pooled context vectors materialised as an embedding table.
- `AsofJoin` — a point-in-time temporal join, each spine row matched as-of within its group.
- `TrainingSet` — the rows a training run reads, projected from a source relation and committed in one canonical full-tuple order; replayed by re-materializing.
- `GraphTrainingSet` — a graph fine-tune's sampled `(anchor, positive, [hard_negative])` pairs, materialized through the SAME producer funnel as `TrainingSet` via a `Batches` (`RecordBatch`-stream) input rather than SQL, committed in the sampler's own emission order (a leading `_ordinal` column, never re-sorted); the format tag (`graph_pairs`/`graph_triplet`) is decided from the recorded `sample.hard_negatives`, never from row content (<https://github.com/f-inverse/jammi-ai/issues/538>); replayed by re-reading the node/edge sources and re-sampling.
- `External` — a consumer-materialized table for a verb the engine does not own; no replay arm (returns `NotRecomputable` by design).
- `EmbeddingDelta` — an incremental refresh of an embedding table (only the changed rows re-embedded, deletion-mask horizons raised); replayed as a full embed into a new table.
- `EmbeddingCompaction` — a versioned embedding table's live rows rewritten as one fragment + one segment; replayed as a full embed into a new table.
- `FineTune` — a LoRA fine-tune run, keyed by the training-set table's definition hash + artifact digest + row count, the base model identity, and the whole `TrainingSpec::FineTune` canonical spec (`spec_canonical` + `spec_schema_version`); model-level cache reuse is not yet supported for this kind — `TrainingSpec::FineTune.cache = Use` is refused, typed, by `admit_training_spec`, the one admission every durable submit edge for a training spec applies (<https://github.com/f-inverse/jammi-ai/issues/562>), and `Bypass` (the only value a submitted job can carry past that refusal) always trains; `TrainingSpec::GraphFineTune` carries no `cache` field at all; replayed by retraining.
<!-- END PRODUCING-DESCRIPTOR-VARIANTS -->

#### The recompute verb — descriptor replay + bounded cascade (`pipeline/recompute.rs`)

`InferenceSession::recompute(table, cascade)`
(`crates/jammi-ai/src/pipeline/recompute.rs`) is the actuator. It computes the transitive
downstream set **once** (`derives_from_closure`) — reported by both arms — then dispatches
on `cascade`:

- **`Cascade::ReportOnly`** (default): recompute the **named table only** (`recompute_one`);
  *report* the downstream-stale set without touching it. The consumer decides.
- **`Cascade::Downstream`**: **one** bounded **topological** sweep on this single explicit
  request (`recompute_downstream_sweep`) — recompute the named table, then every transitive
  dependent in dependency order, each re-resolved freshly so a child reads its parent's
  *new* digest. The order is a stack-safe Kahn pass (`topological_recompute_order`):
  explicit queue + in-degree map, deterministic `sort()` among same-in-degree nodes, a
  diamond recomputed exactly once, a recorded-lineage cycle → `JammiError::DependencyCycle`.
  **No poll, no second pass** after the sweep finishes.

`recompute_one` → `replay_descriptor` (`crates/jammi-ai/src/pipeline/recompute.rs`)
dispatches on the descriptor variant and **always calls the producer with
`CachePolicy::Bypass`** — a recompute that reused a cache would be a no-op, not a recompute.
The replay is **byte-identical when inputs have not moved, on the producing
host** — the descriptor records every output-affecting determinant, but the
producing host's CPU microarchitecture is not one of them, so a replay on a
different CPU host is value-equivalent, not asserted byte-identical (identity
is the catalog row + `definition_hash`; see
`docs/guide/src/materialization-contract.md`).
Per-variant subtleties:
- The many `*_from_manifest` helpers are the reverse of the descriptor-recording `*_for`
  functions — mapping each manifest enum mirror back onto its AI-crate type.
- The intricate case is **`ContextSet`**: its real producer is the
  `assemble_context`→`materialize_context` **pair**, so `recompute_context_set` re-pools
  *every current source row* — reads `(_row_id, vector)` of the source's current embedding
  table (`read_target_rows`), builds one `ContextRequest` per target with that target's own
  vector as `query` and `_row_id` as `exclude_key` (the leakage guard), assembles+pools
  each, routes the pooled rows back through `materialize_context`. A now-empty target is
  skipped, never zero-filled.
- **`Inference`** writes a fresh source-named table per run and returns rows (not a record),
  so the recompute names its output by the **newest `ready`** table for `(source, task,
  model)` — `latest_ready_table_for`.
- **`AsofJoin`** rebuilds the spec via `AsofJoinSpecBuilder` and reports
  `CacheOutcome::Computed` unconditionally (no cache dial).

#### Surfaces (all LIVE)

- **`RecomputeReport`** (`crates/jammi-ai/src/pipeline/recompute.rs`) = the `recomputed`
  tables (`RecomputedTable { original, recomputed, outcome }`) + the `downstream_stale` set.
- **Session:** `Session::recompute` / `staleness` / `derives_from` /
  `verify_materialization` (`crates/jammi-ai/src/local_session.rs`) — all tenant-scoped via
  `get_result_table` (a peer cannot recompute a table it cannot resolve).
- **gRPC:** `PipelineService.Recompute` (`crates/jammi-wire/proto/jammi/v1/pipeline.proto`),
  handler `crates/jammi-server/src/grpc/pipeline.rs` (`PipelineService::recompute`), runs
  inside the request's tenant `scoped`. Wire decode/encode:
  `crates/jammi-ai/src/wire/pipeline.rs` (`RecomputeArgs`, `recompute_from_proto` —
  `UNSPECIFIED` cascade → `ReportOnly` default).
- **Python:** producer methods take `cache=...` (`crates/jammi-python/src/database.rs`);
  sensors `staleness` / `derives_from` / `verify_materialization`; `_recompute_proto`
  decodes a serialized `RecomputeRequest` and returns the serialized `RecomputeReport` — one
  engine call, one decode seam.

#### Extension — adding a cacheable producer

A new result-table producer becomes cacheable by: (1) adding its `ProducingDescriptor`
variant (`crates/jammi-db/src/store/manifest.rs`) with **every** output-affecting
determinant (floats by bit pattern); (2) probing at the top of the verb (`if cache ==
CachePolicy::Use { … probe_cache_record(&def_hash, &inputs) … return Reused }`, the
`EmbeddingPipeline::run` shape, `crates/jammi-ai/src/pipeline/embedding.rs`) and returning
`(record, CacheOutcome)`; (3) writing through `BuildingTable::finish` with the *same*
`(descriptor, env, inputs)` the probe keyed on; (4) adding a `replay_descriptor` arm that
calls the producer with `CachePolicy::Bypass` and the matching `*_from_manifest`
reverse-mappers. Anchor immutable result-table inputs as `ResultDigest` (cacheable) and raw
sources as `UnpinnedAtInstant` (honestly never a hit). Do **not** add a scheduler or a
staleness→recompute loop — that is the platform's, not the engine's
(`crates/jammi-ai/src/pipeline/recompute.rs`).

### 2.5 Encoders (`jammi-encoders`)

- **`AnyEncoder`** — `crates/jammi-encoders/src/any.rs` (the `AnyEncoder` enum): a
  hand-written closed enum `{ Bert, DistilBert, ModernBert, ClipText, OpenClipVision,
  Htsat }`, **not a trait**. Forwards each method to the active variant. The compiler
  forces a match arm in every method — that is the point of the closed enum (no
  trait-object overhead). `Htsat` carries a `Box<HtsatAudio>`
  (`clippy::large_enum_variant`): the audio tower is roughly four times the next-largest
  variant, and an unboxed payload would make *every* `AnyEncoder` — a small BERT one
  included — carry that footprint. **All six variants are first-class:**
  `trainable_params`, `named_trainable_weights`, `set_training`, `load_weights`,
  `dropout_positions`/`restore_dropout_positions` are real on every one, so any of the
  six can be LoRA-fine-tuned and resumed.
- **`Modality` / `EncoderInput` / `OwnedEncoderInput`** —
  `crates/jammi-encoders/src/any.rs` (`Modality`, `EncoderInput`): `Modality` is
  `{ Text, Image, Audio }`; `EncoderInput` is the borrowed batch vocabulary
  (`Text { input_ids, attention_mask }`, `Image { pixel_values }`,
  `Audio { input_features, is_longer }`) that `AnyEncoder::forward_input` dispatches on,
  and `OwnedEncoderInput` is its owning twin for a caller that must materialise a batch
  (borrow it back with `as_input`). Feeding an input whose modality is not the encoder's
  own is a typed refusal, never a reshape. `forward(input_ids, mask)` remains the text
  convenience and is exactly `forward_input(EncoderInput::Text { .. })`.
- **What is defined per variant, and what refuses** — `hidden_size` and `dtype` are
  total (every variant produces a pooled embedding and reads its dtype off a real
  backbone weight). `forward_hidden` and `max_seq_length` are token-sequence properties:
  the BERT family answers, `ClipText` answers `max_seq_length` with the OpenCLIP
  `context_length` and refuses `forward_hidden`, and the two media towers refuse both.
  `max_seq_length` returns `Result` for exactly that reason — every plausible filler
  (`0`, `usize::MAX`, a patch count) would flow into a caller's `min()` as a confidently
  wrong sequence bound. `image_size`/`preprocess_mean`/`preprocess_std` are vision-only
  and `num_mel_bins` audio-only, each a typed refusal elsewhere.
- **`AnyEncoder::probe_input`** — `crates/jammi-encoders/src/any.rs` (the
  `probe_input` fn): the smallest shape-VALID batch for the variant's own geometry
  (text `1 × TEXT_PROBE_TOKENS` ids; image `1 × 3 × image_size²`; audio
  `1 × CLAP_FUSION_CHANNELS × AUDIO_PROBE_TIME_FRAMES × num_mel_bins` with
  `is_longer = [true]`,
  the branch that exercises the AFF fusion path as well as the plain patch-conv). Read
  off the built tower, never a caller-side table, and at the tower's OWN `dtype()` —
  the LoRA/linear seam casts its input to the weight dtype but the media front ends do
  not (candle's `conv2d` refuses an F32 batch against an F16 kernel; HTSAT's leading
  `BatchNorm` refuses it in its centring subtraction), so a probe manufactured at a
  hardcoded `F32` only ever worked for an F32 backbone. This is what keeps the
  fine-tune worker's claim-time acceleration-report probe
  (`crates/jammi-ai/src/fine_tune/worker.rs`, `build_acceleration_report_json`) from
  degrading to `probe_forward_failed` on a media tower — an empty `ops` map there is the
  "absence must fail, never read as clean" case.
- **`AnyEncoder::fusible_site_census`** — `crates/jammi-encoders/src/fusible_census.rs`
  (`FusibleSiteCensus`): `{lora_sites_wrapped, layer_norms, gelu_seam_calls_per_forward}`,
  the per-forward call count for each of the three fusible seams, total over the enum (every
  variant answers, a variant with no such seam answers `0` rather than declining). Each
  field pairs with exactly one `jammi_kernels::admission` key
  (`lora_linear_fused`/`layer_norm_fused`/`gelu_erf_fused`), so a fused-kernel profile's
  positive-proof equation (`fused + eager == calls × batches`) reads its `calls` term
  straight off this struct instead of a reader deriving it by hand from a config. **The
  invariant: every count is WALKED off the built structure — never config arithmetic.**
  `lora_sites_wrapped` counts `jammi_lora::MaybeLoraLinear::takes_lora_linear_admission() ==
  true` over the tower's own site traversal — a NARROWER predicate than `is_lora()`:
  `LoraLinear::forward` branches on `FrozenBase::Dense` vs `FrozenBase::Quantized` before it
  ever reaches `admit()`, so a `Lora` site over a `Quantized` base (a QLoRA backbone) is
  adapted (`is_lora() == true`) but takes no `lora_linear_fused` admission decision and is not
  counted here; `layer_norms` counts the house `LayerNorm` instances the built tower actually
  holds (a family that omits one, e.g. ModernBERT's `None` layer-0 pre-norm or an HTSAT stage
  with no `downsample`, contributes what it actually holds, not what `2 × layers` predicts);
  `gelu_seam_calls_per_forward` counts calls to `activations::gelu_erf` per forward (`0` for
  ModernBERT's GeGLU FFN and for both OpenCLIP towers' `quick_gelu`, which have no fused seam
  at all — two different reasons for the same zero, both stated on the field's own doc). Every
  count is per ONE forward at `training == true`: each seam short-circuits before any
  admission decision in eval, so **an eval forward contributes `0` to both sides of the
  equation** rather than counting as "all eager". `FinetuneRunTier::fusible_site_census`
  (`crates/jammi-bench/src/report.rs`) records the census as bench PROVENANCE, never
  IDENTITY — it is a structural property of the build, not a caller premise two legs must
  agree on.
  `ci/scripts/perf/kernel_census.py` keys each GPU-kernel bucket on
  `COALESCE(demangledName, shortName)`
  rather than `shortName` alone — cutlass's `Kernel2<...>` template wrapper gives every bf16
  GEMM tile instantiation the same literal `shortName`, so keying on `shortName` alone
  collapses distinct instantiations into one anonymous row; the demangled-name key is a
  strict, sum-preserving refinement (a bucket can only split, never merge two old buckets
  into fewer new ones). **A census report keyed on `shortName` alone carries collapsed
  `Kernel2`/`magma_sgemmEx_kernel` rows in `by_kernel_and_grid`** — its
  `by_kernel_name` totals and every other top-line number (`gpu_kernel_us_per_step`,
  wall/front/busy per step) are unaffected; only the per-instantiation breakdown is
  coarser.
- **The three cross-modal towers and their LoRA sites** — each tower has its own
  builder (`ClipText::builder`, `OpenClipVisionTransformer::builder`,
  `HtsatAudio::builder`) with the same knobs the BERT family uses
  (`.lora()`, `.backbone_dtype()`, `.adapter()`), so "LoRA the text tower only" is
  expressible: the sibling is built `LoraBuildConfig::frozen()`. Every wrappable linear
  is a `jammi_lora::MaybeLoraLinear`; an unselected site is
  `Frozen(FrozenBase::Dense(..))` whose forward is `Linear::forward` bit-for-bit, so one
  struct serves inference and training.

  | tower | indexed unit → `layer_idx` | site names (= `target_modules` selector = adapter subpath leaf) | unindexed sites |
  |---|---|---|---|
  | CLIP-text (`clip_text.rs`) | `transformer.resblocks.{n}` → `Some(n)` | `in_proj` (fused QKV, ONE site over the flat `attn.in_proj_weight`/`in_proj_bias`), `out_proj`, `c_fc`, `c_proj` | none (`text_projection` is a bare tensor) |
  | OpenCLIP-vision (`open_clip_vision.rs`) | `transformer.resblocks.{n}` → `Some(n)` | the same four | none (`proj` bare, `conv1` is a conv) |
  | HTSAT-CLAP audio (`htsat_audio.rs`) | `layers.{s}.blocks.{b}` → `Some(s)` (the STAGE) | `query`, `key`, `value`, `attention_output`, `intermediate_dense`, `output_dense` | `reduction` (`layers.{s}.downsample`, `Some(s)`, bias-free); `linear1`, `linear2` (`audio_projection`, `None`) |

  `"all-linear"` selects every site on any tower. A selector that matches nothing (e.g.
  `q_proj` on CLIP) reaches the existing zero-trainable refusal, whose message names the
  selected tower's real site names.
- **`lora_site.rs` WRAPS, the tower loader RESOLVES** —
  `crates/jammi-encoders/src/lora_site.rs` (the `LoraSite` struct), crate-private: it
  holds the trainable `VarBuilder` scoped to the indexed unit, the unit's `layer_idx`,
  the `LoraBuildConfig` and the `VarMap`, and its `wrap` runs
  `should_apply_lora` → `effective_rank` → `LoraLinear::new_with_base`, else
  `MaybeLoraLinear::Frozen(base)`. Base-tensor *resolution* stays with each tower's
  loader (which alone knows the checkpoint layout — CLIP's flat `in_proj_weight` is not
  addressable by a generic helper). The selector name, the adapter subpath and the base
  locator are therefore three independent axes.
- **`open_clip_block.rs` is ONE block, TWO namespaces** —
  `crates/jammi-encoders/src/open_clip_block.rs` (the `ResidualAttentionBlock` struct),
  crate-private: the text and vision towers load the identical
  `LN → fused-QKV MHSA → residual → LN → QuickGelu MLP → residual` architecture under
  the identical checkpoint path, so they share one implementation parameterised by three
  arguments — MLP width, `Option<causal mask>`, and the **adapter key root**. That third
  argument is load-bearing: `jammi-ai` holds ONE `VarMap` per run, candle's
  `VarBuilder::get` returns the already-registered `Var` for a name it has seen, so two
  towers emitting the same key would silently alias one another's `Var`s (half the
  trainable parameters, one gradient stream feeding two towers, an exported "vision"
  adapter that is literally the text weights). The fix is the checkpoint's own
  namespace: vision keys are `visual.resblocks.{n}.{site}`, text keys are
  `resblocks.{n}.{site}`.
- **Mask sentinels are built once at load, in the backbone dtype** — a mask is
  constructed at load in the frozen backbone's dtype, so no forward ever casts one.
  CLIP-text's causal mask uses that dtype's own most-negative FINITE value
  (`crates/jammi-encoders/src/clip_text.rs`, the `dtype_min_sentinel` fn: `f32::MIN`,
  `half::f16::MIN`, `half::bf16::MIN`) — at F32 that is byte-for-byte the tensor it has
  always been, and a non-floating dtype is refused rather than coerced. HTSAT's
  shift-window mask keeps `-100.0` and must NOT be "improved" to a dtype minimum: it is
  HF `ClapAudioLayer`'s own value and `exp(-100) ≈ 3.7e-44` is a denormal in F32, i.e.
  output-affecting. `-100.0` is exact in F32/F16/BF16 alike, and at F32 the `to_dtype`
  is candle's same-dtype early return (`self.clone()`).
- **Every fusible activation goes through the house seam** — a tower never calls
  `Tensor::gelu_erf()` directly. HTSAT's two GELU-erf sites (each Swin block's MLP in
  `SwinBlock::forward`, and the projection head's `"gelu"` arm in
  `ClapAudioProjection::forward_unnormalized_with_training`) both route through
  `crate::activations::gelu_erf(x, training)` — the same seam `BertIntermediate::forward`
  and `DistilBertFfn::forward` use, and the reason `gelu_erf_fused` is reachable on this
  tower with no new kernel work. The seam's contract carries over unchanged: `training ==
  false` is the unchanged eager call byte for byte, so eval bytes and every golden-parity /
  bits-snapshot row taken in eval are what they were before the seam existed, while
  `training == true` makes fused-vs-eager a COUNTED admission decision on tensor state
  (dtype, contiguity, device, non-emptiness), never on model identity. The `training` flag
  is a call-chain PARAMETER sourced from `HtsatAudio::set_training`'s single stored flag and
  threaded to both sites, never a per-sub-struct stored copy — a stored copy is exactly how
  a seam ends up dispatching on a flag the model's own forward has already moved past. The
  two flag-less public entry points (`HtsatAudioEncoder::forward_spine`,
  `ClapAudioProjection::forward_unnormalized`) are eval conveniences defined as their
  `_with_training(.., false)` twins, for boundary-parity harnesses that hold no flag of
  their own. Both sites report to ONE process-wide `gelu_erf_fused` registry entry, so a
  full-tower forward's counter delta is their SUM — one per Swin block, plus one more when
  `projection_hidden_act == "gelu"`; the tower's own module doc carries that arithmetic and
  the per-site oracles (including the `"relu"` negative control) that pin it.
- **A config field that names a computation is dispatched on or REFUSED, never ignored** —
  five `HtsatAudioConfig` fields name a computation the forward path has hard-coded to one
  value, and `HtsatAudioEncoder::load_with` REFUSES, before any tensor is touched, any
  checkpoint that declares the other one, through both entry points (`HtsatAudio::load` and
  `HtsatAudio::builder().build(..)`): `hidden_act` (must be `"gelu"` — `SwinBlock::forward`
  is unconditionally gelu-erf), `enable_fusion` (must be `true` — `HtsatPatchEmbed::forward`
  always builds and applies the AFF fusion blend), `enable_patch_layer_norm` (must be
  `true` — `HtsatPatchEmbed::forward` always applies its trailing LayerNorm),
  `flatten_patch_embeds` (must be `true` — `HtsatPatchEmbed::forward` always flattens the
  patch grid to `[B, num_patches, C]`), and `qkv_bias` (must be `true` —
  `SwinSelfAttention::load_with` always builds bias-carrying `query`/`key`/`value` linears).
  A checkpoint declaring any of the five otherwise would load and then silently compute
  something its own config does not describe — the worst failure shape available, because
  every downstream number still looks well-formed. Every HF `ClapAudioConfig` this tower has
  ever shipped against (`laion/clap-htsat-fused`, this workspace's `htsat_clap_tiny` fixture)
  already declares all five at the one supported value. Contrast `projection_hidden_act`,
  which IS genuinely dispatched on at forward (`"gelu"` and `"relu"` are both real arms) and
  therefore needs no load-time refusal.
- **The de-facto BERT-family contract** — no Rust trait; the three encoders expose an
  *identical inherent-method surface* (`builder`, `forward`, `forward_hidden`,
  `hidden_size`, `max_seq_length`, `trainable_params`, `named_trainable_weights`,
  `set_training`, `load_weights`, `dropout_positions`, `restore_dropout_positions`).
  Reference table per family in the encoders note; e.g. `Bert::forward`
  (`crates/jammi-encoders/src/bert.rs`), `DistilBert::forward`
  (`crates/jammi-encoders/src/distilbert.rs`), `ModernBert::forward`
  (`crates/jammi-encoders/src/modernbert.rs`).
- **`Pooling` + `pool_and_normalize`** — `crates/jammi-encoders/src/pooling.rs` (the
  `Pooling` enum and `pool_and_normalize` fn). Input `[batch,seq,hidden]` + mask →
  output `[batch,hidden]` with **unit-L2 rows**. Every BERT-family `forward` ends here.
  `Max` pooling uses `-1e30`, never `-inf` (`-inf*0 = NaN`).
- **Internal helpers:** `extended_attention_mask` (`crates/jammi-encoders/src/mask.rs`,
  additive `0.0`/`-10000.0`); dual-path `LayerNorm`
  (`crates/jammi-encoders/src/layer_norm.rs`, fused kernel in eval, gradient-safe
  primitive path in training).

### 2.6 LoRA & fine-tuning (`jammi-lora` + `jammi-ai/fine_tune`)

- **`LoraLinear`** — `crates/jammi-lora/src/lora_linear.rs` (the `LoraLinear` struct).
  Math: `base(x) + scaling * dropout(x @ Aᵀ @ Bᵀ)`; `scaling = use_rslora ?
  alpha/sqrt(rank) : alpha/rank`. **Precision invariant:** base path in F32 then cast
  to backbone dtype; **LoRA A/B stay F32 even when the backbone is BF16/F16.**
  `LoraLinear::new` registers A/B Vars deterministically (`Init::Const(0.0)`) then
  **overwrites storage in place** with seeded draws (`set_var`) — only when the Vars are
  actually registered in a VarMap; the mmaped-load path leaves saved weights untouched.
  *This dual-path behavior is the single subtlest contract in the crate.*
- **`MaybeLoraLinear`** — `crates/jammi-lora/src/wrapper.rs` (the `MaybeLoraLinear`
  enum): closed enum `{ Frozen(FrozenBase), Lora(LoraLinear) }`, decided **once at
  construction** (each encoder's own `LoraSite`-shaped helper — the BERT family's
  in-file one, `crates/jammi-encoders/src/lora_site.rs` for the three cross-modal
  towers). `Frozen.forward` delegates to `FrozenBase::forward`; all other methods are
  no-ops on `Frozen`. `MaybeLoraLinear::base()` reaches the frozen base under either
  arm. This is the static-dispatch LoRA injection point — used by `jammi-encoders` to
  hold attention/MLP linears (e.g. `crates/jammi-encoders/src/bert.rs`).
  **One site, one adapter key:** on the training path (`VarMap`-backed `vb`)
  `LoraLinear::new_with_base` snapshots — *before* registering anything — whether the
  `VarMap` already holds `{prefix}.lora_a`/`.lora_b`, and a pre-existing key is a typed
  `LoraError::Config` refusal rather than a silent alias of the first site's `Var`s. That
  is what makes the two-towers-into-one-`VarMap` aliasing §2.5 describes fail loudly. The
  inference path (mmaped-adapter-backed `vb`) registers nothing and is unaffected.
- **`FrozenBase` + `QuantizedLinear`** — `crates/jammi-lora/src/frozen_base.rs`:
  the closed enum naming what a frozen base weight (`MaybeLoraLinear::Frozen`'s
  layer, or `LoraLinear`'s base) is stored as — `Dense(candle_nn::Linear)`
  (every safetensors-loaded path, byte-unchanged) or
  `Quantized(QuantizedLinear)` (a GGUF-quantized weight: `Arc<QTensor>`
  `[out_features, in_features]` plus an optional dense bias). A closed
  `match`, never a trait object, so a third arm added later fails to
  *compile* every consumer that does not handle it. **The uniform F32
  activation rule:** `QuantizedLinear::forward` always casts its input to
  `F32` before the quantized matmul and casts the result back to the input's
  original dtype — the one dtype every backend (CPU, CUDA, Metal) accepts
  without hitting a candle-internal panic (Metal's own quantized-matmul path
  asserts `F32` internally, not a typed error), never a per-device dtype
  table. `FrozenBase::dweight_needed` is the fused `LowRankResidualLinear`
  site's gate for whether `bwd` must compute `dW`: `Dense` routes through
  the existing `frozen_weight_gate` three-way classification unchanged;
  `Quantized` is CONSTANT `false` — a `QTensor` has no `is_variable`
  accessor and is never constructed from a `Var` anywhere in this
  workspace, so the "trainable base" / "ambiguous tracked-non-`Var`" cases
  `frozen_weight_gate` exists to classify for `Dense` have no reachable
  `Quantized` analogue at all.
- **`quant_matmul_grad`** — `crates/jammi-kernels/src/ops/quant_matmul_grad.rs`
  (`QuantMatMulGrad`, a `CustomOp1` wrapping `Arc<candle_core::quantized::
  QTensor>`): the ONLY quantized-weight matmul entry point in the workspace,
  and ALWAYS differentiable — never candle's own `QMatMul`/`apply_op1_no_bwd`.
  `bwd` computes `dx = dy @ W` (gradient wrt the input only — there is no
  gradient wrt a quantized weight; a frozen base is never trained through
  directly, quantized or not, the same `bitsandbytes` `MatMul4Bit` contract)
  and never returns `Ok(None)` for that slot — candle's own
  `Tensor::backward()` drops a `None` gradient SILENTLY, indistinguishable
  from a correctly-computed all-zero gradient, so a `bwd` that returned
  `Ok(None)` here would look successful right up until a real training run
  quietly stopped learning through it. Reached exclusively through
  `super::apply_stateful1` (`BackpropOp::new1` self-prunes for eval — one
  code path serves both regimes, no train/eval branch in this op at all),
  never candle's `apply_op1_no_bwd`. This closes the `QMatMul` half of
  the silent backward-truncation risk for every quantized-weight
  matmul this workspace's own production code loads today (grep-verified: no
  live call site anywhere in the workspace uses `QMatMul::forward` or
  `apply_op1_no_bwd` on a quantized weight); that property is MECHANICALLY
  enforced only within `jammi-kernels/src` (its own forbidden-needle scan,
  `tests/stateful_op_discipline.rs`, walks `jammi-kernels/src` ONLY — not
  `jammi-lora`, `jammi-encoders`, or `jammi-ai`), so nothing structurally
  stops a future call site in one of those other crates from reintroducing
  `QMatMul::forward`/`apply_op1_no_bwd` on a quantized weight — that half of
  the class stays review-enforced, not mechanically closed. The class's
  other entry points (`candle_nn::ops::softmax_last_dim`'s remaining
  eval call site, `candle_nn::ops::sdpa`) are untouched by this op: the first
  is still reachable, the second latent.
- **QLoRA (fine-tuning over a quantized base)** —
  `crates/jammi-ai/src/fine_tune/worker.rs` (`build_encoder_adapters`): a
  GGUF base artifact (`model.gguf` present, `model.safetensors` absent — the
  same FROZEN precedence the resolver applies, §2.7) SELECTS training over
  `FrozenBase::Quantized` sites; there is no separate QLoRA knob, trainer
  mode, or wire field. The base's matmul-site tensors load through the SAME
  `crate::model::backend::gguf::load_gguf_backbone` the inference path uses,
  so a QLoRA fine-tune and an inference load of the same `model.gguf` can
  never silently disagree on which tensors are matmul-site or which dtype
  loaded. LoRA trains its usual low-rank adapters over the frozen quantized
  backbone, never the backbone itself — `FrozenBase::dweight_needed` being
  constant `false` for `Quantized` means the fused LoRA site's `bwd` never
  even attempts a `dW` it structurally could not produce. **Correct by
  construction, not merely tested:** because `quant_matmul_grad`'s `bwd`
  always returns a real, non-`None` gradient rather than silently dropping
  it, a QLoRA training pass cannot silently truncate the gradient reaching
  the adapter the way a `None`-returning or no-bwd quantized matmul would —
  the fail-loud property closes a class of bug that would otherwise look
  like a converging, healthy training run with a permanently-frozen
  adapter.
- **`LoraBuildConfig` + selection helpers** — `crates/jammi-lora/src/config.rs` (the
  `LoraBuildConfig` struct): borrowed-ref, `Copy`, stack-built per call.
  `should_apply_lora` uses **suffix** match (`ends_with`); `effective_rank` uses
  **substring** match (`contains`) — *different semantics, do not conflate*.
  `LoraBuildConfig::frozen()` is the no-LoRA default. Two seeds, split:
  `seed` keys the A/B init draw, `dropout_seed` the mask draw — every tower site
  (`LoraSite::wrap` and the BERT family's own builders) calls
  `LoraLinear::new_with_base_seeded(.., seed, dropout_seed, ..)`; a gang's ranks share
  `seed` and differ in `dropout_seed` (§2.8d), a single-rank run passes one value for
  both.
- **`layers_to_transform` and the UNINDEXED site** — `crates/jammi-lora/src/config.rs`
  (the `should_apply_lora` fn) takes `layer_idx: Option<usize>`, the single authority for
  both halves of the selection. `None` means the site belongs to no numbered repeating
  unit (a CLAP `audio_projection.linear{1,2}` head, say). PEFT's own
  `check_target_module_exists` (`peft/src/peft/tuners/tuners_utils.py`)
  extracts the index with `re.match(r".*?\.[^.]*\.(?P<idx>\d+)\.", key)` — the FIRST
  numbered segment — and sets `target_module_found = False` when there is none, and this
  function follows that rule exactly: `(None, Some(filter))` → `false` (a caller's
  explicit "only these layers" must not silently widen to cover every head site),
  `(None, None)` → fall through to the ordinary selector match. Two consequences worth
  stating: for HTSAT's `layers.{s}.blocks.{b}.…` keys the extracted index is the **stage**
  `s`, not the block `b`; and setting `layers_to_transform` therefore **excludes** every
  unindexed site on that tower.
- **`LoraInitMode`** — `crates/jammi-lora/src/init.rs` (the `LoraInitMode` enum):
  `ZerosB` (default; identity at construction) or `Gaussian`.
- **Persistence** — `AdapterConfig` (`crates/jammi-lora/src/adapter.rs`) +
  `save_adapter`/`load_adapter` (`crates/jammi-lora/src/save_load.rs`):
  `adapter.safetensors` + `adapter_config.json`. `AdapterConfig::from_build` snapshots
  only shape-affecting fields; **run-time-only fields (`lora_dropout`, `init_mode`) are
  deliberately not persisted**. Higher-level `SavedAdapter`
  (`crates/jammi-ai/src/fine_tune/target.rs`) is the on-disk discriminator between
  `ProjectionHead` and `EncoderAdapters` targets.
- **Adapter identity: `model_type` + `tower`, one meaning each** —
  `crates/jammi-lora/src/adapter.rs` (the `AdapterConfig` struct and the `Tower` enum).
  `model_type` is the base ARCHITECTURE id as the shared predicate spells it:
  `bert` | `distilbert` | `modernbert` | `open_clip` | `clap_audio_model`. Three are
  HuggingFace's own values, `clap_audio_model` is HF's id for a CLAP audio config, and
  `open_clip` is this workspace's canonical id for a checkpoint family that ships no
  `model_type` field at all. `tower: Option<Tower>` (`Tower::{Text, Vision, Audio}`,
  serialised `"text"`/`"vision"`/`"audio"`) says WHICH tower of a multi-tower checkpoint
  the adapter installs on, added by `AdapterConfig::with_tower`. It is
  `#[serde(default, skip_serializing_if = "Option::is_none")]`: an adapter with no tower
  emits **no `tower` key at all**, so a single-tower family's `adapter_config.json` is
  byte-identical to what it has always been, and an adapter written without the field
  deserialises unchanged. Adapter/base agreement is checked FAMILY-to-FAMILY, never
  string-to-string (§2.7 / §5).
- **Adapter key layout** — the `named_trainable_weights` keys ARE the adapter
  safetensors keys, per tower:
  - BERT-family: `layer.{n}.{site}.lora_{a,b}`
  - CLIP-text: `resblocks.{n}.{site}.lora_{a,b}`
  - OpenCLIP-vision: `visual.resblocks.{n}.{site}.lora_{a,b}` (the checkpoint's own
    `visual.` namespace — §2.5's "two namespaces")
  - HTSAT: `layers.{s}.blocks.{b}.{site}.lora_{a,b}`,
    `layers.{s}.downsample.reduction.lora_{a,b}`,
    `audio_projection.{linear1,linear2}.lora_{a,b}`

  See §5's "site-name strings are a persistence ABI" gotcha before renaming any of them.
- **`TrainingTarget`** — `crates/jammi-ai/src/fine_tune/target.rs` (the `TrainingTarget`
  enum): `ProjectionHead { head: LoraModel }` or `EncoderAdapters(...)`. Uniform trainer
  surface (`trainable_params`, `set_training`, `named_trainable_weights`, `load_weights`,
  dropout positions, `saved_adapter`).
- **`TrainingFormat::MediaTriplet` — one media shape, keyed by TASK** —
  `crates/jammi-ai/src/fine_tune/data.rs` (the `TrainingFormat` enum): `anchor`,
  `positive`, `negative` columns carrying encoded binary blobs — audio clips
  (WAV/FLAC/MP3/Ogg bytes) or images (PNG/JPEG/… bytes). The modality is **not** carried
  by the variant and is **not** sniffed from the bytes: it is the job's own `ModelTask`
  (`audio_embedding` / `image_embedding`), which the caller already supplies and the
  trainer dispatches its front end on. An encoded WAV and an encoded PNG are both
  `Vec<u8>`, so the column shape genuinely cannot tell them apart and the declared task is
  the authority rather than a byte-header guess. The trainer decodes and forwards the
  three groups as **one joined batch** and then splits it
  (`crates/jammi-ai/src/fine_tune/trainer.rs`, `encode_media_groups`) — the same shape the
  text path already used. A text task over binary columns, or a media task with no binary
  triplet, is the existing typed schema error, with a message naming the expected columns
  per task; `extract_string_column` (`crates/jammi-ai/src/fine_tune/worker.rs`) refuses a
  binary-family column outright, and refuses a cast that would introduce nulls, rather
  than letting the cast fallback fabricate an empty string for a row it could not read.
- **Serving installs a tower adapter, and identity stays single-valued** —
  `crates/jammi-ai/src/model/backend/candle.rs` (`CandleBackend::load`). With an
  encoder adapter present, the OpenCLIP and CLAP arms build through the §2.5 builders with
  `.lora().backbone_dtype(encoder_backbone_dtype).adapter(..)`, exactly as the BERT arms
  already did; `adapter_cfg.tower` picks which tower receives the LoRA and the sibling is
  built frozen. **Both** OpenCLIP towers build at the adapter's own
  `encoder_backbone_dtype`: a fine-tuned model's backbone precision is its adapter's, that
  precision is folded into `ModelIdentity`, and a checkpoint cannot honestly report two.
  With NO adapter both towers keep their root-`vb`-at-`compute_dtype` construction, so an
  unadapted base's served bytes are unchanged. An OpenCLIP adapter that names no tower is
  a typed refusal — a checkpoint with two towers has no defensible default, and guessing
  installs the weights on the wrong one.
- **`FineTuneConfig::validate()`** — `crates/jammi-wire/src/fine_tune.rs`
  (`FineTuneConfig::validate`): the gate (rank>0, alpha>0, dropout∈[0,1), pinball
  ascending levels, …). `seed` defaults to `DEFAULT_FINE_TUNE_SEED = 42` — a constant,
  never entropy.
- **`JobWorker` / `TrainingJob`** — `crates/jammi-ai/src/fine_tune/worker.rs` (the
  `JobWorker` struct: it dispatches every compiled job
  kind, training and compute alike, not only the three training kinds),
  `crates/jammi-ai/src/fine_tune/training_job.rs` (the `TrainingJob` struct): the claim
  loop that owns a training kind's lifecycle and the poll/wait handle a training verb
  returns. See §3.5 for the full submit → claim → train → finalize path and
  §2.7/`docs/guide/src/operability.md` for the `jobs`/`instances`/`workers` schema
  (migration 029) and lease keeper every job kind shares.

### 2.6a Fused training kernels (`jammi-kernels`)

Companion guides: `docs/maintainer/cuda-kernel-guide.md` (writing and proving a
kernel: roofline method, the oracle rules, the benchmarking protocol),
`docs/maintainer/fine-tune-performance-guide.md` (the fine-tune performance track
end to end: the tape-tax diagnosis, each lever's design and measured effect, bf16
rounding placement, why a single-step bf16 gradient cosine cannot judge fidelity, and the
measurement checklists),
and `docs/maintainer/pod-build-guide.md` (the ops manual for the pod build
substrate every lever above was measured on: seed/clone/push-stamp/timing-lock
procedures, exact commands and failure recovery, and the tree-state invariant
catalogue `test_pod_substrate.sh` pins).

**The model.** `crates/jammi-kernels` is a leaf crate: `candle-core`/`candle-nn`
(+ `half`, `libm`, `thiserror`, `tracing`) only, no `jammi-*` dependency, names no
consumer (family L). Every fused op implements `KernelOp`
(`crates/jammi-kernels/src/ops/mod.rs`, the `KernelOp` trait) — a SEALED
supertrait (`Copy + Send + Sync + 'static` + a crate-private `Sealed` marker, so
no downstream crate can implement it for its own type) enforced structurally at
the only sanctioned call points, `apply1`/`apply2`/`apply3`
(`crates/jammi-kernels/src/ops/mod.rs`): a new op that forgets the `Sealed` impl
fails to *compile* the moment anything tries to run it, rather than shipping
unconstrained. `Copy` proves no owned interior-mutable/heap field (no per-instance
cache-in-`fwd`-for-`bwd` state) — it does not, and cannot, prove the absence of a
module-level `static` (a `&'static` reference is itself `Copy`); that class of
statefulness is a review concern, stated in the module doc rather than overclaimed.

- **CPU is real, one call path, the gradcheck substrate.** Every op's CPU
  `fwd`/`bwd` is the actual numeric implementation (not a stub gated out until
  CUDA lands), so `Tensor::backward()` on a CPU tensor drives the exact kernel
  code CUDA will also run — each op's `tests/*_oracles.rs` gradchecks `bwd`
  directly against central finite differences (`gradcheck_*` fns) over this one
  path, rather than needing a separate numeric-gradient shim.
- **CUDA is feature-gated and early-returns.** The crate's `cuda` feature forwards
  to `candle-core/cuda`/`candle-nn/cuda` and pulls in `bindgen_cuda` as an
  optional build-dependency (`crates/jammi-kernels/Cargo.toml`). `build.rs`
  checks `CARGO_FEATURE_CUDA` and returns immediately when it is unset — the
  default (laptop/CI) build never shells out to `nvcc` or requires a CUDA
  toolkit. When the feature is on, `build.rs` pins `compute_cap(80)` (`sm_80`,
  Ampere) both via `CUDA_COMPUTE_CAP` (avoiding an `nvidia-smi` probe at
  construction time — the Docker-build-stage-with-no-driver shape) and via an
  explicit override, and compiles `src/cuda/*.cu` to one PTX blob per kernel;
  the driver JIT-forwards that single `sm_80` PTX to 8.6/8.9/9.0 devices at
  first load (no `-use_fast_math`; `--fmad` contraction is accepted within each
  oracle's stated tolerance, not pinned away globally).
- **The ops** (`crates/jammi-kernels/src/ops/`): `LayerNormFused`
  (`layer_norm.rs`) — bias-free LayerNorm fwd+bwd, reduced over the last dim;
  `bwd` recomputes mean/invvar from `x` (candle 0.11 has no save-for-backward
  channel) via an internal `CustomOp3` helper, `dgamma` skippable via
  construction data. `RopeFused` (`rope.rs`) — rotate-half RoPE fwd+bwd as one
  `CustomOp3`, replacing the ~12-op eager chain; `bwd` reuses the same forward
  kernel with `sin` negated (the flash-attn `conjugate=True` precedent).
  `SoftmaxLastDimFused` (`softmax.rs`) — masked softmax-last-dim with the
  additive mask ADD folded in, collapsing the `[broadcast_add, max, sub, exp,
  sum, div]` chain into one `Op::CustomOp2` graph node (the single largest
  retained-tape tensor in ModernBERT's attention). `GegluFused` (`geglu.rs`) —
  gated-GELU as one `CustomOp1` over the whole packed `Wi` output (the split
  happens inside the kernel), replacing `[narrow, narrow, gelu_erf, mul]`.
  `ScaledCastAdd` (`scaled_cast_add.rs`) — the LoRA-site epilogue `base +
  cast(lora * scaling)`, replacing `[mul, cast, add]`; a generic Tensor-API
  primitive, not LoRA-specific by name (family L), whose one real caller today
  is `jammi-lora`'s `LoraLinear::forward`. `LowRankResidualLinear`
  (`low_rank_residual_linear.rs`) — the whole LoRA SITE (base matmul, dropout,
  both LoRA GEMMs, and `ScaledCastAdd`'s own epilogue, reused as an internal
  step) as one `CustomOp3`, called from `jammi-lora`'s `LoraLinear::forward`
  when its own domain holds; `ScaledCastAdd`'s standalone dispatch counters
  stay at zero on a run where this op's are nonzero (see that op's own module
  doc). `AttentionBlockFused` (`attention_block.rs`) — the WHOLE
  RoPE+`QKᵀ`+mask+softmax+`PV` attention chain as one `CustomOp3`, composed
  from this crate's own existing primitives at the storage level (no new
  `.cu` kernel); has no `window` construction data of its own — a
  local-attention caller pre-combines its padding mask with a sliding-window
  band into ONE additive `mask` argument before calling it.

**Admission (`crates/jammi-kernels/src/admission.rs`): validate-and-fall-back
with typed refusals.** No op decides fusion *policy* in this module — a call
site's own domain check (dtype/shape/contiguity/device capability) reports its
outcome through the shared mechanism:

- **`device_is_supported(d)`** — `true` for CPU always, CUDA only when *this
  build* compiled the `cuda` feature (`cfg!(feature = "cuda")`, a compile-time
  fold); Metal is refused unconditionally (no `metal_fwd` exists, and candle's
  default `metal_fwd` errors rather than falling back — refusing before the
  tensor reaches `apply2`/`apply3` is what keeps the fallback clean). Also
  gates on `MIN_CUDA_COMPUTE_CAP = (8, 0)` via `ComputeCapability::meets_minimum`
  / `probe_cuda_compute_capability` (bf16 tensor cores need Ampere+).
- **Per-op dispatch counters, two mechanisms, additive.** LayerNorm,
  RoPE, softmax and GeGLU each hand-declare their own `pub(crate) static
  X_DISPATCH_COUNTERS: DispatchCounters`, live and tested. Every
  other op (starting with the LoRA epilogue) calls
  `counters_for("its_op_name")` (`admission.rs`), a process-wide, op-keyed
  `HashMap<&'static str, &'static DispatchCounters>` that creates-and-leaks a
  fresh `DispatchCounters` the first time an op name is seen and hands back the
  same `&'static` on every later call — no new hand-declared static needed.
  `DispatchCounters::snapshot()` returns a `DispatchSnapshot { fused, eager }`
  (`Relaxed` atomics).
- **`warn_fallback_once(op, predicate)`** — a `tracing::warn!` emitted at most
  once per process per `(op, predicate)` pair, so a fallback-heavy run does not
  spam.
- **`admit(mode, op, predicate_name, predicate_holds, counters)`** is the single
  entry point: it always records the outcome, then on a failed predicate either
  logs-once-and-falls-back (`AdmissionMode::Fallback`, the default) or returns
  `KernelError::StrictModeFallback` (`AdmissionMode::Strict`) — never a silent
  wrong number. `KernelError` (`crates/jammi-kernels/src/error.rs`) also carries
  a second, unrelated variant for the SAME "typed refusal, never a confident
  wrong number" doctrine at a different point in the pipeline:
  `KernelError::InvalidScale` — an op's own construction-time domain check
  (`SoftmaxLastDimFused::with_scale`, below), which runs before `apply2`/`admit`
  ever see the op, not a failed admission predicate.
- **`JAMMI_KERNELS_STRICT`** — `admission_mode()` reads this env var once per
  process (`OnceLock`); its presence (any value) selects `Strict`. ONE env var
  governs every fused op in every crate that calls `admit`, rather than one per
  op/crate — moved here from `jammi-encoders::layer_norm` (which still
  re-exports `device_is_supported`/`admission_mode` under its old path for
  source compatibility) specifically so `jammi-lora`, which has no dependency
  on `jammi-encoders`, can read the identical switch.
- **The bench report's `*_dispatches` fields are the positive proof.**
  `crates/jammi-bench/src/report.rs` carries `{ln,rope,softmax,geglu,
  lora_epilogue}_{fused,eager}_dispatches` — each pair is a snapshot diff
  (before/after the run) of that op's `DispatchCounters`. A step-time win alone
  cannot distinguish "the fused kernel ran" from "the fused path silently fell
  back and eager was just fast"; `fused > 0 && eager == 0` on every row is what
  "the fused path actually ran" looks like, and is what a `JAMMI_KERNELS_STRICT`
  bench run is required to show (a fallback there is a hard error, not a quiet
  eager number wearing a fused label).

**Training-only gate, eval bit-identity.** Every fused op's call site gates on
`self.training` (or the crate-level `training: bool`), never merely on the
domain check passing: eval/serving *always* runs the pre-existing eager
composition, unconditionally — the fused arm is a NEW branch added for
`(bias.is_none(), training == true)`
(LayerNorm; `crates/jammi-encoders/src/layer_norm.rs`) or `self.training`
(RoPE, softmax, GeGLU in
`crates/jammi-encoders/src/modernbert.rs`; the LoRA epilogue in
`crates/jammi-lora/src/lora_linear.rs::LoraLinear::forward`), never a
rewrite of the existing eval path. Each site's own
`eval_mode_*_is_bit_identical_regardless_of_fused_eligibility`-style test pins
this: eval's output values are byte-for-byte unchanged by the fused kernel's
existence. Outside its own domain, the training arm falls back to the *same*
eager function eval uses, so a domain miss and eval-mode are one code path, not
two independently-maintained ones.

**The eval doctrine: parity/golden lanes.** `jammi-encoders` carries two
feature-gated oracle suites — `tests/parity.rs`
(`#![cfg(feature = "parity-test")]`) and `tests/golden_parity.rs`
(`#![cfg(feature = "golden-parity")]`) — and `jammi-kernels/tests/cuda_parity.rs`
is `required-features = ["live-gpu-tests"]`: it is compiled only where CUDA
device 0 exists, and fails naming the device when it cannot be opened.
`--features golden-parity` runs in CI's hermetic `test` job
(`.github/workflows/ci.yml`): its oracle is a committed PyTorch dump
(`cookbook/fixtures/htsat_clap_tiny/goldens.safetensors`, a tracked binary),
never a network call or a torch install. `parity-test` needs a PyTorch
environment and `live-gpu-tests` a GPU, which no hosted runner has: CI lints
the `live-gpu-tests` surfaces (the `flash-attn-compile` job), and they run on
RunPod through `ci/scripts/runpod_gpu_prove.sh` (`gpu-prove.yml`) or a pod
session (`ci/scripts/gpu-dev.sh`).

**Numerics doctrine: reproduce-the-reference rounding decisions, not
"whatever's convenient."** Each op's bf16 rounding order is a researched,
disclosed choice against a named upstream reference, not this crate's own
"accumulate in f32, round once" default:

- **Softmax's bf16 mask-add** (`ops/softmax.rs`) adds the (bf16-typed) mask in
  bf16 *before* the f32 softmax — matching `candle_nn::ops::softmax`'s own
  bf16-native `broadcast_add` and, per primary-source research against the
  upstream HuggingFace ModernBERT reference (`modeling_modernbert.py` +
  `masking_utils`), that reference's own eager mask path too. The one
  deliberate divergence is the FULLY-masked row: `FullyMaskedPolicy` (see
  below) can force all-zero output instead of the `NaN`/uniform-distribution
  outputs `candle_nn::ops::softmax` produces there — following PyTorch's
  `_safe_softmax` (`aten/src/ATen/native/transformers/attention.cpp`) and
  FlashAttention-2's online-softmax convention (`softmax.h`), both of which
  force zero rather than propagate `NaN`.
- **GeGLU's round-before-multiply** (`ops/geglu.rs`): on bf16, the activation is
  rounded to bf16 *before* the multiply (two rounding points), matching HF's
  `kernels-community` `gelu_and_mul` CUDA kernel (`activation_kernels.cu`,
  which casts to the storage dtype inside `gelu_kernel` before the separate
  multiply step) rather than accumulating the whole `gelu(gate) * up` in f32
  and rounding once. The backward derivation cites ATen's `gelu_backward`
  (erf-mode `kAlpha`/`kBeta` constants) directly.
- **The LoRA epilogue's round-before-add** (`ops/scaled_cast_add.rs`): the
  scaled delta is rounded to `base`'s dtype, then added and rounded once more —
  matching PEFT's reference (`peft/tuners/lora/layer.py`,
  `Linear.forward`: the delta casts down to the base result's dtype *before*
  the add), the opposite rounding-order choice from the "f32-accumulate, round
  once" convention the f32-internal ops otherwise follow (`ops/layer_norm.rs`,
  `ops/rope.rs` — see the CUDA kernel guide's §3.10 regime table), made
  explicitly because here the thing being matched itself rounds twice.
- **Policy is construction data, never a runtime predicate re-derived from
  tensor state at call time inside the op.** `FullyMaskedPolicy`
  (`SoftmaxLastDimFused::fully_masked`), `GeluVariant`
  (`GegluFused`'s variant field — `Tanh` is a typed refusal via
  `check_variant`, not an unimplemented path), and `dgamma_needed`
  (`LayerNormFused`, frozen in at construction from the *call site's*
  `weight.is_variable()`, re-evaluated per call but never inspected by the op
  itself) all follow this rule: the op stays exactly as stateless as every
  other `KernelOp` (the family's `Copy` bound — see `ops`'s module doc's
  `Copy` discussion), and a caller whose masking/
  activation convention does not match a policy's premise simply never
  requests it, rather than the op silently guessing.
- **`SoftmaxLastDimFused::scale` — construction data with a numeric domain,
  not an enum policy, so it gets its own doctrine.** Folds `1/sqrt(head_dim)`
  into the fused softmax op (`scale * scores + mask`, applied strictly before
  the mask add — see `ops/softmax.rs`'s module doc's "scale semantics"
  section), so ModernBERT's training arm retains no separate `Op::Affine`
  node per layer. The field is PRIVATE (unlike `fully_masked`, whose
  `FullyMaskedPolicy` has no invalid inhabitant): the only way to set it is
  `SoftmaxLastDimFused::with_scale(scale: f32) -> Result<Self, KernelError>`,
  which refuses non-finite or non-positive `scale` (`KernelError::InvalidScale`
  — see above), and the only way to read it back is the `scale()` accessor.
  Default `1.0`, an exact no-op at every dtype. `jammi-encoders`'
  `softmax_admission_predicate` gains a `scale_finite_positive` clause so a
  bad scale becomes a counted eager fallback at the call site (Fallback mode)
  or `KernelError::StrictModeFallback` (Strict mode), never a `with_scale`
  refusal surfacing from inside the training arm.
- **The relative-with-floor bf16 metric.** Every bf16 oracle bounds divergence
  as `|a - b| <= REL_TOL * max(|a|, |b|) + ABS_FLOOR` (each op's own
  `bf16_close`/equivalent, e.g. `tests/geglu_oracles.rs`), never bit-exact
  equality and never a bare absolute or relative bound alone: a pure relative
  bound cannot describe a divergence where either side rounds to exact bf16
  zero, and a pure absolute bound is meaningless across bf16's wide dynamic
  range. Combinations proven bit-exact against eager (documented exception,
  not the norm) are stated as such per op (e.g. the LoRA epilogue's
  `(F32,F32)`/`(BF16,F32)` pair — see `ops/scaled_cast_add.rs`'s module doc).

**How to run the A/B.**
`ci/scripts/gpu-dev.sh run <session> bash ci/scripts/perf/finetune_ab.sh`
(or directly over ssh once the checkout is on the pod) —
never a CI job (no GPU on the CI image). It sweeps `{b8 s128, b8 s512, b16
s128} x {dropout 0, dropout 0.05}` across jammi-eager / jammi-fused
(`JAMMI_KERNELS_STRICT=1`) / torch-eager / torch-sdpa legs, emitting one
table (s/step, triplets/s, peak VRAM, the fused dispatch counters, the ratio
vs torch-sdpa, PASS/FAIL/INDETERMINATE against the throughput bar) — see the
script's own header for the full env-var surface (`MODEL_DIR`,
`AB_STEPS`/`AB_WARMUP`, `AB_DRY_RUN`, …). **One binary, no ref-switching:**
every leg — jammi-eager INCLUDED — runs off the SAME tip binary, built ONCE
at the start (`build_binary()`, `--features cuda,jammi-encoders/flash-attn`).
jammi-eager is the tip binary with every fused op forced eager via
`JAMMI_KERNELS_DISABLE=$JAMMI_EAGER_DISABLE_OP_KEYS` (TEN op keys, including
`mem_efficient_attention`, a live per-layer
`admit_cascade`/once-per-forward `op_disabled` site, and
`gelu_erf_fused`, a live standalone `admit` site in
`crate::activations::gelu_erf` —
`ci/scripts/perf/test_finetune_ab_disable_op_keys.py` sweeps the set
mechanically against the real call graph, so an eleventh key cannot be
missed) under `JAMMI_KERNELS_STRICT=1` (disable wins over Strict) plus
`--expect-kernels-disabled` as a negative control — never "the pre-fusion
commit", and never a second build. Both `jammi-fused` legs ALSO pass
`--expect-kernels-disabled ""` — an empty expectation, hard-failing on
any ambient `JAMMI_KERNELS_DISABLE` leaking into the process. **Order-balanced
bar legs:** the two legs the throughput bar gates on (jammi-fused,
torch-sdpa) each run TWICE per config in a fixed A,B,B,A interleaving
(mirrors `gpu_inference_ab.sh`'s own documented drift rationale), gated by a
`TWO_RUN_PROTOCOL_MARKER` file the script writes before any leg runs —
when present, `ab_merge.py` requires all four bar legs and refuses
(`INVALID`) a genuinely MISSING one, rather than silently degrading to the
single-pair estimator an absent marker (an older `raw_dir`) selects.
`ab_merge.py` computes the MIN of the two resulting pair ratios (the
estimator least favourable to jammi) as the bar ratio, reports
`INDETERMINATE` — never PASS/FAIL — when the two pair ratios disagree too
much relative to the 0.9 bar, and separately cross-checks `jammi-fused` vs
`jammi-fused-2` (and the torch-sdpa pair) for premise drift ACROSS the two
runs, independent of the same-run premise checks each pair already
gets.

### 2.6b The training-set loader: committed order, the session memory pool, and the residency bound (`jammi-db` + `jammi-ai/fine_tune`)

A tabular fine-tune's rows are a **producer output**, not a run's private scan (§3.5
narrates the whole submit → claim → train → finalize path; this section is the
data-plane primitive underneath the "Train" step). `materialize_training_set`
(`crates/jammi-db/src/store/mod.rs`) commits the projected columns once into an
immutable `TrainingSet` result table; every reader re-applies the SAME committed order,
and a session reads it either eagerly (collected into memory) or through a per-rank,
residency-bounded stream — which arm a run takes is a single predicate, stated below.

**The committed order: one key list, two renderers, declared at both registration
paths.** `training_set_sort_keys` (`crates/jammi-db/src/store/mod.rs`) is the ONE
source — every projected column, ascending, NULLs first, in declared order — that both
`training_set_order_by` (`crates/jammi-db/src/store/mod.rs`, the SQL `ORDER BY` clause
a reader re-applies) and `training_set_file_sort_order`
(`crates/jammi-db/src/store/mod.rs`, the DataFusion `ListingOptions::with_file_sort_order`
form a provider DECLARES) render from — a reader that hand-wrote either form independently
could silently disagree with the producer's own commitment. `bind_result_table`
(`crates/jammi-db/src/store/mod.rs`) passes the declared order to `register_table`
for every single-fragment `TrainingSet` row, on BOTH registration paths: fresh
materialization (inside `BuildingTable::finish`) and crash recovery
(`load_existing_tables`, on a session that never saw the write) — so a read-back query
plans no `SortExec` regardless of which path bound the table:
`a_training_sets_registration_declares_its_order_so_the_read_back_plans_no_sort`
(`crates/jammi-db/tests/it/materialization.rs`). `training_set_registration_sort_order`
(`crates/jammi-db/src/store/mod.rs`) is where that declaration is actually read back
off the table's `.materialization.json` sidecar; it can legitimately fail to declare one —
no sidecar at all (a pre-migration-021 table), the sidecar present but UNREADABLE (an
object-store error, or a body that fails to parse as the manifest JSON), or a sidecar
whose descriptor is not a `TrainingSet` variant — and all
three arms `warn!`, naming the table and the reason, before returning `Ok(None)`:
registration still succeeds (the reader's explicit `ORDER BY` clause still sorts the read
correctly), only the `SortExec`-free plan is lost for that one row, and the fallback is
never silent: `registration_warns_when_a_training_sets_sidecar_is_absent`
(`crates/jammi-db/tests/it/materialization.rs`) and
`registration_warns_when_a_training_sets_sidecar_is_unreadable`
(`crates/jammi-db/tests/it/materialization.rs`). The unreadable arm must not propagate
the read error via `?`: that would make `bind_result_table` return `Err` WITHOUT ever
calling `register_table` — the row would never enter the session's schema, not merely lose
its ordering hint.
**Cost, unconditionally paid:** `read_materialization_manifest` issues one object-store GET
(plus a body read, when the object exists) per `TrainingSet` row, every time
`bind_result_table` runs for that row — never cached — which means `load_existing_tables`
(session startup / crash recovery) pays one such GET for every `TrainingSet` row currently
in `ready` status.

**The session memory pool.** `[engine] memory_limit` — `memory_limit`
(`crates/jammi-db/src/config/mod.rs`) — is the ONE knob every consumer of a session's
memory is bounded by: an ordinary `SortExec`/`SortPreservingMergeExec`, a training-set
stream's chunk reservation, an eager materialization's collected-batch reservation.
`memory_limit_bytes` (`crates/jammi-db/src/config/mod.rs`) is the ONE reader of the
field, parsing `"<n>%"` (1–100, of `total_physical_memory_bytes`,
`crates/jammi-db/src/config/host_memory.rs` — the lower of the host's physical total and
a readable Linux cgroup ceiling), `"<n>GB"`/`"<n>MB"`/`"<n>KB"` (binary units), or `"<n>"`
(bytes); every unparseable form is a typed `JammiError::Config` naming the key. A resolved
value below the 64 MiB `MEMORY_LIMIT_FLOOR_BYTES`
(`crates/jammi-db/src/config/mod.rs`) is refused too — small enough that DataFusion's
own long-lived pool consumers would be refused on the very first non-trivial query, before
the setting ever bounds the workload it exists to bound. `JammiSession::build` resolves
this ONCE at session construction and installs a `GreedyMemoryPool`
(`crates/jammi-db/src/session.rs`) sized to it on the SAME `RuntimeEnvBuilder` chain
the session's `SessionContext` is built from; `memory_pool`
(`crates/jammi-db/src/session.rs`) is how an engine-side consumer (the training-set
stream, the eager reader) registers its own `MemoryConsumer` against that identical bound,
and `sql_stream` (`crates/jammi-db/src/session.rs`) is `sql`'s streamed twin — the same
tenant-scoped plan, returned as a `SendableRecordBatchStream` a caller drains incrementally
rather than collects. Every over-budget grow — a DataFusion operator's own reservation or
an engine-side consumer's — surfaces as the typed `ResourcesExhausted`
(`crates/jammi-db/src/error.rs`, `{ limit_bytes, detail }`) from the same public path
the query or reservation was made on: never a panic, never a silent wait. At the gRPC
edge, `map_engine_error` (`crates/jammi-server/src/grpc/wire.rs`) maps it to
`Code::ResourceExhausted` (`crates/jammi-server/src/grpc/wire.rs`).

**The writer: one partition, no merge — and the deployment rule.** The training set's
own write plans its full-tuple sort through `single_partition_context`
(`crates/jammi-db/src/session.rs`) inside `plan_training_set_rows`
(`crates/jammi-db/src/store/mod.rs`): a `target_partitions = 1` derivation of the
caller's session state, so the write is ONE external sort at ONE output partition, never a
partitioned local-sort-plus-`SortPreservingMergeExec` merge — there is only ever one
partition to recombine, so no merge operator (with its own real pool reservation on top of
every partition's already-buffered sorted run) is ever planned. The residency this pays is
O(one batch) plus DataFusion's own spill reservation for that single sort, never O(the
whole table) — **the deployment rule**, stated where the write is planned: a single
`[engine] batch_size` batch larger than `[engine] memory_limit`
`cannot be sorted` (`crates/jammi-db/src/store/mod.rs`), because no batch-granular
operator (a spilling external sort, a decoded chunk) can ever hold one. Sized correctly
the arithmetic is comfortable — a fixture with ~100 KB rows under a 64 MiB pool needs
`engine.batch_size = 32` (`32 × ~100,000 B ≈ 3.1 MB` per batch) rather than
`EngineConfig::default`'s `8192` (`8192 × ~100,000 B ≈ 800 MB`, far larger than the pool
regardless of partition count or merge shape) — see
`f1_a_table_whose_eager_read_exceeds_the_pool_trains_to_completion_through_the_stream`
(`crates/jammi-ai/tests/it/training_set_stream.rs`) for the executed numbers. The
session's `RuntimeEnv` carries a disk-backed `DiskManager` by default, so a sort whose
in-progress runs exceed the pool spills rather than failing the write. This
single-partition property is asserted at BOTH `target_partitions ∈ {1, 4}` by
`the_writers_single_partition_derivation_plans_one_sort_and_no_merge`
(`crates/jammi-db/tests/it/materialization.rs`).

The property holds for the source universe a training-set WRITE actually registers: a
`ListingTable` (a registered CSV/Parquet source, single-fragment — its file-group count
follows `target_partitions`), a Postgres/MySQL federated source
(`crates/jammi-db/src/source/postgres.rs`, `crates/jammi-db/src/source/mysql.rs`, planned
through `FederationOptimizerRule`, `crates/jammi-db/src/session.rs` — one partition by
construction), and the mutable provider's own `MemTable::try_new`
(`crates/jammi-db/src/store/mutable/provider.rs`, always built `vec![vec![batch]]` —
one partition). The edge this excludes: a hand-built MULTI-partition `MemTable` under
`single_partition_context` plans a `SortPreservingMergeExec` over N per-partition
`SortExec`s that still collapses to ONE output partition — the writer's own
`partition_count` (`crates/jammi-db/src/store/mod.rs`) guard cannot see that shape,
because a `MemTable`'s partition count is fixed at construction and never collapses just
because `target_partitions` changed (unlike a `ListingTable`'s file groups). No production
source registers one: `MemTable::try_new` has exactly one call site in the workspace, the
mutable provider's own scan above — and it is single-partition. See `ts_session`
(`crates/jammi-db/tests/it/materialization.rs`)'s own doc comment on why
`the_writers_single_partition_derivation_plans_one_sort_and_no_merge` is FILE-backed
rather than `MemTable`-backed.

**A pinned/versioned result table is NOT this universe.** A
VERSIONED result table's provider is `build_masked_provider`
(`crates/jammi-db/src/store/mod.rs`): one `ListingTable` per manifest fragment,
combined by `MaskedTableProvider`'s `scan` (`crates/jammi-db/src/store/masked_provider.rs`)
into a `UnionExec` (`crates/jammi-db/src/store/masked_provider.rs`) when there is more
than one fragment — a shape that follows the manifest's OWN fragment count, never
`target_partitions`, and that the writer's single-partition guard above cannot see at all
(a versioned table is never itself re-sorted through `single_partition_context`). This does
not threaten the property today because no training-set source is a pinned/versioned
provider: every training-set `source_sql` this tree builds is `source_relation`
(`crates/jammi-db/src/sql/ident.rs`, `"<source>".public."<table>"`) — a plain
registered-source relation, reached through `materialize_projection_table`
(`crates/jammi-ai/src/fine_tune/training_set.rs`) and `recompute_training_set`
(`crates/jammi-ai/src/pipeline/recompute.rs`) — and a pinned/versioned provider is
read only through `ResultStore::pinned_provider`/`ctx.read_table`, never through a source's
registered SQL relation. A training set built from a versioned table's rows would need to
name that fact explicitly; nothing in this tree does.

**The per-rank stream.** `crates/jammi-ai/src/fine_tune/stream.rs`'s `TrainingSetStream`
reads the SAME committed order the eager path reads (`read_back_sql`,
`crates/jammi-ai/src/fine_tune/training_set.rs`) but never collects the whole read into
a `Vec<RecordBatch>`: a background pump walks the DataFusion stream batch by batch,
decoding ONLY the rows the current step's chunk needs. `RowWindow`
(`crates/jammi-ai/src/fine_tune/stream.rs`) is the `[start, end)` slice a stream serves
— the training prefix `[0, train_count)` or the validation suffix `[train_count, total)`;
`Slice` (`crates/jammi-ai/src/fine_tune/stream.rs`) is which rows WITHIN that window
this stream keeps, `PerRank(PartitionSpec)` for training or `All { batch }` for validation.
`StreamConfig`'s `new` (`crates/jammi-ai/src/fine_tune/stream.rs`) refuses a zero
prefetch depth (typed); production trains at `PRODUCTION_PREFETCH_DEPTH`
(`crates/jammi-ai/src/fine_tune/stream.rs`, `= 2`) — a named constant, the regression
pin for a `prefetch = 2` deadlock, never a literal at the call site:
`StreamConfig::new` (`crates/jammi-ai/src/fine_tune/worker.rs`). `open`
(`crates/jammi-ai/src/fine_tune/stream.rs`) runs ONE bounded-memory pre-pass over its
whole window BEFORE the first training step — a schema check plus, for a numeric target, a
null/NaN aggregate — so a column-level refusal fires before step 0, not after thousands of
rows of training compute; `next_chunk` (`crates/jammi-ai/src/fine_tune/stream.rs`) is
the blocking call the trainer's per-step loop drives.

**Resident vs Streamed: one predicate.** `whole_set_arm`
(`crates/jammi-ai/src/fine_tune/source.rs`) decides: mining (scores every candidate
against the full corpus) and GradCache (treats the whole dataset as one in-batch-negative
batch) both need every row resident before an epoch begins, so a config taking either arm
gets `Resident` (`crates/jammi-ai/src/fine_tune/source.rs`); every other text arm at
`W = 1` gets `TrainingSource::Streamed`. `worker.rs`'s source selection calls this same
`whole_set_arm` (`crates/jammi-ai/src/fine_tune/worker.rs`) that the trainer's own
dispatch refuses a mismatch against, so the two decisions can never come apart. A
`Streamed` source never collects a `Vec<RecordBatch>` for the training set at all — the
worker calls only `training_set::materialize_projection_table`
(`crates/jammi-ai/src/fine_tune/worker.rs`), never `read_back`/
`read_back_with_reservation` — while a `Resident` loader's construction reads back through
`read_back_with_reservation` (`crates/jammi-ai/src/fine_tune/worker.rs`) — defined at
`read_back_with_reservation` (`crates/jammi-ai/src/fine_tune/training_set.rs`) — and
attaches the live
`MemoryReservation` to the loader via `with_reservation`
(`crates/jammi-ai/src/fine_tune/data.rs`) — held for the loader's own lifetime (moved
into whichever half of a later `split`, `crates/jammi-ai/src/fine_tune/data.rs`,
carries it), not checked-then-released, so the pool's `reserved()` genuinely reflects a
Resident job's residency while it trains. A table whose eager collected size exceeds the
pool therefore still COMPLETES when it trains through the stream
(`f1_a_table_whose_eager_read_exceeds_the_pool_trains_to_completion_through_the_stream`,
`crates/jammi-ai/tests/it/training_set_stream.rs`), and the eager collect of that SAME
table under the SAME pool still refuses, naming `training_set_eager`
(`crates/jammi-ai/tests/it/training_set_stream.rs`); a Resident job's held reservation
is pinned by `p_r_a_resident_loader_holds_its_eager_reservation_while_training_runs`
(`crates/jammi-ai/tests/it/training_set_stream.rs`).

**A task-local tenant scope does not cross `tokio::spawn` or a `block_on` from the
blocking pool.** `tenant` (`crates/jammi-ai/src/fine_tune/source.rs`) on `StreamedSet`
captures the job's tenant via `tenant` (`crates/jammi-ai/src/session.rs`) on
`InferenceSession` while `run_spec` (`crates/jammi-ai/src/fine_tune/worker.rs`) is still
executing inside the caller's `with_tenant_scoped` task-local scope; `open_streamed_source`
(`crates/jammi-ai/src/fine_tune/trainer.rs`) drives the stream's own `open` through
`Handle::block_on` from the `spawn_blocking` pool, which starts a FRESH top-level poll on a
different OS thread — it does NOT inherit the async task's task-local (`current`
(`crates/jammi-db/src/tenant_scope.rs`) on `TenantBinding` only ever reads the override
installed on the CURRENT task, falling back to the session's sticky binding otherwise). So
every query `TrainingSetStream::open` issues — the schema/null-NaN pre-pass, the ordered
`read_back_sql` plan, the pump's own planning — re-enters `with_tenant_scoped` explicitly
INSIDE that `block_on`'s own future, never relying on inheritance, covering every nested
`.await` `open` makes. Tests of this behaviour must scope via `with_tenant_scoped` on BOTH the
submit and the wait, matching production's per-request scoping exactly: the session's STICKY
binding (`bind_tenant`/`with_tenant`) masks the bug class, because `current_tenant` on
`TenantBinding` falls back to it whenever no task-local override is installed on the current
task — including the `spawn_blocking` thread `block_on` runs on — so a sticky-bound session's
blocking-thread call would "accidentally" resolve the right tenant even without the
re-entry above.

### 2.7 Model lifecycle (`jammi-ai/model` + `jammi-db/catalog`)

This section covers two distinct lifecycles that share the word "model" but never touch:
**in-memory residency** (`jammi-ai/model` — load/cache/evict, runtime) and **catalog
identity** (`jammi-db/catalog/model_repo` — register/read/delete, the durable row).
Promotion and retirement are **not** an engine concern — see the "What is *not* here"
note below.

**In-memory residency (`jammi-ai/model`)**

- **`ModelBackend`** — `crates/jammi-ai/src/model/backend/mod.rs` (the `ModelBackend`
  trait): `load` (synchronous, blocking, no cache lock held) + `estimate_memory` (cheap,
  side-effect-free; the **admission currency** — under-estimating risks OOM).
  Implementors dispatch behind `BackendType`.
- **`GpuScheduler` / `GpuPermit`** — `crates/jammi-ai/src/concurrency/gpu_scheduler.rs`
  (the `GpuScheduler` and `GpuPermit` types). `GpuScheduler::try_acquire(bytes) ->
  Option<GpuPermit>` is non-blocking CAS on `reserved_memory`; `Drop for GpuPermit`
  releases budget + notifies (RAII). **Production wires
  `GpuScheduler::new_unlimited()`** (called in `InferenceSession::new`,
  `crates/jammi-ai/src/session.rs`), so admission is inert in deployment [§7]. The async
  `GpuScheduler::acquire` and `GpuPriority` are tests-only.
- **`ModelGuard`** — `crates/jammi-ai/src/model/mod.rs` (the `ModelGuard` struct): the
  handle execution holds. Drop decrements `ref_count`; **eviction only removes
  `ref_count==0` entries** (`ModelCache::evict_one`,
  `crates/jammi-ai/src/model/cache.rs`), so a model an executor holds is never evicted out
  from under it.
- **`ModelCache`** — `crates/jammi-ai/src/model/cache.rs` (the `ModelCache` struct): LRU
  + single-flight + ref-counted entries + GPU-permit-gated admission. The `GpuPermit` is
  `Arc`-shared, not moved: `CacheEntry` holds its own clone (`gpu_permit`) and every live
  `ModelGuard` holds another (`_gpu_permit`), so permit lifetime is **not** entry lifetime
  — the reservation is released only when the *last* `Arc<GpuPermit>` clone drops.
  `evict_one` therefore gates real progress on `ref_count == 0` **and**
  `Arc::strong_count(&entry.gpu_permit) == 1` (the entry's own clone is the only one
  left); an entry that is idle by ref-count but still has an outstanding guard-held clone
  is skipped, not removed.
  `ModelCache::preload` is a thin `get_or_load`-then-`drop` warmer taking an *explicit*
  `(source, task, backend_hint)` — it reads no config list itself. Its callers are the
  server's warm-before-ready step (`crates/jammi-server/src/runtime.rs`
  `preload_models`, driven by `[server] preload_models: Vec<PreloadEntry>` — a bare id
  whose task is resolved from the `models` row at the startup edge, or `{ id, task }`;
  `/readyz` is 503 "preloading i/n" and the claim loop is parked at the session's worker
  gate until every entry is cached; a failed entry is `ServerError::Preload`, exit
  non-zero) and the Python `preload_model` verb. The cache key is task-free, so the head
  is chosen from the task the preload names — never guessed.
- **`ModelSource` / `ModelId`** — `crates/jammi-ai/src/model/mod.rs` (`ModelId` and
  `ModelSource`): `HuggingFace(String)` | `Local(PathBuf)`. **`ModelId` = `Display` of the
  source is the entire cache key** — `task` and `backend_hint` are NOT part of it [§5
  gotcha].
- **`ResolvedModel`** — `crates/jammi-ai/src/model/mod.rs` (the `ResolvedModel` struct):
  the frozen "files located, backend chosen, not loaded" struct (resolver → backend
  contract).

**Architecture identity — the ONE chain (`jammi-ai/src/model/arch.rs`)**

Every "what architecture is this checkpoint, and which files hold it" question in the
workspace goes through this module. Before it, the serving loader, the resolver, the
fine-tune worker and the benchmark harness each carried their own copy of the rules, and
the copies disagreed.

- **`EncoderFamily`** — `crates/jammi-ai/src/model/arch.rs` (the `EncoderFamily` enum):
  `{ Bert, DistilBert, ModernBert, OpenClip, ClapAudio }`. `EncoderFamily::from_config`
  classifies a parsed config in a stated order — CLAP first (its structural signal is the
  most specific), then OpenCLIP (`model_cfg` present), then the text families keyed on
  `model_type` (`bert`/`roberta`/`camembert`/`xlm-roberta` → `Bert`; `distilbert`;
  `modernbert`). `EncoderFamily::towers()` names a family's towers for refusal messages;
  `has_tower(Option<Tower>)` answers whether the checkpoint actually has the tower an
  adapter claims.
- **A DECLARED-but-unknown `model_type` is `None`, i.e. a typed refusal.** There is no
  `_ => Bert` arm: a config that merely happens to deserialize as a `BertConfig` (a GPT-2
  config does) would otherwise train and serve a confidently wrong architecture over
  foreign weights and publish an adapter claiming that architecture. Every caller must
  handle `None` as a refusal.
- **An ABSENT `model_type` is a different question with a different answer** —
  `crates/jammi-ai/src/model/arch.rs` (the `UNDECLARED_MODEL_TYPE_FAMILY` constant),
  which is `Bert`, and the ONE owner of that rule; `config_model_type` reports the same
  id for refusal messages. A non-string `model_type` value counts as undeclared. Every
  reader in the workspace has always loaded such a directory as BERT (they are older
  sentence-transformers exports and hand-written bare BERT configs; HF's own
  `PretrainedConfig` always serialises the key), so answering `None` here would make this
  module the *source* of the train/serve divergence it exists to remove — serving would
  keep loading the checkpoint while the fine-tune worker refused the identical bytes. A
  non-BERT checkpoint that omitted the field still cannot mis-load silently: its geometry
  must deserialize as a `BertConfig` and its tensors must carry BERT's names.
- **Two candidate-name lists, two questions** — `WEIGHTS_CANDIDATE_NAMES` (4 names,
  including `model.onnx`) is the IDENTITY list: every file name that can BE a model's
  weights, used for digest/fingerprint slots. `CANDLE_WEIGHTS_CANDIDATE_NAMES` (3, no
  ONNX) is the RESOLUTION list `weights_candidates` walks in the frozen precedence.
  `config_candidates` walks `CONFIG_CANDIDATE_NAMES` (`config.json`,
  `open_clip_config.json`). Hardcoding `config.json` / `model.safetensors` instead of
  walking these lists is exactly how an OpenCLIP checkpoint — whose files are
  `open_clip_config.json` / `open_clip_model.safetensors` — becomes invisible to a
  consumer.
- **Where the config comes from is part of the contract:** the resolver prefers the
  catalog record's stored `config_json` over disk and the fine-tune worker follows the
  same order, so a job and a serve of the same catalog row can never classify the same
  model differently.
- **The fingerprint arms consume this module without changing their BYTES** —
  `crates/jammi-ai/src/model/backend/candle.rs` (the config and weights fingerprint
  arms); a test pins the fingerprint/digest on the tiny fixtures across the extraction.

**GGUF/k-quant weight loading (`jammi-ai/model/resolver.rs` + `model/backend/gguf.rs`)**

- **`WeightsFormat`** — `crates/jammi-ai/src/model/mod.rs` (the `WeightsFormat`
  enum): `Safetensors` | `Onnx` | `Gguf`, the on-disk STORAGE format of a
  resolved model's weight files, carried on `ResolvedModel.weights_format`.
  Orthogonal to `BackendType`: `Gguf` is a weight-storage format only the
  `Candle` backend loads — the resolver's ORT arm only ever looks for
  `model.onnx`, so `(Ort, Gguf)` is structurally unreachable through the
  resolver, not a case this type itself forbids.
- **The `model.gguf` literal-filename contract** —
  `crates/jammi-ai/src/model/resolver.rs` (`GGUF_WEIGHTS_FILENAME`): mirrors
  `model.safetensors`/`model.onnx`'s own convention — the digest-slot
  machinery (`backend::candle::all_candidate_paths`) stats known names only,
  never sniffs by extension, so a quantized checkpoint must be named exactly
  `model.gguf`. **Precedence is FROZEN:** `model.safetensors` (or
  `open_clip_model.safetensors`) wins, byte-for-byte, over `model.gguf`
  — for a local directory, an HF Hub repo, AND a QLoRA
  fine-tune's own base-artifact resolution (§2.6) alike. Only when neither
  is present does `model.gguf` enter the picture. A directory or repo
  carrying some OTHER `*.gguf` filename is a typed refusal naming every such
  file and pointing at the convention, never a silent extension-sniff or a
  fallback to the first `*.gguf` found.
- **`config.json` is required — llama.cpp-style metadata-embedded GGUFs are
  not a supported checkpoint shape.** Architecture (`model_type`) and layer
  count (`num_hidden_layers`/`num_layers`) are read from the model
  directory's `config.json`, the SAME source every safetensors load already
  uses — never from the GGUF file's own key-value metadata (`general.
  architecture`, `block_count`, etc., the convention some GGUF exporters
  embed instead of shipping a sidecar `config.json`). A `model.gguf` with no
  accompanying `config.json`, or one missing the layer-count key, is a typed
  refusal ("GGUF load requires num_hidden_layers (or num_layers) in
  config.json"), never a metadata-sniffing fallback into the GGUF file
  itself.
- **Three supported architectures; everything else is a typed refusal** —
  `crates/jammi-ai/src/model/backend/gguf.rs` (`GgufArchitecture::
  from_model_type`): BERT-family (`bert`/`roberta`/`camembert`/
  `xlm-roberta`), `distilbert`, `modernbert` — the three text towers
  `jammi-encoders`' `FrozenWeightLookup` seam (`crates/jammi-encoders/src/
  frozen_weight_source.rs`) is wired into. OpenCLIP and HF-CLAP checkpoints
  refuse at `CandleBackend::load`'s GGUF branch ("quantized serving not
  supported for this architecture") — GGUF loading is threaded ONLY through
  the three text towers named above.
- **Matmul-site vs. everything else** — a "matmul-site" tensor is a
  weight/bias one of the encoder's own per-layer linear modules routes
  through `FrozenWeightLookup` (six per BERT-family/DistilBERT layer, four
  per ModernBERT layer — bias-free). A matmul-site tensor stored at a
  genuine k-quant dtype loads as `jammi_lora::FrozenBase::Quantized` and
  NEVER gets dequantized at load — it stays resident as an `Arc<QTensor>`
  (§2.6). Every OTHER tensor — embeddings, LayerNorms, classifier/NER heads,
  and a matmul-site tensor that merely happens to be stored densely
  (`F32`/`F16`/`BF16`, not k-quantized) — is dequantized to the model's
  compute dtype at load and densified into a synthesized in-memory
  safetensors file (`GgufBackbone::densified_path`) that every construction
  site which never consults the lookup (embeddings, norms, heads) reads
  exactly the way it reads a real safetensors checkpoint. A malformed
  header, an element count that is not a multiple of its dtype's own block
  size, an unsupported GGML dtype (neither a k-quant `WeightQuantization`
  format nor `F32`/`F16`/`BF16`), or a missing required matmul-site tensor
  is a typed refusal naming the tensor(s) and dtype, never a silent skip or
  a deep candle panic surfacing mid-load.
- **Resolve-time residency estimation** —
  `crates/jammi-ai/src/model/backend/gguf.rs` (`estimate_gguf_residency`):
  parses ONLY the GGUF header (never tensor data — `gguf_file::Content::
  read` reads tensor bytes on demand, per-name, via a separate call) and
  returns a CONSERVATIVE (≥ true residency) byte figure — a matmul-site
  k-quant tensor is costed at its own storage size plus candle's own
  quantized-CUDA-loader row-padding constant (applied unconditionally,
  regardless of the actual target device, so the estimate stays
  conservative on every device); every other tensor is costed as
  dequantized-to-compute-dtype; the single largest per-tensor dequantize
  TRANSIENT (`stored + 4·N + target·N` — candle's own `QTensor::dequantize`
  always produces `F32` first, before any narrowing cast) is added ONCE,
  covering the worst-case peak buffer overlap during load rather than
  summing every tensor's transient. `CandleBackend::estimate_memory` reuses
  this resolver-computed figure verbatim for a GGUF-resolved model rather
  than re-deriving anything from `weights_paths` (a single `model.gguf`
  file's raw byte size is wildly unrepresentative of resident memory).
- **`ModelIdentity.quantization`** —
  `crates/jammi-db/src/store/manifest.rs` (`ModelIdentity::quantization:
  Option<jammi_numerics::WeightQuantization>`): the MODAL `WeightQuantization`
  among a GGUF backbone's matmul-site tensors (ties broken by that type's
  own `Ord`, i.e. GGUF wire-ID order), `None` for every safetensors/ONNX
  load. Folds in alongside `compute_precision`/`content_digest` (`#[serde
  (default, skip_serializing_if = "Option::is_none")]`, preserving every
  pre-existing `DefinitionHash` byte-for-byte — a `None` serialises to no
  key at all, never a present `null`) because a quantized run is
  output-affecting relative to a full-precision run of the same model over
  the same inputs: two such runs must never collide on one materialization
  identity. `LoadedModel::quantization()` (`crates/jammi-ai/src/model/
  mod.rs`) is the read path both `session.rs` and `pipeline/embedding.rs`
  consult.

**Catalog identity — the durable model row (`jammi-db/catalog/model_repo.rs`)**

The catalog model surface is exactly five verbs. `Catalog` owns the SQL;
`InferenceSession` (local_session) and `CatalogService` (gRPC) are thin projections over
them.

- **`register_model`** — `crates/jammi-db/src/catalog/model_repo.rs`
  (`ModelRepo::register_model`). The *only* writer that creates a model row, and it is
  **not exposed on `local_session` or the gRPC surface** — no public verb calls it (the
  cookbook lifecycle chapter pins `fine_tune` as the "only public registration path",
  `cookbook/book/chapters/.../lifecycle.qmd`). Internally it has several engine-side
  callers, not only training: training (`fine_tune` registers the base model at submission
  — `crates/jammi-ai/src/session.rs` — and the fine-tuned model on completion via the
  worker/trainer, `crates/jammi-ai/src/fine_tune/worker.rs`,
  `crates/jammi-ai/src/fine_tune/trainer.rs`), **and the model-load path auto-registers a
  freshly-loaded model** (`crates/jammi-ai/src/model/cache.rs`) plus the context-predictor
  pipeline (`crates/jammi-ai/src/pipeline/context_predictor.rs`). The accurate invariant
  is **no public/client verb registers; every registration is engine-internal**. INSERTs
  `status = 'registered'` literally; `ON CONFLICT(model_id) DO UPDATE` refreshes
  metadata/backend/task but `artifact_path = COALESCE(excluded, existing)` — a re-register
  can *set* but never *clear* a committed served-path; the finalized served path is written
  solely by the lease-guarded `Catalog::finish_job_with_model` CAS, never by a worker's
  `register_model`. PK is tenant-qualified via `model_pk`: global = `"{name}::{version}"`,
  tenant-scoped = `"{t}::{name}::{version}"`.
- **`get_model`** — `crates/jammi-db/src/catalog/model_repo.rs`
  (`ModelRepo::get_model`). Latest version by name, tenant-filtered (`tenant_id = $t OR
  tenant_id IS NULL`, tenant row preferred). **The reference-resolution path** (training
  base model, eval run, serve/load resolver) — it resolves a model *regardless of lifecycle
  status* so a referencing job always binds.
- **`get_model_version`** — `crates/jammi-db/src/catalog/model_repo.rs`
  (`ModelRepo::get_model_version`). Exact `(name, version)`, same tenant predicate.
- **`delete_model`** — `crates/jammi-db/src/catalog/model_repo.rs`
  (`ModelRepo::delete_model`). **Hard delete** (removes the row outright — there is no
  soft-delete/retire). Resolves the row (`get_model_version` if a version is given, else
  `get_model`), then in **one `Serializable` transaction** runs `scan_model_references`
  before the `DELETE`. The five reference edges (the two static `REFERENCE_EDGES` plus the
  three age-gated `jobs` edges) are `result_tables.model_id`, `jobs.output_model_id`,
  `jobs.model_source` (all keyed by model NAME, no FK), `jobs.model_ref`, `eval_runs.model_id`
  (both keyed by catalog PK, FK-backed); a `jobs` row counts only while non-terminal or
  younger than `[jobs] retention_days`. A non-empty scan returns `DeleteOutcome::Referenced`, raising the typed
  `JammiError::ModelReferenced` (`crates/jammi-db/src/error.rs` → gRPC
  `FailedPrecondition`) — the DB FK is deliberately *never* the rejecter, so a reference
  never leaks as an opaque backend error. Delete is **strict tenant-scoped** (`tenant_id =
  $t OR (tenant_id IS NULL AND $t IS NULL)`): a tenant cannot delete a global or a peer's
  row. Absent row → `ModelNotFound` (`crates/jammi-db/src/error.rs` → gRPC `NotFound`)
  unless `if_exists` is set, then a success no-op.
- **`list_models`** — `crates/jammi-db/src/catalog/model_repo.rs`
  (`ModelRepo::list_models`). Every model visible to the tenant (own + global), ordered by
  `created_at`. The list-facing peer of `get_model`.

**Status is `Registered | Loaded` only** — `ModelStatus`
(`crates/jammi-db/src/catalog/status.rs`) has exactly two variants; `from_str` rejects
anything else. A migration normalizes the legacy `'available'` default to `'registered'`
(`crates/jammi-db/src/catalog/schema.rs`). (Several rustdoc comments —
`crates/jammi-db/src/catalog/model_repo.rs`,
`crates/jammi-wire/proto/jammi/v1/catalog.proto` — still list `"failed"` as an example
status; that string parses *only* for the training/result status enums, not `ModelStatus`.
A model row is **never** stamped `'failed'`: a grep of every `models.status` write finds
only the `'registered'` literal in `register_model` and the migration's
normalize-to-`'registered'` — no `'failed'` writer exists.)

**`ModelRecord` vs `ModelDescriptor` — the client projection.** `ModelRecord`
(`crates/jammi-db/src/catalog/model_repo.rs`) is the full row (version counter,
`base_model_id` lineage, `artifact_path`, `config_json`, `created_at`). `ModelDescriptor`
(`crates/jammi-db/src/catalog/model_repo.rs`, `From<&ModelRecord>`) is the **only** shape
that crosses a client boundary — exactly `{model_id, backend, task, status}`. The
server-internal bookkeeping never reaches a client. The gRPC `Model` message
(`crates/jammi-wire/proto/jammi/v1/catalog.proto`) mirrors this projection.

**Session/gRPC verbs (the consumer surface).**
- `crates/jammi-ai/src/local_session.rs`: `Session::list_models`,
  `Session::describe_model` (`get_model` → `ModelDescriptor`), `Session::delete_model`.
  **No `register_model`, no promote, no retire.**
- `CatalogService` (`crates/jammi-server/src/grpc/catalog.rs`): `list_models`,
  `describe_model`, `delete_model`. Cross-transport parity (`remote == embedded`) is the
  cookbook lifecycle chapter's contract.

**What is *not* here — promotion & retirement are governance, not engine surface.** There
is no `promote_model`, no `retire_model`, and no `Retired`/`'retired'` status anywhere in
the engine. The wire schema *reserves* the removed field tags so a peer can never re-bind
them: `crates/jammi-wire/proto/jammi/v1/catalog.proto` reserves the `"promoted"` field (`//
model promotion is not an engine concern`), and
`crates/jammi-wire/proto/jammi/v1/error.proto` reserves the `"model_retired"` tag.
Lifecycle policy (which version is "the serving one", when a model is decommissioned)
belongs to the consumer's own repo; the engine offers only `register` (via training) →
`read` → hard-`delete`. Any guide or doc claiming a model-promotion or retirement verb is
describing a removed surface.

### 2.8 Server edge (`jammi-server`)

- **`ServiceTier` / `TierSet`** — `crates/jammi-server/src/tiers.rs` (the `ServiceTier`
  enum and `TierSet` struct). `Core` (always mounted — session/embedding/inference/
  pipeline + mutable-table/channel/audit + job submission + `GetServerInfo`; durable job
  submission/status live here, not in a tier of their own), `Event`
  (`TriggerService`), `Eval` (`EvalService`) — `OPTIONAL = [Eval, Event]`; there is no
  `train` tier or cargo feature: `services =
  ["train"]` is a startup error naming the unknown tier. `TierSet::resolve` is
  infallible — every tier is core-compiled, so there is nothing to reject; `TierSet::all`
  is `Self::resolve(ServiceTier::OPTIONAL)`. `TierSet::as_wire` is
  **sorted alphabetically** — the `ServerInfo.services` handshake. **Invariant:
  advertised (`as_wire`) == mounted.** Whether a process *runs* the job claim loop it
  accepts is the separate `[worker] enabled` runtime key (§3.5,
  `docs/guide/src/operability.md`), never a tier and never a build feature.
- **`OssServer` / `serve_grpc_chain`** — `crates/jammi-server/src/runtime.rs` (the
  `OssServer` struct and `serve_grpc_chain` fn). Single Tonic chain shared by production
  and tests. Mounts Flight SQL + `CatalogService` + `JobService` always; engine-backed
  services when `engine.is_some()`; tier-gated `Eval`/`Trigger`. `ChainParts::worker`
  spawns the embedded `JobWorker` claim loop
  iff `[worker] enabled`. **`OssServer::new` calls `InferenceSession::open` (not
  `new`)** so the `annotate` UDTF is registered for Flight SQL.
- **Session/tenant boundary** — `crates/jammi-server/src/grpc/session.rs`:
  `SESSION_HEADER`, `SessionStore` (in-process `HashMap<SessionId, Option<TenantId>>`),
  `TenantResolver` (the async resolver trait, `&MetadataMap` → `Result<TenantScope, Status>`),
  `TenantScope` (`Tenant`/`Global`), `SessionIdTenantResolver` (the engine default —
  `jammi-session-id` header → `SessionStore`). The single async tower layer that applies
  the resolved scope to every gRPC service and the Flight SQL provider is
  `TenantResolverLayer`, in `crates/jammi-server/src/tenant_resolver_layer.rs`. **Invariant:
  a request with no/unknown session header runs unscoped (all-tenants — the explicit
  `Global` scope), never an error** — the load-bearing gotcha behind "bind first" [§5].
- **`AdminAuthorizer`** — `crates/jammi-server/src/grpc/catalog.rs`: a SEPARATE,
  narrower seam from `TenantResolver` above — it gates only `CatalogService.
  Reconcile`'s cross-tenant `all = true` admin pass, never an ordinary verb's
  tenant binding. Synchronous (`fn authorize(&self, metadata: &MetadataMap) ->
  Result<(), Status>`, unlike the `async_trait` `TenantResolver`): a local
  metadata check, not an I/O round-trip. `CatalogServer::new`'s 4th parameter,
  threaded from `GrpcChain.admin_authorizer: Option<Arc<dyn AdminAuthorizer>>`
  (`runtime.rs`'s `build_grpc_chain` — the OSS binary's shipped default,
  `None` — and `:1020`'s `assemble_grpc_chain` exhaustive destructure;
  `flight.rs`'s `serve_flight_with_catalog_service` passes `None`). Shipped
  default `None` refuses EVERY `all = true` request with `PERMISSION_DENIED`
  naming `security.md`; `all = false` never consults it. **Gated verb only —
  gRPC-only by construction** (`Reconcile` has no Flight SQL analogue), unlike
  `TenantResolver`'s one-grant-both-transports shape. Test double:
  `tests/it/common/grpc.rs::AllowAllAdmin`.
- **Per-handler helpers** — `crates/jammi-server/src/grpc/wire.rs`: `session_tenant`,
  **`scoped`** (the concurrency-safe per-task-local tenant scope — handlers must use this,
  never sticky `bind_tenant`), `require_nonempty`, `map_engine_error`/`map_trigger_error`.
  **Invariant: faithful errors** — each `Status` carries the full structured detail so the
  client reconstructs the exact variant.

### 2.8a GangService — multi-host gang admission and `HostAdmission`

The coordinator-to-member admission seam for a multi-host training run.
Proto: `crates/jammi-wire/proto/jammi/v1/gang.proto`, `service GangService`
with one bidi RPC, `RunRank(stream RankControl) returns (stream RankEvent)`.
Handler: `crates/jammi-server/src/grpc/gang.rs`, `GangServer::run_rank`.
Mounted beside `PeerServiceServer` on the internal `[server] peer_bind`
listener only (`crates/jammi-server/src/runtime.rs`, `OssServer::bind`) —
never on the public listener, never wrapped by `TenantResolverLayer`; the
public listener answers `UNIMPLEMENTED` for `/jammi.v1.gang.GangService/*`
(`GANG_LISTENER_ALLOWLIST`, `crates/jammi-server/tests/it/tenant_isolation_oracle.rs`).
Both services on that listener carry the `[server.limits] max_message_bytes`
inbound decode cap, the same per-service setter the public chain applies
(§2.8c, "The decode cap on every listener").

**Two observables, split at admission.** BEFORE admission every determinant
is the call's own result — `Err(Status)`: the ONE fixed `FailedPrecondition`
for every job-row determinant (rung 5 below), `Unavailable` for a catalog fault
or a busy slot, `InvalidArgument` for the wire range edges — and no stream
exists. AFTER admission the call has returned `Ok(stream)`, so every later
outcome is delivered IN the stream: `Admitted`, then exactly one
`Aborted{reason}`, or — for a protocol violation on the admitted stream — a
status trailer. An admitted session's end may name its reason (the caller
already holds the job's own coordinates and was admitted on them); a
pre-admission refusal never does.

**The RunRank admission lattice.** A call is decided in this order, each
rung its own status, and NOTHING on this host is touched before the decision
is complete:

1. **Wire range checks** (`gang.rs`, before any row read): `world == 0` →
   `InvalidArgument("world must be greater than zero")`; `rank >= world` →
   `InvalidArgument("rank must be less than world")`.
2. **Ambient admin scope.** `TenantBinding::is_admin_scope()` — gang
   admission refuses ambient admin scope outright, before any row is read.
3. **The job row is the capability: the row predicate.** `Catalog::get_job_for_rank(job_id)`
   (`crates/jammi-db/src/catalog/jobs_repo.rs`, primary-key-only, no tenant
   predicate, never admin scope) returns the row by primary key alone — it
   decides nothing itself, returns `Ok(None)` only when no job with that id
   exists, and carries the row's OWN `tenant_id` (raw text), its
   `training_set_ref`/`training_set_location` pair, its `world_size`
   (`WorldSizeFact`, decoded from the job's `spec` JSON), and its lease
   (`LeaseFact`, decoded from `lease_expires_at`'s raw stored text). Every
   determinant is decided by the CALLER, `GangServer::run_rank`: `status =
   'running'`; `claimed_by = assign.coordinator_instance_id`; `attempts ==
   assign.attempt`; the lease is `LeaseFact::Live` (a `NULL` lease column, or
   one at/before now, is `Dead`, never live-by-default;
   `crates/jammi-db/src/catalog/lease.rs`'s `decode_lease_expires_at`
   PARSES THE RAW TEXT IN RUST, never a SQL-side `col::timestamptz` cast, so
   a malformed value is `LeaseFact::Undecodable` — refused under its own
   `GangRefusalReason::LeaseUndecodable`, the SAME fixed status as every
   other determinant — rather than surfacing as a genuine read fault on one
   backend and a live-lease row fact on the other);
   `WorldSizeFact::Undecodable` is itself a refusal — a ROW FACT, never a
   fault of the read, so it never maps through `admission_catalog_fault`;
   `assign.world != row.world_size` is itself a refusal — the lattice is
   keyed on the ROW's rank count, never the caller's claim (a caller-keyed
   gate would let a `world_size > 1` job admit under a caller-supplied
   `world = 1`, skipping the next rung entirely).
4. **The world>1 conjunct** (`row.world_size > 1` only — a `world_size == 1`
   row reads no tenant value and no pair at all): (a) the training-set
   identity pair is filled on the row (the coordinator's write-once CAS,
   `Catalog::fill_training_set_identity`); the row's `tenant_id` text
   parses (an unparseable value is a row fact, refused); and (b)
   `resolve_training_set_identity` (`gang.rs`) — the resolution site
   refuses ambient admin scope again, explicitly, before calling the
   resolver; then the ONE tenant-pinned lookup, the STRICT
   `Catalog::get_result_table_for_tenant(training_set_location, tenant)`
   (`crates/jammi-db/src/catalog/result_repo.rs`: `tenant_id = $t OR
   (tenant_id IS NULL AND $t IS NULL)`, an explicit tenant argument, no
   admin arm — never the relaxed `get_result_table`, whose `OR tenant_id IS
   NULL` would hand a real tenant every GLOBAL row of the same name) must
   find a row with `status = 'ready'`; then the sidecar verify
   (verify-at-read): `ResultStore::read_materialization_manifest` of the
   row's `parquet_path` must return a manifest whose `artifact` equals
   `training_set_ref` — a sidecar predating the leaf inventory reads as
   ABSENT and refuses; a mismatching or undecodable one refuses; this host's
   store faulting on the read refuses too. The classification
   (`TrainingSetOutcome`: `Verified` / `AdminScopeRefused` / `Unresolved` /
   `NotReady` / `SidecarAbsent` / `DigestMismatch` / `StoreFault`) is kept
   for the `test-hooks` seam and for re-verification; on the wire every
   non-`Verified` arm is rung 5.
5. **Coordinator freshness.** `Catalog::fresh_instance(coordinator_instance_id,
   lease)` requires the named coordinator's `instances` row present and last
   seen within `instance_liveness_margin(lease)` — `2 × lease`. Decoded in
   RUST from `last_seen_at`'s raw stored text
   (`lease::last_seen_at_is_fresh`), never a SQL-side cast: this column is
   ALWAYS an application-clock stamp on either backend
   (`Catalog::upsert_instance` never writes the database clock here), so a
   value that does not parse reads as not-fresh — a row fact, joining the
   same "absent OR stale" class `fresh_instance` already collapses to one
   answer, never a fault; absent or stale otherwise still
   refuses.
6. Every refusal in rungs 2–5 is the SAME status and message —
   `FailedPrecondition("gang admission refused")` — regardless of which
   determinant failed: the listener discloses neither a job's existence, its
   claimant, its attempt, its tenant, nor another tenant's table
   (non-disclosure).
7. **The slot.** Only now the holder CAS
   (`HostAdmission::admit_rank`, below): `Free` admits; a `ClaimProbe` is
   waited on for at most one heartbeat, then admits if freed or refuses;
   `JobRun` or another `Rank` refuse at once —
   `Unavailable("gang admission: this host's job slot is busy")`, transient,
   no assembly budget consumed. Nothing about the job is decided here, so
   the refusal is the same for every holder kind.
8. **Admitted.** `Admitted` is the stream's first event, emitted only after
   the CAS succeeded; the `RankHold` guard moves into the spawned hold loop.

**`HostAdmission` — the host's admission state** (`crates/jammi-ai/src/fine_tune/worker.rs`,
owned by `InferenceSession`, `InferenceSession::host_admission`): three cells
— the shutdown `phase` (`WorkerPhase`: `Running`/`Draining`/`Releasing`, one
`watch`, read by the claim loop's gate and by every held rank; flipped by
`EmbeddedWorker::begin_drain`/`release_and_stop`, by
`InferenceSession::release_job_leases`, and by the server's DRAIN arm for a
worker-less process), the slot `holder` (`Holder`: `Free` / `ClaimProbe` /
`JobRun` / `Rank{job_id, attempt}`, one `watch`, every transition a
`send_if_modified` compare-and-set, no lock across an `.await`), and this
process's `registry` (its `InstanceRegistration`). The claim loop moves the
holder `Free → ClaimProbe` immediately before `claim_next`
(`HostAdmission::probe_claim`; a held slot skips the claim — a peer never
claims while it holds a rank), `ClaimProbe → JobRun` at the hold site once
the claimed job's lease hold is registered (`HostAdmission::job_running`,
never earlier: the claim→hold prologue stays a probe, so a RELEASE landing
inside it still self-releases at zero net attempts), and `→ Free` when the
run returns (the `ClaimGuard`'s drop, on every exit path). A rank's CAS
(`HostAdmission::try_hold_rank`): `Free → Rank{..}`; the SAME job at a
GREATER attempt supersedes a held `Rank` in place (the elder session is then
refuted by the row at its next tick; its guard's drop leaves the successor's
hold alone); an equal attempt (a duplicate assignment), another job's rank,
`JobRun`, or `ClaimProbe` refuse with what was found (`HolderBusy`). An
inline `run_now` and a direct `JobWorker::run_claimed_job` never touch the
holder — they run beside it. RELEASE's abort decision (`EmbeddedWorker::release_and_stop`,
2e) reads the holder KIND, never a count: `JobRun` aborts the loop task now;
`Free`/`ClaimProbe`/`Rank` wait one heartbeat for the cooperative exit (a
rank is never loop work; its own session ends on the phase). The `/metrics`
gauge `jammi_worker_jobs_in_flight` is `1` iff the holder is `JobRun`.

**The hold loop** (`HeldSession::hold`, `gang.rs`, a spawned task owning the
`RankHold` and the inbound stream) has exactly FIVE arms, of which any one
session takes four — the rank body's end and the park bound are exclusive:

- **inbound** — `Cancel` ends the session `Aborted{Cancelled}`; a second
  `Assign` is a protocol violation, a status trailer
  (`InvalidArgument`), never a second admission; every other frame goes
  through `HeldSession::dispatch_round_frame`, the ONE site the round
  protocol is wired at: a round frame (`RoundInbox::is_round_frame` —
  `RoundResult`, `RoundChunk`, `RoundCommit`, `RoundFault`) is delivered to
  the session's `RoundInbox` and the session stays held; a delivery the
  inbox refuses (the session's `MemberLink` is gone) ends the session with a
  `FailedPrecondition` trailer; an empty frame is a protocol violation
  (`InvalidArgument`). The inbox and the link are built at admission over
  the session's own event sender (`gang_rounds::member_link`), before
  `Admitted` is queued. A `world_size > 1` session hands the link to its
  RANK BODY (§2.8e), spawned at admission; a `world_size == 1` session has
  no body and keeps the link for its whole life. A transport error on the
  inbound side is reported to the inbox (`RoundInbox::fail`) before the
  session ends silently. The client half-closing its send side disables the
  arm; the session stays held.
- **drain** — the host's phase leaving `Running` (a DRAIN or a RELEASE) ends
  the session `Aborted{Drain}` at once: the only host-initiated cut.
- **re-verification tick** — every heartbeat, the SAME determinants
  admission decided are re-read against the live row (`reverify`): the row
  predicate, the training-set identity (`world_size > 1` sessions:
  the pair unchanged, then the strict resolution and the sidecar verify
  again), the coordinator's liveness. Three ends, pairwise distinct on the
  wire, in scope, and in whether they count (`ReverifyEnd`):
  `Refuted` (`REFUTED`, assembly-scoped, COUNTS toward the assembly's
  attempts — a row or artifact fact no longer holds), `Unavailable`
  (`UNAVAILABLE`, assembly-scoped, never counted — the catalog did not
  answer), `StoreUnavailable` (`STORE_UNAVAILABLE`, member-scoped, never
  counted — THIS host's object store faulted: `JammiError::Storage`/`Io` on
  the sidecar read; a sidecar that does not decode is the artifact's fact,
  `Refuted`).
- **the rank body's end** (a body-bearing session) — the body's natural
  end is the session's end: `Outcome{Trained{artifact_digest}}` for a
  completed run, `Outcome{Failed{reason}}` for a typed failure, or the
  body's own pre-collective `Aborted{Refuted | StoreUnavailable |
  Unavailable}` (its prologue re-verifies the training-set identity and
  every leaf of its partition with the SAME classes the tick uses, so the
  wire says the same thing whichever saw it first). A body-bearing session
  never parks: its bounds are the gang deadline on every round wait and the
  re-verification tick.
- **park bound** (a body-less session) — one lease window after admission
  with no rank body to hand the session to, the session ends
  `Aborted{NoBody}`.

Every end is ONE stream event (or one trailer) followed by the stream closing
and the hold's release. A session that ends for any reason but its own body's
end tells the body to stop (its cancel flag — the trainer's epoch-boundary
check) and severs the round inbox (`RoundInbox::sever`: the body's next
collective ends `Disconnected`, and the link's outbound forwarder is aborted
so no frame of the body's rides the stream after the terminal event); the
body's task runs on to that fault and its result is discarded. **The peer
writes nothing to the job row on behalf of a rank**:
`crates/jammi-server/tests/it/gang_terminal_write_oracle.rs`
derives the catalog's `jobs` writers from `jobs_repo.rs` itself and asserts
none is named in `gang.rs`; the wire rows snapshot the row before admission
and after every end; and the body itself runs as `RunnerRole::Rank`, a type
that holds no `LeaseHolder` to write as (§2.8e).

**Non-disclosure and the `test-hooks` seam.** A table-driven oracle asserts
the rung-6 `Status` (code and message bytes) is byte-identical across every
determinant above — seventeen, `GangRefusalReason` — so no leaking message
ever distinguishes them on the wire. Behind `#[cfg(feature = "test-hooks")]`
only, `GangServer::last_refusal_reason()` / `refusal_reason_handle()`
(`gang.rs`) expose which variant a call actually refused for — a test-only
introspection point, never response text; the plain
`cargo test -p jammi-server --test it` lane cannot observe it, and the
`--features test-hooks` lane executes a strictly larger case count as a
result (the holder-contention rows that manufacture a holder through
`HostAdmission::hold_for_test` are `test-hooks` only too). The witness list
(`gang_service.rs`, `every_gang_refusal_reason`) is re-validated by an
exhaustive match over the enum, so a new variant fails that file to compile
until an arm is added.

**A genuine catalog fault during admission is `Unavailable`, not
`FailedPrecondition`.** `admission_catalog_fault` (`gang.rs`) maps
`Catalog::get_job_for_rank`, `Catalog::get_result_table_for_tenant` and
`Catalog::fresh_instance` erroring to `Status::unavailable(..)` — transient,
retriable, distinct from every rung-6 refusal (a non-retriable row fact).
`map_engine_error` is never called on the `RunRank` path
(`crates/jammi-server/tests/it/gang_admission_catalog_fault_oracle.rs` scans
`run_rank`'s own body for it and for at least three `admission_catalog_fault`
sites). At re-verification the same fault is `ReverifyEnd::Unavailable`.

**Tenant handling.** The tenant is DERIVED from the `jobs` row, never accepted
from the caller: `Assign` carries none, no request metadata is read (the
peer listener runs no resolver), ambient admin scope is refused twice, and
the strict resolver takes the row's tenant as an explicit argument that
ambient scope cannot widen. The `GANG_LISTENER_ALLOWLIST` exemption states
this derivation beside its executed cross-tenant-denial cases
(`gang_service.rs`: another tenant's ready, verifying table is refused with
the fixed status; a NULL-tenant row never resolves for a tenant-bound job;
`jammi-session-id` metadata on the call is ignored). Enumerating-caller
oracles (`crates/jammi-server/tests/it/gang_rank_admission_oracle.rs`) pin
`get_job_for_rank`'s and `get_result_table_for_tenant`'s only production
callers to this handler.

**Observability.** `jammi_gang_requests_total{rpc="RunRank"}`
(`crates/jammi-server/src/routes/health.rs`) counts every `RunRank` call
reaching this member, incremented by the whole-server
`crate::metrics_layer::Metrics::call` regardless of how the call is
ultimately decided — the same shape `jammi_peer_requests_total{rpc}` uses for
`PeerService`.

**Config.** Neither the park bound nor the tick is its own key: `[lease]
duration_secs` (`LeaseIntervals::lease()`) is the held session's park bound
and, doubled, `fresh_instance`'s liveness margin; `[lease] heartbeat_secs`
(`LeaseIntervals::heartbeat()`) is the re-verification cadence and the
longest a `ClaimProbe` is waited on — both read once at `OssServer::bind`
and passed into `GangServer::new`.

### 2.8b Gang membership substrate (`instances.peer_addr`/`result_root`)

The catalog-level carrier §2.8a's `fresh_instance` call sits beside: two
columns on `instances` (`crates/jammi-db/src/catalog/instance.rs`,
`crates/jammi-db/src/catalog/jobs_repo.rs`), migration 035, and the ONE
choke point every writer of them funnels through.

- **`instances.peer_addr` / `instances.result_root`** (migration
  `035_instances_peer_addr_result_root`, both nullable `TEXT`, no paired
  `CHECK` — a row with `peer_addr` set and `result_root` NULL is
  representable and simply never a member): `NULL`/`NULL` means "this
  process never joins a gang" — every library/CLI process, and every server
  that never sets `[server] peer_advertise`. Four migration pin sites: the const
  list (`catalog/migrations.rs`), `EXPECTED_MIGRATION_NAMES`
  (`tests/it/migrations.rs`), the ordered-after oracle
  (`migration_035_is_ordered_after_034_and_adds_instances_peer_addr_result_root`,
  parametrized sqlite/postgres), and the `029` ledger-replay test's DELETE
  list (`migration_029_copies_training_jobs_rows_into_jobs_as_queued` —
  `035` ALTERs `instances`, created fresh by `029`'s replayed DDL, so an
  omission there would leave the reopened table missing both columns, RED).
- **`InstanceRegistration`** (`catalog/instance.rs`): the ONE value every
  writer of the `instances` (+ `workers`) row builds — `instance_id`,
  `label`, `host`, `peer_addr: Option<PeerAddr>`, `member_root:
  Option<MemberRoot>`, plus a `worker: Mutex<Option<WorkerFacts>>` cell
  that is the claim-loop half, owned exclusively by `JobWorker`/
  `EmbeddedWorker` (`fine_tune/worker.rs`): `run_until` sets it only AFTER
  its FIRST `upsert_worker` call SUCCEEDS (a failed first upsert must leave the cell
  `None`, never a fact the row does not carry, so a keeper reregister
  racing a still-failing loop start never writes a `workers` row the real
  upsert never itself managed to write), every LATER `set_worker_state`
  writes the cell before the row, `delete_worker` clears it — so a keeper
  reregister racing a state change always re-upserts the `workers` row the
  process is ACTUALLY about to become, never a stale snapshot, and never a
  fact the row does not yet carry. `PeerAddr` is sealed (`parse`/`as_str`/
  `Display` only) and is the SAME type the peer listener uses
  (`index::peer` re-exports it) — the peer and gang listeners can never
  drift into two address types. `PeerAddr::parse` refuses an UNBRACKETED
  IPv6 literal: a bracketed IPv6 host (`[::1]:9000`), an IPv4
  literal, or a DNS hostname are accepted; `2001:db8::1:9000` is refused
  (ambiguous which colon separates host from port).
- **`MembershipConfig::validate` and `InstanceRegistration::from_config`**
  (`catalog/instance.rs`). The member
  row's root is the byte-for-byte output of
  `JammiConfig::resolved_result_root()`, carried VERBATIM: `MembershipConfig::
  validate(&JammiConfig) -> Result<Option<MembershipConfig>>` checks only
  that `peer_advertise` parses as a `PeerAddr` and that `peer_bind` is set
  too (else a typed error naming both keys) — it performs NO filesystem
  access and inspects `result_root`/`artifact_dir` not at all.
  `InstanceRegistration::from_config` runs `MembershipConfig::validate`,
  then, when membership applies, sets `member_root` to
  `MemberRoot::resolved(config)` — the ONE production constructor, wrapping
  `config.resolved_result_root()?` — the SAME string `build_result_store`
  (`jammi-ai/src/session.rs`) hands to `ResultStore::with_root`.
  `MemberRoot::new` (a bare-string wrap, no resolver call, no validation)
  exists ONLY behind `feature = "test-hooks"`, for fixtures — a production
  build never links it, so nothing outside `MemberRoot::resolved` can put an
  arbitrary string in the `instances.result_root` column. **The membership
  path performs NO
  interpretation of the root at all, and the gang-membership listing verb
  does not even read it**: no
  URL parse, no scheme handling, no symlink resolution, no case folding, no
  byte comparison. The row still carries the configured spelling verbatim —
  two spellings of one physical location (`gcs://b/p` vs `gs://b/p`, a
  trailing `/`, a case difference) are two DIFFERENT STRINGS in that
  column — but `list_gang_members`'s admission predicate does not consult
  it at all; root identity across spellings is not decided anywhere, and no
  membership predicate is built on it. The only
  refusal on this path is the non-UTF-8 refusal already inside
  `resolved_result_root` (a non-UTF-8 `artifact_dir`,
  the default arm's only failure mode). `JammiConfig::load_from` calls
  `MembershipConfig::validate` directly (the early-failure check);
  `InferenceSession::wrap_with` (`session.rs`) calls `from_config` once per
  session, before the lease keeper starts and before the result store does
  anything — the universal funnel every `InferenceSession` constructor
  reaches, so a hand-built config (never routed through `load_from`) is
  still covered. `ServerConfig::validate` is NOT the home for any of this:
  it cannot see `artifact_dir`, which `resolved_result_root` needs.
- **`JammiConfig::resolved_result_root()`** (`config/mod.rs`): the ONE
  effective-root derivation (`storage.result_root` when set, else
  `{artifact_dir}/jammi_db` — the SAME derivation `ResultStore::new`'s
  local-root arm performs), fallible: it refuses naming `artifact_dir` when
  the joined path is not valid UTF-8, rather than silently lossy-folding it.
  This is the ONLY function on the membership path that can fail, and the
  ONLY source of the member row's root string.
- **The two read verbs** (`catalog/jobs_repo.rs`, both tenant-unscoped by
  construction — `instances` carries no tenant column): `peer_addr_of(id,
  lease)` is the ONE by-id resolution surface (no kind/self filter —
  any member may resolve any other by id, including a busy or other-kind
  one); `Some` iff the row is present, fresh under
  `instance_liveness_margin(lease)`, and `peer_addr` is non-NULL.
  `list_gang_members(GangListing { kind, self_instance, lease })` (no root
  field) is an `instances JOIN workers` listing: excludes the caller
  itself, excludes `workers.state != 'claiming'` (an INNER join — no
  `workers` row is excluded too, since a member is a fleet worker with a
  claim-loop slot, not merely a live process), excludes a `kinds` token
  that does not match `kind` as a WHOLE comma-split trimmed token (`,`
  is `upsert_worker`'s own encoding), excludes stale/NULL-`peer_addr`
  rows; `result_root` plays NO part in this predicate — two members whose
  `result_root` strings differ (by scheme alias, case, trailing `/`, or
  anything else) ARE gang members of each other. The survivors are sorted
  by `instance_id` BYTE ORDER in Rust (never a SQL `ORDER BY` — backend
  collation is untrusted). A corrupted stored `peer_addr` that fails
  `PeerAddr::parse` is a typed `Catalog` error from either verb, never
  silently mapped to "not a
  member".
- **The lease keeper's reregister** (`catalog/lease_keeper.rs`,
  `LeaseTarget::Instance(Arc<InstanceRegistration>)`): a normal heartbeat is
  `Catalog::touch_instance` (pure UPDATE, never resurrects a pruned row); a
  MISSED touch (`Ok(false)` — the row was pruned during a transient outage)
  calls `Catalog::reregister_instance(&reg)` instead of flipping `lost` — it
  re-upserts the `instances` row AND, when the registration's worker cell
  is `Some`, the `workers` row too, in ONE transaction, so a live process
  rejoins its gang (and its claim-loop membership, if any) with no restart.
  `instance_prune_window(lease)` (`catalog/lease.rs`) = `instance_liveness_
  margin(lease).saturating_add(lease)` = `3 × lease`, STRICTLY beyond the
  `2 × lease` margin `fresh_instance`/the two read verbs judge freshness
  by — `InferenceSession::wrap_with`'s construction-time
  `prune_instances` call uses this function, never a literal
  `saturating_mul(2)`/`(3)` at the call site, so a merely-stale member (in
  `(margin, window]`) keeps its row through at least one more sweep, giving
  the keeper's reregister a chance to land before a prune sweep could ever
  reap it.

### 2.8c The `Peer` collective and the round protocol

The cross-process arm of the collective (`crates/jammi-ai/src/fine_tune/collective/peer.rs`,
`Peer`): rank 0 is the coordinator, in the process that claimed the job;
every other rank is a member on the far end of one admitted `RunRank`
stream. It is the fourth `Collective` implementation beside `Noop`, `Local`
and `Nccl`, selected by configuration like the others; the trainer holds a
`&dyn Collective` and is never `cfg`-forked.

**The blocking-call witness.** Every `Collective` verb takes a
`&BlockingCall` (`crates/jammi-ai/src/fine_tune/collective/mod.rs`,
`BlockingCall`): a thread-bound witness that the caller is on a thread that
may block — a `spawn_blocking` thread or a plain OS thread, never a runtime
worker. It has no public constructor; it is minted only inside the closures
`BlockingCall::spawn_blocking` / `spawn_thread` / `spawn_scoped` run on the
thread they create, and it is `!Send + !Sync`, so it cannot be carried into
a `tokio::spawn`ed future or stored where a worker thread could reach it.
`Peer` is the arm that needs it (its verbs drive stream I/O under
`Handle::block_on`, a panic on a worker thread); the witness sits on the
TRAIT, not on `Peer` alone, because the trainer never names `Peer` — a
guarantee on `Peer`'s inherent methods would be invisible at the one call
site that matters. `Noop`, `Local` and `Nccl` accept it and ignore it. The
trainer threads the witness rather than minting one: `TrainingLoop::run`
takes a `&BlockingCall` (`crates/jammi-ai/src/fine_tune/trainer.rs`) and
passes it down every path that reaches one of `RankContext`'s five wrapper
verbs — the per-step gather, the lockstep flag reduce and both
`canonical_reduce` call sites, the epoch-boundary dropout-position gather —
and the worker mints it at exactly three places, all in
`crates/jammi-ai/src/fine_tune/worker.rs`: the `BlockingCall::spawn_blocking`
that enters rank 0's `run_fine_tune_blocking` (the single rank, a `Local`
gang's rank 0, a `Peer` gang's coordinator) and the `BlockingCall::spawn_thread`
per OTHER rank of an in-process `Local` gang — one OS thread pinned to one
device, entering the runtime's handle so the trainer's `block_on`s work there
as they do on the blocking pool — both in `train_fine_tune` (§2.8d), and the
`BlockingCall::spawn_blocking` that enters a `Peer` member's
`run_fine_tune_blocking` in `run_member_rank`, the rank body (§2.8e); tests
mint theirs with
`spawn_thread`/`spawn_scoped` (the local-gang oracles run each rank on such a
thread). The compile-time claim has an executed oracle:
four doctests on `BlockingCall`'s own docs
(`crates/jammi-ai/src/fine_tune/collective/mod.rs`; `cargo test -p jammi-ai
--doc -- BlockingCall`) — `compile_fail,E0277` for a verb from a
`tokio::spawn`ed future (`BlockingCall` is not `Send`), `compile_fail,E0061`
for a verb with no witness to pass, `compile_fail,E0624` for the private
constructor, and the control (the same verb from `spawn_blocking` compiles
and runs). rustdoc checks the error codes, so a witness made `Send` fails the
`E0277` block by compiling.

**The round descriptor.** `Descriptor` and `TensorSignature` live in
`crates/jammi-ai/src/fine_tune/collective/mod.rs`, shared by `Local` and
`Peer`: `round` (the round index, counted from 0 on every rank — `Local`
stamps it under its rendezvous lock from the shared round's generation,
`Peer` from each rank's own counter, and the wire carries it), `verb` (the
closed `Verb` enum), `world`, `root`, `counts`, one signature per tensor
(the trailing shape only for `all_gather`), and `agreement` — an opaque
digest the caller binds per rank (`Local::with_agreement`,
`Peer::with_agreement`; the trainer binds its canonical trainable-variable
key-name digest, computed on its rank context). `Descriptor::agrees_with` is
the derived equality — every field is a determinant, and the per-field
mutation sweep in `local.rs` destructures the struct with no `..`, so a
field added without an arm there fails to compile. A disagreement on any
field is a typed refusal naming BOTH descriptors on EVERY rank: on `Local`
before the rendezvous publishes, on `Peer` before the coordinator folds.

**The wire.** `crates/jammi-wire/proto/jammi/v1/gang.proto`, additive
arms on the frozen frames: `RankControl` (coordinator → member) gains
`round_result`, `round_chunk`, `round_commit`, `round_fault`; `RankEvent`
(member → coordinator) gains `round_contribution`, `round_chunk`,
`round_ack`, `round_fault`. `RoundDescriptor` mirrors the Rust descriptor
(`round`, the closed `RoundVerb` enum, `world`, `optional root`, a wrapped
`Counts` message so absent and all-zero differ, `TensorSignature { dims,
ElementType }`, `optional agreement`). `ElementType` is `F32`/`F16`/`BF16`;
`RoundVerb`'s `UNSPECIFIED` and every value outside the set are refused
naming both sides, never defaulted. Every numeric field is range-checked by
its reader before use (`world`, `root < world`, every count and dim as a
`usize` with an overflow-checked element product, `chunk_count`, `index <
chunk_count`, the reassembled byte length against the bound the descriptor
implies). Nothing declared before is renamed or removed; the api-freeze
guard decodes only `PACKAGE`/`RPC` tokens, and `api_freeze_baseline.txt` is
unchanged (`crates/jammi-server/tests/it/api_freeze.rs`).

**Rank-ordered coordinator-reduce.** Every member sends its contribution —
its descriptor and its tensors as Arrow IPC (one length-prefixed IPC stream
per tensor: f32 as a `Float32` column, f16 as `Float16`, bf16 as its
`UInt16` bit pattern, exact; every other candle dtype is refused at the seam
before a descriptor exists) — to the coordinator, which folds in RANK ORDER
on its own device with the same operation sequence `Local` runs on rank 0's
device (`Tensor::cat` of the slices, a left fold of `acc.add(term)`, `max`,
the root's tensor), so the two arms are byte-identical over the same inputs
(`fine_tune::collective::peer_tests::peer_fold_over_the_wire_equals_local_fold_byte_for_byte_at_f32_f16_bf16`,
and over a real loopback stream
`crates/jammi-ai/tests/it/peer_gang.rs`). A payload larger than
`[server.limits] max_message_bytes` travels as `RoundChunk`s of at most
`max_message_bytes − 64` bytes each and is reassembled against the byte
bound the agreed descriptor implies; a peer announcing more is refused
before the bytes are buffered.

**The two-phase round.** A round `k` on a member: send the contribution;
wait for the result; decode and HOLD it unapplied; send `RoundAck`; wait for
`RoundCommit`; apply. On the coordinator: collect every member's
contribution; refuse the round on every rank unless every descriptor
equals rank 0's; fold; publish; wait for every member's ACK; send
`RoundCommit` to every member; apply. A fault before the coordinator has
observed the last ACK — a disconnect, a `RoundFault` from any rank, the
gang deadline — leaves NO rank applied for `k`, and every rank's error
names `k`. A fault DURING the commit fan-out is fatal on every rank too:
the coordinator applies nothing, faults every member and refuses every
later round; a member whose stream ended between its ACK and the commit
applied nothing; a member the commit did reach applied `k` and returned
`Ok` — that cannot be retracted — but its next contribution is answered by
the coordinator's fault, so no rank ever continues past a round every rank
did not apply. That one residual state is stated rather than inherited from
the shared-memory arm's publish-then-fault. Every wait on every rank
expires at the gang deadline (`Peer::with_timeout`, the same bound as the
in-process rendezvous) with an error naming the round and what it waited
for; a rank that refuses its own arguments faults its peers with a
`RoundFault` before returning, so no peer waits out the deadline for a
contribution that was never coming; a fault is permanent — every later verb
on the faulted rank refuses quoting it, exactly as `Local` does.

**Links and the server seam.** `Peer` never opens a stream. A `MemberLink`
is the member's end of one admitted stream, a `CoordinatorLink` the
coordinator's; both are built over channels
(`MemberLink::from_channels`, `CoordinatorLink::from_channels`), which is
what the hermetic oracles drive a whole gang through in one process, and
`CoordinatorLink::over_client(channel, assign, max_message_bytes)` opens
`RunRank` on a member, sends the `Assign`, requires `Admitted`, and caps
the client's OWN inbound decode at the same `max_message_bytes` (tonic's
default would otherwise cap that side at 4 MiB regardless of the
deployment). In `jammi-server` the seam is
`crates/jammi-server/src/grpc/gang_rounds.rs`: `member_link(events)` builds
the member's link over an admitted session's own outbound event sender and
returns the `RoundInbox` the session's hold loop delivers round frames to
(`RoundInbox::is_round_frame`, `RoundInbox::deliver`, `RoundInbox::fail` for
a transport error the loop read); `dial_member(addr, assign,
max_message_bytes)` is the coordinator's dial over a `PeerAddr`. The hold
loop itself — admission, the holder CAS, the `select!` whose inbound arm
calls `deliver` — is `GangServer::run_rank`'s admitted-session machinery
(§2.8a). `crates/jammi-server/tests/it/gang_rounds.rs` drives one real
round through a hold-loop-shaped handler, and pins the REAL handler's
trailer for a round frame after a body-less session's link closed (the
`test-hooks` seam `GangServer::take_member_links` hands out a `world_size ==
1` session's link; a `world_size > 1` session's link is its rank body's and
is never offered). The REAL handler's rounds through the REAL rank body —
folding byte-for-byte what `Local` folds — are
`crates/jammi-server/tests/it/gang_coordinator.rs`'s (§2.8e).

**The decode cap on every listener.** `[server.limits] max_message_bytes`
bounds every listener's inbound decode: the public chain's services in
`assemble_grpc_chain` and the `peer_bind` listener's `PeerService` and
`GangService` in `OssServer::bind` (`crates/jammi-server/src/runtime.rs`)
carry the same per-service `max_decoding_message_size`, and the
coordinator's client is the third site. Stated in `encoded_len()` terms and
pinned (`crates/jammi-server/tests/it/gang_rounds.rs`): a frame of exactly
`max_message_bytes` encoded bytes decodes on every listener and both
peer-listener services, so does `n − 1`, and `n + 1` is refused
`OUT_OF_RANGE` naming the configured value. `crates/jammi-server/src/limits.rs`'s
N5 rustdoc quantifies the invariant over listeners.

**The rank's read path.** Before its first collective a rank verifies the
row groups of its partition against the attestation's leaf inventory
(§2.6b's `LeafDigest`s): `verify_partition_leaves(handle, leaves)`
(`crates/jammi-ai/src/fine_tune/collective/peer.rs`) reads one leaf at a
time through `JammiObjectStore::get_range`
(`crates/jammi-db/src/storage/object_store_handle.rs`) — memory is bounded
by the largest row group, never the artifact — and `verify_leaves` is the
same check over any ranged reader. A failure is
`RankReadFault::StoreUnavailable`, MEMBER-scoped by construction (the enum
has no assembly-scoped arm), named by the leaf, mapped to the wire's
`ABORT_REASON_STORE_UNAVAILABLE` by `RankReadFault::abort_reason` — never
counted against the assembly's attempt budget. The coordinator, on a
member's `Aborted` before any contribution, faults the round naming the
reason and folds nothing.

### 2.8d The coordinator body — topology, membership → assignment → dispatch → assembly

Where a claimed training job's ranks run is decided ONCE,
in `JobWorker::run_spec` (`crates/jammi-ai/src/fine_tune/worker.rs`), by
`TopologyDecision::decide(world_size, local_ranks)` from the job's own
identity-relevant `TrainingCommon::world_size` and this host's `[worker]
local_ranks`, and nothing else: `world_size <= 1` is `Single` (today's path,
`RankContext::single_rank`, byte-identical); `1 < world_size <= local_ranks` is
`Local` (every rank in this process over a `LocalGang`, rank `r` pinned to
`[gpu] devices[r]`, each rank its own model-cache entry for its device —
`ModelCache::get_or_load_on` — and its own replicated source,
`TrainingSource::replicate`); `world_size > local_ranks` is `Peer` (this
process is rank 0, the coordinator). `[distributed] max_world_size` plays no
part here: it bounded the job at submit — `RankAdmission`
(`crates/jammi-ai/src/fine_tune/spec.rs`) refuses `world_size >
serveable_world` from configuration alone (the type holds no catalog handle);
a `world_size` within it but beyond this host's own devices submits and is
decided by assembly. A `graph_fine_tune` wider than `local_ranks` is refused
at the coordinator's edge (no training-set table for a member to be admitted
against), never run as a smaller gang. `train_fine_tune` spawns the layout:
rank 0 on the blocking pool, every other local rank on its own
`BlockingCall::spawn_thread`; the other ranks are joined before rank 0's
result is read, and a rank that errored or panicked makes the whole run a
failure — rank 0's artifact is never published over a gang that did not
complete. Rank 0 alone persists the acceleration report. **Every rank's model is built
with its own dropout seed** (`run_fine_tune_blocking`): `RankContext::dropout_seed(config.seed)`
— rank 0 (and the single rank) is the identity, so W=1 is byte-unchanged — goes to the
`_for_rank` head builders (`fine_tune/lora.rs`) and to `LoraBuildConfig::dropout_seed` for an
encoder-adapters target, while the A/B init seed stays `config.seed` on every rank: the ranks
start from byte-identical adapter weights and draw distinct masks (DESIGN.md §4). The
`test-hooks` record `training_test_hooks::rank_targets_for` captures each rank's dropout seed,
its head layers' `dropout_run_seed`s and a pre-step weight digest, and the fan-out oracle
asserts distinct seeds over equal digests through the real `run_spec`.

**The coordinator body** (`JobWorker::coordinate` → `assemble_and_run`), in
order: (1) the write-once CAS of the training-set identity pair —
`Catalog::materialize_or_reuse_training_set(job, worker, attempt, sidecar
digest, table name)`; a `Moved` claim exits with NO write of any kind; (2) the
scaler — computed inside every rank's own `TrainingLoop::run` from the
training set's targets, nothing crosses the wire; the
pre-dispatch gates (the cancel flag → `Cancelled`; the host's phase not
`Running` → `Drain`); (3) membership — `Catalog::list_gang_members(GangListing
{ kind, self_instance, root: this process's own MemberRoot, lease })`: the
verb decides kind, `claiming` state, freshness, self-exclusion and root
identity, the body filters nothing (a host with no `MemberRoot` — `[server]
peer_advertise` unset — cannot coordinate and says so); (4) assignment —
`assign_ranks(&listing, world)`, a PURE function of the listing sorted by
`instance_id` byte order (rank `r` is the `r`-th member; a listing shorter
than `world − 1` is `ShortListed`, never a smaller gang: no substitution);
(5) dispatch — per assigned member, `Catalog::peer_addr_of(instance_id)`
then the installed `MemberDialer` (`HostAdmission::member_dialer`, the
engine's one transport seam; `jammi-server` installs `gang_rounds::GangDialer`
over `dial_member` beside the gang listener in `OssServer::bind`) with the
`Assign { job_id, attempt, rank, world, coordinator_instance_id }`; a member
that does not admit (`Unavailable` for a busy slot, a job-row refusal, a
transport error) ends THIS attempt (every session admitted so far is ended)
and the NEXT attempt re-lists; (6) `Peer::coordinator(links, device,
max_message_bytes).with_timeout([worker] rank_timeout_secs)` and the run as
rank 0 through `train_fine_tune`, as `LeaseHolder::Coordinator` (§2.8e);
(7) **the terminal write on receipt** — rank 0's run returned an artifact,
but the attempt is `Published` only once every member's session has ended
`Outcome{Trained{artifact_digest}}` with rank 0's OWN adapter digest
(`Peer::collect_member_ends` reads each link, under the gang deadline, for
its `Outcome`/`Aborted`/close; `reconcile_member_ends` compares against
`adapter_files_digest` over the files rank 0 is about to publish): a
member's `Failed{reason}` or a differing digest is `TrainingFailed` (the
job's own terminal failure, nothing published), an `Aborted{reason}` is
`MemberAborted`, a closed or silent stream a `LinkFault`; (8) every member
session is ended cooperatively (`Peer::end_members` → one `Cancel` each)
whichever way the run ended, and exactly one `AssemblyOutcome` is recorded
through `Catalog::record_assembly_outcome`. A `Peer` gang RESUMES like an
in-process one: rank 0 and every member discover the job-level resume
checkpoint in the fleet's shared store (the root identity every member was
admitted on — `run_fine_tune_blocking`'s `discover_resume`, on every rank)
and restore it, per-rank dropout positions included, so the successor gang
of a retired attempt starts from the last epoch boundary (the chaos rows
below).

**The exit table is total.** Every way the body ends is a variant of
`CoordinatorEnd` (`worker.rs`); `assembly_outcome(&end)` matches it with no
wildcard arm, so a new end without a row is a compile error. The rows:
`HostCannotCoordinate`/`ShortListed` → `ShortListed`; `CatalogFault`,
`MemberUnreachable`, `MemberRefused`, `PeerRefused`, `LinkFault` →
`Unavailable`; `Cancelled` → `Cancelled`; `Drain` → `Drain`; `MemberAborted
{ reason }` → the outcome of the same name (`Refuted`/`Unavailable`/
`StoreUnavailable`/`NoBody`/`Drain`/`Cancelled`; a value outside the frozen set
reads `Unavailable`) — the typed reason comes from the member's `Aborted`
frame, recorded on the `CoordinatorLink` that read it (`CoordinatorLink::
session_abort`, `Peer::member_aborts`); `TrainingFailed`/`Published` →
`Success` (assembly proceeded to a run; a run's own failure is recorded
`failed` by the caller as at W=1); `Moved` → nothing. `AllRootDivergent` is
never produced: root identity is a predicate inside the listing verb, so an
all-divergent fleet reads as a short listing. After recording, the lease is
settled by the released-vs-failed split (`lease_settlement`, a total match
over `CoordinatorEnd` — DESIGN.md §4 "Failure and release"): a member's
`Aborted{Drain}` hands the lease back at once (`Catalog::release_job_lease`
— `releases + 1`, lease NULL, the row claimable within one idle poll: a
rolling restart of the peer tier costs zero net attempts); every
other mid-run gang fault (a member's `Aborted` for any other reason, a
dropped stream, a rank silent past `[worker] rank_timeout_secs`, a peer's
round fault) leaves the lease to EXPIRE — reclaim arm 1a requeues the row
within the lease window and the successor's `claim_next` spends the attempt
(`attempts + 1`, `releases` unchanged), so a member that keeps failing
exhausts the job's attempts instead of retrying it forever; an assembly end
(no run started) settles by its outcome's counting class
(`AssemblyOutcome::counts_toward_failures`: every one uncounted today, so
released). Either way the attempt returns `WorkerJobError::Abandoned`,
writes nothing terminal, and the row stays `running` for reclaim; the next
claim waits out `next_assembly_after`. `Cancelled` returns through the
existing cancel arm (a request lands `failed`, a lost lease is left for
reclaim).

**The per-attempt watchdog is the coordinator's own `Peer`.** Every
member's stream is read by rank 0's rounds, so a member's `Aborted{reason}`
(recorded typed on that link, `CoordinatorLink::session_abort`), a stream
that dropped, or a rank silent past `[worker] rank_timeout_secs` (the round
deadline) ends rank 0's collective call with the gang faulted — every member
is faulted in the same round (`RoundFault`, `Peer::fault_all`) and its
session ended cooperatively (`Peer::end_members` → `Cancel`, the stream
close) — and the body classifies the end from the links (`MemberAborted` /
`LinkFault`). The `Peer` is built for the attempt and dropped with it, so a
fault retires exactly the attempt it belongs to (the actuator rule: the
engine ships the actuator, never the control loop). Ending a session never aborts a
claim transaction, by the slot discipline: a member's slot is `Rank` for its whole session and a peer never
claims while it holds a rank (`HostAdmission`), so ending a session never
aborts a claim transaction anywhere. The successor attempt resumes from the
job-level resume checkpoint (`{job_id}/_resume/`, rank 0's epoch-boundary
write) and publishes bytes equal to an uninterrupted run. A crashed
coordinator's live `building` training-set row is never met by the
successor at this tip — the producer names every table uniquely, anchors a
registered source `UnpinnedAtInstant` (so the reuse probe never matches),
and the job row's pair is recorded only after the table is `ready` — so the
successor materializes its own table and the orphan is the lease's to reap
after expiry (`ResultStore::recover` → `claim_expired_building_table`); no
`BackOff` disposition exists on the training path
(`crates/jammi-ai/tests/it/gang_coordinator.rs`, the planted-row oracle).
The hermetic chaos rows — a member's stream dropped mid-round, a member
silent past the deadline, a member's host draining mid-round, split brain
(an older attempt's stale runner fenced by the successor's `RunRank`) — run
over a two-host loopback fleet (two engines over one catalog and result
root, the member's real `GangServer::run_rank` on its own runtime) in
`crates/jammi-server/tests/it/gang_chaos.rs`; the process-level SIGKILL rows
(a peer, the coordinator) are `crates/jammi-ai/tests/distributed/gang_chaos.rs`,
advisory in the distributed lane. The oracles:
`crates/jammi-ai/tests/it/gang_coordinator.rs` (a two-rank job within the
serveable world on a one-device host reaches assembly and lands
`ShortListed`, cooled, released; the `Moved` CAS arm writes nothing; a
`local_ranks = 2` job fans out through the real `run_spec` and publishes bytes
equal to a `LocalGang` run of the same fixture),
`crates/jammi-server/tests/it/gang_coordinator.rs` (through the REAL
`GangServer::run_rank` on the production `peer_bind` listener: a member
whose slot is busy answers `Unavailable` — the attempt ends `Unavailable`,
cooled and not counted, the lease released; the next attempt re-lists,
admits, the member's REAL rank body runs rank 1, ends `Outcome{Trained}`,
and rank 0 publishes bytes equal to `Local`'s; a member whose body reports
`Outcome{Failed}` ends the attempt `TrainingFailed`, recorded `failed` under
the `Coordinator` role with nothing published), and `worker.rs`'s own table
oracle over every `CoordinatorEnd`.

### 2.8e The rank body and the runner roles — the single-writer rule as types

The single-writer rule (`docs/plans/67-distributed-training/DESIGN.md` §4) — the lease holder is
the ONE writer of a job's row, its durable checkpoints and its published
artifact; every other rank of a gang writes nothing durable — is stated as
two types in `crates/jammi-ai/src/fine_tune/role.rs`: `LeaseHolder`
(`LoopClaimer` — today's in-process path, the single rank and a `Local`
gang's rank 0, which never traverses the coordinator body; `Coordinator` —
rank 0 of a `Peer` gang) and `RunnerRole` (`Holder(LeaseHolder)`, or `Rank {
rank }` for every rank `>= 1`, in-process or a `Peer` member). EVERY
job-row-writing site on the run path takes a `LeaseHolder` as a required
parameter — the lease-hold registration with its `Releasing` self-release
arm and holder accounting (`register_job_hold_or_release`), the
acceleration-report write (`persist_acceleration_report` and its two
markers), every `record_failed` site, `publish_and_finalize` and so the
`finish_job_with_model` CAS, and the coordinator's own
`record_assembly_outcome`/`release_job_lease` — so a missed site is a
compile error and a `Rank` body, which holds no `LeaseHolder`, has nothing
to pass: the write is unreachable by type. The per-site table, derived by
grep, is `worker.rs`'s module doc ("Runner roles and the job-row writers").
The holder of one attempt is derived ONCE from the claimed spec and this
host's `[worker] local_ranks` (`lease_holder_for`: the `Coordinator`
exactly when a column-source `fine_tune` decides `TopologyDecision::Peer`,
the `LoopClaimer` otherwise — the SAME `decide` call `run_spec` makes) and
threaded to every site; `train_fine_tune` gives rank 0
`RunnerRole::Holder(holder)` and every other local rank `Rank { rank }`.
The trainer's own durable writes carry the same gate:
`TrainingLoopBuilder::runner_role` (derived from the rank context when
unset — rank 0 the loop claimer, the pre-role default; a role that
contradicts the rank is refused at `build`), and `save_resume_checkpoint` /
`save_epoch_checkpoint` write only for a holder — in the trainer, never in
the store, which stays role-agnostic. `W == 1` is the `LoopClaimer`
on every arm and never enters the coordinator body
(`crates/jammi-ai/tests/it/gang_coordinator.rs`, the pinned single-rank
row; the `test-hooks` records `training_test_hooks::lease_holders_for` /
`runner_roles_for`).

**The rank body** (`worker.rs`, `run_member_rank`; spawned by
`GangServer::run_rank` at admission for every `world_size > 1` session,
§2.8a): under the row's own tenant scope, the spec is reconstructed from
the row's `spec` column (`RankAdmissionRow::spec`; a column-source
`fine_tune`, the one kind a `Peer` gang serves — nothing but the coordinates
came from the coordinator), the recorded training set is bound by name and
digest exactly as a retrying coordinator binds it
(`bind_recorded_training_set`, whose refusal classes are the hold loop's
re-verification classes — `Refuted` for an identity that no longer holds,
`StoreUnavailable` for this host's store, `Unavailable` for the catalog),
every leaf of the table is verified against the sidecar's inventory BEFORE
the first collective (`collective::peer::verify_partition_leaves`; under
`BlockByGlobalBatch` every row group carries rows of every rank, so the
partition's leaves are the object's; a bad leaf is the member-scoped
`Aborted{StoreUnavailable}`), the source is bound through the SAME
`bind_training_source` rank 0 used (so the ranks' loaders derive from one
definition), the base model is loaded, the member's `Peer` is built over
the session's link at the deployment's rank timeout and cap, and
`run_fine_tune_blocking` runs on the blocking pool under the witness minted
at ITS OWN `BlockingCall::spawn_blocking` (the third production minting
site, §2.8c) as `RunnerRole::Rank { rank }`: the same target construction,
seed split and acceleration probe as rank 0, no persisted report, no
checkpoint write, no job-row write, no publish. Its natural end is the
session's one terminal event — `Outcome{Trained{artifact_digest}}`, the
digest of the adapter files it holds (`adapter_files_digest`, the ONE
function both sides compute over the file set `publish_artifact`
publishes), or `Outcome{Failed{reason}}` — which the coordinator consumes
in §2.8d's step (7): `Published` only on every member's `Trained` with rank
0's own digest. **The agreement binding**: once the target is built and the
varmap is final, every rank binds `RankContext::canonical_vars_digest` of
`optimizer::sorted_trainable_var_names` on its collective
(`RankContext::bind_agreement` → `Collective::bind_agreement`, at the top
of `TrainingLoop::run`, before the first collective), so a rank whose
canonical layout differs from its peers' is a typed descriptor
disagreement naming both digests on every rank, never a wrong fold
(`trainer.rs`, `runner_role_and_agreement_oracle`). `Noop` and `Nccl`
accept and ignore it; `Local` and `Peer` bind once (the same digest again
is a no-op, a different one a typed error).

### 2.8f `jammi-ballista` — the Ballista compute plane

`jammi-ballista` sits between the
engine (`jammi-ai`/`jammi-db`/`jammi-wire`) and `jammi-server`
(`crates/jammi-ballista/src/lib.rs`'s crate doc): publishable, lockstep
with the rest of the workspace, no cargo feature — a process's role is
`[ballista]` config (§2.1 above), decided at runtime by `jammi-server`.

- **`JammiCodec`** (`codec.rs`, `PhysicalExtensionCodec`) — encodes
  `InferenceExec`/`AnnSearchExec`/`AsofJoinExec`/`KeyCheckExec`/`GangExec`
  as prost messages of a package it compiles itself, `jammi.ballista.v1`
  (`build.rs`) — **not** part of the frozen `jammi.v1.*` surface [§1.3]:
  this package crosses a scheduler/executor boundary INSIDE one cluster's
  own processes, never a client/server wire a foreign consumer decodes, so
  the crate that speaks Ballista's wire owns its shape outright, with none
  of the frozen surface's cross-release compatibility obligations. Every
  buffer this codec writes starts with a 4-byte magic (`codec.rs`'s module
  doc: an illegal prost tag byte, so it can never alias a buffer Ballista's
  own codec wrote); an unmagicked buffer delegates whole to
  `BallistaPhysicalExtensionCodec` — the ONLY way Ballista's own nodes
  cross the wire. Decode rebuilds each operator through its public
  constructor against the DECODING process's own `InferenceSession` (a
  `Weak` reference).
- **`JammiExecutionEngine`** (`engine.rs`) wraps Ballista's
  `DefaultExecutionEngine` and adds two duties before delegating: a stage
  containing a `GangExec` must be single-partition (one gang mechanism,
  never a multi-partition fan-out); a stage whose required device kind (an
  `InferenceExec`'s or a `GangExec`'s stamped `device_kind`) differs from this executor's own
  `InferenceSession::compute_device()` is refused typed (device
  pinning), never silently run on the wrong device.
- **Roles** (`roles.rs`): `host_scheduler`/`host_executor` build a
  `SchedulerRole`/`ExecutorRole` served on jammi's own shutdown — never
  Ballista's own `start_server`/`start_executor_process`, which install
  their own `ctrl_c` handlers and would race the server's two-mode
  shutdown. `host_scheduler` takes a `BallistaCluster`/
  `TaskDistributionPolicy` pair as parameters ONLY so an in-memory cluster
  + a bare policy stay reachable as a test fixture — `jammi-server`'s own
  hosting always passes `BallistaCluster::new(CatalogClusterState,
  CatalogJobState)` and `TaskDistributionPolicy::Custom(DevicePlacement)`;
  there is no knob, the catalog-backed pair is the shipped scheduler,
  never the in-memory one. The scheduler role installs `jammi_ai::
  fine_tune::worker::PlacedGangSubmitter`; the executor role installs
  `PlacedGangRunner` and writes this process's own device claim to its
  `compute_executors` row right after registering.
- **Client** (`client.rs`) — `submit_physical_plan`: the seam a
  scheduler-role process's `PlacedGangSubmitter` calls to place a plan
  instead of running it in-process; matches the plan's own device KIND
  (`InferenceExec::device_kind` or `GangDescriptor::device_kind`, "cpu" is a kind too) and refuses it typed
  BEFORE submitting when no LIVE registered executor lists that kind,
  reading the same catalog `DevicePlacement` reads from and applying the
  same liveness predicate the binder applies (`cluster::executor_is_live`:
  `Active` status and a `heartbeat_at` within `executor_liveness_window()`,
  derived at run time from Ballista's own default executor timeout — a row a SIGKILLed executor left
  behind stops admitting plans after the window, a `Terminating` one at
  once) — `JammiExecutionEngine`'s own device-pinning refusal above
  is the second line, never a silent mis-run.
- **`CatalogClusterState`/`CatalogJobState`** (`cluster.rs`) — the
  catalog-backed `ballista_scheduler::cluster::{ClusterState, JobState}`
  over `jammi_db::catalog::compute_repo`'s generic, distributor-neutral CRUD
  [§2.3]. Execution graphs are never persisted (Ballista 54.1 has no
  graph serialisation): a scheduler restart keeps executor registrations
  and job STATUS rows, but an in-flight job is re-run through jammi's own
  reclaim, never revived by Ballista.
- **`DevicePlacement`** (`placement.rs`, `TaskDistributionPolicy::Custom`)
  — round-robin over executor slots with three refinements: never binds a
  `GangExec` stage to the executor equal to its own `submitter` (deadlock
  avoidance); a stage binds only to an executor whose OWN registered
  devices list its `GangDescriptor.device_kind`/`InferenceExec::
  device_kind()` (a CPU-stamped stage binds a CPU executor, never only a
  GPU refinement); a `GangExec` stage whose job row is already
  `claimed_by` a DIFFERENT executor is never bound at all (the bind-time
  half of the re-launch guard, §2.8g below).

### 2.8g The placed gang — `transfer_claim` and the hand-off arms

Under Ballista placement a `Peer` gang runs as ONE task,
`GangExec { job_id, attempt, world, submitter, device_kind }`
(`crates/jammi-ai/src/operator/gang_exec.rs`), placed by the scheduler on a
device-bearing executor other than the submitter. Two more `HostAdmission`
seams beside `MemberDialer` [§2.8a], `crates/jammi-ai/src/fine_tune/
worker.rs`: `PlacedGangSubmitter` (installed by the scheduler role) and
`PlacedGangRunner` (installed by the executor role) — `jammi-ai` never
depends on `jammi-ballista`.

**The submitting host's holder.** `Holder` (`worker.rs`) gains
`Awaiting { job_id, attempt }` beside `Free`/`ClaimProbe`/`JobRun`/`Rank`
(`worker.rs`): a claimant that is submitting a `GangDescriptor` (the move precedes the submit) or is
awaiting its stream runs no compute for that attempt, so it can still serve
a `RunRank` session for some OTHER attempt — `HostAdmission::
try_hold_rank` admits out of `Awaiting` exactly as it does out of `Free`; a
two-host fleet could not otherwise assemble a `Peer` gang if its only
free-looking host were the one awaiting its own placement result.
`probe_claim` still refuses `Awaiting`, exactly like `JobRun`.

`submit_placed` (`crates/jammi-ai/src/fine_tune/worker.rs`) submits the descriptor and awaits the
stream; its exit arms are total — `submit_placed` (`crates/jammi-ai/src/fine_tune/worker.rs`) documents them: the stream ends
with at least one batch → `WorkerJobError::HandedOff` (the executor owns
the attempt now: no terminal write, no release); the stream ends in an
error or with no batch → re-read the row — `claimed_by` still this
instance → `Abandoned` (left `running` for reclaim, an attempt spent at the
successor's claim); `claimed_by` moved → `HandedOff` (the executor's own
lease expiry requeues it, never this instance's).

`transfer_claim` (`crates/jammi-db/src/catalog/jobs_repo.rs`) is the hand-off: an `UPDATE` guarded by FOUR conjuncts —
`claimed_by = $from` (a stale runner, or a SECOND launch of the same task
via Ballista's own reset-on-`ExecutorLost`, cannot transfer a claim it does
not hold — the re-launch guard's second half, `DevicePlacement`'s bind-time
refusal above being the first); `attempts = $attempts` (an older attempt
cannot transfer past a newer one); `status = 'running'`; the lease is LIVE
(`lease_live_clause`, a POSITIVE `IS NOT NULL AND ... > now`, never the
`OR`-shaped `lease_expired_clause` a RECLAIM sweep uses — a RELEASE's `NULL`
must FAIL a transfer, the opposite of how a reclaim sweep reads that same
`NULL`). `attempts`/`releases` are untouched by design: a hand-off is zero
net attempts, never a re-claim.

`run_placed_gang` (`crates/jammi-ai/src/fine_tune/worker.rs`, called from the
executor role's `PlacedGangRunner`) — (i) takes this host's job slot
through `HostAdmission::probe_claim` (a host already holding a rank, a
loop-claimed job, or another placement refuses typed BEFORE any row write,
); (ii) `transfer_claim`s the row from the descriptor's submitter to
this instance at the SAME `attempts`; (iii) runs `run_claimed_job_under`
VERBATIM as `LeaseHolder::Coordinator` — the SAME body a `Peer` gang's own
claimant runs — so the published bytes are the same as a `Peer` gang's; (iv) maps the
body's `AttemptEnd` to `PlacedOutcome`
(`Trained`/`Failed`; `LeftForReclaim` is a typed `Err`, so the Ballista task
itself ends in error and Ballista's own `task_max_failures = 0` never
re-runs it — jammi's own reclaim, from a future claim, is the only path
back). The writer table (`worker.rs`'s module doc, "Runner roles and the
job-row writers") states this as two more rows: the SUBMITTER after
`HandedOff` writes NOTHING (the row and its lease-keeper registration are
the placed executor's now); the EXECUTOR running `run_placed_gang` writes
as `Coordinator` — the same body as every `LeaseHolder`-gated site above it
[§2.8e].

### 2.9 Numerics (`jammi-numerics`)

- **`NumericsError` / `Result`** — `crates/jammi-numerics/src/error.rs`. The only
  cross-module contract. Validating kernels return `Result` and never panic on bad input;
  pure infallible kernels (`distance`, `pareto`) use `debug_assert_eq!` for length.
- **distance.rs** — `crates/jammi-numerics/src/distance.rs`: `cosine_distance` (reduces in
  **f32**, returns `1.0` on zero-magnitude, **never NaN**); `cosine_similarity`;
  `vector_norm` (reduces in **f64**). The f32/f64 reduction asymmetry is intentional and
  **not interchangeable** [§5].
- **Namespace structs** holding free fns: `RetrievalMetrics`
  (`crates/jammi-numerics/src/retrieval.rs`), `ClassificationMetrics`
  (`crates/jammi-numerics/src/classification.rs`). Metric structs (`RelevanceJudgment`,
  `QueryMetrics`, `AggregateMetrics`, `ClassMetrics`) derive serde and **cross the wire**
  via `jammi-wire` — field changes are wire-schema changes.
- **`ComputePrecision`** — `crates/jammi-numerics/src/precision.rs`: the
  floating-point dtype (`F32`/`F16`/`BF16`) a backbone's weights and
  activations run at; `F32` is its `Default` — maximally compatible, the
  byte-parity oracle's baseline.
- **`WeightQuantization`** — `crates/jammi-numerics/src/quantization.rs` (the
  `WeightQuantization` enum): the GGUF/k-quant weight-STORAGE-format
  vocabulary — `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2K`, `Q3K`, `Q4K`,
  `Q5K`, `Q6K` — a PEER of `ComputePrecision`, never a variant folded into
  it: `ComputePrecision` names the dtype activations and unquantized
  weights/matmuls run at; `WeightQuantization` names how a weight's bytes
  are packed AT REST on disk, a storage concern orthogonal to compute
  dtype — a `q4_0` weight is always dequantized to `F32` before any matmul
  touches it (`jammi_lora::QuantizedLinear`'s uniform-F32-activation rule,
  §2.6). `Ord`/`PartialOrd` are a MANUAL impl keyed on `gguf_wire_id()` (the
  GGML wire ID GGUF itself assigns each dtype, `ggml.h`'s `enum ggml_type`),
  never declaration order, so a sorted `Vec<WeightQuantization>` reproduces
  GGUF's own wire order deterministically. No `Default`: unlike
  `ComputePrecision`'s `F32`, there is no quantization format a caller
  should silently fall into — whether a weight is quantized at all, and to
  which format, is inherent to the GGUF file it loaded from, never a knob
  with a sensible implicit value.
- Sub-module families: `calibration`, `divergence`, `stats`, `gp`, `histogram`, `ner`,
  `pareto`. All deterministic, seeded-RNG-only [§5].

---

## 3. Data-flow walkthroughs

### 3.1 Embedded open + a verb (Rust)

1. `Jammi::open(Target::Local(config))` — `crates/jammi-ai/src/jammi.rs` (`Jammi::open`).
2. → `InferenceSession::open(config)` — `crates/jammi-ai/src/session.rs`
   (`InferenceSession::open`): `new` → `register_query_functions()`, returns `Arc<Self>`.
   `new` builds the artifact store, model resolver, model cache (one shared
   `Arc<GpuScheduler>`), result store, ANN cache. **`ResultStore::recover` runs here,
   before `load_existing_tables`** [§3.7].
3. → `Session::with_configured_worker(engine)`
   (`with_configured_worker` (`crates/jammi-ai/src/local_session.rs`)) —
   spawns via `with_embedded_worker` → `EmbeddedWorker::spawn`, storing it in
   `_worker` (RAII), only when `worker.enabled`
   (`crates/jammi-ai/src/local_session.rs`) reads `true` (default `true`);
   `false` leaves `_worker` as `None` and nothing is spawned.
4. A verb, e.g. `session.search(req)` — `crates/jammi-ai/src/local_session.rs`
   (`Session::search`): destructures `SearchRequest`, picks `engine.search` vs
   `engine.search_by_id`, applies `.filter`/`.select` on the internal `QueryBuilder`,
   `.run().await` → `Vec<RecordBatch>`.

### 3.2 The search path (query vector → hydrated rows)

1. `Session::search` (`crates/jammi-ai/src/local_session.rs`) → `InferenceSession::search`
   (`crates/jammi-ai/src/session.rs`) → `QueryBuilder::new`
   (`crates/jammi-ai/src/query/builder.rs`). (`InferenceSession::search_by_id`,
   `crates/jammi-ai/src/session.rs`, first resolves the example row's vector *inside the
   engine* via `read_vector_by_key` so the vector never crosses the API boundary.)
2. `QueryBuilder::new`: `resolve_embedding_table` picks the table; builds `AnnSearchExec`
   as the plan leaf; **hydration** joins ANN output `(_row_id, _source_id, similarity)`
   back to the source table on `_row_id = _join_key`, casts string cols to VARCHAR, drops
   `_join_key`, re-sorts by `similarity` descending
   (`crates/jammi-ai/src/query/builder.rs`).
3. `AnnSearchExec::execute` (`crates/jammi-ai/src/operator/ann_search_exec.rs`): lazily
   inside `stream::once`, calls `result_store.resolve_search_mode(&table)`
   (`crates/jammi-db/src/store/mod.rs`): `index_path.is_none()` → exact fallback; else
   `open_index` + `load_sidecar` → `Some(SidecarIndex)`; **on any load error logs a warning
   and returns `None` (degrades to exact)**. Then `index.search(query, k)` or
   `exact_vector_search`, converting `(row_id, dist)` → batch with `similarity = 1.0 -
   dist`.
4. `QueryBuilder::run` (`crates/jammi-ai/src/query/builder.rs`) executes the plan, appends
   evidence/provenance columns. Returns `Vec<RecordBatch>` of hydrated rows.

### 3.3 Embedding materialization (write side, builds the index)

`ResultStore::materialize_embedding_table` (`crates/jammi-db/src/store/mod.rs`):
1. `create_table` → name `{source}__{task}__{sanitized_model}__{nanos}_{uuid8}`; derives
   `parquet_url` and (for embedding `Model` tables) `index_url` ending `.idx` **with no
   extension**. INSERTs the row with **status='building'**.
2. Open `ObjectParquetWriter`, build in-memory `SidecarIndex`, write batch + `index.add(key,
   vector)` per row.
3. `writer.close()` → row count; if non-empty, `index.build()` then `save_sidecar(index_url,
   &index)` (writes the three sibling files).
4. `finalize` (`crates/jammi-db/src/store/mod.rs`): `register_table` (DF) +
   `update_result_table_status(Ready, rows)` (stamps `completed_at`).

(The `EmbeddingPipeline` path, `crates/jammi-ai/src/pipeline/embedding.rs`, is the
production driver — `ResultSink::write_batch` filters OK rows and `add`s vectors; `finalize`
calls `idx.build()` only when non-empty.)

### 3.4 annotate(...) — model inference as a SQL relation

Registration: `InferenceSession::register_query_functions`
(`crates/jammi-ai/src/session.rs`) → `ctx.register_udtf("annotate", …)`, holding a
**`Weak<InferenceSession>`** to avoid a reference cycle
(`crates/jammi-ai/src/query/annotate_udtf.rs`, `AnnotateTableFunction`). Plan-time
`TableFunctionImpl::call` parses string args, **loads the model at plan time**
(`block_in_place`+`block_on`) to learn the embedding dim/regression form and build the
output schema [§7]. Execution-time `scan` builds `SELECT <key, content…> FROM <quoted
relation>` (via `quote_ident`/`quote_relation`), plans it through the same `SessionState`,
then wraps the physical input in `annotate_plan` — the one model-over-columns operator
shared with `QueryBuilder::annotate`.

### 3.5 A tabular fine-tune (submit → claim → train → finalize)

**Submit (fast, no compute):** `InferenceSession::fine_tune`
(`crates/jammi-ai/src/session.rs`) validates config, builds `TrainingSpec::FineTune`,
`submit_fine_tune_spec` ensures the base model is registered (FK), serializes the spec to
JSON, and `Catalog::submit_job`/`submit_job_deduped` inserts it into the kind-agnostic
`jobs` table (migration 029) as a **`queued`**, `execution = 'queued'` row — the same
table every training AND compute kind shares. Returns a `TrainingJob` handle (training's
own handle type; a compute verb submitted through `InferenceSession::enqueue` gets the
generalised `jammi_ai::jobs::JobHandle` instead — both name a row in the same table).
**No in-memory state crosses submit→claim — the spec is the only carrier.**

**Worker pickup:** `JobWorker::run_until` (`crates/jammi-ai/src/fine_tune/worker.rs`) each tick: `Catalog::reclaim_expired_jobs` → `Catalog::
claim_next` (takes a lease over `[worker] kinds`, `FOR UPDATE SKIP LOCKED` on Postgres) →
`run_claimed_job`: deserialize spec, pin catalog to the job's tenant, register the claim
with the process's shared `LeaseKeeper` (one dedicated OS thread, its own runtime and
catalog connection — not a per-lease `tokio::spawn` heartbeat), run under tenant scope,
then `publish_and_finalize`. A `CancelJob`/`JobHandle::cancel` request folds into the
SAME `cancel` flag a lease loss trips, via a watcher that polls `jobs.cancel_requested`
at the keeper's own heartbeat cadence.

**Train:** the rank layout is decided first (§2.8d: a single rank, an in-process `Local`
gang, or the coordinator body of a `Peer` gang), then `JobWorker::run_spec` → FineTune arm → `training_set::materialize_projection`
(`crates/jammi-ai/src/fine_tune/training_set.rs`): the projected columns are committed
through `ResultStore::materialize_training_set` as an immutable `TrainingSet` result table
— or an extant `ready` one is bound instead, on the engine's standing reuse key
(definition hash AND every input anchor equal, no unpinned-at-an-instant anchor among
them; a registered source is anchored unpinned, so the tabular path materialises its own
table) — and read back on the SAME `SessionContext` through `read_back_sql`,
`SELECT * FROM <TrainingSetTable::sql_relation> <training_set_order_by(columns)>`, the
reader's half of the order contract. The `GraphFineTune` kind differs in the producer only:
`materialize_graph_training_set` samples the graph and commits the pairs as a training-set
table ordered by a leading `_ordinal`. From the table on the two kinds share one path — the
reader asks the table's descriptor for its committed order
(`ProducingDescriptor::training_set_order_columns`) and the spec for what to decode
(`TrainingSpec::training_set_view`), never which producer wrote the table — so every
topology, a multi-host `Peer` gang included, serves both.
Then the source binding (`bind_training_source`: `Resident` through
`build_training_data_loader` for the whole-set arms, `Streamed` otherwise — the
SAME binding a `Peer` member's rank body performs over the same table, §2.8e) →
`train_fine_tune` → `run_fine_tune_blocking` (on the blocking pool, `catch_unwind`-wrapped):
builds the `TrainingTarget` (empty `target_modules` → projection head; non-empty →
`build_encoder_adapters`, which resolves the backbone through `model::arch` (§2.7) and
dispatches on **`(family, task)` with no default arm**: BERT/DistilBERT/ModernBERT ×
text-embedding/classification/NER/regression → the text towers; `open_clip` ×
`text_embedding` → `ClipText`; `open_clip` × `image_embedding` →
`OpenClipVisionTransformer`; `clap_audio_model` × `audio_embedding` → `HtsatAudio`. Every
other pairing — and every config this crate has no loader for — is a typed refusal naming
both halves and the family's real towers, never a silent fallback onto a family's
"default" tower. The chosen tower is stamped into the saved `AdapterConfig` via
`with_tower`. Each arm calls that tower's own `builder().lora(...)` — **where injection
happens, `crates/jammi-encoders/src/lora_site.rs` and each tower's in-file site
helper**), then
`TrainingLoop::run` (`crates/jammi-ai/src/fine_tune/trainer.rs`). The loop snapshots
`varmap.all_vars()` once, builds AdamW, runs epochs with grad-accum, cooperative
cancellation at epoch boundaries (the SAME `cancel` flag the lease-loss watcher and the
cancel-request watcher above both write), durable resume checkpoints, early stopping,
saves `best`, builds `SavedAdapter`, calls `jammi_lora::save_adapter`. **The loop never
writes terminal status / registers the model / publishes.**

**Finalize (worker, lease-guarded):** `publish_and_finalize`
(`crates/jammi-ai/src/fine_tune/worker.rs`) writes files to a unique per-attempt prefix
`{job_id}/{worker_id}/{attempt}`, `register_model`, `Catalog::finish_job_with_model`
**CAS** flips to `completed` + commits the served path, every retained epoch-checkpoint
row, and the job's terminal status together — only while `job_id AND claimed_by AND
status == 'running' AND attempts` all still match the caller's own attempt. On CAS win,
GC the resume checkpoint. **Finalization is the worker's sole authority.**

### 3.6 get_or_load (model lifecycle, end to end)

`ModelCache::get_or_load(source, task, backend_hint)` (`crates/jammi-ai/src/model/cache.rs`),
retry loop re-taking the write lock:
- **Fast path**: entry hit → `ref_count.fetch_add(1)` → build `ModelGuard` → `touch_lru` →
  return. No resolver/backend/permit churn.
- **Single-flight wait**: another task is loading this id → clone the `Notify`, drop the
  lock, `notified().await`, `continue`.
- **New-load path**: insert into `in_flight`, drop the lock, call `do_load`, re-acquire,
  `in_flight.remove`, **`notify_waiters()` on BOTH Ok and Err arms** (failure must not
  strand waiters).
- `ModelCache::do_load`: `resolver.resolve` → pick backend by `resolved.backend` →
  `estimate_memory` → **admission loop** (`try_acquire`; on `None` take the lock and
  `evict_one`; if nothing evictable, error) → `backend.load` → post-load catalog bookkeeping
  (`complete_generic_registration`,
  `crates/jammi-ai/src/model/cache.rs`) → insert `CacheEntry` (permit moved in) → return
  guard with refcount 1. The bookkeeping write is gated by an ALLOWLIST of the generic,
  non-terminal row kinds it exists to complete, `GENERIC_COMPLETABLE_TYPES`
  (`crates/jammi-ai/src/model/cache.rs`, `&["local", "huggingface", "embedding"]`) — never a
  denylist of the terminal types to protect, which would fail open on every unenumerated
  `model_type` (`fine-tuned`, `context-predictor`, `bert`, `open_clip`, `clap_audio_model`, …). A
  catalog READ error also skips the write outright (`get_model_version`,
  `crates/jammi-ai/src/model/cache.rs`, `warn!` and keep serving) rather than collapsing to
  "no row" and writing over an uninspected row; only when the read succeeds and the row is absent
  or already one of the completable kinds
  (`can_complete`, `crates/jammi-ai/src/model/cache.rs`) does it proceed to `register_model`
  (`crates/jammi-ai/src/model/cache.rs`), which completes a `local`/`huggingface` row or the
  `embedding` FK placeholder — the one case where a `register_model` failure is still logged and
  swallowed rather than propagated. A fine-tuned id can reach this same call (`ModelSource::parse`'s
  HuggingFace fallback matches it like any other non-`local:` string), so without the allowlist
  this generic write would overwrite the served-adapter `artifact_path` and the `base_model_id`
  lineage a terminal producer already committed.

Resolver chain (`crates/jammi-ai/src/model/resolver.rs`, `ModelResolver::resolve`):
`try_catalog_lookup` (`crates/jammi-ai/src/model/resolver.rs`) first (refuses `Retired`;
resolves fine-tuned base recursively + fetches adapter), else `resolve_local`/`resolve_hf_hub`
(locate config, pick backend, gather weights, discover tokenizer, sum file sizes into
`estimated_memory`). Before any of that, an id carrying the reserved `jammi:fine-tuned:` prefix
(`FINE_TUNED_ID_PREFIX`, `crates/jammi-ai/src/model/resolver.rs`) whose row's `model_type` is
NOT `fine-tuned` is refused by name — the backstop for a catalog a pre-fix build already
corrupted, since nothing else ever mints that prefix. A record whose `model_type`
(`crates/jammi-ai/src/model/resolver.rs`) is `fine-tuned` and missing `base_model_id`
(`crates/jammi-ai/src/model/resolver.rs`) or missing `artifact_path`
(`crates/jammi-ai/src/model/resolver.rs`) is refused with a typed error naming the model
id and the missing field, never silently resolved as an ordinary model or served as the
unadapted base.

`load_context_predictor`'s own id-shape backstop
(`record.model_type`, `crates/jammi-ai/src/pipeline/context_predictor.rs`)
mirrors the resolver's `FINE_TUNED_ID_PREFIX` cross-check, but a context-predictor id is
caller-chosen — it carries no reserved prefix a fresh reload can cross-check by shape the way
`try_catalog_lookup` does — so this surface asserts its own row-shape invariant directly,
immediately after reading the row back: any record read under a context-predictor id whose
`model_type` is not `"context-predictor"` is refused by name, naming the id and the row's
actual type, before any of that row's `config_json`/`artifact_path` fields are trusted. This is
the id-shape backstop's other member — the resolver's prefix check defends the `jammi:fine-tuned:`
id space, this defends the context-predictor id space, and each surface owns its own check
rather than trusting the id's shape alone.

The adapter-fetch error contract both reload surfaces share: `fetch_artifact`
(`crates/jammi-db/src/store/artifact.rs`) raises two DISTINCT typed storage outcomes,
never folding them together. A manifest that is ABSENT entirely — nothing was ever
published at that prefix, or a catalog pointer names the wrong one — reclassifies to
`StorageError::NotPublished` (`reclassify_missing_manifest`,
`crates/jammi-db/src/store/artifact.rs`; covered by
`missing_manifest_is_not_published_not_corruption`,
`crates/jammi-db/src/store/artifact.rs`): no manifest is in hand, so there is nothing
to say is corrupt. A manifest that IS present but malformed, or that names a key which is
missing or hash-mismatched on an otherwise-published bundle, is the genuine integrity
failure, `StorageError::Layout` (`reclassify_missing_key`,
`crates/jammi-db/src/store/artifact.rs`; `verify_sha256`,
`crates/jammi-db/src/store/artifact.rs`). Any OTHER storage fault off `fetch_artifact`
— transport/IO, a disabled scheme, driver-init failure, or a permission-denied open on a
present key (which stays `StorageError::Io`, never reclassified —
`permission_fault_on_a_present_key_stays_a_transport_error`,
`crates/jammi-db/src/store/artifact.rs`) — is left unchanged.

Both reload surfaces match on these two variants explicitly and re-type BOTH into the SAME
`JammiError::Model`, naming the model id with a distinct message per variant.
`try_catalog_lookup` (`crates/jammi-ai/src/model/resolver.rs`), `ModelResolver`'s
fine-tuned reload arm, matches `StorageError::NotPublished`
(`crates/jammi-ai/src/model/resolver.rs`) and `StorageError::Layout`
(`crates/jammi-ai/src/model/resolver.rs`) into `JammiError::Model`, and
`load_context_predictor` (`crates/jammi-ai/src/pipeline/context_predictor.rs`) matches
the identical pair — `StorageError::NotPublished`
(`crates/jammi-ai/src/pipeline/context_predictor.rs`) and `StorageError::Layout`
(`crates/jammi-ai/src/pipeline/context_predictor.rs`) — into `JammiError::Model` as
well, never its own `JammiError::Inference`. A catalog record that never recorded an
`artifact_path` at all is a separate, earlier refusal on each surface that never reaches
`fetch_artifact` — the resolver's arm also raises `JammiError::Model`
(`crates/jammi-ai/src/model/resolver.rs`), and so does the predictor's own
`JammiError::Model` (`crates/jammi-ai/src/pipeline/context_predictor.rs`). Any OTHER
storage fault propagates unchanged past both surfaces' own catch-all —
`Err(e) => return Err(e)` (`crates/jammi-ai/src/model/resolver.rs`) and the identical
`Err(e) => return Err(e)` (`crates/jammi-ai/src/pipeline/context_predictor.rs`).

Every corrupted-catalog-record refusal EARLIER in this reload path — before `fetch_artifact` is
even reached — is the SAME `JammiError::Model` variant too: an
absent `config_json`
(`crates/jammi-ai/src/pipeline/context_predictor.rs`), an unparseable `config_json`
(`crates/jammi-ai/src/pipeline/context_predictor.rs`, a DISTINCT message from "absent",
never collapsed), and a parseable-but-incomplete config (missing `head`/`architecture`/
`feature_dim`/`context_k`/`hidden_dim`/`num_heads`/`num_layers`/`head_width`/`value_column`/
`target_scaler`) each name the model id and the specific field. The `varmap.load` arm — a
manifest-verified bundle missing `model.safetensors`
(`crates/jammi-ai/src/pipeline/context_predictor.rs`) — matches `CandleBackend::load`'s
peer refusal for a fine-tuned model's weights file, `"Failed to load safetensors: {e}"`
(`crates/jammi-ai/src/model/backend/candle.rs`), instead of its own
`JammiError::Inference`.

At the gRPC edge, `map_engine_error` (`crates/jammi-server/src/grpc/wire.rs`) maps
`JammiError::Model` (`crates/jammi-server/src/grpc/wire.rs`) to `Code::InvalidArgument`,
maps `JammiError::Inference` (`crates/jammi-server/src/grpc/wire.rs`) to
`Code::Internal`, and lets every unmatched variant — including the propagated
`JammiError::Storage` transport fault — fall through its own catch-all to `Code::Internal`
(`crates/jammi-server/src/grpc/wire.rs`). Because both reload surfaces raise the same
`JammiError::Model` for the same class of outcome, an unpublished OR a corrupted adapter
bundle reads as the SAME `InvalidArgument` whether it is `ModelResolver` or
`load_context_predictor` that hit it, and a genuine transient object-store outage on either
surface reads as the SAME `Internal` — never conflated with the client-visible precondition
failure.

Two DISTINCT proofs pin this, at two DISTINCT layers, and neither substitutes for the other.
The it-tests pin the first layer — *surface → variant on a real resolve*: that
`ModelResolver::try_catalog_lookup` and `load_context_predictor`, driven end-to-end against a
real corrupted/unpublished/permission-faulted bundle on disk, actually PRODUCE the claimed
variant. `crates/jammi-ai/tests/it/models.rs::fine_tuned_adapter_bundle_permission_fault_is_not_a_typed_model_error`
and its context-predictor twin, `crates/jammi-ai/tests/it/context_predictor.rs::context_predictor_reload_permission_fault_is_not_a_typed_model_error`,
inject a real `chmod 0o000` fault against an intact bundle and assert the EXACT variant each
surface raises — `matches!(err, JammiError::Storage(StorageError::Io { .. }))` — never merely
`!matches!(err, JammiError::Model { .. })`, which would pass for any other storage variant
too and prove nothing about which one the surface actually hit.
`crates/jammi-server/src/grpc/wire.rs::tests::adapter_bundle_refusal_codes_agree_across_both_reload_surfaces`
pins the second layer — *variant → code at the wire boundary*: that each of those SAME
variants, once produced (a corrupted pointer, a manifest-verified integrity failure, an
unpublished bundle, and — the SAME `StorageError::Io` the it-tests above prove each surface
actually raises, not a stand-in like `StorageError::DriverInit` — a transport/permission
fault), maps to the right gRPC code on both surfaces: resolver bad-pointer / integrity /
not-published and predictor integrity / not-published all map to `InvalidArgument`; resolver
transport and predictor transport both map to `Internal`.

### 3.7 Crash recovery of building tables

`ResultStore::recover` (`crates/jammi-db/src/store/mod.rs`), at session startup: lists
`Building` rows; for each — Parquet missing → `Failed`; Parquet invalid → delete it +
sidecar, `Failed`; Parquet valid → recount rows, **rebuild the ANN index from Parquet**
(`rebuild_index_from_parquet`), mark `Ready`. **Invariant: a `building` row is a crash
artifact; the Parquet is the source of truth, the sidecar is always rebuildable.** Index
rebuild failure is logged but does not fail recovery — the table goes `Ready` and falls back
to exact search.

### 3.8 Remote verbs (the mirror)

- **Remote `infer`:** `DataClient::infer` (`crates/jammi-client/src/lib.rs`) →
  `transport.service` builds the stub, the `SessionHeader` interceptor stamps the session id
  → the async `TenantResolverLayer` resolves the scope and inserts `SessionTenant` →
  `InferenceServer::infer`
  (`crates/jammi-server/src/grpc/inference.rs`): `session_tenant` → `require_nonempty` →
  `scoped(&self.session, tenant, || session.infer(...))` → `infer_result_to_proto` → client
  `decode_ipc_stream` (or `error_from_status`).
- **Remote `sql` (Flight SQL lane):** `DataClient::sql` (`crates/jammi-client/src/lib.rs`)
  opens a `FlightSqlServiceClient` over the *same* tonic channel and stamps the **bound**
  session id (not a fresh one) → server `TenantBoundProvider` scopes the query to the bound
  tenant (shared-binding path, [§7]).
- **Eval IR metrics (numerics path):** `EvalRunner` encodes each query →
  `result_store.search_vectors` → `cosine_distance` (top-k heap, sorted `(dist,_row_id)`) →
  `RetrievalMetrics::recall_at_ks`/`compute_query`/`aggregate` → serialize through
  `crates/jammi-wire/src/eval/report.rs`.

### 3.9 InferenceService.Predict — the non-delegating serving verb, and where conformal does not live

**Two verbs land on `InferenceService`, and only one is a thin delegate.** `Infer` is the
mirror-shaped verb [§3.8]: proto in → one `Session::infer` call → proto out
(`crates/jammi-server/src/grpc/inference.rs`, `InferenceServer::infer`), reimplementing no
scan or forward logic — `infer` itself is a one-line delegate
(`crates/jammi-ai/src/local_session.rs`, `Session::infer` → `InferenceSession::infer`,
`crates/jammi-ai/src/session.rs`). `Predict`
(`crates/jammi-server/src/grpc/inference.rs`, `InferenceServer::predict`) is the **heavier,
non-delegating** handler. There is no `Session::predict` verb to delegate to: served
prediction is a two-call composition on the engine session (`InferenceSession`), not a single
transport-agnostic verb, so the handler reaches `self.session` directly rather than through
the `Session` wrapper it uses for `infer`.

**Contract — proto surface.** `rpc Predict(PredictRequest) returns (PredictResponse)`
(`crates/jammi-wire/proto/jammi/v1/inference.proto`). Request carries `model_id`, `source`
(the corpus whose embedding table the *live* context is drawn from; need not equal the
training source), `target_key`, optional `split` predicate, optional `EdgeGather edges`,
optional `hybrid_ann_k`. Response is a `oneof { Gaussian | Quantile } distribution`, a
`string source` assembly tag, and `repeated string context_ref`. **The response carries NO
conformal interval** — only the bare predictive distribution plus its provenance.

**The context-serve source it reconstructs.** The handler rebuilds the `ContextServeSource`
enum (`crates/jammi-ai/src/pipeline/context_predictor.rs`) inline from the request's
edge/hybrid fields (`crates/jammi-server/src/grpc/inference.rs`): absent gather ⇒
`ContextServeSource::Ann`; gather with no `hybrid_ann_k` ⇒ `ContextServeSource::Edges(edges)`;
gather with `hybrid_ann_k` ⇒ `ContextServeSource::Hybrid { ann_k, edges, merge: Union }`. This
`match` is a **verbatim duplicate** of the embed (PyO3) binding's reconstruction in
`crates/jammi-python/src/database.rs` — same three arms, same `HybridMerge::Union`. That
symmetry is the "one definition, two transports" property applied to a verb that the wire does
*not* model as a single `Session` method: each surface reassembles the engine type from its own
request shape, so the duplication is structural, not accidental. `ContextServeOptions { source,
split }` (`crates/jammi-ai/src/pipeline/context_predictor.rs`) wraps it; default is `(Ann, no
split)`.

**Data-flow / call-chain (the served-prediction path).** `InferenceServer::predict`
(`crates/jammi-server/src/grpc/inference.rs`) → `session_tenant` + `require_nonempty(model_id,
source, target_key)` → `edge_gather_from_proto(req.edges)`
(`crates/jammi-ai/src/wire/inference.rs`) → reconstruct
`ContextServeSource`/`ContextServeOptions` → inside `scoped(&self.session, tenant, …)`:
1. `InferenceSession::load_context_predictor(model_id, source, options)`
   (`crates/jammi-ai/src/pipeline/context_predictor.rs`) — reads the model's catalog
   `config_json`, rebuilds the `AnyContextPredictor` (Cnp/AttnCnp/Tnp) into a fresh `VarMap`,
   loads persisted safetensors; returns an inference-only `ServedContextPredictor` (forward
   never mutates the varmap).
2. `predict_with_context_predictor_provenanced(&served, target_key)`
   (`crates/jammi-ai/src/pipeline/context_predictor.rs`) — reads the target's stored vector,
   builds a `ContextRequest` whose `source` is `serve.source.to_context_source(context_k)`,
   `exclude_self`/`exclude_key` the target, hydrates the members' `value_column` outcomes,
   `assemble_context` (`crates/jammi-ai/src/pipeline/context_set.rs`), reads member vectors,
   z-scores member `y` with the persisted `TargetScaler`, pads the episode, runs one in-context
   `forward`, then **de-standardises the distribution** back to raw units. Returns
   `PredictionWithProvenance { distribution, source: ContextSourceKind, context_keys }`.
→ `predicted_distribution_to_proto(&prediction.distribution)`
(`crates/jammi-ai/src/wire/inference.rs`) + `context_source_tag(prediction.source)` →
`"ann"|"edges"|"hybrid"` (`crates/jammi-server/src/grpc/inference.rs`) → `PredictResponse`.

**Invariants.** (a) Tenant scope is a task-local installed by `scoped`; both engine calls run
inside the same closure so the load and the forward observe one tenant. (b) Inference-only:
served weights are byte-identical before/after. (c) Never-unattributed coverage: the assembly
`source` fact and the neighbour `context_keys` always ride out of the serving layer — `Predict`
uses the **provenanced** form, not the bare `predict_with_context_predictor`. (d) The serving
`value_column` z-scoring uses the *train-derived* scaler; a config without it is a typed reload
error, never a silent identity de-standardisation.

**Where `predict/conformal.rs` feeds — and its wiring status.**
`crates/jammi-ai/src/predict/conformal.rs` (`ConformalModel`; `ConformalModel::regression` /
`_mondrian` / `_weighted`; `predict_interval`; `IntervalScore::AbsoluteResidual|Cqr`) is the
distribution-free coverage primitive. It is **not on the `Predict` data-flow above** — the gRPC
handler returns the raw distribution and the response proto has no interval field. The
served-predictor conformal bridge is a *separate* surface: `ConformalContextPredictor`
(`crates/jammi-ai/src/pipeline/context_predictor.rs`) + its calibrator
`InferenceSession::calibrate_context_predictor_conformal`, which re-runs
`predict_with_context_predictor` over a held-out `(target_key, observed_y)` calibration set,
picks the `IntervalScore` from the head form — `Gaussian ⇒ AbsoluteResidual`, `Quantile ⇒ Cqr` —
and feeds `ConformalModel::regression{,_mondrian,_weighted}` under the caller-chosen
`ConformalLevers`. Serving an interval is then `ConformalContextPredictor::interval(dist,
group)`.

- **`calibrate_context_predictor_conformal` / `ConformalContextPredictor::interval`: DORMANT**
  from every runtime serving path. Their only callers are integration tests
  (`crates/jammi-ai/tests/it/context_predictor.rs`). No gRPC handler, no `local_session`/`Session`
  method, and no `jammi-python` binding call them. A maintainer might assume the served-predict
  path emits coverage intervals because the conformal wrap is written specifically for
  `ServedContextPredictor`; it does not — the wrap exists but is unwired into any served verb.
- **`ConformalModel` (the primitive): CALLER-DRIVEN.** Its only *runtime-reachable* callers are
  the `jammi-python` embed utilities `conformalize` (classification, via
  `ConformalModel::classification`) / `conformalize_interval` (`ConformalModel::regression` +
  `AbsoluteResidual`) / `conformalize_cqr` (`ConformalModel::regression` + `Cqr`)
  (`crates/jammi-python/src/database.rs`), each constructing a `ConformalModel` per call from
  caller-supplied calibration arrays and returning a set (`conformalize`) or intervals (the other
  two). These are standalone PyO3 methods, not part of the `Predict` (context-predictor) path, and
  have **no gRPC peer** — the conformal primitive is exposed only embedded, for the caller to
  invoke directly. Its *other* in-crate caller is the DORMANT
  `calibrate_context_predictor_conformal` bridge above, which is test-only. (`evidence/conformal.rs`
  references the primitive only in rustdoc, not a call.)

**Why `Predict` is the lone non-delegating handler.** Every other engine-backed gRPC verb has a
matching transport-agnostic `Session` method to delegate to (the single-verb mirror, [§3.8]).
Served prediction does not: it is the composition *load-then-forward* over `InferenceSession`,
plus the request-shaped `ContextServeSource` reconstruction the wire cannot express as one verb.
So the handler does the assembly the embed binding otherwise does — making `Predict` the one place
server-side that reconstructs an engine type inline rather than calling a named verb.

**Extension note.** To put conformal coverage on the wire, the new surface is *not* a tweak to
`Predict`: it needs (1) a calibration submission/verb feeding
`calibrate_context_predictor_conformal`, (2) a persisted `ConformalModel` keyed to the model id,
and (3) interval fields added to `PredictResponse`
(`crates/jammi-wire/proto/jammi/v1/inference.proto`) served via
`ConformalContextPredictor::interval`. Until then, conformal-over-served-prediction stays
test-only: neither a remote nor an embedded `predict` returns a coverage-guaranteed interval.

---

## 4. Extension playbooks

Each recipe names exact files in order. **All of these ship atomically in one PR** [§5
atomic rule].

### 4.1 Add a new wire verb / typed gRPC RPC (the big one — touches the most files)

The embedded path and the remote path build the *same proto request* in shared pure-Python
and decode it through **one** `jammi_ai::wire::*_from_bytes` seam.

**The seam, stated once.** A wire verb's request→engine field map lives in exactly **two
halves that meet at the proto**:

- **`*_from_proto(req)`** — decodes a *decoded* proto message into engine args, validating
  required fields. Called by the **gRPC server handler** (which already holds a decoded
  `Request<…>`).
- **`*_from_bytes(body)`** — `prost::Message::decode` the wire body, then call
  `*_from_proto`. Called by the **embedded PyO3 primitive** (which is handed serialized bytes
  from Python). Example pair: `crates/jammi-ai/src/wire/inference.rs` (`infer_from_bytes` →
  `infer_from_proto`); both exported from `crates/jammi-ai/src/wire/mod.rs`.

So the kwargs→proto map lives **once** in Python
(`clients/python/jammi/_assembly.py`, shared by remote *and* embedded) and the
proto→engine map lives **once** in Rust (`jammi_ai::wire`). The PyO3 layer is a thin set of
`_<verb>_proto(bytes)` primitives.

**Touch-points, in order. All ship atomically in one PR [§5].**

1. **Proto.** Add message(s) + `rpc` to `crates/jammi-wire/proto/jammi/v1/*.proto` (mirror
   `inference.proto`; `rpc Infer(InferRequest) returns (InferResponse)`). A brand-new
   *service file* must also be appended to the `proto_files` list in
   `crates/jammi-wire/build.rs`. Codegen always builds **both** client and server stubs
   (`.build_client(true).build_server(true)`; the generated packages mount at
   `crates/jammi-wire/src/proto.rs`).

2. **Wire conversions — pick the right crate by what the converter touches.**
   - Candle-free, pure proto↔domain (the `request.rs` vocabulary, IPC framing, `ModelTask`
     mapping at `crates/jammi-wire/src/lib.rs`): in `crates/jammi-wire/src/<surface>.rs`,
     re-exported from `crates/jammi-wire/src/lib.rs` (keep **both** directions — orphan
     rule).
   - Anything touching engine-spec vocabulary (`TrainingSpec`, pipeline structs,
     `EdgeGather`, `PredictedDistribution` — needs candle types): in
     `crates/jammi-ai/src/wire/<surface>.rs` (gated behind the `local` feature), re-exported
     from `crates/jammi-ai/src/wire/mod.rs`. **This is where you add the `*_from_proto` /
     `*_from_bytes` pair** (the seam above). `prost` rides jammi-ai's `local` feature so the
     proto type never leaks into the PyO3 crate.

3. **Engine verb.** Implement on `crates/jammi-ai/src/local_session.rs` (the `Session`
   façade), delegating to `InferenceSession`. Server handlers and the embedded binding both
   call `Session`/`InferenceSession`, **never** a hand-rolled path. (Training is special: a
   *single* dispatch `InferenceSession::run_training_spec`
   (`crates/jammi-ai/src/session.rs`) is shared by the gRPC `JobService::submit_job`
   handler and the embedded binding — add new `TrainingSpec` handling there, not in two
   places.)

4. **Server handler.** `crates/jammi-server/src/grpc/<svc>.rs`, fixed shape — copy
   `InferenceServer::infer` (`crates/jammi-server/src/grpc/inference.rs`):
   `let tenant = session_tenant_traced(&request);`
   → `let args = jammi_ai::wire::<verb>_from_proto(request.into_inner())?;`
   → `scoped(&self.session, tenant, || session.<verb>(...)).await.map_err(map_engine_error)?`
   → encode the response. The helpers (`session_tenant_traced`, `require_nonempty`, `scoped`,
   `map_engine_error`) come from `crate::grpc::wire`. The handler decodes through the
   **same** `*_from_proto` seam the embedded binding's `*_from_bytes` drives.

5. **Mount it.** In `serve_grpc_chain` (`crates/jammi-server/src/runtime.rs`) add `builder =
   builder.add_service(FooServiceServer::with_interceptor(FooServer::new(...),
   interceptor.clone()));`, then `mounted.push("FooService");`, plus the `use` import. Place
   it in the right block: **engine-backed** services go under `if let Some(session) = engine
   {` (e.g. Embedding/Inference/Pipeline/Audit); **tier-gated** services go inside that block
   under their `if tiers.contains(ServiceTier::…)` guard (Eval, Train).

6. **Remote mirror (Rust client).** Add the method to `DataClient`
   (`crates/jammi-client/src/lib.rs`) or `CatalogClient` (`crates/jammi-admin/src/lib.rs`):
   build the stub via a `fn <svc>_client()` helper (copy the `inference_client` pattern in
   `crates/jammi-client/src/lib.rs`), map request→proto, send, and decode the structured
   error with `error_from_status`. Add a new `fn <svc>_client()` helper only if the service is
   new.

7. **CLI** (control verbs only). Subcommand under `crates/jammi-cli/src/commands/`, wired into
   the `Commands` enum (`crates/jammi-cli/src/main.rs`) and the `dispatch` match. Data-plane
   verbs do not get a CLI surface.

8. **Python — both wheels, one assembly.**
   - **Shared request assembly (the field map, written once):** add a
     `build_<verb>_request(...)` to `clients/python/jammi/_assembly.py`. Both wheels
     import it (remote: `clients/python/jammi/_database.py`; embedded:
     `clients/python/jammi/_embedded.py`). Surface-only validation (e.g. the graph-only
     embedding-loss guard, `output='quantile' requires levels`) lives here too.
   - **Embedded (`jammi`):** the `EmbeddedBackend` (`clients/python/jammi/_embedded.py`)
     calls `build_<verb>_request(...)`, serializes, and hands the bytes to **one** PyO3
     primitive `_<verb>_proto(bytes)` on `crates/jammi-python/src/database.rs` (e.g.
     `_infer_proto`, `_start_training_proto`), which decodes via
     `jammi_ai::wire::<verb>_from_bytes`. `jammi-python` pulls in **no tonic transport/server
     stack** — it depends on tonic **only for the `tonic::Status` type** the wire-decode seam
     returns (`crates/jammi-python/Cargo.toml`), plus the wire converters + `prost` (behind
     jammi-ai's `local`).
   - **Remote (`jammi.RemoteDatabase`):** the matching method in
     `clients/python/jammi/_database.py` builds via the same `build_<verb>_request`
     then sends over gRPC. **`import jammi` stays native-free** (it imports `jammi_native` lazily, only when a
     `file://` target is opened) — a CI import-direction guard enforces this.
   - `jammi.connect("file://…")` returns the `EmbeddedBackend`; `connect("grpc://…")`
     returns `RemoteDatabase` (`clients/python/jammi/__init__.py`).

9. **Tests + the four guards.** Integration test in `crates/jammi-server/tests/it/` (one file
   per service, registered in `crates/jammi-server/tests/it/main.rs`). Then update the guards a
   new verb trips:
   - **`crates/jammi-python/tests/test_conformance.py`** — pins remote==embedded verb sets and
     identical signatures (`_REMOTE_VERBS`;
     `test_embed_remote_and_client_share_identical_signatures`). A data-plane verb on both
     wheels must appear here.
   - **`crates/jammi-server/tests/it/api_freeze.rs`** — the terminal-0.x freeze guard decodes
     the compiled `FILE_DESCRIPTOR_SET` and asserts the live `(Service, Method)` rpc set EQUALS
     `crates/jammi-server/tests/it/api_freeze_baseline.txt`. **Adding an rpc fails CI until you
     append the matching `RPC <Service>/<Method>` line to the baseline in the same PR** (the
     additive, minor-compatible case made explicit in the diff).
   - **TypeScript client** — see §4.1a (regenerate + extend `surface.test.ts`).
   - **The cookbook chapter guard** — see step 10, mandatory.

10. **The cookbook step — *a verb is not done until its chapter ships and the API guard counts
    it*** (the cookbook is consolidated in-monorepo at `cookbook/book/`).
    - **Add a chapter.** A new recipe directory + `.qmd` under `cookbook/book/chapters/` (e.g.
      `cookbook/book/chapters/20-recompute/recompute.qmd`), listed in the book TOC
      `cookbook/book/_quarto.yml` under the right `part:`.
    - **Bump the API-reference guard count.** Add the verb (as a key → list of the kwargs the
      recipe relies on) to the `REQUIRED` dict in `cookbook/book/scripts/check_api_reference.py`.
      **Mechanism:** the script `connect()`s an ephemeral `file://` engine, resolves each
      `REQUIRED` key as a **bound method on the live instance** (looking through the
      `Database`→`_NativeDatabase` composition), and `inspect.signature`-checks that every listed
      kwarg is a real parameter — failing CI loudly if a signature drifted. The **"guard count"**
      is `checked = len(REQUIRED) + len(MODULE_FUNCTIONS)` (printed as "N surfaces checked"):
      **adding your verb's `REQUIRED` entry IS the count bump** — there is no separate
      magic-number to edit. (`MODULE_FUNCTIONS = ["open_local", "connect"]`.) The guard runs in CI
      on the in-repo book workflow (`.github/workflows/cookbook-book.yml`), so a verb whose chapter
      calls a kwarg the wheel doesn't expose reds CI before render.

**Retrieval-status reality — do not wire a verb into these expecting them live.** The served
`search` verb is **dense-ANN only**: `resolve_search_mode` (`crates/jammi-db/src/store/mod.rs`)
returns `Option<SidecarIndex>` and names no lexical/hybrid/RRF mode. **`LexicalIndex`**
(`crates/jammi-ai/src/index/lexical.rs`) is **DORMANT** — constructed *nowhere* outside its own
module and tests. **`rrf_fuse`** (`crates/jammi-ai/src/query/rrf.rs`, re-exported
`crates/jammi-ai/src/query/mod.rs`) is **CALLER-DRIVEN client-side numerics**, not part of the
served search path: exposed as a stateless verb (`crates/jammi-python/src/database.rs`) computed
locally (`clients/python/jammi/_conformal.py`). **Conformal** (`conformalize` /
`conformalize_interval` / `conformalize_cqr`) is likewise **CALLER-DRIVEN client-side numerics**
(`crates/jammi-python/src/database.rs`; remote computes locally via
`clients/python/jammi/_conformal.py`) — it does **not** ride the wire and does **not** wrap
the served `Predict` predictor [cf. §3.9]. These three sit in `check_api_reference.py`'s `REQUIRED`
(they are real instance methods) but are not gRPC verbs, so they need **no**
proto/handler/mount/freeze-baseline work — only steps 8 and 10.

### 4.1a The TypeScript client codegen step

§4.1 enumerates Rust/CLI/Python touchpoints for a new wire verb; the **TypeScript client** —
`clients/typescript` — mirrors the same proto, so a new `rpc`/message must be regenerated and
re-guarded there too, **in the same atomic PR** [§5].

**Contract: the TS surface is generated, never vendored.** The canonical proto at
`crates/jammi-wire/proto/jammi/v1/*.proto` is the *single* source for every language client
(`clients/typescript/buf.gen.yaml`). The generated code is **gitignored and never committed** —
`clients/typescript/src/gen/` (`.gitignore`) and `dist/`. Neither path is tracked (`git ls-files
clients/typescript/src/gen` → 0 entries). So the codegen step is a *build action*, not an
edit-and-commit step.

**The regen command.** From `clients/typescript/`:

```
npm run generate      # = buf generate ../../crates/jammi-wire/proto   (package.json)
```

`buf generate` reads `buf.gen.yaml` (`version: v2`) and runs the single plugin
**`protoc-gen-es`** (protobuf-es v2) → `out: src/gen`, `target=ts`. There is **no `buf.yaml`
module file** — the proto dir is passed directly as the buf input argument. protoc-gen-es emits
both message types **and** the `GenService` descriptors; Connect-ES's `createClient` consumes those
descriptors directly, so **there is no separate connect plugin**. `generate` is wired as a
**pre-step on every build, typecheck, and test** (`prebuild`/`pretypecheck`/`pretest`,
`clients/typescript/package.json`), so the regen is automatic — but a verb whose proto changed
still requires re-running these so the new descriptors exist.

**Files the maintainer touches by hand (the thin seam).** The generated `*_pb.ts` are NOT
hand-edited. The only hand-written file is `clients/typescript/src/index.ts`, and it only needs an
edit when a **whole new service** is added (not for a new RPC on an existing service): per-service
`import { FooService } from "./gen/jammi/v1/foo_pb.js"`, a matching `export *` re-export, a field on
the `JammiClient` interface, and a `createClient(FooService, transport)` arm in `connect()`. A new
RPC on an existing service needs **zero** hand edits — it appears on the generated service descriptor
automatically.

**Guarding tests (what fails if you skip regen).**
- **`clients/typescript/test/surface.test.ts`** — the verb-surface guard. `verbSurface`
  references **every service's every RPC** with a typed request inside an always-false `if
  (Math.random() < 0)` branch; `tsc` still type-checks the body, so a **missing verb or a drifted
  field shape fails the typecheck**. `connect()` is also asserted to return a client for all
  services and to mint a fresh v4 session id per connection. **Add your new verb's reference into
  `verbSurface`** under the right service block, or the surface is not actually proven. This test is
  **LIVE** in CI (the `pretest` hook regenerates first).
- **There is no TS analogue of `test_conformance.py`** — `surface.test.ts` is the only TS guard;
  it is a *compile-level completeness* proof, not a runtime remote==embedded cross-check.

**CI wiring (LIVE).** Job **`ts-client`** in `.github/workflows/ci.yml` (runs on plain
`ubuntu-latest`, Node 22, NOT the Rust container): `npm ci` → `npm run build` (which runs `buf
generate` via `prebuild`, then `tsc`) → `npm run typecheck` → `npm run test` (vitest, hermetic). Its
stated purpose is to catch "a proto change that breaks TS codegen" on every PR, not only at the
release tag. Release-time publish is `.github/workflows/npm.yml` (same `npm run build/typecheck/test`
then `npm publish --provenance`). `clients/typescript/package.json` is one of the lockstep version
files [§6 release].

**Wiring-status caveat — `test_generated_floor.py` is NOT a TS guard.**
`clients/python/tests/test_generated_floor.py` is a **Python-client** structural guard: it parses
the grpcio/protobuf import-time version guards out of the freshly generated `*_pb2.py`/`*_pb2_grpc.py`
stubs (`clients/python/jammi/_generated/jammi/v1/`) and asserts `pyproject.toml`'s declared
floors satisfy them. It has **nothing to do with the TypeScript client** (protoc-gen-es emits no
runtime version guard). The TS guard is `surface.test.ts` alone.

**Step to add to §4.1 (TS client):** *After step 8 (Python), before step 9 (Tests):* regenerate and
guard the TS client — in `clients/typescript/` run `npm run generate` (= `buf generate
../../crates/jammi-wire/proto`, emitting the gitignored `src/gen/*_pb.ts` via protoc-gen-es); for a
**new service** also add the import / `export *` / `JammiClient` field / `createClient` arm in
`clients/typescript/src/index.ts`; then extend the verb-surface guard
`clients/typescript/test/surface.test.ts` with the new RPC and run `npm run typecheck && npm run
test`.

**Publish-exclusion clarification.** The workspace has 13 members but the release publishes only 11
crates in topological order (`.github/workflows/crates.yml`: jammi-numerics → jammi-db →
jammi-kernels → jammi-lora → jammi-encoders → jammi-wire → jammi-admin → jammi-client →
jammi-ai → jammi-server → jammi-cli). The 3 unpublished members carry `publish = false`
in their manifests: **`jammi-python`**
(`crates/jammi-python/Cargo.toml` — PyO3 cdylib shipped as a maturin wheel via
`.github/workflows/pypi.yml`, not crates.io), **`jammi-test-utils`**
(`crates/jammi-test-utils/Cargo.toml`), and **`jammi-bench`** (`crates/jammi-bench/Cargo.toml` — a
measurement consumer kept out of the published workspace). This differs from `default-members`, which
excludes only `jammi-python` and `jammi-test-utils` but **does** include `jammi-bench` — "default-member"
and "published" are two different exclusion sets.

### 4.2 Add a new source type

1. Variant on `SourceType` (`crates/jammi-db/src/source/mod.rs`, serde snake_case).
2. `crates/jammi-db/src/source/<newtype>.rs` exposing `async fn
   create_<newtype>_tables(source_id, &SourceConnection) -> Result<Vec<(String, Arc<dyn
   TableProvider>)>>` — mirror `crates/jammi-db/src/source/postgres.rs`
   (`create_postgres_tables`). (File-shaped backends instead extend `FileFormat` +
   `create_listing_table`.)
3. Declare the module (feature-gate heavy deps) in `crates/jammi-db/src/source/mod.rs`.
4. Dispatch arm in `JammiSession::register_source_tables` (`crates/jammi-db/src/session.rs`),
   including the `#[cfg(not(feature=…))]` "requires feature" error arm.
5. New connection knobs → fields on `SourceConnection` (`crates/jammi-db/src/source/mod.rs`);
   they JSON-round-trip via `sources.options` automatically. Consider a `tenant_column`.
6. Feature in `crates/jammi-db/Cargo.toml` + integration test in
   `crates/jammi-db/tests/it/sources.rs`. Follow the existing `postgres = [...]` / `mysql =
   [...]` feature pattern that gates the optional `datafusion-table-providers` dependency. *Do
   not* add external SQLite via `datafusion-table-providers` (rusqlite link-version conflict,
   `crates/jammi-db/src/source/mod.rs`) — route SQLite through File+Parquet.

### 4.3 Add a new catalog migration / column

1. `MIGRATION_0NN_*` const in `crates/jammi-db/src/catalog/schema.rs` (DDL portable across
   SQLite+Postgres; non-additive SQLite changes use the create/copy/drop/rename dance — see the
   existing non-additive migrations).
2. **Append** `("0NN_name", schema::MIGRATION_0NN_*)` to the end of `MIGRATIONS` in
   `crates/jammi-db/src/catalog/migrations.rs`. **Never renumber/reorder.**
3. New status value set → extend a typed enum in `crates/jammi-db/src/catalog/status.rs` with
   `Display`+`FromStr`+round-trip test.
4. Update every repo and caller in the same PR.

### 4.4 Add a new ANN index backend (e.g. FAISS)

1. `impl VectorIndex` in a new file under `crates/jammi-db/src/index/` + `pub mod` in
   `crates/jammi-db/src/index/mod.rs`. Honour the contract (keyed by `_row_id`, returns cosine
   distance ascending, `build()` after `add()`s).
2. Persistence: `SidecarKind` variant + extension set in
   `crates/jammi-db/src/storage/sidecar_layout.rs`, plus save/load arms in
   `save_sidecar`/`load_sidecar`.
3. **Dispatch (the real work):** `ResultStore::resolve_search_mode`
   (`crates/jammi-db/src/store/mod.rs`) today returns concrete `Option<SidecarIndex>`. Widen to
   `Option<Box<dyn VectorIndex>>` (or an enum) and update the two call sites:
   `AnnSearchExec::execute` and `ResultStore::search_vectors`. This is the only place the
   abstraction currently leaks the concrete type [§7].
4. Selection key: wire `EmbeddingConfig::default_index_type` (`crates/jammi-db/src/config/mod.rs`)
   — currently dead — through the build site (`crates/jammi-ai/src/pipeline/embedding.rs`).

(Cheapest variant — a **query-time knob** like `search_expansion`: add a field to
`AnnIndexConfig` (`crates/jammi-db/src/config/mod.rs`), map it in `SidecarIndex::index_options`
(`crates/jammi-db/src/index/sidecar.rs`), re-apply on load if query-time-mutable, pin its
default in `crates/jammi-db/src/index/sidecar.rs`.)

### 4.5 Add a new text encoder family

1. `crates/jammi-encoders/src/foo.rs` mirroring `crates/jammi-encoders/src/distilbert.rs`
   (cleanest template): `FooConfig` (serde-renamed to HF field names), per-layer structs
   holding `MaybeLoraLinear` + `crate::layer_norm::LayerNorm`, `Foo` + `FooBuilder` with the
   four builder knobs (`.pooling()`, `.lora()`, `.backbone_dtype()`, `.adapter()`). `build`
   opens `frozen_vb` via `VarBuilder::from_mmaped_safetensors` and `lora_vb` (always **F32**);
   resolve each base tensor in the loader, then hand it to
   `crates/jammi-encoders/src/lora_site.rs`'s `LoraSite::wrap` (or the BERT family's
   in-file equivalent), which runs `should_apply_lora` + `effective_rank` +
   `LoraLinear::new_with_base`. Implement the full §2.5 method surface. End `forward` with
   `pool_and_normalize`. **Pick & document your safetensors key prefix** — and if the
   checkpoint holds more than one tower, give each tower its own adapter key root (§2.5).
2. Register `pub mod foo;` + `pub use foo::{Foo, FooConfig};` in
   `crates/jammi-encoders/src/lib.rs`.
3. Add the `Foo(Foo)` variant to `AnyEncoder` (`crates/jammi-encoders/src/any.rs`) — the
   compiler forces a match arm in every method, including `modality`, `forward_input`,
   `probe_input` and `dtype`.
4. Tests: `crates/jammi-encoders/tests/it/foo.rs` + register in
   `crates/jammi-encoders/tests/it/main.rs`; add a golden/parity fixture if one exists.
5. Wire into the parent crate (same PR): an `EncoderFamily` variant plus its
   `from_config` / `from_adapter_model_type` / `adapter_model_type` / `has_tower` /
   `towers` arms in `crates/jammi-ai/src/model/arch.rs` (§2.7) — that module, not a
   per-call-site `match`, is where the architecture becomes known; then the inference arm
   in `crates/jammi-ai/src/model/backend/candle.rs` (box behind that file's
   `CandleTextForward`, add `"foo"` to the supported-architectures error string) and the
   `(family, task)` arm in `crates/jammi-ai/src/fine_tune/worker.rs`
   (`build_encoder_adapters`). There is no default arm in either dispatch: an unhandled
   pairing must stay a typed refusal.

(A **new pooling strategy** instead: `Pooling` variant
(`crates/jammi-encoders/src/pooling.rs`) + match arm in `pool_and_normalize` + a `*_pool` fn;
auto-available to every encoder.)

### 4.6 Add a new loss

- **Embedding loss:** arm on `EmbeddingLoss` (`crates/jammi-wire/src/fine_tune.rs`) +
  validation; free fn in `crates/jammi-ai/src/fine_tune/trainer.rs` (next to
  `cosine_mse_loss`/`angle_loss`/`mnrl_loss`); wire into the dispatch matching its *batch
  shape* (`dispatch_contrastive_loss` for graded pairs; `compute_loss` for pairs/triplet) —
  wrong shape must be a **typed error**, not a fall-through; route through `matryoshka_wrap` if
  it should be Matryoshka-wrapped.
- **Regression loss:** arm on `RegressionLoss` (`crates/jammi-wire/src/fine_tune.rs`) +
  validation; objective in `crates/jammi-ai/src/fine_tune/regression_loss.rs` (must score
  z-space output vs z-scored target); dispatch in `TrainingLoop::regression_loss`
  (`crates/jammi-ai/src/fine_tune/trainer.rs`); **must update the exhaustive
  `StandardizableHead::for_regression_loss` (`crates/jammi-ai/src/fine_tune/target.rs`) — no
  `_` wildcard, it won't compile until you classify the new arm**; set head width in
  `crates/jammi-ai/src/fine_tune/worker.rs` and `regression_form` in
  `crates/jammi-ai/src/fine_tune/trainer.rs`.
- **Classification loss:** arm on `ClassificationLoss` (`crates/jammi-wire/src/fine_tune.rs`) +
  classify-then-loss path gated in `compute_loss`
  (`crates/jammi-ai/src/fine_tune/trainer.rs`).

### 4.7 Add a new model backend / source / tokenizer (lifecycle)

- **Backend:** `BackendType` variant (`crates/jammi-ai/src/model/mod.rs`); `LoadedModel`
  variant + extend *every* match (`estimate_batch_memory`, `embedding_dim`, `regression_form`,
  `regression_std_scale`, `forward`, …) — no catch-all arm by design;
  `crates/jammi-ai/src/model/backend/<name>.rs` impl `ModelBackend`; register in `Backends`
  (`crates/jammi-ai/src/model/cache.rs`) + construct in `ModelCache::new` + add dispatch in
  BOTH `do_load` and `load_owned_for_test`; teach the resolver to recognize your weights.
  (Cautionary tale: `HttpBackend` does *not* impl `ModelBackend`, so `BackendType::Http` is
  unreachable via the cache, [§7].)
- **Model source:** `ModelSource` variant (`crates/jammi-ai/src/model/mod.rs`) + update
  `Display`/`parse`/`from_canonical` + the `model_type` match in `do_load`; a
  `resolve_<source>` method dispatched in `resolve`.
- **Tokenizer shape:** `TokenizerSource` variant (`crates/jammi-ai/src/model/mod.rs`) +
  `path()` + resolver discovery + `TokenizerWrapper` constructor + backend dispatch.

### 4.8 Add a numerics metric / family

- **New retrieval metric (e.g. MAP@k):** field on `QueryMetrics`
  (`crates/jammi-numerics/src/retrieval.rs`) + mean field on `AggregateMetrics`; compute inside
  `compute_query` **reusing** `top_k`/`relevant_set`/`grade_map` (DRY — the codebase forbids
  re-implementing recall); add the mean to `aggregate`; no RNG / order dependence; **update
  `crates/jammi-wire/src/eval/report.rs` DTOs + the `jammi-ai` runner in the same PR** (field
  change = wire-schema change).
- **New metric family:** `crates/jammi-numerics/src/<family>.rs` (free fns over slices
  returning `Result`, validate up front like `crates/jammi-numerics/src/calibration.rs`) + `pub
  mod` in `crates/jammi-numerics/src/lib.rs` (alphabetical) + test in
  `crates/jammi-numerics/tests/it/<family>.rs`. **No new `jammi-*` dep.**
- **Anything consuming randomness:** take `seed: u64` or `&mut StdRng`, construct via
  `StdRng::seed_from_u64`, **never `thread_rng()`**; sort to a canonical basis before drawing if
  the result must depend only on a multiset (pattern: `bootstrap_ci`).

### 4.9 Workspace / build / release

- **New crate:** `crates/<name>/Cargo.toml` with `version.workspace = true`; add to `members`
  (and `default-members` if a shippable OSS crate); `[workspace.dependencies]` entry pinned to
  the exact version + `path`; insert into the publish topological order in
  `.github/workflows/crates.yml` after every dep; bump in the lockstep version-file set if it
  ships to PyPI/npm.
- **New gated (live) test lane:** empty-list `[features]` entry; gate test code behind `#[cfg(feature
  = "…")]` (never `#[ignore]`); `[[test]]` target with `required-features` if it needs its own
  binary; **skip cleanly** (`tracing::warn`) without the feature; a CI job modeled on
  `test-pg`/`test-broker` + a `--no-run` compile-check in `compile-check-gated`.
- **Cut a release:** PR bumping the version across the lockstep version files
  (`docs/plans/50-open-core-hardening-roadmap/ROADMAP.md`, the version-bump file list) + `cargo
  update --workspace` + `CHANGELOG.md`; run the full gate; on merge tag both `vX.Y.Z` and
  `py-vX.Y.Z`. [§6]

---

## 5. Invariants & gotchas

**The two house rules**

- **Atomic across the workspace.** A behavior change ships across every affected crate in *one
  PR*; split by **capability**, never by **crate** (engine in PR1, ai in PR2 leaves the
  workspace inconsistent between merges). No back-compat shims, no `#[deprecated]`, no compat
  re-exports. A trait change in `jammi-db` includes
  `jammi-ai`/`jammi-server`/`jammi-cli`/`jammi-python` together.
- **Lockstep versioning.** Every publishable crate + every Python/TS/server package ships at
  one `workspace.package.version` (`Cargo.toml`, `[workspace.package] version`). All 13
  manifests use `version.workspace = true`; internal deps pin the same version + a path. The
  embed wheel hard-deps `jammi-client` at the workspace version — a client publish lagging the
  engine leaves it unresolvable.

**Candle split**

- `jammi-wire`/`jammi-admin`/`jammi-client`/`jammi-cli` **must never transitively pull
  candle.** Mechanism: `jammi-lora` is `default-features = false` at the root; candle is opt-in
  via its `candle` feature; `jammi-ai`'s default-on `local` feature flips candle back on. Adding
  a candle-touching dep to any of the four breaks this. Enforced by CI on the **isolated
  per-package build artifact** (`cargo tree` false-positives due to feature unification) [§6].

**Transport / tenancy**

- **Unscoped = all-tenants, silently.** A request with a missing/unknown `jammi-session-id`
  runs unscoped, never erroring (`crates/jammi-server/src/grpc/session.rs`). A client that sets
  `--tenant` but forgets `bind_tenant` reads across tenants. The CLI binds before any verb;
  `DataClient::sql` stamps the bound `session_id`, not a fresh one.
- **Single-session / shared-transport.** `DataClient::over` and `CatalogClient::over` must build
  over the *same* `SessionTransport` (`crates/jammi-client/src/lib.rs`); separate transports
  silently un-scope tenant bindings.
- **Handlers must use `scoped`, never sticky `bind_tenant`** — all handlers share one
  `Arc<InferenceSession>`; `bind_tenant` would race across concurrent requests.
- **gRPC-Web layer order is load-bearing.** `GrpcWebTrailersLayer` *before* `GrpcWebLayer`
  (`crates/jammi-server/src/runtime.rs`); reorder and gRPC-Web error handling breaks (raw gRPC
  unaffected).
- **`TraceContextLayer` sits between `MetricsLayer` and the gRPC-Web layers, on every
  listener path.** `MetricsLayer` → `TraceContextLayer` → `GrpcWebTrailersLayer` →
  `GrpcWebLayer` → … (`crates/jammi-server/src/runtime.rs`, both
  `BoundChain::serve_with_shutdown` and `AssembledChain::into_layered_axum_router`). It opens
  one span per request and continues an incoming W3C `traceparent` via
  `jammi_ai::telemetry::set_parent_from_headers` — reading the raw `http::HeaderMap` BEFORE
  tonic decodes gRPC metadata from the same headers, non-destructively, so tonic's own decode
  downstream is unaffected regardless of layer position; the ordering relative to
  `MetricsLayer`/the gRPC-Web layers is a style choice (mirrors `MetricsLayer`'s placement), not
  a correctness dependency the way the gRPC-Web pair above is.
- **`[server.limits]`'s layer stack is INSIDE the gRPC-Web layers, never tonic's own
  `concurrency_limit_per_connection`/`load_shed` builder knobs.** `RefusalStatusLayer` →
  `GlobalConcurrencyLimitLayer` → `PerConnectionLimitLayer` → `MethodClassLayer` (`crates/
  jammi-server/src/limits.rs`), added via `.layer()` calls AFTER `GrpcWebLayer` in
  `BoundChain::serve_with_shutdown` (`crates/jammi-server/src/runtime.rs`). tonic's own
  per-connection concurrency/load-shed builder methods sit OUTSIDE every user `.layer()`
  call (`tonic-0.14.5/src/transport/server/mod.rs`'s `MakeSvc::call`) — a refusal from
  those would be both uncounted (`jammi_grpc_refused_total` never sees it) and never
  gRPC-Web-framed. Message-size enforcement is NOT a layer at all: every mounted service
  (including Flight SQL) carries its own `max_decoding_message_size`, and that specific
  tonic-codec rejection is `OUT_OF_RANGE`, not `RESOURCE_EXHAUSTED` — verified against
  the vendored tonic 0.14.5 source, not assumed.
- **`as_wire` must equal what's mounted** — the `ServerInfo.services` handshake; a service
  mounted without a tier update lies in the handshake.
- **Faithful errors are a contract.** Every `Status` carries the structured detail via
  `map_*_error`; a bare `Status` loses fidelity on the remote arm.
- **`tenant_id` body fields are vestigial** (clients hard-set `String::new()`) — tenant rides the
  header. Don't start populating them.

**Catalog / storage**

- **Append-only migrations** — renaming/reordering silently re-runs DDL on existing DBs.
- **Tenant reads** filter `tenant_id = <bound> OR tenant_id IS NULL`; NULL rows are **globally
  visible** to every tenant (intentional for seed channels; footgun if a write forgets to bind).
  `tenant_id` is never part of a PRIMARY KEY (Postgres would reject NULL) — it's a UNIQUE
  constraint + a partial unique index `WHERE tenant_id IS NULL`.
- **`TxOptions.read_only` is load-bearing on SQLite** (selects BEGIN mode); **never `block_on` a
  catalog transaction from a runtime worker thread**.
- **`sanitize_model_id` must replace `.`** (a dot makes `with_extension` truncate sidecar
  filenames); **the index base URL carries no extension** (`.idx`); result-table names use nanos
  + 8-char UUID for same-nanosecond uniqueness.
- **A `building` row is a crash artifact**; Parquet is the source of truth, the sidecar is always
  rebuildable [§3.7].

**Index / search**

- **rowmap index == USearch key == insertion order** — anything that reorders or sparsely
  populates `row_map` breaks the key↔id mapping silently (there is no delete on the trait).
- **Search speaks cosine *distance* ascending; the `1.0 - dist` similarity flip happens once** in
  `AnnSearchExec`. New backends must emit distance or they invert the ranking.
- **ANN load failure silently degrades to exact** — correct but slow; the only signal is a
  `warn!`.
- **Metric is hardcoded `Cos`**; `default_distance_metric`/`default_index_type` config is inert
  [§7].
- **`cosine_distance` never yields NaN** (`jammi-db` relies on it); the exact comparator keeps a
  NaN fallback anyway.
- **`search` returns rows, never raw vectors** (philosophy). `search_by_id` resolves the example
  vector inside the engine. A `read_vectors` helper exists but is engine-internal — never expose
  raw vectors on a public/remote surface.

**Encoders**

- **L2 output is a hard contract** — every `forward` returns unit-norm rows; skipping
  normalization silently corrupts cosine similarity.
- **`.contiguous()` after `transpose` is load-bearing** (candle upstream issues) — the comments
  say "must not be removed".
- **`set_training` must toggle LayerNorms too** — eval uses the fused kernel (no defined
  backward); training needs the slow primitive path. Forgetting one yields a working forward but a
  silently-broken backward.
- **Site-name strings are a persistence ABI.** `named_trainable_weights` keys are the adapter
  safetensors keys; the `…lora_sites` helper names (used by dropout-resume) and the inlined
  `named_weights`/`load_weights` prefixes are maintained **independently** — a rename must be
  applied in both places or it silently orphans saved adapters. The same string is also the
  `target_modules` selector a caller writes, so a rename is a user-visible config break too.
- **The cross-modal towers are asymmetric** inside `AnyEncoder`, but only where the
  modality genuinely differs: all three carry real training hooks, and it is the
  *token-sequence* methods that refuse (`forward_hidden` on all three, `max_seq_length` on
  the two media towers). `ClipText` ignores the attention mask (its causal mask is the
  whole story) and pools via EOT-argmax.
- **Two towers of one checkpoint must never share an adapter key root.** `jammi-ai` holds
  one `VarMap` per run and candle's `VarBuilder::get` hands back the already-registered
  `Var` for a name it has seen, so identical keys alias rather than duplicate — half the
  trainable parameters and one gradient stream feeding both towers. The refusal in
  `LoraLinear::new_with_base` (§2.6) is the mechanical guard; the disjoint
  `visual.`-prefixed vision namespace (§2.5) is the design that keeps it from firing.
- **An OpenCLIP adapter must name its tower.** `adapter_config.json` omits the `tower` key
  entirely for a single-tower family (so those files are byte-unchanged), but a
  two-tower checkpoint with `tower: None` is refused at load rather than defaulted.

**LoRA / training**

- **Determinism is name-keyed, not order-keyed.** Every LoRA A/B draw and dropout mask is a pure
  function of `(seed, fully-qualified-param-name)` via `seed_for_param`
  (`crates/jammi-lora/src/seeded.rs`) — never candle's global RNG, never VarMap/HashMap order. On
  one CPU host the same `(seed, rows, config)` → byte-identical adapters; this is a same-box
  guarantee, not a cross-host one (see the numerics note below). The qualified name must match
  candle's `VarBuilder::path` join.
- **In-place Var overwrite is load-bearing** — seeded init and resume restore write into the
  *registered* Var's storage; replacing the field with a fresh clone severs the optimizer binding
  and freezes the weights.
- **Optimizer moments serialize BY NAME** (`varmap.all_vars()` order is unstable across
  processes); never serialize moments positionally.
- **LoRA A/B always F32**, backbone may be BF16/F16.
- **Validation forwards must run with `set_training(false)`** or they consume dropout-mask draws
  and desync the resume stream.
- **Regression trains in z-space**; the `TargetScaler` is persisted and authoritative on resume —
  never recomputed. De-standardisation happens only at serve.
- **Null/NaN regression targets are rejected citing the row**, never coerced to 0.0 (would corrupt
  the scaler μ/σ).
- **Finalization is the worker's sole lease-guarded CAS authority**; the trainer loop never writes
  terminal status; cancellation is cooperative only at epoch boundaries (a `spawn_blocking` thread
  cannot be force-aborted).
- **EmbeddedWorker RAII bounds *claiming*, not *finishing*** — Drop stops new-job claims; an
  in-flight training run completes and writes its status after the guard drops. The worker holds a
  `Weak` to the session.

**Model lifecycle**

- **Cache key = `ModelSource.to_string()` only** — `task`/`backend_hint` are not in `ModelId`; the
  first `get_or_load` for a source pins its backend/task.
- **Single-flight: a waiter must `continue` and re-check the fast path on wake** — the loader may
  have *failed*. Keep `in_flight.remove` + `notify` paired on every exit, both Ok and Err arms.
- **Admission uses `try_acquire` + evict, not `acquire`** (the async `acquire` and `GpuPriority`
  are dead in production). **Eviction only frees `ref_count==0` entries** — fail-fast, no queuing.
- **`GpuPermit` lifetime == `CacheEntry` lifetime** (moved into the entry, field `_gpu_permit`) —
  dropping it early releases budget while the model still occupies memory. **`estimate_memory`
  precision matters** (admission budgets weights only, not activations).
- **Retired-model refusal lives in the resolver, not the catalog read** — `get_model` still returns
  retired rows for reference resolution.

**Numerics**

- **Same-box determinism only** — f32/f64 summation order is fixed *per binary* on the box that
  ran it, but is **not bit-equivalent across x86_64/aarch64, nor across two hosts of the same
  architecture** (candle's vector paths are compile-time `#[cfg(target_feature)]`, and
  `gemm`/`pulp` dispatch the actual SIMD/FMA kernel at runtime off the host's detected features —
  two same-arch boxes with different detected feature sets can pick a different kernel and differ
  in the last bits; measured directly by `bits_snapshot.rs`, which pins per-box, not per
  `(target_arch, target_os)`). Do not add parallel reduction (rayon, non-fixed-lane SIMD) — it
  breaks even the single-box guarantee. Cross-arch reproducibility AND cross-box same-arch
  reproducibility are both explicit non-goals.
- **f32 vs f64 reduction asymmetry is intentional** — `cosine_distance`/`cosine_similarity` in f32,
  `vector_norm`/`cosine_f64` in f64; not interchangeable (shifts last-bit results and can flip a
  tie-break).
- **Canonical-basis-before-resample** (`bootstrap_ci` sorts before drawing); the `statistic_fn` must
  be order-invariant.
- **Infallible kernels use `debug_assert_eq!` for length** — a release-mode length bug is silent;
  preserve the convention, don't "fix" it into `Result` ad hoc.

---

## 6. Build, test & release mechanics

**Toolchain.** Pinned by `rust-toolchain.toml` (channel + rustfmt/clippy), which is the single
source of truth — the CI image build reads it and receives the channel as the `RUST_VERSION`
build-arg, so the image can never bake a version the repo has moved off. `.cargo/config.toml` sets
`rustc-wrapper = "sccache"` globally (sccache disables incremental by design — if sccache is missing,
cargo fails) and, per Linux target, a `rustflags` list: mold (`-fuse-ld=mold`) on both
`x86_64-unknown-linux-gnu` and `aarch64-unknown-linux-gnu`, plus `-C target-feature=+fp16` on the
latter — a COMPILE BASELINE (ARMv8.2-A FEAT_FP16), not a runtime floor: `gemm`'s aarch64 f16 kernels
dispatch at RUNTIME via `is_aarch64_feature_detected!`, so a release build without the flag still
uses them on hardware that has FEAT_FP16; the flag exists only because that same code fails to
COMPILE at opt-level 0 without it. The trade: Raspberry Pi 4 (ARMv8.0) is not a supported
aarch64-linux host for this workspace's artifacts.

Config `rustflags` apply everywhere, local dev AND CI alike — `./.github/actions/setup-rust-ci`
never exports a bare `RUSTFLAGS` (which would REPLACE config `rustflags` for every target, silently
dropping mold and the fp16 floor both); its `deny-warnings` input instead exports
`CARGO_TARGET_<TRIPLE>_RUSTFLAGS=-D warnings` for the runner's own host triple, which JOINS with
(never replaces) config `rustflags` for that same triple — verified against a real cargo. A caller
must never export a bare `RUSTFLAGS` of its own either, at job or step level — that replaces
everything the same way, regardless of which mechanism set it (`dep-dag.yml`'s "clear -D warnings
for one third-party install step" override targets the SAME per-target env var `setup-rust-ci` used,
not a bare `RUSTFLAGS`, for exactly this reason). CI/dev/release
base image: `.docker/ci.Dockerfile` (= `jammi-ai-ci`), a multi-arch index (`linux/amd64` +
`linux/arm64`, one native leg per platform, merged by `_ci-base-image.yml`); the CUDA image extends
it and stays `linux/amd64`-only.

Every multi-arch image this workspace publishes (the CI base image above, and the CPU
`jammi-ai-server` image) is built the same two-leg-plus-merge way, never a single `docker buildx
build --platform linux/amd64,linux/arm64` (which would need QEMU emulation for the non-native
arch): one job per arch, on that arch's own NATIVE runner (`ubuntu-latest` / `ubuntu-24.04-arm`, no
QEMU), each pushing ONLY its own immutable per-arch tag (`sha-<sha>-<arch>`) — never a real,
consumer-facing tag. A separate merge job then combines those two immutable per-arch sources into
the real tags with `docker buildx imagetools create`, dry-running the merge first and asserting the
resulting index's platform set BEFORE the real (pushing) `imagetools create` runs — verify-then-
promote, not promote-then-hope. That merge job is the ONLY job that ever moves a real tag; the two
per-arch build legs never do. `server-image.yml`'s CPU image runs this pattern twice — once
ungated (`build-and-push-main` → `merge-cpu-main`, `main`-dispatch only, `:latest`/`sha-<sha>`) and
once prove-gated (`build-and-push` → `merge-cpu-tag`, `v*` tags only, semver/`:latest`/`sha-<sha>`)
— mirroring `_ci-base-image.yml`'s own single merge job for the CI base image itself.

**Run before pushing (the local gate, mirrors `check`):**
- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --exclude jammi-python` (hermetic — zero network)
- For Python: `maturin develop` then `pytest crates/jammi-python/tests` (never `cargo build` for
  the wheel).

**The PR gate (`.github/workflows/ci.yml`, everything `needs: check`):** `check` (fmt+clippy) →
`test` (hermetic, excludes `jammi-python` because the cdylib needs libpython on the linker path) →
`compile-check-gated` (`--no-run` for live-hub / live-gpu, a *separate job* so accumulated test-binary
graphs don't exhaust runner disk) → `test-clients` (clients + the **two candle-free boundary guards**)
→ `dep-direction` (`check_dep_direction.py`) → `oss-only-build` (`--locked` hermeticity) → `ts-client`
/ `py-client` / `test-python` / `test-broker` (serialised `--test-threads=1`) / `test-pg` (serialised)
→ `test-live` (main-only, advisory).

**The merge path, locally (`ci/scripts/merge_path.sh`):** one runner for the `check`, `test`,
`test-pg`, `guard` and `symbol-index-gates` jobs above plus `docs.yml`'s build, read from the
workflow files at run time (never a copied list) and run in one process: `static` (fmt, the
four clippy surfaces, rustdoc `-D warnings`, the guide build — a missing `mdbook` FAILS unless
`--skip-mdbook`) → `guards` (the guards in `ci/guards.toml` this change can affect, through
`ci/scripts/run_guards.py` — the runner `ci.yml`'s `guard` job calls) → `index` (`symbol-index-gates`' steps,
each `run:` block executed WHOLE) → `tests` (the hermetic lane, the `test-hooks` lane,
golden-parity, and the Postgres lane against `JAMMI_TEST_PG_URL` — a missing database FAILS the
stage unless `--skip-pg` is passed, because a silently skipped lane is how a shared-database
leak once reached CI). It refuses to run when HEAD is the base or the tree is dirty (every
diff-scoped gate would be vacuously green), prints up front which `ci.yml` jobs it does NOT
cover (CI runs those), and exits with the number of failed commands, each named with its log
under `$CARGO_TARGET_DIR/merge-path`.

**CI guard contracts:**
- **`ci/scripts/check_dep_direction.py`** BFS-walks the normal-dependency closure of
  `default-members`; flags any crate whose source is non-crates.io (git/private registry) or
  prefixed `jammi-enterprise`. dev/build deps are not walked.
- **Candle-free boundary** (inline in `.github/workflows/ci.yml`): isolated per-package build of
  `jammi-wire`/`jammi-admin`/`jammi-client`, grep the compiler-artifact stream for
  `candle*|hf-hub|symphonia|tokenizers`; plus `jammi-cli` carries no `jammi-ai` edge. (On the isolated
  artifact, not `cargo tree`.)
- **Conformance:** `crates/jammi-python/tests/test_conformance.py` pins remote == embedded verb sets
  name-for-name.
- **`ci/scripts/check_sqlite_isms.py`** greps `crates/jammi-db/src/**` for hand-written
  backend-specific SQL tokens (SQLite-only `rowid`/`AUTOINCREMENT`/`PRAGMA`/`strftime(`/`glob(`;
  Postgres-only `ctid`). A syntactic first-pass tripwire only — it stops the obvious cheap
  regression; `test-pg` above is what actually enforces backend-behavioral parity.
- **`ci/scripts/check_cuda_run_artifacts.py`** enforces a schema over every `*.json` under
  `crates/jammi-kernels/artifacts/cuda-runs/` (see that directory's own `README.md` for the field
  list, and `docs/maintainer/cuda-kernel-guide.md` §4): a well-typed `schema_version` / `git_sha`
  / `box` / `producer` / `status`, a `producer` that is either statically verifiable (a real
  `#[test] fn` found under its stated `#[ignore]`/`env:<VAR>`/`required-features` gating
  attribute) or a reviewed legacy `kind: "none"` entry in the script's own closed allow-list, and
  `git_sha` an ancestor of `HEAD` (`git merge-base --is-ancestor`) — OR, when the measured tip was
  itself squash-merged, the optional `merged_as` (the squash commit, verified to literally
  reintroduce this same artifact file) + `merged_via_pr` pair naming an ancestor instead, with
  `git_sha` kept verbatim either way.

**Postgres coverage note (heuristic, not a certified gap list).** A name-based grep of
`crates/jammi-db/tests/it/` for the following `jammi-db` catalog/store functions found zero direct
callers by that literal name, so their SQL is exercised only indirectly (if at all) by the
`test-pg` matrix above: `delete_result_tables_for_source`, `get_mutable_table_for_tenant`,
`list_source_descriptors`, `describe_source`, `find_ready_result_tables_anchored_on`,
`delete_artifact_prefix`, `delete_table_files`, `list_all_mutable_tables`, `get_model_version`,
`list_eval_runs`, `latest_eval_run`, the
training-worker checkpoint surface (`put_artifact`/`fetch_artifact`/`put_resume_checkpoint`/
`fetch_resume_checkpoint`/`get_checkpoint`/`set_checkpoint`/`delete_resume_checkpoint`),
`register_table`, `promote_result_table_with_manifest`, `save_sidecar`, `read_keyed_vectors_f32`.
This is a grep over identifier names, not a certified coverage report — a function called through
a wrapper of a different name, or exercised only via a higher-level integration path, would read as
covered under a call-graph analysis and still show up here; conversely nothing here proves the SQL
is actually *un*-exercised end-to-end. Treat it as a to-do list for future direct-coverage work, not
a gate: no CI job asserts against it.

**Test discipline.** Default `cargo test` is fully hermetic. Live tests gate behind a feature and must
**skip cleanly** (`tracing::warn`, never `#[ignore]` / `#[cfg(any())]` / `// TODO`). The GPU suite pins
`require_gpu=true` so a GPU-less build fails fast rather than faking parity. GPU is not testable in CI
(no GPU runners) — compile-checked only; live GPU is an A10G host gate.

**Release (tag-driven, all OIDC trusted publishing, no tokens).** A version bump PR touches the
lockstep version files (`docs/plans/50-open-core-hardening-roadmap/ROADMAP.md`, the version-bump file
list): `Cargo.toml`, `Cargo.lock`, `CHANGELOG.md`, `pyproject.toml`, `clients/python/pyproject.toml`,
`clients/typescript/package.json`, `packaging/server-cpu/pyproject.toml`,
`packaging/server-cu12/pyproject.toml`. On merge, **prove before tagging**: dispatch
`.github/workflows/gpu-prove.yml` on the commit to be released (`--ref main` at the tip, or on the
pushed tag once it exists) and wait for all four shipped arches to go green — **EVERY** release
publishing job (all-or-nothing: not only the CUDA lanes) gates on that recorded
verdict rather than proving anything themselves (`ci/scripts/gpu_prove_verdict.py`, consumed via
`_gpu-proof-required.yml`); a red leg is re-run by hand (`gh run rerun <run_id> --failed`) in that
same prove run. The verdict check is CHECK-ONCE and FAIL-LOUD — no poll, no deadline: a tag push on a
commit whose prove is not ALREADY green fails every release workflow immediately, publishing nothing.
The order is still prove → green → tag, because a tag push commits the version number and nothing
here can retroactively un-push a tag. Then tag both `v*` and `py-v*`
**lightweight, on the same commit, pushed together**:
- **`v*`** → `.github/workflows/crates.yml` (validate → perf-gate → prove-gated `publish` in
  topological order, skip already-published, block on sparse-index propagation; `github-release`
  chains off `publish`) + `.github/workflows/npm.yml` (build+test unconditional; the `Publish` step
  itself is prove-gated) + `.github/workflows/server-image.yml` (the manual `:latest` CPU refresh via
  `workflow_dispatch` on `main` is intentionally ungated — `build-and-push-main` pushes each arch's
  own immutable `sha-<sha>-<arch>` leg, `merge-cpu-main` merges them into the real `:latest`/`sha-<sha>`
  tags via `docker buildx imagetools create`, `gate_kind` "none"; `server-image.yml` carries no
  `push: branches:` trigger, so this arm never fires on a mere merge; both `:latest` tags are
  separately re-pointed by every `v*` release tag itself, via `docker/metadata-action`'s default
  `flavor: latest=auto`, so the CPU `:latest` is never main-only. Publishing an image under a `v*`
  tag is a two-leg + merge pattern (CPU only — the CUDA image builds and pushes in one amd64-only
  job, no merge needed): `build-and-push` pushes ONLY each arch's own immutable
  `sha-<sha>-<arch>` leg (never a real tag), then `merge-cpu-tag` — gated on `build-and-push`'s own
  success — is the ONLY job that ever moves the real semver/`:latest`/`sha-<sha>` tags this arm
  publishes, via the same dry-run-then-`imagetools create` shape as `merge-cpu-main`. The prove-gated
  tag promotions are `build-and-push`, `merge-cpu-tag`, and `build-and-push-cu12`) +
  `.github/workflows/release-binaries.yml` (every asset family — the CLI matrix, the CPU tarball, the
  CUDA tarball — is split into an ungated build leg that always runs and a prove-gated promote leg
  that only attaches to the release on a tag).
- **`py-v*`** → `.github/workflows/pypi.yml` (native wheel) + `.github/workflows/pypi-client.yml`
  (pure-Python client) + `.github/workflows/pypi-server.yml` (CPU wheel — prove-gated too, even
  though it never touches CUDA itself, because it ships in the SAME all-or-nothing lockstep release)
  + `.github/workflows/pypi-server-cuda.yml` (auditwheel deliberately skipped) — all four gated on
  the same verdict, same commit, same tag family, reusing `v*`'s dispatch with no extra prove.

`ci/scripts/check_gpu_prove_once.py`'s `PROMOTION_TABLE` is the reviewed cross-check for every one of
these promotion jobs (workflow, promoting job, gate job); its P6 discovery rule scans EVERY workflow
file (no trigger filtering) and fails by name if a NEW promoting job — one whose steps invoke a
publishing primitive by pattern, directly or via a `uses:`-called local reusable that itself does — is
ever added without a table row.

**Disk pressure is a recurring real failure** — keep `CARGO_TARGET_DIR` on NVMe; the separate
`compile-check-gated` job and `crates.yml --no-verify` exist for this reason.

### 6.1 Navigating the rich graph (build-graph rich → graphify MCP)

For navigating the workspace by *symbol* — who calls `Session::search`, what implements
`VectorIndex`, every site that uses `merge_channels` — build the **rich** `build-graph` (item-level
nodes + semantic reference edges) and serve it to an agent over `graphify`'s stdio MCP server. This is
**local tooling, not a CI lane**: it is built on demand, never in CI, never as a committed artifact. It
is deterministic (zero tokens — pure extraction from build artifacts + rustdoc + rust-analyzer), and on
this workspace yields ~11.2k nodes / ~58k edges across the 13 crates.

This is distinct from the §1.1 `dep-dag` block (crate-level dependency DAG, generated into this doc).
The rich graph is **per-symbol**, lives only under `target/`, and is for interactive navigation.

**One-time setup.** The rich (Layer 2) layer needs a nightly toolchain (for nightly rustdoc) plus
`rust-analyzer` (for the `--references` reference edges); neither is the toolchain pinned by
`rust-toolchain.toml`, so install them alongside it:

```
rustup toolchain install nightly
rustup component add rust-analyzer --toolchain nightly
cargo install build-graph --version 0.1.0 --locked   # pin: graph schema is version-coupled
pip install graphifyy                                 # note the double-y; the import name is `graphify`
```

**Build.** Run the wrapper — it builds the rich graph and **post-verifies it is actually rich**:

```
ci/scripts/build_graph_rich.sh
```

This wraps `cargo build-graph build --rich --references --no-compress --out target/build-graph-rich`.
The `--out` is deliberately **not** `target/build-graph/` — that file belongs to the dep-dag automation
(§1.1); both paths sit under `target/` and are already gitignored (`/target`), so nothing new needs
ignoring. The verification is load-bearing: `cargo build-graph build --rich --references` **silently
degrades to the Layer 1 crate/file graph and still exits 0** when nightly or rust-analyzer is missing
(it logs "rich layer: skipped"). Exit 0 is *not* proof. The wrapper parses the emitted graph and asserts
rich indicators are present — item-level node kinds (`struct`/`method`/`trait`) and semantic edge
relations (`calls`/`implements`/`member_calls`) — and **fails loud** ("rich build degraded to Layer 1 —
need nightly + rust-analyzer") if they are absent. A degraded graph has only `crate`/`file` nodes and
`depends_on`/`contains` edges, so every required indicator is missing.

**Serve.** Point `graphify`'s stdio MCP server at the emitted graph — it consumes `build-graph`'s JSON
natively (find/refs/context/path over the symbol graph):

```
python -m graphify.serve target/build-graph-rich/graph.json
```

**Wire it into an agent.** MCP server configs are **per-agent**; the stanza below is the worked example
for Claude Code (`.mcp.json` at the repo root). Other agents (Codex, Cursor, Gemini CLI, …) use a
different config file and schema — adapt the `command`/`args` accordingly. Do **not** commit this file;
it is a per-developer convenience pointing at a `target/` artifact:

```json
{
  "mcpServers": {
    "jammi-graph": {
      "command": "python",
      "args": ["-m", "graphify.serve", "target/build-graph-rich/graph.json"]
    }
  }
}
```

**Notes / sharp edges.**
- **Communities are advisory, not reproducible.** `graphify`'s community/cluster detection is
  nondeterministic — useful for orientation, never a stable contract. Cite symbols and edges (which are
  deterministic), not cluster ids.
- **`merge-graphs` is unsupported on `build-graph` JSON.** Serve a single graph; do not attempt to merge
  `build-graph` outputs through graphify.

---

## 7. Sharp edges, tech debt & roadmap

Roadmap, sharp edges, tech-debt, and first-PR material live under
`docs/plans/52-maintainer-roadmap/` (see `sharp-edges-and-first-prs.md`). They are kept out of this
reference because they describe where the system is *going*, not what it *is*.
