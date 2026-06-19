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
(candle transformers). Above it sit `jammi-server` (serves the wire over the
engine) and `jammi-python` (a local-only PyO3 cdylib whose remote arm is the
bundled pure-Python client).

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
jammi-ai -> jammi-db, jammi-encoders, jammi-lora, jammi-numerics, jammi-test-utils, jammi-wire
jammi-bench -> jammi-ai, jammi-db, jammi-lora, jammi-numerics
jammi-cli -> jammi-admin, jammi-db
jammi-client -> jammi-admin, jammi-db, jammi-wire
jammi-db -> jammi-numerics, jammi-test-utils
jammi-encoders -> jammi-lora
jammi-lora
jammi-numerics
jammi-python -> jammi-ai, jammi-db
jammi-server -> jammi-admin, jammi-ai, jammi-client, jammi-db, jammi-numerics, jammi-test-utils, jammi-wire
jammi-test-utils -> jammi-db
jammi-wire -> jammi-db, jammi-lora, jammi-numerics
```
<!-- END GENERATED: dep-dag -->

Production dependency edges — each crate's normal `[dependencies]`, dev/test excluded (cf. the
build-link graph above):

```
jammi-numerics   (pure math; no internal deps)
      ▲
jammi-db ────────────────► jammi-numerics
   ▲   ▲
jammi-lora   (NO internal deps; candle OPTIONAL, default-features=false at root)
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
             jammi-encoders(opt, `local`), candle(opt, `local`)
   ▲   ▲
jammi-server ──► jammi-wire, jammi-ai, jammi-db, jammi-numerics  [serves the wire over the engine]
   │
jammi-python ──► jammi-ai, jammi-db, jammi-lora                  [LOCAL-ONLY PyO3 cdylib]

jammi-encoders ──► candle, jammi-lora(features=["candle"])
```

The publish topological order (the canonical DAG statement,
`.github/workflows/crates.yml`, the publish-order list) is:
`jammi-numerics → jammi-db → jammi-lora → jammi-encoders → jammi-wire →
jammi-admin → jammi-client → jammi-ai → jammi-server → jammi-cli`.

Workspace membership (`Cargo.toml`, `[workspace] members`): 13 members;
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
- **`jammi-python` depends on `jammi-ai`, `jammi-db`, `jammi-lora`** — no
  client-substrate crate. Local-only; its remote arm is the bundled pure-Python
  `jammi_client` (`crates/jammi-python/src/lib.rs`, the module setup), so the
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
                                                          jammi-cli, pure-py jammi_client
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
| `TrainingService` | engine + `ServiceTier::Train`, `#[cfg(feature="train")]` |

The `mounted` `Vec` itself is only a `tracing::info!` log line
(`crates/jammi-server/src/runtime.rs`), not the wire advertisement. The handshake
advertises the *tier tokens*, not the service list: `TierSet::as_wire`
(`crates/jammi-server/src/tiers.rs`, [§2.8]) returns the mounted tiers as sorted
wire tokens (`core`/`event`/`eval`/`train`) for `ServerInfo.services`. The
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
| `CatalogService` | `Staleness`/`DerivesFrom` | `grpc/catalog.rs` |
| `CatalogService` | `CreateMutableTable`/`DropMutableTable`/`ListMutableTables` | `grpc/catalog.rs` |
| `CatalogService` | `RegisterTopic`/`DropTopic`/`ListTopics` | `grpc/catalog.rs` |
| `EmbeddingService` | `GenerateEmbeddings`/`EncodeQuery`/`Search` | `grpc/embedding.rs` |
| `InferenceService` | `Infer`/`Predict` | `grpc/inference.rs` |
| `PipelineService` | `BuildNeighborGraph`/`PropagateEmbeddings`/`AssembleContext` | `grpc/pipeline.rs` |
| `PipelineService` | `AsofJoin` | `grpc/pipeline.rs` (`PipelineService::asof_join`) |
| `PipelineService` | `Recompute` | `grpc/pipeline.rs` (`PipelineService::recompute`) |
| `AuditService` | `AuditLog`/`AuditFetchByQueryId`/`AuditFetchRecent` | `grpc/audit.rs` |
| `EvalService` | `EvalEmbeddings`/`EvalPerQuery`/`EvalInference`/`EvalCompare`/`EvalCalibration` | `grpc/eval.rs` |
| `TrainingService` | `StartTraining`/`TrainingStatus` | `grpc/training.rs` |
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

**The four coupling artifacts.**

1. **The API-surface guard — `cookbook/book/scripts/check_api_reference.py`.** This
   is the staleness oracle. It introspects the *live installed* `jammi_ai` wheel
   and asserts every verb the chapters call still exists with the kwargs the
   recipes pass. The mechanism (worth reading; it mirrors the transport-parity
   collapse this guide documents in §2.1/§4.1): it opens an embedded engine
   (`jammi_ai.connect("file://…")`, in `main`) and resolves each verb as a *bound
   method on the live instance* (the `_signature` helper) rather than introspecting
   the wrapper class — because the thin `Database` wrapper holds the native
   `_NativeDatabase` by composition and forwards un-migrated verbs through
   `__getattr__` (documented in `_signature`, corroborated by the engine at
   `crates/jammi-python/src/database.rs`). The contract is the `REQUIRED` dict (46
   verbs) plus `MODULE_FUNCTIONS = ["open_local", "connect"]`; the gate prints
   "`API reference matches installed jammi_ai (N surfaces checked)`" where
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

**How the loop closes in CI (the atomicity guarantee).** The book is tested
against the engine commit it ships beside. The PR gate
(`.github/workflows/cookbook-book.yml`, the PR job) builds the HEAD embed wheel
with `maturin`, force-reinstalls it (`--force-reinstall --no-deps`) over the
unpinned `jammi_ai` dependency, then runs `check_api_reference.py`, the shared-lib
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

**Note on the pin.** The chapters are not pinned to an exact `jammi_ai==X.Y.Z`
PyPI release: `jammi_ai` is consumed **at HEAD**, declared unpinned in
`cookbook/book/pyproject.toml` (the `jammi_ai` dependency) and force-installed from
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
  `InferenceSession::open(config)` → `Session::with_embedded_worker(engine)`. `Target`
  is **Local-only** (`crates/jammi-ai/src/jammi.rs`, the `Target` enum); remote is
  reached through `jammi-client`'s `DataClient`, *not* this front door [§7].
- **`Session`** — `crates/jammi-ai/src/local_session.rs` (the `Session` struct). A
  thin `Arc<InferenceSession>` wrapper (re-exported as `crate::Session`,
  `crates/jammi-ai/src/lib.rs`). Two constructors with a load-bearing distinction:
  - `Session::with_embedded_worker(engine) -> Result<Self>`
    (`crates/jammi-ai/src/local_session.rs`): the **front-door** form — carries
    `Some(worker)`, spawns the training worker (RAII; stops on drop). Must run
    inside a tokio runtime. Returns `JammiError::Config` if `[training]` timing
    violates worker invariants.
  - `Session::new(engine) -> Self` (`crates/jammi-ai/src/local_session.rs`):
    carries `None` — used by **every per-request wrapper** (gRPC handlers, Python
    `Database`), so a worker is **not** spawned per call.
  - **Invariant:** only the front-door session owns a worker; spawning one per
    request would multiply training claimants [§5].
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
  Trust the array order, not the `schema.rs` constant order.
- **Typed status enums** — `crates/jammi-db/src/catalog/status.rs`:
  `ResultTableStatus`, `TrainingJobStatus`, `EvalRunStatus`, `ModelStatus`. Each
  impls `Display`+`FromStr`. **Contract: the DB value set is total over the enum**
  (round-trip test in `status.rs`). `ResultTableKind` (Model/NeighborGraph,
  `crates/jammi-db/src/catalog/result_repo.rs`) is a *separate* discriminator from
  `ModelTask`.
- **`ResultStore`** — `crates/jammi-db/src/store/mod.rs` (the `ResultStore`
  struct): result-table storage coordinator (`root: StorageUrl`, `StorageRegistry`,
  `Arc<Catalog>`, `AnnIndexConfig`). Key methods: `ResultStore::create_table`,
  `finalize`, `recover`, `materialize_embedding_table`, `resolve_search_mode`.
- **`SidecarKind` / `sidecar_extensions`** —
  `crates/jammi-db/src/storage/sidecar_layout.rs`: the single registry the writer,
  reader, and cleanup all consult. Ann → `["usearch","rowmap","manifest.json"]`;
  Lexical → `["tantivy"]`; None → `[]`.
- **SQL identifier quoting** — `crates/jammi-db/src/sql/ident.rs`: `quote_ident`,
  `quote_relation`, `source_relation`. **Invariant: every identifier interpolated
  into a generated SQL string goes through here** (an unquoted hyphen parses as
  minus).
- **Source types** — `crates/jammi-db/src/source/mod.rs`: `SourceType { File,
  Postgres, Mysql }`; `FileFormat { Parquet, Csv, Json, Avro }` (Avro declared but
  unsupported, `crates/jammi-db/src/source/file_format.rs`). `SourceConnection`
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
  struct): USearch HNSW + Jammi-owned rowmap (`ROWMAP_VERSION=1`) + JSON manifest.
  Metric hardcoded `Cos`, quant `F32`. `SidecarIndex::index_options` is the **sole
  place USearch field names appear**.
- **`AnnIndexConfig`** — `crates/jammi-db/src/config.rs` (the `AnnIndexConfig`
  struct): `connectivity` (HNSW M, build-time), `build_expansion`
  (ef_construction, build-time), `search_expansion` (ef_search, query-time,
  mutable). **`0` = backend default.**
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
- **The broker IS a pluggable backend** — `TriggerBroker` trait
  (`crates/jammi-db/src/trigger/broker.rs`, `Send + Sync + 'static`,
  `#[async_trait]`): `register_topic`/`drop_topic`/`publish`/`subscribe`/
  `list_consumers`/`driver_kind`. **Contract: the broker is transport-only — it MUST
  NOT persist** (`crates/jammi-db/src/trigger/broker.rs`, the trait doc); the
  engine's mutable backing table is the authoritative log, and the broker never sees
  `tenant_id` (tenant scope is enforced upstream by catalog lookup + predicate
  injection). Two impls behind `Arc<dyn TriggerBroker>`: `InMemoryBroker`
  (`crates/jammi-db/src/trigger/in_memory.rs`, `BrokerKind::InMemory`, the
  **default**) and `JetStreamBroker` (`crates/jammi-db/src/trigger/jetstream.rs`,
  `BrokerKind::JetStream`) — the latter is gated behind the **`jetstream-broker`
  cargo feature** (`crates/jammi-db/Cargo.toml`; re-exported only under that cfg,
  `crates/jammi-db/src/trigger/mod.rs`; `jammi-server` re-exposes it as
  `jetstream-broker`, `crates/jammi-server/Cargo.toml`). Selection is
  **config-driven**, `build_broker_from_config`
  (`crates/jammi-db/src/session.rs`): `BrokerConfig::InMemory` →
  `InMemoryBroker::new()`; `BrokerConfig::JetStream{…}` → connect; choosing
  JetStream **without** the feature returns a typed `JammiError::Config`, not a
  panic (the `#[cfg(not(feature = "jetstream-broker"))]` `build_jetstream_broker`,
  `crates/jammi-db/src/session.rs`).
- **Publish/subscribe contract + error type.** Error type is `TriggerError`
  (`crates/jammi-db/src/trigger/error.rs`) — a `thiserror` enum: `TopicNotFound`,
  `SchemaConflict`, `UnsupportedSchemaType`, `BatchSchemaMismatch`,
  `PublishTenantMismatch`, `PredicateParse`/`Eval`/`Unsupported`, `OffsetEvicted`,
  `BackingTable(#[from] MutableTableError)`, `Backend(#[from] BackendError)`,
  `Driver`, `Catalog`. Note the **`#[from] MutableTableError`** edge: the trigger
  log *is* a mutable companion table, so the two primitives share an error path.
- **Session verbs** (`crates/jammi-ai/src/local_session.rs`): all return
  `Result<…, TriggerError>`:
  - `Session::register_topic` — registers with **both** the broker driver *and* the
    catalog (`topic_repo`), in that order, because a `publish` resolves the topic
    against the broker; a catalog-only registration would make publish fail with
    `TopicNotFound`.
  - `Session::publish(topic, batch) -> Offset` →
    `Publisher::publish_scoped(topic, tenant, batch)`
    (`crates/jammi-db/src/trigger/publisher.rs`). Tenant comes from the session, not
    the caller.
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
  persist), add a `BrokerKind` variant (`crates/jammi-db/src/trigger/broker.rs`) and
  a `BrokerConfig` arm, wire it in `build_broker_from_config`
  (`crates/jammi-db/src/session.rs`); gate any new dependency behind a cargo feature
  like `jetstream-broker` and return `JammiError::Config` when selected without it.
  No `search()` or result-table code changes — the broker plugs in at the
  session-construction seam.
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
(`crates/jammi-ai/src/pipeline/asof/merge.rs`) → `finalize_with_manifest` with a
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
`finalize_with_manifest` funnel** with `CachePolicy::Bypass` (a recompute that reused
a cache would be a no-op). Byte-identical when inputs haven't moved (the descriptor
records every output-affecting determinant). A pre-contract table with no descriptor
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

`replay_descriptor` dispatches per `ProducingDescriptor` variant — `Inference`,
`Embedding`, `NeighborGraph`, `GraphPropagation`, `ContextSet`, `AsofJoin` — each
rebuilding its producer call from recorded typed params, all with
`CachePolicy::Bypass`. The intricate case is `ContextSet`: its real producer is the
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
- **Recompute a new producer:** any new materialising verb must add a
  `ProducingDescriptor` variant *and* a `replay_descriptor` arm
  (`crates/jammi-ai/src/pipeline/recompute.rs`) — a producer with no replay arm is
  not recomputable. The replay must use `CachePolicy::Bypass`.

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
(`finalize_with_manifest`) — every result table carries a verifiable
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
5. **Attest** through `finalize_with_manifest`
   (`crates/jammi-ai/src/pipeline/asof/verb.rs` → `crates/jammi-db/src/store/mod.rs`) —
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

`MANIFEST_VERSION = 3` (`crates/jammi-db/src/store/manifest.rs`); a version mismatch or a
serde-shape mismatch is a typed `ManifestError`, never a silently-trusted stale hash
(`Manifest::from_json_bytes`). The descriptor is kept **verbatim** in the manifest (not
just the opaque hash) precisely so a reader can *replay* it — that is what the recompute
path reads.

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
- **`MissingManifest`** — no sidecar (a pre-contract table). A truthful unknown, never a
  fabricated match — distinct from a post-contract table that *should* carry one (a torn
  write recovery reconciles).

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
- **New materialized producer (any verb):** write through `finalize_with_manifest` with a
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
field (e.g. `crates/jammi-wire/proto/jammi/v1/pipeline.proto`).

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
`#[serde(tag="producer")]` enum with one variant per producer — `Inference` / `Embedding` /
`NeighborGraph` / `GraphPropagation` / `ContextSet` / `AsofJoin` — each carrying every
output-affecting determinant (float knobs stored by IEEE-754 bit pattern so the descriptor
stays `Eq`/`Hash`, e.g. `min_similarity_bits`, `alpha_bits`).

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
The replay is **byte-identical when inputs have not moved**, because the descriptor records
every output-affecting determinant. Per-variant subtleties:
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
`(record, CacheOutcome)`; (3) writing through `finalize_with_manifest` with the *same*
`(descriptor, env, inputs)` the probe keyed on; (4) adding a `replay_descriptor` arm that
calls the producer with `CachePolicy::Bypass` and the matching `*_from_manifest`
reverse-mappers. Anchor immutable result-table inputs as `ResultDigest` (cacheable) and raw
sources as `UnpinnedAtInstant` (honestly never a hit). Do **not** add a scheduler or a
staleness→recompute loop — that is the platform's, not the engine's
(`crates/jammi-ai/src/pipeline/recompute.rs`).

### 2.5 Encoders (`jammi-encoders`)

- **`AnyEncoder`** — `crates/jammi-encoders/src/any.rs` (the `AnyEncoder` enum): a
  hand-written closed enum `{ Bert, DistilBert, ModernBert, ClipText }`, **not a
  trait**. Forwards each method to the active variant. The compiler forces a match arm
  in every method — that is the point of the closed enum (no trait-object overhead).
  **ClipText is a second-class citizen:** `forward_hidden` returns `Err`,
  `trainable_params` is empty, all training-mode methods are no-ops (frozen, no LoRA).
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
  enum): closed enum `{ Frozen(Linear), Lora(LoraLinear) }`, decided **once at
  construction** (`LoraSite::build`). `Frozen.forward` casts input to the weight dtype;
  all other methods are no-ops on `Frozen`. This is the static-dispatch LoRA injection
  point — used by `jammi-encoders` to hold attention/MLP linears (e.g.
  `crates/jammi-encoders/src/bert.rs`).
- **`LoraBuildConfig` + selection helpers** — `crates/jammi-lora/src/config.rs` (the
  `LoraBuildConfig` struct): borrowed-ref, `Copy`, stack-built per call.
  `should_apply_lora` uses **suffix** match (`ends_with`); `effective_rank` uses
  **substring** match (`contains`) — *different semantics, do not conflate*.
  `LoraBuildConfig::frozen()` is the no-LoRA default.
- **`LoraInitMode`** — `crates/jammi-lora/src/init.rs` (the `LoraInitMode` enum):
  `ZerosB` (default; identity at construction) or `Gaussian`.
- **Persistence** — `AdapterConfig` (`crates/jammi-lora/src/adapter.rs`) +
  `save_adapter`/`load_adapter` (`crates/jammi-lora/src/save_load.rs`):
  `adapter.safetensors` + `adapter_config.json`. `AdapterConfig::from_build` snapshots
  only shape-affecting fields; **run-time-only fields (`lora_dropout`, `init_mode`) are
  deliberately not persisted**. Higher-level `SavedAdapter`
  (`crates/jammi-ai/src/fine_tune/target.rs`) is the on-disk discriminator between
  `ProjectionHead` and `EncoderAdapters` targets.
- **`TrainingTarget`** — `crates/jammi-ai/src/fine_tune/target.rs` (the `TrainingTarget`
  enum): `ProjectionHead { head: LoraModel }` or `EncoderAdapters(...)`. Uniform trainer
  surface (`trainable_params`, `set_training`, `named_trainable_weights`, `load_weights`,
  dropout positions, `saved_adapter`).
- **`FineTuneConfig::validate()`** — `crates/jammi-wire/src/fine_tune.rs`
  (`FineTuneConfig::validate`): the gate (rank>0, alpha>0, dropout∈[0,1), pinball
  ascending levels, …). `seed` defaults to `DEFAULT_FINE_TUNE_SEED = 42` — a constant,
  never entropy.
- **`TrainingWorker` / `TrainingJob`** — `crates/jammi-ai/src/fine_tune/worker.rs` (the
  `TrainingWorker` struct), `crates/jammi-ai/src/fine_tune/training_job.rs` (the
  `TrainingJob` struct): the lifecycle owner and the poll/wait handle.

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
  *moved into* the `CacheEntry` (`_gpu_permit`), so **permit lifetime == entry lifetime**.
  `ModelCache::preload` is a thin `get_or_load`-then-`drop` warmer taking an *explicit*
  `(source, task, backend_hint)` — it does **not** read any config list, and it is called
  only from a test (`crates/jammi-ai/tests/it/models.rs`). `config.preload_models`
  (`crates/jammi-db/src/config.rs`) is **dormant**: documented as "preload at server
  startup" but has no reader anywhere in the engine (defaults empty; no `jammi-server`
  startup wiring consumes it).
- **`ModelSource` / `ModelId`** — `crates/jammi-ai/src/model/mod.rs` (`ModelId` and
  `ModelSource`): `HuggingFace(String)` | `Local(PathBuf)`. **`ModelId` = `Display` of the
  source is the entire cache key** — `task` and `backend_hint` are NOT part of it [§5
  gotcha].
- **`ResolvedModel`** — `crates/jammi-ai/src/model/mod.rs` (the `ResolvedModel` struct):
  the frozen "files located, backend chosen, not loaded" struct (resolver → backend
  contract).

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
  solely by the lease-guarded `finalize_training_job` CAS, never by a worker's
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
  before the `DELETE`. The four reference edges (`REFERENCE_EDGES`) are
  `result_tables.model_id`, `training_jobs.output_model_id` (both keyed by model NAME, no
  FK), `training_jobs.base_model_id`, `eval_runs.model_id` (both keyed by catalog PK,
  FK-backed). A non-empty scan returns `DeleteOutcome::Referenced`, raising the typed
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
  enum and `TierSet` struct). `Core` always mounted; `OPTIONAL = [Eval, Event, Train]`
  (only `Train` is `#[cfg]`-gated). `TierSet::resolve` rejects a requested-but-not-compiled
  tier (`TierError::FeatureNotCompiled`, not a silent drop). `TierSet::as_wire` is **sorted
  alphabetically** — the `ServerInfo.services` handshake. **Invariant: advertised
  (`as_wire`) == mounted.**
- **`OssServer` / `serve_grpc_chain`** — `crates/jammi-server/src/runtime.rs` (the
  `OssServer` struct and `serve_grpc_chain` fn). Single Tonic chain shared by production
  and tests. Mounts Flight SQL + `CatalogService` always; engine-backed services when
  `engine.is_some()`; tier-gated `Eval`/`Train`/`Trigger`. **`OssServer::new` calls
  `InferenceSession::open` (not `new`)** so the `annotate` UDTF is registered for Flight
  SQL.
- **Session/tenant boundary** — `crates/jammi-server/src/grpc/session.rs`:
  `SESSION_HEADER`, `SessionStore` (in-process `HashMap<SessionId, Option<TenantId>>`),
  `TenantInterceptor`. **Invariant: a request with no/unknown session header runs unscoped
  (all-tenants), never an error** — the load-bearing gotcha behind "bind first" [§5].
- **Per-handler helpers** — `crates/jammi-server/src/grpc/wire.rs`: `session_tenant`,
  **`scoped`** (the concurrency-safe per-task-local tenant scope — handlers must use this,
  never sticky `bind_tenant`), `require_nonempty`, `map_engine_error`/`map_trigger_error`.
  **Invariant: faithful errors** — each `Status` carries the full structured detail so the
  client reconstructs the exact variant.

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
3. → `Session::with_embedded_worker(engine)` — spawns `EmbeddedWorker::spawn`, stores it
   in `_worker` (RAII).
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
JSON, and `catalog.create_training_job` into a **`queued`** row. Returns a `TrainingJob`
handle. **No in-memory state crosses submit→claim — the spec is the only carrier.**

**Worker pickup:** `TrainingWorker::run_until` (`crates/jammi-ai/src/fine_tune/worker.rs`)
each tick: `reclaim_expired_training_jobs` → `claim_next_training_job` (takes a lease) →
`run_claimed_job`: deserialize spec, pin catalog to the job's tenant, spawn heartbeat
(renews lease, sets `cancel` on loss), run under tenant scope, then `publish_and_finalize`.

**Train:** `TrainingWorker::run_spec` → FineTune arm → `read_source_columns` (`SELECT …
ORDER BY <full tuple>` for deterministic order) → `build_training_data_loader` →
`train_fine_tune` → `run_fine_tune_blocking` (on the blocking pool, `catch_unwind`-wrapped):
builds the `TrainingTarget` (empty `target_modules` → projection head; non-empty →
`build_encoder_adapters` which resolves the backbone, parses `model_type`, builds the
encoder via `jammi_encoders::{Bert,DistilBert,ModernBert}::builder().lora(...)` — **where
injection happens via `LoraSite::build`, `crates/jammi-encoders/src/bert.rs`**), then
`TrainingLoop::run` (`crates/jammi-ai/src/fine_tune/trainer.rs`). The loop snapshots
`varmap.all_vars()` once, builds AdamW, runs epochs with grad-accum, cooperative
cancellation at epoch boundaries, durable resume checkpoints, early stopping, saves `best`,
builds `SavedAdapter`, calls `jammi_lora::save_adapter`. **The loop never writes terminal
status / registers the model / publishes.**

**Finalize (worker, lease-guarded):** `publish_and_finalize`
(`crates/jammi-ai/src/fine_tune/worker.rs`) writes files to a unique per-attempt prefix
`{job_id}/{worker_id}/{attempt}`, `register_model`, `finalize_training_job` **CAS** flips
to `completed` + commits the served path only while `claimed_by==worker AND
status==running`. On CAS win, GC the resume checkpoint. **Finalization is the worker's sole
authority.**

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
  `evict_one`; if nothing evictable, error) → `backend.load` → best-effort catalog
  `register_model` (failure logged & swallowed) → insert `CacheEntry` (permit moved in) →
  return guard with refcount 1.

Resolver chain (`crates/jammi-ai/src/model/resolver.rs`, `ModelResolver::resolve`):
`try_catalog_lookup` first (refuses `Retired`; resolves fine-tuned base recursively + fetches
adapter), else `resolve_local`/`resolve_hf_hub` (locate config, pick backend, gather weights,
discover tokenizer, sum file sizes into `estimated_memory`).

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
  → server `TenantInterceptor` inserts `SessionTenant` → `InferenceServer::infer`
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
(`clients/python/jammi_client/_assembly.py`, shared by remote *and* embedded) and the
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
   (`crates/jammi-ai/src/session.rs`) is shared by the gRPC `StartTraining` handler and the
   embedded binding — add new `TrainingSpec` handling there, not in two places.)

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
     `build_<verb>_request(...)` to `clients/python/jammi_client/_assembly.py`. Both wheels
     import it (remote: `clients/python/jammi_client/_database.py`; embedded:
     `python/jammi_ai/_database.py`). Surface-only validation (e.g. the graph-only
     embedding-loss guard, `output='quantile' requires levels`) lives here too.
   - **Embedded (`jammi_ai`):** the thin Python `Database` (`python/jammi_ai/_database.py`)
     calls `build_<verb>_request(...)`, serializes, and hands the bytes to **one** PyO3
     primitive `_<verb>_proto(bytes)` on `crates/jammi-python/src/database.rs` (e.g.
     `_infer_proto`, `_start_training_proto`), which decodes via
     `jammi_ai::wire::<verb>_from_bytes`. `jammi-python` pulls in **no tonic transport/server
     stack** — it depends on tonic **only for the `tonic::Status` type** the wire-decode seam
     returns (`crates/jammi-python/Cargo.toml`), plus the wire converters + `prost` (behind
     jammi-ai's `local`).
   - **Remote (`jammi_client.RemoteDatabase`):** the matching method in
     `clients/python/jammi_client/_database.py` builds via the same `build_<verb>_request`
     then sends over gRPC. **`jammi_client` imports no `jammi_ai`** — a CI import-direction
     guard enforces this.
   - `jammi_ai.connect("file://…")` returns the composed `Database`; `connect("grpc://…")`
     returns `RemoteDatabase` (`python/jammi_ai/__init__.py`).

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
locally (`clients/python/jammi_client/_conformal.py`). **Conformal** (`conformalize` /
`conformalize_interval` / `conformalize_cqr`) is likewise **CALLER-DRIVEN client-side numerics**
(`crates/jammi-python/src/database.rs`; remote computes locally via
`clients/python/jammi_client/_conformal.py`) — it does **not** ride the wire and does **not** wrap
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
stubs (`clients/python/jammi_client/_generated/jammi/v1/`) and asserts `pyproject.toml`'s declared
floors satisfy them. It has **nothing to do with the TypeScript client** (protoc-gen-es emits no
runtime version guard). The TS guard is `surface.test.ts` alone.

**Step to add to §4.1 (TS client):** *After step 8 (Python), before step 9 (Tests):* regenerate and
guard the TS client — in `clients/typescript/` run `npm run generate` (= `buf generate
../../crates/jammi-wire/proto`, emitting the gitignored `src/gen/*_pb.ts` via protoc-gen-es); for a
**new service** also add the import / `export *` / `JammiClient` field / `createClient` arm in
`clients/typescript/src/index.ts`; then extend the verb-surface guard
`clients/typescript/test/surface.test.ts` with the new RPC and run `npm run typecheck && npm run
test`.

**Publish-exclusion clarification.** The workspace has 13 members but the release publishes only 10
crates in topological order (`.github/workflows/crates.yml`: jammi-numerics → jammi-db → jammi-lora →
jammi-encoders → jammi-wire → jammi-admin → jammi-client → jammi-ai → jammi-server → jammi-cli). The
3 unpublished members carry `publish = false` in their manifests: **`jammi-python`**
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
4. Selection key: wire `EmbeddingConfig::default_index_type` (`crates/jammi-db/src/config.rs`)
   — currently dead — through the build site (`crates/jammi-ai/src/pipeline/embedding.rs`).

(Cheapest variant — a **query-time knob** like `search_expansion`: add a field to
`AnnIndexConfig` (`crates/jammi-db/src/config.rs`), map it in `SidecarIndex::index_options`
(`crates/jammi-db/src/index/sidecar.rs`), re-apply on load if query-time-mutable, pin its
default in `crates/jammi-db/src/index/sidecar.rs`.)

### 4.5 Add a new text encoder family

1. `crates/jammi-encoders/src/foo.rs` mirroring `crates/jammi-encoders/src/distilbert.rs`
   (cleanest template): `FooConfig` (serde-renamed to HF field names), per-layer structs
   holding `MaybeLoraLinear` + `crate::layer_norm::LayerNorm`, `Foo` + `FooBuilder` with the
   four builder knobs (`.pooling()`, `.lora()`, `.backbone_dtype()`, `.adapter()`). `build`
   opens `frozen_vb` via `VarBuilder::from_mmaped_safetensors` and `lora_vb` (always **F32**);
   use a `LoraSite`-style helper calling `should_apply_lora` + `effective_rank` +
   `LoraLinear::new`. Implement the full §2.5 method surface. End `forward` with
   `pool_and_normalize`. **Pick & document your safetensors key prefix.**
2. Register `pub mod foo;` + `pub use foo::{Foo, FooConfig};` in
   `crates/jammi-encoders/src/lib.rs`.
3. Add the `Foo(Foo)` variant to `AnyEncoder` (`crates/jammi-encoders/src/any.rs`) — the
   compiler forces a match arm in every method.
4. Tests: `crates/jammi-encoders/tests/it/foo.rs` + register in
   `crates/jammi-encoders/tests/it/main.rs`; add a golden/parity fixture if one exists.
5. Wire into the parent crate (same PR): inference arm in
   `crates/jammi-ai/src/model/backend/candle.rs` (box behind that file's `CandleTextForward`,
   add `"foo"` to the supported-architectures error string); fine-tune arm in
   `crates/jammi-ai/src/fine_tune/worker.rs`.

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
  applied in both places or it silently orphans saved adapters.
- **ClipText is asymmetric** inside `AnyEncoder` (errors `forward_hidden`, no-ops training
  methods, ignores the attention mask, pools via EOT-argmax).

**LoRA / training**

- **Determinism is name-keyed, not order-keyed.** Every LoRA A/B draw and dropout mask is a pure
  function of `(seed, fully-qualified-param-name)` via `seed_for_param`
  (`crates/jammi-lora/src/seeded.rs`) — never candle's global RNG, never VarMap/HashMap order. On
  CPU the same `(seed, rows, config)` → byte-identical adapters. The qualified name must match
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

- **Single-architecture determinism only** — f32/f64 summation order is fixed *per binary* but
  **not bit-equivalent across x86_64/aarch64**. Do not add parallel reduction (rayon, non-fixed-lane
  SIMD) — it breaks even the single-arch guarantee. Cross-arch reproducibility is an explicit
  non-goal.
- **f32 vs f64 reduction asymmetry is intentional** — `cosine_distance`/`cosine_similarity` in f32,
  `vector_norm`/`cosine_f64` in f64; not interchangeable (shifts last-bit results and can flip a
  tie-break).
- **Canonical-basis-before-resample** (`bootstrap_ci` sorts before drawing); the `statistic_fn` must
  be order-invariant.
- **Infallible kernels use `debug_assert_eq!` for length** — a release-mode length bug is silent;
  preserve the convention, don't "fix" it into `Result` ad hoc.

---

## 6. Build, test & release mechanics

**Toolchain.** Pinned `1.88.0` + rustfmt/clippy (`rust-toolchain.toml`). `.cargo/config.toml` sets
`rustc-wrapper = "sccache"` globally (sccache disables incremental by design — if sccache is missing,
cargo fails) and `-fuse-ld=mold` for the two linux-gnu targets *only in local dev* (in CI the
`RUSTFLAGS` env var wins). One CI/dev/release base image: `quay.io/pypa/manylinux_2_28_x86_64` →
`.docker/ci.Dockerfile` (= `jammi-ai-ci`); the CUDA image extends it.

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

**Test discipline.** Default `cargo test` is fully hermetic. Live tests gate behind a feature and must
**skip cleanly** (`tracing::warn`, never `#[ignore]` / `#[cfg(any())]` / `// TODO`). The GPU suite pins
`require_gpu=true` so a GPU-less build fails fast rather than faking parity. GPU is not testable in CI
(no GPU runners) — compile-checked only; live GPU is an A10G host gate.

**Release (tag-driven, all OIDC trusted publishing, no tokens).** A version bump PR touches the
lockstep version files (`docs/plans/50-open-core-hardening-roadmap/ROADMAP.md`, the version-bump file
list): `Cargo.toml`, `Cargo.lock`, `CHANGELOG.md`, `pyproject.toml`, `clients/python/pyproject.toml`,
`clients/typescript/package.json`, `packaging/server-cpu/pyproject.toml`,
`packaging/server-cu12/pyproject.toml`. On merge, tag both:
- **`v*`** → `.github/workflows/crates.yml` (validate → publish in topological order, skip
  already-published, block on sparse-index propagation) + `.github/workflows/npm.yml` +
  `.github/workflows/server-image.yml` (GHCR images) + `.github/workflows/release-binaries.yml`.
- **`py-v*`** → `.github/workflows/pypi.yml` (embed wheel) + `.github/workflows/pypi-client.yml`
  (pure-Python client) + `.github/workflows/pypi-server.yml` +
  `.github/workflows/pypi-server-cuda.yml` (auditwheel deliberately skipped).

**Disk pressure is a recurring real failure** — keep `CARGO_TARGET_DIR` on NVMe; the separate
`compile-check-gated` job and `crates.yml --no-verify` exist for this reason.

---

## 7. Sharp edges, tech debt & roadmap

Roadmap, sharp edges, tech-debt, and first-PR material live under
`docs/plans/52-maintainer-roadmap/` (see `sharp-edges-and-first-prs.md`). They are kept out of this
reference because they describe where the system is *going*, not what it *is*.
