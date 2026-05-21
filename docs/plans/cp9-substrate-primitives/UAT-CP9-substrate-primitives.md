# Checkpoint 9: Substrate Primitives (cp9)

**Phases:** Phase 1 (Data-driven provenance channels), Phase 2 (Mutable companion tables), Phase 3 (Tenant scope), Phase 4 (Trigger stream)

**Delivers:** four engine generalisations that turn Jammi into an audit-native, event-driven, multi-tenant substrate without the engine codifying any tenant's domain — data-driven `ChannelContribution`/`merge_channels` over a catalog-registered channel table; durable `MutableTableRegistry` companion tables federated through one DataFusion `SessionContext`; `Option<TenantId>` propagation via a `TenantScopeAnalyzerRule` and write-side `assert_tenant_matches`; topic / publish / subscribe via `TriggerBroker` with backing-table replay.

This guide tests **only new functionality** introduced by cp9. Regression for everything else is automated via `cargo test --workspace`, `cargo test --test smoke`, and `python3 tests/smoke_test.py` per [PLAN-META §Checkpoints](../PLAN-META.md). For the structural decisions that underpin every primitive, see [ADR-00](./ADR-00-tenant-identifier.md), [ADR-01](./ADR-01-wire-surface.md), [ADR-02](./ADR-02-transaction-ownership.md); for the per-phase contracts, see [SPEC-01](./SPEC-01-provenance-channels.md) – [SPEC-04](./SPEC-04-trigger-stream.md).

---

## Overview

After Phase 4 lands, the public Jammi positioning sharpens from *"embeddable AI engine for federated SQL + embeddings + inference"* to *"embeddable AI engine with federated SQL, durable result tables, mutable companion tables, evidence provenance, tenant-scoped sessions, and a trigger-stream event surface."* Every clause in that sentence names a primitive a third tenant would want on its own. This UAT is the human-operator pass that proves it.

---

## Prerequisites

Run everything from the devcontainer at `/workspaces/jammi-ai`. Do not install host-side tooling; the base image carries Rust 1.88.0, `protoc`, `mold`, `sccache`, `mdbook`, and `maturin`.

```bash
# Workspace builds cleanly with all four cp9 phases landed.
cargo build --workspace --all-features

# Default-hermetic test pass (no live network).
cargo test --workspace
cargo test --test smoke
python3 tests/smoke_test.py

# Optional Postgres for SPEC-02 §3.4 / SPEC-03 §7 RLS validation.
# Skip if you are only running Shape A (embedded).
export JAMMI_TEST_PG_URL="postgres://jammi:jammi@localhost:5432/jammi_uat"

# Optional NATS JetStream for the SPEC-04 §10 production broker driver.
# The default in-memory broker covers Shapes A and B without it.
nats-server -js --port 4222 &
export JAMMI_TEST_NATS_URL="nats://127.0.0.1:4222"

# Fresh artifact directory + CLI shortcut. Every shipped CLI subcommand
# honours the JAMMI_ARTIFACT_DIR env var; the legacy `--artifacts` global
# flag was never implemented and is intentionally absent.
export JAMMI_ARTIFACT_DIR=$(mktemp -d)
alias jammi='cargo run --quiet -p jammi-cli --'
```

The four migrations cp9 introduces apply automatically on first session open; the sequence `005_TENANT_SCOPE → 006_CHANNEL_COLUMNS → 007_MUTABLE_TABLES → 008_TOPICS` is verified in the sign-off checklist. Fixtures used below live at `tests/fixtures/patents.parquet` (a generic public-domain corpus), `tests/fixtures/cp9/feature_schema.json` (tenant-neutral SCD-2 schema), and `tests/fixtures/cp9/ranking_schema.json` (tenant-neutral ranking-state schema). The `two_tenant_session_pair` / `mint_tenant_id` helpers under `crates/jammi-test-utils/src/tenant_fixtures.rs` back the engine-level workflow tests.

---

## Phase 1 — Provenance channels

**Goal:** prove that any caller can declare a custom evidence channel in the catalog and that `merge_channels` (which replaces the deleted `add_provenance` per [SPEC-01 §3.6](./SPEC-01-provenance-channels.md)) stitches the declared columns into the query result without engine code changes.

### 1.1 Register a custom channel

```bash
jammi channels register \
  --name scored_by \
  --priority 3 \
  --column ranker:Utf8 \
  --column rank_score:Float32
# → Channel 'scored_by' registered (priority=3).
```

Equivalent Rust (run from a one-off binary or `cargo test --test phase1_walkthrough`):

```rust
use jammi_engine::evidence_channel::ChannelId;
use jammi_engine::catalog::channel_repo::{ChannelSpec, ChannelColumn, ChannelColumnType};

session.catalog().channels().register(&ChannelSpec {
    id: ChannelId::new("scored_by")?,
    priority: 3,
    columns: vec![
        ChannelColumn { name: "ranker".into(),     data_type: ChannelColumnType::Utf8 },
        ChannelColumn { name: "rank_score".into(), data_type: ChannelColumnType::Float32 },
    ],
})?;
```

**Verify the append-only invariant fires:**

```bash
# Retyping the same column must fail with the typed EvidenceChannel error.
jammi channels add-column scored_by --column ranker:Int32
# → Error: Evidence channel error: channel 'scored_by': column 'ranker' was declared Utf8, cannot redeclare as Int32
```

### 1.2 Query merge — exercised by engine tests

The merged-schema property — every output batch's schema contains the four base columns *plus* `retrieved_by` (List<Utf8>), `annotated_by` (List<Utf8>), `similarity` (Float32, from `vector`), and the declared `scored_by` columns; rows surfaced only by `vector` carry `NULL` in every `scored_by` column — is exercised end-to-end by:

- `cargo test -p jammi-ai --test it uat_workflow_a_search_attribution_chain` — three retrievers contribute provenance channels (`vector`, `bm25`, `citation_graph`); the merged result has all six channel columns with NULL for rows that did not contribute.
- `cargo test -p jammi-engine evidence` — the merge unit tests cover priority/ordinal ordering, NULL projection, and tenant-neutral channel registration.

The CLI today does not ship a `jammi search --emit-contribution` invocation; the underlying property is exercised by the engine tests above and through the Rust/Python `db.search(...).run()` surfaces (per the quickstart in `README.md`). A CLI search surface is deferred — see *Wave 3* below.

### 1.3 Cookbook recipe renders

```bash
cd docs/guide && mdbook build && ls book/declare-provenance-channel.html
cargo test --doc -p jammi-ai declare_provenance_channel
```

`SUMMARY.md` lists the recipe between *"Enrich Results with Joins and Annotations"* and *"Fine-Tune for Your Domain"*.

---

## Phase 2 — Mutable companion tables

**Goal:** prove that a `MutableTableRegistry`-registered companion table participates in the same DataFusion `SessionContext` as Parquet result tables, supports transactional DML through one `CatalogBackend::transaction` closure, and survives mid-write SIGKILL with no partial commit visible.

### 2.1 Register a mutable companion table

The reusable tenant-neutral SCD-2 schema at `tests/fixtures/cp9/feature_schema.json` declares the columns `feature_id: Int64 NOT NULL`, `feature_value: Float64 NOT NULL`, `effective_from: Int64 NOT NULL`, `effective_to: Int64 NULL` (millisecond epochs — the same encoding used by the trigger-stream topic columns so the same `Int64` insertion path covers both). Subsequent examples assume `cp tests/fixtures/cp9/feature_schema.json /tmp/`.

```bash
jammi mutable create \
  --name feature_store_dimensions \
  --schema /tmp/feature_schema.json \
  --primary-key feature_id,effective_from \
  --index name=idx_active,columns=feature_id+effective_to,unique=false
# → Mutable table 'feature_store_dimensions' registered (primary_key=[feature_id,effective_from], indexes=[idx_active]).
```

### 2.2 Insert and update rows

```bash
jammi query "
  INSERT INTO mutable.public.feature_store_dimensions
  (feature_id, feature_value, effective_from, effective_to)
  VALUES
  (1842, 9.50, 1746057600000, NULL),
  (1843, 4.25, 1746057600000, NULL)
"
# → count = 2

jammi query "
  UPDATE mutable.public.feature_store_dimensions
     SET effective_to = 1747656000000
   WHERE feature_id = 1842 AND effective_to IS NULL
"
# → count = 1
```

Per [SPEC-02 §3.7](./SPEC-02-mutable-tables.md) (and the DataFusion `TableProvider::insert_into` contract — "a single row in a UInt64 column called 'count'"), each DML statement returns a one-row `count` column. The CLI prints it.

### 2.3 JOIN with a registered source

Federation across the mutable backend and a Parquet result table is exercised by `cargo test -p jammi-engine --test it mutable_federation` (mutable JOIN external source) and by the Phase-1+2 engine workflow `uat_workflow_a_search_attribution_chain`. The CLI's `query` subcommand drives the same `SessionContext`; the three-part name `mutable.public.<id>` is the registered surface (per [SPEC-02 §3.6](./SPEC-02-mutable-tables.md)).

A CLI `generate-embeddings` invocation is not shipped — the embedding-pipeline-to-mutable-JOIN flow exists in the Rust/Python API surface (`db.generate_text_embeddings(...)`, `db.sql(...)`) and is regression-covered by the AI crate's `pipeline_embedding` tests. The CLI surface for embedding generation is deferred — see *Wave 3*.

### 2.4 Atomicity under SIGKILL

```bash
# Spawn a long INSERT that streams 1000 rows; kill it after ~50.
( jammi query "
    INSERT INTO mutable.public.feature_store_dimensions
    SELECT 1000000 + s, 0.0, 1746057600000, NULL
      FROM generate_series(1, 1000) AS t(s)
  " & echo $! > /tmp/jammi.pid ; wait )
KILL_PID=$(cat /tmp/jammi.pid)
sleep 0.2 && kill -9 "$KILL_PID" 2>/dev/null || true

# Reopen the session and confirm zero partial rows.
jammi query "
  SELECT COUNT(*) AS partial
    FROM mutable.public.feature_store_dimensions
   WHERE feature_id >= 1000001
"
# → partial = 0
```

The transaction is the unit; either the closure commits via ADR-02's `CatalogBackend::transaction` or `Drop` rolls back. No row survives a mid-write kill.

### 2.5 Cookbook recipes render

```bash
cd docs/guide && mdbook build && ls book/register-mutable-table.html book/update-mutable-table.html
```

`SUMMARY.md` lists both recipes between the search/enrich recipes and the fine-tune/evaluation recipes. Every Rust sample in both recipes compiles via `mdbook test` (CI runs this automatically).

---

## Phase 3 — Tenant scope

**Goal:** prove that two `JammiSession`s in the same process with different tenant bindings see disjoint views of catalog rows, mutable-table contents, and federated source rows — and that the write-side guard rejects cross-tenant writes with a *typed* `BackendError::TenantMismatch`.

### 3.1 Two sessions, two tenants — embedded

```bash
TENANT_A=$(uuidgen | tr 'A-Z' 'a-z')
TENANT_B=$(uuidgen | tr 'A-Z' 'a-z')

jammi --tenant "$TENANT_A" \
  sources add papers_a --path tests/fixtures/patents.parquet --format parquet

jammi --tenant "$TENANT_B" \
  sources add papers_b --path tests/fixtures/patents.parquet --format parquet

# Each tenant sees its own source — and never the other's.
jammi --tenant "$TENANT_A" sources list
# → src_a (Local)
jammi --tenant "$TENANT_B" sources list
# → src_b (Local)
```

The unscoped session sees only `tenant_id IS NULL` rows (the legacy single-tenant identity per [ADR-00 §"Existing single-tenant users"](./ADR-00-tenant-identifier.md)):

```bash
jammi sources list
# → No sources registered.   ← no sources visible because both registrations are tenant-scoped
```

The CLI tenant-isolation guarantee is regression-tested at the binary boundary by `cli_sources_list_filters_by_tenant_binding` in `crates/jammi-cli/tests/it/cli.rs`.

### 3.2 Isolation across catalog + mutable_table + result_table

`uat_workflow_b_feature_store_scd_isolates_two_tenants` in `crates/jammi-ai/tests/it/uat_workflows.rs` is the executable proof: two tenants maintain `item_dimensions` mutable companion tables; each tenant sees only its own rows through the analyzer rule's predicate injection. Run:

```bash
cargo test -p jammi-ai --test it uat_workflow_b_feature_store_scd_isolates_two_tenants
```

The Flight SQL / gRPC client paths are exercised by `crates/jammi-server/tests/it/flight_tenant.rs` and `grpc_session.rs`; the shape-C harness composes them in `uat_shapes::shape_c_multi_tenant_server_isolates_two_tenants_across_primitives`. Per [SPEC-03 §5.2](./SPEC-03-tenant-scope.md), the `TenantScopeAnalyzerRule` injects `tenant_id = $current OR tenant_id IS NULL` on every `TableScan` whose schema declares the column.

### 3.3 Surface coverage — gRPC + Python + CLI

- **gRPC** — `SessionService.SetTenant` (per ADR-01 §5.4) is exercised by `crates/jammi-server/tests/it/grpc_session.rs` (Tonic interceptor inherits the tenant scope on subsequent RPCs on the same session id).
- **Python** — `db.with_tenant(...)` + `db.tenant()` (per SPEC-03 §6.3) is regression-tested in `python/tests/test_tenant.py`. *(Deferred — Wave 3 for any Python UAT script that combines tenant + new primitive in one process; this is unblocked now that Item 1's PyO3 bindings ship `PyDatabase.{create_mutable_table, drop_mutable_table, register_topic, drop_topic}`.)*
- **CLI** — `jammi --tenant "$TENANT" sources list` and equivalents above; covered by `crates/jammi-cli/tests/it/cli.rs`.

### 3.4 Write-side guard fires when tenants mismatch

A CLI `models register` subcommand is not shipped — model registration today happens through the Python `db.generate_text_embeddings` pipeline and the Rust `Catalog::register_model` API. The write-side `BackendError::TenantMismatch` guard (per [SPEC-03 §7](./SPEC-03-tenant-scope.md)) is regression-tested at the engine layer through `Transaction::assert_tenant_matches` (called from `result_repo.rs`, `model_repo.rs`, `fine_tune_repo.rs`, `eval_repo.rs`) and at the CLI layer indirectly: `sources add` under a different tenant binding inserts under the bound tenant, never the request's tenant — there is no way through the CLI to ask for a mismatch in the first place. *(Deferred — Wave 3 for a `jammi models register` CLI subcommand that exercises the explicit mismatch path.)*

### 3.5 Cookbook recipe renders

```bash
cd docs/guide && mdbook build && ls book/multi-tenant.html
# Optional second recipe:
ls book/scope-source-by-tenant.html 2>/dev/null && echo "scope-source-by-tenant present"
```

---

## Phase 4 — Trigger stream

**Goal:** prove that a catalog-registered topic supports `Publish` / `Subscribe(predicate)` / `ListTopics` per [ADR-01 §5.1](./ADR-01-wire-surface.md), that subscribers receive only batches matching their SQL predicate, and that backing-table replay covers the gap when the broker drops messages or restarts.

### 4.1 Register a topic

```bash
jammi trigger register \
  --name events.changes \
  --schema "op:string,ts_ms:int,key:string,after:string:nullable" \
  --broker-metadata '{"retention_seconds":86400}'
# → Topic 'events.changes' registered (id=<uuid>).

jammi trigger list
# → Name                              Tenant     Columns
#   events.changes                    —          op:string, ts_ms:int, key:string, after:string
```

SQL DDL (`CREATE TOPIC … (…) WITH (broker_metadata = '{…}')`) is wired through `JammiSession::sql` and exercised by `cargo test -p jammi-engine sql_create_topic`; the CLI surface above is the equivalent one-shot for scripts.

### 4.2 Publish events

```bash
cat > /tmp/events.json <<'EOF'
[
  {"op":"c","ts_ms":1747680000000,"key":"row-1","after":"{\"v\":1}"},
  {"op":"u","ts_ms":1747680001000,"key":"row-1","after":"{\"v\":2}"},
  {"op":"d","ts_ms":1747680002000,"key":"row-1","after":null}
]
EOF

jammi trigger publish \
  --topic events.changes --json-file /tmp/events.json
# → Published offset 0.
```

`--row` and `--json-file` are mutually exclusive — passing both, or neither, is rejected at the clap argument-group layer with a non-zero exit. Per-row `--row '<json>'` is still supported for one-shot scripts that already have the JSON inline.

### 4.3 Subscribe with predicate

In a second terminal:

```bash
jammi trigger subscribe \
  --topic events.changes \
  --predicate "op IN ('u','d')" \
  --from-offset 0
```

Expected output:

```text
{"offset":1,"produced_at_us":...,"row":{"op":"u","ts_ms":1747680001000,"key":"row-1","after":"{\"v\":2}"}}
{"offset":2,"produced_at_us":...,"row":{"op":"d","ts_ms":1747680002000,"key":"row-1","after":null}}
(stream stays open; press Ctrl-C to detach)
```

The predicate dialect supported is the positive enumeration from [SPEC-04 §8.2](./SPEC-04-trigger-stream.md) (column refs, comparison + boolean ops, `IS NULL`, `IN`, `LIKE`, `BETWEEN`, plus the whitelisted string functions). Anything outside that set returns `tonic::Status::invalid_argument`.

### 4.4 Replay from backing table after restart

```bash
# Restart the server to drop in-memory broker state.
pkill -f "jammi serve" && jammi serve &
sleep 1

# A new subscription with --from-offset 0 and --no-follow drains every
# event from the backing table without attaching to the live tail.
# The CLI exits once the replay window is exhausted.
jammi trigger subscribe \
  --topic events.changes --from-offset 0 --no-follow
# → {"offset":0,"produced_at_us":...,"row":{"op":"c",...}}
#   {"offset":1,"produced_at_us":...,"row":{"op":"u",...}}
#   {"offset":2,"produced_at_us":...,"row":{"op":"d",...}}
```

Per [SPEC-04 §7.2–7.3](./SPEC-04-trigger-stream.md): the backing table is authoritative; the broker is a delivery accelerator. A subscriber that asks for `from_offset=N` always sees every event with offset ≥ N regardless of broker availability. The `--no-follow` shape is the engine-level `Subscriber::replay_only` exposed at the CLI boundary — regression-covered by the `cli_trigger_subscribe_no_follow_*` tests in `crates/jammi-cli/tests/it/trigger.rs`.

### 4.5 In-memory broker → production broker switch

Swap brokers under the trait by toggling `[trigger_broker]` in `jammi.toml`:

```toml
[trigger_broker]
kind = "jet_stream"
urls = ["nats://127.0.0.1:4222"]
retention_seconds = 86400
```

Restart with `jammi --config jammi.toml serve` and re-run the publish/subscribe commands from §4.2–4.3. **Success criterion:** identical client-visible behaviour across `InMemoryBroker` and `JetStreamBroker`; no `TriggerError`; no schema or surface change visible to the client.

### 4.6 Cookbook recipes render

```bash
cd docs/guide && mdbook build && ls book/publish-events.html book/subscribe-with-filter.html
# Optional third recipe:
ls book/replay-from-backing-table.html 2>/dev/null && echo "replay recipe present"
```

---

## End-to-end third-tenant workflows

Each workflow composes multiple primitives. The discipline test — *"would a Jammi user who has never heard of $tenant care about this addition?"* — must hold for every step; each scenario below is a neutral third-tenant case (search ranking, ML feature store, generic CDC).

### Workflow A — Search-attribution chain (Phase 1 + Phase 2)

**Scenario.** A federated research-paper search tool runs three retrievers in parallel (`vector`, `bm25`, `citation_graph`); results land in a mutable companion table for incremental updates; the UI shows which retrievers found each paper and at what score.

```bash
# Declare two extra channels (vector is engine-seeded).
jammi channels register \
  --name bm25 --priority 1 --column bm25_score:Float32
jammi channels register \
  --name citation_graph --priority 4 \
  --column citation_depth:Int32 --column citation_path_score:Float32

# Mutable companion table for incremental ranking writes.
jammi mutable create \
  --name ranking_state --schema tests/fixtures/cp9/ranking_schema.json \
  --primary-key paper_id,round
```

The end-to-end attribution chain — three retrievers contribute provenance channels; the merged result has all six channel columns with `NULL` for rows that did not contribute; the ranking_state mutable table records the per-round best ranker — is exercised by:

```bash
cargo test -p jammi-ai --test it uat_workflow_a_search_attribution_chain
```

A CLI `search --persist mutable.public.ranking_state` invocation is deferred — the property is regression-covered by the engine test above. *(Wave 3.)*

**Final assertion:** every returned row's `retrieved_by` list is a subset of the channels declared above plus `vector`; no row's attribution columns reference a channel not in the catalog.

### Workflow B — Feature-store SCD (Phase 1 + Phase 2 + Phase 3)

**Scenario.** A multi-tenant ML feature store with Type-2 slowly-changing dimensions. Each tenant's features are isolated by `tenant_id`; an attribution channel `pipeline_source` records which feature-engineering pipeline produced each version.

```bash
TENANT_X=$(uuidgen | tr 'A-Z' 'a-z')
TENANT_Y=$(uuidgen | tr 'A-Z' 'a-z')

# Pipeline-attribution channel (engine-wide; channel ids are global per SPEC-01 §11).
jammi channels register \
  --name pipeline_source --priority 5 \
  --column pipeline_name:Utf8 --column pipeline_version:Utf8

# Each tenant's feature table — same name, isolated rows.
for T in "$TENANT_X" "$TENANT_Y"; do
  jammi --tenant "$T" mutable create \
    --name item_dimensions --schema /tmp/feature_schema.json \
    --primary-key feature_id,effective_from
done

# Tenant X: SCD-2 close-and-open.
jammi --tenant "$TENANT_X" query "
  INSERT INTO mutable.public.item_dimensions VALUES
    (1842, 9.50, 1746057600000, NULL)
"
jammi --tenant "$TENANT_X" query "
  UPDATE mutable.public.item_dimensions
     SET effective_to = 1747656000000
   WHERE feature_id = 1842 AND effective_to IS NULL
"
jammi --tenant "$TENANT_X" query "
  INSERT INTO mutable.public.item_dimensions VALUES
    (1842, 11.75, 1747656000000, NULL)
"

# Tenant Y: unrelated row in the same logical table name.
jammi --tenant "$TENANT_Y" query "
  INSERT INTO mutable.public.item_dimensions VALUES
    (9999, 1.00, 1747612800000, NULL)
"

# Isolation check.
jammi --tenant "$TENANT_X" query \
  "SELECT COUNT(*) FROM mutable.public.item_dimensions"   # → 2
jammi --tenant "$TENANT_Y" query \
  "SELECT COUNT(*) FROM mutable.public.item_dimensions"   # → 1
```

**Expected:** the registration succeeds for both tenants under the same name (the registry stores two rows differentiated by `tenant_id`); the close-and-open lands in disjoint backing-table partitions; a cross-tenant scan with an unscoped session returns 0. Regression-covered by `uat_workflow_b_feature_store_scd_isolates_two_tenants` in `crates/jammi-ai/tests/it/uat_workflows.rs`.

**Final assertion:** the same logical mutable table name, registered twice under two tenants, produces two row sets with empty intersection over `(feature_id, effective_from)`.

### Workflow C — CDC pipeline (Phase 2 + Phase 3 + Phase 4)

**Scenario.** A Debezium-style change-data-capture pipeline. Postgres change envelopes publish to one topic; subscribers filter by `op`; the backing table covers ad-hoc analytics.

```bash
TENANT_OPS=$(uuidgen | tr 'A-Z' 'a-z')
TENANT_OTHER=$(uuidgen | tr 'A-Z' 'a-z')

# Topic carrying CDC envelopes, scoped to the ops tenant.
jammi --tenant "$TENANT_OPS" trigger register \
  --name cdc.orders \
  --schema "op:string,ts_ms:int,key:string,before:string:nullable,after:string:nullable,source_lsn:int" \
  --broker-metadata '{"retention_seconds":604800}'

# Three subscribers — three terminal panes, three different predicates.
jammi --tenant "$TENANT_OPS" trigger subscribe \
  --topic cdc.orders --predicate "op = 'c'" &
jammi --tenant "$TENANT_OPS" trigger subscribe \
  --topic cdc.orders --from-offset 0 &
jammi --tenant "$TENANT_OPS" trigger subscribe \
  --topic cdc.orders --predicate "op IN ('u','d')" &

# Publish ten envelopes.
jammi --tenant "$TENANT_OPS" trigger publish \
  --topic cdc.orders --json-file tests/fixtures/cdc/orders_run_1.json

# Cross-tenant publish — must NOT leak into Tenant OPS's panes.
jammi --tenant "$TENANT_OTHER" trigger register \
  --name cdc.orders \
  --schema "op:string,ts_ms:int,key:string"
jammi --tenant "$TENANT_OTHER" trigger publish \
  --topic cdc.orders --json-file tests/fixtures/cdc/poison.json
```

The predicate-isolation and tenant-isolation properties — subscriber 1 receives exactly the `op = 'c'` envelopes; subscriber 2 receives all 10 envelopes from offset 0; subscriber 3 receives only `op IN ('u','d')` envelopes; Tenant OPS subscribers never see envelopes from `poison.json` — are regression-covered by `uat_workflow_c_cdc_pipeline_isolates_tenants_and_predicates` in `crates/jammi-ai/tests/it/uat_workflows.rs`.

Backing-table direct scan for ad-hoc analytics is exercised at the engine layer (the topic backing table is a regular mutable companion table named `__topic_<uuid7>` and is queryable through the same `MutableTableRegistry::scan_after` surface). A CLI flag for resolving the backing-table name from a topic name (`--output topic_id`) is deferred — *Wave 3*.

**Final assertion:** for every subscriber, `received_offsets ⊆ published_offsets` and `received_envelopes` matches the predicate; zero cross-tenant leak; the backing-table scan agrees with the union of all live subscriptions.

---

## Deployment-shape validation

The three deployment shapes from [PLAN-META](../PLAN-META.md) (Shape D — disaggregated catalog — is deferred). Every primitive must work in every applicable shape.

### Shape A — Embedded

One process, SQLite catalog, in-memory broker. The Phase 1/2/3/4 walkthroughs above all run in Shape A by default.

```bash
# Confirm: no server processes are running.
pgrep -f "jammi serve" && echo "FAIL: server should not be running" || echo "OK: embedded mode"

# Each primitive exercised via the CLI runs against the embedded engine.
jammi channels register --name embedded_ok \
  --priority 9 --column dummy:Boolean
jammi mutable create --name embedded_mut \
  --schema /tmp/feature_schema.json --primary-key feature_id,effective_from
jammi trigger register --name embedded.events --schema "msg:string"
jammi trigger list
```

Regression-covered by `shape_a_embedded_library_exercises_all_primitives` in `crates/jammi-server/tests/it/uat_shapes.rs`.

### Shape B — Single-tenant server

One server process; clients connect over Flight SQL + gRPC; no tenant scope. Regression-covered by `shape_b_single_tenant_flight_server_exercises_primitives` in `crates/jammi-server/tests/it/uat_shapes.rs` (boots `serve_flight()` on an ephemeral port; pre-seeded primitive state stays reachable through the underlying session while the server runs).

```bash
jammi serve &
SHAPE_B_PID=$!; sleep 1

# Phase 1, 2, 4 over the server surface — exercised by the Rust harness
# above. Python UAT scripts that drive every primitive over Flight SQL +
# gRPC from a Python client are *Deferred — Wave 3*; the Wave-3 scripts
# will live at tests/uat/shape_b_*.py.

kill $SHAPE_B_PID
```

### Shape C — Multi-tenant server

One server process; clients set per-connection `jammi.tenant_id` via Flight SQL `SetSessionOptions` or gRPC `SessionService.SetTenant`. Regression-covered by `shape_c_multi_tenant_server_isolates_two_tenants_across_primitives` in `crates/jammi-server/tests/it/uat_shapes.rs` (boots `serve_grpc_with_shutdown` + `serve_flight_with_session_service` on two ephemeral ports sharing one `SessionStore`).

```bash
jammi serve &
SHAPE_C_PID=$!; sleep 1
TENANT_C1=$(uuidgen | tr 'A-Z' 'a-z'); TENANT_C2=$(uuidgen | tr 'A-Z' 'a-z')

# The Phase-3 isolation guarantee at the server surface is exercised by
# the Rust harness above and by crates/jammi-server/tests/it/flight_tenant.rs
# and grpc_session.rs. A Python `shape_c_isolation.py` script that
# combines Flight SQL + gRPC + Python + CLI in one process is *Deferred —
# Wave 3*.

kill $SHAPE_C_PID
```

**Success criterion for all three shapes:** every walkthrough from §"Phase 1" through §"Phase 4" succeeds with no engine code branches keyed on shape. The substrate-primitive surface is shape-invariant.

---

## Deferred — Wave 3

The following surfaces are not part of this checkpoint and are explicitly out of scope for the cp9 UAT pass. They are listed here so operators do not look for them in the shipped CLI / scripts.

- **CLI `jammi search`** (with `--query`, `--k`, `--channel`, `--emit-contribution`, `--persist`) — the search pipeline is reachable through the Rust and Python APIs; a CLI surface is queued for Wave 3.
- **CLI `jammi generate-embeddings`** — reachable through `db.generate_text_embeddings(...)` in Rust/Python; a CLI surface is queued for Wave 3.
- **CLI `jammi models register --tenant`** — model registration today happens through the pipeline path or the Rust `Catalog::register_model` API; the explicit tenant-mismatch CLI flow is queued for Wave 3.
- **CLI `jammi trigger list --filter <name> --output topic_id`** — resolving a backing-table name from a topic name through the CLI is queued for Wave 3.
- **Python UAT scripts `tests/uat/shape_b_*.py` / `shape_c_isolation.py`** — multi-primitive Python clients driving Flight SQL + gRPC are queued for Wave 3. Item 1's PyO3 bindings (`PyDatabase.{create_mutable_table, drop_mutable_table, register_topic, drop_topic}`) ship in this branch, so the dependencies for the Wave-3 scripts are now satisfied.
- **Global `--artifacts <dir>` flag** — replaced by the `JAMMI_ARTIFACT_DIR` env var, which every shipped subcommand already honours; do not look for the flag.

---

## Cookbook

```bash
cd docs/guide && mdbook build
```

The build must complete with zero broken-link warnings. Confirm the new recipes:

| Recipe | Path | Phase | Status |
|---|---|---|---|
| Declare a Custom Provenance Channel | `src/declare-provenance-channel.md` | 1 | Required |
| Register a Mutable Companion Table | `src/register-mutable-table.md` | 2 | Required |
| Run Transactional Updates on a Mutable Table | `src/update-mutable-table.md` | 2 | Required |
| Run Jammi in Multi-Tenant Mode | `src/multi-tenant.md` | 3 | Required |
| Scope a Federated Source by Tenant | `src/scope-source-by-tenant.md` | 3 | Optional |
| Publish Events to a Topic | `src/publish-events.md` | 4 | Required |
| Subscribe with a SQL Predicate Filter | `src/subscribe-with-filter.md` | 4 | Required |
| Replay Events from the Backing Table | `src/replay-from-backing-table.md` | 4 | Optional |

```bash
grep -E 'declare-provenance-channel|register-mutable-table|update-mutable-table|multi-tenant|publish-events|subscribe-with-filter|replay-from-backing-table|scope-source-by-tenant' \
  docs/guide/src/SUMMARY.md
# All required recipes must be listed in the Cookbook section.

# Verify every recipe's "Verify" block actually executes.
cargo test --doc -p jammi-ai cp9_recipes
mdbook test docs/guide
```

Each recipe's *Verify* block (the final section per the cookbook template) must run real code that proves the recipe worked end-to-end — not a `println!` or a `Result::is_ok` token. Operator checks one recipe at random and reads the *Verify* block to confirm.

---

## Sign-off checklist

The operator ticks every box; the PR is merge-blocked until all are checked.

- [ ] Phase 1 golden-path walkthrough passes (channel register, append-only rejection, cookbook recipe renders; the merge property is verified via `uat_workflow_a_*` and the engine's `evidence` test module).
- [ ] Phase 2 golden-path walkthrough passes (register, insert, update, SIGKILL atomicity, cookbook recipes render; the federation JOIN property is verified at the engine layer).
- [ ] Phase 3 golden-path walkthrough passes (two tenants disjoint across catalog + mutable, unscoped session sees only NULL-tenant rows, write-side guard typed error verified at engine layer, cookbook recipe renders).
- [ ] Phase 4 golden-path walkthrough passes (topic register, publish via `--json-file`, predicate subscribe, replay after restart via `--no-follow`, in-memory → JetStream switch, cookbook recipes render).
- [ ] Workflow A — search-attribution chain — `cargo test -p jammi-ai uat_workflow_a_search_attribution_chain` passes; the CLI surface for `jammi search` is documented as *Wave 3*.
- [ ] Workflow B — feature-store SCD — `cargo test -p jammi-ai uat_workflow_b_feature_store_scd_isolates_two_tenants` passes with disjoint row counts (2 and 1) across two tenants on the same mutable-table name.
- [ ] Workflow C — CDC pipeline — `cargo test -p jammi-ai uat_workflow_c_cdc_pipeline_isolates_tenants_and_predicates` passes with predicate-filtered subscribers receiving only their matches and zero cross-tenant leak.
- [ ] Shape A (embedded) verified — every primitive exercised via the CLI against a SQLite catalog and in-memory broker; `cargo test -p jammi-server shape_a_*` passes.
- [ ] Shape B (single-tenant server) verified — `cargo test -p jammi-server shape_b_*` passes; Python UAT scripts deferred (*Wave 3*).
- [ ] Shape C (multi-tenant server) verified — `cargo test -p jammi-server shape_c_*` passes; Python UAT scripts deferred (*Wave 3*).
- [ ] `cargo test --workspace` clean.
- [ ] `cargo test --test smoke` clean.
- [ ] `python3 tests/smoke_test.py` clean.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` clean.
- [ ] `cargo fmt --check` clean.
- [ ] `cd docs/guide && mdbook build` clean (zero warnings, every required recipe present, `SUMMARY.md` updated).
- [ ] `mdbook test docs/guide` clean — every Rust sample in every new recipe compiles.
- [ ] No new `#[allow(...)]` attributes anywhere in the four cp9 PRs: `git log --all --since="cp9 start" -p -- crates/ | grep -E '^\+.*#\[allow' | wc -l` returns 0.
- [ ] No new `let _ = …` over a `Result`, no new `// TODO: later`, no new `#[ignore]` across the four PRs.
- [ ] No tenant-coupled fixtures introduced — the operator scans `crates/`, `tests/`, and `docs/` cp9 diffs and confirms no tenant-domain names, no tenant-shaped CSV columns, and no tenant-specific scripts appear. Flagship tenant names are forbidden by name and by likeness.
- [ ] Migration sequence applies cleanly from an empty catalog: `rm -rf $JAMMI_ARTIFACT_DIR && jammi sources list` succeeds, and the catalog reports migrations `001 → 002 → 003 → 004 → 005 → 006 → 007 → 008` applied in that order.
- [ ] Workspace `version = "0.3.0"` is the single version on every publishable crate (lockstep per *Atomic across the workspace*).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `mdbook build` fails with "unresolved link" | Recipe file exists but `SUMMARY.md` not updated | Add the entry to `docs/guide/src/SUMMARY.md` in the right group per each SPEC's cookbook section |
| `cargo test --doc` fails on a recipe code block | Recipe Rust sample drifted from the API | Fix the recipe, not the engine — docs reflect current state |
| Trigger subscribe returns `tonic::Status::invalid_argument` immediately | Predicate outside the [SPEC-04 §8.2](./SPEC-04-trigger-stream.md) dialect | Rewrite using only column refs / comparisons / `IN` / `LIKE` / whitelisted string functions; subqueries and aggregates are rejected by design |
| `BackendError::TenantMismatch` on a write | Session bound to tenant A; row carries tenant B | Rebind the session or fix the caller to set the row's `tenant_id` to A — never silently rewrite it |
| `OffsetEvicted` from in-memory broker | Subscriber lagged past `channel_capacity` | Engine falls back to backing-table replay (per [SPEC-04 §9](./SPEC-04-trigger-stream.md)); client-side, bump `[trigger_broker.in_memory] channel_capacity` |
| `connection refused` against NATS | `nats-server -js` not running | `nats-server -js --port 4222 &`; confirm `nats stream ls` lists `jammi.topic.*` streams |
| `connection refused` against Postgres | `JAMMI_TEST_PG_URL` set but server not up | Start Postgres or unset the env var; Shape A still works without it |
| `EvidenceChannel("…already declared")` on `register` | Channel id already registered (engine seed includes `vector` and `inference`) | Use `channels add-column` to extend; channel registration is one-shot per [SPEC-01 §3.3](./SPEC-01-provenance-channels.md) |
| `mutable create` fails with "reserved column name" | Schema includes a literal `tenant_id` column | Drop the column — the engine appends it implicitly per [ADR-00 §"Phase 2"](./ADR-00-tenant-identifier.md) |
| `trigger publish` fails with `--row` and `--json-file` together | clap argument-group conflict | Choose one input mode per invocation |
| `trigger publish` fails with "json file … must be a JSON object or an array of JSON objects" | Top-level JSON value is a scalar or null | Wrap the row(s) in a `[ … ]` array, or pass one object literal |
| Federation JOIN errors with "table not found" | Mutable table not registered into the current session | Registration is per-`SessionContext`; reopen or call `create_mutable_table` at startup |
| Predicate-rewrite leaves rows visible across tenants | Federated source registered without `tenant_column` | Set `tenant_column: Some("…")` on `SourceConnection`, or layer Postgres RLS per [SPEC-03 §7](./SPEC-03-tenant-scope.md) |
| Migration fails on existing artifact directory | A hand-edited catalog drifted | Rebuild against a fresh `$JAMMI_ARTIFACT_DIR` (no production users — no backwards compatibility) |

---

## References

- [`README.md`](./README.md) — the cp9 plan-group index.
- [`ADR-00-tenant-identifier.md`](./ADR-00-tenant-identifier.md) — `TenantId(Uuid)` newtype and nullable column rule.
- [`ADR-01-wire-surface.md`](./ADR-01-wire-surface.md) — `TriggerService` and `SessionService` proto surface.
- [`ADR-02-transaction-ownership.md`](./ADR-02-transaction-ownership.md) — backend-owned `transaction(|tx| ...)`.
- [`SPEC-01-provenance-channels.md`](./SPEC-01-provenance-channels.md) — Phase 1 contract.
- [`SPEC-02-mutable-tables.md`](./SPEC-02-mutable-tables.md) — Phase 2 contract.
- [`SPEC-03-tenant-scope.md`](./SPEC-03-tenant-scope.md) — Phase 3 contract.
- [`SPEC-04-trigger-stream.md`](./SPEC-04-trigger-stream.md) — Phase 4 contract.
- [`TEST-01-provenance-channels.md`](./TEST-01-provenance-channels.md), [`TEST-02-mutable-tables.md`](./TEST-02-mutable-tables.md), [`TEST-03-tenant-scope.md`](./TEST-03-tenant-scope.md), [`TEST-04-trigger-stream.md`](./TEST-04-trigger-stream.md) — automated test plans (regression, not UAT).
- [`PLAN-META.md`](../PLAN-META.md) — checkpoint UAT pattern.
