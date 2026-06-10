# Scope a Session to a Tenant

When more than one logical tenant shares a Jammi engine — a SaaS feature
store serving two ML teams, a research workbench shared across three labs,
a notebook product hosting one project per student — every catalog read and
write needs to belong to the right tenant. Jammi's session-scoped tenant
binding does this without the caller having to spell a `WHERE tenant_id = …`
clause on every query.

## Goal

After this recipe you can:

1. Bind a tenant to a session in Rust, Python, and on the CLI.
2. Verify that two sessions on the same process see disjoint rows.
3. Bind a tenant on a remote client via the gRPC `CatalogService` so
   subsequent Flight SQL queries from the same connection observe the
   tenant.

## Setup

Every example below assumes a configured `JammiConfig` (defaults are fine
for the recipe). The tenant identifier is a UUID v4 or v7 string — the
engine refuses the nil UUID (`00000000-…`) at the `TenantId` newtype
boundary.

## Rust

```rust,no_run
# extern crate jammi_db;
# extern crate tokio;
use std::str::FromStr;
use jammi_db::TenantId;
use jammi_db::session::JammiSession;
use jammi_db::config::JammiConfig;

# async fn ex() -> jammi_db::error::Result<()> {
let config = JammiConfig::default();
let alice = TenantId::from_str("018f5a0e-c4c8-7e10-9c4f-3b6f7c5a8e9a")?;

let session = JammiSession::new(config).await?.with_tenant(alice);
// Every catalog read and write on `session` now scopes to Alice.
# Ok(())
# }
```

`with_tenant` is a builder that consumes `self` and returns `Self`, so it
chains naturally. If you hold a session behind `Arc`, use `bind_tenant(&t)`
to update the binding in place — the session shares one `TenantBinding`
across all references.

## Python

```python
import jammi_ai

db = jammi_ai.connect("file:///tmp/jammi")
db.with_tenant("018f5a0e-c4c8-7e10-9c4f-3b6f7c5a8e9a")

# Subsequent calls observe Alice's tenant scope.
db.add_source("inbox", path="/data/alice/inbox.parquet", format="parquet")
db.sql("SELECT * FROM inbox.public.inbox")
```

Pass an empty string to clear: `db.with_tenant("")`.

## CLI

The `--tenant` flag is global; it applies to every subcommand.

```bash
jammi --tenant 018f5a0e-c4c8-7e10-9c4f-3b6f7c5a8e9a sources list
jammi --tenant 018f5a0e-c4c8-7e10-9c4f-3b6f7c5a8e9b models list
```

## Remote clients (gRPC + Flight SQL)

A programmatic client (Python, Go, Java) binds the tenant once per
connection via the `jammi.v1.catalog.CatalogService.SetTenant` RPC. The
server records the tenant against the `jammi-session-id` request metadata
header; every Flight SQL query the same connection issues afterwards
inherits the binding through the `TenantInterceptor` that fronts both
services. Browser clients reach the same `CatalogService` over HTTP/1.1
via the gRPC-Web shim (`application/grpc-web+proto`) — no separate REST
surface, same `jammi-session-id` header semantics.

```python
import grpc
from jammi.v1 import catalog_pb2, catalog_pb2_grpc

channel = grpc.insecure_channel("jammi.example.com:50051")
metadata = [("jammi-session-id", "my-client-uuid")]

client = catalog_pb2_grpc.CatalogServiceStub(channel)
client.SetTenant(
    catalog_pb2.SetTenantRequest(
        tenant=catalog_pb2.Tenant(id="018f5a0e-c4c8-7e10-9c4f-3b6f7c5a8e9a")
    ),
    metadata=metadata,
)
# Subsequent Flight SQL queries on the same channel + jammi-session-id
# observe Alice's tenant scope.
```

## Disjoint views — what to expect

Two sessions on the same process, bound to different tenants, will:

1. **Read each other as invisible:** `list_sources()` returns the calling
   tenant's sources plus any globally-scoped (`tenant_id IS NULL`) sources.
2. **Write into different lanes:** a `register_source` from Alice produces
   a row tagged `tenant_id = alice`; Bob's `list_sources` does not see it.
3. **Share globally-scoped rows:** an unscoped (`tenant_id IS NULL`)
   registration — typically a public reference dataset — is visible to
   every tenant.

The engine enforces the binding at three layers (the SPEC-03 defence-in-depth
discipline):

- **Read-side predicate injection** — `TenantScopeAnalyzerRule` injects
  `tenant_id = $current OR tenant_id IS NULL` on every `TableScan` whose
  schema declares the column.
- **Write-side guard** — every catalog `register_*` and the mutable-table
  sink calls `Transaction::assert_tenant_matches` before INSERT.
- **Storage-side filter** — catalog repo reads also pass the predicate to
  the backend SQL layer, so the wrong tenant's rows never leave the
  database.

A buggy caller that constructs a row with the wrong `tenant_id` gets
`BackendError::TenantMismatch` from the guard layer.

## When the binding doesn't apply

- **External federated sources** without a `tenant_id` column — Jammi's
  analyzer rule has no column to inject against, so those sources show
  every row to every tenant unless the source declaration registers a
  `tenant_column` override. Catalog tables and mutable companion tables
  always carry the column.
- **Cross-tenant `WHERE` clauses** the caller writes by hand — a query
  that contains `WHERE tenant_id = 'other-tenant'` runs against the
  injected predicate plus the user's clause; the analyzer rule does not
  remove user-written predicates.
- **Single-tenant deployments** — bind nothing and every row is global; no
  predicate is injected beyond `tenant_id IS NULL`.

## See also

- The discipline test in [`SPEC-03`](https://github.com/f-inverse/jammi-ai/blob/main/docs/plans/cp9-substrate-primitives/SPEC-03-tenant-scope.md)
- [`Register a Mutable Companion Table`](./external-sources.md) for how a
  mutable companion table also honours the tenant binding on write
