# Security Posture

Jammi is an **engine on a trusted network**, not a security boundary. This page
is the published threat model: precisely **what the engine defends**, **what it
explicitly does not**, the **trusted-network assumption** every deployment
inherits, and the **consumer's responsibilities** for the boundary the engine
deliberately does not own. Every "defends" line below traces to a real test or
code path; every "does not" line is the honest absence of a guarantee the engine
never claims.

The single principle, stated once and not softened: **Jammi authenticates
nothing.** Identity, authorization, and the network perimeter are a consumer's
vocabulary; the engine ships the *seam* a consumer plugs them into, never the
policy. A deployment that exposes the engine's port to an untrusted caller has
removed the boundary the engine assumes is there.

## What the engine defends

Each line names the mechanism and the test or code path that proves it.

| Defence | Mechanism | Traces to |
|---|---|---|
| **Format-version reject-newer** | A persisted artifact stamped with a version newer than this build knows is a *typed rejection*, never a silent misparse into wrong data | `manifest.rs` (`UnsupportedManifestVersion`, the `read`-path guard) + test `newer_manifest_version_is_rejected`; `sidecar.rs` (`IncompatibleFormat`) + test `newer_rowmap_version_is_rejected` |
| **Tenant-scope filtering on every catalog query** | The read-side analyzer injects `tenant_id = $current OR tenant_id IS NULL` on every scan; every `register_*` and the mutable-table sink calls `assert_tenant_matches` before INSERT; the backend SQL layer also carries the predicate. A Jammi-owned result table carries no `tenant_id` column (it is wholly owned by one tenant, or GLOBAL), so a tenant-gating result-table schema provider gates *resolution* on the catalog owner instead — over every lane (Flight `db.sql`, gRPC `sql`, search) a correctly-bound tenant resolves only its own and GLOBAL result tables; a peer's private table resolves not-found | `tenant_scope.rs` (`TenantScopeAnalyzerRule`) + `store::result_schema` (`ResultTableSchemaProvider`) + the catalog repos' `assert_tenant_matches`; proven across the verb surface by `tenant_isolation_oracle.rs::every_case_isolation_holds` (every wire rpc covered, asserted by `every_rpc_is_covered`) |
| **Typed error surfaces** | Failures are typed variants with stable wire status, not opaque strings: a wrong on-disk shape is `JammiError::Schema`, a stale tenant write is `BackendError::TenantMismatch`, an incompatible format is `JammiError::IncompatibleFormat` | the typed-error definitions per crate; the [Format Stability](./format-stability.md) reject paths; the [tenant](./multi-tenant.md) write-guard |
| **The BYO-auth resolver seam (covers every transport)** | Tenant binding is uniformly resolver-driven: a consumer composing the engine via `assemble_grpc_chain` supplies its own `TenantResolver` (async: request metadata → `TenantScope`), which the single async tenant-binding tower layer applies to **every engine gRPC verb** AND the Flight SQL `db.sql` lane (`TenantBoundProvider` drives the *same* resolver). One authenticating resolver, plugged in once, authenticates both transports — closing the cross-transport gap where the gRPC plane was authenticated but Flight bound from the unauthenticated `jammi-session-id` header ([#220](https://github.com/f-inverse/jammi-ai/issues/220), now closed). The seam is the same one downstreams compose with; the BYO-auth seam and the composability seam are one seam. The engine still authenticates nothing on its own — the *default* resolver (`SessionIdTenantResolver`) binds the tenant a caller asserts via `jammi-session-id`; supplying an authenticating resolver is the consumer's job | the seam proof `composability_seam.rs::resolver_seam_scopes_both_transports_and_rejects_missing_credential` (gRPC + Flight isolation and `UNAUTHENTICATED`-on-missing across both transports) and the mirror `grpc_byo_auth.rs::resolver_seam_binds_the_engine_and_rejects_missing_credential`; the lower-level custom-`Interceptor`-in-front pattern (for fronting your own single service) is pinned by the four guarantees below |
| **The `AdminAuthorizer` capability gate — gRPC-only, one grant, one transport** | Unlike the resolver seam above (one grant that covers both transports), `CatalogService.Reconcile`'s cross-tenant `all = true` admin pass is a SEPARATE, narrower gate: a synchronous `AdminAuthorizer` (`fn authorize(&self, metadata: &MetadataMap) -> Result<(), Status>`) a deployment supplies at `GrpcChain.admin_authorizer`. It is gRPC-only **by construction**, not by omission — `Reconcile` has no Flight SQL analogue to gate. The shipped default is `admin_authorizer: None`, which refuses EVERY `all = true` request with `PERMISSION_DENIED` naming this page; a tenant-scoped `Reconcile` (`all = false`) never consults it at all and runs under the caller's own resolved tenant scope like any other verb | `grpc/catalog.rs` (`AdminAuthorizer`, `CatalogServer::admin_authorizer`); default-deny proven by `grpc_remote_session.rs::remote_reconcile_all_is_denied_by_default_without_an_authorizer`; an authorized pass by `grpc_remote_session.rs::remote_reconcile_reports_like_local` and the cross-tenant isolation oracle `tenant_isolation_oracle.rs::assert_reconcile_isolated`, both via the test-only `AllowAllAdmin` (`tests/it/common/grpc.rs`); a worked example implementation is in [Scope a Session to a Tenant → Bring your own auth](./multi-tenant.md#bring-your-own-auth) |

The BYO-auth seam's contract is pinned by `grpc_byo_auth.rs` as a worked
example. The resolver-seam tests above prove it through the engine's own
composability seam across both transports; the custom-interceptor-in-front form
(a consumer fronting its own single service) additionally pins four guarantees,
each its own test:

- **Missing credential → unauthenticated.** No token fails the request before any
  handler runs, so the caller reads nothing — it does not fall through to an
  unscoped read (`missing_credential_is_rejected_not_run_unscoped`).
- **Forged claim → unauthenticated.** A token whose signature does not cover its
  tenant claim is rejected; a forged tenant buys nothing because the signature
  covers the claim (`invalid_credential_is_rejected`).
- **A rejected caller does not fall through.** The interceptor *fails the
  request* rather than binding `None` — there is no path by which an
  unauthenticated caller silently reads another tenant's rows (the same test
  proves a *valid* token for the very tenant the forgery claimed does resolve, so
  the rejection was the signature, not a tenant blocklist).
- **Per-tenant isolation through the seam.** Two callers presenting valid tokens
  for two distinct tenants each see only their own tenant's sources, end to end
  through the authenticating interceptor
  (`two_authenticated_tenants_see_isolated_sources`).

## What the engine explicitly does NOT defend

These are honest absences. The engine never claims them; a deployment that needs
them supplies them above the engine.

- **It authenticates nothing.** There is no built-in credential check on any
  verb. The default `SessionIdTenantResolver` reads the `jammi-session-id` header
  and binds the tenant the caller asserts (or the explicit `Global`/unscoped
  scope when none is bound) — it verifies nothing about who the caller is. A
  deployment that needs authentication supplies its own `TenantResolver` at the
  seam.
- **It ships no authz / RBAC / SSO.** There is no role model, no permission
  check, no policy engine, no identity-provider integration. Authorization is a
  consumer's vocabulary and lives above the seam.
- **`jammi-session-id` is a correlation id, NOT a credential or principal.** It is
  a client-minted, opaque transport correlation id identifying a *connection*,
  not a person. Anyone who presents another session's id assumes that session's
  tenant. It is never an authentication or authorization boundary.
- **No TLS / secrets / IAM.** Transport encryption, secret management, key
  rotation, and cloud IAM are the consumer's runtime, not the engine's — the same
  line the [Design Philosophy](./philosophy.md) draws around load balancing,
  ingress, and orchestration.
- **The peer listener authenticates nothing either (I-PEER).** `[server]
  peer_bind` (unset by default) serves the engine-internal segment-search
  seam to other replicas and trusts the channel — see
  [The peer listener](#the-peer-listener-i-peer) below.

## The peer listener (I-PEER)

`[server] peer_bind` opens a separate internal listener serving
`jammi.v1.peer.PeerService` — the seam a coordinator replica fans a search
out through to the replicas that own a table's segments (see [Beyond one
node](./reference-topologies.md#beyond-one-node-retrieval)). Its threat model
is stated as one invariant, **I-PEER**:

- **Every client of `peer_bind` is a jammi coordinator.** The owner handler
  trusts the channel: the request carries no tenant, the owner binds none and
  reads no `result_tables` row. It enforces exactly one thing at its input
  edge — every requested segment id belongs to the named table (else the whole
  request is refused) and the bundle's stamped precision matches.
- **Tenant scope is enforced once, at the coordinator.** The coordinator's
  `Search` resolved the table through its own tenant-scoped catalog read
  (`tenant_id = $current OR tenant_id IS NULL`) before any fan-out, so a
  coordinator bound to tenant B cannot name tenant A's table — it fails before
  a single peer call. The owner is the second half of that one predicate, not a
  second predicate.
- **The public listener never reaches it.** The peer routes are built outside
  `assemble_grpc_chain`, never wrapped by the tenant-binding layer, never
  advertised by `GetServerInfo`; the public listener answers `UNIMPLEMENTED`
  for `/jammi.v1.peer.PeerService/*` (proven by the tenant-isolation oracle).
- **Binding `peer_bind` on a routable interface without network policy / mTLS
  exposes cross-tenant segment reads** to anyone who can reach the port. The
  listener speaks plaintext gRPC like every other engine port; encryption and
  peer authentication are the runtime's (a mesh, a network policy, mTLS at a
  sidecar), exactly as for the public listener. Default unset = no listener.

## Transport encryption is the deployer's runtime, not the engine's

The engine speaks plaintext gRPC and Flight SQL and ships no TLS code
path; transport encryption is the deployer's runtime. It follows from the
same primitives this page and the [Design Philosophy](./philosophy.md)
already state:

- **B4 ("one binary, every topology") constrains what the *engine* forks
  on, not what fronts it.** Terminating TLS is supplied by the runtime the
  engine deploys into — a proxy, a mesh sidecar, a load balancer — and that
  termination is not a topology-specific code path the engine would need to
  special-case per shape. A `tls` cargo feature would itself be the kind of
  server-only gate B4 refuses: a build-time fork between "the engine" and
  "the engine, but for a server."
- **Passing the discipline test is necessary, not sufficient.** A user who
  has never heard of any consumer does want the wire encrypted — TLS passes
  the [discipline test](./philosophy.md#the-discipline-test) on its own. But
  the boundary table and the paragraph that follows it
  (`docs/guide/src/philosophy.md:116-125`) are the second gate: TLS,
  secrets, IAM, ingress, and load balancing all pass the
  discipline test and are *still* placed in the consumer's runtime, not the
  engine, because the table asks a second question the discipline test does
  not — does owning this turn the engine into infrastructure it isn't. TLS
  termination answers yes.
- **A `[server] tls` key, in the file or the environment, is a typed
  refusal, not a silent no-op.** `ServerConfig` — what `[server]`
  deserializes into — is a `#[serde(default, deny_unknown_fields)]` struct
  (`crates/jammi-db/src/config/mod.rs:1123`), the same discipline
  `JammiConfig` itself carries at its top level
  (`crates/jammi-db/src/config/mod.rs:204`). A `[server] tls = …` stanza in
  a config file, and `JAMMI_SERVER__TLS` in the environment, are not
  silently ignored — each is a typed `JammiError::Config` startup refusal,
  naming the unrecognised key.
- **There is no engine-side certificate to hand the `TenantResolver` seam.**
  Because termination happens outside the engine, the engine never sees a
  peer certificate to map onto a tenant — that mapping, if a deployment
  wants one, lives in the terminator or the proxy in front, not at the
  seam described under [The identity seam](./deploy-server.md#the-identity-seam).
  The CLI's `--target` refuses `grpcs://` and `https://` with a typed error
  naming the accepted schemes (`crates/jammi-cli/src/main.rs:170-187`; the
  CHANGELOG's "drop `grpcs://` and `https://` as accepted `--target`
  schemes" entry (#480), commit `616bb6d4`) rather than advertising a
  transport it cannot speak
  — put a TLS-terminating proxy in front and point `--target` at it in
  plaintext (`grpc://`/`http://`). This is an asymmetry between the two
  clients: the Python SDK's `RemoteTarget` legitimately keeps
  `grpcs://`/`https://` in its own scheme table
  (`clients/python/jammi/_target.py:51-54`) because it is a general client
  library reaching whatever endpoint a deployment publishes (including a
  TLS-terminating proxy), while the CLI is the engine's own admin surface
  and names only the schemes the engine itself speaks.

**Shape B with no mesh** — an on-prem single-tenant deployment that has no
ingress or mesh to terminate TLS for it — still gets encryption: put a
terminator in front of the engine's plaintext listeners. A minimal,
consumer-neutral example with Caddy:

```text
# Caddyfile — terminates TLS and forwards plaintext to the engine's
# loopback-bound listeners (see docker-compose.yml).
#
# `tls internal` issues Caddy's own locally-trusted certificate: a private
# DNS name (no public record) has no ACME challenge path to a public CA,
# so automatic Let's Encrypt/ZeroSSL issuance is not an option here.
jammi.example.com {
    tls internal
    reverse_proxy h2c://127.0.0.1:8081  # gRPC + Flight SQL
}
health.jammi.example.com {
    tls internal
    reverse_proxy 127.0.0.1:8080        # /healthz, /readyz, /metrics
}
```

The reference [`deploy/docker-compose.yml`](https://github.com/f-inverse/jammi-ai/blob/main/deploy/docker-compose.yml)
binds its published ports to `127.0.0.1` for exactly this shape: the
compose stack publishes the engine's ports for a terminator running on the
same host to reach, not for direct exposure to an untrusted network. A
terminator running as a container on the same Compose network instead
reaches the engine by its service name rather than `127.0.0.1` —
`reverse_proxy h2c://jammi-server:8081` — since two containers on the same
Compose network share that network, not the host's loopback interface.

## The trusted-network assumption

Every Jammi deployment that uses the default `SessionIdTenantResolver` assumes a
**trusted network**: a private VPC, a sidecar mesh, or a single-process embedding
where every caller is already inside the trust boundary. On that network, binding
the tenant a caller asserts via `jammi-session-id` is the right, low-friction
trade-off. The moment an untrusted caller can reach the port, that trade-off is
wrong — and closing it is the consumer's job, via an authenticating
`TenantResolver` at the seam, not a flag the engine flips.

## Tenant scope is an organizational mechanism, not an access-control boundary

This distinction is load-bearing. Tenant-scope filtering (above) is an
**organizational** mechanism: it keeps one tenant's catalog rows from appearing
in another tenant's *correctly-bound* reads, so a multi-tenant deployment stays
tidy and a buggy caller that writes the wrong `tenant_id` is refused by
`assert_tenant_matches`. It is **not** an access-control boundary: it does not
decide *which tenant a caller is entitled to act as*. Nothing in the engine
prevents an unauthenticated caller from asserting any tenant it likes via
`jammi-session-id` and reading that tenant's rows. Access control — proving a
caller may act as the tenant it claims — is exactly what the BYO-auth seam adds
*in front of* the scope mechanism. Treating tenant scope as if it were
authorization is the misuse this page exists to forestall.

## The consumer's responsibilities

To put a real tenant boundary in front of untrusted callers, a consumer supplies
the authentication and authorization the engine deliberately omits by
implementing a `TenantResolver` and passing it to `assemble_grpc_chain` — one
plug that binds every engine gRPC verb and the Flight `db.sql` lane through the
single tenant-binding mechanism (the [tenant
recipe](./multi-tenant.md#bring-your-own-auth) and the worked example in
`grpc_byo_auth.rs`):

1. **Authenticate** the principal. In `resolve`, read and verify the caller's
   credential (a bearer token, an exchanged session cookie, a service-to-service
   token). A missing or invalid credential returns `Err(Status::unauthenticated)`
   here, before any handler runs.
2. **Authorize the tenant from the verified claim.** Derive the tenant from the
   *verified* claim — never from a header the caller controls. This is where the
   consumer's policy lives: which tenant this principal may act as. Return
   `Ok(TenantScope::Tenant(t))`.
3. **The engine binds it.** The async tenant-binding layer maps the resolved
   scope onto the `SessionTenant` request extension every verb handler resolves,
   and `TenantBoundProvider` binds it for Flight — the consumer writes only
   `resolve`.

Because `resolve` runs *in front of* every handler, the tenant the engine acts on
is the one the credential proves, not one the caller asserts. **Reject, don't
default:** an authenticating resolver returns `Tenant`/`Err` and NEVER
`TenantScope::Global` — returning `Global` (or, in the lower-level
interceptor-in-front form, binding `None`) on a failed check runs the request
unscoped, which for a `tenant_id IS NULL`-bearing catalog is a global read, so a
rejected caller must fail the request. `TenantScope::Global` is the *explicit*
unscoped choice the default (OSS-cooperative) resolver returns when no tenant is
bound — never a value a rejection falls through to. That framing rule is the
defect `grpc_byo_auth.rs`'s `missing_credential_is_rejected_not_run_unscoped` and
the seam mirror `resolver_seam_binds_the_engine_and_rejects_missing_credential`
guard against.

## Dependency-advisory posture

The engine's dependency tree is gated in CI by `cargo deny` against the RustSec
advisory database, plus a license allowlist and the source/ban guards that
formalize the engine's one-way dependency direction (no proprietary or
non-crates.io crate in the OSS closure). The advisory lane runs the live RustSec
DB on every PR; a documented exception in `deny.toml` records any advisory the
release knowingly carries, with a written rationale, rather than destabilizing
the freeze with a risky bump. The config is `deny.toml` at the repo root.
