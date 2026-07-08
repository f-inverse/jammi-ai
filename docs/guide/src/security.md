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
| **Tenant-scope filtering on every catalog query** | The read-side analyzer injects `tenant_id = $current OR tenant_id IS NULL` on every scan; every `register_*` and the mutable-table sink calls `assert_tenant_matches` before INSERT; the backend SQL layer also carries the predicate | `tenant_scope.rs` (`TenantScopeAnalyzerRule`) + the catalog repos' `assert_tenant_matches`; proven across the verb surface by `tenant_isolation_oracle.rs::every_case_isolation_holds` (every wire rpc covered, asserted by `every_rpc_is_covered`) |
| **Typed error surfaces** | Failures are typed variants with stable wire status, not opaque strings: a wrong on-disk shape is `JammiError::Schema`, a stale tenant write is `BackendError::TenantMismatch`, an incompatible format is `JammiError::IncompatibleFormat` | the typed-error definitions per crate; the [Format Stability](./format-stability.md) reject paths; the [tenant](./multi-tenant.md) write-guard |
| **The BYO-auth resolver seam (covers every transport)** | Tenant binding is uniformly resolver-driven: a consumer composing the engine via `assemble_grpc_chain` supplies its own `TenantResolver` (async: request metadata → `TenantScope`), which the single async tenant-binding tower layer applies to **every engine gRPC verb** AND the Flight SQL `db.sql` lane (`TenantBoundProvider` drives the *same* resolver). One authenticating resolver, plugged in once, authenticates both transports — closing the cross-transport gap where the gRPC plane was authenticated but Flight bound from the unauthenticated `jammi-session-id` header ([#220](https://github.com/f-inverse/jammi-ai/issues/220), now closed). The seam is the same one downstreams compose with; the BYO-auth seam and the composability seam are one seam. The engine still authenticates nothing on its own — the *default* resolver (`SessionIdTenantResolver`) binds the tenant a caller asserts via `jammi-session-id`; supplying an authenticating resolver is the consumer's job | the seam proof `composability_seam.rs::resolver_seam_scopes_both_transports_and_rejects_missing_credential` (gRPC + Flight isolation and `UNAUTHENTICATED`-on-missing across both transports) and the mirror `grpc_byo_auth.rs::resolver_seam_binds_the_engine_and_rejects_missing_credential`; the lower-level custom-`Interceptor`-in-front pattern (for fronting your own single service) is pinned by the four guarantees below |

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
