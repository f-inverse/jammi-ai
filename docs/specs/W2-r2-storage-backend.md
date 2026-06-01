# W2 — R2 as a first-class object-store backend

**Status:** spec — accepted (first-class `r2://`)
**Owner:** TBD
**Estimated effort:** 2–3 days
**Workstream dependencies:** none (storage layer is independent of W00/W1)
**Workstreams blocked by this:** W8 (a Workers + R2 SaaS deploy uses R2 for result-table storage)

> Naming note: the `W2` number is a placeholder — slot it into the roadmap as you
> see fit. The content does not depend on the number.

## Motivation

Result-table storage is one of the five pluggable backends (see
[Design Philosophy](../guide/src/philosophy.md)). Three cloud drivers are first-class
today — `s3://`, `gs://`, `azure://` — each with a typed config struct and a builder.
Cloudflare R2 is a major object store and the natural result-table + source backend for an
edge-function deployment (the engine as a container sidecar, R2 as durable storage). It
deserves the same first-class footing as its peers.

R2 is **S3-compatible**, so it is not a new connector — `object_store` has no native R2
driver; R2 is reached through the S3 driver pointed at an R2 endpoint. The work here is
*ergonomics and explicitness*, not a new capability: encapsulate R2's S3 quirks behind an
`r2://` scheme so a deployer cannot misconfigure them, and document it. This is generic engine
work — Cloudflare R2 is a cloud product, a peer to S3/GCS/Azure, not a consumer of the engine —
so it passes the discipline test and names no tenant.

## Current state (verified at spec time)

R2 **already works today** via the S3 driver with a custom endpoint:

- `crates/jammi-db/src/storage/config.rs` — `S3Config` carries `endpoint: Option<String>`
  ("Custom endpoint URL — for MinIO / LocalStack / S3-compatible services"), plus `region`,
  `access_key_id`, `secret_access_key`, `session_token`, `allow_http`.
- `crates/jammi-db/src/storage/builder.rs:build_s3` wires `with_endpoint`, `with_region`,
  `with_access_key_id`, `with_secret_access_key` onto `AmazonS3Builder`.
- So `s3://<bucket>/<key>` + `CloudConfig::S3(S3Config { endpoint: "https://<account_id>.r2.cloudflarestorage.com", region: "auto", access_key_id, secret_access_key, .. })`
  routes to R2 right now.

What is missing is **first-class footing**:

- `crates/jammi-db/src/storage/url.rs` — `enum Scheme { File, Memory, S3, Gcs, Azure }` and
  `parse_scheme` (`"s3" | "gs"|"gcs" | "azure"|"abfss"`). **No `r2`.**
- No `R2Config`; no `CloudConfig::R2`. A deployer must know R2's endpoint format and the
  `region = "auto"` requirement and hand-build an `S3Config` — an easy-to-get-wrong incantation
  for a backend that should be a named peer.

The downstream paths are already scheme-agnostic and need no change once the driver builds:

- Source registration (`AddSource` with an `r2://…` connection) goes through the same
  `build_object_store` / storage registry `driver_for` path as every other scheme.
- Result-table storage and the **USearch sidecar** round-trip already branch only
  `Scheme::File` vs. everything-else: `crates/jammi-db/src/storage/sidecar_layout.rs:save_sidecar`
  uses `save_sidecar_local` for `file://` and `save_sidecar_remote` (tempfile → object store) for
  all cloud schemes. R2, being non-`file`, takes the remote path automatically.

## Decision

**Ratified: add a first-class `r2://` scheme** as sugar over the S3 driver, rather than leaving
R2 as "use `s3://` with the right endpoint." Rationale:

- **Parity.** `gs://` and `azure://` are first-class even though each is "just an `object_store`
  driver." R2 being a named peer is consistent, not extra surface.
- **Encapsulates the footguns.** R2 requires `region = "auto"` and an account-scoped endpoint
  `https://<account_id>.r2.cloudflarestorage.com`. A first-class scheme derives both from an
  `account_id`, so a deployer supplies `account_id + keys + bucket` and cannot get the region or
  endpoint wrong.
- **Generic.** R2 is a cloud product; the addition passes the discipline test and names no
  consumer.

The alternative — document R2-via-`s3://`-endpoint and add no code — is the strict floor and
stays available (nothing here removes it). It is rejected as the *primary* answer only because it
leaves R2 a second-class incantation while its peers are named schemes.

## Change

### 1. `Scheme::R2` (`crates/jammi-db/src/storage/url.rs`)

```rust
pub enum Scheme { File, Memory, S3, Gcs, Azure, R2 }

// Display / as_str
Self::R2 => "r2",

// parse_scheme
"r2" => Ok(Scheme::R2),
```

### 2. `R2Config` + `CloudConfig::R2` (`crates/jammi-db/src/storage/config.rs`)

```rust
/// Cloudflare R2 connection details. R2 speaks the S3 API; this config derives the
/// S3 endpoint and pins region = "auto" so a deployer cannot misconfigure them.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct R2Config {
    /// Cloudflare account id. The endpoint is derived as
    /// `https://{account_id}.r2.cloudflarestorage.com` unless `endpoint` overrides it.
    pub account_id: Option<String>,
    /// Explicit endpoint override (e.g. an R2 custom domain or a test endpoint).
    pub endpoint: Option<String>,
    /// R2 access key id (an S3-style token minted in the R2 dashboard / API).
    pub access_key_id: Option<String>,
    /// Secret access key paired with `access_key_id`.
    pub secret_access_key: Option<String>,
    /// Allow plain HTTP — test endpoints only; defaults false (fail closed).
    #[serde(default)]
    pub allow_http: bool,
}

// CloudConfig
R2(R2Config),
```

`R2Config::validate()` (called by `CloudConfig::validate`): require `account_id` **or**
`endpoint`; require `access_key_id` and `secret_access_key` together (an access key without its
secret is the partial-credential mistake the existing `validate()` pattern already guards).

### 3. `build_r2` (`crates/jammi-db/src/storage/builder.rs`)

```rust
Scheme::R2 => build_r2(url, config),   // added to the exhaustive match in build_object_store

#[cfg(feature = "storage-r2")]
fn build_r2(url: &StorageUrl, config: Option<&CloudConfig>) -> Result<DynObjectStore, StorageError> {
    let bucket = /* same bucket extraction as build_s3 */;
    let r2 = match config { Some(CloudConfig::R2(r2)) => r2, _ => /* DriverInit: R2 needs config */ };
    let endpoint = r2.endpoint.clone().or_else(|| {
        r2.account_id.as_ref().map(|a| format!("https://{a}.r2.cloudflarestorage.com"))
    }).ok_or(/* DriverInit: account_id or endpoint required */)?;

    let mut builder = object_store::aws::AmazonS3Builder::from_env()
        .with_bucket_name(bucket)
        .with_region("auto")          // R2 requires this
        .with_endpoint(endpoint);
    if let Some(k) = &r2.access_key_id     { builder = builder.with_access_key_id(k); }
    if let Some(s) = &r2.secret_access_key { builder = builder.with_secret_access_key(s); }
    if r2.allow_http { builder = builder.with_allow_http(true); }
    // ... build, map err to DriverInit { scheme: Scheme::R2, .. }
}
```

`storage-r2` is a Cargo feature that reuses the same `object_store/aws` dependency as
`storage-s3` (it is the same driver underneath). `#[cfg(not(feature = "storage-r2"))]` returns
`SchemeNotEnabled` — mirroring the S3/GCS/Azure pattern.

### 4. No change required downstream

Source-read registration, result-table storage, and the USearch sidecar remote round-trip are
already scheme-agnostic (see *Current state*). Once `build_object_store` returns an R2 driver,
`r2://bucket/key` works for `AddSource`, `GenerateAudioEmbeddings` output, and `search` over the
resulting index with no further change.

## Tests

Hermetic by default (no live R2):

- **Pure unit** — `Scheme` round-trips `"r2"`; endpoint derivation from `account_id`
  (`account_id = "abc"` → `https://abc.r2.cloudflarestorage.com`) and the `endpoint` override;
  `R2Config::validate()` rejects a lone `access_key_id` and a config with neither `account_id`
  nor `endpoint`.
- **Driver build** — `build_object_store(r2://bucket/key, CloudConfig::R2{..})` returns a driver
  (does not hit the network); `SchemeNotEnabled` when the feature is off.
- **Live (opt-in, gated like the existing cloud tests)** — round-trip a small object against a
  MinIO/LocalStack S3-compatible endpoint (and, manually, real R2), reusing the S3 live-test
  harness since the driver is shared.

## Docs

- Extend `docs/guide/src/cloud-storage.md` with an R2 section: `account_id + access keys + bucket`,
  `r2://bucket/key`, and the note that R2 is the S3 driver under the hood.
- Note R2 under result-table storage in `docs/guide/src/configuration.md`.

## Discipline check

- "R2" / "Cloudflare R2" is a cloud-product name, a peer to S3/GCS/Azure — **not** a consumer/tenant
  name. The addition is something any deployer on Cloudflare reaches for. Passes the discipline
  test; references no consumer.
- Additive only: no existing scheme, config, or path changes shape; single-tenant and other-cloud
  deployments are untouched.
