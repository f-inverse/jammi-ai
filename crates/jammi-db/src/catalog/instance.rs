//! The gang-membership carrier: [`PeerAddr`] (the wire address a coordinator
//! dials a gang member / segment owner at — moved here from `index::peer`,
//! re-exported there, so the peer listener and the gang listener can never
//! drift into two distinct address types), [`MemberRoot`] (the VERBATIM
//! configured result-table root — see
//! [`crate::config::JammiConfig::resolved_result_root`] — carried on the
//! `instances` row byte-for-byte, paired with its [`RootIdentity`]: the
//! identity of that root ACROSS SPELLINGS, computed once by the process
//! that owns the root and carried on the same row, which IS what the
//! membership predicate compares), [`WorkerFacts`] (the claim-loop half of
//! a registration — owned exclusively by `JobWorker`,
//! `crates/jammi-ai/src/fine_tune/worker.rs`), and
//! [`InstanceRegistration`] — the ONE value every writer of the `instances`
//! (+ `workers`) row builds, through [`InstanceRegistration::from_config`].
//! [`GangListing`] / [`GangMember`] are [`super::Catalog::list_gang_members`]'s
//! request and response shapes; the predicate admits on `kinds` +
//! `workers.state` + `peer_addr` presence + freshness + self-exclusion +
//! ROOT IDENTITY EQUALITY (`docs/plans/67-distributed-training/README.md`
//! unit U5b-1a-A2, the predicate U5b-1a filed and this unit builds).
//!
//! **The root is interpreted in exactly one place, by its owner, at
//! registration** (`MemberRoot::resolved`, the only production constructor;
//! the derivation itself is private to this module): the verbatim string
//! is parsed by the SAME [`crate::storage::StorageUrl`] parser the result
//! store roots itself through (so `gcs://` and `gs://`, `abfss://` and
//! `azure://` fold by the one alias table the store already owns), an
//! object-store key is normalised by the SAME `object_store::path::Path`
//! parser the store hands its keys to, the endpoint or account the store
//! would dial for that scheme is part of the identity, a local root is
//! CREATED (as the store creates it at open) and then canonicalised on the
//! owner's own filesystem, and an in-memory root is refused as unshareable.
//! Nothing on this path re-roots anything: the store still roots at the
//! verbatim string, the row still carries that string verbatim, and the
//! identity is a SEPARATE column used only for equality.

use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::backend::{BackendKind, SqlValue};
use super::jobs_repo::WorkerState;
use super::lease::stale_before_clause;
use crate::error::{JammiError, Result};
use crate::storage::{BuilderSeeds, Scheme, StorageUrl};

/// The address a coordinator dials a gang member / segment owner at
/// (`host:port`, plaintext gRPC — transport encryption is the runtime's,
/// never the engine's). Sealed: the only way to build one is
/// [`PeerAddr::parse`], so a `PeerAddr` value is always a validated
/// `host:port` pair, never an arbitrary string smuggled past construction.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PeerAddr(String);

impl PeerAddr {
    /// Parse `host:port`: a non-empty host (a DNS name, an IPv4 literal, or
    /// a BRACKETED IPv6 literal — this type never resolves it, only
    /// validates the wire-form shape) and a port in `1..=65535`.
    /// `rsplit_once(':')` so a bracketed IPv6 literal's own colons stay
    /// inside the host segment and only the LAST colon is treated as the
    /// host/port separator. An UNBRACKETED IPv6 literal (`2001:db8::1:9000`)
    /// is refused, never silently split on its own last colon: the split is
    /// ambiguous (nothing distinguishes "the last hextet's `:` continues the
    /// address" from "the last `:` is the port separator"), so `[::1]:9000`
    /// is the only accepted IPv6 wire form (P-Y4, contract
    /// `feat_500-C-U5b-1a` §12).
    pub fn parse(input: &str) -> Result<Self> {
        let (host, port) = input.rsplit_once(':').ok_or_else(|| {
            JammiError::Config(format!("peer address '{input}' must be 'host:port'"))
        })?;
        if host.is_empty() {
            return Err(JammiError::Config(format!(
                "peer address '{input}' has an empty host"
            )));
        }
        if host.contains(':') && !(host.starts_with('[') && host.ends_with(']')) {
            return Err(JammiError::Config(format!(
                "peer address '{input}' has an unbracketed IPv6 host; wrap it in \
                 brackets, e.g. '[{host}]:{port}'"
            )));
        }
        let port: u16 = port.parse().map_err(|_| {
            JammiError::Config(format!(
                "peer address '{input}' has an invalid port (must be 1..=65535)"
            ))
        })?;
        if port == 0 {
            return Err(JammiError::Config(format!(
                "peer address '{input}' must have a non-zero port"
            )));
        }
        Ok(Self(input.to_string()))
    }

    /// The validated address, in its wire form (`host:port`).
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for PeerAddr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The VERBATIM configured result-table root — the exact string
/// [`crate::store::ResultStore`] roots itself at (never re-derived, never
/// re-parsed, never re-spelled) — PAIRED with its [`RootIdentity`], computed
/// here, once, from that string by the process that owns the root.
/// [`Self::resolved`] is the ONLY production constructor — it calls
/// `resolved_result_root` itself and derives the identity from its output,
/// so a `MemberRoot` can never carry a string that did not come from the
/// resolver, nor an identity that was not derived from that string.
/// `MemberRoot::new` wraps an arbitrary string (deriving its identity the
/// same way); it is compiled only under `feature = "test-hooks"`
/// (fixtures/tests), never in a production build (not a doc link: the
/// method does not exist in a build without that feature, so an intra-doc
/// link to it fails `cargo doc`'s default-feature pass) — the string
/// constructor being reachable from production code is exactly how an
/// unrelated string ends up in the `instances.result_root` column.
///
/// The row carries both halves: `instances.result_root` (this string,
/// verbatim — a human reads the configured spelling) and
/// `instances.result_root_identity` (the identity — what
/// [`super::Catalog::list_gang_members`] compares, see [`GangListing`]).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemberRoot {
    root: String,
    identity: RootIdentity,
}

impl MemberRoot {
    /// The ONE production constructor: the verbatim root
    /// [`crate::config::JammiConfig::resolved_result_root`] computes for
    /// `config` — the exact string [`crate::store::ResultStore`] roots
    /// itself at — and its [`RootIdentity`].
    /// [`InstanceRegistration::from_config`] is this method's only caller;
    /// nothing else builds a `MemberRoot` in a production build (`new`
    /// below does not exist outside `feature = "test-hooks"`).
    ///
    /// # Errors
    ///
    /// `resolved_result_root`'s own non-UTF-8 refusal, and
    /// the identity derivation's refusals (an in-memory root, a root the store's
    /// URL parser rejects, a local root with no resolvable ancestor).
    pub fn resolved(config: &crate::config::JammiConfig) -> Result<Self> {
        let root = config.resolved_result_root()?;
        let identity = RootIdentity::of(
            &root,
            config.storage.cloud.as_ref(),
            &BuilderSeeds::from_env(),
        )?;
        Ok(Self { root, identity })
    }

    /// Wrap an ALREADY-RESOLVED root string directly, with no resolver call
    /// — fixtures and tests only. Its identity is derived exactly as
    /// [`Self::resolved`] derives it with no `[storage.cloud]` section (a
    /// fixture that needs an endpoint in the identity goes through
    /// [`Self::resolved`] with a real config); a root with no identity (an
    /// in-memory root, an unknown scheme, a local root this process cannot
    /// create) panics here, naming the reason, since a fixture asking for an
    /// unshareable member is a fixture bug. Gated
    /// behind `feature = "test-hooks"` so a production build never links
    /// this constructor: reachable from production code, it would let any
    /// caller put an arbitrary string in the `instances.result_root`
    /// column, defeating the one property this type exists to hold (the
    /// row and the store are the same string, constructible only through
    /// [`Self::resolved`]).
    #[cfg(feature = "test-hooks")]
    pub fn new(root: impl Into<String>) -> Self {
        let root = root.into();
        let identity =
            RootIdentity::of(&root, None, &BuilderSeeds::from_env()).unwrap_or_else(|e| {
                panic!(
                    "the test-only root constructor was given {root:?}, which has no identity: {e}"
                )
            });
        Self { root, identity }
    }

    /// The verbatim root string.
    pub fn as_str(&self) -> &str {
        &self.root
    }

    /// The root's identity across spellings — the membership comparand.
    pub fn identity(&self) -> &RootIdentity {
        &self.identity
    }
}

impl std::fmt::Display for MemberRoot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.root)
    }
}

/// The identity of a result root ACROSS SPELLINGS — what two gang members
/// must share (necessary for shared storage, never sufficient: sufficiency
/// is the attestation VERIFY). Sealed: derived only inside
/// [`MemberRoot::resolved`] (and the `test-hooks` `MemberRoot::new`), by
/// the process that OWNS the root, at registration, on its own filesystem,
/// from the same config the store roots itself from. Never used to root
/// anything; compared for equality by
/// [`super::Catalog::list_gang_members`], which takes the caller's own
/// [`MemberRoot`] — the value its row carries — never a bare identity.
///
/// The rules, by the scheme the store's own URL parser
/// ([`crate::storage::StorageUrl::parse`]) assigns the string — the one
/// alias table in the tree, so this type can never fold a spelling the
/// store would not:
///
/// - **Object stores** (`s3://`, `gs://`|`gcs://`, `azure://`|`abfss://`,
///   `r2://`): `{canonical scheme}://{bucket}/{key}` where the bucket is
///   taken VERBATIM (the store hands it to the driver as spelled and the
///   driver dials it as spelled — the service, not this identity, decides
///   whether a mixed-case name exists), and
///   `key` is the bucket-stripped path normalised by the SAME
///   `object_store::path::Path::parse` the store hands its keys to (one
///   leading delimiter stripped, a trailing one dropped, an EMPTY segment
///   refused exactly as the store refuses it), so `s3://b//p` and
///   `s3://b/p` are one root and `s3://b/p//` is no root at all; then
///   `@{key=value;…}` — the LOCATION DETERMINANTS read back from the very
///   builder the store constructs for that root
///   ([`crate::storage::location_determinants`]: the process environment
///   via `from_env()`, `[storage.cloud]` on top, the order `build_*`
///   applies): for S3/R2 the bucket endpoint `build()` dials, computed by
///   the driver's own expression (endpoint spelling, virtual-hosted style,
///   S3 Express, region — so two regions for one bucket are two identities,
///   a split, never a merge), the Azure account, endpoint (or the Azurite
///   host and account in emulator mode), the Fabric switch, the GCS base
///   URL. The identity spells exactly the
///   variables the driver spells — none through its key tables, and the one
///   bare read the driver makes (`AZURITE_BLOB_STORAGE_URL`) the same way
///   — and reads every value as the driver reads it (its boolean parser's
///   `1`/`true`/`on`/`yes`/`y` in any case; its URL parse), so a value the
///   driver honours is never one the identity misses. Two
///   buckets of one name
///   behind two endpoints or
///   accounts are two locations. Neither the bucket nor the key is case-folded — the driver dials both as
///   spelled.
///   `r2://` and `s3://` stay distinct — different endpoints even when the
///   API is shared.
/// - **Local roots** (`file://` or a bare path, which the parser spells as
///   `file://`): the directory is CREATED first (`create_dir_all`, exactly
///   what the store does at open — idempotent, and the owner is about to
///   open the store there anyway), then canonicalised by the filesystem
///   (symlinks followed, `.`/`..` resolved, the filesystem's own spelling
///   of every component — on a case-insensitive filesystem two spellings
///   of one directory settle to the on-disk one). `file://{that path}`. A
///   relative root is taken against the process's working directory, the
///   same directory the store's own relative root is relative to.
/// - **`memory://`**: refused. An in-memory store lives in this process
///   alone; a member advertising one could never share results with a peer,
///   so the honest answer at registration is a typed refusal, not a row.
///
/// # Errors
///
/// [`JammiError::Config`] naming the root: the parser rejects it (an
/// unknown scheme — the store would reject it too), it is `memory://`, its
/// key has an empty segment, or a local root cannot be created or
/// canonicalised (any filesystem error — permission, a file where a
/// directory is needed, a symlink loop — is a refusal, never a silent
/// fallback to a different identity).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RootIdentity(String);

impl RootIdentity {
    /// Derive the identity of `root` under `cloud` (see the type's doc for
    /// the rules). Crate-private: the only production caller is
    /// [`MemberRoot::resolved`] (plus the `test-hooks` constructor and this
    /// crate's own tests), so an identity never exists apart from the
    /// verbatim root it was derived from.
    pub(crate) fn of(
        root: &str,
        cloud: Option<&crate::storage::CloudConfig>,
        seeds: &BuilderSeeds,
    ) -> Result<Self> {
        let url = StorageUrl::parse(root).map_err(|e| {
            JammiError::Config(format!(
                "result root '{root}' has no identity — the store's own URL parser rejects it: {e}"
            ))
        })?;
        match url.scheme() {
            Scheme::Memory => Err(JammiError::Config(format!(
                "result root '{root}' is an in-memory store: it lives in this process alone and \
                 can never be shared with a gang peer — unset `[server] peer_advertise`, or point \
                 `[storage] result_root` at a root every member can reach"
            ))),
            Scheme::File => {
                let canonical = canonical_local_root(root, url.path())?;
                Ok(Self(format!("file://{canonical}")))
            }
            scheme => {
                let rest = url.path();
                let (bucket, key) = match rest.split_once('/') {
                    Some((b, k)) => (b, k),
                    None => (rest, ""),
                };
                let key = object_store::path::Path::parse(key).map_err(|e| {
                    JammiError::Config(format!(
                        "result root '{root}' has no identity — the store's own key parser \
                         rejects it: {e}"
                    ))
                })?;
                let mut identity = if key.as_ref().is_empty() {
                    format!("{scheme}://{bucket}")
                } else {
                    format!("{scheme}://{bucket}/{}", key.as_ref())
                };
                let determinants = crate::storage::location_determinants_with(&url, cloud, seeds)
                    .map_err(|e| {
                    JammiError::Config(format!(
                        "result root '{root}' has no identity — the store's own builder \
                             rejects it: {e}"
                    ))
                })?;
                if !determinants.is_empty() {
                    identity.push('@');
                    identity.push_str(
                        &determinants
                            .iter()
                            .map(|(k, v)| format!("{k}={v}"))
                            .collect::<Vec<_>>()
                            .join(";"),
                    );
                }
                Ok(Self(identity))
            }
        }
    }

    /// The identity string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for RootIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The local-root rule of [`RootIdentity`]: create, then canonicalise.
/// `root` is the verbatim string, for the error messages.
fn canonical_local_root(root: &str, path: &str) -> Result<String> {
    use std::path::{Path, PathBuf};

    let raw = Path::new(path);
    let absolute: PathBuf = if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| {
                JammiError::Config(format!(
                    "result root '{root}' is relative and the working directory is unreadable: {e}"
                ))
            })?
            .join(raw)
    };
    std::fs::create_dir_all(&absolute).map_err(|e| {
        JammiError::Config(format!(
            "result root '{root}' cannot be created at '{}': {e}",
            absolute.display()
        ))
    })?;
    let resolved = std::fs::canonicalize(&absolute).map_err(|e| {
        JammiError::Config(format!(
            "result root '{root}' cannot be resolved at '{}': {e}",
            absolute.display()
        ))
    })?;
    resolved.to_str().map(str::to_string).ok_or_else(|| {
        JammiError::Config(format!(
            "result root '{root}' resolves to a path that is not valid UTF-8: {}",
            resolved.display()
        ))
    })
}

/// One device a process names as its own: a KIND (`"cpu"` | `"cuda"` |
/// `"metal"`, opaque here — never interpreted by this crate; a placement
/// policy reads it as an exact-match token) and its ORDINAL within that
/// kind (rank 0's device is ordinal 0, etc. — never a global, cross-kind
/// index). Serialized as JSON `{kind, ordinal}` — no other field (67
/// pressure-round delta 3 drops a `memory_bytes` this tree has no source
/// for). The ONE shape both device columns 67's wave-4 migration adds
/// carry a `Vec` of: `workers.devices`
/// ([`WorkerFacts::devices`], the `[worker]` process's own inventory,
/// carried for `ListWorkers` only) and `compute_executors.devices`
/// (`super::compute_repo::ComputeExecutorRecord::devices`, the compute
/// executor's OWN registration fact and the placement join's sole
/// authority) — so a device claim reads identically wherever it is
/// registered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceFact {
    /// The device kind.
    pub kind: String,
    /// The device's ordinal within `kind`.
    pub ordinal: u32,
}

/// Decode a `devices` JSON column (`workers.devices` /
/// `compute_executors.devices`) into its device list. Non-`NULL` text that
/// does not parse as `[{kind, ordinal}]` is a ROW FACT on both backends,
/// never a read FAULT — the shape issue #574 established for
/// `jobs.lease_expires_at` ([`super::lease::LeaseFact`]): the read that
/// found the row still succeeds, with an EMPTY device list AND a
/// `tracing::warn!` naming `row_label` (the executor/instance id) — never a
/// silent `unwrap_or_default()`, which would make the identical failure
/// invisible instead of an observable, attributed row fact.
pub fn decode_devices_json(raw: &str, row_label: &str) -> Vec<DeviceFact> {
    match serde_json::from_str::<Vec<DeviceFact>>(raw) {
        Ok(devices) => devices,
        Err(error) => {
            tracing::warn!(
                row = row_label,
                %error,
                "malformed `devices` JSON; treating this row as having no devices"
            );
            Vec::new()
        }
    }
}

/// The claim-loop half of a registration: the `kinds` this worker claims,
/// its lifecycle [`WorkerState`], and its device inventory — exactly the
/// triple `Catalog::upsert_worker` writes. Owned exclusively by `JobWorker`
/// (`crates/jammi-ai/src/fine_tune/worker.rs`): `run_until` sets it only
/// as ONE fact with every row write (`write_worker_facts` in jammi-ai's
/// `fine_tune::worker`, contract `feat_500-C-U5b-1a` §13): set to the facts
/// about to be UPSERTED, reverted if that upsert fails (`None` again after a
/// failed first write), cleared before `delete_worker` — so
/// [`InstanceRegistration::worker`] reflects only facts a row write of this
/// process succeeded with, or the facts an in-flight upsert is writing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerFacts {
    /// The `,`-joined kind set this worker claims — the ONLY encoding: no
    /// "otherwise producer-encoded" alternative exists.
    /// `Catalog::list_gang_members` splits this column on `,` and trims
    /// each token (`jobs_repo.rs` §"gang-membership listing verb", where
    /// the row's `kinds` is matched against `listing.kind`).
    pub kinds: String,
    /// The claim loop's lifecycle state at this instant.
    pub state: WorkerState,
    /// This `[worker]` process's own device inventory —
    /// `workers.devices`, a `ListWorkers` MIRROR only (read back verbatim on
    /// `jammi.v1.job.WorkerSummary.devices`, field 8 —
    /// `crates/jammi-wire/proto/jammi/v1/job.proto`; never the placement
    /// join's authority: `compute_executors.devices`,
    /// `super::compute_repo::ComputeExecutorRecord::devices`, is that,
    /// since a compute-executor process and a `[worker]` process may be
    /// different processes with different device visibility).
    pub devices: Vec<DeviceFact>,
}

/// The ONE value every writer of the `instances` (+ `workers`) row builds —
/// [`Catalog::upsert_instance`](super::Catalog::upsert_instance) and
/// [`Catalog::reregister_instance`](super::Catalog::reregister_instance)
/// accept no other shape. `peer_addr` / `member_root` are `None` for
/// every process that never joins a gang (a library/CLI process, or a
/// server with no `[server] peer_advertise`) — the row's `peer_addr` /
/// `result_root` columns are written `NULL` in that case.
///
/// `worker` is the claim-loop half (see [`WorkerFacts`]'s doc for its single
/// owner); it starts `None` (unpopulated) and is set by `JobWorker` once its
/// loop actually starts, independently of when the `instances` row itself
/// was written.
#[derive(Debug)]
pub struct InstanceRegistration {
    pub instance_id: String,
    pub label: Option<String>,
    pub host: Option<String>,
    pub peer_addr: Option<PeerAddr>,
    pub member_root: Option<MemberRoot>,
    /// The claim-loop half, mutated in place by its single owner (`JobWorker`)
    /// rather than requiring a whole new registration per state change — see
    /// [`WorkerFacts`]'s doc.
    pub worker: Mutex<Option<WorkerFacts>>,
}

impl InstanceRegistration {
    /// The plain constructor: no config validation, no membership check —
    /// [`InstanceRegistration::from_config`] is the ONE choke point
    /// production call sites use; this constructor is for that function's
    /// own use, and for fixtures/tests that want a registration with an
    /// already-built `PeerAddr`/`MemberRoot` pair (or none).
    pub fn new(
        instance_id: impl Into<String>,
        label: Option<&str>,
        host: Option<&str>,
        peer_addr: Option<PeerAddr>,
        member_root: Option<MemberRoot>,
    ) -> Self {
        Self {
            instance_id: instance_id.into(),
            label: label.map(str::to_string),
            host: host.map(str::to_string),
            peer_addr,
            member_root,
            worker: Mutex::new(None),
        }
    }

    /// Replace the worker half. `None` clears it (`delete_worker`'s call
    /// site: the claim loop has stopped).
    pub fn set_worker(&self, worker: Option<WorkerFacts>) {
        *self.worker.lock().unwrap_or_else(|p| p.into_inner()) = worker;
    }

    /// A snapshot of the worker half at this instant.
    pub fn worker_snapshot(&self) -> Option<WorkerFacts> {
        self.worker
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// The WHOLE registration-eligibility check, in ONE place:
    /// [`MembershipConfig::validate`] (`[server] peer_advertise` parses as a
    /// [`PeerAddr`], `peer_bind` is set) PLUS the VERBATIM root
    /// ([`crate::config::JammiConfig::resolved_result_root`] — no
    /// filesystem access, no interpretation) carried onto the row for
    /// U5b-1a-A2 — or, with `peer_advertise` unset, a non-member
    /// registration with NULL `peer_addr`/`member_root`. `member_root` is
    /// never consulted by [`super::Catalog::list_gang_members`]'s admission
    /// predicate in this unit (see [`GangListing`]'s doc); `peer_addr`
    /// still is (presence + the returned address). The ONLY constructor
    /// [`super::Catalog::upsert_instance`]/[`super::Catalog::
    /// reregister_instance`] accept in production:
    /// `InferenceSession::wrap_with` calls it once per session (every
    /// `InferenceSession` constructor funnels through `wrap_with`), and
    /// [`crate::config::JammiConfig::load_from`] calls
    /// [`MembershipConfig::validate`] directly instead (never this
    /// function), as an early-failure check — so a struct-literal config
    /// built without `load_from` is still covered, at `wrap_with`.
    /// `[server] ServerConfig::validate` is NOT the home either way: it
    /// cannot see `artifact_dir`, which `resolved_result_root` needs.
    pub fn from_config(
        config: &crate::config::JammiConfig,
        instance_id: impl Into<String>,
        label: Option<&str>,
        host: Option<&str>,
    ) -> Result<Self> {
        let Some(membership) = MembershipConfig::validate(config)? else {
            return Ok(Self::new(instance_id, label, host, None, None));
        };
        let member_root = MemberRoot::resolved(config)?;
        Ok(Self::new(
            instance_id,
            label,
            host,
            Some(membership.peer_addr),
            Some(member_root),
        ))
    }
}

/// The WHOLE pure membership check (contract `feat_500-C-U5b-1a` §10, the
/// round-3 excision): `[server] peer_advertise` parses as a [`PeerAddr`]
/// ∧ `peer_bind` is set — no filesystem access, no root interpretation at
/// all (the root is carried verbatim, see [`MemberRoot`]'s doc).
/// [`crate::config::JammiConfig::load_from`] calls [`Self::validate`]
/// directly; `Ok(None)` when `[server] peer_advertise` is unset — a library
/// process never runs any of this.
#[derive(Debug, Clone)]
pub struct MembershipConfig {
    peer_addr: PeerAddr,
}

impl MembershipConfig {
    /// Validate `config`'s membership shape: `peer_advertise` parses as a
    /// [`PeerAddr`]; `peer_bind` is set (a typed error naming BOTH keys
    /// otherwise). Nothing about the result root is checked here — the row
    /// carries [`crate::config::JammiConfig::resolved_result_root`]
    /// verbatim, and that function already refuses a non-UTF-8
    /// `artifact_dir` on its own.
    pub fn validate(config: &crate::config::JammiConfig) -> Result<Option<Self>> {
        let Some(advertise) = &config.server.peer_advertise else {
            // RENDEZVOUS RV4: `[server] placement = "rendezvous"` needs a
            // member row to have any ring to read — refused by NAME, at this
            // ONE choke point, so neither `load_from` nor a struct-literal
            // config reaching `from_config` directly can skip it.
            if config.server.placement == crate::config::PlacementMode::Rendezvous {
                return Err(JammiError::Config(
                    "server.placement = \"rendezvous\" requires server.peer_advertise \
                     (and server.peer_bind) to be set too"
                        .into(),
                ));
            }
            return Ok(None);
        };
        if config.server.peer_bind.is_none() {
            return Err(JammiError::Config(
                "server.peer_advertise requires server.peer_bind to be set too".into(),
            ));
        }
        let peer_addr = PeerAddr::parse(advertise)?;
        Ok(Some(Self { peer_addr }))
    }
}

/// The ONE shared SQL fragment for "a row in `instances` (aliased `alias`)
/// that is LIVE and shares MY result root": `{alias}.peer_addr IS NOT NULL
/// AND {alias}.result_root_identity = ({root_identity_expr}) AND NOT
/// ({stale})`, `{stale}` being [`stale_before_clause`] over
/// `{alias}.last_seen_at` at [`super::lease::instance_liveness_margin`]'s
/// margin. Two callers evaluate exactly this predicate —
/// [`super::Catalog::list_gang_members`] (which joins `workers` and filters
/// `kinds`/`state` in Rust on top; its caller already resolved its own
/// [`MemberRoot`] in Rust, so it passes a BOUND parameter placeholder it
/// pushed itself, e.g. `"$1"`) and [`super::Catalog::list_ring_members`] (the
/// RENDEZVOUS ring, which adds no join, includes the caller's OWN row, and
/// passes a `(SELECT result_root_identity FROM instances WHERE instance_id =
/// …)` self-referencing subquery — never a second, separately-cached copy of
/// its own root identity) — so "live with my root" has one definition, never
/// two independently-drifting `WHERE` clauses. `root_identity_expr` is
/// SPLICED VERBATIM into the returned SQL text (never bound): it is the
/// caller's job to make it a value expression, either a placeholder the
/// caller already bound or a subquery, both of which are `${n}`. Appends only
/// the margin bind, from [`stale_before_clause`], to `params`.
///
/// A row this predicate excludes for staleness, for a missing `peer_addr`,
/// or for a foreign root is excluded exactly the same way — one `WHERE`,
/// no per-reason tag. Neither caller counts WHY a row was excluded (only
/// `RendezvousPlacement` counts a behaviour-changing OUTCOME — the whole
/// ring reading empty — never a per-row exclusion reason); doing so would
/// cost a second statement per caller for a distinction only human
/// debugging, never correctness, would use.
pub(crate) fn live_with_root_clause(
    alias: &str,
    kind: BackendKind,
    margin: Duration,
    root_identity_expr: &str,
    params: &mut Vec<SqlValue<'static>>,
) -> String {
    let stale = stale_before_clause(&format!("{alias}.last_seen_at"), kind, margin, params);
    format!(
        "{alias}.peer_addr IS NOT NULL AND {alias}.result_root_identity = ({root_identity_expr}) \
         AND NOT ({stale})"
    )
}

/// [`super::Catalog::list_gang_members`]'s request: the predicate admits on
/// `kind`, `workers.state == claiming`, `peer_addr` presence, freshness,
/// self-exclusion, AND root identity — a candidate's
/// `instances.result_root_identity` must EQUAL the identity of `root`, the
/// caller's own [`MemberRoot`] (the value ITS row carries — a listing never
/// compares a bare claim; a row whose identity is NULL, written before the
/// column existed or by a process with no membership, never matches).
/// Necessary for shared storage, never sufficient: sufficiency is the
/// attestation VERIFY (U5a-1's admission-time sidecar, U5b-0's leaf
/// inventory).
#[derive(Debug, Clone, Copy)]
pub struct GangListing<'a> {
    /// The `workers.kinds` token this listing matches (a whole,
    /// comma-split, trimmed token — never a substring).
    pub kind: &'a str,
    /// This caller's own `instance_id` — excluded from the result.
    pub self_instance: &'a str,
    /// This caller's own root — the [`MemberRoot`] its registration
    /// carries; only members whose row carries the SAME identity are
    /// returned.
    pub root: &'a MemberRoot,
    /// The deployment's lease window — the same `lease` every other leased
    /// row family renews under. The liveness margin
    /// ([`super::lease::instance_liveness_margin`], `2 * lease`) is applied
    /// INSIDE the verb, exactly [`super::Catalog::fresh_instance`]'s shape.
    pub lease: Duration,
}

/// One returned gang member — no world-size-many round trips: a coordinator
/// with `w` peers gets them all in one call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GangMember {
    pub instance_id: String,
    pub peer_addr: PeerAddr,
}

/// One member of the RENDEZVOUS ring
/// ([`super::Catalog::list_ring_members`]): live, sharing the caller's own
/// result root, `peer_addr` set. Unlike [`GangMember`] the caller's OWN row
/// is a valid member (self-inclusion is the ring's whole point — a segment
/// this process itself wins placement for needs no remote call at all).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingMember {
    pub instance_id: String,
    pub peer_addr: PeerAddr,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_addr_parses_host_port() {
        let a = PeerAddr::parse("10.0.0.1:9000").unwrap();
        assert_eq!(a.as_str(), "10.0.0.1:9000");
        assert_eq!(a.to_string(), "10.0.0.1:9000");
    }

    #[test]
    fn peer_addr_parse_keeps_ipv6_host_intact() {
        // `rsplit_once(':')` — only the LAST colon splits host/port, so a
        // BRACKETED IPv6 literal's own colons stay inside the host segment.
        let a = PeerAddr::parse("[::1]:9000").unwrap();
        assert_eq!(a.as_str(), "[::1]:9000");
    }

    #[test]
    fn peer_addr_parses_a_full_bracketed_ipv6_host() {
        let a = PeerAddr::parse("[2001:db8::1]:9000").unwrap();
        assert_eq!(a.as_str(), "[2001:db8::1]:9000");
    }

    #[test]
    fn peer_addr_parses_a_dns_hostname() {
        let a = PeerAddr::parse("coordinator.internal:9000").unwrap();
        assert_eq!(a.as_str(), "coordinator.internal:9000");
    }

    /// P-Y4 (contract `feat_500-C-U5b-1a` §12): an UNBRACKETED IPv6 literal
    /// is refused, never silently split on its own last colon (which would
    /// treat `9000` as the port and `2001:db8::1` as the host, indistinguishable
    /// from a shorter address whose author simply forgot the brackets).
    #[test]
    fn peer_addr_refuses_an_unbracketed_ipv6_literal() {
        let err = PeerAddr::parse("2001:db8::1:9000").unwrap_err();
        assert!(
            err.to_string().contains("unbracketed"),
            "error must name the unbracketed-IPv6 refusal: {err}"
        );
    }

    #[test]
    fn peer_addr_refuses_an_unbracketed_loopback_ipv6_literal() {
        assert!(PeerAddr::parse("::1:9000").is_err());
    }

    #[test]
    fn peer_addr_refuses_no_colon() {
        let err = PeerAddr::parse("no-port-here").unwrap_err();
        assert!(matches!(err, JammiError::Config(_)));
    }

    #[test]
    fn peer_addr_refuses_empty_host() {
        assert!(PeerAddr::parse(":9000").is_err());
    }

    #[test]
    fn peer_addr_refuses_zero_port() {
        let err = PeerAddr::parse("10.0.0.1:0").unwrap_err();
        assert!(err.to_string().contains("non-zero"), "{err}");
    }

    #[test]
    fn peer_addr_refuses_non_numeric_port() {
        assert!(PeerAddr::parse("10.0.0.1:abc").is_err());
    }

    #[test]
    fn peer_addr_refuses_out_of_range_port() {
        assert!(PeerAddr::parse("10.0.0.1:99999").is_err());
    }

    #[test]
    fn registration_worker_half_starts_unpopulated_and_is_settable() {
        let reg = InstanceRegistration::new("i1", None, None, None, None);
        assert!(reg.worker_snapshot().is_none());
        reg.set_worker(Some(WorkerFacts {
            kinds: "fine_tune".into(),
            state: WorkerState::Claiming,
            devices: vec![],
        }));
        let snap = reg.worker_snapshot().unwrap();
        assert_eq!(snap.kinds, "fine_tune");
        assert_eq!(snap.state, WorkerState::Claiming);
        reg.set_worker(None);
        assert!(reg.worker_snapshot().is_none());
    }
}

#[cfg(test)]
mod root_identity_tests {
    //! The identity's equivalence classes are exactly the spellings the store
    //! would root at one location: scheme aliases, bucket case, and the
    //! store's own key normalisation on object stores (plus the endpoint the
    //! store would dial); creation-then-canonicalisation on local roots.
    //! Everything else stays distinct, and every root the store refuses is
    //! refused here.

    use super::RootIdentity;
    use crate::storage::{BuilderSeeds, CloudConfig, R2Config};

    /// An EMPTY environment: the derivation never reads the test process's
    /// real variables (`BuilderSeeds::from_vars` over nothing).
    fn no_env() -> BuilderSeeds {
        BuilderSeeds::from_vars(std::iter::empty::<(&str, &str)>())
    }

    fn id(root: &str) -> String {
        RootIdentity::of(root, None, &no_env())
            .unwrap()
            .as_str()
            .to_string()
    }

    #[test]
    fn object_store_aliases_and_the_stores_key_normalisation_fold() {
        assert_eq!(id("gcs://bucket/prefix"), id("gs://bucket/prefix"));
        assert_eq!(
            id("abfss://container/prefix"),
            id("azure://container/prefix")
        );
        // The store strips ONE leading delimiter from the key and drops a
        // trailing one (`object_store::path::Path::parse`): the same fold.
        assert_eq!(id("s3://bucket//prefix"), id("s3://bucket/prefix"));
        assert_eq!(id("s3://bucket/prefix/"), id("s3://bucket/prefix"));
        assert_eq!(id("s3://bucket/"), id("s3://bucket"));
        assert_eq!(id("gcs://b/p"), "gs://b/p");
        // With the driver compiled in the dialled bucket endpoint follows `@`;
        // without it the store cannot dial the scheme and there is none.
        assert!(
            id("s3://bucket").starts_with("s3://bucket"),
            "{}",
            id("s3://bucket")
        );
    }

    /// The store's key parser is the oracle: a spelling is a root iff the
    /// store can root at it, and two spellings are one root iff the store
    /// parses them to one key.
    #[test]
    fn a_key_the_store_refuses_is_refused_and_keys_the_store_equates_are_equated() {
        for spelling in [
            "s3://bucket/prefix//",
            "s3://bucket/a//b",
            "s3://bucket/a/./b",
        ] {
            let key = spelling.trim_start_matches("s3://bucket/");
            let store = object_store::path::Path::parse(key);
            match (store, RootIdentity::of(spelling, None, &no_env())) {
                (Ok(k), Ok(identity)) => assert_eq!(
                    identity.as_str(),
                    format!("s3://bucket/{}", k.as_ref()),
                    "{spelling}"
                ),
                (Err(_), Err(e)) => assert!(e.to_string().contains(spelling), "{e}"),
                (store, identity) => {
                    panic!("{spelling}: store={store:?} identity={identity:?} must agree")
                }
            }
        }
    }

    #[test]
    fn object_store_key_case_buckets_and_backends_stay_distinct() {
        assert_ne!(id("s3://bucket/Prefix"), id("s3://bucket/prefix"));
        // The bucket is dialled as spelled by the store and the driver alike.
        assert_ne!(id("s3://BUCKET/prefix"), id("s3://bucket/prefix"));
        assert_ne!(id("s3://a/prefix"), id("s3://b/prefix"));
        // R2 is the S3 driver at an account endpoint: the store refuses an
        // r2:// root with no R2 config (and so does the identity, in a build
        // with the driver); with one, it is a different location from the
        // same-named S3 bucket.
        let r2 = CloudConfig::R2(R2Config {
            account_id: Some("acct".to_string()),
            ..R2Config::default()
        });
        assert_ne!(
            RootIdentity::of("r2://bucket/prefix", Some(&r2), &no_env()).unwrap(),
            RootIdentity::of("s3://bucket/prefix", None, &no_env()).unwrap()
        );
        assert_ne!(id("gs://bucket/prefix"), id("s3://bucket/prefix"));
        assert_ne!(id("s3://bucket/prefix"), id("s3://bucket/prefix/deeper"));
    }

    /// Two buckets of one name behind two endpoints (or two R2 accounts, or
    /// two Azure accounts) are two locations; one endpoint is one location;
    /// `gs://` has one global namespace and no endpoint in its identity.
    /// Needs the drivers compiled in: without them the store cannot dial the
    /// scheme and there are no determinants to differ on.
    #[cfg(all(feature = "storage-s3", feature = "storage-r2"))]
    #[test]
    fn the_endpoint_the_store_would_dial_is_part_of_a_cloud_identity() {
        use crate::storage::S3Config;
        let s3 = |endpoint: Option<&str>| {
            CloudConfig::S3(S3Config {
                endpoint: endpoint.map(str::to_string),
                ..S3Config::default()
            })
        };
        let a = s3(Some("https://minio-a.local:9000"));
        let b = s3(Some("https://minio-b.local:9000"));
        let with = |cloud: &CloudConfig| {
            RootIdentity::of("s3://bucket/prefix", Some(cloud), &no_env())
                .unwrap()
                .as_str()
                .to_string()
        };
        assert_ne!(with(&a), with(&b));
        assert_eq!(with(&a), with(&s3(Some("https://minio-a.local:9000"))));
        assert_ne!(
            with(&a),
            id("s3://bucket/prefix"),
            "an endpoint vs the service default"
        );
        assert_eq!(with(&s3(None)), id("s3://bucket/prefix"));
        let r2 = |account: &str| {
            CloudConfig::R2(R2Config {
                account_id: Some(account.to_string()),
                ..R2Config::default()
            })
        };
        assert_ne!(
            RootIdentity::of("r2://bucket/prefix", Some(&r2("acct-a")), &no_env()).unwrap(),
            RootIdentity::of("r2://bucket/prefix", Some(&r2("acct-b")), &no_env()).unwrap()
        );
        assert_eq!(
            RootIdentity::of("gs://bucket/prefix", Some(&a), &no_env())
                .unwrap()
                .as_str(),
            id("gs://bucket/prefix")
        );
    }

    #[test]
    fn a_memory_root_and_an_unknown_scheme_are_refused_naming_the_root() {
        let err = RootIdentity::of("memory://x", None, &no_env())
            .unwrap_err()
            .to_string();
        assert!(err.contains("memory://x") && err.contains("peer"), "{err}");
        let err = RootIdentity::of("bogus://x", None, &no_env())
            .unwrap_err()
            .to_string();
        assert!(err.contains("bogus://x"), "{err}");
    }

    #[test]
    fn a_local_root_folds_symlinks_dot_segments_trailing_slashes_and_the_file_scheme() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real");
        std::fs::create_dir_all(real.join("jammi_db")).unwrap();
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let real_s = real.to_str().unwrap();
        let link_s = link.to_str().unwrap();
        let base = id(&format!("{real_s}/jammi_db"));
        assert_eq!(id(&format!("{link_s}/jammi_db")), base, "a symlinked root");
        assert_eq!(
            id(&format!("file://{link_s}/jammi_db")),
            base,
            "file:// spelling"
        );
        assert_eq!(id(&format!("{real_s}/jammi_db/")), base, "trailing slash");
        assert_eq!(id(&format!("{real_s}/./jammi_db")), base, "a dot segment");
        assert_eq!(
            id(&format!("{real_s}/other/../jammi_db")),
            base,
            "a parent segment"
        );
        assert!(base.starts_with("file:///"), "{base}");
        assert!(!base.ends_with('/'), "{base}");
    }

    /// The owner CREATES its root (as the store does at open), so the
    /// identity is the canonical path of an existing directory and never
    /// changes afterwards; on a case-insensitive filesystem two spellings of
    /// one directory settle to the on-disk one, and on a case-sensitive one
    /// they are two directories.
    #[test]
    fn a_local_root_is_created_and_a_case_divergent_spelling_settles_to_the_on_disk_one() {
        let dir = tempfile::tempdir().unwrap();
        let upper = dir.path().join("Jammi_DB");
        let lower = dir.path().join("jammi_db");
        assert!(!upper.exists());
        let first = id(upper.to_str().unwrap());
        assert!(upper.is_dir(), "the root is created by the derivation");
        let again = id(upper.to_str().unwrap());
        assert_eq!(first, again, "stable once it exists");
        let second = id(lower.to_str().unwrap());
        let case_insensitive =
            std::fs::canonicalize(&lower).unwrap() == std::fs::canonicalize(&upper).unwrap();
        if case_insensitive {
            assert_eq!(
                first, second,
                "one directory, one identity, the on-disk spelling"
            );
        } else {
            assert_ne!(
                first, second,
                "two directories on a case-sensitive filesystem"
            );
        }
    }

    /// Any filesystem error other than "does not exist yet" is a refusal
    /// naming the root, never a silent fallback to a different identity: a
    /// regular FILE where a directory is needed (ENOTDIR) here.
    #[test]
    fn a_local_root_that_cannot_be_created_is_refused_naming_it() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a-file");
        std::fs::write(&file, b"x").unwrap();
        let root = file.join("jammi_db");
        let root_s = root.to_str().unwrap();
        let err = RootIdentity::of(root_s, None, &no_env())
            .unwrap_err()
            .to_string();
        assert!(
            err.contains(root_s) && err.contains("cannot be created"),
            "{err}"
        );
    }

    #[test]
    fn a_relative_local_root_is_taken_against_the_working_directory() {
        // Relative to the process cwd, inside a scratch dir this test owns.
        let cwd = std::env::current_dir().unwrap();
        let scratch = tempfile::tempdir_in(&cwd).unwrap();
        let rel = scratch.path().strip_prefix(&cwd).unwrap().join("jammi_db");
        let rel_s = rel.to_str().unwrap();
        let expected = id(cwd.join(&rel).to_str().unwrap());
        assert_eq!(id(rel_s), expected);
        assert_eq!(id(&format!("./{rel_s}")), expected);
    }

    /// The determinants come from the builder the store constructs, so every
    /// spelling object_store accepts in the environment is part of the
    /// identity — `AWS_ENDPOINT_URL` (this repo's documented variable),
    /// `AWS_ENDPOINT`, `AWS_ENDPOINT_URL_S3`, the Azure account/endpoint,
    /// the GCS base URL — config on top, exactly as the builder applies it.
    /// The environment is an explicit variable set here, never the process's.
    #[cfg(all(
        feature = "storage-s3",
        feature = "storage-azure",
        feature = "storage-gcs"
    ))]
    #[test]
    fn every_endpoint_spelling_the_store_honours_is_part_of_the_identity() {
        use crate::storage::S3Config;
        let seeds = |vars: &[(&str, &str)]| BuilderSeeds::from_vars(vars.iter().copied());
        let s3 = |vars: &[(&str, &str)]| {
            RootIdentity::of("s3://bucket/prefix", None, &seeds(vars))
                .unwrap()
                .as_str()
                .to_string()
        };
        let default = s3(&[]);
        for spelling in ["AWS_ENDPOINT_URL", "AWS_ENDPOINT", "AWS_ENDPOINT_URL_S3"] {
            let with = s3(&[(spelling, "https://minio-a.local:9000")]);
            assert_ne!(
                with, default,
                "{spelling}: an endpoint vs the service default"
            );
            assert!(
                with.ends_with("@bucket_endpoint=https://minio-a.local:9000/bucket"),
                "{spelling}: {with}"
            );
        }
        assert_ne!(
            s3(&[("AWS_ENDPOINT_URL", "https://minio-a.local:9000")]),
            s3(&[("AWS_ENDPOINT_URL", "https://minio-b.local:9000")]),
            "two endpoints: two locations"
        );
        // Config on top of the environment, as the builder applies it.
        let configured = CloudConfig::S3(S3Config {
            endpoint: Some("https://minio-cfg.local:9000".to_string()),
            ..S3Config::default()
        });
        assert!(RootIdentity::of(
            "s3://bucket/prefix",
            Some(&configured),
            &seeds(&[("AWS_ENDPOINT_URL", "https://minio-env")])
        )
        .unwrap()
        .as_str()
        .ends_with("@bucket_endpoint=https://minio-cfg.local:9000/bucket"));
        let azure = |vars: &[(&str, &str)]| {
            RootIdentity::of("azure://container/prefix", None, &seeds(vars))
                .unwrap()
                .as_str()
                .to_string()
        };
        assert_ne!(
            azure(&[("AZURE_STORAGE_ACCOUNT_NAME", "acct-a")]),
            azure(&[("AZURE_STORAGE_ACCOUNT_NAME", "acct-b")])
        );
        assert_ne!(
            azure(&[("AZURE_STORAGE_ACCOUNT_NAME", "acct-a")]),
            azure(&[
                ("AZURE_STORAGE_ACCOUNT_NAME", "acct-a"),
                ("AZURE_STORAGE_ENDPOINT", "https://blob.local")
            ]),
            "an endpoint override is a different location"
        );
        let gs = |vars: &[(&str, &str)]| {
            RootIdentity::of("gs://bucket/prefix", None, &seeds(vars))
                .unwrap()
                .as_str()
                .to_string()
        };
        assert_ne!(
            gs(&[]),
            gs(&[("GOOGLE_BASE_URL", "http://fake-gcs:4443")]),
            "a repointed GCS"
        );
        // An empty value is unset, as the builder treats it.
        assert_eq!(s3(&[("AWS_ENDPOINT", "")]), default);
    }

    #[test]
    fn distinct_local_roots_stay_distinct() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a");
        let b = dir.path().join("b");
        assert_ne!(id(a.to_str().unwrap()), id(b.to_str().unwrap()));
    }
}
