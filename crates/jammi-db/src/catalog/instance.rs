//! The gang-membership carrier: [`PeerAddr`] (the wire address a coordinator
//! dials a gang member / segment owner at — moved here from `index::peer`,
//! re-exported there, so the peer listener and the gang listener can never
//! drift into two distinct address types), [`CanonicalRoot`] (the
//! canonicalized result-table root two gang members compare byte-for-byte —
//! see [`crate::config::JammiConfig::canonical_result_root`]), [`WorkerFacts`]
//! (the claim-loop half of a registration — owned exclusively by
//! `JobWorker`, `crates/jammi-ai/src/fine_tune/worker.rs`), and
//! [`InstanceRegistration`] — the ONE value every writer of the `instances`
//! (+ `workers`) row builds, through [`InstanceRegistration::from_config`].
//! [`GangListing`] / [`GangMember`] are [`super::Catalog::list_gang_members`]'s
//! request and response shapes.
//!
//! [`MembershipConfig::validate`] is the PURE half of the membership check
//! (no filesystem access at all — [`crate::config::JammiConfig::load_from`]
//! calls it); [`InstanceRegistration::from_config`] is validate PLUS
//! MATERIALIZE (creates the local anchor directory when it is absent,
//! idempotent with [`crate::store::ResultStore`]'s own later
//! `create_dir_all` of the same path) — the split exists so loading a config
//! file never itself creates a directory as a side effect.

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

use super::jobs_repo::WorkerState;
use crate::error::{JammiError, Result};
use crate::storage::{Scheme, StorageUrl};

/// The address a coordinator dials a gang member / segment owner at
/// (`host:port`, plaintext gRPC — transport encryption is the runtime's,
/// never the engine's). Sealed: the only way to build one is
/// [`PeerAddr::parse`], so a `PeerAddr` value is always a validated
/// `host:port` pair, never an arbitrary string smuggled past construction.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PeerAddr(String);

impl PeerAddr {
    /// Parse `host:port`: a non-empty host (a DNS name or an IP literal —
    /// this type never resolves it, only validates the wire-form shape) and
    /// a port in `1..=65535`. `rsplit_once(':')` so an IPv6 literal's own
    /// colons stay inside the host segment and only the LAST colon is
    /// treated as the host/port separator.
    pub fn parse(input: &str) -> Result<Self> {
        let (host, port) = input.rsplit_once(':').ok_or_else(|| {
            JammiError::Config(format!("peer address '{input}' must be 'host:port'"))
        })?;
        if host.is_empty() {
            return Err(JammiError::Config(format!(
                "peer address '{input}' has an empty host"
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

/// A canonicalized result-table root — the string two gang members compare
/// byte-for-byte (never a SQL `=`, never re-canonicalized at compare time).
/// Built exclusively by [`crate::config::JammiConfig::canonical_result_root`];
/// [`Self::new`] exists for that function (and fixtures/tests) rather than
/// being a general-purpose string wrapper any caller may construct from an
/// arbitrary, un-canonicalized value.
///
/// **Necessary, never sufficient, for shared storage**: two byte-identical
/// roots on two filesystems are indistinguishable to this type or to the
/// membership predicate that compares it — sufficiency is established only
/// by the attestation VERIFY (the whole-artifact / per-partition inventory a
/// later unit owns), never by this string alone.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CanonicalRoot(String);

impl CanonicalRoot {
    /// Wrap an already-canonicalized root string.
    pub fn new(canonical: impl Into<String>) -> Self {
        Self(canonical.into())
    }

    /// The canonicalized root string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for CanonicalRoot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The claim-loop half of a registration: the `kinds` this worker claims and
/// its lifecycle [`WorkerState`] — exactly the pair `Catalog::upsert_worker`
/// writes. Owned exclusively by `JobWorker`
/// (`crates/jammi-ai/src/fine_tune/worker.rs`):
/// `run_until` sets it (then issues the first `upsert_worker`), every
/// `set_worker_state` writes the cell BEFORE the row, `delete_worker` clears
/// it — so [`InstanceRegistration::worker`] always reflects what the row is
/// ABOUT to become, never what it already is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerFacts {
    /// The comma-joined (or otherwise producer-encoded) kind set this
    /// worker claims.
    pub kinds: String,
    /// The claim loop's lifecycle state at this instant.
    pub state: WorkerState,
}

/// The ONE value every writer of the `instances` (+ `workers`) row builds —
/// [`Catalog::upsert_instance`](super::Catalog::upsert_instance) and
/// [`Catalog::reregister_instance`](super::Catalog::reregister_instance)
/// accept no other shape. `peer_addr` / `canonical_root` are `None` for
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
    pub canonical_root: Option<CanonicalRoot>,
    /// The claim-loop half, mutated in place by its single owner (`JobWorker`)
    /// rather than requiring a whole new registration per state change — see
    /// [`WorkerFacts`]'s doc.
    pub worker: Mutex<Option<WorkerFacts>>,
}

impl InstanceRegistration {
    /// The plain constructor: no config validation, no membership check —
    /// [`InstanceRegistration::from_config`] (added alongside
    /// [`crate::config::JammiConfig::canonical_result_root`]) is the ONE
    /// choke point production call sites use; this constructor is for that
    /// function's own use, and for fixtures/tests that want a registration
    /// with an already-validated `PeerAddr`/`CanonicalRoot` pair (or none).
    pub fn new(
        instance_id: impl Into<String>,
        label: Option<&str>,
        host: Option<&str>,
        peer_addr: Option<PeerAddr>,
        canonical_root: Option<CanonicalRoot>,
    ) -> Self {
        Self {
            instance_id: instance_id.into(),
            label: label.map(str::to_string),
            host: host.map(str::to_string),
            peer_addr,
            canonical_root,
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

    /// The WHOLE membership check, in ONE place: [`MembershipConfig::validate`]
    /// (PURE — `[server] peer_advertise` parses as a [`PeerAddr`], `peer_bind`
    /// is set, the anchor is well-formed and absolute) PLUS MATERIALIZE (the
    /// local anchor directory is created if absent, then canonicalized) — or,
    /// with `peer_advertise` unset, a non-member registration with NULL
    /// `peer_addr`/`canonical_root`. The ONLY constructor
    /// [`super::Catalog::upsert_instance`]/[`super::Catalog::
    /// reregister_instance`] accept in production:
    /// `InferenceSession::wrap_with` calls it once per session (every
    /// `InferenceSession` constructor funnels through `wrap_with`), running
    /// BEFORE the result store (or anything else) creates a single
    /// directory — its own `create_dir_all` of the anchor is idempotent with
    /// the store's later one of the same path. [`crate::config::JammiConfig
    /// ::load_from`] calls [`MembershipConfig::validate`] directly instead
    /// (never this function): loading a config file must never itself create
    /// a directory as a side effect, purely for the early-failure check — so
    /// a struct-literal config built without `load_from` is still covered, at
    /// `wrap_with`. `[server] ServerConfig::validate` is NOT the home either
    /// way: it cannot see `artifact_dir`, which the anchor needs.
    pub fn from_config(
        config: &crate::config::JammiConfig,
        instance_id: impl Into<String>,
        label: Option<&str>,
        host: Option<&str>,
    ) -> Result<Self> {
        let Some(membership) = MembershipConfig::validate(config)? else {
            return Ok(Self::new(instance_id, label, host, None, None));
        };
        let peer_addr = membership.peer_addr.clone();
        let canonical_root = membership.materialize()?;
        Ok(Self::new(
            instance_id,
            label,
            host,
            Some(peer_addr),
            Some(canonical_root),
        ))
    }
}

/// [`MembershipConfig::validate`]'s PURE output, [`MembershipConfig::
/// materialize`]'s input — contract `feat_500-C-U5b-1a` §9 (round-2
/// redesign): `artifact_dir` is a LOCAL PATH, never reinterpreted as a URL;
/// only an explicit `[storage] result_root` is ever parsed as one.
#[derive(Debug, Clone)]
enum EffectiveRoot {
    /// `[storage] result_root` unset: the anchor IS `artifact_dir`, the
    /// LITERAL `PathBuf` — the SAME value `jammi_db::session`'s catalog
    /// open, `resolved_result_root`, the local cache dir, and `JobWorker`
    /// all use. NEVER parsed as a URL: a `file://`- or cloud-scheme-spelled
    /// `artifact_dir` is refused as RELATIVE by the literal
    /// `Path::is_absolute` check in [`MembershipConfig::validate`] (as a
    /// bare string it does not start with `/`), never silently accepted
    /// through a URL reparse that would place a false-positive member on
    /// the wrong root (the round-2 F-A1/F-A2 findings this redesign
    /// closes).
    Default(PathBuf),
    /// `result_root` explicitly set to a `file://` URL (or a bare path,
    /// which [`StorageUrl::parse`] treats identically): the WHOLE absolute
    /// local path it names — no leaf, since `result_root` already names
    /// the whole effective root ([`crate::store::ResultStore::with_root`]'s
    /// verbatim use).
    ResultRootFile(PathBuf),
    /// `result_root` explicitly set to a cloud scheme:
    /// [`StorageUrl`]'s own verbatim rendering — NO scheme lowercasing (an
    /// uppercase scheme token is refused by [`StorageUrl::parse`] itself,
    /// case-sensitively, exactly as `build_result_store` would refuse the
    /// SAME string — never silently folded), trailing `/` trimmed. No leaf.
    ResultRootCloud(String),
}

/// The PURE half of the membership check (contract `feat_500-C-U5b-1a` §8
/// B3, split by the F1 fix; §9 redesigns its anchor handling): NO
/// filesystem access of any kind. [`crate::config::JammiConfig::load_from`]
/// calls [`Self::validate`] directly — never [`InstanceRegistration::
/// from_config`], which additionally MATERIALIZES the anchor (a
/// directory-creating side effect a config LOAD must never perform merely
/// by being read). `Ok(None)` when `[server] peer_advertise` is unset — a
/// library process never runs any of this.
#[derive(Debug, Clone)]
pub struct MembershipConfig {
    peer_addr: PeerAddr,
    effective_root: EffectiveRoot,
}

impl MembershipConfig {
    /// Validate `config`'s membership shape with NO filesystem read/write:
    /// `peer_advertise` parses as a [`PeerAddr`]; `peer_bind` is set (a typed
    /// error naming BOTH keys otherwise).
    ///
    /// **`artifact_dir` is a LOCAL PATH, never a URL (§9).** When
    /// `[storage] result_root` is UNSET, the anchor is `artifact_dir`'s
    /// LITERAL `PathBuf`: `Path::is_absolute()` must hold (a relative
    /// `artifact_dir` — including the `.jammi` fallback
    /// `default_artifact_dir` returns when `ProjectDirs` is unavailable, and
    /// including a `file://…`- or cloud-scheme-spelled string, which as a
    /// bare path never starts with `/` — is refused naming `artifact_dir`)
    /// and `Path::to_str()` must be `Some` (a non-UTF-8 `artifact_dir` is
    /// refused naming the key too — never a lossy fold on the compared
    /// value).
    ///
    /// **Only an explicit `result_root` is ever parsed as a URL** — VERBATIM,
    /// exactly as `build_result_store` parses it: NO scheme lowercasing (an
    /// uppercase scheme is refused by [`StorageUrl::parse`] itself,
    /// case-sensitively, consistently with the store). `file://` (or a bare
    /// path): the WHOLE path must be absolute, refused naming
    /// `storage.result_root` otherwise. A cloud scheme: [`Scheme`]'s own
    /// rendering, trailing `/` trimmed, no further check. `memory://` is
    /// refused for a gang member.
    pub fn validate(config: &crate::config::JammiConfig) -> Result<Option<Self>> {
        let Some(advertise) = &config.server.peer_advertise else {
            return Ok(None);
        };
        if config.server.peer_bind.is_none() {
            return Err(JammiError::Config(
                "server.peer_advertise requires server.peer_bind to be set too".into(),
            ));
        }
        let peer_addr = PeerAddr::parse(advertise)?;

        let effective_root = match &config.storage.result_root {
            None => {
                // The literal PathBuf — never reinterpreted as a URL. See
                // `EffectiveRoot::Default`'s own doc for why.
                if !config.artifact_dir.is_absolute() {
                    return Err(JammiError::Config(format!(
                        "server.peer_advertise requires artifact_dir '{}' to be absolute when \
                         server.peer_advertise is set",
                        config.artifact_dir.display()
                    )));
                }
                if config.artifact_dir.to_str().is_none() {
                    return Err(JammiError::Config(format!(
                        "server.peer_advertise requires artifact_dir '{}' to be valid UTF-8",
                        config.artifact_dir.display()
                    )));
                }
                EffectiveRoot::Default(config.artifact_dir.clone())
            }
            Some(result_root) => {
                // Parsed VERBATIM — no scheme lowercasing (§9): an
                // uppercase scheme is refused by `StorageUrl::parse` itself,
                // case-sensitively, exactly like `build_result_store`'s own
                // parse of the SAME string.
                let url = StorageUrl::parse(result_root).map_err(|e| {
                    JammiError::Config(format!(
                        "server.peer_advertise requires a valid [storage.result_root]: {e}"
                    ))
                })?;
                match url.scheme() {
                    Scheme::Memory => {
                        return Err(JammiError::Config(
                            "server.peer_advertise cannot be combined with a memory:// result \
                             root"
                                .into(),
                        ));
                    }
                    Scheme::File => {
                        let anchor = PathBuf::from(url.path());
                        if !anchor.is_absolute() {
                            return Err(JammiError::Config(format!(
                                "server.peer_advertise requires [storage.result_root] '{}' to \
                                 be absolute when server.peer_advertise is set",
                                anchor.display()
                            )));
                        }
                        EffectiveRoot::ResultRootFile(anchor)
                    }
                    other_scheme => {
                        let path = url.path().trim_end_matches('/');
                        EffectiveRoot::ResultRootCloud(format!("{other_scheme}://{path}"))
                    }
                }
            }
        };
        Ok(Some(Self {
            peer_addr,
            effective_root,
        }))
    }

    /// MATERIALIZE: for a `file://` anchor (either arm), create it if
    /// absent (`std::fs::create_dir_all` — the deployment's own directory,
    /// which `ResultStore`/the catalog create moments later anyway, so this
    /// is idempotent with that later call over the SAME path), refusing
    /// naming the key if it exists as a non-directory or cannot be created;
    /// then `std::fs::canonicalize`s it (symlinks and `.`/`..` resolved). A
    /// cloud root needs no filesystem step at all. Consumes `self`: each
    /// validated config is materialized exactly once per call site.
    ///
    /// **Partial-tree honesty.** `create_dir_all` is not atomic: if it
    /// creates some missing parent components and then fails on a later
    /// one (a permission fault, a concurrent removal), those parent
    /// directories are left behind on disk even though this call returns
    /// `Err` — the same non-atomicity `std::fs::create_dir_all`'s own doc
    /// states, inherited here rather than papered over.
    pub(crate) fn materialize(self) -> Result<CanonicalRoot> {
        match self.effective_root {
            EffectiveRoot::ResultRootCloud(s) => Ok(CanonicalRoot::new(s)),
            EffectiveRoot::ResultRootFile(anchor) => {
                let canonical = materialize_file_anchor(&anchor, "storage.result_root")?;
                Ok(CanonicalRoot::new(format!("file://{canonical}")))
            }
            EffectiveRoot::Default(artifact_dir) => {
                // The leaf `jammi_db` is appended LEXICALLY, after
                // canonicalizing `artifact_dir` alone — never itself
                // resolved, whether absent, present, or a symlink; exactly
                // `{artifact_dir}/jammi_db`, `resolved_result_root`'s own
                // default-arm string, canonicalized.
                let canonical = materialize_file_anchor(&artifact_dir, "artifact_dir")?;
                Ok(CanonicalRoot::new(format!("file://{canonical}/jammi_db")))
            }
        }
    }
}

/// Create `anchor` if absent (checked for non-directory FIRST, so an
/// anchor that already exists as a plain FILE gets the precise "must be a
/// directory" refusal rather than `create_dir_all`'s own platform-dependent
/// `AlreadyExists`/`EEXIST` error text), then canonicalize it, returning
/// the canonicalized path as a UTF-8 `String` — every failure refused
/// naming `anchor_key`. Shared by both `file://` arms
/// ([`EffectiveRoot::ResultRootFile`] and [`EffectiveRoot::Default`]'s own
/// `artifact_dir`).
fn materialize_file_anchor(anchor: &std::path::Path, anchor_key: &str) -> Result<String> {
    if let Ok(meta) = std::fs::metadata(anchor) {
        if !meta.is_dir() {
            return Err(JammiError::Config(format!(
                "server.peer_advertise requires [{anchor_key}] '{}' to be a directory",
                anchor.display()
            )));
        }
    }
    std::fs::create_dir_all(anchor).map_err(|e| {
        JammiError::Config(format!(
            "server.peer_advertise: failed to create [{anchor_key}] '{}': {e}",
            anchor.display()
        ))
    })?;
    let canonical = std::fs::canonicalize(anchor).map_err(|e| {
        JammiError::Config(format!(
            "server.peer_advertise: failed to canonicalize [{anchor_key}] '{}': {e}",
            anchor.display()
        ))
    })?;
    canonical.into_os_string().into_string().map_err(|os| {
        JammiError::Config(format!(
            "server.peer_advertise: canonicalized [{anchor_key}] is not valid UTF-8: {}",
            std::path::Path::new(&os).display()
        ))
    })
}

/// [`super::Catalog::list_gang_members`]'s request shape.
#[derive(Debug, Clone, Copy)]
pub struct GangListing<'a> {
    /// The `workers.kinds` token this listing matches (a whole,
    /// comma-split, trimmed token — never a substring).
    pub kind: &'a str,
    /// This caller's own `instance_id` — excluded from the result.
    pub self_instance: &'a str,
    /// The caller's own canonicalized result root — a member's `result_root`
    /// must match this BYTE-FOR-BYTE (a Rust comparison, never a SQL `=`).
    pub canonical_root: &'a CanonicalRoot,
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
        // `rsplit_once(':')` — only the LAST colon splits host/port, so an
        // IPv6 literal's own colons stay inside the host segment.
        let a = PeerAddr::parse("[::1]:9000").unwrap();
        assert_eq!(a.as_str(), "[::1]:9000");
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
        }));
        let snap = reg.worker_snapshot().unwrap();
        assert_eq!(snap.kinds, "fine_tune");
        assert_eq!(snap.state, WorkerState::Claiming);
        reg.set_worker(None);
        assert!(reg.worker_snapshot().is_none());
    }
}
