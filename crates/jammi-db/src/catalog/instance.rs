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
//! registration** ([`RootIdentity::of`]): the verbatim string is parsed by
//! the SAME [`crate::storage::StorageUrl`] parser the result store roots
//! itself through (so `gcs://` and `gs://`, `abfss://` and `azure://` fold
//! by the one alias table the store already owns), a local root is
//! resolved on the owner's own filesystem (symlinks, `.`/`..`, the
//! filesystem's own case), and an in-memory root is refused as
//! unshareable. Nothing on this path re-roots anything: the store still
//! roots at the verbatim string, the row still carries that string
//! verbatim, and the identity is a SEPARATE column used only for equality.

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
    /// [`RootIdentity::of`]'s refusals (an in-memory root, a root the store's
    /// URL parser rejects, a local root with no resolvable ancestor).
    pub fn resolved(config: &crate::config::JammiConfig) -> Result<Self> {
        let root = config.resolved_result_root()?;
        let identity = RootIdentity::of(&root)?;
        Ok(Self { root, identity })
    }

    /// Wrap an ALREADY-RESOLVED root string directly, with no resolver call
    /// — fixtures and tests only. Its identity is derived exactly as
    /// [`Self::resolved`] derives it; a root with no identity (an in-memory
    /// root, an unknown scheme) panics here, naming the reason, since a
    /// fixture asking for an unshareable member is a fixture bug. Gated
    /// behind `feature = "test-hooks"` so a production build never links
    /// this constructor: reachable from production code, it would let any
    /// caller put an arbitrary string in the `instances.result_root`
    /// column, defeating the one property this type exists to hold (the
    /// row and the store are the same string, constructible only through
    /// [`Self::resolved`]).
    #[cfg(feature = "test-hooks")]
    pub fn new(root: impl Into<String>) -> Self {
        let root = root.into();
        let identity = RootIdentity::of(&root).unwrap_or_else(|e| {
            panic!("the test-only root constructor was given {root:?}, which has no identity: {e}")
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
/// is the attestation VERIFY). Sealed: built only by [`Self::of`], a total
/// function of the verbatim root string, run by the process that OWNS the
/// root, at registration, on its own filesystem. Never used to root
/// anything; compared for equality by
/// [`super::Catalog::list_gang_members`].
///
/// The rules, by the scheme the store's own URL parser
/// ([`crate::storage::StorageUrl::parse`]) assigns the string — the one
/// alias table in the tree, so this type can never fold a spelling the
/// store would not:
///
/// - **Object stores** (`s3://`, `gs://`|`gcs://`, `azure://`|`abfss://`,
///   `r2://`): `{canonical scheme}://{authority, lowercased}/{key}` with
///   trailing `/`s trimmed from the key and the key's case PRESERVED
///   (bucket and container names are case-insensitive by their services'
///   rules; object keys are not). `r2://` and `s3://` stay distinct — they
///   are different endpoints even when the API is shared.
/// - **Local roots** (`file://` or a bare path, which the parser spells as
///   `file://`): `file://{path}` where `path` is the longest EXISTING prefix
///   canonicalised by the filesystem (symlinks followed, `..` resolved
///   against the real parent, the filesystem's own case) with the
///   not-yet-existing remainder appended lexically (`.` dropped, `..`
///   popped) — a root the store has not created yet has the identity it
///   will have once created. A relative root is taken against the process's
///   working directory, the same directory the store's own relative root is
///   relative to.
/// - **`memory://`**: refused. An in-memory store lives in this process
///   alone; a member advertising one could never share results with a peer,
///   so the honest answer at registration is a typed refusal, not a row.
///
/// # Errors
///
/// [`JammiError::Config`] naming the root: the parser rejects it (an
/// unknown scheme — the store would reject it too), it is `memory://`, or a
/// local root has no resolvable ancestor / is not valid UTF-8 once resolved.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RootIdentity(String);

impl RootIdentity {
    /// Derive the identity of `root` (see the type's doc for the rules).
    pub fn of(root: &str) -> Result<Self> {
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
                let (authority, key) = match rest.split_once('/') {
                    Some((a, k)) => (a, k.trim_end_matches('/')),
                    None => (rest, ""),
                };
                let authority = authority.to_ascii_lowercase();
                Ok(Self(if key.is_empty() {
                    format!("{scheme}://{authority}")
                } else {
                    format!("{scheme}://{authority}/{key}")
                }))
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

/// The local-root rule of [`RootIdentity::of`]: the longest existing prefix
/// of `path` canonicalised by the filesystem, the remainder appended
/// lexically. `root` is the verbatim string, for the error messages.
fn canonical_local_root(root: &str, path: &str) -> Result<String> {
    use std::path::{Component, Path, PathBuf};

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
    let components: Vec<Component<'_>> = absolute.components().collect();
    for n in (1..=components.len()).rev() {
        let prefix: PathBuf = components[..n].iter().collect();
        let Ok(mut resolved) = std::fs::canonicalize(&prefix) else {
            continue;
        };
        for component in &components[n..] {
            match component {
                Component::CurDir | Component::RootDir | Component::Prefix(_) => {}
                Component::ParentDir => {
                    resolved.pop();
                }
                Component::Normal(name) => resolved.push(name),
            }
        }
        return resolved.to_str().map(str::to_string).ok_or_else(|| {
            JammiError::Config(format!(
                "result root '{root}' resolves to a path that is not valid UTF-8: {}",
                resolved.display()
            ))
        });
    }
    Err(JammiError::Config(format!(
        "result root '{root}' has no resolvable ancestor on this filesystem"
    )))
}

/// The claim-loop half of a registration: the `kinds` this worker claims and
/// its lifecycle [`WorkerState`] — exactly the pair `Catalog::upsert_worker`
/// writes. Owned exclusively by `JobWorker`
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

/// [`super::Catalog::list_gang_members`]'s request: the predicate admits on
/// `kind`, `workers.state == claiming`, `peer_addr` presence, freshness,
/// self-exclusion, AND root identity — a candidate's
/// `instances.result_root_identity` must EQUAL `root_identity`, the
/// caller's own (a row whose identity is NULL, written before the column
/// existed or by a process with no membership, never matches). Necessary
/// for shared storage, never sufficient: sufficiency is the attestation
/// VERIFY (U5a-1's admission-time sidecar, U5b-0's leaf inventory).
#[derive(Debug, Clone, Copy)]
pub struct GangListing<'a> {
    /// The `workers.kinds` token this listing matches (a whole,
    /// comma-split, trimmed token — never a substring).
    pub kind: &'a str,
    /// This caller's own `instance_id` — excluded from the result.
    pub self_instance: &'a str,
    /// This caller's own root identity ([`MemberRoot::identity`]); only
    /// members whose row carries the SAME identity are returned.
    pub root_identity: &'a RootIdentity,
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
    //! [`RootIdentity::of`] is a total function of the verbatim root whose
    //! equivalence classes are exactly the spellings the store would root at
    //! the same location: scheme aliases, authority case and trailing
    //! slashes on object stores; symlinks, `.`/`..`, trailing slashes and
    //! not-yet-existing leaves on local roots. Everything else stays distinct.

    use super::RootIdentity;

    fn id(root: &str) -> String {
        RootIdentity::of(root).unwrap().as_str().to_string()
    }

    #[test]
    fn object_store_aliases_authority_case_and_trailing_slashes_fold() {
        assert_eq!(id("gcs://bucket/prefix"), id("gs://bucket/prefix"));
        assert_eq!(
            id("abfss://container/prefix"),
            id("azure://container/prefix")
        );
        assert_eq!(id("s3://BUCKET/prefix"), id("s3://bucket/prefix"));
        assert_eq!(id("s3://bucket/prefix/"), id("s3://bucket/prefix"));
        assert_eq!(id("s3://bucket/prefix//"), id("s3://bucket/prefix"));
        assert_eq!(id("s3://bucket/"), id("s3://bucket"));
        assert_eq!(id("gcs://b/p"), "gs://b/p");
        assert_eq!(id("s3://bucket"), "s3://bucket");
    }

    #[test]
    fn object_store_key_case_buckets_and_backends_stay_distinct() {
        assert_ne!(id("s3://bucket/Prefix"), id("s3://bucket/prefix"));
        assert_ne!(id("s3://a/prefix"), id("s3://b/prefix"));
        assert_ne!(id("r2://bucket/prefix"), id("s3://bucket/prefix"));
        assert_ne!(id("gs://bucket/prefix"), id("s3://bucket/prefix"));
        assert_ne!(id("s3://bucket/prefix"), id("s3://bucket/prefix/deeper"));
    }

    #[test]
    fn a_memory_root_and_an_unknown_scheme_are_refused_naming_the_root() {
        let err = RootIdentity::of("memory://x").unwrap_err().to_string();
        assert!(err.contains("memory://x") && err.contains("peer"), "{err}");
        let err = RootIdentity::of("bogus://x").unwrap_err().to_string();
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

    #[test]
    fn a_local_root_the_store_has_not_created_yet_has_the_identity_it_will_have() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("not").join("yet").join("jammi_db");
        let before = id(root.to_str().unwrap());
        std::fs::create_dir_all(&root).unwrap();
        let after = id(root.to_str().unwrap());
        assert_eq!(before, after);
        assert_ne!(before, id(dir.path().join("not").to_str().unwrap()));
    }

    #[test]
    fn a_relative_local_root_is_taken_against_the_working_directory() {
        let cwd = std::env::current_dir().unwrap();
        let expected = id(cwd.join("rel").join("jammi_db").to_str().unwrap());
        assert_eq!(id("rel/jammi_db"), expected);
        assert_eq!(id("./rel/jammi_db"), expected);
    }

    #[test]
    fn distinct_local_roots_stay_distinct() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a");
        let b = dir.path().join("b");
        assert_ne!(id(a.to_str().unwrap()), id(b.to_str().unwrap()));
    }
}
