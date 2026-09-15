//! The gang-membership carrier: [`PeerAddr`] (the wire address a coordinator
//! dials a gang member / segment owner at — moved here from `index::peer`,
//! re-exported there, so the peer listener and the gang listener can never
//! drift into two distinct address types), [`MemberRoot`] (the VERBATIM
//! configured result-table root — see
//! [`crate::config::JammiConfig::resolved_result_root`] — carried on the
//! `instances` row byte-for-byte, but **not consulted by the membership
//! predicate in this unit**: root identity across spellings, and any
//! membership predicate built on it, are
//! `docs/plans/67-distributed-training/README.md` unit U5b-1a-A2's
//! question, not this one's), [`WorkerFacts`] (the claim-loop half of a
//! registration — owned exclusively by `JobWorker`,
//! `crates/jammi-ai/src/fine_tune/worker.rs`), and
//! [`InstanceRegistration`] — the ONE value every writer of the `instances`
//! (+ `workers`) row builds, through [`InstanceRegistration::from_config`].
//! [`GangListing`] / [`GangMember`] are [`super::Catalog::list_gang_members`]'s
//! request and response shapes; [`GangListing`] carries no root field — the
//! predicate admits on `kinds` + `workers.state` + `peer_addr` presence +
//! freshness + self-exclusion ONLY (contract `feat_500-C-U5b-1a` §12, P-Y1,
//! the round-5 excision).
//!
//! **The membership path performs no interpretation of the root at all,
//! and no longer even reads it**: no filesystem access, no URL parse, no
//! scheme handling, no symlink resolution, no byte comparison. The only
//! refusal on this path is the non-UTF-8 refusal already inside
//! [`crate::config::JammiConfig::resolved_result_root`]. A spelling-identity
//! unit (folding `gcs://`/`gs://`, resolving symlinks, and any membership
//! predicate built on the result), is filed separately —
//! `docs/plans/67-distributed-training/README.md`, unit U5b-1a-A2.

use std::sync::Mutex;
use std::time::Duration;

use super::jobs_repo::WorkerState;
use crate::error::{JammiError, Result};

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

/// The result-table root every member row carries, VERBATIM — the same
/// string [`crate::config::JammiConfig::resolved_result_root`] returns for
/// the deployment, and the SAME string
/// [`crate::store::ResultStore`] roots itself at (never re-derived, never
/// re-parsed, never re-spelled). [`Self::resolved`] is the ONLY production
/// constructor — it calls `resolved_result_root` itself, so a `MemberRoot`
/// can never carry a string that did not come from the resolver.
/// `MemberRoot::new` wraps an arbitrary string with no resolver call at
/// all; it is compiled only under `feature = "test-hooks"`
/// (fixtures/tests), never in a production build (not a doc link: the
/// method does not exist in a build without that feature, so an intra-doc
/// link to it fails `cargo doc`'s default-feature pass) — the string
/// constructor being reachable from production code is exactly how an
/// unrelated string ends up in the `instances.result_root` column.
///
/// **Not consulted by [`super::Catalog::list_gang_members`] in this unit**
/// (contract `feat_500-C-U5b-1a` §12, P-Y1/P-Y2, the round-5 excision):
/// [`GangListing`] carries no root field at all, so two members rooted at
/// byte-DIFFERENT spellings of the same or different locations (`gcs://b/p`
/// vs `gs://b/p`, `file:///a` vs `s3://b`) ARE gang members of each other —
/// this predicate has nothing to say about the root. The column is still
/// written, verbatim, for every member row: root identity across spellings,
/// and any membership predicate built on it, is
/// `docs/plans/67-distributed-training/README.md` unit U5b-1a-A2's
/// question, and a precondition of U5b-1b-ii (gang formation).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemberRoot(String);

impl MemberRoot {
    /// The ONE production constructor: the verbatim root
    /// [`crate::config::JammiConfig::resolved_result_root`] computes for
    /// `config` — the exact string [`crate::store::ResultStore`] roots
    /// itself at. [`InstanceRegistration::from_config`] is this method's
    /// only caller; nothing else builds a `MemberRoot` in a production
    /// build (`new` below does not exist outside `feature = "test-hooks"`).
    pub fn resolved(config: &crate::config::JammiConfig) -> Result<Self> {
        Ok(Self(config.resolved_result_root()?))
    }

    /// Wrap an ALREADY-RESOLVED root string directly, with no resolver call
    /// and no validation — fixtures and tests only. Gated behind
    /// `feature = "test-hooks"` so a production build never links this
    /// constructor: reachable from production code, it would let any
    /// caller put an arbitrary string in the `instances.result_root`
    /// column, defeating the one property this type exists to hold (the
    /// row and the store are the same string, constructible only through
    /// [`Self::resolved`]).
    #[cfg(feature = "test-hooks")]
    pub fn new(root: impl Into<String>) -> Self {
        Self(root.into())
    }

    /// The verbatim root string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for MemberRoot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
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

/// [`super::Catalog::list_gang_members`]'s request shape. Carries **no
/// root field** (contract `feat_500-C-U5b-1a` §12, P-Y1, the round-5
/// excision): the predicate admits on `kind`, `workers.state == claiming`,
/// `peer_addr` presence, freshness, and self-exclusion ONLY. Root identity
/// is `docs/plans/67-distributed-training/README.md` unit U5b-1a-A2's
/// question, not this verb's.
#[derive(Debug, Clone, Copy)]
pub struct GangListing<'a> {
    /// The `workers.kinds` token this listing matches (a whole,
    /// comma-split, trimmed token — never a substring).
    pub kind: &'a str,
    /// This caller's own `instance_id` — excluded from the result.
    pub self_instance: &'a str,
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
