//! Service tiers — the mechanism that lets one `jammi-server` binary scale to
//! many deployment shapes by mounting only the gRPC services a deployment needs.
//!
//! ## The model
//!
//! A *tier* is a named group of gRPC services that a deployment either mounts or
//! does not. One tier is always present; the rest are opt-in:
//!
//! - [`ServiceTier::Core`] — **always** mounted: the control-plane
//!   `CatalogService` (the tenant trio + the `GetServerInfo` handshake, plus the
//!   sources / models / channels / mutable-tables / topic-admin catalog verbs),
//!   `EmbeddingService`, `InferenceService`, `PipelineService`, `AuditService`,
//!   and `TrainingService` — the job submission surface (`StartTraining` /
//!   `TrainingStatus`). These are the serve-path primitives every deployment
//!   needs: bind a tenant, embed, infer, read result/mutable tables, observe
//!   channel state, read audit records, submit a job and read its status.
//!   There is no useful Jammi server without them.
//! - [`ServiceTier::Event`] — `TriggerService` (publish / subscribe). A
//!   downstream tier builds on this trigger stream. (Topic *admin* is a
//!   control-plane catalog verb, always present; only the publish/subscribe
//!   compute stream is event-gated.)
//! - [`ServiceTier::Eval`] — `EvalService` (per-query eval arrays). A tooling
//!   surface, not part of the serve hot path.
//!
//! Submitting a job and *running* it are different questions. Submission is
//! core: every deployment accepts a job and reports its status. Whether THIS
//! process also claims and executes queued jobs is the `[worker] enabled`
//! runtime key ([`jammi_db::config::WorkerConfig`]), read by
//! [`crate::runtime::assemble_grpc_chain`] — not a tier and not a build
//! feature. A request node runs `[worker] enabled = false` and still accepts
//! every submission; a compute node runs `services = []` with `[worker]
//! enabled = true, kinds = [...]` and claims what the request nodes queued.
//!
//! The catalog / mutable-table / channel / audit / job-submission verbs sit in
//! **core**, not in a tier of their own: they are the control-plane +
//! read/write data primitives the serve path depends on (a serve-only box that
//! embeds and queries result tables needs mutable-table reads; audit is
//! introspection every surface emits), so splitting them out would leave a
//! "serve" deployment unable to serve. Only `Event` and `Eval` are
//! role-specific enough to gate.
//!
//! ## Capability matches deployment
//!
//! A deployment advertises exactly the tiers it mounted, over the wire, in
//! [`crate::grpc::proto::catalog::ServerInfo::services`]. Reaching a verb whose
//! tier was not mounted is a truthful tonic `Unimplemented` — the service-mount
//! analog of the client `connect(target)` capability-by-build: the box that did
//! not opt into `eval` does not advertise or answer eval verbs.
//!
//! ## Runtime config, no compile features
//!
//! Every tier compiles into every build. There is no per-tier cargo feature
//! and therefore no compile ceiling for the runtime selection to hit: `[server]
//! services` in `jammi.toml` (or `JAMMI_SERVER__SERVICES`) selects the optional
//! tiers, and the only resolution error is a token naming no tier
//! ([`TierError::Unknown`]). One binary, many shapes, no rebuild — a single
//! published image stays flexible across serve-only, event, eval, and
//! all-in-one deployments without a per-shape rebuild.

use std::collections::BTreeSet;
use std::fmt;
use std::str::FromStr;

use jammi_db::config::ServiceSelection;

/// One mountable group of gRPC services. The wire/config name is the
/// `snake_case` token returned by [`Self::as_str`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ServiceTier {
    /// Always mounted: session/embedding/inference/pipeline + mutable-table/
    /// channel/audit + job submission + the `GetServerInfo` handshake. Cannot
    /// be disabled.
    Core,
    /// `TriggerService` — topic / publish / subscribe event streams.
    Event,
    /// `EvalService` — per-query evaluation arrays.
    Eval,
}

impl ServiceTier {
    /// The optional tiers — every tier a deployment may turn on or off. `Core`
    /// is excluded: it is always mounted.
    pub const OPTIONAL: [ServiceTier; 2] = [ServiceTier::Eval, ServiceTier::Event];

    /// The wire/config token for this tier.
    pub fn as_str(self) -> &'static str {
        match self {
            ServiceTier::Core => "core",
            ServiceTier::Event => "event",
            ServiceTier::Eval => "eval",
        }
    }
}

impl fmt::Display for ServiceTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ServiceTier {
    type Err = TierError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "core" => Ok(ServiceTier::Core),
            "event" => Ok(ServiceTier::Event),
            "eval" => Ok(ServiceTier::Eval),
            other => Err(TierError::Unknown(other.to_string())),
        }
    }
}

/// Errors from resolving a tier selection into a mountable [`TierSet`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TierError {
    /// A config token named no known tier.
    #[error("unknown service tier '{0}'; expected one of: core, event, eval")]
    Unknown(String),
}

/// The resolved set of tiers a deployment mounts. Always contains
/// [`ServiceTier::Core`]. Built by [`Self::resolve`] from the optional tiers a
/// deployment selected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierSet {
    tiers: BTreeSet<ServiceTier>,
}

impl TierSet {
    /// Resolve a selection of *optional* tiers into a mountable set.
    ///
    /// `Core` is added unconditionally; an explicit `Core` in the selection is
    /// harmless. Every tier compiles into every build, so a typed selection
    /// always resolves — the result is the truthful set this binary will mount
    /// and advertise.
    pub fn resolve(optional: impl IntoIterator<Item = ServiceTier>) -> Self {
        let mut tiers = BTreeSet::new();
        tiers.insert(ServiceTier::Core);
        tiers.extend(optional);
        Self { tiers }
    }

    /// Resolve a deployment's [`ServiceSelection`] into a mountable set.
    ///
    /// `All` expands to every optional tier (all-in-one). `Only(tokens)` parses
    /// each token to a [`ServiceTier`], rejecting unknown names. This is the
    /// single bridge from the engine's raw-token config (`jammi-db` knows no
    /// tier vocabulary) to the server's typed tier set.
    pub fn from_config(selection: &ServiceSelection) -> Result<Self, TierError> {
        match selection {
            ServiceSelection::All(_) => Ok(Self::all()),
            ServiceSelection::Only(tokens) => {
                let tiers = tokens
                    .iter()
                    .map(|t| ServiceTier::from_str(t))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Self::resolve(tiers))
            }
        }
    }

    /// The full set: core plus every optional tier. This is the default
    /// deployment shape (all-in-one).
    pub fn all() -> Self {
        Self::resolve(ServiceTier::OPTIONAL)
    }

    /// Whether this set mounts `tier`.
    pub fn contains(&self, tier: ServiceTier) -> bool {
        self.tiers.contains(&tier)
    }

    /// The mounted tiers as wire tokens, **sorted alphabetically** — the value
    /// of `ServerInfo.services` this deployment advertises. Sorted on the token
    /// string (not the enum's `Ord`, which follows declaration order) so the
    /// handshake value is stable and matches the documented contract.
    pub fn as_wire(&self) -> Vec<String> {
        let mut wire: Vec<String> = self.tiers.iter().map(|t| t.as_str().to_string()).collect();
        wire.sort();
        wire
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_is_always_present_even_with_an_empty_selection() {
        let set = TierSet::resolve(std::iter::empty());
        assert!(set.contains(ServiceTier::Core));
        assert_eq!(set.as_wire(), vec!["core".to_string()]);
    }

    #[test]
    fn resolve_adds_requested_tiers() {
        let set = TierSet::resolve([ServiceTier::Event, ServiceTier::Eval]);
        assert!(set.contains(ServiceTier::Core));
        assert!(set.contains(ServiceTier::Event));
        assert!(set.contains(ServiceTier::Eval));
        assert_eq!(
            set.as_wire(),
            vec!["core".to_string(), "eval".to_string(), "event".to_string()]
        );
    }

    #[test]
    fn explicit_core_in_the_selection_is_harmless() {
        let set = TierSet::resolve([ServiceTier::Core, ServiceTier::Event]);
        assert_eq!(set.as_wire(), vec!["core".to_string(), "event".to_string()]);
    }

    #[test]
    fn all_includes_core_and_every_optional() {
        let set = TierSet::all();
        assert!(set.contains(ServiceTier::Core));
        for tier in ServiceTier::OPTIONAL {
            assert!(set.contains(tier), "all() omits {tier}");
        }
        assert_eq!(
            set.as_wire(),
            vec!["core".to_string(), "eval".to_string(), "event".to_string()]
        );
    }

    #[test]
    fn wire_tokens_are_sorted_and_round_trip() {
        let set = TierSet::all();
        let wire = set.as_wire();
        let mut sorted = wire.clone();
        sorted.sort();
        assert_eq!(wire, sorted, "wire tokens are sorted (BTreeSet order)");
        for token in &wire {
            let tier = ServiceTier::from_str(token).expect("token parses");
            assert!(set.contains(tier));
        }
    }

    #[test]
    fn unknown_tier_token_is_an_error() {
        assert_eq!(
            ServiceTier::from_str("registry"),
            Err(TierError::Unknown("registry".to_string()))
        );
    }

    /// `train` is not a tier: job submission is core and job execution is the
    /// `[worker] enabled` runtime key, so a config still naming the former
    /// tier is refused by name rather than silently accepted.
    #[test]
    fn the_former_train_token_is_an_unknown_tier() {
        assert_eq!(
            ServiceTier::from_str("train"),
            Err(TierError::Unknown("train".to_string()))
        );
    }

    #[test]
    fn from_config_all_is_all() {
        let set = TierSet::from_config(&ServiceSelection::default()).expect("default resolves");
        assert_eq!(set, TierSet::all());
    }

    #[test]
    fn from_config_empty_only_is_serve_only() {
        let set =
            TierSet::from_config(&ServiceSelection::Only(vec![])).expect("serve-only resolves");
        assert_eq!(set.as_wire(), vec!["core".to_string()]);
    }

    #[test]
    fn from_config_named_tier() {
        let set = TierSet::from_config(&ServiceSelection::Only(vec!["event".to_string()]))
            .expect("event resolves");
        assert!(set.contains(ServiceTier::Event));
        assert!(!set.contains(ServiceTier::Eval));
    }

    #[test]
    fn from_config_unknown_token_is_an_error() {
        let err = TierSet::from_config(&ServiceSelection::Only(vec!["registry".to_string()]))
            .unwrap_err();
        assert_eq!(err, TierError::Unknown("registry".to_string()));
    }
}
