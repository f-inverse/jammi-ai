//! Service tiers — the mechanism that lets one `jammi-server` binary scale to
//! many deployment shapes by mounting only the gRPC services a deployment needs.
//!
//! ## The model
//!
//! A *tier* is a named group of gRPC services that a deployment either mounts or
//! does not. One tier is always present; the rest are opt-in:
//!
//! - [`ServiceTier::Core`] — **always** mounted: `SessionService` (the tenant
//!   trio + the `GetServerInfo` handshake), `EmbeddingService`,
//!   `InferenceService`, `MutableTableService`, `ChannelService`,
//!   `AuditService`. These are the serve-path primitives every deployment
//!   needs: embed, infer, read result/mutable tables, observe channel state,
//!   and read audit records. There is no useful Jammi server without them.
//! - [`ServiceTier::Train`] — `FineTuneService` (`StartFineTune` /
//!   `FineTuneStatus`). A serve-only box does not train.
//! - [`ServiceTier::Event`] — `TriggerService` (topics / publish / subscribe).
//!   The enterprise tier builds on this trigger stream.
//! - [`ServiceTier::Eval`] — `EvalService` (per-query eval arrays). A tooling
//!   surface, not part of the serve or train hot path.
//!
//! `MutableTable`/`Channel`/`Audit` sit in **core**, not in a tier of their
//! own: they are read/write data primitives the serve path depends on (a
//! serve-only box that embeds and queries result tables needs mutable-table
//! reads; audit is introspection every surface emits), so splitting them out
//! would leave a "serve" deployment unable to serve. Only `FineTune`,
//! `Trigger`, and `Eval` are role-specific enough to gate.
//!
//! ## Capability matches deployment
//!
//! A deployment advertises exactly the tiers it mounted, over the wire, in
//! [`crate::grpc::proto::session::ServerInfo::services`]. Reaching a verb whose
//! tier was not mounted is a truthful tonic `Unimplemented` — the service-mount
//! analog of the client `connect(target)` capability-by-build: the box that did
//! not opt into `train` does not advertise or answer train verbs.
//!
//! ## Runtime config vs compile features
//!
//! Two independent gates compose, and a tier is mounted **iff both pass**:
//!
//! 1. **Compile feature** ([`ServiceTier::compiled_in`]) — a hard ceiling. A
//!    tier whose code is `#[cfg]`-gated out of the binary *cannot* be mounted,
//!    no matter what config says; requesting it is a startup error
//!    ([`TierError::FeatureNotCompiled`]). Today only `Train` carries such a
//!    gate (the server `train` feature, default-on, which compiles the
//!    `FineTuneService` mount). The mechanism is general: adding a `#[cfg]` to
//!    any tier's `compiled_in` arm makes a config request for the compiled-out
//!    tier a truthful error with no other change.
//! 2. **Runtime config** ([`crate::config`-driven `TierSelection`]) — the
//!    deployment's choice *under* that ceiling. One binary, many shapes, no
//!    rebuild: `[server] services` in `jammi.toml` selects the optional tiers.
//!
//! This granularity (runtime toggles layered on the existing compile features)
//! is deliberate: heavy deps stay feature-gated for binary size / build cost,
//! while a single published image stays flexible across serve-only, train,
//! event, and all-in-one deployments without a per-shape rebuild.

use std::collections::BTreeSet;
use std::fmt;
use std::str::FromStr;

use jammi_db::config::ServiceSelection;

/// One mountable group of gRPC services. The wire/config name is the
/// `snake_case` token returned by [`Self::as_str`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ServiceTier {
    /// Always mounted: session/embedding/inference + mutable-table/channel/audit
    /// + the `GetServerInfo` handshake. Cannot be disabled.
    Core,
    /// `FineTuneService` — model training.
    Train,
    /// `TriggerService` — topic / publish / subscribe event streams.
    Event,
    /// `EvalService` — per-query evaluation arrays.
    Eval,
}

impl ServiceTier {
    /// The optional tiers — every tier a deployment may turn on or off. `Core`
    /// is excluded: it is always mounted.
    pub const OPTIONAL: [ServiceTier; 3] =
        [ServiceTier::Eval, ServiceTier::Event, ServiceTier::Train];

    /// The wire/config token for this tier.
    pub fn as_str(self) -> &'static str {
        match self {
            ServiceTier::Core => "core",
            ServiceTier::Train => "train",
            ServiceTier::Event => "event",
            ServiceTier::Eval => "eval",
        }
    }

    /// Whether this tier's mount code is compiled into the running binary.
    ///
    /// This is the hard ceiling the runtime config cannot exceed. `Core`,
    /// `Event`, and `Eval` are always compiled in the OSS build; `Train` is
    /// gated on the server `train` feature (default-on), so a `--no-default-
    /// features` serve-only build genuinely carries no `FineTuneService` mount
    /// and honestly reports `train` as uncompilable.
    pub fn compiled_in(self) -> bool {
        match self {
            ServiceTier::Core | ServiceTier::Event | ServiceTier::Eval => true,
            ServiceTier::Train => cfg!(feature = "train"),
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
            "train" => Ok(ServiceTier::Train),
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
    #[error("unknown service tier '{0}'; expected one of: core, train, event, eval")]
    Unknown(String),
    /// A tier was requested in config but its code is not compiled into this
    /// binary (its feature is off). Truthful refusal rather than a silent drop.
    #[error(
        "service tier '{0}' is not compiled into this binary; \
         rebuild with the '{0}' feature or remove it from `[server] services`"
    )]
    FeatureNotCompiled(ServiceTier),
}

/// The resolved set of tiers a deployment mounts. Always contains
/// [`ServiceTier::Core`]. Built by [`Self::resolve`] from the optional tiers a
/// deployment selected, after reconciling each against its compile gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierSet {
    tiers: BTreeSet<ServiceTier>,
}

impl TierSet {
    /// Resolve a selection of *optional* tiers into a mountable set.
    ///
    /// `Core` is added unconditionally. Each requested optional tier is checked
    /// against [`ServiceTier::compiled_in`]: a tier whose feature is compiled
    /// out is a [`TierError::FeatureNotCompiled`] (the runtime config cannot
    /// exceed the compile ceiling). The result is the truthful set this binary
    /// will mount and advertise.
    pub fn resolve(optional: impl IntoIterator<Item = ServiceTier>) -> Result<Self, TierError> {
        let mut tiers = BTreeSet::new();
        tiers.insert(ServiceTier::Core);
        for tier in optional {
            if tier == ServiceTier::Core {
                // Core is implicit; an explicit `core` in the list is harmless.
                continue;
            }
            if !tier.compiled_in() {
                return Err(TierError::FeatureNotCompiled(tier));
            }
            tiers.insert(tier);
        }
        Ok(Self { tiers })
    }

    /// Resolve a deployment's [`ServiceSelection`] into a mountable set.
    ///
    /// `All` expands to every optional tier compiled into this binary
    /// (all-in-one). `Only(tokens)` parses each token to a [`ServiceTier`]
    /// (rejecting unknown names) and resolves under the compile gate. This is
    /// the single bridge from the engine's raw-token config (`jammi-db` knows no
    /// tier vocabulary) to the server's typed tier set.
    pub fn from_config(selection: &ServiceSelection) -> Result<Self, TierError> {
        match selection {
            ServiceSelection::All(_) => Ok(Self::all_compiled()),
            ServiceSelection::Only(tokens) => {
                let tiers = tokens
                    .iter()
                    .map(|t| ServiceTier::from_str(t))
                    .collect::<Result<Vec<_>, _>>()?;
                Self::resolve(tiers)
            }
        }
    }

    /// The full set: core plus every optional tier compiled into this binary.
    /// This is the default deployment shape (all-in-one) and the embedded
    /// build's capability ceiling.
    pub fn all_compiled() -> Self {
        let optional = ServiceTier::OPTIONAL
            .into_iter()
            .filter(|t| t.compiled_in());
        // `resolve` cannot fail here: every tier passed is compiled-in by the
        // filter above.
        Self::resolve(optional).expect("compiled-in tiers always resolve")
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
        let set = TierSet::resolve(std::iter::empty()).expect("empty resolves");
        assert!(set.contains(ServiceTier::Core));
        assert_eq!(set.as_wire(), vec!["core".to_string()]);
    }

    #[test]
    fn resolve_adds_requested_compiled_in_tiers() {
        // Event and Eval are always compiled in.
        let set = TierSet::resolve([ServiceTier::Event, ServiceTier::Eval]).expect("resolve");
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
        let set = TierSet::resolve([ServiceTier::Core, ServiceTier::Event]).expect("resolve");
        assert_eq!(set.as_wire(), vec!["core".to_string(), "event".to_string()]);
    }

    #[test]
    fn all_compiled_includes_core_and_every_compiled_optional() {
        let set = TierSet::all_compiled();
        assert!(set.contains(ServiceTier::Core));
        assert!(set.contains(ServiceTier::Event));
        assert!(set.contains(ServiceTier::Eval));
        // Train is present iff its feature is compiled in.
        assert_eq!(
            set.contains(ServiceTier::Train),
            ServiceTier::Train.compiled_in()
        );
    }

    #[test]
    fn wire_tokens_are_sorted_and_round_trip() {
        let set = TierSet::all_compiled();
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

    #[test]
    fn from_config_all_is_all_compiled() {
        let set = TierSet::from_config(&ServiceSelection::default()).expect("default resolves");
        assert_eq!(set, TierSet::all_compiled());
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

    /// When `train` is compiled out, requesting it is a truthful refusal — not a
    /// silent drop. This is the runtime-vs-feature reconciliation: config cannot
    /// exceed the compile ceiling.
    #[cfg(not(feature = "train"))]
    #[test]
    fn requesting_a_compiled_out_tier_is_a_truthful_error() {
        let err = TierSet::resolve([ServiceTier::Train]).unwrap_err();
        assert_eq!(err, TierError::FeatureNotCompiled(ServiceTier::Train));
        // `all_compiled` simply omits it — no error, honestly absent.
        assert!(!TierSet::all_compiled().contains(ServiceTier::Train));
    }

    /// When `train` IS compiled in (the default build), requesting it mounts it.
    #[cfg(feature = "train")]
    #[test]
    fn requesting_a_compiled_in_train_tier_mounts_it() {
        let set = TierSet::resolve([ServiceTier::Train]).expect("train resolves when compiled");
        assert!(set.contains(ServiceTier::Train));
    }
}
