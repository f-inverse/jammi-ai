//! The **sensing** layer of incremental recompute: read-only staleness,
//! lineage, and cache-lookup over the materialization contract every result
//! table carries ([`crate::store::manifest`]).
//!
//! This layer *reports*; it never acts. It answers three questions a recompute
//! decision (or a feature store, a lineage UI, an attribution chain) asks
//! generically, all by reading the recorded [`DefinitionHash`] and
//! [`InputAnchor`]s — never by mutating anything or re-running a producer:
//!
//! 1. **Is this artifact still fresh?** [`ResultStore::staleness`] compares a
//!    `ready` table's recorded definition hash and input anchors against what
//!    they are *now*: a [`Staleness`] verdict.
//! 2. **Has this exact definition-over-inputs already been materialised?**
//!    [`ResultStore::lookup_cached`] finds a `ready` table with the same
//!    `(definition_hash, input_anchors)` — a cache hit a producer could reuse
//!    instead of recomputing.
//! 3. **What derives from this table?** [`ResultStore::derives_from`] returns
//!    the one-hop reverse-dependency edges (the tables that anchored on it), the
//!    data a caller walks transitively to find everything downstream of a change.
//!
//! # Honest scoping of what can be resolved *now*
//!
//! Freshness is only as confident as the inputs are reproducibly identifiable.
//! Of the four [`AnchorKind`]s, only two have a live current-state surface this
//! engine can read today:
//!
//! - [`AnchorKind::ResultDigest`] — the input is an immutable result table; its
//!   *current* anchor is its current artifact digest, which this layer reads
//!   from the input's own manifest. A recomputed parent gets a new digest, so a
//!   child anchored on the old one is detected stale by the same comparison —
//!   recursion falls out of the per-input comparison with no special case.
//! - [`AnchorKind::UnpinnedAtInstant`] — the input was an external source with
//!   no version surface, anchored only by a read instant. An instant is not a
//!   reproducible id, so such an input can never be confidently `Fresh`; it
//!   contributes to [`Staleness::Undecidable`] and *never* yields a cache hit.
//!
//! [`AnchorKind::MutableVersion`] and [`AnchorKind::SourceVersion`] are
//! **structurally unreachable in a recorded anchor today**: no live producer
//! emits one as the anchor a downstream table senses against, and — critically —
//! there is **no current-resolution surface** for them (the `mutable_tables`
//! catalog has no monotonic version column to re-read; an external source's
//! as-of column is resolved at scan time, not stored for re-resolution). Rather
//! than fabricate a read against a surface that does not exist, this layer
//! resolves both to [`CurrentAnchor::Undecidable`] and documents it: when a
//! producer first anchors a downstream table on a mutable/source version *and*
//! the catalog grows the surface to re-resolve it, these arms gain a live
//! resolution — the comparison shape is already in place.

use serde::{Deserialize, Serialize};

use crate::catalog::artifact_repo::ArtifactRef;
use crate::catalog::result_repo::ResultTableName;

use crate::catalog::result_repo::ResultTableRecord;
use crate::error::{JammiError, Result};
use crate::storage::StorageUrl;

use super::manifest::{
    exact_reuse_matches, AnchorKind, DefinitionHash, InputAnchor, MaterializationEnv,
    MaterializationManifest, ProducingDescriptor, ReuseCandidate,
};
use super::{PinnedSource, ResultStore};

/// Whether a producer reuses an already-materialised result for its exact
/// `(definition, input anchors)` instead of recomputing it — the **opt-in**
/// memoization dial every result-table producer carries.
///
/// The default is [`Self::Bypass`], never [`Self::Use`]: a producer must never
/// silently hand back a table the caller did not just compute. Surprise reuse is
/// the "honest, not silent" sin — a caller that wanted a *fresh* run and got a
/// cached one, with no signal, cannot tell the difference. Reuse is therefore
/// both explicitly requested (`Use`) *and* explicitly reported (the producer
/// returns a [`CacheOutcome`] so the caller observes which path ran), never
/// inferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CachePolicy {
    /// Probe the cache before computing: on an exact `(definition, inputs)` hit
    /// with an extant artifact, short-circuit and reuse the cached table,
    /// skipping the expensive compute.
    Use,
    /// Always recompute. The default — a producer never reuses a prior result
    /// unless the caller opts in.
    #[default]
    Bypass,
}

/// Which path a producer took, returned so reuse is **observable**, never
/// inferred. A caller that passed [`CachePolicy::Use`] learns from the outcome
/// whether the expensive compute ran ([`Self::Computed`]) or an existing
/// artifact was reused ([`Self::Reused`]) — the honest signal that distinguishes
/// a fresh run from a cache hit.
///
/// One value on every surface: a producer returns it, a job records it in its
/// terminal payload (`{"outcome":"computed"}` /
/// `{"outcome":"reused","reused":{"table":…}}` /
/// `{"outcome":"reused","reused":{"model":…}}`), and the wire carries it as
/// the `jammi.v1.inference.CacheOutcome` message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", content = "reused", rename_all = "snake_case")]
pub enum CacheOutcome {
    /// The producer ran its compute and materialised a new artifact.
    Computed,
    /// An exact hit short-circuited the compute; this already-committed
    /// artifact was reused.
    Reused(ReusedArtifact),
}

/// The identity of the artifact a reuse handed back — a typed reference,
/// never a bare relation name, so a reuse outcome cannot be formatted into a
/// query without going through the identity's own reviewed accessor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReusedArtifact {
    /// An already-`ready` result table.
    Table(ResultTableName),
    /// An already-`published` model artifact.
    Model(ArtifactRef),
}

/// Whether a `ready` result table is still the output of its recorded
/// definition over its recorded inputs' *current* state — a read-only verdict
/// the engine reports and never acts on.
///
/// The variants are ordered by confidence. `Fresh` is the only verdict that
/// asserts reuse is safe; every other arm is a reason a reader must decide for
/// itself (recompute, accept, alarm) — the engine ships the sensor, never the
/// policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "staleness", rename_all = "snake_case")]
pub enum Staleness {
    /// The recorded definition hash equals the current definition's *and* every
    /// recorded input anchor equals its current anchor. The artifact is the
    /// output of its definition over the inputs' present state — reuse is safe.
    Fresh,
    /// At least one determinant changed and *every* changed determinant was
    /// confidently resolvable (no undecidable input clouded the verdict). The
    /// table is provably out of date for the reasons listed.
    Stale {
        /// The confident reasons the artifact is out of date, in input order
        /// (definition first when it changed).
        reasons: Vec<StaleReason>,
    },
    /// Freshness cannot be confidently asserted because one or more inputs have
    /// no reproducible current anchor (an [`AnchorKind::UnpinnedAtInstant`], or
    /// a kind with no current-resolution surface). Any *confidently* resolved
    /// staleness reasons are still reported, so a reader sees both the proven
    /// drift and the inputs that cloud the rest — an honest "I don't fully
    /// know", never a fabricated `Fresh`.
    Undecidable {
        /// The source ids whose current anchor could not be resolved.
        unpinned: Vec<String>,
        /// The staleness reasons that *were* confidently decided despite the
        /// undecidable inputs (e.g. the definition hash changed for certain).
        decided_reasons: Vec<StaleReason>,
    },
    /// The table carries no manifest summary (`definition_hash IS NULL`) — a
    /// pre-contract table created before the materialization contract landed.
    /// A truthful unknown: its freshness cannot be assessed because it has no
    /// recorded definition or anchors, never a fabricated verdict.
    MissingManifest,
}

/// One reason a [`Staleness`] verdict is `Stale` (or a `decided_reason` of an
/// `Undecidable`): a single determinant that diverged from what was recorded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum StaleReason {
    /// The current definition of how this table is produced no longer hashes to
    /// the recorded `definition_hash` — the producing code, parameters, or
    /// environment changed. Carries both hashes for the reader.
    DefinitionChanged {
        /// The `definition_hash` recorded in the table's manifest summary.
        recorded: String,
        /// The current definition hash the caller computed.
        current: String,
    },
    /// An input's current anchor differs from the one recorded — the upstream
    /// state the table was built over advanced (e.g. a parent result table was
    /// recomputed, so its artifact digest changed).
    InputAdvanced {
        /// The input source whose anchor moved.
        source: String,
        /// The anchor recorded at the table's build time.
        recorded: String,
        /// The input's current anchor.
        current: String,
    },
    /// An input recorded in the manifest no longer exists — its source table was
    /// dropped, so the table can never be reproduced from it.
    InputVanished {
        /// The input source that is gone.
        source: String,
    },
}

/// The *current* state-pointer of one recorded input, resolved live — the right
/// side of the per-input comparison [`ResultStore::staleness`] performs. Only
/// the arms that have a live current-resolution surface are present (see the
/// module docs); an input with no resolvable surface is
/// [`Self::Undecidable`], never a fabricated value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "current_anchor", rename_all = "snake_case")]
pub enum CurrentAnchor {
    /// The input is an immutable result table; its current anchor is its current
    /// artifact digest (hex).
    ResultDigest(String),
    /// The input has no reproducible current anchor this engine can read — an
    /// `UnpinnedAtInstant` (an instant is not an id), or a kind with no
    /// current-resolution surface. Freshness against it is undecidable.
    Undecidable,
    /// The input source no longer exists — its result table was dropped.
    Vanished,
}

/// One reverse-dependency edge of the materialization lineage: `derived`
/// anchored on `input` (with anchor kind `kind`), so a change to `input`
/// propagates to `derived`. Returned one hop at a time by
/// [`ResultStore::derives_from`]; a caller walks the relation transitively.
///
/// The lineage is a *view over* the recorded `input_anchors_json` — the single
/// source of truth — not a second edge store: an edge exists iff some `ready`
/// table's manifest summary records `input` as a source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivesFromEdge {
    /// The upstream input source the edge points *from*.
    pub input: String,
    /// The downstream table that anchored on `input`.
    pub derived: String,
    /// The kind of anchor `derived` recorded for `input`.
    pub kind: AnchorKind,
}

impl ResultStore {
    /// The `ready` result tables already materialised by the **exact** same
    /// definition over the **exact** same input anchors, **newest first** — the
    /// shared candidate-resolution [`Self::lookup_cached`] and
    /// [`Self::probe_cache_record`] both build on.
    ///
    /// The candidate set is narrowed by the indexed predicate
    /// `definition_hash = $1 AND status = 'ready'`; the exact anchor match,
    /// the refusal of an [`AnchorKind::UnpinnedAtInstant`] request, and the
    /// newest-first order are [`exact_reuse_matches`]' — the one reuse
    /// predicate every probe in the engine shares. Many rows can share a
    /// `(definition_hash, input_anchors)` key — a producer legitimately
    /// re-materialising the same inputs (an idempotent recompute, or a race)
    /// — and every one of them is a semantically equivalent reuse; this
    /// returns all of them so a caller that needs more than the single newest
    /// (an extant-artifact retry) can fall through to the rest.
    ///
    /// Visible to the rest of `store` (not public) so a producer whose reuse
    /// needs an extra predicate over the candidates — the training-set probe
    /// filters them to [`crate::catalog::result_repo::ResultTableKind::TrainingSet`]
    /// — matches the same `(definition, inputs)` key as
    /// [`Self::probe_cache_record`] instead of restating it.
    pub(super) async fn exact_match_candidates(
        &self,
        definition: &DefinitionHash,
        inputs: &[InputAnchor],
    ) -> Result<Vec<ResultTableRecord>> {
        exact_reuse_matches(inputs, || {
            self.catalog()
                .find_ready_result_tables_by_definition(definition.as_str())
        })
        .await
    }

    /// Find a `ready` result table already materialised by the **exact** same
    /// definition over the **exact** same input anchors — a cache hit a producer
    /// could reuse instead of recomputing. Returns the cached table's name, or
    /// `None` for a miss. Read-only; tenant-scoped like every catalog read.
    ///
    /// When several `ready` rows share the exact `(definition_hash,
    /// input_anchors)` key, this is a pure sensor over the shared exact-match
    /// candidate set (newest-first, per-key): it names the newest one,
    /// deterministically, without touching storage to confirm anything
    /// survives on disk (that confirmation is [`Self::probe_cache_record`]'s
    /// job).
    ///
    /// **An [`AnchorKind::UnpinnedAtInstant`] anchor in the requested set is
    /// never a hit**: no candidate is ever considered for such a request.
    pub async fn lookup_cached(
        &self,
        definition: &DefinitionHash,
        inputs: &[InputAnchor],
    ) -> Result<Option<String>> {
        Ok(self
            .exact_match_candidates(definition, inputs)
            .await?
            .into_iter()
            .next()
            .map(|record| record.table_name))
    }

    /// The **action-layer** cache probe a producer runs at the top of its verb,
    /// before the expensive compute: over every exact `(definition, inputs)`
    /// match (newest first), return the first whose Parquet artifact is still
    /// extant on disk. Returns the reusable table's name on a sound hit, `None`
    /// if no exact match survives.
    ///
    /// The extant-artifact check is the difference between this and the bare
    /// [`Self::lookup_cached`] sensor: a `ready` catalog row whose bytes were
    /// reaped (a torn write that committed `ready` before durability on a power
    /// loss; a half-deleted table) must *not* be handed back as a reuse — the
    /// producer would short-circuit to a table that cannot be read. A cache hit
    /// is only sound when the catalog row *and* its artifact both survive.
    /// Crucially, a reaped artifact is **not** a global miss: when multiple
    /// `ready` rows share the exact key (an idempotent recompute, a race), this
    /// falls through to the next-newest exact match rather than giving up after
    /// the first — a reaped *preferred* candidate must not shadow a sound reuse
    /// that still exists. An [`AnchorKind::UnpinnedAtInstant`] input never
    /// reaches the extant check: no candidate is ever considered for such a
    /// request, so an unpinned-anchored producer is honestly never a hit.
    pub async fn probe_cache(
        &self,
        definition: &DefinitionHash,
        inputs: &[InputAnchor],
    ) -> Result<Option<String>> {
        Ok(self
            .probe_cache_record(definition, inputs)
            .await?
            .map(|record| record.table_name))
    }

    /// [`Self::probe_cache`] returning the reusable table's full
    /// [`ResultTableRecord`] on a sound hit, not just its name — the shape a
    /// producer needs when it short-circuits, so it can hand the reused record
    /// straight back without a second catalog read. `None` on a miss (no exact
    /// match, an unpinned input, or every exact match's artifact reaped).
    /// Read-only and tenant-scoped.
    pub async fn probe_cache_record(
        &self,
        definition: &DefinitionHash,
        inputs: &[InputAnchor],
    ) -> Result<Option<ResultTableRecord>> {
        // Iterate every exact-key candidate newest-first; a row whose artifact
        // was reaped is not a sound reuse — fall through to the next candidate
        // rather than short-circuit to an unreadable table or give up on the
        // whole key.
        for candidate in self.exact_match_candidates(definition, inputs).await? {
            let parquet_url = StorageUrl::parse(&candidate.parquet_path)?;
            let handle = self.open_parquet(&parquet_url)?;
            let path = handle.data_path()?;
            if handle.exists(&path).await? {
                return Ok(Some(candidate));
            }
        }
        Ok(None)
    }

    /// Report whether a `ready` result table is still the output of its recorded
    /// definition over its recorded inputs' *current* state — the read-only
    /// `staleness` sensor. Reports a [`Staleness`]; it acts on nothing (recompute
    /// / accept / alarm is the reader's policy, the `verify_materialization`
    /// stance).
    ///
    /// `Fresh` iff the recorded `definition_hash` equals `current_definition`
    /// *and* every recorded input's current anchor equals its recorded anchor. A
    /// pre-contract row (`definition_hash IS NULL`) is [`Staleness::MissingManifest`].
    /// An input with no reproducible current anchor makes the verdict
    /// [`Staleness::Undecidable`] (never a confident `Fresh`), while still
    /// reporting any confidently-decided staleness reasons.
    pub async fn staleness(
        &self,
        table: &ResultTableRecord,
        current_definition: &DefinitionHash,
    ) -> Result<Staleness> {
        let (Some(recorded_hash), Some(anchors_json)) =
            (&table.definition_hash, &table.input_anchors_json)
        else {
            return Ok(Staleness::MissingManifest);
        };

        let mut decided: Vec<StaleReason> = Vec::new();
        let mut unpinned: Vec<String> = Vec::new();

        if recorded_hash != current_definition.as_str() {
            decided.push(StaleReason::DefinitionChanged {
                recorded: recorded_hash.clone(),
                current: current_definition.as_str().to_string(),
            });
        }

        let recorded_anchors: Vec<InputAnchor> = serde_json::from_str(anchors_json)?;
        for anchor in &recorded_anchors {
            match self.current_anchor(anchor).await? {
                CurrentAnchor::ResultDigest(current) => {
                    if current != anchor.anchor.0 {
                        decided.push(StaleReason::InputAdvanced {
                            source: anchor.source.clone(),
                            recorded: anchor.anchor.0.clone(),
                            current,
                        });
                    }
                }
                CurrentAnchor::Vanished => {
                    decided.push(StaleReason::InputVanished {
                        source: anchor.source.clone(),
                    });
                }
                CurrentAnchor::Undecidable => {
                    unpinned.push(anchor.source.clone());
                }
            }
        }

        if !unpinned.is_empty() {
            return Ok(Staleness::Undecidable {
                unpinned,
                decided_reasons: decided,
            });
        }
        if decided.is_empty() {
            Ok(Staleness::Fresh)
        } else {
            Ok(Staleness::Stale { reasons: decided })
        }
    }

    /// Resolve one recorded [`InputAnchor`] to its *current* state-pointer,
    /// dispatching on the anchor's [`AnchorKind`]:
    ///
    /// - [`AnchorKind::ResultDigest`] → the input result table's current
    ///   artifact digest ([`CurrentAnchor::ResultDigest`]), or
    ///   [`CurrentAnchor::Vanished`] if the table no longer resolves.
    /// - [`AnchorKind::UnpinnedAtInstant`] → [`CurrentAnchor::Undecidable`]: an
    ///   instant is not a reproducible id.
    /// - [`AnchorKind::MutableVersion`] / [`AnchorKind::SourceVersion`] →
    ///   [`CurrentAnchor::Undecidable`]: there is no current-resolution surface
    ///   to read a live version from (see the module docs). This is honest, not
    ///   a fabricated read against a surface that does not exist.
    ///
    /// The parent's current digest is exactly what
    /// [`ResultStore::pin_current_version`] would anchor a new artifact on —
    /// a versioned parent's current version identity (a refresh that changed
    /// content advances it, one that did not leaves every dependent
    /// `Fresh`), a never-refreshed parent's base artifact digest — read from
    /// one pin so the comparison and the producers agree on what "current"
    /// means. Private: the value is a comparison operand for
    /// [`Self::staleness`], never an anchor a caller could pair with a read
    /// of its own.
    async fn current_anchor(&self, anchor: &InputAnchor) -> Result<CurrentAnchor> {
        match anchor.kind {
            AnchorKind::ResultDigest => {
                let Some(parent) = self.catalog().get_result_table(&anchor.source).await? else {
                    return Ok(CurrentAnchor::Vanished);
                };
                let pin = self.pin_current_version(parent).await?;
                Ok(CurrentAnchor::ResultDigest(pin.input_anchor().anchor.0))
            }
            AnchorKind::UnpinnedAtInstant
            | AnchorKind::MutableVersion
            | AnchorKind::SourceVersion => Ok(CurrentAnchor::Undecidable),
        }
    }

    /// The [`ProducingDescriptor`] the pinned table's current content was
    /// produced by — the verbatim verb + typed parameters a refresh derives
    /// its parameters from and a recompute replays: a published version's
    /// delta descriptor (`Embedding` at the base, `EmbeddingDelta` after a
    /// refresh, `EmbeddingCompaction` after a compaction), read from the
    /// manifest the pin already holds; a never-refreshed table's
    /// [`Self::base_descriptor`].
    pub async fn producing_descriptor(&self, pin: &PinnedSource) -> Result<ProducingDescriptor> {
        match pin.published() {
            Some(published) => Ok(published.manifest().delta.descriptor.clone()),
            None => self.base_descriptor(pin.record()).await,
        }
    }

    /// The [`ProducingDescriptor`] a `ready` result table's base artifact
    /// recorded — its `.materialization.json` sidecar (the contract's source
    /// of truth), which never changes after the table is promoted, so this
    /// read resolves no version and needs no pin. It is the replay input for
    /// a table whose current version cannot be resolved: `recompute` is the
    /// documented remedy for `VersionUnavailable`, and every embedding-family
    /// descriptor in a version chain replays as the same full embed of the
    /// base descriptor.
    ///
    /// A table with no manifest sidecar (`definition_hash IS NULL` — a
    /// pre-contract table created before the materialization contract landed)
    /// has no recorded descriptor to replay, so it is a typed
    /// [`JammiError::NotRecomputable`] — a loud refusal, never a re-run
    /// guessed from the table's columns. A reader that only wants to *verify*
    /// reads the opaque hash; a reader that wants to *recompute* reads the
    /// descriptor here.
    pub async fn base_descriptor(&self, table: &ResultTableRecord) -> Result<ProducingDescriptor> {
        Ok(self.base_manifest(table).await?.descriptor)
    }

    /// The environment a `ready` result table's base artifact was produced
    /// in, as its sidecar records it — what a refresh must reproduce. The
    /// same refusal as [`Self::base_descriptor`] for a table without one.
    pub async fn base_env(&self, table: &ResultTableRecord) -> Result<MaterializationEnv> {
        Ok(self.base_manifest(table).await?.env)
    }

    /// The base artifact's sidecar, or [`JammiError::NotRecomputable`] for a
    /// table that has none.
    async fn base_manifest(&self, table: &ResultTableRecord) -> Result<MaterializationManifest> {
        self.recorded_manifest(table)
            .await?
            .ok_or_else(|| JammiError::NotRecomputable {
                table: table.table_name.clone(),
            })
    }

    /// The materialization a result table's base artifact recorded — its
    /// `.materialization.json` sidecar, verbatim — or
    /// [`JammiError::MissingManifest`] for a table that carries none. The
    /// read-only `describe_table` verb: what produced the table (descriptor,
    /// environment, every invoked model's run), over what (input anchors),
    /// and the digests a verifier matches. A versioned table's later versions
    /// share this definition (a refresh refuses on drift); their fragment
    /// digests are attested by the version chain `verify_materialization`
    /// walks.
    pub async fn describe_table(
        &self,
        table: &ResultTableRecord,
    ) -> Result<MaterializationManifest> {
        self.recorded_manifest(table)
            .await?
            .ok_or_else(|| JammiError::MissingManifest {
                table: table.table_name.clone(),
            })
    }

    /// The base artifact's sidecar, `None` when it has none.
    async fn recorded_manifest(
        &self,
        table: &ResultTableRecord,
    ) -> Result<Option<MaterializationManifest>> {
        let parquet_url = StorageUrl::parse(&table.parquet_path)?;
        self.read_materialization_manifest(&parquet_url).await
    }

    /// The full transitive downstream subgraph of `source`: every result table
    /// reachable by following [`Self::derives_from`] edges from `source`, walked
    /// **stack-safely** with an explicit work-stack and a visited set — never
    /// recursion, so an arbitrarily deep lineage chain can never blow the stack.
    ///
    /// The returned edges are the union of every hop's one-hop edges, in
    /// breadth-of-discovery order. A node is expanded at most once (the visited
    /// set), so a diamond (two paths to the same descendant) is walked once, not
    /// twice. A materialization lineage is a DAG by construction — a producer
    /// anchors its inputs before its output exists, so no output can be its own
    /// ancestor — therefore re-entering a node *already on the active descent
    /// path* is a corruption of the recorded anchors, surfaced as a typed
    /// [`JammiError::DependencyCycle`] rather than an infinite walk.
    pub async fn derives_from_closure(&self, source: &str) -> Result<Vec<DerivesFromEdge>> {
        // Iterative depth-first walk with explicit frames, so an arbitrarily deep
        // lineage chain can never blow the Rust call stack. Cycle detection is
        // the DAG back-edge test: a node currently on the active root→node
        // descent path (`on_path`) that is re-encountered as a child closes a
        // cycle. `expanded` records nodes whose subtree is fully walked so a
        // diamond (two paths to the same descendant) is walked once and is *not*
        // mistaken for a cycle — the distinction a flat visited set cannot make.
        //
        // A frame is `(node, its remaining one-hop edges)`. `derives_from` is
        // async, so the edges of a node are fetched once when its frame is pushed
        // (not re-fetched as the frame is revisited).
        struct Frame {
            node: String,
            edges: std::vec::IntoIter<DerivesFromEdge>,
        }

        let mut expanded: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut on_path: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut collected: Vec<DerivesFromEdge> = Vec::new();

        on_path.insert(source.to_string());
        let mut stack: Vec<Frame> = vec![Frame {
            node: source.to_string(),
            edges: self.derives_from(source).await?.into_iter(),
        }];

        while let Some(frame) = stack.last_mut() {
            match frame.edges.next() {
                Some(edge) => {
                    let child = edge.derived.clone();
                    if on_path.contains(&child) {
                        return Err(JammiError::DependencyCycle { table: child });
                    }
                    collected.push(edge);
                    if expanded.contains(&child) {
                        // Already fully walked via another path — a DAG diamond,
                        // not a cycle. Record the edge but don't re-descend.
                        continue;
                    }
                    on_path.insert(child.clone());
                    let edges = self.derives_from(&child).await?.into_iter();
                    stack.push(Frame { node: child, edges });
                }
                None => {
                    // Frame exhausted: its subtree is fully walked. Pop it off the
                    // active path and mark it expanded.
                    let done = stack.pop().expect("frame present in this arm");
                    on_path.remove(&done.node);
                    expanded.insert(done.node);
                }
            }
        }
        Ok(collected)
    }

    /// The one-hop reverse-dependency edges of `source`: every `ready` result
    /// table whose recorded `input_anchors` name `source` as an input. Read-only
    /// and tenant-scoped; a caller walks the relation transitively (with the
    /// stack-safe [`Self::derives_from_closure`] helper) to find the
    /// whole downstream subgraph of a change.
    ///
    /// The candidate set is narrowed by the SQL pre-filter
    /// `input_anchors_json LIKE '%"source":"<name>"%'` — a safe
    /// over-approximation (it can match a different field whose value contains
    /// the substring) refined by an exact decode-and-match in Rust, so the
    /// returned edges are precise. `input_anchors_json` is the single source of
    /// truth; there is no second edge store.
    pub async fn derives_from(&self, source: &str) -> Result<Vec<DerivesFromEdge>> {
        let candidates = self
            .catalog()
            .find_ready_result_tables_anchored_on(source)
            .await?;

        let mut edges = Vec::new();
        for candidate in candidates {
            let Some(ref anchors_json) = candidate.input_anchors_json else {
                continue;
            };
            let anchors: Vec<InputAnchor> = serde_json::from_str(anchors_json)?;
            for anchor in anchors {
                if anchor.source == source {
                    edges.push(DerivesFromEdge {
                        input: source.to_string(),
                        derived: candidate.table_name.clone(),
                        kind: anchor.kind,
                    });
                }
            }
        }
        Ok(edges)
    }
}

impl ReuseCandidate for ResultTableRecord {
    fn recorded_anchors_json(&self) -> Option<&str> {
        self.input_anchors_json.as_deref()
    }

    fn created_at(&self) -> &str {
        &self.created_at
    }

    fn name(&self) -> &str {
        &self.table_name
    }
}
