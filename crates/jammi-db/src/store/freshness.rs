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

use crate::catalog::result_repo::ResultTableRecord;
use crate::error::{JammiError, Result};
use crate::storage::StorageUrl;

use super::manifest::{AnchorKind, DefinitionHash, InputAnchor};
use super::ResultStore;

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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "cache_outcome", rename_all = "snake_case")]
pub enum CacheOutcome {
    /// The producer ran its compute and materialised a new table.
    Computed,
    /// An exact cache hit short-circuited the compute; the named already-`ready`
    /// table was reused.
    Reused {
        /// The reused cached table's name.
        table: String,
    },
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
    /// Find a `ready` result table already materialised by the **exact** same
    /// definition over the **exact** same input anchors — a cache hit a producer
    /// could reuse instead of recomputing. Returns the cached table's name, or
    /// `None` for a miss. Read-only; tenant-scoped like every catalog read.
    ///
    /// The candidate set is narrowed by the indexed predicate
    /// `definition_hash = $1 AND status = 'ready'` (migration 022); the *exact*
    /// `input_anchors` match is a Rust set-equality post-filter over each
    /// candidate's decoded `input_anchors_json`, because an anchor set is a
    /// structured value, not a SQL-comparable scalar.
    ///
    /// **An [`AnchorKind::UnpinnedAtInstant`] anchor in the requested set is
    /// never a hit.** Such an anchor is a wall-clock instant, not a reproducible
    /// id: two reads of the same unpinned source at different instants carry
    /// different anchors and may have seen different data, so equal instants do
    /// not prove equal inputs — a "hit" on one would be fabricated reuse. A
    /// requested set containing any unpinned anchor short-circuits to a miss.
    pub async fn lookup_cached(
        &self,
        definition: &DefinitionHash,
        inputs: &[InputAnchor],
    ) -> Result<Option<String>> {
        // An unpinned input means the request itself is not reproducibly
        // identifiable, so no recorded table can be a sound reuse of it.
        if inputs
            .iter()
            .any(|a| a.kind == AnchorKind::UnpinnedAtInstant)
        {
            return Ok(None);
        }

        let candidates = self
            .catalog()
            .find_ready_result_tables_by_definition(definition.as_str())
            .await?;

        for candidate in candidates {
            // A post-contract `ready` row always carries `input_anchors_json`
            // (written in the same transaction as `definition_hash`); a row
            // without it is pre-contract and could not have matched the
            // definition-hash filter, so this is belt-and-braces, not a band-aid.
            let Some(ref anchors_json) = candidate.input_anchors_json else {
                continue;
            };
            let recorded: Vec<InputAnchor> = serde_json::from_str(anchors_json)?;
            if anchor_sets_equal(&recorded, inputs) {
                return Ok(Some(candidate.table_name));
            }
        }
        Ok(None)
    }

    /// The **action-layer** cache probe a producer runs at the top of its verb,
    /// before the expensive compute: resolve [`Self::lookup_cached`] for the
    /// exact `(definition, inputs)`, then confirm the hit's Parquet artifact is
    /// still extant on disk. Returns the reusable table's name on a sound hit,
    /// `None` on a miss.
    ///
    /// The extant-artifact check is the difference between this and the bare
    /// [`Self::lookup_cached`] sensor: a `ready` catalog row whose bytes were
    /// reaped (a torn write that committed `ready` before durability on a power
    /// loss; a half-deleted table) must *not* be handed back as a reuse — the
    /// producer would short-circuit to a table that cannot be read. A cache hit
    /// is only sound when the catalog row *and* its artifact both survive, so the
    /// probe re-confirms the bytes the cached row points at. An
    /// [`AnchorKind::UnpinnedAtInstant`] input never reaches the extant check:
    /// [`Self::lookup_cached`] already short-circuits it to a miss, so an
    /// unpinned-anchored producer is honestly never a hit.
    pub async fn probe_cache(
        &self,
        definition: &DefinitionHash,
        inputs: &[InputAnchor],
    ) -> Result<Option<String>> {
        let Some(table_name) = self.lookup_cached(definition, inputs).await? else {
            return Ok(None);
        };
        // A hit names a `ready` row; confirm its Parquet bytes are still present.
        // A row whose artifact was reaped is not a sound reuse — fall through to a
        // recompute rather than short-circuit to an unreadable table.
        let Some(record) = self.catalog().get_result_table(&table_name).await? else {
            return Ok(None);
        };
        let parquet_url = StorageUrl::parse(&record.parquet_path)?;
        let handle = self.open_parquet(&parquet_url)?;
        let path = handle.data_path()?;
        if handle.exists(&path).await? {
            Ok(Some(table_name))
        } else {
            Ok(None)
        }
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
    pub async fn current_anchor(&self, anchor: &InputAnchor) -> Result<CurrentAnchor> {
        match anchor.kind {
            AnchorKind::ResultDigest => {
                let Some(parent) = self.catalog().get_result_table(&anchor.source).await? else {
                    return Ok(CurrentAnchor::Vanished);
                };
                let parquet_url = StorageUrl::parse(&parent.parquet_path)?;
                match self.read_materialization_manifest(&parquet_url).await? {
                    Some(manifest) => Ok(CurrentAnchor::ResultDigest(manifest.artifact.0)),
                    // A resolvable result table with no manifest is a pre-contract
                    // parent: its current digest is recomputed from its bytes, the
                    // same fall-back `result_digest_anchor` uses, so the comparison
                    // is against the parent's true present content.
                    None => {
                        let handle = self.open_parquet(&parquet_url)?;
                        let path = handle.data_path()?;
                        let bytes = handle.get_bytes(&path).await?;
                        Ok(CurrentAnchor::ResultDigest(
                            super::manifest::ArtifactDigest::of_bytes(&bytes).0,
                        ))
                    }
                }
            }
            AnchorKind::UnpinnedAtInstant
            | AnchorKind::MutableVersion
            | AnchorKind::SourceVersion => Ok(CurrentAnchor::Undecidable),
        }
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

/// Set equality of two input-anchor lists: same anchors, order-insensitive.
///
/// Input anchors are a *set* of `(source, anchor, kind)` triples — a producer's
/// declaration order is incidental, so two materialisations over the same inputs
/// in a different order are the same cache key. A source appears at most once in
/// a producer's anchor set (it reads each input once), so a length check plus a
/// containment check in each direction is exact without deduplication.
fn anchor_sets_equal(a: &[InputAnchor], b: &[InputAnchor]) -> bool {
    a.len() == b.len() && a.iter().all(|x| b.contains(x)) && b.iter().all(|y| a.contains(y))
}
