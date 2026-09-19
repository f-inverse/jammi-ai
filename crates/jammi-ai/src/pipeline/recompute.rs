//! The **action** half of incremental recompute: re-invoke a result table's
//! recorded producer over the inputs' *current* state.
//!
//! Where the sensing layer ([`jammi_db::store::freshness`]) *reports* — is this
//! table stale, what derives from it — `recompute` *acts*: it reads the table's
//! recorded [`ProducingDescriptor`] (persisted verbatim in the
//! `.materialization.json` sidecar, not merely hashed away), reconstructs the
//! producing verb call from its typed parameters, and runs it through the
//! unmodified `BuildingTable::finish` funnel. The replay always recomputes
//! ([`CachePolicy::Bypass`]) — a recompute that reused a cache would be a no-op,
//! not a recompute — and is byte-identical when the inputs have not moved
//! (because the descriptor records every output-affecting determinant). That
//! byte identity holds on the producing host (same host and architecture) —
//! the descriptor does not record the CPU microarchitecture that ran the fold,
//! so across hosts a replay is value-equivalent up to float rounding and the
//! identity is the catalog row plus `definition_hash`, not the raw bytes.
//!
//! # The two bounded actions, and the line the engine does not cross
//!
//! - [`Cascade::ReportOnly`] (default) — recompute the **named** table only, and
//!   *report* the downstream-stale set (via the sensing layer's
//!   `derives_from_closure`); recompute none of it. The consumer decides what to
//!   do with the report.
//! - [`Cascade::Downstream`] — **one** bounded topological sweep on this single
//!   explicit request: recompute the named table, then every transitive
//!   dependent in dependency order (a parent's new digest lands before its child
//!   recomputes, so the child re-resolves against the fresh parent). No poll, no
//!   re-check, no second pass after the sweep finishes.
//!
//! This is the *last* engine surface of the recompute story. Re-running the
//! sweep on a schedule, or wiring a staleness monitor to trigger it (a
//! sensor→actuator loop), is the consumer's composition — a governing platform
//! built on a published engine version, never the engine itself. The engine
//! ships the actuator; it never ships the control loop that pulls it.
//!
//! # Per-variant dispatch
//!
//! Each [`ProducingDescriptor`] variant reconstructs its producer call from the
//! recorded typed parameters (and, for the derived producers, the table's own
//! catalog row, which carries the originating `source_id`). The intricate case
//! is [`ProducingDescriptor::ContextSet`]: its real producer is the
//! `assemble_context`→`materialize_context` **pair**, so a recompute re-pools
//! every target's context over the source's *current* rows under the recorded
//! recipe, then routes the pooled rows back through `materialize_context` (see
//! `recompute_context_set`).
//!
//! Two producers carry no `CachePolicy` dial at all — `asof_join` and
//! [`ProducingDescriptor::TrainingSet`]'s
//! [`materialize_training_set`](jammi_db::store::ResultStore::materialize_training_set),
//! which owns its own reuse probe. Their replays still always recompute, and for
//! a stated reason rather than by assumption: `asof_join` never reuses, and the
//! training-set probe matches on the `(definition, input anchors)` pair, which an
//! unpinned source anchor — the only anchor a replay of a projected source
//! relation can honestly supply — never satisfies. See `recompute_training_set`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::error::{JammiError, Result};
use jammi_db::store::manifest::{
    AsofBoundary, AsofDirection, AsofTolerance, ContextAggregator, ContextCandidateSource,
    ContextEdgeGather, GraphSampleFields, InputAnchor, ProducingDescriptor, PropagationDirection,
    PropagationOutput, PropagationWeighting, GRAPH_READ_ORDER_RULE_V1, TRAINING_SET_ORDER_RULE_V1,
};
use jammi_db::store::{CacheOutcome, CachePolicy};

use crate::fine_tune::graph_sampler::{EdgeProvenance, GraphFineTuneSources, GraphSampleConfig};
use crate::model::ModelSource;
use crate::pipeline::asof::{
    AsofJoinSpecBuilder, AsofKey, Boundary, MatchDirection, TieBreak, Tolerance,
};
use crate::pipeline::context_set::{
    ContextRequest, ContextSource, HybridMerge, MaterializedContext, SetAggregator,
};
use crate::pipeline::embedding::EmbeddingPipeline;
use crate::pipeline::graph_neighbourhood::{EdgeDirection, EdgeGather, EdgeSourceRef};
use crate::pipeline::graph_propagation::{
    PropagateRequest, PropagationOutput as AiPropagationOutput,
};
use crate::pipeline::neighbor_graph::BuildNeighborGraph;
use crate::session::InferenceSession;

/// Whether a [`recompute`](InferenceSession::recompute) also sweeps the bounded
/// downstream DAG, or only reports it. The default is [`Self::ReportOnly`]: the
/// engine never re-runs a transitive dependent unless the caller explicitly asks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Cascade {
    /// Recompute the named table only; *report* the transitive downstream-stale
    /// set without recomputing any of it. The consumer decides what to do next.
    #[default]
    ReportOnly,
    /// One bounded topological sweep on this single explicit request: recompute
    /// the named table, then every transitive dependent in dependency order. No
    /// re-check or second pass after the sweep finishes.
    Downstream,
}

/// One table a [`recompute`](InferenceSession::recompute) re-produced: the
/// original name it was recomputed *from*, the new table the replay wrote, and
/// the cache outcome of that replay (always
/// [`Computed`](CacheOutcome::Computed) — a recompute bypasses the cache).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecomputedTable {
    /// The original table name the recompute replayed.
    pub original: String,
    /// The freshly materialised table the replay wrote.
    pub recomputed: String,
    /// The cache outcome of the replay.
    pub outcome: CacheOutcome,
}

/// The outcome of a [`recompute`](InferenceSession::recompute): the tables that
/// were re-produced (one for [`Cascade::ReportOnly`]; the named table plus its
/// transitive dependents, in topological order, for [`Cascade::Downstream`]) and
/// the transitive downstream-stale set.
///
/// For [`Cascade::ReportOnly`], `downstream_stale` is *reported but not acted
/// on* — the set the consumer may choose to recompute next. For
/// [`Cascade::Downstream`], it is the same set the sweep just recomputed (so a
/// caller can confirm what the sweep covered).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecomputeReport {
    /// The tables re-produced, in the order they were recomputed.
    pub recomputed: Vec<RecomputedTable>,
    /// The transitive set of tables downstream of the named table (every table
    /// that anchors on it, directly or transitively).
    pub downstream_stale: Vec<String>,
}

impl InferenceSession {
    /// Re-invoke `table`'s recorded producer over the inputs' current state.
    ///
    /// Reads the table's recorded [`ProducingDescriptor`] and reconstructs the
    /// producing verb call from its typed parameters, running it through the
    /// unmodified `BuildingTable::finish` funnel with [`CachePolicy::Bypass`]
    /// (a recompute always recomputes). A pre-contract table (no recorded
    /// descriptor) is the typed [`JammiError::NotRecomputable`] — a loud refusal,
    /// never a re-run guessed from columns.
    ///
    /// `cascade` selects the bounded action: [`Cascade::ReportOnly`] recomputes
    /// the named table only and reports the downstream-stale set;
    /// [`Cascade::Downstream`] additionally sweeps every transitive dependent in
    /// topological order, stack-safely (see `recompute_downstream_sweep`).
    ///
    /// Tenant scope is the caller's: this resolves `table` through the
    /// tenant-filtered catalog (in [`crate::Session::recompute`]), so a peer
    /// cannot recompute a table it cannot resolve.
    pub async fn recompute(
        self: &Arc<Self>,
        table: &ResultTableRecord,
        cascade: Cascade,
    ) -> Result<RecomputeReport> {
        // The transitive downstream set is the same for both arms — reported by
        // `ReportOnly`, swept by `Downstream`. Compute it once from the named
        // table (its dependents anchor on its table name).
        let downstream_stale: Vec<String> = self
            .result_store()
            .derives_from_closure(&table.table_name)
            .await?
            .into_iter()
            .map(|edge| edge.derived)
            .collect();

        let recomputed = match cascade {
            Cascade::ReportOnly => vec![self.recompute_one(table).await?],
            Cascade::Downstream => self.recompute_downstream_sweep(table).await?,
        };

        Ok(RecomputeReport {
            recomputed,
            downstream_stale: dedup_preserving_order(downstream_stale),
        })
    }

    /// Recompute exactly one table from its recorded descriptor — the unit both
    /// cascade arms build on. Reads the descriptor, dispatches on its variant to
    /// reconstruct the producer call, and replays it with [`CachePolicy::Bypass`].
    async fn recompute_one(self: &Arc<Self>, table: &ResultTableRecord) -> Result<RecomputedTable> {
        let descriptor = self.result_store().producing_descriptor(table).await?;
        let (recomputed, outcome) = self.replay_descriptor(table, descriptor).await?;
        Ok(RecomputedTable {
            original: table.table_name.clone(),
            recomputed,
            outcome,
        })
    }

    /// Dispatch on the recorded [`ProducingDescriptor`] to reconstruct and replay
    /// the producing verb. Returns the freshly materialised table's name and the
    /// replay's [`CacheOutcome`] (always `Computed`). Every arm calls its producer
    /// with [`CachePolicy::Bypass`] so the replay genuinely recomputes.
    async fn replay_descriptor(
        self: &Arc<Self>,
        table: &ResultTableRecord,
        descriptor: ProducingDescriptor,
    ) -> Result<(String, CacheOutcome)> {
        match descriptor {
            ProducingDescriptor::Inference {
                model_id,
                task,
                source_id,
                content_columns,
                key_column,
            } => {
                let source = ModelSource::from_canonical(&model_id);
                let (_batches, outcome) = self
                    .infer(
                        &source_id,
                        &source,
                        task,
                        &content_columns,
                        &key_column,
                        CachePolicy::Bypass,
                    )
                    .await?;
                // `infer` writes a fresh source-named result table and returns the
                // rows (not the record). The replay's output is the newest `ready`
                // inference table for this `(source, task, model)` — the table this
                // producer call just promoted.
                let recomputed = self
                    .latest_ready_table_for(&source_id, task, &model_id)
                    .await?;
                Ok((recomputed, outcome))
            }
            ProducingDescriptor::Embedding {
                model_id,
                task,
                source_id,
                columns,
                key_column,
                dimensions: _,
            } => {
                let (record, outcome) =
                    EmbeddingPipeline::new(self.as_ref(), &self.result_store(), task)
                        .run(
                            &source_id,
                            &model_id,
                            &columns,
                            &key_column,
                            CachePolicy::Bypass,
                            // Recompute replays the unmodified materialization
                            // funnel directly — it is not itself a job, so it
                            // creates no `partial_result` link.
                            None,
                        )
                        .await?;
                Ok((record.table_name, outcome))
            }
            // A versioned table's replay is a full embed of the current source
            // into a NEW table (D9): value-equivalent, a new chain root.
            ProducingDescriptor::EmbeddingDelta {
                model_id,
                task,
                source_id,
                columns,
                key_column,
                ..
            }
            | ProducingDescriptor::EmbeddingCompaction {
                model_id,
                task,
                source_id,
                columns,
                key_column,
                ..
            } => {
                let (record, outcome) =
                    EmbeddingPipeline::new(self.as_ref(), &self.result_store(), task)
                        .run(
                            &source_id,
                            &model_id,
                            &columns,
                            &key_column,
                            CachePolicy::Bypass,
                            None,
                        )
                        .await?;
                Ok((record.table_name, outcome))
            }
            ProducingDescriptor::NeighborGraph {
                source_table,
                k,
                min_similarity_bits,
                mutual,
                self_exclude,
                exact,
                exact_max_rows,
                // Not a replay input — it is re-derived fresh from the source
                // table's *current* sidecar-index precision when the build
                // re-runs the index-assisted driver; it exists in the
                // descriptor purely to make that precision an output-affecting
                // determinant of the *original* run's identity.
                index_storage_precision: _,
            } => {
                let params = BuildNeighborGraph {
                    k,
                    min_similarity: min_similarity_bits.map(f32::from_bits),
                    mutual,
                    self_exclude,
                    exact,
                    exact_max_rows,
                    // Not a recorded determinant (a resolved endpoint equals its
                    // `_row_id` either way today), so it carries the build default.
                    resolve_keys: BuildNeighborGraph::default().resolve_keys,
                };
                let (record, outcome) = self
                    .build_neighbor_graph(
                        &table.source_id,
                        Some(&source_table),
                        &params,
                        CachePolicy::Bypass,
                    )
                    .await?;
                Ok((record.table_name, outcome))
            }
            ProducingDescriptor::GraphPropagation {
                source_table,
                edge_source,
                kernel_id: _,
                direction,
                hops,
                alpha_bits,
                weighting,
                output,
                dimensions: _,
            } => {
                let edge_source_ref = EdgeSourceRef::from_binding(edge_source);
                let request = PropagateRequest::new(table.source_id.clone(), edge_source_ref)
                    .with_embedding_table(source_table)
                    .with_direction(edge_direction_from_manifest(direction))
                    .with_hops(hops)
                    .with_weighting(propagation_weighting_from_manifest(weighting))
                    .with_alpha(f64::from_bits(alpha_bits))
                    .with_output(propagation_output_from_manifest(output));
                let (record, outcome) = self
                    .propagate_embeddings(&request, CachePolicy::Bypass)
                    .await?;
                Ok((record.table_name, outcome))
            }
            ProducingDescriptor::ContextSet {
                encoder_id: _,
                source_id,
                embedding_table,
                candidate_source,
                value_columns,
                aggregator,
                exclude_self,
                split,
                dimensions,
            } => {
                let recipe = context_recipe_from_manifest(
                    &source_id,
                    embedding_table,
                    candidate_source,
                    value_columns,
                    aggregator,
                    exclude_self,
                    split,
                )?;
                self.recompute_context_set(&recipe, dimensions).await
            }
            ProducingDescriptor::AsofJoin {
                spine,
                facts,
                spine_by,
                facts_by,
                spine_time,
                facts_time,
                direction,
                boundary,
                tolerance,
                tie_break_column,
                project,
            } => {
                let spec = AsofJoinSpecBuilder::new(
                    AsofKey {
                        by: spine_by,
                        time: spine_time,
                    },
                    AsofKey {
                        by: facts_by,
                        time: facts_time,
                    },
                )
                .direction(asof_direction_from_manifest(direction))
                .boundary(asof_boundary_from_manifest(boundary))
                .tolerance(tolerance.map(asof_tolerance_from_manifest))
                .tie_break(match tie_break_column {
                    Some(column) => TieBreak::ByColumnDesc(column),
                    None => TieBreak::Error,
                })
                .project(project)
                .build();
                let record = self.asof_join(&spine, &facts, &spec).await?;
                // `asof_join` carries no cache dial (it always recomputes), so the
                // replay is unconditionally a fresh `Computed`.
                Ok((record.table_name, CacheOutcome::Computed))
            }
            ProducingDescriptor::TrainingSet {
                source,
                columns,
                task,
                format,
                order_rule,
            } => {
                self.recompute_training_set(table, source, columns, task, format, order_rule)
                    .await
            }
            ProducingDescriptor::FineTune {
                training_set_definition_hash: _,
                training_set_artifact_digest: _,
                training_set_row_count: _,
                spec_canonical,
                spec_schema_version,
                base_model_id: _,
                world_size: _,
                // #500 U4b: this run's own gang topology at claim time —
                // recorded on the descriptor, but not itself an input the
                // recompute arm (K1: retrain) needs to name, since retrain
                // resubmits the SAME spec and lets a fresh claim resolve
                // its OWN worker/gang topology, never replaying the prior
                // attempt's recorded one.
                collective: _,
                local_ranks: _,
            } => {
                self.recompute_fine_tune(table, &spec_canonical, spec_schema_version)
                    .await
            }
            // An external producer is a verb the engine does not own, so there is
            // no faithful call to reconstruct — a loud refusal, never a guessed
            // re-run. Recomputing an external table is the producing consumer's
            // job; the engine only ever republishes what it is handed.
            ProducingDescriptor::External { .. } => Err(JammiError::NotRecomputable {
                table: table.table_name.clone(),
            }),
            ProducingDescriptor::GraphTrainingSet {
                node_source,
                edge_source,
                id_column,
                text_column,
                src_column,
                dst_column,
                task,
                format,
                sample,
                read_order_rule,
            } => {
                self.recompute_graph_training_set(
                    table,
                    node_source,
                    edge_source,
                    id_column,
                    text_column,
                    src_column,
                    dst_column,
                    task,
                    format,
                    sample,
                    read_order_rule,
                )
                .await
            }
        }
    }

    /// Re-invoke the training-set producer
    /// ([`jammi_db::store::ResultStore::materialize_training_set`]) over the
    /// recorded projection — the [`ProducingDescriptor::TrainingSet`] replay.
    ///
    /// Every determinant is taken from the descriptor (the source query, the
    /// projected columns, the task, the format tag) except the `source_id`
    /// lineage column, which is the recomputed table's own catalog row — the
    /// same shape the `NeighborGraph` arm uses.
    ///
    /// # Anchors re-derive from the RECORDED anchor set, not from `source_id`
    ///
    /// The original materialization may have anchored more than one relation
    /// (the graph arm anchors both its node and its edge source); `source_id`
    /// is one lineage column and can name only one of them. The replay instead
    /// reads the table's own `.materialization.json` sidecar for its recorded
    /// `input_anchors` and re-anchors every one of THOSE relation names — at a
    /// fresh, shared instant — so a replay's manifest never reports a narrower
    /// anchor set than the original materialization actually read.
    ///
    /// # Why this always recomputes
    ///
    /// The verb carries no cache dial: it owns its own reuse probe, which
    /// matches on the `(definition, input anchors)` pair. The anchors a replay
    /// can honestly supply for a projected source relation are
    /// [`AnchorKind::UnpinnedAtInstant`](jammi_db::store::manifest::AnchorKind::UnpinnedAtInstant)
    /// — a registered source exposes no version surface to pin — and an
    /// unpinned anchor never matches that probe, so the replay writes a fresh
    /// table and reports [`CacheOutcome::Computed`]. The outcome is returned as
    /// the verb reports it rather than asserted here: reuse is reported, never
    /// inferred, and a future pinned source would legitimately make the replay a
    /// hit.
    ///
    /// # A recorded PINNED anchor is re-resolved PINNED, never silently downgraded
    ///
    /// Every anchor `materialize_projection` itself records today is
    /// [`AnchorKind::UnpinnedAtInstant`] — but the recorded `input_anchors` this
    /// function reads come from the table's own `.materialization.json`
    /// sidecar, and `ProducingDescriptor::FineTune` is the first producer
    /// to anchor a `TrainingSet`-kind table by its content digest
    /// ([`AnchorKind::ResultDigest`]) rather than by an unpinned read instant.
    /// A future producer over a *versioned* source ([`AnchorKind::MutableVersion`])
    /// is the same shape. For each recorded anchor this loop therefore
    /// dispatches on its OWN kind rather than blanket-downgrading every entry
    /// to unpinned:
    ///
    /// - [`AnchorKind::UnpinnedAtInstant`] re-anchors at a fresh instant, as
    ///   before.
    /// - [`AnchorKind::MutableVersion`] / [`AnchorKind::ResultDigest`]
    ///   RE-RESOLVE against the named relation's CURRENT state
    ///   ([`crate::session::InferenceSession::result_store`]'s
    ///   [`jammi_db::store::ResultStore::pin_current_version`]) and stay
    ///   pinned — a re-resolution, never a re-use of the stale recorded value.
    ///   A relation that no longer resolves (deregistered, or the pinned
    ///   version was reaped) is [`JammiError::NotRecomputable`], naming the
    ///   anchor's source, rather than silently treated as unpinned.
    /// - [`AnchorKind::SourceVersion`] names an external/federated source's
    ///   pinned as-of value, which this engine owns no local surface to
    ///   re-verify or refresh — refused the same way, rather than fabricate a
    ///   re-resolution it cannot honestly perform.
    ///
    /// # The four refusals
    ///
    /// - An `order_rule` this build does not commit is
    ///   [`JammiError::NotRecomputable`]. The producer commits exactly
    ///   [`TRAINING_SET_ORDER_RULE_V1`](jammi_db::store::manifest::TRAINING_SET_ORDER_RULE_V1);
    ///   replaying a table committed under some other rule would write rows in
    ///   an order the recorded descriptor does not claim, which is a fabricated
    ///   re-run, not a recompute.
    /// - The `.materialization.json` sidecar this function re-reads for the
    ///   anchor set (below) has gone missing since the caller's own descriptor
    ///   read succeeded — also [`JammiError::NotRecomputable`], never a silent
    ///   "zero recorded anchors" default. The caller (`recompute`'s outer
    ///   dispatch) already reads the SAME sidecar once, through
    ///   `ResultStore::producing_descriptor`, to obtain the `source` /
    ///   `columns` / `task` / `format` / `order_rule` this function is called
    ///   with; this second, independent read is for the anchor set, and a
    ///   sidecar that vanished strictly between the two reads must refuse
    ///   here exactly as it would have refused there.
    /// - A recorded PINNED anchor whose target no longer resolves is
    ///   [`JammiError::NotRecomputable`] (see above) — never silently
    ///   downgraded to unpinned.
    /// - A recorded `source` query that no longer resolves in this session
    ///   (its relation was deregistered, or it never was a durable relation)
    ///   fails at the planner inside the verb, naming the missing relation —
    ///   the failure mode of a training set whose rows were projected from a
    ///   session-scoped relation rather than a durable registered source. No
    ///   producer in this tree names one today: `materialize_projection` (the
    ///   only [`ProducingDescriptor::TrainingSet`] producer) always reads a
    ///   durable registered source, and the graph arm samples in memory and
    ///   never writes a `TrainingSet` table at all
    ///   (<https://github.com/f-inverse/jammi-ai/issues/538> tracks giving it
    ///   a table of its own). This refusal stays because the planner error is
    ///   the honest response to ANY table whose recorded source is not
    ///   durable, not because one is expected today.
    async fn recompute_training_set(
        self: &Arc<Self>,
        table: &ResultTableRecord,
        source: String,
        columns: Vec<String>,
        task: crate::model::ModelTask,
        format: String,
        order_rule: String,
    ) -> Result<(String, CacheOutcome)> {
        if order_rule != TRAINING_SET_ORDER_RULE_V1 {
            return Err(JammiError::NotRecomputable {
                table: table.table_name.clone(),
            });
        }
        // Re-anchor from the RECORDED anchor set's relation names, never from
        // `table.source_id` alone: a graph training set's original
        // materialization anchored BOTH the node and the edge relation, and a
        // replay that only re-derived one would silently drop the other from
        // the new manifest's lineage. `table.source_id` names one relation by
        // construction (the catalog row's single lineage column); the
        // manifest's own `input_anchors` is the only record of the full set.
        let parquet_url = jammi_db::storage::StorageUrl::parse(&table.parquet_path)?;
        // A missing sidecar is not "zero anchors" — it is the same honest
        // refusal as an unrecognised order rule: without the manifest there is
        // no recorded anchor SET to re-anchor from, and silently defaulting to
        // an empty one would replay the table with NONE of its original inputs
        // recorded, which is a fabricated lineage, not a recompute.
        let manifest = self
            .result_store()
            .read_materialization_manifest(&parquet_url)
            .await?
            .ok_or_else(|| JammiError::NotRecomputable {
                table: table.table_name.clone(),
            })?;
        let recorded_anchors = manifest.input_anchors;
        let now = chrono::Utc::now().to_rfc3339();
        let mut inputs: Vec<InputAnchor> = Vec::with_capacity(recorded_anchors.len());
        for anchor in &recorded_anchors {
            inputs.push(self.reresolve_recorded_anchor(table, anchor, &now).await?);
        }
        let materialized = self
            .result_store()
            .materialize_training_set(
                self.context(),
                crate::fine_tune::training_set::training_set_spec(
                    &table.source_id,
                    &source,
                    &columns,
                    task,
                    &format,
                    inputs,
                    self.compute_device(),
                ),
            )
            .await?;
        Ok((
            materialized.table_name().to_string(),
            materialized.outcome.clone(),
        ))
    }

    /// [`ProducingDescriptor::GraphTrainingSet`] replay (GA9, issue #538):
    /// re-read the CURRENT node/edge sources and re-sample, through the SAME
    /// shared core ([`crate::fine_tune::worker::materialize_graph_training_set`])
    /// a fresh run uses — never a second, independent re-implementation of
    /// the sample-then-materialise path.
    ///
    /// Anchors both `node_source` and `edge_source` from the table's OWN
    /// recorded anchor set (`Self::recompute_training_set`'s own doc section
    /// explains why: `table.source_id` names only one relation, and this
    /// variant's original materialization anchored two) — the same
    /// re-resolution policy, dispatching on each recorded anchor's OWN kind
    /// via [`Self::reresolve_recorded_anchor`].
    ///
    /// `min_negatives` and `provenance` are not recorded on the descriptor
    /// (neither is output-affecting for a successful sample — see
    /// [`ProducingDescriptor::GraphTrainingSet`]'s own doc); the replay uses
    /// `min_negatives: 1` (the most permissive floor, which can only make a
    /// replay of an already-successful sample MORE likely to succeed, never
    /// less) and `EdgeProvenance::Declared` (never read by `GraphSampler::
    /// sample`, only by the informational `has_declared_supervision`, which
    /// this replay never calls).
    ///
    /// `task` and `format` ARE recorded (K1: `replay_descriptor`'s match
    /// names every field, never `task: _, format: _` discarding two that
    /// exist precisely so a replay can check itself) and are asserted equal
    /// to what THIS replay independently derives — `task` is always
    /// `TextEmbedding` (the graph arm's own hardcoded choice, no per-run
    /// variation), `format` is GA3's own function of `sample.hard_negatives`
    /// alone — BEFORE any read or write, refusing loudly on a mismatch
    /// rather than silently trusting today's re-derivation to still agree
    /// with what the original run recorded (the root-cause class GA3 exists
    /// to close finding its second home here: a future change to either
    /// derivation could otherwise silently diverge a replay from its
    /// original recorded meaning with no signal at all).
    #[allow(clippy::too_many_arguments)]
    async fn recompute_graph_training_set(
        self: &Arc<Self>,
        table: &ResultTableRecord,
        node_source: String,
        edge_source: String,
        id_column: String,
        text_column: String,
        src_column: String,
        dst_column: String,
        task: jammi_db::model_task::ModelTask,
        format: String,
        sample: GraphSampleFields,
        read_order_rule: String,
    ) -> Result<(String, CacheOutcome)> {
        if read_order_rule != GRAPH_READ_ORDER_RULE_V1 {
            return Err(JammiError::NotRecomputable {
                table: table.table_name.clone(),
            });
        }
        let expected_format =
            crate::fine_tune::data::TrainingFormat::in_batch(sample.hard_negatives > 0)
                .format_tag();
        if task != jammi_db::model_task::ModelTask::TextEmbedding || format != expected_format {
            return Err(JammiError::FineTune(format!(
                "table '{}': recorded task/format ({task:?}/{format}) do not match what this \
                 replay derives fresh (TextEmbedding/{expected_format}) — the graph arm's task \
                 or format derivation has changed since this table was materialised; refusing \
                 rather than silently replaying under a different meaning",
                table.table_name
            )));
        }
        let parquet_url = jammi_db::storage::StorageUrl::parse(&table.parquet_path)?;
        let manifest = self
            .result_store()
            .read_materialization_manifest(&parquet_url)
            .await?
            .ok_or_else(|| JammiError::NotRecomputable {
                table: table.table_name.clone(),
            })?;
        let recorded_anchors = manifest.input_anchors;
        let now = chrono::Utc::now().to_rfc3339();
        let mut inputs: Vec<InputAnchor> = Vec::with_capacity(recorded_anchors.len());
        for anchor in &recorded_anchors {
            inputs.push(self.reresolve_recorded_anchor(table, anchor, &now).await?);
        }

        let sources = GraphFineTuneSources {
            node_source,
            id_column,
            text_column,
            edge_source,
            src_column,
            dst_column,
            provenance: EdgeProvenance::Declared,
        };
        // Domain-validity at the decode edge (family D): `f64::from_bits`
        // reconstructs ANY bit pattern the persisted sidecar happens to
        // hold, including NaN/Infinity, and `GraphSampleConfig::validate`'s
        // `<= 0.0` checks do not catch that (`NaN <= 0.0` is `false`, so a
        // NaN return_p/in_out_q would silently pass validation and reach
        // `biased_choice`'s `1.0 / return_p` weight computation as a
        // confidently-wrong, not a refused, number). Refused here, typed,
        // by name, before `GraphSampleConfig` is even constructed.
        let return_p = f64::from_bits(sample.return_p_bits);
        let in_out_q = f64::from_bits(sample.in_out_q_bits);
        if !return_p.is_finite() || !in_out_q.is_finite() {
            return Err(JammiError::FineTune(format!(
                "table '{}': recorded return_p/in_out_q decode to a non-finite value \
                 ({return_p}/{in_out_q}) — the sidecar's sample fields are corrupt or the bits \
                 were never a valid f64 to begin with; refusing rather than sampling under an \
                 undefined node2vec bias",
                table.table_name
            )));
        }
        let sample_config = GraphSampleConfig {
            walk_length: sample.walk_length as usize,
            walks_per_node: sample.walks_per_node as usize,
            return_p,
            in_out_q,
            hard_negatives: sample.hard_negatives as usize,
            exclude_hops: sample.exclude_hops as usize,
            min_negatives: 1,
            seed: sample.seed,
        };
        let materialized = crate::fine_tune::worker::materialize_graph_training_set(
            self,
            "recompute",
            &sources,
            sample_config,
            inputs,
        )
        .await?;
        Ok((
            materialized.table_name().to_string(),
            materialized.outcome.clone(),
        ))
    }

    /// Re-resolve ONE recorded anchor for [`Self::recompute_training_set`],
    /// dispatching on its own [`AnchorKind`] — see that method's "A recorded
    /// PINNED anchor is re-resolved PINNED" doc section for the policy this
    /// implements.
    async fn reresolve_recorded_anchor(
        self: &Arc<Self>,
        table: &ResultTableRecord,
        anchor: &InputAnchor,
        now: &str,
    ) -> Result<InputAnchor> {
        use jammi_db::store::manifest::AnchorKind;
        match anchor.kind {
            AnchorKind::UnpinnedAtInstant => {
                Ok(InputAnchor::unpinned_at_instant(anchor.source.clone(), now))
            }
            AnchorKind::MutableVersion | AnchorKind::ResultDigest => {
                let current = self
                    .catalog()
                    .get_result_table(&anchor.source)
                    .await?
                    .ok_or_else(|| JammiError::NotRecomputable {
                        table: format!(
                            "{} (pinned input anchor '{}' no longer resolves: the relation is \
                             gone)",
                            table.table_name, anchor.source
                        ),
                    })?;
                self.result_store()
                    .pin_current_version(current)
                    .await
                    .map(|pinned| pinned.input_anchor())
            }
            // An external/federated source's pinned as-of value has no local
            // surface this engine can re-verify or refresh — refusing rather
            // than fabricating a re-resolution it cannot honestly perform.
            AnchorKind::SourceVersion => Err(JammiError::NotRecomputable {
                table: format!(
                    "{} (pinned input anchor '{}' is an external source_version this engine \
                     cannot re-resolve)",
                    table.table_name, anchor.source
                ),
            }),
        }
    }

    /// Re-invoke a `TrainingSpec::FineTune` job — the
    /// [`ProducingDescriptor::FineTune`] replay, which K1 fixes as **retrain**,
    /// never a re-derivation from the recorded fields: `spec_canonical` +
    /// `spec_schema_version` decode back into the exact
    /// [`crate::fine_tune::spec::TrainingSpec::FineTune`] the original job ran
    /// ([`crate::fine_tune::spec::fine_tune_spec_from_canonical`]), which is
    /// then submitted and driven to completion the same way any fresh
    /// fine-tune submission is — over the source's *current* rows, under a
    /// fresh training-set materialization, so a replay reflects the inputs'
    /// present state exactly like every other producer this module replays.
    /// `CachePolicy::Bypass` (baked into the decode) means a replay never
    /// short-circuits into the model-level reuse probe: a recompute that
    /// reused a cache would be a no-op, not a recompute (module doc).
    ///
    /// `table` names the row whose descriptor was read — used only for the
    /// refusal message, since `pipeline::recompute` operates over
    /// `result_tables` rows exclusively (`ResultTableRecord`) while a
    /// fine-tuned model's descriptor lives on its `models` row instead. No
    /// production caller of [`InferenceSession::recompute`] can therefore
    /// hand this arm a `ProducingDescriptor::FineTune` today; the arm exists
    /// so the match stays exhaustive (K7) and so a future model-level
    /// recompute surface, should one land, replays this variant correctly
    /// from day one rather than needing this logic written under pressure
    /// then.
    async fn recompute_fine_tune(
        self: &Arc<Self>,
        table: &ResultTableRecord,
        spec_canonical: &str,
        spec_schema_version: u32,
    ) -> Result<(String, CacheOutcome)> {
        let spec = crate::fine_tune::spec::fine_tune_spec_from_canonical(
            spec_canonical,
            spec_schema_version,
        )
        .map_err(|e| JammiError::NotRecomputable {
            table: format!(
                "{} (undecodable fine-tune spec_canonical: {e})",
                table.table_name
            ),
        })?;
        let job = self.run_training_spec(spec).await?;
        job.wait().await?;
        Ok((job.model_id.clone(), CacheOutcome::Computed))
    }

    /// Re-invoke the `assemble_context`→`materialize_context` **pair** — the real
    /// ContextSet producer — over the source's *current* rows under the recorded
    /// recipe.
    ///
    /// `materialize_context` is a sink that receives pre-pooled rows, so the
    /// determinant is the `assemble_context` recipe (the [`ContextRequest`]), not
    /// the sink. A recompute therefore re-pools every target's context: it reads
    /// every `(_row_id, vector)` of the source's current embedding table, builds
    /// one [`ContextRequest`] per target (the recipe, with that target's own
    /// vector as the `query` and its `_row_id` as the `exclude_key` so the
    /// leakage guard drops the target's own row), assembles + pools each, then
    /// routes the pooled rows back through `materialize_context`. The targets are
    /// the source's *current* rows — the point of a recompute is to re-pool over
    /// the inputs' present state.
    async fn recompute_context_set(
        self: &Arc<Self>,
        recipe: &ContextRequest,
        dimensions: usize,
    ) -> Result<(String, CacheOutcome)> {
        let table = self
            .catalog()
            .resolve_embedding_table(&recipe.source_id, recipe.embedding_table.as_deref())
            .await?;
        // ONE resolution of the source table's current version (M1), shared
        // by every target in the loop below: `read_target_rows` and every
        // `assemble_context_pinned` call read the SAME version, so a
        // version publish racing this recompute can never straddle across
        // targets, and the schema/mask memo (M3) hits for every target
        // after the first.
        let pin = self.result_store().pin_current_version(table).await?;
        let targets = self.read_target_rows(&pin).await?;

        let mut rows: Vec<(String, Vec<f32>)> = Vec::new();
        for (row_id, query) in targets {
            let mut request = recipe.clone();
            request.query = query;
            request.exclude_key = Some(row_id.clone());
            let representation = self.assemble_context_pinned(&request, &pin).await?;
            // A degenerate context (no neighbour survived exclusion/split) has no
            // pooled vector. The original materialisation could only have
            // recorded a row for a target whose context was non-empty, so a
            // recompute likewise skips a now-empty target rather than fabricate a
            // zero vector.
            if let Some(vector) = representation.context_vector {
                rows.push((row_id, vector));
            }
        }

        let (record, outcome) = self
            .materialize_context(
                MaterializedContext {
                    rows: &rows,
                    dimensions,
                    recipe,
                    // A replay's targets are `read_target_rows` of the resolved
                    // table, so every key here IS one of that table's `_row_id`
                    // values — the one caller position that can honestly claim
                    // the table's own origin column for its targets.
                    key_column: pin.record().key_column.as_deref(),
                },
                CachePolicy::Bypass,
            )
            .await?;
        Ok((record.table_name, outcome))
    }

    /// Read every `(_row_id, vector)` of an embedding table into owned rows — the
    /// targets a ContextSet recompute re-pools over. Reads through
    /// [`jammi_db::store::ResultStore::pinned_provider`] — the SAME
    /// resolution every per-target `assemble_context_pinned` call in the
    /// loop above reads from, never a second, independent read of
    /// `current_version` — so a recompute pools over exactly one snapshot of
    /// the source's current rows, never a session-stale prior version and
    /// never a version straddled mid-loop.
    async fn read_target_rows(
        &self,
        pin: &jammi_db::store::PinnedSource,
    ) -> Result<Vec<(String, Vec<f32>)>> {
        let table = pin.record();
        let ctx = self.context();
        let provider = self.result_store().pinned_provider(ctx, pin).await?;
        let batches = ctx
            .read_table(provider)
            .map_err(JammiError::from)?
            .select_columns(&["_row_id", "vector"])
            .map_err(JammiError::from)?
            .collect()
            .await
            .map_err(JammiError::from)?;

        let mut rows = Vec::new();
        for batch in &batches {
            let row_ids = read_row_id_column(batch, &table.table_name)?;
            let mut vectors: Vec<Vec<f32>> = Vec::new();
            jammi_db::store::vectors::extend_with_fixed_size_list_f32(
                batch,
                &table.table_name,
                "vector",
                &mut vectors,
            )?;
            for (row_id, vector) in row_ids.into_iter().zip(vectors) {
                rows.push((row_id, vector));
            }
        }
        Ok(rows)
    }

    /// Resolve the newest `ready` result table for a `(source_id, task,
    /// model_id)` — the table an inference replay just promoted. Inference writes
    /// a fresh source-named table on every run, so the recompute names its output
    /// by the latest ready table the producer wrote.
    ///
    /// `find_result_tables` returns the tenant-scoped matches ordered by
    /// `created_at`; the replay's output is the newest `ready` one (the just-
    /// promoted table). Tenant scope is inherited from the caller, so this never
    /// resolves a peer's table.
    async fn latest_ready_table_for(
        &self,
        source_id: &str,
        task: crate::model::ModelTask,
        model_id: &str,
    ) -> Result<String> {
        self.catalog()
            .find_result_tables(source_id, Some(task), Some(model_id))
            .await?
            .into_iter()
            .rfind(|record| record.status == "ready")
            .map(|record| record.table_name)
            .ok_or_else(|| {
                JammiError::Other(format!(
                    "recompute: no ready table found for source '{source_id}' model '{model_id}' \
                     after replay"
                ))
            })
    }

    /// One bounded **topological** sweep of the named table and every transitive
    /// dependent — the [`Cascade::Downstream`] body.
    ///
    /// The sweep recomputes a parent before any child that anchors on it, so the
    /// child senses the parent's new digest and replays over fresh inputs. The
    /// ordering is a stack-safe iterative Kahn-style topological sort over the
    /// `derives_from` edges restricted to the named table's transitive closure:
    /// the work-stack and the in-degree map are explicit (no recursion), so an
    /// arbitrarily deep lineage chain can never blow the Rust call stack. A
    /// diamond (two parents → one shared child) recomputes the child exactly once,
    /// after both parents; a cycle in the recorded lineage is the typed
    /// [`JammiError::DependencyCycle`] (a materialization lineage is a DAG by
    /// construction, so a cycle is corruption, not a caller condition — the same
    /// well-foundedness `derives_from_closure` relies on).
    async fn recompute_downstream_sweep(
        self: &Arc<Self>,
        root: &ResultTableRecord,
    ) -> Result<Vec<RecomputedTable>> {
        let order = self.topological_recompute_order(root).await?;

        let mut recomputed = Vec::with_capacity(order.len());
        for table_name in order {
            // Re-resolve each node freshly so a child reads its parent's *new*
            // digest (the parent was recomputed earlier in this loop). The root
            // is `root` itself; a dependent is resolved by name through the same
            // tenant-scoped catalog.
            let record = if table_name == root.table_name {
                root.clone()
            } else {
                self.catalog()
                    .get_result_table(&table_name)
                    .await?
                    .ok_or_else(|| {
                        JammiError::Catalog(format!(
                            "recompute sweep: dependent table '{table_name}' vanished mid-sweep"
                        ))
                    })?
            };
            recomputed.push(self.recompute_one(&record).await?);
        }
        Ok(recomputed)
    }

    /// The topological recompute order of `root` and its transitive dependents —
    /// every node appears after all of its in-closure parents. Stack-safe: an
    /// explicit Kahn queue over the closure's edges, never recursion. Raises
    /// [`JammiError::DependencyCycle`] if the recorded lineage is not a DAG (a
    /// node that never reaches in-degree zero closes a cycle).
    async fn topological_recompute_order(&self, root: &ResultTableRecord) -> Result<Vec<String>> {
        // The transitive closure's edges (parent → child). `derives_from_closure`
        // is itself stack-safe and raises `DependencyCycle` on a back-edge, so the
        // edge set it returns is already DAG-shaped over reachable nodes; the Kahn
        // pass below orders them and re-confirms acyclicity defensively.
        let edges = self
            .result_store()
            .derives_from_closure(&root.table_name)
            .await?;

        // Node set = root ∪ every endpoint of every edge.
        let mut nodes: HashSet<String> = HashSet::new();
        nodes.insert(root.table_name.clone());
        let mut children: HashMap<String, Vec<String>> = HashMap::new();
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        in_degree.insert(root.table_name.clone(), 0);

        for edge in &edges {
            // Only edges whose parent is inside the closure matter: an edge from a
            // node outside the root's subgraph (a sibling parent of a shared
            // child) would otherwise inflate the child's in-degree and wedge the
            // Kahn pass below. The closure is rooted at `root`, so its reachable
            // node set is `nodes`; an edge is in-closure iff its parent is reached.
            nodes.insert(edge.input.clone());
            nodes.insert(edge.derived.clone());
            in_degree.entry(edge.input.clone()).or_insert(0);
            *in_degree.entry(edge.derived.clone()).or_insert(0) += 1;
            children
                .entry(edge.input.clone())
                .or_default()
                .push(edge.derived.clone());
        }

        // Kahn's algorithm with an explicit queue — no recursion.
        let mut queue: Vec<String> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(node, _)| node.clone())
            .collect();
        // Deterministic order among same-in-degree nodes keeps the sweep
        // reproducible (the parent of a diamond is processed before exploring its
        // children regardless of HashMap iteration order).
        queue.sort();

        let mut order: Vec<String> = Vec::with_capacity(nodes.len());
        let mut head = 0;
        while head < queue.len() {
            let node = queue[head].clone();
            head += 1;
            order.push(node.clone());
            if let Some(kids) = children.get(&node) {
                let mut ready: Vec<String> = Vec::new();
                for child in kids {
                    let deg = in_degree
                        .get_mut(child)
                        .expect("every edge endpoint has an in-degree entry");
                    *deg -= 1;
                    if *deg == 0 {
                        ready.push(child.clone());
                    }
                }
                ready.sort();
                queue.extend(ready);
            }
        }

        if order.len() != nodes.len() {
            // A node never reached in-degree zero → it is in a cycle. Name the
            // first such node (a stable, deterministic pick) as the cycle table.
            let mut remaining: Vec<String> =
                nodes.into_iter().filter(|n| !order.contains(n)).collect();
            remaining.sort();
            let table = remaining
                .into_iter()
                .next()
                .expect("len mismatch implies a remaining node");
            return Err(JammiError::DependencyCycle { table });
        }

        Ok(order)
    }
}

/// Reconstruct the [`ContextRequest`] recipe from the recorded `ContextSet`
/// descriptor — the recipe every target was pooled under. The per-target `query`
/// and `exclude_key` are *not* part of the recipe (they vary per target and
/// become the output's row keys); they are filled in per target during the
/// recompute over the source's current rows.
fn context_recipe_from_manifest(
    source_id: &str,
    embedding_table: Option<String>,
    candidate_source: ContextCandidateSource,
    value_columns: Vec<String>,
    aggregator: ContextAggregator,
    exclude_self: bool,
    split: Option<String>,
) -> Result<ContextRequest> {
    // `ContextRequest::new` seeds the leakage-safe defaults; the recorded recipe
    // overrides every determinant the descriptor carries. `query`/`exclude_key`
    // are intentionally left at their `new` defaults — they are per-target inputs.
    let mut request = ContextRequest::new(source_id, Vec::new(), 0);
    request.embedding_table = embedding_table;
    request.source = context_source_from_manifest(candidate_source);
    request.value_columns = value_columns;
    request.aggregator = set_aggregator_from_manifest(aggregator);
    request.exclude_self = exclude_self;
    request.split = split;
    Ok(request)
}

/// Map the manifest's candidate-source mirror back onto the AI-crate
/// [`ContextSource`] — the reverse of `candidate_source_for`.
fn context_source_from_manifest(source: ContextCandidateSource) -> ContextSource {
    match source {
        ContextCandidateSource::Ann { k } => ContextSource::Ann { k },
        ContextCandidateSource::Edges { gather } => {
            ContextSource::Edges(edge_gather_from_manifest(gather))
        }
        ContextCandidateSource::Hybrid {
            ann_k,
            gather,
            merge: _,
        } => ContextSource::Hybrid {
            ann_k,
            edges: edge_gather_from_manifest(gather),
            // Only `Union` exists today; the manifest mirror carries it for
            // forward-completeness, so the reverse is the single AI variant.
            merge: HybridMerge::Union,
        },
    }
}

/// Map the manifest's edge-gather mirror back onto the AI-crate [`EdgeGather`] —
/// the reverse of `edge_gather_for`. `hops` is the recorded *effective* depth, so
/// the gather's `hop_cap` is set to it (the depth is already clamped) and the
/// builder default cap is irrelevant to the replay.
fn edge_gather_from_manifest(gather: ContextEdgeGather) -> EdgeGather {
    let mut rebuilt = EdgeGather::new(EdgeSourceRef::from_binding(gather.edge_source));
    rebuilt.hops = gather.hops;
    rebuilt.fanout = gather.fanout;
    rebuilt.direction = edge_direction_from_manifest(gather.direction);
    rebuilt.edge_types = gather.edge_types;
    rebuilt.min_weight = gather.min_weight_bits.map(f64::from_bits);
    rebuilt.as_of = gather.as_of;
    // The recorded depth is already the effective (post-clamp) value, so pin the
    // cap to it: a replay must not re-clamp a faithfully-recorded depth.
    rebuilt.hop_cap = gather.hops.max(1);
    rebuilt
}

/// Map the manifest [`ContextAggregator`] mirror back onto the AI-crate
/// [`SetAggregator`].
fn set_aggregator_from_manifest(aggregator: ContextAggregator) -> SetAggregator {
    match aggregator {
        ContextAggregator::Mean => SetAggregator::Mean,
        ContextAggregator::Sum => SetAggregator::Sum,
        ContextAggregator::Max => SetAggregator::Max,
    }
}

/// Map the manifest [`PropagationDirection`] mirror back onto the AI-crate
/// [`EdgeDirection`] (shared by propagation and the context edge gather).
fn edge_direction_from_manifest(direction: PropagationDirection) -> EdgeDirection {
    match direction {
        PropagationDirection::Out => EdgeDirection::Out,
        PropagationDirection::In => EdgeDirection::In,
        PropagationDirection::Undirected => EdgeDirection::Undirected,
    }
}

/// Map the manifest [`PropagationWeighting`] mirror back onto the AI-crate
/// `PropagationWeighting`.
fn propagation_weighting_from_manifest(
    weighting: PropagationWeighting,
) -> crate::pipeline::graph_propagation::PropagationWeighting {
    use crate::pipeline::graph_propagation::PropagationWeighting as Ai;
    match weighting {
        PropagationWeighting::Uniform => Ai::Uniform,
        PropagationWeighting::DegreeNormalized => Ai::DegreeNormalized,
        PropagationWeighting::EdgeSimilarity => Ai::EdgeSimilarity,
    }
}

/// Map the manifest [`PropagationOutput`] mirror back onto the AI-crate
/// `PropagationOutput`.
fn propagation_output_from_manifest(output: PropagationOutput) -> AiPropagationOutput {
    match output {
        PropagationOutput::Final => AiPropagationOutput::Final,
        PropagationOutput::JumpingKnowledge => AiPropagationOutput::JumpingKnowledge,
    }
}

/// Map the manifest [`AsofDirection`] mirror back onto the AI-crate
/// [`MatchDirection`].
fn asof_direction_from_manifest(direction: AsofDirection) -> MatchDirection {
    match direction {
        AsofDirection::Backward => MatchDirection::Backward,
        AsofDirection::Forward => MatchDirection::Forward,
        AsofDirection::Nearest => MatchDirection::Nearest,
    }
}

/// Map the manifest [`AsofBoundary`] mirror back onto the AI-crate [`Boundary`].
fn asof_boundary_from_manifest(boundary: AsofBoundary) -> Boundary {
    match boundary {
        AsofBoundary::Inclusive => Boundary::Inclusive,
        AsofBoundary::Exclusive => Boundary::Exclusive,
    }
}

/// Map the manifest [`AsofTolerance`] mirror back onto the AI-crate [`Tolerance`].
fn asof_tolerance_from_manifest(tolerance: AsofTolerance) -> Tolerance {
    match tolerance {
        AsofTolerance::Duration(d) => Tolerance::Duration(d),
        AsofTolerance::Steps(s) => Tolerance::Steps(s),
    }
}

/// Drop duplicate table names while preserving first-seen order — the downstream
/// closure can name a diamond descendant once per parent edge, but the reported
/// set is a set.
fn dedup_preserving_order(names: Vec<String>) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    names
        .into_iter()
        .filter(|n| seen.insert(n.clone()))
        .collect()
}

/// Read the `_row_id` column of an embedding batch into owned strings, casting
/// from whatever Arrow string family the Parquet scan returns. Mirrors the
/// neighbor-graph / propagation readers; the context recompute shares the shape.
fn read_row_id_column(batch: &arrow::array::RecordBatch, table: &str) -> Result<Vec<String>> {
    use arrow::array::Array;
    let col = batch
        .column_by_name("_row_id")
        .ok_or_else(|| JammiError::Other(format!("table '{table}' has no _row_id column")))?;
    let casted = arrow::compute::cast(col, &arrow::datatypes::DataType::Utf8)
        .map_err(|e| JammiError::Other(format!("cast _row_id of '{table}' to Utf8: {e}")))?;
    let strings = casted
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .ok_or_else(|| JammiError::Other(format!("_row_id of '{table}' is not a string column")))?;
    Ok((0..strings.len())
        .map(|i| strings.value(i).to_string())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    use jammi_db::source::{FileFormat, SourceConnection, SourceType};

    use crate::model::ModelTask;
    use crate::session::InferenceSession;

    /// The function-level exercise of the race `recompute_training_set`'s own
    /// "The four refusals" doc section names: `recompute`'s outer dispatch
    /// reads the descriptor through `ResultStore::producing_descriptor` (the
    /// "descriptor read"), THEN `recompute_training_set` reads the SAME
    /// sidecar again for the anchor set (the "anchor read"). A sidecar that
    /// disappears in that window cannot be constructed by driving the public
    /// `recompute` entry
    /// point end to end: `producing_descriptor` already refuses with
    /// `NotRecomputable` if the sidecar is absent at ITS read, so an
    /// end-to-end test that deletes the sidecar up front only ever exercises
    /// that earlier, already-correct guard and never reaches this function's
    /// own read at all (confirmed: with this function's fix reverted to
    /// `.map(..).unwrap_or_default()`, an end-to-end `Session::recompute` test
    /// against a table whose sidecar was deleted before the call still returns
    /// `NotRecomputable`, unchanged — the outer guard, not this one, is what it
    /// observes). This test instead calls `recompute_training_set` directly,
    /// with the SAME `source`/`columns`/`task`/`format`/`order_rule` values the
    /// outer dispatch would have destructured from a successful descriptor
    /// read (not routed through `producing_descriptor` itself here, so the
    /// call does not add a second in-tree caller for
    /// `pinned_source_gate.rs`'s machine-checked `PRODUCING_DESCRIPTOR_CALLERS`
    /// to enumerate) — exactly reproducing the state this function sees when
    /// the sidecar vanishes strictly between the two reads. `order_rule` is
    /// the only one of the five this function reads before the anchor read
    /// (the earlier guard at the top of the function), so it is the only one
    /// that has to be the real committed value; the rest are inert on this
    /// path — the function returns before ever using them.
    #[tokio::test(flavor = "multi_thread")]
    async fn recompute_training_set_refuses_when_its_own_manifest_read_finds_no_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let session = Arc::new(
            InferenceSession::new(jammi_test_utils::test_config(dir.path()))
                .await
                .unwrap(),
        );
        let csv = dir.path().join("pairs.csv");
        std::fs::write(&csv, "text_a,text_b,score\nhello,world,1.0\n").unwrap();
        session
            .add_source(
                "training",
                SourceType::File,
                SourceConnection {
                    url: Some(format!("file://{}", csv.display())),
                    format: Some(FileFormat::Csv),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let columns = vec![
            "text_a".to_string(),
            "text_b".to_string(),
            "score".to_string(),
        ];
        // Table only — this test never reads the rows back, so it routes
        // through the table-only form (#500 U2c c3b) rather than discarding
        // an eager `Vec<RecordBatch>` collect it never needed.
        let table = crate::fine_tune::training_set::materialize_projection_table(
            &session,
            "training",
            &columns,
            ModelTask::TextEmbedding,
            "contrastive",
        )
        .await
        .unwrap();

        // Stand-ins for the fields the outer dispatch would have destructured
        // from a successful `producing_descriptor` read. Only `order_rule`
        // has to be the real committed value (`TRAINING_SET_ORDER_RULE_V1`,
        // checked at the top of `recompute_training_set` before the anchor
        // read this test targets); the rest never reach a use on this path —
        // the function returns at the anchor read, before ever running
        // `source` as SQL.
        let source = "SELECT \"text_a\", \"text_b\", \"score\" FROM stand_in".to_string();
        let format = "contrastive".to_string();
        let order_rule = TRAINING_SET_ORDER_RULE_V1.to_string();

        // The window: the sidecar vanishes strictly between the descriptor
        // read (elided above — see the doc comment) and the anchor read
        // `recompute_training_set` is about to make.
        let url = jammi_db::storage::StorageUrl::parse(table.parquet_path()).unwrap();
        let handle = session.result_store().open_parquet(&url).unwrap();
        let sidecar = handle.sibling_path("materialization.json").unwrap();
        assert_eq!(
            handle.delete_if_exists(&sidecar).await.unwrap(),
            jammi_db::storage::DeleteOutcome::Deleted,
            "the sidecar must actually be removed for the window to be real"
        );

        // The whole `ResultTableRecord` `recompute_training_set` takes is
        // fetched through the catalog (#551) — never through
        // `TrainingSetTable`, which carries no whole-row accessor.
        let record = session
            .catalog()
            .get_result_table(table.table_name())
            .await
            .unwrap()
            .expect("the producer promoted a catalog row");
        let err = session
            .recompute_training_set(
                &record,
                source,
                columns,
                ModelTask::TextEmbedding,
                format,
                order_rule,
            )
            .await
            .expect_err(
                "a sidecar missing at the anchor read must refuse, never replay with \
                 zero recorded anchors",
            );
        match err {
            JammiError::NotRecomputable { table: named } => {
                assert_eq!(named, table.table_name());
            }
            other => panic!("expected NotRecomputable, got {other:?}"),
        }
    }
}
