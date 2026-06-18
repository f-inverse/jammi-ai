//! The **action** half of incremental recompute: re-invoke a result table's
//! recorded producer over the inputs' *current* state.
//!
//! Where the sensing layer ([`jammi_db::store::freshness`]) *reports* — is this
//! table stale, what derives from it — `recompute` *acts*: it reads the table's
//! recorded [`ProducingDescriptor`] (persisted verbatim in the
//! `.materialization.json` sidecar, not merely hashed away), reconstructs the
//! producing verb call from its typed parameters, and runs it through the
//! unmodified `finalize_with_manifest` funnel. The replay always recomputes
//! ([`CachePolicy::Bypass`]) — a recompute that reused a cache would be a no-op,
//! not a recompute — and is byte-identical when the inputs have not moved
//! (because the descriptor records every output-affecting determinant).
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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::error::{JammiError, Result};
use jammi_db::store::manifest::{
    AnchorKind, AsofBoundary, AsofDirection, AsofTolerance, ContextAggregator,
    ContextCandidateSource, ContextEdgeGather, ContextEdgeSource, InputAnchor, ProducingDescriptor,
    PropagationDirection, PropagationOutput, PropagationWeighting,
};
use jammi_db::store::{CacheOutcome, CachePolicy};

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
    /// unmodified `finalize_with_manifest` funnel with [`CachePolicy::Bypass`]
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
                let edge_source_ref = self.edge_source_ref_for_replay(table, &edge_source).await?;
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
        }
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
        let targets = self.read_target_rows(&table).await?;

        let mut rows: Vec<(String, Vec<f32>)> = Vec::new();
        for (row_id, query) in targets {
            let mut request = recipe.clone();
            request.query = query;
            request.exclude_key = Some(row_id.clone());
            let representation = self.assemble_context(&request).await?;
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
                },
                CachePolicy::Bypass,
            )
            .await?;
        Ok((record.table_name, outcome))
    }

    /// Read every `(_row_id, vector)` of an embedding table into owned rows — the
    /// targets a ContextSet recompute re-pools over. Reuses the registered
    /// DataFusion table so tenant scope and cloud credentials are inherited.
    async fn read_target_rows(&self, table: &ResultTableRecord) -> Result<Vec<(String, Vec<f32>)>> {
        let batches = self
            .sql(&format!(
                "SELECT _row_id, vector FROM \"jammi.{}\"",
                table.table_name
            ))
            .await?;

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

    /// Reconstruct the [`EdgeSourceRef`] a graph propagation read, for replay. The
    /// `GraphPropagation` descriptor records the edge relation by id; the kind of
    /// the recorded edge **input anchor** disambiguates the two edge-source
    /// shapes: an immutable `neighbor_graph` table anchors as
    /// [`AnchorKind::ResultDigest`], while a registered external edge source
    /// anchors as [`AnchorKind::UnpinnedAtInstant`].
    ///
    /// A registered edge source's column bindings are not part of the propagation
    /// descriptor (only its id is), so they are reconstructed from the engine's
    /// single-source defaults (`src`/`dst`, no type/weight/as-of) — the same
    /// defaults the wire decode applies when a remote propagate omits them.
    async fn edge_source_ref_for_replay(
        &self,
        table: &ResultTableRecord,
        edge_source: &str,
    ) -> Result<EdgeSourceRef> {
        let recorded = self.recorded_edge_anchor_kind(table, edge_source).await?;
        Ok(match recorded {
            Some(AnchorKind::ResultDigest) => EdgeSourceRef::NeighborGraph {
                table_name: edge_source.to_string(),
            },
            // An unpinned (or, defensively, any non-result-digest) edge anchor is
            // a registered external edge source; its column bindings carry the
            // engine defaults.
            _ => EdgeSourceRef::Registered {
                source_id: edge_source.to_string(),
                src_column: "src".to_string(),
                dst_column: "dst".to_string(),
                type_column: None,
                weight_column: None,
                as_of_column: None,
            },
        })
    }

    /// The recorded anchor kind of the propagation's edge-source input, read from
    /// the table's manifest `input_anchors`. `None` when the edge source is not
    /// among the recorded anchors (it should always be, for a propagation).
    async fn recorded_edge_anchor_kind(
        &self,
        table: &ResultTableRecord,
        edge_source: &str,
    ) -> Result<Option<AnchorKind>> {
        let Some(ref anchors_json) = table.input_anchors_json else {
            return Ok(None);
        };
        let anchors: Vec<InputAnchor> = serde_json::from_str(anchors_json)
            .map_err(|e| JammiError::Other(format!("decode input anchors: {e}")))?;
        Ok(anchors
            .into_iter()
            .find(|a| a.source == edge_source)
            .map(|a| a.kind))
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
            .filter(|record| record.status == "ready")
            .next_back()
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
    let mut rebuilt = EdgeGather::new(edge_source_ref_from_manifest(gather.edge_source));
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

/// Map the manifest's edge-source mirror back onto the AI-crate [`EdgeSourceRef`]
/// — the reverse of `edge_source_for`. The `ContextSet` descriptor records the
/// full registered-source column bindings, so this reconstruction is lossless
/// (unlike the propagation edge source, whose columns are not in its descriptor).
fn edge_source_ref_from_manifest(source: ContextEdgeSource) -> EdgeSourceRef {
    match source {
        ContextEdgeSource::NeighborGraph { table_name } => {
            EdgeSourceRef::NeighborGraph { table_name }
        }
        ContextEdgeSource::Registered {
            source_id,
            src_column,
            dst_column,
            type_column,
            weight_column,
            as_of_column,
        } => EdgeSourceRef::Registered {
            source_id,
            src_column,
            dst_column,
            type_column,
            weight_column,
            as_of_column,
        },
    }
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
