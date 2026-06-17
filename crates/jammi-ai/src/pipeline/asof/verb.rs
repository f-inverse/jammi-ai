//! The `asof_join` verb body: resolve two registered relations to physical
//! scans, plan the [`AsofJoinExec`] (with the ordering its merge requires), run
//! it, and publish the result through the single materialization funnel.
//!
//! The verb owns the *lifecycle* (resolve → plan → run → write → attest); the
//! operator owns the *algorithm*. It writes through
//! [`ResultStore::finalize_with_manifest`](jammi_db::store::ResultStore), so an
//! as-of result table carries the same verifiable manifest every other producer
//! does: a typed [`ProducingDescriptor::AsofJoin`] over the join's parameters,
//! and an input anchor for BOTH relations.

use std::sync::Arc;

use arrow::compute::SortOptions;
use datafusion::physical_expr::{expressions::col, LexOrdering, PhysicalSortExpr};
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::ExecutionPlan;
use jammi_db::catalog::result_repo::{ResultTableKind, ResultTableRecord};
use jammi_db::error::{JammiError, Result};
use jammi_db::store::manifest::{
    AsofBoundary, AsofDirection, AsofTolerance, InputAnchor, Materialization, MaterializationEnv,
    ProducingDescriptor,
};
use jammi_db::ModelTask;

use super::exec::AsofJoinExec;
use super::spec::{AsofJoinSpec, Boundary, MatchDirection, TieBreak, Tolerance};
use crate::session::InferenceSession;

/// The provenance id recorded in the result table's `model_id` column — an
/// as-of join invokes no model, but the column is NOT NULL, so a stable sentinel
/// rides it (mirroring the neighbor-graph derivation's sentinel).
const ASOF_JOIN_MODEL_ID: &str = "asof-join";

/// Run an as-of temporal join of two registered relations and materialise the
/// point-in-time-correct result table. Resolves `spine` and `facts` through the
/// session's tenant-scoped catalog, sorts each by (`by`..., `time`[, tie-break]),
/// merges via [`AsofJoinExec`], and finalises through the materialization funnel.
pub async fn run(
    session: &InferenceSession,
    spine: &str,
    facts: &str,
    spec: &AsofJoinSpec,
) -> Result<ResultTableRecord> {
    let spine_plan = scan_relation(session, spine).await?;
    let facts_plan = scan_relation(session, facts).await?;

    // Each side sorted into the merge's required order; an empty `by` is a single
    // global group, which the temporal sort alone orders. The tie-break column
    // lives only on the facts side, so only the right sort carries it.
    let spine_sorted = sort_for_merge(spine_plan, &spec.left.by, &spec.left.time, None)?;
    let facts_sorted = sort_for_merge(
        facts_plan,
        &spec.right.by,
        &spec.right.time,
        tie_break_column(&spec.tie_break),
    )?;

    let exec = AsofJoinExec::try_new(spine_sorted, facts_sorted, spec.clone())
        .map_err(|e| JammiError::Other(format!("asof_join planning failed: {e}")))?;
    let exec: Arc<dyn ExecutionPlan> = Arc::new(exec);

    let task_ctx = session.context().task_ctx();
    let stream = exec
        .execute(0, task_ctx)
        .map_err(|e| JammiError::Other(format!("AsofJoinExec failed: {e}")))?;
    let batches = datafusion::physical_plan::common::collect(stream)
        .await
        .map_err(|e| JammiError::Other(format!("asof_join collect failed: {e}")))?;

    // The output rows belong to the spine's source. `derived_from` is the
    // FK-lineage anchor naming a source *result table*; the as-of inputs are
    // registered sources (not result tables), so it is `None` and the
    // reproducibility lineage rides the manifest's input anchors instead.
    let table_info = session
        .result_store()
        .create_table(
            spine,
            ModelTask::TextEmbedding,
            ResultTableKind::AsofJoin,
            None,
            ASOF_JOIN_MODEL_ID,
            None,
            None,
            None,
        )
        .await?;

    let out_schema = exec.schema();
    let mut writer = session
        .result_store()
        .open_writer(&table_info.parquet_url, Arc::clone(&out_schema))
        .await?;
    for batch in &batches {
        writer.write_batch(batch).await?;
    }
    let row_count = writer.close().await?;

    // The materialization contract: the join's typed parameters as the producing
    // description, the engine/device with an empty model set (the join runs no
    // model), and a read-time anchor for BOTH input relations. A registered
    // source exposes no as-of/version surface in open-core, so each is honestly
    // recorded as `UnpinnedAtInstant` rather than a fabricated pin.
    let descriptor = descriptor_for(spine, facts, spec);
    let env = MaterializationEnv::new(session.compute_device(), Vec::new());
    let now = chrono::Utc::now().to_rfc3339();
    let inputs = vec![
        InputAnchor::unpinned_at_instant(spine, now.clone()),
        InputAnchor::unpinned_at_instant(facts, now),
    ];
    session
        .result_store()
        .finalize_with_manifest(
            session.context(),
            &table_info.table_name,
            &table_info.parquet_url,
            row_count,
            Materialization::new(&descriptor, &env, inputs),
        )
        .await?;

    session
        .catalog()
        .get_result_table(&table_info.table_name)
        .await?
        .ok_or_else(|| {
            JammiError::Catalog(format!(
                "as-of join table '{}' not found after finalization",
                table_info.table_name
            ))
        })
}

/// The tie-break secondary column, when the policy names one.
fn tie_break_column(tie_break: &TieBreak) -> Option<&str> {
    match tie_break {
        TieBreak::ByColumnDesc(c) => Some(c.as_str()),
        TieBreak::Error => None,
    }
}

/// Resolve a registered relation to its physical scan, reusing the session's
/// tenant-scoped SQL path (so a caller cannot point the join at another tenant's
/// relation). `SELECT *` keeps every column — the join projects from the facts
/// side and passes the spine through verbatim.
async fn scan_relation(
    session: &InferenceSession,
    source_id: &str,
) -> Result<Arc<dyn ExecutionPlan>> {
    let table_name = session.find_table_name(source_id)?;
    let relation = jammi_db::sql::source_relation(source_id, &table_name);
    let df = session
        .context()
        .sql(&format!("SELECT * FROM {relation}"))
        .await
        .map_err(|e| JammiError::Other(format!("asof_join scan of '{source_id}' failed: {e}")))?;
    df.create_physical_plan()
        .await
        .map_err(|e| JammiError::Other(format!("asof_join scan plan for '{source_id}': {e}")))
}

/// Wrap a child scan in a [`SortExec`] ordering it by (`by`..., `time`[,
/// tie-break]) ascending — the order [`AsofJoinExec`]'s single-pointer merge
/// requires. Ascending on the tie-break column places the maximal value of a
/// coincident-time run last, where the merge reads it as the winner.
fn sort_for_merge(
    plan: Arc<dyn ExecutionPlan>,
    by: &[String],
    time: &str,
    tie_break: Option<&str>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let schema = plan.schema();
    let options = SortOptions {
        descending: false,
        nulls_first: false,
    };
    let mut exprs: Vec<PhysicalSortExpr> = Vec::with_capacity(by.len() + 2);
    for key in by {
        let expr = col(key, schema.as_ref())
            .map_err(|e| JammiError::Other(format!("asof_join sort key '{key}': {e}")))?;
        exprs.push(PhysicalSortExpr { expr, options });
    }
    let time_expr = col(time, schema.as_ref())
        .map_err(|e| JammiError::Other(format!("asof_join time key '{time}': {e}")))?;
    exprs.push(PhysicalSortExpr {
        expr: time_expr,
        options,
    });
    if let Some(secondary) = tie_break {
        let expr = col(secondary, schema.as_ref())
            .map_err(|e| JammiError::Other(format!("asof_join tie-break '{secondary}': {e}")))?;
        exprs.push(PhysicalSortExpr { expr, options });
    }
    let ordering = LexOrdering::new(exprs)
        .ok_or_else(|| JammiError::Other("asof_join requires a temporal sort key".into()))?;
    Ok(Arc::new(SortExec::new(ordering, plan)))
}

/// Build the typed [`ProducingDescriptor::AsofJoin`] from the join parameters —
/// the deterministic identity the manifest hashes. The temporal-engine enums map
/// to the manifest's transport-neutral mirrors.
fn descriptor_for(spine: &str, facts: &str, spec: &AsofJoinSpec) -> ProducingDescriptor {
    ProducingDescriptor::AsofJoin {
        spine: spine.to_string(),
        facts: facts.to_string(),
        spine_by: spec.left.by.clone(),
        facts_by: spec.right.by.clone(),
        spine_time: spec.left.time.clone(),
        facts_time: spec.right.time.clone(),
        direction: match spec.direction {
            MatchDirection::Backward => AsofDirection::Backward,
            MatchDirection::Forward => AsofDirection::Forward,
            MatchDirection::Nearest => AsofDirection::Nearest,
        },
        boundary: match spec.boundary {
            Boundary::Inclusive => AsofBoundary::Inclusive,
            Boundary::Exclusive => AsofBoundary::Exclusive,
        },
        tolerance: spec.tolerance.map(|t| match t {
            Tolerance::Duration(d) => AsofTolerance::Duration(d),
            Tolerance::Steps(s) => AsofTolerance::Steps(s),
        }),
        tie_break_column: match &spec.tie_break {
            TieBreak::ByColumnDesc(c) => Some(c.clone()),
            TieBreak::Error => None,
        },
        project: spec.project.clone(),
    }
}
