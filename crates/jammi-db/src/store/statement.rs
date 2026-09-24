//! `CREATE TABLE … AS` as the store executes it: a result table named by
//! the statement, produced through the same sink and funnel every other
//! producer uses.
//!
//! `OR REPLACE` never loses the table it replaces. The statement's rows are
//! built under a row of their own that names the table it is to replace
//! ([`CreateResultTableParams::replaces`]); the old table serves under its
//! name until the new artifact is complete and attested, then the promote
//! transaction removes the old row and moves the new one onto the name —
//! a reader resolves one or the other, never none — and the old bytes are
//! reclaimed after it. A failure anywhere before that transaction — the
//! query refusing mid-write, the plane declining, the promote losing its
//! lease — leaves the old table as it was and discards the row and bytes
//! the statement made. A replacement is not a version of the table it
//! replaces: a version is a keyed delta over a base artifact that stays,
//! while `OR REPLACE` produces a different artifact under a possibly
//! different definition and reclaims the old one, which is exactly what a
//! row of its own gives it.

use std::sync::Arc;

use datafusion::execution::TaskContext;
use datafusion::physical_plan::ExecutionPlan;
use tracing::warn;

use crate::catalog::result_repo::{ResultTableCas, ResultTableKind, ResultTableRecord};
use crate::error::{JammiError, Result};
use crate::model_task::ModelTask;
use crate::session::QueryContext;
use crate::store::building::BuildingTable;
use crate::store::manifest::{InputAnchor, Materialization, ProducingDescriptor};
use crate::store::{ResultStore, ResultTableOrigin, SinkKind};

#[cfg(doc)]
use crate::catalog::result_repo::CreateResultTableParams;

/// The provenance id a statement's result table carries in its `model_id`
/// column — a statement invokes no model, but the column is NOT NULL, so a
/// stable sentinel rides it (the shape the as-of join and training-set
/// producers use).
const STATEMENT_MODEL_ID: &str = "statement";

/// What a `CREATE TABLE <name> AS <query>` asks for.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd)]
pub struct CreateTableAs {
    /// The result table's name — the `result_tables` primary key, read as
    /// `"jammi.<name>"`.
    pub name: String,
    /// `IF NOT EXISTS`: an existing table of this name is left as it is.
    pub if_not_exists: bool,
    /// `OR REPLACE`: an existing table of this name serves until the new
    /// one is complete, then the name moves to the new table in one catalog
    /// transaction and the old bytes are reclaimed. A name nothing is under
    /// is simply created.
    pub or_replace: bool,
    /// The `AS <query>` part as SQL — the table's definition, which the
    /// manifest hashes and a recompute re-plans.
    pub query: String,
    /// Every relation the query scans, as spelled — the table's input
    /// anchors, and its lineage's source.
    pub sources: Vec<String>,
}

/// The name a replacement of `name` is built under: unique per statement,
/// and never the name itself, which the table being replaced keeps until
/// the swap.
fn replacement_name(name: &str) -> String {
    let timestamp = chrono::Utc::now().format("%Y%m%dT%H%M%S%9f");
    let suffix = &uuid::Uuid::new_v4().simple().to_string()[..8];
    format!("{name}__replacement__{timestamp}_{suffix}")
}

impl ResultStore {
    /// Materialize `query`'s rows as the result table `statement` names:
    /// the row is created under the caller's tenant, written through the
    /// sink where the compute plane says, attested with a
    /// [`ProducingDescriptor::Statement`] over the query and an
    /// unpinned anchor per scanned relation, and promoted `ready` — under
    /// the name, replacing the table there when `OR REPLACE` found one
    /// (see the module doc). `None` when `IF NOT EXISTS` found the table;
    /// an existing table without `OR REPLACE` or `IF NOT EXISTS` is refused
    /// typed. A statement that fails leaves no row and no bytes of its own
    /// behind.
    pub async fn create_table_as(
        &self,
        ctx: &QueryContext,
        statement: &CreateTableAs,
        query: Arc<dyn ExecutionPlan>,
        context: Arc<TaskContext>,
    ) -> Result<Option<ResultTableRecord>> {
        let name = statement.name.as_str();
        let replaces = match self.catalog.get_result_table(name).await? {
            Some(_) if statement.or_replace => Some(name),
            Some(_) if statement.if_not_exists => return Ok(None),
            Some(_) => {
                return Err(JammiError::Catalog(format!(
                    "result table '{name}' already exists"
                )))
            }
            None => None,
        };
        let table_name = replaces.map_or_else(|| name.to_string(), replacement_name);
        // The lineage column names the first relation the query scans; a
        // query scanning none (`SELECT 1 AS id`) is its own source.
        let source_id = statement.sources.first().map_or(name, String::as_str);
        let building = self
            .create_named_table(
                table_name,
                ResultTableOrigin {
                    source_id,
                    // A statement's rows are filed under no model task; the
                    // column is NOT NULL, so the same filler every non-model
                    // producer stamps.
                    task: ModelTask::TextEmbedding,
                    kind: ResultTableKind::Statement,
                    derived_from: None,
                    model_id: STATEMENT_MODEL_ID,
                    dimensions: None,
                    key_column: None,
                    text_columns: None,
                    job_attempt: None,
                },
                replaces,
            )
            .await?;
        let row = building.table_name().to_string();
        let cas = building.cas();
        match self
            .materialize_statement(ctx, statement, building, query, context)
            .await
        {
            Ok(record) => Ok(Some(record)),
            Err(e) => {
                self.discard_statement_row(&cas, &row).await;
                Err(e)
            }
        }
    }

    /// The statement's write and finish over its `building` row.
    async fn materialize_statement(
        &self,
        ctx: &QueryContext,
        statement: &CreateTableAs,
        mut building: BuildingTable,
        query: Arc<dyn ExecutionPlan>,
        context: Arc<TaskContext>,
    ) -> Result<ResultTableRecord> {
        let summary = self
            .write_result_table(&mut building, SinkKind::Rows, query, context)
            .await?;
        let descriptor = ProducingDescriptor::Statement {
            query: statement.query.clone(),
        };
        let now = chrono::Utc::now().to_rfc3339();
        let inputs = statement
            .sources
            .iter()
            .map(|source| InputAnchor::unpinned_at_instant(source, now.clone()))
            .collect();
        building
            .finish(
                ctx,
                summary.rows as usize,
                Materialization::new(&descriptor, &summary.env, inputs),
            )
            .await
    }

    /// Discard the row a failed statement made, with its bytes: the
    /// `building -> failed` CAS under the statement's own writer (a miss is
    /// the row's answer — the sink already failed it, or recovery holds
    /// it), then the drop of the terminal row. A statement's failed row is
    /// nobody's evidence: the statement reported its error, and no name
    /// reaches the row. What this cannot remove is logged and left to
    /// `reconcile`; the statement's own error is what the caller sees.
    async fn discard_statement_row(&self, cas: &ResultTableCas, row: &str) {
        if let Err(e) = self.catalog.fail_building_table(cas).await {
            warn!(table = row, outcome = %e, "statement: the failed row was not this writer's to fail");
        }
        match self.drop_result_table(row).await {
            Ok(_) | Err(JammiError::RowGone { .. }) => {}
            Err(e) => {
                warn!(table = row, error = %e, "statement: the failed row was not discarded; reconcile reaps it");
            }
        }
    }
}
