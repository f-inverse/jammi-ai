//! `CREATE TABLE … AS` as the store executes it: a result table named by
//! the statement, produced through the same sink and funnel every other
//! producer uses.

use std::sync::Arc;

use datafusion::execution::TaskContext;
use datafusion::physical_plan::ExecutionPlan;

use crate::catalog::result_repo::{ResultTableKind, ResultTableRecord};
use crate::error::{JammiError, Result};
use crate::model_task::ModelTask;
use crate::session::QueryContext;
use crate::store::manifest::{
    ComputeDevice, InputAnchor, Materialization, MaterializationEnv, ProducingDescriptor,
};
use crate::store::{ResultStore, ResultTableOrigin, SinkKind};

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
    /// `OR REPLACE`: an existing table of this name is dropped first.
    pub or_replace: bool,
    /// The query, rendered as its logical plan — the table's definition,
    /// which the manifest hashes.
    pub definition: String,
    /// Every relation the query scans, as spelled — the table's input
    /// anchors, and its lineage's source.
    pub sources: Vec<String>,
}

impl ResultStore {
    /// Materialize `query`'s rows as the result table `statement` names:
    /// the row is created under the caller's tenant, written through the
    /// sink where the compute plane says, attested with a
    /// [`ProducingDescriptor::Statement`] over the definition and an
    /// unpinned anchor per scanned relation, and promoted `ready`. `None`
    /// when `IF NOT EXISTS` found the table; an existing table without
    /// `OR REPLACE` or `IF NOT EXISTS` is refused typed.
    pub async fn create_table_as(
        &self,
        ctx: &QueryContext,
        statement: &CreateTableAs,
        query: Arc<dyn ExecutionPlan>,
        context: Arc<TaskContext>,
    ) -> Result<Option<ResultTableRecord>> {
        let name = statement.name.as_str();
        if self.catalog.get_result_table(name).await?.is_some() {
            if statement.or_replace {
                self.drop_result_table(name).await?;
            } else if statement.if_not_exists {
                return Ok(None);
            } else {
                return Err(JammiError::Catalog(format!(
                    "result table '{name}' already exists"
                )));
            }
        }
        // The lineage column names the first relation the query scans; a
        // query scanning none (a `VALUES` list) is its own source.
        let source_id = statement.sources.first().map_or(name, String::as_str);
        let mut building = self
            .create_named_table(
                name.to_string(),
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
            )
            .await?;
        let summary = self
            .write_result_table(&mut building, SinkKind::Rows, query, context)
            .await?;
        let descriptor = ProducingDescriptor::Statement {
            definition: statement.definition.clone(),
        };
        // A statement runs no model; its rows do not depend on a device.
        let env = MaterializationEnv::new(ComputeDevice::Cpu, Vec::new());
        let now = chrono::Utc::now().to_rfc3339();
        let inputs = statement
            .sources
            .iter()
            .map(|source| InputAnchor::unpinned_at_instant(source, now.clone()))
            .collect();
        let record = building
            .finish(
                ctx,
                summary.rows as usize,
                Materialization::new(&descriptor, &env, inputs),
            )
            .await?;
        Ok(Some(record))
    }
}
