use crate::catalog::backend::{BackendError, Row, SqlValue, TxOptions};
use crate::catalog::Catalog;
use crate::error::{JammiError, Result};
use crate::model_task::ModelTask;
use crate::tenant_scope::TenantBinding;

/// Whether a result table is a direct model output or a derivation of another
/// result table.
///
/// The discriminator is orthogonal to [`ModelTask`]: a [`Model`](Self::Model)
/// row's `task` names a genuine model task and resolves as an embedding or
/// inference output; a derivation row's `task` still names the source
/// embedding's task (so it round-trips through the same column) but the kind
/// excludes it from embedding-table resolution. Keeping the distinction here
/// rather than in `ModelTask` leaves that enum a pristine catalogue of model
/// tasks (S9 §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum ResultTableKind {
    /// An embedding or inference table produced by running a model.
    Model,
    /// A k-nearest-neighbour edge relation derived from an embedding table.
    NeighborGraph,
}

impl ResultTableKind {
    /// Canonical string stored in the `result_tables.kind` column. The single
    /// source of truth — [`try_from_db_str`](Self::try_from_db_str) decodes it.
    pub fn as_db_str(&self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::NeighborGraph => "neighbor_graph",
        }
    }

    /// Decode the canonical string back into a [`ResultTableKind`]. Unknown
    /// spellings raise [`JammiError::Catalog`] naming the offending value.
    pub fn try_from_db_str(s: &str) -> Result<Self> {
        match s {
            "model" => Ok(Self::Model),
            "neighbor_graph" => Ok(Self::NeighborGraph),
            other => Err(JammiError::Catalog(format!(
                "Unknown result-table kind '{other}'. Expected: model, neighbor_graph"
            ))),
        }
    }
}

/// Parameters for creating a new result table entry.
/// `status` defaults to `'building'` via SQL DEFAULT — not passed here.
#[derive(Debug)]
pub struct CreateResultTableParams<'a> {
    pub table_name: &'a str,
    pub source_id: &'a str,
    pub model_id: &'a str,
    pub task: ModelTask,
    pub kind: ResultTableKind,
    pub derived_from: Option<&'a str>,
    pub parquet_path: &'a str,
    pub index_path: Option<&'a str>,
    pub dimensions: Option<i32>,
    pub key_column: Option<&'a str>,
    pub text_columns: Option<&'a str>,
}

/// A row from the `result_tables` catalog table.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ResultTableRecord {
    pub table_name: String,
    pub source_id: String,
    pub model_id: String,
    pub task: ModelTask,
    pub kind: ResultTableKind,
    pub derived_from: Option<String>,
    pub parquet_path: String,
    pub index_path: Option<String>,
    pub dimensions: Option<i32>,
    pub distance_metric: String,
    pub row_count: usize,
    pub status: String,
    pub key_column: Option<String>,
    pub text_columns: Option<String>,
    pub created_at: String,
    pub completed_at: Option<String>,
    /// Owning tenant, or `None` for a GLOBAL (shared) table. Recovery reads
    /// this to re-scope a cross-tenant reconciliation back to the row's own
    /// tenant when it enumerates orphans under an admin scan; tenant-scoped
    /// callers see only their own rows, so for them this is always either
    /// their tenant or `None`.
    pub tenant_id: Option<String>,
    /// The materialization-contract definition hash — the indexable summary of
    /// the `.materialization.json` sidecar's `definition_hash`. `None` for a
    /// pre-contract table (created before migration 021), which verifies as
    /// [`crate::store::manifest::MatchVerdict::MissingManifest`].
    pub definition_hash: Option<String>,
    /// The materialization-contract input anchors as canonical JSON — the
    /// indexable summary of the sidecar's `input_anchors`. `None` for a
    /// pre-contract table.
    pub input_anchors_json: Option<String>,
}

fn parse_row(row: &Row<'_>) -> std::result::Result<ResultTableRecord, BackendError> {
    let task_raw: String = row.get("task")?;
    let task = ModelTask::try_from_db_str(&task_raw).map_err(|e| BackendError::TypeConversion {
        column: "task".into(),
        detail: e.to_string(),
    })?;
    let kind_raw: String = row.get("kind")?;
    let kind =
        ResultTableKind::try_from_db_str(&kind_raw).map_err(|e| BackendError::TypeConversion {
            column: "kind".into(),
            detail: e.to_string(),
        })?;
    Ok(ResultTableRecord {
        table_name: row.get("table_name")?,
        source_id: row.get("source_id")?,
        model_id: row.get("model_id")?,
        task,
        kind,
        derived_from: row.try_get("derived_from")?,
        parquet_path: row.get("parquet_path")?,
        index_path: row.try_get("index_path")?,
        dimensions: row.try_get("dimensions")?,
        distance_metric: row.get("distance_metric")?,
        row_count: row.get::<i32>("row_count")? as usize,
        status: row.get("status")?,
        key_column: row.try_get("key_column")?,
        text_columns: row.try_get("text_columns")?,
        created_at: row.get("created_at")?,
        completed_at: row.try_get("completed_at")?,
        tenant_id: row.try_get("tenant_id")?,
        definition_hash: row.try_get("definition_hash")?,
        input_anchors_json: row.try_get("input_anchors_json")?,
    })
}

impl Catalog {
    /// Insert a new result table record with status = 'building'. Binds
    /// the session's tenant to the row (SPEC-03 §7).
    pub async fn create_result_table(&self, p: CreateResultTableParams<'_>) -> Result<()> {
        let table_name = p.table_name.to_string();
        let source_id = p.source_id.to_string();
        let model_id = p.model_id.to_string();
        let task = p.task.as_db_str();
        let kind = p.kind.as_db_str();
        let derived_from = p.derived_from.map(str::to_string);
        let parquet_path = p.parquet_path.to_string();
        let index_path = p.index_path.map(str::to_string);
        let dimensions = p.dimensions;
        let key_column = p.key_column.map(str::to_string);
        let text_columns = p.text_columns.map(str::to_string);
        let tenant = self.current_tenant();

        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.assert_tenant_matches(tenant, "result_tables")?;
                    tx.execute(
                        "INSERT INTO result_tables (table_name, source_id, model_id, task, kind, \
                         derived_from, parquet_path, index_path, dimensions, key_column, \
                         text_columns, tenant_id) \
                         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)",
                        &[
                            SqlValue::TextOwned(table_name),
                            SqlValue::TextOwned(source_id),
                            SqlValue::TextOwned(model_id),
                            SqlValue::Text(task),
                            SqlValue::Text(kind),
                            SqlValue::from(derived_from),
                            SqlValue::TextOwned(parquet_path),
                            SqlValue::from(index_path),
                            SqlValue::from(dimensions.map(|d| d as i64)),
                            SqlValue::from(key_column),
                            SqlValue::from(text_columns),
                            SqlValue::from(tenant.map(|t| t.to_string())),
                        ],
                    )
                    .await?;
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Update a result table's status and row count. Sets `completed_at` when
    /// transitioning to a terminal state (Ready/Failed).
    ///
    /// Inside a [`crate::session::JammiSession::with_admin_scope`] closure the
    /// tenant predicate is dropped and the row is reconciled by its
    /// `table_name` primary key alone — startup recovery promotes/fails an
    /// orphan it found under the cross-tenant scan without first re-binding to
    /// that tenant. The match is exact because `table_name` is the table's
    /// PRIMARY KEY. Outside admin scope the update is tenant-scoped, so one
    /// tenant can never flip another tenant's row.
    pub async fn update_result_table_status(
        &self,
        name: &str,
        status: super::status::ResultTableStatus,
        rows: usize,
    ) -> Result<()> {
        let completed_at = if matches!(
            status,
            super::status::ResultTableStatus::Ready | super::status::ResultTableStatus::Failed
        ) {
            Some(
                chrono::Utc::now()
                    .format("%Y-%m-%dT%H:%M:%S%.3fZ")
                    .to_string(),
            )
        } else {
            None
        };
        let status_str = status.to_string();
        let name = name.to_string();
        let rows_i64 = rows as i64;
        let admin = TenantBinding::is_admin_scope();
        let tenant = self.current_tenant();

        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    if admin {
                        tx.execute(
                            "UPDATE result_tables SET status = $1, row_count = $2, \
                             completed_at = $3 WHERE table_name = $4",
                            &[
                                SqlValue::TextOwned(status_str),
                                SqlValue::Int(rows_i64),
                                SqlValue::from(completed_at),
                                SqlValue::TextOwned(name),
                            ],
                        )
                        .await?;
                    } else {
                        tx.execute(
                            "UPDATE result_tables SET status = $1, row_count = $2, \
                             completed_at = $3 \
                             WHERE table_name = $4 AND (tenant_id = $5 OR tenant_id IS NULL)",
                            &[
                                SqlValue::TextOwned(status_str),
                                SqlValue::Int(rows_i64),
                                SqlValue::from(completed_at),
                                SqlValue::TextOwned(name),
                                SqlValue::from(tenant.map(|t| t.to_string())),
                            ],
                        )
                        .await?;
                    }
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Flip a result table `building -> ready` **and** persist its
    /// materialization-contract summary columns (`definition_hash`,
    /// `input_anchors_json`) in a single transaction -- the indexable summary of
    /// the `.materialization.json` sidecar, so verification and provenance
    /// queries need not open every sidecar. This is the catalog half of the
    /// single `building -> ready` boundary
    /// ([`crate::store::ResultStore::finalize_with_manifest`]); the sidecar is
    /// written before this commits, so a crash never leaves a `ready` row whose
    /// manifest never landed.
    ///
    /// Like [`Self::update_result_table_status`] this is tenant-scoped outside
    /// an admin scope and PK-only inside one (recovery promotes an orphan it
    /// found cross-tenant). `completed_at` is set because `ready` is terminal.
    pub async fn promote_result_table_with_manifest(
        &self,
        name: &str,
        rows: usize,
        definition_hash: &str,
        input_anchors_json: &str,
    ) -> Result<()> {
        let completed_at = chrono::Utc::now()
            .format("%Y-%m-%dT%H:%M:%S%.3fZ")
            .to_string();
        let name = name.to_string();
        let rows_i64 = rows as i64;
        let definition_hash = definition_hash.to_string();
        let input_anchors_json = input_anchors_json.to_string();
        let admin = TenantBinding::is_admin_scope();
        let tenant = self.current_tenant();

        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    if admin {
                        tx.execute(
                            "UPDATE result_tables SET status = 'ready', row_count = $1, \
                             completed_at = $2, definition_hash = $3, input_anchors_json = $4 \
                             WHERE table_name = $5",
                            &[
                                SqlValue::Int(rows_i64),
                                SqlValue::TextOwned(completed_at),
                                SqlValue::TextOwned(definition_hash),
                                SqlValue::TextOwned(input_anchors_json),
                                SqlValue::TextOwned(name),
                            ],
                        )
                        .await?;
                    } else {
                        tx.execute(
                            "UPDATE result_tables SET status = 'ready', row_count = $1, \
                             completed_at = $2, definition_hash = $3, input_anchors_json = $4 \
                             WHERE table_name = $5 AND (tenant_id = $6 OR tenant_id IS NULL)",
                            &[
                                SqlValue::Int(rows_i64),
                                SqlValue::TextOwned(completed_at),
                                SqlValue::TextOwned(definition_hash),
                                SqlValue::TextOwned(input_anchors_json),
                                SqlValue::TextOwned(name),
                                SqlValue::from(tenant.map(|t| t.to_string())),
                            ],
                        )
                        .await?;
                    }
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Fetch a single result table by name. Tenant-filtered.
    pub async fn get_result_table(&self, name: &str) -> Result<Option<ResultTableRecord>> {
        let name = name.to_string();
        let tenant = self.current_tenant();
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query_opt(
                            "SELECT * FROM result_tables WHERE table_name = $1 \
                               AND (tenant_id = $2 OR tenant_id IS NULL)",
                            &[
                                SqlValue::TextOwned(name),
                                SqlValue::from(tenant.map(|t| t.to_string())),
                            ],
                            parse_row,
                        )
                        .await
                    })
                },
            )
            .await?)
    }

    /// List result tables with a given status, scoped to the session tenant.
    ///
    /// Inside a [`crate::session::JammiSession::with_admin_scope`] closure the
    /// per-row tenant filter is dropped and rows from **every** tenant are
    /// returned — the cross-tenant enumeration startup recovery needs so a
    /// `building` orphan owned by any tenant is reconciled, not only the
    /// session's own and GLOBAL (`tenant_id IS NULL`) rows. Outside admin scope
    /// the query is tenant-scoped exactly as every other read on this table.
    pub async fn list_result_tables_by_status(
        &self,
        status: super::status::ResultTableStatus,
    ) -> Result<Vec<ResultTableRecord>> {
        let status_str = status.to_string();
        let admin = TenantBinding::is_admin_scope();
        let tenant = self.current_tenant();
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        if admin {
                            tx.query(
                                "SELECT * FROM result_tables WHERE status = $1 \
                                 ORDER BY created_at",
                                &[SqlValue::TextOwned(status_str)],
                                parse_row,
                            )
                            .await
                        } else {
                            tx.query(
                                "SELECT * FROM result_tables WHERE status = $1 \
                                   AND (tenant_id = $2 OR tenant_id IS NULL) \
                                 ORDER BY created_at",
                                &[
                                    SqlValue::TextOwned(status_str),
                                    SqlValue::from(tenant.map(|t| t.to_string())),
                                ],
                                parse_row,
                            )
                            .await
                        }
                    })
                },
            )
            .await?)
    }

    /// Find result tables matching source, optional task, optional model;
    /// scoped to the session tenant.
    pub async fn find_result_tables(
        &self,
        source_id: &str,
        task: Option<ModelTask>,
        model_id: Option<&str>,
    ) -> Result<Vec<ResultTableRecord>> {
        let mut sql = "SELECT * FROM result_tables WHERE source_id = $1".to_string();
        let mut params: Vec<SqlValue<'static>> = vec![SqlValue::TextOwned(source_id.to_string())];

        if let Some(t) = task {
            sql.push_str(&format!(" AND task = ${}", params.len() + 1));
            params.push(SqlValue::Text(t.as_db_str()));
        }
        if let Some(m) = model_id {
            sql.push_str(&format!(" AND model_id = ${}", params.len() + 1));
            params.push(SqlValue::TextOwned(m.to_string()));
        }
        let tenant = self.current_tenant();
        sql.push_str(&format!(
            " AND (tenant_id = ${} OR tenant_id IS NULL)",
            params.len() + 1
        ));
        params.push(SqlValue::from(tenant.map(|t| t.to_string())));
        sql.push_str(" ORDER BY created_at");

        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| Box::pin(async move { tx.query(&sql, &params, parse_row).await }),
            )
            .await?)
    }

    /// Delete all result tables for a source. Returns the deleted records
    /// so callers can clean up associated disk files. Scoped strictly to the
    /// session's tenant — a tenant deletes only its own result tables, never a
    /// shared GLOBAL (`tenant_id IS NULL`) table it did not create; only an
    /// unscoped session manages GLOBAL rows. The SELECT carries the same STRICT
    /// predicate as the DELETE so the returned record set (the disk-cleanup
    /// set) is exactly the deleted set, never a superset.
    pub async fn delete_result_tables_for_source(
        &self,
        source_id: &str,
    ) -> Result<Vec<ResultTableRecord>> {
        let sid = source_id.to_string();
        let tenant = self.current_tenant();
        Ok(self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    let tenant_param = SqlValue::from(tenant.map(|t| t.to_string()));
                    let records = tx
                        .query(
                            "SELECT * FROM result_tables WHERE source_id = $1 \
                               AND (tenant_id = $2 OR (tenant_id IS NULL AND $2 IS NULL))",
                            &[SqlValue::TextOwned(sid.clone()), tenant_param.clone()],
                            parse_row,
                        )
                        .await?;
                    tx.execute(
                        "DELETE FROM result_tables WHERE source_id = $1 \
                           AND (tenant_id = $2 OR (tenant_id IS NULL AND $2 IS NULL))",
                        &[SqlValue::TextOwned(sid), tenant_param],
                    )
                    .await?;
                    Ok(records)
                })
            })
            .await?)
    }

    /// Resolve which embedding table to use for a source. Tenant-filtered.
    pub async fn resolve_embedding_table(
        &self,
        source_id: &str,
        table_name: Option<&str>,
    ) -> Result<ResultTableRecord> {
        if let Some(name) = table_name {
            return self
                .get_result_table(name)
                .await?
                .ok_or_else(|| JammiError::Catalog(format!("Result table '{name}' not found")));
        }

        // Derive the embedding-task list from `ModelTask::ALL` so that
        // adding a future embedding variant automatically extends this
        // resolver — the enum is the single source of truth, not a
        // hardcoded `task IN ('text_embedding', 'image_embedding')`
        // literal. Mirrors the dynamic-placeholder idiom that
        // `find_result_tables` above uses for its conditional binds.
        let embedding_tasks: Vec<&'static str> = ModelTask::ALL
            .iter()
            .filter(|t| t.is_embedding())
            .map(|t| t.as_db_str())
            .collect();
        if embedding_tasks.is_empty() {
            return Err(JammiError::Catalog(
                "ModelTask defines no embedding variants — resolver cannot run".into(),
            ));
        }

        let sid = source_id.to_string();
        let tenant = self.current_tenant();

        // $1 = source_id; $2..$(1+N) = embedding tasks; $(2+N) = tenant.
        let mut params: Vec<SqlValue<'static>> = Vec::with_capacity(embedding_tasks.len() + 2);
        params.push(SqlValue::TextOwned(sid));
        let task_placeholders: Vec<String> = (0..embedding_tasks.len())
            .map(|i| format!("${}", i + 2))
            .collect();
        for t in &embedding_tasks {
            params.push(SqlValue::Text(t));
        }
        let tenant_placeholder = format!("${}", params.len() + 1);
        params.push(SqlValue::from(tenant.map(|t| t.to_string())));

        // `kind = 'model'` excludes derived tables (e.g. a neighbor-graph edge
        // relation) whose `task` column still names the source embedding's
        // task — only genuine model outputs resolve as an embedding source.
        let sql = format!(
            "SELECT * FROM result_tables \
             WHERE source_id = $1 AND task IN ({tasks}) \
               AND kind = 'model' \
               AND status = 'ready' \
               AND (tenant_id = {tenant} OR tenant_id IS NULL) \
             ORDER BY created_at DESC, rowid DESC LIMIT 1",
            tasks = task_placeholders.join(", "),
            tenant = tenant_placeholder,
        );

        let found = self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| Box::pin(async move { tx.query_opt(&sql, &params, parse_row).await }),
            )
            .await?;
        found.ok_or_else(|| {
            JammiError::Catalog(format!("No ready embedding table for source '{source_id}'"))
        })
    }

    /// Persist a checkpoint (batch number) for a result table.
    pub async fn set_checkpoint(&self, name: &str, batch: usize) -> Result<()> {
        let name = name.to_string();
        let batch_i64 = batch as i64;
        let tenant = self.current_tenant();
        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.execute(
                        "UPDATE result_tables SET checkpoint = $1 WHERE table_name = $2 \
                           AND (tenant_id = $3 OR tenant_id IS NULL)",
                        &[
                            SqlValue::Int(batch_i64),
                            SqlValue::TextOwned(name),
                            SqlValue::from(tenant.map(|t| t.to_string())),
                        ],
                    )
                    .await?;
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Retrieve the last checkpoint for a result table.
    pub async fn get_checkpoint(&self, name: &str) -> Result<Option<usize>> {
        let name = name.to_string();
        let tenant = self.current_tenant();
        let found: Option<Option<i32>> = self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query_opt(
                            "SELECT checkpoint FROM result_tables WHERE table_name = $1 \
                               AND (tenant_id = $2 OR tenant_id IS NULL)",
                            &[
                                SqlValue::TextOwned(name),
                                SqlValue::from(tenant.map(|t| t.to_string())),
                            ],
                            |row| row.try_get::<i32>("checkpoint"),
                        )
                        .await
                    })
                },
            )
            .await?;
        Ok(found.flatten().map(|c| c as usize))
    }
}
