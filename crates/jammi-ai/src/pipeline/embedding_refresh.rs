//! Incremental embedding — the three actuators over a versioned embedding
//! table (`refresh_embeddings`, `compact_embeddings`, `expire_versions`).
//!
//! A refresh re-embeds only what changed: it classifies every source row by
//! comparing the `_content_hash` the source scan computes today against the
//! hash stored beside the row's current vector (`Added` / `Changed` /
//! `Unchanged`, and `Deleted` = current keys the source no longer has), infers
//! ONLY `Added ∪ Changed` through the same `InferenceExec` the base embed uses,
//! writes them as a new immutable fragment + ANN segment stamped with the new
//! version, raises the deletion mask's horizon to `N - 1` for every superseded
//! or deleted key, and publishes the version in one compare-and-set. Nothing
//! is visible before the publish; a refused refresh (null key, duplicate key,
//! definition drift, missing hashes) leaves the previous version live.
//!
//! These are actuators, not a control loop: when to refresh, compact or expire
//! is the consumer's. Every verb is exposed on the remote surface too (the
//! library is never less capable than the server).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::physical_expr::expressions::{col, CastExpr};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::joins::{HashJoinExec, PartitionMode};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;
use futures::StreamExt;
use jammi_db::catalog::result_repo::{ResultTableKind, ResultTableRecord};
use jammi_db::catalog::status::ResultTableStatus;
use jammi_db::error::{JammiError, NonUniqueScan, NotRefreshableReason, Result};
use jammi_db::index::sidecar::SidecarIndex;
use jammi_db::index::VectorIndex;
use jammi_db::model_task::ModelTask;
use jammi_db::storage::StorageUrl;
use jammi_db::store::content_hash::ContentHash;
use jammi_db::store::manifest::{
    ArtifactDigest, DefinitionHash, DeletePolicy, InputAnchor, MaterializationManifest,
    ProducingDescriptor,
};
use jammi_db::store::schema::CONTENT_HASH_COLUMN;
use jammi_db::store::version::{
    DeletesRef, FragmentRef, SegmentRef, VersionDelta, VersionManifest,
};
use jammi_db::store::{BuildingVersion, ResultStore};
use jammi_db::tenant_scope::TenantBinding;

use crate::operator::inference_exec::InferenceExecBuilder;
use crate::operator::ordered_input::{key_checked, ordered_input};
use crate::pipeline::embedding::{embedding_definition, EmbeddingDefinition};
use crate::pipeline::result_sink::ResultSink;
use crate::session::InferenceSession;

/// The options of one refresh.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RefreshOptions {
    /// What a key the source no longer has becomes (default: tombstoned).
    pub deletes: DeletePolicy,
}

/// Whether a refresh published a new version or found nothing to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RefreshOutcome {
    Published,
    NoChange,
}

/// What a refresh or compaction did. Realized counts: `inferred_rows` is the
/// number of rows the model was invoked on, `dropped_rows` the keys asked for
/// that the model did not realize (a per-row input failure), so `inferred_rows
/// - dropped_rows` is the fragment's row count.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RefreshReport {
    pub table: String,
    /// The published version (`Published`), or the current one (`NoChange`).
    pub version: Option<i64>,
    pub parent_version: Option<i64>,
    pub inferred_rows: u64,
    pub added: u64,
    pub changed: u64,
    pub deleted: u64,
    pub unchanged: u64,
    pub dropped_rows: u64,
    pub live_rows: u64,
    pub masked_rows: u64,
    pub outcome: RefreshOutcome,
}

/// What `expire_versions` removed.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ExpiryReport {
    pub table: String,
    /// The version rows deleted, ascending.
    pub expired_versions: Vec<i64>,
    /// The objects (fragments, deletes, manifests, segment siblings) deleted.
    pub objects_deleted: u64,
}

/// Test-only rendezvous points inside a refresh (`test-hooks`): a test parks
/// a refresh at a documented point and observes the catalog / read paths
/// beside it.
#[cfg(feature = "test-hooks")]
pub mod refresh_test_hooks {
    use std::sync::{Arc, Mutex, PoisonError};

    use tokio::sync::Notify;

    /// Where a refresh may park.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ParkPoint {
        /// After the delta fragment and segment landed, before the publish
        /// transaction — the window §6.13 (concurrent visibility) and §6.12
        /// (an expired version lease beside a live table row) observe.
        BeforePublish,
    }

    struct Armed {
        table: String,
        point: ParkPoint,
        parked: std::sync::atomic::AtomicBool,
        parked_notify: Notify,
        release: Notify,
        released: std::sync::atomic::AtomicBool,
    }

    static ARM: Mutex<Vec<Arc<Armed>>> = Mutex::new(Vec::new());

    /// The test's handle on an armed park point.
    pub struct RefreshPark {
        state: Arc<Armed>,
    }

    /// Arm `point` for the next refresh of `table`.
    pub fn arm(table: &str, point: ParkPoint) -> RefreshPark {
        let state = Arc::new(Armed {
            table: table.to_string(),
            point,
            parked: std::sync::atomic::AtomicBool::new(false),
            parked_notify: Notify::new(),
            release: Notify::new(),
            released: std::sync::atomic::AtomicBool::new(false),
        });
        let mut guard = ARM.lock().unwrap_or_else(PoisonError::into_inner);
        assert!(
            !guard.iter().any(|s| s.table == table),
            "refresh test hook: a park is already armed for '{table}'"
        );
        guard.push(Arc::clone(&state));
        RefreshPark { state }
    }

    impl RefreshPark {
        /// Wait (bounded to 30s) until the refresh has parked.
        pub async fn wait_parked(&self) {
            let notified = self.state.parked_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.state.parked.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            let _ = tokio::time::timeout(std::time::Duration::from_secs(30), notified).await;
        }

        /// Whether the refresh is parked.
        pub fn is_parked(&self) -> bool {
            self.state.parked.load(std::sync::atomic::Ordering::SeqCst)
        }

        /// Release the parked refresh and disarm (idempotent).
        pub fn release(&self) {
            self.state
                .released
                .store(true, std::sync::atomic::Ordering::SeqCst);
            self.state.release.notify_waiters();
            let mut guard = ARM.lock().unwrap_or_else(PoisonError::into_inner);
            guard.retain(|s| !Arc::ptr_eq(s, &self.state));
        }
    }

    impl Drop for RefreshPark {
        fn drop(&mut self) {
            self.release();
        }
    }

    /// One-shot: the FIRST refresh of the armed table to reach `point`
    /// takes the arm and parks; a concurrent second refresh passes through
    /// (§6.7 needs the second publisher to run to completion beside the
    /// parked first).
    pub(super) async fn maybe_park(table: &str, point: ParkPoint) {
        let state = {
            let mut guard = ARM.lock().unwrap_or_else(PoisonError::into_inner);
            let pos = guard
                .iter()
                .position(|s| s.table == table && s.point == point);
            pos.map(|i| guard.remove(i))
        };
        let Some(state) = state else {
            return;
        };
        let released = state.release.notified();
        tokio::pin!(released);
        released.as_mut().enable();
        state
            .parked
            .store(true, std::sync::atomic::Ordering::SeqCst);
        state.parked_notify.notify_waiters();
        if state.released.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        let _ = tokio::time::timeout(std::time::Duration::from_secs(60), released).await;
    }
}

/// The embedding parameters every embedding-family descriptor carries.
struct EmbeddingParams {
    model_id: String,
    task: ModelTask,
    source_id: String,
    columns: Vec<String>,
    key_column: String,
    dimensions: usize,
}

fn embedding_params(table: &str, descriptor: &ProducingDescriptor) -> Result<EmbeddingParams> {
    match descriptor {
        ProducingDescriptor::Embedding {
            model_id,
            task,
            source_id,
            columns,
            key_column,
            dimensions,
        }
        | ProducingDescriptor::EmbeddingDelta {
            model_id,
            task,
            source_id,
            columns,
            key_column,
            dimensions,
            ..
        }
        | ProducingDescriptor::EmbeddingCompaction {
            model_id,
            task,
            source_id,
            columns,
            key_column,
            dimensions,
            ..
        } => Ok(EmbeddingParams {
            model_id: model_id.clone(),
            task: *task,
            source_id: source_id.clone(),
            columns: columns.clone(),
            key_column: key_column.clone(),
            dimensions: *dimensions,
        }),
        _ => Err(JammiError::NotRefreshable {
            table: table.to_string(),
            reason: NotRefreshableReason::NotEmbeddingTable,
        }),
    }
}

/// The current version's row set: `_row_id → content hash`, read through the
/// bound (masked) provider. Duplicate keys → `NonUniqueKey { Parent }`; a NULL
/// or malformed hash → `NotRefreshable { MissingContentHash }`.
async fn current_state(ctx: &SessionContext, table: &str) -> Result<HashMap<String, ContentHash>> {
    use datafusion::sql::TableReference;
    let table_ref = TableReference::bare(format!("jammi.{table}"));
    let df = ctx.table(table_ref).await.map_err(JammiError::from)?;
    if df
        .schema()
        .field_with_unqualified_name(CONTENT_HASH_COLUMN)
        .is_err()
    {
        return Err(JammiError::NotRefreshable {
            table: table.to_string(),
            reason: NotRefreshableReason::MissingContentHash,
        });
    }
    let batches = df
        .select_columns(&["_row_id", CONTENT_HASH_COLUMN])
        .map_err(JammiError::from)?
        .collect()
        .await
        .map_err(JammiError::from)?;
    let mut out: HashMap<String, ContentHash> = HashMap::new();
    let mut dups: HashMap<String, u64> = HashMap::new();
    for batch in &batches {
        let ids = arrow::compute::cast(batch.column(0), &DataType::Utf8)
            .map_err(|e| JammiError::Other(format!("refresh: cast: {e}")))?;
        let ids = ids
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| JammiError::Inference("_row_id is not a string column".into()))?;
        let hashes = arrow::compute::cast(batch.column(1), &DataType::Utf8)
            .map_err(|e| JammiError::Other(format!("refresh: cast: {e}")))?;
        let hashes = hashes
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| JammiError::Inference("_content_hash is not a string column".into()))?;
        for i in 0..batch.num_rows() {
            if hashes.is_null(i) {
                return Err(JammiError::NotRefreshable {
                    table: table.to_string(),
                    reason: NotRefreshableReason::MissingContentHash,
                });
            }
            let hash =
                ContentHash::from_hex(hashes.value(i)).map_err(|_| JammiError::NotRefreshable {
                    table: table.to_string(),
                    reason: NotRefreshableReason::MissingContentHash,
                })?;
            let key = ids.value(i).to_string();
            if out.insert(key.clone(), hash).is_some() {
                *dups.entry(key).or_insert(1) += 1;
            }
        }
    }
    if !dups.is_empty() {
        return Err(non_unique(table, NonUniqueScan::Parent, dups));
    }
    Ok(out)
}

fn non_unique(table: &str, scan: NonUniqueScan, dups: HashMap<String, u64>) -> JammiError {
    let total = dups.len() as u64;
    let mut keys: Vec<(String, u64)> = dups.into_iter().collect();
    keys.sort();
    keys.truncate(10);
    JammiError::NonUniqueKey {
        table: table.to_string(),
        scan,
        keys,
        total,
    }
}

/// The classified delta of one source scan against the current state.
struct Classified {
    added: Vec<String>,
    changed: Vec<String>,
    deleted: Vec<String>,
    unchanged: u64,
}

impl InferenceSession {
    /// Re-embed only the source rows whose content changed since `table`'s
    /// current version, and publish the result as a new version. See the
    /// module doc; every refusal is typed and leaves the previous version live.
    pub async fn refresh_embeddings(
        self: &Arc<Self>,
        table: &str,
        options: RefreshOptions,
    ) -> Result<RefreshReport> {
        let store = self.result_store();
        let ctx = self.context();

        // ── step 0: gates ──────────────────────────────────────────────────
        let record = self.refreshable_record(table).await?;
        let descriptor = store.producing_descriptor(&record).await?;
        let params = embedding_params(table, &descriptor)?;
        let definition = embedding_definition(self, &params.model_id, params.task).await?;
        self.check_definition_drift(&record, &params, &definition)?;

        // ── step 1: the base publish ───────────────────────────────────────
        let record = self.ensure_base_version(&store, record, &params).await?;
        let parent_version = record
            .current_version
            .expect("a base version is published before any delta");

        // ── step 2: the parent manifest ────────────────────────────────────
        let parquet_url = StorageUrl::parse(&record.parquet_path)?;
        let parent = store
            .read_version_manifest(&record.table_name, &parquet_url, parent_version)
            .await?
            .ok_or_else(|| JammiError::VersionUnavailable {
                table: record.table_name.clone(),
                version: parent_version,
            })?;

        // ── step 3: the current state ──────────────────────────────────────
        let current = current_state(ctx, &record.table_name).await?;

        // ── step 4: the source scan, classified ────────────────────────────
        let source_query = self.source_query_for(&params)?;
        let classified = self
            .classify_source(
                &record.table_name,
                &source_query,
                &params,
                &current,
                options.deletes,
            )
            .await?;

        // ── step 5: an empty delta ─────────────────────────────────────────
        if classified.added.is_empty()
            && classified.changed.is_empty()
            && classified.deleted.is_empty()
        {
            return Ok(RefreshReport {
                table: record.table_name.clone(),
                version: Some(parent_version),
                parent_version: parent.parent,
                inferred_rows: 0,
                added: 0,
                changed: 0,
                deleted: 0,
                unchanged: classified.unchanged,
                dropped_rows: 0,
                live_rows: parent.live_rows as u64,
                masked_rows: parent.masked_rows as u64,
                outcome: RefreshOutcome::NoChange,
            });
        }

        // ── step 6: allocate ───────────────────────────────────────────────
        let mut version = store.allocate_version(&record).await?;
        let n = version.version();

        // ── step 7: infer the delta ────────────────────────────────────────
        let mut to_infer: Vec<String> = classified
            .added
            .iter()
            .chain(classified.changed.iter())
            .cloned()
            .collect();
        to_infer.sort();
        let (fragment, realized) = if to_infer.is_empty() {
            (None, HashSet::new())
        } else {
            self.infer_delta(
                &store,
                &version,
                &params,
                &definition,
                &source_query,
                &to_infer,
            )
            .await?
        };
        let dropped_rows = to_infer.iter().filter(|k| !realized.contains(*k)).count() as u64;

        // ── step 9: the deletion mask ──────────────────────────────────────
        let mut mask = store
            .read_deletion_mask(&record.table_name, &parent)
            .await?;
        for key in classified.changed.iter().chain(classified.deleted.iter()) {
            mask.raise(key.clone(), n - 1);
        }
        for key in &realized {
            debug_assert!(!mask.is_masked(key, n), "a key of fragment {n} is masked");
        }
        let deletes = if mask.is_empty() {
            None
        } else {
            let url = version.deletes_url()?;
            let (entries, digest) = mask.write(&store.open_parquet(&url)?).await?;
            Some(DeletesRef {
                url: url.as_str().to_string(),
                entries,
                digest,
            })
        };

        // ── step 10: the manifest ──────────────────────────────────────────
        let mut fragments = parent.fragments.clone();
        let mut segments = parent.segments.clone();
        if let Some((fragment_ref, segment_id)) = fragment {
            fragments.push(fragment_ref);
            segments.push(SegmentRef {
                segment_id,
                version: n,
            });
        }
        let delta_descriptor = ProducingDescriptor::EmbeddingDelta {
            model_id: params.model_id.clone(),
            task: params.task,
            source_id: params.source_id.clone(),
            columns: params.columns.clone(),
            key_column: params.key_column.clone(),
            dimensions: params.dimensions,
            parent_version,
            parent_identity: parent.identity.clone(),
            deletes: options.deletes,
        };
        let anchors = vec![InputAnchor::unpinned_at_instant(
            &params.source_id,
            chrono::Utc::now().to_rfc3339(),
        )];
        let definition_hash = DefinitionHash(record.definition_hash.clone().ok_or_else(|| {
            JammiError::NotRefreshable {
                table: record.table_name.clone(),
                reason: NotRefreshableReason::MissingContentHash,
            }
        })?);
        let identity = VersionManifest::compute_identity(
            &parent.identity,
            &definition_hash,
            &delta_descriptor,
            &fragments,
            deletes.as_ref(),
        )?;
        let mut manifest = VersionManifest {
            version_format: jammi_db::store::version::VERSION_FORMAT,
            table: record.table_name.clone(),
            version: n,
            parent: Some(parent_version),
            definition_hash,
            delta: VersionDelta {
                descriptor: delta_descriptor,
                input_anchors: anchors.clone(),
            },
            fragments,
            segments,
            deletes,
            live_rows: 0,
            masked_rows: 0,
            identity,
            produced_by: jammi_db::store::run_id().to_string(),
            produced_at: chrono::Utc::now().to_rfc3339(),
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
        };
        let physical_rows: usize = manifest.fragments.iter().map(|f| f.rows).sum();
        let live_rows = store.count_live_rows(ctx, &record, &manifest).await?;
        manifest.live_rows = live_rows;
        manifest.masked_rows = physical_rows.saturating_sub(live_rows);
        store
            .write_version_manifest(&parquet_url, &manifest)
            .await?;

        #[cfg(feature = "test-hooks")]
        refresh_test_hooks::maybe_park(
            &record.table_name,
            refresh_test_hooks::ParkPoint::BeforePublish,
        )
        .await;

        // ── step 11: publish ───────────────────────────────────────────────
        let anchors_json = serde_json::to_string(&anchors)?;
        if let Err(e) = version
            .publish(
                &manifest.identity,
                live_rows,
                manifest.masked_rows,
                &anchors_json,
            )
            .await
        {
            // The publish CAS missed (a concurrent publisher moved the parent,
            // or the lease was lost): fail this version and reap its bytes.
            if let Err(abort_err) = version.abort().await {
                tracing::warn!(table = record.table_name, version = n, error = %abort_err, "refresh: abort after a missed publish did not complete");
            }
            return Err(e);
        }
        self.after_publish(&store, &record.table_name, &params.source_id)
            .await?;

        Ok(RefreshReport {
            table: record.table_name.clone(),
            version: Some(n),
            parent_version: Some(parent_version),
            inferred_rows: to_infer.len() as u64,
            added: classified.added.len() as u64,
            changed: classified.changed.len() as u64,
            deleted: classified.deleted.len() as u64,
            unchanged: classified.unchanged,
            dropped_rows,
            live_rows: live_rows as u64,
            masked_rows: manifest.masked_rows as u64,
            outcome: RefreshOutcome::Published,
        })
    }

    /// Step 0's catalog gates: the tenant-scoped read, the STRICT tenant pair
    /// (a scoped tenant can read a GLOBAL table but never refresh it), `ready`,
    /// an embedding table, and a `ready` current version.
    pub(crate) async fn refreshable_record(&self, table: &str) -> Result<ResultTableRecord> {
        let record = self
            .catalog()
            .get_result_table(table)
            .await?
            .ok_or_else(|| JammiError::Catalog(format!("Result table '{table}' not found")))?;
        if !TenantBinding::is_admin_scope() {
            let caller = self.catalog().current_tenant().map(|t| t.to_string());
            if record.tenant_id != caller {
                return Err(JammiError::TenantMismatch {
                    table: table.to_string(),
                });
            }
        }
        if record.status != ResultTableStatus::Ready.to_string() {
            return Err(JammiError::NotRefreshable {
                table: table.to_string(),
                reason: NotRefreshableReason::NotReady,
            });
        }
        if record.kind != ResultTableKind::Model
            || !matches!(
                record.task,
                ModelTask::TextEmbedding | ModelTask::ImageEmbedding | ModelTask::AudioEmbedding
            )
        {
            return Err(JammiError::NotRefreshable {
                table: table.to_string(),
                reason: NotRefreshableReason::NotEmbeddingTable,
            });
        }
        if let Some(v) = record.current_version {
            let row = self.catalog().get_result_table_version(table, v).await?;
            if row.is_none_or(|r| r.status != ResultTableStatus::Ready.to_string()) {
                return Err(JammiError::NotRefreshable {
                    table: table.to_string(),
                    reason: NotRefreshableReason::CurrentVersionUnavailable,
                });
            }
        }
        Ok(record)
    }

    /// D19: the definition rebuilt from the current parameters and the loaded
    /// model must equal the table's recorded `definition_hash`.
    fn check_definition_drift(
        &self,
        record: &ResultTableRecord,
        params: &EmbeddingParams,
        definition: &EmbeddingDefinition,
    ) -> Result<()> {
        let descriptor = ProducingDescriptor::Embedding {
            model_id: params.model_id.clone(),
            task: params.task,
            source_id: params.source_id.clone(),
            columns: params.columns.clone(),
            key_column: params.key_column.clone(),
            dimensions: params.dimensions,
        };
        let current = MaterializationManifest::definition_of(&descriptor, &definition.env)
            .map_err(jammi_db::store::manifest_to_jammi)?;
        match &record.definition_hash {
            Some(recorded) if recorded == current.as_str() => Ok(()),
            Some(recorded) => Err(JammiError::DefinitionDrift {
                table: record.table_name.clone(),
                recorded: recorded.clone(),
                current: current.as_str().to_string(),
            }),
            None => Err(JammiError::NotRefreshable {
                table: record.table_name.clone(),
                reason: NotRefreshableReason::MissingContentHash,
            }),
        }
    }

    /// Step 1: publish the base version of a never-refreshed table (D3) and
    /// bind the versioned provider, returning the re-read record. A table
    /// already versioned is returned unchanged.
    async fn ensure_base_version(
        &self,
        store: &ResultStore,
        record: ResultTableRecord,
        params: &EmbeddingParams,
    ) -> Result<ResultTableRecord> {
        if record.current_version.is_some() {
            return Ok(record);
        }
        let table = record.table_name.clone();
        let parquet_url = StorageUrl::parse(&record.parquet_path)?;
        if record.dimensions.is_none() {
            return Err(JammiError::NotRefreshable {
                table,
                reason: NotRefreshableReason::NotEmbeddingTable,
            });
        }
        let base_manifest = store
            .read_materialization_manifest(&parquet_url)
            .await?
            .ok_or_else(|| JammiError::NotRecomputable {
                table: table.clone(),
            })?;
        let b = record.next_version;
        let segments: Vec<SegmentRef> =
            TenantBinding::admin_scope(self.catalog().list_base_index_segments(&table))
                .await?
                .into_iter()
                .map(|s| SegmentRef {
                    segment_id: s.segment_id,
                    version: b,
                })
                .collect();
        let anchors: Vec<InputAnchor> = match &record.input_anchors_json {
            Some(json) => serde_json::from_str(json)?,
            None => vec![InputAnchor::unpinned_at_instant(
                &params.source_id,
                chrono::Utc::now().to_rfc3339(),
            )],
        };
        let identity = base_manifest.artifact.as_str().to_string();
        let manifest = VersionManifest {
            version_format: jammi_db::store::version::VERSION_FORMAT,
            table: table.clone(),
            version: b,
            parent: None,
            definition_hash: base_manifest.definition_hash.clone(),
            delta: VersionDelta {
                descriptor: base_manifest.descriptor.clone(),
                input_anchors: anchors,
            },
            fragments: vec![FragmentRef {
                url: parquet_url.as_str().to_string(),
                version: b,
                rows: record.row_count,
                digest: base_manifest.artifact.clone(),
            }],
            segments,
            deletes: None,
            live_rows: record.row_count,
            masked_rows: 0,
            identity: identity.clone(),
            produced_by: jammi_db::store::run_id().to_string(),
            produced_at: chrono::Utc::now().to_rfc3339(),
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
        };
        let manifest_url = store
            .write_version_manifest(&parquet_url, &manifest)
            .await?;
        match self
            .catalog()
            .publish_base_version(
                &table,
                b,
                manifest_url.as_str(),
                &identity,
                record.row_count,
            )
            .await
        {
            Ok(()) => {}
            Err(JammiError::CasFailed { .. }) => {
                // A concurrent base publisher won: proceed with its version.
            }
            Err(e) => return Err(e),
        }
        let record = self
            .catalog()
            .get_result_table(&table)
            .await?
            .ok_or_else(|| JammiError::RowGone {
                table: table.clone(),
            })?;
        let Some(current) = record.current_version else {
            return Err(JammiError::CasFailed {
                table,
                status: record.status,
            });
        };
        // Idempotent: the winner's manifest must exist at its path.
        let winner_url = jammi_db::store::layout::version_manifest_url(&parquet_url, current)?;
        let handle = store.open_parquet(&winner_url)?;
        if !handle.exists(&handle.data_path()?).await? && current == b {
            store
                .write_version_manifest(&parquet_url, &manifest)
                .await?;
        }
        store.bind_result_table(self.context(), &record).await?;
        Ok(record)
    }

    fn source_query_for(&self, params: &EmbeddingParams) -> Result<String> {
        let table_name = self.find_table_name(&params.source_id)?;
        Ok(self.build_source_query(
            &params.source_id,
            &table_name,
            &params.key_column,
            &params.columns,
        ))
    }

    async fn source_plan(&self, source_query: &str) -> Result<Arc<dyn ExecutionPlan>> {
        let df = self
            .context()
            .sql(source_query)
            .await
            .map_err(|e| JammiError::Inference(format!("Failed to scan source: {e}")))?;
        df.create_physical_plan()
            .await
            .map_err(|e| JammiError::Inference(format!("Failed to create scan plan: {e}")))
    }

    /// Step 4: drain the whole source once through `KeyCheckExec` (a null key
    /// refuses at end of input, before any classification is trusted), render
    /// the key with the runner's cast, and classify every row against the
    /// current state. Duplicate source keys → `NonUniqueKey { Source }`.
    async fn classify_source(
        &self,
        table: &str,
        source_query: &str,
        params: &EmbeddingParams,
        current: &HashMap<String, ContentHash>,
        deletes: DeletePolicy,
    ) -> Result<Classified> {
        let plan = key_checked(self.source_plan(source_query).await?, &params.key_column)?;
        let mut stream = plan
            .execute(0, self.context().task_ctx())
            .map_err(JammiError::from)?;
        let mut seen: HashSet<String> = HashSet::new();
        let mut dups: HashMap<String, u64> = HashMap::new();
        let mut added = Vec::new();
        let mut changed = Vec::new();
        let mut unchanged = 0u64;
        while let Some(batch) = stream.next().await {
            let batch: RecordBatch = batch.map_err(JammiError::from)?;
            let keys = batch.column_by_name(&params.key_column).ok_or_else(|| {
                JammiError::Inference(format!("key column '{}' missing", params.key_column))
            })?;
            let keys = arrow::compute::cast(keys, &DataType::Utf8)
                .map_err(|e| JammiError::Other(format!("refresh: cast: {e}")))?;
            let keys = keys
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| JammiError::Inference("key column did not render to Utf8".into()))?;
            let hashes = batch.column_by_name(CONTENT_HASH_COLUMN).ok_or_else(|| {
                JammiError::Inference("_content_hash missing from the source scan".into())
            })?;
            let hashes = arrow::compute::cast(hashes, &DataType::Utf8)
                .map_err(|e| JammiError::Other(format!("refresh: cast: {e}")))?;
            let hashes = hashes
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    JammiError::Inference("_content_hash did not render to Utf8".into())
                })?;
            for i in 0..batch.num_rows() {
                if keys.is_null(i) {
                    // `KeyCheckExec` refuses at end of input; never classify a null.
                    continue;
                }
                let key = keys.value(i).to_string();
                if !seen.insert(key.clone()) {
                    *dups.entry(key).or_insert(1) += 1;
                    continue;
                }
                let hash = ContentHash::from_hex(hashes.value(i))?;
                match current.get(&key) {
                    None => added.push(key),
                    Some(h) if *h == hash => unchanged += 1,
                    Some(_) => changed.push(key),
                }
            }
        }
        if !dups.is_empty() {
            return Err(non_unique(table, NonUniqueScan::Source, dups));
        }
        let deleted: Vec<String> = match deletes {
            DeletePolicy::Retain => Vec::new(),
            DeletePolicy::Tombstone => {
                let mut d: Vec<String> = current
                    .keys()
                    .filter(|k| !seen.contains(*k))
                    .cloned()
                    .collect();
                d.sort();
                d
            }
        };
        Ok(Classified {
            added,
            changed,
            deleted,
            unchanged,
        })
    }

    /// Step 7: infer `keys` (an in-memory build side joined onto the source
    /// scan) through the one ordered plan shape into the version's fragment +
    /// segment. Returns the fragment reference and its segment id (`None` when
    /// zero rows were realized — the empty object is deleted), plus the
    /// realized keys.
    async fn infer_delta(
        &self,
        store: &ResultStore,
        version: &BuildingVersion,
        params: &EmbeddingParams,
        definition: &EmbeddingDefinition,
        source_query: &str,
        keys: &[String],
    ) -> Result<(Option<(FragmentRef, i64)>, HashSet<String>)> {
        let source = self.source_plan(source_query).await?;
        let key_schema = Arc::new(Schema::new(vec![Field::new(
            "_refresh_key",
            DataType::Utf8,
            false,
        )]));
        let key_batch = RecordBatch::try_new(
            Arc::clone(&key_schema),
            vec![Arc::new(StringArray::from_iter_values(
                keys.iter().map(String::as_str),
            ))],
        )
        .map_err(|e| JammiError::Other(format!("refresh: key batch: {e}")))?;
        let build: Arc<dyn ExecutionPlan> =
            MemorySourceConfig::try_new_exec(&[vec![key_batch]], key_schema, None)?;
        let left_key = col("_refresh_key", build.schema().as_ref())?;
        let right_key: Arc<dyn PhysicalExpr> = Arc::new(CastExpr::new(
            col(&params.key_column, source.schema().as_ref())?,
            DataType::Utf8,
            None,
        ));
        let join: Arc<dyn ExecutionPlan> = Arc::new(HashJoinExec::try_new(
            build,
            source,
            vec![(left_key, right_key)],
            None,
            &datafusion::common::JoinType::Inner,
            None,
            PartitionMode::CollectLeft,
            datafusion::common::NullEquality::NullEqualsNothing,
        )?);
        let input = ordered_input(join, &params.key_column)?;
        let inference_exec = InferenceExecBuilder::new(
            input,
            definition.model_source.clone(),
            params.task,
            params.columns.clone(),
            params.key_column.clone(),
            params.source_id.clone(),
            Arc::clone(self.model_cache()),
        )
        .batch_size(self.inner_config().inference.batch_size)
        .observer(self.observer().clone())
        .embedding_dim(Some(definition.embedding_dim))
        .passthrough(vec![CONTENT_HASH_COLUMN.to_string()])
        .build()?;

        let fragment_url = version.fragment_url()?;
        let schema = jammi_db::store::schema::embedding_table_schema(definition.embedding_dim);
        let writer = store.open_writer(&fragment_url, schema).await?;
        let ann_config = store.ann_config();
        let sidecar = SidecarIndex::new(
            definition.embedding_dim,
            ann_config,
            version.storage_precision(),
        )?;
        let mut sink = ResultSink::for_version_fragment(writer, sidecar);

        let stream = inference_exec
            .execute(0, self.context().task_ctx())
            .map_err(JammiError::from)?;
        let batches = datafusion::physical_plan::common::collect(stream)
            .await
            .map_err(JammiError::from)?;
        let mut realized: HashSet<String> = HashSet::new();
        for batch in &batches {
            if !version.is_live() {
                drop(sink);
                return Err(JammiError::LeaseLost {
                    table: version.table_name().to_string(),
                });
            }
            realized.extend(sink.write_batch(batch).await?);
        }
        let (rows, index) = sink.finalize().await?;
        let handle = store.open_parquet(&fragment_url)?;
        if rows == 0 {
            handle.delete_if_exists(&handle.data_path()?).await?;
            return Ok((None, realized));
        }
        let segment_id = match index {
            Some(idx) => version.append_segment(&idx).await?.0,
            None => {
                return Err(JammiError::Inference(
                    "refresh: realized rows but built no index".into(),
                ))
            }
        };
        let bytes = handle.get_bytes(&handle.data_path()?).await?;
        let fragment = FragmentRef {
            url: fragment_url.as_str().to_string(),
            version: version.version(),
            rows,
            digest: ArtifactDigest::of_bytes(&bytes),
        };
        Ok((Some((fragment, segment_id)), realized))
    }

    /// After a publish: re-bind the table (the new version's provider),
    /// evict the loaded segment sets, and invalidate the ANN cache for the
    /// source.
    async fn after_publish(&self, store: &ResultStore, table: &str, source_id: &str) -> Result<()> {
        let record = self
            .catalog()
            .get_result_table(table)
            .await?
            .ok_or_else(|| JammiError::RowGone {
                table: table.to_string(),
            })?;
        store.bind_result_table(self.context(), &record).await?;
        self.ann_cache().invalidate_source(source_id)?;
        Ok(())
    }

    /// Rewrite the current version's live rows as one fragment + one segment
    /// (no inference), publishing a new version whose chain identity folds
    /// the parent's. The threshold is the consumer's; `live_rows` /
    /// `masked_rows` on every report are the inputs to that decision.
    pub async fn compact_embeddings(self: &Arc<Self>, table: &str) -> Result<RefreshReport> {
        let store = self.result_store();
        let ctx = self.context();
        let record = self.refreshable_record(table).await?;
        let descriptor = store.producing_descriptor(&record).await?;
        let params = embedding_params(table, &descriptor)?;
        let record = self.ensure_base_version(&store, record, &params).await?;
        let parent_version = record
            .current_version
            .expect("a base version is published before any compaction");
        let parquet_url = StorageUrl::parse(&record.parquet_path)?;
        let parent = store
            .read_version_manifest(&record.table_name, &parquet_url, parent_version)
            .await?
            .ok_or_else(|| JammiError::VersionUnavailable {
                table: record.table_name.clone(),
                version: parent_version,
            })?;
        let dimensions = params.dimensions;

        let mut version = store.allocate_version(&record).await?;
        let n = version.version();

        // Every live row, in `_row_id` order, through the masked provider.
        let batches = ctx
            .sql(&format!(
                "SELECT * FROM \"jammi.{}\" ORDER BY _row_id",
                record.table_name
            ))
            .await
            .map_err(JammiError::from)?
            .collect()
            .await
            .map_err(JammiError::from)?;
        let schema = jammi_db::store::schema::embedding_table_schema(dimensions);
        let fragment_url = version.fragment_url()?;
        let mut writer = store
            .open_writer(&fragment_url, Arc::clone(&schema))
            .await?;
        let mut index =
            SidecarIndex::new(dimensions, store.ann_config(), version.storage_precision())?;
        let mut rows = 0usize;
        for batch in &batches {
            let batch = coerce_to_embedding_schema(batch, &schema)?;
            writer.write_batch(&batch).await?;
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let mut vectors = Vec::new();
            jammi_db::store::vectors::extend_with_fixed_size_list_f32(
                &batch,
                &record.table_name,
                "vector",
                &mut vectors,
            )?;
            for (i, v) in vectors.iter().enumerate() {
                index.add(ids.value(i), v)?;
            }
            rows += batch.num_rows();
        }
        writer.close().await?;
        let handle = store.open_parquet(&fragment_url)?;
        let bytes = handle.get_bytes(&handle.data_path()?).await?;
        let digest = ArtifactDigest::of_bytes(&bytes);
        let mut segments = Vec::new();
        if rows > 0 {
            index.build()?;
            let seg = version.append_segment(&index).await?;
            segments.push(SegmentRef {
                segment_id: seg.0,
                version: n,
            });
        }
        let fragments = vec![FragmentRef {
            url: fragment_url.as_str().to_string(),
            version: n,
            rows,
            digest,
        }];
        let descriptor = ProducingDescriptor::EmbeddingCompaction {
            model_id: params.model_id.clone(),
            task: params.task,
            source_id: params.source_id.clone(),
            columns: params.columns.clone(),
            key_column: params.key_column.clone(),
            dimensions,
            parent_version,
            parent_identity: parent.identity.clone(),
        };
        let definition_hash = DefinitionHash(record.definition_hash.clone().unwrap_or_default());
        let identity = VersionManifest::compute_identity(
            &parent.identity,
            &definition_hash,
            &descriptor,
            &fragments,
            None,
        )?;
        let anchors = vec![InputAnchor::result_digest(
            &record.table_name,
            &ArtifactDigest(parent.identity.clone()),
        )];
        let manifest = VersionManifest {
            version_format: jammi_db::store::version::VERSION_FORMAT,
            table: record.table_name.clone(),
            version: n,
            parent: Some(parent_version),
            definition_hash,
            delta: VersionDelta {
                descriptor,
                input_anchors: anchors,
            },
            fragments,
            segments,
            deletes: None,
            live_rows: rows,
            masked_rows: 0,
            identity,
            produced_by: jammi_db::store::run_id().to_string(),
            produced_at: chrono::Utc::now().to_rfc3339(),
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
        };
        store
            .write_version_manifest(&parquet_url, &manifest)
            .await?;
        // `result_tables.input_anchors_json` stays the parent's: a compaction
        // reads no source.
        let anchors_json = record
            .input_anchors_json
            .clone()
            .unwrap_or_else(|| "[]".into());
        if let Err(e) = version
            .publish(&manifest.identity, rows, 0, &anchors_json)
            .await
        {
            if let Err(abort_err) = version.abort().await {
                tracing::warn!(table = record.table_name, version = n, error = %abort_err, "compact: abort after a missed publish did not complete");
            }
            return Err(e);
        }
        self.after_publish(&store, &record.table_name, &params.source_id)
            .await?;
        Ok(RefreshReport {
            table: record.table_name.clone(),
            version: Some(n),
            parent_version: Some(parent_version),
            inferred_rows: 0,
            added: 0,
            changed: 0,
            deleted: 0,
            unchanged: rows as u64,
            dropped_rows: 0,
            live_rows: rows as u64,
            masked_rows: 0,
            outcome: RefreshOutcome::Published,
        })
    }

    /// Delete every version row `< before` that is not the current version
    /// (`ready` or `failed`), then reap its manifest, deletes and every
    /// fragment / segment stamped with it that the CURRENT manifest does not
    /// list. `{table}.parquet`, `.materialization.json` and `next_version` are
    /// never touched.
    pub async fn expire_versions(
        self: &Arc<Self>,
        table: &str,
        before: i64,
    ) -> Result<ExpiryReport> {
        let store = self.result_store();
        let record = self.refreshable_record(table).await?;
        let Some(current) = record.current_version else {
            return Ok(ExpiryReport {
                table: table.to_string(),
                expired_versions: Vec::new(),
                objects_deleted: 0,
            });
        };
        let parquet_url = StorageUrl::parse(&record.parquet_path)?;
        let manifest = store
            .read_version_manifest(&record.table_name, &parquet_url, current)
            .await?
            .ok_or_else(|| JammiError::VersionUnavailable {
                table: record.table_name.clone(),
                version: current,
            })?;
        let retained_fragments: HashSet<String> =
            manifest.fragments.iter().map(|f| f.url.clone()).collect();
        let retained_segments: HashSet<i64> =
            manifest.segments.iter().map(|s| s.segment_id).collect();
        let mut expired = Vec::new();
        let mut objects_deleted = 0u64;
        for row in self.catalog().list_result_table_versions(table).await? {
            if row.version >= before
                || row.version == current
                || !(row.status == ResultTableStatus::Ready.to_string()
                    || row.status == ResultTableStatus::Failed.to_string())
            {
                continue;
            }
            if !self
                .catalog()
                .delete_result_table_version(table, row.version)
                .await?
            {
                continue;
            }
            objects_deleted += store
                .reap_expired_version(
                    &parquet_url,
                    &record.table_name,
                    row.version,
                    &retained_fragments,
                    &retained_segments,
                )
                .await? as u64;
            expired.push(row.version);
        }
        expired.sort_unstable();
        Ok(ExpiryReport {
            table: table.to_string(),
            expired_versions: expired,
            objects_deleted,
        })
    }
}

/// A `SELECT *` over the masked provider comes back with the scan's view
/// types (`Utf8View`) and the reader's field nullability; a compaction writes
/// the table's canonical embedding schema, so every column is cast to it.
fn coerce_to_embedding_schema(
    batch: &RecordBatch,
    schema: &arrow::datatypes::SchemaRef,
) -> Result<RecordBatch> {
    let mut columns = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        let col = batch
            .column_by_name(field.name())
            .ok_or_else(|| JammiError::Schema {
                table: String::new(),
                column: field.name().clone(),
                expected: format!("{}", field.data_type()),
                actual: "missing".into(),
            })?;
        columns.push(
            arrow::compute::cast(col, field.data_type())
                .map_err(|e| JammiError::Other(format!("compaction: cast: {e}")))?,
        );
    }
    RecordBatch::try_new(Arc::clone(schema), columns)
        .map_err(|e| JammiError::Other(format!("compaction: build batch: {e}")))
}
