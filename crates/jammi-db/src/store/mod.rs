pub mod artifact;
pub mod manifest;
pub mod mutable;
pub mod schema;
pub mod vectors;

pub use artifact::{ArtifactStore, LocalArtifact};
pub use manifest::{
    AnchorKind, AnchorValue, ArtifactDigest, ComputeDevice, DefinitionHash, InputAnchor,
    ManifestError, MatchVerdict, Materialization, MaterializationEnv, MaterializationManifest,
    ModelIdentity, ProducingDescriptor,
};

use std::path::Path;
use std::sync::Arc;

use arrow::array::Array;
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::prelude::SessionContext;
use datafusion::sql::TableReference;
use tracing::warn;

use crate::catalog::result_repo::{CreateResultTableParams, ResultTableKind, ResultTableRecord};
use crate::catalog::status::ResultTableStatus;
use crate::catalog::Catalog;
use crate::config::AnnIndexConfig;
use crate::error::{JammiError, Result};
use crate::index::sidecar::SidecarIndex;
use crate::index::VectorIndex;
use crate::model_task::ModelTask;
use crate::storage::sidecar_layout::SidecarKind;
use crate::storage::{
    self, JammiObjectStore, ObjectParquetWriter, Scheme, StorageRegistry, StorageUrl,
};
use crate::tenant_scope::TenantBinding;

/// The catalog-row provenance of an embedding result table
/// [`ResultStore::materialize_embedding_table`] writes — *what* the table is in
/// the catalog, distinct from the [`Materialization`] descriptor that captures
/// *how* its data was computed.
///
/// Groups the values the catalog row needs verbatim: the `source_id` the output
/// rows belong to, the `model_id` that records the derivation provenance (the
/// context-set encoder or propagation kernel, not a foundation model), the
/// `derived_from` FK-lineage anchor naming the source embedding table this was
/// computed from (`None` when no single source table backs the whole batch),
/// and the embedding `dimensions`. These are *not* derived from the descriptor:
/// the catalog's `source_id` / `derived_from` are its own lineage columns, which
/// a producer may anchor differently from the descriptor's internal source
/// fields, so the row carries them explicitly.
#[derive(Debug)]
pub struct EmbeddingTableSpec<'a> {
    /// The source the output rows belong to (catalog `source_id`).
    pub source_id: &'a str,
    /// The derivation provenance recorded as the catalog `model_id`.
    pub model_id: &'a str,
    /// The source embedding result table this output was derived from — the
    /// FK-lineage anchor. `None` when no single source table backs the batch.
    pub derived_from: Option<&'a str>,
    /// The embedding width of every output vector.
    pub dimensions: usize,
}

/// Returned by [`ResultStore::create_table`] — the generated paths and name
/// for a new result table, before any data has been written.
#[derive(Debug)]
pub struct ResultTableInfo {
    /// Unique table identifier (schema-qualified by the engine when registering
    /// with DataFusion).
    pub table_name: String,
    /// Storage URL for the Parquet object — open via [`ResultStore::open_parquet`].
    pub parquet_url: StorageUrl,
    /// Storage URL for the sidecar-index *base* (no extension; the layout
    /// helpers append `.usearch`/`.rowmap`/`.manifest.json`). `None` for
    /// non-embedding tables.
    pub index_url: Option<StorageUrl>,
}

/// Coordinates Parquet storage, ANN indexes, DataFusion registration,
/// catalog metadata, and crash recovery for result tables.
///
/// Wraps a `StorageUrl` as the root prefix every new table is created under.
/// File scheme keeps the historical `{artifact_dir}/jammi_db/` layout;
/// `s3://bucket/jammi_db/`, `gs://...`, `azure://...` work without code
/// change because every read/write goes through [`StorageRegistry`].
pub struct ResultStore {
    root: StorageUrl,
    registry: StorageRegistry,
    catalog: Arc<Catalog>,
    /// HNSW tuning for every sidecar index this store builds and loads — the
    /// deployment's [`AnnIndexConfig`], applied at build time (recovery and
    /// materialization) and re-applied to the query-time dial on load.
    ann: AnnIndexConfig,
}

/// Sanitize a model ID for use in file names.
///
/// Replaces every character that would be ambiguous in a path with `_`:
/// `/`, `:`, ` ` (component separators / scheme delimiter / shell-unsafe),
/// and `.` (interpreted by [`std::path::Path`] as an extension delimiter,
/// which silently truncates sidecar filenames when the model-id path
/// contains a dot — e.g. a `local:/path/with/.cache/model` source).
fn sanitize_model_id(model_id: &str) -> String {
    model_id
        .chars()
        .map(|c| {
            if c == '/' || c == ':' || c == ' ' || c == '.' {
                '_'
            } else {
                c
            }
        })
        .take(64)
        .collect()
}

impl ResultStore {
    /// Construct a result-store rooted at a local artifact directory. The
    /// directory is created if absent. Equivalent to
    /// `ResultStore::with_root(StorageUrl::parse(artifact_dir.join("jammi_db"))?, …)`
    /// with a default-constructed [`StorageRegistry`].
    pub fn new(artifact_dir: &Path, catalog: Arc<Catalog>, ann: AnnIndexConfig) -> Result<Self> {
        let jammi_db_dir = artifact_dir.join("jammi_db");
        std::fs::create_dir_all(&jammi_db_dir)?;
        let url = StorageUrl::parse(
            jammi_db_dir
                .to_str()
                .ok_or_else(|| JammiError::Config("Non-UTF8 artifact_dir".into()))?,
        )?;
        Ok(Self {
            root: url,
            registry: StorageRegistry::new(),
            catalog,
            ann,
        })
    }

    /// Construct a result-store rooted at an arbitrary [`StorageUrl`] —
    /// the path on `cloud://` schemes a deployment uses for shared
    /// result-table storage. The registry is shared with the engine
    /// session so callers register cloud credentials once.
    pub fn with_root(
        root: StorageUrl,
        registry: StorageRegistry,
        catalog: Arc<Catalog>,
        ann: AnnIndexConfig,
    ) -> Result<Self> {
        if root.scheme() == Scheme::File {
            // Ensure the directory exists so create_table doesn't fail on
            // the first write. Cloud schemes are bucket-rooted and have no
            // directory concept.
            let path = root.path();
            std::fs::create_dir_all(path)?;
        }
        Ok(Self {
            root,
            registry,
            catalog,
            ann,
        })
    }

    /// The catalog this store writes result-table rows through. Read accessor
    /// for callers that hold a `ResultStore` and need the same catalog handle
    /// (e.g. to resolve a `ResultTableRecord` by name before verifying it).
    pub fn catalog(&self) -> &Arc<Catalog> {
        &self.catalog
    }

    /// Open the [`JammiObjectStore`] handle for a result-table Parquet URL.
    pub fn open_parquet(&self, url: &StorageUrl) -> Result<JammiObjectStore> {
        let driver = self.registry.driver_for(url, None)?;
        Ok(JammiObjectStore::new(driver, url.clone()))
    }

    /// Open the handle for a sidecar-index base URL (no extension). The
    /// returned handle's `sibling_path(...)` resolves the `.usearch`,
    /// `.rowmap`, `.manifest.json` siblings.
    pub fn open_index(&self, url: &StorageUrl) -> Result<JammiObjectStore> {
        let driver = self.registry.driver_for(url, None)?;
        Ok(JammiObjectStore::new(driver, url.clone()))
    }

    /// Generate URLs and register a new result table in the catalog with
    /// status = 'building'.
    ///
    /// `kind` discriminates a direct model output from a derivation of another
    /// result table (e.g. a neighbor-graph edge relation); `derived_from` names
    /// the source result table a derivation was computed from (`None` for a
    /// `Model` table). A non-`Model` table gets `index_url = None` — no sidecar
    /// index is built for it — regardless of its `task`.
    #[allow(clippy::too_many_arguments)]
    pub async fn create_table(
        &self,
        source_id: &str,
        task: ModelTask,
        kind: ResultTableKind,
        derived_from: Option<&str>,
        model_id: &str,
        dimensions: Option<i32>,
        key_column: Option<&str>,
        text_columns: Option<&str>,
    ) -> Result<ResultTableInfo> {
        let sanitized = sanitize_model_id(model_id);
        let timestamp = chrono::Utc::now().format("%Y%m%dT%H%M%S%9f");
        // Nanoseconds plus a short uuid suffix make table names unique even
        // when two tokio tasks call create_table within the same nanosecond
        // (concurrent embedding generation on the same source).
        let suffix = &uuid::Uuid::new_v4().simple().to_string()[..8];
        let task_str = task.as_db_str();
        let table_name = format!("{source_id}__{task_str}__{sanitized}__{timestamp}_{suffix}");

        let parquet_url = self.derive_url(&format!("{table_name}.parquet"))?;
        // A sidecar index exists only for a model embedding table. A derived
        // table (a neighbor-graph edge relation) is searched as a plain
        // relation, never through an ANN sidecar — it maps to
        // `SidecarKind::None` at the storage layer.
        let index_url = if matches!(kind, ResultTableKind::Model) && task.is_embedding() {
            // Index base path has no extension — the sidecar layout helpers
            // append .usearch / .rowmap / .manifest.json.
            Some(self.derive_url(&format!("{table_name}.idx"))?)
        } else {
            None
        };

        self.catalog
            .create_result_table(CreateResultTableParams {
                table_name: &table_name,
                source_id,
                model_id,
                task,
                kind,
                derived_from,
                parquet_path: parquet_url.as_str(),
                index_path: index_url.as_ref().map(|u| u.as_str()),
                dimensions,
                key_column,
                text_columns,
            })
            .await?;

        Ok(ResultTableInfo {
            table_name,
            parquet_url,
            index_url,
        })
    }

    /// Open an [`ObjectParquetWriter`] for the result-table Parquet URL.
    pub async fn open_writer(
        &self,
        url: &StorageUrl,
        schema: arrow::datatypes::SchemaRef,
    ) -> Result<ObjectParquetWriter> {
        let handle = self.open_parquet(url)?;
        Ok(ObjectParquetWriter::open(&handle, schema).await?)
    }

    /// Register an existing Parquet object as a DataFusion table.
    pub async fn register_table(
        &self,
        ctx: &SessionContext,
        name: &str,
        url: &StorageUrl,
    ) -> Result<()> {
        register_parquet_table(ctx, &self.registry, name, url).await
    }

    /// Finalize a result table behind its materialization contract: the single
    /// `building -> ready` transition every producer routes through.
    ///
    /// Builds the [`MaterializationManifest`] from the producer's
    /// [`ProducingDescriptor`], the output-affecting [`MaterializationEnv`], the
    /// resolved [`InputAnchor`]s, and the written Parquet's freshly-computed
    /// [`ArtifactDigest`], then performs the publish in this crash-safe order:
    ///
    /// 1. compute the artifact digest over the durable Parquet bytes;
    /// 2. write the `.materialization.json` sidecar (a sibling of the Parquet,
    ///    distinct from the ANN `.manifest.json` index sidecar);
    /// 3. register the table in DataFusion; and
    /// 4. flip `building -> ready` **and** persist the catalog summary columns
    ///    (`definition_hash`, `input_anchors_json`) in one transaction.
    ///
    /// The sidecar is written *before* the status flip — the same boundary the
    /// ANN sidecar uses — so a crash never leaves a `ready` table without a
    /// manifest. A crash between (1) and (2) leaves a `building` row whose
    /// Parquet is valid but whose manifest never landed; recovery reconciles
    /// that to `failed` (the producing descriptor cannot be reconstructed),
    /// never a manifest-less promotion.
    ///
    /// This is the sole `building -> ready` path: there is no manifest-free
    /// finalize. Every result-table producer — inference, the embedding
    /// pipeline, the neighbor-graph derivation, and the
    /// [`Self::materialize_embedding_table`] producers (graph propagation,
    /// context sets) — supplies a descriptor, an environment, and its inputs
    /// here, so no table escapes without an attestation.
    pub async fn finalize_with_manifest(
        &self,
        ctx: &SessionContext,
        name: &str,
        url: &StorageUrl,
        rows: usize,
        materialization: Materialization<'_>,
    ) -> Result<MaterializationManifest> {
        let parquet_handle = self.open_parquet(url)?;
        let parquet_path = parquet_handle.data_path()?;
        let bytes = parquet_handle.get_bytes(&parquet_path).await?;
        let digest = ArtifactDigest::of_bytes(&bytes);

        let manifest = MaterializationManifest::compute(
            materialization.descriptor,
            materialization.env,
            materialization.inputs,
            digest,
            run_id().to_string(),
            chrono::Utc::now().to_rfc3339(),
        )
        .map_err(manifest_to_jammi)?;

        // Crash window the contract must survive: the Parquet is durable but the
        // manifest is not yet written and the status flip has not committed.
        #[cfg(feature = "test-hooks")]
        crate::store::mutable::test_hook::maybe_signal_materialization().await;

        self.write_materialization_sidecar(url, &manifest).await?;

        self.register_table(ctx, name, url).await?;

        let anchors_json = serde_json::to_string(&manifest.input_anchors)
            .map_err(|e| JammiError::Other(format!("serialise input anchors: {e}")))?;
        self.catalog
            .promote_result_table_with_manifest(
                name,
                rows,
                manifest.definition_hash.as_str(),
                &anchors_json,
            )
            .await?;
        Ok(manifest)
    }

    /// Resolve the [`InputAnchor`] for an immutable result-table input: its
    /// content digest is its anchor ([`AnchorKind::ResultDigest`]). Prefers the
    /// digest the input's own manifest already attests (no re-read); falls back
    /// to recomputing it from the input's Parquet bytes for a pre-contract
    /// source table that carries no manifest.
    pub async fn result_digest_anchor(&self, table: &ResultTableRecord) -> Result<InputAnchor> {
        let parquet_url = StorageUrl::parse(&table.parquet_path)?;
        let digest = match self.read_materialization_manifest(&parquet_url).await? {
            Some(m) => m.artifact,
            None => {
                let handle = self.open_parquet(&parquet_url)?;
                let path = handle.data_path()?;
                let bytes = handle.get_bytes(&path).await?;
                ArtifactDigest::of_bytes(&bytes)
            }
        };
        Ok(InputAnchor::result_digest(&table.table_name, &digest))
    }

    /// Read a result table's `.materialization.json` sidecar, if present.
    ///
    /// Returns `Ok(None)` when no sidecar exists — a pre-contract table, or one
    /// whose write was torn before the manifest landed. The caller distinguishes
    /// those via the catalog summary columns.
    pub async fn read_materialization_manifest(
        &self,
        parquet_url: &StorageUrl,
    ) -> Result<Option<MaterializationManifest>> {
        let handle = self.open_parquet(parquet_url)?;
        let sidecar = materialization_sidecar_path(&handle)?;
        if !handle.exists(&sidecar).await? {
            return Ok(None);
        }
        let bytes = handle.get_bytes(&sidecar).await?;
        let manifest =
            MaterializationManifest::from_json_bytes(&bytes).map_err(manifest_to_jammi)?;
        Ok(Some(manifest))
    }

    /// Write a result table's `.materialization.json` sidecar.
    async fn write_materialization_sidecar(
        &self,
        parquet_url: &StorageUrl,
        manifest: &MaterializationManifest,
    ) -> Result<()> {
        let handle = self.open_parquet(parquet_url)?;
        let sidecar = materialization_sidecar_path(&handle)?;
        let bytes = manifest.to_json_bytes().map_err(manifest_to_jammi)?;
        handle.put_bytes(&sidecar, bytes.into()).await?;
        Ok(())
    }

    /// Recompute a `ready` result table's artifact digest and check it (and, if
    /// given, an expected definition hash) against its manifest sidecar. The
    /// read-only `verify_materialization` verb. Returns a [`MatchVerdict`]; it
    /// never acts on one (refuse / alarm / fall back is the consumer's policy).
    ///
    /// The verdict attests the Parquet **data**, never the ANN search index.
    pub async fn verify_materialization(
        &self,
        table: &ResultTableRecord,
        expected_definition: Option<&DefinitionHash>,
    ) -> Result<MatchVerdict> {
        let parquet_url = StorageUrl::parse(&table.parquet_path)?;
        let Some(manifest) = self.read_materialization_manifest(&parquet_url).await? else {
            // No sidecar: a pre-contract table (truthful unknown) — distinct from
            // a post-contract table that *should* carry one (a torn write or a
            // bypassed funnel), which recovery reconciles, not this read path.
            return Ok(MatchVerdict::MissingManifest);
        };

        let handle = self.open_parquet(&parquet_url)?;
        let path = handle.data_path()?;
        let bytes = handle.get_bytes(&path).await?;
        let recomputed = ArtifactDigest::of_bytes(&bytes);

        if recomputed != manifest.artifact {
            return Ok(MatchVerdict::Mismatch {
                expected: manifest.artifact.0,
                found: recomputed.0,
            });
        }

        if let Some(expected) = expected_definition {
            if *expected != manifest.definition_hash {
                return Ok(MatchVerdict::Mismatch {
                    expected: expected.0.clone(),
                    found: manifest.definition_hash.0,
                });
            }
        }

        let unpinned = manifest.unpinned_inputs();
        if unpinned.is_empty() {
            Ok(MatchVerdict::Match)
        } else {
            Ok(MatchVerdict::MatchWithUnpinnedInputs { unpinned })
        }
    }

    /// Reconcile every result table left `building` by a crash, restoring the
    /// crash-consistency invariant of the catalog↔result-storage boundary.
    ///
    /// # Guarantee
    ///
    /// **Crash-consistent eventual reconciliation.** Object storage cannot join
    /// the catalog transaction, so a table is published in two steps: the bytes
    /// (Parquet + sidecar) are written first, then a single catalog row flips
    /// `building → ready`. The status gate makes that boundary crash-safe
    /// without a distributed transaction:
    ///
    /// - **No half-written table is ever queryable.** Only a `ready` row is
    ///   loaded into DataFusion ([`Self::load_existing_tables`]); a `building`
    ///   or `failed` row is never registered, so a crash mid-write leaves
    ///   nothing addressable.
    /// - **Reconciliation is terminal.** This sweep visits every `building` row
    ///   and drives it to exactly one terminal state — `ready` if its bytes are
    ///   a fully-valid closed Parquet (promoted with the *true* footer row
    ///   count, and the ANN sidecar rebuilt from the Parquet so an embedding
    ///   table self-heals even if its sidecar never landed), `failed` otherwise
    ///   (missing bytes, or a torn/partial Parquet whose bytes are then reaped).
    ///   No row is left `building`.
    /// - **A promoted row's `row_count` is the truth on disk**, read from the
    ///   Parquet footer — never the count the writer *intended* before it
    ///   crashed.
    ///
    /// The sweep is idempotent: re-running it after it has reconciled every
    /// `building` row is a no-op.
    ///
    /// # Cross-tenant scope
    ///
    /// Recovery runs under [`crate::session::JammiSession::with_admin_scope`] so
    /// it enumerates and reconciles `building` orphans owned by **every**
    /// tenant, not only the (unscoped, GLOBAL) startup session's own rows. Each
    /// promoted/failed row keeps its own `tenant_id`; the bypass is confined to
    /// this sweep and clears the instant it returns.
    ///
    /// # Durability boundary
    ///
    /// Both catalog backends replay their write-ahead log on restart, so a
    /// *process* crash never loses a committed `building → ready` (or the
    /// `building` insert that recovery later reconciles): the row that was
    /// durably committed before the crash is present after it. The backends
    /// differ only under host **power loss**: Postgres defaults to a synchronous
    /// commit (`fsync`), so a committed transaction survives power loss;
    /// SQLite runs `synchronous=NORMAL` under WAL, which fsyncs at checkpoint
    /// but not on every commit, so a power loss can lose the last committed
    /// transaction(s) since the previous checkpoint. That is a property of the
    /// catalog's durability setting, not of this reconciliation — whatever the
    /// catalog durably retained, recovery reconciles consistently against the
    /// bytes on disk.
    pub async fn recover(&self) -> Result<()> {
        TenantBinding::admin_scope(self.recover_inner()).await
    }

    /// The cross-tenant reconciliation loop, run inside [`Self::recover`]'s
    /// admin scope so the catalog enumeration and the per-row status flips both
    /// see and write across every tenant's `building` rows.
    async fn recover_inner(&self) -> Result<()> {
        let building = self
            .catalog
            .list_result_tables_by_status(ResultTableStatus::Building)
            .await?;
        for table in building {
            let parquet_url = StorageUrl::parse(&table.parquet_path)?;
            let parquet_handle = self.open_parquet(&parquet_url)?;
            let parquet_path = parquet_handle.data_path()?;
            let parquet_exists = parquet_handle.exists(&parquet_path).await?;
            let parquet_valid =
                parquet_exists && storage::reader::is_valid_parquet(&parquet_handle).await?;

            if !parquet_exists {
                warn!(
                    table = table.table_name,
                    "Recovery: Parquet missing, marking failed"
                );
                self.catalog
                    .update_result_table_status(&table.table_name, ResultTableStatus::Failed, 0)
                    .await?;
            } else if !parquet_valid {
                warn!(
                    table = table.table_name,
                    "Recovery: invalid Parquet, deleting and marking failed"
                );
                parquet_handle.delete_if_exists(&parquet_path).await.ok();
                if let Some(ref idx) = table.index_path {
                    let idx_url = StorageUrl::parse(idx)?;
                    let idx_handle = self.open_index(&idx_url)?;
                    storage::sidecar_layout::delete_sidecar(&idx_handle, SidecarKind::Ann)
                        .await
                        .ok();
                }
                self.catalog
                    .update_result_table_status(&table.table_name, ResultTableStatus::Failed, 0)
                    .await?;
            } else if let Some(manifest) = self.read_materialization_manifest(&parquet_url).await? {
                // The manifest sidecar is present (written before the flip), so
                // its summary columns can be backfilled as part of the same
                // promotion the live path performs.
                let row_count = storage::reader::count_parquet_rows(&parquet_handle).await?;
                // Rebuild ANN index if this is an embedding table
                if table.task.is_embedding() {
                    if let Some(ref idx_path) = table.index_path {
                        let idx_url = StorageUrl::parse(idx_path)?;
                        if let Err(e) = self
                            .rebuild_index_from_parquet(
                                &parquet_handle,
                                &idx_url,
                                table.dimensions.unwrap_or(0) as usize,
                            )
                            .await
                        {
                            warn!(
                                table = table.table_name,
                                error = %e,
                                "Recovery: failed to rebuild index, proceeding without"
                            );
                        }
                    }
                }
                let anchors_json = serde_json::to_string(&manifest.input_anchors)
                    .map_err(|e| JammiError::Other(format!("serialise input anchors: {e}")))?;
                self.catalog
                    .promote_result_table_with_manifest(
                        &table.table_name,
                        row_count,
                        manifest.definition_hash.as_str(),
                        &anchors_json,
                    )
                    .await?;
            } else {
                // Valid Parquet but NO manifest sidecar: the write was torn in
                // the window between the Parquet landing and the manifest being
                // written (before the `building -> ready` flip). The contract
                // forbids promoting a table without an attestation, and the
                // producing descriptor cannot be reconstructed here — so this
                // row is reaped to `failed`, not promoted manifest-less.
                warn!(
                    table = table.table_name,
                    "Recovery: valid Parquet but no materialization manifest \
                     (torn write before manifest); deleting and marking failed"
                );
                parquet_handle.delete_if_exists(&parquet_path).await.ok();
                if let Some(ref idx) = table.index_path {
                    let idx_url = StorageUrl::parse(idx)?;
                    let idx_handle = self.open_index(&idx_url)?;
                    storage::sidecar_layout::delete_sidecar(&idx_handle, SidecarKind::Ann)
                        .await
                        .ok();
                }
                self.catalog
                    .update_result_table_status(&table.table_name, ResultTableStatus::Failed, 0)
                    .await?;
            }
        }

        self.reconcile_ready_manifests().await?;
        Ok(())
    }

    /// Reconcile already-`ready` result tables against the materialization
    /// contract: a post-contract row (one whose catalog `definition_hash` is
    /// set, so it was promoted under the contract) whose `.materialization.json`
    /// sidecar is now absent is a corruption — the attestation a verifier would
    /// read is gone. Such a row is driven to `failed` and its bytes reaped,
    /// rather than left queryable with a silently-missing manifest.
    ///
    /// A **pre-contract** row (catalog `definition_hash IS NULL`, created before
    /// migration 021) legitimately has no sidecar; it is left untouched and
    /// verifies as an honest [`MatchVerdict::MissingManifest`]. This is the
    /// distinction the contract requires: a bug (post-contract, no sidecar) is
    /// reaped; a legitimate historical table is preserved.
    async fn reconcile_ready_manifests(&self) -> Result<()> {
        let ready = self
            .catalog
            .list_result_tables_by_status(ResultTableStatus::Ready)
            .await?;
        for table in ready {
            // Only a post-contract row (summary column set) is expected to carry
            // a sidecar; a pre-contract row legitimately does not.
            if table.definition_hash.is_none() {
                continue;
            }
            let parquet_url = StorageUrl::parse(&table.parquet_path)?;
            let handle = self.open_parquet(&parquet_url)?;
            let sidecar = materialization_sidecar_path(&handle)?;
            if handle.exists(&sidecar).await? {
                continue;
            }
            warn!(
                table = table.table_name,
                "Recovery: post-contract ready table is missing its materialization \
                 manifest sidecar; deleting and marking failed"
            );
            let data_path = handle.data_path()?;
            handle.delete_if_exists(&data_path).await.ok();
            if let Some(ref idx) = table.index_path {
                let idx_url = StorageUrl::parse(idx)?;
                let idx_handle = self.open_index(&idx_url)?;
                storage::sidecar_layout::delete_sidecar(&idx_handle, SidecarKind::Ann)
                    .await
                    .ok();
            }
            self.catalog
                .update_result_table_status(&table.table_name, ResultTableStatus::Failed, 0)
                .await?;
        }
        Ok(())
    }

    /// Load every `ready` result table into DataFusion.
    ///
    /// Runs under an admin scope so a restart re-registers `ready` tables for
    /// **every** tenant (a single startup session is unscoped/GLOBAL and would
    /// otherwise miss tenant-owned tables).
    ///
    /// All tenants' `ready` tables share one DataFusion context, so raw `sql()`
    /// over a result table is **not** tenant-isolated — a session can resolve any
    /// `jammi.{name}` it can name. This is by design: the open-core engine targets
    /// a **trusted network** and ships no security boundary of its own.
    /// `tenant_scope` is an *organizational* mechanism (catalog-API access such as
    /// `get_result_table` is tenant-scoped); access
    /// control against a hostile principal is the consumer's responsibility — the
    /// BYO-auth seam / a governing platform — never the engine's. See the guide's
    /// security posture / threat model for the boundary.
    ///
    /// A `ready` row whose bytes are absent (a torn write that committed `ready`
    /// before the bytes were durable on a power loss) is skipped, not
    /// registered, so it is never queryable.
    pub async fn load_existing_tables(&self, ctx: &SessionContext) -> Result<()> {
        TenantBinding::admin_scope(self.load_existing_tables_inner(ctx)).await
    }

    async fn load_existing_tables_inner(&self, ctx: &SessionContext) -> Result<()> {
        let ready = self
            .catalog
            .list_result_tables_by_status(ResultTableStatus::Ready)
            .await?;
        for table in ready {
            let url = match StorageUrl::parse(&table.parquet_path) {
                Ok(u) => u,
                Err(e) => {
                    warn!(
                        table = table.table_name,
                        error = %e,
                        "Result-table parquet_path is not a valid storage URL"
                    );
                    continue;
                }
            };
            let handle = self.open_parquet(&url)?;
            let path = handle.data_path()?;
            if handle.exists(&path).await? {
                if let Err(e) = self.register_table(ctx, &table.table_name, &url).await {
                    warn!(
                        table = table.table_name,
                        error = %e,
                        "Failed to register existing table"
                    );
                }
            }
        }
        Ok(())
    }

    /// Search an embedding table for the nearest neighbors of a query vector.
    /// Uses SidecarIndex (ANN) when available, falls back to exact brute-force search.
    pub async fn search_vectors(
        &self,
        ctx: &SessionContext,
        table: &ResultTableRecord,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(String, f32)>> {
        let index = self.resolve_search_mode(table).await?;
        match index {
            Some(idx) => idx.search(query, k),
            None => {
                crate::index::exact::exact_vector_search(ctx, &table.table_name, query, k).await
            }
        }
    }

    /// Resolve whether to use ANN (sidecar index) or exact search for a table.
    /// Returns `Some(SidecarIndex)` for ANN, `None` for exact fallback.
    pub async fn resolve_search_mode(
        &self,
        table: &ResultTableRecord,
    ) -> Result<Option<SidecarIndex>> {
        let Some(ref idx_path) = table.index_path else {
            return Ok(None);
        };
        let idx_url = StorageUrl::parse(idx_path)?;
        let handle = self.open_index(&idx_url)?;
        match storage::sidecar_layout::load_sidecar(&handle, &self.ann).await {
            Ok(index) => Ok(Some(index)),
            Err(e) => {
                warn!(
                    table = table.table_name,
                    error = %e,
                    "Sidecar index unavailable, falling back to exact search"
                );
                Ok(None)
            }
        }
    }

    /// Persist a fully-built sidecar index next to the table's parquet object.
    pub async fn save_sidecar(&self, url: &StorageUrl, index: &SidecarIndex) -> Result<()> {
        let handle = self.open_index(url)?;
        storage::sidecar_layout::save_sidecar(&handle, index).await
    }

    /// Best-effort delete of a result-table's parquet object + sidecar bundle.
    /// 404 is not an error — callers (e.g. `remove_source`) are paving over
    /// already-cleaned state.
    pub async fn delete_table_files(
        &self,
        parquet_path: &str,
        index_path: Option<&str>,
    ) -> Result<()> {
        let parquet_url = StorageUrl::parse(parquet_path)?;
        let parquet_handle = self.open_parquet(&parquet_url)?;
        let path = parquet_handle.data_path()?;
        parquet_handle.delete_if_exists(&path).await?;

        if let Some(idx) = index_path {
            let idx_url = StorageUrl::parse(idx)?;
            let idx_handle = self.open_index(&idx_url)?;
            storage::sidecar_layout::delete_sidecar(&idx_handle, SidecarKind::Ann).await?;
        }
        Ok(())
    }

    /// Derive a child URL under the result-store root for an artifact name.
    fn derive_url(&self, name: &str) -> Result<StorageUrl> {
        let root_str = self.root.as_str();
        let joined = if root_str.ends_with('/') {
            format!("{root_str}{name}")
        } else {
            format!("{root_str}/{name}")
        };
        Ok(StorageUrl::parse(&joined)?)
    }

    /// Rebuild an ANN sidecar index from a Parquet object backed by an
    /// arbitrary `object_store` scheme. Used by the recovery path.
    async fn rebuild_index_from_parquet(
        &self,
        parquet_handle: &JammiObjectStore,
        index_url: &StorageUrl,
        dimensions: usize,
    ) -> Result<()> {
        if dimensions == 0 {
            return Ok(());
        }

        let batches = storage::reader::read_all_record_batches(parquet_handle).await?;
        let mut index = SidecarIndex::new(dimensions, &self.ann)?;
        for batch in batches {
            let row_ids = batch
                .column_by_name("_row_id")
                .and_then(|c| c.as_any().downcast_ref::<arrow::array::StringArray>());
            let vectors = batch.column_by_name("vector").and_then(|c| {
                c.as_any()
                    .downcast_ref::<arrow::array::FixedSizeListArray>()
            });

            if let (Some(ids), Some(vecs)) = (row_ids, vectors) {
                for i in 0..ids.len() {
                    let row_id = ids.value(i);
                    let v = vecs.value(i);
                    let float_arr = v
                        .as_any()
                        .downcast_ref::<arrow::array::Float32Array>()
                        .ok_or_else(|| JammiError::Other("Vector not Float32".into()))?;
                    let vec: Vec<f32> = (0..float_arr.len()).map(|j| float_arr.value(j)).collect();
                    index.add(row_id, &vec)?;
                }
            }
        }

        if index.len() > 0 {
            index.build()?;
            self.save_sidecar(index_url, &index).await?;
        }
        Ok(())
    }

    /// Materialise pre-pooled per-key vectors into a normal embedding-shaped
    /// result table — the `(_row_id, _source_id, _model_id, vector)` Parquet
    /// plus the sidecar ANN index every embedding table carries.
    ///
    /// The table this writes is indistinguishable from one
    /// [`crate::store::ResultStore::create_table`] produces for an embedding
    /// task: an embedding [`ModelTask`], a dimensioned `vector` column, and a
    /// sidecar index built from those vectors. Callers that pool a retrieval into
    /// a per-target context vector (S16), or aggregate features over a graph
    /// (S12), land it here so the result is searchable and joinable like any
    /// other embedding table. `model_id` is the derivation provenance (e.g. the
    /// context-set encoder, or the propagation kernel), not a foundation model.
    ///
    /// `derived_from` names the source embedding result table this output was
    /// computed from — the FK-lineage anchor. A graph propagation passes its
    /// input embedding table here so the catalog records the derivation; a caller
    /// pooling from a source's *raw* rows (no single source result table) passes
    /// `None`.
    pub async fn materialize_embedding_table(
        &self,
        ctx: &SessionContext,
        spec: EmbeddingTableSpec<'_>,
        rows: &[(String, Vec<f32>)],
        materialization: Materialization<'_>,
    ) -> Result<ResultTableRecord> {
        let EmbeddingTableSpec {
            source_id,
            model_id,
            derived_from,
            dimensions,
        } = spec;

        // A normal embedding result table (S9 vocabulary: kind='model'); the
        // task is the embedding task that drives the sidecar-index sidecar URL.
        let table_info = self
            .create_table(
                source_id,
                ModelTask::TextEmbedding,
                ResultTableKind::Model,
                derived_from,
                model_id,
                Some(dimensions as i32),
                Some("_row_id"),
                None,
            )
            .await?;

        let schema = crate::store::schema::embedding_table_schema(dimensions);
        let batch = embedding_batch(&schema, source_id, model_id, rows, dimensions)?;

        let mut writer = self.open_writer(&table_info.parquet_url, schema).await?;
        let mut index = SidecarIndex::new(dimensions, &self.ann)?;
        if !rows.is_empty() {
            writer.write_batch(&batch).await?;
            for (key, vector) in rows {
                index.add(key, vector)?;
            }
        }
        let row_count = writer.close().await?;

        if index.len() > 0 {
            index.build()?;
            if let Some(ref index_url) = table_info.index_url {
                self.save_sidecar(index_url, &index).await?;
            }
        }

        self.finalize_with_manifest(
            ctx,
            &table_info.table_name,
            &table_info.parquet_url,
            row_count,
            materialization,
        )
        .await?;

        self.catalog
            .get_result_table(&table_info.table_name)
            .await?
            .ok_or_else(|| {
                JammiError::Catalog(format!(
                    "Result table '{}' not found after materialisation",
                    table_info.table_name
                ))
            })
    }
}

/// Build the `(_row_id, _source_id, _model_id, vector)` batch for a
/// materialised embedding table from per-key vectors.
fn embedding_batch(
    schema: &arrow::datatypes::SchemaRef,
    source_id: &str,
    model_id: &str,
    rows: &[(String, Vec<f32>)],
    dimensions: usize,
) -> Result<arrow::array::RecordBatch> {
    use arrow::array::{FixedSizeListArray, Float32Array, StringArray};
    use arrow::datatypes::{DataType, Field};

    for (key, vector) in rows {
        if vector.len() != dimensions {
            return Err(JammiError::Schema {
                table: model_id.to_string(),
                column: "vector".into(),
                expected: format!("FixedSizeList<Float32> width {dimensions}"),
                actual: format!("row '{key}' has width {}", vector.len()),
            });
        }
    }

    let row_ids = StringArray::from_iter_values(rows.iter().map(|(k, _)| k.as_str()));
    let source_ids = StringArray::from_iter_values(rows.iter().map(|_| source_id));
    let model_ids = StringArray::from_iter_values(rows.iter().map(|_| model_id));
    let flat: Vec<f32> = rows.iter().flat_map(|(_, v)| v.iter().copied()).collect();
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let vectors = FixedSizeListArray::try_new(
        item,
        dimensions as i32,
        Arc::new(Float32Array::from(flat)),
        None,
    )
    .map_err(|e| JammiError::Other(format!("materialize: build vector column: {e}")))?;

    arrow::array::RecordBatch::try_new(
        Arc::clone(schema),
        vec![
            Arc::new(row_ids),
            Arc::new(source_ids),
            Arc::new(model_ids),
            Arc::new(vectors),
        ],
    )
    .map_err(|e| JammiError::Other(format!("materialize: build batch: {e}")))
}

/// The per-process producing-run identity stamped on every manifest's
/// `produced_by`. Provenance only — never the reproducibility anchor (that is
/// the input anchors). One id per engine process, generated on first use.
fn run_id() -> &'static str {
    static RUN_ID: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    RUN_ID.get_or_init(|| uuid::Uuid::new_v4().simple().to_string())
}

/// The `.materialization.json` sidecar path beside a result table's Parquet
/// object. Distinct from the ANN `.manifest.json` index sidecar
/// ([`crate::storage::sidecar_layout`]): this attests the Parquet data, that one
/// describes the search index.
fn materialization_sidecar_path(handle: &JammiObjectStore) -> Result<object_store::path::Path> {
    Ok(handle.sibling_path("materialization.json")?)
}

/// Lift a [`ManifestError`] into the engine error type. A storage failure keeps
/// its `Storage` shape; everything else is a `Catalog`-class invariant breach in
/// the contract layer.
fn manifest_to_jammi(e: ManifestError) -> JammiError {
    match e {
        ManifestError::Storage(s) => JammiError::Storage(s),
        ManifestError::Serde(s) => JammiError::Json(s),
        other => JammiError::Catalog(other.to_string()),
    }
}

/// Register a Parquet URL as a DataFusion table under `jammi.{name}`.
///
/// Ensures the underlying object_store driver is registered on the
/// `RuntimeEnv` so the URL resolves on read.
pub(crate) async fn register_parquet_table(
    ctx: &SessionContext,
    registry: &StorageRegistry,
    name: &str,
    url: &StorageUrl,
) -> Result<()> {
    use datafusion::datasource::file_format::options::ParquetReadOptions;

    // Make sure the engine's driver for this URL is the same one DataFusion
    // sees — important for cloud schemes where DataFusion's default
    // registry would otherwise build a credential-less duplicate.
    let driver = registry.driver_for(url, None)?;
    if !matches!(url.scheme(), Scheme::File | Scheme::Memory) {
        let parsed = ::url::Url::parse(url.as_str()).map_err(|e| {
            JammiError::Config(format!("Storage URL '{url}' did not re-parse: {e}"))
        })?;
        ctx.runtime_env().register_object_store(&parsed, driver);
    }

    // Register under a single bare identifier `jammi.{name}` rather than a
    // string DataFusion would re-parse as a multipart reference. A result
    // table name embeds a UTC timestamp (`…T…`) and may carry characters
    // (hyphens from a sanitized local model path) that the SQL tokenizer
    // either rejects — falling back to a case-preserved bare literal — or
    // splits on the dot into a lowercased `jammi` schema. That parser-routing
    // is name-dependent and inconsistent. The query side reaches these tables
    // through the quoted single identifier `"jammi.{name}"`, which is always a
    // bare, case-preserved literal; matching that here makes every result
    // table — embedding and edge alike — register and resolve identically.
    let table_ref = TableReference::bare(format!("jammi.{name}"));
    // Validate the URL parses as a ListingTableUrl before handing to DF.
    let _ = ListingTableUrl::parse(url.as_str())?;
    ctx.register_parquet(table_ref, url.as_str(), ParquetReadOptions::default())
        .await?;
    Ok(())
}
