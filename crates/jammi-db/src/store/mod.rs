pub mod artifact;
pub mod freshness;
pub mod manifest;
pub mod mutable;
pub mod result_schema;
pub mod schema;
pub mod vectors;

pub use artifact::{ArtifactStore, LocalArtifact};
pub use freshness::{
    CacheOutcome, CachePolicy, CurrentAnchor, DerivesFromEdge, StaleReason, Staleness,
};
pub use manifest::{
    AnchorKind, AnchorValue, ArtifactDigest, ComputeDevice, DefinitionHash, InputAnchor,
    ManifestError, MatchVerdict, Materialization, MaterializationEnv, MaterializationManifest,
    ModelIdentity, ProducingDescriptor,
};
pub use result_schema::ResultTableSchemaProvider;

use std::collections::BTreeMap;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

use arrow::array::Array;
use datafusion::catalog::SchemaProvider;
use datafusion::datasource::listing::{ListingTable, ListingTableConfig, ListingTableUrl};
use datafusion::datasource::TableProvider;
use datafusion::execution::options::ReadOptions;
use datafusion::prelude::SessionContext;
use tracing::warn;

use crate::catalog::result_repo::{CreateResultTableParams, ResultTableKind, ResultTableRecord};
use crate::catalog::status::ResultTableStatus;
use crate::catalog::Catalog;
use crate::config::{AnnIndexConfig, StoragePrecision};
use crate::error::{JammiError, Result};
use crate::index::sidecar::SidecarIndex;
use crate::index::VectorIndex;
use crate::model_task::ModelTask;
use crate::storage::sidecar_layout::SidecarKind;
use crate::storage::{
    self, JammiObjectStore, ObjectParquetWriter, Scheme, StorageRegistry, StorageUrl,
};
use crate::tenant::TenantId;
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
    /// The source key-column name recorded as catalog provenance (the catalog
    /// `key_column`). The *physical* key of every embedding table is always
    /// `_row_id`; this names which column of the origin those keys came from,
    /// so lineage survives without changing the output schema. Producers that
    /// key straight off `_row_id` pass `"_row_id"`.
    pub key_column: &'a str,
    /// The source content columns these vectors were computed from, recorded as
    /// the catalog `text_columns` provenance (joined). `None` when no source
    /// columns are attributed (a pooled or externally-produced batch).
    pub text_columns: Option<&'a str>,
}

/// The reserved [`ProducingDescriptor::External`] `params` key
/// [`ResultStore::materialize_computed_embedding_table`] folds a content digest of
/// the normalized rows into. Bare (unnamespaced) so it matches the key
/// `jammi-ai`'s import pipeline has always used for the same purpose —
/// namespacing it would change the `params` `BTreeMap`'s canonical bytes and
/// therefore the [`DefinitionHash`] of every table an existing caller already
/// produced under the old key.
pub const CONTENT_DIGEST_PARAM_KEY: &str = "content_digest";

/// Caller-supplied provenance for a computed embedding table materialized
/// through [`ResultStore::materialize_computed_embedding_table`] — the
/// [`ProducingDescriptor::External`] producer's vocabulary. The engine owns
/// only the *mechanism* (normalize, digest, materialize); the caller owns the
/// *meaning* of `producer_id`, `params`, `env`, and `inputs`, so this struct
/// carries no consumer-specific field.
#[derive(Debug, Clone)]
pub struct ComputedEmbeddingProvenance {
    /// The caller's stable identifier for the producing verb it does not ask
    /// the engine to own — an opaque label naming the external producer (its
    /// own pipeline id, e.g. `"external_import"`).
    pub producer_id: String,
    /// Every output-affecting parameter of the caller's producer, as
    /// canonical string key/value pairs. Completeness is the caller's
    /// contract — an omitted determinant silently aliases two different
    /// productions on one hash. Must **not** contain
    /// [`CONTENT_DIGEST_PARAM_KEY`]: the verb folds that key in itself from
    /// the normalized rows, and a caller-supplied value there would either be
    /// silently overwritten (a footgun) or collide — so this is rejected
    /// loudly instead.
    pub params: BTreeMap<String, String>,
    /// The output-affecting environment (engine version, compute device,
    /// invoked models) the caller's producer ran under.
    pub env: MaterializationEnv,
    /// The as-of state of every input the caller's producer read, in producer
    /// order.
    pub inputs: Vec<InputAnchor>,
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
    /// The tenant-gating schema provider every result table registers into —
    /// installed as the session context's default schema so bare `jammi.{name}`
    /// resolutions honour the catalog owner. Shares the catalog's
    /// [`TenantBinding`], so the read gate matches the catalog API's own
    /// `(tenant_id = $current OR tenant_id IS NULL)` + admin-scope bypass.
    result_schema: Arc<ResultTableSchemaProvider>,
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
        let result_schema = Arc::new(ResultTableSchemaProvider::new(
            catalog
                .tenant_binding()
                .unwrap_or_else(TenantBinding::unscoped),
        ));
        Ok(Self {
            root: url,
            registry: StorageRegistry::new(),
            catalog,
            ann,
            result_schema,
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
        let result_schema = Arc::new(ResultTableSchemaProvider::new(
            catalog
                .tenant_binding()
                .unwrap_or_else(TenantBinding::unscoped),
        ));
        Ok(Self {
            root,
            registry,
            catalog,
            ann,
            result_schema,
        })
    }

    /// The catalog this store writes result-table rows through. Read accessor
    /// for callers that hold a `ResultStore` and need the same catalog handle
    /// (e.g. to resolve a `ResultTableRecord` by name before verifying it).
    pub fn catalog(&self) -> &Arc<Catalog> {
        &self.catalog
    }

    /// The tenant-gating schema provider this store registers result tables
    /// into. A caller composing the session installs it as the query context's
    /// default schema (see [`Self::install_result_schema`]) so bare
    /// `jammi.{name}` resolutions honour the catalog owner.
    pub fn result_schema(&self) -> Arc<ResultTableSchemaProvider> {
        Arc::clone(&self.result_schema)
    }

    /// Install this store's [`ResultTableSchemaProvider`] as `ctx`'s default
    /// schema (`datafusion.public`) — the schema bare `jammi.{name}` result
    /// tables resolve through. Idempotent: re-installing the same provider
    /// preserves the tables it already holds. Registration
    /// ([`Self::register_table`]) calls this itself, so a context that only
    /// ever registers through the store need not call it; a session installs it
    /// eagerly so the provider is present even before the first table lands.
    pub fn install_result_schema(&self, ctx: &SessionContext) -> Result<()> {
        let config = ctx.copied_config();
        let catalog_opts = &config.options().catalog;
        let catalog = ctx.catalog(&catalog_opts.default_catalog).ok_or_else(|| {
            JammiError::Other(format!(
                "default catalog '{}' is not registered on the session context",
                catalog_opts.default_catalog
            ))
        })?;
        catalog
            .register_schema(
                &catalog_opts.default_schema,
                Arc::clone(&self.result_schema) as Arc<dyn SchemaProvider>,
            )
            .map_err(|e| JammiError::Other(format!("install result-table schema provider: {e}")))?;
        Ok(())
    }

    /// The deployment's ANN sidecar-index tuning — the HNSW knobs plus the
    /// `storage_precision` / `oversample` defaults every newly-created
    /// embedding table's catalog row is stamped with. Read accessor for a
    /// caller that builds a `SidecarIndex` directly (rather than through
    /// [`Self::materialize_embedding_table`]) at table-creation time, e.g. the
    /// embedding-generation pipeline.
    pub fn ann_config(&self) -> &AnnIndexConfig {
        &self.ann
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
                // Stamped once, here, from today's deployment default — every
                // later build/load of this table's index reads it back off the
                // catalog row, never off `self.ann` again, so a later config
                // change cannot silently rebuild an existing table at a
                // different precision than this row already promises.
                // `effective_oversample_for` resolves the precision-specific
                // default (Binary's wider Hamming-coarse-stage oversample)
                // when the deployment left `oversample` at its untouched
                // shared default, while still honoring an explicit override.
                storage_precision: self.ann.storage_precision,
                oversample: self
                    .ann
                    .effective_oversample_for(self.ann.storage_precision),
                created_at: crate::catalog::backend::now_sortable(),
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

    /// Register an existing result-table Parquet object under the bare
    /// `jammi.{name}` identifier, gated on its catalog `owner` (the row's
    /// `tenant_id`, or `None` for a GLOBAL table).
    ///
    /// Builds the `ListingTable` provider — replicating the schema inference
    /// [`SessionContext::register_parquet`] performs so the resolved Arrow
    /// schema (Utf8View under the Arrow parquet-reader default) matches — then
    /// inserts it into this store's [`ResultTableSchemaProvider`], ensuring the
    /// provider is installed as `ctx`'s default schema first. The table
    /// resolves through the provider's tenant gate on every read lane, so a
    /// correctly-bound peer that names another tenant's table resolves
    /// not-found.
    pub async fn register_table(
        &self,
        ctx: &SessionContext,
        name: &str,
        url: &StorageUrl,
        owner: Option<TenantId>,
    ) -> Result<()> {
        let provider = build_result_table_provider(ctx, &self.registry, url).await?;
        self.install_result_schema(ctx)?;
        self.result_schema
            .register_result_table(format!("jammi.{name}"), provider, owner);
        Ok(())
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
    /// 3. flip `building -> ready` **and** persist the catalog summary columns
    ///    (`definition_hash`, `input_anchors_json`) in one transaction,
    ///    returning the row's `tenant_id`; and
    /// 4. register the table in DataFusion under that catalog owner, so its
    ///    resolution is tenant-gated by the row's own owner.
    ///
    /// The sidecar is written *before* the status flip — the same boundary the
    /// ANN sidecar uses — so a crash never leaves a `ready` table without a
    /// manifest. Registration is in-memory session state (not a durability
    /// boundary), so it follows the flip: a crash between the flip and
    /// registration leaves a valid `ready` row that the restart's
    /// [`Self::load_existing_tables`] re-registers. A crash between (1) and (2)
    /// leaves a `building` row whose
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

        let anchors_json = serde_json::to_string(&manifest.input_anchors)
            .map_err(|e| JammiError::Other(format!("serialise input anchors: {e}")))?;
        // The flip returns the row's own `tenant_id` (the owner stamped at
        // `create_table`) so registration gates the table on the catalog owner
        // by construction, never on whatever scope happens to run finalize.
        let owner = self
            .catalog
            .promote_result_table_with_manifest(
                name,
                rows,
                manifest.definition_hash.as_str(),
                &anchors_json,
            )
            .await?;

        self.register_table(ctx, name, url, owner).await?;
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
                                table.storage_precision.unwrap_or_default(),
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
    /// otherwise miss tenant-owned tables). Each table keeps its own catalog
    /// owner (`tenant_id`), so admin-scoped bulk loading does not flatten
    /// ownership: query-time resolution still gates each table on the tenant
    /// that owns it.
    ///
    /// All tenants' `ready` tables share one DataFusion context, but each
    /// registers through the [`ResultTableSchemaProvider`] carrying its catalog
    /// owner, so raw `sql()` over a result table applies the **same
    /// organizational tenant-scope** as the catalog API (`get_result_table`)
    /// and the mutable-table lane: a correctly-bound tenant resolves only its
    /// own and GLOBAL (`tenant_id IS NULL`) result tables over every lane
    /// (Flight `db.sql` included), and a peer's private table resolves
    /// not-found. This scopes a correctly-bound tenant's reads; it is an
    /// organizational mechanism, not a hostile-principal boundary — the
    /// trusted-network + BYO-auth posture is unchanged. Access control against a
    /// forged principal remains the consumer's BYO-auth seam / governing
    /// platform, never the engine's. See the guide's security posture for the
    /// boundary.
    ///
    /// A `ready` row whose bytes are absent (a torn write that committed `ready`
    /// before the bytes were durable on a power loss) is skipped, not
    /// registered, so it is never queryable.
    pub async fn load_existing_tables(&self, ctx: &SessionContext) -> Result<()> {
        TenantBinding::admin_scope(self.load_existing_tables_inner(ctx)).await
    }

    async fn load_existing_tables_inner(&self, ctx: &SessionContext) -> Result<()> {
        // Install the gating provider up-front so it is `ctx`'s default schema
        // even when there are zero ready tables to register (so a query on a
        // fresh session resolves not-found through the gate, and source removal
        // finds the provider to clear).
        self.install_result_schema(ctx)?;
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
            // The row's own `tenant_id` is the table's owner — captured here so
            // an admin-scoped bulk load registers each table under the tenant
            // that owns it, never flattened to the loading scope.
            let owner = match table.tenant_id.as_deref() {
                Some(s) => match TenantId::from_str(s) {
                    Ok(t) => Some(t),
                    Err(e) => {
                        warn!(
                            table = table.table_name,
                            error = %e,
                            "Result-table tenant_id is not a valid tenant id; skipping"
                        );
                        continue;
                    }
                },
                None => None,
            };
            let handle = self.open_parquet(&url)?;
            let path = handle.data_path()?;
            if handle.exists(&path).await? {
                if let Err(e) = self
                    .register_table(ctx, &table.table_name, &url, owner)
                    .await
                {
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
        // The catalog row's own persisted precision — never today's deployment
        // default — is what a load must verify against; see
        // `SidecarIndex::load`'s strict scalar_kind check.
        let expected_precision = table.storage_precision.unwrap_or_default();
        match storage::sidecar_layout::load_sidecar(&handle, &self.ann, expected_precision).await {
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
    ///
    /// `precision` must be the table's own persisted
    /// `ResultTableRecord::storage_precision` — **never** today's
    /// `self.ann.storage_precision` deployment default. A crash-recovery
    /// rebuild runs against whatever the *existing* row already promises; a
    /// rebuild at a different precision than that row's manifest previously
    /// recorded would silently corrupt recall (a graph a caller believes is
    /// `Int8` reopened as `F32`, or vice versa).
    async fn rebuild_index_from_parquet(
        &self,
        parquet_handle: &JammiObjectStore,
        index_url: &StorageUrl,
        dimensions: usize,
        precision: StoragePrecision,
    ) -> Result<()> {
        if dimensions == 0 {
            return Ok(());
        }

        let batches = storage::reader::read_all_record_batches(parquet_handle).await?;
        let mut index = SidecarIndex::new(dimensions, &self.ann, precision)?;
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
            key_column,
            text_columns,
        } = spec;

        // A normal embedding result table (S9 vocabulary: kind='model'); the
        // task is the embedding task that drives the sidecar-index sidecar URL.
        // The physical key stays `_row_id` (the output schema is invariant);
        // `key_column` / `text_columns` are the caller's source-side provenance.
        let table_info = self
            .create_table(
                source_id,
                ModelTask::TextEmbedding,
                ResultTableKind::Model,
                derived_from,
                model_id,
                Some(dimensions as i32),
                Some(key_column),
                text_columns,
            )
            .await?;

        let schema = crate::store::schema::embedding_table_schema(dimensions);
        let batch = embedding_batch(&schema, source_id, model_id, rows, dimensions)?;

        let mut writer = self.open_writer(&table_info.parquet_url, schema).await?;
        // Fresh creation (the row was just stamped with today's deployment
        // default in `create_table` above), so the same default applies here —
        // unlike a rebuild, there is no pre-existing catalog promise to honour.
        let mut index = SidecarIndex::new(dimensions, &self.ann, self.ann.storage_precision)?;
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

    /// Materialize consumer-computed, in-memory vectors as a ready, searchable
    /// embedding table under a caller-supplied [`ProducingDescriptor::External`]
    /// provenance — the promotion path for a producer the engine does not
    /// dispatch itself (a perturbation, a reconditioning pass, a migration off
    /// another store, any in-process recompute-avoidance batch).
    ///
    /// Every engine embedding table's storage/search contract is
    /// **cosine/direction-only**: rows are read back only through
    /// [`crate::index::VectorIndex`] cosine search, never as raw-vector reads,
    /// so a vector's *magnitude* is unobservable — only its *direction*
    /// carries meaning. Unit-normalizing the caller's rows before storing and
    /// digesting them is therefore invariant-upholding, never observably
    /// lossy, even for a caller's already-perturbed or reconditioned vectors:
    /// two vectors that differ only in magnitude are the same point under this
    /// contract, so collapsing that unobservable degree of freedom cannot lose
    /// information the table's own read path could ever expose. (This is a
    /// per-call normalization the caller's *rows* undergo, not a claim that
    /// every table this engine stores is unit-norm end-to-end — a graph
    /// propagation landed through [`Self::materialize_embedding_table`]
    /// directly may legitimately carry zero rows it declines to normalize.)
    ///
    /// Upholds the embedding-table invariant the same way
    /// [`Self::materialize_embedding_table`] callers had to hand-roll before
    /// this verb existed: each row is validated to `spec.dimensions` wide
    /// (typed [`JammiError::Schema`] on mismatch) and L2-normalized, rejecting
    /// a zero or non-finite norm (also [`JammiError::Schema`] — such a vector
    /// cannot be cosine-searched). The **normalized copy** — never the
    /// caller's borrowed input — is what gets stored and digested.
    ///
    /// Auto-folds a [`CONTENT_DIGEST_PARAM_KEY`] content digest of the
    /// normalized rows into `provenance.params`, so two materializations
    /// sharing every scalar determinant but different vectors never collide
    /// on one [`DefinitionHash`] (K7 completeness). Fails loud
    /// ([`JammiError::Schema`]) if the caller's `params` already carries that
    /// reserved key — never a silent overwrite.
    pub async fn materialize_computed_embedding_table(
        &self,
        ctx: &SessionContext,
        spec: EmbeddingTableSpec<'_>,
        rows: &[(String, Vec<f32>)],
        mut provenance: ComputedEmbeddingProvenance,
    ) -> Result<ResultTableRecord> {
        if provenance.params.contains_key(CONTENT_DIGEST_PARAM_KEY) {
            return Err(JammiError::Schema {
                table: spec.source_id.to_string(),
                column: CONTENT_DIGEST_PARAM_KEY.to_string(),
                expected: "provenance.params without a caller-supplied content_digest".to_string(),
                actual: "provenance.params already carries the reserved content_digest key"
                    .to_string(),
            });
        }

        let dimensions = spec.dimensions;
        let mut normalized: Vec<(String, Vec<f32>)> = Vec::with_capacity(rows.len());
        for (key, vector) in rows {
            if vector.len() != dimensions {
                return Err(JammiError::Schema {
                    table: spec.source_id.to_string(),
                    column: "vector".to_string(),
                    expected: format!("FixedSizeList<Float32> width {dimensions}"),
                    actual: format!("row '{key}' has width {}", vector.len()),
                });
            }
            let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            if !(norm.is_finite() && norm > 0.0) {
                return Err(JammiError::Schema {
                    table: spec.source_id.to_string(),
                    column: "vector".to_string(),
                    expected: "a non-zero-norm, L2-normalizable vector".to_string(),
                    actual: format!("row '{key}' has norm {norm}"),
                });
            }
            normalized.push((key.clone(), vector.iter().map(|x| x / norm).collect()));
        }

        provenance.params.insert(
            CONTENT_DIGEST_PARAM_KEY.to_string(),
            content_digest(&normalized),
        );

        let descriptor = ProducingDescriptor::External {
            producer_id: provenance.producer_id,
            params: provenance.params,
        };

        self.materialize_embedding_table(
            ctx,
            spec,
            &normalized,
            Materialization::new(&descriptor, &provenance.env, provenance.inputs),
        )
        .await
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

/// A stable content digest over normalized embedding rows: the hex of a
/// SHA-256 folding each row's key bytes and vector bytes in file order.
/// Distinguishes two productions that share every scalar determinant but
/// carry different vectors, so they never alias on one [`DefinitionHash`].
fn content_digest(rows: &[(String, Vec<f32>)]) -> String {
    let mut buf = Vec::new();
    for (key, vector) in rows {
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&(vector.len() as u64).to_le_bytes());
        for x in vector {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    ArtifactDigest::of_bytes(&buf).0
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
/// Fold a [`ManifestError`] into the engine's [`JammiError`] — the single
/// canonical conversion the materialization funnel and the action-layer probes
/// (which compute a [`MaterializationManifest::definition_of`] outside the
/// funnel) both use, so a manifest error surfaces the same typed arm regardless
/// of where it arose.
pub fn manifest_to_jammi(e: ManifestError) -> JammiError {
    match e {
        ManifestError::Storage(s) => JammiError::Storage(s),
        ManifestError::Serde(s) => JammiError::Json(s),
        other => JammiError::Catalog(other.to_string()),
    }
}

/// Build the `ListingTable` provider for a result-table Parquet URL, ready to
/// register under the bare `jammi.{name}` identifier in the
/// [`ResultTableSchemaProvider`].
///
/// Replicates exactly what [`SessionContext::register_parquet`] does — the same
/// driver registration, `ParquetReadOptions::default()` → listing options
/// (resolved against the session's config + table options) → schema inference →
/// `ListingTable` — so the resolved Arrow schema (Utf8View under the parquet
/// reader default) matches the one the old direct-registration path produced.
/// Only the final step differs: rather than registering into the context's
/// default `MemorySchemaProvider` under a re-parsed `TableReference`, the
/// caller inserts this provider into the tenant-gating schema keyed by the
/// single bare `jammi.{name}` literal — the same literal the query side reaches
/// these tables through, which the SQL tokenizer never splits on the embedded
/// timestamp dot or a sanitized model path's hyphen.
async fn build_result_table_provider(
    ctx: &SessionContext,
    registry: &StorageRegistry,
    url: &StorageUrl,
) -> Result<Arc<dyn TableProvider>> {
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

    let config = ctx.copied_config();
    let listing_options =
        ParquetReadOptions::default().to_listing_options(&config, ctx.copied_table_options());
    let table_path = ListingTableUrl::parse(url.as_str())?;
    let resolved_schema = listing_options
        .infer_schema(&ctx.state(), &table_path)
        .await?;
    let table_config = ListingTableConfig::new(table_path)
        .with_listing_options(listing_options)
        .with_schema(resolved_schema);
    Ok(Arc::new(ListingTable::try_new(table_config)?))
}
