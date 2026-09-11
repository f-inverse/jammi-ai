pub mod artifact;
pub mod building;
pub mod building_version;
pub mod content_hash;
pub mod deletes;
pub mod freshness;
pub mod layout;
pub mod manifest;
pub mod masked_provider;
pub mod mutable;
pub mod reconcile;
pub mod result_schema;
pub mod schema;
pub mod segment_set_cache;
pub mod vectors;
pub mod version;

pub use artifact::{ArtifactStore, LocalArtifact};
pub use building::BuildingTable;
pub use building_version::BuildingVersion;
pub use deletes::DeletionMask;
pub use freshness::{
    CacheOutcome, CachePolicy, CurrentAnchor, DerivesFromEdge, StaleReason, Staleness,
};
pub use layout::TenantSegment;
pub use manifest::{
    AnchorKind, AnchorValue, ArtifactDigest, ComputeDevice, DefinitionHash, DeletePolicy,
    InputAnchor, ManifestError, MatchVerdict, Materialization, MaterializationEnv,
    MaterializationManifest, ModelContentDigest, ModelContentDigestUnavailableReason,
    ModelIdentity, ProducingDescriptor,
};
pub use reconcile::{ReconcileOptions, ReconcileReport};
pub use result_schema::ResultTableSchemaProvider;
pub use version::VersionManifest;

use std::collections::{BTreeMap, BTreeSet};
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

use crate::catalog::lease::LeaseIntervals;
use crate::catalog::result_repo::{
    CreateResultTableParams, JobAttempt, ResultTableCas, ResultTableKind, ResultTableRecord,
};
use crate::catalog::status::ResultTableStatus;
use crate::catalog::Catalog;
use crate::config::AnnIndexConfig;
use crate::error::{JammiError, Result};
use crate::index::peer::{AllLocal, NoPeers, PeerFailureCounters, PeerTransport, SegmentPlacement};
use crate::index::placed::{PlacedIndex, SegmentSource};
use crate::index::segment::{SegmentId, SegmentedIndex};
use crate::index::sidecar::SidecarIndex;
use crate::index::VectorIndex;
use crate::model_task::ModelTask;
use crate::storage::index_cache::SegmentIndexCache;
use crate::storage::sidecar_layout::SidecarKind;
use crate::storage::{
    self, DeleteOutcome, JammiObjectStore, ObjectParquetWriter, Scheme, StorageRegistry, StorageUrl,
};
use crate::store::masked_provider::{MaskedFragment, MaskedTableProvider, PlaceholderProvider};
use crate::store::segment_set_cache::{LoadedSegmentSet, SegmentSetCache};
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
    /// so lineage survives without changing the output schema. A reader joins
    /// `source.<key_column> = derived._row_id`, so the name must be a column
    /// the origin really has — a producer keying straight off a source's own
    /// `_row_id` passes `Some("_row_id")`. `None` when the origin key is
    /// unknown or does not apply (keys that correspond to no stored source
    /// row), which is the honest answer rather than a name the origin lacks.
    pub key_column: Option<&'a str>,
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

/// Coordinates Parquet storage, ANN indexes, DataFusion registration,
/// catalog metadata, and crash recovery for result tables.
///
/// Wraps a `StorageUrl` as the root prefix every new table is created under.
/// File scheme keeps the historical `{artifact_dir}/jammi_db/` layout;
/// `s3://bucket/jammi_db/`, `gs://...`, `azure://...` work without code
/// change because every read/write goes through [`StorageRegistry`].
///
/// One store instance is one **writer**: it mints a `writer-{uuid}` at
/// construction and stamps it on every `building` row it creates, so two
/// sessions in one process are distinct writers. `Clone` shares the same
/// writer identity, catalog, registry, and caches — the handle a
/// [`BuildingTable`] keeps to act on its row.
#[derive(Clone)]
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
    /// The content-addressed local cache every ANN index segment is loaded
    /// through. Materialises a remote segment bundle into a local directory
    /// USearch can open, once per immutable segment; a `file://` bundle loads
    /// in place. Shares the store's [`StorageRegistry`]. An `Arc` so a
    /// [`PlacedIndex`] and a peer owner handler can hold the same cache.
    segment_cache: Arc<SegmentIndexCache>,
    /// Loaded segment sets and version manifests per `(table, version)`.
    segment_sets: Arc<SegmentSetCache>,
    /// Which process owns which segment, read at every
    /// [`Self::resolve_search_mode`]. Default [`AllLocal`]: every segment is
    /// this process's — a single node.
    placement: Arc<dyn SegmentPlacement>,
    /// The transport a placed search fans remote segments out through.
    /// Default [`NoPeers`]: every remote call is unreachable, so a store
    /// without a transport is exactly a single-node store.
    peer_transport: Arc<dyn PeerTransport>,
    /// `[server] peer_local_load_bytes` — the marginal-load admission budget
    /// one query may spend loading segments it does not own. `None` =
    /// unbounded.
    peer_local_load_bytes: Option<u64>,
    /// The failure-ladder counters every placed search increments; scraped as
    /// `jammi_peer_search_failures_total{reason}`.
    peer_failures: Arc<PeerFailureCounters>,
    /// This store's writer identity, stamped on every `building` row it
    /// creates and named by every transition on that row.
    writer_id: Arc<str>,
    /// The lease window / heartbeat every [`BuildingTable`] this store creates
    /// (or recovery claims) is held under — the deployment's one
    /// [`crate::config::LeaseConfig`].
    lease: LeaseIntervals,
    /// The model-artifact store rooted at `{root}/models`, sharing this
    /// store's [`StorageRegistry`]. A single storage knob (`root`) serves
    /// both result tables and trained models; `jammi-ai`'s session reads
    /// this handle back through [`Self::artifact_store`] rather than
    /// constructing its own, so the two can never disagree on where models
    /// live relative to result tables.
    artifact_store: Arc<ArtifactStore>,
    /// The process's lease-renewal thread (N3) every [`BuildingTable`] this
    /// store creates or recovery adopts holds its row with, in place of
    /// a per-table `tokio::spawn` heartbeat task.
    /// `None` — the default — means a table this store hands out is renewed
    /// by NOTHING beyond its initial lease window: correct but non-renewing,
    /// acceptable for a short-lived test fixture, never for a production
    /// deployment (the session choke point attaches one via
    /// [`Self::with_lease_keeper`] before serving). `Clone`d cheaply — an
    /// `Arc`, shared by every clone of this store.
    keeper: Option<Arc<crate::catalog::lease_keeper::LeaseKeeper>>,
}

/// The width guard for the NO-INDEX exact fallback — the one search path with
/// no [`SidecarIndex`] behind it, so the authoritative
/// [`crate::index::segment::verify_query_width`] cannot reach it. Here the
/// only width on record is the catalog's `dimensions` column.
///
/// `dimensions` is `Option<i32>`, and `None` is a live state (a row written
/// before the column existed, or a non-embedding table). `None` is therefore
/// an EXPLICIT PASS-THROUGH: there is nothing to check the query against, and
/// refusing would break tables that work today. The scan below is safe either
/// way — `cosine_distance` now refuses a length mismatch outright rather than
/// reading past a vector — so this check buys a TYPED, table-named error
/// instead of a panic, not memory safety.
fn verify_query_width_against_catalog(table: &ResultTableRecord, query: &[f32]) -> Result<()> {
    let Some(dimensions) = table.dimensions else {
        return Ok(());
    };
    let expected = usize::try_from(dimensions).unwrap_or(0);
    if expected != 0 && query.len() != expected {
        return Err(JammiError::Schema {
            table: table.table_name.clone(),
            column: "query".into(),
            expected: format!("{expected} dimensions"),
            actual: format!("{} dimensions", query.len()),
        });
    }
    Ok(())
}

/// Mint a fresh writer identity.
fn new_writer_id() -> Arc<str> {
    Arc::from(format!("writer-{}", uuid::Uuid::new_v4()).as_str())
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

/// `{root}/models` — the artifact store's root, derived from the result
/// store's own root so one storage knob serves both.
fn models_root(root: &StorageUrl) -> Result<StorageUrl> {
    let root_str = root.as_str().trim_end_matches('/');
    Ok(StorageUrl::parse(&format!("{root_str}/models"))?)
}

/// The three terminal-or-untouched outcomes an expired-lease `building` row
/// can classify to. [`ResultStore::classify_expired_row`] is the ONE function
/// that computes this — apply calls it and then performs the outcome
/// ([`ResultStore::reconcile_expired_building_row`]); `reconcile`'s
/// `apply=false` preview calls it ALONE and performs nothing.
///
/// **A promotion is not a reclaim (#484 design revision).** An earlier
/// revision of this type carried a `Promote { keeps, reclaims, dir_prefixes }`
/// shape that tried to predict, at classify time, exactly which of a row's
/// CURRENT segment sidecars [`ResultStore::rebuild_index_from_parquet`]'s
/// destructive purge would delete-and-not-rewrite, and credited that
/// prediction into `orphans`/`bytes_reclaimed` in both modes. That mirror was
/// itself a recurring defect surface: the `Err` arm's credit subtracted a
/// counterfactual `keeps` from what the rebuild ACTUALLY purged, an
/// ERROR-level mismatch oracle existed only to notice when the two predictions
/// diverged (rather than removing the redundant prediction), and the fused
/// Parquet reader it depended on could turn a benign listing-to-read vanish
/// race into a whole-pass abort. The fix is architectural, not another
/// mirror-repair: `Promote` now carries only the row's FULL currently
/// referenced key set (protected, in both modes) and predicts NOTHING about
/// what the rebuild will purge — a promotion's internal rebuild is bookkeeping
/// the promotion performs on itself, never a reclaim this pass reports at all
/// (see [`ReconcileReport::bytes_reclaimed`]'s updated contract). Apply's own
/// [`ResultStore::purge_segments`] call still runs exactly as before (a
/// promotion legitimately needs to clear stale segment state); what changed
/// is that its returned key set is now recorded into a per-pass
/// non-crediting exclusion (`store::reconcile::reconcile_inner`'s
/// `promoted_purged`) rather than differenced against a classify-time
/// prediction and credited.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ExpiredRowOutcome {
    /// Torn/invalid Parquet, or a valid Parquet with no manifest sidecar:
    /// reaped to `failed`, its objects deleted (apply) or previewed as
    /// reclaimable (dry-run) — accounted in `orphans`/`bytes_reclaimed` in
    /// BOTH modes, and NEVER age-gated against `grace` (see
    /// [`ReconcileReport::orphans`]'s two admission routes).
    Reap,
    /// A valid Parquet with its manifest sidecar present: promoted to
    /// `ready` (apply) or previewed as such (dry-run). Carries the row's
    /// FULL currently-referenced key set (the Parquet, the manifest
    /// sidecar, and every CURRENT `index_segments` row's sidecars) —
    /// protected wholesale (dry-run); apply protects the SAME shape simply
    /// by promoting the row and letting it re-query as `ready` before this
    /// pass's `referenced_result_keys` runs. Neither mode predicts, or
    /// reports, anything about what the promotion's own rebuild will purge
    /// and not rewrite — see this type's own doc comment.
    Promote {
        /// Every key this row references RIGHT NOW.
        keeps: BTreeSet<String>,
        /// Directory-shaped sidecar prefixes (`SidecarKind::Lexical`'s
        /// `.tantivy`) this row currently references — carried separately
        /// because [`ReferencedKeys`](crate::store::reconcile::ReferencedKeys)
        /// matches these by PREFIX, never exact equality. Always empty in
        /// practice (an embedding-task building row's segments are
        /// ANN-only — see [`ResultStore::append_segment`]).
        dir_prefixes: BTreeSet<String>,
    },
    /// The Parquet itself is absent: nothing to reap, promote, or protect —
    /// only the `building -> failed` CAS runs (apply).
    Untouched,
}

/// [`ResultStore::rebuild_index_from_parquet`]'s `Err` payload: the
/// underlying error, PAIRED with the root-relative keys its own
/// `purge_segments` call had already deleted before whatever failed next
/// (reading the Parquet's batches, decoding a vector, writing the fresh
/// segment). Empty `purged` when `purge_segments` itself is what failed —
/// nothing is known to have been deleted in that case. The sole caller
/// ([`ResultStore::reconcile_expired_building_row`]) records `purged` into
/// this pass's `promoted_purged` exclusion even on this `Err` arm: those
/// bytes are gone from storage regardless of what failed downstream of the
/// purge, so they must never fall through to the ordinary orphan arm and be
/// double-reported against a listing snapshot taken before they were
/// deleted; a key `purge_segments` FAILED to delete (a real I/O error, never
/// merely absent) is simply not in `purged` at all, and is left exactly
/// where it is for the ordinary age-gated arm — this pass, or a later one —
/// to reclaim normally.
struct RebuildFailure {
    error: JammiError,
    purged: BTreeSet<String>,
}

/// Every root-relative key one call to [`ResultStore::delete_objects_after_cas`]
/// or [`ResultStore::purge_segments`] touched, split by what actually
/// happened to it — never collapsed into one flat set (esc-484): `deleted`
/// is exactly [`DeleteOutcome::Deleted`], the ONLY set `reconcile`'s
/// byte-accounting may ever credit; `errored` is every key whose
/// `delete_if_exists` hit a REAL object-store error (never a mere
/// [`DeleteOutcome::Absent`]) and so was left in place. A key that was
/// merely `Absent` (never written for this row's actual precision/state, or
/// vanished before this call ran) is in NEITHER set — it is not a failure,
/// and it was not a deletion. `abort()`'s own completeness check needs
/// exactly `errored`: [`ResultStore::reap_candidate_keys`]'s superset
/// intentionally enumerates every POSSIBLE sidecar extension regardless of a
/// row's actual precision, most of which are legitimately `Absent` and were
/// never expected to exist — diffing THAT superset against `deleted` alone
/// would flag every merely-inapplicable extension as a false failure, which
/// is exactly the bug this type exists to prevent.
#[derive(Debug, Clone, Default)]
pub(crate) struct DeletionOutcome {
    pub deleted: BTreeSet<String>,
    pub errored: BTreeSet<String>,
}

/// What one call to [`ResultStore::reconcile_expired_building_row`] learned
/// about a row's objects — distinguishing "credit this as an ordinary
/// reclaim" from "this pass's promotion consumed these keys, account for
/// them nowhere" so `reconcile`'s pre-pass can never conflate the two
/// (esc-484 design revision: a promotion is not a reclaim).
pub(crate) enum ExpiredRowDeletion {
    /// [`ExpiredRowOutcome::Untouched`], or a `Promote` row whose claim was
    /// lost to a concurrent writer/recoverer before anything was deleted:
    /// nothing to account.
    Untouched,
    /// [`ExpiredRowOutcome::Reap`]'s actually-deleted keys (a partial delete
    /// failure leaves the failed key out — see
    /// [`ResultStore::delete_objects_after_cas`]) — credited into
    /// `orphans`/`bytes_reclaimed` exactly like any other orphan.
    Reaped(BTreeSet<String>),
    /// An [`ExpiredRowOutcome::Promote`] row's rebuild ACTUALLY purged these
    /// keys (whether or not the rebuild went on to succeed) — EXCLUDED from
    /// this pass's accounting entirely: never `orphans`, `pending`, nor
    /// `bytes_reclaimed`. They were consumed by the promotion, not reclaimed
    /// by the ordinary orphan mechanism. A key `purge_segments` FAILED to
    /// delete (a real I/O error, never merely absent) is never in this set —
    /// it is left exactly where it is, falling through to the ordinary
    /// age-gated arm to retry, in this pass or a later one.
    PromotedPurged(BTreeSet<String>),
}

/// Test-only rendezvous hooks for reconcile's expired-building races
/// (`#484` and follow-ups): a caller can park a running pass at a documented
/// point and release it once test setup has manufactured the race window,
/// pinning an exact TOCTOU rather than merely inferring it from a single
/// fixture. Compiled only under `feature = "test-hooks"`; no production code
/// path observes anything in this module beyond the two `maybe_park_*` calls
/// themselves (no-ops whenever nothing is armed).
#[cfg(feature = "test-hooks")]
pub mod reconcile_test_hooks {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use tokio::sync::Notify;

    /// One-shot rendezvous state for an armed race, keyed by table name so
    /// only the armed table's own pass ever parks.
    struct RaceState {
        table_name: String,
        parked: Arc<AtomicBool>,
        parked_notify: Arc<Notify>,
        release: Arc<Notify>,
        released: Arc<AtomicBool>,
    }

    /// The test's handle on an armed race: wait for the pass to park, then
    /// release it. Dropping the handle releases a parked writer (if any) so
    /// a panicking test never hangs the pass out to the bounded park's
    /// timeout.
    pub struct RaceHandle {
        parked: Arc<AtomicBool>,
        parked_notify: Arc<Notify>,
        release: Arc<Notify>,
        released: Arc<AtomicBool>,
    }

    fn arm(slot: &Mutex<Option<RaceState>>, table_name: &str) -> RaceHandle {
        let state = RaceState {
            table_name: table_name.to_string(),
            parked: Arc::new(AtomicBool::new(false)),
            parked_notify: Arc::new(Notify::new()),
            release: Arc::new(Notify::new()),
            released: Arc::new(AtomicBool::new(false)),
        };
        let handle = RaceHandle {
            parked: Arc::clone(&state.parked),
            parked_notify: Arc::clone(&state.parked_notify),
            release: Arc::clone(&state.release),
            released: Arc::clone(&state.released),
        };
        let mut guard = slot.lock().expect("reconcile test-hook arm lock");
        // Only `maybe_park` clears this slot, and only when the parked
        // pass's table name matches the armed one — `RaceHandle::release`
        // and its `Drop` never touch the slot. So an occupied slot means
        // one of two things: an earlier `RaceHandle` for THIS race point
        // was never released (or was leaked past its test) before a new
        // test tried to arm the same point again, OR the earlier pass never
        // reached this race point for the armed table (no `maybe_park` call
        // matched it, so nothing ever consumed the slot). Either way,
        // silently overwriting it would strand whatever pass is (or later
        // becomes) parked against the stale `RaceState` with no
        // `RaceHandle` left able to release it, hanging that pass out to
        // its own 30s park timeout. Panicking here (test-hooks only; no
        // production path ever calls `arm`) turns that into an immediate,
        // attributable test failure instead.
        assert!(
            guard.is_none(),
            "reconcile test-hook: race already armed for table '{}' when arming '{table_name}' \
             on the same slot — release the earlier RaceHandle before arming again",
            guard.as_ref().map(|s| s.table_name.as_str()).unwrap_or("")
        );
        *guard = Some(state);
        drop(guard);
        handle
    }

    impl RaceHandle {
        /// Wait (bounded to 5s) until the pass has parked at the armed
        /// point.
        pub async fn wait_parked(&self) {
            let notified = self.parked_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.parked.load(Ordering::SeqCst) {
                return;
            }
            let _ = tokio::time::timeout(std::time::Duration::from_secs(5), notified).await;
        }

        /// Whether the pass is currently parked at the armed point.
        pub fn is_parked(&self) -> bool {
            self.parked.load(Ordering::SeqCst)
        }

        /// Release the parked pass (idempotent).
        pub fn release(&self) {
            self.released.store(true, Ordering::SeqCst);
            self.release.notify_waiters();
        }
    }

    impl Drop for RaceHandle {
        fn drop(&mut self) {
            self.release();
        }
    }

    async fn maybe_park(slot: &Mutex<Option<RaceState>>, table_name: &str) {
        let taken = {
            let mut guard = slot.lock().expect("reconcile test-hook arm lock");
            if guard.as_ref().is_some_and(|s| s.table_name == table_name) {
                guard.take()
            } else {
                None
            }
        };
        let Some(state) = taken else {
            return;
        };
        state.parked.store(true, Ordering::SeqCst);
        state.parked_notify.notify_waiters();
        if !state.released.load(Ordering::SeqCst) {
            let released = state.release.notified();
            tokio::pin!(released);
            released.as_mut().enable();
            if !state.released.load(Ordering::SeqCst) {
                let _ = tokio::time::timeout(std::time::Duration::from_secs(30), released).await;
            }
        }
    }

    static MANIFEST_ARM: Mutex<Option<RaceState>> = Mutex::new(None);

    /// Arm the "manifest vanished between classify and perform" race for
    /// `table_name`: the next time
    /// [`super::ResultStore::reconcile_expired_building_row`]'s `Promote` arm
    /// reaches [`maybe_park_before_manifest_reread`] for THIS table, it parks
    /// (bounded to 30s) until [`RaceHandle::release`] — the window in which a
    /// test can delete the row's manifest sidecar out from under it, pinning
    /// the exact TOCTOU the production re-read guards against. Panics if
    /// this race point is already armed — see `arm`.
    pub fn arm_manifest_vanish_race(table_name: &str) -> RaceHandle {
        arm(&MANIFEST_ARM, table_name)
    }

    /// Park if a manifest-vanish race is armed for `table_name` (a no-op
    /// otherwise, and a no-op for every other test/production build). Called
    /// by `reconcile_expired_building_row`'s `Promote` arm right after
    /// `classify_expired_row` returns `Promote` for this row, immediately
    /// before its own re-read of the manifest sidecar — the exact window
    /// that race lands in.
    pub(super) async fn maybe_park_before_manifest_reread(table_name: &str) {
        maybe_park(&MANIFEST_ARM, table_name).await
    }

    static PARQUET_ARM: Mutex<Option<RaceState>> = Mutex::new(None);

    /// Arm the "Parquet vanished during the classify window" race for
    /// `table_name`: the next time [`super::ResultStore::classify_expired_row`]
    /// reaches [`maybe_park_before_parquet_reread`] for THIS table — right
    /// after its own `exists()` check on the Parquet object passes, and
    /// immediately before its single read of the Parquet's bytes
    /// (`storage::reader::validate_and_count_parquet_rows`) — it parks
    /// (bounded to 30s) until [`RaceHandle::release`]: the window in which a
    /// test can delete the row's Parquet out from under it, pinning that this
    /// vanish reclassifies the row to [`super::ExpiredRowOutcome::Reap`]
    /// rather than aborting the whole reconcile pass with an object-store
    /// error. Panics if this race point is already armed — see `arm`.
    pub fn arm_parquet_vanish_race(table_name: &str) -> RaceHandle {
        arm(&PARQUET_ARM, table_name)
    }

    /// Park if a Parquet-vanish race is armed for `table_name` (a no-op
    /// otherwise, and a no-op for every other test/production build). Called
    /// by `classify_expired_row` immediately after its Parquet `exists()`
    /// check passes, before its single read of the Parquet's bytes.
    pub(super) async fn maybe_park_before_parquet_reread(table_name: &str) {
        maybe_park(&PARQUET_ARM, table_name).await
    }

    static POST_CLAIM_PARQUET_ARM: Mutex<Option<RaceState>> = Mutex::new(None);

    /// Arm the "Parquet vanished after claim, before the post-claim row-count
    /// read" race for `table_name` (esc-484 advisory): the next time
    /// [`super::ResultStore::reconcile_expired_building_row`]'s `Promote` arm
    /// reaches [`maybe_park_before_post_claim_row_count`] for THIS table —
    /// after `claim_expired` has already succeeded, immediately before its
    /// `storage::reader::count_parquet_rows` read — it parks (bounded to 30s)
    /// until [`RaceHandle::release`]: the window in which a test can delete
    /// the row's Parquet out from under an already-claimed recoverer, pinning
    /// that this vanish reclassifies the row to a reap (under the CLAIM's own
    /// CAS) rather than aborting the whole reconcile pass with an
    /// object-store error. Panics if this race point is already armed —
    /// see `arm`.
    pub fn arm_post_claim_parquet_vanish_race(table_name: &str) -> RaceHandle {
        arm(&POST_CLAIM_PARQUET_ARM, table_name)
    }

    /// Park if a post-claim Parquet-vanish race is armed for `table_name` (a
    /// no-op otherwise, and a no-op for every other test/production build).
    /// Called by `reconcile_expired_building_row`'s `Promote` arm immediately
    /// after `claim_expired` succeeds, before its post-claim row-count read.
    pub(super) async fn maybe_park_before_post_claim_row_count(table_name: &str) {
        maybe_park(&POST_CLAIM_PARQUET_ARM, table_name).await
    }
}

impl ResultStore {
    /// Construct a result-store rooted at a local artifact directory. The
    /// directory is created if absent. Roots result tables at
    /// `{artifact_dir}/jammi_db/` (unchanged from the historical layout) with
    /// the ANN segment cache and the artifact fetch cache relocated OUT of
    /// that root, at `{artifact_dir}/cache/index` and
    /// `{artifact_dir}/cache/artifact` respectively: the caches are
    /// content-addressed scratch state, not result-table data, so they no
    /// longer sit inside the directory a `reconcile` or backup walks as the
    /// table root. Equivalent to
    /// `ResultStore::with_root(StorageUrl::parse(artifact_dir.join("jammi_db"))?, …, artifact_dir.join("cache"))`
    /// with a default-constructed [`StorageRegistry`]. Old on-disk
    /// `jammi_db/index_cache` / `jammi_db/artifact_cache` directories from
    /// before this change are inert after upgrade — cold caches that
    /// `reconcile` reports as `unattributed` (never deleted).
    pub fn new(artifact_dir: &Path, catalog: Arc<Catalog>, ann: AnnIndexConfig) -> Result<Self> {
        let jammi_db_dir = artifact_dir.join("jammi_db");
        std::fs::create_dir_all(&jammi_db_dir)?;
        let url = StorageUrl::parse(
            jammi_db_dir
                .to_str()
                .ok_or_else(|| JammiError::Config("Non-UTF8 artifact_dir".into()))?,
        )?;
        Self::with_root(
            url,
            StorageRegistry::new(),
            catalog,
            ann,
            artifact_dir.join("cache"),
        )
    }

    /// Construct a result-store rooted at an arbitrary [`StorageUrl`] —
    /// the path on `cloud://` schemes a deployment uses for shared
    /// result-table storage. The registry is shared with the engine
    /// session so callers register cloud credentials once.
    ///
    /// `local_cache_dir` is the **parent** of the two local cache
    /// directories this store derives: `{local_cache_dir}/index` (the ANN
    /// segment cache — a `file://` root loads its segments in place, so it
    /// is unused there) and `{local_cache_dir}/artifact` (the model-artifact
    /// fetch cache the store's own [`ArtifactStore`], rooted at
    /// `{root}/models`, materialises cloud bundles under). Both are local
    /// paths even when `root` is a cloud scheme, since USearch and candle
    /// both read from the local filesystem.
    pub fn with_root(
        root: StorageUrl,
        registry: StorageRegistry,
        catalog: Arc<Catalog>,
        ann: AnnIndexConfig,
        local_cache_dir: std::path::PathBuf,
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
        let segment_cache = Arc::new(SegmentIndexCache::new(
            registry.clone(),
            local_cache_dir.join("index"),
        )?);
        let artifact_store = Arc::new(ArtifactStore::with_root(
            models_root(&root)?,
            registry.clone(),
            local_cache_dir.join("artifact"),
        )?);
        Ok(Self {
            root,
            registry,
            catalog,
            ann,
            result_schema,
            segment_cache,
            segment_sets: Arc::new(SegmentSetCache::new()),
            placement: Arc::new(AllLocal),
            peer_transport: Arc::new(NoPeers),
            peer_local_load_bytes: None,
            peer_failures: Arc::new(PeerFailureCounters::default()),
            writer_id: new_writer_id(),
            lease: LeaseIntervals::default(),
            artifact_store,
            keeper: None,
        })
    }

    /// Attach the process's lease-renewal thread (N3): every
    /// [`BuildingTable`] this store creates or recovery adopts from this
    /// point on holds its row open with `keeper` instead of running its own
    /// heartbeat task. The session choke point calls this once, right after
    /// constructing both, before the store serves any `create_table` call.
    pub fn with_lease_keeper(
        mut self,
        keeper: Arc<crate::catalog::lease_keeper::LeaseKeeper>,
    ) -> Self {
        self.keeper = Some(keeper);
        self
    }

    /// This store's model-artifact store, rooted at `{root}/models` and
    /// sharing this store's [`StorageRegistry`]. `jammi-ai`'s session reads
    /// this handle rather than constructing its own artifact store, so the
    /// two never disagree on where models live relative to result tables.
    pub fn artifact_store(&self) -> Arc<ArtifactStore> {
        Arc::clone(&self.artifact_store)
    }

    /// Set the lease window / heartbeat every [`BuildingTable`] this store
    /// creates is held under (the deployment's
    /// [`crate::config::LeaseConfig::intervals`]). Defaults to the engine's
    /// built-in 30 s / 10 s pair.
    pub fn with_lease_intervals(mut self, intervals: LeaseIntervals) -> Self {
        self.lease = intervals;
        self
    }

    /// The lease timing this store's building tables are held under.
    pub fn lease_intervals(&self) -> LeaseIntervals {
        self.lease
    }

    /// Set which process owns which segment (read at every
    /// [`Self::resolve_search_mode`]). Defaults to [`AllLocal`].
    pub fn with_placement(mut self, placement: Arc<dyn SegmentPlacement>) -> Self {
        self.placement = placement;
        self
    }

    /// Set the transport a placed search fans remote segments out through.
    /// Defaults to [`NoPeers`].
    pub fn with_peer_transport(mut self, transport: Arc<dyn PeerTransport>) -> Self {
        self.peer_transport = transport;
        self
    }

    /// Set `[server] peer_local_load_bytes` — the marginal-load admission
    /// budget one query may spend loading segments it does not own when their
    /// owners are unreachable. `None` (the default) = unbounded.
    pub fn with_peer_local_load_bytes(mut self, budget: Option<u64>) -> Self {
        self.peer_local_load_bytes = budget;
        self
    }

    /// The content-addressed segment cache every segment of this store loads
    /// through — shared with a [`PlacedIndex`] and a peer owner handler.
    pub fn segment_cache(&self) -> &Arc<SegmentIndexCache> {
        &self.segment_cache
    }

    /// The placed-search failure-ladder counters this store increments.
    pub fn peer_failures(&self) -> Arc<PeerFailureCounters> {
        Arc::clone(&self.peer_failures)
    }

    /// The process's lease-renewal keeper this store's `building` tables
    /// hold with, if one has been attached via
    /// [`Self::with_lease_keeper`].
    pub(crate) fn lease_keeper(&self) -> Option<Arc<crate::catalog::lease_keeper::LeaseKeeper>> {
        self.keeper.clone()
    }

    /// This store's writer identity (`writer-{uuid}`).
    pub fn writer_id(&self) -> &str {
        &self.writer_id
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
    /// status = 'building', lease-owned by this store's writer, and return the
    /// [`BuildingTable`] handle whose heartbeat keeps that lease renewed until
    /// [`BuildingTable::finish`] or [`BuildingTable::abort`].
    ///
    /// `kind` discriminates a direct model output from a derivation of another
    /// result table (e.g. a neighbor-graph edge relation); `derived_from` names
    /// the source result table a derivation was computed from (`None` for a
    /// `Model` table). No ANN index is created here for any `kind`: an embedding
    /// table's index materialises lazily as segments through
    /// [`BuildingTable::append_segment`], and a derived table carries none at
    /// all.
    ///
    /// The row's tenant is read once from the catalog binding in force and
    /// captured on the handle, so every later transition — including the
    /// heartbeat's, which runs on a task with no task-local scope — names the
    /// row's own tenant.
    ///
    /// `job_attempt` (N11, esc-107) is threaded straight to
    /// [`crate::catalog::result_repo::CreateResultTableParams::job_attempt`]
    /// — see there for the `jobs.partial_result` compare-and-set this
    /// performs in the SAME transaction as the row's own INSERT, and for why
    /// the CAS needs the full `(job_id, instance_id, attempts)` identity, not
    /// `job_id` alone. `None` for a table created outside the job machinery
    /// (a test fixture, or a caller that materialises with no job of
    /// record).
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
        job_attempt: Option<JobAttempt<'_>>,
    ) -> Result<BuildingTable> {
        let sanitized = sanitize_model_id(model_id);
        let timestamp = chrono::Utc::now().format("%Y%m%dT%H%M%S%9f");
        // Nanoseconds plus a short uuid suffix make table names unique even
        // when two tokio tasks call create_table within the same nanosecond
        // (concurrent embedding generation on the same source).
        let suffix = &uuid::Uuid::new_v4().simple().to_string()[..8];
        let task_str = task.as_db_str();
        let table_name = format!("{source_id}__{task_str}__{sanitized}__{timestamp}_{suffix}");

        // Read the tenant ONCE from the catalog binding in force and use the
        // same segment for both the row's `tenant_id` and this key — a
        // `TenantSegment::parse` of the key's second path component always
        // agrees with the row it names.
        let tenant = self.catalog.current_tenant();
        let seg = TenantSegment::of(tenant.as_ref());
        let parquet_url = layout::result_table_url(&self.root, &seg, &table_name)?;
        if self.root.scheme() == Scheme::File {
            // The tenant-segment subdirectory is new territory: object_store's
            // local-filesystem `put` creates parent directories for the
            // Parquet write itself, but the ANN sidecar's writer is USearch's
            // raw FFI file open (`SidecarIndex::save`), which does NOT create
            // directories — it needs `{root}/{seg}/` to already exist.
            std::fs::create_dir_all(std::path::Path::new(self.root.path()).join(&seg))?;
        }
        let storage_precision = self.ann.storage_precision;

        self.catalog
            .create_result_table(CreateResultTableParams {
                table_name: &table_name,
                source_id,
                model_id,
                task,
                kind,
                derived_from,
                parquet_path: parquet_url.as_str(),
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
                storage_precision,
                oversample: self.ann.effective_oversample_for(storage_precision),
                created_at: crate::catalog::backend::now_sortable(),
                writer_id: Some(&self.writer_id),
                lease: Some(self.lease.lease()),
                job_attempt,
            })
            .await?;

        let building = BuildingTable::adopt(
            self.clone(),
            table_name,
            parquet_url,
            tenant,
            self.writer_id.to_string(),
            storage_precision,
        );

        // The W1 window: the `building` row is committed and heartbeating,
        // no bytes exist yet.
        #[cfg(feature = "test-hooks")]
        crate::store::mutable::test_hook::maybe_signal_table_created(&self.writer_id).await;

        Ok(building)
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
        let provider = build_result_table_provider(ctx, &self.registry, url, None).await?;
        self.install_result_schema(ctx)?;
        self.result_schema
            .add_result_table(format!("jammi.{name}"), provider, owner);
        Ok(())
    }

    /// The attestation half of [`BuildingTable::finish`]: compute the artifact
    /// digest over the durable Parquet bytes at `url`, build the
    /// [`MaterializationManifest`] from the producer's [`ProducingDescriptor`],
    /// the output-affecting [`MaterializationEnv`], and the resolved
    /// [`InputAnchor`]s, and write the `.materialization.json` sidecar (a
    /// sibling of the Parquet, distinct from the ANN `.manifest.json` index
    /// sidecar). Returns the manifest and its input anchors as the canonical
    /// JSON the promote CAS persists as the `input_anchors_json` summary
    /// column.
    ///
    /// The sidecar lands *before* the status flip — the same boundary the ANN
    /// sidecar uses — so a crash never leaves a `ready` table without a
    /// manifest; a crash between the write and the flip leaves a `building`
    /// row whose lease expires, which recovery then promotes from this very
    /// sidecar with the footer's true row count.
    ///
    /// Only [`BuildingTable::finish`] calls this on the writer's path (after a
    /// successful lease renew — K7); it is `pub` so a producer that composes
    /// the funnel by hand in a test can reach the same bytes.
    pub async fn write_attestation(
        &self,
        url: &StorageUrl,
        materialization: Materialization<'_>,
    ) -> Result<(MaterializationManifest, String)> {
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

        self.write_materialization_sidecar(url, &manifest).await?;

        let anchors_json = serde_json::to_string(&manifest.input_anchors)
            .map_err(|e| JammiError::Other(format!("serialise input anchors: {e}")))?;
        Ok((manifest, anchors_json))
    }

    /// Resolve the [`InputAnchor`] for an immutable result-table input: its
    /// content digest is its anchor ([`AnchorKind::ResultDigest`]). Prefers the
    /// digest the input's own manifest already attests (no re-read); falls back
    /// to recomputing it from the input's Parquet bytes for a pre-contract
    /// source table that carries no manifest.
    pub async fn result_digest_anchor(&self, table: &ResultTableRecord) -> Result<InputAnchor> {
        // A versioned table's anchor is its CURRENT version's identity (the
        // base version's identity is the base artifact hex, so publishing
        // the base moves no anchor).
        if let Some(identity) = self.current_version_identity(table).await? {
            return Ok(InputAnchor::result_digest(
                &table.table_name,
                &ArtifactDigest(identity),
            ));
        }
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

    /// The identity of `table`'s current version (`None` for a never-refreshed
    /// table), read off the version row under admin scope (the table was
    /// already resolved through the tenant-scoped read).
    pub async fn current_version_identity(
        &self,
        table: &ResultTableRecord,
    ) -> Result<Option<String>> {
        let Some(version) = table.current_version else {
            return Ok(None);
        };
        let row = TenantBinding::admin_scope(
            self.catalog
                .get_result_table_version(&table.table_name, version),
        )
        .await?;
        match row {
            Some(r) if r.status == ResultTableStatus::Ready.to_string() => {
                Ok(Some(r.identity.unwrap_or_default()))
            }
            _ => Err(JammiError::VersionUnavailable {
                table: table.table_name.clone(),
                version,
            }),
        }
    }

    /// `COUNT(*)` over the masked provider of a (possibly unpublished)
    /// version manifest — the exact live-row count a publish records.
    pub async fn count_live_rows(
        &self,
        ctx: &SessionContext,
        record: &ResultTableRecord,
        manifest: &VersionManifest,
    ) -> Result<usize> {
        let provider = self.build_masked_provider(ctx, record, manifest).await?;
        let df = ctx.read_table(provider)?;
        Ok(df.count().await?)
    }

    /// Read a manifest's deletion mask (empty when it lists none).
    pub async fn read_deletion_mask(
        &self,
        table: &str,
        manifest: &VersionManifest,
    ) -> Result<deletes::DeletionMask> {
        self.load_deletion_mask(table, manifest).await
    }

    /// Expiry's reap of one deleted version row's artifacts: its manifest and
    /// deletes always; its fragment and every segment stamped with it only
    /// when the CURRENT manifest does not list them (a fragment retained by
    /// reference stays). Returns the number of objects deleted.
    pub async fn reap_expired_version(
        &self,
        parquet_url: &StorageUrl,
        table_name: &str,
        version: i64,
        retained_fragments: &std::collections::HashSet<String>,
        retained_segments: &std::collections::HashSet<i64>,
    ) -> Result<usize> {
        let mut deleted = 0usize;
        let mut urls = vec![
            layout::version_manifest_url(parquet_url, version)?,
            layout::version_deletes_url(parquet_url, version)?,
        ];
        let fragment = layout::version_fragment_url(parquet_url, version)?;
        if !retained_fragments.contains(fragment.as_str()) {
            urls.push(fragment);
        }
        for url in urls {
            let handle = self.open_parquet(&url)?;
            if handle.delete_if_exists(&handle.data_path()?).await? == DeleteOutcome::Deleted {
                deleted += 1;
            }
        }
        for seg in self
            .catalog
            .list_index_segments_for_version(table_name, version)
            .await?
        {
            if retained_segments.contains(&seg.segment_id) {
                continue;
            }
            let url = StorageUrl::parse(&seg.index_path)?;
            let handle = self.open_index(&url)?;
            for ext in storage::sidecar_layout::sidecar_extensions(SidecarKind::Ann) {
                let Ok(path) = handle.sibling_path(ext) else {
                    continue;
                };
                if handle.delete_if_exists(&path).await? == DeleteOutcome::Deleted {
                    deleted += 1;
                }
            }
            self.catalog
                .delete_index_segment_row(table_name, seg.segment_id)
                .await?;
        }
        self.segment_sets.evict_table(table_name);
        Ok(deleted)
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

        // A versioned table (§3.6): the base check above is unchanged; then
        // every fragment digest and the deletes digest of the CURRENT version
        // are recomputed from the bytes, the identity chain is recomputed from
        // the parent's recorded identity, and both the version manifest's
        // identity and the catalog row's are compared. A mismatch names the
        // artifact that diverged.
        let mut unpinned = manifest.unpinned_inputs();
        if let Some(version) = table.current_version {
            let Some(vm) = self
                .read_version_manifest(&table.table_name, &parquet_url, version)
                .await?
            else {
                return Err(JammiError::VersionUnavailable {
                    table: table.table_name.clone(),
                    version,
                });
            };
            for fragment in &vm.fragments {
                let found = if fragment.url == table.parquet_path {
                    recomputed.clone()
                } else {
                    let url = StorageUrl::parse(&fragment.url)?;
                    let handle = self.open_parquet(&url)?;
                    let bytes = handle.get_bytes(&handle.data_path()?).await?;
                    ArtifactDigest::of_bytes(&bytes)
                };
                if found != fragment.digest {
                    return Ok(MatchVerdict::Mismatch {
                        expected: fragment.digest.0.clone(),
                        found: found.0,
                    });
                }
            }
            if let Some(deletes) = &vm.deletes {
                let url = StorageUrl::parse(&deletes.url)?;
                let handle = self.open_parquet(&url)?;
                let bytes = handle.get_bytes(&handle.data_path()?).await?;
                let found = ArtifactDigest::of_bytes(&bytes);
                if found != deletes.digest {
                    return Ok(MatchVerdict::Mismatch {
                        expected: deletes.digest.0.clone(),
                        found: found.0,
                    });
                }
            }
            let expected_identity = match vm.delta.descriptor.parent_identity() {
                // The base version: its identity IS the base artifact hex (D3).
                None => manifest.artifact.0.clone(),
                Some(parent_identity) => VersionManifest::compute_identity(
                    parent_identity,
                    &vm.definition_hash,
                    &vm.delta.descriptor,
                    &vm.fragments,
                    vm.deletes.as_ref(),
                )?,
            };
            if expected_identity != vm.identity {
                return Ok(MatchVerdict::Mismatch {
                    expected: expected_identity,
                    found: vm.identity.clone(),
                });
            }
            if let Some(recorded) = self.current_version_identity(table).await? {
                if recorded != vm.identity {
                    return Ok(MatchVerdict::Mismatch {
                        expected: vm.identity.clone(),
                        found: recorded,
                    });
                }
            }
            for anchor in &vm.delta.input_anchors {
                if anchor.kind == AnchorKind::UnpinnedAtInstant
                    && !unpinned.contains(&anchor.source)
                {
                    unpinned.push(anchor.source.clone());
                }
            }
        }
        if unpinned.is_empty() {
            Ok(MatchVerdict::Match)
        } else {
            Ok(MatchVerdict::MatchWithUnpinnedInputs { unpinned })
        }
    }

    /// Reconcile every result table left `building` by a dead writer,
    /// restoring the crash-consistency invariant of the catalog↔result-storage
    /// boundary.
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
    /// - **A live writer is never touched.** The sweep visits only `building`
    ///   rows whose writer lease is **absent or expired**
    ///   ([`Catalog::list_expired_building_tables`]); a row under a live lease
    ///   belongs to a writer in this or another process that is still
    ///   producing it, and every status flip below is a compare-and-set
    ///   carrying the same expired-lease predicate, so a writer that comes
    ///   back mid-sweep and renews wins the row. This is esc-094's fix: a peer
    ///   replica's restart no longer reaps a table another replica is seconds
    ///   from finishing.
    /// - **Reconciliation is terminal for a dead writer's row.** Each such row
    ///   is driven to exactly one terminal state — `ready` if its bytes are a
    ///   fully-valid closed Parquet whose manifest sidecar landed (promoted
    ///   with the *true* footer row count, the ANN sidecar rebuilt from the
    ///   Parquet so an embedding table self-heals even if its segment set never
    ///   landed), `failed` otherwise (missing bytes, a torn/partial Parquet, or
    ///   a valid Parquet with no manifest — the descriptor cannot be
    ///   reconstructed).
    /// - **Every deletion follows a one-row CAS.** The reaper deletes a row's
    ///   objects only after its own `building → failed` CAS affected exactly
    ///   one row; the promote arm first *claims* the row
    ///   ([`Catalog::claim_expired_building_table`] — the recoverer becomes the
    ///   writer, heartbeating a fresh lease) and only then rebuilds and
    ///   promotes under that ownership. A failed delete is logged and left for
    ///   reconcile, never swallowed.
    /// - **A promoted row's `row_count` is the truth on disk**, read from the
    ///   Parquet footer — never the count the writer *intended* before it
    ///   crashed.
    ///
    /// The sweep is idempotent: re-running it after it has reconciled every
    /// expired-lease `building` row is a no-op.
    ///
    /// # Cross-tenant scope
    ///
    /// Recovery runs under [`crate::session::JammiSession::with_admin_scope`]
    /// — the one named implicit-admin pass — so it enumerates, reconciles, and
    /// **deletes the bytes of** expired-lease `building` rows owned by
    /// **every** tenant, not only the (unscoped, GLOBAL) startup session's own
    /// rows, and it does so even when the store is bound to one tenant. Each
    /// promoted/failed row keeps its own `tenant_id`; the bypass is confined
    /// to this sweep and clears the instant it returns.
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
    /// see and write across every tenant's expired-lease `building` rows.
    async fn recover_inner(&self) -> Result<()> {
        let expired = self.catalog.list_expired_building_tables().await?;
        for table in expired {
            self.reconcile_expired_building_row(table).await?;
        }
        self.recover_expired_versions().await?;
        self.reconcile_ready_manifests().await?;
        Ok(())
    }

    /// The version arm of recovery: every `building` VERSION row whose lease
    /// expired is claimed (fencing its writer), failed by CAS, and its
    /// artifacts stamped with that number reaped — never promoted (a delta is
    /// cheap to redo), never touching the table row or the base artifacts.
    async fn recover_expired_versions(&self) -> Result<()> {
        for v in self.catalog.list_expired_building_versions().await? {
            let Some(table) = self.catalog.get_result_table(&v.table_name).await? else {
                continue;
            };
            if !self
                .catalog
                .claim_expired_building_version(
                    &v.table_name,
                    v.version,
                    &self.writer_id,
                    self.lease.lease(),
                )
                .await?
            {
                continue;
            }
            let cas = crate::catalog::version_repo::VersionCas::writer(
                &v.table_name,
                v.version,
                &self.writer_id,
                parse_owner(&table)?,
            );
            match self.catalog.fail_building_version(&cas).await {
                Ok(()) => {}
                Err(e) if is_cas_miss(&e) => {
                    warn!(table = v.table_name, version = v.version, outcome = %e, "Recovery: version row moved on; nothing deleted");
                    continue;
                }
                Err(e) => return Err(e),
            }
            let parquet_url = StorageUrl::parse(&table.parquet_path)?;
            if let Err(e) = self
                .reap_version_artifacts(&parquet_url, &v.table_name, v.version)
                .await
            {
                warn!(table = v.table_name, version = v.version, error = %e, "Recovery: version artifact reap did not complete; reconcile reaps it");
            }
        }
        Ok(())
    }

    /// The recovery arm for ONE expired-lease `building` row: claim it
    /// (fencing whatever writer is or was alive), then drive it to exactly
    /// one terminal state, deleting bytes only after the CAS that licenses
    /// it. Shared by [`Self::recover_inner`] (the admin-scoped, cross-tenant
    /// startup sweep) and [`crate::store::reconcile`]'s pass (esc-094: an
    /// expired-lease `building` row is reaped through THIS arm — claim, then
    /// fail-CAS or promote, then delete — never through reconcile's orphan
    /// arm, which performs no claim and no CAS at all). The binding in force
    /// when this runs determines scope: admin-scoped from `recover_inner`,
    /// or whatever scope the caller (a tenant-bound [`Self::reconcile`], or
    /// admin-scoped [`Self::reconcile_all`]) is already running under —
    /// [`crate::catalog::result_repo::ResultTableCas::expired`] renders the
    /// matching tenant arm either way.
    ///
    /// The read-only classification [`ExpiredRowOutcome`] documents: performs
    /// the existence/validity/manifest-presence checks against the object
    /// store and claims or deletes NOTHING. The single source of truth both
    /// [`Self::reconcile_expired_building_row`] (apply) and `reconcile`'s
    /// dry-run preview branch on. A dry-run cannot predict a concurrent claim
    /// race, so this reports what would happen ABSENT interference — the
    /// same caveat every other preview in `reconcile` carries.
    ///
    /// For a `Promote` row, the payload is simply the row's FULL currently
    /// referenced key set (via [`Self::referenced_result_keys`], scoped to
    /// this one row) — this classification predicts NOTHING about what
    /// [`Self::rebuild_index_from_parquet`]'s destructive purge will or will
    /// not rewrite (see [`ExpiredRowOutcome`]'s own doc comment for why: a
    /// promotion's internal rebuild is not a reclaim this pass reports).
    ///
    /// A missing Parquet is [`ExpiredRowOutcome::Untouched`] when caught by
    /// the `exists()` check below; a Parquet that vanishes in the window
    /// between that check and this function's own single read of its bytes
    /// (`storage::reader::validate_and_count_parquet_rows`, which restores
    /// the "vanish reads as invalid, never as an aborting error" semantics)
    /// re-classifies as [`ExpiredRowOutcome::Reap`] — the identical outcome
    /// an already-torn Parquet gets — rather than propagating an
    /// object-store error that would abort the whole reconcile pass over one
    /// row's benign race.
    async fn classify_expired_row(&self, table: &ResultTableRecord) -> Result<ExpiredRowOutcome> {
        let parquet_url = StorageUrl::parse(&table.parquet_path)?;
        let parquet_handle = self.open_parquet(&parquet_url)?;
        let parquet_path = parquet_handle.data_path()?;
        if !parquet_handle.exists(&parquet_path).await? {
            return Ok(ExpiredRowOutcome::Untouched);
        }
        #[cfg(feature = "test-hooks")]
        reconcile_test_hooks::maybe_park_before_parquet_reread(&table.table_name).await;
        // One read validates AND (were it still needed) would count rows in
        // a single object-store fetch — kept as a single read even though
        // this classification no longer consumes the count. A vanish
        // between the `exists()` check above and this read (the classify
        // window race) resolves through the SAME `None` arm a torn/invalid
        // Parquet already takes, never an `Err` that would abort this pass.
        let is_valid = storage::reader::validate_and_count_parquet_rows(&parquet_handle)
            .await?
            .is_some();
        if !is_valid {
            return Ok(ExpiredRowOutcome::Reap);
        }
        if self
            .read_materialization_manifest(&parquet_url)
            .await?
            .is_none()
        {
            return Ok(ExpiredRowOutcome::Reap);
        }

        let referenced = self
            .referenced_result_keys(std::slice::from_ref(table), &[])
            .await?;
        Ok(ExpiredRowOutcome::Promote {
            keeps: referenced.exact,
            dir_prefixes: referenced.dir_prefixes,
        })
    }

    /// The root-relative ANN sidecar-sibling keys a table's CURRENT
    /// `index_segments` rows name RIGHT NOW — the exact per-segment
    /// enumeration [`Self::purge_segments`] deletes from and
    /// [`Self::reap_candidate_keys`] previews (both call this rather than
    /// hand-copying the loop). Never includes a segment's own base
    /// `index_path` key: no writer creates a file there and no deleter ever
    /// deletes one. A segment whose `index_path` does not parse as a
    /// [`StorageUrl`] is silently excluded here (this is a candidate
    /// PREVIEW, not the destructive delete `purge_segments` performs — that
    /// still hard-errors on the same row, per its own doc comment). Never
    /// called by [`Self::classify_expired_row`] — a `Promote` row's payload
    /// is the protect-side [`Self::referenced_result_keys`] set, not a
    /// deletion-side prediction (see that classification's own doc
    /// comment).
    async fn segment_ann_sidecar_keys(&self, table_name: &str) -> Result<BTreeSet<String>> {
        let mut keys = BTreeSet::new();
        for seg in self.catalog.list_index_segments(table_name).await? {
            let Ok(seg_url) = StorageUrl::parse(&seg.index_path) else {
                continue;
            };
            for ext in storage::sidecar_layout::sidecar_extensions(SidecarKind::Ann) {
                if let Ok(sib) = layout::sidecar_url(&seg_url, ext) {
                    if let Some(rel) = reconcile::relative_to(&self.root, &sib) {
                        keys.insert(rel);
                    }
                }
            }
        }
        Ok(keys)
    }

    /// The recovery arm's outcome for ONE expired-lease `building` row —
    /// see [`ExpiredRowDeletion`] for what the returned value means to
    /// `reconcile`'s pre-pass accounting; [`Self::recover_inner`] ignores it
    /// (a background sweep has no report to account into).
    async fn reconcile_expired_building_row(
        &self,
        table: ResultTableRecord,
    ) -> Result<ExpiredRowDeletion> {
        let tenant = parse_owner(&table)?;
        let cas = ResultTableCas::expired(&table.table_name, tenant);
        let parquet_url = StorageUrl::parse(&table.parquet_path)?;

        match self.classify_expired_row(&table).await? {
            ExpiredRowOutcome::Untouched => {
                warn!(
                    table = table.table_name,
                    "Recovery: Parquet missing, marking failed"
                );
                // No bytes to reap: the CAS is the whole arm. A miss means
                // the writer renewed or a peer recoverer got here first —
                // skip.
                if let Err(e) = self.catalog.fail_building_table(&cas).await {
                    if !is_cas_miss(&e) {
                        return Err(e);
                    }
                    warn!(table = table.table_name, outcome = %e, "Recovery: row moved on; skipped");
                }
                Ok(ExpiredRowDeletion::Untouched)
            }
            ExpiredRowOutcome::Reap => {
                warn!(
                    table = table.table_name,
                    "Recovery: torn or invalid building row, marking failed and deleting"
                );
                Ok(ExpiredRowDeletion::Reaped(
                    self.reap_after_fail_cas(&cas, &parquet_url).await?,
                ))
            }
            ExpiredRowOutcome::Promote { .. } => {
                let parquet_handle = self.open_parquet(&parquet_url)?;
                // The manifest sidecar is present (written before the flip),
                // so its summary columns can be backfilled as part of the
                // same promotion the live path performs. Claim the row FIRST
                // — the recoverer becomes the writer, heartbeating a fresh
                // lease — then rebuild and promote under that ownership.
                //
                // esc-484 item "manifest vanished between classify and
                // perform": a concurrent pass may have reaped this row (or
                // its sidecar was otherwise lost) in the moment between the
                // `classify_expired_row` call above and this re-read — never
                // abort the WHOLE reconcile pass over that race; re-classify
                // this row as `Reap` (exactly what `classify_expired_row`
                // itself would return with no manifest present) instead.
                #[cfg(feature = "test-hooks")]
                reconcile_test_hooks::maybe_park_before_manifest_reread(&table.table_name).await;
                let Some(manifest) = self.read_materialization_manifest(&parquet_url).await? else {
                    warn!(
                        table = table.table_name,
                        "Recovery: classified Promote but its manifest sidecar vanished before \
                         perform; re-classifying as Reap"
                    );
                    return Ok(ExpiredRowDeletion::Reaped(
                        self.reap_after_fail_cas(&cas, &parquet_url).await?,
                    ));
                };
                let Some(recovered) = self.claim_expired(&cas, &table, tenant).await? else {
                    warn!(table = table.table_name, "Recovery: claim lost; skipped");
                    return Ok(ExpiredRowDeletion::Untouched);
                };
                // esc-484 advisory: a further race window opens between the
                // manifest re-read above (now satisfied) and this row-count
                // read — the claim is held, but the Parquet itself can still
                // vanish out from under it before this fetch runs. Never
                // propagate that as an aborting `Err`; re-classify this row
                // as `Reap` (the same outcome an already-torn Parquet gets)
                // under the CLAIM's own CAS, exactly like the manifest-vanish
                // arm above does under the PRE-claim CAS.
                #[cfg(feature = "test-hooks")]
                reconcile_test_hooks::maybe_park_before_post_claim_row_count(&table.table_name)
                    .await;
                let row_count = match storage::reader::count_parquet_rows(&parquet_handle).await {
                    Ok(n) => n,
                    Err(storage::StorageError::Io {
                        source: object_store::Error::NotFound { .. },
                        ..
                    }) => {
                        warn!(
                            table = table.table_name,
                            "Recovery: classified Promote but its Parquet vanished after claim, \
                             before the post-claim row-count read; re-classifying as Reap"
                        );
                        let reaped = self
                            .reap_after_fail_cas(&recovered.cas(), &parquet_url)
                            .await?;
                        recovered.detach();
                        return Ok(ExpiredRowDeletion::Reaped(reaped));
                    }
                    Err(e) => return Err(e.into()),
                };
                // Rebuild the ANN index as a fresh single segment if this is
                // an embedding table (self-healing even if its segment set
                // never landed, or landed torn). Renew before the
                // destructive purge; a renew miss abandons this arm silently
                // (no deletion).
                let mut promoted_purged = BTreeSet::new();
                if table.task.is_embedding() {
                    let renew = self
                        .catalog
                        .renew_lease(&recovered.cas(), self.lease.lease())
                        .await;
                    if let Err(e) = renew {
                        if !is_cas_miss(&e) {
                            return Err(e);
                        }
                        warn!(table = table.table_name, outcome = %e, "Recovery: claim lost before rebuild; skipped");
                        recovered.detach();
                        return Ok(ExpiredRowDeletion::Untouched);
                    }
                    match self
                        .rebuild_index_from_parquet(&recovered, &parquet_handle, &table)
                        .await
                    {
                        Ok(purged) => {
                            // The rebuild succeeded: `purged` is exactly the
                            // keys `purge_segments` actually deleted (some of
                            // which the rebuild immediately rewrote at the
                            // SAME key, e.g. a fresh segment 0 — that key's
                            // fresh bytes are protected normally once this
                            // row re-queries as `ready`, never through this
                            // exclusion). A promotion's internal rebuild is
                            // not a reclaim: recorded here for EXCLUSION from
                            // this pass's accounting, never credited.
                            promoted_purged = purged;
                        }
                        Err(RebuildFailure { error: e, purged }) => {
                            if is_cas_miss(&e) {
                                warn!(table = table.table_name, outcome = %e, "Recovery: claim lost during rebuild; skipped");
                                recovered.detach();
                                return Ok(ExpiredRowDeletion::Untouched);
                            }
                            warn!(
                                table = table.table_name,
                                error = %e,
                                "Recovery: failed to rebuild index, proceeding without; the \
                                 segments its purge already deleted are excluded from this \
                                 pass, not credited"
                            );
                            // A later step (reading the Parquet's batches,
                            // building the index, writing the fresh segment)
                            // can fail AFTER `purge_segments` already ran —
                            // those bytes it actually deleted are gone from
                            // storage regardless, so `purged` must still be
                            // excluded here, never silently dropped into the
                            // ordinary orphan arm against a stale listing
                            // snapshot. A key `purge_segments` itself FAILED
                            // to delete is never in `purged` at all — it
                            // survives on disk, unreferenced (its catalog row
                            // is gone either way), for the ordinary age-gated
                            // arm to reclaim normally, this pass or a later
                            // one.
                            promoted_purged = purged;
                        }
                    }
                }
                let anchors_json = serde_json::to_string(&manifest.input_anchors)
                    .map_err(|e| JammiError::Other(format!("serialise input anchors: {e}")))?;
                let promoted = self
                    .catalog
                    .promote_result_table_with_manifest(
                        &recovered.cas(),
                        row_count,
                        manifest.definition_hash.as_str(),
                        &anchors_json,
                    )
                    .await;
                // The row is terminal (or lost) under this recoverer either
                // way: detach the handle so Drop marks nothing.
                recovered.detach();
                match promoted {
                    Ok(_) => {}
                    Err(e) if is_cas_miss(&e) => {
                        warn!(table = table.table_name, outcome = %e, "Recovery: promote superseded; skipped");
                    }
                    Err(e) => return Err(e),
                }
                Ok(ExpiredRowDeletion::PromotedPurged(promoted_purged))
            }
        }
    }

    /// Claim the expired-lease row `cas` names for this store's writer and
    /// return the [`BuildingTable`] the recoverer now holds (heartbeat
    /// running), or `None` when the claim matched zero rows — the writer
    /// renewed, or a peer recoverer claimed first — in which case nothing was
    /// written.
    async fn claim_expired(
        &self,
        cas: &ResultTableCas,
        table: &ResultTableRecord,
        tenant: Option<TenantId>,
    ) -> Result<Option<BuildingTable>> {
        // A FRESH id per claim, never this process's
        // OWN `self.writer_id` — if the row this claim targets happens to be
        // THIS process's own lapsed writer, re-stamping the SAME id would
        // leave the lapsed `BuildingTable` handle's `Owner::Writer(self.writer_id)`
        // CAS still matching (no fence at all: the two handles would share
        // one identity and race each other for the rest of the row's life —
        // the rebuild below could purge segments the lapsed writer is still
        // appending). A claim is always a distinct identity from every
        // `ResultStore`'s own writer_id, so the CAS the lapsed writer's next
        // renew/append/promote issues always misses.
        let claim_writer_id = format!("{}/claim-{}", self.writer_id, uuid::Uuid::new_v4());
        if !self
            .catalog
            .claim_expired_building_table(cas, &claim_writer_id, self.lease.lease())
            .await?
        {
            return Ok(None);
        }
        let parquet_url = StorageUrl::parse(&table.parquet_path)?;
        Ok(Some(BuildingTable::adopt(
            self.clone(),
            table.table_name.clone(),
            parquet_url,
            tenant,
            claim_writer_id,
            table.storage_precision.unwrap_or_default(),
        )))
    }

    /// The reaper's fail arm: the `building -> failed` CAS under `cas` FIRST,
    /// then — only if it affected exactly one row — the row's objects are
    /// deleted. A CAS miss (the writer renewed, or a peer got here first)
    /// deletes nothing and returns an empty set. Otherwise returns EXACTLY
    /// the root-relative keys [`Self::delete_objects_after_cas`] actually
    /// deleted — a key whose delete failed is left OUT (logged, never
    /// swallowed into a false credit): `reconcile`'s pre-pass unions only
    /// this returned set into `bytes_reclaimed`, so a partial failure here
    /// never over-reports what this pass reclaimed; the un-deleted key falls
    /// to the ordinary orphan arm (or a later reconcile pass) to retry.
    async fn reap_after_fail_cas(
        &self,
        cas: &ResultTableCas,
        parquet_url: &StorageUrl,
    ) -> Result<BTreeSet<String>> {
        match self.catalog.fail_building_table(cas).await {
            Ok(()) => {}
            Err(e) if is_cas_miss(&e) => {
                warn!(table = cas.table, outcome = %e, "Recovery: row moved on; nothing deleted");
                return Ok(BTreeSet::new());
            }
            Err(e) => return Err(e),
        }
        match self.delete_objects_after_cas(parquet_url, cas).await {
            Ok(outcome) => Ok(outcome.deleted),
            Err(e) => {
                warn!(
                    table = cas.table,
                    error = %e,
                    "Recovery: object delete after the fail CAS did not complete; reconcile reaps it"
                );
                Ok(BTreeSet::new())
            }
        }
    }

    /// Reconcile already-`ready` result tables against the materialization
    /// contract: a post-contract row (one whose catalog `definition_hash` is
    /// set, so it was promoted under the contract) whose `.materialization.json`
    /// sidecar is now absent is a corruption — the attestation a verifier would
    /// read is gone. Such a row is driven to `failed` by a `status = 'ready'`
    /// compare-and-set and, only after that CAS affected one row, its bytes
    /// are reaped — rather than left queryable with a silently-missing
    /// manifest.
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
            // The version arm (D14(i)): a current version whose manifest is
            // definitively absent fails the VERSION row only — the table row
            // and its base artifacts are untouched; reads see the placeholder.
            if let Some(version) = table.current_version {
                let manifest_url = layout::version_manifest_url(&parquet_url, version)?;
                let vh = self.open_parquet(&manifest_url)?;
                if !vh.exists(&vh.data_path()?).await? {
                    warn!(
                        table = table.table_name,
                        version,
                        "Recovery: current version manifest is absent; failing the version row"
                    );
                    self.catalog
                        .fail_ready_version(&table.table_name, version)
                        .await?;
                    self.segment_sets.evict_table(&table.table_name);
                }
            }
            let handle = self.open_parquet(&parquet_url)?;
            let sidecar = materialization_sidecar_path(&handle)?;
            if handle.exists(&sidecar).await? {
                continue;
            }
            warn!(
                table = table.table_name,
                "Recovery: post-contract ready table is missing its materialization \
                 manifest sidecar; marking failed and deleting"
            );
            if !self
                .catalog
                .fail_ready_result_table(&table.table_name)
                .await?
            {
                warn!(
                    table = table.table_name,
                    "Recovery: ready row moved on; nothing deleted"
                );
                continue;
            }
            // A `ready` row carries no lease, so the expired-lease owner arm
            // names it for the segment purge.
            let cas = ResultTableCas::expired(&table.table_name, parse_owner(&table)?);
            if let Err(e) = self.delete_objects_after_cas(&parquet_url, &cas).await {
                warn!(
                    table = table.table_name,
                    error = %e,
                    "Recovery: object delete after the fail CAS did not complete; reconcile reaps it"
                );
            }
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
            let _ = owner;
            let handle = self.open_parquet(&url)?;
            let path = handle.data_path()?;
            if handle.exists(&path).await? {
                if let Err(e) = self.bind_result_table(ctx, &table).await {
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

    /// Search an embedding table for the nearest neighbors of a query vector —
    /// the PLACED entry, for the online consumers (the `Search` leaf's peer,
    /// the context-set single-shot retrieval). Uses the placed ANN index when
    /// available, falls back to exact brute-force search over the whole
    /// Parquet otherwise.
    ///
    /// Routes through [`PlacedIndex::search_final_placed`], so a multi-segment
    /// quantized / `Binary` table returns the exact-rescored, cross-segment
    /// comparable top-`k` — never raw per-segment candidate distances — and a
    /// segment a peer owns is searched at that peer. The oversample is the
    /// table's own stamped default (no per-request override on this lane).
    pub async fn search_vectors(
        &self,
        ctx: &SessionContext,
        table: &ResultTableRecord,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(String, f32)>> {
        match self.resolve_search_mode(table).await? {
            Some(index) => {
                let oversample = self.ann.resolve_oversample(None, table.oversample);
                index.search_final_placed(query, k, oversample).await
            }
            None => {
                verify_query_width_against_catalog(table, query)?;
                crate::index::exact::exact_vector_search(ctx, &table.table_name, query, k).await
            }
        }
    }

    /// [`Self::search_vectors`]'s FORCE-LOCAL twin, for the batch consumers
    /// (the eval runner's per-query loop): ignores placement, loads every
    /// segment locally through [`Self::resolve_search_mode_local`] and
    /// searches the sync [`SegmentedIndex::search_final`]. Any replica can
    /// (the content-addressed cache over the shared root); a batch build never
    /// fans out per node.
    pub async fn search_vectors_local(
        &self,
        ctx: &SessionContext,
        table: &ResultTableRecord,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(String, f32)>> {
        match self.resolve_search_mode_local(table).await? {
            Some(index) => {
                let oversample = self.ann.resolve_oversample(None, table.oversample);
                index.search_final(query, k, oversample)
            }
            None => {
                verify_query_width_against_catalog(table, query)?;
                crate::index::exact::exact_vector_search(ctx, &table.table_name, query, k).await
            }
        }
    }

    /// Resolve whether a table's ANN index (its whole segment set) can serve a
    /// PLACED search, or whether the caller must fall back to exact
    /// brute-force. Returns `Some(PlacedIndex)` over every segment, `None` for
    /// exact fallback. The online entry: placement is read here, at every
    /// call, for every segment.
    ///
    /// A table with no segments resolves to `None`. When every segment's
    /// owner list is empty (this process owns them all — the [`AllLocal`]
    /// default, or a single node) the set is loaded exactly as
    /// [`Self::resolve_search_mode_local`] loads it, including its whole-table
    /// exact fallback on any load failure, and searched through the same sync
    /// kernels. When at least one segment is owned by a peer the set is
    /// `Mixed`: local segments are loaded, remote ones are recorded with their
    /// owners and never loaded here; a local load failure in that shape is
    /// [`JammiError::Unavailable`] — a multi-node table is never exact-scanned
    /// silently.
    pub async fn resolve_search_mode(
        &self,
        table: &ResultTableRecord,
    ) -> Result<Option<PlacedIndex>> {
        let segments = self.catalog.list_index_segments(&table.table_name).await?;
        if segments.is_empty() {
            return Ok(None);
        }
        let mut owners = Vec::with_capacity(segments.len());
        for seg in &segments {
            owners.push(
                self.placement
                    .owners(&table.table_name, SegmentId(seg.segment_id))
                    .await,
            );
        }
        let precision = table.storage_precision.unwrap_or_default();
        if owners.iter().all(Vec::is_empty) {
            // Every segment is local: identical to the force-local entry,
            // including its version-aware masked load — a `PlacedIndex` never
            // bypasses the mask the online search verb promises. `sources`
            // (a flat, unversioned `list_index_segments` load) is not used on
            // this arm; the versioned resolver owns segment selection.
            return Ok(self
                .resolve_search_mode_local(table)
                .await?
                .map(|index| PlacedIndex::from_local(
                    index,
                    &table.table_name,
                    Arc::clone(&self.peer_transport),
                    Arc::clone(&self.segment_cache),
                    self.ann,
                    self.peer_local_load_bytes,
                    table.dimensions,
                    Arc::clone(&self.peer_failures),
                )));
        }
        // `Mixed`: load what this process owns, record what a peer owns. A
        // local load failure here is `Unavailable` — a multi-node table is
        // never silently exact-scanned. (Not version-aware: a versioned
        // table's placed/Mixed path is `list_index_segments`' flat,
        // unversioned segment set — the same limitation `resolve_search_mode`
        // carried before the all-local arm above was closed. Multi-node
        // deployments of a versioned, refreshed table are out of scope here.)
        let sources = {
            let mut sources = Vec::with_capacity(segments.len());
            for (seg, owners) in segments.iter().zip(owners) {
                let index_url = StorageUrl::parse(&seg.index_path)?;
                if owners.is_empty() {
                    let index = self
                        .segment_cache
                        .load_segment(&index_url, &self.ann, precision)
                        .await
                        .map_err(|e| JammiError::Unavailable {
                            resource: format!("segment {}/{}", table.table_name, seg.segment_id),
                            reason: format!("local load failed on a placed table: {e}"),
                        })?;
                    sources.push(SegmentSource::Local(SegmentId(seg.segment_id), index));
                } else {
                    sources.push(SegmentSource::Remote {
                        segment_id: SegmentId(seg.segment_id),
                        owners,
                        row_count: seg.row_count,
                        index_url,
                    });
                }
            }
            sources
        };
        Ok(Some(PlacedIndex::with_sources(
            sources,
            &table.table_name,
            precision,
            Arc::clone(&self.peer_transport),
            Arc::clone(&self.segment_cache),
            self.ann,
            self.peer_local_load_bytes,
            table.dimensions,
            Arc::clone(&self.peer_failures),
        )?))
    }

    /// Resolve whether a table's ANN index (its whole segment set) can serve a
    /// FORCE-LOCAL search, or whether the caller must fall back to exact
    /// brute-force. Returns `Some(SegmentedIndex)` merging every segment, `None`
    /// for exact fallback. The batch consumers' entry (the neighbor-graph
    /// build holds the returned index across a whole build): placement is
    /// ignored and every segment is loaded here.
    ///
    /// A table with no segments resolves to `None`. If *any* segment fails to
    /// load — a torn bundle, or a drifted-precision segment failing
    /// [`SidecarIndex::load`]'s strict `scalar_kind` check — the whole table
    /// falls back to exact (`None`), never a `SegmentedIndex` over the surviving
    /// subset: dropping a failed segment would silently make its rows
    /// unsearchable, surfacing "no matches" for rows that exist. Each segment is
    /// loaded through the content-addressed segment cache; the catalog row's own
    /// persisted precision — never the deployment default — is what each load
    /// verifies against.
    pub async fn resolve_search_mode_local(
        &self,
        table: &ResultTableRecord,
    ) -> Result<Option<Arc<SegmentedIndex>>> {
        let expected_precision = table.storage_precision.unwrap_or_default();
        match table.current_version {
            None => {
                // A never-refreshed table: today's path over the base set
                // (`version IS NULL`), cached once the table is `ready` (its
                // base set is frozen from then on).
                let cacheable = table.status == ResultTableStatus::Ready.to_string();
                if cacheable {
                    if let Some(set) = self.segment_sets.get(&table.table_name, None) {
                        return Ok(Some(Arc::clone(&set.index)));
                    }
                }
                let segments = self
                    .catalog
                    .list_base_index_segments(&table.table_name)
                    .await?;
                if segments.is_empty() {
                    return Ok(None);
                }
                let mut loaded = Vec::with_capacity(segments.len());
                for seg in segments {
                    let url = StorageUrl::parse(&seg.index_path)?;
                    match self
                        .segment_cache
                        .load_segment(&url, &self.ann, expected_precision)
                        .await
                    {
                        Ok(index) => loaded.push((SegmentId(seg.segment_id), index)),
                        Err(e) => {
                            warn!(
                                table = table.table_name,
                                segment = seg.segment_id,
                                error = %e,
                                "Segment index unavailable, falling back to whole-table exact search"
                            );
                            return Ok(None);
                        }
                    }
                }
                let index = Arc::new(SegmentedIndex::new(loaded)?);
                if cacheable {
                    self.segment_sets.insert(
                        &table.table_name,
                        None,
                        Arc::new(LoadedSegmentSet {
                            index: Arc::clone(&index),
                            mask: Arc::new(deletes::DeletionMask::empty()),
                        }),
                    );
                }
                Ok(Some(index))
            }
            Some(version) => {
                if let Some(set) = self.segment_sets.get(&table.table_name, Some(version)) {
                    return Ok(Some(Arc::clone(&set.index)));
                }
                // Manifest resolution: definitive absence or a failed row is
                // the typed `VersionUnavailable`; an `exists()` error propagates.
                let manifest = self.resolve_version_manifest(table, version).await?;
                let mask = Arc::new(
                    self.load_deletion_mask(&table.table_name, &manifest)
                        .await?,
                );
                let parquet_url = StorageUrl::parse(&table.parquet_path)?;
                let mut loaded = Vec::with_capacity(manifest.segments.len());
                for seg in &manifest.segments {
                    let url = layout::segment_url(&parquet_url, seg.segment_id)?;
                    match self
                        .segment_cache
                        .load_segment(&url, &self.ann, expected_precision)
                        .await
                    {
                        Ok(index) => loaded.push((SegmentId(seg.segment_id), seg.version, index)),
                        Err(e) => {
                            warn!(
                                table = table.table_name,
                                version,
                                segment = seg.segment_id,
                                error = %e,
                                "Segment index unavailable, falling back to masked exact search"
                            );
                            return Ok(None);
                        }
                    }
                }
                if loaded.is_empty() {
                    return Ok(None);
                }
                let index = Arc::new(SegmentedIndex::new_masked(loaded, Arc::clone(&mask))?);
                self.segment_sets.insert(
                    &table.table_name,
                    Some(version),
                    Arc::new(LoadedSegmentSet {
                        index: Arc::clone(&index),
                        mask,
                    }),
                );
                Ok(Some(index))
            }
        }
    }

    /// The loaded-set cache (evicted per table on bind / publish / delete).
    pub fn segment_sets(&self) -> &Arc<SegmentSetCache> {
        &self.segment_sets
    }

    /// Read a version's `.version.json` through the per-table cache. `Ok(None)`
    /// when the object is definitively absent; an `exists()` error propagates.
    pub async fn read_version_manifest(
        &self,
        table: &str,
        parquet_url: &StorageUrl,
        version: i64,
    ) -> Result<Option<Arc<VersionManifest>>> {
        if let Some(m) = self.segment_sets.get_manifest(table, version) {
            return Ok(Some(m));
        }
        let url = layout::version_manifest_url(parquet_url, version)?;
        let handle = self.open_parquet(&url)?;
        let path = handle.data_path()?;
        if !handle.exists(&path).await? {
            return Ok(None);
        }
        let bytes = handle.get_bytes(&path).await?;
        let manifest = Arc::new(VersionManifest::from_json_bytes(&bytes)?);
        self.segment_sets
            .insert_manifest(table, version, Arc::clone(&manifest));
        Ok(Some(manifest))
    }

    /// Write a version's `.version.json` (idempotent re-PUT at the same path).
    pub async fn write_version_manifest(
        &self,
        parquet_url: &StorageUrl,
        manifest: &VersionManifest,
    ) -> Result<StorageUrl> {
        let url = layout::version_manifest_url(parquet_url, manifest.version)?;
        let handle = self.open_parquet(&url)?;
        let path = handle.data_path()?;
        handle
            .put_bytes(&path, manifest.to_json_bytes()?.into())
            .await?;
        Ok(url)
    }

    /// Resolve the CURRENT version's manifest for a read: the version row
    /// must be `ready` and the manifest present, else the typed
    /// [`JammiError::VersionUnavailable`] (D14(i)). Runs the row read under
    /// admin scope: the caller already resolved `table` through the
    /// tenant-scoped table read, and a version inherits its table's owner.
    async fn resolve_version_manifest(
        &self,
        table: &ResultTableRecord,
        version: i64,
    ) -> Result<Arc<VersionManifest>> {
        let unavailable = || JammiError::VersionUnavailable {
            table: table.table_name.clone(),
            version,
        };
        let row = TenantBinding::admin_scope(
            self.catalog
                .get_result_table_version(&table.table_name, version),
        )
        .await?;
        match row {
            Some(r) if r.status == ResultTableStatus::Ready.to_string() => {}
            _ => return Err(unavailable()),
        }
        let parquet_url = StorageUrl::parse(&table.parquet_path)?;
        self.read_version_manifest(&table.table_name, &parquet_url, version)
            .await?
            .ok_or_else(unavailable)
    }

    /// Load a manifest's deletion mask (empty when the manifest lists none).
    async fn load_deletion_mask(
        &self,
        table: &str,
        manifest: &VersionManifest,
    ) -> Result<deletes::DeletionMask> {
        match &manifest.deletes {
            None => Ok(deletes::DeletionMask::empty()),
            Some(d) => {
                let url = StorageUrl::parse(&d.url)?;
                let handle = self.open_parquet(&url)?;
                deletes::DeletionMask::read(&handle, table).await
            }
        }
    }

    /// The ONE registration path for a ready table (D8): `current_version`
    /// `None` → today's single `ListingTable` over the base Parquet;
    /// `Some(N)` → the [`MaskedTableProvider`] over version `N`'s fragments
    /// under its deletion mask; a version whose manifest cannot be resolved →
    /// the [`PlaceholderProvider`] (planning succeeds, every scan is the typed
    /// `VersionUnavailable`), registered under the row's owner so a peer
    /// tenant still resolves not-found. Evicts the table's loaded segment
    /// sets. Called by startup, `BuildingTable::finish` and `publish_version`.
    pub async fn bind_result_table(
        &self,
        ctx: &SessionContext,
        record: &ResultTableRecord,
    ) -> Result<()> {
        let owner = parse_owner(record)?;
        let url = StorageUrl::parse(&record.parquet_path)?;
        self.segment_sets.evict_table(&record.table_name);
        let Some(version) = record.current_version else {
            return self
                .register_table(ctx, &record.table_name, &url, owner)
                .await;
        };
        let Some(dimensions) = record.dimensions else {
            return Err(JammiError::Catalog(format!(
                "result table '{}' is versioned (current_version = {version}) but carries no                  dimensions — a catalog invariant violation",
                record.table_name
            )));
        };
        let manifest = match self.resolve_version_manifest(record, version).await {
            Ok(m) => m,
            Err(JammiError::VersionUnavailable { .. }) => {
                warn!(
                    table = record.table_name,
                    version,
                    "current version manifest unresolvable; registering a placeholder provider"
                );
                let provider = Arc::new(PlaceholderProvider::new(
                    record.table_name.clone(),
                    version,
                    crate::store::schema::embedding_table_schema(dimensions.max(0) as usize),
                ));
                self.install_result_schema(ctx)?;
                self.result_schema.add_result_table(
                    format!("jammi.{}", record.table_name),
                    provider,
                    owner,
                );
                return Ok(());
            }
            Err(e) => return Err(e),
        };
        let provider = self.build_masked_provider(ctx, record, &manifest).await?;
        self.install_result_schema(ctx)?;
        self.result_schema.add_result_table(
            format!("jammi.{}", record.table_name),
            provider,
            owner,
        );
        Ok(())
    }

    /// The [`MaskedTableProvider`] for `manifest` — one `ListingTable` per
    /// fragment, every non-base fragment pinned to the base fragment's
    /// inferred schema, under the manifest's deletion mask. Unregistered: the
    /// caller registers it (`bind_result_table`) or reads through it directly
    /// (a not-yet-published manifest's live-row count).
    pub async fn build_masked_provider(
        &self,
        ctx: &SessionContext,
        record: &ResultTableRecord,
        manifest: &VersionManifest,
    ) -> Result<Arc<dyn TableProvider>> {
        let mask = Arc::new(
            self.load_deletion_mask(&record.table_name, manifest)
                .await?,
        );
        let mut fragments = Vec::with_capacity(manifest.fragments.len());
        let mut pinned: Option<arrow::datatypes::SchemaRef> = None;
        for fragment in &manifest.fragments {
            let url = StorageUrl::parse(&fragment.url)?;
            let provider =
                build_result_table_provider(ctx, &self.registry, &url, pinned.clone()).await?;
            if pinned.is_none() {
                pinned = Some(provider.schema());
            }
            fragments.push(MaskedFragment {
                provider,
                version: fragment.version,
            });
        }
        let schema = pinned.ok_or_else(|| {
            JammiError::Catalog(format!(
                "result table '{}' version {} lists no fragments",
                record.table_name, manifest.version
            ))
        })?;
        Ok(Arc::new(MaskedTableProvider::new(
            record.table_name.clone(),
            fragments,
            mask,
            schema,
        )))
    }

    /// Persist a fully-built [`SidecarIndex`] as a NEW immutable segment of
    /// `building`'s ANN index and register it under the writer's ownership,
    /// returning the allocated [`SegmentId`]. Existing segments are untouched
    /// — the index's row-set grows without any graph rebuild.
    ///
    /// The id is allocated by reading the current maximum and inserting at
    /// `max + 1` (or `0` for the first segment), retrying on the
    /// `(table_name, segment_id)` primary-key collision a concurrent appender
    /// racing to the same next id would cause — the segment bundle's URL
    /// (`{table}__seg{N}.idx`, a sibling of the row's Parquet) embeds the id,
    /// so allocation and URL derivation share this loop rather than a single
    /// non-atomic `INSERT … SELECT MAX+1`. The catalog row is inserted first
    /// (reserving the id, in the same transaction as the writer's lease
    /// check) and the bundle saved second, so a save failure leaves a segment
    /// row whose bundle is absent — [`Self::resolve_search_mode`] then falls
    /// the whole table back to exact, and recovery rebuilds the set — never a
    /// silently missing row.
    ///
    /// The index's own precision **must** equal the row's persisted
    /// `storage_precision`: a segment built at the deployment default after
    /// that default drifted from the table's promise would be caught only at
    /// load time as a hard failure, so it is rejected here instead. The
    /// segment inherits the table's owning tenant from the row.
    pub async fn append_segment(
        &self,
        building: &BuildingTable,
        index: &SidecarIndex,
    ) -> Result<SegmentId> {
        let precision = building.storage_precision();
        if index.storage_precision() != precision {
            return Err(JammiError::Other(format!(
                "append_segment: index built at {:?} but table '{}' is persisted at {:?} — \
                 a segment must match its table's precision",
                index.storage_precision(),
                building.table_name(),
                precision
            )));
        }
        let row_count = index.len();
        let cas = building.cas();

        loop {
            let next = self
                .catalog
                .max_index_segment_id(building.table_name())
                .await?
                .map_or(0, |m| m + 1);
            let seg_url = layout::segment_url(building.parquet_url(), next)?;
            if self
                .catalog
                .insert_index_segment(&cas, next, seg_url.as_str(), row_count)
                .await?
            {
                self.save_sidecar(&seg_url, index).await?;
                return Ok(SegmentId(next));
            }
            // Lost the race for `next` (another appender inserted it first);
            // re-read the max and retry at the new next id.
        }
    }

    /// Allocate the next version of the READY table `table` under this
    /// store's writer id and lease: the catalog's monotonic allocation
    /// ([`Catalog::allocate_result_table_version`]) plus the lease-held handle
    /// every refresh/compaction write routes through. The handle carries the
    /// table's persisted precision (every segment it appends must match) and
    /// the row's own tenant.
    pub async fn allocate_version(&self, table: &ResultTableRecord) -> Result<BuildingVersion> {
        let parquet_url = StorageUrl::parse(&table.parquet_path)?;
        let allocated = self
            .catalog
            .allocate_result_table_version(&table.table_name, &self.writer_id, self.lease.lease())
            .await?;
        let manifest_url = StorageUrl::parse(&allocated.manifest_path)?;
        let tenant = parse_owner(table)?;
        Ok(BuildingVersion::adopt(
            self.clone(),
            table.table_name.clone(),
            parquet_url,
            allocated.version,
            allocated.parent,
            manifest_url,
            tenant,
            self.writer_id.to_string(),
            table.storage_precision.unwrap_or_default(),
        ))
    }

    /// Persist a fully-built [`SidecarIndex`] as a NEW immutable segment
    /// stamped with `version`'s number, registered under the version's lease
    /// — the same read-max / insert / collision-retry loop as
    /// [`Self::append_segment`], with the version row (not the table row) as
    /// the lease check ([`Catalog::insert_index_segment_for_version`]); the
    /// bundle is saved second so a save failure leaves a row with an absent
    /// bundle for the version's own reap. Precision must equal the table's.
    pub async fn append_segment_for_version(
        &self,
        version: &BuildingVersion,
        index: &SidecarIndex,
    ) -> Result<SegmentId> {
        let precision = version.storage_precision();
        if index.storage_precision() != precision {
            return Err(JammiError::Other(format!(
                "append_segment_for_version: index built at {:?} but table '{}' is persisted at \
                 {:?} — a segment must match its table's precision",
                index.storage_precision(),
                version.table_name(),
                precision
            )));
        }
        let row_count = index.len();
        let cas = version.cas();
        loop {
            let next = self
                .catalog
                .max_index_segment_id(version.table_name())
                .await?
                .map_or(0, |m| m + 1);
            let seg_url = layout::segment_url(version.parquet_url(), next)?;
            if self
                .catalog
                .insert_index_segment_for_version(&cas, next, seg_url.as_str(), row_count)
                .await?
            {
                self.save_sidecar(&seg_url, index).await?;
                return Ok(SegmentId(next));
            }
        }
    }

    /// Reap every artifact stamped with `version` of the table at
    /// `parquet_url`: `__v{N}.parquet`, `__v{N}.deletes.parquet`,
    /// `__v{N}.version.json`, and every `version = N` segment (bundle siblings
    /// then catalog rows, [`Self::purge_segments_for_version`]). NEVER the
    /// base Parquet, its `.materialization.json`, or a `version IS NULL`
    /// segment. The caller has already performed the CAS that licenses this
    /// (the version row's `failed`, or expiry's row delete). 404 is not an
    /// error; a real delete failure lands in `errored`, never swallowed.
    pub(crate) async fn reap_version_artifacts(
        &self,
        parquet_url: &StorageUrl,
        table_name: &str,
        version: i64,
    ) -> Result<DeletionOutcome> {
        let mut deleted = BTreeSet::new();
        let mut errored = BTreeSet::new();
        for url in [
            layout::version_fragment_url(parquet_url, version)?,
            layout::version_deletes_url(parquet_url, version)?,
            layout::version_manifest_url(parquet_url, version)?,
        ] {
            let handle = self.open_parquet(&url)?;
            let path = handle.data_path()?;
            match handle.delete_if_exists(&path).await {
                Ok(DeleteOutcome::Deleted) => {
                    if let Some(rel) = reconcile::relative_to(&self.root, &url) {
                        deleted.insert(rel);
                    }
                }
                Ok(DeleteOutcome::Absent) => {}
                Err(e) => {
                    warn!(
                        table = table_name,
                        version,
                        object = %url,
                        error = %e,
                        "reap_version_artifacts: delete failed; left for reconcile to retry"
                    );
                    if let Some(rel) = reconcile::relative_to(&self.root, &url) {
                        errored.insert(rel);
                    }
                }
            }
        }
        let segments = self.purge_segments_for_version(table_name, version).await?;
        deleted.extend(segments.deleted);
        errored.extend(segments.errored);
        Ok(DeletionOutcome { deleted, errored })
    }

    /// Delete the bundles and catalog rows of every segment stamped with
    /// `version` — the version-scoped peer of `purge_segments`, which stays
    /// table-scoped and reachable only from the table-level building/failed
    /// arms (a versioned table's base set is never purged by a version).
    pub(crate) async fn purge_segments_for_version(
        &self,
        table_name: &str,
        version: i64,
    ) -> Result<DeletionOutcome> {
        let mut deleted = BTreeSet::new();
        let mut errored = BTreeSet::new();
        for seg in self
            .catalog
            .list_index_segments_for_version(table_name, version)
            .await?
        {
            let url = StorageUrl::parse(&seg.index_path).map_err(|e| {
                JammiError::Other(format!(
                    "purge_segments_for_version: table '{table_name}' segment {} has an \
                     unparseable index_path '{}': {e}",
                    seg.segment_id, seg.index_path
                ))
            })?;
            let handle = self.open_index(&url)?;
            for ext in storage::sidecar_layout::sidecar_extensions(SidecarKind::Ann) {
                let Ok(path) = handle.sibling_path(ext) else {
                    continue;
                };
                match handle.delete_if_exists(&path).await {
                    Ok(DeleteOutcome::Deleted) => {
                        if let Ok(sib) = layout::sidecar_url(&url, ext) {
                            if let Some(rel) = reconcile::relative_to(&self.root, &sib) {
                                deleted.insert(rel);
                            }
                        }
                    }
                    Ok(DeleteOutcome::Absent) => {}
                    Err(e) => {
                        warn!(
                            table = table_name,
                            version,
                            segment = seg.segment_id,
                            extension = ext,
                            error = %e,
                            "purge_segments_for_version: sidecar delete failed; left for reconcile"
                        );
                        if let Ok(sib) = layout::sidecar_url(&url, ext) {
                            if let Some(rel) = reconcile::relative_to(&self.root, &sib) {
                                errored.insert(rel);
                            }
                        }
                    }
                }
            }
        }
        self.catalog
            .delete_index_segments_for_version(table_name, version)
            .await?;
        Ok(DeletionOutcome { deleted, errored })
    }

    /// Persist a fully-built sidecar index bundle at `url` (its base, no
    /// extension). The write half every segment save routes through.
    pub async fn save_sidecar(&self, url: &StorageUrl, index: &SidecarIndex) -> Result<()> {
        let handle = self.open_index(url)?;
        storage::sidecar_layout::save_sidecar(&handle, index).await
    }

    /// Best-effort delete of every segment bundle in a table's ANN index set
    /// **and** the segment catalog rows, under the ownership `cas` names. 404
    /// is not an error — the caller may be paving over already-cleaned state.
    /// Enumerates the set from the catalog, so it must run *before* the
    /// `result_tables` row is deleted (the `ON DELETE CASCADE` on
    /// `index_segments` would otherwise reap the rows first and hide the
    /// bundle URLs).
    ///
    /// Deletes `SidecarKind::Ann` siblings ONLY — a `Lexical` `.tantivy`
    /// directory beside a segment (if one ever exists) is never touched here
    /// (index segments this store appends are ANN-only; see
    /// [`Self::append_segment`]). Returns the root-relative keys actually
    /// deleted, per extension, tried independently of one another so a
    /// single failed delete never hides whether its siblings succeeded — the
    /// exact set `reconcile`'s accounting must credit, never a superset — plus
    /// every key that hit a REAL delete error (see [`DeletionOutcome`]).
    ///
    /// A catalog `index_segments` row whose `index_path` does not even parse
    /// as a [`StorageUrl`] is corruption, not a row to quietly skip past: this
    /// returns an error rather than `continue`-ing over it, so a caller (this
    /// row's own `abort`/promote-rebuild, or `reconcile`) learns loudly that
    /// this table's segment set could not be enumerated, rather than
    /// silently under-deleting (and `reconcile`'s accounting silently
    /// under-crediting) a row whose catalog state is already broken.
    async fn purge_segments(&self, cas: &ResultTableCas) -> Result<DeletionOutcome> {
        let mut deleted = BTreeSet::new();
        let mut errored = BTreeSet::new();
        for seg in self.catalog.list_index_segments(&cas.table).await? {
            let url = StorageUrl::parse(&seg.index_path).map_err(|e| {
                JammiError::Other(format!(
                    "purge_segments: table '{}' segment {} has an unparseable index_path \
                     '{}': {e}",
                    cas.table, seg.segment_id, seg.index_path
                ))
            })?;
            let handle = self.open_index(&url)?;
            for ext in storage::sidecar_layout::sidecar_extensions(SidecarKind::Ann) {
                let Ok(path) = handle.sibling_path(ext) else {
                    continue;
                };
                match handle.delete_if_exists(&path).await {
                    Ok(DeleteOutcome::Deleted) => {
                        if let Ok(sib) = layout::sidecar_url(&url, ext) {
                            if let Some(rel) = reconcile::relative_to(&self.root, &sib) {
                                deleted.insert(rel);
                            }
                        }
                    }
                    // Already gone (a concurrent purge, or a race with this
                    // very reconcile pass) — this call removed nothing, so
                    // it is never inserted into `deleted`: the caller's
                    // accounting (`credit_reaped`) must never credit a
                    // sidecar this call did not actually free.
                    Ok(DeleteOutcome::Absent) => {}
                    Err(e) => {
                        warn!(
                            table = cas.table,
                            segment = seg.segment_id,
                            extension = ext,
                            error = %e,
                            "purge_segments: sidecar delete failed; left for reconcile to retry"
                        );
                        if let Ok(sib) = layout::sidecar_url(&url, ext) {
                            if let Some(rel) = reconcile::relative_to(&self.root, &sib) {
                                errored.insert(rel);
                            }
                        }
                    }
                }
            }
        }
        self.catalog.delete_index_segments(cas).await?;
        Ok(DeletionOutcome { deleted, errored })
    }

    /// Delete a result table's objects — the Parquet, its
    /// `.materialization.json` sidecar, and its whole ANN segment set (bundles
    /// and catalog rows) — under the ownership `cas` names. The byte-deletion
    /// half every deletion arm shares (`abort()`, recovery's claim/fail CAS,
    /// `reconcile(apply=true)`): the caller has ALREADY performed the
    /// one-row CAS that licenses this deletion. 404 is not an error.
    ///
    /// Returns a [`DeletionOutcome`] whose `deleted` is EXACTLY the
    /// root-relative keys [`DeleteOutcome::Deleted`] this call actually
    /// removed — never a key whose `delete_if_exists` errored, AND never a
    /// key that was already [`DeleteOutcome::Absent`] (a 404), however that
    /// came to be: never written, already cleaned by a peer, or vanished in
    /// the window between whatever classified this row and this very delete
    /// call (esc-484) — and whose `errored` is every key that hit a REAL
    /// delete failure (see [`DeletionOutcome`]'s own doc comment for why the
    /// two are never merged). Each of the three deletions (Parquet, manifest
    /// sidecar, segment set) is attempted independently, so one failure never
    /// suppresses an attempt at the others; `reconcile`'s pre-pass accounting
    /// credits only `deleted`, which is why the accounting set can never
    /// exceed the TRUE deletion set.
    pub(crate) async fn delete_objects_after_cas(
        &self,
        parquet_url: &StorageUrl,
        cas: &ResultTableCas,
    ) -> Result<DeletionOutcome> {
        let mut deleted = BTreeSet::new();
        let mut errored = BTreeSet::new();
        let parquet_handle = self.open_parquet(parquet_url)?;
        let path = parquet_handle.data_path()?;
        match parquet_handle.delete_if_exists(&path).await {
            Ok(DeleteOutcome::Deleted) => {
                if let Some(rel) = reconcile::relative_to(&self.root, parquet_url) {
                    deleted.insert(rel);
                }
            }
            // Already gone by the time this delete ran (e.g. vanished in the
            // window between `classify_expired_row`'s read and this CAS-
            // licensed reap) — this call freed nothing, so the key is never
            // inserted into `deleted`: crediting it here would report bytes
            // this pass never actually reclaimed (esc-484).
            Ok(DeleteOutcome::Absent) => {}
            Err(e) => {
                warn!(
                    table = cas.table,
                    error = %e,
                    "delete_objects_after_cas: Parquet delete failed; left for reconcile to retry"
                );
                if let Some(rel) = reconcile::relative_to(&self.root, parquet_url) {
                    errored.insert(rel);
                }
            }
        }
        let sidecar = materialization_sidecar_path(&parquet_handle)?;
        match parquet_handle.delete_if_exists(&sidecar).await {
            Ok(DeleteOutcome::Deleted) => {
                if let Ok(sidecar_url) = layout::sidecar_url(parquet_url, "materialization.json") {
                    if let Some(rel) = reconcile::relative_to(&self.root, &sidecar_url) {
                        deleted.insert(rel);
                    }
                }
            }
            Ok(DeleteOutcome::Absent) => {}
            Err(e) => {
                warn!(
                    table = cas.table,
                    error = %e,
                    "delete_objects_after_cas: manifest sidecar delete failed; left for reconcile to retry"
                );
                if let Ok(sidecar_url) = layout::sidecar_url(parquet_url, "materialization.json") {
                    if let Some(rel) = reconcile::relative_to(&self.root, &sidecar_url) {
                        errored.insert(rel);
                    }
                }
            }
        }
        let segments = self.purge_segments(cas).await?;
        deleted.extend(segments.deleted);
        errored.extend(segments.errored);
        Ok(DeletionOutcome { deleted, errored })
    }

    /// The dry-run twin of [`Self::delete_objects_after_cas`] (via
    /// [`Self::purge_segments`]): the exact root-relative key set that
    /// function deletes for `table_name`'s Parquet at `parquet_url` — the
    /// Parquet itself, its `.materialization.json` sidecar, and every CURRENT
    /// `index_segments` row's ANN-ONLY sidecar siblings (never `Lexical` —
    /// see [`Self::purge_segments`]). Derived through the SAME building
    /// blocks the actual deleter uses (`layout::sidecar_url`,
    /// [`crate::storage::sidecar_layout::sidecar_extensions`] at
    /// `SidecarKind::Ann`, [`crate::store::reconcile::relative_to`]) — never
    /// its own hand-copied enumeration — so `reconcile`'s dry-run preview can
    /// never name a key the real deleter would not also delete (or vice
    /// versa): the "accounting set == deletion set" invariant.
    pub(crate) async fn reap_candidate_keys(
        &self,
        parquet_url: &StorageUrl,
        table_name: &str,
    ) -> Result<BTreeSet<String>> {
        let mut keys = BTreeSet::new();
        if let Some(rel) = reconcile::relative_to(&self.root, parquet_url) {
            keys.insert(rel);
        }
        if let Ok(sidecar_url) = layout::sidecar_url(parquet_url, "materialization.json") {
            if let Some(rel) = reconcile::relative_to(&self.root, &sidecar_url) {
                keys.insert(rel);
            }
        }
        keys.extend(self.segment_ann_sidecar_keys(table_name).await?);
        Ok(keys)
    }

    /// Rebuild a table's whole ANN index from its Parquet as a single fresh
    /// segment, under the recoverer's ownership of the row (`recovered`). Used
    /// by the recovery path: it discards any stale segment set the crashed
    /// attempt left (bundles and catalog rows) and writes one authoritative
    /// segment `0` over every Parquet row, so a recovered table's index exactly
    /// covers its data regardless of how many segments the interrupted write
    /// had produced.
    ///
    /// The precision is the table's own persisted
    /// `ResultTableRecord::storage_precision` — **never** today's deployment
    /// default (threaded through [`Self::append_segment`]'s B4 guard). A rebuild
    /// at a different precision than the row promises would silently corrupt
    /// recall (a graph a caller believes is `Int8` reopened as `F32`).
    ///
    /// Returns the root-relative keys [`Self::purge_segments`] actually
    /// deleted (esc-484 design revision: a promotion is not a reclaim). The
    /// caller ([`Self::reconcile_expired_building_row`]) records this set
    /// verbatim into the pass's `promoted_purged` accumulator — never
    /// diffed against a fresh segment `0` it is about to rewrite, and never
    /// checked against `classify_expired_row`'s classification, which
    /// predicts NOTHING about what this rebuild will purge (its `Promote`
    /// payload is simply the row's currently-referenced key set). Those keys
    /// are excluded from this pass's accounting entirely — never `orphans`,
    /// never `bytes_reclaimed` — because they are the promotion's own
    /// internal bookkeeping, not bytes this pass reclaimed on the row's
    /// behalf; a later pass's ordinary age-gated orphan arm is what would
    /// credit them, and only if `purge_segments` itself failed to delete one.
    ///
    /// On `Err`, the [`RebuildFailure`] payload carries the SAME `purged` set
    /// alongside the error (esc-484 item (b)): `purge_segments` runs BEFORE
    /// the Parquet is read and the fresh segment is built, so a later step
    /// failing (a torn Parquet read, a bad vector, the segment write itself)
    /// still leaves those bytes genuinely deleted from storage — the caller
    /// must credit `purged` even when this returns `Err`, never only on
    /// `Ok`.
    async fn rebuild_index_from_parquet(
        &self,
        recovered: &BuildingTable,
        parquet_handle: &JammiObjectStore,
        table: &ResultTableRecord,
    ) -> std::result::Result<BTreeSet<String>, RebuildFailure> {
        let dimensions = table.dimensions.unwrap_or(0) as usize;
        if dimensions == 0 {
            return Ok(BTreeSet::new());
        }

        // Replace any stale segment set from the interrupted attempt — a
        // deletion, so it runs under the claim the recoverer just took.
        let purged = self
            .purge_segments(&recovered.cas())
            .await
            .map_err(|error| RebuildFailure {
                error,
                purged: BTreeSet::new(),
            })?
            .deleted;

        self.write_fresh_segment_zero(recovered, parquet_handle, table, dimensions)
            .await
            .map_err(|error| RebuildFailure {
                error,
                purged: purged.clone(),
            })?;
        Ok(purged)
    }

    /// The read-Parquet / build-index / write-segment-0 tail of
    /// [`Self::rebuild_index_from_parquet`], split out so its ordinary `?`
    /// short-circuiting stays readable — the caller is solely responsible
    /// for pairing any error here with the `purged` set the destructive
    /// purge already produced.
    async fn write_fresh_segment_zero(
        &self,
        recovered: &BuildingTable,
        parquet_handle: &JammiObjectStore,
        table: &ResultTableRecord,
        dimensions: usize,
    ) -> Result<()> {
        let precision = table.storage_precision.unwrap_or_default();
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
            recovered.append_segment(&index).await?;
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
    ///
    /// `job_attempt` (N11, esc-107) is threaded straight to
    /// [`Self::create_table`] — see there for the `jobs.partial_result`
    /// compare-and-set this performs. `None` for a table created outside the
    /// job machinery (a test fixture, a recompute replay, or a caller that
    /// materialises with no job of record).
    pub async fn materialize_embedding_table(
        &self,
        ctx: &SessionContext,
        spec: EmbeddingTableSpec<'_>,
        rows: &[(String, Vec<f32>)],
        materialization: Materialization<'_>,
        job_attempt: Option<crate::catalog::result_repo::JobAttempt<'_>>,
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
        let building = self
            .create_table(
                source_id,
                ModelTask::TextEmbedding,
                ResultTableKind::Model,
                derived_from,
                model_id,
                Some(dimensions as i32),
                key_column,
                text_columns,
                job_attempt,
            )
            .await?;

        // The building row carries this table's persisted precision; the
        // segment is built at THAT precision, read back off the handle
        // (which captured the stamped value) rather than re-derived from
        // `self.ann` — the uniform, drift-proof source `append_segment`'s own
        // guard checks against.
        let precision = building.storage_precision();

        let schema = crate::store::schema::embedding_table_schema(dimensions);
        let batch = embedding_batch(&schema, source_id, model_id, rows, dimensions)?;

        let mut writer = self.open_writer(building.parquet_url(), schema).await?;
        let mut index = SidecarIndex::new(dimensions, &self.ann, precision)?;
        if !rows.is_empty() {
            writer.write_batch(&batch).await?;
            for (key, vector) in rows {
                index.add(key, vector)?;
            }
        }
        let row_count = writer.close().await?;

        if index.len() > 0 {
            index.build()?;
            building.append_segment(&index).await?;
        }

        // Every `?` above unwinds through `BuildingTable`'s Drop (a best-effort
        // `building -> failed` CAS, no byte deletion); `finish` is the single
        // `building -> ready` funnel.
        building.finish(ctx, row_count, materialization).await
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
            // `import_embeddings` is not one of `jammi_ai::jobs::ComputeSpec`'s
            // kinds — it is a GPU-free promotion of caller-supplied vectors,
            // not a job this crate dispatches — so it carries no job of record.
            None,
        )
        .await
    }
}

/// Build the `(_row_id, _source_id, _model_id, vector, _content_hash)` batch
/// for a materialised embedding table from per-key vectors — a NULL hash in
/// every row, since no producer that lands here embedded a source row.
fn embedding_batch(
    schema: &arrow::datatypes::SchemaRef,
    source_id: &str,
    model_id: &str,
    rows: &[(String, Vec<f32>)],
    dimensions: usize,
) -> Result<arrow::array::RecordBatch> {
    crate::store::schema::embedding_batch_with_null_hash(
        schema, source_id, model_id, rows, dimensions,
    )
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
/// This process's producing-run id — `produced_by` on every manifest it
/// writes (provenance, never a hash input).
pub fn run_id() -> &'static str {
    static RUN_ID: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    RUN_ID.get_or_init(|| uuid::Uuid::new_v4().simple().to_string())
}

/// The tenant a `result_tables` row carries, parsed (`None` for GLOBAL).
fn parse_owner(table: &ResultTableRecord) -> Result<Option<TenantId>> {
    table
        .tenant_id
        .as_deref()
        .map(TenantId::from_str)
        .transpose()
        .map_err(|e| {
            JammiError::Other(format!(
                "result table '{}': invalid tenant_id: {e}",
                table.table_name
            ))
        })
}

/// Whether `e` is one of the four typed outcomes of a building-row CAS that
/// matched zero rows — the signal a recovery arm skips on (it never deletes
/// after a miss) rather than propagates.
fn is_cas_miss(e: &JammiError) -> bool {
    matches!(
        e,
        JammiError::RowGone { .. }
            | JammiError::TenantMismatch { .. }
            | JammiError::LeaseLost { .. }
            | JammiError::CasFailed { .. }
    )
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
    pinned_schema: Option<arrow::datatypes::SchemaRef>,
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
    // A versioned table's fragments are pinned to the base fragment's
    // inferred schema so the union's schema is one shape; a fragment whose
    // file disagrees surfaces as a typed schema error at read.
    let resolved_schema = match pinned_schema {
        Some(s) => s,
        None => {
            listing_options
                .infer_schema(&ctx.state(), &table_path)
                .await?
        }
    };
    let table_config = ListingTableConfig::new(table_path)
        .with_listing_options(listing_options)
        .with_schema(resolved_schema);
    Ok(Arc::new(ListingTable::try_new(table_config)?))
}
