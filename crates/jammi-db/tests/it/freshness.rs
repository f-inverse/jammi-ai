//! The sensing layer — read-only staleness, cache-lookup, and reverse-dependency
//! lineage over the materialization contract ([`jammi_db::store::freshness`]).
//!
//! These tests are an adversarial oracle: each asserts a *property* and fails if
//! that property breaks. They prove `staleness` reports `Fresh` for an untouched
//! table and the right reason for each kind of drift (definition change, an
//! input parent recomputed to a new digest, an input vanished, a pre-contract
//! table, an undecidable unpinned input that can never be `Fresh`); that
//! `lookup_cached` is an exact `(definition_hash, input_anchors)` match that
//! misses on a one-bit change and never hits on an unpinned anchor; and that the
//! `derives_from` lineage is correct, walks transitively stack-safely, and
//! surfaces a cycle as a typed `DependencyCycle`.

use std::sync::Arc;

use arrow::array::{FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use datafusion::prelude::SessionContext;
use jammi_db::catalog::backend::BackendKind;
use jammi_db::catalog::backend_postgres::PostgresBackend;
use jammi_db::catalog::backend_sqlite::SqliteBackend;
use jammi_db::catalog::result_repo::{ResultTableKind, ResultTableRecord};
use jammi_db::catalog::status::ResultTableStatus;
use jammi_db::catalog::Catalog;
use jammi_db::config::AnnIndexConfig;
use jammi_db::error::JammiError;
use jammi_db::model_task::ModelTask;
use jammi_db::store::manifest::{
    ArtifactDigest, ComputeDevice, ComputePrecision, DefinitionHash, InputAnchor, Materialization,
    MaterializationEnv, MaterializationManifest, ModelContentDigest, ModelIdentity,
    ProducingDescriptor,
};
use jammi_db::store::schema::embedding_table_schema;
use jammi_db::store::{ResultStore, ResultTableInfo, StaleReason, Staleness};
use tempfile::tempdir;
use test_case::test_case;

const DIMS: usize = 4;

/// A fresh, effectively-unique `mutable_version` anchor version number.
///
/// Every `descriptor()`/`env()` in this module is a fixed literal, so every
/// `materialize()` call in the whole file shares one `DefinitionHash` — the
/// input anchors are the ONLY thing that can distinguish one test's `ready`
/// row from another's in `find_ready_result_tables_by_definition`'s
/// `(definition_hash, input_anchors)` search. On the Postgres lane (one
/// shared database for the whole run, rows never cleaned up), two tests
/// that pinned the SAME literal `mutable_version("docs", N)` anchor would
/// produce indistinguishable rows; `lookup_cached`/`probe_cache` would then
/// nondeterministically resolve to whichever sibling test's row sorts first
/// by `created_at` — including one whose own tempdir (and therefore Parquet
/// bytes) has already been dropped. A fresh version number per call sidesteps
/// the ambiguity entirely: no two tests, and no two runs, ever pin the same
/// anchor.
fn unique_version() -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    jammi_test_utils::unique_suffix().hash(&mut hasher);
    hasher.finish()
}

/// Build a catalog on `backend`, running migrations. Returns `None` for the
/// Postgres arm when `JAMMI_TEST_PG_URL` is unset, so callers skip (never
/// `#[ignore]`) exactly like [`jammi_test_utils::make_test_session`]. Every
/// result table this module creates gets a fresh UUID-suffixed name
/// ([`ResultStore::create_table`]), so the shared Postgres lane never needs a
/// per-test unique source id: lineage/staleness queries here are always keyed
/// on that unique `table_name`, never on a fixed literal.
async fn fresh_catalog(backend: BackendKind, dir: &std::path::Path) -> Option<Arc<Catalog>> {
    let backend_impl = match backend {
        BackendKind::Sqlite => {
            let b = SqliteBackend::open(&dir.join("catalog.db")).await.unwrap();
            jammi_db::catalog::backend::BackendImpl::Sqlite(b)
        }
        BackendKind::Postgres => {
            let url = jammi_test_utils::pg_url_for_tests()?;
            let pg = PostgresBackend::open_with_options(&url, 8, None)
                .await
                .unwrap();
            jammi_db::catalog::backend::BackendImpl::Postgres(pg)
        }
    };
    backend_impl.migrate().await.unwrap();
    Some(Arc::new(Catalog::from_backend(backend_impl)))
}

/// Fetch a backend-parameterized catalog, skipping the test (with a warning)
/// when the Postgres arm has no `JAMMI_TEST_PG_URL`.
macro_rules! fresh_catalog_or_skip {
    ($backend:expr, $dir:expr) => {
        match fresh_catalog($backend, $dir.path()).await {
            Some(c) => c,
            None => {
                eprintln!("skipping {:?}: JAMMI_TEST_PG_URL unset", $backend);
                return;
            }
        }
    };
}

fn store(dir: &std::path::Path, catalog: Arc<Catalog>) -> ResultStore {
    ResultStore::new(dir, catalog, AnnIndexConfig::default()).unwrap()
}

async fn create_building(store: &ResultStore) -> ResultTableInfo {
    store
        .create_table(
            "docs",
            ModelTask::TextEmbedding,
            ResultTableKind::Model,
            None,
            "test-model",
            Some(DIMS as i32),
            Some("_row_id"),
            Some("body"),
        )
        .await
        .unwrap()
}

async fn write_embedding_parquet(store: &ResultStore, info: &ResultTableInfo, n: usize) -> usize {
    let schema = embedding_table_schema(DIMS);
    let row_ids: Vec<String> = (0..n).map(|i| format!("row-{i}")).collect();
    let row_id_arr = StringArray::from_iter_values(row_ids.iter().map(|s| s.as_str()));
    let source_arr = StringArray::from_iter_values((0..n).map(|_| "docs"));
    let model_arr = StringArray::from_iter_values((0..n).map(|_| "test-model"));
    let flat: Vec<f32> = (0..n)
        .flat_map(|i| (0..DIMS).map(move |d| (i * DIMS + d) as f32))
        .collect();
    let item = Arc::new(arrow_schema::Field::new(
        "item",
        arrow_schema::DataType::Float32,
        false,
    ));
    let vectors =
        FixedSizeListArray::try_new(item, DIMS as i32, Arc::new(Float32Array::from(flat)), None)
            .unwrap();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(row_id_arr),
            Arc::new(source_arr),
            Arc::new(model_arr),
            Arc::new(vectors),
        ],
    )
    .unwrap();
    let mut writer = store.open_writer(&info.parquet_url, schema).await.unwrap();
    writer.write_batch(&batch).await.unwrap();
    writer.close().await.unwrap()
}

fn descriptor() -> ProducingDescriptor {
    ProducingDescriptor::Embedding {
        model_id: "test-model".into(),
        task: ModelTask::TextEmbedding,
        source_id: "docs".into(),
        columns: vec!["body".into()],
        key_column: "_row_id".into(),
        dimensions: DIMS,
    }
}

fn env() -> MaterializationEnv {
    MaterializationEnv::new(
        ComputeDevice::Cpu,
        vec![ModelIdentity {
            model_id: "test-model".into(),
            backend: "candle".into(),
            compute_precision: ComputePrecision::F32,
            content_digest: ModelContentDigest::Sha256("it-fixture-digest".into()),
        }],
    )
}

/// Materialise a table through the funnel and return its record + the definition
/// hash the funnel computed.
async fn materialize(
    store: &ResultStore,
    ctx: &SessionContext,
    inputs: Vec<InputAnchor>,
) -> (ResultTableRecord, DefinitionHash) {
    let info = create_building(store).await;
    let rows = write_embedding_parquet(store, &info, 3).await;
    let manifest = store
        .finalize_with_manifest(
            ctx,
            &info.table_name,
            &info.parquet_url,
            rows,
            Materialization::new(&descriptor(), &env(), inputs),
        )
        .await
        .unwrap();
    let record = store
        .catalog()
        .get_result_table(&info.table_name)
        .await
        .unwrap()
        .expect("record after materialize");
    (record, manifest.definition_hash)
}

// === staleness ============================================================

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn fresh_when_definition_and_inputs_are_unchanged(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    // A parent result table, then a child that anchors on the parent's digest.
    let (parent, _) = materialize(&store, &ctx, vec![]).await;
    let parent_anchor = store.result_digest_anchor(&parent).await.unwrap();
    let (child, def) = materialize(&store, &ctx, vec![parent_anchor]).await;

    assert_eq!(
        store.staleness(&child, &def).await.unwrap(),
        Staleness::Fresh,
        "unchanged definition + unchanged parent digest must be Fresh"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn stale_when_the_definition_changes(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    // No inputs, so the ONLY determinant that can move is the definition itself
    // — isolating the DefinitionChanged path. (An input with no current-anchor
    // resolution surface, e.g. a MutableVersion, would correctly cloud this to
    // Undecidable; that interaction is exercised separately.)
    let (record, _def) = materialize(&store, &ctx, vec![]).await;

    // The producing definition changed (a different current hash than recorded).
    let changed = DefinitionHash("a-different-definition-hash".into());
    match store.staleness(&record, &changed).await.unwrap() {
        Staleness::Stale { reasons } => {
            assert!(
                reasons.iter().any(|r| matches!(
                    r,
                    StaleReason::DefinitionChanged { current, .. }
                        if current == "a-different-definition-hash"
                )),
                "expected a DefinitionChanged reason, got {reasons:?}"
            );
        }
        other => panic!("expected Stale(DefinitionChanged), got {other:?}"),
    }
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn stale_when_a_parent_is_recomputed_to_a_new_digest(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    // Parent v1; child anchors on parent's CURRENT digest.
    let (parent, _) = materialize(&store, &ctx, vec![]).await;
    let parent_anchor = store.result_digest_anchor(&parent).await.unwrap();
    let recorded_digest = parent_anchor.anchor.0.clone();
    let (child, def) = materialize(&store, &ctx, vec![parent_anchor]).await;
    assert_eq!(
        store.staleness(&child, &def).await.unwrap(),
        Staleness::Fresh
    );

    // The parent is recomputed: its manifest now attests a NEW artifact digest.
    // (`current_anchor` reads the parent's manifest digest, so re-attesting the
    // parent's sidecar to a new digest models exactly the recompute → new-output
    // chain the staleness comparison must detect in the child.)
    reattest_parent_with_new_digest(&store, &parent, ArtifactDigest::of_bytes(b"parent-v2")).await;

    match store.staleness(&child, &def).await.unwrap() {
        Staleness::Stale { reasons } => {
            assert!(
                reasons.iter().any(|r| matches!(
                    r,
                    StaleReason::InputAdvanced { source, recorded, current }
                        if source == &parent.table_name
                            && recorded == &recorded_digest
                            && current != &recorded_digest
                )),
                "expected InputAdvanced for the recomputed parent, got {reasons:?}"
            );
        }
        other => panic!("expected Stale(InputAdvanced), got {other:?}"),
    }
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn stale_input_vanished_reason(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    // The child anchors on a ResultDigest whose source name resolves to NO table
    // (it was never created / already dropped): current_anchor → Vanished.
    let phantom = InputAnchor::result_digest(
        "a-parent-that-does-not-exist",
        &ArtifactDigest::of_bytes(b"ghost"),
    );
    let (child, def) = materialize(&store, &ctx, vec![phantom]).await;

    match store.staleness(&child, &def).await.unwrap() {
        Staleness::Stale { reasons } => {
            assert!(
                reasons.iter().any(|r| matches!(
                    r,
                    StaleReason::InputVanished { source }
                        if source == "a-parent-that-does-not-exist"
                )),
                "expected InputVanished, got {reasons:?}"
            );
        }
        other => panic!("expected Stale(InputVanished), got {other:?}"),
    }
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn missing_manifest_for_a_pre_contract_table(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));

    // A pre-contract row: bytes + a `ready` row, but NO definition_hash summary.
    let info = create_building(&store).await;
    let rows = write_embedding_parquet(&store, &info, 3).await;
    catalog
        .update_result_table_status(&info.table_name, ResultTableStatus::Ready, rows)
        .await
        .unwrap();
    let record = catalog
        .get_result_table(&info.table_name)
        .await
        .unwrap()
        .unwrap();
    assert!(record.definition_hash.is_none());

    let any_def = DefinitionHash("whatever".into());
    assert_eq!(
        store.staleness(&record, &any_def).await.unwrap(),
        Staleness::MissingManifest,
        "a pre-contract row has no recorded definition — a truthful unknown"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn unpinned_input_is_never_fresh(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    // An UnpinnedAtInstant input has no reproducible current anchor, so the
    // verdict is Undecidable — NEVER a confident Fresh, even though the
    // definition is unchanged.
    let (record, def) = materialize(
        &store,
        &ctx,
        vec![InputAnchor::unpinned_at_instant(
            "federated",
            "2026-06-17T00:00:00Z",
        )],
    )
    .await;

    match store.staleness(&record, &def).await.unwrap() {
        Staleness::Undecidable {
            unpinned,
            decided_reasons,
        } => {
            assert_eq!(unpinned, vec!["federated".to_string()]);
            assert!(
                decided_reasons.is_empty(),
                "definition unchanged, so no decided reasons — only the cloud"
            );
        }
        other => {
            panic!("expected Undecidable, got {other:?} (an unpinned input must never be Fresh)")
        }
    }
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn undecidable_still_reports_a_confidently_decided_definition_change(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let (record, _def) = materialize(
        &store,
        &ctx,
        vec![InputAnchor::unpinned_at_instant(
            "federated",
            "2026-06-17T00:00:00Z",
        )],
    )
    .await;

    // The definition certainly changed AND an input is undecidable: the verdict
    // is Undecidable, but the proven definition drift is still reported so the
    // reader sees both the cloud and the certainty.
    let changed = DefinitionHash("changed-definition".into());
    match store.staleness(&record, &changed).await.unwrap() {
        Staleness::Undecidable {
            unpinned,
            decided_reasons,
        } => {
            assert_eq!(unpinned, vec!["federated".to_string()]);
            assert!(
                decided_reasons
                    .iter()
                    .any(|r| matches!(r, StaleReason::DefinitionChanged { .. })),
                "the confidently-decided definition change must still be reported, got {decided_reasons:?}"
            );
        }
        other => panic!("expected Undecidable with a decided reason, got {other:?}"),
    }
}

// === lookup_cached ========================================================

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn lookup_cached_hits_an_exact_match(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let inputs = vec![InputAnchor::mutable_version("docs", unique_version())];
    let (record, def) = materialize(&store, &ctx, inputs.clone()).await;

    let hit = store.lookup_cached(&def, &inputs).await.unwrap();
    assert_eq!(
        hit.as_deref(),
        Some(record.table_name.as_str()),
        "the exact (definition, anchors) pair must hit its own table"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn lookup_cached_misses_on_a_one_bit_anchor_change(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let base_version = unique_version();
    let (_record, def) = materialize(
        &store,
        &ctx,
        vec![InputAnchor::mutable_version("docs", base_version)],
    )
    .await;

    // Same definition, one anchor value changed: a miss.
    let probe = vec![InputAnchor::mutable_version(
        "docs",
        base_version.wrapping_add(1),
    )];
    assert_eq!(
        store.lookup_cached(&def, &probe).await.unwrap(),
        None,
        "a one-bit change in the input anchors must miss the cache"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn lookup_cached_never_hits_an_unpinned_request(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    // Materialise a table whose recorded anchors include an unpinned instant.
    let unpinned = vec![InputAnchor::unpinned_at_instant(
        "federated",
        "2026-06-17T00:00:00Z",
    )];
    let (_record, def) = materialize(&store, &ctx, unpinned.clone()).await;

    // Even probing with the byte-identical anchor set must miss: an instant is
    // not a reproducible id, so a "hit" would be fabricated reuse.
    assert_eq!(
        store.lookup_cached(&def, &unpinned).await.unwrap(),
        None,
        "an unpinned anchor in the request can never produce a cache hit"
    );
}

// === derives_from + transitive walk ======================================

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn derives_from_reports_the_one_hop_dependents(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let (parent, _) = materialize(&store, &ctx, vec![]).await;
    let parent_anchor = store.result_digest_anchor(&parent).await.unwrap();
    let (child, _) = materialize(&store, &ctx, vec![parent_anchor.clone()]).await;
    // A second, unrelated table anchored on a DIFFERENT source must NOT appear.
    let (_other, _) = materialize(
        &store,
        &ctx,
        vec![InputAnchor::mutable_version("elsewhere", 1)],
    )
    .await;

    let edges = store.derives_from(&parent.table_name).await.unwrap();
    assert_eq!(edges.len(), 1, "exactly one table derives from the parent");
    assert_eq!(edges[0].input, parent.table_name);
    assert_eq!(edges[0].derived, child.table_name);
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn derives_from_closure_walks_transitively(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    // A → B → C chain via ResultDigest anchors.
    let (a, _) = materialize(&store, &ctx, vec![]).await;
    let a_anchor = store.result_digest_anchor(&a).await.unwrap();
    let (b, _) = materialize(&store, &ctx, vec![a_anchor]).await;
    let b_anchor = store.result_digest_anchor(&b).await.unwrap();
    let (c, _) = materialize(&store, &ctx, vec![b_anchor]).await;

    let closure = store.derives_from_closure(&a.table_name).await.unwrap();
    let reached: std::collections::HashSet<&str> =
        closure.iter().map(|e| e.derived.as_str()).collect();
    assert!(
        reached.contains(b.table_name.as_str()) && reached.contains(c.table_name.as_str()),
        "the transitive walk from A must reach both B and C, got {reached:?}"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn derives_from_closure_is_stack_safe_on_a_deep_chain(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    // A long linear chain root -> t1 -> t2 -> ... : an explicit-work-stack walk
    // handles it; a naive recursion would risk a stack-depth blow-up. (Kept
    // modest so the test stays fast while still exercising many hops.)
    const DEPTH: usize = 200;
    let (root, _) = materialize(&store, &ctx, vec![]).await;
    let mut prev = root.clone();
    for _ in 0..DEPTH {
        let anchor = store.result_digest_anchor(&prev).await.unwrap();
        let (next, _) = materialize(&store, &ctx, vec![anchor]).await;
        prev = next;
    }

    let closure = store.derives_from_closure(&root.table_name).await.unwrap();
    assert_eq!(
        closure.len(),
        DEPTH,
        "the closure of a length-{DEPTH} chain has {DEPTH} edges"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn derives_from_closure_surfaces_a_cycle_as_a_typed_error(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    // Build a genuine 2-node cycle in the recorded anchors: X anchors on Y and Y
    // anchors on X. A lineage is a DAG by construction, so this is a corruption —
    // it is forged here by editing the catalog summary directly, the only way to
    // produce the back-edge the walk must reject as a typed DependencyCycle
    // rather than loop forever.
    let (x, _) = materialize(
        &store,
        &ctx,
        vec![InputAnchor::mutable_version("docs", unique_version())],
    )
    .await;
    let (y, _) = materialize(
        &store,
        &ctx,
        vec![InputAnchor::mutable_version("docs", unique_version())],
    )
    .await;

    force_input_anchor(
        &catalog,
        &x.table_name,
        &InputAnchor::result_digest(&y.table_name, &ArtifactDigest::of_bytes(b"y")),
    )
    .await;
    force_input_anchor(
        &catalog,
        &y.table_name,
        &InputAnchor::result_digest(&x.table_name, &ArtifactDigest::of_bytes(b"x")),
    )
    .await;

    let err = store
        .derives_from_closure(&x.table_name)
        .await
        .expect_err("a cycle in the lineage must error, not loop");
    assert!(
        matches!(err, JammiError::DependencyCycle { .. }),
        "a lineage cycle must surface as a typed DependencyCycle, got {err:?}"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn derives_from_closure_collects_a_diamond_descendant_once(backend: BackendKind) {
    // A re-converging DAG (a diamond): two parents P1, P2 both feed one shared
    // child C. The stack-safe closure must collect C's subtree exactly once (it is
    // `expanded` after the first descent) and must NOT mistake the second arrival
    // at C for a back-edge cycle. This is the distinction a flat visited-set walk
    // cannot make — the W-61a audit's follow-up.
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    // root feeds P1 and P2; both P1 and P2 feed the shared child C.
    let (root, _) = materialize(&store, &ctx, vec![]).await;
    let root_anchor = store.result_digest_anchor(&root).await.unwrap();
    let (p1, _) = materialize(&store, &ctx, vec![root_anchor.clone()]).await;
    let (p2, _) = materialize(&store, &ctx, vec![root_anchor]).await;
    let p1_anchor = store.result_digest_anchor(&p1).await.unwrap();
    let p2_anchor = store.result_digest_anchor(&p2).await.unwrap();
    // C anchors on BOTH P1 and P2 — the re-converging node.
    let (c, _) = materialize(&store, &ctx, vec![p1_anchor, p2_anchor]).await;

    let closure = store.derives_from_closure(&root.table_name).await.unwrap();

    // Edges into C: one from P1, one from P2 — both recorded (they are real
    // reverse-dependency edges). But C is *expanded* once, so its own subtree
    // (empty here) is walked once and the walk terminates — no cycle error.
    let into_c: Vec<&str> = closure
        .iter()
        .filter(|e| e.derived == c.table_name)
        .map(|e| e.input.as_str())
        .collect();
    assert_eq!(
        into_c.len(),
        2,
        "both P1→C and P2→C are real edges and both are reported, got {into_c:?}"
    );
    let reached: std::collections::HashSet<&str> =
        closure.iter().map(|e| e.derived.as_str()).collect();
    assert!(
        reached.contains(p1.table_name.as_str())
            && reached.contains(p2.table_name.as_str())
            && reached.contains(c.table_name.as_str()),
        "the diamond walk reaches P1, P2, and the shared C, got {reached:?}"
    );
}

// === probe_cache (action-layer hit confirmation) ===========================

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn probe_cache_hits_an_exact_match_with_an_extant_artifact(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let inputs = vec![InputAnchor::mutable_version("docs", unique_version())];
    let (record, def) = materialize(&store, &ctx, inputs.clone()).await;

    let hit = store.probe_cache(&def, &inputs).await.unwrap();
    assert_eq!(
        hit.as_deref(),
        Some(record.table_name.as_str()),
        "an exact (definition, inputs) match with extant bytes is a sound reuse"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn probe_cache_misses_when_the_artifact_was_reaped(backend: BackendKind) {
    // A `ready` catalog row whose Parquet bytes are gone (a torn write that
    // committed `ready` before durability, or a half-deleted table) must NOT be
    // handed back as a reuse — the producer would short-circuit to a table it
    // cannot read. The bare `lookup_cached` sensor still reports the catalog hit;
    // `probe_cache` re-confirms the bytes and falls through to a miss.
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let inputs = vec![InputAnchor::mutable_version("docs", unique_version())];
    let (record, def) = materialize(&store, &ctx, inputs.clone()).await;

    reap_artifact(&store, &record).await;

    assert_eq!(
        store.lookup_cached(&def, &inputs).await.unwrap().as_deref(),
        Some(record.table_name.as_str()),
        "the bare sensor still reports the catalog-level hit"
    );
    assert_eq!(
        store.probe_cache(&def, &inputs).await.unwrap(),
        None,
        "probe_cache re-confirms the artifact and misses when the bytes are gone"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn probe_cache_record_reuses_an_intact_newer_row_when_an_older_same_key_row_is_reaped(
    backend: BackendKind,
) {
    // Two `ready` rows can legitimately share the exact same
    // `(definition_hash, input_anchors)` key — a producer re-materialising an
    // idempotent recompute, or a race. `descriptor()`/`env()` are fixed
    // literals across this module, so two `materialize()` calls over the SAME
    // `inputs` produce two DIFFERENT `ready` tables sharing one definition hash
    // AND one anchor set — exactly the shared-key scenario.
    //
    // When the OLDER of the two has had its Parquet bytes reaped but the NEWER
    // one's are intact, the cache probe must still resolve the sound reuse: a
    // reaped candidate must not shadow another candidate at the exact same key
    // (esc-023 — the false-miss this test pins down).
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let inputs = vec![InputAnchor::mutable_version("docs", unique_version())];
    let (older, def) = materialize(&store, &ctx, inputs.clone()).await;
    let (newer, def_again) = materialize(&store, &ctx, inputs.clone()).await;
    assert_eq!(
        def, def_again,
        "both materialisations replay the same fixed descriptor/env, so they share one definition hash"
    );

    reap_artifact(&store, &older).await;

    let hit = store
        .probe_cache_record(&def, &inputs)
        .await
        .unwrap()
        .expect("the newer same-key row's artifact is intact — a sound reuse exists");
    assert_eq!(
        hit.table_name, newer.table_name,
        "the reaped OLDER candidate must not shadow the intact NEWER match — was a false cache miss"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn probe_cache_record_falls_through_a_reaped_newest_row_to_an_intact_older_match(
    backend: BackendKind,
) {
    // The non-tautology guard: reaps the NEWEST same-key row instead, leaving
    // an OLDER match intact. Preferring the newest candidate (`ORDER BY
    // created_at DESC`) alone is not sufficient: without the fallthrough
    // iteration over every exact-match candidate, a reaped newest row would
    // shadow the still-intact older one and reproduce the same false-miss bug
    // in the opposite direction. This is the case a DESC-ordering-only change
    // (no iteration) would still get wrong.
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let inputs = vec![InputAnchor::mutable_version("docs", unique_version())];
    let (older, def) = materialize(&store, &ctx, inputs.clone()).await;
    let (newer, def_again) = materialize(&store, &ctx, inputs.clone()).await;
    assert_eq!(
        def, def_again,
        "both materialisations replay the same fixed descriptor/env, so they share one definition hash"
    );

    reap_artifact(&store, &newer).await;

    let hit = store
        .probe_cache_record(&def, &inputs)
        .await
        .unwrap()
        .expect("the older same-key row's artifact is intact — a sound reuse exists");
    assert_eq!(
        hit.table_name, older.table_name,
        "the reaped NEWEST candidate must fall through to the intact OLDER match"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn probe_cache_misses_on_a_one_bit_change(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let base_version = unique_version();
    let (_record, def) = materialize(
        &store,
        &ctx,
        vec![InputAnchor::mutable_version("docs", base_version)],
    )
    .await;

    let probe = vec![InputAnchor::mutable_version(
        "docs",
        base_version.wrapping_add(1),
    )];
    assert_eq!(
        store.probe_cache(&def, &probe).await.unwrap(),
        None,
        "a one-bit anchor change is a different cache key — never a hit"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn probe_cache_record_returns_the_reusable_record_on_a_hit(backend: BackendKind) {
    // The producer-facing variant returns the full `ResultTableRecord` (not just
    // the name) so a producer that short-circuits hands the reused record back
    // without a second catalog read. On a hit it is the cached table's record;
    // on a one-bit-changed probe it is `None`.
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let base_version = unique_version();
    let inputs = vec![InputAnchor::mutable_version("docs", base_version)];
    let (record, def) = materialize(&store, &ctx, inputs.clone()).await;

    let hit = store
        .probe_cache_record(&def, &inputs)
        .await
        .unwrap()
        .expect("an exact match with extant bytes is a sound reuse");
    assert_eq!(hit.table_name, record.table_name);
    assert_eq!(hit.status, "ready");

    let miss = store
        .probe_cache_record(
            &def,
            &[InputAnchor::mutable_version(
                "docs",
                base_version.wrapping_add(1),
            )],
        )
        .await
        .unwrap();
    assert!(
        miss.is_none(),
        "a one-bit anchor change is a different key — no reusable record"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn probe_cache_never_hits_an_unpinned_request(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let unpinned = vec![InputAnchor::unpinned_at_instant(
        "federated",
        "2026-06-17T00:00:00Z",
    )];
    let (_record, def) = materialize(&store, &ctx, unpinned.clone()).await;

    assert_eq!(
        store.probe_cache(&def, &unpinned).await.unwrap(),
        None,
        "an unpinned anchor is never a sound reuse — honestly off"
    );
}

// === helpers ===============================================================

/// Delete a `ready` table's Parquet artifact bytes out from under its
/// still-`ready` catalog row — models a torn write that committed `ready`
/// before durability, or a half-deleted table.
async fn reap_artifact(store: &ResultStore, record: &ResultTableRecord) {
    let url = jammi_db::storage::StorageUrl::parse(&record.parquet_path).unwrap();
    let handle = store.open_parquet(&url).unwrap();
    let path = handle.data_path().unwrap();
    handle.delete_if_exists(&path).await.unwrap();
}

/// Re-attest a parent table's `.materialization.json` sidecar to a new artifact
/// digest — models the parent being recomputed to a new output, which
/// `current_anchor` reads as the parent's current digest.
async fn reattest_parent_with_new_digest(
    store: &ResultStore,
    parent: &ResultTableRecord,
    new_digest: ArtifactDigest,
) {
    let url = jammi_db::storage::StorageUrl::parse(&parent.parquet_path).unwrap();
    let original = store
        .read_materialization_manifest(&url)
        .await
        .unwrap()
        .expect("parent has a manifest");
    let updated = MaterializationManifest {
        artifact: new_digest,
        ..original
    };
    let handle = store.open_parquet(&url).unwrap();
    let sidecar = handle.sibling_path("materialization.json").unwrap();
    handle
        .put_bytes(&sidecar, updated.to_json_bytes().unwrap().into())
        .await
        .unwrap();
}

/// Overwrite a table's `input_anchors_json` catalog summary to a single anchor —
/// used only to forge the otherwise-impossible cyclic lineage the cycle guard
/// must reject. Runs a raw UPDATE through the public backend transaction surface
/// (no production code path writes a cyclic anchor set, so there is — correctly —
/// no engine method to do this).
async fn force_input_anchor(catalog: &Catalog, table: &str, anchor: &InputAnchor) {
    use jammi_db::catalog::backend::{SqlValue, TxOptions};

    let json = serde_json::to_string(&vec![anchor.clone()]).unwrap();
    let table = table.to_string();
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), move |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE result_tables SET input_anchors_json = $1 WHERE table_name = $2",
                    &[SqlValue::TextOwned(json), SqlValue::TextOwned(table)],
                )
                .await
                .map(|_| ())
            })
        })
        .await
        .unwrap();
}
