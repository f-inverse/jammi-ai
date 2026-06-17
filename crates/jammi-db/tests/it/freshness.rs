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
use jammi_db::catalog::backend_sqlite::SqliteBackend;
use jammi_db::catalog::result_repo::{ResultTableKind, ResultTableRecord};
use jammi_db::catalog::status::ResultTableStatus;
use jammi_db::catalog::Catalog;
use jammi_db::config::AnnIndexConfig;
use jammi_db::error::JammiError;
use jammi_db::model_task::ModelTask;
use jammi_db::store::manifest::{
    ArtifactDigest, ComputeDevice, DefinitionHash, InputAnchor, Materialization,
    MaterializationEnv, MaterializationManifest, ModelIdentity, ProducingDescriptor,
};
use jammi_db::store::schema::embedding_table_schema;
use jammi_db::store::{ResultStore, ResultTableInfo, StaleReason, Staleness};
use tempfile::tempdir;

const DIMS: usize = 4;

async fn fresh_catalog(dir: &std::path::Path) -> Arc<Catalog> {
    let backend = SqliteBackend::open(&dir.join("catalog.db")).await.unwrap();
    let backend = jammi_db::catalog::backend::BackendImpl::Sqlite(backend);
    backend.migrate().await.unwrap();
    Arc::new(Catalog::from_backend(backend))
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

#[tokio::test]
async fn fresh_when_definition_and_inputs_are_unchanged() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
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

#[tokio::test]
async fn stale_when_the_definition_changes() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
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

#[tokio::test]
async fn stale_when_a_parent_is_recomputed_to_a_new_digest() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
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

#[tokio::test]
async fn stale_input_vanished_reason() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
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

#[tokio::test]
async fn missing_manifest_for_a_pre_contract_table() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
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

#[tokio::test]
async fn unpinned_input_is_never_fresh() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
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

#[tokio::test]
async fn undecidable_still_reports_a_confidently_decided_definition_change() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
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

#[tokio::test]
async fn lookup_cached_hits_an_exact_match() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let inputs = vec![InputAnchor::mutable_version("docs", 7)];
    let (record, def) = materialize(&store, &ctx, inputs.clone()).await;

    let hit = store.lookup_cached(&def, &inputs).await.unwrap();
    assert_eq!(
        hit.as_deref(),
        Some(record.table_name.as_str()),
        "the exact (definition, anchors) pair must hit its own table"
    );
}

#[tokio::test]
async fn lookup_cached_misses_on_a_one_bit_anchor_change() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let (_record, def) =
        materialize(&store, &ctx, vec![InputAnchor::mutable_version("docs", 7)]).await;

    // Same definition, one anchor value changed (7 -> 8): a miss.
    let probe = vec![InputAnchor::mutable_version("docs", 8)];
    assert_eq!(
        store.lookup_cached(&def, &probe).await.unwrap(),
        None,
        "a one-bit change in the input anchors must miss the cache"
    );
}

#[tokio::test]
async fn lookup_cached_never_hits_an_unpinned_request() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
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

#[tokio::test]
async fn derives_from_reports_the_one_hop_dependents() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
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

#[tokio::test]
async fn derives_from_closure_walks_transitively() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
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

#[tokio::test]
async fn derives_from_closure_is_stack_safe_on_a_deep_chain() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
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

#[tokio::test]
async fn derives_from_closure_surfaces_a_cycle_as_a_typed_error() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    // Build a genuine 2-node cycle in the recorded anchors: X anchors on Y and Y
    // anchors on X. A lineage is a DAG by construction, so this is a corruption —
    // it is forged here by editing the catalog summary directly, the only way to
    // produce the back-edge the walk must reject as a typed DependencyCycle
    // rather than loop forever.
    let (x, _) = materialize(&store, &ctx, vec![InputAnchor::mutable_version("docs", 1)]).await;
    let (y, _) = materialize(&store, &ctx, vec![InputAnchor::mutable_version("docs", 2)]).await;

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

#[tokio::test]
async fn derives_from_closure_collects_a_diamond_descendant_once() {
    // A re-converging DAG (a diamond): two parents P1, P2 both feed one shared
    // child C. The stack-safe closure must collect C's subtree exactly once (it is
    // `expanded` after the first descent) and must NOT mistake the second arrival
    // at C for a back-edge cycle. This is the distinction a flat visited-set walk
    // cannot make — the W-61a audit's follow-up.
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
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

#[tokio::test]
async fn probe_cache_hits_an_exact_match_with_an_extant_artifact() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let inputs = vec![InputAnchor::mutable_version("docs", 7)];
    let (record, def) = materialize(&store, &ctx, inputs.clone()).await;

    let hit = store.probe_cache(&def, &inputs).await.unwrap();
    assert_eq!(
        hit.as_deref(),
        Some(record.table_name.as_str()),
        "an exact (definition, inputs) match with extant bytes is a sound reuse"
    );
}

#[tokio::test]
async fn probe_cache_misses_when_the_artifact_was_reaped() {
    // A `ready` catalog row whose Parquet bytes are gone (a torn write that
    // committed `ready` before durability, or a half-deleted table) must NOT be
    // handed back as a reuse — the producer would short-circuit to a table it
    // cannot read. The bare `lookup_cached` sensor still reports the catalog hit;
    // `probe_cache` re-confirms the bytes and falls through to a miss.
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let inputs = vec![InputAnchor::mutable_version("docs", 11)];
    let (record, def) = materialize(&store, &ctx, inputs.clone()).await;

    // Reap the artifact bytes out from under the still-`ready` catalog row.
    let url = jammi_db::storage::StorageUrl::parse(&record.parquet_path).unwrap();
    let handle = store.open_parquet(&url).unwrap();
    let path = handle.data_path().unwrap();
    handle.delete_if_exists(&path).await.unwrap();

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

#[tokio::test]
async fn probe_cache_misses_on_a_one_bit_change() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let (_record, def) =
        materialize(&store, &ctx, vec![InputAnchor::mutable_version("docs", 7)]).await;

    let probe = vec![InputAnchor::mutable_version("docs", 8)];
    assert_eq!(
        store.probe_cache(&def, &probe).await.unwrap(),
        None,
        "a one-bit anchor change is a different cache key — never a hit"
    );
}

#[tokio::test]
async fn probe_cache_never_hits_an_unpinned_request() {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog(dir.path()).await;
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
