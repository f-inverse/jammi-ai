//! The versioned read path: a deletion mask over immutable segments and
//! fragments, on the ANN merge (`SegmentedIndex::new_masked`) and on the SQL
//! scan (`MaskedTableProvider` / `MaskExec`), plus the placeholder a table
//! whose current manifest cannot be resolved is bound to.
//!
//! The fixture is a hand-built three-version table: the base (version 0,
//! rows `r0..r19`, one segment), a delta (version 1: `r0..r9` re-embedded and
//! `r20..r24` added, `r10..r14` deleted — mask horizons `0`), and a second
//! delta (version 2: `r20` deleted — mask horizon `1`). Every artifact is
//! written through the same store primitives a refresh uses.

use std::sync::Arc;

use arrow::array::{Array, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field};
use datafusion::prelude::SessionContext;
use jammi_db::catalog::result_repo::{ResultTableKind, ResultTableRecord};
use jammi_db::catalog::Catalog;
use jammi_db::config::{AnnIndexConfig, StoragePrecision};
use jammi_db::error::JammiError;
use jammi_db::index::segment::{SegmentId, SegmentedIndex};
use jammi_db::index::sidecar::SidecarIndex;
use jammi_db::index::VectorIndex;
use jammi_db::model_task::ModelTask;
use jammi_db::storage::StorageUrl;
use jammi_db::store::deletes::DeletionMask;
use jammi_db::store::layout;
use jammi_db::store::manifest::{
    ArtifactDigest, ComputeDevice, InputAnchor, Materialization, MaterializationEnv,
    ProducingDescriptor,
};
use jammi_db::store::schema::embedding_table_schema;
use jammi_db::store::version::{
    DeletesRef, FragmentRef, SegmentRef, VersionDelta, VersionManifest,
};
use jammi_db::store::ResultStore;
use tempfile::tempdir;

const DIMS: usize = 4;

fn seg(rows: &[(&str, [f32; 4])], p: StoragePrecision) -> SidecarIndex {
    let mut idx = SidecarIndex::new(DIMS, &AnnIndexConfig::default(), p).unwrap();
    for (id, v) in rows {
        idx.add(id, v).unwrap();
    }
    idx.build().unwrap();
    idx
}

fn ids(hits: &[(String, f32)]) -> Vec<String> {
    hits.iter().map(|(id, _)| id.clone()).collect()
}

/// A key present in two segments is served ONLY by the segment the mask does
/// not cover: mask `(K, 0)` hides seg 0's K (near the query), so the merge
/// returns seg 1's K at seg 1's distance — never seg 0's nearest-wins copy.
#[test]
fn masked_row_never_appears_in_search() {
    let q = [1.0, 0.0, 0.0, 0.0];
    let s0 = seg(
        &[("K", [1.0, 0.0, 0.0, 0.0]), ("a", [0.0, 1.0, 0.0, 0.0])],
        StoragePrecision::F32,
    );
    let s1 = seg(
        &[("K", [0.0, 0.0, 1.0, 0.0]), ("b", [0.0, 0.0, 0.0, 1.0])],
        StoragePrecision::F32,
    );
    let mask = Arc::new(DeletionMask::from_entries([("K".to_string(), 0)]));
    let idx = SegmentedIndex::new_masked(vec![(SegmentId(0), 0, s0), (SegmentId(1), 1, s1)], mask)
        .unwrap();
    let hits = idx.search(&q, 2).unwrap();
    let k = hits
        .iter()
        .find(|(id, _)| id == "K")
        .expect("seg 1's K is live");
    assert!(
        k.1 > 0.5,
        "K must be served from seg 1 (orthogonal to the query), got distance {}",
        k.1
    );
    assert_eq!(hits.iter().filter(|(id, _)| id == "K").count(), 1);
    // Every live row is reachable at k = 3 (a widened fetch past the dead row).
    let all = idx.search(&q, 3).unwrap();
    let mut got = ids(&all);
    got.sort();
    assert_eq!(got, vec!["K", "a", "b"]);
}

/// The rescore reads the OWNING segment's exact vector: K is far in seg 0 and
/// near in seg 1; with seg 0's K masked, seg 1's K rescored against its own
/// exact vector ranks first.
#[test]
fn rescore_reads_the_owning_segment() {
    let q = [1.0, 0.0, 0.0, 0.0];
    let s0 = seg(
        &[("K", [0.0, 0.0, 1.0, 0.0]), ("a", [0.0, 1.0, 0.0, 0.0])],
        StoragePrecision::Int8,
    );
    let s1 = seg(
        &[("K", [1.0, 0.0, 0.0, 0.0]), ("b", [0.0, 0.0, 0.0, 1.0])],
        StoragePrecision::Int8,
    );
    let mask = Arc::new(DeletionMask::from_entries([("K".to_string(), 0)]));
    let idx = SegmentedIndex::new_masked(vec![(SegmentId(0), 0, s0), (SegmentId(1), 1, s1)], mask)
        .unwrap();
    let hits = idx.search_final(&q, 1, 4).unwrap();
    assert_eq!(ids(&hits), vec!["K"]);
    assert!(
        hits[0].1 < 0.01,
        "K's exact vector must come from seg 1 (its owner), got distance {}",
        hits[0].1
    );
}

/// An empty mask leaves the merge byte-identical to the lone sidecar (N=1).
#[test]
fn empty_mask_is_the_unmasked_merge() {
    let rows = [
        ("a", [1.0, 0.0, 0.0, 0.1]),
        ("b", [0.0, 1.0, 0.0, 0.2]),
        ("c", [0.0, 0.0, 1.0, 0.3]),
    ];
    let lone = seg(&rows, StoragePrecision::F32);
    let masked = SegmentedIndex::new_masked(
        vec![(SegmentId(0), 0, seg(&rows, StoragePrecision::F32))],
        Arc::new(DeletionMask::empty()),
    )
    .unwrap();
    for (_, q) in &rows {
        assert_eq!(
            ids(&masked.search(q, 2).unwrap()),
            ids(&lone.search(q, 2).unwrap())
        );
    }
}

// ─── the hand-built three-version table ─────────────────────────────────────

struct Fixture {
    _dir: tempfile::TempDir,
    store: ResultStore,
    ctx: SessionContext,
    record: ResultTableRecord,
    parquet_url: StorageUrl,
}

fn vec_for(key: &str, stamp: u8) -> [f32; 4] {
    // Distinct per (key, version) so a served vector identifies its fragment.
    let n: u32 = key.trim_start_matches('r').parse().unwrap();
    let base = [(n as f32 + 1.0), (stamp as f32) + 0.5, 1.0, 0.25];
    let norm = base.iter().map(|x| x * x).sum::<f32>().sqrt();
    [
        base[0] / norm,
        base[1] / norm,
        base[2] / norm,
        base[3] / norm,
    ]
}

fn batch(rows: &[(&str, [f32; 4])], model: &str) -> RecordBatch {
    let schema = embedding_table_schema(DIMS);
    let n = rows.len();
    let flat: Vec<f32> = rows.iter().flat_map(|(_, v)| v.iter().copied()).collect();
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let vectors =
        FixedSizeListArray::try_new(item, DIMS as i32, Arc::new(Float32Array::from(flat)), None)
            .unwrap();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from_iter_values(rows.iter().map(|(k, _)| *k))),
            Arc::new(StringArray::from_iter_values((0..n).map(|_| "docs"))),
            Arc::new(StringArray::from_iter_values((0..n).map(|_| model))),
            Arc::new(vectors),
            jammi_db::store::content_hash::null_hash_column(n),
        ],
    )
    .unwrap()
}

async fn write_fragment(
    store: &ResultStore,
    url: &StorageUrl,
    b: &RecordBatch,
) -> (usize, ArtifactDigest) {
    let mut w = store.open_writer(url, b.schema()).await.unwrap();
    w.write_batch(b).await.unwrap();
    let rows = w.close().await.unwrap();
    let handle = store.open_parquet(url).unwrap();
    let bytes = handle
        .get_bytes(&handle.data_path().unwrap())
        .await
        .unwrap();
    (rows, ArtifactDigest::of_bytes(&bytes))
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

#[allow(clippy::too_many_arguments)]
fn manifest(
    table: &str,
    version: i64,
    parent: Option<i64>,
    definition_hash: &str,
    fragments: Vec<FragmentRef>,
    segments: Vec<SegmentRef>,
    deletes: Option<DeletesRef>,
    identity: &str,
    live_rows: usize,
    masked_rows: usize,
) -> VersionManifest {
    VersionManifest {
        version_format: jammi_db::store::version::VERSION_FORMAT,
        table: table.into(),
        version,
        parent,
        definition_hash: jammi_db::store::manifest::DefinitionHash(definition_hash.into()),
        delta: VersionDelta {
            descriptor: descriptor(),
            input_anchors: vec![InputAnchor::unpinned_at_instant(
                "docs",
                "1970-01-01T00:00:00Z",
            )],
        },
        fragments,
        segments,
        deletes,
        live_rows,
        masked_rows,
        identity: identity.into(),
        produced_by: "test".into(),
        produced_at: "1970-01-01T00:00:00Z".into(),
        engine_version: "0".into(),
    }
}

/// Build the three-version table (see the module doc) and bind it.
async fn fixture() -> Fixture {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = ResultStore::new(dir.path(), catalog.clone(), AnnIndexConfig::default()).unwrap();
    let ctx = SessionContext::new();

    // Base: r0..r19, one segment, through the funnel.
    let building = store
        .create_table(
            "docs",
            ModelTask::TextEmbedding,
            ResultTableKind::Model,
            None,
            "test-model",
            Some(DIMS as i32),
            Some("_row_id"),
            Some("body"),
            None,
        )
        .await
        .unwrap();
    let base_rows: Vec<(String, [f32; 4])> = (0..20)
        .map(|i| (format!("r{i}"), vec_for(&format!("r{i}"), 0)))
        .collect();
    let base_refs: Vec<(&str, [f32; 4])> =
        base_rows.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    let parquet_url = building.parquet_url().clone();
    let (rows, _) = write_fragment(&store, &parquet_url, &batch(&base_refs, "base")).await;
    building
        .append_segment(&seg(&base_refs, StoragePrecision::F32))
        .await
        .unwrap();
    let env = MaterializationEnv::new(ComputeDevice::Cpu, Vec::new());
    let record = building
        .finish(
            &ctx,
            rows,
            Materialization::new(
                &descriptor(),
                &env,
                vec![InputAnchor::unpinned_at_instant(
                    "docs",
                    "1970-01-01T00:00:00Z",
                )],
            ),
        )
        .await
        .unwrap();
    let table = record.table_name.clone();
    let base_manifest = store
        .read_materialization_manifest(&parquet_url)
        .await
        .unwrap()
        .unwrap();
    let definition_hash = base_manifest.definition_hash.as_str().to_string();
    let base_identity = base_manifest.artifact.as_str().to_string();

    // Version 0: the base publish.
    let v0 = manifest(
        &table,
        0,
        None,
        &definition_hash,
        vec![FragmentRef {
            url: parquet_url.as_str().into(),
            version: 0,
            rows: 20,
            digest: base_manifest.artifact.clone(),
        }],
        vec![SegmentRef {
            segment_id: 0,
            version: 0,
        }],
        None,
        &base_identity,
        20,
        0,
    );
    let v0_url = store
        .write_version_manifest(&parquet_url, &v0)
        .await
        .unwrap();
    catalog
        .publish_base_version(&table, 0, v0_url.as_str(), &base_identity, 20)
        .await
        .unwrap();
    let record = catalog.get_result_table(&table).await.unwrap().unwrap();

    // Version 1: r0..r9 re-embedded + r20..r24 added; r10..r14 deleted.
    let mut v1_handle = store.allocate_version(&record).await.unwrap();
    assert_eq!(v1_handle.version(), 1);
    let v1_rows: Vec<(String, [f32; 4])> = (0..10)
        .chain(20..25)
        .map(|i| (format!("r{i}"), vec_for(&format!("r{i}"), 1)))
        .collect();
    let v1_refs: Vec<(&str, [f32; 4])> = v1_rows.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    let (v1_count, v1_digest) = write_fragment(
        &store,
        &v1_handle.fragment_url().unwrap(),
        &batch(&v1_refs, "v1"),
    )
    .await;
    let seg1 = v1_handle
        .append_segment(&seg(&v1_refs, StoragePrecision::F32))
        .await
        .unwrap();
    let mut mask1 = DeletionMask::empty();
    for i in 0..15 {
        mask1.raise(format!("r{i}"), 0);
    }
    let deletes_url = v1_handle.deletes_url().unwrap();
    let (entries, deletes_digest) = mask1
        .write(&store.open_parquet(&deletes_url).unwrap())
        .await
        .unwrap();
    let v1_fragments = vec![
        v0.fragments[0].clone(),
        FragmentRef {
            url: v1_handle.fragment_url().unwrap().as_str().into(),
            version: 1,
            rows: v1_count,
            digest: v1_digest,
        },
    ];
    let v1_deletes = DeletesRef {
        url: deletes_url.as_str().into(),
        entries,
        digest: deletes_digest,
    };
    let v1_identity = VersionManifest::compute_identity(
        &base_identity,
        &jammi_db::store::manifest::DefinitionHash(definition_hash.clone()),
        &descriptor(),
        &v1_fragments,
        Some(&v1_deletes),
    )
    .unwrap();
    let v1 = manifest(
        &table,
        1,
        Some(0),
        &definition_hash,
        v1_fragments.clone(),
        vec![
            SegmentRef {
                segment_id: 0,
                version: 0,
            },
            SegmentRef {
                segment_id: seg1.0,
                version: 1,
            },
        ],
        Some(v1_deletes.clone()),
        &v1_identity,
        20,
        15,
    );
    store
        .write_version_manifest(&parquet_url, &v1)
        .await
        .unwrap();
    v1_handle.publish(&v1_identity, 20, 15, "[]").await.unwrap();
    let record = catalog.get_result_table(&table).await.unwrap().unwrap();
    assert_eq!(record.current_version, Some(1));

    // Version 2: r20 (a version-1 row) deleted — mask horizon 1, no fragment.
    let mut v2_handle = store.allocate_version(&record).await.unwrap();
    assert_eq!(
        (v2_handle.version(), v2_handle.parent_version()),
        (2, Some(1))
    );
    let mut mask2 = mask1.clone();
    mask2.raise("r20".into(), 1);
    let deletes2_url = v2_handle.deletes_url().unwrap();
    let (entries2, deletes2_digest) = mask2
        .write(&store.open_parquet(&deletes2_url).unwrap())
        .await
        .unwrap();
    let v2_deletes = DeletesRef {
        url: deletes2_url.as_str().into(),
        entries: entries2,
        digest: deletes2_digest,
    };
    let v2_identity = VersionManifest::compute_identity(
        &v1_identity,
        &jammi_db::store::manifest::DefinitionHash(definition_hash.clone()),
        &descriptor(),
        &v1_fragments,
        Some(&v2_deletes),
    )
    .unwrap();
    let v2 = manifest(
        &table,
        2,
        Some(1),
        &definition_hash,
        v1_fragments,
        v1.segments.clone(),
        Some(v2_deletes),
        &v2_identity,
        19,
        16,
    );
    store
        .write_version_manifest(&parquet_url, &v2)
        .await
        .unwrap();
    v2_handle.publish(&v2_identity, 19, 16, "[]").await.unwrap();
    let record = catalog.get_result_table(&table).await.unwrap().unwrap();
    assert_eq!(record.current_version, Some(2));

    store.bind_result_table(&ctx, &record).await.unwrap();
    Fixture {
        _dir: dir,
        store,
        ctx,
        record,
        parquet_url,
    }
}

async fn count(ctx: &SessionContext, sql: &str) -> i64 {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0)
}

async fn rows(ctx: &SessionContext, sql: &str) -> Vec<(String, String)> {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let id =
            arrow::compute::cast(b.column_by_name("_row_id").unwrap(), &DataType::Utf8).unwrap();
        let id = id.as_any().downcast_ref::<StringArray>().unwrap();
        let model =
            arrow::compute::cast(b.column_by_name("_model_id").unwrap(), &DataType::Utf8).unwrap();
        let model = model.as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..b.num_rows() {
            out.push((id.value(i).to_string(), model.value(i).to_string()));
        }
    }
    out
}

/// §6.20 — `COUNT(*)` and `SELECT vector` (which project no key) exclude
/// masked rows; `LIMIT 10` returns 10 LIVE rows; every re-embedded key is
/// served from its newest fragment; the deleted keys never appear.
#[tokio::test]
async fn masked_projection_and_limit() {
    let f = fixture().await;
    let t = format!("\"jammi.{}\"", f.record.table_name);
    assert_eq!(
        count(&f.ctx, &format!("SELECT count(*) FROM {t}")).await,
        19
    );
    let vectors = f
        .ctx
        .sql(&format!("SELECT vector FROM {t}"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(vectors.iter().map(|b| b.num_rows()).sum::<usize>(), 19);
    let limited = rows(&f.ctx, &format!("SELECT * FROM {t} LIMIT 10")).await;
    assert_eq!(
        limited.len(),
        10,
        "LIMIT 10 must yield 10 live rows: {limited:?}"
    );
    let all = rows(&f.ctx, &format!("SELECT * FROM {t} ORDER BY _row_id")).await;
    for (id, model) in &all {
        let n: u32 = id[1..].parse().unwrap();
        assert!(
            !(10..=14).contains(&n) && n != 20,
            "deleted key {id} must not appear"
        );
        if !(10..=20).contains(&n) {
            assert_eq!(
                model, "v1",
                "re-embedded/added key {id} comes from the version-1 fragment"
            );
        } else {
            assert_eq!(
                model, "base",
                "untouched key {id} comes from the base fragment"
            );
        }
    }
    let plan = f
        .ctx
        .sql(&format!("EXPLAIN SELECT count(*) FROM {t}"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let text = arrow::util::pretty::pretty_format_batches(&plan)
        .unwrap()
        .to_string();
    assert!(
        text.contains("MaskExec"),
        "the versioned scan is masked:\n{text}"
    );
    assert!(
        text.contains("UnionExec"),
        "the versioned scan unions fragments:\n{text}"
    );
}

/// The ANN merge over the bound set: a deleted key never surfaces, a
/// re-embedded key is served from the newer segment (its new vector), and the
/// old vector of a re-embedded key never returns that key at the old distance.
#[tokio::test]
async fn masked_ann_merge_over_the_versioned_set() {
    let f = fixture().await;
    let index = f
        .store
        .resolve_search_mode_local(&f.record)
        .await
        .unwrap()
        .expect("the segment set loads");
    // r3 was re-embedded: its new vector finds r3 at distance ~0; its old
    // vector never finds r3 at ~0.
    let hits = index.search_final(&vec_for("r3", 1), 1, 4).unwrap();
    assert_eq!(ids(&hits), vec!["r3"]);
    assert!(hits[0].1 < 1e-4);
    let stale = index.search_final(&vec_for("r3", 0), 3, 4).unwrap();
    assert!(
        stale.iter().all(|(id, d)| id != "r3" || *d > 1e-4),
        "the old vector must never return r3 at the old distance: {stale:?}"
    );
    // r12 was deleted: never returned, even when asked for everything.
    let all = index.search_final(&vec_for("r12", 0), 25, 4).unwrap();
    assert!(
        all.iter().all(|(id, _)| id != "r12" && id != "r20"),
        "{all:?}"
    );
    assert_eq!(all.len(), 19, "every live row is reachable: {all:?}");
}

/// A never-refreshed table binds to the single base provider: no `MaskExec`,
/// no `UnionExec` in its plan.
#[tokio::test]
async fn never_refreshed_table_has_no_mask_in_its_plan() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    let store = ResultStore::new(dir.path(), catalog.clone(), AnnIndexConfig::default()).unwrap();
    let ctx = SessionContext::new();
    let building = store
        .create_table(
            "docs",
            ModelTask::TextEmbedding,
            ResultTableKind::Model,
            None,
            "test-model",
            Some(DIMS as i32),
            Some("_row_id"),
            Some("body"),
            None,
        )
        .await
        .unwrap();
    let rows_: Vec<(String, [f32; 4])> = (0..3)
        .map(|i| (format!("r{i}"), vec_for(&format!("r{i}"), 0)))
        .collect();
    let refs: Vec<(&str, [f32; 4])> = rows_.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    let (n, _) = write_fragment(&store, building.parquet_url(), &batch(&refs, "base")).await;
    let env = MaterializationEnv::new(ComputeDevice::Cpu, Vec::new());
    let record = building
        .finish(&ctx, n, Materialization::new(&descriptor(), &env, vec![]))
        .await
        .unwrap();
    assert_eq!(record.current_version, None);
    let plan = ctx
        .sql(&format!(
            "EXPLAIN SELECT count(*) FROM \"jammi.{}\"",
            record.table_name
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let text = arrow::util::pretty::pretty_format_batches(&plan)
        .unwrap()
        .to_string();
    assert!(
        !text.contains("MaskExec") && !text.contains("UnionExec"),
        "{text}"
    );
}

/// D14(i): a current version whose manifest is definitively absent binds to
/// the placeholder — planning succeeds, every scan and the ANN path are the
/// typed `VersionUnavailable { table, version }`.
#[tokio::test]
async fn unresolvable_current_version_is_typed_unavailable() {
    let f = fixture().await;
    let v2_url = layout::version_manifest_url(&f.parquet_url, 2).unwrap();
    let handle = f.store.open_parquet(&v2_url).unwrap();
    handle
        .delete_if_exists(&handle.data_path().unwrap())
        .await
        .unwrap();
    f.store.bind_result_table(&f.ctx, &f.record).await.unwrap();
    let err = f
        .ctx
        .sql(&format!(
            "SELECT count(*) FROM \"jammi.{}\"",
            f.record.table_name
        ))
        .await
        .unwrap()
        .collect()
        .await
        .expect_err("a scan through the placeholder must fail");
    match JammiError::from(err) {
        JammiError::VersionUnavailable { table, version } => {
            assert_eq!(table, f.record.table_name);
            assert_eq!(version, 2);
        }
        other => panic!("expected VersionUnavailable, got {other:?}"),
    }
    match f.store.resolve_search_mode(&f.record).await {
        Err(JammiError::VersionUnavailable { version: 2, .. }) => {}
        Err(other) => panic!("expected VersionUnavailable, got {other:?}"),
        Ok(_) => panic!("the ANN path must refuse too"),
    }
}

/// §6.19 — an object vanishing mid-scan surfaces as the typed
/// `Storage(Io { NotFound })` naming the fragment, never a partial count.
#[cfg(feature = "test-hooks")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mid_scan_object_vanish_is_a_typed_not_found() {
    let f = fixture().await;
    let table = f.record.table_name.clone();
    let race =
        jammi_db::store::masked_provider::masked_scan_test_hooks::arm_masked_scan_drain(&table);
    let ctx = f.ctx.clone();
    let sql = format!("SELECT count(*) FROM \"jammi.{table}\"");
    let query = tokio::spawn(async move { ctx.sql(&sql).await.unwrap().collect().await });
    race.wait_parked().await;
    let v1_url = layout::version_fragment_url(&f.parquet_url, 1).unwrap();
    let handle = f.store.open_parquet(&v1_url).unwrap();
    handle
        .delete_if_exists(&handle.data_path().unwrap())
        .await
        .unwrap();
    race.release();
    let err = query
        .await
        .unwrap()
        .expect_err("the vanished fragment must fail the query, never a partial count");
    match JammiError::from(err) {
        JammiError::Storage(jammi_db::storage::StorageError::Io {
            path,
            source: object_store::Error::NotFound { .. },
        }) => assert!(
            path.contains("__v1.parquet"),
            "the typed not-found names the fragment: {path}"
        ),
        other => panic!("expected Storage(Io(NotFound)), got {other:?}"),
    }
}
