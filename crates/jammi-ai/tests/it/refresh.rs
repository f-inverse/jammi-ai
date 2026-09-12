//! `refresh_embeddings` / `expire_versions` over a hermetic Parquet source and
//! the `tiny_bert` fixture: the incremental-embedding acceptance oracles.
//!
//! Each test embeds a small source, EDITS the Parquet file in place (the
//! registered source re-lists it on the next scan), refreshes, and asserts the
//! version chain, the deletion mask, the read paths and the typed refusals
//! the unit contract pins.

use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

use arrow::array::{Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use jammi_ai::inference::runner::test_hooks::{forward_calls_for, reset_forward_calls_for};
use jammi_ai::pipeline::embedding_refresh::{RefreshOptions, RefreshOutcome, RefreshReport};
use jammi_ai::pipeline::neighbor_graph::BuildNeighborGraph;
use jammi_ai::pipeline::recompute::Cascade;
use jammi_ai::session::InferenceSession;
use jammi_ai::Session;
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::error::{JammiError, NonUniqueScan, NotRefreshableReason};
use jammi_db::index::sidecar::SidecarIndex;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::storage::StorageUrl;
use jammi_db::store::deletes::DeletionMask;
use jammi_db::store::manifest::{DefinitionHash, MatchVerdict, ProducingDescriptor};
use jammi_db::store::{layout, CachePolicy, StaleReason, Staleness};
use jammi_db::TenantId;
use jammi_test_utils::vq;
use tempfile::TempDir;

use crate::common;

fn tiny_bert_id() -> String {
    format!("local:{}", common::cookbook_fixture("tiny_bert").display())
}

/// Lexically distinct per row (a tiny random-init encoder collapses sentences
/// that differ only by digits onto one vector, which would make every
/// search-by-key tie meaningless).
fn text_for(i: i64) -> String {
    const WORDS: [&str; 41] = [
        "apple", "river", "engine", "violet", "quartz", "harbor", "meadow", "copper", "signal",
        "falcon", "marble", "thunder", "lantern", "orchid", "granite", "saddle", "compass",
        "walnut", "beacon", "cinder", "ember", "glacier", "hollow", "ivory", "jasper", "kettle",
        "lumen", "mosaic", "nectar", "onyx", "pebble", "quiver", "ripple", "sapphire", "timber",
        "umber", "velvet", "willow", "yarrow", "zephyr", "anvil",
    ];
    let n = WORDS.len() as i64;
    let words: Vec<&str> = (0..(2 + i % 11))
        .map(|k| WORDS[((i * (k + 3) + k * k) % n) as usize])
        .collect();
    format!("{} number {i}", words.join(" "))
}

/// Write a `(id Int64 nullable, text Utf8)` Parquet file.
fn write_parquet(path: &Path, rows: &[(Option<i64>, String)]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("text", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(
                rows.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter().map(|(_, t)| t.clone()).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    let tmp = path.with_extension("parquet.tmp");
    let file = std::fs::File::create(&tmp).unwrap();
    let mut w = parquet::arrow::ArrowWriter::try_new(file, schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    std::fs::rename(tmp, path).unwrap();
}

fn rows(n: i64) -> Vec<(Option<i64>, String)> {
    (0..n).map(|i| (Some(i), text_for(i))).collect()
}

struct Harness {
    _dir: TempDir,
    root: PathBuf,
    session: Arc<InferenceSession>,
    source_path: PathBuf,
    /// A per-test source id (the forward counter and the ANN cache are keyed
    /// by it, and the `it` binary runs tests in parallel).
    source: String,
    table: String,
}

fn unique_source() -> String {
    format!("src_{}", common::unique_suffix())
}

async fn open_session(root: &Path, threads: usize) -> Arc<InferenceSession> {
    let mut config = common::test_config(root);
    config.engine.execution_threads = threads;
    Arc::new(InferenceSession::new(config).await.unwrap())
}

async fn add_source(session: &InferenceSession, source: &str, url: String) {
    session
        .add_source(
            source,
            SourceType::File,
            SourceConnection {
                url: Some(url),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
}

/// Embed `n` rows of a fresh source with `tiny_bert`.
async fn harness(n: i64) -> Harness {
    harness_with(n, 1).await
}

async fn harness_with(n: i64, threads: usize) -> Harness {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("engine");
    std::fs::create_dir_all(&root).unwrap();
    let source_path = dir.path().join("src.parquet");
    write_parquet(&source_path, &rows(n));
    let session = open_session(&root, threads).await;
    let source = unique_source();
    add_source(
        &session,
        &source,
        format!("file://{}", source_path.display()),
    )
    .await;
    let (record, _) = session
        .generate_text_embeddings(
            &source,
            &tiny_bert_id(),
            &["text".to_string()],
            "id",
            CachePolicy::Bypass,
            None,
        )
        .await
        .unwrap();
    assert_eq!(record.row_count, n as usize);
    Harness {
        _dir: dir,
        root,
        session,
        source_path,
        source,
        table: record.table_name,
    }
}

impl Harness {
    async fn record(&self) -> ResultTableRecord {
        self.session
            .catalog()
            .get_result_table(&self.table)
            .await
            .unwrap()
            .expect("table present")
    }

    async fn refresh(&self) -> jammi_db::error::Result<RefreshReport> {
        self.session
            .refresh_embeddings(&self.table, RefreshOptions::default())
            .await
    }

    async fn vector_of(&self, key: &str) -> Option<Vec<f32>> {
        let batches = self
            .session
            .sql(&format!(
                "SELECT vector FROM \"jammi.{}\" WHERE _row_id = '{key}'",
                self.table
            ))
            .await
            .unwrap();
        let mut out = Vec::new();
        for b in &batches {
            jammi_db::store::vectors::extend_with_fixed_size_list_f32(
                b,
                &self.table,
                "vector",
                &mut out,
            )
            .unwrap();
        }
        out.into_iter().next()
    }

    async fn count_key(&self, key: &str) -> i64 {
        let batches = self
            .session
            .sql(&format!(
                "SELECT count(*) AS c FROM \"jammi.{}\" WHERE _row_id = '{key}'",
                self.table
            ))
            .await
            .unwrap();
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0)
    }

    async fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        let record = self.record().await;
        self.session
            .result_store()
            .search_vectors(self.session.context(), &record, &vq(query), k)
            .await
            .unwrap()
    }

    async fn mask(&self, version: i64) -> DeletionMask {
        let record = self.record().await;
        let url =
            layout::version_deletes_url(&StorageUrl::parse(&record.parquet_path).unwrap(), version)
                .unwrap();
        let handle = self.session.result_store().open_parquet(&url).unwrap();
        DeletionMask::read(&handle, &self.table).await.unwrap()
    }

    fn fragment_exists(&self, record: &ResultTableRecord, version: i64) -> bool {
        let url = layout::version_fragment_url(
            &StorageUrl::parse(&record.parquet_path).unwrap(),
            version,
        )
        .unwrap();
        common::url_to_path(url.as_str()).exists()
    }
}

/// §6.1 — edit one row of a 10k-row source: exactly one row is inferred, one
/// segment stamped with the new version carries it, the mask holds exactly
/// `(4242, N-1)`, the new vector finds the key and the old one never does at
/// the old distance, the key is served once, `row_count` is unchanged.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edit_one_row_refresh_infers_exactly_one() {
    let h = harness(10_000).await;
    let old_vector = h.vector_of("4242").await.unwrap();
    let before = h.record().await;

    let mut edited = rows(10_000);
    edited[4242].1 = "a completely different sentence about something else".into();
    write_parquet(&h.source_path, &edited);

    let report = h.refresh().await.unwrap();
    assert_eq!(report.outcome, RefreshOutcome::Published);
    assert_eq!(
        (
            report.inferred_rows,
            report.added,
            report.changed,
            report.deleted,
            report.dropped_rows
        ),
        (1, 0, 1, 0, 0),
        "{report:?}"
    );
    let n = report.version.unwrap();
    assert_eq!(report.parent_version, Some(0));
    assert_eq!(n, 1);

    let after = h.record().await;
    assert_eq!(after.current_version, Some(n));
    assert_eq!(
        after.row_count, before.row_count,
        "row_count is unchanged by an edit"
    );
    let segments = h
        .session
        .catalog()
        .list_index_segments(&h.table)
        .await
        .unwrap();
    let new_seg: Vec<_> = segments.iter().filter(|s| s.version == Some(n)).collect();
    assert_eq!(new_seg.len(), 1, "{segments:?}");
    assert_eq!(new_seg[0].row_count, 1);

    let mask = h.mask(n).await;
    assert_eq!(mask.sorted_entries(), vec![("4242".to_string(), n - 1)]);

    let new_vector = h.vector_of("4242").await.unwrap();
    assert_ne!(new_vector, old_vector);
    let hits = h.search(&new_vector, 1).await;
    assert_eq!(hits[0].0, "4242");
    assert!(
        hits[0].1 < 1e-4,
        "the new vector finds 4242 at ~0: {hits:?}"
    );
    let stale = h.search(&old_vector, 5).await;
    assert!(
        stale.iter().all(|(id, d)| id != "4242" || *d > 1e-4),
        "the old vector never returns 4242 at the old distance: {stale:?}"
    );
    assert_eq!(h.count_key("4242").await, 1);
}

/// §6.1 variant — edit a row to empty text: the model refuses it per row, so
/// the refresh publishes a mask-only version (no fragment, no segment),
/// `row_count` drops by one, and the key is never served.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edit_to_empty_text_publishes_a_mask_only_version() {
    let h = harness(300).await;
    let before = h.record().await;
    let mut edited = rows(300);
    edited[42].1 = String::new();
    write_parquet(&h.source_path, &edited);

    let report = h.refresh().await.unwrap();
    assert_eq!(report.outcome, RefreshOutcome::Published);
    assert_eq!(
        (report.changed, report.inferred_rows, report.dropped_rows),
        (1, 1, 1)
    );
    let n = report.version.unwrap();
    let after = h.record().await;
    assert_eq!(after.row_count, before.row_count - 1);
    assert!(
        !h.fragment_exists(&after, n),
        "no fragment for an all-dropped delta"
    );
    assert!(h
        .session
        .catalog()
        .list_index_segments_for_version(&h.table, n)
        .await
        .unwrap()
        .is_empty());
    assert_eq!(
        h.mask(n).await.sorted_entries(),
        vec![("42".to_string(), n - 1)]
    );
    assert_eq!(h.count_key("42").await, 0);
    let any = h.search(&[0.1; 32], 300).await;
    assert!(any.iter().all(|(id, _)| id != "42"));
}

/// §6.2 — delete one key: `deleted == 1`, no new fragment/segment, the mask
/// holds `(7, N-1)`; the masked merge never returns it while a direct probe of
/// segment 0's bundle still does.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_one_key() {
    let h = harness(200).await;
    let vector7 = h.vector_of("7").await.unwrap();
    let edited: Vec<_> = rows(200)
        .into_iter()
        .filter(|(id, _)| *id != Some(7))
        .collect();
    write_parquet(&h.source_path, &edited);

    let report = h.refresh().await.unwrap();
    assert_eq!(
        (
            report.deleted,
            report.inferred_rows,
            report.added,
            report.changed
        ),
        (1, 0, 0, 0)
    );
    let n = report.version.unwrap();
    let after = h.record().await;
    assert!(!h.fragment_exists(&after, n));
    assert_eq!(
        h.mask(n).await.sorted_entries(),
        vec![("7".to_string(), n - 1)]
    );
    assert!(h.search(&vector7, 3).await.iter().all(|(id, _)| id != "7"));
    assert_eq!(h.count_key("7").await, 0);
    assert_eq!(after.row_count, 199);

    // Segment 0's bundle itself still indexes 7: the mask, not a rewrite.
    let segments = h
        .session
        .catalog()
        .list_index_segments(&h.table)
        .await
        .unwrap();
    let seg0 = segments.iter().find(|s| s.segment_id == 0).unwrap();
    let base = common::url_to_path(&seg0.index_path);
    let idx = SidecarIndex::load(
        &base,
        h.session.result_store().ann_config(),
        after.storage_precision.unwrap_or_default(),
    )
    .unwrap();
    assert!(idx.contains("7"));
}

/// §6.3(a) — `recompute` of a versioned table yields a NEW single-segment
/// table with equal top-k, an artifact digest distinct from every version
/// identity, and the chain untouched.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn recompute_of_a_versioned_table_is_a_new_table() {
    let h = harness(200).await;
    let mut edited = rows(200);
    edited[3].1 = "edited three".into();
    write_parquet(&h.source_path, &edited);
    let report = h.refresh().await.unwrap();
    assert_eq!(report.outcome, RefreshOutcome::Published);
    let versions_before = h
        .session
        .catalog()
        .list_result_table_versions(&h.table)
        .await
        .unwrap();

    let svc = Session::new(Arc::clone(&h.session));
    let out = svc.recompute(&h.table, Cascade::ReportOnly).await.unwrap();
    let new_table = out.recomputed[0].recomputed.clone();
    assert_ne!(new_table, h.table);
    let new_record = h
        .session
        .catalog()
        .get_result_table(&new_table)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(new_record.current_version, None);
    assert_eq!(
        h.session
            .catalog()
            .list_index_segments(&new_table)
            .await
            .unwrap()
            .len(),
        1
    );
    let store = h.session.result_store();
    let new_digest = store
        .read_materialization_manifest(&StorageUrl::parse(&new_record.parquet_path).unwrap())
        .await
        .unwrap()
        .unwrap()
        .artifact
        .0;
    for v in &versions_before {
        assert_ne!(v.identity.as_deref(), Some(new_digest.as_str()));
    }
    let versions_after = h
        .session
        .catalog()
        .list_result_table_versions(&h.table)
        .await
        .unwrap();
    assert_eq!(versions_before, versions_after, "the chain is untouched");
    // Value-equivalent, not byte-equal: the replay embeds the edited source
    // in fresh batches, and candle pads per batch, so a vector can move by
    // float noise; and the fixture encoder ties every same-length sentence,
    // so a top-k id ORDER among ties is arbitrary. The oracle compares the
    // whole ranking on a fixed 20-query set: the same id set, and every id's
    // distance equal within noise.
    for q in 0..20i64 {
        let query = h.vector_of(&q.to_string()).await.unwrap();
        let old: std::collections::BTreeMap<String, f32> =
            h.search(&query, 200).await.into_iter().collect();
        let new: std::collections::BTreeMap<String, f32> = store
            .search_vectors(h.session.context(), &new_record, &vq(&query), 200)
            .await
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(old.len(), 200);
        assert_eq!(
            old.keys().collect::<Vec<_>>(),
            new.keys().collect::<Vec<_>>(),
            "query {q}"
        );
        for (id, d_old) in &old {
            let d_new = new[id];
            assert!(
                (d_old - d_new).abs() < 1e-2,
                "query {q} id {id}: {d_old} vs {d_new}"
            );
        }
        assert!(
            old[&q.to_string()] < 1e-4,
            "query {q} finds itself in the chain"
        );
    }
}

/// §6.6(b) — the current version's manifest vanishes: `recover()` fails the
/// VERSION row only; `search` and `SELECT` are the typed `VersionUnavailable`;
/// a peer tenant resolves not-found; `recompute` yields a new table.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn current_manifest_loss_is_typed_unavailable_and_recomputable() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("engine");
    std::fs::create_dir_all(&root).unwrap();
    let source_path = dir.path().join("src.parquet");
    write_parquet(&source_path, &rows(120));
    let session = open_session(&root, 1).await;
    let tenant_a = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap();
    let tenant_b = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9b").unwrap();
    let model = tiny_bert_id();
    let url = format!("file://{}", source_path.display());
    let source = unique_source();
    let (table, _) = session
        .with_tenant_scoped(tenant_a, |_| async {
            add_source(&session, &source, url).await;
            let (r, _) = session
                .generate_text_embeddings(
                    &source,
                    &model,
                    &["text".to_string()],
                    "id",
                    CachePolicy::Bypass,
                    None,
                )
                .await
                .unwrap();
            (r.table_name, ())
        })
        .await;
    let mut edited = rows(120);
    edited[5].1 = "edited five".into();
    write_parquet(&source_path, &edited);
    let report = session
        .with_tenant_scoped(tenant_a, |_| {
            session.refresh_embeddings(&table, RefreshOptions::default())
        })
        .await
        .unwrap();
    let n = report.version.unwrap();

    let record = session
        .with_tenant_scoped(tenant_a, |_| session.catalog().get_result_table(&table))
        .await
        .unwrap()
        .unwrap();
    let parquet_url = StorageUrl::parse(&record.parquet_path).unwrap();
    let manifest_url = layout::version_manifest_url(&parquet_url, n).unwrap();
    std::fs::remove_file(common::url_to_path(manifest_url.as_str())).unwrap();

    let store = session.result_store();
    store.recover().await.unwrap();
    let row = session
        .with_admin_scope(|_| session.catalog().get_result_table_version(&table, n))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(row.status, "failed");
    let record = session
        .with_tenant_scoped(tenant_a, |_| session.catalog().get_result_table(&table))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(record.current_version, Some(n), "current_version unchanged");
    assert_eq!(record.status, "ready");
    assert!(
        common::url_to_path(parquet_url.as_str()).exists(),
        "base bytes intact"
    );

    // Re-bind as a restart would, then both read paths refuse typed.
    session
        .with_admin_scope(|_| store.load_existing_tables(session.context()))
        .await
        .unwrap();
    let count_sql = format!("SELECT count(*) FROM \"jammi.{table}\"");
    let err = session
        .with_tenant_scoped(tenant_a, |_| session.sql(&count_sql))
        .await
        .expect_err("SELECT through the placeholder refuses");
    assert!(
        matches!(err, JammiError::VersionUnavailable { version, .. } if version == n),
        "{err:?}"
    );
    let refused_query = vq(&[0.1; 32]);
    let err = session
        .with_tenant_scoped(tenant_a, |_| {
            store.search_vectors(session.context(), &record, &refused_query, 3)
        })
        .await
        .expect_err("search refuses");
    assert!(
        matches!(err, JammiError::VersionUnavailable { .. }),
        "{err:?}"
    );
    let peer = session
        .with_tenant_scoped(tenant_b, |_| session.sql(&count_sql))
        .await
        .expect_err("a peer tenant cannot resolve the table");
    assert!(
        peer.to_string().contains("not found"),
        "no disclosure: {peer}"
    );
    assert!(
        !matches!(peer, JammiError::VersionUnavailable { .. }),
        "{peer:?}"
    );

    let svc = Session::new(Arc::clone(&session));
    let out = session
        .with_tenant_scoped(tenant_a, |_| svc.recompute(&table, Cascade::ReportOnly))
        .await
        .unwrap();
    assert_ne!(out.recomputed[0].recomputed, table);
}

/// §6.6(a) + §6.12 — a version whose lease expired while parked before its
/// publish: `recover()` fails the version row and reaps its artifacts, leaves
/// the table `ready` with `current_version` unchanged and searchable; the
/// parked refresh's publish then misses typed; the next refresh allocates
/// `k + 1` with `parent_version = current`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn expired_version_lease_is_reaped_and_the_table_stays_ready() {
    use jammi_ai::pipeline::embedding_refresh::refresh_test_hooks::{arm, ParkPoint};
    use jammi_db::catalog::version_repo::VersionCas;

    let h = harness(150).await;
    let mut edited = rows(150);
    edited[9].1 = "edited nine".into();
    write_parquet(&h.source_path, &edited);
    let before = h.record().await;
    let k = before.next_version; // the version the parked refresh will take (base 0 → delta 1)

    let park = arm(&h.table, ParkPoint::BeforePublish);
    let session = Arc::clone(&h.session);
    let table = h.table.clone();
    let parked = tokio::spawn(async move {
        session
            .refresh_embeddings(&table, RefreshOptions::default())
            .await
    });
    park.wait_parked().await;
    assert!(park.is_parked());
    let allocated = k + 1; // base publish took k; the delta is k + 1
    let store = h.session.result_store();
    let cas = VersionCas::writer(&h.table, allocated, store.writer_id(), None);
    h.session
        .catalog()
        .expire_version_lease_for_test(&cas)
        .await
        .unwrap();
    let record_after_base = h.record().await;
    assert!(
        h.fragment_exists(&record_after_base, allocated),
        "the parked delta wrote its fragment"
    );

    store.recover().await.unwrap();
    let row = h
        .session
        .catalog()
        .get_result_table_version(&h.table, allocated)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(row.status, "failed");
    assert!(
        !h.fragment_exists(&record_after_base, allocated),
        "artifacts stamped {allocated} reaped"
    );
    assert!(h
        .session
        .catalog()
        .list_index_segments_for_version(&h.table, allocated)
        .await
        .unwrap()
        .is_empty());
    let table_row = h.record().await;
    assert_eq!(table_row.status, "ready");
    assert_eq!(
        table_row.current_version,
        Some(k),
        "current_version unchanged"
    );
    assert!(common::url_to_path(&table_row.parquet_path).exists());
    assert!(
        !h.search(&[0.1; 32], 3).await.is_empty(),
        "still searchable"
    );

    park.release();
    let err = parked
        .await
        .unwrap()
        .expect_err("the parked publish must miss");
    assert!(
        matches!(
            err,
            JammiError::CasFailed { .. } | JammiError::LeaseLost { .. }
        ),
        "{err:?}"
    );

    let report = h.refresh().await.unwrap();
    assert_eq!(report.version, Some(allocated + 1));
    assert_eq!(report.parent_version, Some(k));
}

/// §6.13 — a refresh parked after its segment landed and before its publish:
/// concurrent `SELECT`/search show exactly the parent version; after the
/// publish, the new one.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_reads_see_the_parent_until_publish() {
    use jammi_ai::pipeline::embedding_refresh::refresh_test_hooks::{arm, ParkPoint};

    let h = harness(100).await;
    let old_vector = h.vector_of("11").await.unwrap();
    let mut edited = rows(100);
    edited[11].1 = "edited eleven".into();
    write_parquet(&h.source_path, &edited);

    let park = arm(&h.table, ParkPoint::BeforePublish);
    let session = Arc::clone(&h.session);
    let table = h.table.clone();
    let parked = tokio::spawn(async move {
        session
            .refresh_embeddings(&table, RefreshOptions::default())
            .await
    });
    park.wait_parked().await;
    assert_eq!(
        h.vector_of("11").await.unwrap(),
        old_vector,
        "SELECT shows the parent"
    );
    // The fixture encoder ties every same-length sentence, so the parent is
    // asserted over the whole ranking: 11 is served at its OLD vector.
    let hits = h.search(&old_vector, 100).await;
    assert!(
        hits.iter().any(|(id, d)| id == "11" && *d < 1e-4),
        "search shows the parent: {hits:?}"
    );
    assert_eq!(h.count_key("11").await, 1);

    park.release();
    let report = parked.await.unwrap().unwrap();
    assert_eq!(report.outcome, RefreshOutcome::Published);
    let new_vector = h.vector_of("11").await.unwrap();
    assert_ne!(new_vector, old_vector, "SELECT shows the new version");
    let hits = h.search(&new_vector, 1).await;
    assert_eq!(hits[0].0, "11");
    let stale = h.search(&old_vector, 100).await;
    assert!(
        stale.iter().all(|(id, d)| id != "11" || *d > 1e-4),
        "{stale:?}"
    );
}

/// §6.7 — two refreshes on one parent allocate distinct numbers; exactly one
/// publishes; the other's publish is `ParentMoved` (the table-row swap's
/// `current_version` no longer equals the parent it allocated against — the
/// SAME lost-race classification an allocation miss gets, never `CasFailed`,
/// whose `status` payload would misname the cause since the row IS `ready`),
/// its row `failed` and its artifacts reaped.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_refreshes_on_one_parent_publish_exactly_once() {
    use jammi_ai::pipeline::embedding_refresh::refresh_test_hooks::{arm, ParkPoint};

    let h = harness(100).await;
    let mut edited = rows(100);
    edited[2].1 = "edited two".into();
    write_parquet(&h.source_path, &edited);

    let park = arm(&h.table, ParkPoint::BeforePublish);
    let s1 = Arc::clone(&h.session);
    let t1 = h.table.clone();
    let first =
        tokio::spawn(async move { s1.refresh_embeddings(&t1, RefreshOptions::default()).await });
    park.wait_parked().await;
    // The second refresh runs to completion beside the parked first.
    let second = h.refresh().await.unwrap();
    assert_eq!(second.outcome, RefreshOutcome::Published);
    park.release();
    let err = first
        .await
        .unwrap()
        .expect_err("the first publisher must miss");
    assert!(matches!(err, JammiError::ParentMoved { .. }), "{err:?}");

    let record = h.record().await;
    assert_eq!(record.current_version, second.version);
    let versions = h
        .session
        .catalog()
        .list_result_table_versions(&h.table)
        .await
        .unwrap();
    let numbers: Vec<i64> = versions.iter().map(|v| v.version).collect();
    assert_eq!(numbers, vec![0, 1, 2], "distinct numbers: {versions:?}");
    let loser = versions
        .iter()
        .find(|v| Some(v.version) != second.version && v.version != 0)
        .unwrap();
    assert_eq!(loser.status, "failed");
    assert!(
        !h.fragment_exists(&record, loser.version),
        "the loser's artifacts are reaped"
    );
}

/// DELTA fix round 2 (audit a98bf51479b523692 F1 / design round a1492adfcf0d0f8e6
/// item 1c) — a concurrent BASE publish must not fail the whole refresh. Two
/// sessions on the SAME root/catalog both race `ensure_base_version` on a
/// never-based table; session 1 is parked right before its own
/// `publish_base_version` CAS, session 2 (unparked — the arm is one-shot, so
/// a second refresh passes straight through) publishes the base to
/// completion, then session 1 is released. Session 1's base-publish CAS now
/// misses on `current_version` no longer `NULL` — the SAME lost-race shape
/// `ParentMoved` already names for the allocation and publish-table-row
/// misses — so `ensure_base_version`'s absorb arm must treat it exactly like
/// that: absorb, adopt the winner's version 0, and let the refresh proceed
/// to its own (empty) delta as `NoChange`, never fail the whole refresh.
/// Before the fix (F1 unified the base-publish miss into `ParentMoved` but
/// left the absorb arm matching only the old `CasFailed` spelling), this
/// returns `Err(ParentMoved { .. })`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_base_publish_does_not_fail_the_refresh() {
    use jammi_ai::pipeline::embedding_refresh::refresh_test_hooks::{arm, ParkPoint};

    let h = harness(20).await;
    assert_eq!(
        h.record().await.current_version,
        None,
        "a fresh table has no base version yet"
    );

    let park = arm(&h.table, ParkPoint::BeforeBasePublish);
    let session1 = Arc::clone(&h.session);
    let table = h.table.clone();
    let parked = tokio::spawn(async move {
        session1
            .refresh_embeddings(&table, RefreshOptions::default())
            .await
    });
    park.wait_parked().await;
    assert!(park.is_parked());

    // Session 2: a second `InferenceSession` on the SAME root/catalog. Its
    // own `refresh_embeddings` reaches the same park point, finds the arm
    // already taken (one-shot), passes through, and publishes the base to
    // completion.
    let session2 = open_session(&h.root, 1).await;
    let report2 = session2
        .refresh_embeddings(&h.table, RefreshOptions::default())
        .await
        .unwrap();
    assert_eq!(report2.outcome, RefreshOutcome::NoChange);
    assert_eq!(report2.version, Some(0));
    assert_eq!(h.record().await.current_version, Some(0));

    park.release();
    let report1 = parked
        .await
        .unwrap()
        .expect("a concurrent base publish must not fail session 1's refresh");
    assert_eq!(report1.outcome, RefreshOutcome::NoChange);
    assert_eq!(
        report1.version,
        Some(0),
        "session 1 adopts the winner's version"
    );

    // Exactly one base row — session 1's absorbed loss never inserted a
    // second version row (its whole `publish_base_version` transaction
    // rolled back at the miss, before any INSERT).
    let versions = h
        .session
        .catalog()
        .list_result_table_versions(&h.table)
        .await
        .unwrap();
    assert_eq!(
        versions.iter().map(|v| v.version).collect::<Vec<_>>(),
        vec![0],
        "exactly one base row: {versions:?}"
    );
    assert_eq!(versions[0].status, "ready");
}

/// O2 (DELTA fix round 1, audit a25424e2aa5e91337 F2) — two SESSIONS on one
/// catalog: session 1 is at version 0 (having just published the base);
/// session 2, a SECOND `InferenceSession` opened on the SAME root/catalog,
/// publishes version 1 adding new keys via its own refresh. Session 1's
/// DataFusion `ctx` is never re-bound past v0 — `bind_result_table` only
/// ever rebinds the PUBLISHING session's own `ctx` (`ensure_base_version` /
/// `after_publish`), never a sibling session's — which is the exact F2
/// stale-binding shape. Session 1 then refreshes with no further source
/// edits: before this fix, step 3 read session 1's own stale `ctx.table(..)`
/// (still v0), so it would classify v1's already-added keys as `Added`
/// again, re-infer them into a THIRD fragment with no mask entry, and
/// publish a bogus version 2 with two live physical rows under each of
/// those `_row_id`s. After the fix, step 3 reads PARENT's own masked
/// provider (loaded fresh from the catalog's `current_version`, not from
/// `ctx`), so session 1 sees v1's rows as already current and the refresh is
/// a genuine `NoChange`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stale_process_binding_does_not_corrupt_a_concurrent_refresh() {
    let h = harness(20).await;
    // Session 1 publishes the base (v = 0) via a no-op refresh.
    let base = h.refresh().await.unwrap();
    assert_eq!(base.outcome, RefreshOutcome::NoChange);
    assert_eq!(h.record().await.current_version, Some(0));

    // Session 2: a second `InferenceSession` on the SAME root/catalog. Its
    // own `refresh_embeddings` call publishes v = 1 after 5 new rows are
    // added to the source.
    let session2 = open_session(&h.root, 1).await;
    let mut rows25 = rows(20);
    rows25.extend((20..25).map(|i| (Some(i), text_for(i))));
    write_parquet(&h.source_path, &rows25);
    let report2 = session2
        .refresh_embeddings(&h.table, RefreshOptions::default())
        .await
        .unwrap();
    assert_eq!(report2.outcome, RefreshOutcome::Published);
    assert_eq!(report2.added, 5);
    assert_eq!(h.record().await.current_version, Some(1));

    // Session 1's ctx was never told about v = 1 — only session 2's own
    // `bind_result_table` call touched session 2's ctx. Session 1 now
    // refreshes with NO further source edits: its delta must already see
    // the 5 keys session 2 added as part of the CURRENT state, never as new
    // `Added` keys to re-infer.
    let report1 = h.refresh().await.unwrap();
    assert_eq!(
        report1.outcome,
        RefreshOutcome::NoChange,
        "session 1's delta must already see v1's rows as current, not `Added`: {report1:?}"
    );

    // No duplicate physical row under any key: the exact live-row count,
    // read through the CURRENT version's OWN masked provider (never
    // session 1's stale `ctx` — a `SELECT` through `ctx` is a DIFFERENT,
    // pre-existing staleness class this fix does not close, see
    // `ResultStore::bind_result_table`'s doc comment for the full
    // read-class/persist-class residual), must be exactly 25, not 30 (25
    // real rows plus 5 duplicates from a re-inferred fragment).
    let record = h.record().await;
    let manifest = h
        .session
        .result_store()
        .read_version_manifest(
            &record.table_name,
            &StorageUrl::parse(&record.parquet_path).unwrap(),
            record.current_version.unwrap(),
        )
        .await
        .unwrap()
        .unwrap();
    let live = h
        .session
        .result_store()
        .count_live_rows(h.session.context(), &record, &manifest)
        .await
        .unwrap();
    assert_eq!(live, 25, "count_live_rows must be exact, no duplicate rows");
}

/// DELTA fix round 2 (audit a98bf51479b523692 F3 / design round
/// a1492adfcf0d0f8e6 "the persisting residual") —
/// `ResultStore::current_version_provider` is the ONE new read seam the five
/// persisting producers (neighbor_graph, recompute, context_set,
/// graph_propagation, context_predictor) now route through instead of a
/// session's registered `jammi.{table}`, precisely so a producer that
/// resolves its artifact's provenance from a FRESH catalog read (e.g.
/// `result_digest_anchor`, which reads `table.current_version`) reads
/// content that agrees with it.
///
/// Same two-session staleness shape as
/// `stale_process_binding_does_not_corrupt_a_concurrent_refresh` (O2):
/// session 1 publishes v0 (20 rows); session 2 (a SECOND `InferenceSession`
/// on the SAME root/catalog) refreshes to v1, adding 5 rows. Session 1's own
/// `ctx` is never rebound past v0 (only the PUBLISHING session's own
/// `bind_result_table` call touches its `ctx`) — a raw SQL scan over session
/// 1 still serves 20 rows. `current_version_provider`, given session 1's
/// `ctx` but a freshly re-read record (`current_version = Some(1)`), must
/// read v1's full 25 rows regardless of session 1's stale registration.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn current_version_provider_reads_the_catalog_version_not_the_stale_session() {
    let h = harness(20).await;
    let base = h.refresh().await.unwrap();
    assert_eq!(base.outcome, RefreshOutcome::NoChange);
    assert_eq!(h.record().await.current_version, Some(0));

    let session2 = open_session(&h.root, 1).await;
    let mut rows25 = rows(20);
    rows25.extend((20..25).map(|i| (Some(i), text_for(i))));
    write_parquet(&h.source_path, &rows25);
    let report2 = session2
        .refresh_embeddings(&h.table, RefreshOptions::default())
        .await
        .unwrap();
    assert_eq!(report2.outcome, RefreshOutcome::Published);
    assert_eq!(h.record().await.current_version, Some(1));

    // Session 1's own registration is still v0 — a raw SQL scan proves it,
    // the exact staleness `bind_result_table`'s doc now names.
    let stale_batches = h
        .session
        .sql(&format!("SELECT _row_id FROM \"jammi.{}\"", h.table))
        .await
        .unwrap();
    let stale_rows: usize = stale_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(stale_rows, 20, "session 1's own registration is still v0");

    // A FRESH catalog read (current_version = Some(1)) pinned (M1) and read
    // through `pinned_provider`, through session 1's OWN `ctx`, must read
    // v1's 25 rows — never session 1's stale 20. (`current_version_provider`
    // itself left the public surface with M1 — this is its replacement.)
    let fresh_record = h.record().await;
    assert_eq!(fresh_record.current_version, Some(1));
    let pin = h
        .session
        .result_store()
        .pin_current_version(fresh_record)
        .await
        .unwrap();
    let provider = h
        .session
        .result_store()
        .pinned_provider(h.session.context(), &pin)
        .await
        .unwrap();
    let batches = h
        .session
        .context()
        .read_table(provider)
        .unwrap()
        .collect()
        .await
        .unwrap();
    let fresh_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        fresh_rows, 25,
        "pinned_provider reads the catalog's current version, not the stale session"
    );
}

/// DELTA round-4 straddle oracle (`CONTRACT-DELTA-fix3.md`'s Oracles
/// section): publish a NEW version BETWEEN `pin_current_version` and the
/// pinned read, and assert the persisted anchor AND the pinned read's
/// content both still name the version pinned at, never a mix where the
/// anchor names the OLD version while a read straddles onto the NEW one.
/// RED at `ebb1c9e6` (round 3): `current_version_identity` (the anchor leg)
/// and `current_version_provider` (the read leg) each independently
/// re-resolve `table.current_version` off the caller's record, so nothing
/// stops the read from serving whatever published in between — this test
/// pins BEFORE the race and would show the read's content moving to v1
/// while the anchor comment claims v0, on that shape.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pin_current_version_survives_a_publish_race_between_pin_and_read() {
    use datafusion::prelude::{col, lit};

    let h = harness(10).await;
    let base = h.refresh().await.unwrap();
    assert_eq!(base.outcome, RefreshOutcome::NoChange);
    assert_eq!(h.record().await.current_version, Some(0));
    let v0_vector = h.vector_of("0").await.expect("row 0 has a v0 vector");

    // Pin v0 — ONE resolution; the anchor and the read both derive from it.
    let pin = h
        .session
        .result_store()
        .pin_current_version(h.record().await)
        .await
        .unwrap();
    assert_eq!(pin.version(), Some(0));
    let pinned_anchor = pin.input_anchor();

    // THE RACE: a version publishes BETWEEN the pin and the read — a second
    // session on the same catalog/root re-embeds row "0" as v1.
    let session2 = open_session(&h.root, 1).await;
    let mut edited = rows(10);
    edited[0].1 = format!("{}-EDITED", edited[0].1);
    write_parquet(&h.source_path, &edited);
    let report2 = session2
        .refresh_embeddings(&h.table, RefreshOptions::default())
        .await
        .unwrap();
    assert_eq!(report2.outcome, RefreshOutcome::Published);
    assert_eq!(h.record().await.current_version, Some(1));

    // The anchor: infallible and unchanged (no second catalog read), and it
    // DIFFERS from a fresh anchor computed against the now-current v1 — the
    // pin did not silently follow the race.
    assert_eq!(
        pin.input_anchor(),
        pinned_anchor,
        "input_anchor is infallible and idempotent"
    );
    let fresh_anchor = h
        .session
        .result_store()
        .result_digest_anchor(&h.record().await)
        .await
        .unwrap();
    assert_ne!(
        fresh_anchor, pinned_anchor,
        "v1 published a new identity; the v0 pin must not have followed it"
    );

    // The read: `pinned_provider` still serves v0's content — row "0"'s
    // PRE-edit vector, never v1's re-embedded one — even though the
    // catalog's `current_version` moved to 1 in between the pin and this
    // read. This is the property itself: the anchor and every row the pin
    // reads derive from the SAME resolution.
    let provider = h
        .session
        .result_store()
        .pinned_provider(h.session.context(), &pin)
        .await
        .unwrap();
    let batches = h
        .session
        .context()
        .read_table(provider)
        .unwrap()
        .filter(col("_row_id").eq(lit("0")))
        .unwrap()
        .select_columns(&["vector"])
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut pinned_read_vectors = Vec::new();
    for batch in &batches {
        jammi_db::store::vectors::extend_with_fixed_size_list_f32(
            batch,
            &h.table,
            "vector",
            &mut pinned_read_vectors,
        )
        .unwrap();
    }
    assert_eq!(
        pinned_read_vectors.len(),
        1,
        "exactly one row for key \"0\" in the pinned v0 read"
    );
    assert_eq!(
        pinned_read_vectors[0], v0_vector,
        "the pinned read must still serve v0's content, not v1's re-embedded row \"0\""
    );
}

/// O2 (DELTA fix round 1, audit a25424e2aa5e91337 F2/V2) — the compaction
/// arm: the MORE DANGEROUS half of the same defect. Session 1 publishes
/// version 0; session 2 (a second `InferenceSession` on the SAME
/// root/catalog) refreshes to version 1, adding new rows. Session 1's `ctx`
/// is never rebound past v0. Session 1 then COMPACTS: before this fix,
/// `compact_embeddings` rewrote the live rows it is about to republish by
/// scanning `ctx.sql("SELECT * FROM \"jammi.{t}\"")` over session 1's stale
/// v0 binding, so the compacted fragment would carry ONLY v0's rows —
/// silently and PERMANENTLY discarding every row session 2 added, while the
/// K7 chain records the result as a legitimate compaction of v1. After the
/// fix, compaction reads through v1's OWN masked provider (loaded fresh from
/// the catalog, never `ctx`), so nothing is lost.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stale_process_binding_does_not_lose_rows_on_compaction() {
    let h = harness(20).await;
    let base = h.refresh().await.unwrap();
    assert_eq!(base.outcome, RefreshOutcome::NoChange);
    assert_eq!(h.record().await.current_version, Some(0));

    let session2 = open_session(&h.root, 1).await;
    let mut rows25 = rows(20);
    rows25.extend((20..25).map(|i| (Some(i), text_for(i))));
    write_parquet(&h.source_path, &rows25);
    let report2 = session2
        .refresh_embeddings(&h.table, RefreshOptions::default())
        .await
        .unwrap();
    assert_eq!(report2.outcome, RefreshOutcome::Published);
    assert_eq!(h.record().await.current_version, Some(1));

    // Session 1, still bound at v0, compacts.
    let report = h.session.compact_embeddings(&h.table).await.unwrap();
    assert_eq!(
        report.live_rows, 25,
        "compaction must not silently drop the rows session 2 added: {report:?}"
    );

    let record = h.record().await;
    let manifest = h
        .session
        .result_store()
        .read_version_manifest(
            &record.table_name,
            &StorageUrl::parse(&record.parquet_path).unwrap(),
            record.current_version.unwrap(),
        )
        .await
        .unwrap()
        .unwrap();
    let live = h
        .session
        .result_store()
        .count_live_rows(h.session.context(), &record, &manifest)
        .await
        .unwrap();
    assert_eq!(live, 25, "no row lost by the compaction");
}

/// §6.8 — the model changed under the table (its content digest moved): the
/// refresh is `DefinitionDrift` at step 0, before any base publish — no version
/// row, `next_version` unchanged.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_drift_is_refused_before_any_allocation() {
    let dir = TempDir::new().unwrap();
    let model_dir = dir.path().join("model");
    copy_dir(&common::cookbook_fixture("tiny_bert"), &model_dir);
    let root = dir.path().join("engine");
    std::fs::create_dir_all(&root).unwrap();
    let source_path = dir.path().join("src.parquet");
    write_parquet(&source_path, &rows(50));
    let session = open_session(&root, 1).await;
    let source = unique_source();
    add_source(
        &session,
        &source,
        format!("file://{}", source_path.display()),
    )
    .await;
    let model = format!("local:{}", model_dir.display());
    let (record, _) = session
        .generate_text_embeddings(
            &source,
            &model,
            &["text".to_string()],
            "id",
            CachePolicy::Bypass,
            None,
        )
        .await
        .unwrap();
    drop(session);

    // A byte change in the model directory (JSON-neutral) moves the content
    // digest the definition folds; a fresh session loads it anew.
    let config = model_dir.join("config.json");
    let mut text = std::fs::read_to_string(&config).unwrap();
    text.push('\n');
    std::fs::write(&config, text).unwrap();
    let session = open_session(&root, 1).await;
    let err = session
        .refresh_embeddings(&record.table_name, RefreshOptions::default())
        .await
        .expect_err("drift must refuse");
    assert!(matches!(err, JammiError::DefinitionDrift { .. }), "{err:?}");
    let after = session
        .catalog()
        .get_result_table(&record.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(after.next_version, 0);
    assert_eq!(after.current_version, None);
    assert!(session
        .catalog()
        .list_result_table_versions(&record.table_name)
        .await
        .unwrap()
        .is_empty());
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let dest = to.join(entry.file_name());
        if entry.path().is_dir() {
            copy_dir(&entry.path(), &dest);
        } else {
            std::fs::copy(entry.path(), dest).unwrap();
        }
    }
}

/// §6.9 — restart parity: after a refresh, a reopened session serves the
/// identical search results and `SELECT` output; the version rows are intact.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restart_serves_the_refreshed_version() {
    let h = harness(120).await;
    let mut edited = rows(120);
    edited[8].1 = "edited eight".into();
    edited.retain(|(id, _)| *id != Some(20));
    write_parquet(&h.source_path, &edited);
    let report = h.refresh().await.unwrap();
    let query = h.vector_of("8").await.unwrap();
    let hits_before = h.search(&query, 5).await;
    let select_before = h
        .session
        .sql(&format!(
            "SELECT _row_id, _model_id FROM \"jammi.{}\" ORDER BY _row_id",
            h.table
        ))
        .await
        .unwrap();
    let versions_before = h
        .session
        .catalog()
        .list_result_table_versions(&h.table)
        .await
        .unwrap();
    let Harness {
        root,
        table,
        _dir,
        source_path,
        session,
        source: _,
    } = h;
    drop(session);

    let session = open_session(&root, 1).await;
    let record = session
        .catalog()
        .get_result_table(&table)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(record.current_version, report.version);
    let hits_after = session
        .result_store()
        .search_vectors(session.context(), &record, &vq(&query), 5)
        .await
        .unwrap();
    assert_eq!(hits_before, hits_after);
    let select_after = session
        .sql(&format!(
            "SELECT _row_id, _model_id FROM \"jammi.{table}\" ORDER BY _row_id"
        ))
        .await
        .unwrap();
    assert_eq!(
        arrow::util::pretty::pretty_format_batches(&select_before)
            .unwrap()
            .to_string(),
        arrow::util::pretty::pretty_format_batches(&select_after)
            .unwrap()
            .to_string()
    );
    assert_eq!(
        session
            .catalog()
            .list_result_table_versions(&table)
            .await
            .unwrap(),
        versions_before
    );
    assert_eq!(
        session
            .catalog()
            .get_result_table(&table)
            .await
            .unwrap()
            .unwrap()
            .row_count,
        119
    );
    drop(source_path);
}

/// §6.10 — a 4-file source under `execution_threads = 4`, one edit per file:
/// every partition's edit is inferred (`inferred_rows == 4`).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn multi_partition_refresh_is_complete() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("engine");
    std::fs::create_dir_all(&root).unwrap();
    let src_dir = dir.path().join("multi");
    std::fs::create_dir_all(&src_dir).unwrap();
    let part = |f: i64| -> Vec<(Option<i64>, String)> {
        (0..100)
            .map(|i| (Some(f * 100 + i), text_for(f * 100 + i)))
            .collect()
    };
    for f in 0..4 {
        write_parquet(&src_dir.join(format!("part{f}.parquet")), &part(f));
    }
    let session = open_session(&root, 4).await;
    let source = unique_source();
    add_source(&session, &source, format!("file://{}", src_dir.display())).await;
    let (record, _) = session
        .generate_text_embeddings(
            &source,
            &tiny_bert_id(),
            &["text".to_string()],
            "id",
            CachePolicy::Bypass,
            None,
        )
        .await
        .unwrap();
    assert_eq!(record.row_count, 400);
    for f in 0..4 {
        let mut p = part(f);
        p[7].1 = format!("edited row in file {f}");
        write_parquet(&src_dir.join(format!("part{f}.parquet")), &p);
    }
    // DataFusion 52's list-files cache (on by default, infinite TTL) keeps a
    // DIRECTORY source's first listing — size and mtime included — for the
    // session's lifetime, so a rewritten file is read through its stale
    // footer offsets. An engine-level property of directory sources (a
    // single-file source is `head`ed fresh on every scan); the test drops the
    // stale listing so the oracle measures the refresh, not the cache.
    if let Some(lfc) = session
        .context()
        .runtime_env()
        .cache_manager
        .get_list_files_cache()
    {
        lfc.clear();
    }
    let report = session
        .refresh_embeddings(&record.table_name, RefreshOptions::default())
        .await
        .unwrap();
    assert_eq!(
        (
            report.inferred_rows,
            report.changed,
            report.added,
            report.deleted
        ),
        (4, 4, 0, 0),
        "{report:?}"
    );
}

/// §6.11 — K1 reachability: after a refresh `producing_descriptor` is
/// `EmbeddingDelta`, and `recompute` dispatches through the new arm.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn producing_descriptor_is_the_delta_after_a_refresh() {
    let h = harness(60).await;
    let mut edited = rows(60);
    edited[1].1 = "edited one".into();
    write_parquet(&h.source_path, &edited);
    let report = h.refresh().await.unwrap();
    let record = h.record().await;
    let descriptor = h
        .session
        .result_store()
        .producing_descriptor(&record)
        .await
        .unwrap();
    match descriptor {
        ProducingDescriptor::EmbeddingDelta {
            parent_version,
            deletes,
            ..
        } => {
            assert_eq!(parent_version, report.parent_version.unwrap());
            assert_eq!(deletes, jammi_db::store::DeletePolicy::Tombstone);
        }
        other => panic!("expected EmbeddingDelta, got {other:?}"),
    }
    let svc = Session::new(Arc::clone(&h.session));
    let out = svc.recompute(&h.table, Cascade::ReportOnly).await.unwrap();
    assert_ne!(out.recomputed[0].recomputed, h.table);
}

/// §6.15 — a duplicated source key on an already-versioned table is
/// `NonUniqueKey { Source }` after the complete scan with no new version row;
/// the initial embed still tolerates it. A parent with two physical rows under
/// one `_row_id` (constructed) is `NonUniqueKey { Parent }`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_unique_keys_are_refused_on_both_scans() {
    let h = harness(80).await;
    let mut edited = rows(80);
    edited[4].1 = "edited four".into();
    write_parquet(&h.source_path, &edited);
    h.refresh().await.unwrap();
    let versions_before = h
        .session
        .catalog()
        .list_result_table_versions(&h.table)
        .await
        .unwrap();

    let mut dup = rows(80);
    dup.push((Some(3), "a second row three".into()));
    dup.push((Some(3), "a third row three".into()));
    dup.push((Some(5), "a second row five".into()));
    write_parquet(&h.source_path, &dup);
    let err = h.refresh().await.expect_err("duplicate source keys refuse");
    match err {
        JammiError::NonUniqueKey {
            scan, keys, total, ..
        } => {
            assert_eq!(scan, NonUniqueScan::Source);
            assert_eq!(total, 2);
            assert_eq!(keys, vec![("3".to_string(), 3), ("5".to_string(), 2)]);
        }
        other => panic!("expected NonUniqueKey, got {other:?}"),
    }
    assert_eq!(
        h.session
            .catalog()
            .list_result_table_versions(&h.table)
            .await
            .unwrap(),
        versions_before,
        "no new version row"
    );

    // The initial embed tolerates duplicates (row_count counts both).
    let (dup_record, _) = h
        .session
        .generate_text_embeddings(
            &h.source,
            &tiny_bert_id(),
            &["text".to_string()],
            "id",
            CachePolicy::Bypass,
            None,
        )
        .await
        .unwrap();
    assert_eq!(dup_record.row_count, 83);

    // A parent with two physical rows under one `_row_id`: the base Parquet
    // rewritten with a duplicated row (test-only corruption of the fragment).
    write_parquet(&h.source_path, &rows(80));
    let record = h.record().await;
    let base_path = common::url_to_path(&record.parquet_path);
    let batches = {
        let file = std::fs::File::open(&base_path).unwrap();
        let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        reader.collect::<Result<Vec<_>, _>>().unwrap()
    };
    let schema = batches[0].schema();
    let tmp = base_path.with_extension("tmp");
    let file = std::fs::File::create(&tmp).unwrap();
    let mut w = parquet::arrow::ArrowWriter::try_new(file, schema, None).unwrap();
    for b in &batches {
        w.write(b).unwrap();
    }
    w.write(&batches[0].slice(0, 1)).unwrap();
    w.close().unwrap();
    std::fs::rename(tmp, &base_path).unwrap();
    // Re-bind so the scan reads the rewritten object.
    h.session
        .result_store()
        .bind_result_table(h.session.context(), &record)
        .await
        .unwrap();
    let err = h
        .refresh()
        .await
        .expect_err("a duplicated parent key refuses");
    assert!(
        matches!(
            err,
            JammiError::NonUniqueKey {
                scan: NonUniqueScan::Parent,
                ..
            }
        ),
        "{err:?}"
    );
}

/// §6.16 — dependents stay fresh: a neighbor graph over T is `Fresh` after a
/// no-edit refresh (which still publishes the base version) and
/// `Stale { InputAdvanced }` after an edit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dependents_stay_fresh_across_a_no_change_refresh() {
    let h = harness(60).await;
    let params = BuildNeighborGraph {
        k: 3,
        min_similarity: None,
        mutual: false,
        self_exclude: true,
        exact: true,
        exact_max_rows: 10_000,
        resolve_keys: false,
    };
    let (graph, _) = h
        .session
        .build_neighbor_graph(&h.source, Some(&h.table), &params, CachePolicy::Bypass)
        .await
        .unwrap();
    let svc = Session::new(Arc::clone(&h.session));
    let g_def = DefinitionHash(graph.definition_hash.clone().unwrap());
    assert_eq!(
        svc.staleness(&graph.table_name, g_def.clone())
            .await
            .unwrap(),
        Staleness::Fresh
    );

    let report = h.refresh().await.unwrap();
    assert_eq!(report.outcome, RefreshOutcome::NoChange);
    assert_eq!(
        h.record().await.current_version,
        Some(0),
        "the base version is published"
    );
    assert_eq!(
        svc.staleness(&graph.table_name, g_def.clone())
            .await
            .unwrap(),
        Staleness::Fresh
    );

    let mut edited = rows(60);
    edited[10].1 = "edited ten".into();
    write_parquet(&h.source_path, &edited);
    let report = h.refresh().await.unwrap();
    assert_eq!(report.outcome, RefreshOutcome::Published);
    match svc.staleness(&graph.table_name, g_def).await.unwrap() {
        Staleness::Stale { reasons } => {
            assert!(
                reasons
                    .iter()
                    .any(|r| matches!(r, StaleReason::InputAdvanced { .. })),
                "{reasons:?}"
            );
        }
        other => panic!("expected Stale {{ InputAdvanced }}, got {other:?}"),
    }
}

/// §6.18 (refresh leg) — the source edited to carry 3 null keys: the refresh
/// of a versioned table is `InvalidKey { id, 3 }`, the model is never
/// invoked, no building version row is left, `next_version` is unchanged.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn null_keys_refuse_the_refresh_before_any_model_call() {
    let h = harness(40).await;
    let mut edited = rows(40);
    edited[1].1 = "edited one".into();
    write_parquet(&h.source_path, &edited);
    h.refresh().await.unwrap();
    let before = h.record().await;

    let mut nulls = rows(40);
    nulls[3].0 = None;
    nulls[8].0 = None;
    nulls[9].0 = None;
    write_parquet(&h.source_path, &nulls);
    reset_forward_calls_for(&h.source);
    let err = h.refresh().await.expect_err("null keys refuse");
    assert!(
        matches!(&err, JammiError::InvalidKey { column, null_count } if column == "id" && *null_count == 3),
        "{err:?}"
    );
    assert_eq!(forward_calls_for(&h.source), 0);
    let after = h.record().await;
    assert_eq!(after.next_version, before.next_version);
    assert!(h
        .session
        .catalog()
        .list_live_building_versions()
        .await
        .unwrap()
        .is_empty());
}

/// §6.22 (embedded leg) — tenant B cannot refresh tenant A's table and a
/// scoped tenant cannot refresh a GLOBAL table (`TenantMismatch`, before the
/// model load), yet can `SELECT` from the GLOBAL one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tenant_scope_gates_the_refresh_before_the_model_load() {
    let h = harness(30).await; // GLOBAL table
    let tenant_b = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9b").unwrap();
    let mut edited = rows(30);
    edited[0].1 = "edited zero".into();
    write_parquet(&h.source_path, &edited);
    reset_forward_calls_for(&h.source);
    let err = h
        .session
        .with_tenant_scoped(tenant_b, |_| {
            h.session
                .refresh_embeddings(&h.table, RefreshOptions::default())
        })
        .await
        .expect_err("a scoped tenant cannot refresh a GLOBAL table");
    assert!(matches!(err, JammiError::TenantMismatch { .. }), "{err:?}");
    assert_eq!(forward_calls_for(&h.source), 0);
    let count_sql = format!("SELECT count(*) FROM \"jammi.{}\"", h.table);
    let count = h
        .session
        .with_tenant_scoped(tenant_b, |_| h.session.sql(&count_sql))
        .await
        .unwrap();
    assert_eq!(
        count[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        30
    );
    assert_eq!(
        h.record().await.current_version,
        None,
        "nothing was published"
    );
}

/// `read_vectors` on a refreshed table reads through the masked provider in
/// `_row_id` order — the same rows `SELECT ... ORDER BY _row_id` returns.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_vectors_follows_the_refreshed_version_in_key_order() {
    let h = harness(25).await;
    let mut edited = rows(25);
    edited[12].1 = "edited twelve".into();
    edited.retain(|(id, _)| *id != Some(6));
    write_parquet(&h.source_path, &edited);
    h.refresh().await.unwrap();
    let record = h.record().await;
    let vectors = h.session.read_vectors(&record).await.unwrap();
    let batches = h
        .session
        .sql(&format!(
            "SELECT vector FROM \"jammi.{}\" ORDER BY _row_id",
            h.table
        ))
        .await
        .unwrap();
    let mut expected = Vec::new();
    for b in &batches {
        jammi_db::store::vectors::extend_with_fixed_size_list_f32(
            b,
            &h.table,
            "vector",
            &mut expected,
        )
        .unwrap();
    }
    assert_eq!(vectors.len(), 24);
    assert_eq!(vectors, expected);
}

/// §3.6 — `verify_materialization` on a versioned table: the base check is
/// unchanged, then every fragment digest, the deletes digest and the identity
/// chain of the CURRENT version are recomputed and compared to the manifest
/// and the catalog row; a tampered delta fragment is a `Mismatch` naming the
/// fragment's digest, and the base table's own verdict is unaffected.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn verify_follows_the_refreshed_version() {
    let h = harness(60).await;
    let mut edited = rows(60);
    edited[3].1 = "edited three for the verify oracle".into();
    edited.retain(|(id, _)| *id != Some(8));
    write_parquet(&h.source_path, &edited);
    let report = h.refresh().await.unwrap();
    let n = report.version.unwrap();
    let record = h.record().await;
    let store = h.session.result_store();
    let verdict = store.verify_materialization(&record, None).await.unwrap();
    assert!(
        matches!(
            &verdict,
            MatchVerdict::Match | MatchVerdict::MatchWithUnpinnedInputs { .. }
        ),
        "a freshly refreshed version verifies: {verdict:?}"
    );
    let wrong = DefinitionHash("not-the-definition".into());
    assert!(matches!(
        store
            .verify_materialization(&record, Some(&wrong))
            .await
            .unwrap(),
        MatchVerdict::Mismatch { .. }
    ));

    // Tamper the delta fragment: the verdict names ITS digest, and the base
    // Parquet (untouched) still passes its own check.
    let parquet_url = StorageUrl::parse(&record.parquet_path).unwrap();
    let manifest = store
        .read_version_manifest(&h.table, &parquet_url, n)
        .await
        .unwrap()
        .unwrap();
    let fragment = manifest.fragments.iter().find(|f| f.version == n).unwrap();
    let path = common::url_to_path(&fragment.url);
    let mut bytes = std::fs::read(&path).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xff;
    std::fs::write(&path, &bytes).unwrap();
    match store.verify_materialization(&record, None).await.unwrap() {
        MatchVerdict::Mismatch { expected, found } => {
            assert_eq!(expected, fragment.digest.0);
            assert_ne!(found, fragment.digest.0);
        }
        other => panic!("a tampered fragment must be a Mismatch, got {other:?}"),
    }
}

/// §6.3(b) — `compact_embeddings` rewrites the live rows as ONE fragment +
/// ONE segment stamped with the new version, no deletes, the same ranking,
/// an identity distinct from the parent's; `expire_versions(before = N)`
/// then removes the older rows and every unreferenced fragment / segment /
/// deletes / manifest while the base Parquet, its manifest and
/// `next_version` stay, and search is unchanged.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compact_yields_single_fragment_value_equivalent() {
    let h = harness(150).await;
    let mut edited = rows(150);
    edited[4].1 = "edited four".into();
    edited[9].1 = "edited nine".into();
    edited.retain(|(id, _)| *id != Some(30));
    write_parquet(&h.source_path, &edited);
    let delta = h.refresh().await.unwrap();
    assert_eq!(delta.outcome, RefreshOutcome::Published);
    let parent = h
        .session
        .catalog()
        .get_result_table_version(&h.table, delta.version.unwrap())
        .await
        .unwrap()
        .unwrap();
    let store = h.session.result_store();
    let record = h.record().await;
    let parquet_url = StorageUrl::parse(&record.parquet_path).unwrap();
    let before_rank: Vec<std::collections::BTreeMap<String, u32>> = {
        let mut v = Vec::new();
        for q in [0i64, 4, 9, 77] {
            let query = h.vector_of(&q.to_string()).await.unwrap();
            v.push(
                h.search(&query, 149)
                    .await
                    .into_iter()
                    .map(|(id, d)| (id, d.to_bits()))
                    .collect(),
            );
        }
        v
    };

    let report = h.session.compact_embeddings(&h.table).await.unwrap();
    let n = report.version.unwrap();
    assert_eq!(report.parent_version, Some(parent.version));
    assert_eq!((report.live_rows, report.masked_rows), (149, 0));
    let manifest = store
        .read_version_manifest(&h.table, &parquet_url, n)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(manifest.fragments.len(), 1);
    assert_eq!(manifest.fragments[0].version, n);
    assert_eq!(manifest.segments.len(), 1);
    assert_eq!(manifest.segments[0].version, n);
    assert!(manifest.deletes.is_none());
    assert_ne!(manifest.identity, parent.identity.clone().unwrap());
    assert!(matches!(
        manifest.delta.descriptor,
        ProducingDescriptor::EmbeddingCompaction { .. }
    ));
    let record = h.record().await;
    assert_eq!(record.current_version, Some(n));
    assert_eq!(record.row_count, 149);
    for (i, q) in [0i64, 4, 9, 77].iter().enumerate() {
        let query = h.vector_of(&q.to_string()).await.unwrap();
        let after: std::collections::BTreeMap<String, u32> = h
            .search(&query, 149)
            .await
            .into_iter()
            .map(|(id, d)| (id, d.to_bits()))
            .collect();
        assert_eq!(
            after, before_rank[i],
            "the compaction carries the vectors byte-for-byte (query {q})"
        );
    }

    // Expire everything below the compaction.
    let next_before = record.next_version;
    let old_versions: Vec<i64> = h
        .session
        .catalog()
        .list_result_table_versions(&h.table)
        .await
        .unwrap()
        .iter()
        .map(|v| v.version)
        .filter(|v| *v < n)
        .collect();
    let expiry = h.session.expire_versions(&h.table, n).await.unwrap();
    assert_eq!(expiry.expired_versions, old_versions);
    assert!(expiry.objects_deleted > 0);
    let remaining: Vec<i64> = h
        .session
        .catalog()
        .list_result_table_versions(&h.table)
        .await
        .unwrap()
        .iter()
        .map(|v| v.version)
        .collect();
    assert_eq!(remaining, vec![n]);
    for v in &old_versions {
        assert!(!common::url_to_path(
            layout::version_manifest_url(&parquet_url, *v)
                .unwrap()
                .as_str()
        )
        .exists());
        assert!(!common::url_to_path(
            layout::version_deletes_url(&parquet_url, *v)
                .unwrap()
                .as_str()
        )
        .exists());
        assert!(!h.fragment_exists(&record, *v));
        assert!(h
            .session
            .catalog()
            .list_index_segments_for_version(&h.table, *v)
            .await
            .unwrap()
            .is_empty());
    }
    let segments = h
        .session
        .catalog()
        .list_index_segments(&h.table)
        .await
        .unwrap();
    assert!(segments.iter().any(|s| s.version == Some(n)));
    assert!(
        segments.iter().any(|s| s.version.is_none()),
        "the base segment is never reaped: {segments:?}"
    );
    assert!(common::url_to_path(&record.parquet_path).exists());
    assert!(common::url_to_path(
        layout::sidecar_url(&parquet_url, "materialization.json")
            .unwrap()
            .as_str()
    )
    .exists());
    let record = h.record().await;
    assert_eq!(
        record.next_version, next_before,
        "expiry never touches the allocator"
    );
    for (i, q) in [0i64, 4, 9, 77].iter().enumerate() {
        let query = h.vector_of(&q.to_string()).await.unwrap();
        let after: std::collections::BTreeMap<String, u32> = h
            .search(&query, 149)
            .await
            .into_iter()
            .map(|(id, d)| (id, d.to_bits()))
            .collect();
        assert_eq!(
            after, before_rank[i],
            "search unchanged after expiry (query {q})"
        );
    }
}

/// DELTA round-4 M5 oracle: a non-ready current version must refuse
/// `expire_versions` rather than reap against an unchecked manifest.
///
/// Honest scope note: this black-box path is ALSO refused earlier, by the
/// pre-existing `refreshable_record` gate at step 0 (`NotRefreshable {
/// CurrentVersionUnavailable }`), so this test proves the CALLER-VISIBLE
/// property the contract names ("a non-ready current version → refuses,
/// `reap_expired_version` never runs") but does not, by itself, isolate the
/// specific `read_version_manifest` → `resolve_version_manifest` swap this
/// same commit makes at `expire_versions`'s OWN manifest read further down —
/// that conversion's independent value is closing the narrower TOCTOU race
/// where the row is still `ready` at `refreshable_record`'s check and only
/// transitions to non-ready before the later read, which needs a park-point
/// test hook this commit does not add. Recorded here rather than claimed as
/// fully isolated.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn expire_versions_refuses_a_non_ready_current_version() {
    let h = harness(20).await;
    let mut edited = rows(20);
    edited[0].1 = "edited zero".into();
    write_parquet(&h.source_path, &edited);
    let report = h.refresh().await.unwrap();
    assert_eq!(report.outcome, RefreshOutcome::Published);
    let n = report.version.unwrap();

    // Break v_n's manifest and let recovery mark its version row `failed`
    // while the table's `current_version` keeps pointing at it — the
    // "non-ready current version" state this oracle targets (the same
    // recovery behaviour `current_manifest_loss_is_typed_unavailable_and_recomputable`
    // proves elsewhere).
    let record = h.record().await;
    let parquet_url = StorageUrl::parse(&record.parquet_path).unwrap();
    let manifest_url = layout::version_manifest_url(&parquet_url, n).unwrap();
    std::fs::remove_file(common::url_to_path(manifest_url.as_str())).unwrap();
    h.session.result_store().recover().await.unwrap();
    let row = h
        .session
        .catalog()
        .get_result_table_version(&h.table, n)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        row.status, "failed",
        "recovery marks the broken version failed"
    );
    let record = h.record().await;
    assert_eq!(
        record.current_version,
        Some(n),
        "current_version is unchanged by recovery — a non-ready CURRENT version"
    );

    let before_versions = h
        .session
        .catalog()
        .list_result_table_versions(&h.table)
        .await
        .unwrap();

    let err = h
        .session
        .expire_versions(&h.table, n + 1)
        .await
        .expect_err("a non-ready current version must refuse expiry, never reap against it");
    assert!(
        matches!(
            err,
            JammiError::NotRefreshable {
                reason: NotRefreshableReason::CurrentVersionUnavailable,
                ..
            }
        ),
        "{err:?}"
    );

    // `reap_expired_version` was never reached: the refused call deleted no
    // catalog version row.
    let after_versions = h
        .session
        .catalog()
        .list_result_table_versions(&h.table)
        .await
        .unwrap();
    assert_eq!(
        before_versions.len(),
        after_versions.len(),
        "a refused expiry deletes no version row"
    );
}
