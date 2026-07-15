// Hermetic tests for int8 vector-storage quantization + retrieve→rescore
// (`StoragePrecision` / `AnnIndexConfig::{storage_precision, oversample}` /
// `SearchRequest::oversample`). Uses the real `patents.parquet` fixture and
// the tiny BERT encoder fixture — no network access required.
//
// The corpus is small (20 rows), so an `Int8` build at the default oversample
// covers the whole table in one rescore pass; the correctness assertions
// compare against an independently-computed exact brute-force cosine ranking
// over the *ground-truth* Parquet vectors, never against the quantized
// index's own (lossy) distances.

use std::sync::Arc;

use arrow::array::{Array, FixedSizeListArray, Float32Array, StringArray};
use arrow::datatypes::DataType;
use jammi_db::config::{AnnIndexConfig, StoragePrecision};
use jammi_db::index::sidecar::SidecarIndex;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_numerics::distance::cosine_distance;
use tempfile::TempDir;

use crate::common;

fn tiny_bert_model() -> String {
    "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap()
}

/// A session config rooted at `dir` with the embedding ANN sidecar built at
/// `precision` and the table-default oversample set explicitly (rather than
/// left at its own default), so every test controls both knobs.
fn config_at(
    dir: &std::path::Path,
    precision: StoragePrecision,
    oversample: usize,
) -> jammi_db::config::JammiConfig {
    let mut config = common::test_config(dir);
    config.embedding.ann.storage_precision = precision;
    config.embedding.ann.oversample = Some(oversample);
    config
}

async fn session_with_patents_at(
    dir: &std::path::Path,
    precision: StoragePrecision,
    oversample: usize,
) -> Arc<jammi_ai::session::InferenceSession> {
    let config = config_at(dir, precision, oversample);
    let session = Arc::new(
        jammi_ai::session::InferenceSession::new(config)
            .await
            .unwrap(),
    );
    session
        .add_source(
            "patents",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    session
        .generate_text_embeddings(
            "patents",
            &tiny_bert_model(),
            &["abstract".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap();
    session
}

/// Read every `(_row_id, vector)` pair of an embedding table straight off its
/// Parquet object — the ground truth a rescore is supposed to recover, read
/// independently of the ANN sidecar entirely.
async fn read_ground_truth(
    session: &jammi_ai::session::InferenceSession,
    table_name: &str,
) -> Vec<(String, Vec<f32>)> {
    let batches = session
        .sql(&format!(
            "SELECT _row_id, vector FROM \"jammi.{table_name}\""
        ))
        .await
        .unwrap();
    let mut rows = Vec::new();
    for batch in &batches {
        let row_id_col = batch.column_by_name("_row_id").unwrap();
        let row_ids = arrow::compute::cast(row_id_col.as_ref(), &DataType::Utf8).unwrap();
        let row_ids = row_ids.as_any().downcast_ref::<StringArray>().unwrap();
        let vectors = batch
            .column_by_name("vector")
            .unwrap()
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        for i in 0..batch.num_rows() {
            let v = vectors.value(i);
            let floats = v.as_any().downcast_ref::<Float32Array>().unwrap();
            rows.push((
                row_ids.value(i).to_string(),
                (0..floats.len()).map(|j| floats.value(j)).collect(),
            ));
        }
    }
    rows
}

/// Brute-force exact cosine top-k over `ground_truth`, the same stable
/// tie-break (`distance` ascending, `row_id` ascending) the exact
/// neighbor-graph driver and the rescore re-sort both use.
fn exact_top_k(ground_truth: &[(String, Vec<f32>)], query: &[f32], k: usize) -> Vec<(String, f32)> {
    let mut scored: Vec<(String, f32)> = ground_truth
        .iter()
        .map(|(id, v)| (id.clone(), cosine_distance(query, v)))
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    scored.truncate(k);
    scored
}

/// (1) The full production path — `session.search` over an `Int8`-precision
/// table with the default oversample (4), which at this 20-row corpus and
/// k=5 covers every row (`candidate_k = 20`) — must recover the *exact*
/// brute-force top-k over the ground-truth `f32` vectors: same row order, and
/// similarity scores matching the exact computation tightly (both paths run
/// the identical `cosine_distance` over the identical exact vectors).
#[tokio::test]
async fn int8_search_with_rescore_matches_exact_f32_baseline() {
    let dir = TempDir::new().unwrap();
    let session = session_with_patents_at(dir.path(), StoragePrecision::Int8, 4).await;

    let table = session
        .catalog()
        .resolve_embedding_table("patents", None)
        .await
        .unwrap();
    assert_eq!(
        table.storage_precision,
        Some(StoragePrecision::Int8),
        "the table's catalog row must record the Int8 build precision"
    );

    let ground_truth = read_ground_truth(&session, &table.table_name).await;
    assert_eq!(ground_truth.len(), 20, "the patents fixture has 20 rows");

    let query = ground_truth[3].1.clone();
    let k = 5;
    let expected = exact_top_k(&ground_truth, &query, k);

    let hits = session
        .search("patents", query, k, None, None)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();

    let (actual_ids, actual_scores) = hit_ids_and_scores(&hits);
    let expected_ids: Vec<&str> = expected.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(
        actual_ids, expected_ids,
        "Int8 retrieve→rescore must recover the exact brute-force top-k row order"
    );
    for ((_, expected_dist), actual_sim) in expected.iter().zip(actual_scores.iter()) {
        let expected_sim = 1.0 - expected_dist;
        assert!(
            (expected_sim - actual_sim).abs() < 1e-4,
            "rescored similarity {actual_sim} must match the exact cosine similarity \
             {expected_sim} (rescore must score off exact vectors, not the quantized ones)"
        );
    }
}

/// (2) An `F32` table stays single-stage: no `.rawf32` rescore companion is
/// ever written beside its sidecar bundle, because the index's own vectors
/// are already exact.
#[tokio::test]
async fn f32_table_writes_no_rescore_companion() {
    let dir = TempDir::new().unwrap();
    let session = session_with_patents_at(dir.path(), StoragePrecision::F32, 4).await;

    let table = session
        .catalog()
        .resolve_embedding_table("patents", None)
        .await
        .unwrap();
    assert_eq!(table.storage_precision, Some(StoragePrecision::F32));

    let seg_url = common::segment0_index_url(&session, &table.table_name).await;
    let base = jammi_test_utils::url_to_path(&seg_url);
    assert!(
        !base.with_extension("rawf32").exists(),
        "an F32-precision index must never write a rescore companion"
    );
}

/// (5) A sidecar whose on-disk `scalar_kind` no longer matches the catalog
/// row's persisted `storage_precision` (e.g. the graph predates a precision
/// change) must never be silently reopened as if it matched — `resolve_search_mode`
/// catches the strict-precision `IncompatibleFormat` and falls back to exact
/// search, so a query against the tampered table still returns the correct
/// (brute-force) answer rather than crashing or silently misreading the
/// quantized graph.
#[tokio::test]
async fn stale_manifest_precision_falls_back_to_exact_search_not_a_crash() {
    let dir = TempDir::new().unwrap();
    let session = session_with_patents_at(dir.path(), StoragePrecision::Int8, 4).await;

    let table = session
        .catalog()
        .resolve_embedding_table("patents", None)
        .await
        .unwrap();
    let ground_truth = read_ground_truth(&session, &table.table_name).await;
    let seg_url = common::segment0_index_url(&session, &table.table_name).await;
    let base = jammi_test_utils::url_to_path(&seg_url);
    let manifest_path = base.with_extension("manifest.json");

    let mut manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
    assert_eq!(manifest["scalar_kind"], "int8");
    manifest["scalar_kind"] = serde_json::json!("f16");
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let query = ground_truth[0].1.clone();
    let k = 3;
    let expected = exact_top_k(&ground_truth, &query, k);

    let hits = session
        .search("patents", query, k, None, None)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();
    let (actual_ids, _) = hit_ids_and_scores(&hits);
    let expected_ids: Vec<&str> = expected.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(
        actual_ids, expected_ids,
        "a stale/mismatched sidecar must fall back to the exact brute-force path, not crash \
         or silently misload"
    );
}

/// (3) An `Int8` table DOES write the companion, and it is the source of
/// exact-vector recovery: the quantized index's own `get` (USearch's lossy
/// reconstruction) differs from the Parquet ground truth for at least one
/// row, while `get_exact` (the mmap'd companion) matches it bit-for-bit.
#[tokio::test]
async fn int8_index_writes_companion_and_get_exact_recovers_ground_truth() {
    let dir = TempDir::new().unwrap();
    let session = session_with_patents_at(dir.path(), StoragePrecision::Int8, 4).await;

    let table = session
        .catalog()
        .resolve_embedding_table("patents", None)
        .await
        .unwrap();
    let seg_url = common::segment0_index_url(&session, &table.table_name).await;
    let base = jammi_test_utils::url_to_path(&seg_url);
    assert!(
        base.with_extension("rawf32").exists(),
        "an Int8-precision index must write the rescore companion"
    );

    let ground_truth = read_ground_truth(&session, &table.table_name).await;
    // Load the segment's own `SidecarIndex` directly: `get` (USearch's lossy
    // reconstruction) vs `get_exact` (the companion) is a per-segment property,
    // not part of the merged `SegmentedIndex` surface.
    let index = SidecarIndex::load(&base, &AnnIndexConfig::default(), StoragePrecision::Int8)
        .expect("the Int8 sidecar segment must load");
    assert_eq!(index.storage_precision(), StoragePrecision::Int8);

    let mut saw_quantization_loss = false;
    for (row_id, exact_vector) in &ground_truth {
        let exact = index.get_exact(row_id).unwrap().unwrap();
        assert_eq!(
            &exact, exact_vector,
            "get_exact must return the bit-exact f32 vector for '{row_id}'"
        );
        let quantized = index.get(row_id).unwrap().unwrap();
        if &quantized != exact_vector {
            saw_quantization_loss = true;
        }
    }
    assert!(
        saw_quantization_loss,
        "at least one row's Int8-quantized reconstruction (`get`) must differ from the exact \
         f32 vector — otherwise this test cannot tell a real quantized build from an F32 one"
    );
}

/// (4) A search request's `oversample` overrides the table's config default:
/// requesting `oversample = 1` (below the table's default of 4) narrows the
/// quantized retrieval to exactly `k` candidates — the same set the raw
/// quantized index itself would return for `k` with no rescore widening at
/// all. If the override did not reach the retrieval stage, the candidate
/// pool would stay at the table's default-derived width instead.
#[tokio::test]
async fn per_request_oversample_overrides_table_default() {
    let dir = TempDir::new().unwrap();
    // Table default oversample = 4 (would retrieve `k * 4` candidates absent
    // an override).
    let session = session_with_patents_at(dir.path(), StoragePrecision::Int8, 4).await;
    let table = session
        .catalog()
        .resolve_embedding_table("patents", None)
        .await
        .unwrap();
    let ground_truth = read_ground_truth(&session, &table.table_name).await;
    let query = ground_truth[7].1.clone();
    let k = 3;

    let index = session
        .result_store()
        .resolve_search_mode(&table)
        .await
        .unwrap()
        .expect("the Int8 sidecar index must load");
    let raw_quantized_top_k = index.search(&query, k).unwrap();
    let mut raw_ids: Vec<String> = raw_quantized_top_k
        .iter()
        .map(|(id, _)| id.clone())
        .collect();
    raw_ids.sort();

    let hits = session
        .search("patents", query, k, None, Some(1))
        .await
        .unwrap()
        .run()
        .await
        .unwrap();
    let (mut overridden_ids, _) = hit_ids_and_scores(&hits);
    let mut overridden_ids: Vec<String> = overridden_ids.drain(..).map(str::to_string).collect();
    overridden_ids.sort();

    assert_eq!(
        overridden_ids, raw_ids,
        "oversample = 1 must narrow the rescored candidate set to exactly the raw quantized \
         top-k — proving the per-request value, not the table's default of 4, reached the \
         retrieval stage"
    );
}

/// (6) The table's own **stamped** `oversample` — not whatever the current
/// deployment default happens to be — drives a rescore's candidate breadth.
/// The table is created under a wide deployment default (`7`, so `k * 7 =
/// 21` candidates cover this whole 20-row corpus and the rescore recovers
/// the *exact* top-k); a second session then reopens the *same* on-disk
/// table+catalog under a narrower deployment default (`1`, so `k * 1 = k`,
/// no widening at all) and searches with no per-request override. If
/// oversample resolution read the live deployment config instead of the
/// table's stamped column, the second session would narrow the candidate
/// set to the raw quantized top-k, which — for this fixture/query — misses
/// one of the true top-k neighbours (see `int8_search_with_rescore` sibling
/// tests establishing quantized-vs-exact divergence on this corpus).
#[tokio::test]
async fn table_stamped_oversample_drives_rescore_not_deployment_config() {
    let dir = TempDir::new().unwrap();
    // Created under a wide deployment default: k=3 * oversample=7 = 21
    // candidates, which covers the whole 20-row corpus, so this table's
    // stamped oversample guarantees an exact rescore.
    let creator_session = session_with_patents_at(dir.path(), StoragePrecision::Int8, 7).await;
    let table = creator_session
        .catalog()
        .resolve_embedding_table("patents", None)
        .await
        .unwrap();
    assert_eq!(
        table.oversample,
        Some(7),
        "the table's catalog row must stamp the oversample in effect at creation"
    );

    let ground_truth = read_ground_truth(&creator_session, &table.table_name).await;
    let k = 3;
    let query = ground_truth[1].1.clone();
    let expected = exact_top_k(&ground_truth, &query, k);
    let mut expected_ids: Vec<String> = expected.iter().map(|(id, _)| id.clone()).collect();
    expected_ids.sort();

    // A second session reopens the SAME on-disk catalog+table under a
    // narrower deployment default — simulating a deployment whose oversample
    // config changed after this table was created.
    let reopened_config = config_at(dir.path(), StoragePrecision::Int8, 1);
    let reopened_session = Arc::new(
        jammi_ai::session::InferenceSession::new(reopened_config)
            .await
            .unwrap(),
    );

    let hits = reopened_session
        .search("patents", query, k, None, None)
        .await
        .unwrap()
        .run()
        .await
        .unwrap();
    let (mut actual_ids, _) = hit_ids_and_scores(&hits);
    let mut actual_ids: Vec<String> = actual_ids.drain(..).map(str::to_string).collect();
    actual_ids.sort();

    assert_eq!(
        actual_ids, expected_ids,
        "a search with no per-request override must use the table's own stamped \
         oversample (7), not the reopening session's deployment default (1) — \
         otherwise the rescore narrows to the raw quantized top-k and misses a \
         true top-k neighbour"
    );
}

/// Every `_row_id` + `similarity` pair from a hydrated search's batches, in
/// row order.
fn hit_ids_and_scores(batches: &[arrow::record_batch::RecordBatch]) -> (Vec<&str>, Vec<f32>) {
    let mut ids = Vec::new();
    let mut scores = Vec::new();
    for batch in batches {
        let row_ids = batch
            .column_by_name("_row_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let similarities = batch
            .column_by_name("similarity")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            ids.push(row_ids.value(i));
            scores.push(similarities.value(i));
        }
    }
    (ids, scores)
}
