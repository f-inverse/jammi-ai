//! The content-hash column, the deterministic base order, and the null-key
//! refusal — the three input-edge contracts every model-facing scan carries
//! (`generate_embeddings`, `infer`; the refresh leg lives in `refresh.rs`).
//!
//! - **`_content_hash`** is the fifth column of every fresh embedding table:
//!   the hex SHA-256 over the embedded source columns rendered exactly as the
//!   model read them, so a later refresh classifies rows by comparing hashes.
//! - **Base determinism.** A multi-file source under `execution_threads = 4`
//!   embeds every row (the hand-built plan coalesces before the blocking
//!   sort — without it the optimizer's round-robin repartition above the
//!   hash projection would leave partition 0 with a quarter of the rows), and
//!   the artifact digest is identical between `execution_threads = 1` and `4`
//!   because the rows are written in one total order.
//! - **Null keys** are one typed refusal, `InvalidKey { column, null_count }`,
//!   with the exact count, raised before the model is invoked even once, with
//!   nothing written.

use std::sync::Arc;

use arrow::array::{Array, Float32Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use jammi_ai::inference::runner::test_hooks::{forward_calls_for, reset_forward_calls_for};
use jammi_ai::model::ModelSource;
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::status::ResultTableStatus;
use jammi_db::error::JammiError;
use jammi_db::model_task::ModelTask;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::storage::StorageUrl;
use jammi_db::store::content_hash::{content_hash_row, ContentHash, ContentValue};
use jammi_db::store::CachePolicy;
use tempfile::TempDir;

use crate::common;

fn tiny_bert_id() -> String {
    format!("local:{}", common::cookbook_fixture("tiny_bert").display())
}

/// Write a `(id Int64 nullable, text Utf8)` Parquet file.
fn write_parquet(path: &std::path::Path, ids: Vec<Option<i64>>, texts: Vec<String>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("text", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(texts)),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut w = parquet::arrow::ArrowWriter::try_new(file, schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

async fn session_with_threads(dir: &TempDir, sub: &str, threads: usize) -> Arc<InferenceSession> {
    let artifact_dir = dir.path().join(sub);
    std::fs::create_dir_all(&artifact_dir).unwrap();
    let mut config = common::test_config(&artifact_dir);
    config.engine.execution_threads = threads;
    Arc::new(InferenceSession::new(config).await.unwrap())
}

async fn add_parquet_source(session: &InferenceSession, source_id: &str, url: String) {
    session
        .add_source(
            source_id,
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

/// A 4-file directory source, 1,000 rows per file, with duplicate keys
/// present (file 3 re-uses the first ten ids of file 0).
fn write_multi_file_source(dir: &std::path::Path) -> String {
    let src_dir = dir.join("multi");
    std::fs::create_dir_all(&src_dir).unwrap();
    for f in 0..4i64 {
        let ids: Vec<Option<i64>> = (0..1000i64)
            .map(|i| Some(if f == 3 && i < 10 { i } else { f * 1000 + i }))
            .collect();
        let texts: Vec<String> = (0..1000)
            .map(|i| format!("file {f} row {i} about topic {}", (f * 1000 + i) % 37))
            .collect();
        write_parquet(&src_dir.join(format!("part{f}.parquet")), ids, texts);
    }
    format!("file://{}", src_dir.display())
}

async fn artifact_digest(session: &InferenceSession, parquet_path: &str) -> String {
    let url = StorageUrl::parse(parquet_path).unwrap();
    session
        .result_store()
        .read_materialization_manifest(&url)
        .await
        .unwrap()
        .expect("manifest sidecar present")
        .artifact
        .0
}

/// Oracle (a): the fifth column exists, decodes as 64 lowercase hex, and
/// equals the pure fold over the source text of the same row.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn embedding_table_carries_the_content_hash_of_its_source_row() {
    let dir = TempDir::new().unwrap();
    let session = session_with_threads(&dir, "a", 1).await;
    let path = dir.path().join("src.parquet");
    let texts: Vec<String> = (0..6).map(|i| format!("sentence number {i}")).collect();
    write_parquet(&path, (0..6).map(Some).collect(), texts.clone());
    add_parquet_source(&session, "src", format!("file://{}", path.display())).await;

    let (record, _) = session
        .generate_text_embeddings(
            "src",
            &tiny_bert_id(),
            &["text".to_string()],
            "id",
            CachePolicy::Bypass,
            None,
        )
        .await
        .unwrap();

    let batches = session
        .sql(&format!(
            "SELECT _row_id, CAST(_content_hash AS VARCHAR) AS h FROM \"jammi.{}\" ORDER BY _row_id",
            record.table_name
        ))
        .await
        .unwrap();
    let mut seen = 0;
    for b in &batches {
        let ids = arrow::compute::cast(b.column(0), &DataType::Utf8).unwrap();
        let ids = ids.as_any().downcast_ref::<StringArray>().unwrap();
        let hs = arrow::compute::cast(b.column(1), &DataType::Utf8).unwrap();
        let hs = hs.as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..b.num_rows() {
            let row: usize = ids.value(i).parse().unwrap();
            let expected = content_hash_row(&[ContentValue::Str(&texts[row])]).to_hex();
            assert_eq!(
                hs.value(i),
                expected,
                "row {row}: hash must equal the pure fold"
            );
            ContentHash::from_hex(hs.value(i)).expect("64 lowercase hex");
            seen += 1;
        }
    }
    assert_eq!(seen, 6);
}

/// §6.17 — base determinism. Every row of a 4-file source is embedded under
/// `execution_threads = 4` (duplicates included: the initial embed tolerates
/// them), and the artifact digest equals the one `execution_threads = 1`
/// writes; `infer` returns the identical `_row_id` sequence and task columns
/// under both, excluding the wall-clock `_latency_ms`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn multi_partition_embed_is_complete_and_thread_count_invariant() {
    let dir = TempDir::new().unwrap();
    let url = write_multi_file_source(dir.path());
    let model = tiny_bert_id();

    let mut digests = Vec::new();
    let mut infer_views: Vec<Vec<String>> = Vec::new();
    for (sub, threads) in [("t1", 1usize), ("t4", 4usize)] {
        let session = session_with_threads(&dir, sub, threads).await;
        add_parquet_source(&session, "multi", url.clone()).await;
        let (record, _) = session
            .generate_text_embeddings(
                "multi",
                &model,
                &["text".to_string()],
                "id",
                CachePolicy::Bypass,
                None,
            )
            .await
            .unwrap();
        assert_eq!(
            record.row_count, 4_000,
            "execution_threads={threads}: every partition's rows must be embedded"
        );
        digests.push(artifact_digest(&session, &record.parquet_path).await);

        let (batches, _) = session
            .infer(
                "multi",
                &ModelSource::parse(&model),
                ModelTask::TextEmbedding,
                &["text".to_string()],
                "id",
                CachePolicy::Bypass,
            )
            .await
            .unwrap();
        let mut view = Vec::new();
        for b in &batches {
            let ids = arrow::compute::cast(b.column_by_name("_row_id").unwrap(), &DataType::Utf8)
                .unwrap();
            let ids = ids.as_any().downcast_ref::<StringArray>().unwrap();
            let status =
                arrow::compute::cast(b.column_by_name("_status").unwrap(), &DataType::Utf8)
                    .unwrap();
            let status = status.as_any().downcast_ref::<StringArray>().unwrap();
            let vectors = b
                .column_by_name("vector")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow::array::FixedSizeListArray>()
                .unwrap()
                .clone();
            for i in 0..b.num_rows() {
                let v = vectors.value(i);
                let v = v.as_any().downcast_ref::<Float32Array>().unwrap();
                let bits: Vec<u32> = (0..v.len()).map(|j| v.value(j).to_bits()).collect();
                view.push(format!("{}|{}|{bits:?}", ids.value(i), status.value(i)));
            }
        }
        assert_eq!(
            view.len(),
            4_000,
            "execution_threads={threads}: infer row count"
        );
        infer_views.push(view);
    }
    assert_eq!(
        digests[0], digests[1],
        "the embed artifact digest must not depend on execution_threads"
    );
    assert_eq!(
        infer_views[0], infer_views[1],
        "infer's `_row_id` sequence and task columns must not depend on execution_threads"
    );
}

/// §6.18 — null keys on the base embed and `infer` paths: the typed refusal
/// with the exact count, zero model invocations, nothing written, no
/// `building` row left behind.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn null_keys_are_refused_typed_before_any_model_call() {
    let dir = TempDir::new().unwrap();
    let session = session_with_threads(&dir, "n", 2).await;
    let path = dir.path().join("nulls.parquet");
    let ids = vec![Some(1), None, Some(3), None, Some(5), None, Some(7)];
    let texts: Vec<String> = (0..7).map(|i| format!("row {i}")).collect();
    write_parquet(&path, ids, texts);
    add_parquet_source(&session, "nulls", format!("file://{}", path.display())).await;
    let model = tiny_bert_id();

    // Warm the model cache so a later forward count of zero is about this
    // call's data path, not a load-time artefact.
    session.encode_text_query(&model, "warm").await.unwrap();

    let objects_before = list_artifacts(dir.path().join("n"));

    reset_forward_calls_for("nulls");
    let err = session
        .generate_text_embeddings(
            "nulls",
            &model,
            &["text".to_string()],
            "id",
            CachePolicy::Bypass,
            None,
        )
        .await
        .expect_err("null keys must refuse the embed");
    assert!(
        matches!(&err, JammiError::InvalidKey { column, null_count } if column == "id" && *null_count == 3),
        "expected InvalidKey {{ id, 3 }}, got {err:?}"
    );
    assert_eq!(
        forward_calls_for("nulls"),
        0,
        "the model must not be invoked before the refusal"
    );

    reset_forward_calls_for("nulls");
    let err = session
        .infer(
            "nulls",
            &ModelSource::parse(&model),
            ModelTask::TextEmbedding,
            &["text".to_string()],
            "id",
            CachePolicy::Bypass,
        )
        .await
        .expect_err("null keys must refuse infer");
    assert!(
        matches!(&err, JammiError::InvalidKey { column, null_count } if column == "id" && *null_count == 3),
        "expected InvalidKey {{ id, 3 }}, got {err:?}"
    );
    assert_eq!(
        forward_calls_for("nulls"),
        0,
        "the model must not be invoked before the refusal"
    );

    // Nothing written under the artifact root, and no `building` row left
    // live: the refusal unwinds through the writer's handle.
    let objects_after = list_artifacts(dir.path().join("n"));
    let new_objects: Vec<_> = objects_after
        .iter()
        .filter(|p| !objects_before.contains(p) && !p.ends_with(".db") && !p.contains("catalog.db"))
        .collect();
    assert!(
        new_objects
            .iter()
            .all(|p| !p.ends_with(".parquet") && !p.contains("__seg")),
        "no result artifact may be written for a refused call, got {new_objects:?}"
    );
    let building = session
        .catalog()
        .list_result_tables_by_status(ResultTableStatus::Building)
        .await
        .unwrap();
    assert!(
        building.is_empty(),
        "no live building row, got {building:?}"
    );
}

fn list_artifacts(root: std::path::PathBuf) -> Vec<String> {
    fn walk(dir: &std::path::Path, out: &mut Vec<String>) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out);
                } else {
                    out.push(p.display().to_string());
                }
            }
        }
    }
    let mut out = Vec::new();
    walk(&root, &mut out);
    out
}
