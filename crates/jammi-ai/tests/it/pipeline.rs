use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::physical_plan::ExecutionPlan;
use parquet::arrow::ArrowWriter;

use crate::common;
use jammi_ai::model::{ModelSource, ModelTask};
use jammi_ai::pipeline::embedding::build_embedding_plan;
use jammi_ai::pipeline::result_sink::filter_ok_and_extract_vectors;
use jammi_ai::session::InferenceSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use tempfile::TempDir;

fn tiny_bert_model() -> String {
    "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap()
}

async fn session_with_patents() -> (Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
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
    (session, dir)
}

// ─── Core generate_embeddings: one run, many assertions ──────────────────────
//
// Consolidates: writes Parquet, status=ready, dimensions match, sidecar index
// built, DataFusion queryable, metadata tracked, Parquet readable standalone.

#[tokio::test]
async fn generate_embeddings_produces_complete_result() {
    let (session, _dir) = session_with_patents().await;

    let record = session
        .generate_text_embeddings(
            "patents",
            &tiny_bert_model(),
            &["abstract".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
            None,
        )
        .await
        .unwrap()
        .0;

    // Parquet file exists with rows
    assert!(record.row_count > 0);
    assert!(
        common::url_to_path(&record.parquet_path).exists(),
        "Parquet file should exist at {}",
        record.parquet_path
    );

    // Status = ready with completed_at
    assert_eq!(record.status, "ready");
    assert!(record.completed_at.is_some());
    let from_catalog = session
        .catalog()
        .get_result_table(&record.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(from_catalog.status, "ready");

    // Dimensions match tiny_bert (hidden_size=32)
    assert_eq!(record.dimensions_raw(), Some(32));

    // Metadata tracked
    assert_eq!(record.source_id, "patents");
    assert_eq!(record.task, ModelTask::TextEmbedding);
    assert!(record.key_column.as_deref() == Some("id"));
    assert!(record.text_columns.as_deref() == Some("abstract"));

    // Sidecar index files exist (the table's single segment 0 bundle)
    let base_url = common::segment0_index_url(&session, &record.table_name).await;
    let base = common::url_to_path(&base_url);
    assert!(
        base.with_extension("usearch").exists(),
        "USearch index missing"
    );
    assert!(base.with_extension("rowmap").exists(), "Rowmap missing");
    assert!(
        base.with_extension("manifest.json").exists(),
        "Manifest missing"
    );

    // Queryable via DataFusion
    let results = session
        .sql(&format!(
            "SELECT _row_id, _source_id FROM \"jammi.{}\" LIMIT 5",
            record.table_name
        ))
        .await
        .unwrap();
    assert!(!results.is_empty());
    assert!(results[0].num_rows() <= 5);
    assert!(results[0].schema().field_with_name("_row_id").is_ok());
    assert!(results[0].schema().field_with_name("_source_id").is_ok());

    // Readable by external Parquet tools (no Jammi context)
    let file = std::fs::File::open(common::url_to_path(&record.parquet_path)).unwrap();
    let builder =
        parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let reader = builder.build().unwrap();
    let mut total_rows = 0;
    for batch in reader {
        let batch = batch.unwrap();
        total_rows += batch.num_rows();
        assert!(batch.schema().field_with_name("_row_id").is_ok());
        assert!(batch.schema().field_with_name("vector").is_ok());
    }
    assert_eq!(total_rows, record.row_count);
}

// ─── Multiple result tables coexist + sidecar fallback ───────────────────────

#[tokio::test]
async fn multiple_tables_and_sidecar_fallback() {
    let (session, _dir) = session_with_patents().await;

    let r1 = session
        .generate_text_embeddings(
            "patents",
            &tiny_bert_model(),
            &["abstract".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
            None,
        )
        .await
        .unwrap()
        .0;

    let r2 = session
        .generate_text_embeddings(
            "patents",
            &tiny_bert_model(),
            &["title".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
            None,
        )
        .await
        .unwrap()
        .0;

    // Two distinct tables
    assert_ne!(r1.table_name, r2.table_name);
    let tables = session
        .catalog()
        .find_result_tables("patents", Some(ModelTask::TextEmbedding), None)
        .await
        .unwrap();
    assert!(tables.len() >= 2);

    // Delete every segment sidecar for r1 — Parquet still queryable
    for base_url in common::segment_index_urls(&session, &r1.table_name).await {
        let base = common::url_to_path(&base_url);
        std::fs::remove_file(base.with_extension("usearch")).ok();
        std::fs::remove_file(base.with_extension("rowmap")).ok();
        std::fs::remove_file(base.with_extension("manifest.json")).ok();
    }
    let results = session
        .sql(&format!(
            "SELECT _row_id, vector FROM \"jammi.{}\" LIMIT 3",
            r1.table_name
        ))
        .await
        .unwrap();
    assert!(
        !results.is_empty(),
        "Parquet should still be queryable after sidecar deletion"
    );
}

// ─── Failed rows filtered in embedding output ────────────────────────────────

#[tokio::test]
async fn failed_rows_skipped_in_embedding_output() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());

    session
        .add_source(
            "patents_nulls",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents_with_nulls.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let record = session
        .generate_text_embeddings(
            "patents_nulls",
            &tiny_bert_model(),
            &["abstract".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
            None,
        )
        .await
        .unwrap()
        .0;

    // patents_with_nulls has 10 rows, 3 with null abstract
    assert!(
        record.row_count < 10,
        "Should have fewer rows than source due to null filtering, got {}",
        record.row_count
    );
    assert_eq!(record.status, "ready");
}

// ─── infer() persists results to Parquet ─────────────────────────────────────

#[tokio::test]
async fn infer_persists_results_to_parquet() {
    let (session, _dir) = session_with_patents().await;
    let model_source = ModelSource::local(common::cookbook_fixture("tiny_bert"));

    let results = session
        .infer(
            "patents",
            &model_source,
            ModelTask::TextEmbedding,
            &["abstract".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;

    assert!(!results.is_empty(), "Should return batches to caller");

    let tables = session
        .catalog()
        .find_result_tables("patents", Some(ModelTask::TextEmbedding), None)
        .await
        .unwrap();
    assert!(!tables.is_empty(), "Should have created a result table");
    assert_eq!(tables[0].status, "ready");
}

// ─── Existing tables loaded on new session ───────────────────────────────────

#[tokio::test]
async fn existing_tables_loaded_on_new_session() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());

    // First session: generate embeddings
    {
        let session = Arc::new(InferenceSession::new(config.clone()).await.unwrap());
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
                None,
            )
            .await
            .unwrap();
    }

    // Second session: result table should be queryable without re-generating
    {
        let session = Arc::new(InferenceSession::new(config).await.unwrap());
        let tables = session
            .catalog()
            .find_result_tables("patents", Some(ModelTask::TextEmbedding), None)
            .await
            .unwrap();
        assert!(!tables.is_empty());

        let record = &tables[0];
        let results = session
            .sql(&format!(
                "SELECT count(*) as cnt FROM \"jammi.{}\"",
                record.table_name
            ))
            .await
            .unwrap();
        assert!(!results.is_empty());
    }
}

// ─── Concurrent embedding generation on same source ──────────────────────────

#[tokio::test]
async fn concurrent_embedding_generation_on_same_source() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
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

    let model = tiny_bert_model();

    let session_a = Arc::clone(&session);
    let model_a = model.clone();
    let task_a = tokio::spawn(async move {
        session_a
            .generate_text_embeddings(
                "patents",
                &model_a,
                &["title".to_string()],
                "id",
                jammi_db::store::CachePolicy::Bypass,
                None,
            )
            .await
    });

    let session_b = Arc::clone(&session);
    let model_b = model.clone();
    let task_b = tokio::spawn(async move {
        session_b
            .generate_text_embeddings(
                "patents",
                &model_b,
                &["abstract".to_string()],
                "id",
                jammi_db::store::CachePolicy::Bypass,
                None,
            )
            .await
    });

    let (result_a, result_b) = tokio::try_join!(task_a, task_b).unwrap();
    let (record_a, _) = result_a.unwrap();
    let (record_b, _) = result_b.unwrap();

    // Both completed successfully with status "ready"
    assert_eq!(record_a.status, "ready");
    assert_eq!(record_b.status, "ready");

    // Both have the expected row count (patents.parquet has 20 rows)
    assert_eq!(record_a.row_count, 20);
    assert_eq!(record_b.row_count, 20);

    // Table names are distinct
    assert_ne!(record_a.table_name, record_b.table_name);

    // Both exist in catalog with status "ready"
    let cat_a = session
        .catalog()
        .get_result_table(&record_a.table_name)
        .await
        .unwrap()
        .unwrap();
    let cat_b = session
        .catalog()
        .get_result_table(&record_b.table_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(cat_a.status, "ready");
    assert_eq!(cat_b.status, "ready");
}

// ─── Large batch embedding completes without OOM ─────────────────────────────

#[tokio::test]
async fn large_batch_embedding_completes_without_oom() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());

    // Generate a 5000-row Parquet file
    let texts = [
        "Quantum computing advances in error correction",
        "Gene editing breakthroughs using CRISPR technology",
        "Renewable energy storage with solid-state batteries",
        "Autonomous vehicle navigation in urban environments",
        "Natural language processing for medical records",
        "Blockchain consensus mechanisms for scalability",
        "Protein folding prediction with deep learning",
        "Satellite internet constellation deployment strategies",
        "Robotic surgery systems with haptic feedback",
        "Carbon capture and sequestration techniques",
        "Superconducting materials at higher temperatures",
        "Brain-computer interface signal processing",
        "Fusion reactor plasma containment methods",
        "Biodegradable polymer synthesis for packaging",
        "Federated learning for privacy-preserving analytics",
        "Lidar-based terrain mapping for exploration",
        "Microbiome engineering for crop resilience",
        "Photonic computing architectures for inference",
        "Desalination membranes using graphene oxide",
        "Swarm intelligence algorithms for logistics",
    ];

    let row_count = 5000;
    let ids: Vec<i64> = (0..row_count as i64).collect();
    let text_values: Vec<&str> = (0..row_count).map(|i| texts[i % texts.len()]).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("text", DataType::Utf8, false),
    ]));

    let batch = arrow::array::RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(text_values)),
        ],
    )
    .unwrap();

    let parquet_path = dir.path().join("large_batch.parquet");
    let file = std::fs::File::create(&parquet_path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let parquet_url = format!("file://{}", parquet_path.display());
    session
        .add_source(
            "large_batch",
            SourceType::File,
            SourceConnection {
                url: Some(parquet_url),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let record = session
        .generate_text_embeddings(
            "large_batch",
            &tiny_bert_model(),
            &["text".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
            None,
        )
        .await
        .unwrap()
        .0;

    assert_eq!(record.row_count, 5000);
    assert_eq!(record.status, "ready");
}

// ─── Opt-in memoization: embeddings are honestly NEVER cacheable ─────────────
//
// An embedding pipeline anchors its raw source `UnpinnedAtInstant` (no version
// surface in open-core), so `probe_cache` never hits: a `Use` request always
// computes. The cache is honestly OFF here, not silently broken.

#[tokio::test]
async fn cache_use_on_embeddings_always_recomputes_unpinned_source() {
    use jammi_db::store::{CacheOutcome, CachePolicy};

    let (session, _dir) = session_with_patents().await;

    let (first, first_outcome) = session
        .generate_text_embeddings(
            "patents",
            &tiny_bert_model(),
            &["abstract".to_string()],
            "id",
            CachePolicy::Use,
            None,
        )
        .await
        .unwrap();
    assert_eq!(first_outcome, CacheOutcome::Computed);

    // A byte-identical second Use request still computes — the source is unpinned,
    // so the probe is honestly always a miss and a fresh table is materialised.
    let (second, second_outcome) = session
        .generate_text_embeddings(
            "patents",
            &tiny_bert_model(),
            &["abstract".to_string()],
            "id",
            CachePolicy::Use,
            None,
        )
        .await
        .unwrap();
    assert_eq!(
        second_outcome,
        CacheOutcome::Computed,
        "an unpinned-anchored producer never reuses — caching is honestly off"
    );
    assert_ne!(
        first.table_name, second.table_name,
        "each Use embedding run materialises a fresh table (no reuse)"
    );
}

// ─── `build_embedding_plan` is THE one plan-building site (contract
// `feat_500-wave4` §2.5) ───────────────────────────────────────────────────
//
// `EmbeddingPipeline::run` (behind `generate_text_embeddings`) calls
// `build_embedding_plan` rather than building its own copy of the plan, so
// collecting the SAME construction inputs through `build_embedding_plan`
// directly — bypassing `run`/`generate` entirely — must produce rows
// byte-identical to what `generate` persisted, over a multi-partition
// (two-file) source.

/// A 2-file directory source, disjoint keys, small enough to embed fast.
fn write_two_file_source(dir: &std::path::Path) -> String {
    let src_dir = dir.join("two_files");
    std::fs::create_dir_all(&src_dir).unwrap();
    let files: [[(i64, &str); 2]; 2] = [
        [(0, "alpha widget"), (1, "beta widget")],
        [(2, "gamma gadget"), (3, "delta gadget")],
    ];
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("text", DataType::Utf8, false),
    ]));
    for (f, rows) in files.iter().enumerate() {
        let ids: Vec<i64> = rows.iter().map(|(id, _)| *id).collect();
        let texts: Vec<String> = rows.iter().map(|(_, t)| t.to_string()).collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(texts)),
            ],
        )
        .unwrap();
        let file = std::fs::File::create(src_dir.join(format!("part{f}.parquet"))).unwrap();
        let mut w = ArrowWriter::try_new(file, schema.clone(), None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }
    format!("file://{}", src_dir.display())
}

#[tokio::test]
async fn build_embedding_plan_collected_in_process_matches_generates_written_rows() {
    let dir = TempDir::new().unwrap();
    let url = write_two_file_source(dir.path());
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    session
        .add_source(
            "two_files",
            SourceType::File,
            SourceConnection {
                url: Some(url),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let model = tiny_bert_model();
    let (record, _) = session
        .generate_text_embeddings(
            "two_files",
            &model,
            &["text".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
            None,
        )
        .await
        .unwrap();
    assert_eq!(record.row_count, 4, "both files' rows land in one table");

    let persisted = session
        .sql(&format!(
            "SELECT _row_id, _source_id, _model_id, vector, _content_hash FROM \"jammi.{}\" \
             ORDER BY _row_id",
            record.table_name
        ))
        .await
        .unwrap();

    // Independently, through `build_embedding_plan` alone — never through
    // `generate_text_embeddings`/`EmbeddingPipeline::run` — with the SAME
    // construction inputs `generate` resolved above.
    let plan: Arc<dyn ExecutionPlan> = build_embedding_plan(
        &session,
        "two_files",
        ModelSource::parse(&model),
        ModelTask::TextEmbedding,
        &["text".to_string()],
        "id",
        record.dimensions().expect("dimensions recorded").get(),
    )
    .await
    .unwrap();
    let task_ctx = session.context().task_ctx();
    let stream = plan.execute(0, task_ctx).unwrap();
    let raw_batches = datafusion::physical_plan::common::collect(stream)
        .await
        .unwrap();
    let mut ok_batches = Vec::new();
    for batch in &raw_batches {
        let (ok_batch, _row_ids, _vectors) = filter_ok_and_extract_vectors(batch).unwrap();
        if ok_batch.num_rows() > 0 {
            ok_batches.push(ok_batch);
        }
    }
    assert!(!ok_batches.is_empty(), "the independent plan realized rows");
    let independent = arrow::compute::concat_batches(&ok_batches[0].schema(), &ok_batches).unwrap();
    let sort_indices =
        arrow::compute::sort_to_indices(independent.column_by_name("_row_id").unwrap(), None, None)
            .unwrap();
    let independent = arrow::compute::take_record_batch(&independent, &sort_indices).unwrap();

    assert_eq!(
        persisted.iter().map(|b| b.num_rows()).sum::<usize>(),
        independent.num_rows(),
        "row counts must match between generate's written table and the independently \
         collected plan"
    );
    let persisted_str = arrow::util::pretty::pretty_format_batches(&persisted)
        .unwrap()
        .to_string();
    let independent_str = arrow::util::pretty::pretty_format_batches(&[independent])
        .unwrap()
        .to_string();
    assert_eq!(
        persisted_str, independent_str,
        "build_embedding_plan collected in-process must equal generate's written rows \
         byte-for-byte — one plan-building site, contract feat_500-wave4 §2.5"
    );
}
