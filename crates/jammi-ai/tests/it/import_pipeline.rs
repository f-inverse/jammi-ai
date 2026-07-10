//! Byte-parity oracle for the `ImportPipeline::run` → `ResultStore` refactor.
//!
//! `ImportPipeline::run` used to hand-roll normalize + content-digest +
//! `External`-descriptor assembly, then call
//! `ResultStore::materialize_embedding_table` directly. That mechanism moved
//! into the generic `ResultStore::materialize_computed_embedding_table` verb;
//! `ImportPipeline::run` now just builds the caller-side provenance and calls
//! it. This file pins that the refactor is **byte-identical**: the same
//! deterministic input, run through the pipeline, must reproduce both the
//! manifest `DefinitionHash` and the output Parquet artifact digest recorded
//! from the pre-refactor code — hardcoded below as committed golden constants
//! captured by running this exact test against `main @ 3164644` (before the
//! refactor landed).
//!
//! The input fixture is generated in-process from a fixed algorithm (no wall
//! clock, no randomness, no UUIDs) rather than committed as a binary blob, so
//! it is exactly reproducible from source and legible in review; the same
//! [`ArrowWriter`] encoding settings the engine uses everywhere else make its
//! bytes deterministic across runs.

use std::sync::Arc;

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

use jammi_ai::session::InferenceSession;
use jammi_db::storage::StorageUrl;
use jammi_db::store::manifest::ArtifactDigest;

use crate::common;

/// Golden `DefinitionHash` captured by running this test's fixture through
/// `ImportPipeline::run` on `main @ 3164644` (pre-refactor). The definition
/// hash excludes the input anchors (and therefore the `UnpinnedAtInstant`
/// timestamp), so it is reproducible across runs and across this refactor.
const GOLDEN_DEFINITION_HASH: &str =
    "6ee12df08ecb3184bf690e6960af0337715090cee6db238efbd4f240bea6a15a";

/// Golden output Parquet artifact digest (SHA-256 hex) captured the same way.
const GOLDEN_ARTIFACT_DIGEST: &str =
    "1093bebfee3cf0ac31368b4cc1c94de117e4bced8bebc481efbefadc00e883e8";

const DIMS: usize = 4;

/// Deterministic `(_row_id, vector)` fixture: five rows, fixed non-unit
/// vectors (so normalization is exercised, not a no-op), written as plain
/// `Utf8` (never `Utf8View`) because the reader downcasts the key column to
/// `StringArray` directly.
fn keyed_vector_rows() -> (Vec<String>, Vec<Vec<f32>>) {
    let row_ids: Vec<String> = (0..5).map(|i| format!("row-{i}")).collect();
    let vectors: Vec<Vec<f32>> = (0..5)
        .map(|i| (0..DIMS).map(|d| ((i * DIMS + d) as f32) + 1.0).collect())
        .collect();
    (row_ids, vectors)
}

/// Write the fixed `(_row_id, vector)` rows to a Parquet object at
/// `dir/precomputed_vectors.parquet` and return its `file://` URL.
fn write_precomputed_vectors_fixture(dir: &std::path::Path) -> StorageUrl {
    let (row_ids, vectors) = keyed_vector_rows();
    let schema = Arc::new(Schema::new(vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                DIMS as i32,
            ),
            false,
        ),
    ]));

    let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let vector_array =
        FixedSizeListArray::try_new(item, DIMS as i32, Arc::new(Float32Array::from(flat)), None)
            .unwrap();

    let batch = arrow::array::RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(row_ids)) as ArrayRef,
            Arc::new(vector_array),
        ],
    )
    .unwrap();

    let parquet_path = dir.join("precomputed_vectors.parquet");
    let file = std::fs::File::create(&parquet_path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    StorageUrl::parse(&format!("file://{}", parquet_path.display())).unwrap()
}

/// Run the import over the fixed fixture and return the resulting manifest
/// `DefinitionHash` and the output artifact's SHA-256 hex digest.
async fn run_import_and_capture(dir: &std::path::Path) -> (String, String) {
    let session = InferenceSession::new(common::test_config(dir))
        .await
        .unwrap();
    let vectors_url = write_precomputed_vectors_fixture(dir);

    let record = session
        .import_embeddings(
            "byte_parity_source",
            "byte-parity-model",
            &vectors_url,
            "_row_id",
            &["body".to_string()],
            DIMS,
        )
        .await
        .unwrap();
    assert_eq!(record.status, "ready");

    let definition_hash = record
        .definition_hash
        .clone()
        .expect("materialize_computed_embedding_table always records a definition hash");

    let output_path = jammi_test_utils::url_to_path(&record.parquet_path);
    let bytes = std::fs::read(&output_path).unwrap();
    let artifact_digest = ArtifactDigest::of_bytes(&bytes).0;

    (definition_hash, artifact_digest)
}

/// The load-bearing proof: the refactored `ImportPipeline::run` (which now
/// delegates to `ResultStore::materialize_computed_embedding_table`) reproduces
/// the exact `DefinitionHash` and artifact digest a base-captured (pre-
/// refactor) run produced over the identical deterministic input.
#[tokio::test]
async fn import_pipeline_is_byte_identical_across_the_refactor() {
    let dir = TempDir::new().unwrap();
    let (definition_hash, artifact_digest) = run_import_and_capture(dir.path()).await;

    assert_eq!(
        definition_hash, GOLDEN_DEFINITION_HASH,
        "DefinitionHash diverged from the pre-refactor golden — the import's \
         producing descriptor or environment changed"
    );
    assert_eq!(
        artifact_digest, GOLDEN_ARTIFACT_DIGEST,
        "output artifact digest diverged from the pre-refactor golden — the \
         written Parquet bytes changed"
    );
}
