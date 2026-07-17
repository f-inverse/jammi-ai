//! Byte-parity oracle for the `ImportPipeline::run` → `ResultStore` refactor.
//!
//! `ImportPipeline::run` used to hand-roll normalize + content-digest +
//! `External`-descriptor assembly, then call
//! `ResultStore::materialize_embedding_table` directly. That mechanism moved
//! into the generic `ResultStore::materialize_computed_embedding_table` verb;
//! `ImportPipeline::run` now just builds the caller-side provenance and calls
//! it. This file pins that the refactor is **byte-identical** in the two ways
//! that carry no workspace-version dependence:
//!
//! 1. the output Parquet artifact digest (the `_row_id`/`_source_id`/
//!    `_model_id`/`vector` bytes) — checked against a committed golden
//!    constant captured by running this exact test against `main @ 3164644`
//!    (before the refactor landed);
//! 2. the manifest's `ProducingDescriptor::External` — the producer id and
//!    every param, including the content digest of the normalized rows —
//!    checked against an independently hand-built reference descriptor, never
//!    read back from the code under test.
//!
//! The manifest's `DefinitionHash` is deliberately **not** pinned here: it
//! folds `MaterializationEnv::engine_version` (`CARGO_PKG_VERSION`), so it
//! legitimately changes on every workspace version bump — that is correct
//! production behaviour (an import's definition hash depends on the engine
//! that produced it), not a refactor regression. A test asserting equality
//! against one hardcoded hash would fail on every release rather than on an
//! actual regression; the descriptor check below still proves the refactor
//! preserved the producing definition, without pinning the version.
//!
//! The input fixture is generated in-process from a fixed algorithm (no wall
//! clock, no randomness, no UUIDs) rather than committed as a binary blob, so
//! it is exactly reproducible from source and legible in review; the same
//! [`ArrowWriter`] encoding settings the engine uses everywhere else make its
//! bytes deterministic across runs.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

use jammi_ai::session::InferenceSession;
use jammi_db::storage::StorageUrl;
use jammi_db::store::manifest::{ArtifactDigest, ProducingDescriptor};

use crate::common;

/// Golden output Parquet artifact digest (SHA-256 hex) captured the same way.
const GOLDEN_ARTIFACT_DIGEST: &str =
    "1093bebfee3cf0ac31368b4cc1c94de117e4bced8bebc481efbefadc00e883e8";

/// Golden content digest (SHA-256 hex) of the fixture's normalized
/// `(_row_id, vector)` rows, captured the same way as
/// [`GOLDEN_ARTIFACT_DIGEST`]. Unlike the `DefinitionHash`, this digest folds
/// only the row keys and L2-normalized vector bytes — no workspace version, no
/// timestamp — so it is reproducible across runs, across this refactor, and
/// across every workspace version bump.
const GOLDEN_CONTENT_DIGEST: &str =
    "4e0844aabcce5ea0e12fc582aeb681d899a619b50c6a6363f364accf1e42aaf1";

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

/// The independently hand-built reference `ProducingDescriptor` an
/// unregressed `ImportPipeline::run` must produce over [`keyed_vector_rows`] —
/// every field is written out literally here, never read back from
/// `crate::pipeline::import` or `ResultStore::materialize_computed_embedding_table`,
/// so a change to either (a renamed/dropped/added param, a different
/// producer id, a different content-digest input) is caught rather than
/// silently mirrored.
fn expected_descriptor() -> ProducingDescriptor {
    let mut params = BTreeMap::new();
    params.insert("source_id".to_string(), "byte_parity_source".to_string());
    // "byte-parity-model" has no `hf://`/`local:`/`file://` prefix and is not
    // an absolute or relative filesystem path, so `ModelSource::parse` treats
    // it as a bare Hub id and its canonical form is the string unchanged.
    params.insert("model_id".to_string(), "byte-parity-model".to_string());
    params.insert("dimensions".to_string(), "4".to_string());
    params.insert("key_column".to_string(), "_row_id".to_string());
    params.insert("text_columns".to_string(), "body".to_string());
    params.insert(
        "content_digest".to_string(),
        GOLDEN_CONTENT_DIGEST.to_string(),
    );

    ProducingDescriptor::External {
        producer_id: "external_import".to_string(),
        params,
    }
}

/// Run the import over the fixed fixture and return the resulting output
/// artifact's SHA-256 hex digest and its manifest's producing descriptor.
async fn run_import_and_capture(dir: &std::path::Path) -> (String, ProducingDescriptor) {
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

    let output_path = jammi_test_utils::url_to_path(&record.parquet_path);
    let bytes = std::fs::read(&output_path).unwrap();
    let artifact_digest = ArtifactDigest::of_bytes(&bytes).0;

    let url = StorageUrl::parse(&record.parquet_path).unwrap();
    let manifest = session
        .result_store()
        .read_materialization_manifest(&url)
        .await
        .unwrap()
        .expect("materialize_computed_embedding_table always writes a manifest sidecar");

    (artifact_digest, manifest.descriptor)
}

/// The load-bearing proof: the refactored `ImportPipeline::run` (which now
/// delegates to `ResultStore::materialize_computed_embedding_table`)
/// reproduces both the exact output artifact digest and the exact producing
/// descriptor a base-captured (pre-refactor) run produced over the identical
/// deterministic input — with no assertion depending on the workspace
/// version.
#[tokio::test]
async fn import_pipeline_is_byte_identical_across_the_refactor() {
    let dir = TempDir::new().unwrap();
    let (artifact_digest, descriptor) = run_import_and_capture(dir.path()).await;

    assert_eq!(
        artifact_digest, GOLDEN_ARTIFACT_DIGEST,
        "output artifact digest diverged from the pre-refactor golden — the \
         written Parquet bytes changed"
    );
    assert_eq!(
        descriptor,
        expected_descriptor(),
        "the producing `External` descriptor diverged from the independently \
         hand-built reference — the import's producer id, params, or \
         content-digest input changed"
    );
}

/// An import attributes its keys to a source column, so the pipeline requires
/// one — the same condition the wire decode enforces, held at the engine edge
/// so a direct Rust caller cannot slip past it. Every other argument here is
/// the one the byte-parity import above uses successfully, so the empty
/// `key_column` is the only thing under test.
#[tokio::test]
async fn import_rejects_an_empty_key_column() {
    let dir = TempDir::new().unwrap();
    let session = InferenceSession::new(common::test_config(dir.path()))
        .await
        .unwrap();
    let vectors_url = write_precomputed_vectors_fixture(dir.path());

    let err = session
        .import_embeddings(
            "byte_parity_source",
            "byte-parity-model",
            &vectors_url,
            "",
            &["body".to_string()],
            DIMS,
        )
        .await
        .expect_err("an empty key_column must be rejected");

    let msg = err.to_string();
    assert!(
        msg.contains("key_column"),
        "the error must name the offending field, got: {msg}"
    );
}
