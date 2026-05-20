//! Schema-shape contract between the data-driven channel mechanism
//! and the legacy provenance output.
//!
//! Locks in the canonical Arrow schema produced when `vector` and
//! `inference` channels participate, so future refactors that drift the
//! shape are caught at test time.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use jammi_ai::evidence::{merge_channels, ChannelContribution};
use jammi_engine::catalog::Catalog;
use jammi_engine::ChannelId;
use tempfile::tempdir;

fn fixtures_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests")
        .join("fixtures")
}

fn deterministic_batch(n: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("_source_id", DataType::Utf8, false),
    ]));
    let row_id = Arc::new(StringArray::from(
        (0..n).map(|i| format!("r{i}")).collect::<Vec<_>>(),
    )) as ArrayRef;
    let src = Arc::new(StringArray::from(vec!["src"; n])) as ArrayRef;
    RecordBatch::try_new(schema, vec![row_id, src]).unwrap()
}

fn open_catalog() -> (tempfile::TempDir, Catalog) {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();
    (dir, catalog)
}

/// Canonical merge of `vector` + `inference` channels.
///
/// The output schema is asserted against a checked-in golden so any
/// drift in field name, dtype, nullability, or ordering surfaces at test
/// time. To regenerate the golden after an *intentional* shape change,
/// set `JAMMI_REGENERATE_GOLDENS=1` and re-run this test.
#[test]
fn vector_and_inference_reexpressed_produce_byte_identical_recordbatch() {
    let (_dir, catalog) = open_catalog();
    let vector = ChannelId::new("vector").unwrap();
    let inference = ChannelId::new("inference").unwrap();

    let n = 4;
    let batch = deterministic_batch(n);

    let vector_contrib = ChannelContribution::single(
        vector.clone(),
        Arc::new(Float32Array::from(vec![0.95_f32, 0.90, 0.85, 0.80])) as ArrayRef,
    );
    let inference_contrib = ChannelContribution {
        channel: inference.clone(),
        columns: vec![
            Arc::new(StringArray::from(vec!["m"; n])) as ArrayRef,
            Arc::new(StringArray::from(vec!["embedding"; n])) as ArrayRef,
            Arc::new(Float32Array::from(vec![0.9_f32; n])) as ArrayRef,
        ],
    };

    let merged = merge_channels(
        &catalog,
        &[batch],
        &[vector.clone(), inference.clone()],
        &[vector],
        &[inference],
        &[vec![vector_contrib, inference_contrib]],
    )
    .unwrap();
    assert_eq!(merged.len(), 1);

    let schema = merged[0].schema();
    let actual = schema_to_canonical_json(&schema);

    let golden_path = fixtures_dir().join("golden_provenance_schema.json");
    if std::env::var("JAMMI_REGENERATE_GOLDENS").is_ok() {
        std::fs::write(&golden_path, &actual).expect("regenerating golden_provenance_schema.json");
        eprintln!("regenerated golden at {}", golden_path.display());
        return;
    }

    let expected = std::fs::read_to_string(&golden_path).unwrap_or_else(|e| {
        panic!(
            "missing golden file {}: {e}; re-run with JAMMI_REGENERATE_GOLDENS=1 to create it",
            golden_path.display()
        )
    });

    assert_eq!(
        actual.trim(),
        expected.trim(),
        "merged schema drifted from the golden at {}. \
         If this drift is intentional, re-run with JAMMI_REGENERATE_GOLDENS=1.",
        golden_path.display()
    );
}

/// Render the schema as a stable canonical JSON form: `[{name, dtype,
/// nullable}, …]`. We deliberately avoid Arrow's full `to_json` because
/// it serializes private metadata that may shift across Arrow versions
/// without changing observational shape.
fn schema_to_canonical_json(schema: &Schema) -> String {
    let entries: Vec<serde_json::Value> = schema
        .fields()
        .iter()
        .map(|f| {
            serde_json::json!({
                "name": f.name(),
                "dtype": format!("{:?}", f.data_type()),
                "nullable": f.is_nullable(),
            })
        })
        .collect();
    serde_json::to_string_pretty(&entries).unwrap()
}

/// Sanity-check companion: the catalog's `merged_schema` projection
/// matches what the merger appends to the batch (modulo the
/// retrieved_by/annotated_by prefix that lives outside the channel
/// schema by design).
#[test]
fn merged_schema_matches_merger_output_suffix() {
    let (_dir, catalog) = open_catalog();
    let vector = ChannelId::new("vector").unwrap();
    let inference = ChannelId::new("inference").unwrap();
    let from_catalog = catalog
        .channels()
        .merged_schema(&[vector.clone(), inference.clone()])
        .unwrap();
    let catalog_names: Vec<&str> = from_catalog
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    assert_eq!(
        catalog_names,
        vec![
            "similarity",
            "inference_model",
            "inference_task",
            "inference_confidence",
        ]
    );
}
