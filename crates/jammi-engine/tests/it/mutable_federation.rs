//! Phase 2 SPEC-02 §11 exit criteria #1 and #4 — federation between a
//! mutable companion table and a Parquet result table.
//!
//! Coverage:
//! - 1000-row `INSERT INTO mutable.public.<id>` via `session.sql(...)` then
//!   a `JOIN` against a Parquet-backed source on a shared key, asserted
//!   over both backends.
//! - Smaller correctness case: mutable `(id, label)` joined with Parquet
//!   `(id, vector)` returns exactly one row per matching id.
//!
//! Parameterised over [`BackendKind`] via `test_case` + `cfg_attr` so the
//! Postgres lane only generates when the `live-postgres-tests` feature is on
//! AND `JAMMI_TEST_PG_URL` is set. The Postgres variant skips at runtime
//! when the env var is unset; the SQLite variant always runs.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use jammi_engine::catalog::backend::BackendKind;
use jammi_engine::session::JammiSession;
use jammi_engine::source::{FileFormat, SourceConnection, SourceType};
use jammi_engine::store::mutable::definition::{MutableTableDefinitionBuilder, MutableTableId};
use jammi_test_utils::make_test_session;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use tempfile::tempdir;
use test_case::test_case;

/// Backend-unique mutable-table identifier. SQLite per-tempdir tests don't
/// strictly need this, but Postgres runs share one schema; the suffix avoids
/// `relation "<name>" already exists` between parameterized variants.
fn unique_id(prefix: &str) -> MutableTableId {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let epoch_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    MutableTableId::new(format!("{prefix}_{epoch_ns:x}_{n:x}")).unwrap()
}

/// Write a Parquet file with `(id Int64, payload Utf8)` rows at `path`.
async fn write_payload_parquet(path: &std::path::Path, n_rows: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let ids = Int64Array::from_iter_values(0..n_rows as i64);
    let payloads: Vec<String> = (0..n_rows).map(|i| format!("payload-{i}")).collect();
    let payloads = StringArray::from_iter_values(payloads);
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(ids) as ArrayRef, Arc::new(payloads) as ArrayRef],
    )
    .unwrap();
    let file = std::fs::File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Write a Parquet file with `(id Int64, vector FixedSizeList<Float32, 3>)` rows.
async fn write_vector_parquet(path: &std::path::Path, ids: &[i64]) {
    let item_field = Arc::new(Field::new("item", DataType::Float32, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::clone(&item_field), 3),
            false,
        ),
    ]));
    let id_array = Int64Array::from(ids.to_vec());
    let flat: Vec<f32> = ids
        .iter()
        .flat_map(|i| [*i as f32, *i as f32 + 0.5, *i as f32 + 1.5])
        .collect();
    let values = Arc::new(Float32Array::from(flat)) as ArrayRef;
    let list = FixedSizeListArray::new(item_field, 3, values, None);
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(id_array) as ArrayRef, Arc::new(list) as ArrayRef],
    )
    .unwrap();
    let file = std::fs::File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

async fn register_mutable_5col(session: &JammiSession, id: &MutableTableId) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Utf8, false),
        Field::new("c", DataType::Float64, false),
        Field::new("d", DataType::Boolean, false),
    ]));
    let def = MutableTableDefinitionBuilder::new(id.clone(), schema)
        .primary_key(vec!["key".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();
}

async fn register_label_table(session: &JammiSession, id: &MutableTableId) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("label", DataType::Utf8, false),
    ]));
    let def = MutableTableDefinitionBuilder::new(id.clone(), schema)
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mutable_thousand_row_insert_then_federation_join(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let Some(session) = make_test_session(backend, dir.path()).await else {
        eprintln!("skipping {backend:?}: JAMMI_TEST_PG_URL unset");
        return;
    };

    let id = unique_id("federation_5col");
    register_mutable_5col(&session, &id).await;

    // 1000-row multi-VALUES INSERT through the SQL surface.
    let mut values = String::new();
    for i in 0..1000_i64 {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!(
            "({i}, {a}, 'b{i}', {c}, {d})",
            a = i * 2,
            c = i as f64 * 0.5,
            d = if i % 2 == 0 { "true" } else { "false" },
        ));
    }
    let stmt = format!(
        "INSERT INTO mutable.public.{name} (key, a, b, c, d) VALUES {values}",
        name = id.as_str(),
    );
    session.sql(&stmt).await.unwrap();

    // Parquet result table with 100 rows whose ids overlap the mutable table.
    let pq_path = dir.path().join("payloads.parquet");
    write_payload_parquet(&pq_path, 100).await;
    session
        .add_source(
            "payloads",
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", pq_path.display())),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let query = format!(
        "SELECT m.a, s.payload \
         FROM mutable.public.{name} m \
         JOIN payloads.public.payloads s ON m.key = s.id \
         ORDER BY m.key",
        name = id.as_str(),
    );
    let batches = session.sql(&query).await.unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total, 100,
        "federation join should yield one row per parquet id",
    );

    let merged = arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap();
    let payload_strings = string_column_to_vec(&merged, "payload");
    assert_eq!(payload_strings[0], "payload-0");
    assert_eq!(payload_strings[99], "payload-99");
    let a = merged
        .column_by_name("a")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(a.value(0), 0);
    assert_eq!(a.value(99), 198);
}

/// Pull a `Utf8`-shaped column out of `batch` as `Vec<String>`, accepting
/// either `StringArray` or `StringViewArray` (Arrow 57's parquet reader emits
/// `StringViewArray` for `Utf8` columns under DataFusion 52). Fails loudly if
/// the column is neither shape.
fn string_column_to_vec(batch: &RecordBatch, name: &str) -> Vec<String> {
    use arrow::array::StringViewArray;
    let col = batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("column {name} missing in {:?}", batch.schema()));
    if let Some(sa) = col.as_any().downcast_ref::<StringArray>() {
        return (0..sa.len()).map(|i| sa.value(i).to_string()).collect();
    }
    if let Some(sv) = col.as_any().downcast_ref::<StringViewArray>() {
        return (0..sv.len()).map(|i| sv.value(i).to_string()).collect();
    }
    panic!(
        "column {name} is neither StringArray nor StringViewArray; got {:?}",
        col.data_type()
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mutable_join_parquet_returns_one_row_per_matching_id(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let Some(session) = make_test_session(backend, dir.path()).await else {
        eprintln!("skipping {backend:?}: JAMMI_TEST_PG_URL unset");
        return;
    };

    let id = unique_id("federation_label");
    register_label_table(&session, &id).await;

    // 5 mutable rows (ids 1..=5).
    let labels = (1..=5_i64)
        .map(|i| format!("({i}, 'label-{i}')"))
        .collect::<Vec<_>>()
        .join(", ");
    session
        .sql(&format!(
            "INSERT INTO mutable.public.{name} (id, label) VALUES {labels}",
            name = id.as_str(),
        ))
        .await
        .unwrap();

    // Parquet with 5 rows (ids 2..=6); intersection is ids 2..=5 → 4 rows.
    let pq_path = dir.path().join("vectors.parquet");
    write_vector_parquet(&pq_path, &[2, 3, 4, 5, 6]).await;
    session
        .add_source(
            "vectors",
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", pq_path.display())),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let query = format!(
        "SELECT m.label, v.vector \
         FROM mutable.public.{name} m \
         JOIN vectors.public.vectors v USING (id) \
         ORDER BY m.id",
        name = id.as_str(),
    );
    let batches = session.sql(&query).await.unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total, 4,
        "intersection of mutable.id ∩ parquet.id = {{2..=5}}"
    );

    let merged = arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap();
    let labels = string_column_to_vec(&merged, "label");
    assert_eq!(labels[0], "label-2");
    assert_eq!(labels[3], "label-5");
    let vector = merged
        .column_by_name("vector")
        .unwrap()
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    assert_eq!(vector.len(), 4);
    assert_eq!(vector.value_length(), 3);
}
