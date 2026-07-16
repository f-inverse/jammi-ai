//! `ResultStore::materialize_computed_embedding_table` — the generic verb a
//! consumer that computes embeddings in-process (a perturbation, a
//! reconditioning pass, a migration off another store) reaches for to publish
//! them as a searchable, joinable embedding table with its own provenance.
//!
//! Covers: the happy path (ready table, findable, searchable, caller params +
//! auto-folded `content_digest` recorded, `derived_from` lineage carried
//! through both the catalog FK and the manifest's `ResultDigest` input
//! anchor); the fail-loud invariant checks (width mismatch, zero/non-finite
//! norm, a caller-supplied reserved `content_digest` key); and the no-collision
//! property (two materializations sharing every scalar param but different
//! vectors get different `DefinitionHash`es).

use std::collections::BTreeMap;
use std::sync::Arc;

use datafusion::prelude::SessionContext;
use jammi_db::catalog::backend::BackendKind;
use jammi_db::catalog::backend_postgres::PostgresBackend;
use jammi_db::catalog::backend_sqlite::SqliteBackend;
use jammi_db::catalog::Catalog;
use jammi_db::config::AnnIndexConfig;
use jammi_db::error::JammiError;
use jammi_db::model_task::ModelTask;
use jammi_db::store::manifest::{
    AnchorKind, ComputeDevice, InputAnchor, MaterializationEnv, ProducingDescriptor,
};
use jammi_db::store::{ComputedEmbeddingProvenance, EmbeddingTableSpec, ResultStore};
use jammi_test_utils::unique_suffix;
use tempfile::tempdir;
use test_case::test_case;

const DIMS: usize = 3;

/// Build a catalog on `backend`, running migrations. Returns `None` for the
/// Postgres arm when `JAMMI_TEST_PG_URL` is unset, so callers skip (never
/// `#[ignore]`) exactly like [`jammi_test_utils::make_test_session`].
async fn fresh_catalog(backend: BackendKind, dir: &std::path::Path) -> Option<Arc<Catalog>> {
    let backend_impl = match backend {
        BackendKind::Sqlite => {
            let b = SqliteBackend::open(&dir.join("catalog.db")).await.unwrap();
            jammi_db::catalog::backend::BackendImpl::Sqlite(b)
        }
        BackendKind::Postgres => {
            let url = jammi_test_utils::pg_url_for_tests()?;
            let pg = PostgresBackend::open_with_options(&url, 8, None)
                .await
                .unwrap();
            jammi_db::catalog::backend::BackendImpl::Postgres(pg)
        }
    };
    backend_impl.migrate().await.unwrap();
    Some(Arc::new(Catalog::from_backend(backend_impl)))
}

/// Fetch a backend-parameterized catalog, skipping the test (with a warning)
/// when the Postgres arm has no `JAMMI_TEST_PG_URL`.
macro_rules! fresh_catalog_or_skip {
    ($backend:expr, $dir:expr) => {
        match fresh_catalog($backend, $dir.path()).await {
            Some(c) => c,
            None => {
                eprintln!("skipping {:?}: JAMMI_TEST_PG_URL unset", $backend);
                return;
            }
        }
    };
}

fn store(dir: &std::path::Path, catalog: Arc<Catalog>) -> ResultStore {
    ResultStore::new(dir, catalog, AnnIndexConfig::default()).unwrap()
}

fn env() -> MaterializationEnv {
    MaterializationEnv::new(ComputeDevice::Cpu, vec![])
}

fn provenance(
    params: BTreeMap<String, String>,
    inputs: Vec<InputAnchor>,
) -> ComputedEmbeddingProvenance {
    ComputedEmbeddingProvenance {
        producer_id: "custom_producer".to_string(),
        params,
        env: env(),
        inputs,
    }
}

fn spec<'a>(source_id: &'a str, derived_from: Option<&'a str>) -> EmbeddingTableSpec<'a> {
    EmbeddingTableSpec {
        source_id,
        model_id: "custom-model",
        derived_from,
        dimensions: DIMS,
        key_column: Some("_row_id"),
        text_columns: None,
    }
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn happy_path_lands_a_ready_searchable_table_with_provenance_and_lineage(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    // Backend-unique source ids: the Postgres lane shares one database across
    // the whole test run, and `find_result_tables` below asserts an EXACT
    // count for `computed_source_id` — a fixed literal would accumulate rows
    // across sibling tests and repeated runs, breaking that count.
    let suffix = unique_suffix();
    let source_source_id = format!("source_docs_{suffix}");
    let computed_source_id = format!("computed_{suffix}");

    // A source table to derive from — its content digest anchors the
    // `ResultDigest` input, and its table name anchors the catalog FK.
    let source_rows = vec![("s1".to_string(), vec![1.0_f32, 0.0, 0.0])];
    let source = store
        .materialize_embedding_table(
            &ctx,
            spec(&source_source_id, None),
            &source_rows,
            jammi_db::store::manifest::Materialization::new(
                &ProducingDescriptor::External {
                    producer_id: "seed".to_string(),
                    params: BTreeMap::new(),
                },
                &env(),
                vec![],
            ),
        )
        .await
        .unwrap();

    let result_digest_anchor = store.result_digest_anchor(&source).await.unwrap();
    assert_eq!(result_digest_anchor.kind, AnchorKind::ResultDigest);

    let mut params = BTreeMap::new();
    params.insert("perturbation_kind".to_string(), "gaussian".to_string());

    let rows = vec![
        ("r1".to_string(), vec![1.0_f32, 0.0, 0.0]),
        ("r2".to_string(), vec![0.0_f32, 2.0, 0.0]),
    ];

    let record = store
        .materialize_computed_embedding_table(
            &ctx,
            spec(&computed_source_id, Some(&source.table_name)),
            &rows,
            provenance(params, vec![result_digest_anchor.clone()]),
        )
        .await
        .unwrap();

    // Ready + findable.
    assert_eq!(record.status, "ready");
    assert_eq!(record.row_count, 2);
    let found = catalog
        .find_result_tables(&computed_source_id, Some(ModelTask::TextEmbedding), None)
        .await
        .unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].table_name, record.table_name);

    // The catalog FK lineage is recorded.
    assert_eq!(
        record.derived_from.as_deref(),
        Some(source.table_name.as_str())
    );

    // The manifest's input anchors carry the `ResultDigest` lineage anchor.
    let url = jammi_db::storage::StorageUrl::parse(&record.parquet_path).unwrap();
    let manifest = store
        .read_materialization_manifest(&url)
        .await
        .unwrap()
        .expect("sidecar present");
    assert_eq!(manifest.input_anchors, vec![result_digest_anchor]);

    // The descriptor is `External` with the caller's params plus the
    // auto-folded, non-empty `content_digest`.
    match &manifest.descriptor {
        ProducingDescriptor::External {
            producer_id,
            params,
        } => {
            assert_eq!(producer_id, "custom_producer");
            assert_eq!(
                params.get("perturbation_kind"),
                Some(&"gaussian".to_string())
            );
            let digest = params
                .get("content_digest")
                .expect("content_digest auto-folded");
            assert!(!digest.is_empty());
        }
        other => panic!("expected External descriptor, got {other:?}"),
    }

    // Searchable through the same cosine-search path every embedding table
    // uses: the query is the (already-unit) first row, so it is its own
    // nearest neighbour at distance ~0.
    let top1 = store
        .search_vectors(&ctx, &record, &[1.0, 0.0, 0.0], 1)
        .await
        .unwrap();
    assert_eq!(top1.len(), 1);
    assert_eq!(top1[0].0, "r1");
    assert!(
        top1[0].1 < 1e-5,
        "expected ~0 cosine distance, got {}",
        top1[0].1
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn width_mismatch_is_a_schema_error(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let rows = vec![("r1".to_string(), vec![1.0_f32, 0.0])]; // width 2, spec wants 3
    let err = store
        .materialize_computed_embedding_table(
            &ctx,
            spec("computed", None),
            &rows,
            provenance(BTreeMap::new(), vec![]),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(err, JammiError::Schema { .. }),
        "expected Schema error, got {err:?}"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn zero_norm_is_a_schema_error(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let rows = vec![("r1".to_string(), vec![0.0_f32, 0.0, 0.0])];
    let err = store
        .materialize_computed_embedding_table(
            &ctx,
            spec("computed", None),
            &rows,
            provenance(BTreeMap::new(), vec![]),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(err, JammiError::Schema { .. }),
        "expected Schema error, got {err:?}"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn non_finite_norm_is_a_schema_error(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let rows = vec![("r1".to_string(), vec![f32::INFINITY, 0.0, 0.0])];
    let err = store
        .materialize_computed_embedding_table(
            &ctx,
            spec("computed", None),
            &rows,
            provenance(BTreeMap::new(), vec![]),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(err, JammiError::Schema { .. }),
        "expected Schema error, got {err:?}"
    );

    // A NaN norm (e.g. from a NaN component) is equally rejected.
    let rows_nan = vec![("r1".to_string(), vec![f32::NAN, 0.0, 0.0])];
    let err_nan = store
        .materialize_computed_embedding_table(
            &ctx,
            spec("computed", None),
            &rows_nan,
            provenance(BTreeMap::new(), vec![]),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(err_nan, JammiError::Schema { .. }),
        "expected Schema error, got {err_nan:?}"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn caller_supplied_reserved_content_digest_key_is_a_hard_error(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let mut params = BTreeMap::new();
    params.insert("content_digest".to_string(), "caller-supplied".to_string());

    let rows = vec![("r1".to_string(), vec![1.0_f32, 0.0, 0.0])];
    let err = store
        .materialize_computed_embedding_table(
            &ctx,
            spec("computed", None),
            &rows,
            provenance(params, vec![]),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(err, JammiError::Schema { .. }),
        "expected a hard Schema error rejecting the reserved key, got {err:?}"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn identical_scalar_params_but_different_vectors_never_collide_on_one_hash(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let catalog = fresh_catalog_or_skip!(backend, dir);
    let store = store(dir.path(), Arc::clone(&catalog));
    let ctx = SessionContext::new();

    let mut params = BTreeMap::new();
    params.insert("shared".to_string(), "scalar".to_string());

    let rows_a = vec![("r1".to_string(), vec![1.0_f32, 0.0, 0.0])];
    let rows_b = vec![("r1".to_string(), vec![0.0_f32, 1.0, 0.0])];

    let record_a = store
        .materialize_computed_embedding_table(
            &ctx,
            spec("computed_a", None),
            &rows_a,
            provenance(params.clone(), vec![]),
        )
        .await
        .unwrap();
    let record_b = store
        .materialize_computed_embedding_table(
            &ctx,
            spec("computed_b", None),
            &rows_b,
            provenance(params, vec![]),
        )
        .await
        .unwrap();

    assert_ne!(
        record_a.definition_hash, record_b.definition_hash,
        "different vectors under identical scalar params must not collide on one DefinitionHash"
    );
}
