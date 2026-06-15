use arrow::datatypes::DataType;
use jammi_db::catalog::channel_repo::{
    ChannelCatalogError, ChannelColumn, ChannelColumnType, ChannelSpec,
};
use jammi_db::catalog::Catalog;
use jammi_db::error::JammiError;
use jammi_db::ChannelId;
use tempfile::tempdir;

async fn open_catalog() -> (tempfile::TempDir, Catalog) {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).await.unwrap();
    (dir, catalog)
}

#[tokio::test]
async fn migration_006_seeds_vector_and_inference_with_exact_columns() {
    let (_dir, catalog) = open_catalog().await;
    let channels = catalog.channels().list().await.unwrap();

    let vector = channels
        .iter()
        .find(|c| c.id.as_str() == "vector")
        .expect("vector channel must be seeded");
    assert_eq!(vector.priority, 1);
    assert_eq!(vector.columns.len(), 1);
    assert_eq!(vector.columns[0].name, "similarity");
    assert_eq!(vector.columns[0].data_type, ChannelColumnType::Float32);

    let inference = channels
        .iter()
        .find(|c| c.id.as_str() == "inference")
        .expect("inference channel must be seeded");
    assert_eq!(inference.priority, 2);
    assert_eq!(inference.columns.len(), 3);
    let names: Vec<&str> = inference.columns.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["inference_model", "inference_task", "inference_confidence"]
    );
}

#[tokio::test]
async fn declared_columns_appear_in_merged_schema() {
    let (_dir, catalog) = open_catalog().await;
    let scored_by = ChannelId::new("scored_by").unwrap();
    catalog
        .channels()
        .register(&ChannelSpec {
            id: scored_by.clone(),
            priority: 3,
            columns: vec![
                ChannelColumn {
                    name: "ranker".into(),
                    data_type: ChannelColumnType::Utf8,
                },
                ChannelColumn {
                    name: "rank_score".into(),
                    data_type: ChannelColumnType::Float32,
                },
            ],
        })
        .await
        .unwrap();

    let schema = catalog
        .channels()
        .merged_schema(&[ChannelId::new("vector").unwrap(), scored_by])
        .await
        .unwrap();

    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, vec!["similarity", "ranker", "rank_score"]);
    assert_eq!(schema.field(0).data_type(), &DataType::Float32);
    assert_eq!(schema.field(1).data_type(), &DataType::Utf8);
    assert_eq!(schema.field(2).data_type(), &DataType::Float32);
    for field in schema.fields() {
        assert!(
            field.is_nullable(),
            "declared columns are nullable by contract"
        );
    }
}

#[tokio::test]
async fn channel_column_order_is_stable_across_catalog_reads() {
    let (_dir, catalog) = open_catalog().await;
    let first = catalog
        .channels()
        .get(&ChannelId::new("inference").unwrap())
        .await
        .unwrap()
        .unwrap();
    let second = catalog
        .channels()
        .get(&ChannelId::new("inference").unwrap())
        .await
        .unwrap()
        .unwrap();
    let first_names: Vec<&str> = first.columns.iter().map(|c| c.name.as_str()).collect();
    let second_names: Vec<&str> = second.columns.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(first_names, second_names);
}

#[tokio::test]
async fn add_columns_then_merged_schema_includes_new_column() {
    let (_dir, catalog) = open_catalog().await;
    let id = ChannelId::new("scored_by").unwrap();
    catalog
        .channels()
        .register(&ChannelSpec {
            id: id.clone(),
            priority: 3,
            columns: vec![ChannelColumn {
                name: "ranker".into(),
                data_type: ChannelColumnType::Utf8,
            }],
        })
        .await
        .unwrap();
    catalog
        .channels()
        .add_columns(
            &id,
            &[ChannelColumn {
                name: "rank_score".into(),
                data_type: ChannelColumnType::Float32,
            }],
        )
        .await
        .unwrap();

    let schema = catalog.channels().merged_schema(&[id]).await.unwrap();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, vec!["ranker", "rank_score"]);
}

/// SPEC-01 §9 — `register` must reject a channel id that's already in the
/// catalog with `ChannelCatalog(AlreadyExists(...))`.
#[tokio::test]
async fn register_rejects_duplicate_channel_id() {
    let (_dir, catalog) = open_catalog().await;
    let spec = ChannelSpec {
        id: ChannelId::new("scored_by").unwrap(),
        priority: 3,
        columns: vec![ChannelColumn {
            name: "ranker".into(),
            data_type: ChannelColumnType::Utf8,
        }],
    };
    catalog.channels().register(&spec).await.unwrap();

    let err = catalog.channels().register(&spec).await.unwrap_err();
    match err {
        JammiError::ChannelCatalog(ChannelCatalogError::AlreadyExists(channel)) => assert_eq!(
            channel, "scored_by",
            "expected 'already exists' for 'scored_by'"
        ),
        other => panic!("expected ChannelCatalog(AlreadyExists), got {other:?}"),
    }
}

/// SPEC-01 §9 — `add_columns` must reject a redeclaration of an existing
/// column with a different `ChannelColumnType`. The production message
/// names both the column and the would-be new type so a Python caller
/// learning the API can see exactly what failed.
#[tokio::test]
async fn add_columns_rejects_int32_retype_of_utf8_column() {
    let (_dir, catalog) = open_catalog().await;
    let id = ChannelId::new("scored_by").unwrap();
    catalog
        .channels()
        .register(&ChannelSpec {
            id: id.clone(),
            priority: 3,
            columns: vec![ChannelColumn {
                name: "ranker".into(),
                data_type: ChannelColumnType::Utf8,
            }],
        })
        .await
        .unwrap();

    let err = catalog
        .channels()
        .add_columns(
            &id,
            &[ChannelColumn {
                name: "ranker".into(),
                data_type: ChannelColumnType::Int32,
            }],
        )
        .await
        .unwrap_err();
    // The Display text is load-bearing — the message names both the column and
    // the would-be new type so a Python caller sees exactly what failed.
    assert!(
        err.to_string().contains("cannot redeclare as Int32") && err.to_string().contains("ranker"),
        "expected 'cannot redeclare as Int32' for 'ranker'; got: {err}"
    );
    match err {
        JammiError::ChannelCatalog(ChannelCatalogError::ColumnConflict {
            column,
            requested,
            ..
        }) => {
            assert_eq!(column, "ranker");
            assert_eq!(requested, ChannelColumnType::Int32);
        }
        other => panic!("expected ChannelCatalog(ColumnConflict), got {other:?}"),
    }
}
