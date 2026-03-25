use crate::common;

use jammi_engine::{
    session::JammiSession,
    source::{FileFormat, SourceConnection, SourceType},
};
use tempfile::tempdir;

#[tokio::test]
async fn register_and_query_multiple_formats() {
    let dir = tempdir().unwrap();
    let config = common::test_config(dir.path());
    let session = JammiSession::new(config).await.unwrap();

    // Parquet
    session
        .add_source(
            "patents",
            SourceType::Local,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let results = session
        .sql("SELECT id, title FROM patents.public.patents LIMIT 5")
        .await
        .unwrap();
    assert!(!results.is_empty());
    assert!(results[0].num_rows() <= 5);
    assert!(results[0].schema().field_with_name("id").is_ok());
    assert!(results[0].schema().field_with_name("title").is_ok());

    // CSV
    session
        .add_source(
            "scores",
            SourceType::Local,
            SourceConnection {
                url: Some(common::fixture_url("scores.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let results = session
        .sql("SELECT name, score FROM scores.public.scores WHERE score > 0.6")
        .await
        .unwrap();
    assert_eq!(results[0].num_rows(), 2);
}

#[tokio::test]
async fn query_with_filter_and_order() {
    let dir = tempdir().unwrap();
    let config = common::test_config(dir.path());
    let session = JammiSession::new(config).await.unwrap();

    session
        .add_source(
            "patents",
            SourceType::Local,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let results = session
        .sql("SELECT title, year FROM patents.public.patents WHERE year >= 2022 ORDER BY year DESC")
        .await
        .unwrap();

    let batch = &results[0];
    assert!(batch.num_rows() >= 1);
    let years = batch
        .column_by_name("year")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap();
    for i in 1..years.len() {
        assert!(
            years.value(i - 1) >= years.value(i),
            "Should be DESC ordered"
        );
    }
}

#[tokio::test]
async fn source_persists_across_sessions() {
    let dir = tempdir().unwrap();
    let config = common::test_config(dir.path());

    {
        let session = JammiSession::new(config.clone()).await.unwrap();
        session
            .add_source(
                "persist",
                SourceType::Local,
                SourceConnection {
                    url: Some(common::fixture_url("patents.parquet")),
                    format: Some(FileFormat::Parquet),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
    }

    {
        let session = JammiSession::new(config).await.unwrap();
        let sources = session.catalog().list_sources().unwrap();
        assert!(sources.iter().any(|s| s.source_id == "persist"));
    }
}

#[tokio::test]
async fn source_crud_list_and_remove() {
    let dir = tempdir().unwrap();
    let config = common::test_config(dir.path());
    let session = JammiSession::new(config).await.unwrap();

    session
        .add_source(
            "src_a",
            SourceType::Local,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    session
        .add_source(
            "src_b",
            SourceType::Local,
            SourceConnection {
                url: Some(common::fixture_url("scores.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let sources = session.catalog().list_sources().unwrap();
    let ids: Vec<&str> = sources.iter().map(|s| s.source_id.as_str()).collect();
    assert!(ids.contains(&"src_a"));
    assert!(ids.contains(&"src_b"));

    session.remove_source("src_a").unwrap();
    let sources = session.catalog().list_sources().unwrap();
    assert!(!sources.iter().any(|s| s.source_id == "src_a"));
    assert!(sources.iter().any(|s| s.source_id == "src_b"));

    // Queries against the removed source should fail.
    let err = session.sql("SELECT * FROM src_a.public.patents").await;
    assert!(err.is_err(), "Query against removed source should fail");

    // Queries against the other source should still work.
    let rows = session
        .sql("SELECT COUNT(*) FROM src_b.public.scores")
        .await
        .unwrap();
    assert!(!rows.is_empty());
}

#[tokio::test]
async fn session_respects_config_batch_size() {
    let dir = tempdir().unwrap();
    let mut config = common::test_config(dir.path());
    config.engine.batch_size = 2;

    let session = JammiSession::new(config).await.unwrap();
    session
        .add_source(
            "patents",
            SourceType::Local,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let results = session
        .sql("SELECT * FROM patents.public.patents")
        .await
        .unwrap();
    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 20);
}
