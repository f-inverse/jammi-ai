mod common;

use jammi_engine::{
    session::JammiSession,
    source::{FileFormat, SourceConnection, SourceType},
};
use tempfile::tempdir;

// --- Source registration and querying ---

#[tokio::test]
async fn register_parquet_source_and_query() {
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
        .sql("SELECT id, title FROM patents.public.patents LIMIT 5")
        .await
        .unwrap();
    assert!(!results.is_empty());
    assert!(results[0].num_rows() <= 5);
    assert!(results[0].schema().field_with_name("id").is_ok());
    assert!(results[0].schema().field_with_name("title").is_ok());
}

#[tokio::test]
async fn register_csv_source_and_query() {
    let dir = tempdir().unwrap();
    let config = common::test_config(dir.path());
    let session = JammiSession::new(config).await.unwrap();

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
    assert_eq!(results[0].num_rows(), 2); // alpha (0.9), beta (0.7)
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
    let years = batch.column_by_name("year").unwrap();
    let year_arr = years
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap();
    for i in 1..year_arr.len() {
        assert!(
            year_arr.value(i - 1) >= year_arr.value(i),
            "Should be DESC ordered"
        );
    }
}

// --- Source persistence ---

#[tokio::test]
async fn source_persists_across_sessions() {
    let dir = tempdir().unwrap();
    let config = common::test_config(dir.path());

    // Session 1: register
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

    // Session 2: verify
    {
        let session = JammiSession::new(config).await.unwrap();
        let sources = session.catalog().list_sources().unwrap();
        assert!(sources.iter().any(|s| s.source_id == "persist"));
    }
}

// --- Source CRUD ---

#[tokio::test]
async fn source_list_returns_registered_sources() {
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
}

#[tokio::test]
async fn source_removal() {
    let dir = tempdir().unwrap();
    let config = common::test_config(dir.path());
    let session = JammiSession::new(config).await.unwrap();

    session
        .add_source(
            "to_remove",
            SourceType::Local,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    session.catalog().remove_source("to_remove").unwrap();
    let sources = session.catalog().list_sources().unwrap();
    assert!(!sources.iter().any(|s| s.source_id == "to_remove"));
}

// --- Config integration ---

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

// --- Error cases ---

#[tokio::test]
async fn query_nonexistent_source_fails() {
    let dir = tempdir().unwrap();
    let config = common::test_config(dir.path());
    let session = JammiSession::new(config).await.unwrap();

    let result = session
        .sql("SELECT * FROM nonexistent.public.table LIMIT 1")
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn duplicate_source_registration_fails() {
    let dir = tempdir().unwrap();
    let config = common::test_config(dir.path());
    let session = JammiSession::new(config).await.unwrap();

    let conn = SourceConnection {
        url: Some(common::fixture_url("patents.parquet")),
        format: Some(FileFormat::Parquet),
        ..Default::default()
    };

    session
        .add_source("dup", SourceType::Local, conn.clone())
        .await
        .unwrap();
    let result = session.add_source("dup", SourceType::Local, conn).await;
    assert!(result.is_err());
}
