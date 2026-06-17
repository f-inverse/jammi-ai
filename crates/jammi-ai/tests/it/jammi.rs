//! The SDK front door, embedded path. `Jammi::open(Target::Local(config))` must
//! yield a working in-process [`Session`] — the one-call "use the SDK, run any
//! shape" entry point. This drives the real source → generate-embeddings →
//! search pipeline over the patents fixture and the tiny BERT cookbook model
//! through the `Session` the factory returns, proving the front door produces a
//! live embedded session, not just a constructed value.

use arrow::array::StringArray;
use jammi_ai::{Jammi, Modality, SearchQuery, SearchRequest, Session, Target};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use tempfile::TempDir;

use crate::common;

fn tiny_bert() -> String {
    "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap()
}

/// `Jammi::open(Target::Local(_))` returns an embedded [`Session`] that drives
/// the full embed → search pipeline end to end against the patents corpus.
#[tokio::test]
async fn open_local_yields_a_working_embedded_session() {
    let dir = TempDir::new().unwrap();
    let session: Session = Jammi::open(Target::Local(common::test_config(dir.path())))
        .await
        .expect("open local session");

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
        .expect("add_source through the opened session");

    let record = session
        .generate_embeddings(
            "patents",
            &tiny_bert(),
            &["abstract".to_string()],
            "id",
            Modality::Text,
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .expect("generate_embeddings through the opened session")
        .0;
    assert_eq!(record.status, "ready");
    assert!(record.row_count > 0, "patents corpus embeds rows");

    let hits = session
        .search(SearchRequest {
            source_id: "patents".to_string(),
            query: SearchQuery::Vector(vec![0.5_f32; 32]),
            k: 5,
            embedding_table: None,
            filter: None,
            select: Vec::new(),
        })
        .await
        .expect("search through the opened session");
    assert!(!hits.is_empty(), "the opened session returns search hits");
    assert!(
        hits.iter().any(|b| b
            .column_by_name("_row_id")
            .is_some_and(|c| { c.as_any().downcast_ref::<StringArray>().is_some() })),
        "hits carry the _row_id provenance column the search verb yields"
    );
}
