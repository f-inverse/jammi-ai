//! A verb reads a relation: a registered source, or, under a name no source
//! is registered by, a table the engine holds, named as SQL reads it — a
//! result table `jammi.<name>`, a mutable table `mutable.public.<name>`
//! (#762). A table made with `CREATE TABLE … AS` is embedded, indexed,
//! inferred over and searched without leaving the engine.

use std::collections::BTreeSet;
use std::str::FromStr;
use std::sync::Arc;

use arrow::array::{AsArray, RecordBatch};
use arrow::datatypes::{DataType, Field, Int64Type, Schema};
use jammi_ai::local_session::{
    BuildLexicalIndex, LexicalAnalyzer, LexicalSearchRequest, SearchQuery, SearchRequest,
};
use jammi_ai::session::InferenceSession;
use jammi_ai::Session;
use jammi_datafusion::{ModelSource, ModelTask};
use jammi_db::error::{JammiError, Missing};
use jammi_db::index::SearchMethod;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::mutable::definition::{MutableTableDefinitionBuilder, MutableTableId};
use jammi_db::store::CachePolicy;
use jammi_db::TenantId;
use tempfile::TempDir;

use crate::common;

const RECENT: &str = "CREATE TABLE recent AS \
     SELECT id, title, abstract FROM patents.public.patents WHERE year >= 2022";

fn tiny_bert() -> String {
    format!("local:{}", common::fixture("tiny_bert").display())
}

async fn patents_session(dir: &TempDir, source_id: &str) -> (Arc<InferenceSession>, Session) {
    let engine = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    engine
        .add_source(
            source_id,
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let session = Session::new(Arc::clone(&engine));
    (engine, session)
}

fn indexed(columns: &[&str]) -> BuildLexicalIndex {
    BuildLexicalIndex {
        columns: columns.iter().map(|c| c.to_string()).collect(),
        key_column: "id".into(),
        analyzer: LexicalAnalyzer::English,
    }
}

fn lexical(source_id: &str, text: &str) -> LexicalSearchRequest {
    LexicalSearchRequest {
        source_id: source_id.into(),
        text: text.into(),
        k: 20,
        lexical_table: None,
        filter: None,
        select: Vec::new(),
    }
}

/// Every value of `name`, rendered as text, in row order.
fn values(batches: &[RecordBatch], name: &str) -> Vec<String> {
    batches
        .iter()
        .flat_map(|b| {
            let column = b.column_by_name(name).unwrap_or_else(|| panic!("{name}"));
            let text = arrow::compute::cast(column, &DataType::Utf8).unwrap();
            text.as_string::<i32>()
                .iter()
                .map(|v| v.unwrap().to_string())
                .collect::<Vec<_>>()
        })
        .collect()
}

async fn ids(engine: &InferenceSession, query: &str) -> Vec<i64> {
    let batches = engine.sql(query).await.unwrap();
    batches
        .iter()
        .flat_map(|b| {
            arrow::compute::cast(b.column(0), &DataType::Int64)
                .unwrap()
                .as_primitive::<Int64Type>()
                .values()
                .to_vec()
        })
        .collect()
}

#[tokio::test]
async fn a_table_made_by_sql_is_embedded_indexed_inferred_over_and_searched() {
    let dir = TempDir::new().unwrap();
    let (engine, session) = patents_session(&dir, "patents").await;
    engine.sql(RECENT).await.unwrap();
    let recent = ids(&engine, "SELECT id FROM \"jammi.recent\" ORDER BY id").await;
    assert!(!recent.is_empty() && recent.len() < 20, "{recent:?}");
    let recent_keys: BTreeSet<String> = recent.iter().map(i64::to_string).collect();

    // A lexical index over the table, searched and hydrated from it.
    let index = session
        .build_lexical_index("jammi.recent", &indexed(&["title", "abstract"]))
        .await
        .unwrap();
    assert_eq!(index.row_count, recent.len());
    assert_eq!(index.source_id, "jammi.recent");
    let found = session
        .lexical_search(lexical("jammi.recent", "quantum"))
        .await
        .unwrap();
    let found = values(&found, "id");
    assert!(!found.is_empty());
    assert!(found.iter().all(|id| recent_keys.contains(id)), "{found:?}");

    // Embeddings over the table, searched and hydrated from it.
    let (embeddings, _) = engine
        .generate_text_embeddings(
            "jammi.recent",
            &tiny_bert(),
            &["abstract".to_string()],
            "id",
            CachePolicy::Bypass,
            None,
        )
        .await
        .unwrap();
    assert_eq!(embeddings.row_count, recent.len());
    assert_eq!(embeddings.source_id, "jammi.recent");
    let neighbours = session
        .search(SearchRequest {
            source_id: "jammi.recent".into(),
            query: SearchQuery::RowKey(recent[0].to_string()),
            k: 3,
            embedding_table: None,
            filter: None,
            select: Vec::new(),
            method: SearchMethod::Exact,
        })
        .await
        .unwrap();
    let neighbours = values(&neighbours, "title");
    assert_eq!(neighbours.len(), 3.min(recent.len()));

    // Inference over the table: one output per row of it.
    let (inferred, _) = engine
        .infer(
            "jammi.recent",
            &ModelSource::parse(&tiny_bert()),
            ModelTask::TextEmbedding,
            &["abstract".to_string()],
            "id",
            CachePolicy::Bypass,
        )
        .await
        .unwrap();
    let inferred: BTreeSet<String> = values(&inferred, "_row_id").into_iter().collect();
    assert_eq!(inferred, recent_keys);
}

#[tokio::test]
async fn a_mutable_table_is_a_verbs_input() {
    let dir = TempDir::new().unwrap();
    let (engine, session) = patents_session(&dir, "patents").await;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("body", DataType::Utf8, false),
    ]));
    let notes = MutableTableDefinitionBuilder::new(MutableTableId::new("notes").unwrap(), schema)
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    engine.create_mutable_table(notes).await.unwrap();
    engine
        .sql(
            "INSERT INTO mutable.public.notes (id, body) VALUES \
             (1, 'turbine blade cooling'), (2, 'quantum error correction'), (3, 'solar cells')",
        )
        .await
        .unwrap();

    let index = session
        .build_lexical_index("mutable.public.notes", &indexed(&["body"]))
        .await
        .unwrap();
    assert_eq!(index.row_count, 3);
    let found = session
        .lexical_search(lexical("mutable.public.notes", "quantum"))
        .await
        .unwrap();
    assert_eq!(values(&found, "id"), ["2"]);
}

#[tokio::test]
async fn a_held_table_is_an_input_only_to_a_caller_that_sees_it() {
    let dir = TempDir::new().unwrap();
    let (engine, session) = patents_session(&dir, "patents").await;
    let alice = TenantId::from_str("018f5a0e-c4c8-7e10-9c4f-3b6f7c5a8e01").unwrap();
    let bob = TenantId::from_str("018f5a0e-c4c8-7e10-9c4f-3b6f7c5a8e02").unwrap();
    engine.bind_tenant(alice);
    engine.sql(RECENT).await.unwrap();
    session
        .build_lexical_index("jammi.recent", &indexed(&["title"]))
        .await
        .unwrap();

    // Another tenant's table, a table that does not exist, and a result
    // table named as no SQL reads it are each a missing source.
    engine.bind_tenant(bob);
    for name in [
        "jammi.recent",
        "jammi.nothing",
        "recent",
        "mutable.public.nothing",
    ] {
        let err = session
            .build_lexical_index(name, &indexed(&["title"]))
            .await
            .unwrap_err();
        assert!(
            matches!(&err, JammiError::NotFound(Missing::Source { source_id }) if source_id == name),
            "{name}: {err}"
        );
    }
}

#[tokio::test]
async fn a_source_registered_under_the_name_is_read_before_a_held_table() {
    let dir = TempDir::new().unwrap();
    let (engine, session) = patents_session(&dir, "jammi.recent").await;
    engine
        .sql(
            "CREATE TABLE recent AS SELECT id, title FROM \"jammi.recent\".public.patents \
             WHERE year >= 2022",
        )
        .await
        .unwrap();
    let index = session
        .build_lexical_index("jammi.recent", &indexed(&["title"]))
        .await
        .unwrap();
    assert_eq!(
        index.row_count, 20,
        "the source's 20 patents, not the table's"
    );
}

#[tokio::test]
async fn sql_copies_no_rows_to_a_file() {
    let dir = TempDir::new().unwrap();
    let (engine, _session) = patents_session(&dir, "patents").await;
    let out = TempDir::new().unwrap();
    let file = out.path().join("patents.parquet");
    let err = engine
        .sql(&format!(
            "COPY (SELECT id FROM patents.public.patents) TO '{}' STORED AS PARQUET",
            file.display()
        ))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("read-only"), "{err}");
    assert!(!file.exists());
}
