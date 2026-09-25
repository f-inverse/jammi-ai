//! Lexical (BM25) retrieval: `build_lexical_index` materialises a source's
//! text as a lexical table, and `lexical_search` ranks it and hydrates the
//! source's rows — over the patents fixture, through the embedded [`Session`].

use std::str::FromStr;
use std::sync::Arc;

use arrow::array::{Array, AsArray, RecordBatch};
use arrow::datatypes::{Float32Type, Int64Type};
use jammi_ai::local_session::{BuildLexicalIndex, LexicalAnalyzer, LexicalSearchRequest};
use jammi_ai::session::InferenceSession;
use jammi_ai::Session;
use jammi_db::catalog::result_repo::ResultTableKind;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::manifest::ProducingDescriptor;
use jammi_db::TenantId;
use tempfile::TempDir;

use crate::common;

async fn patents_session(dir: &TempDir) -> (Arc<InferenceSession>, Session) {
    let engine = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    engine
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
        .unwrap();
    let session = Session::new(Arc::clone(&engine));
    (engine, session)
}

fn title_and_abstract() -> BuildLexicalIndex {
    BuildLexicalIndex {
        columns: vec!["title".into(), "abstract".into()],
        key_column: "id".into(),
        analyzer: LexicalAnalyzer::English,
    }
}

fn request(text: &str, k: usize) -> LexicalSearchRequest {
    LexicalSearchRequest {
        source_id: "patents".into(),
        text: text.into(),
        k,
        lexical_table: None,
        filter: None,
        select: Vec::new(),
    }
}

fn int_column(batches: &[RecordBatch], name: &str) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|b| {
            let column = b.column_by_name(name).unwrap_or_else(|| panic!("{name}"));
            arrow::compute::cast(column, &arrow::datatypes::DataType::Int64)
                .unwrap()
                .as_primitive::<Int64Type>()
                .values()
                .to_vec()
        })
        .collect()
}

fn text_column(batches: &[RecordBatch], name: &str) -> Vec<String> {
    batches
        .iter()
        .flat_map(|b| {
            let column = b.column_by_name(name).unwrap_or_else(|| panic!("{name}"));
            let utf8 = arrow::compute::cast(column, &arrow::datatypes::DataType::Utf8).unwrap();
            let strings = utf8.as_string::<i32>();
            (0..strings.len())
                .map(|i| strings.value(i).to_string())
                .collect::<Vec<_>>()
        })
        .collect()
}

/// The patents carrying a word in their title or abstract, by id.
async fn patents_mentioning(engine: &Arc<InferenceSession>, word: &str, extra: &str) -> Vec<i64> {
    let batches = engine
        .sql(&format!(
            "SELECT id FROM patents.public.patents \
             WHERE (lower(title) LIKE '%{word}%' OR lower(abstract) LIKE '%{word}%') {extra} \
             ORDER BY id"
        ))
        .await
        .unwrap();
    int_column(&batches, "id")
}

#[tokio::test]
async fn a_lexical_search_ranks_the_rows_carrying_the_query_terms() {
    let dir = TempDir::new().unwrap();
    let (engine, session) = patents_session(&dir).await;
    let table = session
        .build_lexical_index("patents", &title_and_abstract())
        .await
        .unwrap();
    assert_eq!(table.kind, ResultTableKind::Lexical);
    assert_eq!(table.row_count, 20);

    let found = session
        .lexical_search(request("quantum", 20))
        .await
        .unwrap();
    let mut ids = int_column(&found, "id");
    let ranks = int_column(&found, "bm25_rank");
    assert_eq!(ranks, (0..ids.len() as i64).collect::<Vec<_>>());
    let scores: Vec<f32> = found
        .iter()
        .flat_map(|b| {
            b.column_by_name("bm25_score")
                .unwrap()
                .as_primitive::<Float32Type>()
                .values()
                .to_vec()
        })
        .collect();
    assert!(scores.windows(2).all(|w| w[0] >= w[1]), "{scores:?}");
    // Hydrated from the source, attributed to the `bm25` channel.
    assert!(text_column(&found, "title").iter().all(|t| !t.is_empty()));
    let retrieved_by = found[0]
        .column_by_name("retrieved_by")
        .unwrap()
        .as_list::<i32>();
    assert_eq!(
        text_column(
            &[RecordBatch::try_from_iter([("c", retrieved_by.value(0))]).unwrap()],
            "c"
        ),
        vec!["bm25"]
    );
    // Exactly the patents mentioning the word, and none that do not.
    ids.sort_unstable();
    assert_eq!(ids, patents_mentioning(&engine, "quantum", "").await);
}

#[tokio::test]
async fn a_filtered_lexical_search_returns_the_k_best_passing_rows() {
    let dir = TempDir::new().unwrap();
    let (engine, session) = patents_session(&dir).await;
    session
        .build_lexical_index("patents", &title_and_abstract())
        .await
        .unwrap();
    let passing = patents_mentioning(&engine, "quantum", "AND year >= 2022").await;
    assert!(passing.len() >= 2, "the fixture has recent quantum patents");

    let unfiltered = int_column(
        &session
            .lexical_search(request("quantum", 20))
            .await
            .unwrap(),
        "id",
    );
    let found = session
        .lexical_search(LexicalSearchRequest {
            filter: Some("year >= 2022".into()),
            select: vec!["id".into(), "year".into(), "bm25_rank".into()],
            ..request("quantum", 2)
        })
        .await
        .unwrap();
    let ids = int_column(&found, "id");
    // The two best-ranked passing rows, in the unfiltered ranking's order.
    let expected: Vec<i64> = unfiltered
        .into_iter()
        .filter(|id| passing.contains(id))
        .take(2)
        .collect();
    assert_eq!(ids, expected);
    let names: Vec<String> = found[0]
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    // The selection, then the evidence: the kept `bm25_rank` once, and the
    // `bm25_score` the selection dropped as null.
    assert_eq!(
        names,
        [
            "id",
            "year",
            "retrieved_by",
            "annotated_by",
            "bm25_score",
            "bm25_rank"
        ]
    );
    assert_eq!(int_column(&found, "bm25_rank").len(), 2);
    assert_eq!(
        found[0].column_by_name("bm25_score").unwrap().null_count(),
        found[0].num_rows()
    );
}

#[tokio::test]
async fn a_lexical_index_records_its_definition_and_recomputes() {
    let dir = TempDir::new().unwrap();
    let (_engine, session) = patents_session(&dir).await;
    let params = BuildLexicalIndex {
        analyzer: LexicalAnalyzer::Raw,
        ..title_and_abstract()
    };
    let table = session
        .build_lexical_index("patents", &params)
        .await
        .unwrap();

    let manifest = session.describe_table(&table.table_name).await.unwrap();
    assert_eq!(
        manifest.descriptor,
        ProducingDescriptor::LexicalIndex {
            source_id: "patents".into(),
            key_column: "id".into(),
            text_columns: vec!["title".into(), "abstract".into()],
            analyzer: LexicalAnalyzer::Raw,
        }
    );

    let report = session
        .recompute(
            &table.table_name,
            jammi_ai::pipeline::recompute::Cascade::ReportOnly,
        )
        .await
        .unwrap();
    let recomputed = &report.recomputed[0].recomputed;
    assert_ne!(recomputed, &table.table_name);
    let replayed = session.describe_table(recomputed).await.unwrap();
    assert_eq!(replayed.definition_hash, manifest.definition_hash);
}

#[tokio::test]
async fn a_named_table_must_be_a_lexical_index() {
    let dir = TempDir::new().unwrap();
    let (_engine, session) = patents_session(&dir).await;
    session
        .build_lexical_index("patents", &title_and_abstract())
        .await
        .unwrap();
    let asof_like = session
        .sql("CREATE TABLE recent AS SELECT * FROM patents.public.patents WHERE year > 2021")
        .await;
    assert!(asof_like.is_ok());
    let named = session
        .lexical_search(LexicalSearchRequest {
            lexical_table: Some("recent".into()),
            ..request("quantum", 3)
        })
        .await
        .unwrap_err();
    assert!(named.to_string().contains("not a lexical index"), "{named}");
}

#[tokio::test]
async fn a_lexical_index_resolves_only_for_its_tenant() {
    let dir = TempDir::new().unwrap();
    let (engine, session) = patents_session(&dir).await;
    let alice = TenantId::from_str("018f5a0e-c4c8-7e10-9c4f-3b6f7c5a8e01").unwrap();
    let bob = TenantId::from_str("018f5a0e-c4c8-7e10-9c4f-3b6f7c5a8e02").unwrap();

    engine.bind_tenant(alice);
    session
        .build_lexical_index("patents", &title_and_abstract())
        .await
        .unwrap();
    assert!(!session
        .lexical_search(request("quantum", 3))
        .await
        .unwrap()
        .is_empty());

    engine.bind_tenant(bob);
    let refused = session
        .lexical_search(request("quantum", 3))
        .await
        .unwrap_err();
    assert!(
        refused.to_string().contains("No ready lexical index"),
        "{refused}"
    );
}
