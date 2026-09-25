//! BM25 lexical index over a lexical table's text.
//!
//! A [`LexicalIndex`] is a `tantivy` inverted index over the `(_row_id, text)`
//! rows of a [`ResultTableKind::Lexical`] table, answering
//! [`LexicalIndex::search`] with each hit's BM25 score and rank — the lexical
//! peer of the ANN index's cosine.
//!
//! ## Lifecycle
//! The index is derived state: a table version's rows fully determine it, so
//! it is built in memory from those rows the first time a process searches
//! that version ([`LexicalIndexes`]) and never persisted. A new version of the
//! table is a new content digest, hence a new index.
//!
//! ## Tenant scope
//! The index applies no row-level filter — isolation is table-level, exactly
//! as the ANN search path: the caller resolves the table through the
//! tenant-scoped catalog and the index holds only that table's rows.
//!
//! ## Query
//! A query is its words: the text runs through the index's own
//! [`LexicalAnalyzer`] and each resulting term is one disjunctive clause, so a
//! row matching any term scores and one matching more, or rarer, terms scores
//! higher. No query syntax is interpreted — a colon, a quote or a minus sign is
//! text, never an operator.
//!
//! [`ResultTableKind::Lexical`]: jammi_db::catalog::result_repo::ResultTableKind::Lexical

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use jammi_db::error::{JammiError, Result};
use jammi_db::index::LexicalAnalyzer;
use jammi_db::store::manifest::InputAnchor;
use tantivy::collector::TopDocs;
use tantivy::query::BooleanQuery;
use tantivy::schema::{IndexRecordOption, Schema, Value, STORED, STRING, TEXT};
use tantivy::tokenizer::{LowerCaser, RemoveLongFilter, SimpleTokenizer, Stemmer, TextAnalyzer};
use tantivy::{doc, Index, IndexWriter, TantivyDocument, Term};

/// Heap budget for the in-memory index writer (tantivy's minimum is 3 MiB; a
/// modest arena keeps a build's footprint bounded).
const WRITER_HEAP_BYTES: usize = 15_000_000;

/// The tokenizer registry name an analyzer registers under. Tantivy keys the
/// body field's tokenizer by name, so the build and the query agree by
/// referencing the same constant.
fn registry_name(analyzer: LexicalAnalyzer) -> &'static str {
    match analyzer {
        LexicalAnalyzer::English => "jammi_english",
        LexicalAnalyzer::Raw => "jammi_raw",
    }
}

/// The tantivy [`TextAnalyzer`] an analyzer denotes.
///
/// Each `.filter(..)` returns a distinctly-typed builder, so the stemmed and
/// unstemmed pipelines `.build()` in their own arms rather than sharing a tail.
fn text_analyzer(analyzer: LexicalAnalyzer) -> TextAnalyzer {
    let base = TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(RemoveLongFilter::limit(40))
        .filter(LowerCaser);
    match analyzer {
        LexicalAnalyzer::English => base
            .filter(Stemmer::new(tantivy::tokenizer::Language::English))
            .build(),
        LexicalAnalyzer::Raw => base.build(),
    }
}

/// One lexical hit: the table row's `_row_id`, its raw BM25 score, and its
/// 0-based rank in the returned list.
#[derive(Debug, Clone, PartialEq)]
pub struct LexicalHit {
    pub row_id: String,
    pub bm25_score: f32,
    pub rank: usize,
}

/// A BM25 inverted index over one table's text, keyed by `_row_id`.
pub struct LexicalIndex {
    index: Index,
    analyzer: LexicalAnalyzer,
    row_id_field: tantivy::schema::Field,
    body_field: tantivy::schema::Field,
    doc_count: u64,
}

impl LexicalIndex {
    /// Build a lexical index over `rows`, each a `(row_id, text)` pair, under
    /// `analyzer`.
    pub fn build<I, R, T>(rows: I, analyzer: LexicalAnalyzer) -> Result<Self>
    where
        I: IntoIterator<Item = (R, T)>,
        R: AsRef<str>,
        T: AsRef<str>,
    {
        let mut schema_builder = Schema::builder();
        // `_row_id` is a stored exact-match key, never tokenised.
        let row_id_field = schema_builder.add_text_field("_row_id", STRING | STORED);
        let body_options = TEXT.set_indexing_options(
            tantivy::schema::TextFieldIndexing::default()
                .set_tokenizer(registry_name(analyzer))
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        );
        let body_field = schema_builder.add_text_field("body", body_options);
        let index = Index::create_in_ram(schema_builder.build());
        index
            .tokenizers()
            .register(registry_name(analyzer), text_analyzer(analyzer));

        let mut doc_count = 0u64;
        let mut writer: IndexWriter = index
            .writer(WRITER_HEAP_BYTES)
            .map_err(|e| JammiError::Lexical(format!("index writer: {e}")))?;
        for (row_id, text) in rows {
            let row_id = row_id.as_ref();
            writer
                .add_document(doc!(row_id_field => row_id, body_field => text.as_ref()))
                .map_err(|e| JammiError::Lexical(format!("add document '{row_id}': {e}")))?;
            doc_count += 1;
        }
        writer
            .commit()
            .map_err(|e| JammiError::Lexical(format!("commit: {e}")))?;

        Ok(Self {
            index,
            analyzer,
            row_id_field,
            body_field,
            doc_count,
        })
    }

    /// Number of documents (rows) in the index.
    pub fn len(&self) -> u64 {
        self.doc_count
    }

    /// Whether the index holds no documents.
    pub fn is_empty(&self) -> bool {
        self.doc_count == 0
    }

    /// The distinct terms `text` analyses to, in first-occurrence order.
    fn query_terms(&self, text: &str) -> Vec<Term> {
        let mut analyzer = text_analyzer(self.analyzer);
        let mut stream = analyzer.token_stream(text);
        let mut terms: Vec<Term> = Vec::new();
        while stream.advance() {
            let term = Term::from_field_text(self.body_field, &stream.token().text);
            if !terms.contains(&term) {
                terms.push(term);
            }
        }
        terms
    }

    /// The top `k` rows for `text` by BM25 score, highest first, each carrying
    /// its 0-based rank. Ties break by `_row_id`, so the ranking is stable
    /// across runs. A query with no terms (empty, or only characters the
    /// analyzer drops) matches nothing.
    pub fn search(&self, text: &str, k: usize) -> Result<Vec<LexicalHit>> {
        let terms = self.query_terms(text);
        if k == 0 || self.doc_count == 0 || terms.is_empty() {
            return Ok(Vec::new());
        }
        let reader = self
            .index
            .reader()
            .map_err(|e| JammiError::Lexical(format!("reader: {e}")))?;
        let searcher = reader.searcher();
        let query = BooleanQuery::new_multiterms_query(terms);

        // Over-fetch so the tie-break sees every doc that could land in the
        // top-k, then truncate.
        let fetch = k.saturating_mul(4);
        let top = searcher
            .search(&query, &TopDocs::with_limit(fetch).order_by_score())
            .map_err(|e| JammiError::Lexical(format!("search: {e}")))?;

        let mut scored: Vec<(f32, String)> = top
            .into_iter()
            .map(|(score, addr)| {
                let doc: TantivyDocument = searcher
                    .doc(addr)
                    .map_err(|e| JammiError::Lexical(format!("fetch doc: {e}")))?;
                let row_id = doc
                    .get_first(self.row_id_field)
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| JammiError::Lexical("indexed doc missing _row_id".into()))?;
                Ok((score, row_id.to_string()))
            })
            .collect::<Result<_>>()?;
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        scored.truncate(k);

        Ok(scored
            .into_iter()
            .enumerate()
            .map(|(rank, (bm25_score, row_id))| LexicalHit {
                row_id,
                bm25_score,
                rank,
            })
            .collect())
    }
}

/// The lexical indexes a process has built, one per lexical table, each tied
/// to the content digest of the table version it was built from. A search over
/// a version this holds reuses its index; a search over a newer version builds
/// that version's index and replaces the older one.
#[derive(Default)]
pub struct LexicalIndexes {
    built: Mutex<HashMap<String, Built>>,
}

/// One table's index and the version it was built from.
struct Built {
    from: InputAnchor,
    index: Arc<LexicalIndex>,
}

impl LexicalIndexes {
    /// The index of the table version `anchor` names, built by `build` when
    /// this process has not built it yet. Two searches racing on a first build
    /// may both build; the indexes are identical, so either may be kept.
    pub async fn get_or_build<F>(&self, anchor: InputAnchor, build: F) -> Result<Arc<LexicalIndex>>
    where
        F: std::future::Future<Output = Result<LexicalIndex>>,
    {
        if let Some(index) = self.current(&anchor)? {
            return Ok(index);
        }
        let index = Arc::new(build.await?);
        self.lock()?.insert(
            anchor.source.clone(),
            Built {
                from: anchor,
                index: Arc::clone(&index),
            },
        );
        Ok(index)
    }

    fn current(&self, anchor: &InputAnchor) -> Result<Option<Arc<LexicalIndex>>> {
        Ok(self
            .lock()?
            .get(&anchor.source)
            .filter(|built| &built.from == anchor)
            .map(|built| Arc::clone(&built.index)))
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, HashMap<String, Built>>> {
        self.built
            .lock()
            .map_err(|_| JammiError::Lexical("lexical index registry poisoned".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<(&'static str, &'static str)> {
        vec![
            ("doc-1", "a method for reducing turbine blade vibration"),
            ("doc-2", "an apparatus for cooling turbine engine blades"),
            ("doc-3", "a recipe for baking sourdough bread"),
            ("doc-4", "turbine turbine turbine engine engine"),
        ]
    }

    #[test]
    fn bm25_ranks_rows_matching_more_query_terms_higher() {
        let idx = LexicalIndex::build(fixture(), LexicalAnalyzer::English).unwrap();
        let hits = idx.search("turbine engine", 10).unwrap();
        let ids: Vec<&str> = hits.iter().map(|h| h.row_id.as_str()).collect();
        // A disjunction: a row carrying either term matches, one carrying
        // neither does not.
        assert!(!ids.contains(&"doc-3"));
        assert!(ids.contains(&"doc-1"));
        // Rows carrying both terms outrank the row carrying one.
        let pos = |id: &str| ids.iter().position(|h| *h == id).unwrap();
        assert!(pos("doc-2") < pos("doc-1") && pos("doc-4") < pos("doc-1"));
    }

    #[test]
    fn query_syntax_is_searched_as_text() {
        let idx = LexicalIndex::build(fixture(), LexicalAnalyzer::English).unwrap();
        let plain = idx.search("turbine engine", 10).unwrap();
        let punctuated = idx.search("\"turbine: -engine", 10).unwrap();
        assert_eq!(plain, punctuated);
    }

    #[test]
    fn a_query_with_no_terms_matches_nothing() {
        let idx = LexicalIndex::build(fixture(), LexicalAnalyzer::English).unwrap();
        assert!(idx.search("", 10).unwrap().is_empty());
        assert!(idx.search("  ?!  ", 10).unwrap().is_empty());
    }

    #[test]
    fn ranks_are_dense_and_zero_based() {
        let idx = LexicalIndex::build(fixture(), LexicalAnalyzer::English).unwrap();
        let hits = idx.search("turbine engine", 10).unwrap();
        for (i, h) in hits.iter().enumerate() {
            assert_eq!(h.rank, i, "rank must equal position");
        }
    }

    #[test]
    fn repeated_term_row_outscores_single_mention() {
        let idx = LexicalIndex::build(fixture(), LexicalAnalyzer::English).unwrap();
        let hits = idx.search("turbine engine", 10).unwrap();
        // doc-4 mentions both terms repeatedly; it should rank above doc-2.
        let pos = |id: &str| hits.iter().position(|h| h.row_id == id).unwrap();
        assert!(pos("doc-4") < pos("doc-2"));
    }

    #[test]
    fn search_is_deterministic_across_runs() {
        let idx = LexicalIndex::build(fixture(), LexicalAnalyzer::English).unwrap();
        let a = idx.search("turbine engine", 10).unwrap();
        let b = idx.search("turbine engine", 10).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn k_caps_the_result_count() {
        let idx = LexicalIndex::build(fixture(), LexicalAnalyzer::English).unwrap();
        let hits = idx.search("turbine", 2).unwrap();
        assert!(hits.len() <= 2);
    }

    #[test]
    fn zero_k_returns_empty() {
        let idx = LexicalIndex::build(fixture(), LexicalAnalyzer::English).unwrap();
        assert!(idx.search("turbine", 0).unwrap().is_empty());
    }

    #[test]
    fn empty_index_returns_empty() {
        let idx =
            LexicalIndex::build(Vec::<(&str, &str)>::new(), LexicalAnalyzer::English).unwrap();
        assert!(idx.is_empty());
        assert!(idx.search("anything", 5).unwrap().is_empty());
    }

    #[test]
    fn english_stemmer_matches_inflected_forms() {
        // "blades" (built) is reachable via "blade" (queried) under the
        // English analyzer's Porter stemmer.
        let idx = LexicalIndex::build(fixture(), LexicalAnalyzer::English).unwrap();
        let hits = idx.search("blade", 10).unwrap();
        let ids: Vec<&str> = hits.iter().map(|h| h.row_id.as_str()).collect();
        assert!(ids.contains(&"doc-2"));
    }

    #[test]
    fn raw_analyzer_does_not_stem() {
        // Under Raw, "blade" does not reach the row that only has "blades".
        let rows = vec![("only-plural", "cooling turbine blades assembly")];
        let idx = LexicalIndex::build(rows, LexicalAnalyzer::Raw).unwrap();
        assert!(idx.search("blade", 10).unwrap().is_empty());
        // The exact surface form still matches.
        assert_eq!(idx.search("blades", 10).unwrap().len(), 1);
    }

    #[test]
    fn analyzer_choice_changes_results() {
        let rows = || vec![("r", "vibrating turbines")];
        let english = LexicalIndex::build(rows(), LexicalAnalyzer::English).unwrap();
        let raw = LexicalIndex::build(rows(), LexicalAnalyzer::Raw).unwrap();
        // "turbine" reaches the row under English stemming, not under Raw.
        assert_eq!(english.search("turbine", 5).unwrap().len(), 1);
        assert!(raw.search("turbine", 5).unwrap().is_empty());
    }
}
