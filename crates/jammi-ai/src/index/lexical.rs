//! BM25 lexical sidecar over the text columns of a result table.
//!
//! A [`LexicalIndex`] is the lexical peer of the USearch ANN sidecar: a
//! `tantivy` inverted index keyed by `_row_id`, written as a `.tantivy`
//! directory beside the table's Parquet object (the
//! `jammi_db::storage::sidecar_layout::SidecarKind::Lexical` sibling). It answers
//! `lexical_search(query_text, k) -> [(row_id, bm25_score, rank)]`, scoring
//! with tantivy's BM25 — the inverse of the ANN sidecar's cosine.
//!
//! ## Lifecycle
//! The lexical sidecar's lifecycle equals the ANN sidecar's: it is built (and
//! rebuilt) with the table. An immutable `result_table` rebuild produces a new
//! sidecar; for a `mutable_table` source, incremental update is the caller's
//! mode (re-ingest the changed rows into a fresh index).
//!
//! ## Tenant scope
//! Lexical search applies no row-level filter — isolation is table-level,
//! exactly as the ANN `search` path: the caller resolves the table through the
//! tenant-scoped catalog and hands this index only that table's rows. The index
//! never crosses a table boundary.
//!
//! ## Analyzer
//! Tokenisation / stemming / stop-words materially change BM25 ranking, so the
//! analyzer is configurable ([`Analyzer`]) with an English default. It is not
//! hardcoded English-only: a [`Analyzer::Raw`] (whitespace, no stemming)
//! analyzer is available for domains the English stemmer mangles.

use std::path::Path;

use jammi_db::error::{JammiError, Result};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Schema, Value, STORED, STRING, TEXT};
use tantivy::tokenizer::{LowerCaser, RemoveLongFilter, SimpleTokenizer, Stemmer, TextAnalyzer};
use tantivy::{doc, Index, IndexWriter, TantivyDocument};

/// Heap budget for the in-memory index writer (tantivy's minimum is 3 MiB; a
/// laptop-default sidecar build is small, so a modest arena keeps the footprint
/// bounded without spilling for typical fixtures).
const WRITER_HEAP_BYTES: usize = 15_000_000;

/// The tokenizer pipeline a [`LexicalIndex`] applies to its body text and to
/// query text — the two must match for BM25 term statistics to line up.
///
/// `English` is the sane default the spec mandates; `Raw` is the escape hatch
/// for domains the English stemmer mangles (codes, identifiers, languages the
/// Porter stemmer was never built for). The analyzer is *config*, never a
/// hardcoded English-only path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Analyzer {
    /// Lowercase + Porter (English) stemming + over-long-token removal. The
    /// default for English prose.
    English,
    /// Lowercase + over-long-token removal, **no** stemming. For text the
    /// English stemmer would corrupt.
    Raw,
}

impl Default for Analyzer {
    fn default() -> Self {
        Self::English
    }
}

impl Analyzer {
    /// The tokenizer registry name this analyzer registers under. Tantivy keys
    /// the body field's tokenizer by name, so the build and the query parser
    /// agree by referencing the same constant.
    fn registry_name(self) -> &'static str {
        match self {
            Self::English => "jammi_english",
            Self::Raw => "jammi_raw",
        }
    }

    /// Build the tantivy [`TextAnalyzer`] this variant denotes.
    ///
    /// Each `.filter(..)` returns a distinctly-typed builder, so the stemmed
    /// and unstemmed pipelines `.build()` in their own arms rather than sharing
    /// a tail `.build()`.
    fn build(self) -> TextAnalyzer {
        let base = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(RemoveLongFilter::limit(40))
            .filter(LowerCaser);
        match self {
            Self::English => base
                .filter(Stemmer::new(tantivy::tokenizer::Language::English))
                .build(),
            Self::Raw => base.build(),
        }
    }
}

/// One lexical hit: the table row's `_row_id`, its raw BM25 score, and its
/// 0-based rank in the returned list (rank is what RRF fuses on).
#[derive(Debug, Clone, PartialEq)]
pub struct LexicalHit {
    pub row_id: String,
    pub bm25_score: f32,
    pub rank: usize,
}

/// A BM25 inverted index over one table's text, keyed by `_row_id`.
pub struct LexicalIndex {
    index: Index,
    analyzer: Analyzer,
    row_id_field: tantivy::schema::Field,
    body_field: tantivy::schema::Field,
    doc_count: u64,
}

impl LexicalIndex {
    /// Build a lexical index over `rows`, each a `(row_id, text)` pair, under
    /// the given [`Analyzer`]. `text` is the concatenation of the row's
    /// indexed text columns (the caller joins them — the index is agnostic to
    /// which columns fed it).
    pub fn build<'a, I>(rows: I, analyzer: Analyzer) -> Result<Self>
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
    {
        let mut schema_builder = Schema::builder();
        // `_row_id` is a stored exact-match key, never tokenised.
        let row_id_field = schema_builder.add_text_field("_row_id", STRING | STORED);
        // The body rides the configurable analyzer (referenced by name below).
        let body_options = TEXT.set_indexing_options(
            tantivy::schema::TextFieldIndexing::default()
                .set_tokenizer(analyzer.registry_name())
                .set_index_option(tantivy::schema::IndexRecordOption::WithFreqsAndPositions),
        );
        let body_field = schema_builder.add_text_field("body", body_options);
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema);
        index
            .tokenizers()
            .register(analyzer.registry_name(), analyzer.build());

        let mut doc_count = 0u64;
        {
            let mut writer: IndexWriter = index
                .writer(WRITER_HEAP_BYTES)
                .map_err(|e| JammiError::Lexical(format!("index writer: {e}")))?;
            for (row_id, text) in rows {
                writer
                    .add_document(doc!(row_id_field => row_id, body_field => text))
                    .map_err(|e| JammiError::Lexical(format!("add document '{row_id}': {e}")))?;
                doc_count += 1;
            }
            writer
                .commit()
                .map_err(|e| JammiError::Lexical(format!("commit: {e}")))?;
        }

        Ok(Self {
            index,
            analyzer,
            row_id_field,
            body_field,
            doc_count,
        })
    }

    /// The analyzer this index was built under (the query path must match it).
    pub fn analyzer(&self) -> Analyzer {
        self.analyzer
    }

    /// Number of documents (rows) in the index.
    pub fn len(&self) -> u64 {
        self.doc_count
    }

    /// Whether the index holds no documents.
    pub fn is_empty(&self) -> bool {
        self.doc_count == 0
    }

    /// Run a BM25 query, returning the top `k` rows ranked by score (highest
    /// first), each carrying its 0-based rank. Ties break deterministically by
    /// `_row_id` so the rank a downstream fuser sees is stable across runs.
    pub fn search(&self, query_text: &str, k: usize) -> Result<Vec<LexicalHit>> {
        if k == 0 || self.doc_count == 0 {
            return Ok(Vec::new());
        }
        let reader = self
            .index
            .reader()
            .map_err(|e| JammiError::Lexical(format!("reader: {e}")))?;
        let searcher = reader.searcher();

        let mut parser = QueryParser::for_index(&self.index, vec![self.body_field]);
        parser.set_conjunction_by_default();
        let query = parser
            .parse_query(query_text)
            .map_err(|e| JammiError::Lexical(format!("parse query {query_text:?}: {e}")))?;

        // Over-fetch so the deterministic tie-break sees every doc that could
        // land in the top-k, then truncate.
        let fetch = k.saturating_mul(4).max(k);
        let top = searcher
            .search(&query, &TopDocs::with_limit(fetch).order_by_score())
            .map_err(|e| JammiError::Lexical(format!("search: {e}")))?;

        let mut scored: Vec<(f32, String)> = Vec::with_capacity(top.len());
        for (score, addr) in top {
            let doc: TantivyDocument = searcher
                .doc(addr)
                .map_err(|e| JammiError::Lexical(format!("fetch doc: {e}")))?;
            let row_id = doc
                .get_first(self.row_id_field)
                .and_then(|v| v.as_str())
                .ok_or_else(|| JammiError::Lexical("indexed doc missing _row_id".into()))?
                .to_string();
            scored.push((score, row_id));
        }

        // Descending score; ties broken ascending by row_id for determinism.
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
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

/// Persist a built index to a `.tantivy` directory at `dir`, then reopen it
/// against that directory. The on-disk directory is the
/// `jammi_db::storage::sidecar_layout::SidecarKind::Lexical` sibling; persistence rebuilds
/// rather than copying the RAM index so the stored term dictionary is the
/// canonical one.
impl LexicalIndex {
    /// Build a lexical index over `rows` directly into the on-disk `.tantivy`
    /// directory `dir`, returning the handle open against it. `dir` must exist.
    /// This is the sidecar-save path: the directory it writes is the object the
    /// storage layout uploads / cleans up by extension.
    pub fn build_in_dir<'a, I>(dir: &Path, rows: I, analyzer: Analyzer) -> Result<Self>
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
    {
        let mut schema_builder = Schema::builder();
        let row_id_field = schema_builder.add_text_field("_row_id", STRING | STORED);
        let body_options = TEXT.set_indexing_options(
            tantivy::schema::TextFieldIndexing::default()
                .set_tokenizer(analyzer.registry_name())
                .set_index_option(tantivy::schema::IndexRecordOption::WithFreqsAndPositions),
        );
        let body_field = schema_builder.add_text_field("body", body_options);
        let schema = schema_builder.build();

        let mmap = tantivy::directory::MmapDirectory::open(dir)
            .map_err(|e| JammiError::Lexical(format!("open sidecar dir {dir:?}: {e}")))?;
        let index = Index::create(mmap, schema, tantivy::IndexSettings::default())
            .map_err(|e| JammiError::Lexical(format!("create index in {dir:?}: {e}")))?;
        index
            .tokenizers()
            .register(analyzer.registry_name(), analyzer.build());

        let mut doc_count = 0u64;
        {
            let mut writer: IndexWriter = index
                .writer(WRITER_HEAP_BYTES)
                .map_err(|e| JammiError::Lexical(format!("index writer: {e}")))?;
            for (row_id, text) in rows {
                writer
                    .add_document(doc!(row_id_field => row_id, body_field => text))
                    .map_err(|e| JammiError::Lexical(format!("add document '{row_id}': {e}")))?;
                doc_count += 1;
            }
            writer
                .commit()
                .map_err(|e| JammiError::Lexical(format!("commit: {e}")))?;
        }

        Ok(Self {
            index,
            analyzer,
            row_id_field,
            body_field,
            doc_count,
        })
    }

    /// Reopen a lexical sidecar previously written by [`build_in_dir`] at
    /// `dir`, under the analyzer it was built with. The analyzer must match the
    /// build's (tantivy stores the tokenizer *name* per field but not its
    /// pipeline, so the query path re-registers it).
    ///
    /// [`build_in_dir`]: LexicalIndex::build_in_dir
    pub fn open_in_dir(dir: &Path, analyzer: Analyzer) -> Result<Self> {
        let mmap = tantivy::directory::MmapDirectory::open(dir)
            .map_err(|e| JammiError::Lexical(format!("open sidecar dir {dir:?}: {e}")))?;
        let index = Index::open(mmap)
            .map_err(|e| JammiError::Lexical(format!("open index in {dir:?}: {e}")))?;
        index
            .tokenizers()
            .register(analyzer.registry_name(), analyzer.build());

        let schema = index.schema();
        let row_id_field = schema
            .get_field("_row_id")
            .map_err(|e| JammiError::Lexical(format!("sidecar missing _row_id field: {e}")))?;
        let body_field = schema
            .get_field("body")
            .map_err(|e| JammiError::Lexical(format!("sidecar missing body field: {e}")))?;

        let reader = index
            .reader()
            .map_err(|e| JammiError::Lexical(format!("reader: {e}")))?;
        let doc_count = reader.searcher().num_docs();

        Ok(Self {
            index,
            analyzer,
            row_id_field,
            body_field,
            doc_count,
        })
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
    fn bm25_returns_keyword_matching_rows() {
        let idx = LexicalIndex::build(fixture(), Analyzer::English).unwrap();
        let hits = idx.search("turbine engine", 10).unwrap();
        let ids: Vec<&str> = hits.iter().map(|h| h.row_id.as_str()).collect();
        // The bread row shares no terms; it must not appear.
        assert!(!ids.contains(&"doc-3"));
        // Conjunction default: only rows carrying BOTH terms match.
        assert!(ids.contains(&"doc-2"));
        assert!(ids.contains(&"doc-4"));
    }

    #[test]
    fn ranks_are_dense_and_zero_based() {
        let idx = LexicalIndex::build(fixture(), Analyzer::English).unwrap();
        let hits = idx.search("turbine engine", 10).unwrap();
        for (i, h) in hits.iter().enumerate() {
            assert_eq!(h.rank, i, "rank must equal position");
        }
    }

    #[test]
    fn repeated_term_row_outscores_single_mention() {
        let idx = LexicalIndex::build(fixture(), Analyzer::English).unwrap();
        let hits = idx.search("turbine engine", 10).unwrap();
        // doc-4 mentions both terms repeatedly; it should rank above doc-2.
        let pos = |id: &str| hits.iter().position(|h| h.row_id == id).unwrap();
        assert!(pos("doc-4") < pos("doc-2"));
    }

    #[test]
    fn search_is_deterministic_across_runs() {
        let idx = LexicalIndex::build(fixture(), Analyzer::English).unwrap();
        let a = idx.search("turbine engine", 10).unwrap();
        let b = idx.search("turbine engine", 10).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn k_caps_the_result_count() {
        let idx = LexicalIndex::build(fixture(), Analyzer::English).unwrap();
        let hits = idx.search("turbine", 2).unwrap();
        assert!(hits.len() <= 2);
    }

    #[test]
    fn zero_k_returns_empty() {
        let idx = LexicalIndex::build(fixture(), Analyzer::English).unwrap();
        assert!(idx.search("turbine", 0).unwrap().is_empty());
    }

    #[test]
    fn empty_index_returns_empty() {
        let idx = LexicalIndex::build(Vec::new(), Analyzer::English).unwrap();
        assert!(idx.is_empty());
        assert!(idx.search("anything", 5).unwrap().is_empty());
    }

    #[test]
    fn english_stemmer_matches_inflected_forms() {
        // "blades" (built) is reachable via "blade" (queried) under the
        // English analyzer's Porter stemmer.
        let idx = LexicalIndex::build(fixture(), Analyzer::English).unwrap();
        let hits = idx.search("blade", 10).unwrap();
        let ids: Vec<&str> = hits.iter().map(|h| h.row_id.as_str()).collect();
        assert!(ids.contains(&"doc-2"));
    }

    #[test]
    fn raw_analyzer_does_not_stem() {
        // Under Raw, "blade" does not reach the row that only has "blades".
        let rows = vec![("only-plural", "cooling turbine blades assembly")];
        let idx = LexicalIndex::build(rows, Analyzer::Raw).unwrap();
        assert!(idx.search("blade", 10).unwrap().is_empty());
        // The exact surface form still matches.
        assert_eq!(idx.search("blades", 10).unwrap().len(), 1);
    }

    #[test]
    fn analyzer_choice_changes_results() {
        let rows = || vec![("r", "vibrating turbines")];
        let english = LexicalIndex::build(rows(), Analyzer::English).unwrap();
        let raw = LexicalIndex::build(rows(), Analyzer::Raw).unwrap();
        // "turbine" reaches the row under English stemming, not under Raw.
        assert_eq!(english.search("turbine", 5).unwrap().len(), 1);
        assert!(raw.search("turbine", 5).unwrap().is_empty());
    }

    #[test]
    fn round_trips_through_a_sidecar_directory() {
        let dir = tempfile::tempdir().unwrap();
        let built = LexicalIndex::build_in_dir(dir.path(), fixture(), Analyzer::English).unwrap();
        let from_build = built.search("turbine engine", 10).unwrap();
        drop(built);

        let reopened = LexicalIndex::open_in_dir(dir.path(), Analyzer::English).unwrap();
        let from_disk = reopened.search("turbine engine", 10).unwrap();
        assert_eq!(from_build, from_disk);
        assert_eq!(reopened.len(), 4);
    }
}
