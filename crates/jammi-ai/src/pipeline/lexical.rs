//! The `build_lexical_index` verb: project a source's text columns into a
//! lexical table — one `(_row_id, text)` row per source row — that
//! `lexical_search` ranks by BM25.
//!
//! The table is the index's data of record. Its inverted index is derived
//! state, rebuilt in memory from these rows
//! ([`crate::index::LexicalIndexes`]), so the table alone pins what a search
//! ranks: the text as it was read, under the analyzer the descriptor records.

use jammi_db::catalog::result_repo::{JobAttempt, Producer, ResultTableKind, ResultTableRecord};
use jammi_db::error::{JammiError, Result};
use jammi_db::index::LexicalAnalyzer;
use jammi_db::sql::{quote_ident, source_relation};
use jammi_db::store::manifest::{InputAnchor, Materialization, ProducingDescriptor};
use jammi_db::store::{ResultTableOrigin, SinkKind};
use serde::{Deserialize, Serialize};

use crate::session::InferenceSession;

/// What a lexical index is built from: a source's text columns, keyed by one
/// of its columns, tokenised by an analyzer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildLexicalIndex {
    /// The text columns, joined in this order by a space into each row's text.
    pub columns: Vec<String>,
    /// The source column each row is keyed by — the key a search hydrates the
    /// source's rows through.
    pub key_column: String,
    /// How the text and every query are tokenised.
    pub analyzer: LexicalAnalyzer,
}

impl BuildLexicalIndex {
    fn descriptor(&self, source_id: &str) -> ProducingDescriptor {
        ProducingDescriptor::LexicalIndex {
            source_id: source_id.to_string(),
            key_column: self.key_column.clone(),
            text_columns: self.columns.clone(),
            analyzer: self.analyzer,
        }
    }
}

/// Materialise `source_id`'s lexical table: every source row's key as
/// `_row_id` and its text columns joined as `text`, resolved through the
/// session's tenant-scoped catalog.
pub async fn run(
    session: &InferenceSession,
    source_id: &str,
    params: &BuildLexicalIndex,
    job_attempt: Option<JobAttempt<'_>>,
) -> Result<ResultTableRecord> {
    if params.columns.is_empty() {
        return Err(JammiError::Config(
            "build_lexical_index requires at least one text column".into(),
        ));
    }
    let relation = source_relation(source_id, &session.find_table_name(source_id).await?);
    let text = params
        .columns
        .iter()
        .map(|c| format!("CAST({} AS VARCHAR)", quote_ident(c)))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT CAST({key} AS VARCHAR) AS _row_id, concat_ws(' ', {text}) AS text FROM {relation}",
        key = quote_ident(&params.key_column),
    );
    let plan = session
        .context()
        .sql(&sql)
        .await
        .map_err(JammiError::from)?
        .create_physical_plan()
        .await
        .map_err(JammiError::from)?;

    let columns = params.columns.join(",");
    let mut building = session
        .result_store()
        .create_table(ResultTableOrigin {
            source_id,
            producer: Producer::Derivation { task: None },
            kind: ResultTableKind::Lexical,
            derived_from: None,
            dimensions: None,
            key_column: Some(&params.key_column),
            text_columns: Some(&columns),
            job_attempt,
        })
        .await?;
    let summary = session
        .result_store()
        .write_result_table(
            &mut building,
            SinkKind::Rows,
            plan,
            session.context().task_ctx(),
        )
        .await?;

    // A registered source exposes no version surface, so its read is recorded
    // honestly as `UnpinnedAtInstant`: the table itself is what pins the text.
    let descriptor = params.descriptor(source_id);
    let inputs = vec![InputAnchor::unpinned_at_instant(
        source_id,
        chrono::Utc::now().to_rfc3339(),
    )];
    building
        .finish(
            session.context(),
            summary.rows as usize,
            Materialization::new(&descriptor, &summary.env, inputs),
        )
        .await
}
