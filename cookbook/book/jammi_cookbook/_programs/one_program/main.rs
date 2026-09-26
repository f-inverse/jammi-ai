//! The cookbook's one program through the Rust API: register a source, embed
//! it, and search it with an in-process `InferenceSession` — the engine linked
//! into this binary, no server.
//!
//! Usage: `one_program <corpus-url> <model> <row-key>`. The artifact directory
//! and every other setting come from `jammi.toml` or the `JAMMI_*` environment.
//! Prints the three rows nearest `<row-key>` as JSON lines, in rank order.

use std::sync::Arc;

use arrow::json::LineDelimitedWriter;
use jammi_ai::session::InferenceSession;
use jammi_ai::SearchMethod;
use jammi_db::config::JammiConfig;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::CachePolicy;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (Some(corpus), Some(model), Some(row_key)) = (args.next(), args.next(), args.next())
    else {
        return Err("usage: one_program <corpus-url> <model> <row-key>".into());
    };

    let session = Arc::new(InferenceSession::new(JammiConfig::load(None)?).await?);

    // 1. Register the source.
    let connection = SourceConnection::parse(&corpus, FileFormat::Parquet)?;
    session
        .add_source("corpus", SourceType::File, connection)
        .await?;

    // 2. Embed its `content` column, one vector per `id`.
    session
        .generate_text_embeddings(
            "corpus",
            &model,
            &["content".to_string()],
            "id",
            CachePolicy::Bypass,
            None,
        )
        .await?;

    // 3. Search: the rows nearest the vector stored for `row_key`.
    let hits = session
        .search_by_id("corpus", &row_key, 3, None, SearchMethod::default())
        .await?
        .select(&["id".to_string(), "similarity".to_string()])?
        .run()
        .await?;

    let mut out = LineDelimitedWriter::new(std::io::stdout().lock());
    out.write_batches(&hits.iter().collect::<Vec<_>>())?;
    out.finish()?;
    Ok(())
}
