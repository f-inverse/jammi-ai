//! `jammi search` subcommand.
//!
//! Runs a nearest-neighbour search over a source's embedding table on the
//! server and prints the hydrated result rows as JSON lines — one object per
//! row, in rank order — so a shell pipeline or a script reads them exactly.

use arrow::json::LineDelimitedWriter;
use clap::Args;
use jammi_client::DataClient;
use jammi_db::index::SearchMethod;
use jammi_wire::request::{SearchQuery, SearchRequest};

#[derive(Args)]
pub struct SearchArgs {
    /// The source whose embedding table is searched
    source: String,
    /// Query by example: rank by the vector stored for this row key
    #[arg(long)]
    row_key: String,
    /// How many nearest rows to return
    #[arg(short, long, default_value_t = 10)]
    k: usize,
    /// Search this embedding table of the source; omit for its newest ready one
    #[arg(long)]
    embedding_table: Option<String>,
    /// A SQL predicate over the hydrated columns: the `k` nearest rows that
    /// satisfy it
    #[arg(long)]
    filter: Option<String>,
    /// The columns to print, comma-separated; omit for every hydrated column
    #[arg(long, value_delimiter = ',')]
    select: Vec<String>,
    /// Score every vector instead of walking the ANN index
    #[arg(long, conflicts_with = "oversample")]
    exact: bool,
    /// A quantized index's retrieve→rescore breadth for this search
    #[arg(long)]
    oversample: Option<usize>,
}

pub async fn run(client: &DataClient, args: SearchArgs) -> Result<(), Box<dyn std::error::Error>> {
    let batches = client
        .search(SearchRequest {
            source_id: args.source,
            query: SearchQuery::RowKey(args.row_key),
            k: args.k,
            embedding_table: args.embedding_table,
            filter: args.filter,
            select: args.select,
            method: if args.exact {
                SearchMethod::Exact
            } else {
                SearchMethod::Approximate {
                    oversample: args.oversample,
                }
            },
        })
        .await?;
    let mut writer = LineDelimitedWriter::new(std::io::stdout().lock());
    writer.write_batches(&batches.iter().collect::<Vec<_>>())?;
    writer.finish()?;
    Ok(())
}
