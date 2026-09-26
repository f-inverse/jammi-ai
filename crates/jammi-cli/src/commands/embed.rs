//! `jammi embed` subcommand.
//!
//! Generates embeddings for a source's columns on the server, persisting one
//! vector per row as a result table, and prints the table it wrote.

use clap::{Args, ValueEnum};
use jammi_client::DataClient;
use jammi_db::store::{CacheOutcome, CachePolicy};
use jammi_wire::request::{EmbeddingRequest, Modality};

/// Which tower embeds the columns.
#[derive(Clone, Copy, ValueEnum)]
pub enum Tower {
    /// The text tower: the columns are joined and embedded as text.
    Text,
    /// The image tower: one column of image bytes.
    Image,
    /// The audio tower: one column of audio bytes.
    Audio,
}

impl From<Tower> for Modality {
    fn from(tower: Tower) -> Self {
        match tower {
            Tower::Text => Modality::Text,
            Tower::Image => Modality::Image,
            Tower::Audio => Modality::Audio,
        }
    }
}

#[derive(Args)]
pub struct EmbedArgs {
    /// The source whose rows are embedded
    source: String,
    /// The encoder: `local:<path>`, a Hub repo id, or a fine-tuned model id
    #[arg(long)]
    model: String,
    /// The content columns, comma-separated
    #[arg(long, value_delimiter = ',', required = true)]
    columns: Vec<String>,
    /// The column whose value keys each embedding row
    #[arg(long)]
    key: String,
    /// Which tower embeds the columns
    #[arg(long, value_enum, default_value_t = Tower::Text)]
    modality: Tower,
    /// Serve the model's leading `dimensions` coordinates (a Matryoshka
    /// prefix); omit for the model's own width
    #[arg(long)]
    dimensions: Option<usize>,
    /// Reuse a ready table the same definition over the same inputs already
    /// produced, rather than recomputing
    #[arg(long)]
    reuse: bool,
}

pub async fn run(client: &DataClient, args: EmbedArgs) -> Result<(), Box<dyn std::error::Error>> {
    let (table, outcome) = client
        .generate_embeddings(EmbeddingRequest {
            source_id: args.source,
            model_id: args.model,
            columns: args.columns,
            key_column: args.key,
            modality: args.modality.into(),
            dimensions: args.dimensions,
            cache: if args.reuse {
                CachePolicy::Use
            } else {
                CachePolicy::Bypass
            },
        })
        .await?;
    println!("table:      {}", table.table_name);
    println!("rows:       {}", table.row_count);
    match table.dimensions() {
        Some(width) => println!("dimensions: {width}"),
        None => println!("dimensions: —"),
    }
    match outcome {
        CacheOutcome::Computed => println!("cache:      computed"),
        CacheOutcome::Reused(_) => println!("cache:      reused"),
    }
    Ok(())
}
