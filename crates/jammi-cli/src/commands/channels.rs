//! `jammi channels` subcommand.
//!
//! Manage the catalog-backed evidence channels that gate which provenance
//! columns merge into a search result. Mirrors `sources` / `models` shape —
//! one subcommand enum, one `run` dispatcher, no shared state across
//! invocations.

use clap::Subcommand;
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::channel_repo::{ChannelColumn, ChannelColumnType, ChannelSpec};
use jammi_db::config::JammiConfig;
use jammi_db::evidence_channel::ChannelId;

#[derive(Subcommand)]
pub enum ChannelAction {
    /// Register a new evidence channel with one or more declared columns.
    Register {
        /// Channel identifier (e.g. `scored_by`). Lowercase ASCII + `_`.
        #[arg(long)]
        name: String,
        /// Merge priority. Lower priorities win when two channels declare the
        /// same column name.
        #[arg(long)]
        priority: i32,
        /// Declared columns in `name:Type` form. Type uses the PascalCase
        /// Arrow names accepted by `ChannelColumnType::from_sql_str`
        /// (`Float32`, `Float64`, `Int32`, `Int64`, `Utf8`, `Boolean`).
        /// Pass `--column` multiple times to declare multiple columns.
        #[arg(long = "column", value_name = "NAME:TYPE")]
        columns: Vec<String>,
    },

    /// Append one or more columns to an already-registered channel. Append-
    /// only: redeclaring an existing column (same name) is rejected even when
    /// the type matches.
    AddColumn {
        /// Channel identifier the columns are being added to.
        name: String,
        /// One column spec per `--column` flag in `name:Type` form.
        #[arg(long = "column", value_name = "NAME:TYPE")]
        columns: Vec<String>,
    },

    /// List registered evidence channels ordered by `(priority, name)`.
    List,
}

pub async fn run(
    config: JammiConfig,
    tenant: Option<jammi_db::TenantId>,
    action: ChannelAction,
) -> Result<(), Box<dyn std::error::Error>> {
    let session = InferenceSession::new(config).await?;
    if let Some(t) = tenant {
        session.bind_tenant(t);
    }

    match action {
        ChannelAction::Register {
            name,
            priority,
            columns,
        } => {
            if columns.is_empty() {
                return Err("at least one --column is required".into());
            }
            let id = ChannelId::new(&name)?;
            let parsed = parse_column_specs(&columns)?;
            let spec = ChannelSpec {
                id,
                priority,
                columns: parsed,
            };
            session.catalog().channels().register(&spec).await?;
            println!("Channel '{name}' registered (priority={priority}).");
        }
        ChannelAction::AddColumn { name, columns } => {
            if columns.is_empty() {
                return Err("at least one --column is required".into());
            }
            let id = ChannelId::new(&name)?;
            let parsed = parse_column_specs(&columns)?;
            session
                .catalog()
                .channels()
                .add_columns(&id, &parsed)
                .await?;
            println!("Channel '{name}' extended with {} column(s).", parsed.len());
        }
        ChannelAction::List => {
            let specs = session.catalog().channels().list().await?;
            if specs.is_empty() {
                println!("No channels registered.");
            } else {
                println!("{:<30} {:<10} Columns", "Name", "Priority");
                println!("{}", "-".repeat(70));
                for spec in specs {
                    let cols: Vec<String> = spec
                        .columns
                        .iter()
                        .map(|c| format!("{}:{}", c.name, channel_type_name(c.data_type)))
                        .collect();
                    println!(
                        "{:<30} {:<10} {}",
                        spec.id.as_str(),
                        spec.priority,
                        cols.join(", ")
                    );
                }
            }
        }
    }
    Ok(())
}

/// Parse a slice of `name:Type` strings into [`ChannelColumn`] entries. Each
/// spec must contain exactly one `:` and the type must be a PascalCase Arrow
/// name accepted by [`ChannelColumnType::from_sql_str`].
fn parse_column_specs(specs: &[String]) -> Result<Vec<ChannelColumn>, Box<dyn std::error::Error>> {
    let mut out = Vec::with_capacity(specs.len());
    for raw in specs {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err("empty --column spec".into());
        }
        let (name, ty) = trimmed.split_once(':').ok_or_else(|| {
            format!("--column '{trimmed}' must be of the form name:Type (e.g. ranker:Utf8)")
        })?;
        let name = name.trim();
        let ty = ty.trim();
        if name.is_empty() {
            return Err(format!("--column '{trimmed}' has an empty column name").into());
        }
        let data_type = ChannelColumnType::from_sql_str(ty)?;
        out.push(ChannelColumn {
            name: name.to_string(),
            data_type,
        });
    }
    Ok(out)
}

fn channel_type_name(t: ChannelColumnType) -> &'static str {
    match t {
        ChannelColumnType::Float32 => "Float32",
        ChannelColumnType::Float64 => "Float64",
        ChannelColumnType::Int32 => "Int32",
        ChannelColumnType::Int64 => "Int64",
        ChannelColumnType::Utf8 => "Utf8",
        ChannelColumnType::Boolean => "Boolean",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_column_specs_happy_path() {
        let cols =
            parse_column_specs(&["ranker:Utf8".to_string(), "rank_score:Float32".to_string()])
                .unwrap();
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0].name, "ranker");
        assert_eq!(cols[0].data_type, ChannelColumnType::Utf8);
        assert_eq!(cols[1].name, "rank_score");
        assert_eq!(cols[1].data_type, ChannelColumnType::Float32);
    }

    #[test]
    fn parse_column_specs_rejects_missing_separator() {
        let err = parse_column_specs(&["ranker".to_string()]).unwrap_err();
        assert!(err.to_string().contains("name:Type"));
    }

    #[test]
    fn parse_column_specs_rejects_empty_name() {
        let err = parse_column_specs(&[":Utf8".to_string()]).unwrap_err();
        assert!(err.to_string().contains("empty column name"));
    }

    #[test]
    fn parse_column_specs_rejects_unknown_type() {
        let err = parse_column_specs(&["x:Decimal".to_string()]).unwrap_err();
        assert!(err.to_string().contains("Decimal"));
    }
}
