//! `jammi mutable` subcommand.
//!
//! Drive the remote [`Session`]'s `create_mutable_table` /
//! `drop_mutable_table` / `list_mutable_tables` verbs from the CLI. Schema is
//! supplied via a JSON file so the same fixture can drive Rust, Python, and CLI
//! tests without duplicating the column definitions. Index specs are inline
//! (`name=X,columns=A+B,unique=false`) to mirror the `trigger` subcommand's
//! style.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};
use clap::Subcommand;
use jammi_ai::Session;
use jammi_db::store::mutable::definition::{
    MutableIndexDef, MutableTableDefinitionBuilder, MutableTableId,
};
use serde::Deserialize;

#[derive(Subcommand)]
pub enum MutableAction {
    /// Register a mutable companion table. The schema is read from a JSON
    /// file (array of `{name, type, nullable}` objects). Types use the
    /// PascalCase Arrow names the engine's catalog encoder accepts.
    Create {
        /// Mutable table identifier (lowercase ASCII / digits / `_`).
        #[arg(long)]
        name: String,
        /// Path to a JSON file describing the schema.
        #[arg(long)]
        schema: PathBuf,
        /// Comma-separated primary-key column list (e.g.
        /// `feature_id,effective_from`).
        #[arg(long)]
        primary_key: String,
        /// Optional secondary index in `name=NAME,columns=A+B,unique=BOOL` form.
        /// Pass `--index` multiple times for multiple indexes.
        #[arg(long = "index", value_name = "NAME=...,columns=...,unique=...")]
        indexes: Vec<String>,
        /// Optional monotonic ordering column (must be Int64 or UInt64). When
        /// set, the table can back a Phase-4 trigger-stream topic.
        #[arg(long)]
        order_column: Option<String>,
    },

    /// Drop a registered mutable companion table.
    Drop {
        /// Mutable table identifier.
        name: String,
    },

    /// List registered mutable tables visible to the current tenant binding.
    List,
}

pub async fn run(
    session: &Session,
    action: MutableAction,
) -> Result<(), Box<dyn std::error::Error>> {
    match action {
        MutableAction::Create {
            name,
            schema,
            primary_key,
            indexes,
            order_column,
        } => {
            let id = MutableTableId::new(&name)?;
            let arrow_schema = parse_schema_file(&schema)?;
            let pk_cols: Vec<String> = primary_key
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect();
            if pk_cols.is_empty() {
                return Err("--primary-key must list at least one column".into());
            }
            // The wire body stays tenant-free: the server stamps the session's
            // bound tenant onto the catalog row under its tenant scope, so the
            // definition the client builds carries no tenant.
            let mut builder = MutableTableDefinitionBuilder::new(id.clone(), arrow_schema)
                .primary_key(pk_cols)
                .tenant(None);
            for spec in &indexes {
                builder = builder.index(parse_index_spec(spec)?);
            }
            if let Some(col) = order_column {
                builder = builder.order_column(col);
            }
            let def = builder.build()?;
            let pk_repr = def.primary_key.join(",");
            let idx_repr = if def.indexes.is_empty() {
                "[]".to_string()
            } else {
                def.indexes
                    .iter()
                    .map(|i| i.name.as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            };
            session.create_mutable_table(def).await?;
            println!(
                "Mutable table '{name}' registered (primary_key=[{pk_repr}], indexes=[{idx_repr}])."
            );
        }
        MutableAction::Drop { name } => {
            let id = MutableTableId::new(&name)?;
            session.drop_mutable_table(&id).await?;
            println!("Mutable table '{name}' dropped.");
        }
        MutableAction::List => {
            let tables = session.list_mutable_tables().await?;
            if tables.is_empty() {
                println!("No mutable tables registered.");
            } else {
                println!("{:<30} {:<25} Columns", "Name", "Primary Key");
                println!("{}", "-".repeat(80));
                for def in tables {
                    let cols: Vec<String> = def
                        .schema
                        .fields()
                        .iter()
                        .map(|f| format!("{}:{}", f.name(), arrow_type_name(f.data_type())))
                        .collect();
                    println!(
                        "{:<30} {:<25} {}",
                        def.id.as_str(),
                        def.primary_key.join(","),
                        cols.join(", ")
                    );
                }
            }
        }
    }
    Ok(())
}

/// JSON wire shape for one schema column.
#[derive(Deserialize)]
struct SchemaColumnFile {
    name: String,
    #[serde(rename = "type")]
    ty: String,
    #[serde(default)]
    nullable: bool,
}

/// Parse a JSON file containing an array of `{name, type, nullable}` records
/// into a [`SchemaRef`].
///
/// Accepted `type` values are the PascalCase Arrow names the engine's catalog
/// encoder supports: `Boolean`, `Int8`/`Int16`/`Int32`/`Int64`,
/// `UInt8`/`UInt16`/`UInt32`/`UInt64`, `Float32`, `Float64`, `Utf8`,
/// `Binary`. Anything else is rejected with a typed error mentioning the
/// supported set so the caller can correct the fixture.
pub(crate) fn parse_schema_file(path: &Path) -> Result<SchemaRef, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("read schema file {}: {e}", path.display()))?;
    let columns: Vec<SchemaColumnFile> = serde_json::from_str(&raw)
        .map_err(|e| format!("parse schema file {}: {e}", path.display()))?;
    if columns.is_empty() {
        return Err("schema must declare at least one column".into());
    }
    let fields: Result<Vec<Field>, Box<dyn std::error::Error>> = columns
        .into_iter()
        .map(|c| {
            let data_type = arrow_type_from_name(&c.ty)?;
            Ok(Field::new(&c.name, data_type, c.nullable))
        })
        .collect();
    Ok(Arc::new(Schema::new(fields?)))
}

/// Parse one secondary-index spec in
/// `name=NAME,columns=A+B,unique=BOOL` form into a [`MutableIndexDef`].
///
/// Splits on `,` (key/value pairs) and `+` (multi-column index). Keys are
/// case-insensitive; values are not. Missing or unrecognised keys produce a
/// typed error.
pub(crate) fn parse_index_spec(spec: &str) -> Result<MutableIndexDef, Box<dyn std::error::Error>> {
    let mut name: Option<String> = None;
    let mut columns: Option<Vec<String>> = None;
    let mut unique: Option<bool> = None;

    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (key, value) = part.split_once('=').ok_or_else(|| {
            format!("--index '{spec}' fragment '{part}' must be of the form key=value")
        })?;
        let key = key.trim().to_ascii_lowercase();
        let value = value.trim();
        match key.as_str() {
            "name" => {
                if value.is_empty() {
                    return Err(format!("--index '{spec}' has empty name").into());
                }
                name = Some(value.to_string());
            }
            "columns" => {
                let cols: Vec<String> = value
                    .split('+')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect();
                if cols.is_empty() {
                    return Err(format!("--index '{spec}' must declare at least one column").into());
                }
                columns = Some(cols);
            }
            "unique" => {
                let parsed: bool = value.parse().map_err(|_| {
                    format!("--index '{spec}' unique value '{value}' must be true or false")
                })?;
                unique = Some(parsed);
            }
            other => {
                return Err(format!(
                    "--index '{spec}' unknown key '{other}' (expected name/columns/unique)"
                )
                .into());
            }
        }
    }

    Ok(MutableIndexDef {
        name: name.ok_or_else(|| format!("--index '{spec}' missing required key 'name'"))?,
        columns: columns
            .ok_or_else(|| format!("--index '{spec}' missing required key 'columns'"))?,
        unique: unique.unwrap_or(false),
    })
}

/// PascalCase Arrow type names accepted by the engine's catalog encoder. This
/// list matches `topic_repo::data_type_name`/`data_type_from_name` so the CLI
/// fixtures round-trip through the catalog without surprise widening.
fn arrow_type_from_name(name: &str) -> Result<DataType, Box<dyn std::error::Error>> {
    Ok(match name {
        "Boolean" => DataType::Boolean,
        "Int8" => DataType::Int8,
        "Int16" => DataType::Int16,
        "Int32" => DataType::Int32,
        "Int64" => DataType::Int64,
        "UInt8" => DataType::UInt8,
        "UInt16" => DataType::UInt16,
        "UInt32" => DataType::UInt32,
        "UInt64" => DataType::UInt64,
        "Float32" => DataType::Float32,
        "Float64" => DataType::Float64,
        "Utf8" => DataType::Utf8,
        "Binary" => DataType::Binary,
        other => {
            return Err(format!(
                "unsupported schema type '{other}'; expected one of \
                 Boolean, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, \
                 UInt64, Float32, Float64, Utf8, Binary"
            )
            .into());
        }
    })
}

fn arrow_type_name(ty: &DataType) -> &'static str {
    match ty {
        DataType::Boolean => "Boolean",
        DataType::Int8 => "Int8",
        DataType::Int16 => "Int16",
        DataType::Int32 => "Int32",
        DataType::Int64 => "Int64",
        DataType::UInt8 => "UInt8",
        DataType::UInt16 => "UInt16",
        DataType::UInt32 => "UInt32",
        DataType::UInt64 => "UInt64",
        DataType::Float32 => "Float32",
        DataType::Float64 => "Float64",
        DataType::Utf8 => "Utf8",
        DataType::Binary => "Binary",
        _ => "?",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_schema(tmp: &TempDir, body: &str) -> PathBuf {
        let path = tmp.path().join("schema.json");
        std::fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn parse_schema_file_happy_path() {
        let tmp = TempDir::new().unwrap();
        let path = write_schema(
            &tmp,
            r#"[
              {"name":"feature_id","type":"Int64","nullable":false},
              {"name":"feature_value","type":"Float64","nullable":true}
            ]"#,
        );
        let schema = parse_schema_file(&path).unwrap();
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.field(0).name(), "feature_id");
        assert_eq!(schema.field(0).data_type(), &DataType::Int64);
        assert!(!schema.field(0).is_nullable());
        assert_eq!(schema.field(1).data_type(), &DataType::Float64);
        assert!(schema.field(1).is_nullable());
    }

    #[test]
    fn parse_schema_file_rejects_unknown_type() {
        let tmp = TempDir::new().unwrap();
        let path = write_schema(&tmp, r#"[{"name":"x","type":"Decimal","nullable":false}]"#);
        let err = parse_schema_file(&path).unwrap_err();
        assert!(err.to_string().contains("Decimal"));
    }

    #[test]
    fn parse_schema_file_rejects_empty_array() {
        let tmp = TempDir::new().unwrap();
        let path = write_schema(&tmp, "[]");
        let err = parse_schema_file(&path).unwrap_err();
        assert!(err.to_string().contains("at least one column"));
    }

    #[test]
    fn parse_index_spec_happy_path() {
        let idx = parse_index_spec("name=idx_active,columns=feature_id+effective_to,unique=false")
            .unwrap();
        assert_eq!(idx.name, "idx_active");
        assert_eq!(idx.columns, vec!["feature_id", "effective_to"]);
        assert!(!idx.unique);
    }

    #[test]
    fn parse_index_spec_defaults_unique_false() {
        let idx = parse_index_spec("name=idx_active,columns=feature_id").unwrap();
        assert!(!idx.unique);
    }

    #[test]
    fn parse_index_spec_rejects_missing_name() {
        let err = parse_index_spec("columns=a").unwrap_err();
        assert!(err.to_string().contains("missing required key 'name'"));
    }

    #[test]
    fn parse_index_spec_rejects_missing_columns() {
        let err = parse_index_spec("name=idx").unwrap_err();
        assert!(err.to_string().contains("missing required key 'columns'"));
    }

    #[test]
    fn parse_index_spec_rejects_unknown_key() {
        let err = parse_index_spec("name=idx,columns=a,extra=true").unwrap_err();
        assert!(err.to_string().contains("unknown key"));
    }
}
