//! `jammi trigger` subcommand.
//!
//! Drive the server's trigger-stream topic-admin surface over the remote
//! [`CatalogClient`]: register / drop / list topics. These are the control-plane
//! lifecycle verbs; the data-plane `publish` / `subscribe` compute verbs are not
//! exposed on the strict client CLI. Every verb runs on the server — the CLI
//! only builds the topic schema from its inline spec and hands the work to the
//! session.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};
use clap::Subcommand;
use jammi_admin::CatalogClient;
use jammi_db::trigger::{TopicDefinition, TopicId};

/// Subcommands under `jammi trigger`.
#[derive(Subcommand)]
pub enum TriggerAction {
    /// List registered topics visible to the current tenant binding.
    List,

    /// Register a new topic. The schema is given inline as a comma-
    /// separated list of `<name>:<type>[:nullable]` triples; supported
    /// types are `int`, `float`, `string`, `bool`.
    Register {
        /// Fully-qualified topic name (e.g. `cdc.orders`).
        #[arg(long)]
        name: String,
        /// Schema definition, e.g. `op:string,ts_ms:int,key:string,after:string:nullable`.
        #[arg(long)]
        schema: String,
        /// Optional broker-driver metadata, parsed as JSON object (e.g.
        /// `'{"retention_seconds":86400}'`).
        #[arg(long, default_value = "{}")]
        broker_metadata: String,
    },

    /// Drop a topic by name. `--if-exists` makes a missing topic a no-op.
    Drop {
        /// Topic name to drop.
        #[arg(long)]
        name: String,
        /// Treat a missing topic as success rather than an error.
        #[arg(long)]
        if_exists: bool,
    },
}

pub async fn run(
    session: &CatalogClient,
    action: TriggerAction,
) -> Result<(), Box<dyn std::error::Error>> {
    match action {
        TriggerAction::List => list_topics(session).await,
        TriggerAction::Register {
            name,
            schema,
            broker_metadata,
        } => register_topic(session, &name, &schema, &broker_metadata).await,
        TriggerAction::Drop { name, if_exists } => drop_topic(session, &name, if_exists).await,
    }
}

async fn list_topics(session: &CatalogClient) -> Result<(), Box<dyn std::error::Error>> {
    let topics = session.list_topics().await?;
    if topics.is_empty() {
        println!("No topics registered.");
        return Ok(());
    }
    println!("{:<40} {:<10} Columns", "Name", "Tenant");
    println!("{}", "-".repeat(80));
    for t in topics {
        let cols: Vec<String> = t
            .schema
            .fields()
            .iter()
            .map(|f| format!("{}:{}", f.name(), simple_type_name(f.data_type())))
            .collect();
        println!(
            "{:<40} {:<10} {}",
            t.name,
            t.tenant
                .map(|t| t.to_string())
                .unwrap_or_else(|| "—".into()),
            cols.join(", ")
        );
    }
    Ok(())
}

async fn register_topic(
    session: &CatalogClient,
    name: &str,
    schema_spec: &str,
    broker_metadata: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let schema = parse_schema_spec(schema_spec)?;
    let broker_metadata: BTreeMap<String, String> = serde_json::from_str(broker_metadata)
        .map_err(|e| format!("broker_metadata must be a JSON object: {e}"))?;
    let topic = TopicDefinition {
        id: TopicId::new(),
        name: name.to_string(),
        schema: Arc::new(schema),
        // The wire body stays tenant-free: the server stamps the session's
        // tenant onto the topic under its tenant scope.
        tenant: None,
        broker_metadata,
    };
    session.register_topic(&topic).await?;
    println!("Topic '{name}' registered (id={}).", topic.id);
    Ok(())
}

async fn drop_topic(
    session: &CatalogClient,
    name: &str,
    if_exists: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Resolve the topic by name to its server-minted id, then drop by id (the
    // identity the `CatalogClient::drop_topic` surface is keyed on). A missing topic
    // is a no-op under `--if-exists` and an error otherwise.
    let topic = session
        .list_topics()
        .await?
        .into_iter()
        .find(|t| t.name == name);
    match topic {
        Some(t) => {
            session.drop_topic(t.id).await?;
            println!("Topic '{name}' dropped.");
            Ok(())
        }
        None if if_exists => {
            println!("Topic '{name}' not found (no-op).");
            Ok(())
        }
        None => Err(format!("topic '{name}' not found").into()),
    }
}

fn parse_schema_spec(spec: &str) -> Result<Schema, Box<dyn std::error::Error>> {
    let mut fields: Vec<Field> = Vec::new();
    for raw in spec.split(',') {
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let mut parts = raw.split(':');
        let name = parts.next().ok_or("missing column name")?.trim();
        let ty = parts.next().ok_or("missing column type")?.trim();
        let nullable = matches!(parts.next().map(str::trim), Some("nullable"));
        let data_type = match ty {
            "int" | "int64" => DataType::Int64,
            "float" | "float64" | "double" => DataType::Float64,
            "string" | "utf8" => DataType::Utf8,
            "bool" | "boolean" => DataType::Boolean,
            other => return Err(format!("unsupported column type: {other}").into()),
        };
        fields.push(Field::new(name, data_type, nullable));
    }
    if fields.is_empty() {
        return Err("schema must declare at least one column".into());
    }
    Ok(Schema::new(fields))
}

fn simple_type_name(ty: &DataType) -> &'static str {
    match ty {
        DataType::Int64 => "int",
        DataType::Float64 => "float",
        DataType::Utf8 => "string",
        DataType::Boolean => "bool",
        _ => "?",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_schema_spec_builds_fields() {
        let schema = parse_schema_spec("op:string,ts_ms:int,key:string:nullable").unwrap();
        assert_eq!(schema.fields().len(), 3);
        assert_eq!(schema.field(0).name(), "op");
        assert_eq!(schema.field(0).data_type(), &DataType::Utf8);
        assert_eq!(schema.field(1).data_type(), &DataType::Int64);
        assert!(schema.field(2).is_nullable());
    }

    #[test]
    fn parse_schema_spec_rejects_unknown_type() {
        let err = parse_schema_spec("x:widget").unwrap_err();
        assert!(err.to_string().contains("unsupported column type"));
    }

    #[test]
    fn parse_schema_spec_rejects_empty() {
        let err = parse_schema_spec("  ").unwrap_err();
        assert!(err.to_string().contains("at least one column"));
    }
}
