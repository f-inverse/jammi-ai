//! `jammi trigger` subcommand.
//!
//! Drive the server's trigger-stream surface over the remote [`Session`]:
//! register / list topics, publish a batch, and subscribe. Every verb runs on
//! the server — the CLI builds the Arrow `RecordBatch` for a publish and parses
//! the subscribe predicate to SQL locally (a parse against an empty
//! `SessionContext`, no catalog), then hands the work to the session.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Array, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use clap::{ArgGroup, Subcommand};
use datafusion::execution::context::SessionContext;
use futures::StreamExt;
use jammi_ai::Session;
use jammi_db::trigger::{DeliveredBatch, Offset, Predicate, TopicDefinition, TopicId};

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

    /// Publish one batch to a topic. Either pass one or more `--row` JSON
    /// objects (one per row), or `--json-file` pointing at a JSON file
    /// containing either an array of row objects or a single row object.
    /// The two modes are mutually exclusive.
    #[command(group(
        ArgGroup::new("publish_input")
            .args(["rows", "json_file"])
            .required(true)
            .multiple(false),
    ))]
    Publish {
        /// Topic name to publish to.
        #[arg(long)]
        topic: String,
        /// One row per `--row` flag. JSON object syntax.
        #[arg(long = "row", value_name = "JSON", group = "publish_input")]
        rows: Vec<String>,
        /// Path to a JSON file containing either an array of row objects or
        /// a single row object. Mutually exclusive with `--row`.
        #[arg(long, value_name = "PATH", group = "publish_input")]
        json_file: Option<PathBuf>,
    },

    /// Subscribe and print every delivered batch as one JSON object per
    /// row. With `--no-follow`, the server drains only the replay window and
    /// closes the stream; otherwise runs until interrupted with Ctrl-C.
    Subscribe {
        /// Topic name to subscribe to.
        #[arg(long)]
        topic: String,
        /// Optional SQL predicate; an empty string matches every row.
        #[arg(long, default_value = "")]
        predicate: String,
        /// Replay starting at this offset. `0` replays from the earliest
        /// retained event; omit to start at the live tail.
        #[arg(long)]
        from_offset: Option<u64>,
        /// Drain only the replay window from the server and exit instead of
        /// attaching to the live broker tail.
        #[arg(long)]
        no_follow: bool,
    },
}

pub async fn run(
    session: &Session,
    action: TriggerAction,
) -> Result<(), Box<dyn std::error::Error>> {
    match action {
        TriggerAction::List => list_topics(session).await,
        TriggerAction::Register {
            name,
            schema,
            broker_metadata,
        } => register_topic(session, &name, &schema, &broker_metadata).await,
        TriggerAction::Publish {
            topic,
            rows,
            json_file,
        } => publish_rows(session, &topic, &rows, json_file.as_deref()).await,
        TriggerAction::Subscribe {
            topic,
            predicate,
            from_offset,
            no_follow,
        } => subscribe_topic(session, &topic, &predicate, from_offset, no_follow).await,
    }
}

async fn list_topics(session: &Session) -> Result<(), Box<dyn std::error::Error>> {
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
    session: &Session,
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

/// Resolve a topic by name from the server's `list_topics`, erroring when no
/// topic with that name is visible to the session's tenant. Publish and
/// subscribe both key off the same resolution so the schema they build a batch
/// against (or filter with) is the server's registered schema.
async fn resolve_topic(
    session: &Session,
    name: &str,
) -> Result<TopicDefinition, Box<dyn std::error::Error>> {
    session
        .list_topics()
        .await?
        .into_iter()
        .find(|t| t.name == name)
        .ok_or_else(|| format!("topic '{name}' not found").into())
}

async fn publish_rows(
    session: &Session,
    name: &str,
    rows: &[String],
    json_file: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let topic = resolve_topic(session, name).await?;

    let batch = match json_file {
        Some(path) => load_rows_from_file(path, &topic.schema)?,
        None => {
            // ArgGroup enforces that one of `rows` / `json_file` is set, so
            // this branch always sees a non-empty rows vector. The empty
            // check is a defensive belt-and-braces.
            if rows.is_empty() {
                return Err("at least one --row argument is required".into());
            }
            let parsed_rows = parse_row_strings(rows)?;
            build_batch(&topic.schema, &parsed_rows)?
        }
    };

    let offset = session.publish(&topic, batch).await?;
    println!("Published offset {}.", offset.value());
    Ok(())
}

async fn subscribe_topic(
    session: &Session,
    name: &str,
    predicate: &str,
    from_offset: Option<u64>,
    no_follow: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let topic = resolve_topic(session, name).await?;
    // The predicate is parsed to SQL against the topic schema using a local
    // empty `SessionContext` — a parse, not a catalog touch — then crosses the
    // wire as that SQL; the server re-parses it against the same schema.
    let ctx = SessionContext::new();
    let pred = Predicate::from_sql(&ctx, Arc::clone(&topic.schema), predicate)?;
    let start = from_offset.map(|v| Offset::new(v, chrono::Utc::now()));

    if !no_follow {
        eprintln!("Listening on topic '{}'. Ctrl-C to exit.", topic.name);
    }
    // `no_follow` selects the server's finite replay-only drain: the stream
    // yields the retained replay window and then ends, so the loop terminates.
    // Without it the stream tails live until the process is interrupted.
    let mut stream = session.subscribe(&topic, pred, start, no_follow).await?;
    while let Some(item) = stream.next().await {
        let delivered = item?;
        emit_delivered(&delivered)?;
    }
    Ok(())
}

/// Render one delivered batch as one JSON object per row to stdout.
fn emit_delivered(delivered: &DeliveredBatch) -> Result<(), Box<dyn std::error::Error>> {
    let rows = batch_to_json_rows(&delivered.batch)?;
    for row in rows {
        println!(
            "{}",
            serde_json::json!({
                "offset": delivered.offset.value(),
                "produced_at_us": delivered.produced_at.timestamp_micros(),
                "row": row,
            })
        );
    }
    Ok(())
}

/// Parse each `--row` JSON string into a row object. Each entry must be a
/// JSON object whose keys match the topic schema.
fn parse_row_strings(
    rows: &[String],
) -> Result<Vec<serde_json::Map<String, serde_json::Value>>, Box<dyn std::error::Error>> {
    rows.iter()
        .map(|s| -> Result<_, Box<dyn std::error::Error>> {
            let value: serde_json::Value = serde_json::from_str(s)?;
            value
                .as_object()
                .cloned()
                .ok_or_else(|| "each --row must be a JSON object".into())
        })
        .collect()
}

/// Read a JSON file containing either an array of row objects or a single
/// row object, validate every row is a JSON object, and build a
/// `RecordBatch` matching `schema`.
pub(crate) fn load_rows_from_file(
    path: &Path,
    schema: &SchemaRef,
) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("read json file {}: {e}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| format!("parse json file {}: {e}", path.display()))?;
    let rows: Vec<serde_json::Map<String, serde_json::Value>> = match value {
        serde_json::Value::Array(items) => items
            .into_iter()
            .map(|v| -> Result<_, Box<dyn std::error::Error>> {
                v.as_object().cloned().ok_or_else(|| {
                    format!(
                        "json file {} contains an array element that is not a JSON object",
                        path.display()
                    )
                    .into()
                })
            })
            .collect::<Result<_, _>>()?,
        serde_json::Value::Object(obj) => vec![obj],
        _ => {
            return Err(format!(
                "json file {} must be a JSON object or an array of JSON objects",
                path.display()
            )
            .into());
        }
    };
    if rows.is_empty() {
        return Err(format!("json file {} contains zero rows", path.display()).into());
    }
    build_batch(schema, &rows)
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

fn build_batch(
    schema: &SchemaRef,
    rows: &[serde_json::Map<String, serde_json::Value>],
) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let mut columns: Vec<arrow::array::ArrayRef> = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        let name = field.name();
        match field.data_type() {
            DataType::Int64 => {
                let values: Vec<Option<i64>> = rows
                    .iter()
                    .map(|r| r.get(name).and_then(|v| v.as_i64()))
                    .collect();
                columns.push(Arc::new(Int64Array::from(values)));
            }
            DataType::Float64 => {
                let values: Vec<Option<f64>> = rows
                    .iter()
                    .map(|r| r.get(name).and_then(|v| v.as_f64()))
                    .collect();
                columns.push(Arc::new(Float64Array::from(values)));
            }
            DataType::Utf8 => {
                let values: Vec<Option<String>> = rows
                    .iter()
                    .map(|r| r.get(name).and_then(|v| v.as_str()).map(str::to_string))
                    .collect();
                columns.push(Arc::new(StringArray::from(values)));
            }
            DataType::Boolean => {
                let values: Vec<Option<bool>> = rows
                    .iter()
                    .map(|r| r.get(name).and_then(|v| v.as_bool()))
                    .collect();
                columns.push(Arc::new(BooleanArray::from(values)));
            }
            other => return Err(format!("CLI does not yet build column type {other:?}").into()),
        }
    }
    Ok(RecordBatch::try_new(Arc::clone(schema), columns)?)
}

fn batch_to_json_rows(
    batch: &RecordBatch,
) -> Result<Vec<serde_json::Map<String, serde_json::Value>>, Box<dyn std::error::Error>> {
    let n = batch.num_rows();
    let mut rows: Vec<serde_json::Map<String, serde_json::Value>> =
        (0..n).map(|_| serde_json::Map::new()).collect();
    for (col_idx, field) in batch.schema().fields().iter().enumerate() {
        let column = batch.column(col_idx);
        for (row_idx, row) in rows.iter_mut().enumerate() {
            let value = column_value_to_json(field.data_type(), column.as_ref(), row_idx);
            row.insert(field.name().clone(), value);
        }
    }
    Ok(rows)
}

fn column_value_to_json(ty: &DataType, column: &dyn Array, row_idx: usize) -> serde_json::Value {
    if column.is_null(row_idx) {
        return serde_json::Value::Null;
    }
    match ty {
        DataType::Int64 => column
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(|a| serde_json::Value::from(a.value(row_idx)))
            .unwrap_or(serde_json::Value::Null),
        DataType::Float64 => column
            .as_any()
            .downcast_ref::<Float64Array>()
            .and_then(|a| serde_json::Number::from_f64(a.value(row_idx)))
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        DataType::Utf8 => column
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|a| serde_json::Value::String(a.value(row_idx).to_string()))
            .unwrap_or(serde_json::Value::Null),
        DataType::Boolean => column
            .as_any()
            .downcast_ref::<BooleanArray>()
            .map(|a| serde_json::Value::Bool(a.value(row_idx)))
            .unwrap_or(serde_json::Value::Null),
        _ => serde_json::Value::Null,
    }
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
    use tempfile::TempDir;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("op", DataType::Utf8, false),
            Field::new("ts_ms", DataType::Int64, false),
            Field::new("key", DataType::Utf8, false),
        ]))
    }

    #[test]
    fn load_rows_from_file_accepts_array() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("rows.json");
        std::fs::write(
            &path,
            r#"[
                {"op":"c","ts_ms":1,"key":"a"},
                {"op":"u","ts_ms":2,"key":"a"}
            ]"#,
        )
        .unwrap();
        let batch = load_rows_from_file(&path, &schema()).unwrap();
        assert_eq!(batch.num_rows(), 2);
    }

    #[test]
    fn load_rows_from_file_accepts_single_object() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("rows.json");
        std::fs::write(&path, r#"{"op":"c","ts_ms":1,"key":"a"}"#).unwrap();
        let batch = load_rows_from_file(&path, &schema()).unwrap();
        assert_eq!(batch.num_rows(), 1);
    }

    #[test]
    fn load_rows_from_file_rejects_malformed_json() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("rows.json");
        std::fs::write(&path, "not json").unwrap();
        let err = load_rows_from_file(&path, &schema()).unwrap_err();
        assert!(err.to_string().contains("parse json file"));
    }

    #[test]
    fn load_rows_from_file_rejects_scalar() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("rows.json");
        std::fs::write(&path, "42").unwrap();
        let err = load_rows_from_file(&path, &schema()).unwrap_err();
        assert!(err.to_string().contains("JSON object"));
    }

    #[test]
    fn load_rows_from_file_rejects_empty_array() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("rows.json");
        std::fs::write(&path, "[]").unwrap();
        let err = load_rows_from_file(&path, &schema()).unwrap_err();
        assert!(err.to_string().contains("zero rows"));
    }
}
