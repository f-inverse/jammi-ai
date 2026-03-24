use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

use jammi_engine::catalog::Catalog;
use jammi_engine::error::{JammiError, Result};

/// Build the dynamic output schema for evidence queries based on which
/// channels participated. Only participating channels contribute fields.
///
/// Base fields (always present):
/// - `_row_id` (Utf8)
/// - `_source_id` (Utf8)
/// - `retrieved_by` (`List<Utf8>`)
/// - `annotated_by` (`List<Utf8>`)
///
/// Channel fields (added per channel):
/// - vector: `similarity` (Float32)
/// - inference: `inference_model` (Utf8), `inference_task` (Utf8), `inference_confidence` (Float32)
pub fn evidence_schema(channels: &[&str], catalog: &Catalog) -> Result<SchemaRef> {
    let list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));

    let mut fields = vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("_source_id", DataType::Utf8, false),
        Field::new("retrieved_by", list_type.clone(), false),
        Field::new("annotated_by", list_type, false),
    ];

    for channel in channels {
        let record = catalog.get_evidence_channel(channel)?;
        let channel_fields = parse_fields_json(&record.schema_json)?;
        fields.extend(channel_fields);
    }

    Ok(Arc::new(Schema::new(fields)))
}

/// Parse a JSON object mapping field names to Arrow type names into Arrow Fields.
///
/// Format: `{"similarity": "Float32"}` → `[Field::new("similarity", DataType::Float32, true)]`
///
/// Supported type names: Float32, Float64, Utf8, Int32, Int64, Boolean.
fn parse_fields_json(json: &str) -> Result<Vec<Field>> {
    let map: serde_json::Map<String, serde_json::Value> = serde_json::from_str(json)
        .map_err(|e| JammiError::Other(format!("Invalid evidence channel schema JSON: {e}")))?;

    map.iter()
        .map(|(name, type_val)| {
            let type_str = type_val
                .as_str()
                .ok_or_else(|| JammiError::Other(format!("Field type must be string: {name}")))?;
            let data_type = match type_str {
                "Float32" => DataType::Float32,
                "Float64" => DataType::Float64,
                "Utf8" => DataType::Utf8,
                "Int32" => DataType::Int32,
                "Int64" => DataType::Int64,
                "Boolean" => DataType::Boolean,
                other => {
                    return Err(JammiError::Other(format!(
                        "Unsupported evidence field type '{other}' for field '{name}'"
                    )))
                }
            };
            Ok(Field::new(name, data_type, true))
        })
        .collect()
}
