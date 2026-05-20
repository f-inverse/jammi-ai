//! Types for declaring and registering mutable companion tables.

use std::fmt;
use std::str::FromStr;

use arrow_schema::{DataType, SchemaRef};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::catalog::backend::BackendError;
use crate::tenant::TenantId;

/// Catalog-unique identifier for one mutable companion table.
///
/// Lowercase ASCII letters, digits, and `_`, length 1..=63. The newtype
/// refuses any other shape at the API boundary, so the name is always safe to
/// interpolate into backend DDL after dialect-specific quoting.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MutableTableId(String);

impl MutableTableId {
    /// Construct a new identifier, validating its shape.
    pub fn new(name: impl Into<String>) -> Result<Self, MutableTableError> {
        let s = name.into();
        if s.is_empty() {
            return Err(MutableTableError::InvalidId("empty".to_string()));
        }
        if s.len() > 63 {
            return Err(MutableTableError::InvalidId(format!(
                "length must be 1..=63 (got {})",
                s.len()
            )));
        }
        if !s
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        {
            return Err(MutableTableError::InvalidId(format!(
                "only lowercase ASCII, digits, and `_` allowed; got: {s}"
            )));
        }
        Ok(MutableTableId(s))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for MutableTableId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for MutableTableId {
    type Err = MutableTableError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

/// Secondary index declaration: one `CREATE INDEX` per entry.
#[derive(Debug, Clone)]
pub struct MutableIndexDef {
    pub name: String,
    pub columns: Vec<String>,
    pub unique: bool,
}

/// Registration descriptor — everything needed to construct the storage
/// table, the catalog row, and the DataFusion `TableProvider`.
///
/// Construct via [`MutableTableDefinitionBuilder`]; the builder validates the
/// schema, primary-key membership, reserved column rules, and `order_column`
/// type.
#[derive(Debug, Clone)]
pub struct MutableTableDefinition {
    pub id: MutableTableId,
    pub schema: SchemaRef,
    /// Column(s) forming the primary key. Must be a non-empty subset of
    /// `schema.fields()`.
    pub primary_key: Vec<String>,
    /// Optional tenant scope. Stored on the catalog row; Phase 3 wires the
    /// predicate-injection layer that reads it.
    pub tenant: Option<TenantId>,
    /// Secondary indexes; one `CREATE INDEX` per entry inside the same
    /// transaction as the `CREATE TABLE`.
    pub indexes: Vec<MutableIndexDef>,
    /// Free-form tenant-owned metadata. The engine stores it but does not
    /// parse it.
    pub user_metadata: serde_json::Value,
    /// Optional monotonic ordering column for `MutableTableRegistry::scan_after`.
    /// Must exist in `schema` and be `Int64` or `UInt64`. Required for backing
    /// a Phase-4 trigger-stream topic; optional otherwise.
    pub order_column: Option<String>,
    /// Chunk size for streaming reads. Default 8192.
    pub chunk_size: usize,
}

/// Builder with build-time validation.
#[derive(Debug)]
pub struct MutableTableDefinitionBuilder {
    id: MutableTableId,
    schema: SchemaRef,
    primary_key: Vec<String>,
    tenant: Option<TenantId>,
    indexes: Vec<MutableIndexDef>,
    user_metadata: serde_json::Value,
    order_column: Option<String>,
    chunk_size: usize,
}

impl MutableTableDefinitionBuilder {
    pub fn new(id: MutableTableId, schema: SchemaRef) -> Self {
        Self {
            id,
            schema,
            primary_key: vec![],
            tenant: None,
            indexes: vec![],
            user_metadata: serde_json::Value::Object(serde_json::Map::new()),
            order_column: None,
            chunk_size: 8192,
        }
    }

    pub fn primary_key(mut self, columns: Vec<String>) -> Self {
        self.primary_key = columns;
        self
    }

    pub fn tenant(mut self, tenant: Option<TenantId>) -> Self {
        self.tenant = tenant;
        self
    }

    pub fn index(mut self, idx: MutableIndexDef) -> Self {
        self.indexes.push(idx);
        self
    }

    pub fn user_metadata(mut self, value: serde_json::Value) -> Self {
        self.user_metadata = value;
        self
    }

    pub fn order_column(mut self, column: impl Into<String>) -> Self {
        self.order_column = Some(column.into());
        self
    }

    pub fn chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    pub fn build(self) -> Result<MutableTableDefinition, MutableTableError> {
        if self.primary_key.is_empty() {
            return Err(MutableTableError::Schema(
                "primary key must not be empty".to_string(),
            ));
        }

        let field_names: std::collections::HashSet<&str> = self
            .schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();

        for pk in &self.primary_key {
            if !field_names.contains(pk.as_str()) {
                return Err(MutableTableError::MissingPrimaryKey(pk.clone()));
            }
        }

        for f in self.schema.fields() {
            if f.name() == "tenant_id" {
                return Err(MutableTableError::ReservedColumn(f.name().clone()));
            }
            if f.name().starts_with('_') {
                return Err(MutableTableError::ReservedColumn(f.name().clone()));
            }
        }

        if let Some(ref col) = self.order_column {
            let field = self.schema.field_with_name(col).map_err(|_| {
                MutableTableError::Schema(format!("order_column '{col}' not in schema"))
            })?;
            if !matches!(field.data_type(), DataType::Int64 | DataType::UInt64) {
                return Err(MutableTableError::Schema(format!(
                    "order_column '{col}' must be Int64 or UInt64, got {:?}",
                    field.data_type()
                )));
            }
        }

        Ok(MutableTableDefinition {
            id: self.id,
            schema: self.schema,
            primary_key: self.primary_key,
            tenant: self.tenant,
            indexes: self.indexes,
            user_metadata: self.user_metadata,
            order_column: self.order_column,
            chunk_size: self.chunk_size,
        })
    }
}

/// Mutable-table error taxonomy.
#[derive(Debug, Error)]
pub enum MutableTableError {
    #[error("invalid mutable table id: {0}")]
    InvalidId(String),
    #[error("schema validation: {0}")]
    Schema(String),
    #[error("primary key column not present in schema: {0}")]
    MissingPrimaryKey(String),
    #[error("reserved column name: {0}")]
    ReservedColumn(String),
    #[error("mutable table not found: {0}")]
    NotFound(MutableTableId),
    #[error("mutable table already exists: {0}")]
    AlreadyExists(MutableTableId),
    #[error("table has no order_column declared; required by scan_after")]
    NoOrderColumn,
    #[error("backend: {0}")]
    Backend(#[from] BackendError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{Field, Schema};
    use std::sync::Arc;

    fn schema_with(cols: &[(&str, DataType)]) -> SchemaRef {
        let fields: Vec<Field> = cols
            .iter()
            .map(|(name, ty)| Field::new(*name, ty.clone(), false))
            .collect();
        Arc::new(Schema::new(fields))
    }

    #[test]
    fn id_accepts_lowercase_digits_underscore() {
        assert!(MutableTableId::new("ab1_c2").is_ok());
        assert!(MutableTableId::new("a").is_ok());
        assert!(MutableTableId::new("123").is_ok());
    }

    #[test]
    fn id_rejects_uppercase_or_special() {
        assert!(MutableTableId::new("Foo").is_err());
        assert!(MutableTableId::new("foo-bar").is_err());
        assert!(MutableTableId::new("foo bar").is_err());
        assert!(MutableTableId::new("").is_err());
        let too_long = "a".repeat(64);
        assert!(MutableTableId::new(too_long).is_err());
    }

    #[test]
    fn builder_rejects_empty_primary_key() {
        let id = MutableTableId::new("t").unwrap();
        let s = schema_with(&[("a", DataType::Int64)]);
        let err = MutableTableDefinitionBuilder::new(id, s)
            .build()
            .unwrap_err();
        assert!(matches!(err, MutableTableError::Schema(_)));
    }

    #[test]
    fn builder_rejects_pk_not_in_schema() {
        let id = MutableTableId::new("t").unwrap();
        let s = schema_with(&[("a", DataType::Int64)]);
        let err = MutableTableDefinitionBuilder::new(id, s)
            .primary_key(vec!["missing".to_string()])
            .build()
            .unwrap_err();
        assert!(matches!(err, MutableTableError::MissingPrimaryKey(_)));
    }

    #[test]
    fn builder_rejects_reserved_tenant_id_column() {
        let id = MutableTableId::new("t").unwrap();
        let s = schema_with(&[("a", DataType::Int64), ("tenant_id", DataType::Utf8)]);
        let err = MutableTableDefinitionBuilder::new(id, s)
            .primary_key(vec!["a".to_string()])
            .build()
            .unwrap_err();
        assert!(matches!(err, MutableTableError::ReservedColumn(_)));
    }

    #[test]
    fn builder_rejects_underscore_prefixed_column() {
        let id = MutableTableId::new("t").unwrap();
        let s = schema_with(&[("a", DataType::Int64), ("_rowid", DataType::Int64)]);
        let err = MutableTableDefinitionBuilder::new(id, s)
            .primary_key(vec!["a".to_string()])
            .build()
            .unwrap_err();
        assert!(matches!(err, MutableTableError::ReservedColumn(_)));
    }

    #[test]
    fn builder_rejects_order_column_with_wrong_type() {
        let id = MutableTableId::new("t").unwrap();
        let s = schema_with(&[("a", DataType::Int64), ("seq", DataType::Float32)]);
        let err = MutableTableDefinitionBuilder::new(id, s)
            .primary_key(vec!["a".to_string()])
            .order_column("seq")
            .build()
            .unwrap_err();
        assert!(matches!(err, MutableTableError::Schema(_)));
    }

    #[test]
    fn builder_happy_path_with_index_order_and_tenant() {
        let id = MutableTableId::new("dim").unwrap();
        let s = schema_with(&[
            ("k", DataType::Int64),
            ("v", DataType::Utf8),
            ("seq", DataType::Int64),
        ]);
        let t = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap();
        let def = MutableTableDefinitionBuilder::new(id, s)
            .primary_key(vec!["k".to_string()])
            .tenant(Some(t))
            .index(MutableIndexDef {
                name: "idx_v".into(),
                columns: vec!["v".into()],
                unique: false,
            })
            .order_column("seq")
            .build()
            .unwrap();
        assert_eq!(def.primary_key, vec!["k".to_string()]);
        assert_eq!(def.indexes.len(), 1);
        assert_eq!(def.tenant, Some(t));
        assert_eq!(def.order_column.as_deref(), Some("seq"));
    }
}
