//! Postgres implementation of [`MutableBackend`].

use std::sync::Arc;

use arrow_schema::DataType;

use crate::catalog::backend::BackendImpl;
use crate::sql::quote_ident;

use super::definition::{MutableIndexDef, MutableTableDefinition};
use super::MutableBackend;

pub struct PostgresMutableBackend {
    backend: Arc<BackendImpl>,
}

impl PostgresMutableBackend {
    pub fn new(backend: Arc<BackendImpl>) -> Self {
        Self { backend }
    }
}

impl MutableBackend for PostgresMutableBackend {
    fn create_table_ddl(&self, def: &MutableTableDefinition) -> String {
        let mut cols: Vec<String> = def
            .schema
            .fields()
            .iter()
            .map(|f| {
                let nullable = if f.is_nullable() { "" } else { " NOT NULL" };
                format!(
                    "{} {}{}",
                    quote_ident(f.name()),
                    pg_type(f.data_type()),
                    nullable
                )
            })
            .collect();
        // Stored as TEXT to match migration 005's catalog convention; the
        // sink and tenant-predicate paths emit stringified UUIDs.
        cols.push("tenant_id TEXT".to_string());

        let pk = def
            .primary_key
            .iter()
            .map(|c| quote_ident(c))
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            "CREATE TABLE {} ({}, PRIMARY KEY ({}))",
            quote_ident(def.id.as_str()),
            cols.join(", "),
            pk
        )
    }

    fn create_index_ddl(&self, def: &MutableTableDefinition, idx: &MutableIndexDef) -> String {
        let unique = if idx.unique { "UNIQUE " } else { "" };
        let cols = idx
            .columns
            .iter()
            .map(|c| quote_ident(c))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "CREATE {}INDEX {} ON {}({})",
            unique,
            quote_ident(&idx.name),
            quote_ident(def.id.as_str()),
            cols
        )
    }

    fn drop_table_ddl(&self, def: &MutableTableDefinition) -> String {
        format!(
            "DROP TABLE IF EXISTS {} CASCADE",
            quote_ident(def.id.as_str())
        )
    }

    fn insert_dml(&self, def: &MutableTableDefinition, columns: &[&str], n_rows: usize) -> String {
        let mut all_cols: Vec<String> = columns.iter().map(|c| quote_ident(c)).collect();
        all_cols.push("tenant_id".to_string());
        let cols_clause = all_cols.join(", ");

        let per_row = all_cols.len();
        let mut row_clauses: Vec<String> = Vec::with_capacity(n_rows);
        for r in 0..n_rows {
            let placeholders: Vec<String> = (0..per_row)
                .map(|i| format!("${}", r * per_row + i + 1))
                .collect();
            row_clauses.push(format!("({})", placeholders.join(", ")));
        }

        format!(
            "INSERT INTO {} ({}) VALUES {}",
            quote_ident(def.id.as_str()),
            cols_clause,
            row_clauses.join(", ")
        )
    }

    fn update_dml(
        &self,
        def: &MutableTableDefinition,
        set_columns: &[&str],
        where_predicate: &str,
    ) -> String {
        let set_clause: String = set_columns
            .iter()
            .enumerate()
            .map(|(i, c)| format!("{} = ${}", quote_ident(c), i + 1))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "UPDATE {} SET {} WHERE {}",
            quote_ident(def.id.as_str()),
            set_clause,
            where_predicate
        )
    }

    fn delete_dml(&self, def: &MutableTableDefinition, where_predicate: &str) -> String {
        format!(
            "DELETE FROM {} WHERE {}",
            quote_ident(def.id.as_str()),
            where_predicate
        )
    }

    fn scan_dml(
        &self,
        def: &MutableTableDefinition,
        projection: &[&str],
        predicate: Option<&str>,
        limit: Option<usize>,
    ) -> String {
        let proj = if projection.is_empty() {
            "*".to_string()
        } else {
            projection
                .iter()
                .map(|c| quote_ident(c))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let mut sql = format!("SELECT {} FROM {}", proj, quote_ident(def.id.as_str()));
        if let Some(p) = predicate {
            sql.push_str(" WHERE ");
            sql.push_str(p);
        }
        if let Some(l) = limit {
            sql.push_str(&format!(" LIMIT {l}"));
        }
        sql
    }

    fn catalog_backend(&self) -> &BackendImpl {
        &self.backend
    }
}

fn pg_type(ty: &DataType) -> &'static str {
    match ty {
        DataType::Boolean => "BOOLEAN",
        DataType::Int8 | DataType::Int16 => "SMALLINT",
        DataType::Int32 | DataType::UInt8 | DataType::UInt16 => "INTEGER",
        DataType::Int64 | DataType::UInt32 | DataType::UInt64 => "BIGINT",
        DataType::Float16 | DataType::Float32 => "REAL",
        DataType::Float64 => "DOUBLE PRECISION",
        DataType::Utf8 | DataType::LargeUtf8 => "TEXT",
        DataType::Binary | DataType::LargeBinary => "BYTEA",
        DataType::Timestamp(_, _) => "TIMESTAMPTZ",
        DataType::Date32 | DataType::Date64 => "DATE",
        _ => "JSONB",
    }
}
