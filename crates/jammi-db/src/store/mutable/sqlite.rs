//! SQLite implementation of [`MutableBackend`].

use std::sync::Arc;

use arrow_schema::DataType;

use crate::catalog::backend::BackendImpl;
use crate::sql::quote_ident;

use super::definition::{MutableIndexDef, MutableTableDefinition};
use super::MutableBackend;

pub struct SqliteMutableBackend {
    backend: Arc<BackendImpl>,
}

impl SqliteMutableBackend {
    pub fn new(backend: Arc<BackendImpl>) -> Self {
        Self { backend }
    }
}

impl MutableBackend for SqliteMutableBackend {
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
                    sqlite_type(f.data_type()),
                    nullable
                )
            })
            .collect();
        // ADR-00: always emit the tenant_id column.
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
        format!("DROP TABLE IF EXISTS {}", quote_ident(def.id.as_str()))
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

/// Map an Arrow `DataType` to a SQLite column type name. Only the subset
/// Phase 2 supports; richer types live in future migrations.
fn sqlite_type(ty: &DataType) -> &'static str {
    match ty {
        DataType::Boolean => "INTEGER",
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => "INTEGER",
        DataType::Float16 | DataType::Float32 | DataType::Float64 => "REAL",
        DataType::Utf8 | DataType::LargeUtf8 => "TEXT",
        DataType::Binary | DataType::LargeBinary => "BLOB",
        DataType::Date32 | DataType::Date64 | DataType::Timestamp(_, _) => "TEXT",
        _ => "BLOB",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::mutable::definition::{MutableTableDefinitionBuilder, MutableTableId};
    use arrow_schema::{Field, Schema};
    use std::sync::Arc as StdArc;

    fn def() -> MutableTableDefinition {
        let schema = StdArc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        MutableTableDefinitionBuilder::new(MutableTableId::new("widgets").unwrap(), schema)
            .primary_key(vec!["id".into()])
            .build()
            .unwrap()
    }

    fn backend() -> SqliteMutableBackend {
        // The mutable backend exposes only string renderers in tests; the
        // BackendImpl isn't dereferenced for DDL/DML rendering. We construct
        // it from a real sqlite pool just to satisfy the constructor.
        let rt = tokio::runtime::Runtime::new().unwrap();
        let tempdir = tempfile::tempdir().unwrap();
        let path = tempdir.path().join("t.db");
        let sb = rt
            .block_on(crate::catalog::backend_sqlite::SqliteBackend::open(&path))
            .unwrap();
        std::mem::forget(tempdir); // outlive the test (test process drops at exit).
        SqliteMutableBackend::new(StdArc::new(BackendImpl::Sqlite(sb)))
    }

    #[test]
    fn create_table_ddl_emits_implicit_tenant_id() {
        let b = backend();
        let ddl = b.create_table_ddl(&def());
        assert!(ddl.contains("\"tenant_id\" TEXT") || ddl.contains("tenant_id TEXT"));
        assert!(ddl.contains("PRIMARY KEY (\"id\")"));
        assert!(ddl.starts_with("CREATE TABLE \"widgets\""));
    }

    #[test]
    fn insert_dml_placeholder_count_matches_rows_times_cols_plus_tenant() {
        let b = backend();
        let cols = ["id", "name"];
        let dml = b.insert_dml(&def(), &cols, 3);
        // 3 rows * (2 user cols + 1 tenant_id) = 9 placeholders
        let placeholders: Vec<_> = (1..=9).map(|i| format!("${i}")).collect();
        for p in &placeholders {
            assert!(dml.contains(p), "missing {p} in {dml}");
        }
        assert!(!dml.contains("$10"), "extra placeholder in {dml}");
    }

    #[test]
    fn drop_table_ddl_uses_if_exists_no_cascade() {
        let b = backend();
        let ddl = b.drop_table_ddl(&def());
        assert_eq!(ddl, r#"DROP TABLE IF EXISTS "widgets""#);
    }

    #[test]
    fn quote_ident_neutralises_injection_payload() {
        assert_eq!(quote_ident(r#"a"b"#), r#""a""b""#);
    }
}
