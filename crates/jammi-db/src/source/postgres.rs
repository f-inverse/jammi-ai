//! Postgres source provider via `datafusion-table-providers`.
//!
//! Gated behind the `postgres` feature flag.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::catalog::TableProvider;
use datafusion::common::TableReference;
use datafusion_table_providers::postgres::PostgresTableFactory;
use datafusion_table_providers::sql::db_connection_pool::postgrespool::PostgresConnectionPool;
use secrecy::SecretString;

use crate::error::{JammiError, Result};
use crate::source::SourceConnection;

/// Create table providers for all public tables in a Postgres database.
///
/// Returns `(table_name, Arc<dyn TableProvider>)` pairs with filter/projection/limit
/// pushdown handled automatically by `datafusion-table-providers`.
pub async fn create_postgres_tables(
    source_id: &str,
    connection: &SourceConnection,
) -> Result<Vec<(String, Arc<dyn TableProvider>)>> {
    let url = connection
        .url
        .as_deref()
        .ok_or_else(|| JammiError::Source {
            source_id: source_id.into(),
            message: "Postgres source requires a connection URL".into(),
        })?;

    let mut params = HashMap::new();
    params.insert(
        "connection_string".to_string(),
        SecretString::from(url.to_string()),
    );
    for (k, v) in &connection.options {
        params.insert(k.clone(), SecretString::from(v.clone()));
    }

    let pool = PostgresConnectionPool::new(params)
        .await
        .map_err(|e| JammiError::Source {
            source_id: source_id.into(),
            message: format!("Failed to connect to Postgres: {e}"),
        })?;
    let pool = Arc::new(pool);
    let factory = PostgresTableFactory::new(pool);

    let table_names = discover_table_names(source_id, url).await?;
    let mut tables = Vec::new();
    for name in table_names {
        let table_ref = TableReference::bare(&name);
        let provider = factory
            .table_provider(table_ref)
            .await
            .map_err(|e| JammiError::Source {
                source_id: source_id.into(),
                message: format!("Failed to create Postgres table provider for '{name}': {e}"),
            })?;
        tables.push((name, provider));
    }

    Ok(tables)
}

/// Discover table names from information_schema via a lightweight query.
async fn discover_table_names(source_id: &str, url: &str) -> Result<Vec<String>> {
    let (client, conn) = tokio_postgres::connect(url, tokio_postgres::NoTls)
        .await
        .map_err(|e| JammiError::Source {
            source_id: source_id.into(),
            message: format!("Postgres connection for table discovery failed: {e}"),
        })?;

    tokio::spawn(async move {
        if let Err(e) = conn.await {
            tracing::error!("Postgres background connection error: {e}");
        }
    });

    let rows = client
        .query(
            "SELECT table_name FROM information_schema.tables \
             WHERE table_schema = 'public' AND table_type = 'BASE TABLE' \
             ORDER BY table_name",
            &[],
        )
        .await
        .map_err(|e| JammiError::Source {
            source_id: source_id.into(),
            message: format!("Failed to query information_schema: {e}"),
        })?;

    Ok(rows.iter().map(|r| r.get::<_, String>(0)).collect())
}
