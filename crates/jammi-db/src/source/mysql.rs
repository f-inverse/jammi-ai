//! MySQL source provider via `datafusion-table-providers`.
//!
//! Gated behind the `mysql` feature flag.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::catalog::TableProvider;
use datafusion::common::TableReference;
use datafusion_table_providers::mysql::MySQLTableFactory;
use datafusion_table_providers::sql::db_connection_pool::mysqlpool::MySQLConnectionPool;
use secrecy::SecretString;

use crate::error::{JammiError, Result};
use crate::source::SourceConnection;

/// Create table providers for all tables in a MySQL database.
pub async fn create_mysql_tables(
    source_id: &str,
    connection: &SourceConnection,
) -> Result<Vec<(String, Arc<dyn TableProvider>)>> {
    let url = connection
        .url
        .as_deref()
        .ok_or_else(|| JammiError::Source {
            source_id: source_id.into(),
            message: "MySQL source requires a connection URL".into(),
        })?;

    let mut params = HashMap::new();
    params.insert(
        "connection_string".to_string(),
        SecretString::from(url.to_string()),
    );
    for (k, v) in &connection.options {
        params.insert(k.clone(), SecretString::from(v.clone()));
    }

    let pool = MySQLConnectionPool::new(params)
        .await
        .map_err(|e| JammiError::Source {
            source_id: source_id.into(),
            message: format!("Failed to connect to MySQL: {e}"),
        })?;
    let pool = Arc::new(pool);
    let factory = MySQLTableFactory::new(pool);

    // MySQL: discover tables from the connection's default database
    let table_names = discover_mysql_tables(source_id, url).await?;
    let mut tables = Vec::new();
    for name in table_names {
        let table_ref = TableReference::bare(&name);
        let provider = factory
            .table_provider(table_ref)
            .await
            .map_err(|e| JammiError::Source {
                source_id: source_id.into(),
                message: format!("Failed to create MySQL table provider for '{name}': {e}"),
            })?;
        tables.push((name, provider));
    }

    Ok(tables)
}

async fn discover_mysql_tables(source_id: &str, url: &str) -> Result<Vec<String>> {
    let opts = mysql_async::Opts::from_url(url).map_err(|e| JammiError::Source {
        source_id: source_id.into(),
        message: format!("Invalid MySQL URL: {e}"),
    })?;
    let pool = mysql_async::Pool::new(opts);

    use mysql_async::prelude::Queryable;
    let mut conn = pool.get_conn().await.map_err(|e| JammiError::Source {
        source_id: source_id.into(),
        message: format!("MySQL connection for table discovery failed: {e}"),
    })?;

    let rows: Vec<String> = conn
        .query("SHOW TABLES")
        .await
        .map_err(|e| JammiError::Source {
            source_id: source_id.into(),
            message: format!("Failed to discover MySQL tables: {e}"),
        })?;

    pool.disconnect().await.ok();
    Ok(rows)
}
