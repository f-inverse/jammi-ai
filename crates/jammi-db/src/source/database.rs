//! What the database sources share: the URL a database source connects
//! through, and one table provider per table of a schema, listed through the
//! source's own connection pool — so discovery connects exactly as the
//! providers do, with the same TLS mode, host and credentials.

use std::error::Error;
use std::future::Future;
use std::sync::Arc;

use datafusion::catalog::TableProvider;
use datafusion::common::TableReference;
use datafusion_table_providers::sql::db_connection_pool::dbconnection::get_tables;
use datafusion_table_providers::sql::db_connection_pool::DbConnectionPool;

use crate::error::{JammiError, Result};
use crate::source::SourceConnection;

/// The connection URL of a database source, which it cannot connect without.
pub(crate) fn url<'a>(source_id: &str, connection: &'a SourceConnection) -> Result<&'a str> {
    connection.url.as_deref().ok_or_else(|| JammiError::Source {
        source_id: source_id.into(),
        message: "a database source requires a connection URL".into(),
    })
}

/// A provider for every table of `schema`, each built by `provider`.
pub(crate) async fn schema_tables<T, P, F, Fut>(
    source_id: &str,
    pool: &(dyn DbConnectionPool<T, P> + Send + Sync),
    schema: &str,
    provider: F,
) -> Result<Vec<(String, Arc<dyn TableProvider>)>>
where
    T: 'static,
    P: 'static,
    F: Fn(TableReference) -> Fut,
    Fut: Future<Output = std::result::Result<Arc<dyn TableProvider>, Box<dyn Error + Send + Sync>>>,
{
    let failed = |message: String| JammiError::Source {
        source_id: source_id.into(),
        message,
    };
    let connection = pool
        .connect()
        .await
        .map_err(|e| failed(format!("connecting to list the tables of `{schema}`: {e}")))?;
    let mut names = get_tables(connection, schema)
        .await
        .map_err(|e| failed(format!("listing the tables of `{schema}`: {e}")))?;
    names.sort();
    let mut tables = Vec::with_capacity(names.len());
    for name in names {
        let table = provider(TableReference::bare(name.as_str()))
            .await
            .map_err(|e| failed(format!("building the provider for table `{name}`: {e}")))?;
        tables.push((name, table));
    }
    Ok(tables)
}
