//! MySQL source provider via `datafusion-table-providers`.
//!
//! Gated behind the `mysql` feature flag.

use std::collections::HashMap;
use std::error::Error;
use std::future::Future;
use std::sync::Arc;

use datafusion::catalog::TableProvider;
use datafusion::common::TableReference;
use datafusion_table_providers::mysql::sql_table::MySQLTable;
use datafusion_table_providers::sql::db_connection_pool::dbconnection::get_tables;
use datafusion_table_providers::sql::db_connection_pool::mysqlpool::MySQLConnectionPool;
use datafusion_table_providers::sql::db_connection_pool::DbConnectionPool;
use secrecy::SecretString;
use url::Url;

use crate::error::{JammiError, Result};
use crate::federation::remote::{self, Remote};
use crate::source::database::{self, TlsModes};
use crate::source::{SourceConnection, SourceTable};

/// The URL parameter that sets the TLS mode: the driver's own, which
/// `mysql_async` does not parse.
const TLS_MODE: &str = "sslmode";

/// The driver's TLS modes: `required`, which verifies the server's
/// certificate and the host it names, unless the URL names another.
const TLS: TlsModes = TlsModes {
    default: "required",
    verifying: &["required"],
};

/// Create table providers for all tables of the MySQL database the URL names,
/// each handing MySQL only what it runs with the engine's meaning ([`remote`]).
pub async fn create_mysql_tables(
    source_id: &str,
    connection: &SourceConnection,
) -> Result<Vec<(SourceTable, Arc<dyn TableProvider>)>> {
    let failed = |message: String| JammiError::Source {
        source_id: source_id.into(),
        message,
    };
    let ca = match &connection.ca_certificate {
        Some(ca) => Some(database::ca_file(source_id, ca).await?),
        None => None,
    };
    let (params, schema) =
        pool_params(&database::url(source_id, connection)?, ca.as_deref()).map_err(failed)?;
    let pool = MySQLConnectionPool::new(params)
        .await
        .map_err(|e| failed(format!("Failed to connect to MySQL: {e}")))?;
    let pool = Arc::new(pool);
    let pool = &pool;
    let tables = schema_tables(source_id, pool.as_ref(), &schema, |table| async move {
        let mysql = Arc::new(MySQLTable::new(pool, table.clone()).await?);
        Ok(remote::table_provider(
            Remote::MySql,
            table,
            Arc::clone(&mysql) as _,
            mysql,
            None,
        ))
    })
    .await?;
    Ok(tables
        .into_iter()
        .map(|(name, provider)| {
            (
                SourceTable::new(source_id, SourceTable::PUBLIC, &name),
                provider,
            )
        })
        .collect())
}

/// The driver's pool parameters for a `mysql://` URL, and the database whose
/// tables the source serves.
///
/// `mysql_async` parses the URL; its `sslmode` parameter — `disabled`,
/// `preferred` or `required` (the default), the driver's own and unknown to
/// `mysql_async` — is taken out first and handed to the driver beside it. The
/// driver's `preferred` still refuses a server without TLS, so a local server
/// that has none is reached with `sslmode=disabled`; `preferred` does not
/// verify the server's certificate. `ca` is the file the source's CA
/// certificate was written to ([`database::ca_file`]): the driver reads it as
/// `sslrootcert`, and the source connects with a mode that verifies the
/// server ([`TLS`]).
fn pool_params(
    url: &str,
    ca: Option<&std::path::Path>,
) -> std::result::Result<(HashMap<String, SecretString>, String), String> {
    let mut parsed = Url::parse(url).map_err(|e| format!("invalid MySQL URL: {e}"))?;
    let (named, rest): (Vec<_>, Vec<_>) = parsed
        .query_pairs()
        .into_owned()
        .partition(|(key, _)| key == TLS_MODE);
    let mode = TLS.mode(named.last().map(|(_, mode)| mode.as_str()), ca.is_some())?;
    let tls = std::iter::once((TLS_MODE.to_string(), mode))
        .chain(ca.map(|ca| ("sslrootcert".to_string(), ca.display().to_string())));
    parsed.set_query(None);
    if !rest.is_empty() {
        parsed.query_pairs_mut().extend_pairs(&rest);
    }
    let opts = mysql_async::Opts::from_url(parsed.as_str())
        .map_err(|e| format!("invalid MySQL URL: {e}"))?;
    let schema = opts
        .db_name()
        .ok_or("a MySQL source URL names its database: mysql://user@host/<database>")?
        .to_string();
    let params = [("connection_string".to_string(), parsed.to_string())]
        .into_iter()
        .chain(tls)
        .map(|(key, value)| (key, SecretString::from(value)))
        .collect();
    Ok((params, schema))
}

/// A provider for every table of `schema`, each built by `provider`, listed
/// through the source's own connection pool — so discovery connects exactly
/// as the providers do, with the same TLS mode, host and credentials.
async fn schema_tables<T, P, F, Fut>(
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

#[cfg(test)]
mod tests {
    use super::*;
    use secrecy::ExposeSecret;

    fn exposed(params: &HashMap<String, SecretString>) -> HashMap<&str, &str> {
        params
            .iter()
            .map(|(k, v)| (k.as_str(), v.expose_secret()))
            .collect()
    }

    #[test]
    fn the_tls_mode_is_the_drivers_and_the_rest_is_the_url() {
        let (params, schema) = pool_params(
            "mysql://reader:pw@127.0.0.1:3307/shop?sslmode=disabled",
            None,
        )
        .unwrap();
        assert_eq!(schema, "shop");
        assert_eq!(
            exposed(&params),
            HashMap::from([
                ("connection_string", "mysql://reader:pw@127.0.0.1:3307/shop"),
                ("sslmode", "disabled"),
            ])
        );
    }

    #[test]
    fn a_ca_certificate_is_the_drivers_root_and_verifies_the_server() {
        let ca = std::path::Path::new("/tmp/jammi-source-ca/x.pem");
        let (params, _) = pool_params("mysql://u@h/shop", Some(ca)).unwrap();
        assert_eq!(
            exposed(&params),
            HashMap::from([
                ("connection_string", "mysql://u@h/shop"),
                ("sslmode", "required"),
                ("sslrootcert", "/tmp/jammi-source-ca/x.pem"),
            ])
        );
        assert!(pool_params("mysql://u@h/shop?sslmode=preferred", Some(ca)).is_err());
        assert!(pool_params("mysql://u@h/shop?sslmode=disabled", Some(ca)).is_err());
    }

    #[test]
    fn a_url_without_a_database_or_with_an_unknown_parameter_is_refused() {
        assert!(pool_params("mysql://u@h", None).is_err());
        assert!(pool_params("mysql://u@h/shop?frobnicate=1", None).is_err());
    }
}
