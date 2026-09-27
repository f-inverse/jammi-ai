//! MySQL source provider via `datafusion-table-providers`.
//!
//! Gated behind the `mysql` feature flag.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::catalog::TableProvider;
use datafusion_table_providers::mysql::MySQLTableFactory;
use datafusion_table_providers::sql::db_connection_pool::mysqlpool::MySQLConnectionPool;
use secrecy::SecretString;
use url::Url;

use crate::error::{JammiError, Result};
use crate::source::{database, SourceConnection};

/// The URL parameter that sets the TLS mode: the driver's own, which
/// `mysql_async` does not parse.
const TLS_MODE: &str = "sslmode";

/// Create table providers for all tables of the MySQL database the URL names.
pub async fn create_mysql_tables(
    source_id: &str,
    connection: &SourceConnection,
) -> Result<Vec<(String, Arc<dyn TableProvider>)>> {
    let failed = |message: String| JammiError::Source {
        source_id: source_id.into(),
        message,
    };
    let (params, schema) =
        pool_params(database::url(source_id, connection)?, &connection.options).map_err(failed)?;
    let pool = MySQLConnectionPool::new(params)
        .await
        .map_err(|e| failed(format!("Failed to connect to MySQL: {e}")))?;
    let pool = Arc::new(pool);
    let factory = MySQLTableFactory::new(Arc::clone(&pool));
    database::schema_tables(source_id, pool.as_ref(), &schema, |table| {
        factory.table_provider(table)
    })
    .await
}

/// The driver's pool parameters for a `mysql://` URL, and the database whose
/// tables the source serves.
///
/// `mysql_async` parses the URL; its `sslmode` parameter — `disabled`,
/// `preferred` or `required` (the default), the driver's own and unknown to
/// `mysql_async` — is taken out first and handed to the driver beside it. The
/// driver's `preferred` still refuses a server without TLS, so a local server
/// that has none is reached with `sslmode=disabled`. `options` are the
/// driver's own parameters and override what the URL says.
fn pool_params(
    url: &str,
    options: &HashMap<String, String>,
) -> std::result::Result<(HashMap<String, SecretString>, String), String> {
    let mut parsed = Url::parse(url).map_err(|e| format!("invalid MySQL URL: {e}"))?;
    let (tls, rest): (Vec<_>, Vec<_>) = parsed
        .query_pairs()
        .into_owned()
        .partition(|(key, _)| key == TLS_MODE);
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
        .chain(options.clone())
        .map(|(key, value)| (key, SecretString::from(value)))
        .collect();
    Ok((params, schema))
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
            &HashMap::new(),
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
    fn source_options_override_the_url() {
        let options = HashMap::from([("sslmode".to_string(), "required".to_string())]);
        let (params, _) = pool_params("mysql://u@h/shop?sslmode=disabled", &options).unwrap();
        assert_eq!(params["sslmode"].expose_secret(), "required");
    }

    #[test]
    fn a_url_without_a_database_or_with_an_unknown_parameter_is_refused() {
        assert!(pool_params("mysql://u@h", &HashMap::new()).is_err());
        assert!(pool_params("mysql://u@h/shop?frobnicate=1", &HashMap::new()).is_err());
    }
}
