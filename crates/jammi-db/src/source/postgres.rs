//! Postgres source provider via `datafusion-table-providers`.
//!
//! Gated behind the `postgres` feature flag.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::catalog::TableProvider;
use datafusion_table_providers::postgres::PostgresTableFactory;
use datafusion_table_providers::sql::db_connection_pool::postgrespool::PostgresConnectionPool;
use secrecy::SecretString;

use crate::error::{JammiError, Result};
use crate::pg_uri::{Keyword, PgUri};
use crate::source::{database, SourceConnection};

/// The schema whose tables a Postgres source serves.
const SCHEMA: &str = "public";

/// Create table providers for all public tables in a Postgres database.
///
/// Returns `(table_name, Arc<dyn TableProvider>)` pairs with filter/projection/limit
/// pushdown handled automatically by `datafusion-table-providers`.
pub async fn create_postgres_tables(
    source_id: &str,
    connection: &SourceConnection,
) -> Result<Vec<(String, Arc<dyn TableProvider>)>> {
    let failed = |message: String| JammiError::Source {
        source_id: source_id.into(),
        message,
    };
    let params =
        pool_params(database::url(source_id, connection)?, &connection.options).map_err(failed)?;
    let pool = PostgresConnectionPool::new(params)
        .await
        .map_err(|e| failed(format!("Failed to connect to Postgres: {e}")))?;
    let pool = Arc::new(pool);
    let factory = PostgresTableFactory::new(Arc::clone(&pool));
    database::schema_tables(source_id, pool.as_ref(), SCHEMA, |table| {
        factory.table_provider(table)
    })
    .await
}

/// The driver's pool parameters for a libpq connection URI.
///
/// The driver reads its connection in keyword form only — a URI passed as
/// one would lose every component — so the URI's keywords (read as libpq
/// reads them, [`PgUri`]) become the driver's, and an absent `sslmode` is
/// `prefer`, libpq's default, as it is to the catalog's driver reading the
/// same URL. The driver takes no client certificate, so a URI naming one is
/// refused rather than connected without it. `options` are the driver's own
/// parameters and override what the URI says.
fn pool_params(
    url: &str,
    options: &HashMap<String, String>,
) -> std::result::Result<HashMap<String, SecretString>, String> {
    let uri = PgUri::parse(url).map_err(|e| e.to_string())?;
    let from_uri = uri
        .params()
        .map(|(keyword, value)| Ok((driver_parameter(keyword)?.to_string(), value.to_string())))
        .collect::<std::result::Result<Vec<_>, String>>()?;
    Ok([("sslmode".to_string(), "prefer".to_string())]
        .into_iter()
        .chain(from_uri)
        .chain(options.clone())
        .map(|(key, value)| (key, SecretString::from(value)))
        .collect())
}

/// The driver's parameter for a libpq keyword.
fn driver_parameter(keyword: Keyword) -> std::result::Result<&'static str, String> {
    match keyword {
        Keyword::Host => Ok("host"),
        Keyword::Port => Ok("port"),
        Keyword::User => Ok("user"),
        Keyword::Password => Ok("pass"),
        Keyword::Dbname => Ok("db"),
        Keyword::Sslmode => Ok("sslmode"),
        Keyword::Sslrootcert => Ok("sslrootcert"),
        Keyword::ApplicationName => Ok("application_name"),
        Keyword::Options => Ok("options"),
        Keyword::Sslcert | Keyword::Sslkey => Err(format!(
            "a Postgres source takes no client certificate (`{}`)",
            <&str>::from(keyword)
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use secrecy::ExposeSecret;

    fn params(url: &str) -> HashMap<String, String> {
        pool_params(url, &HashMap::new())
            .unwrap()
            .into_iter()
            .map(|(k, v)| (k, v.expose_secret().to_string()))
            .collect()
    }

    fn expected(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn a_uri_becomes_the_drivers_keyword_parameters() {
        assert_eq!(
            params("postgresql://reader:s%40cret@db.internal:6543/sales"),
            expected(&[
                ("host", "db.internal"),
                ("port", "6543"),
                ("user", "reader"),
                ("pass", "s@cret"),
                ("db", "sales"),
                ("sslmode", "prefer"),
            ])
        );
    }

    #[test]
    fn an_empty_authority_takes_its_socket_from_the_query() {
        // The form a local server hands out: no host before the path at all.
        assert_eq!(
            params("postgresql://postgres:@/postgres?host=/tmp/pgdata"),
            expected(&[
                ("host", "/tmp/pgdata"),
                ("user", "postgres"),
                ("db", "postgres"),
                ("sslmode", "prefer"),
            ])
        );
    }

    #[test]
    fn tls_parameters_reach_the_driver_verbatim() {
        let got = params(
            "postgres://u@h/d?sslmode=verify-full&sslrootcert=/etc/ssl/ca.pem&application_name=etl",
        );
        assert_eq!(got["sslmode"], "verify-full");
        assert_eq!(got["sslrootcert"], "/etc/ssl/ca.pem");
        assert_eq!(got["application_name"], "etl");
    }

    #[test]
    fn source_options_override_the_uri() {
        let options = HashMap::from([("sslmode".to_string(), "disable".to_string())]);
        let got = pool_params("postgres://u@h/d?sslmode=require", &options).unwrap();
        assert_eq!(got["sslmode"].expose_secret(), "disable");
    }

    #[test]
    fn a_uri_the_driver_cannot_honour_is_refused() {
        assert!(pool_params("not a url", &HashMap::new()).is_err());
        assert!(pool_params("postgres://u@h/d?host=a,b", &HashMap::new()).is_err());
        assert!(pool_params("postgres://u@h/d?sslcert=/etc/ssl/me.pem", &HashMap::new()).is_err());
    }
}
