//! Postgres source provider via `datafusion-table-providers`.
//!
//! Gated behind the `postgres` feature flag.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::catalog::TableProvider;
use datafusion_table_providers::postgres::PostgresTableFactory;
use datafusion_table_providers::sql::db_connection_pool::postgrespool::PostgresConnectionPool;
use percent_encoding::percent_decode_str;
use secrecy::SecretString;

use crate::error::{JammiError, Result};
use crate::source::{database, SourceConnection};

/// The schema whose tables a Postgres source serves.
const SCHEMA: &str = "public";

/// The URI query parameters a source takes, each with the driver parameter it
/// sets.
const QUERY_PARAMETERS: [(&str, &str); 9] = [
    ("host", "host"),
    ("port", "port"),
    ("user", "user"),
    ("password", "pass"),
    ("dbname", "db"),
    ("sslmode", "sslmode"),
    ("sslrootcert", "sslrootcert"),
    ("application_name", "application_name"),
    ("options", "options"),
];

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
/// The driver reads a connection string in libpq's keyword form only — a URI
/// passed as one would lose every component — so the URI is taken apart here
/// with libpq's meaning: the authority and path give the host, port, user,
/// password and database; a query parameter overrides the component it names
/// (a `host` that is a path names a Unix-socket directory, and the authority
/// may then be empty); and an absent `sslmode` is `prefer`, as it is to libpq
/// and to the catalog's driver reading the same URL. A query parameter the
/// driver has no use for is refused rather than dropped. `options` are the
/// driver's own parameters and override what the URI says.
///
/// The grammar is libpq's, not the WHATWG URL standard's, which refuses an
/// empty host after a user (`postgresql://user@/db?host=/run/pg`).
fn pool_params(
    url: &str,
    options: &HashMap<String, String>,
) -> std::result::Result<HashMap<String, SecretString>, String> {
    let decoded = |part: &str| percent_decode_str(part).decode_utf8_lossy().into_owned();
    let rest = ["postgresql://", "postgres://"]
        .iter()
        .find_map(|scheme| url.strip_prefix(scheme))
        .ok_or_else(|| format!("`{url}` is not a postgres:// or postgresql:// URI"))?;
    let (rest, query) = rest.split_once('?').unwrap_or((rest, ""));
    let (authority, db) = rest.split_once('/').unwrap_or((rest, ""));
    let (userinfo, hostport) = authority
        .rsplit_once('@')
        .map_or((None, authority), |(userinfo, hostport)| {
            (Some(userinfo), hostport)
        });
    let (user, password) = match userinfo.map(|u| u.split_once(':')) {
        Some(Some((user, password))) => (Some(user), Some(password)),
        Some(None) => (userinfo, None),
        None => (None, None),
    };
    let (host, port) = match hostport.strip_prefix('[') {
        Some(bracketed) => {
            let (host, after) = bracketed
                .split_once(']')
                .ok_or_else(|| format!("unclosed `[` in the host of `{url}`"))?;
            (host, after.strip_prefix(':'))
        }
        None => hostport
            .rsplit_once(':')
            .map_or((hostport, None), |(host, port)| (host, Some(port))),
    };
    let authority = [
        ("host", Some(host)),
        ("port", port),
        ("user", user),
        ("db", Some(db)),
    ]
    .into_iter()
    .filter_map(|(key, value)| Some((key.to_string(), decoded(value.filter(|v| !v.is_empty())?))))
    .chain(password.map(|p| ("pass".to_string(), decoded(p))));
    let query = query
        .split('&')
        .filter(|pair| !pair.is_empty())
        .map(|pair| {
            let (key, value) = pair
                .split_once('=')
                .ok_or_else(|| format!("the Postgres URL parameter `{pair}` has no value"))?;
            let param = QUERY_PARAMETERS
                .iter()
                .find(|(name, _)| *name == key)
                .map(|(_, param)| param.to_string())
                .ok_or_else(|| format!("the Postgres URL parameter `{key}` is not supported"))?;
            Ok((param, decoded(value)))
        })
        .collect::<std::result::Result<Vec<_>, String>>()?;
    let params: HashMap<String, String> = [("sslmode".to_string(), "prefer".to_string())]
        .into_iter()
        .chain(authority)
        .chain(query)
        .chain(options.clone())
        .collect();
    if params.get("host").is_some_and(|host| host.contains(',')) {
        return Err("a Postgres source connects to one host".into());
    }
    Ok(params
        .into_iter()
        .map(|(key, value)| (key, SecretString::from(value)))
        .collect())
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
    fn a_host_query_parameter_names_a_unix_socket_directory() {
        assert_eq!(
            params("postgres://postgres@localhost/catalog?host=/tmp/pg.sock.d"),
            expected(&[
                ("host", "/tmp/pg.sock.d"),
                ("user", "postgres"),
                ("db", "catalog"),
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
                ("pass", ""),
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
    fn a_malformed_multi_host_or_unsupported_uri_is_refused() {
        assert!(pool_params("not a url", &HashMap::new()).is_err());
        assert!(pool_params("postgres://u@h/d?frobnicate=1", &HashMap::new()).is_err());
        assert!(pool_params("postgres://u@h/d?host=a,b", &HashMap::new()).is_err());
    }
}
