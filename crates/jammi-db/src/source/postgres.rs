//! Postgres source provider via `datafusion-table-providers`.
//!
//! Gated behind the `postgres` feature flag.

mod columns;
mod pool;
mod scan;

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::catalog::TableProvider;
use datafusion::common::TableReference;
use datafusion_table_providers::postgres::DynPostgresConnectionPool;
use datafusion_table_providers::sql::db_connection_pool::postgrespool::PostgresConnectionPool;
use secrecy::SecretString;

use crate::error::{JammiError, Result};
use crate::pg_uri::{Keyword, PgUri};
use crate::source::database::{self, TlsModes};
use crate::source::{SourceConnection, SourceTable};

use self::pool::{Address, ReachablePool};

/// Create table providers for every table, view and materialized view of
/// every schema of the database the connecting role may use, each under its
/// own schema.
///
/// Returns `(table, Arc<dyn TableProvider>)` pairs, each a
/// [`crate::federation::remote::Remote::Postgres`] table: Postgres is sent
/// only what it runs as the engine does, and each column is read as the
/// engine reads its type ([`columns`]).
pub async fn create_postgres_tables(
    source_id: &str,
    connection: &SourceConnection,
) -> Result<Vec<(SourceTable, Arc<dyn TableProvider>)>> {
    let failed = |message: String| JammiError::Source {
        source_id: source_id.into(),
        message,
    };
    let url = database::url(source_id, connection)?;
    let ca = match &connection.ca_certificate {
        Some(ca) => Some(database::ca_file(source_id, ca).await?),
        None => None,
    };
    let params = pool_params(url, ca.as_deref()).map_err(failed)?;
    let address = address(url).map_err(failed)?;
    let pool = PostgresConnectionPool::new(params)
        .await
        .map_err(|e| failed(format!("Failed to connect to Postgres: {e}")))?;
    let pool = Arc::new(pool);
    let tables = columns::columns(&pool)
        .await
        .map_err(|e| failed(format!("reading the database's columns: {e}")))?;
    let scans: Arc<DynPostgresConnectionPool> = Arc::new(ReachablePool::new(pool, address));
    Ok(tables
        .into_iter()
        .map(|((schema, table), columns)| {
            let reference = TableReference::partial(schema.as_str(), table.as_str());
            let provider = columns::remote_table(&scans, reference, &columns);
            (SourceTable::new(source_id, &schema, &table), provider)
        })
        .collect())
}

/// Where the database a libpq URI names listens: its `host` and `port`
/// (5432 when absent), a `host` that is a path naming the directory of a
/// Unix-domain socket, as libpq reads it.
fn address(url: &str) -> std::result::Result<Address, String> {
    let uri = PgUri::parse(url).map_err(|e| e.to_string())?;
    let (mut host, mut port) = (None, 5432);
    for (keyword, value) in uri.params() {
        match keyword {
            Keyword::Host => host = Some(value.to_string()),
            Keyword::Port => {
                port = value
                    .parse()
                    .map_err(|_| format!("`{value}` is not a port"))?;
            }
            _ => {}
        }
    }
    let host = host.ok_or("a Postgres URI must name its host")?;
    Ok(if host.starts_with('/') {
        Address::Socket(std::path::Path::new(&host).join(format!(".s.PGSQL.{port}")))
    } else {
        Address::Tcp { host, port }
    })
}

/// The driver's pool parameters for a libpq connection URI.
///
/// The driver reads its connection in keyword form only — a URI passed as
/// one would lose every component — so the URI's keywords (read as libpq
/// reads them, [`PgUri`]) become the driver's, and an absent `sslmode` is
/// `prefer`, libpq's default, as it is to the catalog's driver reading the
/// same URL. The driver takes no client certificate, so a URI naming one is
/// refused rather than connected without it. `ca` is the file the source's CA
/// certificate was written to ([`database::ca_file`]): the driver reads it as
/// `sslrootcert`, the source connects with a mode that verifies the server
/// ([`TLS`]), and a URI naming its own `sslrootcert` as well is refused.
///
/// Every session runs at UTC ([`SESSION`]), whatever `options` the URI
/// names besides.
///
/// The driver writes [`INTERPOLATED`] values into a keyword string unquoted
/// and hands the rest to typed setters, so those values arrive quoted
/// ([`quoted`]) — a socket directory or database name with a space in it
/// would otherwise split into a second keyword.
fn pool_params(
    url: &str,
    ca: Option<&std::path::Path>,
) -> std::result::Result<HashMap<String, SecretString>, String> {
    let uri = PgUri::parse(url).map_err(|e| e.to_string())?;
    let mut params = uri
        .params()
        .map(|(keyword, value)| Ok((driver_parameter(keyword)?.to_string(), value.to_string())))
        .collect::<std::result::Result<HashMap<_, _>, String>>()?;
    if ca.is_some() && params.contains_key("sslrootcert") {
        return Err(
            "the URI names `sslrootcert` and the source holds a CA certificate; name the CA once"
                .into(),
        );
    }
    let mode = TLS.mode(params.get("sslmode").map(String::as_str), ca.is_some())?;
    params.insert("sslmode".into(), mode);
    if let Some(ca) = ca {
        params.insert("sslrootcert".into(), ca.display().to_string());
    }
    let session = params
        .remove("options")
        .map_or_else(|| SESSION.to_string(), |named| format!("{named} {SESSION}"));
    params.insert("options".into(), session);
    Ok(params
        .into_iter()
        .map(|(key, value)| {
            let value = if INTERPOLATED.contains(&key.as_str()) {
                quoted(&value)
            } else {
                value
            };
            (key, SecretString::from(value))
        })
        .collect())
}

/// libpq's TLS modes: `prefer` when the URI names none, as libpq's own default,
/// and `verify-full` — the certificate and the host it names — for a source
/// holding a CA certificate.
const TLS: TlsModes = TlsModes {
    default: "prefer",
    verifying: &["verify-full", "verify-ca"],
};

/// The session settings every connection runs with, last on its `options`
/// so they hold over any the URI names: `TimeZone` is UTC, as the engine's
/// zoned timestamps are, and as a `timestamp` column is read and compared
/// (its instant at UTC).
const SESSION: &str = "-c TimeZone=UTC";

/// The driver parameters it writes into its keyword connection string as
/// given; every other parameter reaches the driver's config through a setter.
const INTERPOLATED: [&str; 4] = ["host", "port", "user", "db"];

/// `value` as a single-quoted keyword-string value, backslash-escaping `\`
/// and `'` (libpq's keyword/value grammar, which the driver's parser reads).
fn quoted(value: &str) -> String {
    format!("'{}'", value.replace('\\', r"\\").replace('\'', r"\'"))
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
        pool_params(url, None)
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
                ("host", "'db.internal'"),
                ("port", "'6543'"),
                ("user", "'reader'"),
                ("pass", "s@cret"),
                ("db", "'sales'"),
                ("sslmode", "prefer"),
                ("options", "-c TimeZone=UTC"),
            ])
        );
    }

    #[test]
    fn an_empty_authority_takes_its_socket_from_the_query() {
        // The form a local server hands out: no host before the path at all.
        assert_eq!(
            params("postgresql://postgres:@/postgres?host=/tmp/pgdata"),
            expected(&[
                ("host", "'/tmp/pgdata'"),
                ("user", "'postgres'"),
                ("db", "'postgres'"),
                ("sslmode", "prefer"),
                ("options", "-c TimeZone=UTC"),
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
    fn interpolated_values_reach_the_driver_whole() {
        // The driver's own keyword string, built as it builds it: each
        // interpolated value followed by a space.
        let got = params(r"postgresql://o%27brien:@/my%20db?host=/run/pg%20data%5Cx");
        let keywords: String = [("host", "host"), ("user", "user"), ("db", "dbname")]
            .iter()
            .map(|(param, keyword)| format!("{keyword}={} ", got[*param]))
            .collect();
        let config: tokio_postgres::Config = keywords.parse().unwrap();
        assert_eq!(
            config.get_hosts(),
            [tokio_postgres::config::Host::Unix(r"/run/pg data\x".into())]
        );
        assert_eq!(config.get_user(), Some("o'brien"));
        assert_eq!(config.get_dbname(), Some("my db"));
    }

    #[test]
    fn a_ca_certificate_is_the_drivers_root_and_verifies_the_server() {
        let ca = std::path::Path::new("/tmp/jammi-source-ca/x.pem");
        let got = pool_params("postgres://u@h/d", Some(ca)).unwrap();
        assert_eq!(
            got["sslrootcert"].expose_secret(),
            "/tmp/jammi-source-ca/x.pem"
        );
        assert_eq!(got["sslmode"].expose_secret(), "verify-full");
        let got = pool_params("postgres://u@h/d?sslmode=verify-ca", Some(ca)).unwrap();
        assert_eq!(got["sslmode"].expose_secret(), "verify-ca");
        assert!(pool_params("postgres://u@h/d?sslmode=require", Some(ca)).is_err());
        assert!(pool_params("postgres://u@h/d?sslrootcert=/etc/ca.pem", Some(ca)).is_err());
    }

    #[test]
    fn the_database_is_reached_where_libpq_would_reach_it() {
        assert_eq!(
            address("postgres://u@db.internal:6543/d").unwrap(),
            Address::Tcp {
                host: "db.internal".into(),
                port: 6543
            }
        );
        assert_eq!(
            address("postgres://u@[::1]/d").unwrap(),
            Address::Tcp {
                host: "::1".into(),
                port: 5432
            }
        );
        assert_eq!(
            address("postgresql://postgres:@/postgres?host=/tmp/pgdata").unwrap(),
            Address::Socket("/tmp/pgdata/.s.PGSQL.5432".into())
        );
    }

    /// Every session runs at UTC; options the URI names come first, so the
    /// engine's setting holds over them.
    #[test]
    fn every_session_runs_at_utc_after_the_uris_own_options() {
        assert_eq!(params("postgres://u@h/d")["options"], "-c TimeZone=UTC");
        assert_eq!(
            params("postgres://u@h/d?options=-c%20statement_timeout%3D5000")["options"],
            "-c statement_timeout=5000 -c TimeZone=UTC"
        );
    }

    #[test]
    fn a_uri_the_driver_cannot_honour_is_refused() {
        assert!(pool_params("not a url", None).is_err());
        assert!(pool_params("postgres://u@h/d?host=a,b", None).is_err());
        assert!(pool_params("postgres://u@h/d?sslcert=/etc/ssl/me.pem", None).is_err());
    }
}
