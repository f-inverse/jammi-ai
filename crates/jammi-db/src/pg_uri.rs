//! libpq's connection URI, read once for every Postgres connection the
//! engine makes — the catalog, the trigger broker and Postgres sources.
//!
//! The grammar and its meaning are libpq's
//! (<https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING-URIS>):
//! `postgresql://[userspec@][hostspec][/dbname][?paramspec]`. It is not the
//! WHATWG URL standard's, which refuses an empty host after a user — the form
//! libpq uses for a Unix-socket connection
//! (`postgresql://user@/db?host=/run/postgresql`).
//!
//! What the engine takes from libpq's meaning, for every consumer alike:
//!
//! - every part is percent-decoded, and nothing else — a `+` is a `+`;
//! - a named parameter is a libpq keyword and overrides the URI component it
//!   names; a parameter that is not a keyword the engine reads is refused,
//!   as libpq refuses a name that is not a keyword (JDBC's `ssl=true` reads
//!   as `sslmode=require`, as it does to libpq);
//! - an empty value is no value, so an empty component or parameter leaves
//!   the keyword to its default (a `user:@` password is no password);
//! - a host that is an absolute path names a Unix-socket directory;
//! - a URI names one host: libpq's multi-host failover list is refused,
//!   since no driver the engine connects through takes one.

use std::collections::BTreeMap;
use std::str::FromStr;

use sqlx::postgres::PgConnectOptions;

/// A libpq connection keyword the engine reads from a connection URI.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, strum::EnumString, strum::IntoStaticStr,
)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum Keyword {
    Host,
    Port,
    User,
    Password,
    Dbname,
    Sslmode,
    Sslrootcert,
    Sslcert,
    Sslkey,
    ApplicationName,
    Options,
}

/// libpq's `sslmode` values.
const SSL_MODES: [&str; 6] = [
    "disable",
    "allow",
    "prefer",
    "require",
    "verify-ca",
    "verify-full",
];

/// A connection URI that is not a libpq URI the engine reads. The message
/// never repeats the URI, which may carry a password.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub(crate) struct PgUriError(String);

fn refused(message: impl Into<String>) -> PgUriError {
    PgUriError(message.into())
}

/// A parsed libpq connection URI: the value of every keyword it sets.
pub(crate) struct PgUri {
    params: BTreeMap<Keyword, String>,
}

impl PgUri {
    pub(crate) fn parse(uri: &str) -> Result<Self, PgUriError> {
        let rest = ["postgresql://", "postgres://"]
            .iter()
            .find_map(|scheme| uri.strip_prefix(scheme))
            .ok_or_else(|| refused("a Postgres URL begins with postgresql:// or postgres://"))?;
        let (hierarchy, query) = rest.split_once('?').unwrap_or((rest, ""));
        let (authority, dbname) = hierarchy.split_once('/').unwrap_or((hierarchy, ""));
        let (userinfo, hostport) = authority
            .rsplit_once('@')
            .map_or((None, authority), |(userinfo, hostport)| {
                (Some(userinfo), hostport)
            });
        let (user, password) = match userinfo.map(|u| u.split_once(':').ok_or(u)) {
            Some(Ok((user, password))) => (Some(user), Some(password)),
            Some(Err(user)) => (Some(user), None),
            None => (None, None),
        };
        let (host, port) = match hostport.strip_prefix('[') {
            Some(bracketed) => {
                let (host, after) = bracketed
                    .split_once(']')
                    .ok_or_else(|| refused("an unclosed `[` in the Postgres URL's host"))?;
                let port = match after {
                    "" => None,
                    after => Some(after.strip_prefix(':').ok_or_else(|| {
                        refused("a bracketed Postgres host is followed by `:port` or nothing")
                    })?),
                };
                (host, port)
            }
            None => hostport
                .rsplit_once(':')
                .map_or((hostport, None), |(host, port)| (host, Some(port))),
        };
        let components = [
            (Keyword::Host, Some(host)),
            (Keyword::Port, port),
            (Keyword::User, user),
            (Keyword::Password, password),
            (Keyword::Dbname, Some(dbname)),
        ]
        .into_iter()
        .filter_map(|(keyword, value)| Some(Ok((keyword, value?))));
        let parameters = query
            .split('&')
            .filter(|pair| !pair.is_empty())
            .map(parameter);
        components
            .chain(parameters)
            .try_fold(BTreeMap::new(), |mut params, setting| {
                let (keyword, raw) = setting?;
                let value = decoded(raw)?;
                if value.is_empty() {
                    params.remove(&keyword);
                } else {
                    params.insert(keyword, checked(keyword, value)?);
                }
                Ok(params)
            })
            .map(|params| Self { params })
    }

    /// Every keyword the URI sets, with its value.
    pub(crate) fn params(&self) -> impl Iterator<Item = (Keyword, &str)> {
        self.params
            .iter()
            .map(|(keyword, value)| (*keyword, value.as_str()))
    }
}

/// sqlx's connect options for a libpq connection URI.
///
/// sqlx reads libpq's keywords as its URL's query parameters, then fills an
/// absent password from the password file; its builder can do neither (it
/// takes `options` only as `-c` pairs, and applies the password file only
/// before any field is set). So the parsed URI reaches sqlx as a URL that
/// carries every keyword as a form-encoded query parameter — the one form
/// its WHATWG reader takes for every libpq URI, the socket form included.
pub(crate) fn connect_options(uri: &str) -> Result<PgConnectOptions, sqlx::Error> {
    let uri = PgUri::parse(uri).map_err(|e| sqlx::Error::Configuration(Box::new(e)))?;
    let query = url::form_urlencoded::Serializer::new(String::new())
        .extend_pairs(
            uri.params()
                .map(|(keyword, value)| (<&str>::from(keyword), value)),
        )
        .finish();
    format!("postgres:///?{query}").parse()
}

/// One `name=value` query parameter, its value still encoded.
fn parameter(pair: &str) -> Result<(Keyword, &str), PgUriError> {
    let (name, value) = pair
        .split_once('=')
        .ok_or_else(|| refused(format!("the Postgres URL parameter `{pair}` has no value")))?;
    match decoded(name)?.as_str() {
        "ssl" if value == "true" => Ok((Keyword::Sslmode, "require")),
        name => Keyword::from_str(name)
            .map(|keyword| (keyword, value))
            .map_err(|_| {
                refused(format!(
                    "the Postgres URL parameter `{name}` is not supported"
                ))
            }),
    }
}

/// `raw` percent-decoded, as libpq decodes it: a `%` begins two hex digits,
/// and the result is UTF-8.
fn decoded(raw: &str) -> Result<String, PgUriError> {
    let well_formed = raw.match_indices('%').all(|(at, _)| {
        raw.as_bytes()
            .get(at + 1..at + 3)
            .is_some_and(|hex| hex.iter().all(u8::is_ascii_hexdigit))
    });
    if !well_formed {
        return Err(refused("a malformed percent-encoding in the Postgres URL"));
    }
    percent_encoding::percent_decode_str(raw)
        .decode_utf8()
        .map(String::from)
        .map_err(|_| refused("a Postgres URL part that does not decode to UTF-8"))
}

/// `value` when it is a value `keyword` takes.
fn checked(keyword: Keyword, value: String) -> Result<String, PgUriError> {
    let refusal = match keyword {
        Keyword::Host if value.contains(',') => Some("a Postgres URL names one host"),
        Keyword::Port if value.parse::<u16>().is_err() => {
            Some("the Postgres URL's port is not a port number")
        }
        Keyword::Sslmode if !SSL_MODES.contains(&value.as_str()) => {
            Some("the Postgres URL's sslmode is not one of libpq's")
        }
        _ => None,
    };
    refusal.map_or(Ok(value), |refusal| Err(refused(refusal)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(uri: &str) -> Vec<(&'static str, String)> {
        PgUri::parse(uri)
            .unwrap()
            .params()
            .map(|(keyword, value)| (keyword.into(), value.to_string()))
            .collect()
    }

    fn expected(pairs: &[(&'static str, &str)]) -> Vec<(&'static str, String)> {
        pairs.iter().map(|(k, v)| (*k, v.to_string())).collect()
    }

    #[test]
    fn the_authority_and_path_name_the_host_port_user_password_and_database() {
        assert_eq!(
            params("postgresql://reader:s%40cret@db.internal:6543/sales"),
            expected(&[
                ("host", "db.internal"),
                ("port", "6543"),
                ("user", "reader"),
                ("password", "s@cret"),
                ("dbname", "sales"),
            ])
        );
    }

    #[test]
    fn a_bracketed_host_is_an_ipv6_address() {
        assert_eq!(
            params("postgres://u@[2001:db8::1234]:5433/d"),
            expected(&[
                ("host", "2001:db8::1234"),
                ("port", "5433"),
                ("user", "u"),
                ("dbname", "d"),
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
                ("dbname", "catalog"),
            ])
        );
        // The percent-encoded path in the host part means the same.
        assert_eq!(
            params("postgresql://%2Fvar%2Flib%2Fpostgresql/d"),
            expected(&[("host", "/var/lib/postgresql"), ("dbname", "d")])
        );
    }

    /// The form a local server hands out: credentials beside no host at
    /// all, the socket as a parameter. An empty password is no password.
    #[test]
    fn an_empty_authority_takes_its_socket_from_the_query() {
        assert_eq!(
            params("postgresql://postgres:@/postgres?host=/tmp/pgdata"),
            expected(&[
                ("host", "/tmp/pgdata"),
                ("user", "postgres"),
                ("dbname", "postgres"),
            ])
        );
    }

    #[test]
    fn only_percent_encoding_is_decoded() {
        assert_eq!(
            params("postgres://app:p%40ss+w@/db?options=-c%20geqo%3Doff"),
            expected(&[
                ("user", "app"),
                ("password", "p@ss+w"),
                ("dbname", "db"),
                ("options", "-c geqo=off"),
            ])
        );
    }

    #[test]
    fn a_parameter_overrides_its_component_and_an_empty_one_clears_it() {
        assert_eq!(
            params("postgres://u@h:1/d?port=2&dbname=e&user=&ssl=true"),
            expected(&[
                ("host", "h"),
                ("port", "2"),
                ("dbname", "e"),
                ("sslmode", "require")
            ])
        );
    }

    #[test]
    fn what_libpq_would_refuse_or_no_driver_takes_is_refused() {
        [
            "not a url",
            "mysql://u@h/d",
            "postgres://u@h/d?frobnicate=1",
            "postgres://u@h/d?sslmode",
            "postgres://u@h/d?sslmode=sometimes",
            "postgres://u@h:port/d",
            "postgres://u@h/d?host=a,b",
            "postgres://u@a:1,b:2/d",
            "postgres://u@[::1/d",
            "postgres://u:p%4@h/d",
        ]
        .into_iter()
        .for_each(|uri| assert!(PgUri::parse(uri).is_err(), "{uri} parsed"));
    }

    #[test]
    fn a_refusal_never_repeats_the_password() {
        let error = PgUri::parse("postgres://u:hunter2@h:x/d").err().unwrap();
        assert!(!error.to_string().contains("hunter2"), "{error}");
    }

    /// libpq's Unix-socket URI — credentials beside an empty host, the
    /// socket directory as the `host` parameter — reaches sqlx as the
    /// socket, the user and the database.
    #[test]
    fn a_socket_uri_with_credentials_connects_as_libpq_would() {
        let opts = connect_options("postgresql://postgres:@/jammi?host=/tmp/pg_sock").unwrap();
        assert_eq!(
            opts.get_socket().map(|p| p.to_string_lossy().into_owned()),
            Some("/tmp/pg_sock".to_string())
        );
        assert_eq!(opts.get_username(), "postgres");
        assert_eq!(opts.get_database(), Some("jammi"));
    }

    /// A password reaches sqlx exactly, its `@` and literal `+` included —
    /// sqlx form-decodes its query, where an unescaped `+` is a space.
    #[test]
    fn a_password_reaches_sqlx_exactly() {
        let opts = connect_options("postgres://app:p%40ss+w@/db?host=/run/pg").unwrap();
        assert!(
            format!("{opts:?}").contains(r#"password: Some("p@ss+w")"#),
            "{opts:?}"
        );
    }

    #[test]
    fn a_tcp_uri_connects_to_its_host_and_port() {
        let opts =
            connect_options("postgres://u:p@localhost:5433/db?application_name=etl").unwrap();
        assert_eq!(opts.get_host(), "localhost");
        assert_eq!(opts.get_port(), 5433);
        assert_eq!(opts.get_username(), "u");
        assert_eq!(opts.get_database(), Some("db"));
        assert_eq!(opts.get_application_name(), Some("etl"));
        assert!(opts.get_socket().is_none());
        let opts = connect_options("postgresql:///db?host=/tmp&user=u").unwrap();
        assert_eq!(opts.get_username(), "u");
        assert!(opts.get_socket().is_some());
    }

    #[test]
    fn a_uri_sqlx_cannot_take_is_a_configuration_error() {
        assert!(matches!(
            connect_options("postgres://u@h/d?frobnicate=1"),
            Err(sqlx::Error::Configuration(_))
        ));
    }
}
