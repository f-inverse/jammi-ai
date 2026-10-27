//! The pool a Postgres source's scans connect through ([`ReachablePool`]).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use datafusion_table_providers::sql::db_connection_pool::dbconnection::DbConnection;
use datafusion_table_providers::sql::db_connection_pool::{DbConnectionPool, JoinPushDown};

use crate::error::JammiError;

/// How long a connect waits on the driver's pool before the database's
/// address is probed: a pool with an idle connection answers well within it.
const PROBE_AFTER: Duration = Duration::from_millis(250);

/// How long a probe of the database's address waits for an answer.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// Where a database listens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Address {
    Tcp {
        host: String,
        port: u16,
    },
    /// A Unix-domain socket file.
    Socket(PathBuf),
}

impl Address {
    /// Whether anything answers at the address: `Ok` once a connection is
    /// accepted, the refusal or the failure to resolve otherwise.
    async fn probe(&self) -> std::io::Result<()> {
        let accepted = async {
            match self {
                Address::Tcp { host, port } => {
                    tokio::net::TcpStream::connect((host.as_str(), *port)).await?;
                }
                Address::Socket(path) => {
                    tokio::net::UnixStream::connect(path).await?;
                }
            }
            Ok(())
        };
        tokio::time::timeout(PROBE_TIMEOUT, accepted)
            .await
            .unwrap_or_else(|_| {
                Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    format!("no answer within {} s", PROBE_TIMEOUT.as_secs()),
                ))
            })
    }
}

impl std::fmt::Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Address::Tcp { host, port } if host.contains(':') => write!(f, "[{host}]:{port}"),
            Address::Tcp { host, port } => write!(f, "{host}:{port}"),
            Address::Socket(path) => write!(f, "{}", path.display()),
        }
    }
}

/// A Postgres source's pool: the driver's, failing fast when the database
/// cannot be reached.
///
/// A driver pool answers a connect at once from an idle connection; with
/// none, it opens one, and a driver that retries a refused connect until its
/// own timeout answers a database that is down only after that wait, with
/// advice about its capacity. So a connect the pool has not answered within
/// [`PROBE_AFTER`] probes the database's address directly: an address that
/// refuses, does not resolve or does not answer fails the connect at once,
/// [`JammiError::Unavailable`] naming the address and why; an address that
/// answers leaves the connect waiting on the pool, which is then busy, not
/// unreachable.
pub(super) struct ReachablePool<Q> {
    pool: Arc<Q>,
    address: Address,
}

impl<Q> ReachablePool<Q> {
    pub(super) fn new(pool: Arc<Q>, address: Address) -> Self {
        Self { pool, address }
    }
}

#[async_trait]
impl<T, P, Q> DbConnectionPool<T, P> for ReachablePool<Q>
where
    P: 'static,
    Q: DbConnectionPool<T, P> + Send + Sync,
{
    async fn connect(
        &self,
    ) -> std::result::Result<Box<dyn DbConnection<T, P>>, Box<dyn std::error::Error + Send + Sync>>
    {
        let connect = self.pool.connect();
        tokio::pin!(connect);
        tokio::select! {
            connection = &mut connect => connection,
            () = tokio::time::sleep(PROBE_AFTER) => match self.address.probe().await {
                Ok(()) => connect.await,
                Err(cause) => Err(Box::new(JammiError::Unavailable {
                    resource: format!("the database at {}", self.address),
                    reason: cause.to_string(),
                })),
            },
        }
    }

    fn join_push_down(&self) -> JoinPushDown {
        self.pool.join_push_down()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_address_names_itself_as_a_client_would_reach_it() {
        let tcp = |host: &str| Address::Tcp {
            host: host.into(),
            port: 5432,
        };
        assert_eq!(tcp("db.internal").to_string(), "db.internal:5432");
        assert_eq!(tcp("::1").to_string(), "[::1]:5432");
        assert_eq!(
            Address::Socket(PathBuf::from("/tmp/pg/.s.PGSQL.5432")).to_string(),
            "/tmp/pg/.s.PGSQL.5432"
        );
    }

    #[tokio::test]
    async fn a_closed_port_is_refused_at_once() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let refused = Address::Tcp {
            host: "127.0.0.1".into(),
            port,
        }
        .probe()
        .await
        .unwrap_err();
        assert_eq!(refused.kind(), std::io::ErrorKind::ConnectionRefused);
    }
}
