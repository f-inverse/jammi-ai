//! A query over a Postgres source whose database has stopped answering fails
//! at once, classed unavailable and naming where the database was — not
//! after the driver pool's whole connect timeout, blaming its capacity.

use std::time::{Duration, Instant};

use jammi_db::catalog::backend::BackendKind;
use jammi_db::error::ErrorClass;
use jammi_db::source::SourceDefinition;
use jammi_test_utils::{make_test_session, unique_suffix};
use sqlx::{Connection, Executor, PgConnection};
use tempfile::tempdir;
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinSet;
use url::Url;

/// A TCP proxy on a loopback port to `target`, until its task is aborted:
/// then its listener and every connection through it close, as a stopped
/// database's would.
async fn proxy(target: (String, u16)) -> (u16, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let task = tokio::spawn(async move {
        let mut connections = JoinSet::new();
        loop {
            let (mut inbound, _) = listener.accept().await.unwrap();
            let target = target.clone();
            connections.spawn(async move {
                let mut outbound = TcpStream::connect((target.0.as_str(), target.1))
                    .await
                    .unwrap();
                // A connection ends when either side closes it.
                tokio::io::copy_bidirectional(&mut inbound, &mut outbound)
                    .await
                    .ok();
            });
        }
    });
    (port, task)
}

#[tokio::test]
async fn a_query_over_a_stopped_database_fails_at_once_naming_it() {
    let source_dir = tempdir().unwrap();
    let url = jammi_test_utils::postgres_database_url(source_dir.path()).await;
    let mut db = PgConnection::connect(&url).await.unwrap();
    db.execute("CREATE TABLE t (id INTEGER); INSERT INTO t VALUES (1);")
        .await
        .unwrap();
    db.close().await.unwrap();

    let mut through = Url::parse(&url).unwrap();
    let target = (
        through.host_str().unwrap().to_string(),
        through.port().unwrap_or(5432),
    );
    let (port, proxy) = proxy(target).await;
    through.set_host(Some("127.0.0.1")).unwrap();
    through.set_port(Some(port)).unwrap();

    let dir = tempdir().unwrap();
    let session = make_test_session(BackendKind::Sqlite, dir.path()).await;
    let source = format!("pg_{}", unique_suffix());
    let definition = SourceDefinition::from_url(through.as_str(), None).unwrap();
    session
        .add_source(&source, definition.source_type, definition.connection)
        .await
        .unwrap();
    let query = format!("SELECT id FROM {source}.public.t");
    session.sql(&query).await.expect("the database answers through the proxy");

    proxy.abort();
    proxy.await.ok();

    let started = Instant::now();
    let err = session.sql(&query).await.expect_err("the database is gone");
    let elapsed = started.elapsed();
    assert!(elapsed < Duration::from_secs(2), "failed after {elapsed:?}: {err}");
    assert_eq!(err.class(), ErrorClass::Unavailable, "{err}");
    let message = err.to_string();
    assert!(message.contains(&format!("127.0.0.1:{port}")), "{message}");
}
