//! Reachability tests for [`jammi_db::catalog::Catalog::ping`].
//!
//! SQLite covers the hermetic happy path. The Postgres parametrisation runs
//! only when the workspace was built with the `live-postgres-tests` feature
//! (which the CI `test-pg` job sets) and `JAMMI_TEST_PG_URL` resolves to a
//! reachable server; both a happy-path and an unreachable-URL negative case
//! live behind the same gate.

use jammi_db::catalog::backend::BackendKind;
use jammi_db::catalog::Catalog;

#[tokio::test]
async fn sqlite_ping_succeeds_against_open_pool() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).await.unwrap();
    catalog.ping().await.expect("ping should succeed");
    assert_eq!(catalog.backend_arc().backend_kind(), BackendKind::Sqlite);
}

#[tokio::test]
async fn sqlite_ping_is_idempotent() {
    // Ping is idempotent and cheap — call it twice to demonstrate the second
    // call does not run any migrations or take the catalog's mutation lock.
    let dir = tempfile::tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).await.unwrap();
    catalog.ping().await.expect("first ping");
    catalog.ping().await.expect("second ping");
}

#[tokio::test]
async fn sqlite_ping_holds_after_catalog_dropped_while_arc_alive() {
    // The backend wraps a `SqlitePool` shared via Arc. Dropping the catalog
    // does not invalidate an Arc-cloned backend handle — the pool stays
    // alive until the last handle releases it. The shared-pool contract is
    // what jammi-server relies on to multiplex one backend across multiple
    // sessions, so an extra ping after `drop(catalog)` is the right
    // regression test.
    let dir = tempfile::tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).await.unwrap();
    let backend = catalog.backend_arc();
    drop(catalog);
    backend
        .ping()
        .await
        .expect("backend still reachable via arc");
}

#[cfg(feature = "live-postgres-tests")]
mod postgres {
    use super::*;
    use jammi_db::catalog::backend::{BackendError, BackendImpl};
    use jammi_db::catalog::backend_postgres::PostgresBackend;
    use jammi_test_utils::pg_url_for_tests;

    #[tokio::test]
    async fn postgres_ping_succeeds_against_live_url() {
        let Some(url) = pg_url_for_tests() else {
            tracing::warn!(
                "JAMMI_TEST_PG_URL not set; skipping postgres_ping_succeeds_against_live_url"
            );
            return;
        };
        let pg = PostgresBackend::open_with_options(&url, 4, None)
            .await
            .expect("open postgres backend");
        let backend = BackendImpl::Postgres(pg);
        backend.migrate().await.expect("migrate postgres catalog");
        let catalog = Catalog::from_backend(backend);
        catalog.ping().await.expect("postgres ping");
        assert_eq!(catalog.backend_arc().backend_kind(), BackendKind::Postgres);
    }

    #[tokio::test]
    async fn postgres_ping_fails_against_unreachable_host() {
        // Use a known-closed port on localhost; `connect_with` returns a
        // pool that fails on first use, so even if the connect succeeds at
        // pool-creation time the `SELECT 1` will surface the failure.
        let url = "postgres://jammi:jammi@127.0.0.1:1/jammi_test";
        match PostgresBackend::open_with_options(url, 1, None).await {
            Ok(pg) => {
                let backend = BackendImpl::Postgres(pg);
                let err = backend
                    .ping()
                    .await
                    .expect_err("ping must fail against unreachable host");
                // Pool failures classify as Unavailable; raw protocol errors
                // pass through as Sqlx. Either is an acceptable signal of
                // "backend is down" — the caller's job is "is `is_err()` true?"
                assert!(
                    matches!(err, BackendError::Unavailable(_) | BackendError::Sqlx(_)),
                    "unexpected variant: {err:?}"
                );
            }
            Err(err) => {
                // `connect_with` itself rejected the URL. That's also a fine
                // negative signal.
                assert!(
                    matches!(err, BackendError::Unavailable(_) | BackendError::Sqlx(_)),
                    "unexpected variant: {err:?}"
                );
            }
        }
    }
}
