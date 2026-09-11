pub use jammi_test_utils::*;

/// Start a [`jammi_db::catalog::lease_keeper::LeaseKeeper`] whose connect
/// factory reopens a FRESH `Catalog` on `backend` every time it is invoked
/// (from inside the keeper thread's own runtime): the on-disk SQLite catalog
/// under `dir`, or the shared Postgres test database. Called only after the
/// test's own session opened on the same backend (so on the Postgres lane
/// `JAMMI_TEST_PG_URL` is known to be set) — never a runtime skip.
pub async fn keeper_for_backend(
    backend: jammi_db::catalog::backend::BackendKind,
    dir: std::path::PathBuf,
    intervals: jammi_db::catalog::lease::LeaseIntervals,
) -> std::sync::Arc<jammi_db::catalog::lease_keeper::LeaseKeeper> {
    use jammi_db::catalog::backend::{BackendImpl, BackendKind};
    use jammi_db::catalog::backend_postgres::PostgresBackend;
    use jammi_db::catalog::lease_keeper::LeaseKeeper;
    use jammi_db::catalog::Catalog;

    let url = match backend {
        BackendKind::Sqlite => None,
        BackendKind::Postgres => Some(
            pg_url_for_tests()
                .expect("the Postgres lane's session already opened, so JAMMI_TEST_PG_URL is set"),
        ),
    };
    LeaseKeeper::start(
        move || {
            let dir = dir.clone();
            let url = url.clone();
            Box::pin(async move {
                match url {
                    None => Catalog::open(&dir).await,
                    Some(url) => {
                        let pg = PostgresBackend::open_with_options(&url, 8, None).await?;
                        Ok(Catalog::from_backend(BackendImpl::Postgres(pg)))
                    }
                }
            })
        },
        intervals,
    )
    .await
    .expect("the keeper connects within the lease window")
}
