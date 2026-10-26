//! The MySQL databases the MySQL source's live tests read: one per test, on
//! the server at `JAMMI_TEST_MYSQL_URL`.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use mysql_async::prelude::Queryable;

/// The prefix every per-test database carries, followed by the unix second
/// it was created at, which the sweep reads its age from.
const PREFIX: &str = "jammi_t_";

/// A per-test database older than this belongs to a run that has ended.
const TTL: Duration = Duration::from_secs(24 * 60 * 60);

/// A fresh database on the test server holding what `script` creates, as the
/// URL a MySQL source registers. The server's certificate is the self-signed
/// one MySQL generates at its first start, so the URL connects with
/// `sslmode=preferred`, which does not verify it.
///
/// # Panics
/// When `JAMMI_TEST_MYSQL_URL` is unset, the server refuses the connection, or
/// `script` fails.
pub async fn mysql_database_holding(script: &str) -> String {
    let server = jammi_test_resources::env("JAMMI_TEST_MYSQL_URL");
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("the clock reads after the epoch")
        .as_secs();
    let database = format!(
        "{PREFIX}{now}_{}",
        &uuid::Uuid::new_v4().simple().to_string()[..12]
    );
    let mut admin = mysql_async::Conn::from_url(format!(
        "{server}?require_ssl=true&verify_ca=false&verify_identity=false"
    ))
    .await
    .expect("connect to JAMMI_TEST_MYSQL_URL");
    sweep(&mut admin, now).await;
    admin
        .query_drop(format!(
            "CREATE DATABASE `{database}`; USE `{database}`; {script}"
        ))
        .await
        .unwrap_or_else(|e| panic!("set up the test database: {e}"));
    admin
        .disconnect()
        .await
        .expect("close the setup connection");
    let mut url = url::Url::parse(&server).expect("JAMMI_TEST_MYSQL_URL parses as a URL");
    url.set_path(&format!("/{database}"));
    url.set_query(Some("sslmode=preferred"));
    url.to_string()
}

/// Drop every per-test database older than [`TTL`].
async fn sweep(admin: &mut mysql_async::Conn, now: u64) {
    let names: Vec<String> = admin
        .exec(
            "SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE ?",
            (format!("{PREFIX}%"),),
        )
        .await
        .expect("list the per-test databases");
    for name in names {
        let created = name[PREFIX.len()..]
            .split('_')
            .next()
            .and_then(|secs| secs.parse::<u64>().ok());
        if created.is_some_and(|created| now.saturating_sub(created) > TTL.as_secs()) {
            admin
                .query_drop(format!("DROP DATABASE IF EXISTS `{name}`"))
                .await
                .unwrap_or_else(|e| panic!("DROP DATABASE {name}: {e}"));
        }
    }
}
