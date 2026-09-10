//! `[broker.postgres]` config resolution (contract item 2).
//!
//! Hermetic: every test here fails (or defaults) before any network call —
//! `PostgresBroker::connect` validates `idle_poll_secs`/`url` shape before
//! opening a connection, and the "no url, SQLite catalog" case is rejected in
//! `session.rs` before `connect` is ever reached. The live "url defaults from
//! `[catalog.postgres]`" success path is covered end-to-end by
//! `broker_parity.rs`'s Postgres arm (every parity test builds its broker via
//! `PostgresBroker::connect` directly, over `JAMMI_TEST_PG_URL`).

use jammi_db::config::{BrokerConfig, CatalogConfig, JammiConfig, Secret};
use jammi_db::session::JammiSession;

fn parse(toml_src: &str) -> JammiConfig {
    JammiConfig::parse_from(toml_src, std::iter::empty::<(String, String)>()).unwrap()
}

#[test]
fn broker_config_round_trip_postgres_explicit_url() {
    let cfg = parse(
        "[broker.postgres]\n\
         url = \"postgres://u:p@h/db\"\n\
         idle_poll_secs = 2\n",
    );
    assert_eq!(
        cfg.broker,
        BrokerConfig::Postgres {
            url: Some(Secret::from("postgres://u:p@h/db")),
            idle_poll_secs: 2,
        }
    );
}

#[test]
fn broker_config_postgres_defaults() {
    let cfg = parse("[broker.postgres]\n");
    assert_eq!(
        cfg.broker,
        BrokerConfig::Postgres {
            url: None,
            idle_poll_secs: 5,
        }
    );
}

#[test]
fn broker_postgres_url_is_redacted_like_catalog_postgres_url() {
    let cfg = parse("[broker.postgres]\nurl = \"postgres://u:s3cr3t@h/db\"\n");
    assert!(!format!("{cfg:?}").contains("s3cr3t"));
}

#[test]
fn broker_postgres_bogus_key_refuses_naming_it() {
    let err = JammiConfig::parse_from(
        "[broker.postgres]\nbogus = 1\n",
        std::iter::empty::<(String, String)>(),
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("bogus"), "{msg}");
}

/// `[broker.postgres]` with no `url`, over a SQLite catalog,
/// is a typed config error naming BOTH keys — there is no default to fall
/// back to. Never touches the network: the check runs before
/// `PostgresBroker::connect`.
#[tokio::test]
async fn sqlite_catalog_and_no_broker_url_names_both_keys() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = jammi_test_utils::test_config(dir.path());
    cfg.catalog = CatalogConfig::Sqlite { path: None };
    cfg.broker = BrokerConfig::Postgres {
        url: None,
        idle_poll_secs: 5,
    };
    let msg = match JammiSession::new(cfg).await {
        Ok(_) => panic!("expected a Config error naming both `url` keys"),
        Err(err) => err.to_string(),
    };
    assert!(msg.contains("broker.postgres"), "{msg}");
    assert!(msg.contains("catalog.postgres"), "{msg}");
}

/// Validation edge: `idle_poll_secs = 0` is a typed config error. Never touches the
/// network: `PostgresBroker::connect` validates this before dialing.
#[tokio::test]
async fn idle_poll_secs_zero_is_a_typed_config_error() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = jammi_test_utils::test_config(dir.path());
    cfg.broker = BrokerConfig::Postgres {
        url: Some(Secret::from("postgres://u:p@h/db")),
        idle_poll_secs: 0,
    };
    let msg = match JammiSession::new(cfg).await {
        Ok(_) => panic!("expected a Config error for idle_poll_secs = 0"),
        Err(err) => err.to_string(),
    };
    assert!(msg.contains("idle_poll_secs"), "{msg}");
    assert!(msg.contains(">= 1"), "{msg}");
}

/// Validation edge: a non-`postgres://` `url` is a typed config error. Never touches
/// the network: `PostgresBroker::connect` validates the scheme before
/// dialing.
#[tokio::test]
async fn non_postgres_url_is_a_typed_config_error() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = jammi_test_utils::test_config(dir.path());
    cfg.broker = BrokerConfig::Postgres {
        url: Some(Secret::from("nats://nats.svc:4222")),
        idle_poll_secs: 5,
    };
    let msg = match JammiSession::new(cfg).await {
        Ok(_) => panic!("expected a Config error for a non-postgres:// url"),
        Err(err) => err.to_string(),
    };
    assert!(msg.contains("postgres://"), "{msg}");
}
