//! `[broker.postgres]` config resolution (contract item 2).
//!
//! Hermetic: every test here fails (or defaults) before any network call —
//! `idle_poll_secs = 0`, a non-`postgres://` `url`, and "no url, SQLite
//! catalog" are all rejected at the `session.rs` config seam, before
//! `PostgresBroker::connect` is ever reached, as a typed
//! [`jammi_db::error::JammiError::Config`] (the actual enum VARIANT, not
//! merely a message substring `PostgresBroker::connect`'s own
//! `TriggerError::Driver` — wrapped by `?` into `JammiError::Trigger` — would
//! also satisfy). The live "url defaults from `[catalog.postgres]`" success
//! path is covered end-to-end by `broker_parity.rs`'s Postgres arm (every
//! parity test builds its broker via `PostgresBroker::connect` directly, over
//! `JAMMI_TEST_PG_URL`).

use jammi_db::config::{BrokerConfig, CatalogConfig, JammiConfig, Secret};
use jammi_db::error::JammiError;
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

/// Validation edge: `idle_poll_secs = 0` is a typed config error -- the
/// actual [`JammiError::Config`] VARIANT, not merely a message that happens
/// to contain the right substring. Validated at the `session.rs` config
/// seam BEFORE `PostgresBroker::connect` is ever reached (never touches the
/// network): `PostgresBroker::connect`'s own validation of the same
/// condition returns `TriggerError::Driver`, which `?` wraps in
/// `JammiError::Trigger` -- a DIFFERENT variant whose `Display` text would
/// still satisfy a substring-only assertion, which is exactly the gap this
/// test closes.
#[tokio::test]
async fn idle_poll_secs_zero_is_a_typed_config_error() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = jammi_test_utils::test_config(dir.path());
    cfg.broker = BrokerConfig::Postgres {
        url: Some(Secret::from("postgres://u:p@h/db")),
        idle_poll_secs: 0,
    };
    let err = match JammiSession::new(cfg).await {
        Ok(_) => panic!("expected a Config error for idle_poll_secs = 0"),
        Err(err) => err,
    };
    match &err {
        JammiError::Config(msg) => {
            assert!(msg.contains("idle_poll_secs"), "{msg}");
            assert!(msg.contains(">= 1"), "{msg}");
        }
        other => panic!("expected JammiError::Config, got {other:?}"),
    }
}

/// Validation edge: a non-`postgres://` `url` is a typed config error -- the
/// actual [`JammiError::Config`] VARIANT, not merely a message substring
/// (same rationale as [`idle_poll_secs_zero_is_a_typed_config_error`]).
/// Validated at the `session.rs` config seam before `PostgresBroker::connect`
/// is ever reached (never touches the network).
#[tokio::test]
async fn non_postgres_url_is_a_typed_config_error() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = jammi_test_utils::test_config(dir.path());
    cfg.broker = BrokerConfig::Postgres {
        url: Some(Secret::from("nats://nats.svc:4222")),
        idle_poll_secs: 5,
    };
    let err = match JammiSession::new(cfg).await {
        Ok(_) => panic!("expected a Config error for a non-postgres:// url"),
        Err(err) => err,
    };
    match &err {
        JammiError::Config(msg) => assert!(msg.contains("postgres://"), "{msg}"),
        other => panic!("expected JammiError::Config, got {other:?}"),
    }
}

/// `BrokerKind::as_str()` must equal the `[broker.<kind>]` config tag for
/// EVERY variant, not just Postgres (the rest of this file's focus): checked
/// by actually parsing a minimal valid TOML stanza tagged with
/// `BrokerKind::as_str()`'s own output and confirming it resolves to the
/// matching `BrokerConfig` variant. A hand relabeling of one side (e.g.
/// renaming the `Postgres` variant's serde tag without updating `as_str()`,
/// or vice versa) fails this rather than only surfacing as a runtime
/// `ServerInfo.broker` string nobody happens to compare against the config
/// tag in one place.
#[test]
fn broker_kind_as_str_matches_every_config_tag() {
    use jammi_db::trigger::BrokerKind;

    // `in_memory`: the bare-string form (`broker = "in_memory"`).
    assert_eq!(BrokerKind::InMemory.as_str(), "in_memory");
    let cfg = parse(&format!("broker = \"{}\"\n", BrokerKind::InMemory.as_str()));
    assert_eq!(cfg.broker, BrokerConfig::InMemory);

    // `jet_stream`: requires `url`.
    assert_eq!(BrokerKind::JetStream.as_str(), "jet_stream");
    let cfg = parse(&format!(
        "[broker.{}]\nurl = \"nats://n:4222\"\n",
        BrokerKind::JetStream.as_str()
    ));
    assert!(
        matches!(cfg.broker, BrokerConfig::JetStream { .. }),
        "the `[broker.{}]` tag must resolve to `BrokerConfig::JetStream`",
        BrokerKind::JetStream.as_str()
    );

    // `postgres`: every field defaults.
    assert_eq!(BrokerKind::Postgres.as_str(), "postgres");
    let cfg = parse(&format!("[broker.{}]\n", BrokerKind::Postgres.as_str()));
    assert!(
        matches!(cfg.broker, BrokerConfig::Postgres { .. }),
        "the `[broker.{}]` tag must resolve to `BrokerConfig::Postgres`",
        BrokerKind::Postgres.as_str()
    );
}
