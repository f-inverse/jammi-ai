use super::*;

#[test]
fn catalog_config_round_trip_sqlite_default() {
    let toml_src = r#"
        [catalog.sqlite]
    "#;
    let cfg: JammiConfig = toml::from_str(toml_src).unwrap();
    assert_eq!(cfg.catalog, CatalogConfig::Sqlite { path: None });
}

#[test]
fn catalog_config_round_trip_sqlite_with_path() {
    let toml_src = r#"
        [catalog.sqlite]
        path = "/srv/jammi/catalog.db"
    "#;
    let cfg: JammiConfig = toml::from_str(toml_src).unwrap();
    assert_eq!(
        cfg.catalog,
        CatalogConfig::Sqlite {
            path: Some(PathBuf::from("/srv/jammi/catalog.db"))
        }
    );
}

#[test]
fn catalog_config_round_trip_postgres() {
    let toml_src = r#"
        [catalog.postgres]
        url = "postgres://u:p@h/db"
        pool_size = 16
        max_lifetime_secs = 1800
    "#;
    let cfg: JammiConfig = toml::from_str(toml_src).unwrap();
    assert_eq!(
        cfg.catalog,
        CatalogConfig::Postgres {
            url: "postgres://u:p@h/db".into(),
            pool_size: 16,
            max_lifetime_secs: Some(1800),
        }
    );
}

#[test]
fn catalog_config_postgres_defaults() {
    let toml_src = r#"
        [catalog.postgres]
        url = "postgres://u:p@h/db"
    "#;
    let cfg: JammiConfig = toml::from_str(toml_src).unwrap();
    assert_eq!(
        cfg.catalog,
        CatalogConfig::Postgres {
            url: "postgres://u:p@h/db".into(),
            pool_size: 8,
            max_lifetime_secs: None,
        }
    );
}

#[test]
fn broker_config_round_trip_in_memory() {
    let toml_src = r#"
        [broker.in_memory]
    "#;
    let cfg: JammiConfig = toml::from_str(toml_src).unwrap();
    assert_eq!(cfg.broker, BrokerConfig::InMemory);
}

#[test]
fn broker_config_round_trip_jetstream() {
    let toml_src = r#"
        [broker.jet_stream]
        url = "nats://nats.svc:4222"
        retention_seconds = 86400
        credentials = "-----BEGIN NATS USER JWT-----\nabc\n------END NATS USER JWT------"
    "#;
    let cfg: JammiConfig = toml::from_str(toml_src).unwrap();
    assert_eq!(
        cfg.broker,
        BrokerConfig::JetStream {
            url: "nats://nats.svc:4222".into(),
            retention_seconds: 86400,
            credentials: Some(Secret::from(
                "-----BEGIN NATS USER JWT-----\nabc\n------END NATS USER JWT------"
            )),
        }
    );
    assert!(!format!("{cfg:?}").contains("abc"));
}

/// `broker.credentials` carries what `async_nats::ConnectOptions::
/// with_credentials` takes — the `.creds` file CONTENTS — not a path. The
/// file form reads the file at load, so what reaches the session builder
/// is the JWT + seed text with its one trailing newline trimmed.
#[test]
fn jetstream_credentials_are_contents_not_a_path() {
    let dir = tempfile::tempdir().unwrap();
    let creds_path = dir.path().join("nats.creds");
    let contents = "-----BEGIN NATS USER JWT-----\neyJ0.abc\n------END NATS USER JWT------\n\
                    -----BEGIN USER NKEY SEED-----\nSUAB\n------END USER NKEY SEED------";
    std::fs::write(&creds_path, format!("{contents}\n")).unwrap();
    let toml_src = format!(
        r#"
        [broker.jet_stream]
        url = "nats://nats.svc:4222"
        credentials = {{ file = {:?} }}
    "#,
        creds_path.to_str().unwrap()
    );
    let cfg: JammiConfig = toml::from_str(&toml_src).unwrap();
    let BrokerConfig::JetStream { credentials, .. } = &cfg.broker else {
        panic!("expected jet_stream, got {:?}", cfg.broker);
    };
    let resolved = credentials.as_ref().expect("credentials set");
    assert_eq!(resolved.expose(), contents);
    assert_ne!(resolved.expose(), creds_path.to_str().unwrap());
    assert!(!format!("{cfg:?}").contains("SUAB"));
}

#[test]
fn broker_config_jetstream_defaults() {
    let toml_src = r#"
        [broker.jet_stream]
        url = "nats://nats.svc:4222"
    "#;
    let cfg: JammiConfig = toml::from_str(toml_src).unwrap();
    assert_eq!(
        cfg.broker,
        BrokerConfig::JetStream {
            url: "nats://nats.svc:4222".into(),
            retention_seconds: 7 * 24 * 60 * 60,
            credentials: None,
        }
    );
}

#[test]
fn jammi_config_default_uses_sqlite_and_in_memory() {
    let cfg = JammiConfig::default();
    assert_eq!(cfg.catalog, CatalogConfig::Sqlite { path: None });
    assert_eq!(cfg.broker, BrokerConfig::InMemory);
}

/// The redaction oracle (H9 scope, extended per the phase-4 audit): a
/// `Debug` render of the whole config — the thing a startup log line or a
/// panic message prints — must never carry a secret, at ANY secret
/// position the config has: the catalog URL, broker credentials, an HTTP
/// header, `models.hub_token`, and every `storage.cloud.*` credential field
/// across all four cloud variants, in both the inline AND the `{ file = … }`
/// spelling. Each cloud variant gets its own document (the field is a
/// singular `Option<CloudConfig>`, so only one variant can be active at
/// once), sharing one `base` fragment for the non-cloud positions.
#[test]
fn jammi_config_debug_never_prints_a_secret() {
    let dir = tempfile::tempdir().unwrap();
    let write_secret = |name: &str, contents: &str| -> String {
        let path = dir.path().join(name);
        std::fs::write(&path, contents).unwrap();
        path.to_str().unwrap().to_string()
    };
    let s3_secret_file = write_secret("s3-secret", "s3-file-secret-xyz");
    let gcs_sa_file = write_secret(
        "gcs-sa",
        "{\"type\":\"service_account\",\"key\":\"gcs-file-secret-xyz\"}",
    );
    let azure_secret_file = write_secret("azure-secret", "azure-file-clientsecret-xyz");

    let plaintexts = [
        "hunter2-pw",
        "Bearer tok-4f9a-secret",
        "nats-jwt-secret-abc",
        "nats-url-token-secret",
        "hf-inline-hubtoken-xyz",
        "s3-inline-secret-xyz",
        "s3-file-secret-xyz",
        "s3-inline-sessiontoken-xyz",
        "r2-inline-secret-xyz",
        "gcs-file-secret-xyz",
        "azure-inline-accountkey-xyz",
        "azure-inline-sastoken-xyz",
        "azure-file-clientsecret-xyz",
    ];

    let base = r#"
        [catalog.postgres]
        url = "postgres://jammi:hunter2-pw@db.internal:5432/jammi"

        [broker.jet_stream]
        url = "nats://nats-user:nats-url-token-secret@nats.svc:4222"
        credentials = "nats-jwt-secret-abc"

        [inference.http.headers]
        Authorization = "Bearer tok-4f9a-secret"

        [models]
        hub_token = "hf-inline-hubtoken-xyz"
    "#;

    // S3: secret_access_key via the `{ file = … }` form, session_token
    // inline — both secret-valued S3 fields covered in one document.
    let s3_src = format!(
        "{base}\n[storage.cloud.s3]\nsecret_access_key = {{ file = {s3_secret_file:?} }}\n\
         session_token = \"s3-inline-sessiontoken-xyz\"\n"
    );
    // R2: secret_access_key inline.
    let r2_src = format!(
        "{base}\n[storage.cloud.r2]\naccount_id = \"acct\"\n\
         secret_access_key = \"r2-inline-secret-xyz\"\n"
    );
    // GCS: service_account via the `{ file = … }` form.
    let gcs_src =
        format!("{base}\n[storage.cloud.gcs]\nservice_account = {{ file = {gcs_sa_file:?} }}\n");
    // Azure: account_key inline, client_secret via the `{ file = … }` form —
    // both spellings in one document.
    let azure_src = format!(
        "{base}\n[storage.cloud.azure]\naccount_name = \"acct\"\n\
         account_key = \"azure-inline-accountkey-xyz\"\n\
         client_secret = {{ file = {azure_secret_file:?} }}\n\
         tenant_id = \"t\"\nclient_id = \"c\"\n"
    );
    // Azure `sas_token`, inline, in its own document (mutually exclusive
    // with `account_key` per `AzureConfig::validate` — untested here since
    // this test never calls `validate()`, but kept separate for clarity).
    let azure_sas_src = format!(
        "{base}\n[storage.cloud.azure]\naccount_name = \"acct\"\n\
         sas_token = \"azure-inline-sastoken-xyz\"\n"
    );

    for src in [s3_src, r2_src, gcs_src, azure_src, azure_sas_src] {
        let cfg: JammiConfig = toml::from_str(&src).unwrap_or_else(|e| panic!("{src}\n{e}"));
        let rendered = format!("{cfg:?}");
        let pretty = format!("{cfg:#?}");
        for secret in plaintexts {
            assert!(
                !rendered.contains(secret),
                "Debug leaked {secret:?} (src:\n{src}):\n{rendered}"
            );
            assert!(
                !pretty.contains(secret),
                "Debug (pretty) leaked {secret:?} (src:\n{src}):\n{pretty}"
            );
        }
    }
}

/// Phase-4 audit item 1 (HIGH): the file layer is parsed AFTER `${VAR}`
/// interpolation, so a TOML syntax error on the interpolated text must
/// never render via `Display` (which quotes a code frame of the offending
/// source line verbatim — the exact source line an unquoted `url =
/// ${POSTGRES_URL}` expansion turns into the secret itself). An unquoted
/// expansion is exactly this: `postgres://u:hunter2-file-secret@h/db` is
/// not valid bare TOML syntax, so this is a genuine parse error, not a
/// contrived one.
#[test]
fn file_parse_error_after_env_interpolation_never_echoes_the_expanded_secret() {
    let env = vec![(
        "PG".to_string(),
        "postgres://u:hunter2-file-secret@h/db".to_string(),
    )];
    let toml_src = "[catalog.postgres]\nurl = ${PG}\n";
    let err = JammiConfig::parse_from(toml_src, env).unwrap_err();
    match err {
        JammiError::Config(msg) => {
            assert!(!msg.contains("hunter2-file-secret"), "msg = {msg}");
            assert!(
                msg.contains("line") && msg.contains("column"),
                "expected a line:column locator, msg = {msg}"
            );
        }
        other => panic!("expected JammiError::Config, got {other:?}"),
    }
}

/// Phase-4 audit item 2 (MEDIUM): a whole-value env override at a seq/map
/// position that fails serde's `invalid_type` check must not echo the
/// value — `invalid type: string "…", expected …` bakes the value into the
/// message text with no way to strip it after the fact, so the whole
/// message is replaced with a fixed "expected <shape>" phrase instead.
#[test]
fn env_whole_value_type_mismatch_at_a_map_position_never_echoes_the_value() {
    let err = JammiConfig::parse_from(
        "",
        vec![(
            "JAMMI_INFERENCE__HTTP__HEADERS".to_string(),
            "\"Bearer hunter2-env-secret\"".to_string(),
        )],
    )
    .unwrap_err();
    match err {
        JammiError::Config(msg) => assert!(!msg.contains("hunter2-env-secret"), "msg = {msg}"),
        other => panic!("expected JammiError::Config, got {other:?}"),
    }
}

/// The missing arm: `Node::parse_as_toml`'s OWN error path (layers.rs's
/// "env value is not valid TOML" branch) — genuinely malformed TOML syntax
/// at a seq/map-position env leaf, as opposed to the sibling test above
/// (valid TOML, wrong shape). Only the FILE arm of this same safe-rendering
/// discipline was covered before this test (`describe_toml_error` is
/// exercised by `file_parse_error_after_env_interpolation_never_echoes_the_expanded_secret`);
/// this pins the ENV-leaf arm names the offending variable and never echoes
/// the value.
#[test]
fn env_leaf_invalid_toml_syntax_never_echoes_the_value_and_names_the_variable() {
    let err = JammiConfig::parse_from(
        "",
        vec![(
            "JAMMI_INFERENCE__HTTP__HEADERS".to_string(),
            // An unterminated string literal: not valid TOML at all, so this
            // hits `Node::parse_as_toml`'s own parse-failure branch rather
            // than a downstream `invalid_type` check.
            "\"unterminated-hunter2-env-secret".to_string(),
        )],
    )
    .unwrap_err();
    match err {
        JammiError::Config(msg) => {
            assert!(!msg.contains("hunter2-env-secret"), "msg = {msg}");
            assert!(
                msg.contains("JAMMI_INFERENCE__HTTP__HEADERS"),
                "msg = {msg}"
            );
        }
        other => panic!("expected JammiError::Config, got {other:?}"),
    }
}

/// Phase-4 audit item 2's FILE-arm half: a file value can itself be a
/// `${VAR}` expansion, so the same `invalid_type` leak is reachable with NO
/// env override at all — `[inference.http] headers = "${TOKEN}"` expands to
/// a bare string at a map position.
#[test]
fn file_value_type_mismatch_at_a_map_position_never_echoes_an_expanded_secret() {
    let env = vec![(
        "TOKEN".to_string(),
        "Bearer hunter2-file-value-secret".to_string(),
    )];
    let toml_src = "[inference.http]\nheaders = \"${TOKEN}\"\n";
    let err = JammiConfig::parse_from(toml_src, env).unwrap_err();
    match err {
        JammiError::Config(msg) => {
            assert!(!msg.contains("hunter2-file-value-secret"), "msg = {msg}")
        }
        other => panic!("expected JammiError::Config, got {other:?}"),
    }
}

/// Phase-4 audit item 3 (MEDIUM), the direct bare-env case: an env
/// whole-value at an ENUM position that does not itself name a variant
/// must not be echoed as the "unknown variant" — the message names only
/// the variable and the expected variant list.
#[test]
fn env_whole_value_at_an_enum_position_never_echoes_the_value() {
    let err = JammiConfig::parse_from(
        "",
        vec![(
            "JAMMI_CATALOG".to_string(),
            "{ postgres = { url = \"postgres://u:hunter2-enum-secret@h/db\" } }".to_string(),
        )],
    )
    .unwrap_err();
    match err {
        JammiError::Config(msg) => {
            assert!(!msg.contains("hunter2-enum-secret"), "msg = {msg}");
            assert!(msg.contains("JAMMI_CATALOG"), "msg = {msg}");
        }
        other => panic!("expected JammiError::Config, got {other:?}"),
    }
}

/// Phase-4 audit item 3, the `Node::Override` enum-lowering path: the SAME
/// leak, but reached with a file layer present (forcing the merge to
/// record `Node::Override` instead of a bare `Node::Env`) — the lowering
/// arm used to synthesize a table keyed by the raw value before checking
/// variant membership, letting the generic multi-key path re-echo it (and
/// misattribute the origin to "the config file").
#[test]
fn env_whole_value_at_an_enum_position_via_override_never_echoes_the_value() {
    let err = JammiConfig::parse_from(
        "[catalog.sqlite]\n",
        vec![(
            "JAMMI_CATALOG".to_string(),
            "{ postgres = { url = \"postgres://u:hunter2-override-secret@h/db\" } }".to_string(),
        )],
    )
    .unwrap_err();
    match err {
        JammiError::Config(msg) => {
            assert!(!msg.contains("hunter2-override-secret"), "msg = {msg}");
            assert!(msg.contains("JAMMI_CATALOG"), "msg = {msg}");
        }
        other => panic!("expected JammiError::Config, got {other:?}"),
    }
}

/// Phase-4 audit item 4 (LOW): `broker.jet_stream.url` is `Secret`-typed —
/// a NATS URL can carry userinfo/token auth inline
/// (`nats://user:pass@host`), the same class of leak `catalog.postgres.url`
/// guards against.
#[test]
fn broker_jetstream_url_is_redacted_like_catalog_postgres_url() {
    let toml_src = r#"
        [broker.jet_stream]
        url = "nats://nats-user:hunter2-nats-url-secret@nats.svc:4222"
    "#;
    let cfg: JammiConfig = toml::from_str(toml_src).unwrap();
    let BrokerConfig::JetStream { url, .. } = &cfg.broker else {
        panic!("expected jet_stream, got {:?}", cfg.broker);
    };
    assert_eq!(
        url.expose(),
        "nats://nats-user:hunter2-nats-url-secret@nats.svc:4222"
    );
    assert!(!format!("{cfg:?}").contains("hunter2-nats-url-secret"));
}

/// Phase-4 audit item 5 (LOW): `ServiceSelection`'s array form trims and
/// filters empties exactly like its comma-list string form — a trailing
/// newline or blank token from a templated array value is not a value.
#[test]
fn services_array_form_trims_and_filters_empties_like_the_comma_list_form() {
    #[derive(Debug, serde::Deserialize)]
    struct Holder {
        services: ServiceSelection,
    }
    let padded: Holder = toml::from_str(r#"services = [" event ", "", "eval"]"#).unwrap();
    assert_eq!(
        padded.services,
        ServiceSelection::Only(vec!["event".into(), "eval".into()])
    );
}

#[test]
fn signing_key_config_round_trip_env() {
    let toml_src = r#"
        signing_key = "env"
    "#;
    let cfg: JammiConfig = toml::from_str(toml_src).unwrap();
    assert_eq!(cfg.signing_key, SigningKeyConfig::Env);
}

#[test]
fn signing_key_config_round_trip_file() {
    let toml_src = r#"
        [signing_key.file]
        path = "/run/secrets/audit-master-key"
    "#;
    let cfg: JammiConfig = toml::from_str(toml_src).unwrap();
    assert_eq!(
        cfg.signing_key,
        SigningKeyConfig::File {
            path: PathBuf::from("/run/secrets/audit-master-key")
        }
    );
}

#[test]
fn http_headers_deserialise_inline_and_file_forms() {
    let dir = tempfile::tempdir().unwrap();
    let token_path = dir.path().join("hf-token");
    std::fs::write(&token_path, "hf_filetoken\n").unwrap();
    let toml_src = format!(
        r#"
        [inference.http.headers]
        Authorization = "Bearer x"
        X-Api-Key = {{ file = {:?} }}
    "#,
        token_path.to_str().unwrap()
    );
    let cfg: JammiConfig = toml::from_str(&toml_src).unwrap();
    let headers = &cfg.inference.http.headers;
    assert_eq!(headers.len(), 2);
    assert_eq!(headers["Authorization"].expose(), "Bearer x");
    assert_eq!(headers["X-Api-Key"].expose(), "hf_filetoken");
    let rendered = format!("{cfg:?}");
    assert!(!rendered.contains("Bearer x") && !rendered.contains("hf_filetoken"));
    // Keys are still visible: the header NAME is not a secret.
    assert!(rendered.contains("Authorization"));
}

#[test]
fn jammi_config_default_signing_key_is_env() {
    assert_eq!(JammiConfig::default().signing_key, SigningKeyConfig::Env);
}

#[test]
fn signing_key_absent_defaults_to_env() {
    // `JammiConfig` is `#[serde(default)]`, so TOML without `[signing_key]`
    // parses to the env-backed default.
    let cfg: JammiConfig = toml::from_str("artifact_dir = \"/tmp/jammi\"").unwrap();
    assert_eq!(cfg.signing_key, SigningKeyConfig::Env);
}

#[test]
fn storage_config_default_is_local() {
    let cfg = JammiConfig::default();
    assert!(cfg.storage.result_root.is_none());
    assert!(cfg.storage.cloud.is_none());
}

#[test]
fn storage_config_round_trip_r2() {
    // Secrets (access_key_id / secret_access_key) are deliberately absent:
    // they come from the container's AWS_* env vars at driver-build time.
    let toml_src = r#"
        [storage]
        result_root = "r2://jammi-results/prod"

        [storage.cloud.r2]
        account_id = "abc123def456"
    "#;
    let cfg: JammiConfig = toml::from_str(toml_src).unwrap();
    assert_eq!(
        cfg.storage.result_root.as_deref(),
        Some("r2://jammi-results/prod")
    );
    match cfg.storage.cloud {
        Some(CloudConfig::R2(r2)) => {
            assert_eq!(r2.account_id.as_deref(), Some("abc123def456"));
            assert!(r2.access_key_id.is_none());
            assert!(r2.secret_access_key.is_none());
        }
        other => panic!("expected R2 cloud config, got {other:?}"),
    }
}

#[test]
fn storage_config_round_trip_s3() {
    let toml_src = r#"
        [storage]
        result_root = "s3://jammi-results/prod"

        [storage.cloud.s3]
        region = "us-east-1"
    "#;
    let cfg: JammiConfig = toml::from_str(toml_src).unwrap();
    assert_eq!(
        cfg.storage.result_root.as_deref(),
        Some("s3://jammi-results/prod")
    );
    match cfg.storage.cloud {
        Some(CloudConfig::S3(s3)) => {
            assert_eq!(s3.region.as_deref(), Some("us-east-1"));
            assert!(s3.access_key_id.is_none());
        }
        other => panic!("expected S3 cloud config, got {other:?}"),
    }
}

#[test]
fn load_rejects_partial_r2_credentials() {
    // account_id + access_key_id but no secret — the fail-closed
    // CloudConfig::validate must reject this at load time.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("jammi.toml");
    std::fs::write(
        &path,
        r#"
            [storage]
            result_root = "r2://bucket/prefix"

            [storage.cloud.r2]
            account_id = "abc"
            access_key_id = "only-the-key"
        "#,
    )
    .unwrap();
    let err = JammiConfig::load_from(Some(&path), std::iter::empty()).unwrap_err();
    assert!(
        matches!(err, JammiError::Storage(_)),
        "expected a Storage validation error, got {err:?}"
    );
}

#[test]
fn lease_config_defaults_match_engine_constants() {
    // The defaults must reproduce the engine's built-in lease timing so a
    // config without a `[lease]` section behaves identically. These two
    // values are the contract; every lease holder derives its `Duration`s
    // from them.
    let l = LeaseConfig::default();
    assert_eq!(l.duration_secs, 30);
    assert_eq!(l.heartbeat_secs, 10);
    let intervals = l.intervals().unwrap();
    assert_eq!(intervals.lease(), std::time::Duration::from_secs(30));
    assert_eq!(intervals.heartbeat(), std::time::Duration::from_secs(10));
    assert_eq!(intervals, LeaseIntervals::default());
}

#[test]
fn training_config_defaults_match_engine_constants() {
    // The worker's own default (1 s idle poll) plus the shared lease.
    let t = TrainingConfig::default();
    assert!(t.run_worker);
    assert_eq!(t.idle_poll_secs, 1);

    let intervals = t
        .worker_intervals(LeaseConfig::default().intervals().unwrap())
        .unwrap();
    assert_eq!(intervals.lease, std::time::Duration::from_secs(30));
    assert_eq!(intervals.heartbeat, std::time::Duration::from_secs(10));
    assert_eq!(intervals.idle_poll, std::time::Duration::from_secs(1));
}

#[test]
fn training_config_absent_defaults() {
    // A config without `[training]` / `[lease]` parses to the engine defaults.
    let cfg: JammiConfig = toml::from_str("artifact_dir = \"/tmp/jammi\"").unwrap();
    assert_eq!(cfg.training, TrainingConfig::default());
    assert_eq!(cfg.lease, LeaseConfig::default());
}

#[test]
fn lease_and_training_config_round_trip() {
    let toml_src = r#"
        [lease]
        duration_secs = 8
        heartbeat_secs = 2

        [training]
        idle_poll_secs = 1
    "#;
    let cfg: JammiConfig = toml::from_str(toml_src).unwrap();
    assert_eq!(
        cfg.lease,
        LeaseConfig {
            duration_secs: 8,
            heartbeat_secs: 2,
        }
    );
    assert_eq!(
        cfg.training,
        TrainingConfig {
            // Not spelled in the TOML above: the container-level
            // `#[serde(default)]` must fill it from `Default`, on.
            run_worker: true,
            idle_poll_secs: 1,
        }
    );
    let intervals = cfg
        .training
        .worker_intervals(cfg.lease.intervals().unwrap())
        .unwrap();
    assert_eq!(intervals.lease, std::time::Duration::from_secs(8));
    assert_eq!(intervals.heartbeat, std::time::Duration::from_secs(2));
}

#[test]
fn old_training_lease_keys_are_refused() {
    // The lease keys moved to `[lease]`; a TOML still naming them under
    // `[training]` is a typed parse error (fail-closed), never a silent
    // fall-back to the default 30 s / 10 s timing.
    let toml_src = r#"
        [training]
        lease_duration_secs = 8
        heartbeat_interval_secs = 2
    "#;
    let err = toml::from_str::<JammiConfig>(toml_src).unwrap_err();
    assert!(
        err.to_string().contains("lease_duration_secs"),
        "the error must name the refused key, got: {err}"
    );
    // And an unknown key under `[lease]` is refused the same way.
    let err = toml::from_str::<JammiConfig>("[lease]\nlease_duration_secs = 8\n").unwrap_err();
    assert!(
        err.to_string().contains("lease_duration_secs"),
        "got: {err}"
    );
}

// ── `run_worker`: whether THIS process runs the claim loop ──────────────

#[test]
fn training_config_default_runs_the_worker() {
    // An unconfigured deployment is a whole one: it accepts jobs AND works
    // them. Opting out has to be an explicit act, so the default is on.
    assert!(TrainingConfig::default().run_worker);
}

#[test]
fn training_config_toml_without_run_worker_defaults_to_true() {
    // A `[training]` section that predates the key — the shape every
    // already-deployed config file has — still parses, and parses to on.
    let toml_src = r#"
        [training]
        idle_poll_secs = 1
    "#;
    let cfg: JammiConfig = toml::from_str(toml_src).unwrap();
    assert!(cfg.training.run_worker);
    assert_eq!(cfg.training, TrainingConfig::default());

    // And a file with no `[training]` section at all.
    let bare: JammiConfig = toml::from_str("artifact_dir = \"/tmp/jammi\"").unwrap();
    assert!(bare.training.run_worker);
}

#[test]
fn training_config_toml_run_worker_false_parses_to_false() {
    let toml_src = r#"
        [training]
        run_worker = false
    "#;
    let cfg: JammiConfig = toml::from_str(toml_src).unwrap();
    assert!(!cfg.training.run_worker);
    // Switching the loop off leaves the timing at its default — and it is
    // still validated, so a config that switches the loop back on later
    // cannot smuggle in timing that was never checked.
    assert_eq!(cfg.training.idle_poll_secs, 1);
    assert!(cfg
        .training
        .worker_intervals(cfg.lease.intervals().unwrap())
        .is_ok());
}

#[test]
fn training_config_equality_distinguishes_run_worker() {
    // The derived `PartialEq`/`Eq` must actually see the new field: two
    // configs identical but for `run_worker` are NOT equal. Without this,
    // the `assert_eq!(cfg.training, ...)` assertions above would hold
    // vacuously for a field equality ignored.
    let on = TrainingConfig::default();
    let off = TrainingConfig {
        run_worker: false,
        ..TrainingConfig::default()
    };
    assert_ne!(on, off);
    assert_eq!(off, off.clone());
    assert_eq!(on, TrainingConfig::default());
}

#[test]
fn parse_env_bool_accepts_the_four_spellings() {
    // Every spelling a shell, a container manifest, or an orchestrator
    // template actually emits — case-insensitive, whitespace-trimmed.
    for raw in ["true", "TRUE", "True", "1", " true ", "\ttrue\n"] {
        assert!(
            parse_env_bool("JAMMI_TEST__BOOL", raw).unwrap(),
            "raw = {raw:?}"
        );
    }
    for raw in ["false", "FALSE", "False", "0", " false ", "\tfalse\n"] {
        assert!(
            !parse_env_bool("JAMMI_TEST__BOOL", raw).unwrap(),
            "raw = {raw:?}"
        );
    }
}

#[test]
fn parse_env_bool_rejects_everything_outside_the_domain() {
    // The control: each of these is a real thing an operator types, and
    // every one must be a typed error rather than a silent guess. The empty
    // and whitespace-only cases matter most — `export VAR=` is the classic
    // accidental "unset", and reading it as `false` would stop a process
    // claiming work with nothing in the config file to explain it.
    for raw in [
        "",
        " ",
        "\t",
        "yes",
        "no",
        "on",
        "off",
        "y",
        "n",
        "2",
        "-1",
        "0.0",
        "01",
        "truee",
        "true false",
        "null",
        "none",
        "\"true\"",
    ] {
        let err = parse_env_bool("JAMMI_TRAINING__RUN_WORKER", raw).unwrap_err();
        match err {
            JammiError::Config(msg) => {
                assert!(
                    msg.contains("JAMMI_TRAINING__RUN_WORKER"),
                    "message must name the variable; raw = {raw:?}, msg = {msg}"
                );
                assert!(
                    msg.contains("true, false, 1, 0"),
                    "message must name the accepted set; raw = {raw:?}, msg = {msg}"
                );
            }
            other => panic!("expected JammiError::Config for {raw:?}, got {other:?}"),
        }
    }
}

#[test]
fn env_override_run_worker_flips_the_file_in_both_directions() {
    let dir = tempfile::tempdir().unwrap();
    let on_path = dir.path().join("on.toml");
    std::fs::write(&on_path, "[training]\nrun_worker = true\n").unwrap();
    let off_path = dir.path().join("off.toml");
    std::fs::write(&off_path, "[training]\nrun_worker = false\n").unwrap();

    // true in the file, false in the env → the env wins.
    let cfg = JammiConfig::load_from(
        Some(&on_path),
        vec![(
            "JAMMI_TRAINING__RUN_WORKER".to_string(),
            "false".to_string(),
        )],
    )
    .unwrap();
    assert!(
        !cfg.training.run_worker,
        "env `false` must override file `true`"
    );

    // false in the file, true in the env → the env wins the other way.
    // Spelled `1` so the numeric form is proven through `load_from`, not
    // only in the parser's own unit test.
    let cfg = JammiConfig::load_from(
        Some(&off_path),
        vec![("JAMMI_TRAINING__RUN_WORKER".to_string(), "1".to_string())],
    )
    .unwrap();
    assert!(
        cfg.training.run_worker,
        "env `1` must override file `false`"
    );

    // No override → the file's value stands, in both directions. Without
    // this leg an arm that unconditionally wrote `true` would still pass
    // above.
    assert!(
        !JammiConfig::load_from(Some(&off_path), std::iter::empty())
            .unwrap()
            .training
            .run_worker
    );
    assert!(
        JammiConfig::load_from(Some(&on_path), std::iter::empty())
            .unwrap()
            .training
            .run_worker
    );
}

#[test]
fn env_override_run_worker_unparsable_is_a_typed_load_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("jammi.toml");
    std::fs::write(&path, "[training]\nrun_worker = false\n").unwrap();

    let err = JammiConfig::load_from(
        Some(&path),
        vec![(
            "JAMMI_TRAINING__RUN_WORKER".to_string(),
            "maybe".to_string(),
        )],
    )
    .unwrap_err();

    match err {
        JammiError::Config(msg) => {
            assert!(msg.contains("JAMMI_TRAINING__RUN_WORKER"), "msg = {msg}");
            // Names the variable only, never the value (a bool-typed env
            // leaf sits at the same position shape a secret-typed sibling
            // does elsewhere in this config, so this error must stay safe
            // to paste into a startup log regardless of which field it is).
            assert!(!msg.contains("maybe"), "msg = {msg}");
        }
        other => panic!("expected JammiError::Config, got {other:?}"),
    }

    // With no override the same file loads cleanly — the error came from
    // the override, not from the file.
    assert!(
        !JammiConfig::load_from(Some(&path), std::iter::empty())
            .unwrap()
            .training
            .run_worker
    );
}

#[test]
fn env_override_lease_lands_through_the_struct_derived_layer() {
    // `[lease]` is an ordinary top-level section: `JAMMI_LEASE__<FIELD>`
    // reaches `LeaseConfig` through the same struct-derived env layer every
    // other section uses, overriding the file per-field and leaving the
    // sibling field at the file's value.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("jammi.toml");
    std::fs::write(&path, "[lease]\nduration_secs = 30\nheartbeat_secs = 10\n").unwrap();

    let cfg = JammiConfig::load_from(
        Some(&path),
        vec![("JAMMI_LEASE__DURATION_SECS".to_string(), "90".to_string())],
    )
    .unwrap();
    assert_eq!(
        cfg.lease,
        LeaseConfig {
            duration_secs: 90,
            heartbeat_secs: 10,
        }
    );

    // Both fields from env, over an empty file layer.
    let cfg = JammiConfig::parse_from(
        "",
        vec![
            ("JAMMI_LEASE__DURATION_SECS".to_string(), "8".to_string()),
            ("JAMMI_LEASE__HEARTBEAT_SECS".to_string(), "2".to_string()),
        ],
    )
    .unwrap();
    assert_eq!(
        cfg.lease.intervals().unwrap().lease(),
        std::time::Duration::from_secs(8)
    );
    assert_eq!(
        cfg.lease.intervals().unwrap().heartbeat(),
        std::time::Duration::from_secs(2)
    );

    // No override -> the file's value stands.
    assert_eq!(
        JammiConfig::load_from(Some(&path), std::iter::empty())
            .unwrap()
            .lease,
        LeaseConfig::default()
    );
}

#[test]
fn env_override_lease_violating_margin_is_a_typed_load_error() {
    // The env layer feeds the SAME post-load validation the file does
    // (`load_from`, not the parse-only `parse_from`): an env-supplied
    // heartbeat with no margin under the lease is refused at load, not at
    // the first heartbeat.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("jammi.toml");
    std::fs::write(&path, "").unwrap();
    let err = JammiConfig::load_from(
        Some(&path),
        vec![("JAMMI_LEASE__HEARTBEAT_SECS".to_string(), "15".to_string())],
    )
    .unwrap_err();
    assert!(
        matches!(&err, JammiError::Config(m) if m.contains("heartbeat_secs")),
        "got {err:?}"
    );
}

#[test]
fn env_override_lease_unknown_field_refuses() {
    // `deny_unknown_fields` holds through the env layer too: the former
    // `[training]` spelling of the lease keys is not an alias under
    // `[lease]` either.
    let err = JammiConfig::parse_from(
        "",
        vec![(
            "JAMMI_LEASE__LEASE_DURATION_SECS".to_string(),
            "8".to_string(),
        )],
    )
    .unwrap_err();
    assert!(
        matches!(&err, JammiError::Config(m) if m.contains("lease_duration_secs")),
        "the error must name the refused key, got {err:?}"
    );
}

#[test]
fn lease_config_rejects_heartbeat_without_margin() {
    // heartbeat == lease (no margin) — a live holder's lease would expire
    // between beats. Must be rejected, not clamped.
    let equal = LeaseConfig {
        duration_secs: 10,
        heartbeat_secs: 10,
    };
    let err = equal.intervals().unwrap_err();
    assert!(
        matches!(&err, JammiError::Config(m) if m.contains("heartbeat_secs")),
        "expected a Config error naming the heartbeat field, got {err:?}"
    );

    // heartbeat * 2 just over lease (margin not cleared) — also rejected.
    let too_close = LeaseConfig {
        duration_secs: 19,
        heartbeat_secs: 10,
    };
    assert!(matches!(too_close.intervals(), Err(JammiError::Config(_))));

    // heartbeat > lease — clearly rejected.
    let inverted = LeaseConfig {
        duration_secs: 5,
        heartbeat_secs: 30,
    };
    assert!(matches!(inverted.intervals(), Err(JammiError::Config(_))));

    // The exact 2× boundary (heartbeat * 2 == lease) is REJECTED: a
    // renewal coincident with expiry races a reclaim.
    let exact = LeaseConfig {
        duration_secs: 20,
        heartbeat_secs: 10,
    };
    assert!(
        matches!(exact.intervals(), Err(JammiError::Config(_))),
        "heartbeat * 2 == lease must be rejected under the strict margin"
    );

    // Strictly under half (heartbeat * 2 < lease) is accepted.
    let strict = LeaseConfig {
        duration_secs: 21,
        heartbeat_secs: 10,
    };
    assert!(strict.intervals().is_ok());
}

#[test]
fn lease_config_margin_check_is_overflow_safe() {
    // An operator-controlled heartbeat whose doubling overflows `u64` must
    // be rejected with a Config error — never a debug-build panic, never a
    // release-build silent wrap-to-zero that accepts bogus timing.
    let absurd = LeaseConfig {
        duration_secs: 30,
        heartbeat_secs: u64::MAX / 2 + 1,
    };
    assert!(
        matches!(absurd.intervals(), Err(JammiError::Config(_))),
        "a heartbeat whose doubling overflows u64 must be a Config error"
    );
}

#[test]
fn training_config_rejects_zero_idle_poll() {
    let cfg = TrainingConfig {
        idle_poll_secs: 0,
        ..Default::default()
    };
    let err = cfg
        .worker_intervals(LeaseConfig::default().intervals().unwrap())
        .unwrap_err();
    assert!(
        matches!(&err, JammiError::Config(m) if m.contains("idle_poll_secs")),
        "expected a Config error naming the idle-poll field, got {err:?}"
    );
}

#[test]
fn lease_config_rejects_zero_heartbeat_and_zero_duration() {
    let zero_hb = LeaseConfig {
        duration_secs: 30,
        heartbeat_secs: 0,
    };
    assert!(matches!(zero_hb.intervals(), Err(JammiError::Config(_))));
    let zero_lease = LeaseConfig {
        duration_secs: 0,
        heartbeat_secs: 0,
    };
    let err = zero_lease.intervals().unwrap_err();
    assert!(matches!(&err, JammiError::Config(m) if m.contains("must be > 0")));
}

#[test]
fn load_rejects_invalid_training_timing() {
    // The load path enforces the invariant: a heartbeat with no margin in
    // the TOML is a hard load error.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("jammi.toml");
    std::fs::write(
        &path,
        r#"
            [lease]
            duration_secs = 5
            heartbeat_secs = 5

            [training]
            idle_poll_secs = 1
        "#,
    )
    .unwrap();
    let err = JammiConfig::load_from(Some(&path), std::iter::empty()).unwrap_err();
    assert!(matches!(err, JammiError::Config(_)), "got {err:?}");
}

#[test]
fn interpolate_env_vars_happy_path() {
    let out = interpolate_env_vars("url = \"${JAMMI_TEST_INTERP_HAPPY}\"", |n| {
        (n == "JAMMI_TEST_INTERP_HAPPY").then(|| "from-env".to_string())
    })
    .unwrap();
    assert_eq!(out, "url = \"from-env\"");
}

#[test]
fn interpolate_env_vars_missing_is_typed_error() {
    let err = interpolate_env_vars("url = \"${JAMMI_TEST_INTERP_DEFINITELY_NOT_SET}\"", |_| {
        None
    })
    .unwrap_err();
    match err {
        JammiError::Config(msg) => {
            assert!(
                msg.contains("JAMMI_TEST_INTERP_DEFINITELY_NOT_SET"),
                "msg = {msg}"
            );
        }
        other => panic!("expected JammiError::Config, got {other:?}"),
    }
}

#[test]
fn interpolate_env_vars_escape_double_dollar() {
    let out = interpolate_env_vars("password = \"$$secret$$\"", |_| None).unwrap();
    assert_eq!(out, "password = \"$secret$\"");
}

#[test]
fn interpolate_env_vars_unterminated_brace_errors() {
    let err = interpolate_env_vars("url = \"${UNCLOSED\"", |_| None).unwrap_err();
    assert!(matches!(err, JammiError::Config(_)), "{err:?}");
}

#[test]
fn interpolate_env_vars_bare_dollar_preserved() {
    let out = interpolate_env_vars("hint = \"price is $5\"", |_| None).unwrap();
    assert_eq!(out, "hint = \"price is $5\"");
}

#[test]
fn interpolate_env_vars_invalid_name_errors() {
    let err = interpolate_env_vars("url = \"${1bad}\"", |_| None).unwrap_err();
    match err {
        JammiError::Config(msg) => assert!(msg.contains("Invalid env-var name"), "{msg}"),
        other => panic!("expected JammiError::Config, got {other:?}"),
    }
}

#[test]
fn load_interpolates_before_parse() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("jammi.toml");
    std::fs::write(
        &path,
        r#"
            [catalog.postgres]
            url = "${JAMMI_TEST_LOAD_URL}"
            pool_size = 4
        "#,
    )
    .unwrap();
    let cfg = JammiConfig::load_from(
        Some(&path),
        vec![(
            "JAMMI_TEST_LOAD_URL".to_string(),
            "postgres://u:p@h/db".to_string(),
        )],
    )
    .unwrap();
    assert_eq!(
        cfg.catalog,
        CatalogConfig::Postgres {
            url: "postgres://u:p@h/db".into(),
            pool_size: 4,
            max_lifetime_secs: None,
        }
    );
}

#[test]
fn storage_precision_default_is_f32() {
    assert_eq!(StoragePrecision::default(), StoragePrecision::F32);
}

#[test]
fn storage_precision_display_and_from_str_round_trip() {
    for precision in [
        StoragePrecision::F32,
        StoragePrecision::F16,
        StoragePrecision::Int8,
        StoragePrecision::Binary,
    ] {
        let s = precision.to_string();
        assert_eq!(s.parse::<StoragePrecision>().unwrap(), precision);
    }
}

#[test]
fn storage_precision_from_str_rejects_unknown_value() {
    assert!("fp8".parse::<StoragePrecision>().is_err());
}

#[test]
fn storage_precision_maps_onto_usearch_scalar_kind() {
    assert_eq!(
        StoragePrecision::F32.to_scalar_kind(),
        usearch::ScalarKind::F32
    );
    assert_eq!(
        StoragePrecision::F16.to_scalar_kind(),
        usearch::ScalarKind::F16
    );
    assert_eq!(
        StoragePrecision::Int8.to_scalar_kind(),
        usearch::ScalarKind::I8
    );
    assert_eq!(
        StoragePrecision::Binary.to_scalar_kind(),
        usearch::ScalarKind::B1
    );
}

#[test]
fn only_f32_skips_rescore() {
    assert!(!StoragePrecision::F32.needs_rescore());
    assert!(StoragePrecision::F16.needs_rescore());
    assert!(StoragePrecision::Int8.needs_rescore());
    assert!(StoragePrecision::Binary.needs_rescore());
}

#[test]
fn ann_index_config_default_oversample_is_unset() {
    assert_eq!(AnnIndexConfig::default().oversample, None);
    assert_eq!(AnnIndexConfig::default().effective_oversample(), 4);
}

#[test]
fn effective_oversample_clamps_a_misconfigured_zero_to_one() {
    let ann = AnnIndexConfig {
        oversample: Some(0),
        ..AnnIndexConfig::default()
    };
    assert_eq!(ann.effective_oversample(), 1);
}

#[test]
fn resolve_oversample_prefers_request_then_table_then_config() {
    let ann = AnnIndexConfig {
        oversample: Some(4),
        ..AnnIndexConfig::default()
    };
    // A per-request override wins over both.
    assert_eq!(ann.resolve_oversample(Some(9), Some(6)), 9);
    // No request → the table's own stamped default drives it, not the
    // deployment config default.
    assert_eq!(ann.resolve_oversample(None, Some(6)), 6);
    // Neither → the deployment's effective oversample (pre-migration-023
    // fallback).
    assert_eq!(ann.resolve_oversample(None, None), 4);
}

#[test]
fn resolve_oversample_clamps_a_zero_override_or_default_to_one() {
    let ann = AnnIndexConfig {
        oversample: Some(4),
        ..AnnIndexConfig::default()
    };
    // A 0 override must never shrink the candidate set below k.
    assert_eq!(ann.resolve_oversample(Some(0), Some(6)), 1);
    let misconfigured = AnnIndexConfig {
        oversample: Some(0),
        ..AnnIndexConfig::default()
    };
    assert_eq!(misconfigured.resolve_oversample(None, None), 1);
}

#[test]
fn storage_precision_default_oversample_is_precision_specific() {
    assert_eq!(StoragePrecision::F32.default_oversample(), 4);
    assert_eq!(StoragePrecision::F16.default_oversample(), 4);
    assert_eq!(StoragePrecision::Int8.default_oversample(), 4);
    assert_eq!(StoragePrecision::Binary.default_oversample(), 32);
}

#[test]
fn effective_oversample_for_uses_precision_default_when_unset() {
    // An untouched (`None`) deployment config defers to the precision's
    // own default: Binary widens to 32, the other three stay at 4.
    let ann = AnnIndexConfig::default();
    assert_eq!(ann.effective_oversample_for(StoragePrecision::Binary), 32);
    assert_eq!(ann.effective_oversample_for(StoragePrecision::F32), 4);
    assert_eq!(ann.effective_oversample_for(StoragePrecision::Int8), 4);
}

#[test]
fn effective_oversample_for_honors_an_explicit_deployment_override() {
    // An explicit `Some` deployment override wins over the
    // precision-specific default, even for Binary — an operator's
    // explicit config is never silently widened.
    let ann = AnnIndexConfig {
        oversample: Some(8),
        ..AnnIndexConfig::default()
    };
    assert_eq!(ann.effective_oversample_for(StoragePrecision::Binary), 8);
    assert_eq!(ann.effective_oversample_for(StoragePrecision::F32), 8);
}

#[test]
fn effective_oversample_for_honors_an_explicit_four_on_binary_not_widened_to_thirty_two() {
    // The exact case the adversarial audit flagged: a deployment that has
    // EXPLICITLY configured `oversample = 4` on a `Binary` table must be
    // honored verbatim as 4, never silently widened to Binary's own
    // per-precision default of 32.
    let ann = AnnIndexConfig {
        oversample: Some(4),
        ..AnnIndexConfig::default()
    };
    assert_eq!(ann.effective_oversample_for(StoragePrecision::Binary), 4);
}

#[test]
fn effective_oversample_for_none_on_binary_resolves_to_thirty_two() {
    // The unset (`None`) counterpart: with no explicit deployment
    // override, a Binary table still stamps the wider per-precision
    // default of 32.
    let ann = AnnIndexConfig {
        oversample: None,
        ..AnnIndexConfig::default()
    };
    assert_eq!(ann.effective_oversample_for(StoragePrecision::Binary), 32);
}

// ── esc-095: layered `JAMMI_*` env overrides ─────────────────────────────

/// esc-095's oracle: an env-only selection of Postgres + JetStream must
/// actually run Postgres + JetStream — before this fix, `apply_env_overrides`
/// hand-listed 18 variables and silently ignored `JAMMI_CATALOG__*` /
/// `JAMMI_BROKER__*` entirely, so this exact scenario ran SQLite.
#[test]
fn env_only_postgres_and_jetstream_selection_round_trips() {
    let cfg = JammiConfig::parse_from(
        "",
        vec![
            (
                "JAMMI_CATALOG__POSTGRES__URL".to_string(),
                "postgres://u:p@h/db".to_string(),
            ),
            (
                "JAMMI_BROKER__JET_STREAM__URL".to_string(),
                "nats://nats.svc:4222".to_string(),
            ),
        ],
    )
    .unwrap();
    assert_eq!(
        cfg.catalog,
        CatalogConfig::Postgres {
            url: "postgres://u:p@h/db".into(),
            pool_size: 8,
            max_lifetime_secs: None,
        }
    );
    assert_eq!(
        cfg.broker,
        BrokerConfig::JetStream {
            url: "nats://nats.svc:4222".into(),
            retention_seconds: 7 * 24 * 60 * 60,
            credentials: None,
        }
    );
}

/// The control leg: empty env, empty file → the untouched defaults
/// (discriminant equality on both arms), so the oracle above is proven
/// against a real baseline rather than vacuously.
#[test]
fn env_only_oracle_control_empty_env_stays_default() {
    let cfg = JammiConfig::parse_from("", std::iter::empty()).unwrap();
    assert_eq!(cfg.catalog, CatalogConfig::Sqlite { path: None });
    assert_eq!(cfg.broker, BrokerConfig::InMemory);
}

#[test]
fn env_unknown_top_level_section_refuses_naming_the_variable() {
    // `JAMMI_CATLOG__KIND` — one segment short of `JAMMI_CATALOG__...` — is
    // exactly esc-095's typo shape: it must refuse, not silently no-op.
    let err = JammiConfig::parse_from(
        "",
        vec![("JAMMI_CATLOG__KIND".to_string(), "postgres".to_string())],
    )
    .unwrap_err();
    match err {
        JammiError::Config(msg) => assert!(msg.contains("JAMMI_CATLOG__KIND"), "msg = {msg}"),
        other => panic!("expected JammiError::Config, got {other:?}"),
    }
}

#[test]
fn env_unknown_key_under_a_known_section_refuses_naming_it() {
    let err = JammiConfig::parse_from(
        "",
        vec![
            (
                "JAMMI_CATALOG__POSTGRES__URL".to_string(),
                "postgres://u:p@h/db".to_string(),
            ),
            (
                "JAMMI_CATALOG__POSTGRES__BOGUS".to_string(),
                "1".to_string(),
            ),
        ],
    )
    .unwrap_err();
    match err {
        JammiError::Config(msg) => {
            assert!(msg.contains("bogus"), "msg = {msg}");
            assert!(
                msg.contains("JAMMI_CATALOG__POSTGRES__BOGUS"),
                "msg = {msg}"
            );
        }
        other => panic!("expected JammiError::Config, got {other:?}"),
    }
}

#[test]
fn env_value_outside_domain_refuses_naming_key_and_var() {
    let err = JammiConfig::parse_from(
        "",
        vec![("JAMMI_GPU__MEMORY_FRACTION".to_string(), "90%".to_string())],
    )
    .unwrap_err();
    match err {
        JammiError::Config(msg) => {
            assert!(msg.contains("JAMMI_GPU__MEMORY_FRACTION"), "msg = {msg}");
            assert!(
                msg.contains("gpu.memory_fraction") || msg.contains("memory_fraction"),
                "msg = {msg}"
            );
        }
        other => panic!("expected JammiError::Config, got {other:?}"),
    }
}

#[test]
fn env_kernels_disable_and_other_runtime_knobs_are_ignored() {
    // A single-underscore name that is not `JAMMI_<top-level-field>` is a
    // runtime knob outside this layer's namespace — never an error, never
    // reflected into the config.
    let cfg = JammiConfig::parse_from(
        "",
        vec![
            ("JAMMI_KERNELS_DISABLE".to_string(), "1".to_string()),
            ("JAMMI_AUDIT_MASTER_KEY".to_string(), "deadbeef".to_string()),
            (
                "JAMMI_TEST_PG_URL".to_string(),
                "postgres://ignored".to_string(),
            ),
            ("JAMMI_CONFIG".to_string(), "/nonexistent.toml".to_string()),
        ],
    )
    .unwrap();
    let baseline = JammiConfig::parse_from("", std::iter::empty()).unwrap();
    assert_eq!(cfg.catalog, baseline.catalog);
    assert_eq!(cfg.broker, baseline.broker);
    assert_eq!(cfg.artifact_dir, baseline.artifact_dir);
    assert_eq!(cfg.training, baseline.training);
}

#[test]
fn bare_env_and_file_catalog_selection_shapes() {
    let via_env = JammiConfig::parse_from(
        "",
        vec![("JAMMI_CATALOG".to_string(), "sqlite".to_string())],
    )
    .unwrap();
    assert_eq!(via_env.catalog, CatalogConfig::Sqlite { path: None });

    let via_file = JammiConfig::parse_from("catalog = \"sqlite\"\n", std::iter::empty()).unwrap();
    assert_eq!(via_file.catalog, CatalogConfig::Sqlite { path: None });
}

#[test]
fn bare_env_storage_cloud_selects_default_variant() {
    let cfg = JammiConfig::parse_from(
        "",
        vec![("JAMMI_STORAGE__CLOUD".to_string(), "s3".to_string())],
    )
    .unwrap();
    match cfg.storage.cloud {
        Some(CloudConfig::S3(s3)) => {
            assert!(s3.region.is_none());
            assert!(s3.endpoint.is_none());
            assert!(s3.access_key_id.is_none());
            assert!(s3.secret_access_key.is_none());
            assert!(s3.session_token.is_none());
            assert!(!s3.allow_http);
        }
        other => panic!("expected S3(default), got {other:?}"),
    }
}

#[test]
fn file_postgres_plus_env_pool_size_merges() {
    let cfg = JammiConfig::parse_from(
        "[catalog.postgres]\nurl = \"postgres://u:p@h/db\"\n",
        vec![(
            "JAMMI_CATALOG__POSTGRES__POOL_SIZE".to_string(),
            "32".to_string(),
        )],
    )
    .unwrap();
    assert_eq!(
        cfg.catalog,
        CatalogConfig::Postgres {
            url: "postgres://u:p@h/db".into(),
            pool_size: 32,
            max_lifetime_secs: None,
        }
    );
}

#[test]
fn file_sqlite_plus_env_postgres_env_wins() {
    let cfg = JammiConfig::parse_from(
        "[catalog.sqlite]\n",
        vec![(
            "JAMMI_CATALOG__POSTGRES__URL".to_string(),
            "postgres://u:p@h/db".to_string(),
        )],
    )
    .unwrap();
    assert_eq!(
        cfg.catalog,
        CatalogConfig::Postgres {
            url: "postgres://u:p@h/db".into(),
            pool_size: 8,
            max_lifetime_secs: None,
        }
    );
}

#[test]
fn file_s3_plus_env_r2_env_wins() {
    let cfg = JammiConfig::parse_from(
        "[storage.cloud.s3]\nregion = \"us-east-1\"\n",
        vec![(
            "JAMMI_STORAGE__CLOUD__R2__ACCOUNT_ID".to_string(),
            "acct".to_string(),
        )],
    )
    .unwrap();
    match cfg.storage.cloud {
        Some(CloudConfig::R2(r2)) => assert_eq!(r2.account_id.as_deref(), Some("acct")),
        other => panic!("expected R2, got {other:?}"),
    }
}

#[test]
fn two_file_variants_errors_naming_both() {
    let err = JammiConfig::parse_from(
        "[catalog.sqlite]\n[catalog.postgres]\nurl = \"x\"\n",
        std::iter::empty(),
    )
    .unwrap_err();
    match err {
        JammiError::Config(msg) => {
            assert!(msg.contains("sqlite"), "msg = {msg}");
            assert!(msg.contains("postgres"), "msg = {msg}");
        }
        other => panic!("expected JammiError::Config, got {other:?}"),
    }
}

/// End-to-end smoke check only: both source orderings still refuse through
/// the public `parse_from` entry point. This is NOT an order-independence
/// oracle — `parse_from` collects its `env` iterator into a `BTreeMap`
/// before the env layer is ever built (mod.rs's `load_from`/`parse_from`),
/// so both `Vec` literals below collapse onto the identical sorted input by
/// the time anything order-sensitive runs. See
/// `env_layer_leaf_table_collision_is_order_independent_and_names_each_role`
/// below for the actual order oracle, driven directly against
/// `env_map::build_env_layer` (which iterates whatever order it is given).
#[test]
fn env_bare_selection_plus_nested_key_collision_both_orders() {
    for env in [
        vec![
            ("JAMMI_CATALOG".to_string(), "sqlite".to_string()),
            (
                "JAMMI_CATALOG__POSTGRES__URL".to_string(),
                "postgres://u:p@h/db".to_string(),
            ),
        ],
        vec![
            (
                "JAMMI_CATALOG__POSTGRES__URL".to_string(),
                "postgres://u:p@h/db".to_string(),
            ),
            ("JAMMI_CATALOG".to_string(), "sqlite".to_string()),
        ],
    ] {
        let err = JammiConfig::parse_from("", env.clone()).unwrap_err();
        match err {
            JammiError::Config(msg) => {
                assert!(msg.contains("JAMMI_CATALOG"), "env = {env:?}, msg = {msg}");
            }
            other => panic!("env = {env:?}: expected JammiError::Config, got {other:?}"),
        }
    }
}

/// T6's real order oracle: driven directly against `env_map::build_env_layer`
/// (which — unlike `parse_from` — iterates its input in exactly the order
/// given, never through a `BTreeMap` first), so the two orderings are
/// genuinely different inputs. The collision is refused either way, but
/// which variable plays "already placed" versus "the one that collides"
/// flips with insertion order, and each arm names its own role's variable(s)
/// explicitly — pinned here so that role assignment cannot silently drift.
#[test]
fn env_layer_leaf_table_collision_is_order_independent_and_names_each_role() {
    let leaf = ("JAMMI_CATALOG".to_string(), "sqlite".to_string());
    let nested = (
        "JAMMI_CATALOG__POSTGRES__URL".to_string(),
        "postgres://u:p@h/db".to_string(),
    );

    // Leaf first: the SECOND (nested) variable is refused as nesting under
    // a path the FIRST (leaf) variable already set as a value.
    let leaf_first_err =
        super::env_map::build_env_layer(vec![leaf.clone(), nested.clone()]).unwrap_err();
    assert_eq!(
        leaf_first_err.0,
        "JAMMI_CATALOG__POSTGRES__URL nests under a path JAMMI_CATALOG already sets as a value"
    );

    // Nested first: the SECOND (leaf) variable is refused as setting a
    // value at a path the FIRST (nested) variable already nests under.
    let nested_first_err = super::env_map::build_env_layer(vec![nested, leaf]).unwrap_err();
    assert_eq!(
        nested_first_err.0,
        "JAMMI_CATALOG sets a value at a path that [\"JAMMI_CATALOG__POSTGRES__URL\"] also nests under"
    );
}

#[test]
fn file_typo_section_plus_env_postgres_names_the_typo() {
    let err = JammiConfig::parse_from(
        "[catalog.postgrez]\nurl = \"typo\"\n",
        vec![(
            "JAMMI_CATALOG__POSTGRES__URL".to_string(),
            "postgres://u:p@h/db".to_string(),
        )],
    )
    .unwrap_err();
    match err {
        JammiError::Config(msg) => assert!(msg.contains("postgrez"), "msg = {msg}"),
        other => panic!("expected JammiError::Config, got {other:?}"),
    }
}

#[test]
fn broker_in_memory_bogus_key_refuses_naming_it() {
    let err =
        JammiConfig::parse_from("[broker.in_memory]\nbogus = 1\n", std::iter::empty()).unwrap_err();
    match err {
        JammiError::Config(msg) => assert!(msg.contains("bogus"), "msg = {msg}"),
        other => panic!("expected JammiError::Config, got {other:?}"),
    }
}

/// H14's struct-position rule: an env leaf cannot stand in for a whole
/// struct payload.
#[test]
fn env_whole_value_override_of_a_struct_position_is_refused() {
    let err = JammiConfig::parse_from(
        "[catalog.postgres]\nurl = \"postgres://file\"\n",
        vec![("JAMMI_CATALOG__POSTGRES".to_string(), "x".to_string())],
    )
    .unwrap_err();
    match err {
        JammiError::Config(msg) => assert!(msg.contains("JAMMI_CATALOG__POSTGRES"), "msg = {msg}"),
        other => panic!("expected JammiError::Config, got {other:?}"),
    }
}

/// H14's struct-position rule, the BARE-env case: with NO file layer at
/// all, `Node::merge` never has a lower (file) node to disagree with, so
/// the env leaf reaches `deserialize_struct` directly as a bare
/// `Node::Env`, never wrapped in `Node::Override`. That case must refuse
/// identically to the with-file-layer case above — the auditor's exact
/// reproduction: `JAMMI_MODELS='{ offline = true }'` and
/// `JAMMI_SERVER='{ health_listen = "1.2.3.4:1" }'` against an otherwise
/// empty config, each of which (pre-fix) was silently ACCEPTED as a
/// whole-struct value and reset every sibling field of that struct to its
/// default.
#[test]
fn env_whole_value_struct_position_is_refused_even_with_no_file_layer() {
    let err = JammiConfig::parse_from(
        "",
        vec![("JAMMI_MODELS".to_string(), "{ offline = true }".to_string())],
    )
    .unwrap_err();
    match err {
        JammiError::Config(msg) => assert!(msg.contains("JAMMI_MODELS"), "msg = {msg}"),
        other => panic!("expected JammiError::Config, got {other:?}"),
    }

    let err = JammiConfig::parse_from(
        "",
        vec![(
            "JAMMI_SERVER".to_string(),
            "{ health_listen = \"1.2.3.4:1\" }".to_string(),
        )],
    )
    .unwrap_err();
    match err {
        JammiError::Config(msg) => assert!(msg.contains("JAMMI_SERVER"), "msg = {msg}"),
        other => panic!("expected JammiError::Config, got {other:?}"),
    }
}

/// H14's map-position rule: a whole-value env override REPLACES a file map,
/// it does not merge into it.
#[test]
fn env_whole_value_override_of_a_map_position_replaces_not_merges() {
    let cfg = JammiConfig::parse_from(
        "[inference.http.headers]\na = \"1\"\n",
        vec![(
            "JAMMI_INFERENCE__HTTP__HEADERS".to_string(),
            "{ b = \"2\" }".to_string(),
        )],
    )
    .unwrap();
    let headers = &cfg.inference.http.headers;
    assert_eq!(headers.len(), 1, "headers = {headers:?}");
    assert_eq!(headers.get("b").map(Secret::expose), Some("2"));
    assert!(!headers.contains_key("a"));
}

/// H1's fourth oracle: a nested per-key env override MERGES into a file
/// map at the same path (never replaces the whole map) — the seq/map
/// "whole value" rule above only fires when the env var names the map's OWN
/// path, not a deeper key under it.
#[test]
fn file_header_map_plus_nested_env_header_merges() {
    let cfg = JammiConfig::parse_from(
        "[inference.http.headers]\na = \"1\"\n",
        vec![(
            "JAMMI_INFERENCE__HTTP__HEADERS__B".to_string(),
            "2".to_string(),
        )],
    )
    .unwrap();
    let headers = &cfg.inference.http.headers;
    assert_eq!(headers.get("a").map(Secret::expose), Some("1"));
    assert_eq!(headers.get("b").map(Secret::expose), Some("2"));
}

#[test]
fn env_integer_field_parses_leading_zeros_string_field_keeps_them_verbatim() {
    let cfg = JammiConfig::parse_from(
        "",
        vec![
            ("JAMMI_ENGINE__BATCH_SIZE".to_string(), "007".to_string()),
            ("JAMMI_ENGINE__MEMORY_LIMIT".to_string(), "007".to_string()),
        ],
    )
    .unwrap();
    assert_eq!(cfg.engine.batch_size, 7);
    assert_eq!(cfg.engine.memory_limit, "007");
}

#[test]
fn services_grammar_all_forms() {
    #[derive(Debug, serde::Deserialize)]
    struct Holder {
        services: ServiceSelection,
    }
    let all: Holder = toml::from_str("services = \"all\"").unwrap();
    assert_eq!(all.services, ServiceSelection::All(AllSentinel::All));

    // H7: case-sensitive — "ALL" is a one-token tier list, not the sentinel.
    let shout: Holder = toml::from_str("services = \"ALL\"").unwrap();
    assert_eq!(shout.services, ServiceSelection::Only(vec!["ALL".into()]));

    let list: Holder = toml::from_str("services = \"train,event\"").unwrap();
    assert_eq!(
        list.services,
        ServiceSelection::Only(vec!["train".into(), "event".into()])
    );

    let empty: Holder = toml::from_str("services = []").unwrap();
    assert_eq!(empty.services, ServiceSelection::Only(vec![]));

    let array: Holder = toml::from_str(r#"services = ["event", "eval"]"#).unwrap();
    assert_eq!(
        array.services,
        ServiceSelection::Only(vec!["event".into(), "eval".into()])
    );
}

/// H7 stays case-sensitive; a trailing newline (what a secrets file or a
/// heredoc-sourced env var actually carries) is trimmed before both the
/// case-sensitive `"all"` check and the comma split — it is not part of the
/// value.
#[test]
fn services_all_trims_whitespace_but_stays_case_sensitive() {
    #[derive(Debug, serde::Deserialize)]
    struct Holder {
        services: ServiceSelection,
    }
    let trimmed: Holder = toml::from_str("services = \"all\\n\"").unwrap();
    assert_eq!(trimmed.services, ServiceSelection::All(AllSentinel::All));

    let padded = JammiConfig::parse_from(
        "",
        vec![("JAMMI_SERVER__SERVICES".to_string(), "  all  ".to_string())],
    )
    .unwrap();
    assert_eq!(
        padded.server.services,
        ServiceSelection::All(AllSentinel::All)
    );

    // Trimming never loosens H7's case sensitivity: padded "ALL" is still a
    // one-token tier list, not the sentinel.
    let shout: Holder = toml::from_str("services = \" ALL \"").unwrap();
    assert_eq!(shout.services, ServiceSelection::Only(vec!["ALL".into()]));
}

#[test]
fn env_services_all_and_comma_list() {
    let all = JammiConfig::parse_from(
        "",
        vec![("JAMMI_SERVER__SERVICES".to_string(), "all".to_string())],
    )
    .unwrap();
    assert_eq!(all.server.services, ServiceSelection::All(AllSentinel::All));

    let list = JammiConfig::parse_from(
        "",
        vec![(
            "JAMMI_SERVER__SERVICES".to_string(),
            "train,event".to_string(),
        )],
    )
    .unwrap();
    assert_eq!(
        list.server.services,
        ServiceSelection::Only(vec!["train".into(), "event".into()])
    );
}

/// X1 oracle: a persisted `sources.options` row in the pre-existing,
/// internally tagged shape — including an unknown, forward-compat key — must
/// still reload through the REAL, unmodified `CloudConfig`.
#[test]
fn persisted_cloud_config_old_shape_with_unknown_key_still_reloads() {
    let json = r#"{"kind":"s3","region":"us-east-1","future_field":"x"}"#;
    let cfg: CloudConfig = serde_json::from_str(json).unwrap();
    match cfg {
        CloudConfig::S3(s3) => assert_eq!(s3.region.as_deref(), Some("us-east-1")),
        other => panic!("expected S3, got {other:?}"),
    }
}

/// The gcs config-side `service_account` field (inline or `{ file }`)
/// collapses onto the persisted `service_account_json` — never
/// `service_account_path` — because by conversion time it is always
/// resolved plain text.
#[test]
fn gcs_service_account_maps_onto_service_account_json() {
    // A TOML literal string (single-quoted) holds the JSON credential
    // verbatim, with no escaping gymnastics.
    let toml_src = "[storage.cloud.gcs]\nservice_account = '{ \"type\": \"service_account\" }'\n";
    let cfg = JammiConfig::parse_from(toml_src, std::iter::empty()).unwrap();
    match cfg.storage.cloud {
        Some(CloudConfig::Gcs(gcs)) => {
            assert_eq!(
                gcs.service_account_json.as_ref().map(Secret::expose),
                Some("{ \"type\": \"service_account\" }")
            );
            assert!(gcs.service_account_path.is_none());
        }
        other => panic!("expected Gcs, got {other:?}"),
    }
}

/// `${VAR}` interpolation is sourced from the SAME env map `parse_from` was
/// handed — never `std::env` — so it is testable without touching the
/// process environment at all.
#[test]
fn parse_from_interpolates_from_the_passed_env_map() {
    let cfg = JammiConfig::parse_from(
        "[catalog.postgres]\nurl = \"${PG_URL}\"\n",
        vec![("PG_URL".to_string(), "postgres://u:p@h/db".to_string())],
    )
    .unwrap();
    assert_eq!(
        cfg.catalog,
        CatalogConfig::Postgres {
            url: "postgres://u:p@h/db".into(),
            pool_size: 8,
            max_lifetime_secs: None,
        }
    );
}

/// H15: the resolution order is injectable — every step probed against a
/// tempdir, never the real `/etc`.
#[test]
fn resolve_config_path_in_honors_the_documented_order() {
    let cwd_dir = tempfile::tempdir().unwrap();
    let etc_dir = tempfile::tempdir().unwrap();
    let platform_dir = tempfile::tempdir().unwrap();
    let explicit_dir = tempfile::tempdir().unwrap();

    let roots = ConfigRoots {
        cwd: cwd_dir.path().to_path_buf(),
        etc_dir: etc_dir.path().to_path_buf(),
        platform_dir: Some(platform_dir.path().to_path_buf()),
    };

    // Nothing anywhere → None.
    assert_eq!(resolve_config_path_in(None, &roots, &BTreeMap::new()), None);

    // Only the platform dir has one → platform dir wins.
    let platform_file = platform_dir.path().join("config.toml");
    std::fs::write(&platform_file, "").unwrap();
    assert_eq!(
        resolve_config_path_in(None, &roots, &BTreeMap::new()),
        Some(platform_file.clone())
    );

    // etc_dir also has one → etc_dir beats the platform dir.
    let etc_file = etc_dir.path().join("jammi.toml");
    std::fs::write(&etc_file, "").unwrap();
    assert_eq!(
        resolve_config_path_in(None, &roots, &BTreeMap::new()),
        Some(etc_file.clone())
    );

    // cwd also has one → cwd beats etc_dir.
    let cwd_file = cwd_dir.path().join("jammi.toml");
    std::fs::write(&cwd_file, "").unwrap();
    assert_eq!(
        resolve_config_path_in(None, &roots, &BTreeMap::new()),
        Some(cwd_file.clone())
    );

    // JAMMI_CONFIG also points somewhere → JAMMI_CONFIG beats cwd.
    let env_file = explicit_dir.path().join("env.toml");
    std::fs::write(&env_file, "").unwrap();
    let mut env = BTreeMap::new();
    env.insert(
        "JAMMI_CONFIG".to_string(),
        env_file.to_str().unwrap().to_string(),
    );
    assert_eq!(
        resolve_config_path_in(None, &roots, &env),
        Some(env_file.clone())
    );

    // An explicit path that exists beats everything, including JAMMI_CONFIG.
    let explicit_file = explicit_dir.path().join("explicit.toml");
    std::fs::write(&explicit_file, "").unwrap();
    assert_eq!(
        resolve_config_path_in(Some(&explicit_file), &roots, &env),
        Some(explicit_file.clone())
    );

    // An explicit path that does NOT exist falls through to the next step
    // (JAMMI_CONFIG), rather than erroring or silently defaulting.
    let missing = explicit_dir.path().join("missing.toml");
    assert_eq!(
        resolve_config_path_in(Some(&missing), &roots, &env),
        Some(env_file)
    );
}

/// `parse_from("", [])` is the documented "defaults" control: no file, no
/// env, exactly `JammiConfig::default()`.
#[test]
fn parse_from_empty_is_the_defaults_control() {
    let cfg = JammiConfig::parse_from("", std::iter::empty()).unwrap();
    assert_eq!(cfg.artifact_dir, JammiConfig::default().artifact_dir);
    assert_eq!(cfg.catalog, CatalogConfig::Sqlite { path: None });
    assert_eq!(cfg.broker, BrokerConfig::InMemory);
    assert_eq!(cfg.signing_key, SigningKeyConfig::Env);
    assert!(cfg.storage.cloud.is_none());
    assert_eq!(cfg.models, ModelsConfig::default());
}

// ── unknown credentials_path key oracle ──────────────────────────────────

/// The retired `credentials_path` key (#483) is no longer silently ignored:
/// `BrokerConfig::JetStream` has `deny_unknown_fields`, so a config that
/// still spells the retired key is refused, naming it.
#[test]
fn broker_jetstream_credentials_path_is_a_refused_unknown_key() {
    let err = JammiConfig::parse_from(
        "[broker.jet_stream]\nurl = \"nats://nats.svc:4222\"\ncredentials_path = \"/var/run/secrets/nats.creds\"\n",
        std::iter::empty(),
    )
    .unwrap_err();
    match err {
        JammiError::Config(msg) => assert!(msg.contains("credentials_path"), "msg = {msg}"),
        other => panic!("expected JammiError::Config, got {other:?}"),
    }
}

// ── ModelsConfig (H5) ─────────────────────────────────────────────────────

#[test]
fn models_config_round_trips_and_defaults() {
    let cfg = JammiConfig::parse_from(
        r#"
        [models]
        hub_endpoint = "https://huggingface.co"
        hub_cache_dir = "/var/cache/jammi"
        hub_token = "hf_inline"
        offline = true
    "#,
        std::iter::empty(),
    )
    .unwrap();
    assert_eq!(
        cfg.models.hub_endpoint.as_deref(),
        Some("https://huggingface.co")
    );
    assert_eq!(
        cfg.models.hub_cache_dir,
        Some(PathBuf::from("/var/cache/jammi"))
    );
    assert!(matches!(
        cfg.models.hub_token,
        Some(SecretSource::Inline(ref s)) if s == "hf_inline"
    ));
    assert_eq!(cfg.models.offline, Some(true));

    let default_cfg = JammiConfig::parse_from("", std::iter::empty()).unwrap();
    assert_eq!(default_cfg.models, ModelsConfig::default());
    assert_eq!(
        default_cfg.models.offline, None,
        "an omitted `offline` key must round-trip as None, not a bare `false` -- \
         jammi-ai's HubSource::from_config's HF_HUB_OFFLINE fallback only applies when \
         this is None"
    );
    assert!(default_cfg.models.hub_token.is_none());
}

/// A literal `offline = false` must round-trip as `Some(false)`, distinct
/// from the omitted-field `None` above -- the `Option<bool>` is what lets
/// `HubSource::from_config`'s `HF_HUB_OFFLINE` fallback tell "the operator
/// explicitly forced online" apart from "the operator never said".
#[test]
fn models_config_explicit_offline_false_is_some_not_none() {
    let cfg = JammiConfig::parse_from("[models]\noffline = false\n", std::iter::empty()).unwrap();
    assert_eq!(cfg.models.offline, Some(false));
}

#[test]
fn models_config_unknown_key_is_refused() {
    let err = JammiConfig::parse_from("[models]\nbogus = 1\n", std::iter::empty()).unwrap_err();
    match err {
        JammiError::Config(msg) => assert!(msg.contains("bogus"), "msg = {msg}"),
        other => panic!("expected JammiError::Config, got {other:?}"),
    }
}

#[test]
fn unknown_server_key_such_as_tls_is_a_typed_load_error() {
    // `ServerConfig` carries no TLS knob at all — the engine has no
    // certificate/key/termination config of its own (security.md's
    // "no silent plaintext fallback" claim). `#[serde(deny_unknown_fields)]`
    // on `ServerConfig` is the oracle backing that claim: a `[server] tls`
    // stanza must fail closed at load time, naming the offending path,
    // never silently drop the unrecognised section and boot plaintext.
    let toml_src = r#"
        [server]
        tls = { cert_path = "x" }
    "#;
    let err = JammiConfig::parse_from(toml_src, std::iter::empty()).unwrap_err();
    match err {
        JammiError::Config(msg) => {
            assert!(
                msg.contains("server.tls"),
                "expected the error to name `server.tls`, got: {msg}"
            );
        }
        other => panic!("expected JammiError::Config, got {other:?}"),
    }
}

// ── Top-level field namespace (T5) ────────────────────────────────────────

#[test]
fn top_level_fields_matches_jammi_config() {
    // Set-equal to `JammiConfig`'s own field list — hand-maintained
    // alongside `super::env_map::TOP_LEVEL_FIELDS`; a field added to one
    // without the other is exactly the drift this pins.
    let expected: &[&str] = &[
        "artifact_dir",
        "engine",
        "gpu",
        "inference",
        "embedding",
        "fine_tuning",
        "lease",
        "training",
        "cache",
        "server",
        "logging",
        "catalog",
        "broker",
        "signing_key",
        "storage",
        "models",
    ];
    assert_eq!(super::env_map::TOP_LEVEL_FIELDS, expected);

    // Behavioural half: every one of them is reachable as a bare
    // `JAMMI_<FIELD>` AND as `JAMMI_<FIELD>__...` without an "unknown
    // top-level section" error.
    for field in expected {
        let var = format!("JAMMI_{}", field.to_uppercase());
        let res = env_map::build_env_layer(vec![(var.clone(), "x".to_string())]);
        assert!(res.is_ok(), "{var} must be namespace-accepted: {res:?}");
    }
}

#[test]
fn bare_env_artifact_dir_round_trips() {
    let cfg = JammiConfig::parse_from(
        "",
        vec![("JAMMI_ARTIFACT_DIR".to_string(), "/srv/jammi".to_string())],
    )
    .unwrap();
    assert_eq!(cfg.artifact_dir, PathBuf::from("/srv/jammi"));
}
