//! A source's secret is never written down (#756). The catalog keeps a
//! source's URL and its cloud credentials so that every later build can
//! connect, so a database URL naming its password, or credentials holding a
//! secret, would put it in the catalog, its WAL and every backup of it. Each
//! names its secret with a `${JAMMI_SOURCE_<NAME>}` placeholder instead,
//! read from the environment of the process that builds the source; one that
//! writes the secret itself, or names any other variable, is refused before
//! anything is persisted.

use jammi_db::catalog::backend::BackendKind;
use jammi_db::config::Secret;
use jammi_db::error::JammiError;
use jammi_db::source::{FileFormat, SourceConnection, SourceDefinition, SourceType};
use jammi_db::storage::{CloudConfig, S3Config};
use jammi_test_utils::{make_test_session, unique_suffix};
use tempfile::tempdir;
use test_case::test_case;

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn a_database_url_writing_its_password_is_refused_before_anything_is_persisted(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;
    // Nothing listens on port 9: a URL the engine tried to connect with
    // fails there, and one it refuses never gets that far.
    for url in [
        "postgresql://ada:hunter2@127.0.0.1:9/sales",
        "postgresql://ada@127.0.0.1:9/sales?password=hunter2",
        "mysql://ada:hunter2@127.0.0.1:9/sales",
    ] {
        let id = format!("db_{}", unique_suffix());
        let definition = SourceDefinition::from_url(url, None).unwrap();
        let err = session
            .add_source(&id, definition.source_type, definition.connection)
            .await
            .unwrap_err();
        let JammiError::InvalidArgument { argument, reason } = &err else {
            panic!("{url}: a password in the URL must be refused as an invalid url, got: {err}");
        };
        assert_eq!(argument, "url", "{reason}");
        assert!(
            reason.contains("${"),
            "the refusal shows the placeholder form: {reason}"
        );
        assert!(
            !reason.contains("hunter2"),
            "the refusal never repeats the password: {reason}"
        );
        let sources = session.catalog().list_sources().await.unwrap();
        assert!(!sources.iter().any(|s| s.source_id == id), "{url}");
    }
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn cloud_credentials_writing_a_secret_are_refused_and_so_is_any_catalog_write(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;
    let id = format!("bucket_{}", unique_suffix());
    let err = session
        .add_source(
            &id,
            SourceType::File,
            SourceConnection {
                url: Some("s3://a-bucket/events.parquet".into()),
                format: Some(FileFormat::Parquet),
                cloud: Some(CloudConfig::S3(S3Config {
                    access_key_id: Some("AKIA-TEST".into()),
                    secret_access_key: Some(Secret::new("wJalr-literal-secret")),
                    ..Default::default()
                })),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    let JammiError::InvalidArgument { argument, reason } = &err else {
        panic!("a written cloud secret must be refused as an invalid argument, got: {err}");
    };
    assert_eq!(argument, "cloud", "{reason}");
    assert!(reason.contains("secret_access_key"), "{reason}");
    assert!(!reason.contains("wJalr-literal-secret"), "{reason}");

    // The catalog refuses it too, whoever writes the row.
    let written =
        SourceDefinition::from_url("postgresql://ada:hunter2@127.0.0.1:9/sales", None).unwrap();
    let err = session
        .catalog()
        .register_source(&id, written.source_type, &written.connection)
        .await
        .unwrap_err();
    assert!(
        matches!(&err, JammiError::InvalidArgument { argument, .. } if argument == "url"),
        "{err}"
    );
    let sources = session.catalog().list_sources().await.unwrap();
    assert!(!sources.iter().any(|s| s.source_id == id));
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn a_placeholder_naming_a_variable_set_aside_for_no_source_is_refused(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;
    // The environment is the server's: were any variable nameable, a caller
    // could have the server send its own secrets, as a source's password, to
    // a host of the caller's choosing.
    let url = SourceDefinition::from_url(
        "postgresql://ada:${AWS_SECRET_ACCESS_KEY}@127.0.0.1:9/sales",
        None,
    )
    .unwrap();
    let cloud = SourceConnection {
        url: Some("s3://a-bucket/events.parquet".into()),
        format: Some(FileFormat::Parquet),
        cloud: Some(CloudConfig::S3(S3Config {
            access_key_id: Some("AKIA-TEST".into()),
            secret_access_key: Some(Secret::new("${JAMMI_AUDIT_MASTER_KEY}")),
            ..Default::default()
        })),
        ..Default::default()
    };
    for (argument, source_type, connection) in [
        ("url", url.source_type, url.connection),
        ("cloud", SourceType::File, cloud),
    ] {
        let id = format!("named_{}", unique_suffix());
        let err = session
            .add_source(&id, source_type, connection)
            .await
            .unwrap_err();
        let JammiError::InvalidArgument {
            argument: refused,
            reason,
        } = &err
        else {
            panic!("{argument}: must be refused as an invalid argument, got: {err}");
        };
        assert_eq!(refused, argument, "{reason}");
        assert!(reason.contains("JAMMI_SOURCE_"), "{reason}");
        let sources = session.catalog().list_sources().await.unwrap();
        assert!(!sources.iter().any(|s| s.source_id == id), "{argument}");
    }
}

/// Every byte of every file under `dir`.
#[cfg(any(feature = "live-postgres-tests", feature = "live-s3-tests"))]
fn files_under(dir: &std::path::Path) -> Vec<(std::path::PathBuf, Vec<u8>)> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            files.extend(files_under(&path));
        } else {
            let bytes = std::fs::read(&path).unwrap();
            files.push((path, bytes));
        }
    }
    files
}

#[cfg(feature = "live-postgres-tests")]
#[tokio::test]
async fn a_password_named_by_a_placeholder_connects_and_is_never_written_down() {
    let source_dir = tempdir().unwrap();
    let url = jammi_test_utils::postgres_database_holding(
        source_dir.path(),
        "CREATE TABLE ratings (id BIGINT PRIMARY KEY); INSERT INTO ratings VALUES (1), (2), (3);",
    )
    .await;
    let source_url = jammi_test_utils::source_url(&url);
    assert!(
        source_url.contains("${"),
        "the test database has a password: {url}"
    );
    let password = std::env::var(jammi_test_utils::SOURCE_PASSWORD_VAR).unwrap();

    let dir = tempdir().unwrap();
    let id = format!("pg_{}", unique_suffix());
    let count = format!("SELECT COUNT(*) AS n FROM {id}.public.ratings");
    let rows = |batches: Vec<arrow::array::RecordBatch>| {
        batches
            .iter()
            .map(|b| arrow::util::display::array_value_to_string(b.column(0), 0).unwrap())
            .collect::<Vec<_>>()
    };
    {
        let session = make_test_session(BackendKind::Sqlite, dir.path()).await;
        let definition = SourceDefinition::from_url(&source_url, None).unwrap();
        session
            .add_source(&id, definition.source_type, definition.connection)
            .await
            .unwrap();
        assert_eq!(rows(session.sql(&count).await.unwrap()), ["3"]);
    }
    // A session opened over the same catalog connects from the stored row.
    let session = make_test_session(BackendKind::Sqlite, dir.path()).await;
    assert_eq!(rows(session.sql(&count).await.unwrap()), ["3"]);
    drop(session);

    // The password as a URL holds it, `:<password>@`: the test server's
    // password is a word the catalog holds elsewhere. The placeholder is
    // there, so the source's row was written and searched.
    let holds = |bytes: &[u8], needle: &str| {
        bytes
            .windows(needle.len())
            .any(|window| window == needle.as_bytes())
    };
    let files = files_under(dir.path());
    assert!(
        files
            .iter()
            .any(|(_, bytes)| holds(bytes, jammi_test_utils::SOURCE_PASSWORD_VAR)),
        "the catalog holds the source's URL, with its placeholder"
    );
    for (path, bytes) in files {
        assert!(
            !holds(&bytes, &format!(":{password}@")),
            "{} holds the source's password",
            path.display()
        );
    }
}

#[cfg(feature = "live-s3-tests")]
#[tokio::test]
async fn a_cloud_secret_named_by_a_placeholder_reads_and_is_never_written_down() {
    use jammi_db::storage::{ObjectParquetWriter, StorageRegistry, StorageUrl};

    let env = jammi_test_resources::env;
    // `s3://bucket/prefix`, or the bare bucket the local test store exports.
    let bucket = env("JAMMI_TEST_S3_BUCKET");
    let base = match bucket.contains("://") {
        true => bucket,
        false => format!("s3://{bucket}"),
    };
    let key = format!(
        "{}/jammi-source-secret-{}.parquet",
        base.trim_end_matches('/'),
        uuid::Uuid::new_v4().simple()
    );
    let url = StorageUrl::parse(&key).unwrap();
    let store = |secret_access_key: &str| {
        CloudConfig::S3(S3Config {
            region: Some(env("AWS_REGION")),
            endpoint: Some(env("JAMMI_TEST_S3_ENDPOINT")),
            access_key_id: Some(env("AWS_ACCESS_KEY_ID")),
            secret_access_key: Some(Secret::new(secret_access_key)),
            allow_http: true,
            ..Default::default()
        })
    };
    // The test writes its object with the store's secret itself; the source
    // reading it names the secret.
    let secret = env("AWS_SECRET_ACCESS_KEY");
    let handle = StorageRegistry::new()
        .handle_for(&url, Some(&store(&secret)))
        .unwrap();
    let batch = arrow::array::RecordBatch::try_from_iter([(
        "id",
        std::sync::Arc::new(arrow::array::Int64Array::from(vec![1, 2, 3])) as _,
    )])
    .unwrap();
    let mut writer = ObjectParquetWriter::open(&handle, batch.schema())
        .await
        .unwrap();
    writer.write_batch(&batch).await.unwrap();
    writer.close().await.unwrap();

    // The source names its secret, in a variable of its own.
    std::env::set_var("JAMMI_SOURCE_TEST_SECRET", &secret);
    let cloud = store("${JAMMI_SOURCE_TEST_SECRET}");
    let dir = tempdir().unwrap();
    let session = make_test_session(BackendKind::Sqlite, dir.path()).await;
    let id = format!("bucket_{}", unique_suffix());
    session
        .add_source(
            &id,
            SourceType::File,
            SourceConnection {
                url: Some(key.clone()),
                format: Some(FileFormat::Parquet),
                cloud: Some(cloud),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let table = key.rsplit('/').next().unwrap().trim_end_matches(".parquet");
    let batches = session
        .sql(&format!(
            "SELECT COUNT(*) AS n FROM {id}.public.\"{table}\""
        ))
        .await
        .unwrap();
    assert_eq!(
        arrow::util::display::array_value_to_string(batches[0].column(0), 0).unwrap(),
        "3"
    );
    drop(session);

    for (path, bytes) in files_under(dir.path()) {
        assert!(
            !bytes
                .windows(secret.len())
                .any(|window| window == secret.as_bytes()),
            "{} holds the source's cloud secret",
            path.display()
        );
    }
    handle
        .vanish_for_test(&handle.data_path().unwrap())
        .await
        .unwrap();
}
