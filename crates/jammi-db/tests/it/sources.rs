use crate::common;

use jammi_db::catalog::backend::{BackendImpl, BackendKind};
use jammi_db::catalog::backend_postgres::PostgresBackend;
use jammi_db::catalog::backend_sqlite::SqliteBackend;
use jammi_db::{
    session::JammiSession,
    source::{FileFormat, SourceConnection, SourceType},
};
use jammi_test_utils::{make_test_session, unique_suffix};
use tempfile::tempdir;
use test_case::test_case;

/// Fetch a backend-parameterized session, skipping the test (with a warning,
/// never `#[ignore]`) when the Postgres arm has no `JAMMI_TEST_PG_URL`.
macro_rules! session_or_skip {
    ($backend:expr, $dir:expr) => {
        match make_test_session($backend, $dir.path()).await {
            Some(s) => s,
            None => {
                eprintln!("skipping {:?}: JAMMI_TEST_PG_URL unset", $backend);
                return;
            }
        }
    };
}

/// Build a raw catalog backend on `backend`, without wrapping it in a
/// session — used only by the one test that needs a caller-customized
/// [`jammi_db::config::JammiConfig`] (a non-default `engine.batch_size`)
/// alongside the backend choice.
async fn build_backend(backend: BackendKind, dir: &std::path::Path) -> Option<BackendImpl> {
    match backend {
        BackendKind::Sqlite => {
            let b = SqliteBackend::open(&dir.join("catalog.db")).await.unwrap();
            Some(BackendImpl::Sqlite(b))
        }
        BackendKind::Postgres => {
            let url = jammi_test_utils::pg_url_for_tests()?;
            let pg = PostgresBackend::open_with_options(&url, 8, None)
                .await
                .unwrap();
            Some(BackendImpl::Postgres(pg))
        }
    }
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn register_and_query_multiple_formats(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = session_or_skip!(backend, dir);

    // Backend-unique source ids: the Postgres lane shares one `sources` table
    // across the whole test run, and `add_source` hard-errors on a duplicate
    // `source_id` — a fixed literal would collide with a sibling test (or a
    // prior run) that registered the same name.
    let suffix = unique_suffix();
    let patents_id = format!("patents_{suffix}");
    let scores_id = format!("scores_{suffix}");

    // Parquet
    session
        .add_source(
            &patents_id,
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let results = session
        .sql(&format!(
            "SELECT id, title FROM {patents_id}.public.patents LIMIT 5"
        ))
        .await
        .unwrap();
    assert!(!results.is_empty());
    assert!(results[0].num_rows() <= 5);
    assert!(results[0].schema().field_with_name("id").is_ok());
    assert!(results[0].schema().field_with_name("title").is_ok());

    // CSV
    session
        .add_source(
            &scores_id,
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("scores.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let results = session
        .sql(&format!(
            "SELECT name, score FROM {scores_id}.public.scores WHERE score > 0.6"
        ))
        .await
        .unwrap();
    assert_eq!(results[0].num_rows(), 2);
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn query_with_filter_and_order(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = session_or_skip!(backend, dir);
    let patents_id = format!("patents_{}", unique_suffix());

    session
        .add_source(
            &patents_id,
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let results = session
        .sql(&format!(
            "SELECT title, year FROM {patents_id}.public.patents WHERE year >= 2022 ORDER BY year DESC"
        ))
        .await
        .unwrap();

    let batch = &results[0];
    assert!(batch.num_rows() >= 1);
    let years = batch
        .column_by_name("year")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap();
    for i in 1..years.len() {
        assert!(
            years.value(i - 1) >= years.value(i),
            "Should be DESC ordered"
        );
    }
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn source_persists_across_sessions(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let persist_id = format!("persist_{}", unique_suffix());

    {
        let session = session_or_skip!(backend, dir);
        session
            .add_source(
                &persist_id,
                SourceType::File,
                SourceConnection {
                    url: Some(common::fixture_url("patents.parquet")),
                    format: Some(FileFormat::Parquet),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
    }

    {
        let session = session_or_skip!(backend, dir);
        let sources = session.catalog().list_sources().await.unwrap();
        assert!(sources.iter().any(|s| s.source_id == persist_id));
    }
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn source_crud_list_and_remove(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = session_or_skip!(backend, dir);

    let suffix = unique_suffix();
    let src_a = format!("src_a_{suffix}");
    let src_b = format!("src_b_{suffix}");

    session
        .add_source(
            &src_a,
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    session
        .add_source(
            &src_b,
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("scores.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let sources = session.catalog().list_sources().await.unwrap();
    let ids: Vec<&str> = sources.iter().map(|s| s.source_id.as_str()).collect();
    assert!(ids.contains(&src_a.as_str()));
    assert!(ids.contains(&src_b.as_str()));

    session.remove_source(&src_a).await.unwrap();
    let sources = session.catalog().list_sources().await.unwrap();
    assert!(!sources.iter().any(|s| s.source_id == src_a));
    assert!(sources.iter().any(|s| s.source_id == src_b));

    // Queries against the removed source should fail.
    let err = session
        .sql(&format!("SELECT * FROM {src_a}.public.patents"))
        .await;
    assert!(err.is_err(), "Query against removed source should fail");

    // Queries against the other source should still work.
    let rows = session
        .sql(&format!("SELECT COUNT(*) FROM {src_b}.public.scores"))
        .await
        .unwrap();
    assert!(!rows.is_empty());
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn session_respects_config_batch_size(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let backend_impl = match build_backend(backend, dir.path()).await {
        Some(b) => b,
        None => {
            eprintln!("skipping {backend:?}: JAMMI_TEST_PG_URL unset");
            return;
        }
    };
    let mut config = common::test_config(dir.path());
    config.engine.batch_size = 2;

    let session = JammiSession::with_backend(config, backend_impl)
        .await
        .unwrap();
    let patents_id = format!("patents_{}", unique_suffix());
    session
        .add_source(
            &patents_id,
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let results = session
        .sql(&format!("SELECT * FROM {patents_id}.public.patents"))
        .await
        .unwrap();
    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 20);
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn register_and_query_jsonl_file_source(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = session_or_skip!(backend, dir);

    // A 2-row newline-delimited JSON fixture, written directly into the
    // test's tempdir (this format has no shared fixture under
    // `tests/fixtures/`, unlike parquet/csv above).
    let fixture_path = dir.path().join("events.jsonl");
    std::fs::write(
        &fixture_path,
        "{\"id\": 1, \"name\": \"alpha\"}\n{\"id\": 2, \"name\": \"beta\"}\n",
    )
    .unwrap();

    let events_id = format!("events_{}", unique_suffix());
    session
        .add_source(
            &events_id,
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", fixture_path.display())),
                format: Some(FileFormat::JsonLines),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let results = session
        .sql(&format!(
            "SELECT id, name FROM {events_id}.public.events ORDER BY id"
        ))
        .await
        .unwrap();
    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 2);
    assert!(results[0].schema().field_with_name("id").is_ok());
    assert!(results[0].schema().field_with_name("name").is_ok());
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn directory_source_with_jsonl_format_excludes_json_named_file(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = session_or_skip!(backend, dir);

    // A directory holding both a `.jsonl` file (2 rows) and a `.json`-named
    // file (3 rows, same schema) — the jsonl-format directory listing must
    // glob-match only the `.jsonl` file, not the `.json` sibling.
    let listing_dir = dir.path().join("events_dir");
    std::fs::create_dir_all(&listing_dir).unwrap();
    std::fs::write(listing_dir.join("a.jsonl"), "{\"id\": 1}\n{\"id\": 2}\n").unwrap();
    std::fs::write(
        listing_dir.join("b.json"),
        "{\"id\": 3}\n{\"id\": 4}\n{\"id\": 5}\n",
    )
    .unwrap();

    let events_id = format!("events_dir_{}", unique_suffix());
    session
        .add_source(
            &events_id,
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", listing_dir.display())),
                format: Some(FileFormat::JsonLines),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let results = session
        .sql(&format!("SELECT id FROM {events_id}.public.events_dir"))
        .await
        .unwrap();
    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 2,
        "jsonl-format directory listing must match only the .jsonl file, not the .json sibling"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn directory_source_with_only_ndjson_files_registered_as_jsonl_falls_back_and_serves_rows(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let session = session_or_skip!(backend, dir);

    // A directory holding ONLY `.ndjson`-named files: `.jsonl` (jsonl's
    // default extension) has zero matches, so the adaptive fallback tries
    // `.ndjson` next and finds this directory's 2 rows — `.ndjson` has no
    // wire/CLI surface of its own, so this fallback is the only reachable
    // path onto it. Registration SUCCEEDS and the rows are queryable, not an
    // error (the enshrined "zero matches" test this replaces predates the
    // adaptive fallback the lead specified).
    let listing_dir = dir.path().join("ndjson_only");
    std::fs::create_dir_all(&listing_dir).unwrap();
    std::fs::write(listing_dir.join("a.ndjson"), "{\"id\": 1}\n{\"id\": 2}\n").unwrap();

    let events_id = format!("ndjson_only_{}", unique_suffix());
    session
        .add_source(
            &events_id,
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", listing_dir.display())),
                format: Some(FileFormat::JsonLines),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let results = session
        .sql(&format!("SELECT id FROM {events_id}.public.ndjson_only"))
        .await
        .unwrap();
    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 2,
        "an .ndjson-only directory registered as jsonl must fall back to .ndjson and serve rows"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn directory_source_with_neither_jsonl_nor_ndjson_is_a_loud_error_naming_both_extensions(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let session = session_or_skip!(backend, dir);

    // Neither `.jsonl` nor its `.ndjson` fallback has any match — the typed
    // error must name BOTH extensions tried, not just the first.
    let listing_dir = dir.path().join("neither");
    std::fs::create_dir_all(&listing_dir).unwrap();
    std::fs::write(listing_dir.join("a.json"), "{\"id\": 1}\n{\"id\": 2}\n").unwrap();

    let events_id = format!("neither_{}", unique_suffix());
    let err = session
        .add_source(
            &events_id,
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", listing_dir.display())),
                format: Some(FileFormat::JsonLines),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains(".jsonl"),
        "error must name the first extension tried: {message}"
    );
    assert!(
        message.contains(".ndjson"),
        "error must name the fallback extension tried: {message}"
    );

    let sources = session.catalog().list_sources().await.unwrap();
    assert!(!sources.iter().any(|s| s.source_id == events_id));
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn directory_source_with_both_jsonl_and_ndjson_prefers_jsonl_and_ignores_ndjson(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let session = session_or_skip!(backend, dir);

    // A directory holding BOTH a `.jsonl` file (2 rows) and an `.ndjson` file
    // (3 rows): `.jsonl` has a non-empty match, so the adaptive fallback never
    // even lists `.ndjson` — deterministic, `.jsonl` always wins. The
    // `.ndjson` rows are silently excluded, the ordinary "wrong extension"
    // listing semantic every format already has (see
    // `directory_source_with_jsonl_format_excludes_json_named_file`), not a
    // new one this adaptive resolution introduces.
    let listing_dir = dir.path().join("mixed");
    std::fs::create_dir_all(&listing_dir).unwrap();
    std::fs::write(listing_dir.join("a.jsonl"), "{\"id\": 1}\n{\"id\": 2}\n").unwrap();
    std::fs::write(
        listing_dir.join("b.ndjson"),
        "{\"id\": 3}\n{\"id\": 4}\n{\"id\": 5}\n",
    )
    .unwrap();

    let events_id = format!("mixed_{}", unique_suffix());
    session
        .add_source(
            &events_id,
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", listing_dir.display())),
                format: Some(FileFormat::JsonLines),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let results = session
        .sql(&format!("SELECT id FROM {events_id}.public.mixed"))
        .await
        .unwrap();
    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 2,
        ".jsonl must win over .ndjson when a directory holds both — the .ndjson rows are excluded"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn single_file_source_with_zero_extension_match_is_a_loud_error(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = session_or_skip!(backend, dir);

    // A `.json`-named file registered as `jsonl` (default `.jsonl` extension,
    // with an explicit `file_extension` override so the adaptive `.ndjson`
    // fallback does not apply — this pins the plain single-extension guard,
    // not the jsonl-only adaptive path covered by the tests above).
    let fixture_path = dir.path().join("events.json");
    std::fs::write(&fixture_path, "{\"id\": 1}\n{\"id\": 2}\n").unwrap();

    let events_id = format!("single_mismatch_{}", unique_suffix());
    let err = session
        .add_source(
            &events_id,
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", fixture_path.display())),
                format: Some(FileFormat::JsonLines),
                file_extension: Some(".jsonl".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains(".jsonl"),
        "error must name the applied extension filter: {message}"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn single_zero_byte_jsonl_file_is_a_loud_error_not_a_silent_empty_table(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let session = session_or_skip!(backend, dir);

    // `touch events.jsonl` / an interrupted export: the file exists, its
    // extension matches, but it carries 0 bytes and therefore no schema.
    // DataFusion's own `infer_schema` already drops size-0 files before
    // merging a schema; the zero-match GUARD must apply the identical
    // predicate so it can't diverge from what schema inference actually
    // sees — otherwise a 0-byte file passes the guard but still yields an
    // empty, column-less table.
    let fixture_path = dir.path().join("events.jsonl");
    std::fs::write(&fixture_path, "").unwrap();

    let events_id = format!("zero_byte_single_{}", unique_suffix());
    let err = session
        .add_source(
            &events_id,
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", fixture_path.display())),
                format: Some(FileFormat::JsonLines),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains(".jsonl"),
        "error must name the applied extension filter: {err}"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn directory_of_only_zero_byte_jsonl_files_is_a_loud_error_not_a_silent_empty_table(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let session = session_or_skip!(backend, dir);

    // Same as the single-file case above, but for the directory listing arm:
    // a directory whose only `.jsonl`-extension file is 0 bytes must resolve
    // to zero USABLE matches, not "one file matched the extension" — DataFusion
    // still lists the 0-byte file before its own size filter drops it, so a
    // guard that only checks the extension (not size) would register a
    // column-less, row-less table here.
    let listing_dir = dir.path().join("zero_byte_dir");
    std::fs::create_dir_all(&listing_dir).unwrap();
    std::fs::write(listing_dir.join("empty.jsonl"), "").unwrap();

    let events_id = format!("zero_byte_dir_{}", unique_suffix());
    let err = session
        .add_source(
            &events_id,
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", listing_dir.display())),
                format: Some(FileFormat::JsonLines),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains(".jsonl"),
        "error must name the applied extension filter: {err}"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn reload_survives_a_source_whose_files_vanished_after_registration(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let source_id = format!("vanished_{}", unique_suffix());
    let fixture_path = dir.path().join("events.jsonl");

    {
        let session = session_or_skip!(backend, dir);
        std::fs::write(&fixture_path, "{\"id\": 1}\n{\"id\": 2}\n").unwrap();
        session
            .add_source(
                &source_id,
                SourceType::File,
                SourceConnection {
                    url: Some(format!("file://{}", fixture_path.display())),
                    format: Some(FileFormat::JsonLines),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        // Registration succeeded with the file present.
        let results = session
            .sql(&format!("SELECT id FROM {source_id}.public.events"))
            .await
            .unwrap();
        assert_eq!(results[0].num_rows(), 2);
    }

    // Delete the backing file BEFORE the next session's `reload_sources`
    // pass — a zero-match listing on reload.
    std::fs::remove_file(&fixture_path).unwrap();

    // `reload_sources` (called from session construction) must surface the
    // SAME typed zero-match error loudly (a `tracing::warn!` per its existing
    // per-source `if let Err(e) = ... { warn!; continue; }` guard) rather than
    // failing the whole session open — one vanished source's files must not
    // make every OTHER source, or the session itself, unreachable.
    let session = session_or_skip!(backend, dir);

    // The catalog row survives (reload only skips DataFusion registration on
    // failure, it never deletes the persisted source) …
    let sources = session.catalog().list_sources().await.unwrap();
    assert!(
        sources.iter().any(|s| s.source_id == source_id),
        "a reload failure must not silently drop the source's catalog row"
    );
    // … but the table is unreachable for SQL, since reload never registered
    // it in DataFusion — a loud, typed miss rather than a stale success.
    let err = session
        .sql(&format!("SELECT id FROM {source_id}.public.events"))
        .await;
    assert!(
        err.is_err(),
        "a source whose files vanished must not still be queryable after reload"
    );
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn session_tenant_defaults_to_none_and_with_tenant_sets_it(backend: BackendKind) {
    use jammi_db::TenantId;
    use std::str::FromStr;

    let dir = tempdir().unwrap();
    let session = session_or_skip!(backend, dir);
    assert!(
        session.tenant().is_none(),
        "fresh session has no tenant scope"
    );

    let t = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap();
    let session = session.with_tenant(t);
    assert_eq!(session.tenant(), Some(t));
}
