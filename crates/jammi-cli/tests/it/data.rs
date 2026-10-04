//! CLI integration test for the data verbs, `embed` and `search`, over the
//! wire: a source registered with `sources add`, embedded with `embed`, and
//! searched with `search` against a hermetic `jammi-server`, the encoder the
//! shipped local `tiny_bert` checkpoint (no network).

use std::path::{Path, PathBuf};

use crate::server_harness::TestServer;

fn workspace_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .join(relative)
}

fn stdout(server: &TestServer, args: &[&str]) -> String {
    let out = server.cli().args(args).output().expect("run jammi");
    assert!(
        out.status.success(),
        "`jammi {}` failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf-8 stdout")
}

/// `embed` writes a result table of one vector per row, and `search` returns
/// the `k` rows nearest a row's own vector as JSON lines in rank order — the
/// row itself first, at similarity 1.
#[test]
fn cli_embed_then_search_returns_the_nearest_rows_as_json_lines() {
    let corpus = workspace_path("cookbook/fixtures/tiny_corpus.parquet");
    let model = format!(
        "local:{}",
        workspace_path("tests/fixtures/tiny_bert").display()
    );
    let server = TestServer::spawn();

    stdout(
        &server,
        &[
            "sources",
            "add",
            "corpus",
            "--url",
            corpus.to_str().expect("utf-8 path"),
            "--format",
            "parquet",
        ],
    );
    let embedded = stdout(
        &server,
        &[
            "embed",
            "corpus",
            "--model",
            &model,
            "--columns",
            "content",
            "--key",
            "id",
        ],
    );
    assert!(embedded.contains("rows:       20"), "{embedded}");
    assert!(embedded.contains("cache:      computed"), "{embedded}");

    let found = stdout(
        &server,
        &[
            "search",
            "corpus",
            "--row-key",
            "1",
            "-k",
            "3",
            "--select",
            "id,similarity",
        ],
    );
    let rows: Vec<serde_json::Value> = found
        .lines()
        .map(|line| serde_json::from_str(line).expect("a JSON line per row"))
        .collect();
    assert_eq!(rows.len(), 3, "{found}");
    assert_eq!(rows[0]["id"], 1, "the row itself ranks first: {found}");
    let similarities: Vec<f64> = rows
        .iter()
        .map(|row| row["similarity"].as_f64().expect("a similarity"))
        .collect();
    assert!((similarities[0] - 1.0).abs() < 1e-5, "{found}");
    assert!(
        similarities.windows(2).all(|pair| pair[0] >= pair[1]),
        "rank order: {found}"
    );
}
