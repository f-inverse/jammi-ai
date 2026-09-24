//! A model served at a remote endpoint runs through the same surfaces a
//! local model does — a plan's embedding, a single-row query — and the
//! materialization records its declaration as the run. Every proof runs
//! against a hermetic mock endpoint that answers each input with a vector
//! derived from its text, so a vector landing on the wrong row is visible.

use std::collections::BTreeMap;
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use jammi_ai::model::backend::remote::RemoteModel;
use jammi_ai::session::InferenceSession;
use jammi_datafusion::{ModelSource, ModelTask};
use jammi_db::config::{RemoteModelConfig, RemoteProtocol, SecretSource};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::storage::StorageUrl;
use jammi_db::store::manifest::{ModelIdentity, ModelRun, RemoteRun};
use jammi_db::store::CachePolicy;
use tempfile::TempDir;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

use crate::common;

const DIMS: usize = 4;
const SECRET: &str = "Bearer s3cret-token";

/// The vector the mock endpoint answers `text` with: its length, then the
/// code of each of its first three characters — distinct per text, so a
/// vector on the wrong row is caught.
fn expected_vector(text: &str) -> Vec<f32> {
    let mut v = vec![text.chars().count() as f32];
    v.extend(text.chars().take(DIMS - 1).map(|c| c as u32 as f32));
    v.resize(DIMS, 0.0);
    v
}

/// An OpenAI-compatible embeddings endpoint, answering every input with
/// [`expected_vector`] in reverse order — the protocol's `index` is what
/// places a vector, never its position in the response.
struct Embedder;

impl Respond for Embedder {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
        let inputs: Vec<String> = serde_json::from_value(body["input"].clone()).unwrap();
        let data: Vec<serde_json::Value> = inputs
            .iter()
            .enumerate()
            .rev()
            .map(|(index, text)| {
                serde_json::json!({"object": "embedding", "index": index,
                                   "embedding": expected_vector(text)})
            })
            .collect();
        ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "object": "list", "data": data, "model": body["model"],
        }))
    }
}

fn declaration(server: &MockServer) -> RemoteModelConfig {
    RemoteModelConfig {
        protocol: RemoteProtocol::OpenaiEmbeddings,
        url: format!("{}/v1/embeddings", server.uri()),
        model: "encoder-small".into(),
        dimensions: NonZeroUsize::new(DIMS).unwrap(),
        revision: "2026-09".into(),
        headers: BTreeMap::from([(
            "Authorization".to_string(),
            SecretSource::Inline(SECRET.into()),
        )]),
        timeout_secs: NonZeroU64::new(5).unwrap(),
        max_in_flight: NonZeroUsize::new(2).unwrap(),
        max_retries: 0,
    }
}

async fn session_with(dir: &TempDir, declared: RemoteModelConfig) -> Arc<InferenceSession> {
    let mut config = common::test_config(dir.path());
    config.models.remote.insert("encoder".into(), declared);
    Arc::new(InferenceSession::new(config).await.unwrap())
}

async fn mount_embedder(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(header("authorization", SECRET))
        .respond_with(Embedder)
        .mount(server)
        .await;
}

fn text_column(texts: &[&str]) -> Vec<ArrayRef> {
    vec![Arc::new(StringArray::from(texts.to_vec())) as ArrayRef]
}

/// A remote model embeds a source through the same plan a local model runs,
/// every vector on its own row, and the materialization records the
/// declaration — endpoint, name, width, revision — as the run. A query is
/// encoded by the same endpoint into the same vector the table holds.
#[tokio::test]
async fn a_remote_model_embeds_a_source_and_its_declaration_is_the_recorded_run() {
    let server = MockServer::start().await;
    mount_embedder(&server).await;
    let dir = TempDir::new().unwrap();
    let declared = declaration(&server);
    let session = session_with(&dir, declared.clone()).await;

    let texts = ["alpha", "beta", "gamma", "delta", "epsilon"];
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("text", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from_iter_values(0..texts.len() as i64)),
            Arc::new(StringArray::from(texts.to_vec())),
        ],
    )
    .unwrap();
    let parquet = dir.path().join("src.parquet");
    let mut writer = parquet::arrow::ArrowWriter::try_new(
        std::fs::File::create(&parquet).unwrap(),
        schema,
        None,
    )
    .unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    session
        .add_source(
            "src",
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", parquet.display())),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let (record, _) = session
        .generate_text_embeddings(
            "src",
            "remote:encoder",
            &["text".to_string()],
            "id",
            CachePolicy::Bypass,
            None,
        )
        .await
        .unwrap();

    let vectors = common::read_table_vectors(&session, &record).await;
    assert_eq!(vectors.len(), texts.len());
    for (row, text) in texts.iter().enumerate() {
        assert_eq!(
            vectors[&row.to_string()],
            expected_vector(text),
            "row {row}"
        );
    }

    let manifest = session
        .result_store()
        .read_materialization_manifest(&StorageUrl::parse(&record.parquet_path).unwrap())
        .await
        .unwrap()
        .expect("the table's manifest");
    assert_eq!(
        manifest.env.models,
        vec![ModelIdentity {
            model_id: "remote:encoder".into(),
            run: ModelRun::Remote(RemoteRun {
                protocol: RemoteProtocol::OpenaiEmbeddings,
                url: declared.url.clone(),
                model: "encoder-small".into(),
                dimensions: DIMS as u64,
                revision: "2026-09".into(),
            }),
        }]
    );

    assert_eq!(
        session
            .encode_text_query("remote:encoder", "gamma")
            .await
            .unwrap(),
        expected_vector("gamma")
    );
}

/// A row with no text is never sent: the request carries only the rows
/// with input, and the empty row keeps its own refusal.
#[tokio::test]
async fn an_empty_row_is_marked_and_never_sent() {
    let server = MockServer::start().await;
    mount_embedder(&server).await;
    let model = RemoteModel::from_config("encoder", &declaration(&server)).unwrap();

    let prepared = model
        .prepare(
            &text_column(&["one", "", "three"]),
            ModelTask::TextEmbedding,
        )
        .unwrap();
    let out = model.forward(prepared).await.unwrap();

    assert_eq!(out.row_status, vec![true, false, true]);
    assert_eq!(out.row_errors[1], "Empty or null text input");
    let flat = &out.float_outputs[0];
    assert_eq!(flat[..DIMS], expected_vector("one")[..]);
    assert_eq!(flat[2 * DIMS..], expected_vector("three")[..]);
    let sent: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(sent["input"], serde_json::json!(["one", "three"]));
    assert_eq!(sent["model"], "encoder-small");
}

/// Answer every request with `body` under `status`.
async fn mount_fixed(server: &MockServer, status: u16, body: serde_json::Value) {
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(status).set_body_json(body))
        .mount(server)
        .await;
}

async fn forward_err(model: &RemoteModel, texts: &[&str]) -> String {
    let prepared = model
        .prepare(&text_column(texts), ModelTask::TextEmbedding)
        .unwrap();
    model
        .forward(prepared)
        .await
        .expect_err("the response must be refused")
        .to_string()
}

/// A response that answers an input twice, leaves one out, or returns a
/// vector of another width is refused naming the input or the row — never
/// paired with rows by position, never spliced into a neighbour.
#[tokio::test]
async fn a_short_repeated_or_ragged_response_is_refused_by_row() {
    let cases = [
        (
            serde_json::json!({"data": [{"index": 0, "embedding": [1.0, 2.0, 3.0, 4.0]}]}),
            "no vector for input 1",
        ),
        (
            serde_json::json!({"data": [
                {"index": 0, "embedding": [1.0, 2.0, 3.0, 4.0]},
                {"index": 0, "embedding": [1.0, 2.0, 3.0, 4.0]}]}),
            "index 0 answered twice",
        ),
        (
            serde_json::json!({"data": [
                {"index": 0, "embedding": [1.0, 2.0, 3.0, 4.0]},
                {"index": 1, "embedding": [1.0, 2.0, 3.0]}]}),
            "row 1 has width 3, expected 4",
        ),
        (
            serde_json::json!({"data": [
                {"index": 0, "embedding": [1.0, 2.0, 3.0, 4.0]},
                {"index": 7, "embedding": [1.0, 2.0, 3.0, 4.0]}]}),
            "index 7 is outside the 2 input(s) sent",
        ),
    ];
    for (body, expected) in cases {
        let server = MockServer::start().await;
        mount_fixed(&server, 200, body).await;
        let model = RemoteModel::from_config("encoder", &declaration(&server)).unwrap();
        let err = forward_err(&model, &["first", "second"]).await;
        assert!(err.contains(expected), "expected {expected:?} in {err}");
        assert!(err.contains("remote:encoder"), "names the model: {err}");
    }
}

/// A refusal the endpoint states (a non-2xx other than 429 or 5xx) fails
/// the forward with the status and the endpoint's own words — and the
/// credential sent with the request appears in no error and no `Debug`.
#[tokio::test]
async fn a_refusing_endpoint_fails_the_forward_and_no_credential_is_printed() {
    let server = MockServer::start().await;
    mount_fixed(
        &server,
        401,
        serde_json::json!({"error": {"message": "invalid api key"}}),
    )
    .await;
    let declared = declaration(&server);
    let model = RemoteModel::from_config("encoder", &declared).unwrap();

    let err = forward_err(&model, &["text"]).await;
    assert!(err.contains("401"), "{err}");
    assert!(err.contains("invalid api key"), "{err}");
    for printed in [err, format!("{model:?}"), format!("{declared:?}")] {
        assert!(
            !printed.contains("s3cret"),
            "a credential was printed: {printed}"
        );
    }
    let sent = &server.received_requests().await.unwrap()[0];
    assert_eq!(
        sent.headers.get("authorization").unwrap().to_str().unwrap(),
        SECRET,
        "the credential is sent"
    );
}

/// A 429 is retried after the delay the endpoint names; with no retry left
/// it fails naming the status and the attempts made.
#[tokio::test]
async fn a_rate_limited_request_is_retried_until_the_retries_run_out() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "0"))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    mount_embedder(&server).await;

    let retrying = RemoteModel::from_config(
        "encoder",
        &RemoteModelConfig {
            max_retries: 1,
            ..declaration(&server)
        },
    )
    .unwrap();
    let prepared = retrying
        .prepare(&text_column(&["retry"]), ModelTask::TextEmbedding)
        .unwrap();
    let out = retrying.forward(prepared).await.unwrap();
    assert_eq!(out.float_outputs[0], expected_vector("retry"));
    assert_eq!(server.received_requests().await.unwrap().len(), 2);

    let exhausted_server = MockServer::start().await;
    mount_fixed(&exhausted_server, 429, serde_json::json!({})).await;
    let exhausted = RemoteModel::from_config("encoder", &declaration(&exhausted_server)).unwrap();
    let err = forward_err(&exhausted, &["retry"]).await;
    assert!(
        err.contains("429") && err.contains("after 1 attempt(s)"),
        "{err}"
    );
}

/// A request that outlasts the declared timeout fails the forward naming
/// the model, not a hang.
#[tokio::test]
async fn a_request_past_its_timeout_fails_the_forward() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(5)))
        .mount(&server)
        .await;
    let model = RemoteModel::from_config(
        "encoder",
        &RemoteModelConfig {
            timeout_secs: NonZeroU64::new(1).unwrap(),
            ..declaration(&server)
        },
    )
    .unwrap();
    let started = std::time::Instant::now();
    let err = forward_err(&model, &["slow"]).await;
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "the timeout bounded the wait"
    );
    assert!(err.contains("remote:encoder"), "{err}");
}

/// A name no declaration carries, and a task the protocol does not carry,
/// are refused before any request is sent.
#[tokio::test]
async fn an_undeclared_name_and_an_uncarried_task_are_refused() {
    let server = MockServer::start().await;
    mount_embedder(&server).await;
    let dir = TempDir::new().unwrap();
    let session = session_with(&dir, declaration(&server)).await;

    let err = session
        .encode_text_query("remote:missing", "text")
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("[models.remote.missing]"), "{err}");

    let err = session
        .model_cache()
        .describe(&ModelSource::remote("encoder"), ModelTask::Classification)
        .await
        .err()
        .expect("classification is not an embeddings request")
        .to_string();
    assert!(err.contains("text embedding only"), "{err}");
    assert!(server.received_requests().await.unwrap().is_empty());
}
