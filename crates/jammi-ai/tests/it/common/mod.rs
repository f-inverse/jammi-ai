pub use jammi_test_utils::*;

use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use jammi_ai::model::hub::HubSource;
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::model_repo::{ModelLocation, ModelRecord};
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::config::ModelsConfig;
use jammi_db::error::{JammiError, Result as JammiResult};
use jammi_db::storage::StorageUrl;
use jammi_db::store::{ArtifactStore, PinnedSource};
use jammi_numerics::retrieval::AggregateMetrics;

/// The ANN index-segment bundle base URLs of `table_name`, in segment order —
/// the load-side handle for tests that inspect a table's on-disk sidecar bundle
/// now that a table's index is a set of segments rather than one `index_path`.
/// Pin `record`'s current version — the value every version-bearing verb
/// (`read_vectors`, `verify_materialization`, `producing_descriptor`) takes,
/// and a test's only way to learn which version a producer left current.
pub async fn pin(session: &InferenceSession, record: ResultTableRecord) -> PinnedSource {
    session
        .result_store()
        .pin_current_version(record)
        .await
        .expect("the current version resolves")
}

/// The current version of the table named `table`, resolved by one pin.
pub async fn current_version(session: &InferenceSession, table: &str) -> Option<i64> {
    let record = session
        .catalog()
        .get_result_table(table)
        .await
        .unwrap()
        .expect("table present");
    pin(session, record).await.version()
}

pub async fn segment_index_urls(
    session: &jammi_ai::session::InferenceSession,
    table_name: &str,
) -> Vec<String> {
    session
        .catalog()
        .list_index_segments(table_name)
        .await
        .unwrap()
        .into_iter()
        .map(|s| s.index_path)
        .collect()
}

/// The single segment's bundle base URL for a freshly-embedded table (one embed
/// pass writes exactly one segment). Panics unless there is exactly one.
pub async fn segment0_index_url(
    session: &jammi_ai::session::InferenceSession,
    table_name: &str,
) -> String {
    let mut urls = segment_index_urls(session, table_name).await;
    assert_eq!(
        urls.len(),
        1,
        "expected exactly one index segment for '{table_name}'"
    );
    urls.pop().unwrap()
}

/// Build an [`ArtifactStore`] rooted at a hermetic `memory://` URL with a fresh
/// local fetch cache, for resolver-level unit tests that construct a
/// `ModelResolver` directly (rather than through a full `InferenceSession`). The
/// cache dir leaks for the test binary's lifetime — acceptable in a test.
pub fn test_artifact_store() -> Arc<ArtifactStore> {
    let cache = tempfile::tempdir().unwrap().keep();
    Arc::new(
        ArtifactStore::with_root(
            jammi_db::storage::StorageUrl::memory("test-artifacts"),
            jammi_db::storage::StorageRegistry::new(),
            cache,
        )
        .unwrap(),
    )
}

/// Build a [`HubSource`] rooted at a fresh tempdir with no configured
/// endpoint/token — for resolver-level integration tests that construct a
/// `ModelResolver` directly rather than through a full `InferenceSession`.
/// Never touches the real network or process environment: no test in this
/// suite that needs a live Hub round-trip uses this fixture (see
/// `hub_source.rs`, which builds its own `HubSource` per test against a
/// wiremock server). The cache dir leaks for the test binary's lifetime —
/// acceptable in a test, same as [`test_artifact_store`].
pub fn test_hub_source() -> HubSource {
    let root = tempfile::tempdir().unwrap().keep();
    HubSource::from_config(
        &ModelsConfig {
            hub_cache_dir: Some(root),
            ..Default::default()
        },
        &|_: &str| None,
    )
    .unwrap()
}

/// Return the four aggregate retrieval metrics paired with their wire-format
/// snake_case names. Tests that need to iterate over every metric (range
/// checks, baseline-vs-candidate diffs, determinism comparisons) consume this
/// instead of indexing fields by string — the array literal makes adding a
/// new metric to [`AggregateMetrics`] a single-file edit that the compiler
/// guides exhaustively.
pub fn aggregate_named_metrics(agg: &AggregateMetrics) -> [(&'static str, f64); 4] {
    [
        ("recall_at_k", agg.recall_at_k),
        ("precision_at_k", agg.precision_at_k),
        ("mrr", agg.mrr),
        ("ndcg", agg.ndcg),
    ]
}

// =============================================================================
// The cold-serve control shared by every `*_serves_cold_after_restart` test:
// three cross-modal towers in `tower_adapters.rs` and the BERT-family peer in
// `fine_tune.rs`. It lives in exactly one place, so every site runs the SAME
// oracle:
//
//   - positive controls, in the WARM session, before any cold assertion:
//     `v_base`/`v_warm` both fully finite and non-degenerate (L2 > 1e-6);
//     `max|v_warm - v_base| >= 1e-3` over finite pairs; a second
//     `serve(base_id)` bit-identical to the first (base-side determinism);
//   - the cold mechanism assertion: `v_cold == v_warm` bit-for-bit;
//   - the mechanism co-assertion: after the restart, the fine-tuned id's
//     catalog record (read through `Catalog::get_model`, the db-API read
//     path) still carries `model_type ==
//     "fine-tuned"`, a referenced artifact, `base_model_id.is_some()`;
//   - the negative control: with the published bundle's `adapter.safetensors`
//     deleted, a COLD serve through a brand-new `InferenceSession` (the real
//     resolver, never a hand-built `ResolvedModel`) refuses with a typed
//     `JammiError::Model` naming the missing file and returns no vector.
// =============================================================================

/// A boxed, higher-ranked "serve this model" closure:
/// `Fn(&InferenceSession) -> future<Result<Vec<f32>>>`, generic over the
/// session's own borrow so the SAME closure re-applies unchanged to the warm
/// session, the cold session, and (for the negative control) a third
/// post-mutation session, without the caller re-borrowing anything it owns.
/// [`text_serve`]/[`image_serve`]/[`audio_serve`] build one per probe by
/// moving in an owned model id and payload.
pub type ServeFn = Box<
    dyn for<'s> Fn(
            &'s InferenceSession,
        ) -> Pin<Box<dyn Future<Output = JammiResult<Vec<f32>>> + Send + 's>>
        + Send
        + Sync,
>;

/// A [`ServeFn`] over [`InferenceSession::encode_text_query`].
pub fn text_serve(model_id: impl Into<String>, probe: &'static str) -> ServeFn {
    let model_id = model_id.into();
    Box::new(move |s: &InferenceSession| {
        let model_id = model_id.clone();
        Box::pin(async move { s.encode_text_query(&model_id, probe).await })
    })
}

/// A [`ServeFn`] over [`InferenceSession::encode_image_query`]. `bytes` is
/// `Arc`-shared rather than cloned per call — a probe image can be sizeable
/// and this closure is invoked at least three times (warm repeat, cold, the
/// negative control's post-mutation serve).
pub fn image_serve(model_id: impl Into<String>, bytes: Arc<Vec<u8>>) -> ServeFn {
    let model_id = model_id.into();
    Box::new(move |s: &InferenceSession| {
        let model_id = model_id.clone();
        let bytes = Arc::clone(&bytes);
        Box::pin(async move { s.encode_image_query(&model_id, &bytes).await })
    })
}

/// A [`ServeFn`] over [`InferenceSession::encode_audio_query`]. See
/// [`image_serve`] for why `bytes` is `Arc`-shared.
pub fn audio_serve(model_id: impl Into<String>, bytes: Arc<Vec<u8>>) -> ServeFn {
    let model_id = model_id.into();
    Box::new(move |s: &InferenceSession| {
        let model_id = model_id.clone();
        let bytes = Arc::clone(&bytes);
        Box::pin(async move { s.encode_audio_query(&model_id, &bytes).await })
    })
}

/// Positive control (a): every component of `v` is finite and `v`'s
/// L2 norm is `> 1e-6` — rules out a degenerate (NaN-laced, all-zero, or
/// near-zero) embedding making the downstream difference/bit-equality
/// checks pass vacuously.
pub fn assert_finite_and_nondegenerate(v: &[f32], label: &str) {
    for (i, x) in v.iter().enumerate() {
        assert!(
            x.is_finite(),
            "{label}: component {i} is non-finite ({x}) — every component of a served \
             embedding must be finite"
        );
    }
    let norm = v
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    assert!(
        norm > 1e-6,
        "{label}: L2 norm {norm} is not > 1e-6 — a degenerate (near-zero) embedding"
    );
}

/// Positive control (b): `max|other[i] - base[i]|`, taken ONLY over index
/// pairs where both components are finite (a control must fail on every bad
/// path, including non-finite — `NaN > c` is `false`, so
/// letting a non-finite component silently poison the max would make this
/// assertion pass vacuously on exactly the input it exists to reject). Panics
/// if `base`/`other` share no finite pair at all, rather than reporting a
/// vacuous `max_diff = 0.0`.
pub fn assert_min_diff_over_finite_pairs(base: &[f32], other: &[f32], min_diff: f32, label: &str) {
    assert_eq!(
        base.len(),
        other.len(),
        "{label}: base and other embeddings must have the same width"
    );
    let mut max_diff = 0.0f32;
    let mut finite_pairs = 0usize;
    for (a, b) in base.iter().zip(other) {
        if a.is_finite() && b.is_finite() {
            finite_pairs += 1;
            max_diff = max_diff.max((a - b).abs());
        }
    }
    assert!(
        finite_pairs > 0,
        "{label}: no finite (base, other) pair to compare — at least one vector is entirely \
         non-finite"
    );
    assert!(
        max_diff >= min_diff,
        "{label}: max|Δ| over {finite_pairs} finite pairs = {max_diff}, expected >= {min_diff}"
    );
}

/// The cold-restart mechanism assertion:
/// `served` and `reference` must be bit-for-bit identical.
pub fn assert_bit_equal(served: &[f32], reference: &[f32], label: &str) {
    assert_eq!(
        served.len(),
        reference.len(),
        "{label}: served and reference embeddings must have the same width"
    );
    for (i, (a, b)) in served.iter().zip(reference).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label}: component {i} differs bit-for-bit (served {a}, reference {b})"
        );
    }
}

/// Mechanism co-assertion: after a cold restart, `model_id`'s catalog record
/// — read through [`jammi_db::catalog::Catalog::get_model`], the db-API read
/// path — still carries the three fields `ModelResolver::try_catalog_lookup`'s
/// fine-tuned arm depends on. This pins the ROW, not just the served vector:
/// a regression that clobbers `model_type`/the artifact reference/`base_model_id`
/// (e.g. in `ModelCache::do_load`'s post-load bookkeeping) but happens to
/// leave both warm and cold resolving to the
/// SAME corrupted row would still pass a bit-equality-only oracle.
pub async fn assert_fine_tuned_record_intact(
    session: &InferenceSession,
    model_id: &str,
    label: &str,
) {
    let record = session
        .catalog()
        .get_model(model_id)
        .await
        .expect("catalog lookup")
        .unwrap_or_else(|| {
            panic!("{label}: fine-tuned model '{model_id}' is not registered in the catalog after restart")
        });
    assert_eq!(
        record.model_type, "fine-tuned",
        "{label}: catalog record's model_type must stay 'fine-tuned' after restart, got '{}'",
        record.model_type
    );
    assert!(
        matches!(record.location, Some(ModelLocation::Artifact(_))),
        "{label}: catalog record must still reference its artifact after restart"
    );
    assert!(
        record.base_model_id.is_some(),
        "{label}: catalog record's base_model_id must stay Some after restart"
    );
}

/// Negative control: with the published bundle's `adapter.safetensors`
/// deleted, a COLD serve through a brand-new [`InferenceSession`] opened over
/// `session_root` (the real resolver, never a hand-built `ResolvedModel`)
/// must refuse — a typed `JammiError::Model` whose message names the missing
/// file — and must never return a vector, finite or not.
///
/// `bundle_dir` is read straight off the artifact `model_id`'s catalog row references: a
/// `file://` prefix resolves to that exact directory with no copy (see
/// `ArtifactStore::fetch_artifact`'s in-place path), so deleting the file
/// there deletes the SAME file every earlier serve in the test read.
pub async fn assert_deleted_adapter_refuses_by_name(
    session: &InferenceSession,
    session_root: &Path,
    model_id: &str,
    label: &str,
    serve_tuned: &ServeFn,
) {
    let record = session
        .catalog()
        .get_model(model_id)
        .await
        .expect("catalog lookup")
        .expect("fine-tuned model registered in catalog");
    let prefix_url = served_bundle_url(&record);
    let bundle_dir = std::path::PathBuf::from(prefix_url.path());
    let weights_path = bundle_dir.join("adapter.safetensors");
    std::fs::remove_file(&weights_path).unwrap_or_else(|e| {
        panic!(
            "{label}: failed to delete the published bundle's adapter.safetensors at \
             {weights_path:?}: {e}"
        )
    });

    let broken_session = InferenceSession::new(test_config(session_root))
        .await
        .unwrap();
    match serve_tuned(&broken_session).await {
        Ok(v) => panic!(
            "{label}: serving '{model_id}' after deleting its adapter.safetensors returned a \
             vector (len {}) instead of refusing — a broken bundle must never silently serve",
            v.len()
        ),
        Err(JammiError::Model { message, .. }) => {
            assert!(
                message.contains("adapter.safetensors"),
                "{label}: refusal message must name the missing file 'adapter.safetensors', \
                 got: {message}"
            );
        }
        Err(other) => panic!(
            "{label}: expected a typed JammiError::Model naming the missing file, got a \
             different error variant: {other:?}"
        ),
    }
}

/// Named-field parameter bundle for [`assert_cold_restart_controls`].
///
/// A positional signature would hold three same-typed `&[f32]`
/// (`v_base`/`v_warm`/`v_cold`) and two same-typed `&ServeFn`
/// (`serve_base`/`serve_tuned`) back to back — a caller transposing any pair
/// would compile silently and assert the wrong control. Named fields make a
/// transposition a field-name typo instead. Follows this repo's
/// params-struct convention for a naturally-wide argument list (see
/// `crates/jammi-ai/src/fine_tune/worker.rs`'s `ModelRegistration`) rather
/// than `#[allow(clippy::too_many_arguments)]`.
pub struct ColdRestartControls<'a> {
    /// The session's on-disk root, needed to locate and delete
    /// `adapter.safetensors` for the negative control.
    pub session_root: &'a Path,
    /// The training instance's still-open session (serves `v_base`/`v_warm`).
    pub warm_session: &'a InferenceSession,
    /// A second, freshly-opened session over the same catalog/artifact dir
    /// (serves `v_cold`).
    pub cold_session: &'a InferenceSession,
    /// The fine-tuned model id under test.
    pub model_id: &'a str,
    /// A short human-readable label for this call's assertions/panics.
    pub label: &'a str,
    /// The base model's embedding, captured in the warm session.
    pub v_base: &'a [f32],
    /// The fine-tuned model's embedding, captured in the warm session.
    pub v_warm: &'a [f32],
    /// The fine-tuned model's embedding, captured in the cold session.
    pub v_cold: &'a [f32],
    /// Re-serves the base model (used for the "served twice" positive
    /// control), applied to `warm_session`.
    pub serve_base: &'a ServeFn,
    /// Re-serves the fine-tuned model, applied to `cold_session` for the
    /// negative control after `adapter.safetensors` is deleted.
    pub serve_tuned: &'a ServeFn,
}

/// Runs the FULL cold-restart control set against one
/// already-completed cold-restart round trip. Called identically by
/// `tower_adapters.rs`'s three cross-modal `*_serves_cold_after_restart`
/// tests and `fine_tune.rs`'s BERT-family peer.
pub async fn assert_cold_restart_controls(controls: ColdRestartControls<'_>) {
    let ColdRestartControls {
        session_root,
        warm_session,
        cold_session,
        model_id,
        label,
        v_base,
        v_warm,
        v_cold,
        serve_base,
        serve_tuned,
    } = controls;

    // Positive controls (warm session).
    assert_finite_and_nondegenerate(v_base, &format!("{label}: v_base"));
    assert_finite_and_nondegenerate(v_warm, &format!("{label}: v_warm"));
    assert_min_diff_over_finite_pairs(v_base, v_warm, 1e-3, &format!("{label}: v_warm vs v_base"));
    let base_repeat = serve_base(warm_session).await.unwrap_or_else(|e| {
        panic!("{label}: second serve(base_id) in the warm session failed: {e}")
    });
    assert_bit_equal(v_base, &base_repeat, &format!("{label}: base served twice"));

    // The cold mechanism assertion.
    assert_bit_equal(v_warm, v_cold, &format!("{label}: cold vs warm"));

    // Mechanism co-assertion, read through the cold (post-restart) session.
    assert_fine_tuned_record_intact(cold_session, model_id, label).await;

    // Negative control.
    assert_deleted_adapter_refuses_by_name(
        cold_session,
        session_root,
        model_id,
        label,
        serve_tuned,
    )
    .await;
}

/// The 70,000-row multi-row-group `(anchor, positive)` fixture, ONE builder
/// shared by the training-set ordering oracles: 70,000 rows so the writer's 65,536-row
/// group boundary is crossed (more than one row group), scrambled by a
/// permutation with no fixed point in the sort order, and every `anchor`
/// value appears twice so `positive` is the tie-breaker (a key-column-only
/// sort would not be total).
///
/// `session` must already exist (each `#[tokio::test]` owns its own
/// session/tempdir — a session is bound to its runtime, so ONE shared
/// fixture *session* across tests is not meaningful; "one fixture per
/// binary" is met as ONE fixture DEFINITION, called by every ordering
/// oracle).
/// Registers the CSV source under `"pairs"` and materialises the
/// `(anchor, positive)` projection.
///
/// `split`, when `true`, issues `SET datafusion.optimizer.repartition_file_min_size
/// = 1` before materialising — DataFusion only splits ONE file across
/// partitions above the default 10 MiB threshold, and a 70k-row ZSTD table is
/// far under that, so without this knob a multi-partition read gets a single
/// file group and any ordering oracle built on it is vacuous (never
/// exercising a genuinely interleaved scan).
pub struct MultiRowGroupFixture {
    /// The materialised training-set table.
    pub table: jammi_db::store::TrainingSetTable,
    /// The projected columns, in order — what [`Self::table`] was
    /// materialised with.
    pub columns: Vec<String>,
    /// Every `(anchor, positive)` row, in WRITE order (before the producer's
    /// own commit sort) — sort this with [`std::vec::Vec::sort`] to get the
    /// canonical (`full_tuple_v1`) committed order, since the fixture carries
    /// no NULLs (a plain tuple sort is the whole key).
    pub written: Vec<(String, String)>,
    /// The number of Parquet row groups the committed file actually has,
    /// measured off the footer — MEASURED, never assumed, so a future writer
    /// change that alters the row-group boundary fails this fixture's own
    /// callers loudly rather than silently making their oracle vacuous.
    pub row_groups: usize,
}

impl MultiRowGroupFixture {
    /// The rows in their canonical (`full_tuple_v1`) committed order: every
    /// projected column, ascending, NULLs first — a plain tuple sort, since
    /// this fixture carries no NULLs.
    pub fn canonical_order(&self) -> Vec<(String, String)> {
        let mut sorted = self.written.clone();
        sorted.sort();
        sorted
    }
}

/// Build [`MultiRowGroupFixture`] over `session` — see the struct's own doc.
pub async fn multi_row_group_pairs(
    session: &Arc<InferenceSession>,
    dir: &std::path::Path,
    split: bool,
) -> MultiRowGroupFixture {
    use jammi_ai::model::ModelTask;
    use jammi_db::source::{FileFormat, SourceConnection, SourceType};

    const ROWS: usize = 70_000;
    let mut lines = String::from("anchor,positive\n");
    let mut written = Vec::with_capacity(ROWS);
    for i in 0..ROWS {
        let n = (i * 37) % ROWS;
        let anchor = format!("a{:05}", n / 2);
        let positive = format!("p{n:05}");
        lines.push_str(&format!("{anchor},{positive}\n"));
        written.push((anchor, positive));
    }
    let csv = dir.join("pairs.csv");
    std::fs::write(&csv, lines).unwrap();

    session
        .add_source(
            "pairs",
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", csv.display())),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    if split {
        session
            .sql("SET datafusion.optimizer.repartition_file_min_size = 1")
            .await
            .unwrap();
    }

    let columns = vec!["anchor".to_string(), "positive".to_string()];
    let (table, _batches) = jammi_ai::fine_tune::training_set::materialize_projection(
        session,
        "pairs",
        &columns,
        ModelTask::TextEmbedding,
        "pairs",
    )
    .await
    .unwrap();

    let url = jammi_db::storage::StorageUrl::parse(table.parquet_path()).unwrap();
    let handle = session.result_store().open_parquet(&url).unwrap();
    let bytes = handle
        .get_bytes(&handle.data_path().unwrap())
        .await
        .unwrap();
    let builder =
        parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(bytes).unwrap();
    let row_groups = builder.metadata().num_row_groups();

    MultiRowGroupFixture {
        table,
        columns,
        written,
        row_groups,
    }
}

/// A `(text, target)` regression fixture whose row PAYLOAD is padded to
/// roughly `pad_bytes` per row — built for the streamed-residency oracle
/// (`training_set_stream.rs`), which needs a table whose EAGER collected
/// size genuinely exceeds `[engine] memory_limit`'s 64 MiB floor (the
/// smallest pool the normal config-validated session-build path can ever
/// produce — `EngineConfig::MEMORY_LIMIT_FLOOR_BYTES`) while a handful of
/// per-step STREAMED chunks stay tiny. `rows` is deliberately small (the
/// padding, not the row count, is what drives total size) so the fixture
/// writes/reads in low single-digit seconds.
pub async fn padded_regression_fixture(
    session: &Arc<InferenceSession>,
    dir: &std::path::Path,
    rows: usize,
    pad_bytes: usize,
) -> (jammi_db::store::TrainingSetTable, Vec<String>) {
    use jammi_ai::model::ModelTask;
    use jammi_db::source::{FileFormat, SourceConnection, SourceType};

    let pad: String = "x".repeat(pad_bytes);
    let mut lines = String::from("text,target\n");
    for i in 0..rows {
        lines.push_str(&format!("row{i:06}{pad},{}\n", i as f32));
    }
    let csv = dir.join("padded.csv");
    std::fs::write(&csv, lines).unwrap();

    session
        .add_source(
            "padded",
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", csv.display())),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let columns = vec!["text".to_string(), "target".to_string()];
    let (table, _batches) = jammi_ai::fine_tune::training_set::materialize_projection(
        session,
        "padded",
        &columns,
        ModelTask::Regression,
        "regression",
    )
    .await
    .unwrap();
    (table, columns)
}

/// The storage URL of the bundle a trained model's catalog row references —
/// what `ArtifactStore::fetch_artifact` reloads it from.
pub fn served_bundle_url(record: &ModelRecord) -> StorageUrl {
    match &record.location {
        Some(ModelLocation::Artifact(artifact)) => artifact.url().clone(),
        other => panic!(
            "model '{}' must reference an artifact, got {other:?}",
            record.model_id
        ),
    }
}

/// A served fine-tuned model produced the way production produces one: a
/// fine-tune job over `base_id` is submitted and claimed, `files` are staged
/// as its attempt's bundle, and the finalize publishes the artifact and
/// writes `model_id`'s row referencing it. Returns the bundle's prefix.
pub async fn finalize_fine_tuned_model(
    catalog: &jammi_db::catalog::Catalog,
    store: &ArtifactStore,
    model_id: &str,
    base_id: &str,
    files: &[(String, bytes::Bytes)],
) -> StorageUrl {
    use jammi_db::catalog::jobs_repo::{
        FinishJobWithModelParams, ModelRow, ProducedModel, SubmitJobParams,
    };
    use jammi_db::catalog::model_repo::RegisterModelParams;
    use jammi_db::catalog::status::JobExecution;

    const WORKER: &str = "fixture-worker";
    if catalog.get_model(base_id).await.unwrap().is_none() {
        catalog
            .register_model(RegisterModelParams {
                model_id: base_id,
                version: 1,
                model_type: "embedding",
                backend: "candle",
                task: jammi_ai::model::ModelTask::TextEmbedding,
                base_model_id: None,
                external_location: None,
                config_json: None,
            })
            .await
            .unwrap();
    }
    let base_pk = catalog
        .get_model(base_id)
        .await
        .unwrap()
        .unwrap()
        .catalog_pk;
    let job_id = uuid::Uuid::new_v4().to_string();
    catalog
        .submit_job(SubmitJobParams {
            job_id: &job_id,
            kind: "fine_tune",
            execution: JobExecution::Queued,
            spec: "{}",
            model_ref: Some(&base_pk),
            output_model_id: Some(model_id),
            model_source: None,
            priority: 0,
        })
        .await
        .unwrap();
    let attempt = catalog
        .claim_next(WORKER, &["fine_tune"], std::time::Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("the queued job is claimable")
        .attempts;
    let staged = store
        .stage_attempt_artifact(catalog, &job_id, WORKER, attempt, files)
        .await
        .unwrap();
    let prefix = staged.artifact().url().clone();
    let finalized = catalog
        .finish_job_with_model(FinishJobWithModelParams {
            job_id: &job_id,
            instance_id: WORKER,
            attempts: attempt,
            result: "{}",
            output: ProducedModel {
                row: ModelRow {
                    model_id,
                    version: 1,
                    model_type: "fine-tuned",
                    backend: "candle",
                    task: jammi_ai::model::ModelTask::TextEmbedding,
                    base_model_id: Some(base_id),
                    config_json: None,
                },
                artifact: staged,
                materialization: None,
            },
            epoch_checkpoints: Vec::new(),
        })
        .await
        .unwrap();
    assert!(finalized, "the lease holder finalizes");
    prefix
}
