//! Gang test fixtures shared by the `jammi-ai` and `jammi-server` integration
//! suites: the pairs fixture, the gang config, and the in-process `LocalGang`
//! reference a published adapter is compared against byte for byte.
//!
//! One source file, compiled into both test crates — `jammi-server` includes it
//! by `#[path]` — so a fixture is fixed once. Graph fixtures stay with each
//! suite: the two suites sample different graphs on purpose.

use std::sync::Arc;
use std::time::Duration;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use jammi_ai::fine_tune::collective::{BlockingCall, LocalGang};
use jammi_ai::fine_tune::data::TrainingDataLoader;
use jammi_ai::fine_tune::lora::build_projection_head_for_rank;
use jammi_ai::fine_tune::partition::{PartitionRule, PartitionSpec};
use jammi_ai::fine_tune::source::TrainingSource;
use jammi_ai::fine_tune::spec::{TrainingCommon, TrainingSpec};
use jammi_ai::fine_tune::target::TrainingTarget;
use jammi_ai::fine_tune::trainer::{RankContext, TrainingLoopBuilder};
use jammi_ai::fine_tune::{EarlyStoppingMetric, FineTuneConfig, FineTuneMethod};
use jammi_ai::model::{ModelSource, ModelTask};
use jammi_ai::session::InferenceSession;
use jammi_db::storage::{StorageRegistry, StorageUrl};
use jammi_db::store::{ArtifactStore, CachePolicy};
use jammi_test_utils::tiny_vocab_text;
use tempfile::TempDir;

/// Eight `(anchor, positive)` rows the tiny model can tell apart — row from
/// row, and anchor from positive (`tiny_vocab_text`). Single-digit indices, so
/// the file order and the training set's full-tuple order
/// (`training_set_order_by`) agree.
pub(crate) fn pairs() -> Vec<(String, String)> {
    (0..8)
        .map(|i| (tiny_vocab_text('a', i), tiny_vocab_text('p', i)))
        .collect()
}

/// The gang oracle's config: two epochs, per-rank batch 2, no dropout,
/// no validation split, a fixed seed.
pub(crate) fn gang_config(epochs: usize) -> FineTuneConfig {
    gang_config_with_dropout(epochs, 0.0)
}

/// [`gang_config`] at an explicit `lora_dropout` — the seed-split oracle
/// needs a live mask source (`LoraLinear::dropout_run_seed` is `None` at
/// `lora_dropout == 0`).
pub(crate) fn gang_config_with_dropout(epochs: usize, lora_dropout: f64) -> FineTuneConfig {
    FineTuneConfig {
        epochs,
        batch_size: 2,
        validation_fraction: 0.0,
        warmup_steps: 0,
        gradient_accumulation_steps: 1,
        lora_rank: 2,
        lora_dropout,
        seed: 99,
        early_stopping_metric: EarlyStoppingMetric::TrainLoss,
        early_stopping_patience: 10_000,
        learning_rate: 1e-4,
        ..Default::default()
    }
}

pub(crate) fn tiny_bert_model() -> String {
    "local:".to_string()
        + jammi_test_utils::cookbook_fixture("tiny_bert")
            .to_str()
            .unwrap()
}

/// The pairs as a CSV source file under `dir`, returned as a `file://` URL.
pub(crate) fn write_pairs_csv(dir: &std::path::Path) -> String {
    let path = dir.join("pairs.csv");
    let mut body = String::from("anchor,positive\n");
    for (anchor, positive) in pairs() {
        body.push_str(&format!("{anchor},{positive}\n"));
    }
    std::fs::write(&path, body).unwrap();
    format!("file://{}", path.display())
}

pub(crate) fn two_rank_spec() -> TrainingSpec {
    TrainingSpec::FineTune {
        source: "pairs".into(),
        columns: vec!["anchor".into(), "positive".into()],
        method: FineTuneMethod::Lora,
        task: ModelTask::TextEmbedding,
        common: TrainingCommon {
            base_model: tiny_bert_model(),
            config: gang_config(2),
            world_size: 2,
            cache: CachePolicy::Bypass,
        },
    }
}

/// A catalog holding a claimed `running` row for a trainer built directly —
/// `trainer.rs`'s own `test_fixtures::claimed_job` shape.
pub(crate) async fn claimed_loop_env(tag: &str) -> (Arc<jammi_db::catalog::Catalog>, TempDir) {
    let dir = TempDir::new().unwrap();
    let catalog = Arc::new(jammi_db::catalog::Catalog::open(dir.path()).await.unwrap());
    let model_id = format!("{tag}-model");
    catalog
        .register_model(jammi_db::catalog::model_repo::RegisterModelParams {
            model_id: &model_id,
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            external_location: None,
            config_json: None,
        })
        .await
        .unwrap();
    catalog
        .submit_job(jammi_db::catalog::jobs_repo::SubmitJobParams {
            job_id: tag,
            kind: "fine_tune",
            execution: jammi_db::catalog::status::JobExecution::Queued,
            spec: "{}",
            model_ref: Some(&format!("{model_id}::1")),
            output_model_id: None,
            model_source: None,
            priority: 0,
        })
        .await
        .unwrap();
    catalog
        .claim_next(
            &format!("{tag}-worker"),
            &["fine_tune"],
            Duration::from_secs(60),
        )
        .await
        .unwrap()
        .expect("queued job claimable");
    (catalog, dir)
}

pub(crate) fn file_store() -> Arc<ArtifactStore> {
    let root_dir = TempDir::new().unwrap().keep();
    let cache = TempDir::new().unwrap().keep();
    let root = StorageUrl::parse(root_dir.to_str().unwrap()).unwrap();
    Arc::new(ArtifactStore::with_root(root, StorageRegistry::new(), cache).unwrap())
}

/// The reference: the gang oracle's shape — a two-rank `LocalGang` on
/// the CPU, each rank a `TrainingLoop` built directly (projection head over
/// tiny_bert through `build_projection_head_for_rank` at the rank's own
/// `RankContext::dropout_seed`, `PartitionSpec::for_gang(r, 2, 2, ..)`),
/// driven through the real `TrainingLoop::run` on its own
/// `BlockingCall::spawn_thread`; returns rank 0's saved `adapter.safetensors`
/// bytes. `session` supplies the base model through its model cache (the
/// same resolve + load path the worker uses).
pub(crate) async fn reference_rank0_adapter_bytes(
    session: &Arc<InferenceSession>,
    tag: &str,
    loader: fn() -> TrainingDataLoader,
    config: FineTuneConfig,
) -> Vec<u8> {
    let source = ModelSource::parse(&tiny_bert_model());
    let guard = session
        .model_cache()
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();
    let base = Arc::clone(&guard.model);
    let hidden = guard.model.embedding_dim().unwrap();
    drop(guard);
    let gang = LocalGang::new(vec![Device::Cpu, Device::Cpu]).unwrap();
    let store = file_store();
    let runtime = tokio::runtime::Handle::current();
    let job_id = format!("{tag}-reference");
    let mut threads = Vec::new();
    for rank in 0..2u32 {
        let local = gang.rank(rank).unwrap();
        let partition =
            PartitionSpec::for_gang(rank as usize, 2, 2, PartitionRule::BlockByGlobalBatch)
                .unwrap();
        let rank_ctx = RankContext::new(Arc::new(local), partition);
        let dropout_seed = rank_ctx.dropout_seed(config.seed);
        let (catalog, dir) = claimed_loop_env(&format!("{tag}-ref-{rank}")).await;
        let base = Arc::clone(&base);
        let store = Arc::clone(&store);
        let runtime = runtime.clone();
        let job_id = job_id.clone();
        let config = config.clone();
        threads.push(BlockingCall::spawn_thread(move |call| {
            let _runtime = runtime.enter();
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
            let head = build_projection_head_for_rank(hidden, &config, &varmap, &vb, dropout_seed)
                .unwrap();
            let mut training_loop =
                TrainingLoopBuilder::new(TrainingTarget::ProjectionHead { head }, varmap, config)
                    .device(Device::Cpu)
                    .job_id(job_id)
                    .catalog(catalog)
                    .artifact_dir(dir.path().to_path_buf())
                    .base_model(base)
                    .artifact_store(store)
                    .rank_context(rank_ctx)
                    .build()
                    .unwrap();
            let result = training_loop
                .run(&call, TrainingSource::Resident(loader()))
                .unwrap_or_else(|e| panic!("reference rank {rank} must complete: {e}"));
            let bytes =
                std::fs::read(result.artifact_dir.path().join("adapter.safetensors")).unwrap();
            drop(dir);
            bytes
        }));
    }
    let mut rank0 = None;
    for (rank, thread) in threads.into_iter().enumerate() {
        let bytes = thread.join().unwrap();
        if rank == 0 {
            rank0 = Some(bytes);
        }
    }
    rank0.expect("rank 0 ran")
}
