//! Producer→consumer parity for [`RankAdmissionRow::world_size`]: the
//! decode `jammi-db`'s `get_job_for_rank` applies to a job's `spec` column
//! must read exactly what `jammi-ai`'s REAL [`TrainingSpec`] producer
//! writes — never a hand-written spec literal standing in for either side.
//! `jammi-server` is the one crate in this workspace that depends on both
//! `jammi-ai` (the producer) and `jammi-db` (the consumer, via
//! `Catalog::submit_job`/`get_job_for_rank`), so this test lives here rather
//! than in either producer or consumer crate alone.
//!
//! [`RankAdmissionRow::world_size`]: jammi_db::catalog::jobs_repo::RankAdmissionRow::world_size

use std::sync::Arc;
use std::time::Duration;

use jammi_ai::fine_tune::spec::{TrainingCommon, TrainingSpec, DEFAULT_WORLD_SIZE};
use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod};
use jammi_ai::pipeline::context_predictor::{
    ContextArchitecture, ContextPredictorTrainConfig, GaussianObjective, PredictiveHead,
};
use jammi_db::catalog::backend::BackendKind;
use jammi_db::catalog::jobs_repo::{SubmitJobParams, WorldSizeFact};
use jammi_db::catalog::status::JobExecution;
use jammi_db::ModelTask;

/// A REAL `TrainingSpec::FineTune`, naming `world_size = 2` under `common`
/// (the shape `TrainingCommon` actually persists) — never a hand-written
/// JSON literal standing in for it.
fn fine_tune_spec_world_two() -> TrainingSpec {
    TrainingSpec::FineTune {
        source: "patents".into(),
        columns: vec!["abstract".into()],
        method: FineTuneMethod::Lora,
        task: ModelTask::TextEmbedding,
        common: TrainingCommon {
            base_model: "local:tiny".into(),
            config: FineTuneConfig::default(),
            world_size: 2,
            cache: jammi_db::store::CachePolicy::Bypass,
        },
        // The variant's cache policy field; this test decodes only
        // `world_size`, so the value is the default every submit edge takes.
    }
}

/// A REAL `TrainingSpec::ContextPredictor` — the one producer variant this
/// engine ships that carries no `common: TrainingCommon` block at all, so
/// its serialized JSON genuinely names no `world_size` key anywhere
/// (top-level or nested): `TrainingCommon::world_size` itself always
/// serializes (no `skip_serializing_if`), so no CURRENT
/// `FineTune`/`GraphFineTune` value can produce a spec missing the field —
/// this variant is the honest way to exercise the absent-default arm
/// without editing any producer's JSON by hand. Field values are otherwise
/// arbitrary (this test never validates or runs the spec, only decodes its
/// `world_size`), mirroring `context_predictor.rs`'s own `high_offset_spec`
/// test fixture shape.
fn context_predictor_spec() -> TrainingSpec {
    TrainingSpec::ContextPredictor {
        source: "episodes".into(),
        predictor_spec: ContextPredictorTrainConfig {
            model_id: "oracle".into(),
            architecture: ContextArchitecture::AttnCnp,
            key_column: "_row_id".into(),
            task_column: "task".into(),
            value_column: "year".into(),
            context_k: 6,
            hidden_dim: 16,
            num_heads: 2,
            num_layers: 2,
            head: PredictiveHead::Gaussian {
                objective: GaussianObjective::Crps,
            },
            epochs: 1,
            learning_rate: 0.02,
            grad_clip: 1.0,
            test_task_fraction: 0.25,
            min_task_count: 2,
            seed: 7,
        },
    }
}

/// The producer→consumer parity oracle. A REAL `TrainingSpec::FineTune`
/// (`common.world_size = 2`), serialized exactly as `submit_job` would
/// persist it and submitted through `Catalog::submit_job` (never a
/// hand-written spec literal on this path), decodes through
/// `get_job_for_rank` to `WorldSizeFact::Decoded(2)`. A second, genuinely
/// different REAL producer variant — `TrainingSpec::ContextPredictor`,
/// which names no `world_size` anywhere — decodes to the documented absent
/// default, `Decoded(DEFAULT_WORLD_SIZE)`.
///
/// Parameterized over BOTH backends: the SQLite arm always; the Postgres arm
/// under `live-postgres-tests`, against `JAMMI_TEST_PG_URL` (this crate carries
/// no `test_case` dev-dependency, so the two arms are two named test fns over
/// one body). Job ids are unique per run:
/// the Postgres lane shares one database across the whole run.
#[tokio::test]
async fn get_job_for_rank_world_size_matches_the_real_training_spec_producer_sqlite() {
    parity_over(BackendKind::Sqlite).await;
}

#[cfg(feature = "live-postgres-tests")]
#[tokio::test]
async fn get_job_for_rank_world_size_matches_the_real_training_spec_producer_postgres() {
    parity_over(BackendKind::Postgres).await;
}

async fn parity_over(kind: BackendKind) {
    let dir = tempfile::tempdir().unwrap();
    let session = jammi_test_utils::make_test_session(kind, dir.path()).await;
    let catalog = Arc::clone(session.catalog());
    let suffix = jammi_test_utils::unique_suffix();
    let world_two_id = format!("job-parity-world-two-{suffix}");
    let world_absent_id = format!("job-parity-world-absent-{suffix}");
    let coord = format!("coord-parity-{suffix}");

    let world_two_json =
        serde_json::to_string(&fine_tune_spec_world_two()).expect("TrainingSpec serializes");
    catalog
        .submit_job(SubmitJobParams {
            job_id: &world_two_id,
            kind: "fine_tune",
            execution: JobExecution::Queued,
            spec: &world_two_json,
            model_ref: None,
            output_model_id: None,
            model_source: None,
            priority: 0,
        })
        .await
        .unwrap();
    catalog
        .claim_next(&coord, &["fine_tune"], Duration::from_secs(30))
        .await
        .unwrap()
        .expect("must claim the queued job");
    let row = catalog
        .get_job_for_rank(&world_two_id)
        .await
        .unwrap()
        .expect("the row exists");
    assert_eq!(
        row.world_size,
        WorldSizeFact::Decoded(2),
        "a real TrainingSpec::FineTune producer naming world_size = 2 must decode to \
         Decoded(2) through the SAME path the gang handler reads"
    );

    let absent_json =
        serde_json::to_string(&context_predictor_spec()).expect("TrainingSpec serializes");
    assert!(
        !absent_json.contains("world_size"),
        "the ContextPredictor producer must genuinely omit world_size, not merely \
         set it to a default value: {absent_json}"
    );
    catalog
        .submit_job(SubmitJobParams {
            job_id: &world_absent_id,
            kind: "context_predictor",
            execution: JobExecution::Queued,
            spec: &absent_json,
            model_ref: None,
            output_model_id: None,
            model_source: None,
            priority: 0,
        })
        .await
        .unwrap();
    catalog
        .claim_next(&coord, &["context_predictor"], Duration::from_secs(30))
        .await
        .unwrap()
        .expect("must claim the queued job");
    let row_absent = catalog
        .get_job_for_rank(&world_absent_id)
        .await
        .unwrap()
        .expect("the row exists");
    assert_eq!(
        row_absent.world_size,
        WorldSizeFact::Decoded(DEFAULT_WORLD_SIZE),
        "a real producer variant naming no world_size at all must decode to the \
         documented absent default"
    );
}
