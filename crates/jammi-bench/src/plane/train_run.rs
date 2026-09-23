//! The train-run rungs on a fleet: `placed` (the job claimed by one process
//! and placed on an executor in another) and `shape-d` (the job submitted
//! through the deployed topology's query tier over the public surface and
//! claimed by a compute process). Both hand back the same
//! [`TrainedRun`] the in-process rungs do, with where the run ran proven
//! from the catalog and the fleet's logs.

use std::sync::Arc;

use jammi_ai::fine_tune::spec::{TrainingCommon, TrainingSpec, DEFAULT_WORLD_SIZE};
use jammi_ai::fine_tune::training_job::fine_tuned_model_id;
use jammi_ai::fine_tune::FineTuneMethod;
use jammi_ai::session::InferenceSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::CachePolicy;
use jammi_wire::request::FineTuneRequest;

use super::fleet::RunningFleet;
use crate::finetune_run::{
    published_run, write_training_source, FinetuneRunParams, RunContext, Rung, TrainedRun,
    STREAMED_SOURCE,
};

/// The `placed` or `shape-d` rung of `params`.
pub fn train_on_fleet(
    params: &FinetuneRunParams,
    ctx: &RunContext,
) -> Result<TrainedRun, Box<dyn std::error::Error + Send + Sync>> {
    tokio::runtime::Handle::current().block_on(async {
        match params.rung {
            Rung::Placed => train_placed(params, ctx).await,
            Rung::ShapeD => train_shape_d(params, ctx).await,
            Rung::Resident | Rung::Streamed => Err(format!(
                "the {} rung runs in this process, not on a fleet",
                params.rung.as_str()
            )
            .into()),
        }
    })
}

/// The training rows as a file source every member on this host reads.
fn local_training_source(
    ctx: &RunContext,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let path = ctx.work_dir.join("training_source.jsonl");
    write_training_source(&ctx.train_rows, &path)?;
    Ok(RunningFleet::local_url(&path))
}

fn training_spec(params: &FinetuneRunParams, ctx: &RunContext, source: &str) -> TrainingSpec {
    TrainingSpec::FineTune {
        source: source.to_string(),
        columns: params.objective.columns(),
        method: FineTuneMethod::Lora,
        task: params.task.model_task(),
        common: TrainingCommon {
            base_model: format!("local:{}", params.model_dir.display()),
            config: ctx.config.clone(),
            world_size: DEFAULT_WORLD_SIZE,
            cache: CachePolicy::Bypass,
        },
    }
}

async fn register_source(
    session: &Arc<InferenceSession>,
    url: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let name = format!("{STREAMED_SOURCE}_{}", crate::capture::unique_suffix());
    session
        .add_source(
            &name,
            SourceType::File,
            SourceConnection {
                url: Some(url.to_string()),
                format: Some(FileFormat::JsonLines),
                ..Default::default()
            },
        )
        .await?;
    Ok(name)
}

/// `placed`: submitted into the shared catalog, claimed by the submitter
/// process, placed on the executor process — proven by the claim's
/// transfer, the submitter's hand-off line and the scheduler's binding.
async fn train_placed(
    params: &FinetuneRunParams,
    ctx: &RunContext,
) -> Result<TrainedRun, Box<dyn std::error::Error + Send + Sync>> {
    let device = params.cuda_device.map_or(-1, |o| o as i32);
    let leg = format!("train-run-placed-seed{}", params.seed);
    let mut fleet = RunningFleet::spawn_placed(&params.plane, &leg, device, &["fine_tune"]).await?;
    let source_url = local_training_source(ctx)?;
    let source = register_source(&fleet.session, &source_url).await?;
    let job = fleet
        .session
        .run_training_spec(training_spec(params, ctx, &source))
        .await?;
    let job_id = job.job_id.clone();
    let (_, ran_on) = fleet.placed_training_ran_on(&job_id).await?;
    let mut trained = published_run(&fleet.session, &job_id, &job.model_id).await?;
    trained.ran_on = ran_on;
    Ok(trained)
}

/// `shape-d`: submitted through the query tier's public surface — the
/// source registered and the job submitted over gRPC, as a user's would be
/// — claimed and trained by a compute process, never the query or
/// scheduler process.
async fn train_shape_d(
    params: &FinetuneRunParams,
    ctx: &RunContext,
) -> Result<TrainedRun, Box<dyn std::error::Error + Send + Sync>> {
    let device = params.cuda_device.map_or(-1, |o| o as i32);
    let leg = format!("train-run-shape-d-seed{}", params.seed);
    let mut fleet = match &params.plane.query_addr {
        Some(query_addr) => RunningFleet::join_shape_d(query_addr, &leg).await?,
        None => RunningFleet::spawn_shape_d(&params.plane, &leg, device).await?,
    };
    let source_url = match &params.plane.source_url {
        Some(url) => url.clone(),
        None => local_training_source(ctx)?,
    };
    let query_addr = fleet
        .query_addr
        .clone()
        .ok_or("the shape-d fleet has no query tier")?;
    let endpoint = tonic::transport::Endpoint::from_shared(format!("http://{query_addr}"))?;
    let admin = jammi_admin::CatalogClient::connect(endpoint.clone()).await?;
    let source = format!("{STREAMED_SOURCE}_{}", crate::capture::unique_suffix());
    admin
        .add_source(
            &source,
            SourceType::File,
            SourceConnection {
                url: Some(source_url),
                format: Some(FileFormat::JsonLines),
                ..Default::default()
            },
        )
        .await?;
    let client = jammi_client::DataClient::connect(endpoint).await?;
    let job_id = client
        .submit_fine_tune(FineTuneRequest {
            source,
            base_model: format!("local:{}", params.model_dir.display()),
            columns: params.objective.columns(),
            method: FineTuneMethod::Lora,
            task: params.task.model_task(),
            config: Some(ctx.config.clone()),
            world_size: None,
            cache: CachePolicy::Bypass,
        })
        .await?
        .0;
    let model_id = fine_tuned_model_id(&job_id);
    let (_, ran_on) = fleet.shape_d_training_ran_on(&job_id).await?;
    let mut trained = published_run(&fleet.session, &job_id, &model_id).await?;
    trained.ran_on = ran_on;
    Ok(trained)
}
