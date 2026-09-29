//! The train-run rung on a fleet: `shape-d`, the job submitted through the
//! deployed topology's query tier over the public surface and claimed by a
//! compute process. It hands back the same [`TrainedRun`] the in-process
//! rungs do, with where the run ran proven from the catalog and the fleet's
//! logs.

use jammi_ai::fine_tune::training_job::fine_tuned_model_id;
use jammi_ai::fine_tune::FineTuneMethod;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::CachePolicy;
use jammi_wire::request::FineTuneRequest;

use super::running::RunningFleet;
use crate::finetune_run::{
    published_run, write_training_source, FinetuneRunParams, RunContext, Rung, TrainedRun,
    STREAMED_SOURCE,
};

/// The `shape-d` rung of `params`.
pub fn train_on_fleet(
    params: &FinetuneRunParams,
    ctx: &RunContext,
) -> Result<TrainedRun, Box<dyn std::error::Error + Send + Sync>> {
    tokio::runtime::Handle::current().block_on(async {
        match params.rung {
            Rung::ShapeD => train_shape_d(params, ctx).await,
            Rung::Resident | Rung::Streamed => Err(format!(
                "the {} rung runs in this process, not on a fleet",
                params.rung.as_str()
            )
            .into()),
        }
    })
}

/// The training rows as a JSONL source, put in `fleet`'s shared store.
async fn training_source(
    fleet: &RunningFleet,
    ctx: &RunContext,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let path = ctx.work_dir.join("training_source.jsonl");
    write_training_source(&ctx.train_rows, &path)?;
    fleet.publish_input(&path).await
}

/// `shape-d`: submitted through the query tier's public surface — the
/// source registered and the job submitted over gRPC, as a user's would be
/// — claimed and trained by a compute process, never the query process.
async fn train_shape_d(
    params: &FinetuneRunParams,
    ctx: &RunContext,
) -> Result<TrainedRun, Box<dyn std::error::Error + Send + Sync>> {
    let device = params.cuda_device.map_or(-1, |o| o as i32);
    let leg = format!("train-run-shape-d-seed{}", params.seed);
    let mut fleet = match &params.fleet.query_addr {
        Some(query_addr) => RunningFleet::join_shape_d(query_addr, &leg).await?,
        None => RunningFleet::spawn_shape_d(&params.fleet, &leg, device).await?,
    };
    let source_url = training_source(&fleet, ctx).await?;
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
