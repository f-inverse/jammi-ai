use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use serde::Deserialize;

use jammi_ai::fine_tune::FineTuneMethod;

use crate::error::AppError;
use crate::state::AppState;

#[derive(Deserialize)]
pub struct FineTuneRequest {
    pub source_id: String,
    pub base_model: String,
    pub columns: Vec<String>,
    #[serde(default = "default_method")]
    pub method: FineTuneMethod,
    #[serde(default)]
    pub task: String,
}

fn default_method() -> FineTuneMethod {
    FineTuneMethod::Lora
}

/// POST /fine-tune — start a fine-tuning job.
pub async fn start_fine_tune(
    State(state): State<Arc<AppState>>,
    Json(req): Json<FineTuneRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let job = state
        .session()
        .fine_tune(
            &req.source_id,
            &req.base_model,
            &req.columns,
            req.method,
            &req.task,
            None,
        )
        .await?;
    Ok(Json(serde_json::json!({
        "job_id": job.job_id,
        "status": job.status,
    })))
}

/// GET /fine-tune — list fine-tuning jobs.
pub async fn list_fine_tune_jobs(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, AppError> {
    let jobs = state.session().catalog().list_fine_tune_jobs()?;
    Ok(Json(serde_json::json!({ "jobs": jobs })))
}
