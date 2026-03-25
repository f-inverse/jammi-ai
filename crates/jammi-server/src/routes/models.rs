use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use serde::Deserialize;

use jammi_ai::model::{ModelSource, ModelTask};

use crate::error::AppError;
use crate::state::AppState;

/// GET /models — list registered models.
pub async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, AppError> {
    let models = state.session.catalog().list_models()?;
    Ok(Json(serde_json::json!({ "models": models })))
}

#[derive(Deserialize)]
pub struct PreloadRequest {
    pub model_id: String,
    #[serde(default = "default_task")]
    pub task: String,
}

fn default_task() -> String {
    "embedding".into()
}

/// POST /models/preload — preload a model into the cache.
pub async fn preload_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PreloadRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let source = ModelSource::parse(&req.model_id);
    let task = match req.task.as_str() {
        "embedding" => ModelTask::Embedding,
        "classification" => ModelTask::Classification,
        "summarization" => ModelTask::Summarization,
        "text_generation" => ModelTask::TextGeneration,
        other => {
            return Err(AppError {
                status: axum::http::StatusCode::BAD_REQUEST,
                error: format!("Unknown task: {other}"),
                code: "invalid_task".into(),
            });
        }
    };

    state
        .session
        .model_cache()
        .preload(&source, task, None)
        .await?;
    Ok(Json(
        serde_json::json!({ "model_id": req.model_id, "status": "loaded" }),
    ))
}
