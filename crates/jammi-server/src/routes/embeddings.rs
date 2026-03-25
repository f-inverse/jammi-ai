use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use serde::Deserialize;

use crate::error::AppError;
use crate::state::AppState;

#[derive(Deserialize)]
pub struct GenerateEmbeddingsRequest {
    pub source_id: String,
    pub model_id: String,
    pub columns: Vec<String>,
    pub key_column: String,
}

/// POST /embeddings/generate — generate embeddings for a source.
pub async fn generate_embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateEmbeddingsRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let result = state
        .session()
        .generate_embeddings(&req.source_id, &req.model_id, &req.columns, &req.key_column)
        .await?;
    Ok(Json(serde_json::json!({
        "table_name": result.table_name,
        "status": result.status,
        "row_count": result.row_count,
    })))
}
