use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;

use jammi_engine::source::{SourceConnection, SourceType};

use crate::error::AppError;
use crate::state::AppState;

#[derive(Deserialize)]
pub struct AddSourceRequest {
    pub source_id: String,
    pub source_type: SourceType,
    #[serde(flatten)]
    pub connection: SourceConnection,
}

/// POST /sources — register a new data source.
pub async fn add_source(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AddSourceRequest>,
) -> Result<impl IntoResponse, AppError> {
    state
        .session
        .add_source(&req.source_id, req.source_type, req.connection)
        .await?;
    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({ "source_id": req.source_id })),
    ))
}

/// GET /sources — list registered sources.
pub async fn list_sources(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, AppError> {
    let sources = state.session.catalog().list_sources()?;
    Ok(Json(serde_json::json!({ "sources": sources })))
}

/// DELETE /sources/:id — remove a source.
pub async fn remove_source(
    State(state): State<Arc<AppState>>,
    Path(source_id): Path<String>,
) -> Result<impl IntoResponse, AppError> {
    state.session.catalog().remove_source(&source_id)?;
    Ok(StatusCode::NO_CONTENT)
}
