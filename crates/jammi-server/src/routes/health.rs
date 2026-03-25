use axum::Json;
use serde_json::{json, Value};

/// GET /health — returns `{"status": "ok"}` without touching the database.
pub async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}
