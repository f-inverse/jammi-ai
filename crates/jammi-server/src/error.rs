use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

/// JSON error response: `{ "error": "<string>", "code": "<string>" }`.
/// Both fields are always present and non-null.
#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: String,
}

/// Application error that converts into an HTTP response.
pub struct AppError {
    pub status: StatusCode,
    pub error: String,
    pub code: String,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = ErrorResponse {
            error: self.error,
            code: self.code,
        };
        (self.status, Json(body)).into_response()
    }
}

impl From<jammi_engine::error::JammiError> for AppError {
    fn from(err: jammi_engine::error::JammiError) -> Self {
        let message = err.to_string();
        let code = match &err {
            jammi_engine::error::JammiError::Config(_) => "config_error",
            jammi_engine::error::JammiError::Source { .. } => "source_error",
            jammi_engine::error::JammiError::Model { .. } => "model_error",
            jammi_engine::error::JammiError::Inference(_) => "inference_error",
            jammi_engine::error::JammiError::FineTune(_) => "fine_tune_error",
            jammi_engine::error::JammiError::Catalog(_) => "catalog_error",
            _ => "internal_error",
        };
        AppError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            error: message,
            code: code.into(),
        }
    }
}

/// Fallback handler for unmatched routes.
pub async fn fallback_handler() -> AppError {
    AppError {
        status: StatusCode::NOT_FOUND,
        error: "Not found".into(),
        code: "not_found".into(),
    }
}
