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

/// Fallback handler for unmatched routes.
pub async fn fallback_handler() -> AppError {
    AppError {
        status: StatusCode::NOT_FOUND,
        error: "Not found".into(),
        code: "not_found".into(),
    }
}
