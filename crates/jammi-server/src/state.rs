use std::sync::Arc;

use jammi_ai::session::InferenceSession;

/// Shared application state for all HTTP handlers.
pub struct AppState {
    pub session: Arc<InferenceSession>,
}
