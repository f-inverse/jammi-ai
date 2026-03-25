use std::sync::Arc;

use jammi_ai::session::InferenceSession;

/// Shared application state for all HTTP handlers.
pub struct AppState {
    session: Arc<InferenceSession>,
}

impl AppState {
    pub fn new(session: Arc<InferenceSession>) -> Self {
        Self { session }
    }

    pub fn session(&self) -> &InferenceSession {
        &self.session
    }
}
