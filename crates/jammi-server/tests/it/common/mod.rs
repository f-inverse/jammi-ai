use std::path::Path;
use std::sync::Arc;

use jammi_ai::session::InferenceSession;
use jammi_engine::config::JammiConfig;
use jammi_server::state::AppState;

/// Create an AppState backed by a temp directory for testing.
pub async fn test_app_state(artifact_dir: &Path) -> Arc<AppState> {
    let config = JammiConfig {
        artifact_dir: artifact_dir.to_path_buf(),
        gpu: jammi_engine::config::GpuConfig {
            device: -1,
            ..Default::default()
        },
        inference: jammi_engine::config::InferenceConfig {
            batch_size: 8,
            ..Default::default()
        },
        logging: jammi_engine::config::LoggingConfig {
            level: "warn".into(),
            ..Default::default()
        },
        ..Default::default()
    };
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    Arc::new(AppState { session })
}
