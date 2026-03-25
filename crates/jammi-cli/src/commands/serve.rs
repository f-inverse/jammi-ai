use std::sync::Arc;

use jammi_ai::session::InferenceSession;
use jammi_engine::config::JammiConfig;
use jammi_server::state::AppState;

pub async fn run(config: JammiConfig) -> Result<(), Box<dyn std::error::Error>> {
    let listen_addr = config.server.listen.parse()?;

    let session = Arc::new(InferenceSession::new(config).await?);
    let state = Arc::new(AppState { session });

    jammi_server::serve(state, listen_addr).await?;
    Ok(())
}
