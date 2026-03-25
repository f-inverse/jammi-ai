use std::sync::Arc;

use jammi_ai::session::InferenceSession;
use jammi_engine::config::JammiConfig;

pub async fn run(config: JammiConfig) -> Result<(), Box<dyn std::error::Error>> {
    let health_addr = config.server.health_listen.parse()?;
    let flight_addr = config.server.flight_listen.parse()?;

    let session = Arc::new(InferenceSession::new(config).await?);
    let ctx = session.context().clone();

    // Keep session alive for the Flight SQL server's lifetime.
    let _session = session;

    tokio::try_join!(
        async {
            jammi_server::serve(health_addr)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
        },
        async { jammi_server::flight::serve_flight(&ctx, flight_addr).await },
    )?;

    Ok(())
}
