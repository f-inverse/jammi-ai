use jammi_db::config::JammiConfig;
use jammi_server::runtime::OssServer;
use jammi_server::telemetry::init_tracing;

/// `jammi serve` — delegate to the OSS server's runtime. The CLI keeps
/// the friendly subcommand surface; the server orchestration lives in
/// `jammi-server::runtime` so the `jammi-server` binary and the CLI
/// share one entry-point.
///
/// Installs the global tracing subscriber from `[logging]` before the
/// server starts, so `jammi serve` emits the same device-selection and
/// request logs as the standalone `jammi-server` binary — including when
/// stdout is redirected to a file (containers, systemd) rather than a
/// terminal.
pub async fn run(config: JammiConfig) -> Result<(), Box<dyn std::error::Error>> {
    init_tracing(&config);
    OssServer::new(config).await?.run().await?;
    Ok(())
}
