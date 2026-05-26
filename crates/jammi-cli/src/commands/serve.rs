use jammi_db::config::JammiConfig;
use jammi_server::runtime::OssServer;

/// `jammi serve` — delegate to the OSS server's runtime. The CLI keeps
/// the friendly subcommand surface; the server orchestration lives in
/// `jammi-server::runtime` so the `jammi-server` binary and the CLI
/// share one entry-point.
pub async fn run(config: JammiConfig) -> Result<(), Box<dyn std::error::Error>> {
    OssServer::new(config).await?.run().await?;
    Ok(())
}
