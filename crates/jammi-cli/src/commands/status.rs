//! `jammi status` subcommand.
//!
//! Reports the server's capabilities handshake — version, compiled feature
//! flags, addressable storage backends, and mounted gRPC service tiers — by
//! reading [`Session::server_info`]. Reachability is implicit: the RPC
//! succeeding means the server answered, so there is no separate ping.

use jammi_ai::Session;

pub async fn run(session: &Session) -> Result<(), Box<dyn std::error::Error>> {
    let info = session.server_info().await?;
    println!("version:          {}", info.version);
    println!("features:         {}", join_or_dash(&info.features));
    println!("storage_backends: {}", join_or_dash(&info.storage_backends));
    println!("services:         {}", join_or_dash(&info.services));
    Ok(())
}

/// Render a capability list as a comma-separated line, or `—` when empty, so an
/// embedded-shaped server with no mounted tiers prints a readable placeholder
/// rather than a blank.
fn join_or_dash(values: &[String]) -> String {
    if values.is_empty() {
        "—".to_string()
    } else {
        values.join(", ")
    }
}
