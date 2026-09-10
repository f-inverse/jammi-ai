//! A generic HTTP readiness probe for `jammi-server serve`.
//!
//! This is the same shape as Postgres's `pg_isready`: a small, dependency-light
//! CLI check a process supervisor runs out-of-band to decide whether an
//! already-running server is ready to take traffic. It is NOT a health-check
//! *protocol* — it is one GET against the server's own `/readyz` endpoint,
//! translated into a process exit code a shell script can branch on.
//!
//! Any supervisor that drives readiness from a subprocess exit code can use
//! this directly: systemd's `ExecStartPost=`, Nomad's `check { type = "script"
//! }`, and Docker/Compose's `HEALTHCHECK` (`healthcheck.test`) all invoke an
//! arbitrary command and look only at whether it exited zero. Kubernetes is
//! the one platform that needs nothing extra: its `readinessProbe.httpGet`
//! already speaks HTTP directly against `/readyz`, so this subcommand adds no
//! value there — orchestrating *which* platform runs the probe, and how, is
//! the deployer's concern, not the engine's (the engine ships the mechanism,
//! not the policy).
//!
//! The default probe target is derived from the SAME resolved configuration
//! `serve` binds to — `[server] health_listen` — so a probe invoked without
//! `--url` always asks the question "is the server *this process* would have
//! started actually ready", never a guess about some other deployment's port.

use std::time::Duration;

use jammi_db::config::ServerConfig;

/// Default timeout for a single probe request, in seconds.
pub const DEFAULT_TIMEOUT_SECS: u64 = 5;

/// The fixed readiness path every probe GETs.
const READYZ_PATH: &str = "/readyz";

/// GET `url` once with `timeout` and classify the outcome: `Ok(())` iff the
/// response status is exactly `200`; `Err(message)` for every other status
/// (the message carries the status line and response body) and for any
/// transport failure (connection refused, DNS failure, timeout — the
/// message carries the underlying error text). The caller (`main`) turns
/// this into the process exit code: `0` for `Ok`, `1` for `Err`, printing
/// the message to stderr in the failure case.
pub async fn check(url: &str, timeout: Duration) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|e| format!("failed to build HTTP client: {e}"))?;
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("request to {url} failed: {e}"))?;
    let status = response.status();
    if status.as_u16() == 200 {
        return Ok(());
    }
    let body = response.text().await.unwrap_or_default();
    Err(format!("{url} returned {status}: {body}"))
}

/// Build the default probe URL from a resolved [`ServerConfig`]'s
/// `health_listen` bind address, the same value `serve` binds its health
/// listener to. A wildcard bind host is rewritten to its loopback
/// equivalent — a probe dialing `0.0.0.0` or `[::]` would connect to
/// nothing, since those are bind-only addresses, never connect targets:
/// `0.0.0.0` → `127.0.0.1`, `::` → `::1`. Any other host (an explicit
/// loopback or routable address) passes through unchanged. Appends the
/// fixed `/readyz` path.
pub fn default_url(server: &ServerConfig) -> String {
    format!(
        "http://{}{READYZ_PATH}",
        rewrite_wildcard_bind(&server.health_listen)
    )
}

/// Rewrite an unspecified (wildcard) bind host in a `host:port` /
/// `[host]:port` string to its loopback equivalent, leaving the port and
/// every other host form untouched. Falls back to the input unchanged
/// when it is not a `SocketAddr` this process can parse (the probe still
/// tries to dial it — a clearer connection error beats silently
/// swallowing an unparseable configured value).
fn rewrite_wildcard_bind(bind_addr: &str) -> String {
    use std::net::SocketAddr;

    match bind_addr.parse::<SocketAddr>() {
        Ok(SocketAddr::V4(addr)) if addr.ip().is_unspecified() => {
            format!("127.0.0.1:{}", addr.port())
        }
        Ok(SocketAddr::V6(addr)) if addr.ip().is_unspecified() => {
            format!("[::1]:{}", addr.port())
        }
        Ok(addr) => addr.to_string(),
        Err(_) => bind_addr.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_probe_url_rewrites_wildcard_bind() {
        let ipv4 = ServerConfig {
            health_listen: "0.0.0.0:8080".into(),
            ..ServerConfig::default()
        };
        assert_eq!(default_url(&ipv4), "http://127.0.0.1:8080/readyz");

        let ipv6 = ServerConfig {
            health_listen: "[::]:8080".into(),
            ..ServerConfig::default()
        };
        assert_eq!(default_url(&ipv6), "http://[::1]:8080/readyz");
    }

    #[test]
    fn default_probe_url_passes_through_a_concrete_host() {
        let server = ServerConfig {
            health_listen: "127.0.0.1:9090".into(),
            ..ServerConfig::default()
        };
        assert_eq!(default_url(&server), "http://127.0.0.1:9090/readyz");
    }
}
