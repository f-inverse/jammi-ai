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
//! `serve` binds to — `[server] health_listen`, resolved through the
//! identical `--config` / `JAMMI_CONFIG` / `./jammi.toml` /
//! `/etc/jammi/jammi.toml` / platform-config-dir chain `serve` runs — so a
//! probe invoked without `--url` always targets exactly the address `serve`
//! would bind given the same inputs. When neither `--url` nor any config
//! source resolves to something other than the defaults, the probe targets
//! those same defaults, matching what a default `serve` invocation would
//! bind: that IS the parity this subcommand promises, not a guess about some
//! other deployment's port.

use std::time::Duration;

use jammi_db::config::ServerConfig;

/// Default timeout for a single probe request, in seconds.
pub const DEFAULT_TIMEOUT_SECS: u64 = 5;

/// The fixed readiness path every probe GETs.
const READYZ_PATH: &str = "/readyz";

/// The most a failure body is read and echoed to stderr — a misbehaving or
/// hostile endpoint must not let a probe invocation buffer an unbounded
/// response.
const MAX_BODY_BYTES: usize = 4 * 1024;

/// GET `url` once with `timeout` and classify the outcome: `Ok(())` iff the
/// response status is exactly `200`; `Err(message)` for every other status
/// (the message carries the status line and response body, the latter
/// bounded to `MAX_BODY_BYTES`) and for any transport failure (connection
/// refused, DNS failure, timeout, a redirect — the message carries the
/// underlying error text). The caller (`main`) turns this into the process
/// exit code: `0` for `Ok`, `1` for `Err`, printing the message to stderr in
/// the failure case.
///
/// Redirects are never followed: a `3xx` is a failure like any other
/// non-`200` status, since a probe that silently follows a redirect to a
/// healthy endpoint would report the wrong service as ready.
pub async fn check(url: &str, timeout: Duration) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .timeout(timeout)
        .redirect(reqwest::redirect::Policy::none())
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
    let body = match response.bytes().await {
        Ok(bytes) => truncated_body(&bytes),
        Err(e) => format!("<failed to read response body: {e}>"),
    };
    Err(format!("{url} returned {status}: {body}"))
}

/// Decode `bytes` as UTF-8 (lossily — a probe target's failure body is
/// diagnostic text, not a contract), bounded to [`MAX_BODY_BYTES`] with a
/// truncation marker appended when the body exceeds it.
fn truncated_body(bytes: &[u8]) -> String {
    if bytes.len() <= MAX_BODY_BYTES {
        String::from_utf8_lossy(bytes).into_owned()
    } else {
        let mut text = String::from_utf8_lossy(&bytes[..MAX_BODY_BYTES]).into_owned();
        text.push_str("... [truncated]");
        text
    }
}

/// Build the default probe URL from a resolved [`ServerConfig`]'s
/// `health_listen` bind address, the same value `serve` binds its health
/// listener to. A wildcard bind host is rewritten to its loopback
/// equivalent — a probe dialing `0.0.0.0` or `[::]` would connect to
/// nothing, since those are bind-only addresses, never connect targets:
/// `0.0.0.0` → `127.0.0.1`, `::` → `::1`. Any other host (an explicit
/// loopback or routable address) passes through unchanged. Appends the
/// fixed `/readyz` path.
///
/// Fails closed, in the same "Invalid health_listen address" shape
/// `ServerConfig::validate` uses, when `health_listen` is not a parseable
/// `SocketAddr` — a probe must not dial a value it cannot understand.
pub fn default_url(server: &ServerConfig) -> Result<String, String> {
    Ok(format!(
        "http://{}{READYZ_PATH}",
        rewrite_wildcard_bind(&server.health_listen)?
    ))
}

/// Rewrite an unspecified (wildcard) bind host in a `host:port` /
/// `[host]:port` string to its loopback equivalent, leaving the port and
/// every other host form untouched. Fails closed — the same
/// "Invalid health_listen address" message shape
/// `crates/jammi-db/src/config/mod.rs`'s `ServerConfig::validate` uses —
/// when the input is not a `SocketAddr` this process can parse, rather than
/// passing an unparseable value through to a confusing connection error.
fn rewrite_wildcard_bind(bind_addr: &str) -> Result<String, String> {
    use std::net::SocketAddr;

    match bind_addr.parse::<SocketAddr>() {
        Ok(SocketAddr::V4(addr)) if addr.ip().is_unspecified() => {
            Ok(format!("127.0.0.1:{}", addr.port()))
        }
        Ok(SocketAddr::V6(addr)) if addr.ip().is_unspecified() => {
            Ok(format!("[::1]:{}", addr.port()))
        }
        Ok(addr) => Ok(addr.to_string()),
        Err(e) => Err(format!("Invalid health_listen address '{bind_addr}': {e}")),
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
        assert_eq!(
            default_url(&ipv4).expect("valid health_listen"),
            "http://127.0.0.1:8080/readyz"
        );

        let ipv6 = ServerConfig {
            health_listen: "[::]:8080".into(),
            ..ServerConfig::default()
        };
        assert_eq!(
            default_url(&ipv6).expect("valid health_listen"),
            "http://[::1]:8080/readyz"
        );
    }

    #[test]
    fn default_probe_url_passes_through_a_concrete_host() {
        let server = ServerConfig {
            health_listen: "127.0.0.1:9090".into(),
            ..ServerConfig::default()
        };
        assert_eq!(
            default_url(&server).expect("valid health_listen"),
            "http://127.0.0.1:9090/readyz"
        );
    }

    #[test]
    fn default_probe_url_fails_closed_on_an_unparseable_health_listen() {
        let server = ServerConfig {
            health_listen: "not-an-address".into(),
            ..ServerConfig::default()
        };
        let err = default_url(&server)
            .expect_err("an unparseable health_listen must fail closed, not be dialed as-is");
        assert!(
            err.starts_with("Invalid health_listen address 'not-an-address': "),
            "expected the same message shape ServerConfig::validate uses, got: {err}"
        );
    }

    #[test]
    fn truncated_body_marks_truncation_past_the_bound() {
        let short = b"ready check failed";
        assert_eq!(truncated_body(short), "ready check failed");

        let long = vec![b'x'; MAX_BODY_BYTES + 100];
        let out = truncated_body(&long);
        assert!(out.ends_with("... [truncated]"));
        assert!(out.len() < long.len());
    }
}
