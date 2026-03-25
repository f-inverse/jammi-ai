//! vLLM backend — managed process for high-throughput LLM inference.
//!
//! Spawns `vllm serve {model_id} --port {port}` as a child process,
//! performs a two-phase health check (poll `/health`, then verify `/v1/models`),
//! and records the PID for cleanup.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use jammi_engine::error::{JammiError, Result};
use tokio::process::{Child, Command};

/// A running vLLM server process.
pub struct VllmProcess {
    child: Child,
    pub port: u16,
    pub model_id: String,
    _pid_file: PathBuf,
}

impl Drop for VllmProcess {
    fn drop(&mut self) {
        // Best-effort kill on drop
        let _ = self.child.start_kill();
        let _ = std::fs::remove_file(&self._pid_file);
    }
}

/// vLLM backend that manages server process lifecycle.
pub struct VllmBackend {
    artifact_dir: PathBuf,
}

impl VllmBackend {
    pub fn new(artifact_dir: &Path) -> Self {
        Self {
            artifact_dir: artifact_dir.to_path_buf(),
        }
    }

    /// Check that the `vllm` binary is on PATH.
    pub fn check_vllm_installed() -> Result<()> {
        let output = std::process::Command::new("which")
            .arg("vllm")
            .output()
            .map_err(|e| JammiError::Backend(format!("Failed to check for vllm binary: {e}")))?;

        if !output.status.success() {
            return Err(JammiError::Backend(
                "vllm binary not found on PATH. Install with: pip install vllm".into(),
            ));
        }
        Ok(())
    }

    /// Start a vLLM server for the given model with default port (8000).
    pub async fn start_server(&self, model_id: &str) -> Result<VllmProcess> {
        self.start_server_with_config(model_id, 8000, Duration::from_secs(300), &[])
            .await
    }

    /// Start a vLLM server with explicit configuration.
    pub async fn start_server_with_config(
        &self,
        model_id: &str,
        port: u16,
        timeout: Duration,
        extra_args: &[String],
    ) -> Result<VllmProcess> {
        Self::check_vllm_installed()?;

        let mut cmd = Command::new("vllm");
        cmd.arg("serve")
            .arg(model_id)
            .arg("--port")
            .arg(port.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());

        for arg in extra_args {
            cmd.arg(arg);
        }

        let child = cmd
            .spawn()
            .map_err(|e| JammiError::Backend(format!("Failed to spawn vllm process: {e}")))?;

        let pid = child.id().unwrap_or(0);

        // Write PID file for crash recovery
        let pid_dir = self.artifact_dir.join("vllm");
        std::fs::create_dir_all(&pid_dir).ok();
        let pid_file = pid_dir.join(format!("{}.pid", model_id.replace('/', "_")));
        std::fs::write(&pid_file, pid.to_string()).ok();

        let process = VllmProcess {
            child,
            port,
            model_id: model_id.to_string(),
            _pid_file: pid_file,
        };

        // Two-phase health check
        Self::wait_for_health(port, model_id, timeout).await?;

        Ok(process)
    }

    /// Poll `/health` until the server responds with 200, then verify
    /// the model is listed in `/v1/models`.
    pub async fn wait_for_health(port: u16, model_id: &str, timeout: Duration) -> Result<()> {
        let client = reqwest::Client::new();
        let health_url = format!("http://127.0.0.1:{port}/health");
        let models_url = format!("http://127.0.0.1:{port}/v1/models");

        let deadline = tokio::time::Instant::now() + timeout;
        let poll_interval = Duration::from_secs(2);

        // Phase 1: wait for /health to return 200
        loop {
            if tokio::time::Instant::now() >= deadline {
                return Err(JammiError::Backend(format!(
                    "vLLM server for '{model_id}' failed to start within {}s (health check timeout on port {port})",
                    timeout.as_secs()
                )));
            }

            match client.get(&health_url).send().await {
                Ok(resp) if resp.status().is_success() => break,
                _ => tokio::time::sleep(poll_interval).await,
            }
        }

        // Phase 2: verify model is listed
        let resp = client
            .get(&models_url)
            .send()
            .await
            .map_err(|e| JammiError::Backend(format!("Failed to query /v1/models: {e}")))?;

        if !resp.status().is_success() {
            return Err(JammiError::Backend(format!(
                "vLLM /v1/models returned {}",
                resp.status()
            )));
        }

        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| JammiError::Backend(format!("Failed to parse /v1/models: {e}")))?;

        let models = body["data"]
            .as_array()
            .ok_or_else(|| JammiError::Backend("Invalid /v1/models response format".into()))?;

        let found = models.iter().any(|m| m["id"].as_str() == Some(model_id));

        if !found {
            return Err(JammiError::Backend(format!(
                "Model '{model_id}' not found in vLLM server's /v1/models response"
            )));
        }

        Ok(())
    }
}
