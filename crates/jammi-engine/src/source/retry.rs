use std::future::Future;
use std::time::Duration;

use backon::{ExponentialBuilder, Retryable};

use crate::error::{JammiError, Result};

/// Configuration for exponential backoff retry.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Total number of attempts (1 initial + retries). Default: 4.
    pub max_attempts: usize,
    /// Delay before the first retry. Default: 1s.
    pub initial_backoff: Duration,
    /// Upper bound on backoff duration. Default: 30s.
    pub max_backoff: Duration,
    /// Exponential growth factor. Default: 2.0.
    pub backoff_multiplier: f32,
    /// Add random jitter to delays. Default: true.
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 4,
            initial_backoff: Duration::from_secs(1),
            max_backoff: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

impl RetryConfig {
    pub fn validate(&self) -> Result<()> {
        if self.max_attempts == 0 {
            return Err(JammiError::Config("max_attempts must be > 0".into()));
        }
        if self.backoff_multiplier < 1.0 {
            return Err(JammiError::Config(
                "backoff_multiplier must be >= 1.0".into(),
            ));
        }
        Ok(())
    }
}

fn build_backoff(config: &RetryConfig) -> ExponentialBuilder {
    let mut backoff = ExponentialBuilder::default()
        .with_min_delay(config.initial_backoff)
        .with_max_delay(config.max_backoff)
        .with_factor(config.backoff_multiplier)
        .with_max_times(config.max_attempts.saturating_sub(1));
    if config.jitter {
        backoff = backoff.with_jitter();
    }
    backoff
}

/// Retry a fallible async operation with exponential backoff.
/// Error messages include the source name for debuggability.
pub async fn retry_with_backoff<F, Fut, T>(
    source_name: &str,
    config: &RetryConfig,
    operation: F,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    config.validate()?;
    let backoff = build_backoff(config);
    let source_name = source_name.to_string();
    let max_attempts = config.max_attempts;

    operation
        .retry(backoff)
        .await
        .map_err(|e| JammiError::Source {
            source_id: source_name,
            message: format!("Failed after {max_attempts} attempts: {e}"),
        })
}

/// Classify whether an error is transient (retryable).
/// 5xx, 429, connection refused, timeout → retry.
/// Other 4xx, auth failures → fail immediately.
pub fn is_retryable(error: &JammiError) -> bool {
    match error {
        JammiError::Source { message, .. } | JammiError::Backend(message) => {
            message.contains("500")
                || message.contains("502")
                || message.contains("503")
                || message.contains("504")
                || message.contains("429")
                || message.contains("connection refused")
                || message.contains("timeout")
        }
        JammiError::Io(_) => true,
        _ => false,
    }
}

/// Retry with transient-error classification. Non-retryable errors fail immediately.
pub async fn retry_transient<F, Fut, T>(
    source_name: &str,
    config: &RetryConfig,
    mut operation: F,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    config.validate()?;
    let backoff = build_backoff(config);
    let source_name = source_name.to_string();
    let max_attempts = config.max_attempts;

    (|| operation())
        .retry(backoff)
        .when(is_retryable)
        .await
        .map_err(|e| JammiError::Source {
            source_id: source_name,
            message: format!("Failed after {max_attempts} attempts: {e}"),
        })
}
