pub mod catalog;
pub mod config;
pub mod error;

use config::LoggingConfig;

pub fn init_tracing(config: &LoggingConfig) {
    use tracing_subscriber::{fmt, EnvFilter};

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.level));

    match config.format.as_str() {
        "json" => {
            fmt().with_env_filter(filter).json().init();
        }
        _ => {
            fmt().with_env_filter(filter).init();
        }
    }
}
