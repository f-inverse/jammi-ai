//! LoRA A/B initialisation strategies.

use serde::{Deserialize, Serialize};

/// How the LoRA A and B matrices are initialised at construction time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoraInitMode {
    /// A: Kaiming-uniform. B: zeros. Layer is identity at construction — the
    /// adapter contributes nothing until training updates B away from zero.
    #[default]
    ZerosB,
    /// Both A and B drawn from `Normal(0, 0.02)`. Layer is non-identity at
    /// construction, so the adapter perturbs outputs from step zero.
    Gaussian,
}
