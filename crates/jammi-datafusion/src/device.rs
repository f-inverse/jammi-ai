use serde::{Deserialize, Serialize};

/// The kind of device a stage runs on, discarding any ordinal — the kind a
/// process's device inventory lists each of its devices under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeDeviceKind {
    /// CPU.
    Cpu,
    /// A CUDA device, any ordinal.
    Cuda,
    /// An Apple Metal device, any ordinal.
    Metal,
}

impl ComputeDeviceKind {
    /// This kind's spelling on the wire and in a device inventory.
    pub fn wire_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
        }
    }
}

impl std::fmt::Display for ComputeDeviceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.wire_str())
    }
}
