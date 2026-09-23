use serde::{Deserialize, Serialize};

use crate::error::Error;

/// The kind of device a plan runs on, discarding any ordinal — the
/// determinant an [`InferenceSpec`](crate::InferenceSpec) names and a
/// placement compares against. Ordinals are never compared: a plan built on
/// CUDA ordinal 0 runs on an executor whose only CUDA device is ordinal 1
/// (the ordinal is not output-affecting, and the wire form carries none).
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
    /// This kind's spelling on the wire and in a device inventory — the ONE
    /// mapping a registered device is read against, wherever a plan's
    /// required kind is matched to it.
    pub fn wire_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
        }
    }

    /// The inverse of [`Self::wire_str`]: the kind a token names, or `None`
    /// for a token no kind spells.
    pub fn parse(token: &str) -> Option<Self> {
        match token {
            "cpu" => Some(Self::Cpu),
            "cuda" => Some(Self::Cuda),
            "metal" => Some(Self::Metal),
            _ => None,
        }
    }
}

impl std::fmt::Display for ComputeDeviceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.wire_str())
    }
}

impl std::str::FromStr for ComputeDeviceKind {
    type Err = Error;

    fn from_str(token: &str) -> Result<Self, Self::Err> {
        Self::parse(token).ok_or_else(|| Error::UnknownDeviceKind(token.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_spelling_round_trips_every_kind() {
        for kind in [
            ComputeDeviceKind::Cpu,
            ComputeDeviceKind::Cuda,
            ComputeDeviceKind::Metal,
        ] {
            assert_eq!(ComputeDeviceKind::parse(kind.wire_str()), Some(kind));
            assert_eq!(kind.wire_str().parse::<ComputeDeviceKind>().unwrap(), kind);
            assert_eq!(kind.to_string(), kind.wire_str());
        }
        assert!(matches!(
            "tpu".parse::<ComputeDeviceKind>(),
            Err(Error::UnknownDeviceKind(token)) if token == "tpu"
        ));
    }
}
