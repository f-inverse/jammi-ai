//! The GGUF weight-STORAGE-format vocabulary: a candle-free enum naming the
//! quantized weight `dtype`s this workspace can load a `QTensor` from.
//!
//! [`WeightQuantization`] is a PEER of [`crate::ComputePrecision`], never a
//! variant folded into it: `ComputePrecision` names the dtype activations
//! and *unquantized* weights/matmuls run at (`f32`/`f16`/`bf16` — a compute
//! dtype every backend can materialize a dense `Tensor` in);
//! `WeightQuantization` names how a weight's BYTES are packed at rest on
//! disk (GGUF's k-quant block formats — `q4_0` through `q6_k`), a storage
//! concern orthogonal to compute dtype: a `q4_0` weight is always
//! dequantized to `f32` (or narrower) before any matmul touches it — see
//! `jammi_lora::QuantizedLinear`'s "uniform F32 activation rule" — so the
//! two vocabularies compose (a quantized weight, a compute precision for
//! its activations) rather than one subsuming the other. It lives here, not
//! in `jammi-kernels` or `jammi-lora` (both of which carry the `candle`
//! quantized tensor stack), for the same reason `ComputePrecision` does: a
//! crate that only needs to NAME a storage format — config plumbing,
//! `jammi-wire`, `jammi-db` — never pulls a tensor library to do so. The
//! `WeightQuantization -> candle_core::quantized::GgmlDType` conversion is a
//! plain function at the candle boundary, never a `From` impl here (neither
//! type is local to this crate — the orphan rule forbids it, mirroring
//! `ComputePrecision`'s own module doc on this point).
//!
//! # Wire order (family J: determinism needs a stable tie-break key)
//!
//! [`WeightQuantization::gguf_wire_id`] returns the GGML wire ID GGUF itself
//! assigns each dtype (`ggml.h`'s `enum ggml_type`, mirrored by candle's own
//! `GgmlDType::to_u32`/`from_u32` in `candle-core` 0.11.0
//! `src/quantized/mod.rs:299-339` — verified against that table directly,
//! not re-derived: `q4_0=2, q4_1=3, q5_0=6, q5_1=7, q8_0=8, q2k=10, q3k=11,
//! q4k=12, q5k=13, q6k=14`; candle's own enum DECLARATION order interleaves
//! `Q8_1`/`Q8K` — dtypes this crate does not name at all, since no k-quant
//! consumer in this workspace loads them yet — between the ones this crate
//! covers, so wire ID and this crate's variant-declaration order coincide
//! only because the variants below are listed in ascending wire-ID order on
//! purpose, not by construction). [`Ord`]/[`PartialOrd`] are a MANUAL impl
//! keyed on `gguf_wire_id()` rather than a bare `#[derive(PartialOrd, Ord)]`
//! over declaration order: a derive would happen to agree today (the
//! variants below ARE declared in ascending wire-ID order) but would
//! silently decouple from the wire ID — the actual, externally-meaningful
//! total order a GGUF-adjacent consumer (a sorted manifest, a deterministic
//! iteration over a weight-quantization map) needs to reproduce — the
//! moment a future edit reordered the variant list for readability. Keying
//! `Ord` on the table directly makes that coupling explicit and immune to
//! reordering.

use std::cmp::Ordering;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::NumericsError;

/// The GGUF/GGML k-quant weight storage format a quantized weight tensor
/// (a candle `QTensor`, at the boundary crate) is packed in. See the module
/// doc for why this is a storage concern, a PEER of
/// [`crate::ComputePrecision`] rather than a variant of it.
///
/// No [`Default`]: unlike `ComputePrecision` (whose `F32` default is a real
/// "maximally compatible, unquantized" fallback every backbone can run),
/// there is no quantization format a caller should silently fall into —
/// whether a weight is quantized at all, and to which format, is inherent
/// to the GGUF file it was loaded from, never a knob with a sensible
/// implicit value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WeightQuantization {
    /// 4-bit, block size 32, one `f16` scale per block, no zero point.
    Q4_0,
    /// 4-bit, block size 32, one `f16` scale + one `f16` min per block.
    Q4_1,
    /// 5-bit, block size 32, one `f16` scale per block, no zero point.
    Q5_0,
    /// 5-bit, block size 32, one `f16` scale + one `f16` min per block.
    Q5_1,
    /// 8-bit, block size 32, one `f16` scale per block — the lowest-error
    /// (largest) format this crate names.
    Q8_0,
    /// 2-bit k-quant, block size 256 (`QK_K`).
    Q2K,
    /// 3-bit k-quant, block size 256.
    Q3K,
    /// 4-bit k-quant, block size 256 — a DIFFERENT packing from [`Self::Q4_0`]
    /// at the same bit width (super-block scales rather than one scale per
    /// 32-element block); GGUF names it a distinct dtype, so this crate
    /// does too.
    Q4K,
    /// 5-bit k-quant, block size 256.
    Q5K,
    /// 6-bit k-quant, block size 256 — the highest-fidelity k-quant format
    /// this crate names.
    Q6K,
}

impl WeightQuantization {
    /// The GGML wire ID GGUF itself assigns this dtype (module doc: the
    /// jammi-owned total order every [`Ord`]/[`PartialOrd`] comparison
    /// keys off of). Verified against `candle-core` 0.11.0's own
    /// `GgmlDType::to_u32`/`from_u32` table
    /// (`src/quantized/mod.rs:299-339`) — an explicit, in-code copy rather
    /// than a re-export, since this crate is candle-free (module doc).
    pub fn gguf_wire_id(self) -> u32 {
        match self {
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
        }
    }

    /// Every variant, in [`Self::gguf_wire_id`] ascending order — the
    /// canonical enumeration a caller building a manifest/table over every
    /// known format should iterate, rather than hand-listing variants
    /// (which could silently drift from this module's own variant list).
    pub const ALL: [Self; 10] = [
        Self::Q4_0,
        Self::Q4_1,
        Self::Q5_0,
        Self::Q5_1,
        Self::Q8_0,
        Self::Q2K,
        Self::Q3K,
        Self::Q4K,
        Self::Q5K,
        Self::Q6K,
    ];
}

impl PartialOrd for WeightQuantization {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WeightQuantization {
    /// Keyed on [`Self::gguf_wire_id`] (module doc), never on declaration
    /// order — the stable tie-break key family J's determinism contract
    /// requires for e.g. sorting a `Vec<WeightQuantization>` reproducibly.
    fn cmp(&self, other: &Self) -> Ordering {
        self.gguf_wire_id().cmp(&other.gguf_wire_id())
    }
}

impl fmt::Display for WeightQuantization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Q4_0 => write!(f, "q4_0"),
            Self::Q4_1 => write!(f, "q4_1"),
            Self::Q5_0 => write!(f, "q5_0"),
            Self::Q5_1 => write!(f, "q5_1"),
            Self::Q8_0 => write!(f, "q8_0"),
            Self::Q2K => write!(f, "q2k"),
            Self::Q3K => write!(f, "q3k"),
            Self::Q4K => write!(f, "q4k"),
            Self::Q5K => write!(f, "q5k"),
            Self::Q6K => write!(f, "q6k"),
        }
    }
}

impl FromStr for WeightQuantization {
    type Err = NumericsError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "q4_0" => Ok(Self::Q4_0),
            "q4_1" => Ok(Self::Q4_1),
            "q5_0" => Ok(Self::Q5_0),
            "q5_1" => Ok(Self::Q5_1),
            "q8_0" => Ok(Self::Q8_0),
            "q2k" => Ok(Self::Q2K),
            "q3k" => Ok(Self::Q3K),
            "q4k" => Ok(Self::Q4K),
            "q5k" => Ok(Self::Q5K),
            "q6k" => Ok(Self::Q6K),
            other => Err(NumericsError::InvalidInput(format!(
                "unknown weight quantization '{other}'. Expected: q4_0, q4_1, q5_0, q5_1, q8_0, \
                 q2k, q3k, q4k, q5k, q6k"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_and_from_str_round_trip() {
        for wq in WeightQuantization::ALL {
            let s = wq.to_string();
            assert_eq!(s.parse::<WeightQuantization>().unwrap(), wq);
        }
    }

    #[test]
    fn from_str_rejects_unknown_value() {
        let err = "q4_k_m".parse::<WeightQuantization>().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown weight quantization"));
        assert!(msg.contains("q4_0"));
        assert!(msg.contains("q6k"));
    }

    #[test]
    fn from_str_rejects_compute_precision_tokens() {
        // A `ComputePrecision` token must not silently parse as a
        // `WeightQuantization` — the two vocabularies are peers, never
        // aliases (module doc).
        assert!("f32".parse::<WeightQuantization>().is_err());
        assert!("bf16".parse::<WeightQuantization>().is_err());
    }

    #[test]
    fn serde_uses_lowercase_underscore_preserving_tokens() {
        assert_eq!(
            serde_json::to_value(WeightQuantization::Q4_0).unwrap(),
            serde_json::json!("q4_0")
        );
        assert_eq!(
            serde_json::to_value(WeightQuantization::Q4K).unwrap(),
            serde_json::json!("q4k")
        );
        assert_eq!(
            serde_json::from_value::<WeightQuantization>(serde_json::json!("q6k")).unwrap(),
            WeightQuantization::Q6K
        );
        assert_eq!(
            serde_json::from_value::<WeightQuantization>(serde_json::json!("q5_1")).unwrap(),
            WeightQuantization::Q5_1
        );
    }

    #[test]
    fn serde_rejects_unknown_token() {
        assert!(serde_json::from_value::<WeightQuantization>(serde_json::json!("fp8")).is_err());
    }

    /// Table-vs-candle pin: this crate's own `gguf_wire_id` table must
    /// match the exact values verified against `candle-core` 0.11.0's
    /// `GgmlDType::to_u32` (module doc). A future edit to either table
    /// drifting from the other would silently break cross-crate wire
    /// compatibility without this test.
    #[test]
    fn gguf_wire_id_matches_the_verified_candle_table() {
        let expected: &[(WeightQuantization, u32)] = &[
            (WeightQuantization::Q4_0, 2),
            (WeightQuantization::Q4_1, 3),
            (WeightQuantization::Q5_0, 6),
            (WeightQuantization::Q5_1, 7),
            (WeightQuantization::Q8_0, 8),
            (WeightQuantization::Q2K, 10),
            (WeightQuantization::Q3K, 11),
            (WeightQuantization::Q4K, 12),
            (WeightQuantization::Q5K, 13),
            (WeightQuantization::Q6K, 14),
        ];
        for (wq, id) in expected {
            assert_eq!(wq.gguf_wire_id(), *id, "{wq:?}");
        }
    }

    /// Determinism (family J): sorting a shuffled `Vec<WeightQuantization>`
    /// with the derived-nothing, table-keyed `Ord` impl reproduces
    /// ascending wire-ID order — the stable tie-break key a caller needing
    /// deterministic iteration over a set of formats depends on.
    #[test]
    fn ord_sorts_into_ascending_wire_id_order() {
        // Deliberately NOT already-sorted input.
        let mut shuffled = vec![
            WeightQuantization::Q6K,
            WeightQuantization::Q4_0,
            WeightQuantization::Q8_0,
            WeightQuantization::Q2K,
            WeightQuantization::Q5_1,
            WeightQuantization::Q4K,
            WeightQuantization::Q4_1,
            WeightQuantization::Q3K,
            WeightQuantization::Q5K,
            WeightQuantization::Q5_0,
        ];
        shuffled.sort();
        assert_eq!(shuffled, WeightQuantization::ALL.to_vec());
        let ids: Vec<u32> = shuffled.iter().map(|wq| wq.gguf_wire_id()).collect();
        let mut sorted_ids = ids.clone();
        sorted_ids.sort_unstable();
        assert_eq!(ids, sorted_ids);
    }

    #[test]
    fn ord_is_total_and_consistent_with_eq() {
        for a in WeightQuantization::ALL {
            for b in WeightQuantization::ALL {
                let by_cmp = a.cmp(&b);
                let by_partial = a.partial_cmp(&b).unwrap();
                assert_eq!(by_cmp, by_partial);
                assert_eq!(by_cmp == Ordering::Equal, a == b);
            }
        }
    }

    #[test]
    fn all_covers_every_variant_exactly_once() {
        let mut ids: Vec<u32> = WeightQuantization::ALL
            .iter()
            .map(|wq| wq.gguf_wire_id())
            .collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), WeightQuantization::ALL.len());
    }
}
