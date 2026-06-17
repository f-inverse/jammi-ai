//! Wire conversions for the opt-in producer memoization dial — the shared
//! `CachePolicy` request field and `CacheOutcome` response field every
//! result-table producer verb carries. Lives in one module because the enum is
//! shared across the embedding / inference / pipeline services (the proto
//! defines it once in `jammi.v1.inference`), so its decode/encode is shared too
//! rather than duplicated per service converter.

use jammi_db::store::{CacheOutcome, CachePolicy};
use jammi_wire::proto::inference as pb;
use tonic::Status;

/// Decode the wire [`pb::CachePolicy`] into the engine [`CachePolicy`].
/// `UNSPECIFIED` maps to the engine default ([`CachePolicy::Bypass`]) — the
/// documented mapping, matching how every other `*_UNSPECIFIED` arm resolves to
/// its engine default. An out-of-range enum value is a loud `invalid_argument`,
/// never a silent fall-through to a default.
pub fn cache_policy_from_proto(policy: i32) -> Result<CachePolicy, Status> {
    match pb::CachePolicy::try_from(policy) {
        Ok(pb::CachePolicy::Unspecified) | Ok(pb::CachePolicy::Bypass) => Ok(CachePolicy::Bypass),
        Ok(pb::CachePolicy::Use) => Ok(CachePolicy::Use),
        Err(_) => Err(Status::invalid_argument("unknown cache policy")),
    }
}

/// Encode the engine [`CacheOutcome`] into the wire enum value, so reuse is
/// observable on the wire. The engine never produces `UNSPECIFIED`; the variant
/// is the wire default a decoder rejects.
pub fn cache_outcome_to_proto(outcome: &CacheOutcome) -> i32 {
    match outcome {
        CacheOutcome::Computed => pb::CacheOutcome::Computed as i32,
        CacheOutcome::Reused { .. } => pb::CacheOutcome::Reused as i32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unspecified_and_bypass_both_map_to_bypass() {
        assert_eq!(
            cache_policy_from_proto(pb::CachePolicy::Unspecified as i32).unwrap(),
            CachePolicy::Bypass
        );
        assert_eq!(
            cache_policy_from_proto(pb::CachePolicy::Bypass as i32).unwrap(),
            CachePolicy::Bypass
        );
    }

    #[test]
    fn use_maps_to_use() {
        assert_eq!(
            cache_policy_from_proto(pb::CachePolicy::Use as i32).unwrap(),
            CachePolicy::Use
        );
    }

    #[test]
    fn an_out_of_range_policy_is_a_loud_error() {
        let err = cache_policy_from_proto(99).unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn outcome_round_trips_to_the_wire_variant() {
        assert_eq!(
            cache_outcome_to_proto(&CacheOutcome::Computed),
            pb::CacheOutcome::Computed as i32
        );
        assert_eq!(
            cache_outcome_to_proto(&CacheOutcome::Reused { table: "t".into() }),
            pb::CacheOutcome::Reused as i32
        );
    }
}
