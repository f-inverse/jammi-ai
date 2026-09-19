//! Wire conversions for the opt-in producer memoization dial — the shared
//! `CachePolicy` request field every result-table producer verb carries.
//! Lives in one module because the enum is shared across the embedding /
//! inference / pipeline services (the proto defines it once in
//! `jammi.v1.inference`), so its decode/encode is shared too rather than
//! duplicated per service converter. The response half, `CacheOutcome`, is
//! `jammi_wire::cache_outcome_to_proto` / `cache_outcome_from_proto`.

use jammi_db::store::CachePolicy;
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

/// Encode the engine [`CachePolicy`] into the wire enum.
///
/// `Bypass` (the engine default) encodes as `UNSPECIFIED` (`0`), NOT the
/// explicit `BYPASS` variant: a caller that never touches the request's
/// `cache` field (every non-Rust client, and the Rust client's own
/// default-constructed request) leaves the wire field at proto3's implicit
/// zero value, which is `UNSPECIFIED`, so every writer of a `Bypass` policy —
/// including the engine's own re-encode of a `Bypass` spec (`recompute`'s
/// replay, `training_spec_to_proto`) — must agree on that ONE encoding
/// rather than two different wire bytes for the identical decoded value.
/// `Use` still encodes to the explicit `USE` variant (there is no
/// "unspecified but reused" state to collapse it into). [`cache_policy_from_proto`]
/// decodes both `UNSPECIFIED` and `BYPASS` to [`CachePolicy::Bypass`], so
/// decode is unaffected by which of the two equally-valid encodings of
/// `Bypass` this function chooses to emit.
pub fn cache_policy_to_proto(policy: CachePolicy) -> pb::CachePolicy {
    match policy {
        CachePolicy::Use => pb::CachePolicy::Use,
        CachePolicy::Bypass => pb::CachePolicy::Unspecified,
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
    fn cache_policy_encode_decode_round_trips_both_variants() {
        for policy in [CachePolicy::Use, CachePolicy::Bypass] {
            let encoded = cache_policy_to_proto(policy) as i32;
            assert_eq!(cache_policy_from_proto(encoded).unwrap(), policy);
        }
    }

    /// One spec has one wire encoding, on both sides of the transport: a
    /// client that never sets the request's `cache` field transmits proto3's
    /// implicit zero value (`UNSPECIFIED`), and the engine's OWN re-encode of
    /// a `Bypass` spec (`training_spec_to_proto`'s remote-send path, a
    /// recompute replay) must emit the SAME `0`, never the explicit,
    /// distinguishable `BYPASS = 2` — two different bytes on the wire for
    /// what decodes to the identical engine value would defeat any
    /// byte-comparison of "the same spec" across the two paths.
    #[test]
    fn engine_re_encode_of_bypass_matches_an_unset_clients_wire_bytes() {
        let engine_encoded = cache_policy_to_proto(CachePolicy::Bypass) as i32;
        assert_eq!(
            engine_encoded, 0,
            "Bypass must re-encode to the wire's implicit unset value (0), matching a client \
             that never touches the `cache` field, got {engine_encoded}"
        );
        assert_eq!(
            engine_encoded,
            pb::CachePolicy::Unspecified as i32,
            "0 must be exactly UNSPECIFIED, never the explicit BYPASS variant"
        );
    }
}
