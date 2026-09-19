//! The release feature manifest's `cu12-tarball` lane (`ci/release-feature-manifest.json`),
//! read for the tests that hold it against the kernels' probed-op table. Shared by
//! this binary and the `gpu_capability` suite.

use jammi_kernels::admission::{ProbedOpKind, PROBED_OPS};

/// The manifest lane this test reads — the same lane `runpod_gpu_prove.sh`'s
/// capability-surface build derives its feature list from.
pub(crate) const MANIFEST_LANE: &str = "cu12-tarball";

/// `ci/release-feature-manifest.json`, located relative to this crate's
/// manifest dir (`crates/jammi-ai` → workspace root → `ci/`).
pub(crate) const MANIFEST_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../ci/release-feature-manifest.json"
);

/// The manifest capability list every [`ProbedOpKind::TwoArm`] /
/// [`ProbedOpKind::Cascade`] row belongs to.
pub(crate) const MANIFEST_FUSED_OP_ADMISSION: &str = "fused_op_admission";

/// The manifest capability list every [`ProbedOpKind::InternalSubkernel`] row
/// belongs to — the category the campaign lead's manifest reclassification
/// introduces for kernels that are launched unconditionally from inside an
/// already-admitted parent's fused arm and therefore have NO admission gate
/// of their own to assert `Holds` against.
pub(crate) const MANIFEST_INTERNAL_SUBKERNELS: &str = "internal_subkernels";

/// The manifest capability naming the dtypes this build's flash cascade
/// preempts `attention_block`/`mem_efficient_attention` for. Not an op list —
/// and [`manifest_capability_categories_match_probed_ops_by_kind`] asserts
/// that by checking its entries against [`ComputePrecision`]'s own dtype
/// tokens, so allowing it past the op-bearing-category check below cannot be
/// used to smuggle an unprovable op in under a dtype-shaped name.
pub(crate) const MANIFEST_FLASH_DTYPES: &str = "flash_dtypes";

/// `(op, the parent whose fused dispatch proves it ran)`, DERIVED from
/// [`PROBED_OPS`]'s [`ProbedOpKind::InternalSubkernel`] rows — the manifest's
/// new `internal_subkernels` category.
pub(crate) fn internal_subkernel_ops() -> Vec<(&'static str, &'static str)> {
    PROBED_OPS
        .iter()
        .filter_map(|op| match op.kind() {
            ProbedOpKind::InternalSubkernel { parent } => Some((op.report_key(), parent)),
            _ => None,
        })
        .collect()
}

pub(crate) fn load_manifest() -> serde_json::Value {
    let raw = std::fs::read_to_string(MANIFEST_PATH)
        .unwrap_or_else(|e| panic!("read {MANIFEST_PATH}: {e}"));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{MANIFEST_PATH} must be valid JSON: {e}"))
}

pub(crate) fn manifest_string_list(manifest: &serde_json::Value, capability: &str) -> Vec<String> {
    manifest["lanes"][MANIFEST_LANE]["capabilities"][capability]
        .as_array()
        .unwrap_or_else(|| {
            panic!("manifest lane {MANIFEST_LANE:?} is missing capabilities.{capability}")
        })
        .iter()
        .map(|v| {
            v.as_str()
                .unwrap_or_else(|| panic!("capabilities.{capability} entry {v:?} is not a string"))
                .to_string()
        })
        .collect()
}

/// `capabilities.internal_subkernels`, which — unlike its sibling capability
/// lists — is an OBJECT keyed by op, not a string array:
/// `{"<op>": {"parent": "<op>", "launch_site": "<path>"}, ..}`
/// (`ci/scripts/check_release_manifest.py` validates that every `parent`
/// resolves and every `launch_site` exists).
///
/// The shape is the point, not an inconsistency to paper over: an internal
/// subkernel has no admission gate of its own and is PROVABLE only through its
/// parent's dispatch, so the manifest records the proof RELATION alongside the
/// name. A plain string list could name the op but not say what proves it —
/// the same "claimed but unprovable" shape campaign #446 finding 2 was about.
/// Returns `(op, parent)` pairs, sorted by op.
pub(crate) fn manifest_internal_subkernels(manifest: &serde_json::Value) -> Vec<(String, String)> {
    let obj = manifest["lanes"][MANIFEST_LANE]["capabilities"][MANIFEST_INTERNAL_SUBKERNELS]
        .as_object()
        .unwrap_or_else(|| {
            panic!(
                "manifest lane {MANIFEST_LANE:?}'s capabilities.{MANIFEST_INTERNAL_SUBKERNELS} \
                 must be an OBJECT keyed by op (each value carrying `parent` + `launch_site`), \
                 not an array — see ci/scripts/check_release_manifest.py"
            )
        });
    let mut out: Vec<(String, String)> = obj
        .iter()
        .map(|(op, entry)| {
            let parent = entry["parent"].as_str().unwrap_or_else(|| {
                panic!(
                    "capabilities.{MANIFEST_INTERNAL_SUBKERNELS}[{op:?}] must carry a string \
                     `parent` — the op whose fused dispatch is what proves this subkernel ran"
                )
            });
            (op.clone(), parent.to_string())
        })
        .collect();
    out.sort();
    out
}
