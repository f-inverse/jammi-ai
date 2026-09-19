//! The release manifest's capability categories name exactly the probed ops of
//! each kind, so no op is claimed that nothing can prove and no admission
//! decision goes unclaimed.

use jammi_ai::fine_tune::ComputePrecision;
use jammi_kernels::admission::{ProbedOpKind, PROBED_OPS};

use crate::release_manifest::{
    internal_subkernel_ops, load_manifest, manifest_string_list, MANIFEST_FLASH_DTYPES,
    MANIFEST_FUSED_OP_ADMISSION, MANIFEST_LANE,
};

/// The manifest capability list every [`ProbedOpKind::InternalSubkernel`] row
/// belongs to — the category for kernels that are launched unconditionally from inside an
/// already-admitted parent's fused arm and therefore have NO admission gate
/// of their own to assert `Holds` against.
const MANIFEST_INTERNAL_SUBKERNELS: &str = "internal_subkernels";

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
/// a "claimed but unprovable" entry.
/// Returns `(op, parent)` pairs, sorted by op.
fn manifest_internal_subkernels(manifest: &serde_json::Value) -> Vec<(String, String)> {
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

/// The ONLY manifest capabilities allowed to enumerate names: the two proof
/// mechanisms this file cross-checks against [`PROBED_OPS`] by kind, plus
/// [`MANIFEST_FLASH_DTYPES`] (dtype tokens, asserted as such). Every OTHER
/// capability must be a scalar flag — see
/// [`manifest_capability_categories_match_probed_ops_by_kind`] for why an
/// unrecognized collection is a RED rather than a tolerated addition.
const MANIFEST_NAME_BEARING_CAPABILITIES: &[&str] = &[
    MANIFEST_FUSED_OP_ADMISSION,
    MANIFEST_INTERNAL_SUBKERNELS,
    MANIFEST_FLASH_DTYPES,
];

/// The dtype tokens [`MANIFEST_FLASH_DTYPES`] may contain, DERIVED from
/// [`ComputePrecision`]'s own `Display` (the same rendering
/// [`capability_surface`] compares a declared flash dtype against) — never
/// re-typed as string literals here.
fn dtype_tokens() -> Vec<String> {
    [
        ComputePrecision::F32,
        ComputePrecision::BF16,
        ComputePrecision::F16,
    ]
    .iter()
    .map(ToString::to_string)
    .collect()
}

/// Set-EQUALITY between `ci/release-feature-manifest.json`'s capability
/// categories and [`PROBED_OPS`] grouped by [`ProbedOpKind`] — the structural
/// guard that stops the manifest and the probed-op table from drifting apart
/// (the manifest is one more copy of the same fact the table states).
///
/// Deliberately NOT a GPU test: this is a pure data cross-check between a JSON
/// file and a `const`, so it lives in the hermetic `it` suite and runs on
/// every `cargo test -p jammi-ai`.
///
/// TWO op categories exist, and they are read in the SHAPE each actually
/// has — `fused_op_admission` as a string list, `internal_subkernels` as an
/// OBJECT keyed by op (see [`manifest_internal_subkernels`], and
/// `ci/scripts/check_release_manifest.py` which validates each entry's
/// `parent`/`launch_site`). That asymmetry is deliberate on the manifest's
/// side and is asserted THROUGH here, not flattened away: a subkernel's
/// `parent` is the only evidence it ran at all.
///
/// The third assertion is a CLOSURE check over the whole `capabilities`
/// object: no capability OTHER than those two may name an op. A compiled-only
/// bucket — kernels proven by compiling, never by dispatching — carries no
/// proof mechanism (census artifact
/// `crates/jammi-kernels/artifacts/cuda-runs/2026-09-01-axpy-census-bdeb80c-a100-pcie.json`).
/// Checking only the two named categories would let such a bucket appear
/// under any new name and go unnoticed here, which is exactly the drift this test
/// exists to catch — so the check is over the capability object's KEYS, not
/// over a list of names this file already knows.
///
/// A RED here names exactly which side is missing which key (or which parent
/// the two disagree on); it is never a reason to weaken the assertion to a
/// subset check, because a subset check is precisely what let the f16 keys go
/// missing.
#[test]
fn manifest_capability_categories_match_probed_ops_by_kind() {
    let manifest = load_manifest();

    let mut expected_admission: Vec<&str> = PROBED_OPS
        .iter()
        .filter(|op| matches!(op.kind(), ProbedOpKind::TwoArm | ProbedOpKind::Cascade))
        .map(|op| op.report_key())
        .collect();
    expected_admission.sort_unstable();
    let mut declared_admission = manifest_string_list(&manifest, MANIFEST_FUSED_OP_ADMISSION);
    declared_admission.sort();
    assert_eq!(
        declared_admission,
        expected_admission
            .iter()
            .map(|s| (*s).to_string())
            .collect::<Vec<String>>(),
        "manifest lane {MANIFEST_LANE:?}'s {MANIFEST_FUSED_OP_ADMISSION} must name EXACTLY the \
         PROBED_OPS rows that dispatch through admit()/admit_cascade(). A key only the manifest \
         names is a capability nothing can prove; a key only PROBED_OPS names is a real \
         admission decision the release manifest does not claim (which is how the f16 \
         cast-epilogue keys went missing). See this test's doc for the pending manifest edit."
    );

    // `internal_subkernels` is checked on BOTH the op set AND the proof
    // relation. Set-equality alone would let the manifest name the right op
    // while attributing it to the wrong parent — and the parent IS the whole
    // evidence chain for these rows (they have no admission gate; "it ran" is
    // inferred entirely from the parent dispatching fused). A manifest that
    // said `scaled_cast_add`'s parent were, say, `attention_block_flash` would
    // be claiming a proof that does not exist, which set-equality could not
    // see. Comparing `(op, parent)` pairs makes the table and the manifest
    // agree on the relation, not just the name.
    let mut expected_subkernels: Vec<(String, String)> = internal_subkernel_ops()
        .into_iter()
        .map(|(op, parent)| (op.to_string(), parent.to_string()))
        .collect();
    expected_subkernels.sort();
    let declared_subkernels = manifest_internal_subkernels(&manifest);
    assert_eq!(
        declared_subkernels, expected_subkernels,
        "manifest lane {MANIFEST_LANE:?}'s {MANIFEST_INTERNAL_SUBKERNELS} must name EXACTLY the \
         PROBED_OPS rows with no admission gate of their own, AND agree with the table on each \
         one's `parent` — the op whose fused dispatch is the only thing that proves the \
         subkernel ran. A name-only match with a wrong parent is a claimed proof that does not \
         exist."
    );

    // NO OTHER OP-BEARING CATEGORY. The two assertions above pin the two
    // categories by NAME; on their own they say nothing about a THIRD one, so
    // a proof-less bucket (see this test's doc) could appear beside them.
    // This check is therefore over the capability object's own keys: a
    // capability that is not one of [`MANIFEST_NAME_BEARING_CAPABILITIES`]
    // may not enumerate anything at all.
    //
    // "Enumerates something" is judged by JSON SHAPE, not by matching names
    // against PROBED_OPS: a bucket of unprovable ops is unprovable precisely
    // because PROBED_OPS does NOT name its members, so a name-overlap test
    // would be blind to the one case that matters. A non-empty array or
    // object names things; so does a string (a one-op bucket needs no
    // brackets). A bool or a number cannot. An EMPTY array/object is
    // tolerated — it claims no capability, so it is dead schema for
    // `ci/scripts/check_release_manifest.py` to prune, not a false claim this
    // test can see on a device.
    let capabilities = manifest["lanes"][MANIFEST_LANE]["capabilities"]
        .as_object()
        .unwrap_or_else(|| {
            panic!("manifest lane {MANIFEST_LANE:?} must carry a `capabilities` object")
        });
    let unaccounted: Vec<String> = capabilities
        .iter()
        .filter(|(key, _)| !MANIFEST_NAME_BEARING_CAPABILITIES.contains(&key.as_str()))
        .filter(|(_, value)| match value {
            serde_json::Value::Array(items) => !items.is_empty(),
            serde_json::Value::Object(entries) => !entries.is_empty(),
            serde_json::Value::String(_) => true,
            _ => false,
        })
        .map(|(key, value)| format!("{key}: {value}"))
        .collect();
    assert!(
        unaccounted.is_empty(),
        "manifest lane {MANIFEST_LANE:?} names something in a capability that carries no proof \
         mechanism — {unaccounted:?}. Only {MANIFEST_FUSED_OP_ADMISSION} (its own admission \
         delta) and {MANIFEST_INTERNAL_SUBKERNELS} (its parent's) may name ops, and \
         {MANIFEST_FLASH_DTYPES} may name dtypes. A kernel that compiles but dispatches \
         through nothing is not a capability this release surface may claim: wire a real \
         admit()/admit_cascade() site (plus its PROBED_OPS row), give it an admitted parent \
         that launches it, or DELETE it. If this key genuinely names no op, it \
         must be a scalar flag, or be added to MANIFEST_NAME_BEARING_CAPABILITIES with its \
         own content assertion the way {MANIFEST_FLASH_DTYPES} has one below."
    );

    // Non-vacuity for the allowance above: `flash_dtypes` is let past the
    // closure check because it names DTYPES, and that is asserted, not
    // assumed — otherwise an op list could be smuggled in under this key and
    // the closure check would wave it through.
    let tokens = dtype_tokens();
    let declared_flash_dtypes = manifest_string_list(&manifest, MANIFEST_FLASH_DTYPES);
    let non_dtype: Vec<&String> = declared_flash_dtypes
        .iter()
        .filter(|d| !tokens.contains(d))
        .collect();
    assert!(
        non_dtype.is_empty(),
        "manifest lane {MANIFEST_LANE:?}'s {MANIFEST_FLASH_DTYPES} must name only \
         ComputePrecision dtype tokens {tokens:?} — {non_dtype:?} are not dtypes, and this \
         capability is exempt from the op-bearing check above precisely on the grounds that it \
         names dtypes"
    );
}
