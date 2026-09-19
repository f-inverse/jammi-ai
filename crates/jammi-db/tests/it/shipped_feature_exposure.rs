//! The set of cargo features a shipped `jammi-server` artifact carries is DERIVED here and
//! gated against `deny.toml`, never hand-maintained as prose that drifts from the release
//! lanes. Four rounds of this exact unit blocked on the same class: a hand-written sentence
//! in `deny.toml` naming "the release image and the CUDA lanes" as the features-exposed set,
//! silently understating it every time a lane was added or a literal `--features` list moved.
//!
//! All SIX shipped `jammi-server` artifact families (the CUDA tarball, wheel and
//! image, and the CPU tarball, wheel and image) live under `ci/release-feature-
//! manifest.json`'s `lanes` object — a CPU family carries no `capabilities` block
//! (`check_release_manifest.py` enforces that block is required iff a lane's own
//! `cargo_features` names `cuda`/`flash-attn`, forbidden otherwise), but its `cargo_features`
//! sits in exactly the same place as a CUDA lane's. This test derives all six families' feature
//! lists from the manifest ALONE — no second, hand-duplicated per-family feature list here.
//!
//! This lives as an it-test rather than a `ci/scripts/check_*.py` gate for the same reason
//! `bans_names_resolve.rs` does: gate scripts are a human-amend-only surface
//! (`SWARM_GATE_TOUCHED`), `serde_json` is already a jammi-db dependency, and the hermetic
//! `cargo test --workspace` lane already runs this binary on every PR.
//!
//! What this test asserts, all fail-closed (never a silent skip):
//!   1. Every one of the six shipped families is present under the manifest's `lanes` object
//!      (a deleted/renamed family is a FINDING here).
//!   2. `deny.toml` carries a GENERATED, marker-delimited exposure line for every advisory whose
//!      reason names a cargo-gated feature, byte-equal to what today's manifest data computes.
//!      `exposure = [...]` as a structured key inside `[[advisories.ignore]]` is not an option:
//!      measured on cargo-deny 0.20.2, it is `error[unexpected-keys]` and rejects the WHOLE
//!      config. This test never parses `deny.toml`'s prose — it generates the expected line and
//!      checks the file contains that exact line, byte for byte.
//!   3. The Dockerfile's two `ARG CARGO_FEATURES` declarations (the CPU and CUDA builder
//!      stages) carry NO default, and each builder stage's own `RUN` instruction carries the
//!      `${CARGO_FEATURES:?...}` required-argument guard — a literal line scan (the Dockerfile
//!      is not YAML or JSON, so there is no real structured parser for it in this crate's
//!      dependency set) over an HONESTLY STATED, narrow universe: two fixed marker strings,
//!      never a general feature-literal sweep.
//!
//! A feature literal placed in a workflow scalar — a `build-args:` value, a
//! `--features=<list>` or `-F <list>` invocation, an input default, a matrix value — is
//! refused by `ci/scripts/check_workflow_feature_literals.py`, which parses every release
//! workflow (and every local reusable workflow and composite action it reaches) and decides
//! by value over the whole document; each build site reads its lane's list with a
//! `jq -r '.lanes["<key>"].cargo_features | ...'` invocation.
//! `cu12_features` in `ci/scripts/runpod_gpu_prove.sh` carries a literal cargo feature tuple
//! outside that universe on purpose: it is compared against the manifest-derived value with
//! its own loud `PROVE_SURFACE_DRIFT` error rather than reading the manifest directly.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The workspace root, walked up from this crate's manifest directory
/// (`<root>/crates/jammi-db`). Panics rather than skipping if the layout ever changes, so this
/// test cannot silently stop checking anything (same pattern `bans_names_resolve.rs` uses).
fn workspace_root() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .ancestors()
        .find(|dir| dir.join("deny.toml").is_file() && dir.join("Cargo.lock").is_file())
        .unwrap_or_else(|| {
            panic!(
                "no ancestor of {} holds both deny.toml and Cargo.lock",
                manifest.display()
            )
        });
    root.to_path_buf()
}

fn read_to_string(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// The six shipped `jammi-server` artifact families, in the order the exposure
// line lists them. All six live under the manifest's `lanes` key --
// this is a list of NAMES only, never a second copy of any family's feature
// list (that stays exclusively in ci/release-feature-manifest.json).
// ---------------------------------------------------------------------------

const FAMILY_ORDER: &[&str] = &[
    "cpu-wheel",
    "cpu-tarball",
    "cpu-image",
    "cu12-tarball",
    "cu12-wheel",
    "cu12-image",
];

/// Every advisory this test derives an exposure line for, and the single cargo feature that
/// gates the code path the advisory is in. Lives here (not in `ci/release-feature-
/// manifest.json`, which has no advisory-facing section) because nothing besides this test and
/// `deny.toml` needs it, and it is exactly one place either way.
const GATED_ADVISORIES: &[(&str, &str)] = &[
    ("RUSTSEC-2026-0194", "storage-cloud"),
    ("RUSTSEC-2026-0195", "storage-cloud"),
];

fn manifest_json(root: &Path) -> serde_json::Value {
    let text = read_to_string(&root.join("ci/release-feature-manifest.json"));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("ci/release-feature-manifest.json does not parse as JSON: {e}"))
}

fn manifest_lanes(manifest: &serde_json::Value) -> &serde_json::Map<String, serde_json::Value> {
    manifest
        .get("lanes")
        .and_then(|l| l.as_object())
        .unwrap_or_else(|| panic!("ci/release-feature-manifest.json has no `lanes` object"))
}

fn manifest_lane_features(manifest: &serde_json::Value, lane: &str) -> Vec<String> {
    manifest
        .get("lanes")
        .and_then(|l| l.get(lane))
        .and_then(|l| l.get("cargo_features"))
        .and_then(|f| f.as_array())
        .unwrap_or_else(|| {
            panic!("ci/release-feature-manifest.json has no lanes.{lane}.cargo_features array")
        })
        .iter()
        .map(|v| {
            v.as_str()
                .unwrap_or_else(|| panic!("lanes.{lane}.cargo_features has a non-string entry"))
                .to_string()
        })
        .collect()
}

/// Every family (of the six) that carries `feature`, in `FAMILY_ORDER` -- driven ENTIRELY by
/// the manifest (all six families, CPU and CUDA alike, live under `lanes`).
fn families_carrying(
    feature: &str,
    features_by_family: &BTreeMap<&str, Vec<String>>,
) -> Vec<&'static str> {
    FAMILY_ORDER
        .iter()
        .filter(|k| {
            features_by_family
                .get(*k)
                .is_some_and(|feats| feats.iter().any(|f| f == feature))
        })
        .copied()
        .collect()
}

fn generated_exposure_line(advisory_id: &str, families: &[&str]) -> String {
    format!("# EXPOSURE[{advisory_id}] = {}", families.join(", "))
}

#[test]
fn every_shipped_family_is_present_in_the_manifest() {
    // A deleted/renamed family is a FINDING here.
    let root = workspace_root();
    let manifest = manifest_json(&root);
    let lanes = manifest_lanes(&manifest);
    let missing: Vec<&str> = FAMILY_ORDER
        .iter()
        .filter(|fam| !lanes.contains_key(**fam))
        .copied()
        .collect();
    assert!(
        missing.is_empty(),
        "ci/release-feature-manifest.json's `lanes` is missing shipped famil{}: {:?} -- all \
         six shipped families (CPU and CUDA alike) must live under `lanes`",
        if missing.len() == 1 { "y" } else { "ies" },
        missing
    );
}

#[test]
fn deny_toml_carries_the_generated_exposure_line_for_every_gated_advisory() {
    let root = workspace_root();
    let manifest = manifest_json(&root);
    let features_by_family: BTreeMap<&str, Vec<String>> = FAMILY_ORDER
        .iter()
        .map(|fam| (*fam, manifest_lane_features(&manifest, fam)))
        .collect();

    let deny_text = read_to_string(&root.join("deny.toml"));

    let mut missing: Vec<String> = Vec::new();
    for (advisory_id, feature) in GATED_ADVISORIES {
        let families = families_carrying(feature, &features_by_family);
        assert!(
            !families.is_empty(),
            "advisory {advisory_id} names gated feature `{feature}`, but no shipped family's \
             cargo_features carries it -- either the advisory map or a family's feature list is \
             stale"
        );
        let expected = generated_exposure_line(advisory_id, &families);
        if !deny_text.lines().any(|l| l.trim() == expected) {
            missing.push(expected);
        }
    }
    assert!(
        missing.is_empty(),
        "deny.toml is missing (or has a stale) GENERATED exposure line -- add these exact \
         lines verbatim inside the `[[advisories]] ignore` array's reason block (never a \
         structured `exposure = [...]` key: measured on cargo-deny 0.20.2 that rejects the \
         WHOLE config with `error[unexpected-keys]`):\n{}",
        missing.join("\n")
    );
}

#[test]
fn generated_exposure_line_lists_every_family_carrying_the_gated_feature() {
    // Self-contained proof of the generator's own logic: a feature carried by every family
    // must produce a line naming every family, in FAMILY_ORDER.
    let features_by_family: BTreeMap<&str, Vec<String>> = FAMILY_ORDER
        .iter()
        .map(|fam| (*fam, vec!["storage-cloud".to_string()]))
        .collect();
    let families = families_carrying("storage-cloud", &features_by_family);
    assert_eq!(
        families,
        vec![
            "cpu-wheel",
            "cpu-tarball",
            "cpu-image",
            "cu12-tarball",
            "cu12-wheel",
            "cu12-image",
        ]
    );
    assert_eq!(
        generated_exposure_line("RUSTSEC-TEST-0000", &families),
        "# EXPOSURE[RUSTSEC-TEST-0000] = cpu-wheel, cpu-tarball, cpu-image, cu12-tarball, cu12-wheel, cu12-image"
    );
}

#[test]
fn families_carrying_excludes_a_family_missing_the_feature() {
    // Negative control: a family whose cargo_features does NOT name the feature is excluded,
    // never vacuously included.
    let mut features_by_family: BTreeMap<&str, Vec<String>> = FAMILY_ORDER
        .iter()
        .map(|fam| (*fam, vec!["storage-cloud".to_string()]))
        .collect();
    features_by_family.insert("cpu-wheel", vec!["jetstream-broker".to_string()]);
    let families = families_carrying("storage-cloud", &features_by_family);
    assert!(
        !families.contains(&"cpu-wheel"),
        "a family that does not carry the feature must not appear: {families:?}"
    );
}

// ---------------------------------------------------------------------------
// The Dockerfile arm: a literal line scan over an HONESTLY STATED, narrow
// universe (two fixed marker strings) -- the Dockerfile is not YAML or
// JSON, so there is no real structured parser for it available here.
// ---------------------------------------------------------------------------

/// Every line beginning (after leading whitespace) with `ARG CARGO_FEATURES`, as
/// `(1-indexed line number, the line's own trimmed text)`.
fn dockerfile_cargo_features_arg_lines(text: &str) -> Vec<(usize, String)> {
    text.lines()
        .enumerate()
        .filter(|(_, line)| line.trim_start().starts_with("ARG CARGO_FEATURES"))
        .map(|(i, line)| (i + 1, line.trim().to_string()))
        .collect()
}

#[test]
fn dockerfile_cargo_features_arg_carries_no_default() {
    let root = workspace_root();
    let text = read_to_string(&root.join("Dockerfile"));
    let declarations = dockerfile_cargo_features_arg_lines(&text);
    assert_eq!(
        declarations.len(),
        2,
        "expected exactly two `ARG CARGO_FEATURES` declarations (the CPU and CUDA builder \
         stages), found {}: {:?}",
        declarations.len(),
        declarations
    );
    for (lineno, line) in &declarations {
        assert_eq!(
            line, "ARG CARGO_FEATURES",
            "Dockerfile:{lineno}: `ARG CARGO_FEATURES` must carry NO default (a bare \
             declaration) -- found `{line}`. A default masks a missing/mistyped \
             --build-arg, building successfully with the wrong feature list instead of \
             failing loudly at the RUN step's own required-argument guard."
        );
    }
}

#[test]
fn dockerfile_cargo_features_run_carries_the_required_guard() {
    let root = workspace_root();
    let text = read_to_string(&root.join("Dockerfile"));
    let guard = "${CARGO_FEATURES:?";
    let guard_count = text.matches(guard).count();
    assert_eq!(
        guard_count, 2,
        "expected the `{guard}...}}` required-argument guard exactly twice in the Dockerfile \
         (once per builder stage's own RUN instruction: the CPU stage and the CUDA stage), \
         found {guard_count} -- an absent or removed guard lets a `docker build` with no \
         `--build-arg CARGO_FEATURES=...` proceed with an EMPTY feature list instead of \
         failing loudly"
    );
}
