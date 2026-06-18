//! The API-stability freeze-guard.
//!
//! The terminal-0.x engineering bar freezes the public wire surface: the nine
//! `jammi.v1.*` proto packages and the `(Service, Method)` rpc paths they serve
//! (see `docs/guide/src/api-stability.md`). This guard makes that freeze
//! enforceable rather than aspirational: it decodes the compiled
//! [`FILE_DESCRIPTOR_SET`] — the authoritative machine-readable description of
//! what the binary actually serves, the same source the tenant-isolation oracle
//! derives its rpc inventory from — and asserts the live surface EQUALS the
//! committed baseline in `api_freeze_baseline.txt`.
//!
//! Teeth: removing or renaming a stable rpc (or dropping a `jammi.v1.*` package)
//! makes the live set decoded from the descriptor diverge from the baseline, and
//! the equality assertion fails naming the rpc/package that disappeared.
//! *Adding* an rpc fails too unless the baseline is updated in the same PR — the
//! additive, minor-compatible case made explicit and visible in the diff.
//! Dropping a baseline line is exactly the breaking change the semver commitment
//! forbids outside a major, so it can never happen silently.
//!
//! The persisted-format identity is frozen the same way: `MANIFEST_VERSION` (the
//! materialization manifest's format-of-record version) is pinned here, so a
//! bump that is not a deliberate, reviewed format change reds CI.

use std::collections::BTreeSet;

use jammi_db::store::manifest::MANIFEST_VERSION;
use jammi_wire::FILE_DESCRIPTOR_SET;
use prost::Message;
use prost_types::FileDescriptorSet;

/// Every frozen wire package is under this prefix. The descriptor also carries
/// `google.*` / `arrow.*` imports; only `jammi.v1.*` is the engine's own frozen
/// surface.
const WIRE_PACKAGE_PREFIX: &str = "jammi.v1";

/// The committed frozen baseline, compiled in so the test needs no runtime file
/// resolution.
const BASELINE: &str = include_str!("api_freeze_baseline.txt");

/// The frozen wire surface as a set of opaque tokens: `PACKAGE <pkg>` for each
/// `jammi.v1.*` package and `RPC <Service>/<Method>` for each served rpc. One
/// representation drives both the live decode and the baseline parse, so the
/// two are compared in identical shape.
fn live_surface() -> BTreeSet<String> {
    let set = FileDescriptorSet::decode(FILE_DESCRIPTOR_SET)
        .expect("the compiled jammi.v1 descriptor must decode");
    let mut surface = BTreeSet::new();
    for file in &set.file {
        let package = file.package();
        if package != WIRE_PACKAGE_PREFIX
            && !package.starts_with(&format!("{WIRE_PACKAGE_PREFIX}."))
        {
            continue;
        }
        surface.insert(format!("PACKAGE {package}"));
        for service in &file.service {
            for method in &service.method {
                surface.insert(format!("RPC {}/{}", service.name(), method.name()));
            }
        }
    }
    surface
}

/// Parse the committed baseline into the same token set, ignoring blank lines
/// and `#` comments.
fn baseline_surface() -> BTreeSet<String> {
    BASELINE
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .collect()
}

#[test]
fn wire_surface_equals_the_frozen_baseline() {
    let live = live_surface();
    let baseline = baseline_surface();

    let removed: Vec<&String> = baseline.difference(&live).collect();
    assert!(
        removed.is_empty(),
        "FROZEN wire surface entry removed/renamed — this is a breaking change \
         (major only). Live descriptor no longer serves:\n  {removed:#?}"
    );

    let added: Vec<&String> = live.difference(&baseline).collect();
    assert!(
        added.is_empty(),
        "New wire surface entry not in the frozen baseline. If this is an \
         additive (minor-compatible) rpc/package, append it to \
         `api_freeze_baseline.txt` in this same PR:\n  {added:#?}"
    );
}

#[test]
fn manifest_format_version_is_frozen() {
    // The materialization manifest's format-of-record version. A bump is a
    // deliberate, reviewed format change (see docs/guide/src/format-stability.md
    // and api-stability.md); pinning it here means an accidental bump reds CI.
    assert_eq!(
        MANIFEST_VERSION, 3,
        "MANIFEST_VERSION changed from the frozen value 3 — a persisted-format \
         version bump must be a deliberate, reviewed format change, not an \
         accidental edit"
    );
}
