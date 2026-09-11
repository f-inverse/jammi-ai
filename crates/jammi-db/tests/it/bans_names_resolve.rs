//! Every `[bans] deny` name/crate in `deny.toml` resolves to a package in `Cargo.lock`.
//!
//! cargo-deny matches a ban's `name` LITERALLY and reports nothing at all when
//! it matches no package: `datafusion-*`, `hashbr*`, `datafussion` and
//! `this-crate-does-not-exist-at-all` all come back `bans ok`. So a typo, or an
//! upstream rename of a crate the engine line depends on, silently retires the
//! ban that was supposed to hold that crate to one version — the version fences
//! cannot catch this, because they are keyed on the same unmatched name.
//!
//! `name` and `crate` are NOT synonyms (measured against cargo-deny 0.20.2, a
//! real `libc 0.2.189`): `{ crate = "libc:>=0.2.0" }` takes a package SPEC
//! (`name[:version-req]`) and fires correctly; `{ name = "libc:>=0.2.0" }`
//! matches `name` LITERALLY against the unversioned package name and comes
//! back `bans ok` — a genuinely dead ban. So this test resolves `crate` by
//! splitting on `:` and keeping only the package-name half, but resolves
//! `name` VERBATIM and REDs if a `name` value contains `:` (that shape can
//! only be a dead ban here, never a working one).
//!
//! This lives as an it-test rather than a `ci/scripts/check_*.py` gate because
//! the gate scripts are a human-amend-only surface, `toml` is already a
//! jammi-db dependency, and the hermetic `cargo test --workspace` lane already
//! runs this binary on every PR. It never skips: a missing or unparsable
//! `deny.toml`/`Cargo.lock` is a loud failure.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The workspace root, walked up from this crate's manifest directory
/// (`<root>/crates/jammi-db`). Panics rather than skipping if the layout ever
/// changes, so this test cannot silently stop checking anything.
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

fn read_toml(path: &Path) -> toml::Value {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    text.parse::<toml::Value>()
        .unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()))
}

/// Resolve one `[bans] deny` entry to the bare package name cargo-deny will match against
/// `Cargo.lock`. `crate` takes a package SPEC (`name[:version-req]`) and is split on `:`;
/// `name` is matched LITERALLY by cargo-deny, so it is used verbatim and rejected here if it
/// contains `:` (a `name` with a version-req suffix can only be a dead ban — see this module's
/// doc). Pure (no disk access) so it can be unit-tested directly against synthetic entries.
fn resolve_ban_name(entry: &toml::Value) -> Result<String, String> {
    if let Some(name) = entry.get("name").and_then(|n| n.as_str()) {
        if name.contains(':') {
            return Err(format!(
                "[bans] deny entry `name = \"{name}\"` contains ':' — cargo-deny matches \
                 `name` LITERALLY (unlike `crate`, which takes a package SPEC \
                 `name[:version-req]`), so this ban silently matches nothing and is a dead \
                 ban; use `crate = \"{name}\"` for a versioned spec"
            ));
        }
        return Ok(name.to_string());
    }
    if let Some(spec) = entry.get("crate").and_then(|n| n.as_str()) {
        return Ok(spec.split(':').next().unwrap_or(spec).to_string());
    }
    Err("entry has neither a string `name` nor a string `crate`".to_string())
}

#[test]
fn bans_deny_names_resolve_to_packages_in_the_lockfile() {
    let root = workspace_root();

    let deny = read_toml(&root.join("deny.toml"));
    let entries = deny
        .get("bans")
        .and_then(|b| b.get("deny"))
        .and_then(|d| d.as_array())
        .expect("deny.toml has a [bans] deny array");
    assert!(
        !entries.is_empty(),
        "deny.toml's [bans] deny array is empty — the engine-line bound is gone"
    );
    let banned: BTreeSet<String> = entries
        .iter()
        .map(|e| resolve_ban_name(e).unwrap_or_else(|msg| panic!("{msg}")))
        .collect();

    let lock = read_toml(&root.join("Cargo.lock"));
    let packages: BTreeSet<&str> = lock
        .get("package")
        .and_then(|p| p.as_array())
        .expect("Cargo.lock has a [[package]] array")
        .iter()
        .filter_map(|p| p.get("name").and_then(|n| n.as_str()))
        .collect();

    let unmatched: Vec<&String> = banned
        .iter()
        .filter(|n| !packages.contains(n.as_str()))
        .collect();
    assert!(
        unmatched.is_empty(),
        "these deny.toml [bans] deny names match no package in Cargo.lock, so \
         cargo-deny silently enforces nothing for them (a typo, or an upstream \
         rename): {unmatched:?}"
    );
}

#[test]
fn crate_key_spec_form_resolves_to_the_bare_package_name() {
    let entry: toml::Value = toml::from_str(r#"crate = "arrow:>=59.0.0""#).unwrap();
    assert_eq!(resolve_ban_name(&entry).unwrap(), "arrow");
}

#[test]
fn name_key_containing_a_colon_is_flagged_as_a_dead_ban() {
    // Measured against a real `libc 0.2.189`: cargo-deny matches `name` literally, so
    // `{ name = "libc:>=0.2.0" }` returns `bans ok` -- a ban that silently enforces nothing.
    let entry: toml::Value = toml::from_str(r#"name = "libc:>=0.2.0""#).unwrap();
    assert!(
        resolve_ban_name(&entry).is_err(),
        "a `name` value containing ':' must be flagged, not silently truncated to a bare name \
         that happens to match something else in the lockfile"
    );
}

#[test]
fn name_key_without_a_colon_still_resolves_verbatim() {
    let entry: toml::Value = toml::from_str(r#"name = "datafussion""#).unwrap();
    assert_eq!(resolve_ban_name(&entry).unwrap(), "datafussion");
}
