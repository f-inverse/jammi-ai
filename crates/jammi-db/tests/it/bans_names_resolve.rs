//! Every `[bans] deny` name in `deny.toml` resolves to a package in `Cargo.lock`.
//!
//! cargo-deny matches a ban's `name` LITERALLY and reports nothing at all when
//! it matches no package: `datafusion-*`, `hashbr*`, `datafussion` and
//! `this-crate-does-not-exist-at-all` all come back `bans ok`. So a typo, or an
//! upstream rename of a crate the engine line depends on, silently retires the
//! ban that was supposed to hold that crate to one version — the version fences
//! cannot catch this, because they are keyed on the same unmatched name.
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
    let banned: BTreeSet<&str> = entries
        .iter()
        .map(|e| {
            // cargo-deny 0.20 accepts either spelling of the package key.
            e.get("name")
                .or_else(|| e.get("crate"))
                .and_then(|n| n.as_str())
                .expect("every [bans] deny entry has a string `name` or `crate`")
        })
        .collect();

    let lock = read_toml(&root.join("Cargo.lock"));
    let packages: BTreeSet<&str> = lock
        .get("package")
        .and_then(|p| p.as_array())
        .expect("Cargo.lock has a [[package]] array")
        .iter()
        .filter_map(|p| p.get("name").and_then(|n| n.as_str()))
        .collect();

    let unmatched: Vec<&&str> = banned.iter().filter(|n| !packages.contains(**n)).collect();
    assert!(
        unmatched.is_empty(),
        "these deny.toml [bans] deny names match no package in Cargo.lock, so \
         cargo-deny silently enforces nothing for them (a typo, or an upstream \
         rename): {unmatched:?}"
    );
}
