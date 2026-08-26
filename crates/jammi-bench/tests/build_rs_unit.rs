//! Unit tests for `build.rs`'s own pure functions, driven directly against
//! that file's REAL source via `#[path]` (never a hand-copied duplicate that
//! could silently drift) — `build.rs` is not a normal `cargo test` target
//! (it is a standalone binary Cargo compiles and RUNS once, as the build
//! script; `cargo test -p jammi-bench --list` confirms it names nothing
//! from `build.rs`), so an integration test file is the only way to exercise
//! its logic under `cargo test`. `build_script::main` (the real build
//! script entry point) is never called here — only the pure functions it
//! calls (`is_40_lowercase_hex`, `env_or_unknown`, `workspace_root`) are, so
//! this suite never touches `cargo:rustc-env`/`cargo:rerun-if-changed`
//! output or spawns `git`.
//!
//! Round-2 audit (B5): "`JAMMI_BUILD_SHA=deadbeef` ⇒ `unknown` (a build.rs
//! unit case)" — `deadbeef_falls_through_shape_check` below is that case,
//! isolated to the ONE function (`is_40_lowercase_hex`) whose shape check is
//! what actually rejects it, so this suite stays a true unit test rather
//! than needing a real subprocess build (that end-to-end shape — "fresh
//! build, no git, `JAMMI_BUILD_SHA=deadbeef` in the environment, baked sha
//! is literally `unknown`" — is `provenance_baked.rs`'s
//! `deadbeef_env_var_falls_through_to_unknown_when_git_is_unavailable`,
//! which builds a REAL scratch crate; see that test's own doc for why the
//! two are complementary, not redundant).

#[path = "../build.rs"]
#[allow(dead_code)]
mod build_script;

use build_script::{env_or_unknown, is_40_lowercase_hex, workspace_root};
use std::path::PathBuf;

#[test]
fn is_40_lowercase_hex_accepts_only_the_exact_shape() {
    assert!(is_40_lowercase_hex(&"a".repeat(40)));
    assert!(is_40_lowercase_hex(
        "0123456789abcdef0123456789abcdef01234567"
    ));
    assert!(!is_40_lowercase_hex(&"A".repeat(40)), "uppercase rejected");
    assert!(!is_40_lowercase_hex(&"a".repeat(39)), "too short rejected");
    assert!(!is_40_lowercase_hex(&"a".repeat(41)), "too long rejected");
    assert!(!is_40_lowercase_hex(""), "empty rejected");
}

/// Contract §6 F3's second half, as a unit case: `deadbeef` is 8 characters
/// — it fails `is_40_lowercase_hex`'s LENGTH check, the exact function
/// `build.rs::main` calls before ever falling through to `git`. RED at base
/// (before this suite existed): nothing in the tree checked this shape at
/// all.
#[test]
fn deadbeef_falls_through_shape_check() {
    assert!(
        !is_40_lowercase_hex("deadbeef"),
        "JAMMI_BUILD_SHA=deadbeef must NOT be accepted verbatim — it must fall through to \
         the git/unknown fallback, per build.rs's own documented precedence"
    );
}

#[test]
fn env_or_unknown_falls_back_on_empty_and_absent() {
    // `std::env::set_var`/`remove_var` are process-global — safe here
    // because this key is unique to this test and this test crate's suite
    // does not race any other test over it (unlike the OnceLock-memoized
    // `jammi_kernels::admission` state `finetune_step_kernel_disable.rs`'s
    // module doc warns about — `env_or_unknown` re-reads the env on every
    // call, there is no memoization to race).
    let key = "JAMMI_BUILD_RS_TEST_ENV_OR_UNKNOWN_PROBE";
    unsafe {
        std::env::remove_var(key);
    }
    assert_eq!(env_or_unknown(key), "unknown", "absent env var falls back");
    unsafe {
        std::env::set_var(key, "");
    }
    assert_eq!(env_or_unknown(key), "unknown", "empty env var falls back");
    unsafe {
        std::env::set_var(key, "x86_64-unknown-linux-gnu");
    }
    assert_eq!(env_or_unknown(key), "x86_64-unknown-linux-gnu");
    unsafe {
        std::env::remove_var(key);
    }
}

#[test]
fn workspace_root_derives_two_levels_up_from_manifest_dir() {
    assert_eq!(
        workspace_root("/repo/crates/jammi-bench"),
        PathBuf::from("/repo")
    );
    // A shallow/unexpected layout falls back to the manifest dir itself
    // rather than panicking.
    assert_eq!(
        workspace_root("/onlyonelevel"),
        PathBuf::from("/onlyonelevel")
    );
}
