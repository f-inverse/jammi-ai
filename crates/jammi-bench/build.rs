//! Bakes this binary's build-time identity into compile-time environment
//! variables the crate reads back with `env!()` — never `std::env::var()` at
//! run time (unification contract C1). Three literals, `cargo:rustc-env`'d
//! once per build:
//!
//!   - `JAMMI_BUILD_SHA` — `<sha>`, `<sha>-dirty`, or `"unknown"`.
//!   - `JAMMI_BUILD_TARGET` — Cargo's own `$TARGET` for this build.
//!   - `JAMMI_BUILD_PROFILE` — Cargo's own `$PROFILE` for this build.
//!
//! `JAMMI_BUILD_SHA` precedence: an already-set `JAMMI_BUILD_SHA` env var is
//! accepted ONLY if it is exactly 40 lowercase hex characters (`deadbeef` —
//! or anything else that is not that exact shape — is REJECTED and falls
//! through, never trusted verbatim); otherwise a build-time `git rev-parse
//! HEAD` against this crate's own directory, with dirtiness folded in;
//! otherwise the literal `"unknown"`. This is domain validation at the
//! INPUT edge (K2): the binary itself never re-validates `JAMMI_BUILD_SHA`
//! at run time, because by the time it runs there is nothing left to
//! validate — the value baked here is a `'static` compile-time literal.
//!
//! `provenance_baked.rs`'s `runtime_env_and_cwd_are_inert` case is the
//! mechanical proof that nothing here leaks into a run-time read: a later
//! `JAMMI_BUILD_SHA` in the process environment, or a fresh `cwd`, changes
//! NOTHING about an already-built binary's reported identity.
//!
//! Git worktrees: `.git` is a FILE (not a directory) inside a linked
//! worktree (`git worktree add`, e.g. `fa2_ab.sh:5`), so the literal path
//! `<manifest_dir>/../../.git/HEAD` does not exist there. `git rev-parse
//! --git-path HEAD` resolves the REAL location (inside the worktree's own
//! private git-dir) in both the plain-repo and worktree case, and is what
//! `cargo:rerun-if-changed` watches — together with the ref file `HEAD`
//! itself points at (`git symbolic-ref -q HEAD` → `git rev-parse --git-path
//! <ref>`), so a checkout onto a different branch/commit re-runs this
//! script even when `.git/HEAD`'s own byte content does not change (a
//! detached-HEAD checkout writes the sha directly into `HEAD`, so watching
//! `HEAD` alone already covers that case; the symbolic-ref case needs the
//! second watch to catch `git commit` on a branch, which touches the ref
//! file, not `HEAD`).

use std::env;
use std::process::Command;

/// Whether `s` is exactly 40 lowercase hex characters — the ONLY shape a
/// caller-supplied `JAMMI_BUILD_SHA` is trusted verbatim under. Deliberately
/// stricter than "looks like a sha": a shallow-checkout or `pull_request`
/// merge-ref sha would still pass this shape check (it IS 40 hex chars) —
/// this function proves shape only, never provenance; see this file's
/// module doc for why no CI job in this unit relies on that distinction.
fn is_40_lowercase_hex(s: &str) -> bool {
    s.len() == 40
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Run `git` with `args` against `cwd`, returning trimmed stdout on a
/// successful (exit 0) invocation and `None` on any failure (git missing,
/// not a worktree, non-zero exit) — fail-CLOSED into the caller's own
/// `"unknown"` fallback, never a panic that would break every build in an
/// environment without `git` on `PATH` (a packaged source tarball, for
/// instance).
fn git_output(cwd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let s = s.trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// Resolve this build's git identity: the HEAD sha, with `-dirty` appended
/// when the TRACKED tree has uncommitted changes. `None` when `git` is
/// unavailable or this crate is not inside a git worktree at all (a
/// packaged/vendored source tree).
fn git_build_sha(manifest_dir: &str) -> Option<String> {
    // Watch `.git/HEAD` (worktree-correct path — P7) so a checkout moves
    // the fingerprint.
    if let Some(head_path) = git_output(manifest_dir, &["rev-parse", "--git-path", "HEAD"]) {
        println!("cargo:rerun-if-changed={head_path}");
    }
    // A symbolic HEAD (the common case, `refs/heads/<branch>`) also needs
    // its OWN ref file watched: `git commit` on a branch rewrites that ref
    // file, not `.git/HEAD` itself (which just contains `ref:
    // refs/heads/<branch>`, unchanged by a commit on that branch).
    if let Some(symref) = git_output(manifest_dir, &["symbolic-ref", "-q", "HEAD"]) {
        if let Some(ref_path) = git_output(manifest_dir, &["rev-parse", "--git-path", &symref]) {
            println!("cargo:rerun-if-changed={ref_path}");
        }
    }

    let sha = git_output(manifest_dir, &["rev-parse", "HEAD"])?;
    if !is_40_lowercase_hex(&sha) {
        // A shallow/annotated/weird ref that didn't resolve to a real
        // sha — fail closed rather than baking a non-sha string.
        return None;
    }

    // Dirtiness: TRACKED paths only (`--untracked-files=no`) — untracked
    // scratch (this tree's own `?? scratchpad/`, or any agent's temp
    // output) must never dirty a build; only uncommitted changes to
    // tracked files do.
    let dirty = git_output(
        manifest_dir,
        &["status", "--porcelain", "--untracked-files=no"],
    )
    .is_some();

    Some(if dirty { format!("{sha}-dirty") } else { sha })
}

fn main() {
    println!("cargo:rerun-if-env-changed=JAMMI_BUILD_SHA");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_default();

    let sha = env::var("JAMMI_BUILD_SHA")
        .ok()
        .filter(|s| is_40_lowercase_hex(s))
        .or_else(|| git_build_sha(&manifest_dir))
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=JAMMI_BUILD_SHA={sha}");
    println!(
        "cargo:rustc-env=JAMMI_BUILD_TARGET={}",
        env::var("TARGET").unwrap_or_default()
    );
    println!(
        "cargo:rustc-env=JAMMI_BUILD_PROFILE={}",
        env::var("PROFILE").unwrap_or_default()
    );
}
