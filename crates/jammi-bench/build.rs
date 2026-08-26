//! Bakes this binary's build-time identity into compile-time environment
//! variables the crate reads back with `env!()` — never `std::env::var()` at
//! run time (unification contract C1). Three literals, `cargo:rustc-env`'d
//! once per build:
//!
//!   - `JAMMI_BUILD_SHA` — `<sha>`, `<sha>-dirty`, or `"unknown"`.
//!   - `JAMMI_BUILD_TARGET` — Cargo's own `$TARGET` for this build, or
//!     `"unknown"` if that env var was unset or empty.
//!   - `JAMMI_BUILD_PROFILE` — Cargo's own `$PROFILE` for this build, or
//!     `"unknown"` under the same fallback.
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
//! Round-2 audit (advisory A1) — the CONSEQUENCE a valid override has: a
//! caller-supplied `JAMMI_BUILD_SHA` that passes the 40-lowercase-hex shape
//! check is baked VERBATIM, with NO `-dirty` suffix EVER appended — dirtiness
//! is a fact this file establishes only through its OWN `git status` read,
//! which the override path skips entirely. This is contract-sanctioned (C1.1
//! names exactly this precedence, and C16.3's planned server image
//! `JAMMI_BUILD_SHA=${{ github.sha }}` relies on it — a CI checkout of a
//! specific, known-clean sha, where "was the checkout tree itself dirty" is
//! not a question that sha can answer). The consequence: an override sha
//! that DISAGREES with what `git` would have resolved (a stale/wrong value a
//! caller passed by mistake) is baked with no cross-check against git AT
//! ALL, clean or dirty — a phase-2 producer's `provenance.build_sha == $SHA`
//! comparison (contract C5.1) is what has to catch that mismatch, not this
//! file; `build.rs` only ever answers "is the value I was GIVEN shaped like
//! a sha", never "is the value I was given TRUE".
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
//!
//! ## The `-dirty` staleness bug this file fixes (round-2 audit, B2)
//!
//! Watching ONLY `.git/HEAD` and the branch ref (as this file did before
//! this fix) is necessary but not sufficient: Cargo reruns a build script
//! exactly when one of its declared `rerun-if-changed` targets changes (or
//! on a fresh build) — it does NOT rerun the build script merely because
//! the CRATE itself is being recompiled. Editing a tracked file
//! (`src/main.rs`, say) WITHOUT committing forces cargo to recompile
//! `jammi-bench` (the source changed), but touches neither `.git/HEAD` nor
//! the branch ref — so `build.rs` does NOT rerun, and the binary bakes
//! whatever `build_sha` the LAST build script invocation computed, which
//! can be the CLEAN value from before the edit. Reproduced (audit probe
//! rows 3/4): same tree, same commit, two builds — one via a forced
//! build-script rerun (env var toggle), one via only a tracked-source edit
//! — produced `<sha>-dirty` and bare `<sha>` respectively. A phase-2
//! producer's `provenance.build_sha == $SHA` cross-check (contract C5.1/
//! C6.4) would PASS on a binary built from an edited, uncommitted tree —
//! the exact stale-binary hazard the unit exists to close.
//!
//! Fix: ALSO watch the tracked SOURCE the dirtiness computation itself
//! depends on — the workspace's `crates/` directory (recursive; Cargo
//! resolves a directory `rerun-if-changed` target by walking it for the
//! most-recently-modified file — an mtime `stat()` walk, not a content
//! read, so the cost is proportional to file COUNT, not byte size) plus
//! `Cargo.lock`/`Cargo.toml` at the workspace root. `workspace_root()`
//! derives the watch root from `CARGO_MANIFEST_DIR`'s own filesystem
//! position (`<root>/crates/<name>`) rather than a `git` call, so the
//! watch is emitted unconditionally, even in the git-less tarball case
//! where `git_build_sha` itself returns `None`.
//!
//! **What `-dirty` means, exactly (round-3 audit advisory 2).** The
//! dirtiness QUERY (`git_build_sha`'s own `git status`) is pathspec-
//! restricted to `:/crates`, `:/Cargo.lock`, `:/Cargo.toml` — the SAME
//! three targets the watch above names, not the whole repository. The two
//! scopes are kept identical DELIBERATELY: if the query covered more than
//! the watch (the whole repo, as an earlier revision of this file did), a
//! dirty file OUTSIDE the watched set (a `docs/` edit, say) would bake
//! `-dirty` on THAT build, but committing that same edit later touches
//! nothing the watch covers — no rerun happens, and the now-STALE
//! `-dirty` marker survives past the point the tree actually went clean
//! (the SAME staleness SHAPE the retrigger bug above has, just running in
//! the opposite direction: falsely dirty instead of falsely clean). So:
//! `-dirty` means "this binary's own build-relevant tree — `crates/`,
//! `Cargo.lock`, `Cargo.toml` — has an uncommitted change to a TRACKED
//! file", never "the whole monorepo checkout is dirty" and never "an
//! UNTRACKED file exists anywhere" (`--untracked-files=no`).
//!
//! `provenance_baked.rs`'s `edited_tracked_file_forces_dirty_on_rebuild`
//! test is the regression proof: a scratch probe crate is built once
//! (clean), a tracked file is edited WITHOUT committing, the SAME crate is
//! rebuilt, and the second build's baked sha MUST carry `-dirty` — this is
//! the auditor's probe rows 3/4 turned into a standing CI assertion.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Whether `s` is exactly 40 lowercase hex characters — the ONLY shape a
/// caller-supplied `JAMMI_BUILD_SHA` is trusted verbatim under. Deliberately
/// stricter than "looks like a sha": a shallow-checkout or `pull_request`
/// merge-ref sha would still pass this shape check (it IS 40 hex chars) —
/// this function proves shape only, never provenance; see this file's
/// module doc for why no CI job in this unit relies on that distinction.
pub(crate) fn is_40_lowercase_hex(s: &str) -> bool {
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

/// This crate's own filesystem position is always `<workspace_root>/crates/
/// <name>` in every environment this file's module doc names (main
/// checkout, linked worktree, bundle clone, git-less tarball) — a
/// FILESYSTEM fact, not a `git` one, so it resolves even when `git` is
/// entirely unavailable (the tarball case, where `git_build_sha` below
/// returns `None` but the `rerun-if-changed` watches below still need a
/// root to point at). Falls back to `manifest_dir` itself (watching only
/// this one crate) if the directory structure is ever unexpectedly
/// shallow — never panics.
pub(crate) fn workspace_root(manifest_dir: &str) -> PathBuf {
    let manifest_path = Path::new(manifest_dir);
    manifest_path
        .parent() // crates/
        .and_then(Path::parent) // <workspace_root>
        .map(Path::to_path_buf)
        .unwrap_or_else(|| manifest_path.to_path_buf())
}

/// Resolve this build's git identity: the HEAD sha, with `-dirty` appended
/// when the TRACKED tree has uncommitted changes. `None` when `git` is
/// unavailable, this crate is not inside a git worktree at all (a
/// packaged/vendored source tree), OR the dirtiness check itself could not
/// be run (fail-closed — see this function's `dirty` binding: a `git
/// status` failure must never be silently read as "clean").
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
    // tracked files do. Round-2 audit (advisory A2): the earlier form
    // (`git_output(..).is_some()`) collapsed a FAILED `git status`
    // invocation into "clean" — indistinguishable from "ran and found
    // nothing dirty". Run the command directly here so a failed exit
    // status propagates `None` all the way out to `"unknown"` instead of
    // being silently read as clean, the same fail-closed posture the sha
    // resolution above already takes.
    //
    // Round-3 audit (advisory 2): the QUERY is pathspec-restricted to
    // EXACTLY what `main()`'s `rerun-if-changed` WATCHES — `:/crates`,
    // `:/Cargo.lock`, `:/Cargo.toml` (the `:/` magic pathspec means "from
    // the top of the working tree", so this resolves correctly regardless
    // of `cwd` = `manifest_dir`, a nested subdirectory). Before this fix
    // the query covered the WHOLE repo (no pathspec) while the watch
    // covered only this narrower set — the two scopes disagreeing is
    // itself a staleness hazard in the OTHER direction from B2: a dirty
    // file OUTSIDE `crates/`/`Cargo.lock`/`Cargo.toml` (a `docs/` edit,
    // say) would bake `-dirty` on ITS OWN build, but committing that same
    // edit later would never trigger a rerun (nothing the watch covers
    // changed), leaving a now-STALE `-dirty` baked past the point the tree
    // actually went clean. Restricting the query to the watch's own scope
    // makes the two facts move together by construction: any tree state
    // that flips this query's answer is, by definition, a tree state that
    // also would have re-run this script.
    let status_ran = Command::new("git")
        .args([
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            ":/crates",
            ":/Cargo.lock",
            ":/Cargo.toml",
        ])
        .current_dir(manifest_dir)
        .output()
        .ok()?;
    if !status_ran.status.success() {
        return None;
    }
    let dirty = !String::from_utf8_lossy(&status_ran.stdout)
        .trim()
        .is_empty();

    Some(if dirty { format!("{sha}-dirty") } else { sha })
}

/// Read `var`, falling back to `"unknown"` when it is unset OR empty —
/// round-2 audit (advisory A3): `unwrap_or_default()` used to bake `""`,
/// which `report::assert_identity_fields_present` could not tell apart from
/// a genuine value (a `NonNull` field is only checked for JSON `null`, and
/// `""` is not `null`). Cargo always sets `TARGET`/`PROFILE` for a real
/// build-script invocation, so this fallback only ever fires under a
/// non-standard invocation (this file executed directly, or under a
/// stripped-down probe environment) — fail-closed there rather than baking
/// a blank string a downstream reader could mistake for real build
/// metadata.
pub(crate) fn env_or_unknown(var: &str) -> String {
    env::var(var)
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn main() {
    println!("cargo:rerun-if-env-changed=JAMMI_BUILD_SHA");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_default();

    // The B2 fix: watch the tracked SOURCE the dirtiness computation reads,
    // not only the git-ref plumbing below — see this file's module doc.
    // Emitted UNCONDITIONALLY (before any `git` call), so it holds even in
    // the git-less tarball case.
    let root = workspace_root(&manifest_dir);
    println!("cargo:rerun-if-changed={}", root.join("crates").display());
    println!(
        "cargo:rerun-if-changed={}",
        root.join("Cargo.lock").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("Cargo.toml").display()
    );

    let sha = env::var("JAMMI_BUILD_SHA")
        .ok()
        .filter(|s| is_40_lowercase_hex(s))
        .or_else(|| git_build_sha(&manifest_dir))
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=JAMMI_BUILD_SHA={sha}");
    println!(
        "cargo:rustc-env=JAMMI_BUILD_TARGET={}",
        env_or_unknown("TARGET")
    );
    println!(
        "cargo:rustc-env=JAMMI_BUILD_PROFILE={}",
        env_or_unknown("PROFILE")
    );
}
