//! Unification contract C1.6 — the mechanical proof that this binary's
//! build-time identity is genuinely baked at COMPILE time, never read at
//! run time. Driven through the REAL `jammi-bench` CLI entry point (the same
//! `env!("CARGO_BIN_EXE_jammi-bench")` pattern
//! `finetune_step_kernel_disable.rs:42` already establishes), not
//! `report::Provenance::baked()` called in-process — a Rust-level call
//! would prove nothing about the compiled BINARY's own baked env vars.
//!
//! Deliberately does NOT assert `build_sha` is 40-hex, and does NOT assert
//! it is not `"unknown"` — either shape (a genuine git worktree sha, a
//! `-dirty` suffix, or `"unknown"` on a git-less checkout) is a legitimate
//! outcome for a dev tree; this suite proves the SHAPE and the run-time
//! inertness, never a specific value (contract C1.2's own pin: "never
//! asserts 40-hex, never asserts not-unknown").

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../jammi-encoders/tests/fixtures/tiny_modernbert_head64")
}

fn provenance_command() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_jammi-bench"));
    cmd.arg("provenance");
    cmd
}

fn finetune_step_command(model_dir: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_jammi-bench"));
    cmd.args([
        "finetune-step",
        "--model-dir",
        &model_dir.to_string_lossy(),
        "--batch",
        "2",
        "--seq",
        "6",
        "--steps",
        "1",
        "--warmup",
        "0",
        "--lora-rank",
        "2",
        "--target-modules",
        "Wqkv,Wo",
    ]);
    cmd
}

/// The four fields `report::Provenance` serializes, and the shape check
/// each one must pass — never a specific value.
fn assert_provenance_shape(provenance: &serde_json::Value) {
    let build_sha = provenance["build_sha"]
        .as_str()
        .expect("build_sha present and a string");
    // LOWERCASE hex only — matches `build.rs::is_40_lowercase_hex` exactly
    // (round-2 audit advisory A5: `is_ascii_hexdigit()` accepts BOTH cases,
    // so this shape check used to accept a `build_sha` build.rs itself
    // could never actually produce).
    let is_lowercase_hex40 = |s: &str| {
        s.len() == 40
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    };
    let shape_ok = build_sha == "unknown"
        || is_lowercase_hex40(build_sha)
        || (build_sha.len() == 46
            && build_sha.ends_with("-dirty")
            && is_lowercase_hex40(&build_sha[..40]));
    assert!(
        shape_ok,
        "build_sha {build_sha:?} matches none of unknown / 40-hex / 40-hex-dirty"
    );
    assert!(
        provenance["target"].as_str().is_some_and(|s| !s.is_empty()),
        "target must be a non-empty string; provenance={provenance}"
    );
    assert!(
        provenance["profile"]
            .as_str()
            .is_some_and(|s| !s.is_empty()),
        "profile must be a non-empty string; provenance={provenance}"
    );
    let build_features = provenance["build_features"]
        .as_array()
        .expect("build_features present and an array");
    assert!(
        build_features.iter().all(|v| v.is_string()),
        "every build_features entry must be a string; provenance={provenance}"
    );
    assert_eq!(
        provenance["report_schema_version"].as_u64(),
        Some(2),
        "report_schema_version must be 2; provenance={provenance}"
    );
}

/// C1.6(a): `jammi-bench provenance` prints a standalone JSON object with
/// the shape above.
#[test]
fn provenance_subcommand_prints_the_baked_identity() {
    let output = provenance_command()
        .output()
        .expect("spawn jammi-bench provenance");
    assert!(
        output.status.success(),
        "provenance subcommand must never fail — stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let provenance: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("invalid JSON from `jammi-bench provenance`: {e}\n{stdout}"));
    assert_provenance_shape(&provenance);
}

/// C1.6(b): the SAME object (byte-for-byte, field for field) appears
/// verbatim under `report.provenance` of a real CPU `finetune-step` run —
/// one baked identity, shared by every emit site (`Report::new`), not two
/// independently-computed copies that could drift.
#[test]
fn report_carries_provenance() {
    let dir = model_dir();
    assert!(
        dir.join("config.json").exists(),
        "fixture missing: {}",
        dir.display()
    );

    let standalone_output = provenance_command()
        .output()
        .expect("spawn jammi-bench provenance");
    assert!(standalone_output.status.success());
    let standalone: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&standalone_output.stdout))
            .expect("valid JSON from `jammi-bench provenance`");

    let report_output = finetune_step_command(&dir)
        .output()
        .expect("spawn jammi-bench finetune-step");
    assert!(
        report_output.status.success(),
        "finetune-step failed — stderr={}",
        String::from_utf8_lossy(&report_output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&report_output.stdout))
            .expect("valid JSON from `jammi-bench finetune-step`");
    let embedded = report
        .get("provenance")
        .expect("report.provenance present — RED at base (key absent)");
    assert_provenance_shape(embedded);
    assert_eq!(
        embedded, &standalone,
        "report.provenance must be byte-for-byte the SAME object `jammi-bench provenance` \
         prints standalone — a mismatch here means two independently-computed copies exist"
    );
}

/// C1.6(c): run-time inertness. Re-running the ALREADY-COMPILED binary with
/// a DIFFERENT `JAMMI_BUILD_SHA` in the process environment, from a FRESH
/// `cwd` (a brand-new `git init` temp repo, never this workspace's own
/// `.git`), yields a byte-identical `provenance` object. Proves neither the
/// environment nor the filesystem is consulted at run time — every field
/// baked by `build.rs` is a compile-time `env!()` literal (the `tip_sha()`
/// defect this replaces DID re-read `git` at run time; this test is what a
/// regression back to that shape would fail).
#[test]
fn runtime_env_and_cwd_are_inert() {
    let baseline_output = provenance_command()
        .output()
        .expect("spawn jammi-bench provenance (baseline)");
    assert!(baseline_output.status.success());
    let baseline: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&baseline_output.stdout))
            .expect("valid JSON from the baseline run");

    let tmp = tempfile::tempdir().expect("create fresh temp dir");
    let init = Command::new("git")
        .args(["init", "--quiet"])
        .current_dir(tmp.path())
        .output()
        .expect("spawn git init");
    assert!(
        init.status.success(),
        "git init in the fresh temp dir failed — stderr={}",
        String::from_utf8_lossy(&init.stderr)
    );

    // A 40-hex sha that is NOT this workspace's own HEAD (all `f`s can
    // never collide with a real sha256-derived... well, sha1 hex digest by
    // construction being astronomically unlikely, and is trivially
    // distinguishable from any plausible real value by eye in a failure
    // message).
    let injected_output = provenance_command()
        .env("JAMMI_BUILD_SHA", "f".repeat(40))
        .current_dir(tmp.path())
        .output()
        .expect("spawn jammi-bench provenance (env+cwd injected)");
    assert!(
        injected_output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&injected_output.stderr)
    );
    let injected: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&injected_output.stdout))
            .expect("valid JSON from the env+cwd-injected run");

    assert_eq!(
        baseline, injected,
        "a run-time JAMMI_BUILD_SHA and a fresh cwd must change NOTHING about this ALREADY-\
         COMPILED binary's reported provenance — every field is a compile-time env!() literal, \
         never a run-time std::env::var/git read"
    );
}

/// The REAL production `build.rs` source, embedded at TEST compile time —
/// never a hand-maintained duplicate that could silently drift from what
/// this crate actually ships. Used by the two probe-crate regression tests
/// below (round-2 audit B2/B5).
const BUILD_RS_SOURCE: &str = include_str!("../build.rs");

/// Scaffold a minimal, single-member cargo workspace at `root` whose one
/// crate (`crates/probe`) links the REAL `build.rs` source verbatim, and a
/// trivial `src/main.rs` that prints the three baked env vars. Zero
/// dependencies (so `cargo build`/`run` never touches the network), and
/// mirrors this real repo's own two-levels-deep layout
/// (`<workspace_root>/crates/<name>`) exactly — `build.rs::workspace_root`
/// derives its `rerun-if-changed` watch root from that same assumption, so
/// the probe has to share it for the test to mean anything about the real
/// crate.
fn scaffold_probe_crate(root: &Path) {
    let crate_dir = root.join("crates").join("probe");
    fs::create_dir_all(crate_dir.join("src")).expect("mkdir crates/probe/src");

    fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = [\"crates/probe\"]\nresolver = \"2\"\n",
    )
    .expect("write workspace Cargo.toml");

    fs::write(
        crate_dir.join("Cargo.toml"),
        "[package]\nname = \"probe\"\nversion = \"0.0.0\"\nedition = \"2021\"\nbuild = \"build.rs\"\n\n\
         [[bin]]\nname = \"probe\"\npath = \"src/main.rs\"\n",
    )
    .expect("write probe Cargo.toml");

    fs::write(crate_dir.join("build.rs"), BUILD_RS_SOURCE).expect("write probe build.rs");

    fs::write(
        crate_dir.join("src").join("main.rs"),
        "fn main() {\n    \
         println!(\"JAMMI_BUILD_SHA={}\", env!(\"JAMMI_BUILD_SHA\"));\n    \
         println!(\"JAMMI_BUILD_TARGET={}\", env!(\"JAMMI_BUILD_TARGET\"));\n    \
         println!(\"JAMMI_BUILD_PROFILE={}\", env!(\"JAMMI_BUILD_PROFILE\"));\n\
         }\n",
    )
    .expect("write probe main.rs");
}

fn git(dir: &Path, args: &[&str]) {
    let out = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap_or_else(|e| panic!("spawn git {args:?} in {}: {e}", dir.display()));
    assert!(
        out.status.success(),
        "git {args:?} failed in {}: stderr={}",
        dir.display(),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Same as `git`, but returns trimmed stdout — for the handful of new
/// git-state fixtures below that need the resulting sha (`rev-parse HEAD`),
/// not just success/failure.
fn git_capture(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap_or_else(|e| panic!("spawn git {args:?} in {}: {e}", dir.display()));
    assert!(
        out.status.success(),
        "git {args:?} failed in {}: stderr={}",
        dir.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// `git commit` with a fixed, deterministic identity (this suite never
/// depends on the ambient environment having `user.email`/`user.name`
/// configured — several CI images intentionally don't).
fn git_commit(dir: &Path, args: &[&str]) {
    let mut full = vec![
        "-c",
        "user.email=probe@example.com",
        "-c",
        "user.name=probe",
    ];
    full.extend_from_slice(args);
    git(dir, &full);
}

/// Build AND run the probe crate's `probe` binary, returning its stdout.
/// `--offline`: zero dependencies means this never legitimately needs the
/// network, so this stays hermetic even off a real network. A dedicated
/// `CARGO_TARGET_DIR` inside `root` keeps this fully isolated from the
/// shared workspace target dir the REST of this test suite's own
/// `env!("CARGO_BIN_EXE_jammi-bench")` binary was built into.
fn cargo_run_probe(root: &Path, extra_env: &[(&str, &str)]) -> String {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let mut cmd = Command::new(cargo);
    cmd.args(["run", "--quiet", "--offline"])
        .current_dir(root.join("crates").join("probe"))
        .env("CARGO_TARGET_DIR", root.join("target"))
        // Never let THIS test process's own environment leak a
        // JAMMI_BUILD_SHA into the probe build unless the caller asked for
        // one — the probe's own precedence must be exercised cleanly.
        .env_remove("JAMMI_BUILD_SHA");
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("spawn cargo run for the probe crate: {e}"));
    assert!(
        out.status.success(),
        "cargo run (probe crate) failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).to_string()
}

/// Runs `cargo build -v --offline` for the probe crate at `root` and reports
/// whether the verbose transcript shows cargo actually INVOKING the compiled
/// build-script binary (`Running \`.../build-script-build\``) — the one
/// cargo-verbose signal that distinguishes "the script reran" from "cargo
/// reused the cached `cargo:rustc-env` output from a prior run" (empirically
/// confirmed: a build script watching an unchanged file prints this line on
/// the first build and NOT on an unchanged second build; the false-positive
/// direction — printing it on every build — is what state (b)'s
/// nonexistent-loose-ref window causes, and is a documented cost, not a
/// correctness bug; see `packed_ref_repo_still_tracks_the_next_commit`).
fn cargo_build_verbose_ran_build_script(root: &Path) -> bool {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let out = Command::new(cargo)
        .args(["build", "-v", "--offline"])
        .current_dir(root.join("crates").join("probe"))
        .env("CARGO_TARGET_DIR", root.join("target"))
        .env_remove("JAMMI_BUILD_SHA")
        .output()
        .unwrap_or_else(|e| panic!("spawn cargo build -v for the probe crate: {e}"));
    assert!(
        out.status.success(),
        "cargo build -v (probe crate) failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    combined.contains("build-script-build`")
}

fn parse_field(output: &str, key: &str) -> String {
    output
        .lines()
        .find_map(|l| l.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("no {key}= line in probe stdout: {output:?}"))
        .to_string()
}

/// Round-2 audit B2 (class A, hard red) turned into a standing regression
/// test — the auditor's own probe rows 3/4 reproduced here: a scratch probe
/// crate (real `build.rs` source, `scaffold_probe_crate`'s own doc) is
/// committed clean, built once (must NOT be `-dirty`), then a TRACKED file
/// is edited WITHOUT committing and the SAME crate is rebuilt. Before the
/// fix (watching only `.git/HEAD` + the branch ref), this second build
/// bakes the STALE clean sha — the crate recompiles (the source changed)
/// but the build SCRIPT never reruns, so nothing recomputes dirtiness. This
/// test is RED against that shape and GREEN once `build.rs` also watches
/// the workspace's `crates/` directory (this file's own module doc, and
/// `build.rs`'s "The `-dirty` staleness bug this file fixes" section, name
/// the fix).
#[test]
fn edited_tracked_file_forces_dirty_on_rebuild() {
    let tmp = tempfile::tempdir().expect("create scratch workspace");
    let root = tmp.path();
    scaffold_probe_crate(root);

    git(root, &["init", "--quiet"]);
    git(
        root,
        &[
            "-c",
            "user.email=probe@example.com",
            "-c",
            "user.name=probe",
            "add",
            "-A",
        ],
    );
    git(
        root,
        &[
            "-c",
            "user.email=probe@example.com",
            "-c",
            "user.name=probe",
            "commit",
            "--quiet",
            "-m",
            "initial",
        ],
    );

    let baseline = cargo_run_probe(root, &[]);
    let baseline_sha = parse_field(&baseline, "JAMMI_BUILD_SHA");
    assert!(
        !baseline_sha.ends_with("-dirty"),
        "baseline (freshly committed, nothing edited) build must NOT be dirty: {baseline_sha:?}"
    );

    // Edit a TRACKED file WITHOUT committing — the exact shape the auditor's
    // probe rows 3/4 exercised.
    let main_rs = root
        .join("crates")
        .join("probe")
        .join("src")
        .join("main.rs");
    let mut content = fs::read_to_string(&main_rs).expect("read probe main.rs");
    content.push_str("\n// edited, uncommitted — forces a recompile without a new commit\n");
    fs::write(&main_rs, content).expect("edit probe main.rs");

    let rebuilt = cargo_run_probe(root, &[]);
    let rebuilt_sha = parse_field(&rebuilt, "JAMMI_BUILD_SHA");
    assert!(
        rebuilt_sha.ends_with("-dirty"),
        "editing a TRACKED file without committing, then rebuilding, must bake `-dirty` — got \
         {rebuilt_sha:?}. A `build.rs` that watches only `.git/HEAD` + the branch ref for \
         `rerun-if-changed` FAILS this: the crate recompiles (the source changed) but the build \
         script itself never reruns, so the STALE clean sha from the first build gets reused \
         (round-2 audit B2 — the auditor's probe rows 3/4)."
    );
    assert_eq!(
        rebuilt_sha.trim_end_matches("-dirty"),
        baseline_sha,
        "the sha PREFIX must be unchanged (no new commit happened between the two builds) — only \
         the -dirty suffix should differ"
    );
}

/// Contract §6 F3's second half, end-to-end: `JAMMI_BUILD_SHA=deadbeef` (not
/// 40-hex — falls through the shape check) in a workspace with NO `.git` at
/// all (the tarball case `build.rs`'s module doc names) must bake the
/// literal `"unknown"`. `build_rs_unit.rs`'s `deadbeef_falls_through_shape_
/// check` proves the FIRST half of this (the shape check itself, as a pure
/// unit test); this test proves the SAME claim through a REAL `cargo build`
/// invocation of the actual production `build.rs`, closing the gap between
/// "the function says no" and "the binary this repo ships actually bakes
/// `unknown`".
#[test]
fn deadbeef_env_var_falls_through_to_unknown_when_git_is_unavailable() {
    let tmp = tempfile::tempdir().expect("create scratch workspace");
    let root = tmp.path();
    scaffold_probe_crate(root);
    // Deliberately NO `git init` — a git-less tree.

    let out = cargo_run_probe(root, &[("JAMMI_BUILD_SHA", "deadbeef")]);
    let sha = parse_field(&out, "JAMMI_BUILD_SHA");
    assert_eq!(
        sha, "unknown",
        "JAMMI_BUILD_SHA=deadbeef (8 chars, not 40-hex) must fall through the env-var shape \
         check, and with no git available at all must resolve to the literal \"unknown\""
    );
}

// ---------------------------------------------------------------------------
// Unification contract §6 F3, class closure (esc: `.github/workflows/ci.yml`
// run 32995714545 — build-sha hermeticity leg RED). The class is "git states
// in which a new commit changes none of build.rs's rerun-if-changed targets
// (or changes them in a way cargo does not see)". Each state below is a
// fixture pinning WHAT build.rs watches and WHAT a commit in that state
// actually moves; `ci.yml`'s own F3 leg reproduces state (a) for real (the
// one state that needs an actual `actions/checkout` clone, per that step's
// own comment), the rest are covered here as fast local regression tests.
//
//   (a) detached HEAD        — `detached_head_commit_moves_baked_sha_and_forces_rerun`
//   (b) packed refs          — `packed_ref_repo_still_tracks_the_next_commit`
//   (c) linked worktree      — `linked_worktree_commit_moves_baked_sha_and_forces_rerun`
//   (d) commit outside watch — `commit_outside_watched_roots_still_moves_baked_sha`
//   (e) amend                — `amend_moves_baked_sha_and_forces_rerun` (a
//       rebase ends the same way: a sequence of resets/commits that leaves
//       HEAD/the branch ref pointing at a new sha — mechanically identical
//       to what amend exercises here, so it is not duplicated as a third
//       near-identical integration test)
//   (f) mtime granularity    — engineered in `ci.yml`'s F3 step via `sleep 1`
//       between reading `A` and the commit (a filesystem-clock property, not
//       a `build.rs` decision, so there is no local `#[test]` for it — see
//       that step's own comment)
//   (g) stale-source reuse   — ruled out STRUCTURALLY by every test in this
//       block asserting the exact rebuilt VALUE (not just "it changed") and
//       by `build.rs::git_build_sha` never reading a ref FILE's bytes as its
//       sha source (only `git rev-parse HEAD`, live, every build) — a
//       regression to "bake whatever the ref file's content looked like"
//       would fail every `_and_forces_rerun` assertion below on the exact
//       resulting sha, not merely on whether a rebuild happened
//
// Must-still-count (the leg's other half):
// `clean_rebuild_with_no_git_change_does_not_rerun_build_script` pins that a
// rebuild with NOTHING changed must NOT rerun `build.rs` at all.
// ---------------------------------------------------------------------------

/// State (a): `actions/checkout@v4` leaves a CI job's clone in a DETACHED
/// HEAD — `.git/HEAD` holds a bare sha directly, there is no branch ref file
/// to watch instead (`build.rs`'s own module doc names this exact state).
/// This is the LOCAL proxy for `ci.yml`'s own F3 step (a fast scratch repo,
/// not a real `actions/checkout` clone) — the mutation-check vehicle: make
/// `build.rs` stop watching `.git/HEAD` and this test goes red.
#[test]
fn detached_head_commit_moves_baked_sha_and_forces_rerun() {
    let tmp = tempfile::tempdir().expect("create scratch workspace");
    let root = tmp.path();
    scaffold_probe_crate(root);
    git(root, &["init", "--quiet", "-b", "main"]);
    git_commit(root, &["add", "-A"]);
    git_commit(root, &["commit", "--quiet", "-m", "initial"]);
    let first_head = git_capture(root, &["rev-parse", "HEAD"]);

    git(root, &["checkout", "--quiet", "--detach", &first_head]);
    let symref = Command::new("git")
        .args(["symbolic-ref", "-q", "HEAD"])
        .current_dir(root)
        .output()
        .expect("spawn git symbolic-ref");
    assert!(
        !symref.status.success(),
        "HEAD must be detached (no symbolic ref) for this fixture to mean anything"
    );

    let baseline = cargo_run_probe(root, &[]);
    let baseline_sha = parse_field(&baseline, "JAMMI_BUILD_SHA");
    assert_eq!(
        baseline_sha, first_head,
        "a detached-HEAD build must resolve the checked-out commit's own sha"
    );

    // The exact ci.yml F3-leg move: a real commit while detached rewrites
    // `.git/HEAD` ITSELF (no branch ref exists to watch instead).
    git_commit(
        root,
        &["commit", "--quiet", "--allow-empty", "-m", "detached probe"],
    );
    let new_head = git_capture(root, &["rev-parse", "HEAD"]);
    assert_ne!(new_head, first_head, "fixture invalid: HEAD did not move");

    let rebuilt = cargo_run_probe(root, &[]);
    let rebuilt_sha = parse_field(&rebuilt, "JAMMI_BUILD_SHA");
    assert_eq!(
        rebuilt_sha.trim_end_matches("-dirty"),
        new_head,
        "a real commit in a DETACHED HEAD state — the exact state ci.yml's F3 leg's own CI \
         checkout is in — must move the baked sha to the new HEAD"
    );
}

/// State (b): `.git/refs/heads/<branch>` is absent (folded into
/// `.git/packed-refs`) until a commit writes a fresh loose ref. Cargo's own
/// `rerun-if-changed` fingerprint treats a WATCHED PATH THAT DOES NOT EXIST
/// as always-stale (confirmed empirically against real cargo, not assumed:
/// a build script watching a nonexistent path reruns on every subsequent
/// build, never caching "clean") — so a packed branch ref is naturally safe
/// by construction: worst case it reruns `build.rs` more than strictly
/// necessary, never less. This test pins that BOTH halves hold: packing
/// alone changes nothing, and a commit landing on a previously-packed branch
/// still forces the sha forward.
#[test]
fn packed_ref_repo_still_tracks_the_next_commit() {
    let tmp = tempfile::tempdir().expect("create scratch workspace");
    let root = tmp.path();
    scaffold_probe_crate(root);
    git(root, &["init", "--quiet", "-b", "main"]);
    git_commit(root, &["add", "-A"]);
    git_commit(root, &["commit", "--quiet", "-m", "initial"]);

    let baseline = cargo_run_probe(root, &[]);
    let baseline_sha = parse_field(&baseline, "JAMMI_BUILD_SHA");

    git(root, &["pack-refs", "--all", "--prune"]);
    let loose_ref = root.join(".git").join("refs").join("heads").join("main");
    assert!(
        !loose_ref.exists(),
        "fixture invalid: pack-refs --prune should have removed the loose ref file at {}",
        loose_ref.display()
    );

    let after_pack = cargo_run_probe(root, &[]);
    assert_eq!(
        parse_field(&after_pack, "JAMMI_BUILD_SHA"),
        baseline_sha,
        "packing refs alone (no content change) must not change the resolved sha"
    );

    git_commit(
        root,
        &["commit", "--quiet", "--allow-empty", "-m", "second"],
    );
    let new_head = git_capture(root, &["rev-parse", "HEAD"]);
    assert!(
        loose_ref.exists(),
        "fixture invalid: a commit on a packed branch must (re)create the loose ref file"
    );

    let rebuilt = cargo_run_probe(root, &[]);
    let rebuilt_sha = parse_field(&rebuilt, "JAMMI_BUILD_SHA");
    assert_eq!(
        rebuilt_sha.trim_end_matches("-dirty"),
        new_head,
        "a commit on a previously-packed branch must still move the baked sha to the new HEAD"
    );
}

/// State (c): a linked worktree (`.git` is a FILE, not a directory —
/// `build.rs`'s module doc names this exact shape, citing `fa2_ab.sh:5`; the
/// agent-worktree layout this very repo uses for parallel work is this
/// state, checked out on a NEW branch rather than detached). Confirms
/// `git rev-parse --git-path HEAD` resolves the worktree's OWN private
/// `HEAD` (so a commit made INSIDE the worktree is seen) while
/// `--git-path refs/heads/<branch>` still resolves into the SHARED main
/// repo's `.git` dir (branch refs are not per-worktree) — both paths are
/// real, existing files cargo can watch either way.
#[test]
fn linked_worktree_commit_moves_baked_sha_and_forces_rerun() {
    let tmp = tempfile::tempdir().expect("create scratch workspace");
    let root = tmp.path().join("main");
    fs::create_dir_all(&root).expect("mkdir main checkout");
    scaffold_probe_crate(&root);
    git(&root, &["init", "--quiet", "-b", "main"]);
    git_commit(&root, &["add", "-A"]);
    git_commit(&root, &["commit", "--quiet", "-m", "initial"]);

    let baseline = cargo_run_probe(&root, &[]);
    let baseline_sha = parse_field(&baseline, "JAMMI_BUILD_SHA");
    assert!(!baseline_sha.ends_with("-dirty"));

    let worktree = tmp.path().join("wt");
    git(
        &root,
        &[
            "worktree",
            "add",
            "--quiet",
            "-b",
            "probe-wt",
            worktree.to_str().expect("worktree path is valid utf-8"),
        ],
    );
    let dot_git = worktree.join(".git");
    assert!(
        dot_git.is_file(),
        "fixture invalid: `.git` inside a linked worktree must be a FILE (pointer), not a \
         directory — got a non-file at {}",
        dot_git.display()
    );

    let wt_baseline = cargo_run_probe(&worktree, &[]);
    let wt_baseline_sha = parse_field(&wt_baseline, "JAMMI_BUILD_SHA");
    assert_eq!(
        wt_baseline_sha, baseline_sha,
        "a fresh linked-worktree build must resolve the SAME HEAD sha as the checkout it was \
         created from"
    );

    git_commit(
        &worktree,
        &[
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "second (worktree)",
        ],
    );
    let new_head = git_capture(&worktree, &["rev-parse", "HEAD"]);

    let wt_rebuilt = cargo_run_probe(&worktree, &[]);
    let wt_rebuilt_sha = parse_field(&wt_rebuilt, "JAMMI_BUILD_SHA");
    assert_eq!(
        wt_rebuilt_sha.trim_end_matches("-dirty"),
        new_head,
        "a commit made INSIDE a linked worktree must move THAT worktree's own baked sha to its \
         own new HEAD — `--git-path HEAD` must resolve the worktree-private HEAD, not the main \
         checkout's"
    );
}

/// State (d): a commit that touches ONLY a file outside the dirtiness
/// query's own pathspec (`:/crates`, `:/Cargo.lock`, `:/Cargo.toml`) must
/// still move the baked sha — the sha-tracking watch (`.git/HEAD` + branch
/// ref) is a separate mechanism from the dirtiness query, not gated by it.
#[test]
fn commit_outside_watched_roots_still_moves_baked_sha() {
    let tmp = tempfile::tempdir().expect("create scratch workspace");
    let root = tmp.path();
    scaffold_probe_crate(root);
    fs::write(root.join("NOTES.md"), "v1\n").expect("write NOTES.md");
    git(root, &["init", "--quiet", "-b", "main"]);
    git_commit(root, &["add", "-A"]);
    git_commit(root, &["commit", "--quiet", "-m", "initial"]);

    let baseline = cargo_run_probe(root, &[]);
    let baseline_sha = parse_field(&baseline, "JAMMI_BUILD_SHA");
    assert!(!baseline_sha.ends_with("-dirty"));

    // A commit to NOTES.md ONLY — outside crates/Cargo.lock/Cargo.toml, so
    // the dirtiness QUERY's own pathspec never sees it either.
    fs::write(root.join("NOTES.md"), "v2\n").expect("edit NOTES.md");
    git_commit(root, &["add", "NOTES.md"]);
    git_commit(root, &["commit", "--quiet", "-m", "notes v2"]);
    let new_head = git_capture(root, &["rev-parse", "HEAD"]);

    let rebuilt = cargo_run_probe(root, &[]);
    let rebuilt_sha = parse_field(&rebuilt, "JAMMI_BUILD_SHA");
    assert!(
        !rebuilt_sha.ends_with("-dirty"),
        "committing NOTES.md leaves crates/Cargo.lock/Cargo.toml clean, so this must NOT be \
         dirty: {rebuilt_sha:?}"
    );
    assert_eq!(
        rebuilt_sha, new_head,
        "a commit that touches ONLY files outside the watched crates/Cargo.lock/Cargo.toml \
         roots must still move the baked sha — the branch-ref watch is independent of the \
         dirtiness pathspec"
    );
}

/// State (e): `git commit --amend` rewrites the branch ref to a NEW commit
/// object (a different sha even when the tree is unchanged, since the
/// message/timestamp differ) without going through an ordinary fast-forward
/// commit. A `rebase` ends the same way (HEAD/the branch ref pointing at a
/// new sha via a sequence of resets/commits) — mechanically identical from
/// `build.rs`'s perspective, so it is not exercised as a separate test.
#[test]
fn amend_moves_baked_sha_and_forces_rerun() {
    let tmp = tempfile::tempdir().expect("create scratch workspace");
    let root = tmp.path();
    scaffold_probe_crate(root);
    git(root, &["init", "--quiet", "-b", "main"]);
    git_commit(root, &["add", "-A"]);
    git_commit(root, &["commit", "--quiet", "-m", "initial"]);

    let baseline = cargo_run_probe(root, &[]);
    let baseline_sha = parse_field(&baseline, "JAMMI_BUILD_SHA");

    git_commit(
        root,
        &[
            "commit",
            "--quiet",
            "--amend",
            "--allow-empty",
            "-m",
            "amended",
        ],
    );
    let new_head = git_capture(root, &["rev-parse", "HEAD"]);
    assert_ne!(
        new_head,
        baseline_sha.trim_end_matches("-dirty"),
        "fixture invalid: amend must produce a different commit object"
    );

    let rebuilt = cargo_run_probe(root, &[]);
    let rebuilt_sha = parse_field(&rebuilt, "JAMMI_BUILD_SHA");
    assert_eq!(
        rebuilt_sha.trim_end_matches("-dirty"),
        new_head,
        "amending the last commit must move the baked sha to the amended HEAD"
    );
}

/// Must-still-count — the F3 leg's other half: a clean rebuild with no
/// commit and no tracked-file edit must NOT rerun `build.rs` at all.
/// Distinguished from "reused the same cached VALUE" (which a broken
/// build.rs that recomputed but happened to get the same answer would also
/// pass) by observing cargo's own verbose transcript for the literal
/// `Running \`.../build-script-build\`` line.
#[test]
fn clean_rebuild_with_no_git_change_does_not_rerun_build_script() {
    let tmp = tempfile::tempdir().expect("create scratch workspace");
    let root = tmp.path();
    scaffold_probe_crate(root);
    git(root, &["init", "--quiet", "-b", "main"]);
    git_commit(root, &["add", "-A"]);
    git_commit(root, &["commit", "--quiet", "-m", "initial"]);

    assert!(
        cargo_build_verbose_ran_build_script(root),
        "the FIRST build of a fresh crate must invoke the build script"
    );
    assert!(
        !cargo_build_verbose_ran_build_script(root),
        "a rebuild with NOTHING changed (no commit, no tracked-file edit) must NOT rerun \
         build.rs — cargo must reuse the cached JAMMI_BUILD_SHA/TARGET/PROFILE from the prior \
         run"
    );
}
