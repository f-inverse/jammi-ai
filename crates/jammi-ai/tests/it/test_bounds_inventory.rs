//! #527/#567/#578: an ENUMERATING oracle over every literal wall-clock bound
//! FORM in this crate's four test targets (`it`, `distributed`,
//! `gpu_capability`, `metal_quantized_gpu`) — not a sample, and not scoped
//! to "training progress" at the SCAN step (that judgement is not
//! syntactically decidable), but at the REVIEW step: every found site is
//! matched against [`REVIEWED_SITES`], a hand-reviewed classification into
//! one of four checkable classes:
//!
//! * `Class::A` — an OBSERVED-EVENT rendezvous
//!   (`jammi_ai::fine_tune::worker::loop_test_hooks::{arm_observed,
//!   Event}`, or the pre-existing `arm_pause_before_spawn_blocking` oneshot)
//!   now gates the property; the literal bound this site names is only the
//!   backstop wrapping that rendezvous. Checked: the site's own text window
//!   names the rendezvous.
//! * `Class::B` — DERIVED from the config the test itself set
//!   (lease/heartbeat/poll cadence). Checked: the site's own text window
//!   names the config identifier the bound derives from.
//! * `Class::C` — a generous BACKSTOP against a wedged or starved machine.
//!   Checked: the site's own text window contains the fixed sentence
//!   `"wedged or starved machine"`.
//! * `Class::D` — NOT a training-progress wait at all (GPU scheduling,
//!   cache reload, admission/claim-loop state, catalog listing, worker
//!   shutdown mechanics already gated by this crate's OWN internal
//!   rendezvous points, data-streaming deadlock guards, fault-injection
//!   timing). No text predicate — only a one-line reason, always present.
//!
//! **Honesty about the predicate.** The scan step is a plain substring
//! search over raw source lines (`timeout(Duration::from_secs(`,
//! `sleep(Duration::from_secs(`, `Instant::now() + Duration::from_secs(`,
//! `: Duration = Duration::from_secs(`) — syntactic, not AST-aware, and
//! stated as such: it cannot tell a bound inside a doc comment from one in
//! live code (none of today's sites happen to collide with that), and it
//! cannot itself decide "is this training progress" (that is exactly what
//! [`REVIEWED_SITES`] is for). `from_millis` poll cadences are excluded BY
//! RULE — the pattern only ever matches `from_secs(`.
//!
//! A new literal bound anywhere in the four targets is either missing from
//! [`REVIEWED_SITES`] (the [`every_literal_wall_clock_bound_is_reviewed`]
//! assertion fails, naming the file:line) or a REMOVED/MOVED site makes a
//! stale entry fail to match (the same assertion's other arm) — so this
//! file must be updated in the SAME commit as any new/moved/removed bound.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/jammi-ai has two ancestors: crates/, then the repo root")
        .to_path_buf()
}

/// Every git-TRACKED `.rs` file under `root`-relative `pathspec` (a
/// directory or a single file), sorted. Mirrors `pinned_source_gate.rs`'s
/// `tracked_rs_files` (duplicated rather than imported: that file's helper
/// is a private `fn`, and the two gates' universes are allowed to diverge).
fn tracked_rs_files(root: &Path, pathspec: &str) -> Vec<String> {
    let output = Command::new("git")
        .args([
            "-C",
            root.to_str().expect("utf8 repo root"),
            "ls-files",
            "--",
            pathspec,
        ])
        .output()
        .expect("spawn git ls-files");
    assert!(
        output.status.success(),
        "git ls-files -- {pathspec} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // THIS file's own doc comments and `FORMS`/`REVIEWED_SITES` literals
    // necessarily spell out the substrings being scanned for -- excluded
    // from its own universe, or it would find (and need to review) itself.
    let mut files: Vec<String> = String::from_utf8(output.stdout)
        .expect("utf8 git ls-files output")
        .lines()
        .filter(|line| line.ends_with(".rs") && !line.ends_with("/test_bounds_inventory.rs"))
        .map(str::to_string)
        .collect();
    files.sort();
    files
}

/// The four test-target roots this oracle's universe covers (`ci.yml`'s
/// hermetic lane runs `it`; `distributed`/`gpu_capability` are gated
/// live lanes; `metal_quantized_gpu` is `Cargo.toml`'s fourth `[[test]]`).
const TARGET_ROOTS: &[&str] = &[
    "crates/jammi-ai/tests/it",
    "crates/jammi-ai/tests/distributed",
    "crates/jammi-ai/tests/gpu_capability",
    "crates/jammi-ai/tests/metal_quantized_gpu.rs",
];

/// The four literal-bound FORMS this oracle scans for, as a substring to
/// search each tracked line for. `from_millis` cadences never match any of
/// these (excluded by rule, per this file's own doc).
const FORMS: &[&str] = &[
    "timeout(Duration::from_secs(",
    "sleep(Duration::from_secs(",
    "Instant::now() + Duration::from_secs(",
    ": Duration = Duration::from_secs(",
];

/// One found (file, 1-based line) pair, scanning every [`TARGET_ROOTS`]
/// file for every [`FORMS`] substring.
fn scan_universe() -> BTreeSet<(String, u32)> {
    let root = repo_root();
    let mut found = BTreeSet::new();
    for target in TARGET_ROOTS {
        let files = tracked_rs_files(&root, target);
        assert!(
            !files.is_empty(),
            "git ls-files -- {target} returned no tracked .rs files -- the repo root or the \
             pathspec is wrong, which would make this oracle vacuously pass"
        );
        for rel in files {
            let text = std::fs::read_to_string(root.join(&rel))
                .unwrap_or_else(|e| panic!("{rel} is git-tracked but could not be read ({e})"));
            for (idx, line) in text.lines().enumerate() {
                if FORMS.iter().any(|form| line.contains(form)) {
                    found.insert((rel.clone(), (idx + 1) as u32));
                }
            }
        }
    }
    found
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    /// Observed-event rendezvous; the literal here is the backstop only.
    A,
    /// Derived from the test's own lease/heartbeat/poll config.
    B,
    /// A generous backstop against a wedged or starved machine.
    C,
    /// Not a training-progress wait.
    D,
}

/// One reviewed site. `window` is the INCLUSIVE 1-based line range (within
/// the same file) this entry's `marker` must be found in (concatenated,
/// newline-joined) for classes A/B/C; ignored (kept `(line, line)`) for
/// class D, which is checked only for a non-empty `reason`.
struct Site {
    file: &'static str,
    line: u32,
    class: Class,
    window: (u32, u32),
    marker: &'static str,
    reason: &'static str,
}

const WEDGED: &str = "wedged or starved machine";

/// #527/#567/#578's full inventory — see `contracts/testbounds-impl.md`'s
/// B3 table for the prose version of this same list.
const REVIEWED_SITES: &[Site] = &[
    // ---- timeout(Duration::from_secs( ------------------------------------
    Site { file: "crates/jammi-ai/tests/it/fine_tune.rs", line: 2648, class: Class::C, window: (2644, 2654), marker: WEDGED, reason: "" },
    Site { file: "crates/jammi-ai/tests/it/fine_tune.rs", line: 2901, class: Class::C, window: (2901, 2903), marker: WEDGED, reason: "" },
    Site { file: "crates/jammi-ai/tests/it/jobs_cancel.rs", line: 190, class: Class::D, window: (190, 190), marker: "", reason: "a queued INFERENCE job's cancel-then-claim dispatch (never_dispatched_infer); no training compute involved" },
    Site { file: "crates/jammi-ai/tests/it/jobs_cancel.rs", line: 501, class: Class::A, window: (501, 501), marker: "cancel_observed", reason: "" },
    Site { file: "crates/jammi-ai/tests/it/jobs_cancel.rs", line: 511, class: Class::A, window: (511, 518), marker: "poll cadence", reason: "" },
    Site { file: "crates/jammi-ai/tests/it/jobs_cancel.rs", line: 649, class: Class::A, window: (636, 652), marker: "arm_pause_before_spawn_blocking", reason: "" },
    Site { file: "crates/jammi-ai/tests/it/jobs_cancel.rs", line: 855, class: Class::B, window: (855, 855), marker: "heartbeat_secs", reason: "" },
    Site { file: "crates/jammi-ai/tests/it/scheduling.rs", line: 98, class: Class::D, window: (98, 98), marker: "", reason: "GPU memory scheduler admission queueing, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/scheduling.rs", line: 264, class: Class::D, window: (264, 264), marker: "", reason: "GPU memory scheduler admission queueing, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/training_set_stream.rs", line: 305, class: Class::D, window: (305, 305), marker: "", reason: "a prefetch/streaming deadlock backstop over data loading, not training compute progress" },
    Site { file: "crates/jammi-ai/tests/it/training_set_stream.rs", line: 790, class: Class::D, window: (790, 790), marker: "", reason: "a prefetch/streaming deadlock backstop over data loading, not training compute progress" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", line: 259, class: Class::C, window: (259, 261), marker: WEDGED, reason: "" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", line: 290, class: Class::D, window: (290, 290), marker: "", reason: "an IDLE worker's DRAIN (no job in flight), not training progress" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", line: 391, class: Class::D, window: (391, 391), marker: "", reason: "session.close() shutdown, not training progress" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", line: 427, class: Class::A, window: (415, 429), marker: "bundle_landed", reason: "" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", line: 440, class: Class::B, window: (440, 443), marker: "heartbeats", reason: "" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", line: 542, class: Class::D, window: (542, 542), marker: "", reason: "a compute/materialization RELEASE, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", line: 656, class: Class::D, window: (656, 656), marker: "", reason: "claim->hold prologue RELEASE mechanics, gated by this file's own loop_test_hooks parks" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", line: 749, class: Class::D, window: (749, 749), marker: "", reason: "claim->hold prologue RELEASE mechanics, gated by this file's own loop_test_hooks parks" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", line: 845, class: Class::D, window: (845, 845), marker: "", reason: "release_job_leases's own doc/message: it does not wait on any loop at all" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", line: 1260, class: Class::D, window: (1260, 1260), marker: "", reason: "parked-iteration RELEASE mechanics, gated by this file's own loop_test_hooks parks" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", line: 1474, class: Class::B, window: (1474, 1477), marker: "heartbeats", reason: "" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", line: 1656, class: Class::D, window: (1656, 1656), marker: "", reason: "a RELEASE-vs-DRAIN handle-race over a compute materialization, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", line: 1751, class: Class::D, window: (1751, 1751), marker: "", reason: "a schema-fault RELEASE test on fine_tune(1) (near-instant), not a training-progress wait" },
    Site { file: "crates/jammi-ai/tests/it/cache_staleness.rs", line: 781, class: Class::D, window: (781, 781), marker: "", reason: "a GPU-budget cache reload wait, not training" },
    // ---- sleep(Duration::from_secs( ---------------------------------------
    Site { file: "crates/jammi-ai/tests/it/instance_identity.rs", line: 473, class: Class::D, window: (473, 473), marker: "", reason: "worker-registry (list_workers) timing, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/instance_identity.rs", line: 515, class: Class::D, window: (515, 515), marker: "", reason: "worker-registry (list_workers) timing, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/instance_identity.rs", line: 564, class: Class::D, window: (564, 564), marker: "", reason: "worker-registry (list_workers) timing, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/instance_identity.rs", line: 606, class: Class::D, window: (606, 606), marker: "", reason: "worker-registry (list_workers) timing, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/peer_gang.rs", line: 381, class: Class::D, window: (381, 381), marker: "", reason: "a peer wire round-trip silence simulation, not training compute" },
    Site { file: "crates/jammi-ai/tests/distributed/gang_chaos.rs", line: 149, class: Class::D, window: (149, 149), marker: "", reason: "a fault-injection delay before kill9, not an assertion bound -- the pass/fail wait is harness::await_job's TERMINAL_TIMEOUT, reviewed separately" },
    Site { file: "crates/jammi-ai/tests/distributed/gang_chaos.rs", line: 193, class: Class::D, window: (193, 193), marker: "", reason: "a fault-injection delay before kill9, not an assertion bound -- the pass/fail wait is harness::await_job's TERMINAL_TIMEOUT, reviewed separately" },
    // ---- Instant::now() + Duration::from_secs( (the deadline-loop form) --
    Site { file: "crates/jammi-ai/tests/it/host_admission.rs", line: 95, class: Class::D, window: (95, 95), marker: "", reason: "host admission/slot holder state, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/host_admission.rs", line: 345, class: Class::D, window: (345, 345), marker: "", reason: "claim-loop idle-poll count, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/fine_tune.rs", line: 2573, class: Class::C, window: (2573, 2581), marker: WEDGED, reason: "" },
    Site { file: "crates/jammi-ai/tests/it/fine_tune.rs", line: 2866, class: Class::C, window: (2866, 2873), marker: WEDGED, reason: "" },
    Site { file: "crates/jammi-ai/tests/it/instance_identity.rs", line: 38, class: Class::D, window: (38, 38), marker: "", reason: "catalog worker-listing poll, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/instance_identity.rs", line: 286, class: Class::D, window: (286, 286), marker: "", reason: "catalog worker-listing poll, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/instance_identity.rs", line: 382, class: Class::D, window: (382, 382), marker: "", reason: "catalog worker-listing poll, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/jobs_cancel.rs", line: 676, class: Class::D, window: (676, 676), marker: "", reason: "watcher-task cleanup after an abort, not training progress" },
    Site { file: "crates/jammi-ai/tests/it/jobs_cancel.rs", line: 693, class: Class::D, window: (693, 693), marker: "", reason: "catalog handle refcount cleanup after an abort, not training progress" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", line: 178, class: Class::D, window: (178, 178), marker: "", reason: "the claim->hold admission transition (in_flight), not training compute; every call site awaits want=1, near-instant" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", line: 347, class: Class::B, window: (347, 347), marker: "FAST_TIMING.heartbeat", reason: "" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", line: 373, class: Class::B, window: (373, 373), marker: "FAST_TIMING.heartbeat", reason: "" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", line: 477, class: Class::B, window: (477, 477), marker: "FAST_TIMING.heartbeat", reason: "" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", line: 902, class: Class::D, window: (902, 902), marker: "", reason: "a workers-row upsert-on-spawn poll, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", line: 1694, class: Class::D, window: (1694, 1694), marker: "", reason: "a lease-keeper thread death poll, not training compute" },
    // ---- named const `: Duration = Duration::from_secs(` -------------------
    Site { file: "crates/jammi-ai/tests/distributed/harness.rs", line: 531, class: Class::C, window: (531, 596), marker: WEDGED, reason: "" },
];

fn site_text(root: &Path, site: &Site) -> String {
    let full = std::fs::read_to_string(root.join(site.file))
        .unwrap_or_else(|e| panic!("{}: {e}", site.file));
    let lines: Vec<&str> = full.lines().collect();
    let (start, end) = site.window;
    let start = (start as usize).saturating_sub(1);
    let end = (end as usize).min(lines.len());
    lines[start..end].join("\n")
}

#[test]
fn every_literal_wall_clock_bound_is_reviewed() {
    let scanned = scan_universe();
    assert!(
        scanned.len() >= 30,
        "the scan found only {} sites across {TARGET_ROOTS:?} -- expected at least 30; a git \
         root or CWD mismatch would make this oracle hollow-green, so a suspiciously small \
         universe is a hard failure rather than a silent pass",
        scanned.len()
    );

    let reviewed: BTreeSet<(String, u32)> = REVIEWED_SITES
        .iter()
        .map(|s| (s.file.to_string(), s.line))
        .collect();
    assert_eq!(
        reviewed.len(),
        REVIEWED_SITES.len(),
        "REVIEWED_SITES has a duplicate (file, line) entry"
    );

    let missing: Vec<&(String, u32)> = scanned.difference(&reviewed).collect();
    assert!(
        missing.is_empty(),
        "new, UNREVIEWED literal wall-clock bound(s) (add each to REVIEWED_SITES with its \
         class): {missing:?}"
    );

    let stale: Vec<&(String, u32)> = reviewed.difference(&scanned).collect();
    assert!(
        stale.is_empty(),
        "REVIEWED_SITES entries no longer found at their recorded line (moved, edited, or \
         removed -- re-locate and update the entry): {stale:?}"
    );
}

#[test]
fn every_class_a_b_c_site_carries_its_class_marker() {
    let root = repo_root();
    for site in REVIEWED_SITES {
        match site.class {
            Class::A | Class::B | Class::C => {
                assert!(
                    !site.marker.is_empty(),
                    "{}:{} is class {:?} but carries no marker to check",
                    site.file,
                    site.line,
                    site.class
                );
                let text = site_text(&root, site);
                assert!(
                    text.contains(site.marker),
                    "{}:{} is reviewed as class {:?} but its window ({}..={}) does not contain \
                     the required marker {:?}:\n{text}",
                    site.file,
                    site.line,
                    site.class,
                    site.window.0,
                    site.window.1,
                    site.marker
                );
            }
            Class::D => {
                assert!(
                    !site.reason.is_empty(),
                    "{}:{} is class D (not a training-progress wait) but carries no reason",
                    site.file,
                    site.line
                );
            }
        }
    }
}
