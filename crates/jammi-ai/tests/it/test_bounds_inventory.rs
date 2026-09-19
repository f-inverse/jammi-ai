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
//! assertion fails, naming the site) or a REMOVED site makes a stale entry
//! fail to match (the same assertion's other arm) — so this file must be
//! updated in the SAME commit as any new/removed bound. A site is keyed by
//! ITS OWN IDENTITY — `(file, enclosing item, ordinal within that item)`,
//! the enclosing item found by `syn` — never by a line number: an edit
//! anywhere above a site (a doc comment, an unrelated helper) moves its
//! line and must not move the review; the line is carried only for the
//! failure message. The marker window for classes A/B/C is the enclosing
//! item's own span (attributes and doc comments included), so a backstop's
//! justification lives next to the code it justifies.

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

/// One found literal bound: `(file, enclosing item, ordinal within that
/// item)` with the 1-based line carried for messages, and the enclosing
/// item's own inclusive line span (the marker window for classes A/B/C).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Hit {
    file: String,
    item: String,
    ordinal: u32,
    line: u32,
    span: (u32, u32),
}

/// Every `fn`/`const`/`static` item in `text` (top-level, in `impl`s, in
/// nested `mod`s, in trait impls), as `(name, start line, end line)` with
/// the span covering the item's attributes and doc comments -- `syn` over
/// the real AST, never a text scan for `fn `.
fn item_regions(file: &str, text: &str) -> Vec<(String, u32, u32)> {
    struct V(Vec<(String, u32, u32)>);
    fn span_of<T: syn::spanned::Spanned>(node: &T) -> (u32, u32) {
        let s = node.span();
        (s.start().line as u32, s.end().line as u32)
    }
    impl<'ast> syn::visit::Visit<'ast> for V {
        fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
            let (a, b) = span_of(i);
            self.0.push((i.sig.ident.to_string(), a, b));
            syn::visit::visit_item_fn(self, i);
        }
        fn visit_impl_item_fn(&mut self, i: &'ast syn::ImplItemFn) {
            let (a, b) = span_of(i);
            self.0.push((i.sig.ident.to_string(), a, b));
            syn::visit::visit_impl_item_fn(self, i);
        }
        fn visit_item_const(&mut self, i: &'ast syn::ItemConst) {
            let (a, b) = span_of(i);
            self.0.push((i.ident.to_string(), a, b));
            syn::visit::visit_item_const(self, i);
        }
        fn visit_item_static(&mut self, i: &'ast syn::ItemStatic) {
            let (a, b) = span_of(i);
            self.0.push((i.ident.to_string(), a, b));
            syn::visit::visit_item_static(self, i);
        }
    }
    let parsed = syn::parse_file(text)
        .unwrap_or_else(|e| panic!("{file}: syn could not parse this source ({e})"));
    let mut v = V(Vec::new());
    syn::visit::Visit::visit_file(&mut v, &parsed);
    v.0
}

/// The innermost item whose span contains `line` (smallest span wins), or
/// `<module-scope>` when no item contains it.
fn enclosing_item(regions: &[(String, u32, u32)], line: u32) -> (String, (u32, u32)) {
    regions
        .iter()
        .filter(|(_, a, b)| *a <= line && line <= *b)
        .min_by_key(|(_, a, b)| b - a)
        .map(|(n, a, b)| (n.clone(), (*a, *b)))
        .unwrap_or_else(|| ("<module-scope>".to_string(), (line, line)))
}

/// Every literal bound in every [`TARGET_ROOTS`] file, scanning each tracked
/// line for every [`FORMS`] substring and attributing each hit to its
/// enclosing item; ordinals count hits within one item in source order.
fn scan_universe() -> Vec<Hit> {
    let root = repo_root();
    let mut found = Vec::new();
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
            let regions = item_regions(&rel, &text);
            let mut per_item: std::collections::BTreeMap<String, u32> =
                std::collections::BTreeMap::new();
            for (idx, line_text) in text.lines().enumerate() {
                if FORMS.iter().any(|form| line_text.contains(form)) {
                    let line = (idx + 1) as u32;
                    let (item, span) = enclosing_item(&regions, line);
                    let ordinal = per_item.entry(item.clone()).or_insert(0);
                    *ordinal += 1;
                    found.push(Hit {
                        file: rel.clone(),
                        item,
                        ordinal: *ordinal,
                        line,
                        span,
                    });
                }
            }
        }
    }
    found.sort();
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

/// One reviewed site, keyed by identity: the file, the enclosing `fn`/
/// `const`/`static` item (`<module-scope>` if none) and the ordinal of this
/// bound within that item in source order. For classes A/B/C the `marker`
/// must appear inside the enclosing item's span (doc comments included);
/// class D is checked only for a non-empty `reason`.
struct Site {
    file: &'static str,
    item: &'static str,
    ordinal: u32,
    class: Class,
    marker: &'static str,
    reason: &'static str,
}

const WEDGED: &str = "wedged or starved machine";

/// #527/#567/#578's full inventory — see `contracts/testbounds-impl.md`'s
/// B3 table for the prose version of this same list.
const REVIEWED_SITES: &[Site] = &[
    // ---- timeout(Duration::from_secs( ------------------------------------
    Site { file: "crates/jammi-ai/tests/it/fine_tune.rs", item: "cancelled_run_reclaims_epoch_checkpoints_that_actually_existed", ordinal: 2, class: Class::C, marker: WEDGED, reason: "" },
    Site { file: "crates/jammi-ai/tests/it/fine_tune.rs", item: "finalize_reclaims_a_persistently_failed_prune_and_warns", ordinal: 2, class: Class::C, marker: WEDGED, reason: "" },
    Site { file: "crates/jammi-ai/tests/it/jobs_cancel.rs", item: "a_cancel_requested_while_queued_ends_the_job_without_a_worker", ordinal: 1, class: Class::D, marker: "", reason: "a queued INFERENCE job cancelled before any claim (never_dispatched_infer): the row is already terminal, so the wait reads one catalog row; no training compute involved" },
    Site { file: "crates/jammi-ai/tests/it/jobs_cancel.rs", item: "a_requeued_job_already_flagged_is_cancelled_right_after_the_claim", ordinal: 1, class: Class::D, marker: "", reason: "a flagged queued INFERENCE job's claim-then-cancel at the post-claim checkpoint (never_dispatched_infer); no training compute involved" },
    Site { file: "crates/jammi-ai/tests/it/jobs_cancel.rs", item: "a_claimed_training_jobs_cancel_request_is_honoured_at_the_next_epoch_boundary", ordinal: 1, class: Class::A, marker: "cancel_observed", reason: "" },
    Site { file: "crates/jammi-ai/tests/it/jobs_cancel.rs", item: "a_claimed_training_jobs_cancel_request_is_honoured_at_the_next_epoch_boundary", ordinal: 2, class: Class::A, marker: "poll cadence", reason: "" },
    Site { file: "crates/jammi-ai/tests/it/jobs_cancel.rs", item: "a_dropped_run_claimed_jobs_future_leaves_no_leaked_cancel_watcher_or_catalog_handle", ordinal: 1, class: Class::A, marker: "arm_pause_before_spawn_blocking", reason: "" },
    Site { file: "crates/jammi-ai/tests/it/jobs_cancel.rs", item: "a_lease_loss_on_the_owning_worker_lands_the_lease_lost_outcome_never_the_cancel_message", ordinal: 1, class: Class::B, marker: "heartbeat_secs", reason: "" },
    Site { file: "crates/jammi-ai/tests/it/scheduling.rs", item: "blocking_acquire_and_panic_safety", ordinal: 1, class: Class::D, marker: "", reason: "GPU memory scheduler admission queueing, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/scheduling.rs", item: "concurrent_acquire_stress_and_liveness", ordinal: 1, class: Class::D, marker: "", reason: "GPU memory scheduler admission queueing, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/training_set_stream.rs", item: "p4_liveness_over_every_held_chunk_at_every_accepted_prefetch", ordinal: 1, class: Class::D, marker: "", reason: "a prefetch/streaming deadlock backstop over data loading, not training compute progress" },
    Site { file: "crates/jammi-ai/tests/it/training_set_stream.rs", item: "p3_streamed_read_completes_under_a_small_pool_while_eager_fails", ordinal: 1, class: Class::D, marker: "", reason: "a prefetch/streaming deadlock backstop over data loading, not training compute progress" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", item: "stop_and_join_lets_the_epoch_bundle_land", ordinal: 1, class: Class::C, marker: WEDGED, reason: "" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", item: "stop_and_join_returns_within_idle_poll_when_idle", ordinal: 1, class: Class::D, marker: "", reason: "an IDLE worker's DRAIN (no job in flight), not training progress" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", item: "dropping_the_guard_after_a_cancelled_stop_and_join_aborts_the_task", ordinal: 3, class: Class::D, marker: "", reason: "session.close() shutdown, not training progress" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", item: "release_and_stop_leaves_running_with_null_lease_and_no_new_bundle", ordinal: 1, class: Class::A, marker: "bundle_landed", reason: "" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", item: "release_and_stop_leaves_running_with_null_lease_and_no_new_bundle", ordinal: 2, class: Class::B, marker: "heartbeats", reason: "" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", item: "release_mid_materialization_resumes_on_the_successor_without_backoff", ordinal: 1, class: Class::D, marker: "", reason: "a compute/materialization RELEASE, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", item: "release_with_the_loop_paused_inside_claim_next_does_not_abort", ordinal: 1, class: Class::D, marker: "", reason: "claim->hold prologue RELEASE mechanics, gated by this file's own loop_test_hooks parks" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", item: "release_with_the_loop_paused_in_the_claim_to_hold_prologue_self_releases", ordinal: 1, class: Class::D, marker: "", reason: "claim->hold prologue RELEASE mechanics, gated by this file's own loop_test_hooks parks" },
    Site { file: "crates/jammi-ai/tests/it/gang_placed.rs", item: "release_landing_between_probe_claim_and_transfer_self_releases_a_placed_gang", ordinal: 1, class: Class::D, marker: "", reason: "run_placed_gang's own claim->hold prologue RELEASE mechanics, gated by this file's own loop_test_hooks park" },
    Site { file: "crates/jammi-ai/tests/it/gang_placed.rs", item: "release_landing_between_the_epoch_read_and_probe_claim_is_still_refused", ordinal: 1, class: Class::D, marker: "", reason: "run_placed_gang's own birth-snapshot/probe_claim ordering RELEASE mechanics, gated by this file's own loop_test_hooks park" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", item: "release_job_leases_reaches_a_live_foreign_loops_prologue_and_self_releases", ordinal: 1, class: Class::D, marker: "", reason: "release_job_leases's own doc/message: it does not wait on any loop at all" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", item: "release_landing_during_the_reclaim_window_is_caught_by_the_second_gate_read", ordinal: 1, class: Class::D, marker: "", reason: "parked-iteration RELEASE mechanics, gated by this file's own loop_test_hooks parks" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", item: "release_and_stop_report_matches_release_job_leases_on_the_pair_that_actually_differs", ordinal: 1, class: Class::B, marker: "heartbeats", reason: "" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", item: "a_release_racing_an_in_flight_drain_reads_stop_unwitnessed", ordinal: 1, class: Class::D, marker: "", reason: "a RELEASE-vs-DRAIN handle-race over a compute materialization, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", item: "release_and_stops_second_sweep_reports_jobs_none_building_some_from_a_real_fault", ordinal: 1, class: Class::D, marker: "", reason: "a schema-fault RELEASE test on fine_tune(1) (near-instant), not a training-progress wait" },
    Site { file: "crates/jammi-ai/tests/it/cache_staleness.rs", item: "stale_reload_while_guard_live_waits_for_release_under_a_realistic_budget", ordinal: 1, class: Class::D, marker: "", reason: "a GPU-budget cache reload wait, not training" },
    // ---- sleep(Duration::from_secs( ---------------------------------------
    Site { file: "crates/jammi-ai/tests/it/instance_identity.rs", item: "a_failed_first_upsert_worker_leaves_the_cell_none_so_the_keeper_writes_no_workers_row", ordinal: 1, class: Class::D, marker: "", reason: "worker-registry (list_workers) timing, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/instance_identity.rs", item: "a_failed_first_upsert_worker_leaves_the_cell_none_so_the_keeper_writes_no_workers_row", ordinal: 2, class: Class::D, marker: "", reason: "worker-registry (list_workers) timing, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/instance_identity.rs", item: "a_gang_member_survives_a_forced_instance_delete_after_one_keeper_pass", ordinal: 1, class: Class::D, marker: "", reason: "worker-registry (list_workers) timing, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/instance_identity.rs", item: "a_drained_worker_is_not_resurrected_as_a_member_after_a_forced_delete", ordinal: 1, class: Class::D, marker: "", reason: "worker-registry (list_workers) timing, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/peer_gang.rs", item: "a_silent_member_over_the_wire_expires_the_coordinators_wait_at_the_deadline_naming_the_round", ordinal: 1, class: Class::D, marker: "", reason: "a peer wire round-trip silence simulation, not training compute" },
    // ---- Instant::now() + Duration::from_secs( (the deadline-loop form) --
    Site { file: "crates/jammi-ai/tests/it/host_admission.rs", item: "wait_holder", ordinal: 1, class: Class::D, marker: "", reason: "host admission/slot holder state, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/host_admission.rs", item: "an_idle_loop_never_claims_while_a_rank_is_held", ordinal: 1, class: Class::D, marker: "", reason: "claim-loop idle-poll count, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/fine_tune.rs", item: "cancelled_run_reclaims_epoch_checkpoints_that_actually_existed", ordinal: 1, class: Class::C, marker: WEDGED, reason: "" },
    Site { file: "crates/jammi-ai/tests/it/fine_tune.rs", item: "finalize_reclaims_a_persistently_failed_prune_and_warns", ordinal: 1, class: Class::C, marker: WEDGED, reason: "" },
    Site { file: "crates/jammi-ai/tests/it/instance_identity.rs", item: "await_workers", ordinal: 1, class: Class::D, marker: "", reason: "catalog worker-listing poll, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/instance_identity.rs", item: "wait_until_gang_member", ordinal: 1, class: Class::D, marker: "", reason: "catalog worker-listing poll, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/instance_identity.rs", item: "peer_advertise_without_peer_bind_fails_open_naming_both_keys", ordinal: 1, class: Class::D, marker: "", reason: "catalog worker-listing poll, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/jobs_cancel.rs", item: "a_dropped_run_claimed_jobs_future_leaves_no_leaked_cancel_watcher_or_catalog_handle", ordinal: 2, class: Class::D, marker: "", reason: "watcher-task cleanup after an abort, not training progress" },
    Site { file: "crates/jammi-ai/tests/it/jobs_cancel.rs", item: "a_dropped_run_claimed_jobs_future_leaves_no_leaked_cancel_watcher_or_catalog_handle", ordinal: 3, class: Class::D, marker: "", reason: "catalog handle refcount cleanup after an abort, not training progress" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", item: "wait_in_flight", ordinal: 1, class: Class::D, marker: "", reason: "the claim->hold admission transition (in_flight), not training compute; every call site awaits want=1, near-instant" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", item: "dropping_the_guard_after_a_cancelled_stop_and_join_aborts_the_task", ordinal: 1, class: Class::B, marker: "FAST_TIMING.heartbeat", reason: "" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", item: "dropping_the_guard_after_a_cancelled_stop_and_join_aborts_the_task", ordinal: 2, class: Class::B, marker: "FAST_TIMING.heartbeat", reason: "" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", item: "release_and_stop_leaves_running_with_null_lease_and_no_new_bundle", ordinal: 3, class: Class::B, marker: "FAST_TIMING.heartbeat", reason: "" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", item: "a_second_spawn_on_the_same_session_is_refused_structurally", ordinal: 1, class: Class::D, marker: "", reason: "a workers-row upsert-on-spawn poll, not training compute" },
    Site { file: "crates/jammi-ai/tests/it/jobs_shutdown.rs", item: "release_job_leases_is_unobserved_when_the_keeper_thread_is_dead", ordinal: 1, class: Class::D, marker: "", reason: "a lease-keeper thread death poll, not training compute" },
    Site { file: "crates/jammi-ai/tests/distributed/placed_search.rs", item: "wait_ready", ordinal: 1, class: Class::D, marker: "", reason: "a worker /readyz readiness poll deadline in the placed-search leg's own harness (catalog connect + migrate + tier mount on a cold runner), not training compute" },
    // ---- named const `: Duration = Duration::from_secs(` -------------------
    Site { file: "crates/jammi-ai/tests/distributed/harness.rs", item: "TERMINAL_TIMEOUT", ordinal: 1, class: Class::C, marker: WEDGED, reason: "" },
];

fn span_text(root: &Path, file: &str, span: (u32, u32)) -> String {
    let full = std::fs::read_to_string(root.join(file)).unwrap_or_else(|e| panic!("{file}: {e}"));
    let lines: Vec<&str> = full.lines().collect();
    let start = (span.0 as usize).saturating_sub(1);
    let end = (span.1 as usize).min(lines.len());
    lines[start..end].join("\n")
}

fn key(file: &str, item: &str, ordinal: u32) -> (String, String, u32) {
    (file.to_string(), item.to_string(), ordinal)
}

#[test]
fn every_literal_wall_clock_bound_is_reviewed() {
    let scanned = scan_universe();
    if std::env::var_os("JAMMI_TEST_BOUNDS_PRINT").is_some() {
        for h in &scanned {
            eprintln!("HIT\t{}\t{}\t{}\t{}", h.file, h.item, h.ordinal, h.line);
        }
    }
    assert!(
        scanned.len() >= 30,
        "the scan found only {} sites across {TARGET_ROOTS:?} -- expected at least 30; a git \
         root or CWD mismatch would make this oracle hollow-green, so a suspiciously small \
         universe is a hard failure rather than a silent pass",
        scanned.len()
    );
    let scanned_keys: BTreeSet<(String, String, u32)> = scanned
        .iter()
        .map(|h| key(&h.file, &h.item, h.ordinal))
        .collect();
    assert_eq!(
        scanned_keys.len(),
        scanned.len(),
        "two hits share one (file, item, ordinal) key"
    );

    let reviewed: BTreeSet<(String, String, u32)> = REVIEWED_SITES
        .iter()
        .map(|s| key(s.file, s.item, s.ordinal))
        .collect();
    assert_eq!(
        reviewed.len(),
        REVIEWED_SITES.len(),
        "REVIEWED_SITES has a duplicate (file, item, ordinal) entry"
    );

    let missing: Vec<String> = scanned
        .iter()
        .filter(|h| !reviewed.contains(&key(&h.file, &h.item, h.ordinal)))
        .map(|h| format!("{}::{} #{} (line {})", h.file, h.item, h.ordinal, h.line))
        .collect();
    assert!(
        missing.is_empty(),
        "new, UNREVIEWED literal wall-clock bound(s) (add each to REVIEWED_SITES with its \
         class): {missing:?}"
    );

    let stale: Vec<&(String, String, u32)> = reviewed.difference(&scanned_keys).collect();
    assert!(
        stale.is_empty(),
        "REVIEWED_SITES entries no longer found (the bound was removed, or its enclosing item \
         was renamed or split -- re-locate and update the entry): {stale:?}"
    );
}

#[test]
fn every_class_a_b_c_site_carries_its_class_marker() {
    let root = repo_root();
    let scanned = scan_universe();
    for site in REVIEWED_SITES {
        let hit = scanned
            .iter()
            .find(|h| key(&h.file, &h.item, h.ordinal) == key(site.file, site.item, site.ordinal));
        match site.class {
            Class::A | Class::B | Class::C => {
                assert!(
                    !site.marker.is_empty(),
                    "{}::{} #{} is class {:?} but carries no marker to check",
                    site.file,
                    site.item,
                    site.ordinal,
                    site.class
                );
                let Some(hit) = hit else {
                    // The other test names the stale entry; nothing to check here.
                    continue;
                };
                let text = span_text(&root, &hit.file, hit.span);
                assert!(
                    text.contains(site.marker),
                    "{}::{} #{} (line {}) is reviewed as class {:?} but its enclosing item's span \
                     ({}..={}) does not contain the required marker {:?}:\n{text}",
                    site.file,
                    site.item,
                    site.ordinal,
                    hit.line,
                    site.class,
                    hit.span.0,
                    hit.span.1,
                    site.marker
                );
            }
            Class::D => {
                assert!(
                    !site.reason.is_empty(),
                    "{}::{} #{} is class D (not a training-progress wait) but carries no reason",
                    site.file,
                    site.item,
                    site.ordinal
                );
            }
        }
    }
}

/// The review key survives an edit ABOVE the site and does not survive the
/// site moving to a different item: the same source with two blank lines
/// prepended yields identical keys; a bound inside a different fn is a
/// different key.
#[test]
fn review_key_is_line_independent_and_item_sensitive() {
    let src = "fn a() { let _ = timeout(Duration::from_secs(5)); }\nfn b() { let _ = timeout(Duration::from_secs(5)); }\n";
    let regions = item_regions("<synthetic>", src);
    let shifted = format!("\n\n{src}");
    let regions2 = item_regions("<synthetic>", &shifted);
    assert_eq!(enclosing_item(&regions, 1).0, "a");
    assert_eq!(
        enclosing_item(&regions2, 3).0,
        "a",
        "an edit above the site must not change its key"
    );
    assert_eq!(
        enclosing_item(&regions, 2).0,
        "b",
        "a bound in another fn is another key"
    );
    assert_ne!(
        enclosing_item(&regions, 1).1,
        enclosing_item(&regions2, 3).1,
        "the span still moves"
    );
}
