# 62 — Embedding-surface unit: records

Carries the hand-off notes CONTRACT.md's E6 item requires: what this unit deliberately does
NOT ship, what its follow-ups inherit, the esc-058 discovery this unit's own work surfaced, and
the KO-7 follow-up pointer. States current fact, not the unit's history.

## Hardware evidence — current state

This section records what pod/hardware evidence actually backs the CPU-hermetic half's
provisional bounds and the K4/E5 leg, as landed at this branch's head. It is a current-state
snapshot, not a narrative: each claim below cites the artifact or code that verifies it,
and any claim this unit does NOT yet have committed evidence for is stated as open, not
asserted.

**1. Invariance gating proofs — 8/8 batch compositions, four arches, tree `48ed4495`+.**
`crates/jammi-encoders/tests/it/batch_composition_invariance.rs`'s
`gpu_composition_floor`/`GPU_TRUTH_DRIFT_BOUND` constants (`:585`, `:591`, `:691`) are derived
from the committed four-arch archival captures under
`docs/plans/62-embedding-surface/measurements/`, one self-identifying file per arch (first
line is a `HEADER` carrying the probed compute capability, driver device name, and crate
version):

- `measurements/gpu-floors-a100.txt:1` — `compute_capability=8.0 device_name=NVIDIA A100-SXM4-80GB`
- `measurements/gpu-floors-h100.txt:1` — `compute_capability=9.0 device_name=NVIDIA H100 80GB HBM3`
- `measurements/gpu-floors-a40.txt:1` — `compute_capability=8.6 device_name=NVIDIA A40`
- `measurements/gpu-floors-l40s.txt:1` — `compute_capability=8.9 device_name=NVIDIA L40S`

Each file's `OVERALL OVER 8 COMPOSITIONS / 88 ROW-MEASUREMENTS` line (e.g.
`gpu-floors-a100.txt:114`) confirms all 8 fixture compositions read back clean on every arch:
bit-exact `0.0` alone-vs-batch on sm80/sm86/sm90 (`EXACT_ARCH_COMPOSITION_FLOOR`,
`batch_composition_invariance.rs:585`), within `SM89_COMPOSITION_FLOOR` (`:591`, `4.2e-3`) on
sm89 (L40S). The sm89 window-radius red control is loudly SKIPPED, not silently passed: it is
inadmissible on this arch by construction (its own measured minimum separation,
`1.139471768897301e-3`, is below `SM89_COMPOSITION_FLOOR`) —
`pooled_embedding_red_control_window_radius_off_by_one_bf16_cuda` (`:836-865`) probes
`ComputeCapability::new(8, 9)` at runtime and returns early with a named `eprintln!` reason
before any assertion, which fires by construction on exactly the L40S capability the HEADER
line above records; the row-length control stays universal and is never skipped
(`gpu_composition_floor`'s own doc, `:525-563`, names the composition-scoped sm89
admissibility explicitly). Tree `48ed4495` (`Merge bound derivation: per-arch composition
floors + drift bound from four-arch measurements, sm89 admissibility scoping, fail-loud
unknown arch`) is an ancestor of this branch's head; the constants above are unchanged since.

**2. K4 transport leg — fix landed, post-fix pod re-confirmation not yet a committed artifact.**
The Utf8/Utf8View keying defect pod evidence surfaced (K4, a100, tree `48ed4495`:
`remote_flight_read_matches_local_readback_bitwise_on_gpu` panicking at
`grpc_remote_session_gpu.rs:222` on a `Utf8` vs `Utf8View` schema mismatch) is fixed on this
branch by commit `0e5f777d` (merged `0fcc8317`): `keyed_vectors`
(`crates/jammi-server/tests/it/grpc_remote_session_gpu.rs:236-264`) now casts `_row_id` to
`Utf8` before the `StringArray` downcast on both read paths, and the module doc
(`:216-235`) records the Utf8-vs-Utf8View difference as a transport/read-path representation
observation (DataFusion's `schema_force_view_types` parquet-reader default applying
independently per side), not a logical divergence — the `FixedSizeList<Float32>` vector
column keeps its unmodified bitwise, zero-tolerance comparison. `JAMMI_REQUIRE_CUDA`
fail-loud coverage of every skip path in this module (F-7, ancestor of `48ed4495`) is already
landed, so a pod job run under it cannot read green having executed zero assertions. OPEN:
no committed artifact in this repo (this module is an ordinary `it`-suite test, not a
`crates/jammi-kernels/artifacts/cuda-runs/*.json` producer) records an actual post-fix
`JAMMI_REQUIRE_CUDA=1` pod run of `remote_flight_read_matches_local_readback_bitwise_on_gpu`
and `encode_query_two_compute_gpu_repeat_determinism_is_recorded_not_gated`
(`grpc_remote_session_gpu.rs:274`, `:371`) on the a100 — the fix is landed and addresses the
exact failure the tree-`48ed4495` pod run hit, but a "2/2 green post-fix" pod re-run is not
independently verifiable from repo state alone.

**3. Encode identity artifact — GREEN, verified.**
`crates/jammi-kernels/artifacts/cuda-runs/2026-08-29-encode-ab-a4fad082-a100-sxm4.json`:
`"status": "GREEN"`, `"leg_premise_violations": []`, `"git_sha":
"a4fad082f895394b4bf7176bb11606945b5ac3d4"` (an ancestor of this branch's head, confirmed via
`git merge-base --is-ancestor`), legs `r1`/`r2` both `"outcome": "OK"` on
`NVIDIA A100-SXM4-80GB`, produced by `ci/scripts/perf/encode_ab.sh` (`"kind": "script"`) per
the stacked_sweep git-pinned-worktree convention. `python3 ci/scripts/check_cuda_run_artifacts.py`
passes over the full `cuda-runs/` directory including this file (exit 0).

**4. Linux CPU-floor confirmation — OPEN, not yet done.**
`PROVISIONAL_CPU_FLOOR` (`batch_composition_invariance.rs:257`) and its doc (`:237-256`)
state plainly, at this branch's head, that the `f32`-CPU alone-vs-padded-batch bound was
measured ONLY on a macOS dev box and is "NOT YET measured on the CI-representative Linux
runner or the pod train (contract E4, Step 5) -- that measurement must replace this one, not
merely confirm it, before this bound is treated as final" (`:253-256`). No committed artifact,
measurement file, or commit in this branch's history records an actual run of
`pooled_embedding_alone_matches_padded_batch_real_row_f32_cpu` or its two red controls
(`:319`, `:359`) on any of the four Linux pods used for the CUDA sweep above (or any other
Linux host) — the four `measurements/gpu-floors-*.txt` captures are all `measure_gpu_floors_print_only`
output, a CUDA-gated ignored test that does not exercise the `f32_cpu` tests at all. The
macOS-measured provisional floor's caveat is therefore still OPEN, not answered, as of this
branch's head; this record states that plainly rather than asserting a confirmation this repo
does not evidence.

**5. Oracle-advisory release-note paragraph.** Verified present and unedited: "## esc-057 fix
— the required release-note advisory (content_digest joins identity)" below, with its
**REQUIRED RELEASE-NOTE content** paragraph intact. Section ordering already reads
chronologically by discovery (C15 hand-off, C16 inheritance, esc-058, esc-057, K4 advisory,
KO-7 follow-up); left as-is.

## C15 hand-off — no ServerInfo change, no counters on the wire

This unit ships NO `ServerInfo` change and NO dispatch counter of any kind on any wire surface.
The fused attention arms are training-only by design (`if self.training` gates dispatch), so a
serving-path dispatch-counter assertion is unsatisfiable by construction and was never built —
v2 reshape 1 deleted the planned assertion, and E4's invariance oracle and E5's K4 leg both
carry an explicit no-dispatch-counter-assertion invariant for the same reason. `ServerInfo`'s
own front door (61/CONTRACT.md C15: only genuinely static-per-process fields may be appended;
`dispatch_counters`/`kernels_disabled_fired` never go on the wire) is therefore untouched by
this unit and remains open for its own follow-up to walk through.

## C16 inheritance — how-well preconditions discharge in unit 63, not here

61/CONTRACT.md C16 records that no feasible how-well statistic exists on the repo's data without
a public per-pair evaluation seam, a non-default early-stopping metric, and a non-hinge
objective or measured tie fraction. This unit does not touch that surface. `docs/plans/63-how-well/CONTRACT.md`
is authored and binding for that follow-up; its H2 (exact two-sided sign test, domain: numerics)
and H3 (committed held-out fixture under `cookbook/fixtures/`, domains: cookbook + docs-ci) are
already landed on side branches. C16's preconditions discharge there, not in this unit.

## esc-058 — warm `ModelCache` staleness, fix landed on this branch

An adjacent-but-distinct defect to esc-057 (identity-completeness — K7's equal-hash-implies-
equal-bytes holds warm): a warm `ModelCache` hit could serve a pre-mutation digest and
pre-mutation vectors after the underlying model directory was mutated in place within the same
process, because `get_or_load`'s fast path never re-read disk and `content_digest()` was
computed once at load time and never re-derived from current bytes. What breaks in this class is
the digest's correspondence to the model directory's CURRENT state, not its completeness as a
determinant set (esc-058 is `class_id: cache-coherence-digest-describes-disk`, distinct from
esc-057's `identity-completeness`).

The fix is landed on this branch (commit `f0712069`, merged at `0ca0b1e6`): `get_or_load`'s warm
fast path (`crates/jammi-ai/src/model/cache.rs:234-324`) clones the currently cached entry's shared
handles under a short-held READ lock, drops the lock, then `stat`-probes a load-time
`ModelFingerprint` (`ModelFingerprint::probe`, `crates/jammi-ai/src/model/backend/candle.rs:1026-1083`,
computed at load by `compute_model_fingerprint`, `candle.rs:1102-1136`, invoked — in a pinned
fingerprint-then-digest order (audit round 62, F-4'') — via `compute_model_identity_facets`
(`candle.rs:1167-1173`), called from `CandleBackend::load` at `candle.rs:1931`; exposed via
`LoadedModel::probe_freshness` (`pub(crate)`), `crates/jammi-ai/src/model/mod.rs:408-413`) UNLOCKED,
then RE-VALIDATES under a write lock (`Arc::ptr_eq` against the current cache entry) before serving
the cached `Arc<LoadedModel>` — a narrow race (a concurrent evict/reload winning while this task
probes) simply retries from the top against whatever is there now; the single-flight path below
still ensures at most one loader per id: `Ok(true)` serves the snapshot if it is still the current
entry (`cache.rs:252-279`); `Ok(false)` evicts ONLY the same probed entry — unconditional on
`ref_count`, deliberately NOT routed through the idle-only `evict_one` path, since serving stale
bytes is a correctness bug rather than a capacity one — and falls through to the same single-flight
reload path that re-resolves and re-hashes current bytes (`cache.rs:280-315`); `Err` — a
fingerprinted file vanished or became unreadable — surfaces as a typed refusal (K2), never silently
treated as fresh or stale (`cache.rs:316-322`). `stat` only, never a re-hash, so the fast path stays
cheap. Honest residual,
documented on `ModelFingerprint`: `(len, mtime)` is a staleness TRIPWIRE, not a cryptographic
guarantee — a same-length, same-mtime content swap is invisible to it; the `ModelContentDigest`
recomputed on every actual reload remains the sole authoritative attestation.

Red-green test: `crates/jammi-ai/tests/it/cache_staleness.rs::warm_hit_after_in_place_mutation_reloads_fresh_digest_and_vectors`
(`closes_escape: esc-058`) drives the symptom_spec's exact observable through the real
`ModelCache::get_or_load` (warm hit) and `ModelCache::load_owned_for_test` (cold control) over a
mutated model fixture directory. Ledger status is `eval_added`, not yet `closed` — per the
ledger's own lifecycle it promotes to `closed` only after this branch merges and the cited test
is green on main.

## esc-057 fix — the required release-note advisory (content_digest joins identity)

Folding `content_digest` into `ModelIdentity` (esc-057, above) means every `DefinitionHash`
computed before this unit shipped was computed WITHOUT that determinant. The moment a
process upgrades onto a build that carries the fix, the first `verify_materialization` call
(or replay) against any PRE-EXISTING `ready` table backed by a model-producing descriptor
re-derives `definition_hash` WITH the new digest folded in, and that hash will not equal the
one recorded at write time — every such row reports a mismatch, unconditionally, on first
touch after the upgrade, regardless of whether the underlying model directory actually
changed. This is not a regression to work around: it is the fail-safe direction by
construction (stale — or in this case, incompletely-attested — is DETECTED, never silently
trusted as still valid), and `recompute` is the documented remedy, exactly the same recovery
path a genuine `esc-057` collision would require. This is the esc-057 fix working as designed,
not a new defect it introduces.

**REQUIRED RELEASE-NOTE content for the version that ships this fix**: state plainly that
upgrading past this version invalidates every previously-recorded `DefinitionHash` for a
model-producing (`Inference`/`Embedding`) `ProducingDescriptor` — the first
`verify_materialization`/replay against each such table after the upgrade will report a
mismatch even with an untouched model directory — and that `recompute` is the expected,
one-time remedy per affected table. Silence on this point would read as a spurious mass
mismatch report on upgrade day; naming it in the release note turns it into the expected,
documented consequence of closing a real identity-completeness gap.

## K4 device leg — narrowing the JAMMI_REQUIRE_CUDA scope claim (round-2 advisory, open)

`crates/jammi-server/tests/it/grpc_remote_session_gpu.rs`'s module doc states: "a pod job that
meant to prove this leg can never read green having executed zero assertions" as the guarantee
`JAMMI_REQUIRE_CUDA` provides. That statement is overbroad as written: `JAMMI_REQUIRE_CUDA`
only converts every skip/early-return path INSIDE the module (the `InferenceSession::open`
failure inside `start_gpu_engine_server`, and both tests' `let Some(server) = … else { return;
}`) into a hard panic — it says nothing about whether the module was COMPILED IN at all. A
build invoked without the `live-gpu-tests` cargo feature (the feature gating the `mod` line
that pulls this file into the `it` test binary in the first place, per the module's own
"Gating" section) compiles this entire module OUT; no env var, including
`JAMMI_REQUIRE_CUDA`, runs inside code that was never compiled, so a feature-omitted invocation
reads green having executed zero assertions from this leg with `JAMMI_REQUIRE_CUDA` set — the
exact failure mode the module doc claims cannot happen.

This is a reviewed, deliberate scope decision (KO-7 scope pattern: true-but-deliberate, not an
oversight), not a gap this unit corrects — feature-omission is out of `JAMMI_REQUIRE_CUDA`'s
reach BY CONSTRUCTION (it is a runtime env var; feature selection is a compile-time `cargo`
flag), and is covered instead by the recorded pod invocation's explicit `--features
cuda,live-gpu-tests` (see the module's own "Live run" command block) — the pod-validation
discipline this repo already follows never invokes a GPU-gated `it` binary without naming its
required features explicitly. The accurate scope statement: **the leg is
require-env-hardened GIVEN the `live-gpu-tests` feature is compiled in; feature-omission is a
separate, already-covered concern, not a hole in `JAMMI_REQUIRE_CUDA`'s own guarantee.**

docs-ci owns `docs/`, not `crates/jammi-server/tests/it/grpc_remote_session_gpu.rs` — this row
records the accurate scope statement without editing the test file. Flagged for the next
wire-server-domain touch of this module: narrow the module doc's "can never read green having
executed zero assertions" sentence to name the `JAMMI_REQUIRE_CUDA`-given-the-feature-compiled-in
scope explicitly, rather than the current unqualified claim.

## KO-7 scan-root widening — its own follow-up, not this unit

`check_kernel_oracles.py`'s KO-7 rule (unrun-is-RED, total over recognized skip shapes) excludes
`gpu_capability/**` and `jammi-server` it-GPU test modules from its scan roots, so a skip inside
either is invisible to the rule. This is a reviewed, deliberate scope decision (KO-7 scope is
true-but-deliberate), not an oversight this unit corrects — OQ7's ruling is that widening KO-7's
scan roots is its own human-merged tightening PR, never folded into a feature unit's diff. No
such PR exists yet; the scope gap stands as recorded.
