# 62 — Embedding-surface unit: records

Carries the hand-off notes CONTRACT.md's E6 item requires: what this unit deliberately does
NOT ship, what its follow-ups inherit, the esc-058 discovery this unit's own work surfaced, and
the KO-7 follow-up pointer. States current fact, not the unit's history.

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
fast path (`crates/jammi-ai/src/model/cache.rs:79-142`) now `stat`-probes a load-time
`ModelFingerprint` (`ModelFingerprint::probe`, `crates/jammi-ai/src/model/backend/candle.rs:727-780`,
computed at load by `compute_model_fingerprint`, `candle.rs:790-807`, called from `CandleBackend::load`
at `candle.rs:1555`; exposed via `LoadedModel::probe_freshness`, `crates/jammi-ai/src/model/mod.rs:391-396`)
before serving the cached `Arc<LoadedModel>`: `Ok(true)` serves it (`cache.rs:96-106`); `Ok(false)`
evicts the stale entry through the existing removal machinery and falls through to the same
single-flight reload path that re-resolves and re-hashes current bytes (`cache.rs:107-133`);
`Err` — a fingerprinted file vanished or became unreadable — surfaces as a typed refusal (K2),
never silently treated as fresh or stale (`cache.rs:134-140`). `stat` only, never a re-hash, so
the fast path stays cheap. Honest residual, documented on `ModelFingerprint`: `(len, mtime)` is a
staleness TRIPWIRE, not a cryptographic guarantee — a same-length, same-mtime content swap is
invisible to it; the `ModelContentDigest` recomputed on every actual reload remains the sole
authoritative attestation.

Red-green test: `crates/jammi-ai/tests/it/cache_staleness.rs::warm_hit_after_in_place_mutation_reloads_fresh_digest_and_vectors`
(`closes_escape: esc-058`) drives the symptom_spec's exact observable through the real
`ModelCache::get_or_load` (warm hit) and `ModelCache::load_owned_for_test` (cold control) over a
mutated model fixture directory. Ledger status is `eval_added`, not yet `closed` — per the
ledger's own lifecycle it promotes to `closed` only after this branch merges and the cited test
is green on main.

## KO-7 scan-root widening — its own follow-up, not this unit

`check_kernel_oracles.py`'s KO-7 rule (unrun-is-RED, total over recognized skip shapes) excludes
`gpu_capability/**` and `jammi-server` it-GPU test modules from its scan roots, so a skip inside
either is invisible to the rule. This is a reviewed, deliberate scope decision (KO-7 scope is
true-but-deliberate), not an oversight this unit corrects — OQ7's ruling is that widening KO-7's
scan roots is its own human-merged tightening PR, never folded into a feature unit's diff. No
such PR exists yet; the scope gap stands as recorded.
