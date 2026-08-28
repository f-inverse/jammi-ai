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

## esc-058 — discovered during E2, triaged valid-defect, fix on a sibling branch

While building E2's content-digest fold, an adjacent-but-distinct defect surfaced: a warm
`ModelCache` hit serves a pre-mutation digest and pre-mutation vectors after the underlying
model directory is mutated in place within the same process, because `get_or_load`'s fast path
(`cache.rs:91-99`) never re-reads disk and `content_digest()` is computed once at load time
(`candle.rs:404-412,636-648`) and never re-derived from current bytes. This is a distinct class
from esc-057 (identity-completeness — K7's equal-hash-implies-equal-bytes holds warm): what
breaks here is the digest's correspondence to the model directory's CURRENT state, not its
completeness as a determinant set. Recorded as `esc-058-warm-model-cache-serves-stale-digest-and-vectors`,
status `open`, triaged valid-defect (issue-triage, 2026-08-28, during unit-62 E2). Its fix rides
this PR train on a sibling branch (`ai/62-esc058-cache-staleness`) and is not part of this docs
pass.

## KO-7 scan-root widening — its own follow-up, not this unit

`check_kernel_oracles.py`'s KO-7 rule (unrun-is-RED, total over recognized skip shapes) excludes
`gpu_capability/**` and `jammi-server` it-GPU test modules from its scan roots, so a skip inside
either is invisible to the rule. This is a reviewed, deliberate scope decision (KO-7 scope is
true-but-deliberate), not an oversight this unit corrects — OQ7's ruling is that widening KO-7's
scan roots is its own human-merged tightening PR, never folded into a feature unit's diff. No
such PR exists yet; the scope gap stands as recorded.
