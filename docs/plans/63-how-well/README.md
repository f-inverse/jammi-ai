# 63 — How-well unit (C16 lift)

**Status:** plan CONVERGED (pressure-tested, REFINE folded — PLAN.md's v2 deltas are binding: twelve numbered corrections including the committed-fixture requirement, N=12 / >=11-of-12 exact sign test, final-epoch endpoint, disabled early stopping, measured determinism floor, required kernel-mutant RED column, own dispatch/label-only workflow). NOT yet implemented. Sequenced after unit 62, before v0.48.0.

Lifts CONTRACT 61/C16: gates learning OUTCOME (held-out loss, paired by seed) for the fused attention arms. The public per-pair evaluation seam is a refactor of the existing private Trainer::evaluate. Total pod cost of everything on the table: under ~$10 of A100 time.
