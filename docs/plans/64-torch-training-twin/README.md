# 64 — End-to-end PyTorch training twin (front-door record)

**Status:** SCOPED (gap-analyzer verdict: invariant-crossing, 2026-08-28). NOT planned, NOT
implemented. Dispatch is CONTINGENT on unit 63's H5 dynamic-range verdict being ADEQUATE —
if 63 records a negative dynamic-range result, this unit is void (no instrument to twin).
Sequenced after 63 lands.

## The ask

Unit 63 proves the fused arms match jammi's own ALLOFF arm in learning outcome; the
grad-oracle/torch-parity artifacts anchor jammi's operators to the PyTorch reference at
step level. This unit adds the direct end-to-end link: train the committed fixture in
PyTorch under 63's protocol and show jammi's whole-training outcome matches PyTorch's.

## Binding scope corrections (gap-analyzer, verified against the tree)

1. jammi NEVER shuffles (data.rs — .chunks over committed order, deterministic split): the
   batch partition is a CONSTRAINT imposed on the twin (torch DataLoader shuffle=False,
   partition verified via heldout_batch_partition_sha256), not noise to absorb.
2. "Same committed checkpoint bytes" is per the stacked_sweep precedent: shared pod-local
   MODEL_DIR + recorded sha256 both sides (weights are not in the repo); the sha is a
   required identity field and premise leg.
3. The unshared RNG is exactly: the LoRA-A init draw AND lora_dropout's per-step mask.
4. The standing ruling in compare_grad_oracle.py:13-21 (a jammi-vs-torch loss comparison
   "is not a substitute") MUST be explicitly superseded, with the reasoning stated: this
   instrument is distributional over seeds and never pairs trajectories.
5. The guide's own instrument (fine-tune-performance-guide.md:324) is CONTAINMENT over
   three arms, not a two-sample test — torch enters as a THIRD ARM of 63's matrix.

## Provisional lead rulings (the unit's own pressure-test refines; changes are amendments)

- PRIMARY instrument: containment per guide:324 — the torch arm inside ALLOFF's measured
  seed spread and vice versa; a quantitative equivalence margin (TOST-shaped) only as a
  SECONDARY, and only if the margin derives from measured floors, off-sample verified on
  BOTH sides (torch-side off-sample seeds are budgeted, not borrowed).
- Comparison arm: ALLOFF vs torch (the baseline anchor). Transitive composition to the
  fused arm is stated WITH additive margins (delta_total = delta_63 + delta_64), never as
  free transitivity.
- Same-operating-point rule (esc-045): BOTH samples measured in ONE session on one
  pod/driver/wheel set — 63's artifacts are corroboration, never the second sample.
  Honest budget: ~24 runs + off-sample, not ~12.
- lora_dropout: primary twin runs at lora_dropout=0.0 BOTH sides (a controlled comparison
  with its own honest identity tuple), plus a jammi-only 0.05-vs-0.0 spread-contribution
  measurement so the deviation from 63's protocol is quantified, pre-registered.
- Objective: the one 63's probe SELECTED, only (twin code is expensive; both-objectives is
  scope creep until needed). Tokenizer parity is verified at TOKEN-ID level over the
  fixture (a cheap equality leg), not just byte-identical tokenizer.json; truncation/
  padding policy pinned both sides.
- Script home: crates/jammi-bench/reference/ (the torch-reference precedent); NO
  requirements/pyproject pin files (recorded B2 ruling) — versions in README + artifact
  provenance. Evidence-only, never a merge gate (torch is not on the CI image); the
  artifact registers via its own gate-adjacent commit (human-merged).
- OQ4 (torch_encode.py, the ENCODE twin) is NOT discharged by this unit and stays open.
- The twin's plan must enumerate reproduced-vs-absorbed trainer behaviors explicitly
  (schedule, warmup, weight decay, grad-accum, clipping, eval cadence) — nothing absorbed
  silently.

## Invariants crossed (from the scope verdict)

B1 (any new numerics equivalence primitive + the twin script must pass the discipline
test), B2 (reference-dir conventions; no pin files; no cookbook/book reference from
crates/**), K7 (twin artifact identity folds torch/transformers/peft versions, checkpoint
sha, tokenizer digest, fixture shas, objective params, seed set), K2 (typed refusals on
the statistic's input edges), B6 (one-PR atomicity with gate-adjacent edits staged
human-merged). Plus disciplines: KO-4 no-number-without-producer, esc-045 floors,
guide:358 range-before-data.
