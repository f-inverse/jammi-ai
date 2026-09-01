# F16 backbone-dtype sweep (campaign #443, W4-bench) — status: INVALID (premise violation fired; see below)

Every number below is bound to the artifact that produced it by a `claims63:` tag on
the line above it, and checked by `ci/scripts/check_measurement_claims.py` (this README
is registered in that gate's `MEASUREMENT_FILES`). The artifacts are committed beside
this file: `finetune_run_ab_report.json` and `raw/*.json`.

## OPERATOR-APPROVED REDUCED SEED SET — disclosed prominently

<!-- claims63: default=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json; c1=count(#/seeds, 'len') -->
This run used `FINETUNE_RUN_AB_SEEDS` = seeds 1, 2, 3 (**N=3**), a pre-registered, disclosed
<!-- claims63: c1=docs/plans/63-how-well/measurements/campaign-v2/finetune_run_ab_report.json#/decision/gate_seed_count -->
reduction from the pre-registered N=12 gate set the committed **bf16** goldens
(`../campaign-v1/`, `../campaign-v2/`) used. This reduction was operator-approved for
campaign #443 (W4-bench: f16 perf legs through the provenance machinery) specifically
to exercise the merger/provenance machinery under a new `backbone_dtype`, not to
re-litigate the bf16 A/B verdict at full N. **No sign test / GREEN-RED-INVALID decision
at this N should be read as a statistically powered replication of campaign-v2** — as it
happens this run's own premise violation (see "Why `status` is INVALID" below) means no
sign test ran at all, so this caveat is design context for what a hypothetical
<!-- claims63: default=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json; c1=count(#/seeds, 'len') -->
premise-clean run at N=3 would still not deliver, not a report-cited cause of the
`INVALID` this artifact actually recorded.

## Run

- Producer: `ci/scripts/perf/finetune_run_ab.sh`, extended by this same change to add
  `FINETUNE_RUN_AB_BACKBONE_DTYPE` (default `bf16`, byte-identical to the script's prior
  hardcoded `--backbone-dtype bf16` when unset) — passed to **every** leg `run_leg` runs
<!-- claims63: default=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json; c1=count(#/seeds, 'len') -->
  (both A/B arms, all 3 seeds, AND the lr=0 control), preserving identity-field #10
  (`backbone_dtype`, `FINETUNE_RUN_IDENTITY_FIELDS` in `identity_fields.py`)
  cross-arm/cross-seed homogeneity — confirmed by this run's own committed
  `finetune_run_ab_report.json#/cross_seed_identity_violations` = `[]` (empty).
- Env for this run: `FINETUNE_RUN_AB_SEEDS` = seeds 1, 2, 3; `FINETUNE_RUN_AB_BACKBONE_DTYPE=f16`;
  `FINETUNE_RUN_AB_LR0_SEEDS` = seeds 1, 2 (the standard "lr=0 x2 seeds" control, matching
  `campaign-v2`'s own choice), everything else at the script's own defaults (MNRL
<!-- claims63: default=docs/plans/63-how-well/measurements/f16-sweep/raw/seed1__fused__r1.json; c1=#/tiers/finetune_run/epochs; c2=#/tiers/finetune_run/batch; c3=#/tiers/finetune_run/lr -->
  objective, 3 epochs, batch 32, `--lr` unset ⇒ CLI default 2e-4, early-stopping
<!-- claims63: default=docs/plans/63-how-well/measurements/f16-sweep/raw/seed1__fused__r1.json; c1=#/tiers/finetune_run/early_stopping_patience -->
  patience 10000, and `--cuda` at the pod's only device (`device_name`, recorded per leg).
- Pod: RunPod session `a100b` (NVIDIA A100 80GB PCIe), tree `w4b`
  (`ci/scripts/gpu-dev.sh push a100b --tree w4b` + `target a100b w4b`, cutlass
  rsynced in separately). MODEL_DIR: `answerdotai/ModernBERT-large`, fetched fresh via
  `huggingface_hub.snapshot_download` into `/root/checkpoints/ModernBERT-large`
  (config.json + model.safetensors + tokenizer.json present).
- Build: `cargo build --release -p jammi-bench --features cuda,jammi-encoders/flash-attn`
  (the script's own build step). Every committed leg's own `provenance.build_features`
  reads `["bench-cuda", "cuda", "flash-attn"]` — flash-attn WAS compiled in, matching
  the campaign-v2/campaign-v1 build feature list exactly (`raw/seed1__fused__r1.json#/provenance/build_features`).
- **Provenance cross-check residual (round-6 audit item D, `ci/scripts/gpu-dev.sh`'s own
  documented gap):** a tree populated purely by `push` carries no `.git` at all, so
  `finetune_run_ab.sh`'s own `git -C "$REPO_ROOT" rev-parse HEAD` provenance cross-check
  (CONTRACT C5.1) cannot resolve against the pushed tree as-is. Worked around by
  `git init`-ing a SINGLE, pod-local commit inside `/root/trees/w4b` immediately after
  push+cutlass-rsync (`git add -A && git commit`), then building with the resulting
  pod-local commit sha as `JAMMI_BUILD_SHA` for the FIRST build only (the script's own
  subsequent internal rebuild recomputes the identical sha itself, since the tree is
  unchanged) — every committed leg's `provenance.build_sha` (`1ca806b914d860674bf06cd871d1ea2640c35e1d`)
  is therefore a **pod-local, single-commit git identity**, self-consistent with what
  the script's own `git rev-parse HEAD` cross-check saw on this pod, but it is NOT a
  publicly-resolvable commit on this repo's real history — the REAL source state this
  pod ran is the actual commit this change is committed at on the `worktree-agent-ae8e6e21665dedf14`
  branch (see the hand-off SHA), whose exact working-tree content was rsynced to the pod
  unmodified (verified: `verify-train-pairs.py`/leg exit codes below all clean, and the
  script's own build ran the tree as-pushed, no local pod edits made to any tracked
  source file before the sweep started).

## Headline numbers — per-seed fused-vs-alloff deltas (measured, f16)

| seed | fused `held_out_example_mean` | alloff `held_out_example_mean` | d_i = fused - alloff | r1/r2 delta (fused, alloff) |
|---|---|---|---|---|
<!-- claims63: c1=docs/plans/63-how-well/measurements/f16-sweep/raw/seed1__fused__r1.json#/tiers/finetune_run/seed; c2=docs/plans/63-how-well/measurements/f16-sweep/raw/seed1__fused__r1.json#/tiers/finetune_run/held_out_example_mean; c3=docs/plans/63-how-well/measurements/f16-sweep/raw/seed1__alloff__r1.json#/tiers/finetune_run/held_out_example_mean; c4=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json#/per_seed/1/d_i; c5=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json#/per_seed/1/r1_r2_delta/fused; c6=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json#/per_seed/1/r1_r2_delta/alloff -->
| 1 | 3.335803858935833  | 3.368664890527725  | -0.032861 | 0.0, 0.0 |
<!-- claims63: c1=docs/plans/63-how-well/measurements/f16-sweep/raw/seed2__fused__r1.json#/tiers/finetune_run/seed; c2=docs/plans/63-how-well/measurements/f16-sweep/raw/seed2__fused__r1.json#/tiers/finetune_run/held_out_example_mean; c3=docs/plans/63-how-well/measurements/f16-sweep/raw/seed2__alloff__r1.json#/tiers/finetune_run/held_out_example_mean; c4=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json#/per_seed/2/d_i; c5=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json#/per_seed/2/r1_r2_delta/fused; c6=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json#/per_seed/2/r1_r2_delta/alloff -->
| 2 | 3.2585181072354317 | 3.318568527698517  | -0.060050 | 0.0, 0.0 |
<!-- claims63: c1=docs/plans/63-how-well/measurements/f16-sweep/raw/seed3__fused__r1.json#/tiers/finetune_run/seed; c2=docs/plans/63-how-well/measurements/f16-sweep/raw/seed3__fused__r1.json#/tiers/finetune_run/held_out_example_mean; c3=docs/plans/63-how-well/measurements/f16-sweep/raw/seed3__alloff__r1.json#/tiers/finetune_run/held_out_example_mean; c4=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json#/per_seed/3/d_i; c5=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json#/per_seed/3/r1_r2_delta/fused; c6=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json#/per_seed/3/r1_r2_delta/alloff -->
| 3 | 3.3537766486406326 | 3.2915777936577797 | +0.062199 | 0.0, 0.0 |

(`held_out_example_mean` from `raw/seed{N}__{arm}__r1.json#/tiers/finetune_run/held_out_example_mean`;
`d_i` and `r1/r2 delta` copied verbatim from `finetune_run_ab_report.json#/per_seed/{N}/d_i` and
`#/per_seed/{N}/r1_r2_delta`.)

<!-- claims63: c1=negcount('docs/plans/63-how-well/measurements/f16-sweep/raw/seed*__fused__r1.json', 'docs/plans/63-how-well/measurements/f16-sweep/raw/seed*__alloff__r1.json', 'held_out_example_mean'); c2=paircount('docs/plans/63-how-well/measurements/f16-sweep/raw/seed*__fused__r1.json', 'docs/plans/63-how-well/measurements/f16-sweep/raw/seed*__alloff__r1.json', 'held_out_example_mean') -->
2 of 3 seeds read negative (fused better / lower held-out loss) — directionally
<!-- claims63: default=docs/plans/63-how-well/measurements/campaign-v2/finetune_run_ab_report.json; c1=#/decision/mean_d; c2=#/decision/n_neg; c3=#/decision/gate_seed_count -->
consistent with campaign-v2's own bf16 trend (`mean_d=-0.02008`, 8/12 negative), but
<!-- claims63: default=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json; c1=count(#/seeds, 'len') -->
**N=3 is far too small to draw any sign-test conclusion** — this is a raw-number
report, not a re-run of the bf16 verdict at reduced power.

<!-- claims63: default=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json; c1=#/determinism_floor/max_delta -->
- **Determinism floor: exactly 0.0** (`#/determinism_floor/max_delta`, printed as
  `max_r1_r2_delta` in `finetune_run_ab_table.txt`'s own display label) — every
  r1/r2 same-seed repeat is bit-identical at f16 too, the same bar `campaign-v1`/
<!-- claims63: default=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json; c1=#/determinism_floor/cross_seed_spread -->
  `campaign-v2` hold at bf16. `cross_seed_spread` = 0.052409212771387624
  (`#/determinism_floor/cross_seed_spread`, informational only — not a gate).
<!-- claims63: default=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json; c1=count(#/lr0_control/violations, 'len') -->
- **lr=0 RED control: 0 violations**, both seeds, both arms
  (`#/lr0_control/violations` = `[]`). Every control leg's own `learning_happened_delta`
<!-- claims63: default=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json; c1=#/lr0_control/per_seed/1/per_arm/fused/learning_happened_delta -->
  reads exactly `0.0` over a CONSTANT, finite `train_probe_series`
<!-- claims63: default=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json; c1=min(#/lr0_control/per_seed/1/per_arm/fused/train_probe_series); c2=count(#/lr0_control/per_seed/1/per_arm/fused/train_probe_series, 'len'); c3=absdiff(max(#/lr0_control/per_seed/1/per_arm/fused/train_probe_series), min(#/lr0_control/per_seed/1/per_arm/fused/train_probe_series)) -->
  (`[3.325727254152298] * 4` for both seeds/arms, max-min spread exactly 0.0) — the floor bites on a genuinely
  flat, non-diverged trajectory, not a NaN/inf path a naive `> threshold` check would
  have let through silently (family-F non-vacuous-control discipline).
- **Cross-seed/cross-arm identity homogeneity: clean**
  (`#/cross_seed_identity_violations` = `[]`, `#/wrong_seed_count` = `false`) — every
<!-- claims63: default=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json; c1=count(#/seeds, 'len') -->
  leg (both arms, all 3 main seeds, both lr=0 control seeds) reports the same
  `backbone_dtype="f16"`, proving `FINETUNE_RUN_AB_BACKBONE_DTYPE`'s single-read,
  shared-`run_leg` plumbing actually reached every leg, not just the `fused` arm.
- **The flash cascade genuinely dispatches under f16 on real A100 hardware**: every
<!-- claims63: c1=docs/plans/63-how-well/measurements/f16-sweep/raw/seed1__fused__r1.json#/tiers/finetune_run/attention_block_flash_fused_dispatches -->
  `fused` leg reports `attention_block_flash_fused_dispatches=3276` (identical count to
  the bf16 campaigns' own fused legs, e.g. `campaign-v1/probe/p3-fused-mnrl.json`) with
  `flash_compiled=true`; every `alloff` leg reports the flash cascade DECLINED
<!-- claims63: default=docs/plans/63-how-well/measurements/f16-sweep/raw/seed1__alloff__r1.json; c1=#/tiers/finetune_run/attention_block_flash_fused_dispatches; c2=#/tiers/finetune_run/attention_block_flash_declined_dispatches -->
  (`attention_block_flash_fused_dispatches=0`, `..._declined_dispatches=3276`) with the
<!-- claims63: c1=docs/plans/63-how-well/measurements/f16-sweep/raw/seed1__alloff__r1.json#/tiers/finetune_run/attention_block_fused_dispatches -->
  block-arm fallback correctly engaging instead (`attention_block_fused_dispatches=3276`)
  — the fused/alloff differential is real and correctly distinguished at f16, not a
  null experiment.

## Why `status` is INVALID

**The report's actual, sole recorded cause: every `fused` leg fails `ab_merge.py`'s own
CONTRACT 63 Frame premise (unit-63 round-4 audit finding 1).** Every seed's
<!-- claims63: default=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json; c1=count(#/per_seed/1/leg_premise_violations, 'len') -->
`#/per_seed/{N}/leg_premise_violations` holds exactly 2 entries (`fused r1`, `fused r2`),
each reading `"...counters claim a dispatch the declared dtype forbids
(flash_capability_gates admits only BF16, modernbert.rs's own dtype_is_bf16 gate)..."`
— at the time this sweep's own merge ran, `ab_merge.py`'s
`finetune_run_dispatch_proof_violations` premise pre-registered the flash-cascade
differential as **BF16-only**, and every
`fused` leg here declared `backbone_dtype="f16"` while counting a positive
`attention_block_flash_fused_dispatches`, which that premise (correctly, as written at
the time) refused as internally contradictory. Because zero seeds clear the premise,
`#/decision` is `null` — so this report carries no `clean_seed_count` field at all —
and `#/sign_test` is `null`; the merger never reaches the point of comparing a
<!-- claims63: c1=docs/plans/63-how-well/measurements/campaign-v2/finetune_run_ab_report.json#/decision/gate_seed_count -->
clean-seed count against the pre-registered N=12 gate. **There is no separate,
independently-fired `gate_seed_count`-based reason in this report** —
`finetune_run_ab_report.json` has no top-level `gate_seed_count` field (it would live
under `#/decision`, which is `null` here), and `#/wrong_seed_count` reads `false`.

<!-- claims63: default=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json; c1=count(#/seeds, 'len') -->
**Design context, not a second report-cited reason:** the N=3 reduction disclosed above
is a real, pre-registered, operator-approved fact of THIS sweep's design, independent of
whether the premise above fired. Had every leg cleared the premise instead, `ab_merge.py`'s
<!-- claims63: c1=docs/plans/63-how-well/measurements/campaign-v2/finetune_run_ab_report.json#/decision/gate_seed_count -->
own decision-rule text is explicit that its sign-test threshold is "pre-registered FOR 12
<!-- claims63: default=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json; c1=count(#/seeds, 'len') -->
seeds; never rescaled silently" — so a hypothetical premise-clean run at N=3 would still
not have produced a verdict comparable to campaign-v2's own pre-registered, fully-powered
<!-- claims63: default=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json; c1=docs/plans/63-how-well/measurements/campaign-v2/finetune_run_ab_report.json#/decision/gate_seed_count; c2=count(#/seeds, 'len') -->
N=12 read. This is stated here as a caveat on what N=3 CAN and CANNOT deliver even in the
best case, not as a JSON-field this report actually cites for its `INVALID` status.

**Since amended (docs-ci, not performed by this task):** `docs/plans/63-how-well/
CONTRACT.md`'s "Amendment 2026-09-01" (commit `f97326ed`, folded into this worktree after
this artifact was committed) widens the flash-cascade differential's admitted dtype set
from `{BF16}` to `{BF16, F16}`, tracking `flash_capability_gates`'s own
`dtype_is_bf16_or_f16` widening, and updates `ab_merge.py`'s premise checks to match —
motivated explicitly by this sweep's own dispatch-count evidence (above). That amendment's
own commit message states it deliberately does NOT re-run or re-merge this committed
artifact: this README's `INVALID` status is honest history for the premise this sweep's
own merge actually ran under, not retroactively recomputed. A future f16 fused-arm sweep
merged under the amended `ab_merge.py` would no longer trip this specific premise —
whether it produces a non-`INVALID` verdict would then depend on the (separate, still
pre-registered-for-12) seed-count gate the design-context paragraph above describes.

## Files

- `finetune_run_ab_report.json` / `finetune_run_ab_table.txt` — `ab_merge.py`'s merged
  output, `finetune-run` mode, invocation
  `finetune-run raw/ out/` over seeds 1, 2, 3 with lr=0 control seeds 1, 2 (no
  `--allow-missing-lr0-control`, no `--mutant-legs`).
<!-- claims63: default=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json; c1=count(#/seeds, 'len') -->
- `raw/` — one leg report per (seed, arm, repeat): 3 seeds x {fused,alloff} x {r1,r2},
<!-- claims63: default=docs/plans/63-how-well/measurements/f16-sweep/finetune_run_ab_report.json; c1=count(#/lr0_control/seeds, 'len') -->
  plus the lr=0 control legs, 2 seeds x {fused,alloff}; each leg has its own
  `.exit`/`.stderr` sidecar, every one reading exit 0.
