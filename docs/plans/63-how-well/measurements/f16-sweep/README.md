# F16 backbone-dtype sweep (campaign #443, W4-bench) — status: INVALID (both reasons pre-registered, expected)

**Numbers-not-registered-with-the-citation-checker notice:** unlike `campaign-v1`/
`campaign-v2`/`dose-ladder`/`red-proof` in this directory, this README does NOT use
the `<!-- claims63: ... -->` citation convention `ci/scripts/check_measurement_claims.py`
enforces — that checker's `MEASUREMENT_FILES`/`FILE_TOKEN_DENOMINATOR` allowlist is
docs-ci-owned, and wiring a new file into it is a distinct, coordinated change this
bench-scoped task did not make. Every number below is instead a direct, literal copy
from the committed `finetune_run_ab_report.json`/`raw/*.json` in this same directory —
re-derive any of them with `python3 -c "import json; print(json.load(open('finetune_run_ab_report.json'))['...'])"`
against the committed file to check.

## OPERATOR-APPROVED REDUCED SEED SET — disclosed prominently

This run used `FINETUNE_RUN_AB_SEEDS=1,2,3` (**N=3**), a pre-registered, disclosed
reduction from the pre-registered N=12 gate set the committed **bf16** goldens
(`../campaign-v1/`, `../campaign-v2/`) used. This reduction was operator-approved for
campaign #443 (W4-bench: f16 perf legs through the provenance machinery) specifically
to exercise the merger/provenance machinery under a new `backbone_dtype`, not to
re-litigate the bf16 A/B verdict at full N. **No sign test / GREEN-RED-INVALID decision
at this N should be read as a statistically powered replication of campaign-v2** — see
"Why status is INVALID" below, reason (a).

## Run

- Producer: `ci/scripts/perf/finetune_run_ab.sh`, extended by this same change to add
  `FINETUNE_RUN_AB_BACKBONE_DTYPE` (default `bf16`, byte-identical to the script's prior
  hardcoded `--backbone-dtype bf16` when unset) — passed to **every** leg `run_leg` runs
  (both A/B arms, all 3 seeds, AND the lr=0 control), preserving identity-field #10
  (`backbone_dtype`, `FINETUNE_RUN_IDENTITY_FIELDS` in `identity_fields.py`)
  cross-arm/cross-seed homogeneity — confirmed by this run's own committed
  `finetune_run_ab_report.json#/cross_seed_identity_violations` = `[]` (empty).
- Env for this run: `FINETUNE_RUN_AB_SEEDS=1,2,3`, `FINETUNE_RUN_AB_BACKBONE_DTYPE=f16`,
  `FINETUNE_RUN_AB_LR0_SEEDS=1,2` (the standard "lr=0 x2 seeds" control, matching
  `campaign-v2`'s own choice), everything else at the script's own defaults (MNRL
  objective, 3 epochs, batch 32, `--lr` unset ⇒ CLI default 2e-4, early-stopping
  patience 10000, `--cuda 0`).
- Pod: RunPod session `a100b` (NVIDIA A100 80GB PCIe), tree `w4b`
  (`ci/scripts/gpu-dev.sh push a100b --tree w4b` + `target a100b w4b`, cutlass
  rsynced in separately). MODEL_DIR: `answerdotai/ModernBERT-large`, fetched fresh via
  `huggingface_hub.snapshot_download` into `/root/checkpoints/ModernBERT-large`
  (config.json + model.safetensors + tokenizer.json present).
- Build: `cargo build --release -p jammi-bench --features cuda,jammi-encoders/flash-attn`
  (the script's own build step). Every committed leg's own `provenance.build_features`
  reads `["bench-cuda", "cuda", "flash-attn"]` — flash-attn WAS compiled in, matching
  the campaign-v2/campaign-v1 build feature list exactly (`raw/seed1__fused__r1.json#/provenance`).
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

| seed | fused `held_out_example_mean` | alloff `held_out_example_mean` | d_i = fused − alloff | r1/r2 delta (fused, alloff) |
|---|---|---|---|---|
| 1 | 3.335803858935833  | 3.368664890527725  | −0.032861 | 0.0, 0.0 |
| 2 | 3.2585181072354317 | 3.318568527698517  | −0.060050 | 0.0, 0.0 |
| 3 | 3.3537766486406326 | 3.2915777936577797 | +0.062199 | 0.0, 0.0 |

(`held_out_example_mean` from `raw/seed{N}__{arm}__r1.json#/tiers/finetune_run/held_out_example_mean`;
`d_i` and `r1/r2 delta` copied verbatim from `finetune_run_ab_report.json#/per_seed/{N}/d_i` and
`#/per_seed/{N}/r1_r2_delta`.)

2 of 3 seeds read negative (fused better / lower held-out loss) — directionally
consistent with campaign-v2's own bf16 trend (`mean_d=-0.02008`, 8/12 negative), but
**N=3 is far too small to draw any sign-test conclusion** — this is a raw-number
report, not a re-run of the bf16 verdict at reduced power.

- **Determinism floor: exactly 0.0** (`#/determinism_floor/max_delta`, printed as
  `max_r1_r2_delta` in `finetune_run_ab_table.txt`'s own display label) — every
  r1/r2 same-seed repeat is bit-identical at f16 too, the same bar `campaign-v1`/
  `campaign-v2` hold at bf16. `cross_seed_spread` = 0.052409212771387624
  (`#/determinism_floor/cross_seed_spread`, informational only — not a gate).
- **lr=0 RED control: 0 violations**, both seeds, both arms
  (`#/lr0_control/violations` = `[]`). Every control leg's own `learning_happened_delta`
  reads exactly `0.0` over a CONSTANT, finite `train_probe_series`
  (`[3.325727254152298] * 4` for both seeds/arms) — the floor bites on a genuinely
  flat, non-diverged trajectory, not a NaN/inf path a naive `> threshold` check would
  have let through silently (family-F non-vacuous-control discipline).
- **Cross-seed/cross-arm identity homogeneity: clean**
  (`#/cross_seed_identity_violations` = `[]`, `#/wrong_seed_count` = `false`) — every
  leg (both arms, all 3 main seeds, both lr=0 control seeds) reports the same
  `backbone_dtype="f16"`, proving `FINETUNE_RUN_AB_BACKBONE_DTYPE`'s single-read,
  shared-`run_leg` plumbing actually reached every leg, not just the `fused` arm.
- **The flash cascade genuinely dispatches under f16 on real A100 hardware**: every
  `fused` leg reports `attention_block_flash_fused_dispatches=3276` (identical count to
  the bf16 campaigns' own fused legs, e.g. `campaign-v1/probe/p3-fused-mnrl.json`) with
  `flash_compiled=true`; every `alloff` leg reports the flash cascade DECLINED
  (`attention_block_flash_fused_dispatches=0`, `..._declined_dispatches=3276`) with the
  block-arm fallback correctly engaging instead (`attention_block_fused_dispatches=3276`)
  — the fused/alloff differential is real and correctly distinguished at f16, not a
  null experiment.

## Why `status` is INVALID (both reasons pre-registered, neither a plumbing bug)

1. **`gate_seed_count` = 3, not the pre-registered 12.** `ab_merge.py`'s own decision
   rule text is explicit: "the rule is pre-registered FOR 12 seeds; never rescaled
   silently" — this is the disclosed reduction above, working exactly as designed
   (`sign_test: n/a`, `decision: null`).
2. **Every `fused` leg fails `ab_merge.py`'s own CONTRACT 63 Frame premise (unit-63
   round-4 audit F-1):** `leg_premise_violations` for every seed reads `VIOLATED (2)` —
   the merger's `dispatch_arm_consistency` check pre-registers the flash-cascade
   differential as **BF16-only** ("CONTRACT 63 Frame pre-registers the flash cascade
   as this arm's own admitted branch, and that branch is BF16-only ... a `fused` leg
   run at any other dtype cannot exercise the pre-registered differential" —
   `ab_merge.py`'s own comment, unchanged by this task). This is a genuine,
   **correctly-firing, pre-registered CONTRACT gap**, not a merger bug and not
   something this bench-scoped change silently patches: the underlying engine
   capability (`flash_capability_gates` / `dtype_is_bf16_or_f16`,
   `jammi-encoders/src/modernbert.rs`, landed on the `fix/443-fa2-f16-capability`
   branch this task merged in) now admits F16 alongside BF16 for real dispatch — as
   the dispatch-count evidence above shows — but `ab_merge.py`'s own CONTRACT 63
   Frame premise text has NOT been re-registered to recognize F16 as an admitted
   flash-cascade dtype for THIS specific A/B differential. Extending CONTRACT 63
   Frame (and `ab_merge.py`'s premise check) to admit F16 is a distinct, docs-ci-owned
   governance action — flagged here as a follow-up, not performed by this task.

Both reasons are independently sufficient for INVALID; a future N=12, CONTRACT-amended
f16 re-run would need reason (2) resolved first, or its own fused-arm verdict will
read INVALID regardless of seed count.

## Files

- `finetune_run_ab_report.json` / `finetune_run_ab_table.txt` — `ab_merge.py`'s merged
  output, `finetune-run` mode, invocation
  `finetune-run raw/ out/ 1,2,3 1,2` (no `--allow-missing-lr0-control`,
  no `--mutant-legs`).
- `raw/` — all 16 leg reports (12 main: 3 seeds x {fused,alloff} x {r1,r2}, 4 lr=0
  control: 2 seeds x {fused,alloff}), each with its own `.exit`/`.stderr` sidecar; all
  16 legs exited 0.
