# perf/multi-tensor-adamw-r2's own forced-arm A/B — raw runs (provenance)

Raw `jammi-bench finetune-step` output, run on `jammi-a100d` from the
`perf/multi-tensor-adamw-r2` worktree at tip `d95980597afb5373832f6aabdea7eaafb2574248`
(the commit immediately preceding this artifact's own commit — see the
parent artifact JSON's `git_sha`).

- `b8_s128_fused.json.raw`: `JAMMI_KERNELS_STRICT=1 /root/target-ai-core-r2/release/jammi-bench finetune-step --model-dir /root/checkpoints/ModernBERT-large --lora-rank 16 --lora-alpha 32 --target-modules Wqkv,Wo,Wi --backbone-dtype bf16 --cuda 0 --seed 42 --batched-forward true --steps 25 --warmup 5 --batch 8 --seq 128 --lora-dropout 0`
- `b8_s128_disabled.json.raw`: same, plus `JAMMI_KERNELS_DISABLE=adamw_step_fused --expect-kernels-disabled adamw_step_fused`

Renamed `.json.raw` (not `.json`) for the same reason as the sibling
`../a100b/PROVENANCE.md` explains: these are per-leg raw dumps whose headline
numbers are already folded into the parent artifact's
`finetune_step_forced_arm_ab` section (which IS schema-compliant); keeping
the raw dumps out of the schema gate's `*.json` glob avoids a second,
redundant schema wrapper around data the parent artifact already states.
Unlike the `a100b` siblings, these two files' `git_sha` WOULD be resolvable
(this worktree's own tip) — they are kept as `.json.raw` purely for the
above redundancy reason, not because of an unresolved ancestor.
