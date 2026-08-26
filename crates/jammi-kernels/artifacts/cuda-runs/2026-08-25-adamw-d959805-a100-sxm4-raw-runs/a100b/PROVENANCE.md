# a100b full-step A/B — raw runs (provenance)

These eight `*.json.raw` files are the lead's own `jammi-bench finetune-step`
raw output, run by hand on the lead's exclusive `a100b` box, forced onto each
arm via `JAMMI_KERNELS_DISABLE=adamw_step_fused` (disabled) vs its absence
(fused), two repeats (`r1`/`r2`) at two shapes (`s128`, `s512`), batch 8,
bf16 backbone. Copied here verbatim (byte-for-byte, `cp`, no field edited)
from `scratchpad/pod/a100b-adamw-ab/*.json` in the main checkout.

Source paths (main checkout, not this worktree):
- `scratchpad/pod/a100b-adamw-ab/b8_s128_disabled.r1.json`
- `scratchpad/pod/a100b-adamw-ab/b8_s128_disabled.r2.json`
- `scratchpad/pod/a100b-adamw-ab/b8_s128_fused.r1.json`
- `scratchpad/pod/a100b-adamw-ab/b8_s128_fused.r2.json`
- `scratchpad/pod/a100b-adamw-ab/b8_s512_disabled.r1.json`
- `scratchpad/pod/a100b-adamw-ab/b8_s512_disabled.r2.json`
- `scratchpad/pod/a100b-adamw-ab/b8_s512_fused.r1.json`
- `scratchpad/pod/a100b-adamw-ab/b8_s512_fused.r2.json`

## Why `.json.raw`, not `.json`

`ci/scripts/check_cuda_run_artifacts.py` (not yet merged; `ci/cuda-run-artifact-gate`
@ 61cfd25) validates every `*.json` under `cuda-runs/` — including
`*-raw-runs/` legs — against a schema requiring `schema_version`, `git_sha`
(40-hex, an ancestor of `HEAD`) or a reviewed `git_sha_unresolved` +
`producer.kind == "none"`, `box`, `producer`, and `status`. These eight files
are raw `jammi-bench` tier output with none of those fields, and their git
provenance is genuinely UNRESOLVED from here: they were run by the lead on a
box/branch state this worktree has no record of (not `perf/multi-tensor-adamw-r2`'s
own history — this branch deliberately excludes PR #373's device-clip work
these runs may have been measured alongside). `git_sha_unresolved` requires
`producer.kind == "none"`, and the gate's `LEGACY_NONE_ALLOWLIST` is closed to
new entries — so there is no schema-compliant way to name these files `.json`
without either fabricating a `git_sha` this worktree cannot verify, or
requesting a new allow-list entry from the file that owns that list (out of
this crate's shared-declaration scope). Renamed `.json.raw` instead: preserved
verbatim as reference data, outside the schema gate's `*.json` glob, so the
gate is never asked to accept an unverifiable claim.

## What the numbers show (folded into the parent artifact's `a100b_full_step_ab_reference`)

Full-step (forward+backward+AdamW) p50, batch 8, bf16, `--target-modules
Wqkv,Wo,Wi`, `--lora-rank 16`:

| shape | disabled (eager) r1/r2 | fused r1/r2 |
|---|---|---|
| s512 | 0.675926908 / 0.675146455 | 0.658921588 / 0.659359990 |
| s128 | 0.198615414 / 0.199047347 | 0.182207528 / 0.182322336 |

`adamw_fused_dispatches`/`adamw_eager_dispatches`: 0/6720 (disabled) vs
6720/0 (fused) on every one of the eight runs — 224 trainable tensors × 30
total step-loop iterations (steps + warmup, per each run's own params).
