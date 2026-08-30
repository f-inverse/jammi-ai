# Provenance — `2026-08-30-full-sweep-acce7b3d-a100-pcie/`

The first full, end-to-end run of the committed `ci/scripts/perf/finetune_ab.sh`
producer to completion — the real pod run that found the three defects
recorded as `esc-067-committed-producer-never-executed-end-to-end` in
`.jammi/escapes.jsonl`, executed AFTER those defects were fixed on this
branch (`fix/352-finetune-ab-harness`), so this run is the closing evidence
for the throughput+no-OOM clause the fix set out to discharge, not a defect
reproduction.

## Executed sha

`acce7b3d060d5f7fc7ff5f1f8f0b903a2fcbff71` — "fix(perf): finetune_ab
round-2 audit fold-in — ratio-level two-run completeness, honest scanner
unresolved-reporting (#352)". The two commits that landed AFTER this run
(`eaa7d53f`, the esc-067 ledger row + `closes_escape` citations; `cc826336`,
the `evidence_ref` correction) carry ZERO functional delta against the
three files this producer's own control flow and comparator logic live in
— verifiable directly:

```
git diff acce7b3d..cc826336 -- ci/scripts/perf/finetune_ab.sh ci/scripts/perf/ab_merge.py ci/scripts/perf/identity_fields.py
```

The only hunk that diff produces is a `#`-prefixed comment block added to
`finetune_ab.sh`'s own header (the `closes_escape:` line) — `ab_merge.py`
and `identity_fields.py` are byte-identical between the two shas. This
artifact is therefore valid evidence for the producer as it stands on
`fix/352-finetune-ab-harness` today, not just as it stood at `acce7b3d`.

## Environment

Pod: NVIDIA A100 80GB PCIe. Software: torch 2.13.0+cu126. Both are
OPERATOR-ASSERTED for this run — `ab_merge.py`'s merged report schema
carries no `device_name`/torch-version field today (unlike
`GpuInferenceTier`'s own provenance block), so neither string is
independently re-derivable from `finetune_ab_report.json` the way every
NUMBER quoted below is; stated here as the executing session's own record,
not a claim this README's numbers make traceable.

## Invocation shape

From `finetune_ab_report.json`'s own top level: `"steps": "20", "warmup":
"5"` (both `argv`-sourced strings, never cast to `int` by the merge stage —
quoted here exactly as the JSON stores them), `"pass_ratio_bar": 0.9,
"two_run_protocol": true`, `"lora_init": {"torch":
"jammi", "jammi": "jammi (LoraInitMode::ZerosB; not configurable via
finetune-step's CLI)", ...}` — `AB_TORCH_LORA_INIT=jammi` (torch re-draws
its LoRA `A` from jammi's own bound), the setting `finetune_ab.sh`'s own
header names as the precondition for `loss_final_ratio` to mean anything
as a loss-trajectory-equivalence signal, rather than the throughput-only
default (`peft`). `two_run_protocol: true` confirms `TWO_RUN_PROTOCOL_MARKER`
was written and every config's bar legs are the mandatory four-leg A,B,B,A
shape.

Six configs, the sweep's own fixed `{b8 s128, b8 s512, b16 s128} x
{dropout 0, dropout 0.05}`: `b16-s128-d0`, `b16-s128-d0p05`, `b8-s128-d0`,
`b8-s128-d0p05`, `b8-s512-d0`, `b8-s512-d0p05`.

## Verdicts (4 PASS, 2 INDETERMINATE, 0 FAIL, 0 INVALID)

Every number below is `finetune_ab_report.json#/configs/<slug>/`'s own
`bar_pair_ratios`/`bar_ratio_min_of_two_least_favourable_to_jammi`/
`bar_ratio_indeterminate`/`verdict` fields, quoted directly (also visible
pre-formatted in `finetune_ab_table.txt`'s own summary table, lines 52-58).

| config | pair1 (fused/sdpa) | pair2 (fused-2/sdpa-2) | bar ratio (min) | verdict |
|---|---|---|---|---|
| `b16-s128-d0` | 1.025 | 1.028 | 1.025 | **PASS** |
| `b16-s128-d0p05` | 0.975 | 0.982 | 0.975 | **PASS** |
| `b8-s128-d0` | 1.055 | 0.969 | 0.969 | **INDETERMINATE** |
| `b8-s128-d0p05` | 0.924 | 1.017 | 0.924 | **INDETERMINATE** |
| `b8-s512-d0` | 1.077 | 1.086 | 1.077 | **PASS** — both `s512` dropout arms clear the bar |
| `b8-s512-d0p05` | 1.022 | 1.035 | 1.022 | **PASS** — both `s512` dropout arms clear the bar |

Both `b8-s128` rows are `INDETERMINATE` (never PASS/FAIL) because the
within-run pair spread exceeds the combined estimate's own distance to the
0.9 bar (`bar_ratio_classification`'s own straddle-or-spread rule — see
`ab_merge.py`): `b8-s128-d0` reads `spread=0.087` against
`bar-distance-from-0.9=0.069` (`0.087 > 0.069`); `b8-s128-d0p05` reads
`spread=0.093` against `bar-distance-from-0.9=0.024` (`0.093 > 0.024`),
both printed verbatim in the verdict string itself
(`finetune_ab_report.json#/configs/b8-s128-d0/verdict` and
`.../b8-s128-d0p05/verdict`). This matches the repo's own PRIOR `b8·s128`
spread observation, independently recorded before this branch existed —
`docs/maintainer/fine-tune-performance-guide.md`'s stacked-sweep table
("`b8·s128 | 0.1307 | 0.1319 (r1/r2 spread 8.3%)`") and its own caveat
prose ("the `b8·s128` torch leg's own spread exceeds the margin at that
shape"): `b8·s128` is a shape this repo has flagged as noisy near the 0.9
bar before, on a DIFFERENT sweep and a DIFFERENT sha — this run's own
`INDETERMINATE` classification is the A,B,B,A protocol correctly refusing
to paper over that same noise with a single-sample PASS/FAIL, not a new or
surprising finding.

`jammi_fused_dispatch_proof` AND `jammi_fused_dispatch_proof_second_run`
both read `true` for all six configs (`finetune_ab_report.json#/configs/
<slug>/jammi_fused_dispatch_proof{,_second_run}`) — the fused-dispatch
positive-proof channel cleared on BOTH runs of the bar pair, every config;
no config carries an `INVALID` verdict. `leg_premise_violations` and
`leg_premise_violations_second_run` both read `[]` (checked, clean) for
every config — the same-run premise checks found no drift. NOTE:
`leg_premise_violations_cross_run` reads `null` (never `[]`) on every
config despite every relevant leg (`jammi-fused`, `jammi-fused-2`,
`torch-sdpa`, `torch-sdpa-2`) being `OK` throughout this run — this is a
genuine ambiguity in `ab_merge.py`'s own F3 cross-run check (it only ever
assigns a NON-`None` value on the branch that FINDS a violation, so
"checked and clean" and "never checked" both currently read `null`); it
does not affect any verdict above (the override this field feeds is
`if cross_run_premise_violations_list:`, falsy either way), but this
README does not claim the cross-run check positively confirmed agreement
here — only that no violation is recorded. Flagged as a follow-up finding,
not fixed in this commit (fixing it would break this README's own
"zero functional delta since `acce7b3d`" claim above).

## jammi-eager (context leg) OOM — four configs, not one

`jammi-eager` is a single, non-repeated context leg (never part of the bar
ratio) that forces every fused op eager via the full nine-key
`JAMMI_KERNELS_DISABLE` list — see `finetune_ab_table.txt`'s own per-row
`kernels_disabled_requested`/`kernels_disabled_fired` lines for
`b8-s128-d0`/`b8-s128-d0p05`, both reading the complete nine-key set,
confirming the negative control fired as designed. Fully-eager mode has no
fused-kernel memory savings, and its `vram_delta` dwarfs the fused leg's at
matched shape: `b8-s128-d0`'s own `jammi-eager` row reads `57,329,844,224`
bytes (~53 GiB) against `jammi-fused`'s `3,877,634,048` (~3.6 GiB) —
`finetune_ab_table.txt` lines 21/24. On an 80 GiB device, that memory
profile OOMs at the three LARGER shapes:

* `b16-s128-d0` — `jammi-eager` `OOM` (`finetune_ab_table.txt` line 7's
  own row; its `->` error line, 8: `DriverError(CUDA_ERROR_OUT_OF_MEMORY,
  "out of memory")`)
* `b16-s128-d0p05` — `jammi-eager` `OOM` (row line 14, error line 15, same
  driver error)
* `b8-s512-d0` — `jammi-eager` `OOM` (row line 37, error line 38: `LoRA:
  LoRA tensor: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`)
* `b8-s512-d0p05` — `jammi-eager` `OOM` (row line 44, error line 45:
  `Tensor: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`)

Only `b8-s128-d0`/`b8-s128-d0p05` — the smallest shape swept — completed
`jammi-eager` as `OK`. Each OOM'd row is classified per-row (outcome
`OOM`, table row printed, sweep continues — `run_leg`'s own "one config
OOM-ing tells you something; it must not hide the other five" doctrine)
without invalidating the sweep or any OTHER config's verdict: `jammi-eager`
never feeds `bar_pair_ratio`/`fused_proof`/the leg-premise checks the six
verdicts above are computed from, so all six PASS/INDETERMINATE
classifications stand regardless of `jammi-eager`'s own outcome.

## Clause mapping (`finetune_ab.sh`'s own header)

This run discharges #352's FIRST clause only — throughput + no-OOM (the
PASS/INDETERMINATE bar above, against a synthetic cost-fixture step). The
SECOND clause, loss-TRAJECTORY equivalence (jammi-fused vs jammi-eager,
a real trainer, >= 5 seeds), is discharged separately by the pre-registered
real-trainer instrument at `docs/plans/63-how-well/measurements/
campaign-v2` — never by this producer, which never runs a real trainer or
a held-out eval; the `loss_first`/`loss_last`/`loss_final_ratio` columns
this run's own table prints are, verbatim from `finetune_ab.sh`'s own
header, "SAME DATA, COST FIXTURE — NOT A QUALITY RESULT" (the phrase
`finetune_ab.sh` uses to point at `finetune_step.rs`'s own "Honesty about
what is measured" section, which THIS producer's table columns only
partially discharge).

## Files

* `finetune_ab_table.txt` — the printed table, byte-identical to what the
  real subprocess wrote.
* `finetune_ab_report.json` — the merged JSON report, byte-identical to
  what `ab_merge.py`'s real entry point wrote.

Both copied verbatim from the executing session's own output (no hand
edits) — the same "tracked-input fixture, never a hand-rolled dict"
discipline `ci/scripts/perf/fixtures/p6_fa2_dense_raw_runs/PROVENANCE.md`
already documents for this directory's sibling convention.
