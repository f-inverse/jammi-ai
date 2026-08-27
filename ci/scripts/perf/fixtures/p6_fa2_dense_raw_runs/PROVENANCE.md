# Provenance — `p6_fa2_dense_raw_runs/` fixtures

`s128_flash_on_1.json` and `s128_flash_off_1.json` were copied byte-for-byte
from the two SMALLEST files (70 and 74 source lines respectively) in the 8
real, committed `jammi-bench finetune-step` raw-run reports at:

```
crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-p6-b3-dense-raw-runs/{s128,s512}_flash_{on,off}_{1,2}.json
```

on `origin/perf/p6-fa2-dense` @ commit `5886c6b627c7a943df384a8aaadb7c3d6714e79c`
("perf(kernels): THE NUMBER -- P6 Stage B B3-dense finetune-step, flash vs
block arm, a100c SXM4"), copied via:

```
git show origin/perf/p6-fa2-dense:crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-p6-b3-dense-raw-runs/s128_flash_on_1.json
git show origin/perf/p6-fa2-dense:crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-p6-b3-dense-raw-runs/s128_flash_off_1.json
```

These fixtures PREDATE the cuda-runs schema stamp: the committed copies
under `crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-p6-b3-dense-raw-runs/`
were later stamped with `schema_version`/`git_sha`/`box`/`status`/`producer`
fields, so the fixtures are no longer byte-identical to them — the fixtures
keep the bare pre-stamp `Report` shape as originally copied. The producing
branch has since merged into `main` (the
FA2 dense arm, #389, merge commit `6c526f9`), and `main`'s
`crates/jammi-bench/src/report.rs`/`finetune_step.rs` do carry the
flash-named dispatch-counter fields today. These two files are the REAL shape
`ab_merge.py`'s `dispatch_pairs`/`fused_proof` broke on (a docs-ci co-sign
of that branch found `ci/scripts/perf/ab_merge.py::dispatch_pairs()` raises
`KeyError` on `attention_block_flash_fused_dispatches`, which has no
`attention_block_flash_eager_dispatches` sibling — its fallback counter is
named `attention_block_flash_declined_dispatches` instead) — used here as
REAL, committed, tracked-input fixtures for
`test_ab_merge.py::CascadePairFixtureTests` (never a hand-rolled dict
standing in for what a real `finetune-step` report on that branch actually
emits), per this crate's own "tracked-input fixtures" implementer-
acceptance clause.

`s128_flash_on_1.json`'s `finetune_step` block (`batch=8, seq=128,
bf16`) reads `attention_block_flash_fused_dispatches: 840`,
`attention_block_flash_declined_dispatches: 0`,
`attention_block_fused_dispatches: 0`, `flash_compiled: true`,
`kernels_disabled_requested: []` — the flash-ON leg.

`s128_flash_off_1.json` (same config, `JAMMI_KERNELS_DISABLE=
attention_block_flash`) reads `attention_block_flash_fused_dispatches: 0`,
`attention_block_flash_declined_dispatches: 840`,
`attention_block_fused_dispatches: 840`, `flash_compiled: true`,
`kernels_disabled_requested: ["attention_block_flash"]`,
`kernels_disabled_fired: ["attention_block_flash"]` — the flash-OFF
reference leg.

Neither file is modified from what `git show` produced (Bash's `>`
redirection was used to write them, no hand edits) — the "nothing ran"
(case 3) and "`flash_compiled=False` but a disable was requested" (case 4)
fixtures in `test_ab_merge.py` are NOT real recorded runs (no such run
exists — the first is a degenerate schema-regression shape, the second a
build-configuration contradiction), so those two cases construct their
`finetune_step` block by taking `s128_flash_off_1.json`'s dict and
overriding only the specific fields each case names, never inventing an
unrelated shape.
