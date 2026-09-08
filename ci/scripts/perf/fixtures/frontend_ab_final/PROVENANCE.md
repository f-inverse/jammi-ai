# Provenance — `frontend_ab_final/` fixtures

The FINAL, pre-registered #421 follow-on ("media front-end parallelization")
A/B for `perf/421-frontend` (tip `0a8562c4ff6205a69150580ff70a8cd098a9ebba`,
base `c1b0b0bad1f79a4ad6c298400e6ea19cc1ca633c`, box "NVIDIA A100 80GB PCIe,
570.172.08, 51a25f682a8d"), run on pod p421c via `ci/scripts/perf/
frontend_ab.sh` (`FRONTEND_AB_REPEATS=3`, `FRONTEND_AB_SERIAL_TAIL_RATIO=
0.0033`, `AB_RC=0`).

`raw/<tower>__<role>__<repeat>.json` is a hand-trimmed cut of the REAL,
complete `jammi-bench finetune-run` report for each of the 12 legs (2
towers × 2 roles × 3 repeats) — kept ONLY to the fields
`frontend_ab_artifact.py`'s own reader (`_read_raw_leg`) actually reads:
`host.logical_cpus`, `provenance.build_sha`, and
`tiers.finetune_run.{steps_measured, media_front_end_wall_s,
train_run_wall_s, rayon_pool_threads}` (the last key present only on `tip`
legs — a real pre-#421-follow-on `base` binary never emits it). Every
number is byte-identical to the untrimmed rehearsal raw-leg file (never a
hand-typed or rounded transcription) — mirrors `fixtures/
frontend_ab_rehearsal/`'s own "envelope-trimmed" convention and the reason
given there (`jammi-bench`'s own generic report shape carries ~90 other
`tiers.finetune_run` fields this reader has no business asserting on).
`raw/<...>.exit` is each leg's own recorded exit code (all `0` on this
run).

`report.json` is `frontend_ab_merge.py`'s own COMPLETE, UNMODIFIED merged
report for this run (small enough to commit whole — no trimming needed).

Consumed by the producer invocation recorded in the committed artifact's
own `producer.invocation` field:

```
python3 ci/scripts/perf/frontend_ab_artifact.py \
  --raw-dir ci/scripts/perf/fixtures/frontend_ab_final/raw \
  --report-json ci/scripts/perf/fixtures/frontend_ab_final/report.json \
  --identity ci/scripts/perf/frontend_ab_final_identity.json \
  --measured-serial-tail-s 0.00474603615 \
  --out crates/jammi-kernels/artifacts/cuda-runs/2026-09-08-frontend-0a8562c4-a100-pcie.json
```

`test_frontend_ab_artifact.py::RealFixtureRegressionTests` drives this
fixture through `frontend_ab_artifact.build_report` (the real reader) end
to end, against this checkout's own git history — never a hand-rolled
dict standing in for what a real report looks like — asserting the
re-derived HTSAT bar's own verdict is `UNRESOLVED` under BOTH the
driver-default and the run's own measured serial-tail ratio, matching the
committed artifact's own `verdict.unit_verdict`.
