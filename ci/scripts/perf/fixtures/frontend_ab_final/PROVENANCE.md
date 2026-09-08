# Provenance — `frontend_ab_final/` fixtures

The FINAL, pre-registered #421 follow-on ("media front-end parallelization")
A/B for `perf/421-frontend` (tip `0a8562c4ff6205a69150580ff70a8cd098a9ebba`,
base `c1b0b0bad1f79a4ad6c298400e6ea19cc1ca633c`, box "NVIDIA A100 80GB PCIe,
570.172.08, 51a25f682a8d"), run on pod **p421c** via `ci/scripts/perf/
frontend_ab.sh` (`FRONTEND_AB_REPEATS=3`, `FRONTEND_AB_SERIAL_TAIL_RATIO=
0.0033`, `AB_RC=0`) — this is the FINAL close-out run, not the earlier
pod-p421b rehearsal `fixtures/frontend_ab_rehearsal/` names (that rehearsal
found `frontend_ab_merge.py`'s own nested-tier reader defect; it never
produced a committed close-out artifact).

## What is trimmed, what is kept, and why

`raw/<tower>__<role>__<repeat>.json` is a TRIMMED cut of the REAL, complete
`jammi-bench finetune-run` report for each of the 12 legs (2 towers x 2
roles x 3 repeats), produced by the committed `trim_legs.py` in this same
directory (never a hand-rolled dict, never a one-off ungoverned script).
Kept fields — exactly the ones `frontend_ab_artifact.py`'s own reader
(`_read_raw_leg`) actually reads:

- `host.logical_cpus`
- `provenance.build_sha`
- `tiers.finetune_run.steps_measured`
- `tiers.finetune_run.media_front_end_wall_s`
- `tiers.finetune_run.train_run_wall_s`
- `tiers.finetune_run.task` (cross-checked against the tower the filename
  claims: `htsat` -> `audio_embedding`, `clip-vision` -> `image_embedding`)
- `tiers.finetune_run.device_name` (cross-checked against `report.json`'s
  own `box` device prefix, `"NVIDIA A100 80GB PCIe"`)
- `tiers.finetune_run.rayon_pool_threads` — `tip` legs only; a real
  pre-#421-follow-on `base` binary never emits this key at all, which is
  itself asserted (a `base` leg carrying it is a refusal)

Dropped — the REST of a real `tiers.finetune_run` object (~90 other keys:
checkpoint identity hashes, LoRA hyperparameters, target-module lists,
per-dispatch-site fused/eager counters, `engine_version`, `host.
total_ram_mib`, etc.) — `jammi-bench`'s own generic report shape, not a
specific consumer's data shape, is what this driver's own bar arithmetic
is contracted against, and this reader has no business asserting on fields
it never reads. Every KEPT number is byte-identical to its own untrimmed
source (verified by `trim_legs.py --check` against the pod's own untrimmed
raw-leg directory — never a hand-typed or rounded transcription).

`raw/<...>.exit` is each leg's own recorded exit code (all `0` on this
run), copied verbatim (stripped of trailing whitespace, one trailing
newline added) by the same `trim_legs.py`.

`report.json` is `frontend_ab_merge.py`'s own COMPLETE, UNMODIFIED merged
report for this run (small enough to commit whole — no trimming needed).

`serial_tail.txt` is the job log's own verbatim serial-tail-phase output
(the `=== phase: serial tail (CUDA) ===` block) plus the two `build_sha`
lines and `PROV_RC=0` from the `=== phase: builds ===` block immediately
before it — five lines, copied byte-for-byte, never retyped. Read by
`frontend_ab_artifact.py --serial-tail`, which extracts ONLY the
`task=audio_embedding ... t_s=...` line's own `t_s` (the CLIP-vision line
is carried for completeness, matching `notes.recorded_deviations`'s own
"captured but not used by any bar" note; the two `build_sha` lines and
`PROV_RC=0` are context for a human reader, not machine-read by this
producer).

## Untrimmed sources — sha256 (never committed to this repo)

Computed from the pod-p421c job's own scratch directory
(`scratchpad/fe-final/raw/*.json`, this run's untrimmed
`jammi-bench finetune-run` reports) at the time this fixture was cut:

```
749a345e2d64317ab9bfc856875b29376117ffae02f9d1c8e6341895b840dd15  clip-vision__base__r1.json
f4da11bbd756517de933b1987dd7bbcde2494e55a6d49b670b43a886d275d08e  clip-vision__base__r2.json
143c9da20dfe41f4d4398b2625e5d5b3599303dfc5e4972bb9df70b03c5dc538  clip-vision__base__r3.json
10ae57a7559ca60d4a5cf133d2928a1d0b17204277ffce7f03356f7b56077758  clip-vision__tip__r1.json
2f67cd8505092db1dfaad5e3996149c13604f85839a1f649bece36c1d4f2fd65  clip-vision__tip__r2.json
c3d9c09ff2c2325aab1c2d56b36ac29cb8e54ec5b63890c25beb876f6eb227ea  clip-vision__tip__r3.json
9b8819512f0d4c00fa9c1e81bdcfefcaf1fb3191f586c6b5139758562ae63fa5  htsat__base__r1.json
5390030bdd1c18b44dc2eeb4239493cb44fb4338a54776c2734144178230ec2a  htsat__base__r2.json
f8c1fac0ea678471fa7c0af1236a82a1bb047ab6546459344b7aa4bd51eb3a14  htsat__base__r3.json
b6852410d930992361f5822e4edb165a14367ab8442fa3c6de302bc8ef83758b  htsat__tip__r1.json
422c395c62d2e94fe124ac2e32e4d27d38edffb4815c6b5369dee1c5249a722d  htsat__tip__r2.json
0837329901ee5a56f973bfd483125d1a94f0f14af10926ce6789f3e8a8bbc40e  htsat__tip__r3.json
```

These sources never enter this repo (only their trimmed cuts, above, and
these sha256s do) — regenerating or re-verifying this fixture requires the
pod's own scratch directory, which is why `trim_legs.py --check` is
documented as runnable only where the untrimmed sources still exist; CI
never has them, and never needs to (`test_frontend_ab_artifact.py::
RealFixtureRegressionTests` below instead covers the COMMITTED bytes end
to end, which is all CI ever sees).

## Consumed by

The producer invocation recorded in the committed artifact's own
`producer.invocation` field:

```
python3 ci/scripts/perf/frontend_ab_artifact.py \
  --raw-dir ci/scripts/perf/fixtures/frontend_ab_final/raw \
  --report-json ci/scripts/perf/fixtures/frontend_ab_final/report.json \
  --identity ci/scripts/perf/frontend_ab_final_identity.json \
  --serial-tail ci/scripts/perf/fixtures/frontend_ab_final/serial_tail.txt \
  --out crates/jammi-kernels/artifacts/cuda-runs/2026-09-08-frontend-0a8562c4-a100-pcie.json
```

`test_frontend_ab_artifact.py::RealFixtureRegressionTests` drives this
fixture through `frontend_ab_artifact.build_report` (the real reader) end
to end, against this checkout's own git history — never a hand-rolled
dict standing in for what a real report looks like — asserting the
re-derived HTSAT bar's own verdict is `UNRESOLVED` under BOTH the
driver-default and the run's own measured serial-tail ratio, matching the
committed artifact's own `verdict.unit_verdict`. CI's own
`frontend_ab_artifact suite` matrix entry (`ci.yml`) sets
`JAMMI_CI_FRONTEND_AB_FINAL_FIXTURE_EXPECTED=1` and `fetch_depth: "0"`, so
this class's own two `skipTest` escape hatches (fixture directory absent;
this checkout's history missing `base_sha`/`tip_sha`) turn into a hard RED
in CI rather than a silent skip — see that test file's own
`RealFixtureRegressionTests` docstring and `ci.yml`'s own matrix-entry
comment for the doctrine this mirrors.
