# Provenance — `frontend_ab_rehearsal/` fixtures

`htsat_tip_r1_envelope.json` is a hand-trimmed cut of the REAL, complete
`jammi-bench finetune-run` report `htsat__tip__r1.json` produced by the
real A/B rehearsal for issue #421's follow-on ("media front-end
parallelization") on pod p421b, run against `ci/scripts/perf/frontend_ab.sh`
— the same rehearsal that found this driver's own report-reading defect
(`frontend_ab_merge.py`'s own module doc): every one of the eight
`finetune-run` legs completed, but the driver's OLD inline-heredoc reader
read `steps_measured` / `media_front_end_wall_s` / `train_run_wall_s` off
the report's top level, when the real report nests them under
`tiers.finetune_run` (mirroring every other `jammi-bench` tier this repo
reads). Every leg therefore read back `FAIL "report missing
'steps_measured'"` although nothing had actually failed.

The full `htsat__tip__r1.json` this fixture was trimmed from carries ~90
other `tiers.finetune_run` fields (checkpoint identity hashes, target
modules, per-dispatch-site fused/eager counters, etc.) that this driver's
own reader never reads and has no business asserting on — `jammi-bench`'s
own generic report shape, not a specific consumer's data shape, is what
this driver's own bar arithmetic is contracted against. This fixture keeps
ONLY the fields `frontend_ab_merge.py::load_leg` actually reads (the
"envelope"): `host.logical_cpus` and `tiers.finetune_run.{steps_measured,
media_front_end_wall_s, train_run_wall_s, rayon_pool_threads}` — every
number here is byte-identical to the source report's own JSON literal
(`grep`-verified against the untrimmed rehearsal raw-leg file), never a
hand-typed or rounded transcription.

`test_frontend_ab_merge.py`'s `RealEnvelopeFixtureTests` drives this
fixture through `frontend_ab_merge.load_leg` (the real reader
`frontend_ab.sh` calls via `frontend_ab_merge.main`), asserting the
computed `front_per_step`/`train_per_step`/`rayon_pool_threads` match the
source numbers exactly — proving the reader's nested-tier fix actually
reads a real report shape, not merely a shape this repo's own DRY_RUN stub
was rewritten to fabricate.
