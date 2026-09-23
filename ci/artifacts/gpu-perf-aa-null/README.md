# gpu-perf-aa-null — five A/A runs of the `encode` serving path (2026-08-30)

Five runs on rented RunPod pods, both A100 device models, all of one `main`
tip (`6980b8301b1bd104fbed2804af14115f2c0f3f2f`) built twice from two
independent clones and served four times in the order `a1, b1, b2, a2` —
the same revision on both sides of an edge. Each file records, per leg, the
served embed lane's `p50_ms`, `p99_ms` and `rows_per_s`, its identity fields
and its provenance.

These files are the committed oracle of the `encode` ladder's **revision
edge** (`jammi-bench ladder encode --revision direct`):
`crates/jammi-bench/src/ladder/tests.rs` reads each run as the `direct` rung
against itself, `a1`/`a2` as the base build's two repeats and `b1`/`b2` as
the revised build's, and reproduces each run's ratio of pooled medians to
three decimals of the ratio the run was originally merged with (the mean of
its two adjacent-pair ratios). What the runs show, read that way:

| run | cost (b ÷ a) | repeat band | inside the band |
|---|---:|---:|---|
| `pcie-p1` | 1.049 | ×1.045 | no |
| `pcie-p2` | 0.945 | ×1.084 | yes |
| `pcie-p3` | 0.992 | ×1.060 | yes |
| `sxm4-r1` | 0.871 | ×1.087 | no |
| `sxm4-r2` | 0.881 | ×1.117 | no |

Three of five same-revision pairs fall outside the band a single build's
repeats measure: **two builds of one sha differ by more than one build's
repeats do**. That is why a revision edge carries its own A/A twin — a third
build of the base revision, `<rung>@rebuilt`, measured in the same session —
and judges the revised cost against the wider of the repeat band and the
A/A band, never against a band carried over from another session or another
pod. Only per-leg medians are on record here, so the ratios are reproducible
and their intervals are not.

`pcie-p1` and `pcie-p2` ran concurrently on one pod (their leg start times
interleave within a second and a half); every reading above is what the
file records, contention included.
