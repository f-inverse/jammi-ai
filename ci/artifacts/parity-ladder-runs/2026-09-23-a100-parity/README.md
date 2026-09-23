# `2026-09-23-a100-parity/` — the parity ladder on A100s, after the compute plane

One GPU session of every ladder the engine defines, run on rented A100 80 GB PCIe pods at commit
`168c71dc`, with the PyTorch twin rebuilt on the same box as the engine legs it is paired with.
Every directory below is one legs directory the comparator read; beside the legs sit the verdict it
produced (`ladder_verdict.json`) and its table (`ladder_table.txt`). Legs are the JSON files only;
the vectors, adapters and sampled pair tables they name stayed on the pods.

| directory | workload | rungs | verdict |
|---|---|---|---|
| `train-run/` | `train-run` | torch → resident, 12 seeds × 2 takes, lr = 0 controls at seeds 1 and 2, one seed per pod; `natural/` holds the torch twin's natural-width legs beside them | **GREEN** |
| `train-run-plane/` | `train-run` | resident → streamed → placed → shape-d, seed 1 × 2 takes | resident → streamed GREEN; the `placed` legs refused ([#624](https://github.com/f-inverse/jammi-ai/issues/624)) |
| `train-step/` | `train-step` | torch → fused, six shapes × 2 takes | **GREEN** |
| `encode-plane/` | `encode` | plan → plan-partitioned → placed, rows 16 / 1,024 / 16,384 × 2 takes | **GREEN**; the shape-d rung's serve was not placed ([#625](https://github.com/f-inverse/jammi-ai/issues/625)) |
| `propagate/` | `propagate` | torch → torch-geometric → plan → plan-partitioned → placed, 5,120 / 20,480 / 81,920 edges × 2 takes | torch → torch-geometric GREEN; the engine edges INVALID on stationarity ([#627](https://github.com/f-inverse/jammi-ai/issues/627)) |
| `graph-sample/` | `graph-sample` | torch → sampler against node2vec's law, three graphs × 2 takes; verdict and table only, the legs carry the per-cell counts and run to 10 MB each | INVALID: the law fit's cells are below Cochran's floor ([#626](https://github.com/f-inverse/jammi-ai/issues/626)) |
| `predictor-train-run/` | `predictor-train-run` | torch → in-process → placed → shape-d, seeds 1–3 × 2 takes | INVALID on stationarity and on the seed count ([#627](https://github.com/f-inverse/jammi-ai/issues/627)) |
| `structure/` | `structure` | torch → plan → plan-partitioned → placed, 2,048 / 8,192 / 32,768 nodes × 2 takes, warmup 12 / 32 iterations | the legs the session filed; the earlier session's verdict at warmup 3 / 8 was INVALID on stationarity ([#627](https://github.com/f-inverse/jammi-ai/issues/627)) |

## What the session established

**Learning.** Over twelve seeds the engine's held-out loss after the same four epochs is
non-inferior to PyTorch's, and equivalent within the derived margin: mean d = +0.0105 over the
twelve units with interval [+0.0064, +0.0150], against a margin of ±0.153 (half the effect the
reference itself learned, 0.309). Ten of twelve seeds sit a little above torch, two below; the sign
test does not reach either direction at α = 0.0064. Both learning-rate-zero controls held.

**Speed.** The engine's run is 0.606× PyTorch's by medians (interval [0.605, 0.607], outside the
0.8 % noise band), 0.625× by minima: jammi trains the same run in 61 % of the time, or 1.65× faster.
Steps-only, `train-step` reads 0.885× by medians at six shapes, and the fused kernels' gradients
agree with torch's on every real adapter pair.

**Space.** 0.516× the device memory and 0.805× the host memory of the same PyTorch run; on
`train-step`, 0.755× and 0.701×.

**The plane.** An embedding plan fanned over four partitions costs 0.95× the single plan
(indeterminate inside the noise band) and 1.61× on a Ballista executor; a training job through
the streaming loader costs 1.0007× the resident run.

Every number is read off the verdicts here, and `crates/jammi-bench/budgets.json` is derived from
the GREEN ones (`ci/scripts/perf/budgets_from_verdicts.py`).

## What the legs forced

- **The twin padded on a ladder the engine no longer has.** The train-run twin's `--width bucketed`
  leg padded its training batches up the power-of-two bucket ladder and its evaluation batches at
  natural width; the engine pads every batch, training and evaluation, up `ShapeLadder` (eight
  rungs per octave). On this checkpoint the engine's batches sit at 288–352 tokens where the old
  twin padded every one to 512, so the first campaign's torch legs were both a different
  computation (their token digests disagreed with the engine's on every leg) and a slower one
  than PyTorch need be. Every twin that pads now shares one mirror, `shape_ladder.py`; the
  corrected twin reproduces the engine's two digests exactly, and the twelve seeds here are the
  rerun.
- **Two memory instruments are not one column.** The train-step twin filed torch's allocator
  high-water mark as `peak_vram_bytes` against the engine's driver-pool window, and opened its own
  window after an untimed step the pool never gives back; the first table read 82× on the space
  axis. Both twins measure through one shared window now, opened where the engine opens its own.
- **The eager composition is not a control.** With every fused-kernel family off, the step holds
  45 GiB at 8×128 where the fused arm holds 3.7 and exceeds the device at 8×512 and 16×128:
  candle's autograd keeps every operator's output alive for the backward and materializes a
  gradient for every operand, and each LoRA site's eager epilogue widens to `f32`. The training
  ladders judge the product against PyTorch and nothing else.
- **A cold device is not stationary.** The first fused leg after any idle gap starts on a boosted
  clock and slows 3–5 % across the run, which the stationarity gate refuses; the producers soak the
  device with an untimed run of the first leg before the first timed one.
- **The epoch walls are one shape.** The parity test that runs both producers on one box read the
  twin's `epoch_walls` without the engine's `epoch` field; the twin carries it.
