# Superseded

`2026-08-25-p6-b1-flash-timing-bb400c8-a100-sxm4.json` was found BLOCKING by the `10b1f3b`
adversarial audit (`.jammi/ledger/perf-s2-20260825.jsonl`, "B1 (FA2 op) audit @10b1f3b", finding 3):

- The timed bracket allocated `o`/`lse`/scratch/`d_qkv` INSIDE the loop and memset `dq_accum`
  TWICE per backward call (once in `BwdScratch::alloc`, once again — unconditionally — inside
  `flash_varlen_bwd_into`), so the reported number is wrapper-plus-double-memset cost, not kernel
  time — it was being compared elsewhere against `nsys` kernel-only numbers, an apples-to-oranges
  comparison.
- Only 5 warmup iterations; the mean-p50 gap traced to one max outlier over 25 samples.
- `box`/`gpu`/`driver` silently defaulted to `"unknown"` on failure; `tip_sha` came from an
  unchecked env var rather than `git rev-parse HEAD`.
- The non-deterministic backward leg was bimodal (min 0.26ms, p50 0.376ms) with no root cause
  found or stated.

Per this directory's own README ("a re-proof of the same tip on another box is a new file, never
an overwrite"), this file is kept as-is rather than edited. The fix round's replacement measurement
— separate KERNEL (zero-allocation, preallocated buffers) and WRAPPER (public-API, allocates every
call) brackets, `>= 20` warmup / `>= 200` iterations, min+median with a steady-state refusal,
run twice for reproducibility, real (non-"unknown") provenance, and the non-deterministic leg
DROPPED (unreachable from the only production entry point) rather than republished unexplained —
is `2026-08-25-p6-b1-flash-timing-<tip_sha7>-a100-sxm4.json`, produced by
`crates/jammi-kernels/tests/flash_decisive_timing.rs` on `perf/p6-fa2-op`.
