# GPU floor measurement artifact (unit 62)

Producer: `measure_gpu_floors_print_only` (crates/jammi-encoders/tests/it/batch_composition_invariance.rs),
invoked as `JAMMI_REQUIRE_CUDA=1 cargo test -p jammi-encoders --features cuda --test it
measure_gpu_floors -- --ignored --nocapture --test-threads=1`, tree `a4fad082` for the archival capture (self-identifying: each file's first line is a HEADER carrying the probed compute capability, driver-reported device name, and crate version; measured values byte-identical to the `67ba2394` runs the derived constants cite), one archival run per
arch, full stdout lines extracted verbatim per pod:

- `gpu-floors-a100.txt` — NVIDIA A100-SXM4-80GB (sm80), pod cjjh6oaqehvpwi
- `gpu-floors-h100.txt` — NVIDIA H100 80GB HBM3 (sm90), pod gufh54wmqox1rw
- `gpu-floors-l40s.txt` — NVIDIA L40S (sm89), pod kccwbawx92pou1
- `gpu-floors-a40.txt`  — NVIDIA A40 (sm86), pod qlc5z76zh98v6c
- `gpu-floors-rtx6000ada.txt` — NVIDIA RTX 6000 Ada Generation (sm89), measured 2026-09-28 at `37a1b59c`
  after the nightly prove's sm_89 leg first landed on this card: `alone_vs_batch` max `4.569318750356236e-3`
  (rows of length <= 15 exact, longer rows not), window-radius control `1.23e-3`..`1.71e-3` BELOW that
  noise, row-length control's weakest composition `5.21e-3` below floor*5 -- neither red control has
  power at that noise. The diverging operation is cuBLAS's split-K GEMM: on this card and on an RTX
  4090 a `[m, 32] x [32, 64]` product splits K sixteen ways from `m = 17` with its partial sums
  reduced in `bf16` (`REDUCTION_SCHEME_OUTPUT_TYPE`), and not at all at `m = 528`, so a lone row
  longer than 16 and the same row in the batch take different kernels. An L40S and an L4 never split
  those shapes (cuBLASLt level-5 logs, 2026-09-28).
- `gpu-floors-rtx4090.txt` — NVIDIA GeForce RTX 4090 (sm89), measured 2026-09-28 at `9c699873`, the
  tree whose CUDA devices open through `jammi_kernels::device::open_cuda`
  (`CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`): `alone_vs_batch` `0e0` on all 8 compositions,
  bit-identical to the L40S; row-length control's weakest composition `4.69e-3`, window-radius
  control's `2.59e-4`, both above floor*5 = `5e-5`. The RTX 6000 Ada under the same math mode
  (scratch run on `37a1b59c`) is exact on the gating composition and on compositions 0-4.

These files are the producer citations for `EXACT_ARCH_COMPOSITION_FLOOR`,
`SM89_COMPOSITION_FLOOR`, `GPU_TRUTH_DRIFT_BOUND`, and the per-arch red-control
admissibility statements in that test file. 8 batch compositions x per-row ratios plus both
red controls per composition; the sm89 row-length per-composition line-set is the basis for
the composition-scoped admissibility statement (gating composition 0 = 6.881763611768685e-2,
clearing floor*5 = 2.1e-2 by 3.28x; compositions 2 (6.9996589149257556e-3), 5 (5.418507501917013e-3), and 7 (1.0134661986953957e-2) fall below that threshold).
