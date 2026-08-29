# GPU floor measurement artifact (unit 62)

Producer: `measure_gpu_floors_print_only` (crates/jammi-encoders/tests/it/batch_composition_invariance.rs),
invoked as `JAMMI_REQUIRE_CUDA=1 cargo test -p jammi-encoders --features cuda --test it
measure_gpu_floors -- --ignored --nocapture --test-threads=1`, tree `bcd274c18bdb` (measurement
code unchanged since `67ba2394`, whose values the derived constants cite), one archival run per
arch, full stdout lines extracted verbatim per pod:

- `gpu-floors-a100.txt` — NVIDIA A100-SXM4-80GB (sm80), pod cjjh6oaqehvpwi
- `gpu-floors-h100.txt` — NVIDIA H100 80GB HBM3 (sm90), pod gufh54wmqox1rw
- `gpu-floors-l40s.txt` — NVIDIA L40S (sm89), pod kccwbawx92pou1
- `gpu-floors-a40.txt`  — NVIDIA A40 (sm86), pod qlc5z76zh98v6c

These files are the producer citations for `EXACT_ARCH_COMPOSITION_FLOOR`,
`SM89_COMPOSITION_FLOOR`, `GPU_TRUTH_DRIFT_BOUND`, and the per-arch red-control
admissibility statements in that test file. 8 batch compositions x per-row ratios plus both
red controls per composition; the sm89 row-length per-composition line-set is the basis for
the composition-scoped admissibility statement (gating composition 0 = 6.881763611768685e-2,
clearing floor*5 = 2.1e-2 by 3.28x; compositions 2 and 5 fall below that threshold).
