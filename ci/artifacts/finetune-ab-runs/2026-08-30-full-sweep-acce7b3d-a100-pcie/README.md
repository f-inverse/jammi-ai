# `2026-08-30-full-sweep-acce7b3d-a100-pcie/` — a step sweep on an A100 80GB PCIe

One end-to-end sweep of the LoRA optimizer step at
`acce7b3d060d5f7fc7ff5f1f8f0b903a2fcbff71`: six shapes (`{b8 s128, b8 s512,
b16 s128} × {dropout 0, 0.05}`), 20 measured steps after 5 warmup, torch
2.13.0+cu126, `--lora-init jammi` on the PyTorch side. Per shape the merged
report kept one `s_per_step_p50`, one peak-VRAM figure and the dispatch
counters of each leg: the PyTorch step twice and the fused step twice in
`A, B, B, A` order, and the every-family-off reference step once where it
fit in memory (it ran out of memory at four of the six shapes).

Only per-leg medians are on record — no per-step series — so what is
reproducible from this directory is each shape's point ratio and reading,
never an interval.

These files are the committed oracle of the `train-step` ladder
(`jammi-bench ladder train-step`): `crates/jammi-bench/src/ladder/tests.rs`
reads each shape as one unit with `torch`, `reference` and `fused` legs, and
reaches every reading the merged report reached, where the ladder reaches it:

| shape | fused ÷ torch (medians) | repeat band | reading |
|---|---:|---:|---|
| `b16-s128-d0` | 0.974 | ×1.003 | PASS: outside the band, torch ÷ fused ≥ 0.9 |
| `b16-s128-d0p05` | 1.022 | ×1.008 | PASS |
| `b8-s128-d0` | 0.988 | ×1.087 | INDETERMINATE: inside the band |
| `b8-s128-d0p05` | 1.030 | ×1.106 | INDETERMINATE: inside the band |
| `b8-s512-d0` | 0.925 | ×1.011 | PASS |
| `b8-s512-d0p05` | 0.972 | ×1.011 | PASS |

INDETERMINATE is a cost inside the noise band the two repeats of each rung
measure against each other — the two repeats disagree by more than the
ratio is from 1; the report's own reading rested on the same two pairs. A
shape whose reference leg is missing is refused by name on both edges that
touch that rung, and the end-to-end pair is read directly at every shape.
The fused legs' counters prove the fused arm and the flash cascade; the
reference legs' counters prove the every-family-off arm.
