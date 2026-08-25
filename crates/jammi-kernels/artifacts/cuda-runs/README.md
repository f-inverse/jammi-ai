# CUDA run artifacts

One JSON per proven branch tip: the machine-checked record that the crate's `--features cuda`
gates actually executed on hardware. No CI lane has a GPU, so a `cuda_device()`-gated test is not
coverage until a file here records it running — the file is the evidence, the commit message is a
pointer to it.

Each artifact records the exact `git_sha` of the checkout that ran, the device / driver / `nvcc`,
the gate outcomes (`cuda clippy → cuda_parity → per-crate tests with the feature on`), every parity
test by name with its status, and every bench leg's headline numbers with the per-op dispatch
counters (`fused > 0 && eager == 0` is the positive proof the fused path ran; a leg without it is
INVALID, not a datum). Produced by the pod job's `proof_artifact.py` from the captured cargo logs;
a run that parsed zero parity tests is written as `INVALID`, never as green.

Naming: `<date>-<unit>-<sha7>-<gpu>.json`. Append-only: a re-proof of the same tip on another box
is a new file, never an overwrite.
