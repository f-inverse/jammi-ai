# CUDA run artifacts

One JSON per proven branch tip: the machine-checked record that the crate's `--features cuda`
gates actually executed on hardware. No CI lane has a GPU, so a `cuda_device()`-gated test is not
coverage until a file here records it running — the file is the evidence, the commit message is a
pointer to it.

Each artifact records the exact `git_sha` of the checkout that ran, the device / driver / `nvcc`,
the gate outcomes (`cuda clippy → cuda_parity → per-crate tests with the feature on`), every parity
test by name with its status, and every bench leg's headline numbers with the per-op dispatch
counters (`fused > 0 && eager == 0` is the positive proof the fused path ran; a leg without it is
INVALID, not a datum). Produced by the pod job's `ci/scripts/perf/proof_artifact.py` from the
captured cargo logs; a run that parsed zero parity tests is written as `INVALID`, never as green.

Naming: `<date>-<unit>-<sha7>-<gpu>.json`. Append-only: a re-proof of the same tip on another box
is a new file, never an overwrite.

## Schema

`ci/scripts/check_cuda_run_artifacts.py` enforces this on every `*.json` under this directory
(recursing into any `*-raw-runs/` subdirectory too — a per-leg raw `jammi-bench` dump inherits
its parent artifact's `git_sha` and `producer`, it does not go schema-free just because it is a
leg rather than the top-level artifact). Required top-level fields:

- **`schema_version`** (int) — `1` today.
- **`git_sha`** (40 lowercase hex chars) — the exact commit the run measured, checked with
  `git merge-base --is-ancestor <git_sha> HEAD`. A green artifact whose sha is not an ancestor of
  the branch is evidence about the ORACLE, not the code (`docs/maintainer/cuda-kernel-guide.md`
  §4) — the gate fails it. **OR**, for a reviewed legacy artifact only, **`git_sha_unresolved`**
  (whatever short/malformed ref the pod session actually saw) paired with `producer.kind ==
  "none"` — never fabricate a 40-hex value that was not actually resolved.
- **`box`** (string) — the physical/pod identifier the run measured on.
- **`producer`** — `{path, kind, invocation, gating}`:
  - `kind`: `"cargo-test"` (a single named `#[test] fn`, invoked with `--exact <fn>` so the gate
    can statically confirm the fn exists, sits under `#[test]`, and carries the stated gating
    attribute) | `"script"` (a tracked producer script, e.g. this directory's own
    `proof_artifact.py`) | `"none"` (a reviewed legacy artifact predating this schema — the
    gate's own `LEGACY_NONE_ALLOWLIST` is closed; a NEW artifact may never default to this).
  - `gating`: `"#[ignore]"` | `"env:<VAR>"` | `"required-features"` | `"none"` — how the named
    test/script stays out of a plain `cargo test`/CI run.
- **`status`** (string).

Run the gate: `python3 ci/scripts/check_cuda_run_artifacts.py`. Self-test (RED cases for every
rule, on a throwaway fixture repo — never this checkout):
`python3 ci/scripts/check_cuda_run_artifacts.py --self-test`.
