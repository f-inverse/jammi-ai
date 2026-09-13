# Mechanism contract — the CI CUDA image carries NCCL; a `jammi-ai --features cuda` clippy arm

Unit `ci/500-cuda-image-nccl` (plan 67, the U4a precondition; `docs/plans/67-distributed-training/UNITS.md`
§ U4a names it). The committed design pass for this unit lives beside this file in
`docs/rigor/ci_500-cuda-image-nccl.jsonl`.

## Why a mechanism changes

Plan 67's U4a makes `jammi-ai`'s `cuda` feature (`crates/jammi-ai/Cargo.toml:276`) reach
`candle-core/nccl`; cudarc's `dynamic-linking` build script then emits `-lnccl` at link time. The
CI CUDA image installs `cuda-toolkit-12-6` (`.docker/ci-cuda.Dockerfile:68`), which carries no NCCL.
Every lane that links a CUDA binary in that image would fail the moment U4a lands: the
`builder-cuda` stage (`Dockerfile:90`, driven by `server-image.yml`), `_pypi-server.yml:100` +
`:126` (a `cargo build` directly inside the container), `release-binaries.yml`'s CUDA leg, and the
RunPod lanes, which run the same image (`ci/scripts/runpod_lib.sh:98`).

## M1 — the image carries `libnccl` + `libnccl-devel` at the runtime base's exact build

- **Value:** the two package NVRs at `.docker/ci-cuda.Dockerfile:69-70`
  (`libnccl-2.23.4-1+cuda12.6`, `libnccl-devel-2.23.4-1+cuda12.6`), in the same `dnf install`
  call as the toolkit, under the same `install_weak_deps=False` discipline.
- **Readers:** every `cargo build`/`cargo test` that links a `cuda`-feature binary in this image
  (the lanes above), through the linker's `-lnccl`; the twin is the runtime base
  `nvidia/cuda:12.6.3-runtime-ubi8` (`Dockerfile:258`), whose `NV_LIBNCCL_PACKAGE` is the same NVR,
  so the builder links against the soname the runtime resolves.
- **Failure modes, measured:** a bare `libnccl-devel` resolves to `2.31.2-1+cuda13.4` (a CUDA 13
  build against a 12.6 toolkit) — refused by pinning; dnf's glob forms (`libnccl-*+cuda12.6`)
  do not resolve at all — refused by spelling the NVR; the NVR absent from the repo → the image
  build fails (fail closed at `image-cuda.yml`, which rebuilds `:latest` on a push to `main`).
  `rpm -q --recommends` is empty for both packages, so `install_weak_deps=False` drops nothing.
- **Executor:** `image-cuda.yml` on the merge push; the PR's own `flash-attn-compile` job runs on
  the pre-rebuild image, which is sufficient because clippy never links (measured: a crate whose
  build script names a non-existent link library passes `cargo clippy --tests` and fails
  `cargo test --no-run`).

## M2 — the nvcc lane type-checks `jammi-ai`'s cuda arm under `-D warnings`

- **Value:** the step at `.github/workflows/ci.yml:894-895`
  (`cargo clippy -p jammi-ai --features cuda --tests -- -D warnings`) in the `flash-attn-compile`
  job (`.github/workflows/ci.yml:720`).
- **Readers:** the merge path itself; `ci/scripts/check_lint_surface_closure.py` parses the step
  into a lane (it credits coverage from `cargo metadata`, keeping no registry of step names).
- **Property:** no merge-path lane lints this crate's cuda arm under `-D warnings` and none
  compiles it under `cfg(test)` (the CUDA release builds run `cargo build` with `deny_warnings`
  defaulting to false; the GPU pod lane compiles the cuda tests off the merge path). This step
  closes that gap; a red step reddens a required check on every PR, so its exit code is evidence,
  not a claim — PR #529's own run of the job exited 0.
- **Failure modes:** a lint in the cuda × `cfg(test)` surface → the required check fails on the
  PR that introduces it (fail closed by construction).

## Adjacent gap found, assigned to U4a

`release-binaries.yml:504` bundles a hand-listed set of sonames from the toolkit directory; U4a's
`DT_NEEDED libnccl.so.2` lives in `/usr/lib64` and is on no list. U4a derives the set from the
binary's `DT_NEEDED` entries with a post-copy loader check.
