# Mechanism contract — the CI CUDA image carries NCCL; a `jammi-ai --features cuda` clippy arm

Unit `ci/500-cuda-image-nccl` (plan 67, the U4a precondition; `docs/plans/67-distributed-training/UNITS.md`
§ U4a names it). The committed design pass for this unit lives beside this file in
`docs/rigor/ci_500-cuda-image-nccl.jsonl`.

## Why a mechanism changes

Plan 67's U4a makes `jammi-ai`'s `cuda` feature (`crates/jammi-ai/Cargo.toml:276`) reach
`candle-core/nccl`; cudarc's `dynamic-linking` build script then emits `-lnccl` at link time. The
CI CUDA image installs `cuda-toolkit-12-6` (`.docker/ci-cuda.Dockerfile:75`), which carries no NCCL.
Every lane that links a CUDA binary in that image would fail the moment U4a lands: the
`builder-cuda` stage (`Dockerfile:90`, driven by `server-image.yml`),
`.github/workflows/_pypi-server.yml:100` + `.github/workflows/_pypi-server.yml:126` (a `cargo build`
directly inside the container), `release-binaries.yml`'s CUDA leg
(`.github/workflows/release-binaries.yml:405`, `:460`), and the RunPod lanes, which run the same
image (`ci/scripts/runpod_lib.sh:98`).

## M1 — the image carries `libnccl` + `libnccl-devel` at a `+cuda12.6` build

- **Value:** the two package NVRs at `.docker/ci-cuda.Dockerfile:76-77`
  (`libnccl-2.23.4-1+cuda12.6`, `libnccl-devel-2.23.4-1+cuda12.6`), in the same `dnf install`
  call as the toolkit, under the same `install_weak_deps=False` discipline.
- **Readers:** every `cargo build`/`cargo test` that links a `cuda`-feature binary in this image
  (the lanes above), through the linker's `-lnccl`.
- **Invariant:** CUDA-minor agreement, local to this file — the NCCL build must be a `+cuda12.6`
  build because `cuda-toolkit-12-6` is what it is compiled against. NOT a soname invariant:
  NCCL's soname is `libnccl.so.2` for every 2.x, so no cross-file pin is implied.
  2.23.4-1+cuda12.6 is the same build the runtime base `nvidia/cuda:12.6.3-runtime-ubi8`
  (`Dockerfile:258`) ships, which is why that build was chosen among the three `+cuda12.6` builds
  the repo carries (2.22.3, 2.23.4, 2.24.3) — a preference, not an obligation binding the two files.
- **Failure modes, measured:** a bare `libnccl-devel` resolves to `2.31.2-1+cuda13.4` (a CUDA 13
  build against a 12.6 toolkit) — refused by pinning; dnf's glob forms (`libnccl-*+cuda12.6`)
  do not resolve at all — refused by spelling the NVR; the NVR absent from the repo → the image
  build fails (fail closed at `image-cuda.yml`, which rebuilds `:latest` on a push to `main`).
  `rpm -q --recommends` is empty for both packages, so `install_weak_deps=False` drops nothing.
  Floating the version (bare name + `exclude=` for the cuda13 builds) is rejected on
  reproducibility: this image is republished as a mutable `:latest`, so a float re-resolves
  silently on a later rebuild and the build that produced a given binary stops being recoverable
  from the file. `COPY --from=nvidia/cuda:12.6.3-runtime-ubi8` is rejected because that image
  ships the runtime package only — no `nccl.h`, no `libnccl.so` dev symlink.
- **Executor:** `image-cuda.yml` on the merge push. Nothing on THIS PR's merge path observes the
  rebuilt image's content: `flash-attn-compile` runs on the pre-rebuild `:latest`, and clippy never
  links (measured: a crate whose build script names a non-existent link library passes
  `cargo clippy --tests` and fails `cargo test --no-run`), so a green run here is evidence about
  the source tree, not about the published image. The ordering gate is PR-B's, which carries the
  preflight `rpm -q libnccl libnccl-devel && test -e /usr/include/nccl.h && test -e
  /usr/lib64/libnccl.so` in `flash-attn-compile`: it reds on any image that predates this rebuild,
  including the one this unit's own PR ran against, which is exactly the ordering it enforces.

## M2 — the nvcc lane type-checks `jammi-ai`'s cuda arm under `-D warnings`

- **Value:** the step at `.github/workflows/ci.yml:912-913`
  (`cargo clippy -p jammi-ai --features cuda --tests -- -D warnings`) in the `flash-attn-compile`
  job (`.github/workflows/ci.yml:720`). Measured cost on job 103688700438: 5m50s of a 19m20s job.
- **Readers:** the merge path itself; and `ci/scripts/check_lint_surface_closure.py`, which is what
  DETECTS the step's loss. `jammi-ai` declares no `required-features` target reaching `cuda`, so
  that gate's `cargo metadata` closure half credits this step for nothing; the row
  `jammi-ai cuda tests` in `ci/scripts/lint_surface_required_lanes.txt` is the reader that fails
  when no merge-path lane matches it. Deleting the step, dropping its `--tests` or its
  `-D warnings`, marking it `continue-on-error`, or moving it into a workflow that does not run
  on a PR touching `crates/jammi-ai/**` each red the gate, naming the row — the row is credited
  only by a lane whose hosting workflow carries a `pull_request`-to-main trigger admitting that
  crate's own sources, so a push-to-main-only host (`image-cuda.yml`) and a narrowly
  `paths:`-filtered one (`pypi-server-cuda.yml`) satisfy nothing.
- **Property:** no other merge-path lane lints this crate's cuda arm under `-D warnings`, and none
  compiles it under `cfg(test)`. The two CUDA release builds that compile its cuda `src/` are
  path-filtered (`.github/workflows/pypi-server-cuda.yml:22-27`,
  `.github/workflows/server-image.yml:33-37`; neither list contains `crates/jammi-ai/**`) and run
  `cargo build` with `deny_warnings` defaulting to false; the GPU pod lane compiles the cuda tests
  off the merge path. This step closes that gap; a red step reddens a required check on every PR,
  so its exit code is evidence, not a claim — PR #529's own run of the job exited 0.
- **Failure modes:** a lint in the cuda × `cfg(test)` surface → the required check fails on the
  PR that introduces it (fail closed by construction).

## M3 — the cu12 link-set check classifies every `DT_NEEDED` soname

- **Value:** `packaging/server-cu12/verify_link_set.py`'s `classify()` — a soname is acceptable
  only as `PLATFORM`, `DRIVER_PROVIDED` or `COVERED`, and anything else fails the wheel build —
  together with `check()`'s empty-extraction arm: zero `DT_NEEDED` entries is a FAIL naming the
  binary and the tool, never the same "OK" a fully classified binary prints, because a `readelf`
  dump the parser does not match would otherwise read as a clean link set.
- **Readers:** the cu12 wheel lane (`.github/workflows/_pypi-server.yml:144`) on the real binary;
  `ci/scripts/test_verify_link_set.py` hermetically, wired into `ci.yml`'s `Guard` matrix.
- **Property:** the predicate it replaces skipped every soname whose stem began with neither
  `libcu` nor `libnv`, so a binary hard-linking `libnccl.so.2` printed "link-set check OK" while
  the wheel bundled no NCCL. The classification is total, so U4a's NCCL cannot reach a wheel
  silently: it reds until a human classifies it.
- **Failure modes:** a base-image library the allowlist omits reds the lane. That is the intended
  direction — `PLATFORM` is the measured non-CUDA `DT_NEEDED` set of the shipped binary (the
  `server-cu12-binary` artifact of run 34717957779, which is where `libmvec.so.1` came from), and
  a soname outside it is a decision for a human, not a skip.

## Adjacent gap found, assigned to U4a

The cu12/CUDA-bundle class is six sites, all of which U4a must move together when NCCL becomes a
`DT_NEEDED` entry:

1. `.github/workflows/release-binaries.yml:503-504` — the CUDA tarball copies a hand-listed set of
   sonames out of `/usr/local/cuda-12.6/lib64`; `libnccl.so.2` lives in `/usr/lib64` and is on no
   list. U4a derives the set from the binary's `DT_NEEDED` entries with a post-copy loader check.
2. `.github/workflows/release-binaries.yml:377` — the same list restated in prose as what the
   tarball bundles.
3. `packaging/server-cu12/verify_link_set.py` — `COVERED` gains an NCCL entry when, and only when,
   the wheel actually bundles the component. Until then M3's classification reds the lane, which is
   the desired ordering.
4. `packaging/server-cu12/jammi_server/_entry.py:23` — `_CUDA_COMPONENTS`, the `LD_LIBRARY_PATH`
   prefix list the console script builds.
5. `packaging/server-cu12/pyproject.toml:40-43` — the `nvidia-*-cu12` dependency pins.
6. `packaging/server-cu12/README.md:12` — the same component list stated for the reader.

## Adjacent gap found, deferred

The merge-path corpus holds twelve `cargo clippy ... -D warnings` lanes. Deleting each in turn and
re-running both halves of `check_lint_surface_closure.py` (the sweep is mechanical: drop one lane
from `lanes_from_workflows`' output, re-run `find_gaps` and `find_missing_required_lanes`) splits
them five/seven — five are the only cover for some `required-features` target, so the closure half
REDs on their loss unaided; seven are derived from nothing. M2's registry rows three of the seven,
the cuda-feature lanes, whose loss is the most expensive to discover (a cuda lint regression
surfaces only on a GPU pod run, off the merge path).

The other four carry no row, and their loss reddens no gate that reads a clippy invocation at all
— measured: the candidate detector set is `grep -rl clippy ci/scripts` (this gate,
`check_execution_surface_reachability.py`, and pod-side artifacts that never read `ci.yml`'s merge
lanes), and deleting each of the four lines in turn in the worktree, re-running both halves of
`check_lint_surface_closure.py` plus `check_execution_surface_reachability.py` and
`check_ci_guard_wiring.py`, exits 0 in all sixteen runs. The four: `.github/workflows/ci.yml:43`
(`--workspace --all-targets`), `:90` (`-p jammi-server --tests --features test-hooks`), `:113`
(`-p jammi-db --features postgres,mysql --all-targets`), `:516` (`-p jammi-wire -p jammi-admin
-p jammi-client --all-targets`). Rowing them is its own unit: a row's `feature-set` and
`target-selection` fields are a claim about WHICH lint surface the lane uniquely protects, not a
transcription of its command line, and deriving that claim four times is not a change to make in
passing inside this one. `ci/scripts/lint_surface_required_lanes.txt`'s header states the same
gap where a reader adding a row will meet it.
