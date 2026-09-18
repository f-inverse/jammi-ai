# jammi-server-cu12

GPU build of the Jammi engine server, packaged as a pip-installable wheel that
ships the CUDA-enabled `jammi-server` binary behind a `jammi-server` console
script.

```console
pip install jammi-server-cu12
jammi-server --help
```

The CUDA libraries the binary links (`libcudart`, `libcublas`, `libcublasLt`,
`libcurand`, `libnvrtc`, and `libnccl`, the collective-communication library the
multi-GPU training path uses) are pulled in as `nvidia-*-cu12` wheel
dependencies; the console script prepends their `lib/` directories to
`LD_LIBRARY_PATH` before exec'ing the binary, so no system CUDA install is
required (only an NVIDIA driver on the host). This package and `jammi-server`
(CPU) both provide the `jammi-server` command — install exactly one.

## The release tarball

`jammi-server-cu12-<version>-x86_64-unknown-linux-gnu.tar.gz` (built by
`.github/workflows/release-binaries.yml`'s `server-cu12-build` job) ships the
same binary as this wheel, packaged as a self-contained directory with a
launcher instead of a console script: it bundles the CUDA runtime shared
libraries the binary links, derived from the binary's own transitive
`DT_NEEDED` closure plus a measured floor for `dlopen`-only dependencies
(`ci/scripts/bundle_cuda_libs.sh`), never a hand-kept name list — a
dependency a future feature adds is picked up automatically, and one that
cannot be satisfied fails the build by name rather than shipping silently
incomplete.

That self-sufficiency claim — the bundled `lib/` is EVERYTHING the loader
needs on a host with only an NVIDIA driver and no system CUDA install, never
relying on a copy the *build* host happens to also carry — is proven in the
release lane by two arms, not asserted by construction: (1) DETECTION runs
the real dynamic loader against the real staged binary and refuses any
bundle-able member resolved from outside the staged `lib/`; (2) THE JAIL
hides every host copy behind a real `chroot`, so a bundle-able member has
nothing else to resolve from at all. Both arms are required on every build;
see `bundle_cuda_libs.sh`'s own module doc for the full argument and the
measured defects each arm's shape closes. `libnvrtc-builtins` (the one
`dlopen`-only member of the floor) is staged but is not, and cannot be,
proven by either loader arm — NVRTC loads it at runtime rather than linking
it, so no loader trace ever names it.

## Requirements

- **NVIDIA GPU** of compute capability **8.0 or newer** (Ampere A100, A10/A6000,
  Ada L4/L40S, Hopper H100). The kernels are built at `compute_80` and the driver
  JIT-forwards them to 8.6 / 8.9 / 9.0. Turing (7.5, e.g. Tesla T4) is **not
  supported** by this build.
- **NVIDIA driver r560 or newer** (Linux: ≥ `560.28.03`). The kernels ship as PTX
  compiled with the CUDA 12.6 toolkit, which the deployment driver JIT-compiles at
  model load; a driver below the CUDA 12.6 line (for example `550.x`, which tops
  out at CUDA 12.4) cannot compile that PTX and the server exits at startup with a
  clear driver-too-old error rather than a raw `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`
  at first inference. `nvidia-smi` shows the installed driver and its max CUDA
  version.
