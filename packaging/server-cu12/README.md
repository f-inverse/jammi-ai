# jammi-server-cu12

GPU build of the Jammi engine server, packaged as a pip-installable wheel that
ships the CUDA-enabled `jammi-server` binary behind a `jammi-server` console
script.

```console
pip install jammi-server-cu12
jammi-server --help
```

The CUDA runtime libraries (`libcudart`, `libcublas`, `libcublasLt`,
`libcurand`, `libnvrtc`) are pulled in as `nvidia-*-cu12` wheel dependencies; the
console script prepends their `lib/` directories to `LD_LIBRARY_PATH` before
exec'ing the binary, so no system CUDA install is required (only an NVIDIA driver
on the host). This package and `jammi-server` (CPU) both provide the
`jammi-server` command — install exactly one.

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
