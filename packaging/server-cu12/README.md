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
