# BASE_IMAGE has no default ON PURPOSE, the same fail-closed doctrine
# ci.Dockerfile's own BASE_IMAGE/RUST_VERSION ARGs are held to: the workflow
# passes it explicitly (image-cuda.yml's own `base_image_amd64` input,
# defaulting to `ghcr.io/f-inverse/jammi-ai-ci:latest`).
#
# The CUDA base is amd64-only: image-cuda.yml passes no `platforms`, so this
# FROM pins the platform explicitly and an arm64 caller fails loudly at the
# base-image resolution step instead of silently emulating.
ARG BASE_IMAGE
FROM --platform=linux/amd64 ${BASE_IMAGE}

# GCC 13: CUDA 12.6 supports GCC ≤ 13.2; manylinux_2_28 ships GCC 14.2.
# Install gcc-toolset-13 and put it on PATH so nvcc (which ignores CC/CXX
# and finds the host compiler via PATH) sees GCC 13.
#
# `libnccl` + `libnccl-devel`: `jammi-ai`'s `cuda` feature reaches
# `candle-core/nccl`, whose cudarc `dynamic-linking` build script emits
# `cargo:rustc-link-lib=dylib=nccl`. Every stage that LINKS a CUDA binary in
# this image therefore needs `libnccl.so` and `nccl.h` on disk at link time —
# `Dockerfile`'s `builder-cuda` stage (driving `server-image.yml`,
# `pypi-server-cuda.yml`/`_pypi-server.yml`, and `release-binaries.yml`'s CUDA
# leg). `cuda-toolkit-12-6` carries neither; the cuda-rhel8 repo added just
# above does, so both halves ride in the SAME `dnf install` call under the same
# `install_weak_deps=False` discipline.
#
# The version is a fully-qualified NVR rather than a bare package name, for two
# measured reasons. A bare `libnccl-devel` resolves to the newest build the
# cuda-rhel8 repo holds — 2.31.2-1+cuda13.4, a CUDA 13 NCCL against this
# image's CUDA 12.6 toolkit — and dnf's glob forms (`libnccl-*+cuda12.6`,
# `libnccl-devel-*cuda12.6`) do not resolve at all ("No match for argument"),
# so a CUDA-minor pin has to be spelled out in full. 2.23.4-1+cuda12.6 is the
# exact build `nvidia/cuda:12.6.3-runtime-ubi8` already ships, and that is the
# GPU runtime stage these binaries are copied INTO (`Dockerfile`'s
# `runtime-cuda`), so the soname the builder links against and the one the
# runtime resolves are the same library.
RUN dnf install -y gcc-toolset-13-gcc gcc-toolset-13-gcc-c++ \
                   'dnf-command(config-manager)' \
    && dnf config-manager --add-repo \
       https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && dnf install -y --setopt=install_weak_deps=False \
       cuda-toolkit-12-6 \
       libnccl-2.23.4-1+cuda12.6 \
       libnccl-devel-2.23.4-1+cuda12.6 \
    && dnf clean all \
    && rm -rf /var/cache/dnf

ENV CC=/opt/rh/gcc-toolset-13/root/usr/bin/gcc \
    CXX=/opt/rh/gcc-toolset-13/root/usr/bin/g++ \
    PATH="/usr/local/cuda-12.6/bin:/opt/rh/gcc-toolset-13/root/usr/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH}"

# candle compiles its CUDA kernels to single-architecture PTX at this compute
# capability. sm_80 (Ampere / A100) is the floor: PTX built for compute_80
# JIT-forward-runs on every supported datacenter GPU — A100 (8.0), A10/A6000
# (8.6), L40S (8.9), H100 (9.0) — and it is the lowest cap that keeps candle's
# bf16 kernels (gated on `__CUDA_ARCH__ >= 800`). Building for a higher cap
# (e.g. 86) produces PTX that fails to load on 8.0 hardware. Turing (7.5) is out
# of scope — bf16 is unsupported below sm_80.
ENV CUDA_COMPUTE_CAP=80
