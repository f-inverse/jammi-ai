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
# `libnccl` + `libnccl-devel`: the standing invariant is that this image
# carries an NCCL a CUDA link can find, because `cuda-toolkit-12-6` below
# carries none — neither `libnccl.so` nor `nccl.h`, and link time needs both.
# The path by which `jammi-ai`'s `cuda` feature reaches NCCL is
# `candle-core/nccl` (candle keeps `nccl` a feature of its own, so the reach is
# whether `crates/jammi-ai/Cargo.toml`'s `cuda` list names it); with it,
# cudarc's `dynamic-linking` build script emits
# `cargo:rustc-link-lib=dylib=nccl` and every surface that LINKS a CUDA binary
# out of this image resolves `-lnccl` here. The cuda-rhel8 repo added just
# above carries both halves, so they ride in the SAME `dnf install` call under
# the same `install_weak_deps=False` discipline as the toolkit.
#
# Those link surfaces, exactly — only ONE of the three is a docker build:
#   * `server-image.yml` builds through the top-level `Dockerfile`'s
#     `builder-cuda` stage with `docker buildx` (build-args at
#     `server-image.yml:747-749` on the `v*`-tag leg and at `:815-817` on the
#     build-only `pull_request` leg); that stage's own `FROM` is this image.
#   * `pypi-server-cuda.yml` -> `_pypi-server.yml` takes this image as the job
#     `container:` (`_pypi-server.yml:100`) and links DIRECTLY inside it with
#     `cargo build --release -p jammi-server` (`:126`) — no docker build at all.
#   * `release-binaries.yml`'s CUDA leg does the same: `container:` this image
#     (`:405`), `cargo build --release -p jammi-server` (`:460`).
# The RunPod GPU lanes run this same image as well
# (`ci/scripts/runpod_lib.sh:98`, `RP_IMAGE` defaulting to
# `ghcr.io/f-inverse/jammi-ai-ci-cuda:latest`), so the pod-side link surface is
# covered by this same rebuild.
#
# The constraint the version expresses is CUDA-MINOR AGREEMENT, and it is local
# to this file: the NCCL build installed here must be a `+cuda12.6` build,
# because `cuda-toolkit-12-6` on the line below is what it is compiled and
# linked against. It is not a soname constraint — NCCL's soname is
# `libnccl.so.2` for every 2.x release, so the runtime would resolve the
# library either way; what a CUDA-13 NCCL breaks is the toolkit pairing, not
# the name.
#
# Spelling the full NVR is forced, for two measured reasons. A bare
# `libnccl-devel` resolves to the newest build the cuda-rhel8 repo holds —
# 2.31.2-1+cuda13.4, a CUDA 13 NCCL against this image's CUDA 12.6 toolkit —
# and dnf's glob forms (`libnccl-*+cuda12.6`, `libnccl-devel-*cuda12.6`) do not
# resolve at all ("No match for argument"). The repo carries three `+cuda12.6`
# NCCL builds (2.22.3, 2.23.4, 2.24.3), so "the cuda12.6 one" is not a unique
# specification either; 2.23.4-1+cuda12.6 is the same build the runtime base
# `nvidia/cuda:12.6.3-runtime-ubi8` ships (`Dockerfile:258`), which makes it the
# natural choice of the three, not an obligation binding the two files.
#
# The alternative of floating the version (a bare name plus an `exclude=` for
# the cuda13 builds) is rejected on reproducibility, not on correctness: this
# image is rebuilt and republished as a mutable `:latest`, so a float would
# silently re-resolve to a different NCCL on some later rebuild, and the build
# that produced a given binary would no longer be recoverable from this file.
# Copying the library out of the runtime image (`COPY --from=nvidia/cuda:
# 12.6.3-runtime-ubi8`) is rejected too: that image ships the runtime package
# only — no `nccl.h` and no `libnccl.so` development symlink — and link time
# needs both.
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
