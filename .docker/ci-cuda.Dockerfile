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
# `libnccl` + `libnccl-devel`: plan 67's U4a
# (`docs/plans/67-distributed-training/UNITS.md` § U4a) adds `candle-core/nccl`
# to `jammi-ai`'s `cuda` feature, and this image carrying NCCL is that unit's
# stated precondition. At this commit the feature does NOT reach NCCL —
# `crates/jammi-ai/Cargo.toml:276`'s `cuda` list has no nccl entry, and
# candle-core 0.11 keeps `nccl` a separate feature — so nothing links against
# it yet. When that entry is present, cudarc's `dynamic-linking` build script
# emits `cargo:rustc-link-lib=dylib=nccl`, and every surface that LINKS a CUDA
# binary needs `libnccl.so` and `nccl.h` on disk at link time. `cuda-toolkit-12-6`
# carries neither; the cuda-rhel8 repo added just above does, so both halves
# ride in the SAME `dnf install` call under the same `install_weak_deps=False`
# discipline.
#
# Those link surfaces, exactly — only ONE of the three is a docker build:
#   * `server-image.yml` builds through the top-level `Dockerfile`'s
#     `builder-cuda` stage with `docker buildx` (build-args at `:747-749` on
#     the `v*`-tag leg and `:815-817` on the build-only `pull_request` leg);
#     that stage's own `FROM` is this image.
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
#
# That makes this pin one half of a TWIN. The other half is the runtime base
# `nvidia/cuda:12.6.3-runtime-ubi8` at `Dockerfile:258`, whose own
# `NV_LIBNCCL_PACKAGE` is the same 2.23.4-1+cuda12.6. The two move together: a
# CUDA-minor bump of THIS image moves the NCCL pin here in the same change, and
# the runtime base it is matched against with it. Neither side can be left to
# resolve on its own — the cuda-rhel8 repo carries more than one NCCL build for
# `+cuda12.6` (2.23.4 and 2.24.3), and a bare name resolves to a cuda13 build,
# as measured above.
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
