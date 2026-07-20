FROM ghcr.io/f-inverse/jammi-ai-ci:latest

# GCC 13: CUDA 12.6 supports GCC ≤ 13.2; manylinux_2_28 ships GCC 14.2.
# Install gcc-toolset-13 and put it on PATH so nvcc (which ignores CC/CXX
# and finds the host compiler via PATH) sees GCC 13.
RUN dnf install -y gcc-toolset-13-gcc gcc-toolset-13-gcc-c++ \
                   'dnf-command(config-manager)' \
    && dnf config-manager --add-repo \
       https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && dnf install -y --setopt=install_weak_deps=False \
       cuda-toolkit-12-6 \
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
