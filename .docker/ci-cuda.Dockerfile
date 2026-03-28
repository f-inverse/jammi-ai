FROM ghcr.io/f-inverse/jammi-ai-ci:latest

# CUDA toolkit (dev libraries + nvcc, no GPU driver needed at build time).
# The base CI image is manylinux_2_28 (AlmaLinux 8) — use NVIDIA's RHEL 8 repo.
# CUDA 13.0 is the latest version supported by candle's cudarc 0.19.3.
RUN dnf install -y 'dnf-command(config-manager)' \
    && dnf config-manager --add-repo \
       https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && dnf install -y --setopt=install_weak_deps=False \
       cuda-toolkit-13-0 \
    && dnf clean all \
    && rm -rf /var/cache/dnf

# manylinux_2_28 ships GCC 14 but CUDA 13.0 may not support it.
# Install gcc-toolset-13 (GCC 13, supported by all CUDA 12.x/13.x) and
# set it as the default compiler for nvcc and Rust cc builds.
RUN dnf install -y gcc-toolset-13-gcc gcc-toolset-13-gcc-c++ \
    && dnf clean all \
    && rm -rf /var/cache/dnf

ENV PATH="/usr/local/cuda-13.0/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}" \
    CUDA_COMPUTE_CAP=86 \
    CC=/opt/rh/gcc-toolset-13/root/usr/bin/gcc \
    CXX=/opt/rh/gcc-toolset-13/root/usr/bin/g++
