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
    LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH}" \
    CUDA_COMPUTE_CAP=86
