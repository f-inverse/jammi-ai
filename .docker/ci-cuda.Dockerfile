FROM ghcr.io/f-inverse/jammi-ai-ci:latest

# CUDA toolkit (dev libraries + nvcc, no GPU driver needed at build time).
# The base CI image is manylinux_2_28 (AlmaLinux 8) — use NVIDIA's RHEL 8 repo.
# CUDA 13.2 supports the GCC 14 shipped in manylinux_2_28.
RUN dnf install -y 'dnf-command(config-manager)' \
    && dnf config-manager --add-repo \
       https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && dnf install -y --setopt=install_weak_deps=False \
       cuda-toolkit-13-2 \
    && dnf clean all \
    && rm -rf /var/cache/dnf

ENV PATH="/usr/local/cuda-13.2/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH}" \
    CUDA_COMPUTE_CAP=86
