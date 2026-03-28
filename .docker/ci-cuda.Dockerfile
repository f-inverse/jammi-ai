FROM quay.io/pypa/manylinux_2_28_x86_64

# CUDA toolkit (dev libraries + nvcc, no GPU driver needed at build time)
# AlmaLinux 8 is RHEL 8 compatible — use NVIDIA's RHEL 8 repo.
RUN dnf install -y 'dnf-command(config-manager)' \
    && dnf config-manager --add-repo \
       https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && dnf install -y --setopt=install_weak_deps=False \
       cuda-toolkit-12-6 \
    && dnf clean all \
    && rm -rf /var/cache/dnf

ENV PATH="/usr/local/cuda-12.6/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH}" \
    CUDA_COMPUTE_CAP=86

# protoc: prost-build (via substrait) invokes protoc at build time.
ARG PROTOC_VERSION=28.3
RUN curl -fsSL "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-x86_64.zip" \
        -o /tmp/protoc.zip \
    && unzip /tmp/protoc.zip -d /usr/local bin/protoc 'include/*' \
    && rm /tmp/protoc.zip

# mold linker for faster linking
ARG MOLD_VERSION=2.35.1
RUN curl -fsSL "https://github.com/rui314/mold/releases/download/v${MOLD_VERSION}/mold-${MOLD_VERSION}-x86_64-linux.tar.gz" \
    | tar -xz --strip-components=1 -C /usr/local

# Rust toolchain
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH="/usr/local/cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain 1.88.0 --profile minimal \
    && rustup component add rustfmt clippy

# sccache
ARG SCCACHE_VERSION=0.14.0
RUN curl -fsSL "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
    | tar -xz --strip-components=1 -C /usr/local/cargo/bin \
        "sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl/sccache"

# maturin
RUN cargo install maturin --locked \
    && rm -rf /usr/local/cargo/registry /usr/local/cargo/git
