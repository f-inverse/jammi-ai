# syntax=docker/dockerfile:1.7

# Runtime variant selector for the final image. This MUST be a GLOBAL arg (declared before the
# first FROM) so it is in scope for the `FROM ${RUNTIME_VARIANT}` selector at the bottom:
#   runtime-generic        (default) — operator-supplied config + volume (CPU)
#   runtime-selfcontained  — standalone, baked config + encoder (e.g. Cloudflare Containers) (CPU)
#   runtime-cuda           — CUDA build: GPU-accelerated inference, NVIDIA runtime base
#
# CUDA lives only on the server image (M2 §1, §5d). The CPU variants build on the generic
# CI base (`jammi-ai-ci`) and a distroless runtime; the CUDA variant builds on the CUDA CI
# base (`jammi-ai-ci-cuda`, which carries the CUDA 12.6 toolkit + CUDA_COMPUTE_CAP=86) and a
# CUDA runtime base that ships `libcudart` for candle's cudarc backend. Each runtime stage
# copies from the builder it needs, so the single `RUNTIME_VARIANT` selector still resolves
# the whole image — no Dockerfile fork.
ARG RUNTIME_VARIANT=runtime-generic

# ---- builder ----
# The CI base image carries the full Rust toolchain (rustc 1.88.0,
# protoc, mold, sccache). Pinning to `:latest` is intentional —
# the CI image is rebuilt on toolchain bumps and the OSS server
# inherits that update lockstep with the workspace.
FROM ghcr.io/f-inverse/jammi-ai-ci:latest AS builder

WORKDIR /workspace
COPY . .

# BuildKit cache mounts keep the cargo registry and target dir
# warm between image builds — a cold first build is ~30 minutes;
# a warm rebuild is ~3 minutes. The `target/` mount is sharded by
# the workspace name so concurrent builds don't fight over the
# same lock file.
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/workspace/target,sharing=locked \
    cargo build --release \
        --package jammi-server \
        --bin jammi-server \
        --features jetstream-broker,storage-cloud \
    && cp target/release/jammi-server /tmp/jammi-server \
    && strip /tmp/jammi-server

# ---- builder: cuda ----
# The CUDA CI base extends `jammi-ai-ci` with the CUDA 12.6 toolkit (nvcc), GCC 13
# (CUDA 12.6 supports GCC ≤ 13.2), and `CUDA_COMPUTE_CAP=86` — the same image the
# (now-retired) CUDA wheel lane built against. `candle-core/cuda` reads CUDA_COMPUTE_CAP
# at build time to target the GPU architecture; CC/CXX/PATH for nvcc are baked into the base.
FROM ghcr.io/f-inverse/jammi-ai-ci-cuda:latest AS builder-cuda

WORKDIR /workspace
COPY . .

# Same cache-mount strategy as the CPU builder. The only delta is `--features cuda`,
# which pulls in candle's CUDA backend (compiled for compute capability 86).
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/workspace/target,sharing=locked \
    cargo build --release \
        --package jammi-server \
        --bin jammi-server \
        --features cuda,jetstream-broker,storage-cloud \
    && cp target/release/jammi-server /tmp/jammi-server \
    && strip /tmp/jammi-server

# ---- runtime base ----
# Distroless `cc` ships glibc and libstdc++ (Rust binaries linked
# against the system C++ runtime via tonic + tokio's syscall layer
# expect both). Image size lands ~50MB with a stripped binary.
#
# This base is shared by the two CPU variants. The generic variant
# (`runtime-generic`) expects an operator-supplied config at
# `/etc/jammi/jammi.toml` and a mounted volume at `/var/lib/jammi`.
# The self-contained variant (`runtime-selfcontained`) bakes both in
# so it boots standalone on a runtime that provides neither (e.g.
# Cloudflare Containers). The CUDA variant (`runtime-cuda`) has its
# own NVIDIA runtime base below — distroless ships no CUDA libraries.
FROM gcr.io/distroless/cc-debian12 AS runtime-base

# Bring just the stripped binary across — none of the source tree,
# cargo registry, or toolchain follow into the final image.
COPY --from=builder /tmp/jammi-server /usr/local/bin/jammi-server

# Health side-channel on 8080, gRPC + Flight SQL on 8081.
EXPOSE 8080 8081

ENTRYPOINT ["/usr/local/bin/jammi-server"]

# ---- runtime: generic (default) ----
# The image users `docker pull` and run with their own config +
# volume. Behaviour is identical to before this variant existed.
FROM runtime-base AS runtime-generic

# Persistent state: catalog DB, model weights, indices.
# Distroless's nonroot user is uid 65532 — a docker named volume
# (rather than a bind mount) lets docker provision ownership
# automatically. Bind-mount deploys must `chown 65532:65532` the
# host directory.
VOLUME ["/var/lib/jammi"]
USER nonroot:nonroot

CMD ["--config", "/etc/jammi/jammi.toml"]

# ---- runtime: self-contained ----
# Boots with zero external config and no mounted volume: the baked
# config (`deploy/jammi.selfcontained.toml`) points `artifact_dir`
# under `/tmp` (the only path the distroless nonroot user can write
# without a provisioned volume) and the baked `htsat_clap_tiny`
# encoder fixture lets `EncodeAudioQuery` / `GenerateAudioEmbeddings`
# run offline. Clients pass the encoder per request as
# `model_id = "local:/opt/jammi/models/htsat_clap_tiny"`.
#
# No `VOLUME` here — declaring one on a runtime that provides no
# volume just yields an anonymous mount the deploy can't reach.
FROM runtime-base AS runtime-selfcontained

COPY deploy/jammi.selfcontained.toml /etc/jammi/jammi.toml
COPY cookbook/fixtures/htsat_clap_tiny /opt/jammi/models/htsat_clap_tiny

USER nonroot:nonroot

CMD ["--config", "/etc/jammi/jammi.toml"]

# ---- runtime: cuda ----
# GPU runtime base. `nvidia/cuda:*-runtime-ubi8` ships `libcudart` (and the rest of the
# CUDA runtime libraries candle's cudarc backend dlopen's) on a glibc-2.28 UBI8 userland —
# matching the manylinux_2_28 / AlmaLinux 8 lineage the CUDA CI base built the binary
# against, so the binary's glibc symbols resolve. The `-runtime-` (not `-devel-`) image
# carries the shared libraries without the toolkit, keeping the image lean.
#
# GPU access at run time requires the NVIDIA Container Toolkit on the host
# (`docker run --gpus all …`); set `gpu.device = 0` in jammi.toml (or `JAMMI_GPU__DEVICE=0`).
# This path is GPU-only at runtime and is NOT exercised in CI (no GPU on CI runners) — the
# Dockerfile compiling is the CI gate; GPU inference is verified out-of-band.
FROM nvidia/cuda:12.6.3-runtime-ubi8 AS runtime-cuda

# Bring just the stripped CUDA-enabled binary across from the CUDA builder.
COPY --from=builder-cuda /tmp/jammi-server /usr/local/bin/jammi-server

# Health side-channel on 8080, gRPC + Flight SQL on 8081.
EXPOSE 8080 8081

# Persistent state: catalog DB, model weights, indices.
VOLUME ["/var/lib/jammi"]

# UBI8 has no pre-provisioned `nonroot` user; create one with the same uid (65532) the
# distroless variants use so volume-ownership guidance stays identical across images.
RUN groupadd --gid 65532 nonroot \
    && useradd --uid 65532 --gid 65532 --home-dir /home/nonroot --create-home nonroot
USER 65532:65532

ENTRYPOINT ["/usr/local/bin/jammi-server"]
CMD ["--config", "/etc/jammi/jammi.toml"]

# ---- final ----
# Resolve the variant chosen by the global `RUNTIME_VARIANT` arg at the top of this file
# (`--build-arg RUNTIME_VARIANT=runtime-selfcontained` to bake config + encoder;
#  `--build-arg RUNTIME_VARIANT=runtime-cuda` for the GPU server image).
FROM ${RUNTIME_VARIANT}
