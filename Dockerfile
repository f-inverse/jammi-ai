# syntax=docker/dockerfile:1.7

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
        --features jetstream-broker \
    && cp target/release/jammi-server /tmp/jammi-server \
    && strip /tmp/jammi-server

# ---- runtime base ----
# Distroless `cc` ships glibc and libstdc++ (Rust binaries linked
# against the system C++ runtime via tonic + tokio's syscall layer
# expect both). Image size lands ~50MB with a stripped binary.
#
# This base is shared by both image variants. The generic variant
# (`runtime-generic`) expects an operator-supplied config at
# `/etc/jammi/jammi.toml` and a mounted volume at `/var/lib/jammi`.
# The self-contained variant (`runtime-selfcontained`) bakes both in
# so it boots standalone on a runtime that provides neither (e.g.
# Cloudflare Containers).
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
# without a provisioned volume) and the baked `tiny_clap` encoder
# fixture lets `EncodeAudioQuery` / `GenerateAudioEmbeddings` run
# offline. Clients pass the encoder per request as
# `model_id = "local:/opt/jammi/models/tiny_clap"`.
#
# No `VOLUME` here — declaring one on a runtime that provides no
# volume just yields an anonymous mount the deploy can't reach.
FROM runtime-base AS runtime-selfcontained

COPY deploy/jammi.selfcontained.toml /etc/jammi/jammi.toml
COPY cookbook/fixtures/tiny_clap /opt/jammi/models/tiny_clap

USER nonroot:nonroot

CMD ["--config", "/etc/jammi/jammi.toml"]

# ---- final ----
# Select the variant with `--build-arg RUNTIME_VARIANT=...`:
#   runtime-generic        (default) — operator-supplied config + volume
#   runtime-selfcontained  — standalone, baked config + encoder
ARG RUNTIME_VARIANT=runtime-generic
FROM ${RUNTIME_VARIANT}
