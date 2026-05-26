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

# ---- runtime ----
# Distroless `cc` ships glibc and libstdc++ (Rust binaries linked
# against the system C++ runtime via tonic + tokio's syscall layer
# expect both). Image size lands ~50MB with a stripped binary.
FROM gcr.io/distroless/cc-debian12

# Bring just the stripped binary across — none of the source tree,
# cargo registry, or toolchain follow into the final image.
COPY --from=builder /tmp/jammi-server /usr/local/bin/jammi-server

# Health side-channel on 8080, gRPC + Flight SQL on 8081.
EXPOSE 8080 8081

# Persistent state: catalog DB, model weights, indices.
# Distroless's nonroot user is uid 65532 — a docker named volume
# (rather than a bind mount) lets docker provision ownership
# automatically. Bind-mount deploys must `chown 65532:65532` the
# host directory.
VOLUME ["/var/lib/jammi"]
USER nonroot:nonroot

ENTRYPOINT ["/usr/local/bin/jammi-server"]
CMD ["--config", "/etc/jammi/jammi.toml"]
