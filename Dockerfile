# syntax=docker/dockerfile:1.7

# The `jammi-server` container images. Every image packages binaries it does
# not compile: a build passes `--build-context builder=<dir>` (or
# `builder-cuda=<dir>` for the CUDA image), where `<dir>/out` holds the
# `jammi-server` and `jammi` binaries one build of
# `ci/release-feature-manifest.json` produced — in CI, the ones `_server.yml`
# and `_cli.yml` built (`stage-server-binaries` lays the directory out); by
# hand, `ci/dev.sh cargo build --release ...` in the CI image, which links the
# manylinux_2_28 floor the images' runtimes promise. There is one definition
# of how a server binary is compiled, and it is not in this file.
#
# The runtime variant is selected by the global `RUNTIME_VARIANT` argument:
#   runtime-generic        (default) operator-supplied config + a mounted volume (CPU)
#   runtime-selfcontained  a baked config; boots with no config and no volume (CPU)
#   runtime-cuda           candle's CUDA backend on an NVIDIA runtime base
ARG RUNTIME_VARIANT=runtime-generic

# ---- runtime base ----
# Distroless `cc` ships glibc and libstdc++ (the binaries link the system C++
# runtime through tonic and tokio's syscall layer). Shared by the two CPU
# variants: both binaries and the turnkey `jammi-server` entrypoint. The
# generic variant boots zero-config (local SQLite catalog under
# `/var/lib/jammi`) and accepts an operator config via `--config`; the
# self-contained variant bakes a config so it boots on a runtime that provides
# neither a config nor a volume.
FROM gcr.io/distroless/cc-debian12 AS runtime-base

# The `builder` context's `out/`: the stripped `jammi-server` (the entrypoint)
# and the strict-client `jammi` CLI, whose admin verbs run against the server
# in the same image (`jammi --target grpc://… …`).
COPY --from=builder /out/jammi-server /usr/local/bin/jammi-server
COPY --from=builder /out/jammi /usr/local/bin/jammi

# Health side-channel on 8080, gRPC + Flight SQL on 8081.
EXPOSE 8080 8081

# Turnkey: `docker run <image>` runs `jammi-server serve`, which resolves its
# config through the same chain the binary uses everywhere else — `--config`,
# `JAMMI_CONFIG`, `./jammi.toml`, `/etc/jammi/jammi.toml`, the platform config
# dir, finally `JammiConfig::default()`. `WORKDIR /` (distroless sets none)
# makes `./jammi.toml` mean `/jammi.toml`: a bind mount there outranks a baked
# `/etc/jammi/jammi.toml` by this order. `docker run <image> --config
# /etc/jammi/jammi.toml` overrides the default `CMD` (`--config` is a
# top-level flag `serve` accepts, so no `serve` keyword is required).
WORKDIR /
ENTRYPOINT ["/usr/local/bin/jammi-server"]
CMD ["serve"]

# ---- runtime: generic (default) ----
FROM runtime-base AS runtime-generic

# Every runtime stage states its own boot command, so a reader never chases
# an inherited default across a `FROM`.
CMD ["serve"]

# Persistent state: catalog DB, model weights, indices, and the Hugging Face
# Hub cache (`HF_HOME`, below). Zero-config `jammi-server` writes its SQLite
# catalog here, so the directory must be writable by the nonroot user (uid
# 65532) even with no volume mounted: the `builder` context's empty
# `out/jammi-data/` is copied in with that ownership (distroless has no shell
# to `mkdir`). A named volume inherits it; a bind mount must be
# `chown 65532:65532`.
COPY --from=builder --chown=65532:65532 /out/jammi-data /var/lib/jammi
VOLUME ["/var/lib/jammi"]

# Point zero-config `jammi-server` at the declared volume rather than the
# user's XDG data dir (unwritable for uid 65532 here). An operator config's
# `artifact_dir` still wins through `--config`.
ENV JAMMI_ARTIFACT_DIR=/var/lib/jammi

# `HubSource`'s cache-root fallback (`[models] hub_cache_dir` > `HF_HOME` >
# the home directory) would otherwise reach the home-directory arm, and the
# nonroot user has no writable `HOME`. The same volume, so a model pulled
# once survives a restart.
ENV HF_HOME=/var/lib/jammi/hf

USER nonroot:nonroot

# ---- runtime: self-contained ----
# Boots with zero external config and no mounted volume: the baked config
# (`deploy/jammi.selfcontained.toml`) points `artifact_dir` under `/tmp`, the
# only path the nonroot user can write without a provisioned volume. Models
# are named per request: a Hub id is fetched on first use into `HF_HOME`, and
# a deployment that must serve without network builds `FROM` this stage,
# copies its checkpoint in, and names it `local:<path>`. No `VOLUME`: on a
# runtime that provides none, a declared volume is an anonymous mount the
# deploy cannot reach.
FROM runtime-base AS runtime-selfcontained

COPY deploy/jammi.selfcontained.toml /etc/jammi/jammi.toml

# Under `/tmp` with the rest of this stage's state, the one writable root.
ENV HF_HOME=/tmp/jammi/hf

USER nonroot:nonroot

# No `--config`: `serve`'s resolution chain finds the file baked above at
# `/etc/jammi/jammi.toml`; a bind-mounted `/jammi.toml` or `JAMMI_CONFIG`
# outranks it.
CMD ["serve"]

# ---- runtime: cuda ----
# `nvidia/cuda:*-runtime-ubi8` ships `libcudart` and the rest of the CUDA
# runtime libraries candle's cudarc backend dlopens, on a glibc-2.28 UBI8
# userland — the manylinux_2_28 / AlmaLinux 8 lineage the CUDA CI image built
# the binary against, so its glibc symbols resolve. The `-runtime-` image
# carries the shared libraries without the toolkit. GPU access at run time
# needs the NVIDIA Container Toolkit on the host (`docker run --gpus all`).
# `--platform=linux/amd64` is explicit: the base is amd64 only, so an arm
# builder fails loudly instead of emulating.
FROM --platform=linux/amd64 nvidia/cuda:12.6.3-runtime-ubi8 AS runtime-cuda

COPY --from=builder-cuda /out/jammi-server /usr/local/bin/jammi-server
COPY --from=builder-cuda /out/jammi /usr/local/bin/jammi

EXPOSE 8080 8081

# UBI8 has no `nonroot` user; the same uid (65532) the distroless variants
# use, so volume-ownership guidance is identical across images.
RUN groupadd --gid 65532 nonroot \
    && useradd --uid 65532 --gid 65532 --home-dir /home/nonroot --create-home nonroot \
    && mkdir -p /var/lib/jammi /var/lib/jammi/.nv-cache \
    && chown -R 65532:65532 /var/lib/jammi

VOLUME ["/var/lib/jammi"]
USER 65532:65532

ENV JAMMI_ARTIFACT_DIR=/var/lib/jammi

# `HOME` is unset at run time for uid 65532 despite `--create-home`, so
# `HubSource`'s home-directory fallback resolves nothing: the same volume.
ENV HF_HOME=/var/lib/jammi/hf

# The CUDA JIT (PTX→SASS) cache on the data volume: the image ships
# single-arch PTX at CUDA_COMPUTE_CAP=80, so on 8.6/8.9/9.0 the driver JITs it
# at first model load, and uid 65532's default cache dir (`$HOME/.nv`) is not
# writable here. The volume amortises the JIT across restarts.
ENV CUDA_CACHE_PATH=/var/lib/jammi/.nv-cache

WORKDIR /
ENTRYPOINT ["/usr/local/bin/jammi-server"]
CMD ["serve"]

# ---- final ----
FROM ${RUNTIME_VARIANT}
