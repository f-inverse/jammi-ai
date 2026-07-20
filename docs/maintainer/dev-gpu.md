# GPU development on RunPod

jammi's default `cargo test` is CPU-hermetic; the GPU path is exercised only by
the gated `live-gpu-tests` lane. You don't need a local GPU — rent one per task
on RunPod, and let it auto-terminate.

## One-time setup

- A RunPod API key (RunPod → Settings → API Keys). Store it locally:
  ```bash
  mkdir -p ~/.config/runpod && printf '%s' 'YOUR_KEY' > ~/.config/runpod/key && chmod 600 ~/.config/runpod/key
  ```
  (CI reads the same key from the `RUNPOD_API_KEY` GitHub Actions secret.)
- `ci/scripts/gpu-shell.sh` generates its own ephemeral SSH key per run — nothing
  to register.

## Interactive debugging — `gpu-shell`

Drop into a shell on a real GPU of the arch a ticket needs; the pod is
**terminated when you exit**:

```bash
ci/scripts/gpu-shell.sh a100     # sm_80 — the #277 floor (default)
ci/scripts/gpu-shell.sh l40s     # sm_89 (Ada) — fp8 work (#308)
ci/scripts/gpu-shell.sh h100     # sm_90 (Hopper)
ci/scripts/gpu-shell.sh a40      # sm_86 (Ampere workstation)
```

The pod boots the CUDA CI image (full toolchain). Inside, load the container's
build env (an SSH shell doesn't inherit Dockerfile `ENV`), then build/run:

```bash
while IFS= read -r -d '' e; do export "$e"; done < /proc/1/environ
git clone --depth 1 https://github.com/f-inverse/jammi-ai && cd jammi-ai
cargo test -p jammi-ai --features cuda,live-gpu-tests --test gpu_capability -- --nocapture --test-threads=1
```

To reproduce the **shipped runtime image** (e.g. the uid-65532 JIT-cache case in
#305) instead of the toolchain image:

```bash
RP_IMAGE=nvidia/cuda:12.6.3-runtime-ubi8 ci/scripts/gpu-shell.sh a100
```

## Automated proof — CI

The `gpu-prove` workflow (`_gpu-prove-gate.yml`) runs `grpc_embedding_gpu` +
`gpu_capability` on a real A100 via the same shared primitive
(`ci/scripts/runpod_lib.sh`), gating every CUDA release (build → prove →
promote). Trigger it per-PR with the `run-gpu` label, or it runs nightly.

## Notes

- **A100 capacity on RunPod is intermittent** — `gpu-shell` fails over across
  cloud tiers and PCIe/SXM variants; if all are exhausted it exits `75`. Retry.
- Every pod is terminated on exit (an `EXIT` trap in `runpod_lib.sh`). If a run
  is killed uncleanly, check for stragglers in the RunPod console.
