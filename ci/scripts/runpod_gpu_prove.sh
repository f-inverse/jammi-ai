#!/usr/bin/env bash
# GPU prove-lane: build jammi from source on a real GPU and run the gated GPU
# suites — the served client/server proof (grpc_embedding_gpu) and the
# engine-core correctness suite (gpu_capability). Doubles as the #277 regression
# gate: builds candle's kernels at THIS LEG'S NATIVE CUDA_COMPUTE_CAP (see
# NATIVE_COMPUTE_CAP below — overrides the image's baked cap, issue #434) and
# runs them on a real device. Shared deploy/run/teardown lives in runpod_lib.sh.
#
# GPU_PROVE_ARCH selects WHICH shipped CUDA arch (crates/jammi-kernels/build.rs
# GENCODE_ARCHES: sm_80/sm_86/sm_89/sm_90 today) this lane proves; default is
# today's A100 (sm_80) behavior. The sm_XX -> `rp_deploy_arch` key mapping
# below is the ONE place that translation lives; `rp_deploy_arch` itself
# (runpod_lib.sh) is the ONE place the arch key -> actual RunPod GPU-type-id
# candidate list lives — this script never hand-types a GPU-type-id string.
#
# Exit 0 = suites passed; 75 = no capacity for the requested arch (neutral skip).
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This lane's own pods never need 8h — the remote job is capped by RP_TIMEOUT
# (50m by default), so a tighter deadline bounds the damage of an orphan.
RP_TTL_HOURS="${RP_TTL_HOURS:-3}"
# shellcheck source=ci/scripts/runpod_lib.sh
source "$DIR/runpod_lib.sh"

GIT_REPO="${GIT_REPO:-https://github.com/${GITHUB_REPOSITORY:-f-inverse/jammi-ai}.git}"
GIT_REF="${GIT_REF:-${GITHUB_SHA:-main}}"

# sm_XX (the GENCODE_ARCHES / check_gpu_parity_matrix.py silicon-axis naming)
# -> the `rp_deploy_arch` candidate-list key (runpod_lib.sh), and -> the bare
# numeric NATIVE_COMPUTE_CAP this leg's device actually is. NATIVE_COMPUTE_CAP
# overrides the CI image's baked CUDA_COMPUTE_CAP (.docker/ci-cuda.Dockerfile)
# in the remote build env below: candle-kernels 0.11 builds the quantized
# fast-path kernels as single-arch SASS (no PTX) from that env var, so every
# leg building at the image's one baked cap produces kernels that silently
# cannot launch on the other legs' devices (issue #434). Comment on each line
# names the SASS target the leg proves.
GPU_PROVE_ARCH="${GPU_PROVE_ARCH:-sm_80}"
case "$GPU_PROVE_ARCH" in
  sm_80) RP_DEPLOY_ARCH=a100 NATIVE_COMPUTE_CAP=80 ;; # Ampere floor — proves sm_80, #277.
  sm_86) RP_DEPLOY_ARCH=a40  NATIVE_COMPUTE_CAP=86 ;; # Ampere workstation class — proves sm_86.
  sm_89) RP_DEPLOY_ARCH=l4_l40s NATIVE_COMPUTE_CAP=89 ;; # Ada — proves sm_89, fp8 #308. L4 first (canonical commodity inference card, ~half L40S rental); L40S is a capacity-only fallback — same sm_89 SASS, identical correctness proof.
  sm_90) RP_DEPLOY_ARCH=h100 NATIVE_COMPUTE_CAP=90 ;; # Hopper — proves sm_90.
  *)
    echo "::error::unknown GPU_PROVE_ARCH '${GPU_PROVE_ARCH}' (want: sm_80|sm_86|sm_89|sm_90)"
    exit 2
    ;;
esac

# Sweep before renting anything. This workflow sets cancel-in-progress, so a
# superseded run is SIGKILLed and never runs its EXIT trap; the pod it had just
# rented is orphaned. Running the sweep here bounds any such orphan to the gap
# until the next prove run rather than "until the account empties" — which is
# what happened on 2026-07-24.
rp_sweep

rp_init
echo "=== provisioning a live ${RP_DEPLOY_ARCH} (${GPU_PROVE_ARCH}) ==="
rp_deploy_arch "$RP_DEPLOY_ARCH" || exit $?

echo "=== running GPU prove suites on ${RP_HOST}:${RP_PORT} ==="
rp_run_remote <<REMOTE
export CARGO_TERM_COLOR=never
export CARGO_BUILD_RUSTC_WRAPPER=  # wrapper-off (ledger row 17: no cross-target-dir reuse, ~+33% wall on this image)
# Override the image's baked CUDA_COMPUTE_CAP with this leg's NATIVE arch.
# candle-kernels 0.11 builds the quantized fast-path as single-arch SASS (no
# PTX) from this var; leaving the baked cap in place would build every leg's
# fast kernels for the image's one arch, which silently cannot launch on the
# other legs' devices (issue #434). The load-time canary (crates/jammi-kernels)
# is the shipped-artifact guard for that failure mode; this makes each prove
# leg build and test its own native arch instead.
export CUDA_COMPUTE_CAP=${NATIVE_COMPUTE_CAP}
echo "::group::device"; nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv; echo "CUDA_COMPUTE_CAP=\${CUDA_COMPUTE_CAP:-<unset>}"; echo "::endgroup::"
# Hard assertion: nvidia-smi's reported compute_cap (e.g. "8.0") and the
# NATIVE_COMPUTE_CAP override above (e.g. "80") must name the SAME device --
# a mismatch here means the rented pod is not the arch this leg thinks it is,
# and every kernel built below would be silently wrong for it (issue #434's
# exact failure mode, one layer earlier). nvidia-smi's dotted form is
# normalized (dot stripped) before comparing against CUDA_COMPUTE_CAP's bare
# digit form.
compute_cap_raw="\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '[:space:]')"
compute_cap_norm="\${compute_cap_raw//./}"
if [ "\${compute_cap_norm}" != "\${CUDA_COMPUTE_CAP:-}" ]; then
  echo "::error::compute_cap mismatch: nvidia-smi reports compute_cap=\${compute_cap_raw} (normalized \${compute_cap_norm}) but CUDA_COMPUTE_CAP=\${CUDA_COMPUTE_CAP:-<unset>} -- rented device does not match this leg's requested arch, refusing to build"
  exit 97
fi
cd /root && rm -rf jammi-ai
git clone --depth 1 -b "${GIT_REF}" "${GIT_REPO}" jammi-ai 2>&1 | tail -1
cd jammi-ai
rc=0
# The vendored FlashAttention-2 build (\`flash-attn\`, now in the manifest's
# cu12-tarball feature list read below) needs the CUTLASS submodule; a plain
# shallow \`git clone\` above does not fetch submodules.
git submodule update --init --depth 1 crates/jammi-kernels/third_party/cutlass \
  || { echo "::error::CUTLASS submodule init failed (network/remote unreachable?) — refusing to attempt the flash-attn build" >&2; exit 1; }

echo "::group::capability-surface build (ci/release-feature-manifest.json, cu12-tarball lane, never a hardcoded copy)"
# esc-074: the manifest is the single source of truth for the CUDA release
# lanes' exact feature list; this leg reads it (python3, always present in
# the CUDA CI image this pod boots from) rather than restating it. Fails
# loud on a missing/malformed manifest — never a silent empty-string build.
cu12_features="\$(python3 -c "
import json
d = json.load(open('ci/release-feature-manifest.json'))
print(','.join(d['lanes']['cu12-tarball']['cargo_features']))
")"
if [ -z "\${cu12_features}" ]; then
  echo "::error::ci/release-feature-manifest.json produced an empty cu12-tarball cargo_features list" >&2
  exit 1
fi
# jammi-ai lacks jammi-server-only plumbing (jetstream-broker, storage-cloud)
# — the manifest's own \`server_only_cargo_features.features\` documents the
# subtraction so this mapping lives in one place, not a second hardcoded
# copy here.
ai_features="\$(python3 -c "
import json
d = json.load(open('ci/release-feature-manifest.json'))
lane = set(d['lanes']['cu12-tarball']['cargo_features'])
server_only = set(d['server_only_cargo_features']['features'])
print(','.join(sorted(lane - server_only)))
")"
if [ -z "\${ai_features}" ]; then
  echo "::error::deriving the jammi-ai-applicable feature subset from ci/release-feature-manifest.json produced an empty list" >&2
  exit 1
fi
echo "cu12-tarball cargo_features=\${cu12_features}"
echo "jammi-ai-applicable subset=\${ai_features}"
cargo build --release -p jammi-server --bin jammi-server --features "\${cu12_features}" || rc=\$?
cargo test -p jammi-ai --features "\${ai_features},live-gpu-tests" --test gpu_capability --no-run || rc=\$?
echo "::endgroup::"

echo "::group::capability-surface proof (jammi-ai gpu_capability capability_surface, JAMMI_KERNELS_STRICT=1)"
# \`capability_surface\` is delivered by ai-core in a LATER wave of THIS SAME
# PR (campaign #443, the W3+ waves after this one) — this group must FAIL
# LOUD, never silently skip, if that test is absent from this ref: a name
# filter that matches zero tests exits 0 (\"running 0 tests ... test result:
# ok\"), which would otherwise read as a false-green capability proof.
cap_out="\$(JAMMI_KERNELS_STRICT=1 cargo test -p jammi-ai --features "\${ai_features},live-gpu-tests" --test gpu_capability capability_surface -- --nocapture 2>&1)"
cap_rc=\$?
echo "\${cap_out}"
if echo "\${cap_out}" | grep -q "running 0 tests"; then
  echo "::error::jammi-ai gpu_capability's capability_surface test matched ZERO tests on this ref — ai-core's capability-surface delivery (campaign #443) has not landed here; refusing to read a 0-test run as a pass" >&2
  rc=1
elif [ "\${cap_rc}" -ne 0 ]; then
  rc=\${cap_rc}
fi
echo "::endgroup::"

echo "::group::served client/server GPU proof (jammi-server)"
cargo test -p jammi-server --features cuda,live-gpu-tests --test it grpc_embedding_gpu -- --nocapture --test-threads=1 || rc=\$?
echo "::endgroup::"
echo "::group::engine-core GPU correctness (jammi-ai gpu_capability)"
# \`--skip capability_surface\`: this generic sweep does not set
# JAMMI_KERNELS_STRICT=1 (other gpu_capability tests legitimately exercise
# the FALLBACK admission path, which strict mode would break), but
# capability_surface.rs's own module doc REQUIRES strict mode and asserts
# it at the top of the test (capability_surface.rs:425) — a real,
# fail-closed guard, correctly tripping "wrong mode" here. It is NOT
# skipped because it is unimportant: the dedicated capability-surface
# group directly above already runs it, correctly, under
# JAMMI_KERNELS_STRICT=1. Never weaken capability_surface's own guard and
# never set strict mode on this generic sweep to work around it — do the
# opposite (name-exclude it from the one group that cannot satisfy its
# precondition).
cargo test -p jammi-ai --features cuda,live-gpu-tests --test gpu_capability -- --nocapture --test-threads=1 --skip capability_surface || rc=\$?
echo "::endgroup::"
echo "::group::GPU embedding perf — recorded observability, non-gating (jammi-bench gpu-inference-scale)"
cargo run -p jammi-bench --release --features cuda -- gpu-inference-scale || rc=\$?
echo "::endgroup::"
echo "::group::jammi-kernels lib tests, default features (records the x86_64 Linux run this pod is the only artifact for)"
cargo test -p jammi-kernels -- --nocapture --test-threads=1 || rc=\$?
echo "::endgroup::"
echo "::group::jammi-kernels lib tests, --features cuda (this pod's GPU is the device the suite needs)"
cargo test -p jammi-kernels --features cuda -- --nocapture --test-threads=1 || rc=\$?
echo "::endgroup::"
echo "::group::jammi-kernels clippy, --all-targets --features cuda (the only lane that can compile cuda_parity — required-features = [\"cuda\"] pulls dep:bindgen_cuda, a real CUDA toolchain the hermetic runner does not have)"
cargo clippy -p jammi-kernels --all-targets --features cuda -- -D warnings || rc=\$?
echo "::endgroup::"
echo "PROVE_EXIT=\${rc}"; exit \$rc
REMOTE
rc=$?
echo "=== GPU prove suites exit=${rc} ==="
exit "$rc"
