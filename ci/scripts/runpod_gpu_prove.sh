#!/usr/bin/env bash
# GPU prove-lane: build jammi from source on a real GPU and run the gated GPU
# suites — the served client/server proof (grpc_embedding_gpu) and the
# engine-core correctness suite (gpu_capability). Doubles as the compute-cap
# floor regression gate: builds candle's kernels at THIS LEG'S NATIVE
# CUDA_COMPUTE_CAP (see NATIVE_COMPUTE_CAP below — overrides the image's
# baked cap) and
# runs them on a real device. Shared deploy/run/teardown lives in runpod_lib.sh.
#
# GPU_PROVE_ARCH selects WHICH shipped CUDA arch (crates/jammi-kernels/build.rs
# GENCODE_ARCHES: sm_80/sm_86/sm_89/sm_90) this lane proves; default is
# A100 (sm_80). The sm_XX -> `rp_deploy_arch` key mapping
# below is the ONE place that translation lives; `rp_deploy_arch` itself
# (runpod_lib.sh) is the ONE place the arch key -> actual RunPod GPU-type-id
# candidate list lives — this script never hand-types a GPU-type-id string.
#
# Proof surface == shipped surface: every CUDA-bearing cargo
# invocation below carries a LITERAL `--features` tuple -- never a shell
# variable -- immediately preceded by a `PROVE_TUPLE crate=<c> kind=<k>
# features=<literal>` echo, so `check_execution_surface_reachability.py`'s
# `is_gated`/`extract_tuples_from_line` (which operate on the SOURCE TEXT,
# never evaluate a variable) register the real tuple, and
# `check_flash_attn_closure.py` can assert SET EQUALITY between
# `ci/release-feature-manifest.json`'s `prove_lane.crates.<c>.kinds`
# declaration and these invocations (`ci/scripts/prove_surface.py`'s shared
# canonicalization computes the expected literal for each declared pair).
# The manifest is also READ at runtime (see the capability-surface-build
# group) as a TRIPWIRE ONLY -- comparing manifest-derived features against
# the literal below and failing loud on drift -- never used to BUILD the
# `--features` argument itself.
#
# `rp_run_remote_watched` (runpod_lib.sh) layers an
# inactivity watchdog on top of the ssh budget; `PROVE_GROUPS` below names
# the gating groups this driver's own pass/fail rule reads
# `PROVE_GROUP_RC` markers for (`device` and `bench` are NOT members -- see
# that array's own comment). The `jammi-kernels` clippy lane runs in
# `ci.yml`'s own hermetic `Clippy jammi-kernels --features flash-attn
# --all-targets` step (nvcc, no GPU needed), not here -- see
# `check_lint_surface_closure.py`'s own module doc.
#
# Exit 0 = suites passed (see the driver rule below for what "passed" means
# once the bench-cut exception is folded in); 75 = no capacity for the
# requested arch (neutral skip); 76 = watchdog inactivity kill with a
# gating group unresolved; 124 = budget cut with a gating group unresolved.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This lane's own pods never need 8h — the remote job is capped by RP_TIMEOUT
# (100m by default, see RP_TIMEOUT below), so a tighter deadline bounds the
# damage of an orphan.
RP_TTL_HOURS="${RP_TTL_HOURS:-3}"
# shellcheck source=ci/scripts/runpod_lib.sh
source "$DIR/runpod_lib.sh"

# The prove lane's own budget, exported HERE ONLY (never in
# runpod_lib.sh, whose own `${RP_TIMEOUT:-3000}` default stays 3000s for
# every OTHER caller -- gpu-dev.sh). Two-term backstop:
# `RP_TIMEOUT >= 1.5 * max healthy wall` AND
# `RP_TIMEOUT >= max healthy wall + 3 * RP_INACTIVITY` -- the inactivity
# watchdog is the hang detector, so the backstop only needs to outlast the
# slowest healthy leg plus a late-detected hang, not a from-scratch multiple
# of it. Platform ceiling: with the tag-ref
# 3-attempt retry budget (`gpu-prove.yml`'s own header), `3 * 80m deploy-
# worst-case + 2 * 5m overhead + RP_TIMEOUT/60 <= 360m` bounds RP_TIMEOUT
# well above 6000s, so this value is not constrained by the job-timeout
# ceiling.
export RP_TIMEOUT="${RP_TIMEOUT:-6000}"

GIT_REPO="${GIT_REPO:-https://github.com/${GITHUB_REPOSITORY:-f-inverse/jammi-ai}.git}"
GIT_REF="${GIT_REF:-${GITHUB_SHA:-main}}"
# The remote checkout text, computed at source time (beside NATIVE_COMPUTE_CAP)
# so a fixture that sources this file and renders the heredoc sees it: the
# exact commit when PROVE_EXPECT_SHA is set, else the ref (runpod_lib.sh's
# rp_remote_checkout_lines — one helper for every leg).
REMOTE_CHECKOUT_LINES="$(rp_remote_checkout_lines "${GIT_REF}" "${GIT_REPO}")"

# sm_XX (the GENCODE_ARCHES / check_gpu_parity_matrix.py silicon-axis naming)
# -> the `rp_deploy_arch` candidate-list key (runpod_lib.sh), -> the bare
# numeric NATIVE_COMPUTE_CAP this leg's device actually is, and ->
# LEG_DEVICE_80GB, whether EVERY device on that candidate list is 80GB-class.
#
# NATIVE_COMPUTE_CAP overrides the CI image's baked CUDA_COMPUTE_CAP
# (.docker/ci-cuda.Dockerfile) in the remote build env below: candle-kernels
# 0.11 builds the quantized fast-path kernels as single-arch SASS (no PTX) from
# that env var, so every leg building at the image's one baked cap produces
# kernels that silently cannot launch on the other legs' devices.
#
# LEG_DEVICE_80GB selects the suites whose reference shapes are calibrated for
# an 80GB-class device (the encoders-cuda group's eager training-memory
# bounds, and the encoder-level flash oracles that hold three arms of
# ModernBERT-large at once). The memory a suite needs is stated once, in the
# suite, which refuses a smaller device by name: a leg declared `yes` here
# that rents a smaller device FAILS naming it, never skips. Ampere
# workstation (sm_86) and Ada (sm_89) have no 80GB-class device at all; every
# leg's device holds the padded flash oracle, which runs on all four.
#
# Comment on each line names the SASS target the leg proves. A function, so
# `test_gpu_prove_lane.sh` renders any leg's remote script from this one table.
prove_leg_facts() { # $1 = sm_XX
  case "$1" in
    sm_80) RP_DEPLOY_ARCH=a100 NATIVE_COMPUTE_CAP=80 LEG_DEVICE_80GB=yes ;; # Ampere floor — proves sm_80.
    sm_86) RP_DEPLOY_ARCH=a40  NATIVE_COMPUTE_CAP=86 LEG_DEVICE_80GB=no ;; # Ampere workstation class — proves sm_86.
    sm_89) RP_DEPLOY_ARCH=l40s NATIVE_COMPUTE_CAP=89 LEG_DEVICE_80GB=no ;; # Ada — proves sm_89, fp8, on a 48GB-class device: the flash oracle below loads ModernBERT-large, which a 24GB L4 cannot hold beside its arms.
    sm_90) RP_DEPLOY_ARCH=h100 NATIVE_COMPUTE_CAP=90 LEG_DEVICE_80GB=yes ;; # Hopper — proves sm_90.
    *)
      echo "::error::unknown GPU_PROVE_ARCH '$1' (want: sm_80|sm_86|sm_89|sm_90)"
      return 2
      ;;
  esac
}
GPU_PROVE_ARCH="${GPU_PROVE_ARCH:-sm_80}"
prove_leg_facts "$GPU_PROVE_ARCH" || exit 2

# The six PROOF groups this leg's driver rule below gates on --
# `::group::` names in the heredoc below, verbatim. `device` and `bench` are
# NOT members: `device` carries no proof (it is the compute_cap tripwire),
# and `bench` is deliberately NON-GATING -- a cut/hang inside
# `bench` with every group below still `rc=0` passes the leg (see
# `rp_prove_verdict` below). A group added to the script without a
# corresponding entry here (or vice versa) is caught by
# `test_gpu_prove_lane.sh`'s own closure fixture. Declared BEFORE the
# sourced-execution guard below so `test_gpu_prove_lane.sh` can `source`
# this file (RUNPOD_API_KEY/GPU_PROVE_ARCH pre-set, no network/ssh calls
# made merely by sourcing) and see it, exactly as it sees `rp_prove_verdict`.
PROVE_GROUPS=(capability-surface-build capability-surface-proof served-client-server-proof engine-core-sweep engine-lib-cuda kernels-cuda encoders-cuda lora-cuda bench-cuda)

# The driver rule: decide a leg's real verdict
# from `rp_run_remote_watched`'s own return status (`$1`, taken after ITS OWN
# final drain to EOF) plus the `PROVE_GROUP_RC`/`PROVE_EXIT` markers actually
# landed in the log file (`$2`). A plain function (not inlined after the
# heredoc below) so `test_gpu_prove_lane.sh` can drive it directly, against
# hand-built log fixtures, without renting a pod or even sourcing this
# file's own top-level deploy flow.
#
# Precedence (never the reverse): a normal (in-suite) exit's status is
# returned VERBATIM, whatever it is; a ZERO status may be turned into a
# FAILURE by a `PROVE_GROUPS` member missing its marker or reporting
# rc != 0; a cut/hang with NO `PROVE_EXIT=` line (a genuine budget-cut or
# inactivity-kill, never an in-suite decision) applies the bench-cut
# exception: if every `PROVE_GROUPS` member already shows rc=0, the leg
# PASSES (0 + a `::warning::` naming the follow-up), else the cut/hang code
# (76/124) is returned. Returns the final verdict via `return` (0-255).
rp_prove_verdict() {
  local raw_rc="$1" log="$2"
  declare -A grc_map=()
  local line
  # `|| [ -n "$line" ]`: a bare `while IFS= read -r line;
  # do ...; done < "$log"` silently DROPS the log's final line whenever it
  # has no trailing newline (`read` returns non-zero at EOF, which ends the
  # loop BEFORE the body runs for that last read, even though `read` already
  # populated `line`) -- exactly the shape an abrupt ssh cut leaves behind
  # (the remote's own last byte written is mid-line). This form still
  # processes that final, unterminated line. `rp_parse_prove_marker`
  # (runpod_lib.sh) is the ONE shared grammar/parser both this function and
  # `rp_run_remote_watched`'s own live-stream bookkeeping use -- never a
  # second, independently-drifting copy of the same match+extract logic.
  while IFS= read -r line || [ -n "$line" ]; do
    if rp_parse_prove_marker "$line"; then
      grc_map["$RP_PARSED_MARKER_NAME"]="$RP_PARSED_MARKER_RC"
    fi
  done < "$log"

  local has_prove_exit=0
  if grep -q '^PROVE_EXIT=' "$log" 2>/dev/null; then
    has_prove_exit=1
  fi

  local all_proof_pass=1
  local missing_or_failed=()
  local g v
  for g in "${PROVE_GROUPS[@]}"; do
    v="${grc_map[$g]:-}"
    if [ -z "$v" ] || ! [[ "$v" =~ ^[0-9]+$ ]] || [ "$v" -ne 0 ]; then
      all_proof_pass=0
      missing_or_failed+=("${g}=${v:-<missing>}")
    fi
  done

  local rc="$raw_rc"
  if [ "$raw_rc" -eq 0 ]; then
    # A zero status may be turned into a failure by the marker rule, never
    # the reverse — defends against a `PROVE_EXIT=0` that disagrees with its
    # own groups' markers (should never happen given the heredoc's own tail,
    # but a bare zero is never trusted blindly).
    if [ "$all_proof_pass" -ne 1 ]; then
      echo "::error::GPU prove: ssh exited 0 but PROVE_GROUPS member(s) missing or non-zero: ${missing_or_failed[*]:-<none>} — PROVE_EXIT disagrees with its own markers" >&2
      rc=1
    fi
  elif [ "$has_prove_exit" -eq 1 ]; then
    # In-suite exit (the remote reached its own final `echo PROVE_EXIT=…;
    # exit`) — returned VERBATIM, never relabeled, no matter the code.
    rc="$raw_rc"
  elif [ "$raw_rc" -eq 76 ] || [ "$raw_rc" -eq 124 ]; then
    # A genuine cut/hang (no PROVE_EXIT reached) — the bench-cut exception.
    if [ "$all_proof_pass" -eq 1 ]; then
      echo "::warning::GPU prove: bench cut/hung after every proof group already passed (raw rc=${raw_rc}) — the proof stands; the bench leg is a measurement, not a gate" >&2
      rc=0
    else
      rc="$raw_rc"
    fi
  else
    rc="$raw_rc"
  fi
  return "$rc"
}

# Everything below runs only when this file is EXECUTED, never when it is
# `source`d (test_gpu_prove_lane.sh sources it to reach `rp_prove_verdict`/
# `rp_run_remote_watched`/`PROVE_GROUPS` without renting a pod or making any
# network call merely by sourcing).
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then

# Sweep before renting anything. This workflow sets cancel-in-progress, so a
# superseded run is SIGKILLed and never runs its EXIT trap; the pod it had just
# rented is orphaned. Running the sweep here bounds any such orphan to the gap
# until the next prove run rather than "until the account empties". A non-zero rc here is pre-run HYGIENE, never
# this run's own proof failure (the sweep enumerates and reaps PAST orphans;
# it has no bearing on whether THIS run's own pod later proves anything) --
# logged loudly rather than silently discarded so a real "could not
# enumerate" is visible without gating this run on it.
rp_sweep || echo "::warning::pre-run rp_sweep (orphan hygiene) failed rc=$? -- not a proof failure, continuing"

rp_init
echo "=== provisioning a live ${RP_DEPLOY_ARCH} (${GPU_PROVE_ARCH}) ==="
rp_deploy_arch "$RP_DEPLOY_ARCH" || exit $?

echo "=== running GPU prove suites on ${RP_HOST}:${RP_PORT} ==="
LOG="$(mktemp)"
# RP_WATCH_POLL_S: FIXTURE/DIAGNOSTIC-ONLY escape hatch for
# rp_run_remote_watched's own poll-interval parameter (default 5s). A real
# leg never sets it (defaults to 5); test_gpu_prove_lane.sh's fixtures set it to
# a sub-second value so the REAL executed exit path stays fast and
# deterministic under its own watchdog scenarios, exactly like every other
# fixture's own fast-poll argument.
rp_run_remote_watched "" "${RP_WATCH_POLL_S:-5}" <<REMOTE | tee "$LOG"
export CARGO_TERM_COLOR=never
export CARGO_BUILD_RUSTC_WRAPPER=  # wrapper-off (sccache: no cross-target-dir reuse, ~+33% wall on this image)
# Override the image's baked CUDA_COMPUTE_CAP with this leg's NATIVE arch.
# candle-kernels 0.11 builds the quantized fast-path as single-arch SASS (no
# PTX) from this var; leaving the baked cap in place would build every leg's
# fast kernels for the image's one arch, which silently cannot launch on the
# other legs' devices. The load-time canary (crates/jammi-kernels)
# is the shipped-artifact guard for that failure mode; this makes each prove
# leg build and test its own native arch instead.
export CUDA_COMPUTE_CAP=${NATIVE_COMPUTE_CAP}
echo "::group::device"; nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv; echo "CUDA_COMPUTE_CAP=\${CUDA_COMPUTE_CAP:-<unset>}"; echo "::endgroup::"
# Hard assertion: nvidia-smi's reported compute_cap (e.g. "8.0") and the
# NATIVE_COMPUTE_CAP override above (e.g. "80") must name the SAME device --
# a mismatch here means the rented pod is not the arch this leg thinks it is,
# and every kernel built below would be silently wrong for it (the
# single-arch-SASS failure mode above, one layer earlier). nvidia-smi's dotted form is
# normalized (dot stripped) before comparing against CUDA_COMPUTE_CAP's bare
# digit form.
compute_cap_raw="\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '[:space:]')"
compute_cap_norm="\${compute_cap_raw//./}"
if [ "\${compute_cap_norm}" != "\${CUDA_COMPUTE_CAP:-}" ]; then
  echo "::error::compute_cap mismatch: nvidia-smi reports compute_cap=\${compute_cap_raw} (normalized \${compute_cap_norm}) but CUDA_COMPUTE_CAP=\${CUDA_COMPUTE_CAP:-<unset>} -- rented device does not match this leg's requested arch, refusing to build"
  exit 97
fi
${REMOTE_CHECKOUT_LINES}
echo "PROVE_SHA=\$(git rev-parse HEAD)"
rc=0
# The vendored FlashAttention-2 build (\`flash-attn\`, in the manifest's
# cu12-tarball feature list read below) needs the CUTLASS submodule; a plain
# shallow \`git clone\` above does not fetch submodules.
git submodule update --init --depth 1 crates/jammi-kernels/third_party/cutlass \
  || { echo "::error::CUTLASS submodule init failed (network/remote unreachable?) — refusing to attempt the flash-attn build" >&2; exit 1; }

# capability-surface-build: builds the shipped cu12-tarball server binary and
# compile-checks the jammi-ai capability-surface test binary. The manifest
# read below is a TRIPWIRE ONLY -- it never feeds the literal
# \`--features\` arguments the two cargo invocations carry; a divergence
# between the manifest and the literals fails this group loud rather than
# silently building a narrower (or wider) surface than the manifest claims.
echo "::group::capability-surface-build"
grc=0
cu12_features="\$(python3 -c "
import json
d = json.load(open('ci/release-feature-manifest.json'))
print(','.join(d['lanes']['cu12-tarball']['cargo_features']))
")"
if [ -z "\${cu12_features}" ]; then
  echo "::error::ci/release-feature-manifest.json produced an empty cu12-tarball cargo_features list" >&2
  grc=1
fi
ai_features="\$(python3 -c "
import json
d = json.load(open('ci/release-feature-manifest.json'))
lane = set(d['lanes']['cu12-tarball']['cargo_features'])
server_only = set(d['server_only_cargo_features']['features'])
print(','.join(sorted(lane - server_only)))
")"
if [ -z "\${ai_features}" ]; then
  echo "::error::deriving the jammi-ai-applicable feature subset from ci/release-feature-manifest.json produced an empty list" >&2
  grc=1
fi
echo "cu12-tarball cargo_features=\${cu12_features}"
echo "jammi-ai-applicable subset=\${ai_features}"
if [ "\${cu12_features}" != "cuda,flash-attn,jetstream-broker,storage-cloud" ]; then
  echo "::error::PROVE_SURFACE_DRIFT: manifest-derived cu12-tarball cargo_features (\${cu12_features}) no longer matches the literal jammi-server RELEASE tuple this leg builds (cuda,flash-attn,jetstream-broker,storage-cloud) -- update the literal (and its PROVE_TUPLE echo) in the SAME unit as the manifest edit" >&2
  grc=1
fi
if [ "\${ai_features}" != "cuda,flash-attn" ]; then
  echo "::error::PROVE_SURFACE_DRIFT: manifest-derived jammi-ai-applicable subset (\${ai_features}) no longer matches the literal jammi-ai TEST tuple's non-prove_only half (cuda,flash-attn)" >&2
  grc=1
fi
echo "PROVE_TUPLE crate=jammi-server kind=release features=cuda,flash-attn,jetstream-broker,storage-cloud"
cargo build --release -p jammi-server --bin jammi-server --features cuda,flash-attn,jetstream-broker,storage-cloud || grc=\$?
echo "PROVE_TUPLE crate=jammi-ai kind=test features=cuda,flash-attn,live-gpu-tests"
cargo test -p jammi-ai --features cuda,flash-attn,live-gpu-tests --test gpu_capability --no-run || grc=\$?
[ "\$grc" -ne 0 ] && rc=\$grc
echo "PROVE_GROUP_RC name=capability-surface-build rc=\${grc}"
echo "::endgroup::"

# capability-surface-proof: this group must FAIL LOUD, never silently skip,
# if \`capability_surface\` is absent from this ref: a name filter
# that matches zero tests exits 0 ("running 0 tests ... test result: ok"),
# which would otherwise read as a false-green capability proof.
echo "::group::capability-surface-proof"
grc=0
echo "PROVE_TUPLE crate=jammi-ai kind=test features=cuda,flash-attn,live-gpu-tests"
cap_out="\$(JAMMI_KERNELS_STRICT=1 cargo test -p jammi-ai --features cuda,flash-attn,live-gpu-tests --test gpu_capability capability_surface -- --nocapture 2>&1)"
cap_rc=\$?
echo "\${cap_out}"
if echo "\${cap_out}" | grep -q "running 0 tests"; then
  echo "::error::jammi-ai gpu_capability's capability_surface test matched ZERO tests on this ref — the capability-surface test is absent here; refusing to read a 0-test run as a pass" >&2
  grc=1
elif [ "\${cap_rc}" -ne 0 ]; then
  grc=\${cap_rc}
fi
[ "\$grc" -ne 0 ] && rc=\$grc
echo "PROVE_GROUP_RC name=capability-surface-proof rc=\${grc}"
echo "::endgroup::"

# The groups below run only tests that need the GPU: a target that is GPU-only
# as a whole (\`required-features = ["live-gpu-tests"]\`), or the \`gpu\` module a
# CPU target keeps its GPU tests in, selected by \`gpu::\`. Every CPU test runs
# on the hosted CI runners instead. \`ran_tests\` fails a run in which any test
# binary ran no test: a filter or target that selects nothing proves nothing.
# \$1 = the cargo exit status; reads the run's log from /tmp/gpu_tests.log.
ran_tests() {
  [ "\$1" -eq 0 ] || return "\$1"
  if ! grep -q 'test result: ok' /tmp/gpu_tests.log || grep -Eq 'test result: ok\. 0 passed' /tmp/gpu_tests.log; then
    echo "::error::a test binary in this run ran no test" >&2
    return 1
  fi
}

# served-client-server-proof: the shipped served attention surface and the
# remote-session read-back on the device.
echo "::group::served-client-server-proof"
grc=0
echo "PROVE_TUPLE crate=jammi-server kind=test features=cuda,flash-attn,jetstream-broker,live-gpu-tests,storage-cloud"
cargo test -p jammi-server --features cuda,flash-attn,jetstream-broker,live-gpu-tests,storage-cloud --test it -- gpu:: --nocapture --test-threads=1 2>&1 | tee /tmp/gpu_tests.log
ran_tests \${PIPESTATUS[0]} || grc=\$?
[ "\$grc" -ne 0 ] && rc=\$grc
echo "PROVE_GROUP_RC name=served-client-server-proof rc=\${grc}"
echo "::endgroup::"

# engine-core-sweep: \`--skip capability_surface\` -- this sweep does not set
# JAMMI_KERNELS_STRICT=1 (other gpu_capability tests exercise the FALLBACK
# admission path, which strict mode would break); capability_surface requires
# strict mode and runs, under it, in capability-surface-proof above.
echo "::group::engine-core-sweep"
grc=0
echo "PROVE_TUPLE crate=jammi-ai kind=test features=cuda,flash-attn,live-gpu-tests"
cargo test -p jammi-ai --features cuda,flash-attn,live-gpu-tests --test gpu_capability -- --nocapture --test-threads=1 --skip capability_surface 2>&1 | tee /tmp/gpu_tests.log
ran_tests \${PIPESTATUS[0]} || grc=\$?
[ "\$grc" -ne 0 ] && rc=\$grc
echo "PROVE_GROUP_RC name=engine-core-sweep rc=\${grc}"
echo "::endgroup::"

# engine-lib-cuda: jammi-ai's own CUDA-device unit tests.
echo "::group::engine-lib-cuda"
grc=0
echo "PROVE_TUPLE crate=jammi-ai kind=test features=cuda,flash-attn,live-gpu-tests"
cargo test -p jammi-ai --features cuda,flash-attn,live-gpu-tests --lib -- gpu:: --test-threads=1 2>&1 | tee /tmp/gpu_tests.log
ran_tests \${PIPESTATUS[0]} || grc=\$?
[ "\$grc" -ne 0 ] && rc=\$grc
echo "PROVE_GROUP_RC name=engine-lib-cuda rc=\${grc}"
echo "::endgroup::"

# kernels-cuda: jammi-kernels' device suites, the FlashAttention-2 ones
# included, then the GPU tests its lib and admission-class oracle keep in
# \`gpu\` modules.
echo "::group::kernels-cuda"
grc=0
echo "PROVE_TUPLE crate=jammi-kernels kind=test features=cuda,flash-attn,live-gpu-tests"
cargo test -p jammi-kernels --features cuda,flash-attn,live-gpu-tests --test cuda_parity --test flash_smoke --test flash_op_oracles --test flash_torch_parity --test flash_torch_parity_f16 -- --test-threads=1 2>&1 | tee /tmp/gpu_tests.log
ran_tests \${PIPESTATUS[0]} || grc=\$?
echo "PROVE_TUPLE crate=jammi-kernels kind=test features=cuda,flash-attn,live-gpu-tests"
cargo test -p jammi-kernels --features cuda,flash-attn,live-gpu-tests --lib --test empty_non_contiguous_admission_class_oracle -- gpu:: --test-threads=1 2>&1 | tee /tmp/gpu_tests.log
ran_tests \${PIPESTATUS[0]} || grc=\$?
[ "\$grc" -ne 0 ] && rc=\$grc
echo "PROVE_GROUP_RC name=kernels-cuda rc=\${grc}"
echo "::endgroup::"

# encoders-cuda: the encoder device tests, with the flash oracles compiled in
# (\`live-flash-oracle-tests\`): ModernBERT-large's padded flash arm against
# the block arm on real rows, the arch's flash validation. The checkpoint is
# fetched once per leg; the oracles read it from JAMMI_FLASH_ORACLE_MODEL_DIR.
# The eager training-memory bounds and the encoder-level oracles (three arms
# of the model at once) name the device memory their shapes are calibrated
# for and refuse a smaller device; the legs whose every device is 80GB-class
# (LEG_DEVICE_80GB, this script's per-leg table) run them, every other leg
# skips the encoder-level oracles by name and runs the padded oracle alone.
# Each PROVE_TUPLE echo sits inside the same condition as its invocation, so
# a leg's log claims the invocation only where it ran.
echo "::group::encoders-cuda"
grc=0
export JAMMI_FLASH_ORACLE_MODEL_DIR=${RP_REMOTE_ROOT}/checkpoints/ModernBERT-large
mkdir -p "\$JAMMI_FLASH_ORACLE_MODEL_DIR"
for f in config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json; do
  [ -s "\$JAMMI_FLASH_ORACLE_MODEL_DIR/\$f" ] || curl -fsSL "https://huggingface.co/answerdotai/ModernBERT-large/resolve/main/\$f" -o "\$JAMMI_FLASH_ORACLE_MODEL_DIR/\$f" || grc=\$?
done
if [ "${LEG_DEVICE_80GB}" = yes ]; then
  echo "PROVE_TUPLE crate=jammi-encoders kind=test features=cuda,flash-attn,live-flash-oracle-tests,live-gpu-tests"
  cargo test -p jammi-encoders --features cuda,flash-attn,live-flash-oracle-tests,live-gpu-tests --lib --test it -- gpu:: --test-threads=1 2>&1 | tee /tmp/gpu_tests.log
  ran_tests \${PIPESTATUS[0]} || grc=\$?
  echo "PROVE_TUPLE crate=jammi-encoders kind=test features=cuda,flash-attn,live-flash-oracle-tests,live-gpu-tests"
  cargo test -p jammi-encoders --features cuda,flash-attn,live-flash-oracle-tests,live-gpu-tests --test eager_training_memory -- --test-threads=1 2>&1 | tee /tmp/gpu_tests.log
  ran_tests \${PIPESTATUS[0]} || grc=\$?
else
  echo "PROVE_TUPLE crate=jammi-encoders kind=test features=cuda,flash-attn,live-flash-oracle-tests,live-gpu-tests"
  cargo test -p jammi-encoders --features cuda,flash-attn,live-flash-oracle-tests,live-gpu-tests --lib --test it -- gpu:: --test-threads=1 --skip flash_arm_encoder_level --skip flash_vs_block_per_layer_vram --skip flash_arm_fault_harness 2>&1 | tee /tmp/gpu_tests.log
  ran_tests \${PIPESTATUS[0]} || grc=\$?
fi
[ "\$grc" -ne 0 ] && rc=\$grc
echo "PROVE_GROUP_RC name=encoders-cuda rc=\${grc}"
echo "::endgroup::"

# lora-cuda: the fused LoRA epilogue against the PEFT reference on the device.
echo "::group::lora-cuda"
grc=0
echo "PROVE_TUPLE crate=jammi-lora kind=test features=cuda,live-gpu-tests"
cargo test -p jammi-lora --features cuda,live-gpu-tests --test epilogue_peft_rounding -- --test-threads=1 2>&1 | tee /tmp/gpu_tests.log
ran_tests \${PIPESTATUS[0]} || grc=\$?
[ "\$grc" -ne 0 ] && rc=\$grc
echo "PROVE_GROUP_RC name=lora-cuda rc=\${grc}"
echo "::endgroup::"

# bench-cuda: the harness's padded fine-tune step on the device.
echo "::group::bench-cuda"
grc=0
echo "PROVE_TUPLE crate=jammi-bench kind=test features=cuda,flash-attn,live-gpu-tests"
cargo test -p jammi-bench --features cuda,flash-attn,live-gpu-tests --test finetune_step_padded_cuda -- --test-threads=1 2>&1 | tee /tmp/gpu_tests.log
ran_tests \${PIPESTATUS[0]} || grc=\$?
[ "\$grc" -ne 0 ] && rc=\$grc
echo "PROVE_GROUP_RC name=bench-cuda rc=\${grc}"
echo "::endgroup::"

# bench: the encode workload's rungs on the device — recorded legs,
# deliberately NON-GATING -- \`bench_rc\`
# never touches \`rc\`, and this group runs LAST so a cut/hang here, with every
# group above already at rc=0, never blocks the leg (see the driver rule in
# runpod_gpu_prove.sh, after this heredoc). Widened to \`cuda,flash-attn\`
# (prove_lane's jammi-bench \`release\` pair).
echo "::group::bench"
bench_rc=0
echo "PROVE_TUPLE crate=jammi-bench kind=release features=cuda,flash-attn"
cargo run -p jammi-bench --release --features cuda,flash-attn -- encode-step --cuda 0 --rung direct --rung plan --rung plan-partitioned --rows 256 || bench_rc=\$?
echo "BENCH_EXIT=\${bench_rc}"
echo "PROVE_GROUP_RC name=bench rc=\${bench_rc}"
echo "::endgroup::"

echo "PROVE_EXIT=\${rc}"; exit \$rc
REMOTE
raw_rc="${PIPESTATUS[0]}"

rp_prove_verdict "$raw_rc" "$LOG"
rc=$?

rm -f "$LOG"
echo "=== GPU prove suites exit=${rc} (raw=${raw_rc}) ==="
exit "$rc"

fi # end sourced-execution guard
