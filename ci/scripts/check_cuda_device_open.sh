#!/usr/bin/env bash
# Every CUDA device opens through `jammi_kernels::device::open_cuda`. That
# constructor holds cuBLAS's reductions to the compute type; a device opened
# any other way keeps cuBLAS's default math mode, under which an RTX 4090 or
# RTX 6000 Ada rounds split-K partial sums to bf16 and a row embeds
# differently alone than in a batch. The GPU prove's cards (A100, A40, L40S,
# H100) do not split those shapes, so no GPU lane would notice a bypass.
#
# Scope: Rust sources under crates/. Comment lines are prose, not calls.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 2

OPENS='(Device::new_cuda|Device::new_cuda_with_stream)([^A-Za-z0-9_]|$)|CudaDevice::new(_with_stream)?\('
COMMENT_LINE='^[^:]+:[0-9]+:[[:space:]]*//'

rc=0
hits="$(git grep -n -E "$OPENS" -- 'crates/*.rs' ':!crates/jammi-kernels/src/device.rs')" || rc=$?
case "$rc" in
  0) ;;
  1) exit 0 ;;
  *) echo "git grep failed (exit $rc): the tree was not scanned" >&2; exit 2 ;;
esac
rc=0
calls="$(grep -v -E "$COMMENT_LINE" <<<"$hits")" || rc=$?
case "$rc" in
  0)
    printf '%s\n' "$calls"
    echo "a CUDA device opened here keeps cuBLAS's default math mode: open it with jammi_kernels::device::open_cuda (tests: jammi_test_resources::cuda_device)" >&2
    exit 1
    ;;
  1) exit 0 ;;
  *) echo "grep failed (exit $rc) filtering comment lines" >&2; exit 2 ;;
esac
