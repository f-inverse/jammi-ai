#!/usr/bin/env bash
# Build the `jammi-ai-native-cu12` wheel and check the extension it carries.
#
# One recipe for every place the wheel is built: the release workflow
# (`.github/workflows/pypi-native-cuda.yml`, in the CUDA CI image) and a GPU
# pod verifying it (`docs/maintainer/dev-gpu.md`). Needs the CUDA toolkit, the
# CUTLASS submodule (`flash-attn`) and maturin; writes the wheel to
# `packaging/native-cu12/dist/` and prints its path last.
#
# auditwheel is skipped: its repair would copy the CUDA runtime into the wheel,
# which instead depends on NVIDIA's `nvidia-*-cu12` wheels and finds their
# libraries through the RUNPATH `crates/jammi-python/build.rs` sets. The checks
# below stand in for it — the glibc floor the manylinux_2_28 tag promises, the
# machine, every DT_NEEDED library classified and delivered, and the RUNPATH.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT/packaging/native-cu12"
rm -rf dist
maturin build --release --out dist --compatibility manylinux_2_28 --auditwheel skip

wheel="$(ls dist/jammi_ai_native_cu12-*.whl)"
work="$(mktemp -d)"
python3 -m zipfile -e "$wheel" "$work"
so="$(find "$work" -name '*.so' -print -quit)"
if [ -z "$so" ]; then
  echo "::error::no extension .so inside $wheel" >&2
  exit 1
fi

bash "$ROOT/ci/scripts/assert_glibc_floor.sh" "$so" 2.28
bash "$ROOT/ci/scripts/assert_elf_machine.sh" "$so" x86_64
python3 "$ROOT/packaging/server-cu12/verify_link_set.py" "$so"
dynamic="$(readelf -d "$so")"
case "$dynamic" in
  *'$ORIGIN/../nvidia/cuda_runtime/lib'*) ;;
  *)
    echo "::error::$so carries no RUNPATH to the nvidia-*-cu12 wheels' libraries:" >&2
    echo "$dynamic" >&2
    exit 1
    ;;
esac
echo "$ROOT/packaging/native-cu12/$wheel"
