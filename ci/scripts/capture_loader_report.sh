#!/usr/bin/env bash
# Captures a REAL `ldd`-shaped loader report from the real cu12
# `jammi-server` tarball's staged `lib/`, verbatim, to stdout — the fixture
# `ci/scripts/fixtures/cu12_loader_report_real.txt` needs and
# `ci/scripts/test_bundle_cuda_libs.sh`'s real-report arm (1a) verifies. Never
# hand-typed, never inferred: see that fixture's own header for why.
#
# Runs `ldd` with `LD_LIBRARY_PATH` set to the staged `lib/` directory ONLY —
# the same shape the tarball's own launcher uses at `exec` time
# (`.github/workflows/release-binaries.yml`'s `LAUNCH` heredoc). This capture
# step does NOT restrict the loader's search the way the chroot/`unshare`
# shape `bundle_verify_loader_resolution`'s module doc describes would (see
# `ci/scripts/bundle_cuda_libs.sh`'s stop-rule note for why that shape is not
# available on this repo's CI runners today) — it is a CAPTURE of whatever
# the real loader reports on the host it runs on, which is exactly why the
# capture must run on a driver-only-style host (an NVIDIA-driver-only Linux
# box with no system CUDA toolkit installed) rather than a CUDA-toolkit build
# image: on a build image, the toolkit's OWN copies of these libraries sit on
# the default loader search path regardless of `LD_LIBRARY_PATH`, and the
# report would look clean even for a tarball that is missing a library
# entirely — defect (1) this whole arm exists to close, one level up: at
# capture time rather than verification time.
#
# Usage:
#   capture_loader_report.sh <staged-lib-dir> <binary>
#
# Output: the raw `ldd` report on stdout, verbatim — redirect it into
# `ci/scripts/fixtures/cu12_loader_report_real.txt` in place of that file's
# `captured: pending` placeholder header.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: capture_loader_report.sh <staged-lib-dir> <binary>" >&2
  exit 2
fi

lib_dir="$1"
binary="$2"

if [ ! -d "$lib_dir" ]; then
  echo "::error::capture_loader_report.sh: no such directory: ${lib_dir}" >&2
  exit 1
fi
if [ ! -e "$binary" ]; then
  echo "::error::capture_loader_report.sh: no such file: ${binary}" >&2
  exit 1
fi
if ! command -v ldd >/dev/null 2>&1; then
  echo "::error::capture_loader_report.sh: 'ldd' is not on PATH -- run this on the Linux host the tarball targets, not a macOS box." >&2
  exit 1
fi

# `ldd` exits non-zero when it reports any "not found" line (expected here,
# on a driver-only host, for the driver's own libraries) -- the report itself
# is still the thing this script exists to capture, so its exit status is
# deliberately not propagated.
LD_LIBRARY_PATH="$lib_dir" ldd "$binary" || true
