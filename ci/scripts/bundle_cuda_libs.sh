#!/usr/bin/env bash
# Stage the shared libraries the CUDA (`cu12`) `jammi-server` tarball must
# carry, DERIVED from the binary's own dynamic-link requirements rather than
# from a hand-kept list of names.
#
# The tarball ships a launcher that puts `lib/` on `LD_LIBRARY_PATH` and execs
# the binary, so the tarball is only self-sufficient if `lib/` holds every
# SONAME the loader will look for on a host that has an NVIDIA driver and
# nothing else. `release-binaries.yml`'s CUDA leg used to state that set as six
# literal names (`libcudart libcublas libcublasLt libcurand libnvrtc
# libnvrtc-builtins`). A literal list is a copy of a fact that lives in the
# binary, and it drifts the moment a feature adds a link: `jammi-ai`'s `cuda`
# feature includes `candle-core/nccl`, cudarc's build script emits
# `cargo:rustc-link-lib=dylib=nccl` for it, and the resulting binary carries a
# `DT_NEEDED libnccl.so.2` that no name on that list covers — a green build
# shipping a tarball that cannot `exec` on a driver-only host, discovered by a
# user rather than by CI. Reading the requirement off the binary removes the
# copy; every future link is carried automatically, and one that cannot be
# satisfied is a loud failure here instead of a silent omission.
#
# The single mechanism here is DERIVATION (`bundle_copy_sources`): the staged
# set is the transitive `DT_NEEDED` closure of the binary, minus the
# host-provided set, each soname resolved to a real file in an explicit,
# ordered search path. TRANSITIVE, not direct: a library the binary itself
# does not name can still be needed by a library the closure DOES resolve,
# and the loader needs it too. A soname that resolves nowhere fails this
# script, naming it.
#
# This script does NOT ask the real loader whether the staged tree satisfies
# the binary at run time (`LD_LIBRARY_PATH=<lib> ldd <binary>`, a mechanism
# this script carried and then retired): a runtime loader verification is a
# real, separate question — `LD_LIBRARY_PATH` PREPENDS to the loader's
# search rather than RESTRICTING it, so asking `ldd` on a build host that
# happens to carry a library too can report a soname resolved when the
# tarball itself never staged it — and is filed as a follow-up rather than
# asserted here.
#
# Usage:
#   bundle_cuda_libs.sh <binary> <stage-lib-dir> [search-path]
#     `search-path` is a colon-separated list of directories, tried in order;
#     it defaults to the CUDA CI image's layout — the CUDA 12.6 toolkit
#     (`/usr/local/cuda-12.6/lib64`, which carries libcudart/libcublas/…) then
#     `/usr/lib64` (which is where the image's `libnccl` RPM installs
#     `libnccl.so.2`; the toolkit dir carries no NCCL at all). Copies with
#     `cp -L`, so a development symlink is materialised as the real object.
#
# The functions are `source`-able and every one of them but `bundle_needed_
# sonames` is pure (filesystem reads only, no ELF tooling), which is what makes
# this derivation testable off a Linux host at all: `ci/scripts/
# test_bundle_cuda_libs.sh` sources this file, replaces that one function with
# a fixture, and drives the rest over a fake library tree. The workflow leg
# that calls this script runs only on a `v*` tag — without that suite the
# derivation would first be exercised during a release.
set -euo pipefail

# The CUDA CI image's layout (`.docker/ci-cuda.Dockerfile`): the 12.6 toolkit
# first, then the system library directory the `libnccl` RPM installs into.
BUNDLE_DEFAULT_SEARCH_PATH="${BUNDLE_DEFAULT_SEARCH_PATH:-/usr/local/cuda-12.6/lib64:/usr/lib64}"

# Sonames the tarball must NOT carry, in two kinds.
#
# The platform half (glibc and the GCC/C++ runtimes, plus the dynamic loader
# itself) is the host's: these are present on every glibc host the tarball
# targets, the release lane asserts its own glibc floor separately
# (`assert_glibc_floor.sh`), and bundling a second copy of `libstdc++` or
# `libgcc_s` ahead of the host's on `LD_LIBRARY_PATH` is an ABI hazard rather
# than a service. The driver half (`libcuda.so.1`, `libnvidia-*`) ships with
# the user's NVIDIA driver and MUST match it — bundling a build-host copy would
# override the driver the GPU is actually running.
#
# This is the same partition `packaging/server-cu12/verify_link_set.py` states
# for the wheel (its `PLATFORM` + `DRIVER_PROVIDED` sets); the two files answer
# the same question for two artifacts. Anything not matched here is the
# tarball's responsibility, and if it cannot be resolved this script fails
# rather than skipping it — an unknown soname is a decision for a human, never
# a silent omission.
bundle_is_platform_soname() {
  case "$1" in
    libc.so.* | libm.so.* | libmvec.so.* | libdl.so.* | librt.so.* | libpthread.so.* | libgcc_s.so.* | libstdc++.so.*)
      return 0
      ;;
    ld-linux-*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

bundle_is_driver_soname() {
  case "$1" in
    libcuda.so.* | libnvidia-*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

bundle_is_host_provided() {
  bundle_is_platform_soname "$1" || bundle_is_driver_soname "$1"
}

# The one function that reads an ELF file, and therefore the one the hermetic
# suite replaces. Prints the file's `DT_NEEDED` sonames, one per line, in link
# order; prints nothing for a file with none.
bundle_needed_sonames() {
  # `readelf -d` lines read: `0x... (NEEDED)  Shared library: [libcudart.so.12]`
  readelf -d "$1" | sed -n 's/.*(NEEDED).*\[\(.*\)\].*/\1/p'
}

# The directory of the first entry in `search_path` that holds `soname`, or a
# non-zero exit if no entry does. First match wins, so the search path's ORDER
# is meaningful: the toolkit's own copy of a library beats a system one.
bundle_resolve_soname() {
  local soname="$1"
  local search_path="$2"
  local dir
  local IFS=:
  for dir in $search_path; do
    [ -n "$dir" ] || continue
    if [ -e "$dir/$soname" ]; then
      printf '%s\n' "$dir"
      return 0
    fi
  done
  return 1
}

# The copy sources for a binary's link closure: every file to stage, absolute,
# one per line. Arguments: the colon-separated search path, then the sonames to
# start from (the binary's own `DT_NEEDED` list).
#
# For each soname that is not host-provided, EVERY versioned object of that
# stem in the resolving directory is emitted — `libcudart.so.12` (the SONAME
# the loader asks for) and `libcudart.so.12.6.77` (the real object the first is
# a symlink to) alike — because `cp -L` of the SONAME alone would land a file
# named for the SONAME whose own internal soname matches, but a sibling library
# linked against the fully-versioned name would then find nothing. The closure
# continues through each resolved object's own `DT_NEEDED`.
#
# Fails (exit 1), naming every soname, if any non-host soname resolves nowhere
# in the search path. That refusal is the point: the alternative — staging what
# was found and shipping — is exactly the silent omission this script exists to
# end.
bundle_copy_sources() {
  local search_path="$1"
  shift
  local queue=("$@")
  local seen=" "
  local sources=""
  local missing=""
  local i=0
  local soname stem dir resolved so dep

  while [ "$i" -lt "${#queue[@]}" ]; do
    soname="${queue[$i]}"
    i=$((i + 1))
    case "$seen" in
      *" $soname "*) continue ;;
    esac
    seen="$seen$soname "
    if bundle_is_host_provided "$soname"; then
      continue
    fi
    if ! dir="$(bundle_resolve_soname "$soname" "$search_path")"; then
      missing="$missing $soname"
      continue
    fi
    resolved="$dir/$soname"
    stem="${soname%%.so*}"
    for so in "$dir/$stem.so".*; do
      [ -e "$so" ] || continue
      case "$sources" in
        *"$so"$'\n'*) continue ;;
      esac
      sources="$sources$so"$'\n'
    done
    for dep in $(bundle_needed_sonames "$resolved"); do
      queue[${#queue[@]}]="$dep"
    done
  done

  if [ -n "$missing" ]; then
    echo "::error::bundle_cuda_libs.sh: no directory in '${search_path}' holds:${missing}" >&2
    echo "The tarball cannot be assembled without them — the binary would not exec on a driver-only host." >&2
    return 1
  fi

  printf '%s' "$sources"
}

bundle_main() {
  if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "usage: bundle_cuda_libs.sh <binary> <stage-lib-dir> [search-path]" >&2
    return 2
  fi
  local binary="$1"
  local lib_dir="$2"
  local search_path="${3:-$BUNDLE_DEFAULT_SEARCH_PATH}"
  local sources src

  mkdir -p "$lib_dir"
  # Deliberately unquoted: the binary's DT_NEEDED list is one soname per line
  # and is passed as separate arguments (sonames contain no whitespace).
  # shellcheck disable=SC2046
  sources="$(bundle_copy_sources "$search_path" $(bundle_needed_sonames "$binary"))"
  if [ -z "$sources" ]; then
    echo "::error::bundle_cuda_libs.sh: ${binary} names no bundle-able DT_NEEDED library at all — a CUDA build always links at least the CUDA runtime, so this is a defect in the build, not an empty-but-correct set." >&2
    return 1
  fi
  while IFS= read -r src; do
    [ -n "$src" ] || continue
    echo "bundle_cuda_libs.sh: staging ${src}"
    cp -L "$src" "$lib_dir/"
  done <<EOF
$sources
EOF
}

# Sourced by the suite (which needs the functions and not the side effects);
# executed by the release lane.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  bundle_main "$@"
fi
