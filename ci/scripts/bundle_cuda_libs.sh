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
# Two mechanisms, kept separate on purpose because they answer two different
# questions and neither can stand in for the other:
#
#   1. DERIVATION (`bundle_copy_sources`, over `bundle_resolve_closure`): the
#      staged set is the transitive `DT_NEEDED` closure of the binary, minus
#      the host-provided set, each soname resolved to a real file in an
#      explicit, ordered search path. TRANSITIVE, not direct: a library the
#      binary itself does not name can still be needed by a library the
#      closure DOES resolve, and the loader needs it too. A soname that
#      resolves nowhere fails this script, naming it.
#   2. THE FLOOR (`bundle_stage_floor`): seven stems this tarball has carried
#      since before this derivation existed — `libcudart libcublas
#      libcublasLt libcurand libnvrtc libnvrtc-builtins libnccl` — resolved
#      and staged the same way (first match in the search path wins),
#      INDEPENDENTLY of whatever the `DT_NEEDED` closure above happens to
#      reach. This is not redundant with (1): MEASURED (`readelf -d` against
#      the CUDA 12.6 toolkit's own `libnvrtc.so.12`, run inside the
#      `nvidia/cuda:12.6.3-devel-ubi8` image — see
#      `ci/scripts/test_bundle_cuda_libs.sh` for the recorded `NEEDED` lines)
#      shows that `libnvrtc.so.12` names NO `DT_NEEDED` entry for
#      `libnvrtc-builtins` at all — NVRTC `dlopen`s its builtins library at
#      runtime rather than linking it, so no `DT_NEEDED` closure, however
#      faithfully it is walked, can ever discover it. A derivation-only
#      design would silently ship a tarball that cannot JIT-compile a kernel
#      the first time a user's workload calls into NVRTC. The floor is the
#      one thing in this script that is NOT derived from the binary; it is
#      the fixed fact a `DT_NEEDED` walk is structurally blind to.
#
# After both, a post-copy STAGE-SET ASSERTION (`bundle_assert_staged`) checks
# — pure filesystem, no ELF reader — that every soname the closure named
# exists as a regular file of exactly that SONAME under `$lib_dir`. This is
# not the same claim as "the derivation resolved it": the copy step
# (`bundle_copy_sources`) stages by matching a glob against the resolved
# soname's STEM, and a soname whose shape defeats that glob (an unversioned
# `DT_NEEDED libfoo.so`, whose own name has no trailing `.` for the glob to
# match — closed below, but a FUTURE regression in the same shape is exactly
# what this assertion is insurance against) can resolve cleanly and still
# leave `$lib_dir` without the file the derivation claims it satisfied. The
# floor's own refusal (naming a stem no search directory holds at all) plays
# the same role for the fixed seven.
#
# What NEITHER mechanism establishes: a `dlopen`-loaded dependency other than
# the floor's own — NVRTC has one more this script does not special-case, and
# any future library that takes the same shape — is invisible to a
# `DT_NEEDED` walk and stays invisible unless a human adds it to the floor.
# Nor does staging a same-named file under `$lib_dir` prove the REAL loader,
# at run time, on the tarball's own launcher (`LD_LIBRARY_PATH` prepended,
# never restricted), actually resolves each entry FROM there rather than from
# some copy the runtime host happens to carry too — that question needs the
# real loader asked against the real binary, which this hermetic, ELF-free
# script cannot do and does not attempt; it is filed as a follow-up runtime
# verification, not silently assumed.
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

# The floor: stems no `DT_NEEDED` closure walk is trusted to reach on its own.
# See the module doc for why `libnvrtc-builtins` in particular can never be a
# closure member — it is a measured fact, not a guess.
BUNDLE_FLOOR_STEMS="libcudart libcublas libcublasLt libcurand libnvrtc libnvrtc-builtins libnccl"

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

# The directory of the first entry in `search_path` that holds ANY file whose
# name starts `$stem.so.` — used only by the floor, which starts from a bare
# STEM (it has no single soname string to look for: `libnvrtc-builtins` ships
# as `libnvrtc-builtins.so.12.6` on this toolkit and could ship as any other
# version), not from a resolved `DT_NEEDED` entry.
bundle_resolve_stem_dir() {
  local stem="$1"
  local search_path="$2"
  local dir so
  local IFS=:
  for dir in $search_path; do
    [ -n "$dir" ] || continue
    for so in "$dir/$stem.so".*; do
      [ -e "$so" ] || continue
      printf '%s\n' "$dir"
      return 0
    done
  done
  return 1
}

# The shared closure walk: BFS over a binary's transitive `DT_NEEDED` graph,
# starting from `queue`, skipping the host-provided set. Both `bundle_copy_
# sources` (which turns each resolved entry into copy-source file paths) and
# the post-copy stage-set assertion (which only needs the soname NAMES) drive
# this one walk, so there is exactly one place that decides what is "in the
# closure" and one place that can resolve a soname wrongly.
#
# Arguments: the colon-separated search path, then the sonames to start from.
# Emits one line per RESOLVED, non-host-provided soname reached: `<soname>
# <dir>`, `dir` the directory `bundle_resolve_soname` found it in — in
# resolution order, first-resolved first. Fails (exit 1), naming every
# soname, if any non-host soname resolves nowhere in the search path.
bundle_resolve_closure() {
  local search_path="$1"
  shift
  local queue=("$@")
  local seen=" "
  local resolved_lines=""
  local missing=""
  local i=0
  local soname dir resolved dep

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
    resolved_lines="${resolved_lines}${soname} ${dir}"$'\n'
    for dep in $(bundle_needed_sonames "$resolved"); do
      queue[${#queue[@]}]="$dep"
    done
  done

  if [ -n "$missing" ]; then
    echo "::error::bundle_cuda_libs.sh: no directory in '${search_path}' holds:${missing}" >&2
    echo "The tarball cannot be assembled without them — the binary would not exec on a driver-only host." >&2
    return 1
  fi

  printf '%s' "$resolved_lines"
}

# The copy sources for a binary's link closure: every file to stage, absolute,
# one per line. Arguments: the colon-separated search path, then the sonames to
# start from (the binary's own `DT_NEEDED` list). Built over `bundle_resolve_
# closure`'s walk, so it fails the same way, naming the same missing sonames.
#
# For each resolved, non-host soname, the exact resolved file (`$dir/$soname`)
# is staged UNCONDITIONALLY, closing the unversioned-soname hole: a `DT_NEEDED`
# entry that IS its own final object (`libfoo.so`, no trailing version) can
# never match the stem glob below (`"$dir/$stem.so".*` demands a LITERAL `.`
# immediately after `.so`, which `libfoo.so` itself, with nothing after it,
# does not have), so relying on the glob alone would resolve the soname
# (`-e "$dir/$soname"` passes) yet stage nothing and report nothing for it —
# silence on a real `DT_NEEDED` entry the tarball genuinely cannot `exec`
# without. EVERY versioned SIBLING object of that stem in the resolving
# directory is ALSO emitted — `libcudart.so.12` (the SONAME the loader asks
# for) and `libcudart.so.12.6.77` (the real object the first is a symlink to)
# alike — because `cp -L` of the SONAME alone would land a file named for the
# SONAME whose own internal soname matches, but a sibling library linked
# against the fully-versioned name would then find nothing.
bundle_copy_sources() {
  local search_path="$1"
  shift
  local closure closure_rc
  closure="$(bundle_resolve_closure "$search_path" "$@")"
  closure_rc=$?
  if [ "$closure_rc" -ne 0 ]; then
    return "$closure_rc"
  fi

  local soname dir resolved stem so sources=""
  while IFS=' ' read -r soname dir; do
    [ -n "$soname" ] || continue
    resolved="$dir/$soname"
    case "$sources" in
      *"$resolved"$'\n'*) : ;;
      *) sources="$sources$resolved"$'\n' ;;
    esac
    stem="${soname%%.so*}"
    for so in "$dir/$stem.so".*; do
      [ -e "$so" ] || continue
      case "$sources" in
        *"$so"$'\n'*) continue ;;
      esac
      sources="$sources$so"$'\n'
    done
  done <<EOF
$closure
EOF

  printf '%s' "$sources"
}

# The floor: independent of whatever the binary's own `DT_NEEDED` closure
# reaches, these seven stems have been part of every cu12 tarball shipped
# before this derivation existed. Resolved and staged the SAME way the
# derivation resolves anything (first search-path match wins, every versioned
# object of the stem is copied) — but starting from a fixed STEM, never from a
# `DT_NEEDED` entry, which is exactly the point for `libnvrtc-builtins`:
# measurement (the module doc; `readelf -d` against the real toolkit's
# `libnvrtc.so.12`) shows NVRTC never names it as a link-time dependency at
# all, so no closure walk, however correct, can ever reach it — this is the
# only mechanism that stages it. A stem no search directory holds at all is a
# named FAIL: the same refusal the derivation raises, for the one set this
# script does not trust `DT_NEEDED` to find.
bundle_stage_floor() {
  local search_path="$1"
  local lib_dir="$2"
  local stem dir so missing=""
  for stem in $BUNDLE_FLOOR_STEMS; do
    if ! dir="$(bundle_resolve_stem_dir "$stem" "$search_path")"; then
      missing="${missing} ${stem}"
      continue
    fi
    for so in "$dir/$stem.so".*; do
      [ -e "$so" ] || continue
      cp -L "$so" "$lib_dir/"
    done
  done
  if [ -n "$missing" ]; then
    echo "::error::bundle_cuda_libs.sh: the floor library set is missing from every search directory in '${search_path}':${missing} — this tarball has shipped these seven stems since before this derivation existed; libnvrtc-builtins in particular is dlopen'd by libnvrtc rather than linked (measured: readelf -d against the CUDA 12.6 toolkit's libnvrtc.so.12 names no such NEEDED entry), so no DT_NEEDED closure can ever be trusted to reach it, and this is the mechanism that stages it regardless." >&2
    return 1
  fi
}

# The post-copy stage-set assertion: for every non-host-provided soname the
# `DT_NEEDED` closure named, a regular file of EXACTLY that SONAME must exist
# under `$lib_dir` once the copies are done. Pure filesystem test — `[ -f
# ... ]`, nothing else — which is what makes it hermetic with no `ldd`, no ELF
# reader, on either the fixture tree or the real stage. This is NOT the same
# claim `bundle_copy_sources` already makes by exiting 0: that call proves
# every soname RESOLVED somewhere in the search path, not that the COPY step
# actually landed a same-named file for each one under `$lib_dir` — the
# versioned-object glob at the heart of that copy stages by STEM, and a
# soname whose shape defeats it (or any future regression that reintroduces
# the same shape) can resolve cleanly while `$lib_dir` quietly stays short a
# file the derivation itself claims it satisfied.
#
# Arguments: `lib_dir`, then every soname to require, one per remaining
# argument. Fails (exit 1), naming every soname with no regular file of that
# exact name under `lib_dir`.
bundle_assert_staged() {
  local lib_dir="$1"
  shift
  local soname missing=""
  for soname in "$@"; do
    if [ ! -f "${lib_dir}/${soname}" ]; then
      missing="${missing} ${soname}"
    fi
  done
  if [ -n "$missing" ]; then
    echo "::error::bundle_cuda_libs.sh: the stage directory does not carry a regular file for:${missing} — the derivation resolved them but no same-named file exists under ${lib_dir} after the copies." >&2
    return 1
  fi
}

bundle_main() {
  if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "usage: bundle_cuda_libs.sh <binary> <stage-lib-dir> [search-path]" >&2
    return 2
  fi
  local binary="$1"
  local lib_dir="$2"
  local search_path="${3:-$BUNDLE_DEFAULT_SEARCH_PATH}"
  local needed sources src closure soname dir
  local required=()

  mkdir -p "$lib_dir"
  needed="$(bundle_needed_sonames "$binary")"
  # Deliberately unquoted: the binary's DT_NEEDED list is one soname per line
  # and is passed as separate arguments (sonames contain no whitespace).
  # shellcheck disable=SC2046
  sources="$(bundle_copy_sources "$search_path" $needed)"
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

  bundle_stage_floor "$search_path" "$lib_dir"

  # shellcheck disable=SC2046
  closure="$(bundle_resolve_closure "$search_path" $needed)"
  while IFS=' ' read -r soname dir; do
    [ -n "$soname" ] || continue
    required[${#required[@]}]="$soname"
  done <<EOF
$closure
EOF
  # `${required[@]}` on a zero-element array is an unbound-variable error
  # under `set -u` on bash 3.2 (macOS's shipped bash, which the hermetic
  # suite is required to run under) even though bash 4+ tolerates it; `sources`
  # being non-empty above already guarantees at least one closure member, but
  # the guard costs nothing and keeps this call portable regardless.
  if [ "${#required[@]}" -gt 0 ]; then
    bundle_assert_staged "$lib_dir" "${required[@]}"
  else
    bundle_assert_staged "$lib_dir"
  fi
  echo "bundle_cuda_libs.sh: every non-host-provided soname of the DT_NEEDED closure, plus the floor, is staged under ${lib_dir} as a regular file of its own SONAME (checked by filesystem presence, not by asking the runtime loader — see the module doc for what that leaves uncovered)."
}

# Sourced by the suite (which needs the functions and not the side effects);
# executed by the release lane.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  bundle_main "$@"
fi
