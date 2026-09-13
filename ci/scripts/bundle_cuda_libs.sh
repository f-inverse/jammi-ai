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
# Two properties, and they are separate mechanisms on purpose:
#
#   1. DERIVATION (`bundle_copy_sources`): the staged set is the transitive
#      `DT_NEEDED` closure of the binary, minus the host-provided set, each
#      soname resolved to a real file in an explicit, ordered search path.
#      TRANSITIVE, not direct: `libnvrtc.so.12` is what the binary names, but
#      `libnvrtc-builtins.so.12` is what `libnvrtc.so.12` itself names, and the
#      loader needs both. A soname that resolves nowhere fails this script.
#   2. VERIFICATION (`bundle_verify_stage`): after the copies, the real loader
#      is asked to resolve the staged binary's own dependencies
#      (`LD_LIBRARY_PATH=<lib> ldd <binary>`) — and PREPENDS, never RESTRICTS,
#      the loader's search, so `not found` is not the property to check: a
#      soname the tarball never staged still resolves `Ok` here if the build
#      HOST happens to carry it too (a `libnccl` the builder's own
#      `/usr/lib64` ships, say), and that "clean" report ships a tarball that
#      cannot `exec` on a driver-only host with no such copy. What this arm
#      establishes instead: for every `DT_NEEDED` entry that is neither
#      platform- nor driver-provided, the PATH the loader actually resolved it
#      to (not merely whether it resolved) is under `$lib_dir`
#      (`realpath`-normalised, so a symlinked stage dir still compares equal);
#      anything else — resolved from elsewhere, or plainly `not found` — is a
#      named FAIL. This arm depends on none of the derivation's reasoning, so
#      it also catches a closure this script walked wrongly. What it can NOT
#      catch: a `dlopen`-loaded dependency never appears in `ldd`'s output at
#      all — `DT_NEEDED` says nothing about it, and neither does this check;
#      that failure mode is UNCOVERED by this script.
#
#      Two more failure shapes that a bare "no defect line" reading would
#      miss, because neither one is a `not found` or a wrong-directory entry:
#      `ldd` itself can exit non-zero (not a dynamic executable, a crashed
#      loader, a truncated invocation) — that exit status is asserted, never
#      discarded, and a non-zero exit is a named FAIL before the report's text
#      is read at all. Separately, a report that runs cleanly to exit 0 can
#      still be VACUOUS — naming none, or only some, of the sonames the
#      binary's own direct `DT_NEEDED` list carries (a `linux-vdso.so.1`-only
#      report, say) — which a rule that only inspects the lines present would
#      call clean because there is no bad line to find; the report is
#      therefore cross-checked as a SET against `bundle_needed_sonames`'s own
#      output for the binary, and any name that set carries but the report
#      never mentions is a named FAIL.
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
#
# Split into the two kinds separately (rather than one combined predicate)
# because `bundle_verify_stage`'s loader arm treats them differently: BOTH may
# resolve from anywhere the loader finds them (neither is required to be under
# `$lib_dir`), but only the driver half may also come back `not found` — a
# platform library that failed to resolve at all would be a broken host, not a
# tolerated gap.
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

# `realpath`-normalise a directory: the OS's own answer
# (`cd <dir> && pwd -P`) when `dir` exists on THIS filesystem — which resolves
# any symlink in the path, so a symlinked stage dir (a runner whose temp root
# is itself a symlink, e.g. macOS's `/tmp` -> `/private/tmp`) compares equal
# to its target — and the literal string, unchanged, when it does not: the
# hermetic suite's fixture paths (`/usr/lib64/…`, `/stage/lib/…`) are never
# created on disk, and a literal fixture string has no symlink to resolve in
# the first place, so leaving it as given is the correct answer, not a
# fallback that happens to work.
bundle_realpath_dir() {
  local dir="$1" resolved
  if resolved="$(cd -- "$dir" 2>/dev/null && pwd -P)"; then
    printf '%s\n' "$resolved"
  else
    printf '%s\n' "$dir"
  fi
}

# `realpath`-normalise a FILE, following the FILE's own symlink chain, not
# only the directory it sits in. `bundle_realpath_dir` alone is not enough
# here: `cd`+`pwd -P` resolves every symlink in a path it can `cd` into, but
# it can only `cd` into a DIRECTORY, so a soname staged as a symlink pointing
# to a path OUTSIDE `$lib_dir` (`$STAGE/libnccl.so.2 -> $elsewhere/libnccl.
# so.2`) would have its directory component normalise to `$STAGE` — correct,
# and useless, since the question is where the FILE resolves, not where its
# directory entry sits. This walks the symlink chain by hand (`readlink`,
# resolving a relative target against the link's own directory, bailing out
# on a cycle rather than looping forever) until it reaches a non-symlink, then
# hands the final directory to `bundle_realpath_dir` and re-appends the
# basename. Falls back to the literal path, unchanged, when nothing on this
# filesystem answers — same fallback contract as `bundle_realpath_dir`, for
# the same reason (the hermetic suite's fixture paths are strings, never
# files).
bundle_realpath_file() {
  local path="$1" dir target seen=" " real_dir
  while [ -L "$path" ]; do
    case "$seen" in
      *" $path "*)
        printf '%s\n' "$path"
        return 0
        ;;
    esac
    seen="$seen$path "
    dir="$(dirname -- "$path")"
    target="$(readlink -- "$path")"
    case "$target" in
      /*) path="$target" ;;
      *) path="${dir}/${target}" ;;
    esac
  done
  if [ -e "$path" ]; then
    dir="$(dirname -- "$path")"
    real_dir="$(bundle_realpath_dir "$dir")"
    printf '%s/%s\n' "${real_dir%/}" "$(basename -- "$path")"
    return 0
  fi
  printf '%s\n' "$path"
}

# The lines of a loader report that are DEFECTS, given the report and the
# stage `lib_dir` it was generated against. For each `DT_NEEDED` entry the
# report names:
#   * a driver soname (`libcuda.so.*`, `libnvidia-*`) is never a defect,
#     `not found` included — the release job runs on a GPU-less runner with
#     no driver installed, and the driver is deliberately not bundled (see
#     `bundle_is_driver_soname`).
#   * a platform soname (glibc, libstdc++/libgcc, the dynamic loader itself)
#     is never a defect for WHERE it resolved (the host's own copy, wherever
#     that is, is correct), but IS a defect if it comes back `not found` — a
#     platform library missing from the loader's search is a broken host, not
#     a tolerated gap.
#   * everything else — the tarball's own responsibility, the exact set
#     `bundle_copy_sources` stages — is a defect both when `not found` (an
#     `LD_LIBRARY_PATH` prefix cannot invent a file that was never staged) AND
#     when it resolves from anywhere other than under `lib_dir`: `LD_LIBRARY_
#     PATH` PREPENDS to the loader's search, it does not RESTRICT it, so a
#     soname the tarball never staged can still come back resolved — from the
#     builder's own system copy, say — and a rule that only greps for `not
#     found` calls that report clean while shipping a tarball that cannot
#     `exec` on a driver-only host lacking that copy.
# Split out from `bundle_verify_stage` so the rule is exercised by the
# hermetic suite over fixture loader output, on a host that has neither `ldd`
# nor an ELF binary to point it at. Prints nothing when the report is clean;
# otherwise prints one `soname => defect` line per problem, naming both the
# soname and the path (or `not found`, or "missing from loader report") it
# defects on.
#
# The optional third argument is the SET of sonames the report is required to
# name — `bundle_needed_sonames`'s own output for the binary the report was
# generated against, one per line. Without it (the two-argument form) this
# function checks only the lines the report actually contains, which is
# exactly the shape that calls a VACUOUS report clean: an empty string, "not a
# dynamic executable", or a `linux-vdso.so.1`-only report all contain zero bad
# lines, because they contain no lines about any staged dependency at all.
# With the third argument, every name in that set is required to appear as
# SOME entry in the report (resolved, not-found, or otherwise) — `derived -
# reported = ∅`, checked as a set, not by scanning for one known-bad shape —
# and any name the set carries but the report never mentions is its own named
# defect line. `bundle_verify_stage` always supplies it; the suite's direct
# calls that omit it are testing the WHERE-resolved rule in isolation.
bundle_unresolved_from_loader_output() {
  local loader_output="$1"
  local lib_dir="$2"
  local derived_sonames="${3:-}"
  local real_lib_dir line soname rest resolved_path real_resolved_path real_resolved_dir
  local defect="" reported=" " needed
  real_lib_dir="$(bundle_realpath_dir "$lib_dir")"
  real_lib_dir="${real_lib_dir%/}"

  while IFS= read -r line; do
    case "$line" in
      *'=>'*) : ;;
      # No `=>` at all: the vdso pseudo-entry, or the loader naming itself by
      # full path. Neither is a staged dependency; nothing to check.
      *) continue ;;
    esac
    soname="$(printf '%s' "${line%%=>*}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    rest="$(printf '%s' "${line#*=>}" | sed -e 's/^[[:space:]]*//')"
    reported="${reported}${soname} "

    if [ "$rest" = "not found" ]; then
      if bundle_is_driver_soname "$soname"; then
        continue
      fi
      defect="${defect}${soname} => not found
"
      continue
    fi

    if bundle_is_driver_soname "$soname" || bundle_is_platform_soname "$soname"; then
      continue
    fi

    # Strip the trailing ` (0xADDRESS)` the loader appends, leaving the path,
    # then resolve the FILE's own symlink chain — not merely the directory it
    # sits in — before comparing against `$lib_dir` (`bundle_realpath_file`).
    resolved_path="${rest% (*}"
    real_resolved_path="$(bundle_realpath_file "$resolved_path")"
    real_resolved_dir="$(dirname -- "$real_resolved_path")"
    real_resolved_dir="${real_resolved_dir%/}"
    case "$real_resolved_dir" in
      "$real_lib_dir" | "$real_lib_dir"/*) continue ;;
    esac
    defect="${defect}${soname} => ${resolved_path} (resolved outside ${lib_dir})
"
  done <<EOF
$loader_output
EOF

  if [ -n "$derived_sonames" ]; then
    while IFS= read -r needed; do
      [ -n "$needed" ] || continue
      case "$reported" in
        *" $needed "*) continue ;;
      esac
      defect="${defect}${needed} => missing from loader report
"
    done <<EOF
$derived_sonames
EOF
  fi

  printf '%s' "$defect"
}

# Ask the real loader whether the staged tree satisfies the staged binary, and
# fail on any defect `bundle_unresolved_from_loader_output` names.
#
# The tool's own exit status is asserted BEFORE its output is read at all — a
# swallowed `|| true` here would read a non-zero `ldd` (a crash, "not a
# dynamic executable", a truncated invocation) as an empty, and therefore
# clean, report. It is also given the binary's own direct `DT_NEEDED` list, so
# `bundle_unresolved_from_loader_output`'s set cross-check can catch a report
# that exits 0 but is otherwise vacuous.
bundle_verify_stage() {
  local binary="$1"
  local lib_dir="$2"
  local loader_output unresolved needed
  local ldd_rc=0
  loader_output="$(LD_LIBRARY_PATH="$lib_dir" ldd "$binary" 2>&1)" || ldd_rc=$?
  if [ "$ldd_rc" -ne 0 ]; then
    echo "::error::bundle_cuda_libs.sh: ldd exited ${ldd_rc} against ${binary} — the loader could not even be asked whether the stage satisfies the binary, which is not a clean report:" >&2
    printf '%s\n' "$loader_output" >&2
    return 1
  fi
  needed="$(bundle_needed_sonames "$binary")"
  unresolved="$(bundle_unresolved_from_loader_output "$loader_output" "$lib_dir" "$needed")"
  if [ -n "$unresolved" ]; then
    echo "::error::bundle_cuda_libs.sh: the staged tarball does not satisfy its own binary — a bundled dependency did not resolve, resolved from somewhere other than ${lib_dir}, or the loader report never named it at all:" >&2
    printf '%s\n' "$unresolved" >&2
    echo "Full loader output:" >&2
    printf '%s\n' "$loader_output" >&2
    return 1
  fi
  printf '%s\n' "$loader_output"
  echo "bundle_cuda_libs.sh: every non-platform, non-driver DT_NEEDED entry resolved under ${lib_dir} (checked by path, not merely by presence; driver and platform libraries excepted), and the loader named every entry the binary directly needs."
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

  bundle_verify_stage "$binary" "$lib_dir"
}

# Sourced by the suite (which needs the functions and not the side effects);
# executed by the release lane.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  bundle_main "$@"
fi
