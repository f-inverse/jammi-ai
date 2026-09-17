#!/usr/bin/env bash
# Stage the shared libraries the CUDA (`cu12`) `jammi-server` tarball must
# carry, DERIVED from the binary's own dynamic-link requirements rather than
# from a hand-kept list of names, and VERIFY at run time (not merely by
# filesystem presence) that the real dynamic loader resolves the bundled
# entries from the stage directory rather than from a copy the build host
# happens to carry too.
#
# The tarball ships a launcher that puts `lib/` on `LD_LIBRARY_PATH` and execs
# the binary, so the tarball is only self-sufficient if `lib/` holds every
# SONAME the loader will look for on a host that has an NVIDIA driver and
# nothing else. `release-binaries.yml`'s CUDA leg used to state that set as a
# HAND LIST of seven literal names (`libcudart libcublas libcublasLt libcurand
# libnvrtc libnvrtc-builtins libnccl`) after a first derivation attempt (#535)
# and its loader-verification arm (#534) were both excised for measured
# defects — recorded here, fixed here, never re-derived from intent (see
# `ci/scripts/test_bundle_cuda_libs.sh`'s module doc for the fixture that
# reproduces each one). A literal list is a copy of a fact that lives in the
# binary, and it drifts the moment a feature adds a link: `jammi-ai`'s `cuda`
# feature includes `candle-core/nccl`, cudarc's build script emits
# `cargo:rustc-link-lib=dylib=nccl` for it, and the resulting binary carries a
# `DT_NEEDED libnccl.so.2` that no name on a stale list would cover — a green
# build shipping a tarball that cannot `exec` on a driver-only host, discovered
# by a user rather than by CI. Reading the requirement off the binary removes
# the copy; every future link is carried automatically, and one that cannot be
# satisfied is a loud failure here instead of a silent omission.
#
# Three mechanisms, kept separate on purpose because they answer three
# different questions and none can stand in for another:
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
#      the fixed fact a `DT_NEEDED` walk is structurally blind to. The floor
#      may never silently overwrite a destination filename the derivation
#      already staged from a DIFFERENT source object — see `bundle_stage_
#      floor`'s own doc for how that is keyed (destination basename + real
#      source identity, never which directory either side started in).
#   3. LOADER VERIFICATION (`bundle_verify_loader_resolution`): everything
#      above is filesystem-only — it proves a same-named file exists, never
#      that the REAL loader, at run time, resolves each bundled entry FROM
#      the stage directory rather than from a copy the runtime host happens
#      to carry too. See that function's own doc for the four measured
#      defects (#534) its predecessor had and how this shape avoids each one.
#
# Before any of the above: `bundle_assert_no_runpath` checks the binary
# carries no `DT_RPATH`/`DT_RUNPATH` dynamic-section entry — either would let
# the loader resolve a bundled library from wherever it points regardless of
# `LD_LIBRARY_PATH`, which (3) below assumes cannot happen. Checked once, up
# front, by `bundle_main`, before any staging begins.
#
# After (1) and (2), a post-copy STAGE-SET ASSERTION (`bundle_assert_staged`)
# checks — pure filesystem, no ELF reader — that every soname the closure
# named exists as a regular file of exactly that SONAME under `$lib_dir`. This
# is not the same claim as "the derivation resolved it": the copy step
# (`bundle_copy_sources`) stages by matching a glob against the resolved
# soname's STEM, and a soname whose shape defeats that glob (an unversioned
# `DT_NEEDED libfoo.so`, whose own name has no trailing `.` for the glob to
# match — closed below, but a FUTURE regression in the same shape is exactly
# what this assertion is insurance against) can resolve cleanly and still
# leave `$lib_dir` without the file the derivation claims it satisfied. The
# floor's own refusal (naming a stem no search directory holds at all) plays
# the same role for the fixed seven. `bundle_main` treats every phase's
# non-zero exit as its own failure — see its own doc for why that must be
# true of EVERY phase, not just the last one checked.
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
# the derivation and floor testable off a Linux host at all: `ci/scripts/
# test_bundle_cuda_libs.sh` sources this file, replaces that one function with
# a fixture, and drives the rest over a fake library tree. The loader-
# verification functions are pure TEXT parsing (no `ldd`, no ELF, no
# subprocess) for the same reason — see their own doc. The workflow leg that
# calls this script runs only on a `v*` tag — without this suite the
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

# A portable `realpath`: neither GNU `realpath` nor `readlink -f` is
# guaranteed present (macOS's shipped `readlink` has no `-f`), but `python3`
# is already a hard dependency of this repo's CI scripts (the workflow YAML
# parse this suite runs, the `cu12` wheel's own component-contract check).
# Used only to compare SOURCE OBJECTS by identity for the floor's collision
# check (`bundle_stage_floor`) — never to resolve a soname.
bundle_realpath() {
  python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1"
}

# The one function that reads an ELF file, and therefore the one the hermetic
# suite replaces. Prints the file's `DT_NEEDED` sonames, one per line, in link
# order; prints nothing for a file with none.
bundle_needed_sonames() {
  # `readelf -d` lines read: `0x... (NEEDED)  Shared library: [libcudart.so.12]`
  readelf -d "$1" | sed -n 's/.*(NEEDED).*\[\(.*\)\].*/\1/p'
}

# The other ELF-reading function — the hermetic suite replaces this one too,
# the same way it replaces `bundle_needed_sonames`. Prints the raw `readelf
# -d` dynamic-section text verbatim.
bundle_dynamic_section() {
  readelf -d "$1" 2>&1
}

# `LD_LIBRARY_PATH` is the loader's LAST search step (after `DT_RPATH`,
# `LD_PRELOAD`, the cache, and `DT_RUNPATH` sits between the cache and the
# default path — https://man7.org/linux/man-pages/man8/ld.so.8.html "Search
# order"). A `DT_RPATH`/`DT_RUNPATH` entry baked into the binary itself could
# point at the build host's own toolkit directory, in which case the loader
# resolves a bundled library from THERE regardless of what `LD_LIBRARY_PATH`
# says — the launcher's whole `LD_LIBRARY_PATH="${here}/lib:..."` strategy,
# and every claim `bundle_verify_loader_resolution` makes about "resolved
# from `lib_dir`", assumes no such entry exists. Checked once, up front, so
# a build that somehow acquires one fails loudly here instead of silently
# defeating the launcher on a host whose directory layout happens to differ
# from the build host's.
bundle_assert_no_runpath() {
  local binary="$1"
  local out
  out="$(bundle_dynamic_section "$binary")"
  if printf '%s\n' "$out" | grep -Eq '\((RPATH|RUNPATH)\)'; then
    echo "::error::bundle_cuda_libs.sh: ${binary} carries an RPATH/RUNPATH dynamic-section entry — the loader would consult it ahead of (RPATH) or interleaved with (RUNPATH) LD_LIBRARY_PATH, which defeats this arm's whole 'resolved from lib_dir' argument:" >&2
    printf '%s\n' "$out" | grep -E '\((RPATH|RUNPATH)\)' >&2
    return 1
  fi
  return 0
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

# Every regular file in `dir` that is stem `stem`'s own object: every
# versioned sibling (`$stem.so.*`, the glob the derivation's copy step
# already used) PLUS the exact UNVERSIONED object `$stem.so` itself when it
# exists — a fallback the versioned glob alone cannot provide, because
# `$stem.so` (nothing after `.so`) never matches a pattern that demands a
# literal `.` immediately after it. `bundle_copy_sources` already applies
# this same fallback to a resolved soname (see its own doc for why); this is
# the shared one place both the floor's SEARCH (`bundle_resolve_stem_dir`)
# and its COPY step draw from, so there is exactly one definition of "this
# stem's objects in this directory" for the floor to ever disagree with
# itself about (advisory carried from #535: the floor's resolvers used to
# lack this fallback entirely).
bundle_stem_objects() {
  local dir="$1"
  local stem="$2"
  local so
  for so in "$dir/$stem.so".*; do
    [ -e "$so" ] && printf '%s\n' "$so"
  done
  if [ -e "$dir/$stem.so" ]; then
    printf '%s\n' "$dir/$stem.so"
  fi
}

# The directory of the first entry in `search_path` that holds ANY object of
# `bundle_stem_objects` — used only by the floor, which starts from a bare
# STEM (it has no single soname string to look for: `libnvrtc-builtins` ships
# as `libnvrtc-builtins.so.12.6` on this toolkit and could ship as any other
# version, or even unversioned), not from a resolved `DT_NEEDED` entry.
bundle_resolve_stem_dir() {
  local stem="$1"
  local search_path="$2"
  local dir
  local IFS=:
  for dir in $search_path; do
    [ -n "$dir" ] || continue
    if [ -n "$(bundle_stem_objects "$dir" "$stem")" ]; then
      printf '%s\n' "$dir"
      return 0
    fi
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
# derivation resolves anything (first search-path match wins, every object
# `bundle_stem_objects` names for the stem is copied) — but starting from a
# fixed STEM, never from a `DT_NEEDED` entry, which is exactly the point for
# `libnvrtc-builtins`: measurement (the module doc; `readelf -d` against the
# real toolkit's `libnvrtc.so.12`) shows NVRTC never names it as a link-time
# dependency at all, so no closure walk, however correct, can ever reach it —
# this is the only mechanism that stages it. A stem no search directory holds
# at all is a named FAIL: the same refusal the derivation raises, for the one
# set this script does not trust `DT_NEEDED` to find.
#
# `closure_sources` (optional): the EXACT list of absolute source file paths
# `bundle_copy_sources` already resolved for the derivation, one per line —
# the same value `bundle_main` stages under `$lib_dir` before ever calling
# this function. The floor refuses to stage any object whose DESTINATION
# BASENAME (the filename it would land at under `$lib_dir`) the derivation
# ALREADY staged FROM A DIFFERENT SOURCE OBJECT — compared by realpath
# (`bundle_realpath`), never by which directory either path started in: the
# SAME object reached through two different search-path entries (a symlink
# bridging directories, or a duplicate listing of one real file) is not a
# conflict and is simply skipped rather than copied twice; only two
# DIFFERENT objects racing for the same destination filename refuses. #535's
# advisory named this hole; every floor candidate is checked against the
# derivation's own resolved sources BEFORE any file moves, so a collision
# refuses instead of racing whichever copy ran last.
bundle_stage_floor() {
  local search_path="$1"
  local lib_dir="$2"
  local closure_sources="${3:-}"
  local stem dir so so_base so_real closure_real
  local missing="" conflicts="" to_copy=""

  for stem in $BUNDLE_FLOOR_STEMS; do
    if ! dir="$(bundle_resolve_stem_dir "$stem" "$search_path")"; then
      missing="${missing} ${stem}"
      continue
    fi
    while IFS= read -r so; do
      [ -n "$so" ] || continue
      so_base="$(basename "$so")"
      if [ -n "$closure_sources" ] && closure_real="$(bundle_closure_realpath_for_basename "$closure_sources" "$so_base")"; then
        so_real="$(bundle_realpath "$so")"
        if [ "$closure_real" != "$so_real" ]; then
          conflicts="${conflicts}
  ${so_base}: the floor would stage ${so} (real: ${so_real}), the derivation already staged a DIFFERENT object at that destination name (real: ${closure_real})"
        fi
        # Same object either way (same realpath) — already covered by the
        # derivation's own copy; staging it again is redundant, not wrong,
        # so it is simply skipped rather than copied a second time.
        continue
      fi
      to_copy="${to_copy}${so}"$'\n'
    done <<EOF
$(bundle_stem_objects "$dir" "$stem")
EOF
  done

  if [ -n "$conflicts" ]; then
    echo "::error::bundle_cuda_libs.sh: the floor would write a destination filename the derivation already staged from a DIFFERENT source object (never silently overwritten):${conflicts}" >&2
    return 1
  fi
  if [ -n "$missing" ]; then
    echo "::error::bundle_cuda_libs.sh: the floor library set is missing from every search directory in '${search_path}':${missing} — this tarball has shipped these seven stems since before this derivation existed; libnvrtc-builtins in particular is dlopen'd by libnvrtc rather than linked (measured: readelf -d against the CUDA 12.6 toolkit's libnvrtc.so.12 names no such NEEDED entry), so no DT_NEEDED closure can ever be trusted to reach it, and this is the mechanism that stages it regardless." >&2
    return 1
  fi

  while IFS= read -r so; do
    [ -n "$so" ] || continue
    cp -L "$so" "$lib_dir/"
  done <<EOF
$to_copy
EOF
}

# Returns (stdout) the realpath of the `closure_sources` entry (one absolute
# path per line, the same shape `bundle_copy_sources` returns) whose basename
# matches `wanted_base`, or a non-zero exit if none does. The one place that
# answers "did the derivation already claim this destination filename, and
# from what real object" — `bundle_stage_floor`'s only caller.
bundle_closure_realpath_for_basename() {
  local closure_sources="$1"
  local wanted_base="$2"
  local src
  while IFS= read -r src; do
    [ -n "$src" ] || continue
    if [ "$(basename "$src")" = "$wanted_base" ]; then
      bundle_realpath "$src"
      return 0
    fi
  done <<EOF
$closure_sources
EOF
  return 1
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

# ---------------------------------------------------------------------------
# Loader verification (#534): everything above proves a same-named FILE
# exists under `$lib_dir`. None of it proves that the REAL dynamic loader, at
# run time, resolves a bundled entry FROM `$lib_dir` rather than from a copy
# the build/CI host happens to carry too — a distinction that matters because
# the tarball's own launcher PREPENDS `lib/` to `LD_LIBRARY_PATH` rather than
# restricting resolution to it, so a "clean" `ldd` run on a host that itself
# carries a system copy of, say, `libnccl` proves nothing about a driver-only
# host that carries no such copy.
#
# A prior revision of this arm (`bundle_verify_stage` /
# `bundle_unresolved_from_loader_output`, via
# `LD_LIBRARY_PATH=<lib> ldd <binary>`) was excised (#534) after four measured
# defects made it UNSOUND rather than merely incomplete — reproduced as
# fixtures in `ci/scripts/test_bundle_cuda_libs.sh`:
#
#   (1) LD_LIBRARY_PATH PREPENDS, it does not restrict — a soname the tarball
#       never staged can still resolve `Ok` if the host happens to carry a
#       copy too. The property this arm asserts is "no host copy outside
#       `<lib>` satisfies a bundle-able member" — a chroot/`unshare`d mount
#       that hides host copies from the loader entirely is ONE mechanism for
#       that property, not the property itself, so this arm ships in TWO
#       homes rather than picking one:
#         (1a) THE RELEASE LANE (`release-binaries.yml`'s `server-cu12-build`
#              job, in the CUDA container, as root) runs the REAL loader
#              (`LD_LIBRARY_PATH=<lib> ldd <binary>`) against the REAL staged
#              binary and pipes that real report through THIS SAME parser —
#              `bundle_verify_loader_resolution`, the `case "$resolved_path"
#              in "$lib_dir"/*)` branch, refuses any non-platform, non-driver
#              member resolved from outside `<lib>`. An `unshare --mount`
#              chroot around that real `ldd` call would tighten this further
#              (hiding host copies rather than merely detecting one that
#              WOULD have been used) but is not available today —
#              `server-cu12-build` runs in `resolve-base.image_cuda` with no
#              `--privileged`/`--cap-add` (confirmed by reading the job) —
#              and remains OPEN as the chroot half of #534 until a runner
#              grants that privilege; STOP RULE INVOKED for that mechanism
#              only, never for the refuse-outside-`<lib>` property.
#         (1b) THE HERMETIC SUITE (`ci/scripts/test_bundle_cuda_libs.sh`, on
#              the bare-runner Guard matrix and the lead's macOS merge path)
#              parses COMMITTED fixture report TEXT — including the real
#              report (1a) itself captures and uploads as a workflow
#              artifact, see `ci/scripts/fixtures/cu12_loader_report_real.txt`
#              — through this exact same parser, with no `ldd`, no ELF, no
#              network, anywhere in this file's test.
#   (2) VACUOUS PASS on a crashed/empty/vdso-only report — closed by requiring
#       the report to carry a RESOLVED line for every one of the binary's own
#       `DT_NEEDED` names (platform members included): an empty, "not a
#       dynamic executable", or `linux-vdso.so.1`-only report contains none
#       of them and fails on that rule alone, no special-casing needed.
#   (3) THE LOADER'S OWN NO-`=>` LINE (naming itself) used to be skipped as
#       "nothing to check", which flipped a CORRECT stage into a false
#       failure once rule (2) demanded every platform member be present too.
#       `bundle_parse_loader_report` treats any report line with no `=>` as a
#       RESOLVED entry for the basename of its path.
#   (4) `lib_dir='/'` — the path-prefix comparison degenerates to "everything
#       is under lib_dir" if lib_dir is the filesystem root; handled by an
#       explicit refusal before the comparison ever runs.
#
# Driver libraries are legitimately `not found` on a driver-less CI host and
# are tolerated — classified the SAME way everything else in this file
# classifies a soname: `bundle_is_driver_soname`/`bundle_is_platform_soname`,
# the ONE partition this script keeps (see their own doc, above) and the ONE
# `ci/scripts/test_cu12_component_contract.py` binds against
# `packaging/server-cu12/verify_link_set.py`'s `PLATFORM`/`DRIVER_PROVIDED`/
# `COVERED` sets — never a second, parallel enumeration restated here for the
# loader arm alone, which would be exactly the kind of copy this whole unit
# exists to remove (the tarball's `BUNDLE_FLOOR_STEMS` HAND list was one; a
# second hand list here would be another).
#
# Parses one `ld.so --list`/`ldd`-shaped report, read from stdin. Prints one
# line per entry: `<soname> RESOLVED <path>` or `<soname> NOTFOUND`. Three
# line shapes:
#   `soname => /path/to/soname (0x...)`   the ordinary resolved shape.
#   `soname => not found`                 unresolved.
#   `/path/to/soname (0x...)`             NO `=>` at all — the loader naming
#     itself (defect 3). The soname is the basename of the path.
# A blank line (leading/trailing whitespace only) is skipped.
bundle_parse_loader_report() {
  local line soname rest path
  while read -r line; do
    [ -n "$line" ] || continue
    case "$line" in
      *" => "*)
        soname="${line%% => *}"
        rest="${line#*" => "}"
        case "$rest" in
          "not found"*) printf '%s NOTFOUND\n' "$soname" ;;
          *)
            path="${rest%% (*}"
            printf '%s RESOLVED %s\n' "$soname" "$path"
            ;;
        esac
        ;;
      *)
        path="${line%% (*}"
        soname="$(basename "$path")"
        printf '%s RESOLVED %s\n' "$soname" "$path"
        ;;
    esac
  done
}

# Verifies a captured loader report against the binary's own `DT_NEEDED`
# names. Arguments: `lib_dir` (the tarball's stage directory), `report` (the
# captured text, one loader line per line), then the binary's full
# `DT_NEEDED` sonames (platform members included) as the remaining arguments.
# See the section doc above for the four defects this shape closes.
bundle_verify_loader_resolution() {
  local lib_dir="$1"
  local report="$2"
  shift 2
  local needed=("$@")

  # Defect (4): lib_dir='/' must be refused explicitly, before the
  # prefix-comparison below ever runs — for lib_dir='/' every absolute path
  # is (falsely) "under" it, which would silently pass defect (1) instead of
  # catching it.
  case "$lib_dir" in
    /)
      echo "::error::bundle_cuda_libs.sh: loader verification refuses lib_dir='/' — a filesystem-root stage directory makes 'resolved under lib_dir' true of every absolute path, which would silently defeat the host-copy check this arm exists to run." >&2
      return 1
      ;;
  esac
  lib_dir="${lib_dir%/}"

  local parsed
  parsed="$(printf '%s\n' "$report" | bundle_parse_loader_report)"

  local name found entry_soname entry_state entry_path resolved_path
  local absent="" violations=""
  for name in "${needed[@]}"; do
    found=0
    resolved_path=""
    while IFS=' ' read -r entry_soname entry_state entry_path; do
      [ -n "$entry_soname" ] || continue
      if [ "$entry_soname" = "$name" ] && [ "$entry_state" = "RESOLVED" ]; then
        found=1
        resolved_path="$entry_path"
      fi
    done <<EOF
$parsed
EOF
    if [ "$found" -eq 0 ]; then
      if bundle_is_driver_soname "$name"; then
        continue
      fi
      absent="${absent} ${name}"
      continue
    fi
    if ! bundle_is_platform_soname "$name" && ! bundle_is_driver_soname "$name"; then
      case "$resolved_path" in
        "$lib_dir"/*) : ;;
        *) violations="${violations}
  ${name} resolved from ${resolved_path}, not ${lib_dir}" ;;
      esac
    fi
  done

  if [ -n "$absent" ]; then
    echo "::error::bundle_cuda_libs.sh: the loader report names no resolved entry for:${absent} — an empty, crashed, or vacuous report must fail this arm, never pass it." >&2
    return 1
  fi
  if [ -n "$violations" ]; then
    echo "::error::bundle_cuda_libs.sh: the loader resolved a bundled library from OUTSIDE the stage directory — this proves a build-host copy satisfied it, not the tarball's own:${violations}" >&2
    return 1
  fi
  return 0
}

bundle_main() {
  if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "usage: bundle_cuda_libs.sh <binary> <stage-lib-dir> [search-path]" >&2
    return 2
  fi
  local binary="$1"
  local lib_dir="$2"
  local search_path="${3:-$BUNDLE_DEFAULT_SEARCH_PATH}"
  local needed sources src closure closure_rc soname dir
  local required=()

  bundle_assert_no_runpath "$binary" || return 1

  mkdir -p "$lib_dir"
  needed="$(bundle_needed_sonames "$binary")"
  # Deliberately unquoted: the binary's DT_NEEDED list is one soname per line
  # and is passed as separate arguments (sonames contain no whitespace).
  # shellcheck disable=SC2086
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

  # Every phase below returns its own status, and `bundle_main` returns
  # non-zero the moment any of them does (#535's core defect: the retired
  # revision discarded these three return values and printed the success
  # sentence unconditionally, so a tree missing `libnccl` and
  # `libnvrtc-builtins` still reported success under the suite's own
  # `set +e`). Program mode (this file executed, not sourced) does not rely
  # on `errexit` for this either — `errexit` does not survive `source`, which
  # is exactly how the hermetic suite drives this file, so every phase here
  # is checked explicitly regardless of which context calls it.
  # shellcheck disable=SC2086
  closure="$(bundle_resolve_closure "$search_path" $needed)"
  closure_rc=$?
  if [ "$closure_rc" -ne 0 ]; then
    return "$closure_rc"
  fi

  bundle_stage_floor "$search_path" "$lib_dir" "$sources" || return 1

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
    bundle_assert_staged "$lib_dir" "${required[@]}" || return 1
  else
    bundle_assert_staged "$lib_dir" || return 1
  fi
  echo "bundle_cuda_libs.sh: every non-host-provided soname of the DT_NEEDED closure, plus the floor, is staged under ${lib_dir} as a regular file of its own SONAME (checked by filesystem presence; loader verification is a separate, later step — see bundle_verify_loader_resolution)."
}

# Sourced by the suite (which needs the functions and not the side effects);
# executed by the release lane.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  bundle_main "$@"
fi
