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
# nothing else. That set is DERIVED, not a hand list of literal names (see
# `ci/scripts/test_bundle_cuda_libs.sh`'s module doc for the fixture that
# reproduces each defect the design below avoids). A literal list is a copy
# of a fact that lives in the
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
#   2. THE FLOOR (`bundle_stage_floor`): seven stems this tarball always
#      carries — `libcudart libcublas
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
#      defects a plain `ldd` check has and how this shape avoids each one.
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
    *)
      bundle_is_loader_soname "$1"
      ;;
  esac
}

# The dynamic loader itself, named as a `DT_NEEDED` entry (`ld-linux-*`) — a
# subset of "platform", broken out on its own because the jail arm
# (`bundle_verify_jail_report`) treats it differently from every other
# platform member: it is the process `ci/scripts/jail_trace.py` execve's to
# RUN the tolerant `LD_TRACE_LOADED_OBJECTS` trace, placed at and invoked
# from its own `PT_INTERP` path (`bundle_binary_interp`) rather than staged
# alongside the other platform copies under `/platform` — so its self-
# reported entry is PINNED to that exact path (checked for EQUALITY, never
# merely "under some directory" the way every other member's is).
bundle_is_loader_soname() {
  case "$1" in
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

# LEXICAL normalization of an absolute path
# string — collapses `.`/`..` components (`posixpath.normpath`, never
# touches the filesystem, never resolves a symlink) so a value like
# `/lib/../usr/lib/x.so` is judged by where it actually points, never by a
# bare textual prefix match (`case "$p" in "/lib"/*)` matches that literal
# string even though it normalizes to `/usr/lib/x.so`, OUTSIDE the jail's
# `/lib`). Used only by `bundle_verify_jail_report`'s resolved-path checks —
# the jail-report TEXT is judged as fed, so this must stay filesystem-free
# to keep that function hermetically testable off paths that do not exist
# on the host running the suite.
bundle_normalize_path() {
  python3 -c 'import posixpath, sys; print(posixpath.normpath(sys.argv[1]))' "$1"
}

# The device+inode pair for a real file, portable across GNU/BSD `stat`
# flag differences (macOS's shipped `stat` and Linux's disagree on `-f`).
# Used only to compare ON-DISK PROVENANCE (a real hardlink, not merely a
# same-looking name) between two files — never a soname classification.
bundle_stat_ino() {
  python3 -c 'import os, sys; st = os.stat(sys.argv[1]); print(f"{st.st_dev}:{st.st_ino}")' "$1"
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

# The third and last ELF-reading function, used only by the jail arm
# (`bundle_build_jail`): the binary's `PT_INTERP` path (`readelf -l`'s
# "Requesting program interpreter" line) — `bundle_build_jail` copies the
# loader into the jail AT this exact path (e.g. `/lib64/ld-linux-x86-64.so.2`,
# never an arbitrary name like `/ld.so`) and `ci/scripts/jail_trace.py`
# `os.execve`s it there directly (invoked from any OTHER path,
# the loader's own self-reported trace line gains a `=>` — `<true-soname> =>
# <invoked-path>` — that `bundle_verify_jail_report` correctly reads as "not
# at its own PT_INTERP path" and refuses; only invoked from its real
# `PT_INTERP` path does the self line degenerate to the bare, no-`=>` shape).
# The binary itself is never exec'd directly inside the jail — its own
# baked-in `PT_INTERP` reference is resolved by the KERNEL at exec time via
# the normal `/lib64/...` lookup, which is exactly why the loader must
# actually BE there rather than merely present under some other name.
bundle_binary_interp() {
  readelf -l "$1" | sed -n 's/.*Requesting program interpreter: \(.*\)\]$/\1/p'
}

# The dynamic string token `$ORIGIN` (glibc
# `ld.so(8)`, "Dynamic string tokens") resolves, AT RUNTIME, to the
# directory the ACTUAL LOADED OBJECT itself lives in — never a build-host
# path baked in at link time. An RPATH/RUNPATH VALUE composed of one or
# more colon-separated components that are EACH exactly `$ORIGIN` (or the
# brace-quoted `${ORIGIN}`) therefore always resolves INSIDE whatever
# directory currently holds the object — the stage's own `<lib>/` today,
# the jail's own `/lib` after `bundle_build_jail` copies it there — which
# REINFORCES this arm's "resolved from lib_dir" argument rather than
# defeating it. NVIDIA ships `libcublas`, `libcublasLt`, and `libcurand`
# (at least) in the CUDA 12.6 toolkit with exactly `RUNPATH: [$ORIGIN]` —
# refusing every RPATH/RUNPATH by presence alone would fail every real
# cu12 release, not only a genuine build-time-path leak; the domain this
# function checks is the RPATH/RUNPATH's OWN component shape, never its
# mere presence. A COMPONENT other than the bare `$ORIGIN`/`${ORIGIN}`
# token — an absolute
# path, `$ORIGIN` with any suffix (`$ORIGIN/..`, `$ORIGIN/../lib` — these
# escape the object's own directory, which `$ORIGIN` ALONE never does), or
# any component mixed into the same colon-separated value — is refused,
# by component, never by shape-guessing the whole string.
bundle_is_origin_only_rpath() {
  local value="$1"
  local comp
  local IFS=':'
  for comp in $value; do
    # shellcheck disable=SC2016  # deliberately literal -- $ORIGIN is glibc's
    # own dynamic string token, never a shell variable to expand.
    case "$comp" in
      '$ORIGIN' | '${ORIGIN}') : ;;
      *) return 1 ;;
    esac
  done
  return 0
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
# from the build host's. A PURELY `$ORIGIN`-relative entry
# (`bundle_is_origin_only_rpath`) is the one exception — see that
# function's own doc.
bundle_assert_no_runpath() {
  local binary="$1"
  local out
  out="$(bundle_dynamic_section "$binary")"
  local line kind value violations=""
  while IFS= read -r line; do
    case "$line" in
      *'(RPATH)'*) kind="RPATH" ;;
      *'(RUNPATH)'*) kind="RUNPATH" ;;
      *) continue ;;
    esac
    value="$(printf '%s\n' "$line" | sed -n 's/.*\[\(.*\)\]$/\1/p')"
    if ! bundle_is_origin_only_rpath "$value"; then
      violations="${violations}
  (${kind}) [${value}]"
    fi
  done <<EOF
$out
EOF
  if [ -n "$violations" ]; then
    echo "::error::bundle_cuda_libs.sh: ${binary} carries an RPATH/RUNPATH dynamic-section entry with a component other than \$ORIGIN — the loader would consult it ahead of (RPATH) or interleaved with (RUNPATH) LD_LIBRARY_PATH, and a component that is not PURELY \$ORIGIN-relative can point outside the directory the object actually lives in, which defeats this arm's whole 'resolved from lib_dir' argument:${violations}" >&2
    return 1
  fi
  return 0
}

# Extends `bundle_assert_no_runpath`'s guarantee
# from the binary ALONE to every regular file directly under `dir` — a
# STAGED CUDA `.so` (not the binary) carrying its own `DT_RPATH`/`DT_RUNPATH`
# could resolve ITS OWN transitive dependencies from a build-time toolkit
# path regardless of `--library-path`, which is exactly the same hazard
# `bundle_assert_no_runpath` already refuses for the binary, just on a
# different object — including the `$ORIGIN`-only carve-out
# (`bundle_is_origin_only_rpath`), since real CUDA toolkit `.so`s
# (`libcublas`, `libcublasLt`, `libcurand`) carry exactly that RUNPATH and a
# rule refusing it here too would fail every real cu12 release. Checked
# over the jail's OWN `lib_dir` before anything is hardlinked into a jail —
# a toolkit library with a component OTHER than `$ORIGIN` fails loudly here
# rather than silently defeating the "every bundle-able member resolves
# from `/lib`" argument the jail arm exists to prove.
bundle_assert_no_runpath_dir() {
  local dir="$1"
  local f rc=0
  for f in "$dir"/*; do
    [ -f "$f" ] || continue
    if ! bundle_assert_no_runpath "$f"; then
      rc=1
    fi
  done
  return "$rc"
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
# itself about.
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
# DIFFERENT objects racing for the same destination filename refuses. Every
# floor candidate is checked against the
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
# Loader verification: everything above proves a same-named FILE
# exists under `$lib_dir`. None of it proves that the REAL dynamic loader, at
# run time, resolves a bundled entry FROM `$lib_dir` rather than from a copy
# the build/CI host happens to carry too — a distinction that matters because
# the tarball's own launcher PREPENDS `lib/` to `LD_LIBRARY_PATH` rather than
# restricting resolution to it, so a "clean" `ldd` run on a host that itself
# carries a system copy of, say, `libnccl` proves nothing about a driver-only
# host that carries no such copy.
#
# A bare `LD_LIBRARY_PATH=<lib> ldd <binary>` check is UNSOUND rather than
# merely incomplete, for four measured reasons — reproduced as fixtures in
# `ci/scripts/test_bundle_cuda_libs.sh`:
#
#   (1) LD_LIBRARY_PATH PREPENDS, it does not restrict — a soname the tarball
#       never staged can still resolve `Ok` if the host happens to carry a
#       copy too. The property this arm asserts is "no host copy outside
#       `<lib>` satisfies a bundle-able member" — TWO independent mechanisms
#       run in the release lane for it, because DETECTING a host copy that
#       WOULD have been used and HIDING every host copy so none CAN be used
#       are different strengths of the same argument, and neither stands in
#       for the other:
#         (1a) DETECTION, THE RELEASE LANE (`release-binaries.yml`'s
#              `server-cu12-build` job, in the CUDA container, as root) runs
#              the REAL loader (`LD_LIBRARY_PATH=<lib> ldd <binary>`) against
#              the REAL staged binary and pipes that real report through THIS
#              SAME parser — `bundle_verify_loader_resolution`, the
#              `case "$resolved_path" in "$lib_dir"/*)` branch, refuses any
#              non-platform, non-driver member resolved from outside `<lib>`.
#              Runs FIRST, unconditionally, before the jail arm below —
#              nothing here ever falls back to it silently if the jail arm
#              fails; both are required and neither result substitutes for
#              the other's.
#         (1b) THE JAIL, THE RELEASE LANE, arm (1a)'s tightening (the
#              chroot half): rather than merely detecting a host copy that
#              WOULD satisfy a bundled member, this arm HIDES every host
#              copy so none CAN — a `chroot` jail (`bundle_build_jail`)
#              containing NOTHING but the real binary; the staged `<lib>/`
#              HARDLINKED in wholesale (`cp -al` — the stage can be multi-GB,
#              and this REQUIRES the jail and the stage to already sit on
#              the same, container-native filesystem, never the bind-mounted
#              checkout a `container:` job's own workspace actually is);
#              same-named copies of the PLATFORM CLOSURE of the binary AND
#              every staged object (`bundle_jail_platform_closure` — a
#              bundled CUDA library can itself need a platform member the
#              binary never names directly), sourced from arm (1a)'s own
#              already-captured report via `bundle_platform_sources_from_
#              report` — one real loader run is the one fact this script
#              trusts about where the host's own glibc lives, never a second
#              `ldconfig`/`ldd` lookup, copied byte-for-byte into a SEPARATE
#              `/platform` directory — never
#              hardlinked, never written into the `lib/` directory the
#              shipped stage's own inodes live under, so a destination-name
#              collision with the shipped stage's own files is structurally
#              impossible rather than merely unlikely); and the loader
#              itself, copied to and invoked AT its own `PT_INTERP` path
#              (`bundle_binary_interp`) rather than any other name — copied
#              elsewhere, its own self-reported trace line gains a `=>` the
#              shipped parser misreads as "resolved from outside the jail"
#              on an otherwise CORRECT jail. Driver members (`libcuda.so.1`,
#              `libnvidia-*`) are DELIBERATELY ABSENT from the jail.
#              `bundle_assert_jail_file_set` checks
#              the real jail's file set against exactly this plan, wired
#              into the release lane between the build and the trace.
#
#              The report is produced by `ci/scripts/jail_trace.py`, never
#              `ld.so --list`: `--list` is FATAL (exit 127, one error line,
#              no report at all) on the FIRST missing library, and this jail
#              deliberately ships with the driver libraries missing — a
#              mechanism that refuses to produce a report the moment one
#              name is absent cannot ever prove "every driver name is `not
#              found`". `jail_trace.py` instead forks, `os.chroot`s the
#              CHILD into the jail, sets `LD_TRACE_LOADED_OBJECTS=1` in the
#              child's OWN environment strictly AFTER the chroot syscall,
#              then `os.execve`s the loader at its `PT_INTERP` path with
#              `--library-path /lib <binary>` — a TOLERANT trace, `not
#              found` lines and all, exit 0. Setting that environment
#              variable on a `chroot <jail> ...` COMMAND LINE instead (a
#              shell one-liner) would trace `chroot`'s OWN loader ON THE
#              HOST before the `chroot()` syscall ever runs — a silent
#              VACUOUS PASS `jail_trace.py`'s own module doc names as the
#              reason it is a separate Python process rather than a shell
#              invocation. Docker's default capability set includes
#              `CAP_SYS_CHROOT` (probed against `docker run --rm
#              ubuntu:24.04` — no `--privileged`/`--cap-add` needed), so this
#              container needs no extra privilege for `os.chroot` itself to
#              succeed — if it is nonetheless denied (EPERM), `jail_trace.py`
#              exits 2 and the step FAILS naming the missing capability
#              rather than falling back to (1a)'s already-passed result.
#
#              The report is verified by `bundle_verify_jail_report`: a
#              STRICTER rule than (1a)'s in TWO ways. First, inside the jail
#              there is no host copy left to distinguish from a bundled
#              one — EVERY driver name resolving AT ALL is refused (proof
#              the jail failed to exclude it), every OTHER non-loader name
#              must resolve to a NORMALIZED path (`bundle_normalize_path`
#              collapses `..`/`.` components lexically, so a value like
#              `/lib/../usr/lib/x` is judged by where it really points, not
#              by a bare textual prefix) under `/lib`, and the loader's own
#              self-reported entry must resolve to EXACTLY its `PT_INTERP`
#              path — never merely tolerated, since placing/invoking it
#              correctly is itself part of what this arm proves; the
#              `linux-vdso.so.1` kernel-injected entry is an explicit,
#              named carve-out (never a real file, never checked). Second,
#              the quantifier is EVERY LINE the
#              report carries, never only the binary's own direct
#              `DT_NEEDED` names — a member that is only a TRANSITIVE
#              dependency of a STAGED object still appears
#              in a real trace, and judging only the binary's own named set
#              would let a missing or host-leaked TRANSITIVE member
#              (`libm.so.6 => not found`; a driver resolving from
#              `/usr/lib64` when it is not itself a direct `DT_NEEDED`
#              entry) pass silently. This arm's own "bundle-able" quantifier
#              is the binary's `DT_NEEDED` set intersected with the non-host
#              set; `libnvrtc-builtins` (the floor's own dlopen-only member
#              — see the module doc above) is never a `DT_NEEDED` entry,
#              never traced by any loader run, and stays UNPROVEN by this
#              arm, documented rather than silently assumed. The jail
#              report, alongside the SAME lane run's measured `DT_NEEDED`
#              set, is captured and uploaded as a workflow artifact on
#              every run — see `ci/scripts/fixtures/cu12_jail_report_real.txt`.
#         (1c) THE HERMETIC SUITE (`ci/scripts/test_bundle_cuda_libs.sh`, on
#              a guard in `ci/guards.toml`)
#              parses COMMITTED fixture report TEXT for BOTH (1a)'s and
#              (1b)'s rules — including the real reports each arm captures
#              and uploads as a workflow artifact, see
#              `ci/scripts/fixtures/cu12_loader_report_real.txt` and
#              `ci/scripts/fixtures/cu12_jail_report_real.txt` — through the
#              exact same parsers, with no `ldd`, no `chroot`, no ELF, no
#              network, anywhere in this file's test. The jail BUILDER's own
#              file-set rule (`bundle_jail_expected_relpaths` /
#              `bundle_assert_jail_file_set`: the jail contains exactly the
#              staged `<lib>/` UNION the platform copies under the SEPARATE
#              `/platform` directory UNION the loader (at its own
#              `PT_INTERP` path) and the binary, nothing else — no path a
#              host copy could have leaked in at) is likewise exercised as a
#              pure function over a fixture listing, never a real `chroot`,
#              and is WIRED into the release
#              lane itself, between the build and the trace, not merely
#              defined and left uncalled.
#   (2) VACUOUS PASS on a crashed/empty/vdso-only report — closed by requiring
#       the report to carry a RESOLVED line for every one of the binary's own
#       `DT_NEEDED` names (platform members included): an empty, "not a
#       dynamic executable", or `linux-vdso.so.1`-only report contains none
#       of them and fails on that rule alone, no special-casing needed.
#   (3) THE LOADER'S OWN NO-`=>` LINE (naming itself) is NOT skipped as
#       "nothing to check": skipping it flips a CORRECT stage into a false
#       failure once rule (2) demands every platform member be present too.
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

# ---------------------------------------------------------------------------
# The jail arm (the chroot half, arm 1b — see the section doc above).
# `BUNDLE_JAIL_LIB_DIR`/`BUNDLE_JAIL_PLATFORM_DIR` are fixed, never
# parameters, and are TWO SEPARATE directories on purpose: `bundle_build_
# jail` hardlinks the tarball's own `lib_dir` into `BUNDLE_JAIL_LIB_DIR`
# with `cp -al` — sharing INODES with the shipped stage — and NEVER writes
# into that directory again afterward. Every platform member the jail ALSO
# carries (the loader itself excepted — see `bundle_verify_jail_report`'s
# own doc) is copied byte-for-byte (`cp -L`, not hardlinked) into the
# SEPARATE `BUNDLE_JAIL_PLATFORM_DIR`. This is the ISOLATION PROPERTY this
# split states: a write aimed at a platform member's destination path can
# NEVER land on an inode the shipped stage still owns, because the two
# directories never share a namespace to collide in — this does NOT depend
# on the derivation's closure never staging a platform-named file (a fact
# that happens to hold but is never itself enforced): `cp` overwriting an
# existing hardlinked destination IN PLACE would silently corrupt the
# shipped stage's own copy if that fact ever slipped, and keeping platform
# copies in a namespace the hardlinked directory never shares makes that
# corruption structurally impossible regardless. `ci/scripts/jail_trace.py`
# is invoked with `--library-path BUNDLE_JAIL_LIB_DIR:BUNDLE_JAIL_
# PLATFORM_DIR` (glibc's loader searches a colon-separated list in order),
# so both directories are visible to the real trace.
# ---------------------------------------------------------------------------
BUNDLE_JAIL_LIB_DIR="/lib"
BUNDLE_JAIL_PLATFORM_DIR="/platform"

# Pure: given the binary's full `DT_NEEDED` list (platform members
# included), the SUBSET that are platform members (`bundle_is_platform_
# soname`) — a building block `bundle_jail_platform_closure` folds over
# multiple ELF objects; kept as its own pure function so the FILTER itself
# (as opposed to which files it is applied to) stays hermetically testable
# off a plain fixture list.
bundle_jail_platform_basenames() {
  local needed=("$@")
  local n
  for n in "${needed[@]}"; do
    if bundle_is_platform_soname "$n"; then
      printf '%s\n' "$n"
    fi
  done
}

# NOT pure — reads ELF (`bundle_needed_sonames`) over the binary AND every
# already-staged object under `lib_dir`: a
# bundled CUDA library can itself need a platform member the BINARY never
# names directly (e.g. a toolkit `.so` naming `libdl`/`libm`
# entries the binary's own direct `DT_NEEDED` does not) — a jail built only
# from the binary's own platform set can come up short a copy the loader's
# transitive trace legitimately needs. One level over the UNION of (binary,
# every staged file), not a further recursive walk: `bundle_verify_jail_
# report`'s own runtime trace (`LD_TRACE_LOADED_OBJECTS`, which resolves the
# REAL, full transitive graph) is what actually proves the result is
# complete — this function is a generous, cheap over-approximation of what
# to STAGE, never itself the proof. Emits deduplicated platform basenames.
bundle_jail_platform_closure() {
  local binary="$1"
  local lib_dir="$2"
  local all_needed=() n f
  while IFS= read -r n; do
    [ -n "$n" ] || continue
    all_needed[${#all_needed[@]}]="$n"
  done <<EOF
$(bundle_needed_sonames "$binary")
EOF
  for f in "$lib_dir"/*; do
    [ -e "$f" ] || continue
    while IFS= read -r n; do
      [ -n "$n" ] || continue
      all_needed[${#all_needed[@]}]="$n"
    done <<EOF
$(bundle_needed_sonames "$f")
EOF
  done
  bundle_jail_platform_basenames "${all_needed[@]}" | LC_ALL=C sort -u
}

# Pure text function: given arm (1a)'s already-captured loader report (real
# HOST paths, captured before any jail exists) and a list of wanted sonames,
# emits `<soname> <path>` for every one the report RESOLVED. This is the one
# place the jail arm learns where the host's own glibc lives — reusing arm
# (1a)'s real run rather than a second host lookup (`ldconfig -p`, a fresh
# `ldd`) keeps there being exactly one real loader invocation this script
# trusts for host paths (arm 1a's trace, being a REAL `ldd`, already
# resolves the full transitive graph, so every platform member `bundle_
# jail_platform_closure` names is expected to appear here too).
bundle_platform_sources_from_report() {
  local report="$1"
  shift
  local wanted=("$@")
  local parsed name soname state path
  parsed="$(printf '%s\n' "$report" | bundle_parse_loader_report)"
  for name in "${wanted[@]}"; do
    while IFS=' ' read -r soname state path; do
      [ -n "$soname" ] || continue
      if [ "$soname" = "$name" ] && [ "$state" = "RESOLVED" ]; then
        printf '%s %s\n' "$soname" "$path"
      fi
    done <<EOF
$parsed
EOF
  done
}

# Pure: the exact set of paths, relative to the jail root, the jail must
# carry for `bundle_verify_jail_report` to have any chance of passing — the
# SAME set `bundle_assert_jail_file_set` checks the real jail tree against
# after `bundle_build_jail` runs. Arguments, all plain TEXT (this function
# reads no filesystem, no ELF, which is what keeps it hermetic):
#   - `lib_listing`: newline-separated basenames already staged under the
#     tarball's OWN `$lib_dir` (`bundle_main`'s output — `ls`, or a fixture
#     listing in the hermetic suite). Every one is hardlinked into the jail
#     under `BUNDLE_JAIL_LIB_DIR` (`lib/`) wholesale.
#   - `platform_basenames`: newline-separated PLATFORM basenames the jail
#     must ALSO carry a same-named copy of under `BUNDLE_JAIL_PLATFORM_DIR`
#     (`platform/`, a SEPARATE directory from `lib/` — the isolation
#     property the section doc above states) — `bundle_jail_platform_
#     closure`'s output, passed in as plain text so THIS function stays
#     pure.
#   - `interp_relpath`: the loader's own `PT_INTERP` path, WITHOUT the
#     leading `/` (e.g. `lib64/ld-linux-x86-64.so.2`) — the loader must
#     sit at this exact path inside the jail and be invoked there, never an
#     arbitrary name, or its own self-reported trace line gains a `=>` the
#     shipped parser misreads as "resolved from outside" on an otherwise
#     correct jail.
bundle_jail_expected_relpaths() {
  local lib_listing="$1" platform_basenames="$2" interp_relpath="$3"
  local f
  printf '%s\n' "jammi-server" "$interp_relpath"
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    printf 'lib/%s\n' "$f"
  done <<EOF
$lib_listing
EOF
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    printf 'platform/%s\n' "$f"
  done <<EOF
$platform_basenames
EOF
}

# The real builder — filesystem writes only, not hermetically exercised
# directly (it needs a real ELF binary for `bundle_binary_interp` and real
# `cp -al` hardlink semantics); the hermetic suite instead exercises the
# PURE functions above it draws from, plus `bundle_assert_jail_file_set`
# over a fixture tree built by hand.
#
# Arguments: `binary` (the real, already-staged `jammi-server`), `lib_dir`
# (the tarball's OWN stage directory — `bundle_main`'s output, already the
# `DT_NEEDED` closure union the floor), `jail_dir` (created fresh), `report`
# (arm 1a's captured loader report, for the platform members' real host
# paths).
#
# `lib_dir` is hardlinked into the jail with `cp -al` (GNU coreutils;
# this function runs only inside the Linux CUDA container, never the
# hermetic macOS suite) rather than copied byte-for-byte — the stage is
# multi-GB — which REQUIRES `jail_dir` and `lib_dir` to already sit on the
# SAME, container-native filesystem (a hardlink cannot cross a filesystem
# boundary): see this function's caller in `release-binaries.yml` for why
# the stage itself is built under a container-native path (`/root/...`),
# never the bind-mounted checkout a `container:` job's `$GITHUB_WORKSPACE`
# actually is.
bundle_build_jail() {
  local binary="$1" lib_dir="$2" jail_dir="$3" report="$4"

  # A staged CUDA `.so` carrying its own
  # RPATH/RUNPATH would let it resolve ITS OWN dependencies from a
  # build-time toolkit path regardless of `--library-path` — checked BEFORE
  # anything is hardlinked into the jail, so a toolkit library that somehow
  # acquired one fails loudly here rather than silently defeating the
  # "every bundle-able member resolves from `/lib`" argument.
  bundle_assert_no_runpath_dir "$lib_dir" || return 1

  mkdir -p "$jail_dir" "${jail_dir}${BUNDLE_JAIL_PLATFORM_DIR}"
  cp -L "$binary" "${jail_dir}/jammi-server"
  # Hardlinked, sharing inodes with the shipped stage — nothing below
  # this line ever writes into `${jail_dir}${BUNDLE_JAIL_LIB_DIR}` again.
  cp -al "$lib_dir" "${jail_dir}${BUNDLE_JAIL_LIB_DIR}"

  local platform_basenames
  platform_basenames="$(bundle_jail_platform_closure "$binary" "$lib_dir")"

  local platform_sources
  # shellcheck disable=SC2086
  platform_sources="$(bundle_platform_sources_from_report "$report" $platform_basenames)"

  # Platform copies land in the SEPARATE `BUNDLE_JAIL_PLATFORM_DIR`,
  # copied byte-for-byte (`cp -L`, never hardlinked) — this directory shares
  # no inode, and no destination-name collision is even possible, with the
  # hardlinked `BUNDLE_JAIL_LIB_DIR` above.
  local name found src missing_platform=""
  while IFS= read -r name; do
    [ -n "$name" ] || continue
    found=0
    while IFS=' ' read -r n src; do
      [ -n "$n" ] || continue
      if [ "$n" = "$name" ]; then
        cp -L "$src" "${jail_dir}${BUNDLE_JAIL_PLATFORM_DIR}/${name}"
        found=1
      fi
    done <<EOF
$platform_sources
EOF
    if [ "$found" -eq 0 ]; then
      missing_platform="${missing_platform} ${name}"
    fi
  done <<EOF
$platform_basenames
EOF
  if [ -n "$missing_platform" ]; then
    echo "::error::bundle_cuda_libs.sh: arm 1a's loader report has no resolved host path for platform member(s):${missing_platform} — the jail cannot be built without a real copy of each." >&2
    return 1
  fi

  local interp interp_rel
  interp="$(bundle_binary_interp "$binary")"
  if [ -z "$interp" ]; then
    echo "::error::bundle_cuda_libs.sh: ${binary} carries no PT_INTERP — cannot place the loader in the jail at its own path." >&2
    return 1
  fi
  interp_rel="${interp#/}"
  mkdir -p "${jail_dir}/$(dirname "$interp_rel")"
  cp -L "$interp" "${jail_dir}/${interp_rel}"

  bundle_assert_jail_class_provenance "$lib_dir" "$jail_dir" || return 1
}

# The expected-directory-per-CLASS rule everywhere else in this file
# (`bundle_jail_expected_relpaths`, `bundle_verify_jail_report`) is a NAME
# classification (`bundle_is_platform_soname` on the SONAME) — it never
# looks at where a file on disk actually CAME FROM. Two independent
# mechanisms that both encode "platform -> /platform, bundle-able -> /lib"
# could in principle both agree on the same wrong belief. This function
# checks the REAL jail tree against TWO independent facts neither of those
# functions inspects: first, by NAME — a PLATFORM-classified soname found
# under `BUNDLE_JAIL_LIB_DIR` (the bundle-able directory), or a
# non-platform soname found under `BUNDLE_JAIL_PLATFORM_DIR`, is refused
# regardless of anything else; second, by ON-DISK PROVENANCE — `lib_dir` is
# the tarball's own staged directory (the SAME inodes `bundle_build_jail`
# hardlinks into `BUNDLE_JAIL_LIB_DIR`), so a file under the jail's
# `BUNDLE_JAIL_LIB_DIR` must share an inode with some file in `lib_dir`
# (genuinely staged from the tarball, not a host copy that happened to
# land there), and a file under `BUNDLE_JAIL_PLATFORM_DIR` must NOT
# (genuinely copied from the host — `cp -L` — never a hardlink from the
# stage). A platform member's copy found sharing an inode with the stage
# under `BUNDLE_JAIL_LIB_DIR` would mean it was accidentally hardlinked in
# bulk with the tarball's own closure rather than copied individually from
# the host; a bundle-able object's copy found under
# `BUNDLE_JAIL_PLATFORM_DIR` sharing no stage inode would mean the
# platform-copy step wrote something that was never actually resolved
# from the host report at all.
bundle_assert_jail_class_provenance() {
  local lib_dir="$1"
  local jail_dir="$2"
  local f stage_inodes=""
  for f in "$lib_dir"/*; do
    [ -e "$f" ] || continue
    stage_inodes="${stage_inodes}$(bundle_stat_ino "$f")
"
  done

  local base ino violations=""
  for f in "${jail_dir}${BUNDLE_JAIL_LIB_DIR}"/*; do
    [ -e "$f" ] || continue
    base="$(basename "$f")"
    if bundle_is_platform_soname "$base"; then
      violations="${violations}
  ${base}: a PLATFORM member found under ${BUNDLE_JAIL_LIB_DIR} (the bundle-able directory)"
      continue
    fi
    ino="$(bundle_stat_ino "$f")"
    case "$stage_inodes" in
      *"$ino"$'\n'*) : ;;
      *)
        violations="${violations}
  ${base}: under ${BUNDLE_JAIL_LIB_DIR} but its inode matches no file in the staged tarball directory — not actually hardlinked from the stage" ;;
    esac
  done
  for f in "${jail_dir}${BUNDLE_JAIL_PLATFORM_DIR}"/*; do
    [ -e "$f" ] || continue
    base="$(basename "$f")"
    if ! bundle_is_platform_soname "$base"; then
      violations="${violations}
  ${base}: a BUNDLE-ABLE (non-platform) member found under ${BUNDLE_JAIL_PLATFORM_DIR} (the platform directory)"
      continue
    fi
    ino="$(bundle_stat_ino "$f")"
    case "$stage_inodes" in
      *"$ino"$'\n'*)
        violations="${violations}
  ${base}: under ${BUNDLE_JAIL_PLATFORM_DIR} but its inode MATCHES a staged tarball file — accidentally hardlinked from the stage instead of copied from the host" ;;
      *) : ;;
    esac
  done

  if [ -n "$violations" ]; then
    echo "::error::bundle_cuda_libs.sh: a jail file's on-disk provenance does not match its class directory:${violations}" >&2
    return 1
  fi
  return 0
}

# Real filesystem walk of `jail_dir`, compared for EXACT set equality against
# `expected` (paths relative to `jail_dir`, one per remaining argument) — the
# jail BUILDER's own file-set rule (see the section doc above). Anything the
# jail carries that is not on the expected list is refused BY NAME (a host
# file leaking in at a path `bundle_build_jail` never wrote — e.g. an
# absolute-looking `usr/lib/libnccl.so.2` sitting outside `lib/` — is exactly
# the shape a future regression in this builder would take), and anything
# expected but absent is refused the same way `bundle_assert_staged` already
# refuses a missing derivation file.
bundle_assert_jail_file_set() {
  local jail_dir="$1"
  shift
  local expected_nl
  expected_nl="$(printf '%s\n' "$@")"
  local actual_nl
  actual_nl="$(cd "$jail_dir" && find . \( -type f -o -type l \) | sed 's#^\./##')"

  local rel extra="" missing=""
  while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    if ! printf '%s\n' "$expected_nl" | grep -Fxq -- "$rel"; then
      extra="${extra} ${rel}"
    fi
  done <<EOF
$actual_nl
EOF
  while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    if [ ! -e "${jail_dir}/${rel}" ]; then
      missing="${missing} ${rel}"
    fi
  done <<EOF
$expected_nl
EOF

  if [ -n "$extra" ]; then
    echo "::error::bundle_cuda_libs.sh: the jail carries a file the builder never staged, outside the allowed set:${extra} — a host copy or stray path leaking into the jail defeats the whole 'nothing but what we staged' argument this arm exists to prove." >&2
    return 1
  fi
  if [ -n "$missing" ]; then
    echo "::error::bundle_cuda_libs.sh: the jail is missing a file the plan requires:${missing}" >&2
    return 1
  fi
  return 0
}

# Verifies a captured JAIL report (`ci/scripts/jail_trace.py`'s tolerant
# `LD_TRACE_LOADED_OBJECTS` trace, run inside the jail via `os.chroot` —
# never `ld.so --list`, see that script's own doc for why — parsed by the
# SAME `bundle_parse_loader_report` every other arm uses) against the
# binary's own full `DT_NEEDED` names. STRICTER than `bundle_verify_loader_
# resolution`'s rule: inside the jail there is no host copy left standing
# to distinguish a bundled member from — every name resolves from a
# jail-internal path this function pins down exactly, or it is a defect.
# `loader_path` is the loader's OWN `PT_INTERP` path — it must sit there,
# never an arbitrary name, and its self-reported RESOLVED path is checked
# against exactly this value rather than exempted.
#
# The quantifier is EVERY LINE the report carries, not only the names in
# the binary's OWN `DT_NEEDED` list — a member that is only a TRANSITIVE
# dependency of a STAGED object (never the binary's own direct
# `DT_NEEDED`) still appears in a real trace, and an unjudged line is
# exactly how a jail missing that member (`libm.so.6 => not found`, the
# binary genuinely cannot run) or leaking a host path for it (a driver
# resolving from `/usr/lib64` when it is not itself a direct `DT_NEEDED`
# entry) could slip through unjudged. Two passes over the SAME parsed
# report:
#
#   PASS 1 (presence, closes a vacuous pass): every name in the
#   binary's own `DT_NEEDED` set (`needed`, driver members exempted — a
#   driver-less jail may not even print a line for one) must appear
#   SOMEWHERE in the report, in ANY state. An empty, crashed, or
#   `linux-vdso.so.1`-only report names none of them and fails here alone.
#
#   PASS 2 (the positive rule, over EVERY parsed line, not just
#   `needed`): for each entry the report actually carries —
#     - `linux-vdso.so.1` is an explicit, named CARVE-OUT, CONDITIONED
#       exactly like the loader's own carve-out just below, never matched
#       on the soname alone — matching on the name alone would let three
#       spoofed shapes through unexamined: `linux-vdso.so.1 =>
#       /usr/lib/evil.so`, `linux-vdso.so.1 => not found`, and
#       `linux-vdso.so.1 => linux-vdso.so.1 (0x1)` — the third parses to
#       the SAME (RESOLVED, "linux-vdso.so.1") pair the genuine no-`=>`
#       self-line produces, so it is detected separately, off the RAW
#       report text, never through the already-parsed triple alone.
#       Exempt ONLY for the EXACT synthetic shape a real trace produces —
#       a RESOLVED, no-`=>` self-line whose "path" IS the bare soname
#       itself. Any OTHER state/path/shape for this name is NOT exempt and
#       falls through to the SAME classification every other line gets,
#       which refuses it.
#     - the LOADER's own entry (`bundle_is_loader_soname`) is the other
#       explicit carve-out: it must resolve, and its resolved path must
#       equal `loader_path` EXACTLY (normalized — `bundle_normalize_path`)
#       — a jail-construction bug that places or invokes it anywhere else
#       is refused by name, never merely tolerated.
#     - a DRIVER member (`bundle_is_driver_soname`) resolving AT ALL, from
#       ANY line, is a FAIL — the jail deliberately staged none, so any
#       resolution (whether or not the binary itself names it directly)
#       proves the jail failed to exclude the host's copy.
#     - EVERY OTHER line — bundle-able or platform, named directly by the
#       binary or reached only transitively through a staged object — must
#       resolve (`not found` is a FAIL by name, exactly as much a failure
#       as a resolution from the wrong place) to a NORMALIZED path under
#       the directory `bundle_build_jail` actually staged that CLASS of
#       member into — `BUNDLE_JAIL_PLATFORM_DIR` for a platform member
#       (`bundle_is_platform_soname`, the loader itself already consumed by
#       the carve-out above), `BUNDLE_JAIL_LIB_DIR` for everything else
#       (bundle-able) — read from the SAME two constants `bundle_build_jail`
#       uses, never a literal restated here, so the two can never drift
#       apart (a real jail built with platform copies under `/platform`
#       correctly traces `libc.so.6 => /platform/libc.so.6`, and a
#       verifier that only ever accepted `BUNDLE_JAIL_LIB_DIR` would fail
#       that CORRECT jail). Normalized via `bundle_normalize_path` before
#       the prefix check, so a value like `/lib/../usr/lib/x.so` (which a
#       composed `DT_RUNPATH` could in principle produce) is judged by
#       where it actually points, never by a bare textual prefix match
#       that a `..` component defeats.
#
# `needed` (pass 1's presence set) is the binary's own full `DT_NEEDED`
# set — this arm proves self-sufficiency for that set (plus whatever pass 2
# additionally judges from the real trace). `libnvrtc-builtins` (floor-only,
# `dlopen`'d, never a `DT_NEEDED` entry, never traced by any loader run — see
# the module doc) is NOT in this set and stays UNPROVEN by this arm,
# documented rather than silently assumed.
bundle_verify_jail_report() {
  local loader_path="$1"
  local report="$2"
  shift 2
  local needed=("$@")
  local parsed
  parsed="$(printf '%s\n' "$report" | bundle_parse_loader_report)"

  # The vDSO carve-out below must also
  # reject a THIRD spoof shape — a `=>`-line whose reported PATH TEXT
  # happens to equal the bare soname itself (`linux-vdso.so.1 =>
  # linux-vdso.so.1 (0x1)`), which parses to the SAME (RESOLVED,
  # "linux-vdso.so.1") pair the genuine no-`=>` self-line produces. A real
  # trace never emits this shape (a REAL `=>` resolution always carries an
  # ABSOLUTE path); detected directly off the RAW report text, since
  # `bundle_parse_loader_report`'s output alone cannot distinguish "no `=>`
  # at all" from "a `=>` whose path happens to read the same as the name".
  local vdso_has_arrow=0
  case "$report" in
    *"linux-vdso.so.1 => "*) vdso_has_arrow=1 ;;
  esac

  # PASS 1 — presence over the binary's own DT_NEEDED set only.
  local name found entry_soname entry_state entry_path
  local absent=""
  for name in "${needed[@]}"; do
    if bundle_is_driver_soname "$name"; then
      continue
    fi
    found=0
    while IFS=' ' read -r entry_soname entry_state entry_path; do
      [ -n "$entry_soname" ] || continue
      if [ "$entry_soname" = "$name" ]; then
        found=1
      fi
    done <<EOF
$parsed
EOF
    if [ "$found" -eq 0 ]; then
      absent="${absent} ${name}"
    fi
  done
  if [ -n "$absent" ]; then
    echo "::error::bundle_cuda_libs.sh: the jail report names no line at all for:${absent} — an empty, crashed, or vacuous report must fail this arm, never pass it." >&2
    return 1
  fi

  # PASS 2 — the positive rule, over EVERY line the report actually carries.
  local normalized_loader_path
  normalized_loader_path="$(bundle_normalize_path "$loader_path")"
  local unexpected_driver="" violations="" expected_dir
  while IFS=' ' read -r entry_soname entry_state entry_path; do
    [ -n "$entry_soname" ] || continue
    # The vDSO carve-out is exempt ONLY for
    # the EXACT synthetic shape a real trace produces — a RESOLVED, no-`=>`
    # self-line whose "path" is the bare soname itself (`bundle_parse_
    # loader_report`'s shape when there is no directory component at all:
    # `linux-vdso.so.1 (0x...)` parses to soname==path==`linux-vdso.so.1`).
    # Matching on the SONAME ALONE would let a SPOOFED line carrying
    # that name in any other shape — `linux-vdso.so.1 => /usr/lib/evil.so`,
    # `linux-vdso.so.1 => not found` — through unexamined. Conditioned
    # exactly like the loader
    # arm's own carve-out just below: any OTHER state/path for this name
    # falls through to the SAME classification every other line gets
    # (never a special vdso-only violation), which is enough to refuse it
    # (an unresolvable/wrongly-placed bundle-able-shaped name).
    if [ "$entry_soname" = "linux-vdso.so.1" ] && [ "$entry_state" = "RESOLVED" ] && [ "$entry_path" = "linux-vdso.so.1" ] && [ "$vdso_has_arrow" -eq 0 ]; then
      continue
    fi
    if bundle_is_loader_soname "$entry_soname"; then
      if [ "$entry_state" != "RESOLVED" ]; then
        violations="${violations}
  ${entry_soname} (the loader itself) has no resolved entry at all"
      elif [ "$(bundle_normalize_path "$entry_path")" != "$normalized_loader_path" ]; then
        violations="${violations}
  ${entry_soname} (the loader itself) resolved from ${entry_path}, not its own PT_INTERP path ${loader_path}"
      fi
      continue
    fi
    if bundle_is_driver_soname "$entry_soname"; then
      if [ "$entry_state" = "RESOLVED" ]; then
        unexpected_driver="${unexpected_driver} ${entry_soname} (resolved ${entry_path})"
      fi
      continue
    fi
    if [ "$entry_state" != "RESOLVED" ]; then
      violations="${violations}
  ${entry_soname} not found (every non-driver, non-loader, non-vdso line must resolve)"
      continue
    fi
    # Lead-probed fix: the expected directory is keyed by CLASS — a
    # platform member (the loader already excluded above) is staged into
    # `BUNDLE_JAIL_PLATFORM_DIR` by `bundle_build_jail`, never
    # `BUNDLE_JAIL_LIB_DIR`; a bundle-able member is the reverse. Read from
    # the same two constants `bundle_build_jail` itself uses (never a
    # literal), so the builder and the verifier cannot drift apart again.
    if bundle_is_platform_soname "$entry_soname"; then
      expected_dir="$BUNDLE_JAIL_PLATFORM_DIR"
    else
      expected_dir="$BUNDLE_JAIL_LIB_DIR"
    fi
    case "$(bundle_normalize_path "$entry_path")" in
      "${expected_dir}"/*) : ;;
      *)
        violations="${violations}
  ${entry_soname} resolved from ${entry_path}, not ${expected_dir}" ;;
    esac
  done <<EOF
$parsed
EOF

  if [ -n "$unexpected_driver" ]; then
    echo "::error::bundle_cuda_libs.sh: a driver member resolved INSIDE the jail, where none was ever staged:${unexpected_driver} — this proves the jail did not actually exclude host driver copies, defeating the whole point of running under chroot." >&2
    return 1
  fi
  if [ -n "$violations" ]; then
    echo "::error::bundle_cuda_libs.sh: the jail resolved a member from the wrong place, or not found at all:${violations}" >&2
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
  # non-zero the moment any of them does (discarding these three return
  # values and printing the success sentence unconditionally would report
  # success for a tree missing `libnccl` and `libnvrtc-builtins` under the
  # suite's own `set +e`). Program mode (this file executed, not sourced) does not rely
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
