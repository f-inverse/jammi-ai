#!/usr/bin/env bash
# Packages a manifest server build's stripped `target/release/jammi-server`
# as that build's release tarball, and writes `asset=<file>` to
# $GITHUB_OUTPUT:
#
#   server-cpu   `package_release_bin.sh`: the binary alone, as
#                `jammi-server-<version>-<arch>-unknown-linux-gnu.tar.gz`.
#   server-cu12  the self-contained CUDA tarball,
#                `jammi-server-cu12-<version>-x86_64-unknown-linux-gnu.tar.gz`:
#                the binary, every CUDA runtime library it needs on a host
#                with only an NVIDIA driver, and an `LD_LIBRARY_PATH` launcher.
#                The set to bundle is DERIVED (`bundle_cuda_libs.sh`) from the
#                binary's own transitive `DT_NEEDED` closure plus a measured
#                floor of dlopen-only members, never a hand-kept list; the
#                real loader and a chroot jail then check it resolves. The
#                driver's own libraries (`libcuda.so.1`, `libnvidia-*`) are
#                deliberately NOT bundled — they must match the host's
#                driver. Nothing runs the binary: the loader checks only ask
#                WHERE each entry would resolve. Writes
#                `cu12_loader_report_real.txt` and `cu12_jail_report_real.txt`
#                beside it, the real reports a maintainer refreshes
#                `ci/scripts/fixtures/` from.
#
# Runs in the image the build compiled in (`_server.yml`'s binary job): the
# CUDA arm stages libraries from the toolkit that image carries.
#
# Usage: package_server_tarball.sh <build> <arch>
set -euo pipefail

build="${1:?usage: package_server_tarball.sh <build> <arch>}"
arch="${2:?usage: package_server_tarball.sh <build> <arch>}"

case "$build" in
  server-cpu)
    exec bash "$(dirname "${BASH_SOURCE[0]}")/package_release_bin.sh" jammi-server "${arch}-unknown-linux-gnu"
    ;;
  server-cu12)
    if [ "$arch" != x86_64 ]; then
      echo "::error::package_server_tarball.sh: the CUDA build ships x86_64 only, got '$arch'" >&2
      exit 1
    fi
    ;;
  *)
    echo "::error::package_server_tarball.sh: no tarball recipe for build '$build'" >&2
    exit 1
    ;;
esac

# Same parse + shape assert as package_release_bin.sh (sed echoes
# the line unchanged on no-substitution — a garbage asset name
# would upload silently).
version=$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')
case "$version" in
  [0-9]*.[0-9]*.[0-9]*) ;;
  *) echo "::error::could not parse a semver workspace version out of Cargo.toml (got: '${version}')" >&2; exit 1 ;;
esac
# ONE variable for both the asset name's triple and the
# assert's expected arch -- a second hardcoded `x86_64` here
# could drift from the asset name's triple with no gate to catch
# it.
triple="${arch}-unknown-linux-gnu"
asset="jammi-server-cu12-${version}-${triple}.tar.gz"

# This tarball's own name asserts x86_64-unknown-linux-gnu, so the
# binary's ELF machine must be x86_64 before it ships under that
# name. A CPU build gets this via `package_release_bin.sh`'s
# ELF-machine assert; this arm is bespoke (CUDA lib bundling +
# launcher assembly) and never goes through that script, so it calls
# the same shared `assert_elf_machine.sh` directly.
bash ci/scripts/assert_elf_machine.sh target/release/jammi-server "$arch"

# Container-native path, deliberately OUTSIDE $GITHUB_WORKSPACE:
# a `container:` job's workspace is bind-mounted in from the
# runner HOST (a different filesystem/device than the container's
# own writable layer), and the jail arm below needs to `cp -al` (hardlink) this multi-GB stage into the
# jail — a hardlink cannot cross that boundary. `/root` is this
# container's own filesystem, and a `container:` job runs as root.
stage="/root/jammi-server-cu12"
mkdir -p "${stage}/lib"
strip target/release/jammi-server
cp target/release/jammi-server "${stage}/bin-jammi-server"

# DERIVED staging: source the script rather
# than exec it, so this shell keeps `bundle_needed_sonames` and
# `bundle_verify_loader_resolution` in scope for the loader
# verification right below — the release lane's own `set -euo
# pipefail` stays in effect (unlike the hermetic suite, which takes
# it back off deliberately to assert on exit codes; nothing here
# does that). `bundle_main` itself asserts no `DT_RPATH`/
# `DT_RUNPATH` first, then stages the binary's transitive
# `DT_NEEDED` closure (minus the host-provided set) union the
# measured dlopen-only floor, then asserts every required soname
# landed as a same-named file — failing by name, never silently,
# on anything it cannot satisfy. No search path is passed: this
# image's default (`/usr/local/cuda-12.6/lib64` then `/usr/lib64`,
# where the `libnccl` RPM installs) is `bundle_cuda_libs.sh`'s own
# `BUNDLE_DEFAULT_SEARCH_PATH`.
. ci/scripts/bundle_cuda_libs.sh
bundle_main target/release/jammi-server "${stage}/lib"

# Loader verification, arm 1a — DETECTION, the release lane's own
# real execution; arm 1b right below is the jail (the chroot
# half), and `ci/scripts/test_bundle_cuda_libs.sh` is arm 1c, the
# hermetic fixture-parse half for both; see `bundle_cuda_libs.sh`'s
# module doc. Runs the REAL loader against the REAL stage, in this
# container (toolkit present, no NVIDIA driver — the `not found`
# driver lines this arm tolerates by name are expected here), and
# pipes the report through the SAME parser the hermetic suite
# drives over committed fixture text. Runs FIRST, unconditionally —
# arm 1b below never runs instead of this, only in addition to it;
# a failure here (`set -euo pipefail`) stops the step before arm
# 1b, the launcher, or the tarball are ever built.
loader_report="$(LD_LIBRARY_PATH="${stage}/lib" ldd target/release/jammi-server 2>&1 || true)"
# Captured verbatim for `ci/scripts/fixtures/cu12_loader_report_real.txt`
# (the real-report fixture) — uploaded as a workflow artifact below
# so a maintainer can refresh that fixture without renting a pod:
# this container already has the toolkit and no driver, which is
# exactly the shape the shipped arm sees on a release. Written
# BEFORE the verify call so a failing verify still leaves the
# report on disk for the `if: always()` upload below.
printf '%s\n' "$loader_report" > cu12_loader_report_real.txt
# shellcheck disable=SC2046
bundle_verify_loader_resolution "${stage}/lib" "$loader_report" $(bundle_needed_sonames target/release/jammi-server)

# Loader verification, arm 1b — THE JAIL (the chroot half — see
# `ci/scripts/jail_trace.py`'s own module doc for the measured
# reasons). Arm
# 1a above can only DETECT a host copy that would have satisfied a
# bundled member; this arm HIDES every host copy so none CAN, by
# running a TOLERANT `LD_TRACE_LOADED_OBJECTS` trace inside a real
# `chroot` jail — never `ld.so --list`, which is FATAL (exit
# 127, no report at all) on the FIRST missing library, and this
# jail deliberately ships with the driver libraries absent. The
# jail (`bundle_build_jail`) contains nothing but: the real
# binary; the staged `lib/` HARDLINKED in wholesale (`cp -al`
# — why `stage` above is a container-native path — and NEVER
# written to again afterward); same-named copies of the
# platform closure of the binary AND every staged object (a
# bundled CUDA library can itself need a platform member the
# binary never names directly), sourced from arm 1a's own report
# above (`bundle_platform_sources_from_report`) and copied
# byte-for-byte into the SEPARATE `/platform` directory (never
# hardlinked, never written into the `lib/` directory the shipped
# stage's own inodes live under, so a destination-name collision
# with the shipped stage is structurally impossible rather than
# merely unlikely); and the loader ITSELF at its real `PT_INTERP`
# path (anywhere else, its own self-reported trace line gains
# a `=>` the shipped parser misreads as a defect on an otherwise
# correct jail). Driver members are deliberately absent.
# `bundle_assert_jail_file_set` checks the
# real jail's file set against exactly this plan before the trace
# ever runs. Docker's default capability set includes
# `CAP_SYS_CHROOT` (probed against `docker run --rm ubuntu:24.04`,
# no `--privileged`/`--cap-add` needed), so this container needs no
# extra privilege for `os.chroot` itself to succeed — if it is
# nonetheless denied (EPERM), `jail_trace.py` exits 2 and this step
# FAILS naming the missing capability rather than silently falling
# back to arm 1a's already-passed result above.
interp="$(bundle_binary_interp target/release/jammi-server)"
interp_rel="${interp#/}"
jail="/root/jammi-server-cu12-jail"
rm -rf "$jail"
bundle_build_jail target/release/jammi-server "${stage}/lib" "$jail" "$loader_report"

# The jail BUILDER's own file-set rule,
# wired into the lane between the build and the trace, so a
# regression that stages an extra host-shaped path (or drops an
# expected one) is caught here — before the trace ever runs, not
# only in the hermetic suite over a fixture tree.
lib_listing="$(cd "${stage}/lib" && ls)"
# shellcheck disable=SC2046
platform_basenames="$(bundle_jail_platform_closure target/release/jammi-server "${stage}/lib")"
expected_relpaths="$(bundle_jail_expected_relpaths "$lib_listing" "$platform_basenames" "$interp_rel")"
# shellcheck disable=SC2086
bundle_assert_jail_file_set "$jail" $expected_relpaths

# `python3` itself is a dynamically-
# linked executable — if THIS shell already carries
# LD_TRACE_LOADED_OBJECTS, the loader that starts `python3`
# traces `python3`'s OWN dependency list and exits 0 WITHOUT ever
# running jail_trace.py's code at all (measured:
# `LD_TRACE_LOADED_OBJECTS=1 python3 -c 'print(2)'` prints
# python3's own trace, never "2", exit 0) — a check inside
# jail_trace.py cannot defend against this specific case, since
# its own code never runs; refused HERE, before python3 is ever
# exec'd, is the only place this hazard can actually be caught.
if [ -n "${LD_TRACE_LOADED_OBJECTS+x}" ]; then
  echo "::error::LD_TRACE_LOADED_OBJECTS is already set in this shell — invoking python3 in this state would trace python3 ITSELF (jail_trace.py's own code would never run), not the jail. Refusing before python3 is exec'd." >&2
  exit 1
fi
jail_report="$(python3 ci/scripts/jail_trace.py "$jail" "$interp" /jammi-server "${BUNDLE_JAIL_LIB_DIR}:${BUNDLE_JAIL_PLATFORM_DIR}")" && jail_rc=0 || jail_rc=$?
if [ "$jail_rc" -eq 2 ]; then
  echo "::error::jail_trace.py: os.chroot(${jail}) was denied (EPERM) — CAP_SYS_CHROOT is likely missing from this job's container; the jail arm cannot run here." >&2
  exit 1
fi
if [ "$jail_rc" -eq 3 ]; then
  echo "::error::jail_trace.py: the trace failed before producing a report (bad path, missing loader, or exec failure) — see the step log above for the child's own diagnostic." >&2
  exit 1
fi
if [ "$jail_rc" -ne 0 ]; then
  echo "::error::jail_trace.py exited ${jail_rc}, neither the success (0) nor either named failure (2, 3) code — treating as a defect." >&2
  exit 1
fi
# The report and the measured DT_NEEDED set it must be judged
# against are captured TOGETHER, from the SAME lane run, as one
# fixture — a later drift between a hand-kept BINARY_NEEDED and
# what this run actually measured can never desync the two halves
# of the oracle.
{
  printf '%s\n' "$jail_report"
  printf '%s\n' "# --- BINARY_NEEDED ---"
  bundle_needed_sonames target/release/jammi-server
} > cu12_jail_report_real.txt
# shellcheck disable=SC2046
bundle_verify_jail_report "$interp" "$jail_report" $(bundle_needed_sonames target/release/jammi-server)
rm -rf "$jail"

# The launcher: a host with only an NVIDIA driver has no system CUDA,
# so put the bundled libs on the loader path before exec. `exec`
# replaces the shell so signals and the exit code pass through.
cat > "${stage}/jammi-server" <<'LAUNCH'
#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="${here}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "${here}/bin-jammi-server" "$@"
LAUNCH
chmod +x "${stage}/jammi-server"

# `-C`, not a bare `tar -czf "$asset" "$stage"`: `$stage` is an
# absolute, container-native path (`/root/...`), and taring it
# directly would embed that absolute path in the archive rather
# than the relative `jammi-server-cu12/...` layout the launcher
# and every consumer of this tarball expect.
tar -czf "$asset" -C "$(dirname "$stage")" "$(basename "$stage")"
echo "asset=$asset" >> "$GITHUB_OUTPUT"
