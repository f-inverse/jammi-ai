#!/usr/bin/env bash
# A server build's PyPI wheel from its compiled binary: the binary staged
# into `packaging/<build>`, its runtime link set verified against the wheel's
# dependency wheels when the manifest says those deliver it
# (`wheel_verifies_link_set`), the wheel built, and relabelled to the platform
# the binary requires (a bin-bearing wheel is not py3-none-any; hatchling
# sees only the Python sources).
#
#   bash ci/scripts/build_server_wheel.sh BUILD BINARY PLATFORM_TAG
#
# BUILD is a build of `ci/release-feature-manifest.json` (server-cpu,
# server-cu12); BINARY the stripped `jammi-server`; PLATFORM_TAG e.g.
# manylinux_2_28_x86_64. The wheel lands in `packaging/<build>/dist/`.
set -euo pipefail

build="${1:?BUILD}"
binary="${2:?BINARY}"
platform_tag="${3:?PLATFORM_TAG}"
root="$(git rev-parse --show-toplevel)"
package="$root/packaging/$build"
manifest="$root/ci/release-feature-manifest.json"

verify="$(jq -r --arg b "$build" '.builds[$b].wheel_verifies_link_set' "$manifest")"
case "$verify" in
  true | false) ;;
  *) echo "::error::build '$build' states no wheel_verifies_link_set (true/false) in $manifest" >&2; exit 1 ;;
esac

# Artifacts do not preserve the exec bit.
install -m 0755 "$binary" "$package/jammi_server/bin/jammi-server"

# Fail here, not on a user's machine at execve, if the binary links a library
# the wheel's runtime dependencies do not deliver; DT_NEEDED survives stripping.
if [ "$verify" = true ]; then
  python3 "$package/verify_link_set.py" "$binary"
fi

cd "$package"
python3 -m pip install --quiet build wheel
python3 -m build --wheel --outdir dist
python3 -m wheel tags --platform-tag "$platform_tag" --remove dist/*.whl
