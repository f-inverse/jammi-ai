#!/usr/bin/env bash
# Clones the pod's build-substrate SEED into a fresh CARGO_TARGET_DIR for a
# tree. A clone is a PURE COPY — there is no deletion step, no drift window:
# the seed is member-free by construction (pod_seed_target.sh), so every
# workspace-member artifact in the clone comes from the CLONE's own build,
# never from the seed. `cp -a` (reflink where the filesystem supports it)
# because a clone is throwaway per-tree state, not a shared object cache — a
# hardlink clone was reproduced to CORRUPT the seed (round-1 pressure-test:
# writing through a hardlinked path mutates the seed's own copy), so this
# never hardlinks.
#
# Usage: pod_target_clone.sh <seed-dir> <dest-dir> [tree-dir] [--verify]
#   --verify: after the caller's own first build against <dest-dir>, pass a
#     `cargo build -v` LOG on stdin; this asserts it names NO `Fresh jammi-*`
#     unit line — a member-free seed means every member unit must actually
#     compile (Compiling, not Fresh) on the clone's first build. This is an
#     ADDITIONAL, OPT-IN form a human runs after a real build — it is not a
#     substitute for the UNCONDITIONAL filesystem-level check below (round-3
#     audit N2): --verify only ever runs when a caller remembers to run it
#     and only ever catches member units that were Fresh on ONE specific
#     build; the filesystem check runs on every single clone regardless and
#     catches a leftover artifact whether or not anyone ever rebuilds.
#   [tree-dir]: where `cargo metadata` resolves the workspace member list
#     from for the member-freedom check (default /root/jammi-ai — the
#     bootstrap checkout, always present, whose workspace member LIST is
#     what "member-free" is checked against regardless of which tree the
#     clone itself belongs to).
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$DIR/pod_seed_target.sh"

usage() { echo "usage: $(basename "$0") <seed-dir> <dest-dir> [tree-dir] [--verify]" >&2; exit 2; }

VERIFY=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --verify) VERIFY=1 ;;
    *) ARGS+=("$a") ;;
  esac
done
[ "${#ARGS[@]}" -ge 2 ] || usage
SEED_DIR="${ARGS[0]}"
DEST_DIR="${ARGS[1]}"
TREE_DIR_FOR_METADATA="${ARGS[2]:-/root/jammi-ai}"

if [ "$VERIFY" = "1" ]; then
  # Reads a `cargo build -v` log from stdin (the caller already ran the
  # clone's first build); this branch does no cloning of its own.
  LOG="$(cat)"
  if printf '%s\n' "$LOG" | grep -Eq '^[[:space:]]*Fresh[[:space:]]+jammi-'; then
    echo "::error::clone verify FAILED — a member unit reported Fresh on its first build (seed poisoned the clone with a member artifact):" >&2
    printf '%s\n' "$LOG" | grep -E '^[[:space:]]*Fresh[[:space:]]+jammi-' >&2
    exit 1
  fi
  echo "clone verify OK — no Fresh jammi-* unit on the clone's first build"
  exit 0
fi

COMPLETE_MARKER="${SEED_DIR}.jammi-seed-complete"
[ -f "$COMPLETE_MARKER" ] || {
  echo "::error::refusing to clone: no seed at ${SEED_DIR} (missing ${COMPLETE_MARKER}) — run pod_seed_target.sh first" >&2
  exit 3
}

[ -d "$SEED_DIR" ] || {
  echo "::error::seed marker exists (${COMPLETE_MARKER}) but ${SEED_DIR} itself is missing" >&2
  exit 3
}

mkdir -p "$(dirname "$DEST_DIR")"
[ -e "$DEST_DIR" ] && { echo "::error::destination ${DEST_DIR} already exists — refusing to clone over it" >&2; exit 2; }

# `cp --help` advertising `--reflink` is the portability probe: GNU coreutils
# on a CoW filesystem (btrfs, xfs+reflink, some overlay setups) supports it;
# the RunPod image's rootfs may or may not. `--reflink=auto` falls back to a
# real copy silently when the filesystem cannot CoW, so this is always safe
# to pass when the flag itself is recognised — never assumed.
if cp --help 2>&1 | grep -q -- '--reflink'; then
  echo "cp: using --reflink=auto (CoW where the filesystem supports it, else a real copy)"
  cp -a --reflink=auto "$SEED_DIR" "$DEST_DIR"
else
  echo "cp: no --reflink support advertised — plain -a copy"
  cp -a "$SEED_DIR" "$DEST_DIR"
fi
rc=$?
[ "$rc" -eq 0 ] || { echo "::error::cp -a ${SEED_DIR} -> ${DEST_DIR} failed (exit $rc)" >&2; exit "$rc"; }

# `du` (apparent disk usage) vs `du --apparent-size` (logical byte count):
# printed side by side so a reflink clone's TRUE marginal disk cost is
# visible even though its logical size equals the seed's.
du_default="$(du -sh "$DEST_DIR" 2>/dev/null | awk '{print $1}')" # tripwire-ok: best-effort report-only size; the ${du_default:-?} fallback right below prints a visible "?" sentinel, never a silent blank
du_apparent="$(du -sh --apparent-size "$DEST_DIR" 2>/dev/null | awk '{print $1}')" # tripwire-ok: same as du_default above (also covers hosts whose du lacks --apparent-size)
echo "clone at ${DEST_DIR}: du=${du_default:-?} du(apparent-size)=${du_apparent:-?}"

# UNCONDITIONAL, round-3 audit N2: every clone is checked, not just the ones
# a human remembers to --verify after a build. A seed that was NOT actually
# member-free (the failure --verify alone can never catch until someone
# rebuilds against it) is caught the instant it is cloned.
if ! pod_seed_assert_member_free "$DEST_DIR" "$TREE_DIR_FOR_METADATA"; then
  echo "::error::clone at ${DEST_DIR} is NOT member-free — the seed it came from is poisoned; removing the clone" >&2
  rm -rf "$DEST_DIR"
  exit 1
fi

# esc-077: stamp a marker INSIDE the destination so `gpu-dev.sh run`'s
# preflight (and a human) can tell a genuine seed-clone from a raw/cold
# CARGO_TARGET_DIR at a glance — mirroring this script's own SEED-marker
# check above (COMPLETE_MARKER, :62) rather than inventing a new marker
# shape. Records the seed's completion-marker CONTENT (mtime + sha256, not
# just its path) so a later re-seed of the SAME seed dir (a new, different
# completion marker) is distinguishable from the seed this clone was
# actually taken from — self-contained python3 (hashlib/os/datetime), no
# new bash-side hashing dependency beyond what this substrate already
# assumes elsewhere (pod_seed_target.sh's own JSON-marker idiom).
CLONE_MARKER="${DEST_DIR}/.jammi-clone-of-seed"
python3 -c '
import hashlib, json, os, sys, datetime
seed_dir, complete_marker, dest_dir = sys.argv[1], sys.argv[2], sys.argv[3]
h = hashlib.sha256()
with open(complete_marker, "rb") as f:
    h.update(f.read())
mtime = os.path.getmtime(complete_marker)
now = datetime.datetime.now(tz=datetime.timezone.utc)
print(json.dumps({
    "seed_dir": seed_dir,
    "seed_complete_marker": complete_marker,
    "seed_complete_marker_mtime": datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "seed_complete_marker_sha256": h.hexdigest(),
    "dest_dir": dest_dir,
    "clone_timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
}, indent=2))
' "$SEED_DIR" "$COMPLETE_MARKER" "$DEST_DIR" > "$CLONE_MARKER"
rc=$?
[ "$rc" -eq 0 ] || { echo "::error::failed to stamp clone marker at ${CLONE_MARKER} (exit $rc)" >&2; exit "$rc"; }
echo "clone marker stamped: ${CLONE_MARKER}"

echo "=== clone complete: ${DEST_DIR} ==="
