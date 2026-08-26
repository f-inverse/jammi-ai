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
# Usage: pod_target_clone.sh <seed-dir> <dest-dir> [--verify]
#   --verify: after the caller's own first build against <dest-dir>, pass a
#     `cargo build -v` LOG on stdin; this asserts it names NO `Fresh jammi-*`
#     unit line — a member-free seed means every member unit must actually
#     compile (Compiling, not Fresh) on the clone's first build.
set -uo pipefail

usage() { echo "usage: $(basename "$0") <seed-dir> <dest-dir> [--verify]" >&2; exit 2; }

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
du_default="$(du -sh "$DEST_DIR" 2>/dev/null | awk '{print $1}')"
du_apparent="$(du -sh --apparent-size "$DEST_DIR" 2>/dev/null | awk '{print $1}')"
echo "clone at ${DEST_DIR}: du=${du_default:-?} du(apparent-size)=${du_apparent:-?}"
echo "=== clone complete: ${DEST_DIR} ==="
