#!/usr/bin/env bash
# The ONE definition of `push`'s exclude set, plus the laptop-side provenance
# stamp that names exactly what a `push` sent.
#
# `check_cuda_run_artifacts.py`'s git_sha rule is UNCHANGED by any of this: a
# COMMITTED artifact still requires a pushed (i.e. reachable-from-a-remote-
# branch) sha. The push stamp below is iteration provenance ONLY — "what did
# this pod actually receive, right now" for a human debugging a live session —
# never a substitute for that rule.
#
# Usage:
#   pod_push_stamp.sh excludes
#       Prints the exclude set, one pattern per line. gpu-dev.sh's `push`
#       turns each line into an `--exclude '<pattern>'` rsync argument — the
#       ONLY place the pattern list itself is written down, so the real
#       rsync and the stamp's own manifest hash below can never drift apart.
#
#   pod_push_stamp.sh compute <repo-root> <session>
#       Prints the stamp JSON to stdout: {laptop_head, porcelain_sha256,
#       diff_head_sha256, manifest_sha256, ts, session}. The caller (gpu-dev.sh
#       `push`) writes this to a local temp file and rsyncs/scps it to
#       <tree>/.jammi-push-stamp.json right after the real push, so the stamp
#       and the bytes it describes land together.
#
#       manifest_sha256 is computed ENTIRELY LOCALLY (no network, no pod
#       reachability needed — testable hermetically) by dry-running rsync
#       from <repo-root> into a FRESH EMPTY temp directory under the SAME
#       exclude set `excludes` prints: an empty destination makes rsync
#       report every non-excluded path via `--out-format='%n'`, deterministic
#       regardless of whatever the real pod's tree currently holds. Each
#       reported path is then (relative path, file mode, sha256 of its
#       CONTENT) — sorted by path, concatenated, sha256'd once more. This is
#       NOT `git write-tree`: a `write-tree`-based hash would not name bytes
#       an LFS filter or a gitlink (cutlass) rewrites in the working tree —
#       round-5 pressure-test finding — so this hashes what rsync would
#       actually SEND, not what git's index records.
set -uo pipefail

# The one exclude list. cutlass is excluded from push (and from this
# manifest) because it is a git submodule OWNED by tree provisioning, never
# by the working-tree sync — `target`/`push --with-cutlass` provisions it
# with `git -C <tree> submodule update --init --depth 1
# crates/jammi-kernels/third_party/cutlass`, so pushing it as plain files
# would both be wasteful (a full CUTLASS checkout) and wrong (rsync --delete
# would then delete the pod's own submodule checkout on every push that
# omits it).
pod_push_excludes() {
  cat <<'EXC'
.claude
.sccache
.gpu-pull
scratchpad
target
.git
.venv*
crates/jammi-kernels/third_party/cutlass
EXC
}

pod_push_manifest_sha256() { # $1=repo-root
  local repo="$1" empty manifest
  empty="$(mktemp -d)"
  local -a rsync_excludes=()
  while IFS= read -r pat; do
    [ -n "$pat" ] || continue
    rsync_excludes+=(--exclude "$pat")
  done < <(pod_push_excludes)
  manifest="$(mktemp)"
  # --dry-run against an EMPTY destination: rsync reports every path it
  # would create, independent of any real pod's current state.
  rsync -a --dry-run --out-format='%n' "${rsync_excludes[@]}" "${repo%/}/" "$empty/" \
    | while IFS= read -r rel; do
        [ -n "$rel" ] || continue
        local f="${repo%/}/${rel}"
        # Directory entries appear in the listing too (rsync -a recurses);
        # only files carry content to hash, and only files get a manifest
        # line — a directory's presence is already implied by its files'
        # own paths.
        [ -f "$f" ] || continue
        local mode sha
        mode="$(stat -f '%Lp' "$f" 2>/dev/null || stat -c '%a' "$f" 2>/dev/null || echo '?')"
        sha="$(shasum -a 256 "$f" 2>/dev/null | awk '{print $1}')"
        printf '%s\t%s\t%s\n' "$rel" "$mode" "$sha"
      done | sort > "$manifest"
  shasum -a 256 "$manifest" | awk '{print $1}'
  rm -rf "$empty" "$manifest"
}

pod_push_compute() { # $1=repo-root $2=session
  local repo="$1" session="$2" head porcelain_sha diff_sha manifest_sha ts
  head="$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo unknown)"
  porcelain_sha="$(git -C "$repo" status --porcelain 2>/dev/null | shasum -a 256 | awk '{print $1}')"
  diff_sha="$(git -C "$repo" diff HEAD 2>/dev/null | shasum -a 256 | awk '{print $1}')"
  manifest_sha="$(pod_push_manifest_sha256 "$repo")"
  ts="$(date -u +%FT%TZ)"
  python3 -c '
import json, sys
print(json.dumps({
  "laptop_head": sys.argv[1],
  "porcelain_sha256": sys.argv[2],
  "diff_head_sha256": sys.argv[3],
  "manifest_sha256": sys.argv[4],
  "ts": sys.argv[5],
  "session": sys.argv[6],
}, indent=2))
' "$head" "$porcelain_sha" "$diff_sha" "$manifest_sha" "$ts" "$session"
}

usage() { echo "usage: $(basename "$0") excludes | compute <repo-root> <session>" >&2; exit 2; }

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  [ $# -ge 1 ] || usage
  case "$1" in
    excludes) pod_push_excludes ;;
    compute)
      [ $# -ge 3 ] || usage
      pod_push_compute "$2" "$3" ;;
    *) usage ;;
  esac
fi
