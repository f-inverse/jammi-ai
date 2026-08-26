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
#       diff_head_sha256, manifest_sha256, cutlass_gitlink, ts, session}. The
#       caller (gpu-dev.sh `push`) writes this to a local temp file and
#       rsyncs/scps it to <tree>/.jammi-push-stamp.json right after the real
#       push, so the stamp and the bytes it describes land together.
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
#
#       cutlass_gitlink = `git rev-parse HEAD:crates/jammi-kernels/third_party/cutlass`
#       (empty if the path is not a gitlink at HEAD) — round-3 audit N1: the
#       gitlink is EXCLUDED from the manifest/push (below) precisely because
#       it is a submodule, not plain files, which previously meant the
#       pushed tree carried NO RECORD of which cutlass commit it actually
#       needs (the gitlink already moved once, 0ee65de). `target
#       --with-cutlass` reads this field back and refuses to copy a
#       mismatched cutlass into the tree — see `pod_push_cutlass_matches`.
#
#   pod_push_stamp.sh cutlass-check <stamp-json-path> <actual-sha>
#       Returns 0 if the stamp's own cutlass_gitlink equals <actual-sha>, 1
#       on a genuine mismatch (prints both shas), 2 if the stamp is
#       missing/unreadable/has no cutlass_gitlink field (a stale pre-N1
#       stamp, or the tree was never pushed) — the SAME script `target
#       --with-cutlass` invokes remotely (it ships with the checkout, see
#       rp_bootstrap), so the hermetic tests below and the real pod
#       invocation run byte-identical logic, never two copies that can
#       drift apart.
set -uo pipefail

# The one exclude list. cutlass is excluded from push (and from this
# manifest) because it is a git submodule OWNED by tree provisioning, never
# by the working-tree sync — `target --with-cutlass` provisions it (`cp -a`
# from /root/jammi-ai's own initialised submodule, verified against this
# stamp's cutlass_gitlink field — never `git submodule` inside the tree
# itself, which for a tree populated purely by `push` carries no `.git` at
# all), so pushing it as plain files would both be wasteful (a full CUTLASS
# checkout) and wrong (rsync --delete would then delete the pod's own
# submodule checkout on every push that omits it).
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
  local repo="$1" session="$2" head porcelain_sha diff_sha manifest_sha cutlass_gitlink ts
  head="$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo unknown)"
  porcelain_sha="$(git -C "$repo" status --porcelain 2>/dev/null | shasum -a 256 | awk '{print $1}')"
  diff_sha="$(git -C "$repo" diff HEAD 2>/dev/null | shasum -a 256 | awk '{print $1}')"
  manifest_sha="$(pod_push_manifest_sha256 "$repo")"
  cutlass_gitlink="$(git -C "$repo" rev-parse HEAD:crates/jammi-kernels/third_party/cutlass 2>/dev/null || true)"
  ts="$(date -u +%FT%TZ)"
  python3 -c '
import json, sys
print(json.dumps({
  "laptop_head": sys.argv[1],
  "porcelain_sha256": sys.argv[2],
  "diff_head_sha256": sys.argv[3],
  "manifest_sha256": sys.argv[4],
  "cutlass_gitlink": sys.argv[5] or None,
  "ts": sys.argv[6],
  "session": sys.argv[7],
}, indent=2))
' "$head" "$porcelain_sha" "$diff_sha" "$manifest_sha" "$cutlass_gitlink" "$ts" "$session"
}

# round-3 audit N1. $1=stamp-json-path $2=actual-sha. Returns 0 (match) / 1
# (genuine mismatch, both shas printed) / 2 (no usable stamp — missing,
# unreadable, or no cutlass_gitlink field).
pod_push_cutlass_matches() {
  local stamp="$1" actual="$2" stamp_sha
  if [ ! -f "$stamp" ]; then
    echo "::error::no push stamp at ${stamp} — push to this tree first" >&2
    return 2
  fi
  stamp_sha="$(python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
print(d.get("cutlass_gitlink") or "")
' "$stamp" 2>/dev/null)"
  if [ -z "$stamp_sha" ]; then
    echo "::error::push stamp at ${stamp} has no cutlass_gitlink field (a stale pre-N1 stamp — push again)" >&2
    return 2
  fi
  if [ "$stamp_sha" != "$actual" ]; then
    echo "::error::cutlass gitlink MISMATCH — stamp says ${stamp_sha}, actual is ${actual}" >&2
    return 1
  fi
  echo "cutlass gitlink OK — stamp and actual both ${actual}"
  return 0
}

usage() { echo "usage: $(basename "$0") excludes | compute <repo-root> <session> | cutlass-check <stamp-json-path> <actual-sha>" >&2; exit 2; }

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  [ $# -ge 1 ] || usage
  case "$1" in
    excludes) pod_push_excludes ;;
    compute)
      [ $# -ge 3 ] || usage
      pod_push_compute "$2" "$3" ;;
    cutlass-check)
      [ $# -ge 3 ] || usage
      pod_push_cutlass_matches "$2" "$3" ;;
    *) usage ;;
  esac
fi
