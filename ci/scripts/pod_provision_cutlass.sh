#!/usr/bin/env bash
# Provisions cutlass INTO an already-pushed tree, from /root/jammi-ai's own
# initialised submodule, verified against the TREE's own push stamp — the
# `target --with-cutlass` remote body, extracted into a real file this
# suite can source and run against a REAL two-commit submodule fixture
# (round-4 audit finding: the only coverage for this logic used to be two
# `grep`s on gpu-dev.sh's heredoc TEXT — a proxy never run against a real
# instance; shellcheck cannot see it either, since a heredoc body is just
# text to it). gpu-dev.sh's `target --with-cutlass` case now does nothing
# but `bash ci/scripts/pod_provision_cutlass.sh ...` — this file IS the
# remote logic, not a copy of it, so test_pod_substrate.sh's `(m/A1 match)`,
# `(m/A1 drift)`, `(m/A1 deinit)`, `(m/A1 fetch-failure)`, and
# `(m/A1 revert-RED)` legs and the real pod invocation run byte-identical
# code.
#
# `cp -a` from /root/jammi-ai's OWN initialised submodule — never `git
# submodule update` INSIDE the destination tree (round-2 audit finding 1):
# a tree populated by `push` (rsync, which excludes `.git` — see
# pod_push_stamp.sh) carries no `.git` at all, so `git submodule` there
# fails with "not a git repository" on every tree except the default
# bootstrap checkout. /root/jammi-ai IS always a real git clone
# (rp_bootstrap's own, untouched by push), so its submodule is initialised
# there once — but round-3 audit N1: /root/jammi-ai's CURRENT gitlink is
# not necessarily the commit the DESTINATION tree's own ref actually needs
# (the gitlink has already moved once, 0ee65de) — a tree on an FA2 branch
# pinning a DIFFERENT cutlass commit than whatever /root/jammi-ai (usually
# main) happens to have checked out would silently receive the WRONG
# headers. The tree's own push stamp (pod_push_stamp.sh's cutlass_gitlink
# field, written at push time from THAT tree's actual HEAD) is the source
# of truth: verified via pod_push_cutlass_matches (the SAME script
# test_pod_substrate.sh's `(m/N1)` leg exercises, never a second copy of
# the comparison logic) against /root/jammi-ai's submodule AFTER `submodule
# update`; on a mismatch, fetch+checkout the STAMPED commit into
# /root/jammi-ai's own submodule (network — fails loudly if unreachable)
# and re-verify before copying; refuses the copy on any remaining
# mismatch, naming both shas.
#
# round-4 audit A1: `git rev-parse HEAD:<gitlink-path>` reads the
# SUPERPROJECT's own recorded pin for that path — a property of /root/
# jammi-ai's OWN HEAD commit, entirely UNAFFECTED by whether `submodule
# update` actually ran, or what the submodule's working directory is
# actually checked out to. It is not a proxy for "what commit does the
# submodule dir cp -a would copy actually hold" — `git -C <submodule-dir>
# rev-parse HEAD` is that. This also means an OLDER remediation arm
# (compare `HEAD:<path>` before/after fetch+checkout) could never succeed:
# checking out a different commit INSIDE the submodule cannot change what
# `HEAD:path` reports in the superproject.
#
# round-5 audit A1 (the actual defect this file exists to fix): the round-4
# fix ALSO added `set -euo pipefail` to the remote block — correct on its
# own — but left `pod_push_stamp.sh cutlass-check` as a BARE simple
# command whose non-zero exit (a genuine MISMATCH, rc=1) then aborted the
# whole remote shell under `set -e` BEFORE `CHECK_RC=$?` could ever read
# it: the entire mismatch-remediation arm (fetch+checkout+re-verify) was
# DEAD CODE, a regression from the pre-round-4 form where it ran. Reproduced
# against a real two-commit submodule fixture (test_pod_substrate.sh's
# `(m/A1 drift)` and `(m/A1 revert-RED)` legs): `set -euo pipefail` + a bare command
# stops at MISMATCH, rc=1, remediation never reached; `if <cmd>; then ...;
# else CHECK_RC=$?; fi` — an `if`-condition is a `set -e`-EXEMPT context —
# reaches the remediation arm, fetches+checks out the stamped commit,
# re-verifies OK, and proceeds. `set -e` for the REST of this script's
# arms (the every-other-command-must-abort-on-failure contract `target
# --with-cutlass` depends on) is unaffected: only the ONE command whose
# non-zero exit is a MEANINGFUL, handled outcome (not a bug) is wrapped.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() { echo "usage: $(basename "$0") <source-tree-dir> [super-dir]" >&2; exit 2; }
[ $# -ge 1 ] || usage
TREE_SOURCE_DIR="$1"
SUPER_DIR="${2:-/root/jammi-ai}"

git -C "$SUPER_DIR" submodule update --init --depth 1 crates/jammi-kernels/third_party/cutlass

[ -d "$TREE_SOURCE_DIR" ] || { echo "::error::tree source dir '${TREE_SOURCE_DIR}' does not exist — push to it first (target --with-cutlass provisions cutlass INTO an existing tree, it does not create one)" >&2; exit 1; }

CUTLASS_DIR="$SUPER_DIR/crates/jammi-kernels/third_party/cutlass"
[ -d "$CUTLASS_DIR/.git" ] || [ -f "$CUTLASS_DIR/.git" ] \
  || { echo "::error::${CUTLASS_DIR} has no .git after submodule update — deinitialised or never checked out; refusing the copy" >&2; exit 1; }

STAMP="$TREE_SOURCE_DIR/.jammi-push-stamp.json"
ACTUAL_SHA="$(git -C "$CUTLASS_DIR" rev-parse HEAD)"

# `set -e`-EXEMPT context (family A: every arm of the exit-state lattice
# must be reachable) — see this file's own module doc above for the
# regression this if/else closes.
if bash "$DIR/pod_push_stamp.sh" cutlass-check "$STAMP" "$ACTUAL_SHA"; then
  CHECK_RC=0
else
  CHECK_RC=$?
fi

if [ "$CHECK_RC" -eq 1 ]; then
  STAMP_SHA="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("cutlass_gitlink") or "")' "$STAMP")"
  echo "attempting to fetch+checkout the stamp's pinned cutlass commit ${STAMP_SHA} into the submodule at ${CUTLASS_DIR}..."
  git -C "$CUTLASS_DIR" fetch --depth 1 origin "$STAMP_SHA" \
    && git -C "$CUTLASS_DIR" checkout --quiet "$STAMP_SHA" \
    || { echo "::error::could not fetch/checkout cutlass ${STAMP_SHA} into ${CUTLASS_DIR} (network unreachable?) — refusing the copy" >&2; exit 1; }
  ACTUAL_SHA="$(git -C "$CUTLASS_DIR" rev-parse HEAD)"
  bash "$DIR/pod_push_stamp.sh" cutlass-check "$STAMP" "$ACTUAL_SHA" \
    || { echo "::error::even after fetch+checkout, the SUBMODULE's own HEAD still does not match the stamp — refusing the copy" >&2; exit 1; }
elif [ "$CHECK_RC" -ne 0 ]; then
  exit 1
fi

mkdir -p "$TREE_SOURCE_DIR/crates/jammi-kernels/third_party"
rm -rf "$TREE_SOURCE_DIR/crates/jammi-kernels/third_party/cutlass"
cp -a "$CUTLASS_DIR" "$TREE_SOURCE_DIR/crates/jammi-kernels/third_party/cutlass"

# round-4 addendum: $CUTLASS_DIR's own `.git` is a SUBMODULE GITLINK
# pointer file (not a full repo), and `cp -a` copies it verbatim into the
# destination tree — a plain directory tree that is not itself registered
# as owning that gitlink. $TREE_SOURCE_DIR is itself a real git checkout
# (rp_bootstrap's default tree, or a pushed tree whose OWN .git already
# exists); a second, foreign, un-registered .git nested inside it makes
# `git status`/`git add` run from that tree's root treat the path as an
# embedded-repository boundary it cannot resolve, failing fatally. The
# gitlink file is never needed in the copy (this path is deliberately NOT
# git-managed inside the destination tree at all — `target --with-cutlass`
# provisions it by `cp -a`, never by `git submodule` inside the tree,
# exactly because a pushed tree carries no .git of its own to attach a
# submodule to). Strip it and assert it is gone.
rm -rf "$TREE_SOURCE_DIR/crates/jammi-kernels/third_party/cutlass/.git"
[ -e "$TREE_SOURCE_DIR/crates/jammi-kernels/third_party/cutlass/.git" ] \
  && { echo "::error::cutlass/.git still present in the destination tree after stripping — refusing to leave a foreign gitlink in a git-backed tree" >&2; exit 1; }

echo "cutlass provisioned into ${TREE_SOURCE_DIR}/crates/jammi-kernels/third_party/cutlass"
