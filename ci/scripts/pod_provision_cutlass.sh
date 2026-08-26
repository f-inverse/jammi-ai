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
# code. round-6 audit item 1: `pod_build_timings.sh`'s A2 acceptance run
# now ALSO calls this file (see that script's own citation) — this IS the
# ONE provisioning surface for cutlass in ANY tree, never a second,
# independent `git submodule update --init` run in-tree.
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
# headers. The tree's own EXPECTED PIN is the source of truth: verified
# via pod_push_cutlass_matches (the SAME script test_pod_substrate.sh's
# `(m/N1)` leg exercises, never a second copy of the comparison logic)
# against /root/jammi-ai's submodule AFTER `submodule update`; on a
# mismatch, fetch+checkout the pinned commit into /root/jammi-ai's own
# submodule (network — fails loudly if unreachable) and re-verify before
# copying; refuses the copy on any remaining mismatch, naming both shas.
#
# round-6 audit item 1 (the class this fix closes: "the scripts assume a
# git state of the tree that a pushed/provisioned tree does not have"):
# WHERE that expected pin comes from now DEPENDS on the destination
# tree's own shape, decided HERE (the one provisioning surface), never by
# a caller running its own separate git command against the tree:
#   - a tree that is ITSELF a real git checkout (has its own `.git`,
#     `.gitmodules` declaring this path — e.g. pod_build_timings.sh's own
#     FA2-tip checkout, or any bundle/clone) carries its OWN recorded
#     gitlink pin at `HEAD:<path>`, live, correct by construction (it
#     moved WITH the checkout), and requiring NO separate stamp file.
#   - a tree with no `.git` of its own (the pure rsync-`push`ed case,
#     which strips `.git` specifically) falls back to the push-stamp
#     JSON, as before.
# Either way, the DESTINATION path itself is populated by `rm -rf` +
# `cp -a` — plain filesystem operations, never `git submodule update`
# run directly against it — so a path that is ALREADY populated (by an
# earlier provisioning call, poisoned or not) is simply overwritten, the
# exact failure mode a live a100c run hit: `pod_build_timings.sh` used to
# run `git submodule update --init` directly on its own tree's cutlass
# path AFTER an earlier `target --with-cutlass` had already copy-
# provisioned (`.git`-stripped) content there — git refuses to touch a
# non-empty, non-submodule-shaped directory (rc=1, wall=819s wasted
# before failing). Filesystem overwrite has no such refusal.
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
# `HEAD:path` reports in the superproject. round-6 audit item 3
# correction: the BARE form of this command (no `--verify --quiet`)
# ECHOES its own argument text to stdout on a missing path (rc=128) —
# reproduced directly (`git rev-parse HEAD:no/such/path 2>/dev/null`
# prints the literal string "HEAD:no/such/path") — every call site in
# this file and in pod_push_stamp.sh now uses `--verify --quiet`, which
# is silent on a miss.
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
CUTLASS_PATH="crates/jammi-kernels/third_party/cutlass"

usage() { echo "usage: $(basename "$0") <source-tree-dir> [super-dir]" >&2; exit 2; }
[ $# -ge 1 ] || usage
TREE_SOURCE_DIR="$1"
SUPER_DIR="${2:-/root/jammi-ai}"

[ -d "$TREE_SOURCE_DIR" ] || { echo "::error::tree source dir '${TREE_SOURCE_DIR}' does not exist — push to it first (target --with-cutlass provisions cutlass INTO an existing tree, it does not create one)" >&2; exit 1; }

# round-6 fix (lead probe item 3): this call needs NETWORK access and a
# reachable submodule remote (a fresh `--init` clones it) — under this
# script's own `set -e`, a failure here used to abort with git's own raw
# stderr and no step name at all, leaving a reader to guess whether the
# failure was network, the destination tree, or something else entirely.
# Named explicitly.
if ! git -C "$SUPER_DIR" submodule update --init --depth 1 "$CUTLASS_PATH"; then
  echo "::error::pod_provision_cutlass: submodule update failed (network/remote unreachable?) for ${CUTLASS_PATH} in ${SUPER_DIR}" >&2
  exit 1
fi

CUTLASS_DIR="$SUPER_DIR/$CUTLASS_PATH"
[ -d "$CUTLASS_DIR/.git" ] || [ -f "$CUTLASS_DIR/.git" ] \
  || { echo "::error::${CUTLASS_DIR} has no .git after submodule update — deinitialised or never checked out; refusing the copy" >&2; exit 1; }

# round-6 audit item C (a real a100e failure): "provisioned" an EMPTY
# cutlass dir once when the superproject's OWN submodule files had been
# deleted out from under it by an unrelated push at 15:51Z — the old code
# validated ONLY `.git` presence + the HEAD sha, never that the checked-
# out CONTENT actually matches what that sha's tree says should be there.
# Validate CONTENT, not just a git ref, UNCONDITIONALLY (before the self-
# target guard below, since a self-targeting call — e.g. $JAMMI_TREE_DIR
# defaulting to the SAME /root/jammi-ai a100e's own SUPER_DIR was — is
# exactly the shape that incident hit): a real, non-empty cutlass
# checkout always carries `include/cutlass/cutlass.h`, and the on-disk
# file count must be >= `git ls-tree -r HEAD | wc -l` for the SAME commit
# (derived from the pinned commit's own tree, never a hand-typed floor) —
# anything less means files are missing beneath a technically-valid
# `.git`/HEAD, and must refuse loudly rather than copy (or trust, in the
# self-target case) an empty/partial checkout.
[ -f "$CUTLASS_DIR/include/cutlass/cutlass.h" ] \
  || { echo "::error::${CUTLASS_DIR} has a .git and a HEAD but is missing include/cutlass/cutlass.h — the submodule checkout is EMPTY or partial (a real a100e incident: another unit's push deleted its content out from under it); refusing the copy" >&2; exit 1; }
CUTLASS_TREE_FILE_COUNT="$(git -C "$CUTLASS_DIR" ls-tree -r HEAD --name-only | wc -l | tr -d ' ')"
CUTLASS_DISK_FILE_COUNT="$(find "$CUTLASS_DIR" -type f -not -path '*/.git/*' -not -name '.git' | wc -l | tr -d ' ')"
[ "$CUTLASS_DISK_FILE_COUNT" -ge "$CUTLASS_TREE_FILE_COUNT" ] \
  || { echo "::error::${CUTLASS_DIR}'s on-disk file count (${CUTLASS_DISK_FILE_COUNT}) is LESS than its own pinned commit's tree file count (${CUTLASS_TREE_FILE_COUNT}, from 'git ls-tree -r HEAD') — the checkout is missing files beneath a technically-valid HEAD; refusing the copy" >&2; exit 1; }

# Self-target guard: if SUPER_DIR and TREE_SOURCE_DIR are literally the
# SAME directory (e.g. a caller defaults both to /root/jammi-ai), the
# `rm -rf` below would delete SUPER_DIR's own submodule content before
# `cp -a` could read it FROM that same, now-deleted path — a real
# self-destruction risk, not a hypothetical one. SUPER_DIR's own
# `submodule update` + content validation above already made it correct
# for ITSELF; there is nothing left to copy.
if [ "$(cd "$SUPER_DIR" && pwd)" = "$(cd "$TREE_SOURCE_DIR" && pwd)" ]; then
  echo "cutlass provisioning: source-tree-dir and super-dir are the SAME path (${TREE_SOURCE_DIR}) — SUPER_DIR's own submodule checkout IS the tree's cutlass content; nothing to copy"
  exit 0
fi

ACTUAL_SHA="$(git -C "$CUTLASS_DIR" rev-parse HEAD)"

# round-6 audit item 1: decide the EXPECTED PIN's source before comparing
# — never assume every TREE_SOURCE_DIR was populated by `push` (rsync, no
# .git). A tree that is ITSELF a real git checkout (just fetched+checked
# out, or a bundle/clone) already carries its OWN recorded gitlink pin at
# `HEAD:<path>` — read live, with zero staleness risk (it moved WITH the
# checkout, unlike a push-stamp file written before it) — and needs no
# separate stamp. `pod_push_stamp.sh cutlass-check` is still the ONE
# comparison function (never a second copy of the logic): when the pin
# comes from the tree's own index, write it into a throwaway stamp file
# with the SAME shape `cutlass-check` already reads.
STAMP_IS_TEMP=0
# tripwire-ok (both lines below): a missing .gitmodules, or a path
# declared there with no resolvable gitlink, are REAL states this
# decision handles explicitly (the else-branch falls back to the push
# stamp; an unresolvable gitlink on a DECLARED path is refused loudly
# right below) -- never a silent "assume it's a stamp-based tree".
if [ -d "$TREE_SOURCE_DIR/.git" ] && grep -q "$CUTLASS_PATH" "$TREE_SOURCE_DIR/.gitmodules" 2>/dev/null; then
  PIN_SOURCE="the tree's own git index (git-backed tree)"
  TREE_PIN="$(git -C "$TREE_SOURCE_DIR" rev-parse --verify --quiet "HEAD:${CUTLASS_PATH}" || true)" # tripwire-ok: an unresolvable gitlink on a path .gitmodules DOES declare is refused loudly on the very next line, never silently treated as "no pin"
  [ -n "$TREE_PIN" ] || { echo "::error::${TREE_SOURCE_DIR} is git-backed and .gitmodules declares ${CUTLASS_PATH}, but HEAD has no gitlink there — refusing to provision against an unpinned tree" >&2; exit 1; }
  STAMP="$(mktemp)"
  python3 -c 'import json,sys; json.dump({"cutlass_gitlink": sys.argv[1]}, open(sys.argv[2], "w"))' "$TREE_PIN" "$STAMP"
  STAMP_IS_TEMP=1
else
  PIN_SOURCE="the tree's push stamp"
  STAMP="$TREE_SOURCE_DIR/.jammi-push-stamp.json"
fi

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
  echo "cutlass MISMATCH (expected pin source: ${PIN_SOURCE}) — attempting to fetch+checkout the pinned commit ${STAMP_SHA} into the submodule at ${CUTLASS_DIR}..."
  git -C "$CUTLASS_DIR" fetch --depth 1 origin "$STAMP_SHA" \
    && git -C "$CUTLASS_DIR" checkout --quiet "$STAMP_SHA" \
    || { [ "$STAMP_IS_TEMP" = 1 ] && rm -f "$STAMP"; echo "::error::could not fetch/checkout cutlass ${STAMP_SHA} into ${CUTLASS_DIR} (network unreachable?) — refusing the copy" >&2; exit 1; }
  ACTUAL_SHA="$(git -C "$CUTLASS_DIR" rev-parse HEAD)"
  bash "$DIR/pod_push_stamp.sh" cutlass-check "$STAMP" "$ACTUAL_SHA" \
    || { [ "$STAMP_IS_TEMP" = 1 ] && rm -f "$STAMP"; echo "::error::even after fetch+checkout, the SUBMODULE's own HEAD still does not match the pin (source: ${PIN_SOURCE}) — refusing the copy" >&2; exit 1; }
elif [ "$CHECK_RC" -ne 0 ]; then
  [ "$STAMP_IS_TEMP" = 1 ] && rm -f "$STAMP"
  exit 1
fi
[ "$STAMP_IS_TEMP" = 1 ] && rm -f "$STAMP"

mkdir -p "$TREE_SOURCE_DIR/crates/jammi-kernels/third_party"
# shellcheck disable=SC2115  # both vars are non-empty by construction: usage() enforces $1, SUPER_DIR always defaults non-empty, and `set -u` above would already have aborted on an unset one
rm -rf "${TREE_SOURCE_DIR:?}/${CUTLASS_PATH:?}"
cp -a "$CUTLASS_DIR" "$TREE_SOURCE_DIR/$CUTLASS_PATH"

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
# submodule to). Strip it and assert it is gone — this holds EVEN when
# TREE_SOURCE_DIR is itself git-backed and the pin came from its OWN
# index: the git-backed tree's git INDEX still records the correct
# gitlink entry regardless of whether the WORKING TREE at that path
# happens to carry a nested `.git` of its own; leaving one there is a
# strictly worse state (the embedded-repository boundary problem above),
# never a better one.
# shellcheck disable=SC2115  # same as the rm -rf above -- both vars are non-empty by construction
rm -rf "${TREE_SOURCE_DIR:?}/${CUTLASS_PATH:?}/.git"
[ -e "$TREE_SOURCE_DIR/$CUTLASS_PATH/.git" ] \
  && { echo "::error::cutlass/.git still present in the destination tree after stripping — refusing to leave a foreign gitlink in a git-backed tree" >&2; exit 1; }

echo "cutlass provisioned into ${TREE_SOURCE_DIR}/${CUTLASS_PATH} (expected pin source: ${PIN_SOURCE})"
