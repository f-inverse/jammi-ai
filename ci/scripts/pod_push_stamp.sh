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

# round-4 addendum (on-pod incident, a100c A2 run at b3cafda): `shasum` is
# ABSENT on the pod image — a raw `shasum -a 256 ... 2>/dev/null` on a host
# without it produces an EMPTY string via command substitution, no error
# the caller notices, so every hash this file computes would have been
# silently vacuous there. `pod_push_stamp.sh`'s own `compute`/`excludes`
# subcommands run on the laptop (macOS ships `shasum`), but `cutlass-check`
# ships to and runs ON the pod (gpu-dev.sh's `target --with-cutlass`), so
# this file gets the same coreutils-first, shasum-fallback, loud-refusal-
# never-silent-empty helper as pod_seed_target.sh/pod_build_timings.sh —
# duplicated (not sourced from pod_seed_target.sh) to keep this file
# self-contained, matching its existing design.
pod_push_sha256_of_file() { # $1=file (or "-" is not supported; pipe into pod_push_sha256_of_stdin instead)
  if command -v sha256sum >/dev/null 2>&1; then  # tripwire-ok: command -v's own existence probe -- absence is the EXPECTED, checked branch (elif/fallback/error right here), never a silent pass
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then  # tripwire-ok: command -v's own existence probe -- absence is the EXPECTED, checked branch (elif/fallback/error right here), never a silent pass
    shasum -a 256 "$1" | awk '{print $1}'
  else
    echo "::error::pod_push_sha256_of_file: neither sha256sum nor shasum found on PATH — cannot hash $1" >&2
    return 2
  fi
}
pod_push_sha256_of_stdin() {
  if command -v sha256sum >/dev/null 2>&1; then  # tripwire-ok: command -v's own existence probe -- absence is the EXPECTED, checked branch (elif/fallback/error right here), never a silent pass
    sha256sum | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then  # tripwire-ok: command -v's own existence probe -- absence is the EXPECTED, checked branch (elif/fallback/error right here), never a silent pass
    shasum -a 256 | awk '{print $1}'
  else
    echo "::error::pod_push_sha256_of_stdin: neither sha256sum nor shasum found on PATH" >&2
    return 2
  fi
}

# round-5 addendum (round-4 audit addendum "required-tools preflight fails
# loudly" — PARTIAL: present for pod_seed_target.sh/pod_build_timings.sh,
# ABSENT here, the one file that ships to and runs ON the pod). Fails
# loudly, naming every missing tool at once, BEFORE `compute`/`cutlass-
# check` ever runs — never a silent empty hash discovered only by reading
# the stamp JSON later (see new_findings[0] of the round-4 audit verdict:
# a PATH with no sha256sum/shasum previously produced
# `"manifest_sha256": ""` at rc=0).
pod_push_assert_required_tools() {
  local missing="" t
  for t in git python3 rsync stat awk sort; do
    command -v "$t" >/dev/null 2>&1 || missing="${missing}${missing:+ }${t}"  # tripwire-ok: command -v's own existence probe -- absence is the EXPECTED, checked branch (elif/fallback/error right here), never a silent pass
  done
  command -v sha256sum >/dev/null 2>&1 || command -v shasum >/dev/null 2>&1 || missing="${missing}${missing:+ }sha256sum-or-shasum"  # tripwire-ok: command -v's own existence probe -- absence is the EXPECTED, checked branch (elif/fallback/error right here), never a silent pass
  if [ -n "$missing" ]; then
    echo "::error::pod_push_assert_required_tools: missing required tool(s): ${missing}" >&2
    return 1
  fi
}

# round-6 fix (audit item A — the REAL root cause of the manifest_sha256
# nondeterminism a100c/a100e showed on Linux, correcting the round-5
# narrative which blamed LC_ALL alone): GNU `stat -f FORMAT` does NOT
# mean "use this format string" — GNU's `-f` means "display FILE SYSTEM
# status" (the opposite of BSD's `-f`, which IS the format flag). On a
# GNU stat, the OLD fallthrough `stat -f '%Lp' "$f" 2>/dev/null || stat
# -c '%a' "$f"` therefore printed a 5-6 LINE filesystem info block
# (including a live "Free:" block-count line that changes between any
# two invocations, even on the SAME host) to STDOUT, THEN failed
# (rc!=0, since '%Lp' isn't a real file) — `2>/dev/null` only silences
# STDERR, so that multi-line stdout survives into the `||` fallback's
# OWN captured output: reproduced against a real 1,347-file manifest,
# the "mode" field this line captured was 6 lines long (5 fs-status
# lines + the real mode from the correct `stat -c` fallback), inflating
# the manifest to 8,082 lines, and two back-to-back calls differed ONLY
# in the live "Free:" line — the actual source of the a100c/a100e
# divergence. `LC_ALL=C` on the final `sort` is STILL correct and
# load-bearing (a real, independently reproduced cross-locale
# divergence for a DIFFERENT filename-collation reason — see that fix's
# own citation), but it was NOT the cause of the observed a100c/a100e
# nondeterminism; this was. Fixed by detecting the stat FLAVOUR ONCE,
# memoized — never a fallthrough chain whose FAILING branch can still
# emit stdout before failing.
_POD_PUSH_STAT_FLAVOR=""
pod_push_stat_mode() { # $1=file -> file mode (octal, no leading 0), one line, '?' on failure
  local f="$1"
  if [ -z "$_POD_PUSH_STAT_FLAVOR" ]; then
    if stat --version >/dev/null 2>&1; then  # tripwire-ok: stat --version's own existence probe -- absence (non-GNU stat) is the EXPECTED, checked branch (falls to the bsd form), never a silent pass
      _POD_PUSH_STAT_FLAVOR="gnu"
    else
      _POD_PUSH_STAT_FLAVOR="bsd"
    fi
  fi
  # tripwire-ok (both branches below): the flavour was ALREADY resolved
  # above via `stat --version`, so a failure HERE means the file itself
  # is unreadable/missing, not a flag-format mismatch — '?' is a
  # visible, non-empty sentinel (never a silently blank mode field).
  if [ "$_POD_PUSH_STAT_FLAVOR" = "gnu" ]; then
    stat -c '%a' "$f" 2>/dev/null || echo '?' # tripwire-ok: flavour already resolved above; a failure here means the FILE is unreadable/missing, not a flag mismatch -- '?' is a visible sentinel
  else
    stat -f '%Lp' "$f" 2>/dev/null || echo '?' # tripwire-ok: same as the gnu branch above
  fi
}

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

# round-5 fix (a100c on-pod A2 run at 80c7f59, real evidence: two INDEPENDENT
# `git clone`s of the identical bundle at the SAME commit produced two
# DIFFERENT manifest_sha256 values — a100c b2cb2d7a..., a100e 448cc436...
# — while laptop_head/porcelain_sha256/diff_head_sha256/cutlass_gitlink all
# agreed, see the real stamps at scratchpad/a2-timings/80c7f59/{a100c-
# failure,a100e}/.jammi-push-stamp.json — session-local captures, untracked;
# the reproducible tripwire is test_pod_substrate.sh's own locale leg cited
# below). `LC_ALL=C` forces a fixed,
# byte-value collation on the final `sort` regardless of the host's
# ambient locale — a REAL, independently reproduced divergence (two
# locales genuinely sort a crafted filename set differently on the same
# box — see test_pod_substrate.sh's `(v/push revert-RED)` leg) — and
# stays load-bearing for that reason. round-6 correction: it was NOT,
# however, the cause of the SPECIFIC a100c/a100e divergence cited above
# — that was `pod_push_stat_mode`'s own predecessor, a GNU-`stat`-vs-
# BSD-`stat` flag collision that inflated the manifest with live
# filesystem free-block-count lines (see that function's own citation
# for the full mechanism, reproduced against a real 1,347-file
# manifest). Both fixes are real and both stay; this note no longer
# credits the wrong one for the specific incident that motivated it.
pod_push_manifest_sha256() { # $1=repo-root
  local repo="$1" empty manifest rel_count fail_marker
  empty="$(mktemp -d)"
  # A sentinel FILE, not a subshell exit code: the `| while read` loop
  # below runs in a PIPE subshell, so a bare `exit` inside it only ends
  # that subshell — under `pipefail` (this script's own `set -uo
  # pipefail`) the pipeline's reported status would in fact still be
  # `sort`'s (the rightmost command, which itself always exits 0), masking
  # the real failure. A file created inside the subshell is visible to
  # THIS function once the pipeline finishes (subshells share the
  # filesystem), so it is the reliable way to smuggle a failure out of a
  # pipe subshell.
  fail_marker="$(mktemp -u)"
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
        mode="$(pod_push_stat_mode "$f")" # round-6 fix (audit item A): the OLD stat -f/-c fallthrough here was the REAL root cause of the a100c/a100e manifest_sha256 nondeterminism -- see pod_push_stat_mode's own citation
        sha="$(pod_push_sha256_of_file "$f")" || { echo "::error::pod_push_manifest_sha256: failed to hash ${f}" >&2; touch "$fail_marker"; continue; }
        printf '%s\t%s\t%s\n' "$rel" "$mode" "$sha"
      done | LC_ALL=C sort > "$manifest"
  if [ -e "$fail_marker" ]; then
    rm -f "$fail_marker" "$manifest"; rm -rf "$empty"
    echo "::error::pod_push_manifest_sha256: one or more files failed to hash — see the ::error:: line(s) above; refusing to compute a manifest digest over a partial file set" >&2
    return 1
  fi
  rel_count="$(wc -l < "$manifest" | tr -d ' ')"
  # round-5 fix (A4's own class, sibling producer): an EMPTY manifest
  # (zero non-excluded FILES under repo-root — e.g. a repo-root that does
  # not exist, or one whose entire content is excluded) previously hashed
  # sha256("") and returned it as though it were a real, computed manifest
  # digest — indistinguishable, at the JSON level, from a genuine push.
  # Loud refusal instead: never let an empty match set read as computed.
  if [ "${rel_count:-0}" -eq 0 ]; then
    echo "::error::pod_push_manifest_sha256: empty manifest — rsync's dry-run listing (excludes applied) matched ZERO files under ${repo} — refusing to hash sha256('') as though it were a real manifest digest" >&2
    rm -rf "$empty" "$manifest"
    return 1
  fi
  pod_push_sha256_of_file "$manifest"
  rm -rf "$empty" "$manifest"
}

pod_push_compute() { # $1=repo-root $2=session
  local repo="$1" session="$2" head porcelain_sha diff_sha manifest_sha cutlass_gitlink ts
  pod_push_assert_required_tools || return 1
  head="$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo unknown)" # tripwire-ok: "unknown" is a visible, non-empty sentinel for a non-git repo-root — never a silent empty string
  # round-6 fix (audit item 2 — a real regression from round-5's own
  # tripwire fix): git status/diff on a non-git repo-root FAILS ("fatal:
  # not a git repository"), and that failure used to be silently
  # tolerated by piping straight into the hasher regardless — but hashing
  # an EMPTY stdin (git's own failure produces no stdout) yields
  # sha256(""), a VALID-LOOKING, non-empty hash byte-identical to a
  # genuinely CLEAN tree's own porcelain_sha, sitting beside a real
  # laptop_head. The :227 empty-hash backstop can never catch this
  # (sha256("") is non-empty) — the exact "empty reads as computed" class
  # A4/A2's own fixes closed elsewhere, reopened here. Discriminated by
  # whether `head` itself resolved (computed one line up, from the SAME
  # $repo): if HEAD has no answer, this repo-root is not a real checked-
  # out git repo at all — the SAME condition head's own "unknown"
  # sentinel already names, and status/diff failing there is EXPECTED,
  # tolerated the same way. If HEAD DID resolve (a real repo, real
  # history) but status/diff STILL fail — a locked/unreadable index, a
  # corrupted ref, a concurrent git process — that is a genuine,
  # unexpected failure that must FAIL LOUDLY, never silently read as a
  # valid-looking sha256("").
  if [ "$head" = "unknown" ]; then
    porcelain_sha="unknown"
    diff_sha="unknown"
  else
    porcelain_out="$(git -C "$repo" status --porcelain 2>&1)"; porcelain_rc=$?
    if [ "$porcelain_rc" -ne 0 ]; then
      echo "::error::pod_push_compute: 'git status --porcelain' failed in a real repo (HEAD=${head}, rc=${porcelain_rc}): ${porcelain_out}" >&2
      return 1
    fi
    porcelain_sha="$(printf '%s' "$porcelain_out" | pod_push_sha256_of_stdin)"

    diff_out="$(git -C "$repo" diff HEAD 2>&1)"; diff_rc=$?
    if [ "$diff_rc" -ne 0 ]; then
      echo "::error::pod_push_compute: 'git diff HEAD' failed in a real repo (HEAD=${head}, rc=${diff_rc}): ${diff_out}" >&2
      return 1
    fi
    diff_sha="$(printf '%s' "$diff_out" | pod_push_sha256_of_stdin)"
  fi
  manifest_sha="$(pod_push_manifest_sha256 "$repo")" || { echo "::error::pod_push_compute: pod_push_manifest_sha256 failed — see the ::error:: above; refusing to emit a stamp with a missing/empty manifest_sha256" >&2; return 1; }
  # round-5 fix (class-shaped tripwire): loud, never-silent-empty is now
  # required for EVERY hash this stamp carries, not just manifest_sha256 —
  # porcelain_sha/diff_sha above already `return 1` on a hashing failure;
  # this asserts none of the THREE hashed fields came back empty (the
  # shape a missing sha256sum/shasum on PATH produces at every call site
  # at once, reproduced in scratchpad/audit-pb-r4/push/ — a session-local
  # capture, untracked).
  if [ -z "$porcelain_sha" ] || [ -z "$diff_sha" ] || [ -z "$manifest_sha" ]; then
    echo "::error::pod_push_compute: refusing to emit a stamp with an empty hash field (porcelain_sha256='${porcelain_sha}' diff_head_sha256='${diff_sha}' manifest_sha256='${manifest_sha}') — a hashing tool is likely missing from PATH; see pod_push_assert_required_tools above" >&2
    return 1
  fi
  # round-6 fix (audit item 3): bare `git rev-parse HEAD:<path>` on a
  # MISSING path does not fail silently with empty output — it ECHOES
  # its own argument text to STDOUT (rc=128), reproduced directly:
  # `git rev-parse HEAD:no/such/path 2>/dev/null` on a real repo prints
  # the literal string "HEAD:no/such/path". The old `2>/dev/null || true`
  # form let that literal string become `cutlass_gitlink`, corrupting the
  # tri-state this field feeds `pod_push_cutlass_matches`/cutlass-check:
  # a non-empty (but bogus) cutlass_gitlink reads as "stamp HAS a pin",
  # so a genuine cutlass-check mismatch against the REAL submodule sha
  # returns 1 ("genuine mismatch"), not 2 ("no usable stamp") — and
  # `target --with-cutlass`'s remediation arm then tries to fetch+checkout
  # the bogus refspec "HEAD:no/such/path" into the submodule. `--verify
  # --quiet` is the correct form: it prints NOTHING and returns 1 on a
  # missing/unresolvable path, never echoing the argument — exactly the
  # "no gitlink at this path" case cutlass_gitlink is documented as
  # nullable for.
  cutlass_gitlink="$(git -C "$repo" rev-parse --verify --quiet "HEAD:crates/jammi-kernels/third_party/cutlass" 2>/dev/null || true)" # tripwire-ok: --verify --quiet is silent (empty stdout) on a missing path, never echoing the argument — a non-gitlink path (no cutlass submodule at HEAD) is a REAL, valid state; cutlass_gitlink is documented as nullable (compute's own JSON emits `None` for it)
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
' "$stamp" 2>/dev/null)" # tripwire-ok: the python body already catches the ONLY exception this can raise (malformed JSON) explicitly via try/except -> sys.exit(0) with empty stdout; 2>/dev/null only suppresses python's own already-handled traceback noise, never a real diagnostic — the empty-stdout case is itself checked (stamp_sha empty -> return 2) right below
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
