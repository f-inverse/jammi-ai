#!/usr/bin/env bash
# Builds and cleans a MEMBER-FREE pod build-substrate seed: a CARGO_TARGET_DIR
# whose registry/dependency artifacts are fully built (so a clone of it never
# re-compiles a single third-party crate) but carries ZERO workspace-member
# artifacts (so a clone always re-compiles jammi's own code from the CLONE's
# actual source, never links in a stale copy of it from the seed — the exact
# drift class round-4's pressure-test reproduced against `cargo clean -p`,
# which enumerates from the CLONE's source and therefore misses a
# renamed/deleted target's own leftover seed artifact).
#
# Dev-loop only — this is never part of the CI prove lane (runpod_gpu_prove.sh
# builds cold, on its own throwaway pod, by design: a poisoned seed must never
# be able to pass the gate that is supposed to catch it).
#
# Tuples built into the seed (every one of these becomes fully-compiled
# third-party artifacts, and every workspace-member artifact from building
# them is then swept by the member-free clean below):
#   T1  cargo build --release -p jammi-bench --features cuda
#       (+ --features cuda,jammi-kernels/flash-attn once, ONLY when the pod's
#        checkout is on `main` AND jammi-kernels actually declares a
#        flash-attn feature (detected live via `cargo metadata`, never
#        hand-asserted — see pod_seed_pkg_has_feature) — flash-attn
#        additionally compiles the vendored FlashAttention-2 CUTLASS
#        kernels, real nvcc minutes, so it is not paid on every branch's
#        seed)
#   T2  cargo test --no-run for the exact crates/features
#       runpod_gpu_prove.sh's own suites use (kept in lockstep with that
#       script by naming the same -p/--features/--test here)
#   T3  cargo clippy -p jammi-kernels --all-targets --features cuda
#
# After all tuples build: capture every build script's stdout (the announced
# env surface) BEFORE cleaning, member-free-clean the seed (cargo clean
# --workspace, both profiles used above, PLUS an explicit `rm -rf
# */incremental` since cargo's own cleaner does not remove
# `incremental/build_script_build-*`), assert cargo metadata shows no
# non-member path/patch package, cross-check the captured announcement
# against pod_seed_key_inputs.toml, then write the completion marker.
#
# Usage: pod_seed_target.sh [--reseed] [--no-lock]
# Env:
#   JAMMI_SEED_DIR       seed CARGO_TARGET_DIR (default /root/.jammi-seed)
#   JAMMI_TREE_DIR        the source tree to build FROM (default /root/jammi-ai)
#   JAMMI_SEED_LOCK_WAIT_SECS  seconds to wait for pod_timing_lock.sh when this
#                         script was NOT already invoked from inside a
#                         lock-held tmux pane (default 1800; see --no-lock)
# --reseed: rebuild even if .jammi-seed-complete already exists.
# --no-lock: skip this script's own pod_timing_lock.sh wrap — for the ONE
#   caller that is already running inside a lock-held tmux pane (`run
#   --timing`'s own launcher command line acquires the lock BEFORE this
#   script starts; re-acquiring the SAME flock from a child process would
#   deadlock against the parent that already holds it).
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

JAMMI_SEED_DIR="${JAMMI_SEED_DIR:-/root/.jammi-seed}"
JAMMI_TREE_DIR="${JAMMI_TREE_DIR:-/root/jammi-ai}"
JAMMI_SEED_LOCK_WAIT_SECS="${JAMMI_SEED_LOCK_WAIT_SECS:-1800}"
MANIFEST="${JAMMI_SEED_MANIFEST:-$DIR/pod_seed_key_inputs.toml}"

COMPLETE_MARKER="${JAMMI_SEED_DIR}.jammi-seed-complete"
FAILED_MARKER="${JAMMI_SEED_DIR}.jammi-seed-failed"

# round-4 addendum (on-pod incident, a100c A2 run at b3cafda): `shasum` is
# ABSENT on the pod's own image. `shasum -a 256 "$f" 2>/dev/null | awk
# '{print $1}'` on a host with no `shasum` binary produces an EMPTY string
# via command substitution — no error surfaces to the caller, so
# pod_seed_target.sh's own manifest_sha256 and pod_build_timings.sh's
# byte-equality hashes were BOTH silently vacuous on the real pod (the same
# empty-match-set vacuity round-4 audit A4 fixed for the byte-equality
# comparison itself — an empty hash must never read as "computed", let
# alone "matched"). Prefer coreutils `sha256sum` (present on the pod
# image); fall back to `shasum -a 256` (present on macOS dev/CI hosts,
# absent on the pod) only if `sha256sum` itself is missing; loudly refuse
# (rc=2, never a silent empty string) if NEITHER exists.
pod_sha256_of_file() { # $1=file
  if command -v sha256sum >/dev/null 2>&1; then  # tripwire-ok: command -v's own existence probe -- absence is the EXPECTED, checked branch (elif/fallback/error right here), never a silent pass
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then  # tripwire-ok: command -v's own existence probe -- absence is the EXPECTED, checked branch (elif/fallback/error right here), never a silent pass
    shasum -a 256 "$1" | awk '{print $1}'
  else
    echo "::error::pod_sha256_of_file: neither sha256sum nor shasum found on PATH — cannot hash $1" >&2
    return 2
  fi
}
pod_sha256_of_stdin() {
  if command -v sha256sum >/dev/null 2>&1; then  # tripwire-ok: command -v's own existence probe -- absence is the EXPECTED, checked branch (elif/fallback/error right here), never a silent pass
    sha256sum | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then  # tripwire-ok: command -v's own existence probe -- absence is the EXPECTED, checked branch (elif/fallback/error right here), never a silent pass
    shasum -a 256 | awk '{print $1}'
  else
    echo "::error::pod_sha256_of_stdin: neither sha256sum nor shasum found on PATH" >&2
    return 2
  fi
}

# round-4 addendum: a preflight that asserts every external tool the seed
# build actually calls exists BEFORE spending real compile minutes,
# failing loudly and NAMING every missing tool at once (never one-at-a-
# time discovery via a cryptic mid-build "command not found" thirty
# minutes in). Scoped to what pod_seed_target.sh itself calls — flock's
# own presence is already asserted, loudly, by pod_timing_lock.sh itself
# (this script's own re-exec wrapper) whenever the lock path is actually
# taken, so it is not duplicated here.
pod_seed_assert_required_tools() {
  local missing="" t
  for t in cargo git python3; do
    command -v "$t" >/dev/null 2>&1 || missing="${missing}${missing:+ }${t}"  # tripwire-ok: command -v's own existence probe -- absence is the EXPECTED, checked branch (elif/fallback/error right here), never a silent pass
  done
  command -v sha256sum >/dev/null 2>&1 || command -v shasum >/dev/null 2>&1 || missing="${missing}${missing:+ }sha256sum-or-shasum"  # tripwire-ok: command -v's own existence probe -- absence is the EXPECTED, checked branch (elif/fallback/error right here), never a silent pass
  if [ -n "$missing" ]; then
    echo "::error::pod_seed_assert_required_tools: missing required tool(s): ${missing}" >&2
    return 1
  fi
}

# ── manifest parsing / RED-test-shared logic ────────────────────────────────
# Every array literal in pod_seed_key_inputs.toml (inputs/computed_forms/
# vars/commands) is a flat list of double-quoted strings on lines between the
# key and the closing `]`. A hand-rolled scan rather than a TOML library: the
# manifest's own shape is fixed and simple, and this avoids depending on a
# specific Python's tomllib availability on every pod image.
#
# Prints one name per line: every literal name, PLUS the non_key wildcard
# entries verbatim (e.g. "CARGO_FEATURE_*"), from every section.
pod_seed_manifest_names() { # $1=manifest toml path
  python3 - "$1" <<'PY'
import sys, re
text = open(sys.argv[1], encoding="utf-8").read()
# Strip TOML comments (a line's content after an unescaped '#') so a citation
# containing a stray '#' cannot corrupt the array scan below.
lines = []
for line in text.splitlines():
    out, in_str = [], False
    i = 0
    while i < len(line):
        c = line[i]
        if c == '"' and (i == 0 or line[i - 1] != '\\'):
            in_str = not in_str
        if c == '#' and not in_str:
            break
        out.append(c)
        i += 1
    lines.append("".join(out))
text = "\n".join(lines)
# Every array literal: key = [ ...multi-line... ]
for m in re.finditer(r'=\s*\[(.*?)\]', text, re.S):
    for s in re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1)):
        print(s)
PY
}

# name allowed <=> exact match in the manifest's flat name set, OR matches a
# "PREFIX*" wildcard entry (only CARGO_FEATURE_* exists today, but the match
# is generic).
pod_seed_name_allowed() { # $1=name $2=manifest-names-file (one per line)
  local name="$1" names_file="$2" line
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    case "$line" in
      *'*')
        case "$name" in "${line%\*}"*) return 0 ;; esac ;;
      "$name") return 0 ;;
    esac
  done < "$names_file"
  return 1
}

# RED test (i)/(ii) shared scanner: every name-shaped string literal
# ([A-Z][A-Z0-9_]{2,}) in $1, classified as manifest-listed or not. With
# MODE=rerun_only, restricts to `cargo:rerun-if-env-changed=<NAME>` literals
# only (test (ii)'s narrower scope). Prints one UNLISTED name per line to
# stdout; empty output = every literal found is accounted for.
pod_seed_scan_source() { # $1=source file $2=names-file $3=mode(all|rerun_only)
  python3 - "$1" "$3" <<'PY' | while IFS= read -r name; do
import sys, re
path, mode = sys.argv[1], sys.argv[2]
src = open(path, encoding="utf-8", errors="replace").read()
if mode == "rerun_only":
    names = re.findall(r'rerun-if-env-changed=([A-Z][A-Z0-9_]{2,})', src)
else:
    names = re.findall(r'"([A-Z][A-Z0-9_]{2,})"', src)
for n in sorted(set(names)):
    print(n)
PY
    pod_seed_name_allowed "$name" "$2" || echo "$name"
  done
}

# RED test (iii): the seed's own runtime cross-check. $1=dir containing
# captured `<profile>__<pkg-dirname>.output` files (see capture_build_output
# below; test_pod_substrate.sh's own (n4) block points this at fixtures).
# Every `cargo:rerun-if-env-changed=<NAME>` line actually announced by a
# real build-script run must be in the manifest. Prints unlisted names.
#
# round-3 audit N4: an EMPTY capture dir (glob matches nothing — bash
# leaves the literal `dir/*` pattern unexpanded, `[ -f "$f" ]` on that
# literal fails, `continue` skips every iteration) previously fell straight
# through the loop with bad=0 — a seed whose capture step produced nothing
# was stamped complete having checked NOTHING. `capture_count` — real files
# actually iterated — must be >= 1: this IS still checked below (empty
# CAPTURE DIR, never a valid pass).
#
# round-5 correction (a100c on-pod A2 run at 80c7f59, real evidence at
# scratchpad/a2-timings/80c7f59/a100c-failure/a2c.stdout — a session-local
# capture, untracked): a PRIOR round's
# fix additionally flagged every INDIVIDUAL zero-byte captured `output`
# file as an error ("captured at the wrong moment"). That assumption was
# FALSE: cargo creates a `build/<pkg>-*/output` file for every build
# script it actually runs, REGARDLESS of whether that script prints
# anything to stdout — a real seed build on this workspace's own
# `--features jammi-kernels/cuda` graph legitimately captures a zero-byte
# `output` for at least chrono-tz, esaxx-rs, pulldown-cmark, rustls,
# scratch, snap, stacker, and prometheus (build scripts whose ENTIRE job is
# a compile-time codegen step or a `println!("cargo:rustc-cfg=...")`-free
# no-op — nothing `cargo:`-shaped to announce), and the file EXISTING (even
# at zero bytes) is exactly the evidence the capture step ran at the RIGHT
# moment, not the wrong one. Flagging these as errors is a false positive
# that would abort every real seed build on this workspace. The honest
# rule: a captured file's mere EXISTENCE (checked via `capture_count`,
# still required to be >= 1 in aggregate below) is the "capture ran"
# witness; per-file byte count carries no information on its own.
pod_seed_check_stdout_subset() { # $1=capture-dir $2=manifest-toml
  local capture_dir="$1" manifest="$2" names_file f name bad=0 capture_count=0
  names_file="$(mktemp)"
  pod_seed_manifest_names "$manifest" > "$names_file"
  [ -d "$capture_dir" ] || { echo "::error::no build-output capture dir at ${capture_dir}" >&2; rm -f "$names_file"; return 1; }
  for f in "$capture_dir"/*; do
    [ -f "$f" ] || continue
    capture_count=$((capture_count + 1))
    while IFS= read -r name; do
      [ -n "$name" ] || continue
      pod_seed_name_allowed "$name" "$names_file" || { echo "$name (from $(basename "$f"))"; bad=1; }
    done < <(grep -o 'cargo:rerun-if-env-changed=[A-Za-z_][A-Za-z0-9_]*' "$f" | sed 's/^cargo:rerun-if-env-changed=//')
  done
  if [ "$capture_count" -eq 0 ]; then
    echo "::error::cross-check saw no build-script output at all (capture_count=0) — the seed's own capture step produced nothing to check; this must never read as a pass" >&2
    bad=1
  fi
  rm -f "$names_file"
  return "$bad"
}

# Copy every build script's captured stdout out of the seed BEFORE cleaning —
# cargo's own cleaner removes `build/<pkg>-*/output` along with the rest of
# `build/`, so this is the only chance to read it. $3 is the ACTUAL profile
# subdirectory name (debug|release) — the glob is scoped to exactly that
# subtree, never a `*` at that position: an unscoped `*/build/*/output`
# glob matches BOTH debug/ and release/ regardless of which profile_label
# the caller passed, so two calls (one per profile) would each capture the
# SAME full (debug+release) file set under two differently-labelled copies
# — duplicated content, never a genuine per-profile split. Caught by
# inspection (audit-round 2 advisory), not by a fixture that only ever saw
# one profile's build directory.
pod_seed_capture_build_output() { # $1=seed target dir $2=dest capture dir $3=profile subdir (debug|release)
  local seed="$1" dest="$2" profile_label="$3" d base
  mkdir -p "$dest"
  for d in "$seed/$profile_label"/build/*/output; do
    [ -f "$d" ] || continue
    base="$(basename "$(dirname "$d")")"
    cp "$d" "$dest/${profile_label}__${base}.output"
  done
}

# Detects, rather than assumes, whether <pkg> declares a feature named
# <feature> — read live from `cargo metadata`, never hand-asserted. This is
# what T1b's flash-attn leg (below) and pod_build_timings.sh's own FA2 leg
# both gate on: `--features cuda,jammi-encoders/flash-attn` was wrong on its
# face (jammi-encoders declares no such feature; flash-attn lives on
# jammi-kernels, forwarded through jammi-bench's own direct dependency on
# it) — a hardcoded feature PATH string is exactly the kind of assumption
# that silently rots when a feature moves crates; detecting it converts that
# rot into "T1b skipped" instead of "the default pod's seed always fails".
# Returns 0 (declared) / 1 (package found, feature genuinely NOT declared)
# / 2 (could not determine at all — the metadata query failed, or the
# package itself was not found in the graph). round-3 audit Class B: codes
# 1 and 2 used to be the SAME code, so a caller could not distinguish "this
# feature really doesn't exist" from "I have no idea, the query broke".
# round-4 addendum: the two callers now diverge on rc=2, deliberately. The
# SEED's own T1b gate (pod_seed_target_main) ABORTS the whole seed on rc=2
# — a broken metadata query silently downgraded to "feature absent" is
# exactly how the on-pod incident stamped a seed complete WITHOUT its FA2
# artifacts. pod_build_timings.sh's OWN FA2 *measurement* leg (a separate,
# additional clone+build purely for A2's timing acceptance, not the seed
# itself) still WARNS and skips on rc=2 — that call decides only whether to
# additionally measure an optional metric, never whether the seed the run
# already validated (via its own real `pod_seed_target.sh --no-lock`
# invocation at step (i), which now aborts on the same rc=2) is valid.
# round-4 addendum (on-pod incident, a100c A2 run at b3cafda): every
# `--frozen` metadata call site in this file used to run `2>/dev/null`,
# discarding the ACTUAL cargo error — the real failure was `error: failed
# to download android_system_properties v0.1.5 — attempting to make an
# HTTP request, but --frozen was specified` (rc 101; `cargo metadata`
# resolves the FULL cross-platform dependency graph by default, which
# needs source for platform-conditional crates the pod's own build never
# fetches) — and left only an empty string for every caller to puzzle
# over, with no diagnosis anywhere. One seam: every `--frozen` metadata
# call in this file goes through this function, which captures stderr and
# treats non-zero exit OR empty stdout as failure (cargo can print nothing
# useful to stdout while still degenerate-exiting 0), printing the exact
# command and the real stderr — never silently returning empty. $@ = extra
# `cargo metadata` args (e.g. --features jammi-kernels/cuda). Prints
# metadata JSON to stdout on success; prints nothing to stdout and returns
# 2 on failure (callers already treat "no valid metadata" as "could not
# determine", never as a hand-asserted "genuinely absent").
pod_seed_cargo_metadata_frozen() {
  local out err
  err="$(mktemp)"
  if ! out="$(cargo metadata --frozen --format-version 1 "$@" 2>"$err")" || [ -z "$out" ]; then
    echo "::error::cargo metadata --frozen --format-version 1 $* failed (or produced no output) — real stderr:" >&2
    cat "$err" >&2
    rm -f "$err"
    return 2
  fi
  rm -f "$err"
  printf '%s' "$out"
}

pod_seed_pkg_has_feature() { # $1=pkg $2=feature
  pod_seed_cargo_metadata_frozen | python3 -c '
import sys, json
pkg, feat = sys.argv[1], sys.argv[2]
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(2)
for p in d.get("packages", []):
    if p["name"] == pkg:
        sys.exit(0 if feat in (p.get("features") or {}) else 1)
sys.exit(2)
' "$1" "$2"
}

# FILESYSTEM-LEVEL member-freedom check (round-3 audit N2, pattern fixed by
# round-4 audit A2). pod_seed_target.sh always documented "member-free
# seed" but never actually verified it at the one place that matters — the
# target dir's own contents. The metadata-only check elsewhere in this file
# (no non-member path/patch PACKAGE — a Cargo.lock/[patch] hygiene
# question) is the OPPOSITE direction and cannot catch a member's own
# compiled artifact surviving a clean; the incremental/ emptiness check
# covers exactly one subdirectory cargo's own cleaner is already known to
# miss (this file's own module doc, above). This is the ONE mechanical,
# always-on definition: after ANY clean or clone, no
# `{debug,release}/{.fingerprint,deps,build,incremental}` entry may be named
# after a WORKSPACE MEMBER, where "workspace member" is read from `cargo
# metadata`'s own `workspace_members` — never a "jammi-*" glob/prefix guess,
# so a member crate that happened not to start with "jammi-" would still be
# caught.
#
# round-4 audit A2: the round-3 version checked only cargo's hyphenated
# form (.fingerprint/build) and bare underscored form (deps/ NON-library
# entries) — it MISSED every crate's own COMPILED LIBRARY, named
# `lib<underscored>-<hash>.rlib`/`.rmeta` (a "lib" PREFIX glued onto the
# underscored form). The round-3 doc comment claimed this was "verified
# against a real cargo build/clean cycle" — that claim was FALSE (the only
# fixture ever built was a BINARY crate, which has no [lib] target and
# therefore no .rlib/.rmeta output at all, so the gap could never have
# shown up in it). Reproduced for real before fixing (jammi_seed_target.sh
# probe against a genuine `cargo build` + `cargo build --release` of a
# library crate "jammi-zzlib"):
#   $ CARGO_TARGET_DIR=tgt cargo build -q && CARGO_TARGET_DIR=tgt cargo build --release -q
#   $ pod_seed_assert_member_free tgt .   # round-3 pattern
#   -> rc=1, but 4/4 real files matching `find tgt -name 'libjammi_zzlib-*'`
#      (debug+release .rlib/.rmeta) are ABSENT from the printed violation
#      list — invisible to the scanner despite genuinely existing.
# Fixed by adding the "lib"+underscored stem; the SAME probe against the
# SAME fixture directory now lists all four .rlib/.rmeta paths.
#
# round-5 audit (family O — a comment may not claim coverage that does not
# exist): the line above used to claim "the hermetic test for this
# function (test_pod_substrate.sh) builds this exact real library-crate
# fixture itself, rather than asserting from a written claim" — that claim
# was ITSELF false for two consecutive rounds (round-4 audit finding;
# round-4's own attempted fix restated the same false claim without
# closing it). It is now true, and the standing rule this round enforces
# mechanically (test_pod_substrate.sh's own claim-tripwire) is what keeps
# it true going forward: test_pod_substrate.sh's `(q/A2)` leg builds a real
# two-member cargo workspace (lib jammi-zzlib + bin jammi-zzbin), runs a
# REAL `cargo build` + `cargo build --release`, takes its artifact list
# from a REAL `find` (never a hand-typed filename), and asserts every real
# lib*.rlib/.rmeta/.d/.fingerprint/build entry trips this function — and
# that after `cargo clean --workspace` (both profiles) + the incremental/
# rm, none do. $1=target_dir (a CARGO_TARGET_DIR — debug/ and release/ scanned)
# $2=tree_dir (where `cargo metadata` resolves the workspace; default cwd).
# Prints every violating path and fails loudly; never opt-in, run
# UNCONDITIONALLY from pod_seed_target.sh (before the completion stamp),
# pod_target_clone.sh (right after every clone), and pod_build_timings.sh
# (before T1) — pod_target_clone.sh's old `--verify` (a `cargo build -v` log
# grep) stays as an ADDITIONAL, opt-in form a human can still run after a
# real build; it is not replaced, since it catches a DIFFERENT thing (the
# clone's OWN first build actually recompiling, not merely "no leftover
# artifact from the seed").
pod_seed_assert_member_free() { # $1=target_dir $2=tree_dir (optional; default .)
  local target_dir="$1" tree_dir="${2:-.}" meta
  [ -d "$target_dir" ] || { echo "::error::pod_seed_assert_member_free: no such target_dir: ${target_dir}" >&2; return 2; }
  # round-4 audit new_findings (guard-state-collapse, folded in round 5):
  # `[ -d "$target_dir" ]` alone accepts ANY existing directory, including
  # one with NEITHER a debug/ NOR a release/ subtree at all — a degenerate
  # target_dir (e.g. an empty dir right after `mkdir -p`, before a single
  # cargo command has ever run against it) yields bad=[] below and rc=0: a
  # VACUOUS pass at the clone gate (pod_target_clone.sh, right after `cp
  # -a`), the exact "empty match set reads as a computed pass" shape A4
  # fixed for byte-equality and N4 fixed for the env-surface capture, never
  # carried across to this sibling function. `capture_count >= 1`-style
  # non-vacuity: at least ONE of the two cargo profile subdirectories must
  # actually exist before "no member-named entry found" is allowed to mean
  # anything.
  if [ ! -d "$target_dir/debug" ] && [ ! -d "$target_dir/release" ]; then
    echo "::error::pod_seed_assert_member_free: ${target_dir} has NEITHER debug/ NOR release/ — this is not a built CARGO_TARGET_DIR at all, so 'member-free' is meaningless (a vacuous pass), not a genuine check" >&2
    return 2
  fi
  meta="$(cd "$tree_dir" && pod_seed_cargo_metadata_frozen)"
  if [ -z "$meta" ]; then
    echo "::error::pod_seed_assert_member_free: cargo metadata produced no output from ${tree_dir} — registry not fetched?" >&2
    return 2
  fi
  printf '%s' "$meta" | python3 -c '
import sys, json, os, re
target_dir = sys.argv[1]
d = json.load(sys.stdin)
members_by_id = {p["id"]: p["name"] for p in d["packages"]}
# round-4 audit A2 (reproduced against a REAL cargo library build — see
# the `(q/A2)` leg in test_pod_substrate.sh for the permanent, executable
# form): the OLD stem set was {hyphenated, underscored} only, which matches
# .fingerprint/build (hyphenated) and deps/incremental NON-library entries
# (underscored, e.g. jammi_zzlib-<hash>.d) — but the COMPILED LIBRARY
# output of a crate is named with a "lib" PREFIX glued directly onto the
# underscored form: libjammi_zzlib-<hash>.rlib / .rmeta. The old pattern
# required the basename to START WITH the member name; "libjammi_zzlib-..."
# starts with "lib", not "jammi", so EVERY .rlib/.rmeta for EVERY library
# member was invisible — reproduced: 4 real .rlib/.rmeta files from a real
# cargo build plus cargo build --release went unflagged before this fix
# (now caught, same fixture). Stems are therefore {hyphenated, underscored,
# "lib" + underscored} per member — derived from cargo naming rules, not a
# guess.
member_names = set()
for mid in d["workspace_members"]:
    name = members_by_id.get(mid)
    if name:
        underscored = name.replace("-", "_")
        member_names.add(name)
        member_names.add(underscored)
        member_names.add("lib" + underscored)
if not member_names:
    print("::error::pod_seed_assert_member_free: cargo metadata reported ZERO workspace members — refusing to run a vacuous check", file=sys.stderr)
    sys.exit(2)
# A directory ENTRY is member-named if its basename starts with one of the
# stems above followed by a boundary character (cargo separates a crate
# stem from its fingerprint hash/suffix with a hyphen in every artifact
# type observed; `_` is also accepted, defense in depth, since a member
# name is always a specific "jammi-*"-prefixed string with no realistic
# collision risk) — never a bare prefix match, so member "jammi-kernels"
# does not false-positive-match an unrelated "jammi-kernels-utils" entry
# that merely shares the prefix.
pat = re.compile(r"^(" + "|".join(re.escape(n) for n in sorted(member_names, key=len, reverse=True)) + r")[-_]")
bad = []
for profile in ("debug", "release"):
    for sub in (".fingerprint", "deps", "build", "incremental"):
        d2 = os.path.join(target_dir, profile, sub)
        if not os.path.isdir(d2):
            continue
        for entry in os.listdir(d2):
            if pat.match(entry):
                bad.append(os.path.join(d2, entry))
if bad:
    print("::error::member-named artifact(s) survived — NOT member-free:", file=sys.stderr)
    for b in sorted(bad):
        print("  " + b, file=sys.stderr)
    sys.exit(1)
' "$target_dir"
}

# RED tests (i)/(ii), FULL scope (round-2 audit finding 5): every package
# with a build script in the RESOLVED dependency graph — not a hand-picked
# subset of three files. `--features jammi-kernels/cuda` brings bindgen_cuda
# (a build-DEPENDENCY consumed from inside jammi-kernels' own build.rs, so
# it carries no `custom-build` target of its own and is scanned separately,
# see the caller) and cudarc (which DOES carry its own build.rs, and reads
# CUDA_HOME/CUDA_PATH/CUDA_ROOT/CUDA_TOOLKIT_ROOT_DIR/CUDNN_LIB/
# CUDARC_CUDA_VERSION/CONDA_PREFIX — a real, previously-unlisted CUDA-toolchain
# input this full enumeration is what actually catches) into the graph.
#
# A package is "cc-allowlisted" — its OWN build.rs literals are skipped,
# never individually scanned — IFF it depends on the `cc` crate, checked
# from THIS SAME metadata graph (never a hardcoded package-name list, so a
# newly-added cc-based dependency is covered automatically): such a
# package's build.rs typically just calls `cc::Build::new()...compile()`,
# and its literals are C preprocessor macro / `#[cfg(...)]` names `cc`
# itself consumes (`Build::define()`), not env reads — cc's OWN env surface
# is [cc_1_2_57] in the manifest, hand-enumerated separately (see that
# section's own citation for why it cannot be scanned mechanically).
#
# Prints "<name> (from <pkg>)" per unlisted literal, and a one-line
# "<N> package(s) individually scanned, <M> cc-allowlisted" summary to
# stderr (what a RED test reports as its scanned count).
pod_seed_scan_all_vendored_buildrs() { # $1=manifest-toml $2=mode(all|rerun_only)
  local manifest="$1" mode="$2" names_file bad=0 scanned=0 allowlisted=0 fetch_rc=0
  names_file="$(mktemp)"
  pod_seed_manifest_names "$manifest" > "$names_file"

  local meta; meta="$(pod_seed_cargo_metadata_frozen --features jammi-kernels/cuda)"
  if [ -z "$meta" ]; then
    # Self-heal once (round-2 audit finding 6): a bare/fresh checkout with
    # no registry fetch yet is the ordinary shape on a maintainer's machine
    # or a CI runner that skipped the fetch step — `cargo fetch --locked`
    # is offline-safe to attempt (it only pulls what Cargo.lock already
    # pins) and cheap when already warm. Fail loudly, naming the real cause,
    # only if the retry ALSO comes up empty. (round-4 addendum: this retry
    # is now defense-in-depth — pod_seed_target_main's own one-time,
    # network-allowed `cargo metadata --locked` priming call, run before
    # T1, is meant to make this branch unreachable in the real seed build;
    # this function is also called standalone by this suite's own RED
    # tests, which have no such priming step run first.)
    echo "cargo metadata produced no output — attempting 'cargo fetch --locked' once, then retrying" >&2
    # round-5 addendum (advisory folded, prior round): the OLD form
    # (`cargo fetch --locked >&2 2>&1`, rc unread) discarded the fetch's
    # own exit code — a genuine NETWORK failure here was misdiagnosed three
    # lines down as "is a Rust toolchain on PATH?", which is not the real
    # cause and sends a human debugging this the wrong direction. `fetch_rc`
    # is captured explicitly and both branches below cite the ACTUAL
    # command that failed.
    fetch_rc=0
    cargo fetch --locked >&2 2>&1 || fetch_rc=$?
    meta="$(pod_seed_cargo_metadata_frozen --features jammi-kernels/cuda)"
  fi
  if [ -z "$meta" ]; then
    if [ "${fetch_rc:-0}" != 0 ]; then
      echo "::error::cargo metadata produced no output; the self-heal 'cargo fetch --locked' ALSO failed (rc=${fetch_rc}) — see cargo's own stderr above for the real cause, not necessarily a missing toolchain" >&2
    else
      echo "::error::cargo metadata produced no output even after 'cargo fetch --locked' succeeded — is a Rust toolchain on PATH?" >&2
    fi
    rm -f "$names_file"
    return 2
  fi

  while IFS=$'\t' read -r pkg src_path is_cc; do
    [ -n "$pkg" ] || continue
    case "$pkg" in jammi-kernels|jammi-wire) continue ;; esac
    if [ "$is_cc" = "1" ]; then
      allowlisted=$((allowlisted + 1))
      continue
    fi
    scanned=$((scanned + 1))
    if [ ! -f "$src_path" ]; then
      echo "::error::build.rs for ${pkg} not found at ${src_path}" >&2
      bad=1
      continue
    fi
    while IFS= read -r unlisted; do
      [ -n "$unlisted" ] || continue
      echo "${unlisted} (from ${pkg})"
      bad=1
    done < <(pod_seed_scan_source "$src_path" "$names_file" "$mode")
  done < <(printf '%s' "$meta" | python3 -c '
import sys, json
d = json.load(sys.stdin)
# Keyed by src_path (not bare name): Cargo.lock can legitimately pin TWO
# different versions of the same crate (a diamond dependency) with
# DIFFERENT build.rs content (cudarc 0.17.8 vs 0.19.8 in this graph carry
# different env surfaces) — a name-keyed dict silently drops every version
# but the last one iterated, scanning less than the real graph.
seen = {}
for p in d["packages"]:
    for t in p.get("targets", []):
        if t.get("kind") == ["custom-build"]:
            seen[t["src_path"]] = (p["name"], set(dd["name"] for dd in p.get("dependencies", [])))
for src, (name, deps) in sorted(seen.items()):
    print("%s\t%s\t%s" % (name, src, "1" if "cc" in deps else "0"))
')
  echo "vendored scan (mode=${mode}): ${scanned} package(s) individually scanned, ${allowlisted} cc-allowlisted" >&2
  rm -f "$names_file"
  return "$bad"
}

# Writes a DIAGNOSTIC-BEARING failure marker (round-4 addendum — on-pod
# incident, a100c A2 run at b3cafda): a plain N-line tail of the captured
# build log is not sufficient. Reproduced against the incident's own shape:
# a `cargo build` whose actual compiler error scrolled off the end of a
# 100-line tail because OTHER crates kept printing "Checking"/"Compiling"
# lines for tens of seconds after the real failure, and an nvcc OOM-kill
# leaves no "error:" line at all — only the shell's own "Killed" report.
# $1=log_path (the captured build log) $2=marker_path (FAILED_MARKER)
# $3=exit_code (the subshell's own $?).
pod_seed_write_failure_marker() {
  local log="$1" marker="$2" rc="$3"
  local dedup; dedup="$(mktemp)"
  # round-5 addendum (a100c on-pod A2 run at 80c7f59, real failure marker
  # at scratchpad/a2-timings/80c7f59/a100c-failure/.jammi-seed.jammi-seed-
  # failed — a session-local capture, untracked): the env-surface
  # cross-check (pod_seed_check_stdout_subset)
  # repeats the SAME "<NAME> (from <file>)" line once per unlisted
  # literal occurrence — the real incident's own tail was 30 copies of the
  # same handful of lines, crowding out the SINGLE real
  # `::error::a build script announced ... absent from ...` line that sat
  # right after them. `uniq` collapses only ADJACENT duplicate lines
  # (never reorders, so a genuine change in content is never hidden) —
  # applied ONCE, up front, so both the diagnostic-pattern grep and the
  # final tail below read the deduplicated log.
  uniq "$log" > "$dedup" 2>/dev/null || cp "$log" "$dedup" # tripwire-ok: uniq is coreutils-standard; a failure here just means "skip dedup, use the raw log" (cp), never a silent empty marker
  local diag
  # round-5 addendum: `::error::`/`::warning::` — this script's OWN loud-
  # refusal convention, used throughout pod_seed_target.sh/pod_push_
  # stamp.sh/pod_build_timings.sh — were ABSENT from the diagnostic-pattern
  # set. The a100c incident's real failure cause was exactly one of these
  # (`::error::a build script announced ... absent from
  # ci/scripts/pod_seed_key_inputs.toml`, pod_seed_target.sh:762) and the
  # marker printed "no line matched the diagnostic patterns" despite the
  # true cause sitting right there in the log — reproduced: `grep -c
  # '::error::' a2c.stdout` (the real incident's own captured stdout)
  # finds 18 real matches the OLD pattern set never saw.
  diag="$(grep -n -E '^(error(\[E[0-9]+\])?:|warning: unused|note: )|failed to |nvcc fatal|^Killed|::error::|::warning::' "$dedup" 2>/dev/null | head -400)" # tripwire-ok: 2>/dev/null is grep's own no-such-file guard on a controlled temp path; a real match failure surfaces as an empty $diag, handled explicitly below, never silently
  local phase
  phase="$(grep -E '^=== ' "$dedup" 2>/dev/null | tail -1)" # tripwire-ok: same as above — empty phase is handled via the ${phase:-...} fallback right below
  {
    echo "seed build FAILED (exit ${rc})"
    echo "phase in progress at failure: ${phase:-<none printed before the failure>}"
    echo "df -h /:"
    df -h / 2>/dev/null || echo "  (df unavailable)" # tripwire-ok: df is a diagnostic nicety, not the failure cause itself — absence is reported, never silent
    echo "free -g (or vm_stat where free is unavailable):"
    free -g 2>/dev/null || vm_stat 2>/dev/null || echo "  (memory snapshot unavailable)" # tripwire-ok: same — best-effort diagnostic, absence reported
    echo "--- diagnostic lines (error/warning/note/failed to/nvcc fatal/Killed/::error::/::warning::), each with 20 lines of trailing context, up to 400 lines total, ADJACENT DUPLICATES COLLAPSED ---"
    if [ -n "$diag" ]; then
      grep -n -E -A20 '^(error(\[E[0-9]+\])?:|warning: unused|note: )|failed to |nvcc fatal|^Killed|::error::|::warning::' "$dedup" 2>/dev/null | head -400 # tripwire-ok: same controlled-temp-path guard as the $diag assignment above
    else
      echo "  (no line matched the diagnostic patterns — see the last-40-lines tail below; the patterns themselves may need widening for this failure shape)"
    fi
    echo "--- last 40 lines of the captured log (adjacent duplicates collapsed) ---"
    tail -40 "$dedup"
  } > "$marker"
  rm -f "$dedup"
}

# ── the real seed build (main) ──────────────────────────────────────────────
pod_seed_target_main() {
  local reseed=0 no_lock=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --reseed) reseed=1; shift ;;
      --no-lock) no_lock=1; shift ;;
      *) echo "::error::unknown argument '$1'" >&2; return 2 ;;
    esac
  done

  if [ "$no_lock" != "1" ]; then
    # An ARRAY, never a quoted `"$(cond && echo --reseed)"` — the quoted
    # form passes an EMPTY-STRING argument (not "no argument") whenever
    # reseed=0, and pod_seed_target.sh's own arg loop then rejects it as
    # "unknown argument ''" — dead code on the documented default
    # invocation (no --reseed) AND on the -w lock path, since that IS the
    # re-exec (round-2 audit finding 4). An array with nothing pushed
    # expands to zero words, exactly "no argument" when reseed=0.
    local -a reseed_args=()
    [ "$reseed" = "1" ] && reseed_args=(--reseed)
    # round-3 audit Class B: `"${reseed_args[@]}"` on a DECLARED-BUT-EMPTY
    # array under `set -u` is an unbound-variable error on bash < 4.4
    # (macOS's shipped /bin/bash is 3.2 — a GPLv3-licensing artifact, not a
    # hypothetical). `"${arr[@]+"${arr[@]}"}"` is the portable "expand if
    # set, else nothing" idiom that has always worked correctly under
    # nounset, on every bash this tooling might run under (laptop or pod).
    JAMMI_TIMING_LABEL="seed" JAMMI_TIMING_JOB="seed" \
      exec "$DIR/pod_timing_lock.sh" acquire -w "$JAMMI_SEED_LOCK_WAIT_SECS" -- \
        env JAMMI_SEED_DIR="$JAMMI_SEED_DIR" JAMMI_TREE_DIR="$JAMMI_TREE_DIR" \
        "$DIR/pod_seed_target.sh" --no-lock "${reseed_args[@]+"${reseed_args[@]}"}"
  fi

  # A dry run parses args + (when --no-lock is absent) re-execs through the
  # lock exactly like a real invocation, then stops here — before touching
  # git/cargo — so the re-exec's own argv shape (the fix above) is testable
  # hermetically without a real build. Never set in production.
  if [ "${JAMMI_SEED_DRY_RUN:-0}" = "1" ]; then
    echo "dry-run: args parsed OK (reseed=${reseed} no_lock=${no_lock}) — real build skipped"
    return 0
  fi

  # round-4 addendum: fail loudly, naming every missing tool, BEFORE
  # spending any real compile time — never a cryptic "command not found"
  # discovered one tool at a time, thirty minutes into a build.
  pod_seed_assert_required_tools || return 1

  # Gate on EITHER marker (contract: `--reseed` overrides). A FAILED marker
  # left un-checked here let `shell`/`up` silently RETRY a known-broken seed
  # build on every single invocation (start_seed_build runs unconditionally
  # after bootstrap) — burning real compile minutes each time instead of
  # surfacing "this needs a human to look at it, or --reseed" once.
  if [ "$reseed" != "1" ] && [ -f "$COMPLETE_MARKER" ]; then
    echo "seed already complete ($COMPLETE_MARKER) — nothing to do (--reseed to force)"
    return 0
  fi
  if [ "$reseed" != "1" ] && [ -f "$FAILED_MARKER" ]; then
    echo "seed previously FAILED ($FAILED_MARKER) — not retrying automatically (--reseed to force); log tail:"
    tail -20 "$FAILED_MARKER"
    return 1
  fi
  rm -f "$FAILED_MARKER"

  local log rc
  log="$(mktemp)"
  # A REAL SUBSHELL `( )`, never a `{ }` group: a group runs in THIS SAME
  # process, so `exit 1` inside it (every build step below) would kill the
  # WHOLE SCRIPT immediately — skipping `rc=$?`, the FAILED_MARKER writer,
  # and the log-tail print entirely (the failure arm was dead code; round-2
  # audit finding 2). A subshell's `exit` only ends the subshell, leaving
  # its exit status in `$?` for the line right after `)` to read. `rc` is
  # declared `local` BEFORE the subshell runs (not combined with the
  # capture, and nothing else executes between the subshell and `rc=$?`) —
  # `local rc=$?` on one line, or `local rc` immediately after the group,
  # both read `$?` from the WRONG command (the `local` builtin's own always-
  # 0 success, not the build's real status) — the exact defect this
  # shape avoids.
  (
    set -euo pipefail
    cd "$JAMMI_TREE_DIR" || { echo "no tree at $JAMMI_TREE_DIR"; exit 1; }
    sha="$(git rev-parse HEAD)"; ref="$(git rev-parse --abbrev-ref HEAD)"
    # round-6 fix (lead probe, same class as pod_build_timings.sh's own
    # FA2 gate fix): `ref` above is kept as-is for the completion
    # marker's own informational "ref" field (a genuinely detached HEAD
    # SHOULD read literally "HEAD" there) — but gating T1b on `ref =
    # "main"` has the identical detached-HEAD hole: a seed built from a
    # bundle/sha checkout AT main's exact commit reads abbrev-ref "HEAD",
    # never "main", so T1b silently never runs. Gated on the RESOLVED
    # sha instead, with an explicit `JAMMI_SEED_IS_MAIN=1` override for a
    # caller (the bootstrap) that already KNOWS it is on main without
    # needing to re-derive it, and an honest reason when origin/main
    # itself cannot be resolved at all (never conflated with "genuinely
    # not on main").
    _seed_main_sha="$(git rev-parse --verify --quiet origin/main 2>/dev/null || true)" # tripwire-ok: no origin/main remote-tracking ref is a REAL state (e.g. a bundle clone with no such remote) -- named explicitly in t1b_reason below, never silently treated as "not main"
    _seed_is_main="false"
    if [ "${JAMMI_SEED_IS_MAIN:-0}" = "1" ]; then
      _seed_is_main="true"
      _seed_main_reason="JAMMI_SEED_IS_MAIN=1 override"
    elif [ -z "$_seed_main_sha" ]; then
      _seed_main_reason="could not resolve origin/main (no such remote-tracking ref — a bundle/sha-only checkout has none)"
    elif [ "$sha" = "$_seed_main_sha" ]; then
      _seed_is_main="true"
      _seed_main_reason="resolved sha matches origin/main"
    else
      _seed_main_reason="resolved sha ${sha} != origin/main (${_seed_main_sha})"
    fi
    capture="$(mktemp -d)"

    export CARGO_TARGET_DIR="$JAMMI_SEED_DIR"
    export CARGO_INCREMENTAL=0
    export CARGO_BUILD_RUSTC_WRAPPER=

    # round-4 addendum (on-pod incident, a100c A2 run at b3cafda): the seed
    # is the ONE place network access is expected and allowed — every
    # `--frozen` metadata call downstream of here (pod_seed_pkg_has_feature,
    # pod_seed_assert_member_free, pod_seed_scan_all_vendored_buildrs, the
    # non-member-package check) needs the FULL cross-platform dependency
    # graph `cargo metadata` resolves by default already fetched, which
    # `cargo build`/`cargo test` alone do NOT provide (they fetch only for
    # the CURRENT platform; `cargo metadata` without `--filter-platform`
    # walks every platform-conditional dependency in Cargo.lock, e.g. an
    # Android-only transitive crate this pod's own build never touches).
    # Reproduced: T1-T3 all succeeded, the member-free clean ran, and THEN
    # the non-member-package metadata check died on `error: failed to
    # download android_system_properties v0.1.5 — attempting to make an
    # HTTP request, but --frozen was specified` — every downstream
    # `--frozen` call would have failed the same way, some of them (pre-
    # this-addendum) SILENTLY, via pod_seed_pkg_has_feature's own rc=2
    # "could not determine" being treated as "skip T1b" rather than "abort"
    # (fixed below), so a poisoned pod state could stamp complete WITHOUT
    # the FA2 artifacts and nobody would know. One priming call, network
    # allowed, run exactly once, before ANYTHING `--frozen` is asked of
    # cargo:
    echo "=== priming cargo metadata resolution (network allowed once; every --frozen call below depends on this having already fetched) ==="
    metadata_prime_err="$(mktemp)"
    if ! cargo metadata --locked --format-version 1 --features jammi-kernels/cuda >/dev/null 2>"$metadata_prime_err"; then
      echo "::error::cargo metadata --locked (the seed's one-time network-allowed priming call) failed:" >&2
      cat "$metadata_prime_err" >&2
      rm -f "$metadata_prime_err"
      exit 1
    fi
    rm -f "$metadata_prime_err"

    echo "=== T1: release -p jammi-bench --features cuda ==="
    cargo build --release -p jammi-bench --features cuda || exit 1
    # round-5 addendum: the completion marker used to hardcode
    # `"tuples": ["T1","T2","T3"]` — it could not express whether T1b/FA2
    # actually ran (round-4's commit message itself named "a seed stamped
    # complete WITHOUT the FA2 artifacts and nobody would know" as the
    # defect this addendum closes for the rc=2 case; the rc=1/not-on-main
    # arms still produced a byte-identical marker either way).
    # pod_build_timings.sh's own FA2 measurement leg writes
    # `flash_attn_leg_wall_s` into the acceptance JSON with no way for a
    # reader to know which seed tuples were actually built — t1b_ran/
    # t1b_reason below make that legible in the committed marker itself.
    t1b_ran="false"
    t1b_reason="not main (${_seed_main_reason}) — T1b is main-only by design"
    if [ "$_seed_is_main" = "true" ]; then
      feat_rc=0
      pod_seed_pkg_has_feature jammi-kernels flash-attn || feat_rc=$?
      if [ "$feat_rc" -eq 0 ]; then
        # T1b compiles the vendored FA2 CUTLASS kernels, and jammi-kernels'
        # build.rs hard-fails when the cutlass submodule is not checked out.
        # A `git clone`-shaped tree (gpu-dev.sh's own bootstrap of
        # /root/jammi-ai) carries the GITLINK but not the checkout, so a
        # fresh pod's auto-seed on main would always die in T1b — provision
        # it here, before spending compile time (network, like the metadata
        # priming above; a tree whose checkout is complete skips this). A
        # non-git tree missing the header still fails loudly in build.rs
        # exactly as before — never a silent T1b skip.
        cutlass_inc="crates/jammi-kernels/third_party/cutlass/include"
        # The guard's predicate is BUILD.RS'S predicate — the same file
        # (include/cutlass/cutlass.h, crates/jammi-kernels/build.rs:311),
        # never a coarser dir-exists check: an include/ dir whose
        # cutlass/cutlass.h is gone (interrupted checkout, partial copy)
        # must count as unprovisioned, or the seed skips provisioning and
        # T1b dies with the exact panic this arm exists to eliminate.
        if [ ! -f "$cutlass_inc/cutlass/cutlass.h" ] && git rev-parse --git-dir >/dev/null 2>&1; then # tripwire-ok: the probe's EXIT CODE is the branch condition ("is this tree a git repo at all"); a non-git tree is a legitimate state whose only correct handling is the else-path (build.rs fails loudly on a genuinely missing include/), and git's "not a git repository" stderr would be pure noise on that ordinary path
          echo "=== T1b prerequisite: provisioning the CUTLASS submodule (git submodule update --init --force --checkout --depth 1) ==="
          # --force --checkout: this arm only runs when include/ is MISSING,
          # i.e. the checkout is already broken — a half-deleted worktree
          # whose git metadata still claims the pinned commit would make a
          # plain `update` a silent no-op (observed live on a100.2).
          git submodule update --init --force --checkout --depth 1 crates/jammi-kernels/third_party/cutlass || {
            echo "::error::CUTLASS submodule provisioning failed — T1b (flash-attn) cannot build; see git's own stderr above" >&2
            exit 1
          }
        fi
        echo "=== T1b (main only): release -p jammi-bench --features cuda,jammi-kernels/flash-attn ==="
        cargo build --release -p jammi-bench --features cuda,jammi-kernels/flash-attn || exit 1
        t1b_ran="true"
        t1b_reason="declared (cargo metadata) and built (${_seed_main_reason})"
      elif [ "$feat_rc" -eq 1 ]; then
        echo "=== T1b skipped: jammi-kernels declares no flash-attn feature (cargo metadata) ==="
        t1b_reason="jammi-kernels declares no flash-attn feature (cargo metadata, ${_seed_main_reason})"
      else
        # round-4 addendum: rc=2 ("could not determine") used to be treated
        # the SAME as rc=1 ("genuinely absent") — silently skip T1b. That is
        # exactly the failure mode the on-pod incident hit: a broken
        # `--frozen` metadata query read as "no flash-attn feature", the
        # seed stamped complete WITHOUT the FA2/T1b artifacts, and nothing
        # about the completion marker said so. "Could not determine" is not
        # a safe default to "absent" — abort the whole seed loudly instead,
        # naming the ambiguity, so a broken metadata query can never
        # silently downgrade what the seed actually contains.
        echo "::error::could not determine whether jammi-kernels declares flash-attn (cargo metadata query failed or the package was not found) — refusing to guess 'absent'; see pod_seed_cargo_metadata_frozen's own ::error:: above for the real cause" >&2
        exit 1
      fi
    fi

    echo "=== T2: cargo test --no-run for runpod_gpu_prove.sh's own suites ==="
    cargo test -p jammi-server --features cuda,live-gpu-tests --test it --no-run || exit 1
    cargo test -p jammi-ai --features cuda,live-gpu-tests --test gpu_capability --no-run || exit 1
    cargo test -p jammi-kernels --no-run || exit 1
    cargo test -p jammi-kernels --features cuda --no-run || exit 1

    echo "=== T3: clippy -p jammi-kernels --all-targets --features cuda ==="
    cargo clippy -p jammi-kernels --all-targets --features cuda -- -D warnings || exit 1

    echo "=== capturing build-script stdout before the clean ==="
    pod_seed_capture_build_output "$JAMMI_SEED_DIR" "$capture" debug
    pod_seed_capture_build_output "$JAMMI_SEED_DIR" "$capture" release

    echo "=== member-free clean ==="
    cargo clean --workspace --frozen || exit 1
    cargo clean --workspace --release --frozen || exit 1
    rm -rf "${JAMMI_SEED_DIR}"/*/incremental
    local leftover=""
    while IFS= read -r -d '' incdir; do
      local inner; inner="$(find "$incdir" -mindepth 1 -print -quit 2>/dev/null)" # tripwire-ok: find on a dir this loop just enumerated cannot meaningfully fail; a genuine miss surfaces as inner="" -> leftover stays empty -> NOT a silent pass, the caller still requires the loop to have run
      [ -z "$inner" ] || leftover="${leftover}${inner}\n"
    done < <(find "$JAMMI_SEED_DIR" -type d -name incremental -print0 2>/dev/null) # tripwire-ok: JAMMI_SEED_DIR is asserted to exist by every caller of pod_seed_target_main before this point; an empty match set here means zero incremental/ dirs, which is the SUCCESS case (nothing to check), not a hidden failure
    [ -z "$leftover" ] || { echo "::error::incremental/ not empty after rm -rf: $leftover"; exit 1; }

    echo "=== asserting no non-member path/patch package (cargo metadata) ==="
    pod_seed_cargo_metadata_frozen | python3 -c '
import sys, json
d = json.load(sys.stdin)
members = set(d["workspace_members"])
bad = [p["name"] for p in d["packages"] if p.get("source") is None and p["id"] not in members]
if bad:
    print("::error::non-member path/patch package(s) with source=null: %s" % bad)
    sys.exit(1)
' || exit 1

    echo "=== asserting the seed is member-free at the filesystem level (round-3 audit N2) ==="
    pod_seed_assert_member_free "$JAMMI_SEED_DIR" "$JAMMI_TREE_DIR" || exit 1

    echo "=== cross-checking the announced env surface against the manifest ==="
    pod_seed_check_stdout_subset "$capture" "$MANIFEST" || {
      echo "::error::a build script announced (rerun-if-env-changed) a var absent from $MANIFEST — see the lines above"
      exit 1
    }

    local size_bytes; size_bytes="$(du -sk "$JAMMI_SEED_DIR" 2>/dev/null | awk '{print $1*1024}')" # tripwire-ok: best-effort size for the marker only, never gates pass/fail
    # round-5 fix (class-shaped tripwire): this `2>/dev/null` used to
    # discard pod_sha256_of_file's own `::error::` line (the one THIS same
    # function prints when neither sha256sum nor shasum exists) — the
    # subshell still aborted under `set -e` either way (pod_seed_assert_
    # required_tools already guarantees a hashing tool exists before this
    # point is ever reached in practice), but a caller reading the FAILED
    # marker's tail would have seen no cause at all for a failure at this
    # exact line — the same "aborts with no diagnosis" shape the a100c
    # incident hit for a DIFFERENT command. Never silence a producing
    # command's stderr.
    local manifest_sha256; manifest_sha256="$(pod_sha256_of_file "$MANIFEST")"
    python3 -c '
import json, sys
tuples = ["T1", "T2", "T3"]
t1b_ran = sys.argv[7] == "true"
if t1b_ran:
    tuples.append("T1b")
print(json.dumps({
  "ref": sys.argv[1], "sha": sys.argv[2],
  "date": sys.argv[3], "tuples": tuples,
  "rustflags": sys.argv[4], "size_bytes": int(sys.argv[5]),
  "manifest_sha256": sys.argv[6], "seed_source": "built",
  "t1b_flash_attn_ran": t1b_ran, "t1b_flash_attn_reason": sys.argv[8],
}, indent=2))
' "$ref" "$sha" "$(date -u +%FT%TZ)" "${RUSTFLAGS:-}" "${size_bytes:-0}" "${manifest_sha256:-}" "$t1b_ran" "$t1b_reason" \
      > "$COMPLETE_MARKER"
    echo "=== seed complete: $COMPLETE_MARKER ==="
  ) > "$log" 2>&1
  rc=$?
  if [ "$rc" != 0 ]; then
    # round-4 addendum (on-pod incident, a100c A2 run at b3cafda): a plain
    # 100-line TAIL missed the diagnostic entirely — the actual error was
    # buried under later "Checking"/"Compiling" lines from OTHER crates
    # still finishing in parallel before the whole `cargo build` returned
    # non-zero, and nvcc OOM-kills print no "error:" line of their own at
    # all (the shell's own "Killed" report is the only trace). The marker
    # now carries: every line matching a diagnostic shape (error, error[EN],
    # unused-warning, note, "failed to", nvcc's own "fatal", or the shell's
    # "Killed") WITH surrounding context, the phase header (this script's
    # own `=== ... ===` echo) that was in progress when the subshell exited,
    # the exit code, and a resource snapshot at failure time — never a bare
    # tail alone. `pod_seed_write_failure_marker` is a standalone function
    # (not inlined here) so it has its own hermetic test — test_pod_
    # substrate.sh's `(n/addendum)` leg: a fixture log whose only
    # diagnostic line sits 200 lines above the tail must survive into the
    # marker (RED on a tail-only revert, same leg).
    pod_seed_write_failure_marker "$log" "$FAILED_MARKER" "$rc"
    cat "$log"
    # The marker's own "seed build FAILED" framing is ALSO echoed to stdout
    # (not just the raw captured build log above) — a human or CI log
    # watching this in real time sees the failure verdict immediately,
    # rather than having to notice the last compiler line never said
    # "complete".
    echo "seed build FAILED (exit $rc) — see $FAILED_MARKER for the full tail"
    rm -f "$log"
    return "$rc"
  fi
  cat "$log"
  rm -f "$log"
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  pod_seed_target_main "$@"
fi
