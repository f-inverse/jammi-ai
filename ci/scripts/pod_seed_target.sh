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
# actually iterated — must be >= 1, and every one of them non-empty (a
# captured-but-truly-empty output file means the capture ran at the wrong
# moment relative to the real build, not that the build script announced
# nothing worth tracking).
pod_seed_check_stdout_subset() { # $1=capture-dir $2=manifest-toml
  local capture_dir="$1" manifest="$2" names_file f name bad=0 capture_count=0
  names_file="$(mktemp)"
  pod_seed_manifest_names "$manifest" > "$names_file"
  [ -d "$capture_dir" ] || { echo "::error::no build-output capture dir at ${capture_dir}" >&2; rm -f "$names_file"; return 1; }
  for f in "$capture_dir"/*; do
    [ -f "$f" ] || continue
    capture_count=$((capture_count + 1))
    if [ ! -s "$f" ]; then
      echo "::error::captured build-script output file is EMPTY: $f (captured at the wrong moment, or the build script genuinely never ran)" >&2
      bad=1
    fi
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
# feature really doesn't exist" from "I have no idea, the query broke" —
# both silently read as "skip the optional leg", which is the right ACTION
# either way, but the WRONG message ("declares no flash-attn feature" is a
# false claim when the truth is "could not ask").
pod_seed_pkg_has_feature() { # $1=pkg $2=feature
  cargo metadata --frozen --format-version 1 2>/dev/null | python3 -c '
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

# FILESYSTEM-LEVEL member-freedom check (round-3 audit N2). pod_seed_target.sh
# always documented "member-free seed" but never actually verified it at the
# one place that matters — the target dir's own contents. The metadata-only
# check elsewhere in this file (no non-member path/patch PACKAGE — a
# Cargo.lock/[patch] hygiene question) is the OPPOSITE direction and cannot
# catch a member's own compiled artifact surviving a clean; the incremental/
# emptiness check covers exactly one subdirectory cargo's own cleaner is
# already known to miss (this file's own module doc, above). This is the
# ONE mechanical, always-on definition: after ANY clean or clone, no
# `{debug,release}/{.fingerprint,deps,build,incremental}` entry may be named
# after a WORKSPACE MEMBER, where "workspace member" is read from `cargo
# metadata`'s own `workspace_members` — never a "jammi-*" glob/prefix guess,
# so a member crate that happened not to start with "jammi-" would still be
# caught. Both cargo's hyphenated form (.fingerprint/build directory names,
# e.g. `jammi-kernels-<hash>`) and its underscored form (deps/ compiled
# artifact names, e.g. `jammi_kernels-<hash>.rlib`) are checked, verified
# against a real cargo build/clean cycle (not merely a naming-convention
# guess). $1=target_dir (a CARGO_TARGET_DIR — debug/ and release/ scanned)
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
  meta="$(cd "$tree_dir" && cargo metadata --frozen --format-version 1 2>/dev/null)"
  if [ -z "$meta" ]; then
    echo "::error::pod_seed_assert_member_free: cargo metadata produced no output from ${tree_dir} — registry not fetched?" >&2
    return 2
  fi
  printf '%s' "$meta" | python3 -c '
import sys, json, os, re
target_dir = sys.argv[1]
d = json.load(sys.stdin)
members_by_id = {p["id"]: p["name"] for p in d["packages"]}
member_names = set()
for mid in d["workspace_members"]:
    name = members_by_id.get(mid)
    if name:
        member_names.add(name)
        member_names.add(name.replace("-", "_"))
if not member_names:
    print("::error::pod_seed_assert_member_free: cargo metadata reported ZERO workspace members — refusing to run a vacuous check", file=sys.stderr)
    sys.exit(2)
# A directory ENTRY is member-named if its basename starts with
# "<member>-" (cargo always separates a crate name from its fingerprint
# hash/suffix with exactly one hyphen) — a boundary check, so member
# "jammi-kernels" does not false-positive-match an unrelated
# "jammi-kernels-utils" entry that merely shares the prefix.
pat = re.compile(r"^(" + "|".join(re.escape(n) for n in sorted(member_names, key=len, reverse=True)) + r")-")
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
  local manifest="$1" mode="$2" names_file bad=0 scanned=0 allowlisted=0
  names_file="$(mktemp)"
  pod_seed_manifest_names "$manifest" > "$names_file"

  local meta; meta="$(cargo metadata --frozen --format-version 1 --features jammi-kernels/cuda 2>/dev/null)"
  if [ -z "$meta" ]; then
    # Self-heal once (round-2 audit finding 6): a bare/fresh checkout with
    # no registry fetch yet is the ordinary shape on a maintainer's machine
    # or a CI runner that skipped the fetch step — `cargo fetch --locked`
    # is offline-safe to attempt (it only pulls what Cargo.lock already
    # pins) and cheap when already warm. Fail loudly, naming the real cause,
    # only if the retry ALSO comes up empty.
    echo "cargo metadata produced no output — attempting 'cargo fetch --locked' once, then retrying" >&2
    cargo fetch --locked >&2 2>&1
    meta="$(cargo metadata --frozen --format-version 1 --features jammi-kernels/cuda 2>/dev/null)"
  fi
  if [ -z "$meta" ]; then
    echo "::error::cargo metadata produced no output even after 'cargo fetch --locked' — is a Rust toolchain on PATH?" >&2
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
    capture="$(mktemp -d)"

    export CARGO_TARGET_DIR="$JAMMI_SEED_DIR"
    export CARGO_INCREMENTAL=0
    export CARGO_BUILD_RUSTC_WRAPPER=

    echo "=== T1: release -p jammi-bench --features cuda ==="
    cargo build --release -p jammi-bench --features cuda || exit 1
    if [ "$ref" = "main" ]; then
      feat_rc=0
      pod_seed_pkg_has_feature jammi-kernels flash-attn || feat_rc=$?
      if [ "$feat_rc" -eq 0 ]; then
        echo "=== T1b (main only): release -p jammi-bench --features cuda,jammi-kernels/flash-attn ==="
        cargo build --release -p jammi-bench --features cuda,jammi-kernels/flash-attn || exit 1
      elif [ "$feat_rc" -eq 1 ]; then
        echo "=== T1b skipped: jammi-kernels declares no flash-attn feature (cargo metadata) ==="
      else
        echo "=== T1b skipped: could not determine whether jammi-kernels declares flash-attn (cargo metadata query failed or the package was not found) — treating as absent, not as confirmed absent ==="
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
      local inner; inner="$(find "$incdir" -mindepth 1 -print -quit 2>/dev/null)"
      [ -z "$inner" ] || leftover="${leftover}${inner}\n"
    done < <(find "$JAMMI_SEED_DIR" -type d -name incremental -print0 2>/dev/null)
    [ -z "$leftover" ] || { echo "::error::incremental/ not empty after rm -rf: $leftover"; exit 1; }

    echo "=== asserting no non-member path/patch package (cargo metadata) ==="
    cargo metadata --frozen --format-version 1 2>/dev/null | python3 -c '
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

    local size_bytes; size_bytes="$(du -sk "$JAMMI_SEED_DIR" 2>/dev/null | awk '{print $1*1024}')"
    local manifest_sha256; manifest_sha256="$(shasum -a 256 "$MANIFEST" 2>/dev/null | awk '{print $1}')"
    python3 -c '
import json, sys
print(json.dumps({
  "ref": sys.argv[1], "sha": sys.argv[2],
  "date": sys.argv[3], "tuples": ["T1", "T2", "T3"],
  "rustflags": sys.argv[4], "size_bytes": int(sys.argv[5]),
  "manifest_sha256": sys.argv[6], "seed_source": "built",
}, indent=2))
' "$ref" "$sha" "$(date -u +%FT%TZ)" "${RUSTFLAGS:-}" "${size_bytes:-0}" "${manifest_sha256:-}" \
      > "$COMPLETE_MARKER"
    echo "=== seed complete: $COMPLETE_MARKER ==="
  ) > "$log" 2>&1
  rc=$?
  if [ "$rc" != 0 ]; then
    { echo "seed build FAILED (exit $rc) — log tail:"; tail -100 "$log"; } > "$FAILED_MARKER"
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
