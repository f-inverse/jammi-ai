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
#       (+ --features cuda,jammi-encoders/flash-attn once, ONLY when the pod's
#        checkout is on `main` — flash-attn additionally compiles the vendored
#        FlashAttention-2 CUTLASS kernels, real nvcc minutes, so it is not
#        paid on every branch's seed)
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
# below; the hermetic test in test_pod_substrate.sh points this at a
# fixture). Every `cargo:rerun-if-env-changed=<NAME>` line actually announced
# by a real build-script run must be in the manifest. Prints unlisted names.
pod_seed_check_stdout_subset() { # $1=capture-dir $2=manifest-toml
  local capture_dir="$1" manifest="$2" names_file f name bad=0
  names_file="$(mktemp)"
  pod_seed_manifest_names "$manifest" > "$names_file"
  [ -d "$capture_dir" ] || { echo "::error::no build-output capture dir at ${capture_dir}" >&2; rm -f "$names_file"; return 1; }
  for f in "$capture_dir"/*; do
    [ -f "$f" ] || continue
    while IFS= read -r name; do
      [ -n "$name" ] || continue
      pod_seed_name_allowed "$name" "$names_file" || { echo "$name (from $(basename "$f"))"; bad=1; }
    done < <(grep -o 'cargo:rerun-if-env-changed=[A-Za-z_][A-Za-z0-9_]*' "$f" | sed 's/^cargo:rerun-if-env-changed=//')
  done
  rm -f "$names_file"
  return "$bad"
}

# Copy every build script's captured stdout out of the seed BEFORE cleaning —
# cargo's own cleaner removes `build/<pkg>-*/output` along with the rest of
# `build/`, so this is the only chance to read it.
pod_seed_capture_build_output() { # $1=seed target dir $2=dest capture dir $3=profile label
  local seed="$1" dest="$2" profile_label="$3" d base
  mkdir -p "$dest"
  for d in "$seed"/*/build/*/output; do
    [ -f "$d" ] || continue
    base="$(basename "$(dirname "$d")")"
    cp "$d" "$dest/${profile_label}__${base}.output"
  done
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
    JAMMI_TIMING_LABEL="seed" JAMMI_TIMING_JOB="seed" \
      exec "$DIR/pod_timing_lock.sh" acquire -w "$JAMMI_SEED_LOCK_WAIT_SECS" -- \
        env JAMMI_SEED_DIR="$JAMMI_SEED_DIR" JAMMI_TREE_DIR="$JAMMI_TREE_DIR" \
        "$DIR/pod_seed_target.sh" --no-lock "$([ "$reseed" = 1 ] && echo --reseed)"
  fi

  if [ "$reseed" != "1" ] && [ -f "$COMPLETE_MARKER" ]; then
    echo "seed already complete ($COMPLETE_MARKER) — nothing to do (--reseed to force)"
    return 0
  fi
  rm -f "$FAILED_MARKER"

  local log; log="$(mktemp)"
  {
    set -uo pipefail
    cd "$JAMMI_TREE_DIR" || { echo "no tree at $JAMMI_TREE_DIR"; exit 1; }
    local sha ref; sha="$(git rev-parse HEAD)"; ref="$(git rev-parse --abbrev-ref HEAD)"
    local capture; capture="$(mktemp -d)"

    export CARGO_TARGET_DIR="$JAMMI_SEED_DIR"
    export CARGO_INCREMENTAL=0
    export CARGO_BUILD_RUSTC_WRAPPER=

    echo "=== T1: release -p jammi-bench --features cuda ==="
    cargo build --release -p jammi-bench --features cuda || exit 1
    if [ "$ref" = "main" ]; then
      echo "=== T1b (main only): release -p jammi-bench --features cuda,jammi-encoders/flash-attn ==="
      cargo build --release -p jammi-bench --features cuda,jammi-encoders/flash-attn || exit 1
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
  } > "$log" 2>&1
  local rc
  rc=$?
  if [ "$rc" != 0 ]; then
    { echo "seed build FAILED (exit $rc) — log tail:"; tail -100 "$log"; } > "$FAILED_MARKER"
    cat "$log"
    rm -f "$log"
    return "$rc"
  fi
  cat "$log"
  rm -f "$log"
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  pod_seed_target_main "$@"
fi
