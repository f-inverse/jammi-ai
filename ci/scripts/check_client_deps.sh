#!/usr/bin/env bash
# Client-substrate dependency-boundary guards (wired from ci.yml's
# `test-clients` job).
#
# Two modes, one mechanism:
#
#   substrate — the candle-free client substrate: the wire transport
#     (`jammi-wire`), the control-plane client (`jammi-admin`), and the
#     data-plane client (`jammi-client`) must pull NO candle / hf-hub /
#     symphonia / tokenizers ML stack.
#   cli — the `jammi` CLI is a strict control-plane client: it depends on
#     `jammi-admin`, NOT `jammi-ai`, so it must stay candle-free AND carry no
#     `jammi-ai` edge at all.
#
# The workspace build feature-unifies `local` on via jammi-server /
# jammi-python, so `cargo tree` would falsely flag candle; the real
# per-package build set is what `cargo build -p X` compiles, so both guards
# inspect the compiler-artifact stream of the actual isolated build.
#
# Fail-closed on the build itself: the JSON stream goes to a file and cargo's
# own exit code is checked (via `set -e`) before any grep runs. A plain
# `cargo build --message-format=json | grep` pipeline has no pipefail, so a
# FAILED build could still exit 0 through grep's status and report the
# boundary green on a build that never completed.
#
# Usage:
#   check_client_deps.sh substrate|cli
#   check_client_deps.sh --self-test
#     (drives the assertion logic below against synthetic, multi-hundred-KB
#     compiler-artifact fixtures; no cargo, no network)
set -euo pipefail

# The forbidden embedded-engine ML stack, matched against each compiled
# package's manifest_path (`…/<name>-<version>/Cargo.toml` for registry
# crates) — PACKAGE identity, not target names, which a package is free to
# rename (jammi-cli's binary target is `jammi`; an ML dep could do the
# same). Every family gets the wildcard: these are families, not single
# crates (candle-core/candle-nn/..., the ten symphonia-* members,
# tokenizers and any split-off), so anchor-closing any alternative would
# hide its sub-crates.
ML_DEP_RE='"manifest_path":"[^"]*/(candle[^/"]*|hf[-_]hub[^/"]*|symphonia[^/"]*|tokenizers[^/"]*)/Cargo\.toml"'

# Sets the global `packages`/`ml_error`/`ml_ok` for a mode. Returns
# (never exits, so --self-test can probe an unknown mode too) 0 on a known
# mode, 2 otherwise.
set_mode_vars() {
  case "$1" in
    substrate)
      packages=(-p jammi-wire -p jammi-admin -p jammi-client)
      ml_error="the client substrate compiled an embedded-engine ML dep — the candle-free boundary regressed"
      ml_ok="wire / admin / client build set is candle-free"
      ;;
    cli)
      packages=(-p jammi-cli)
      ml_error="the jammi CLI compiled an embedded-engine ML dep — the strict-client boundary regressed"
      ml_ok="jammi-cli build set is candle-free"
      ;;
    *)
      echo "unknown mode: ${1} (expected substrate|cli)" >&2
      return 2
      ;;
  esac
}

# Filters a raw cargo `--message-format=json` stream (`$1`) down to its
# `compiler-artifact` lines, into `$2`. "Parsed zero artifacts" and "no ML
# dep found" must never be the same state: an empty stream (a cargo JSON
# reshape, a wrapper eating stdout) means the guard cannot see the build
# set, so it refuses rather than reporting the boundary green on evidence it
# never had. Returns (never exits) 1 on an empty/absent stream.
extract_artifacts() {
  grep '"reason":"compiler-artifact"' "$1" > "$2"
}

# The real assertion. `$1` = mode (substrate|cli), `$2` = a file already
# filtered to compiler-artifact lines (by `extract_artifacts`, or a
# synthetic fixture built the same shape by `--self-test`). Returns (never
# exits, so --self-test can call this repeatedly without killing its own
# shell under `set -e`) 0 on a clean boundary, 1 on any violation or missing
# package, 2 on an unknown mode.
#
# Every grep below reads `$2` — a FILE — directly, never a variable piped
# through `printf | grep -q`: on a multi-hundred-KB stream, `grep -q` exits
# at its first match and SIGPIPEs the upstream writer, which `pipefail`
# turns into a spurious 141 that flips the verdict — a false "package never
# appeared" (this function's package-presence loop, on ANY mode) or a
# false-GREEN swallowed real violation (the cli-only jammi-ai check below,
# the dangerous direction: an early match reads as no match at all). A file
# has no writer to kill, so no grep here can SIGPIPE.
assert_boundary() {
  local mode="$1" artifacts="$2"
  set_mode_vars "$mode" || return 2

  # The guard must also have SEEN the packages it is guarding (workspace
  # packages live at `crates/<pkg>/Cargo.toml`; a package's targets need
  # not carry its name — jammi-cli's binary target is renamed to `jammi` —
  # so a target-name match would red a clean build and leave the boundary
  # assertions below unreachable).
  local arg
  for arg in "${packages[@]}"; do
    [ "$arg" = "-p" ] && continue
    if ! grep -qE "\"manifest_path\":\"[^\"]*/${arg}/Cargo\.toml\"" "$artifacts"; then
      echo "::error::requested package ${arg} never appeared in the compiler-artifact stream — the guard did not observe the build it is gating" >&2
      return 1
    fi
  done

  if grep -ioE "$ML_DEP_RE" "$artifacts"; then
    echo "::error::${ml_error}"
    return 1
  fi
  echo "$ml_ok"

  if [ "$mode" = "cli" ]; then
    if grep -qE '"manifest_path":"[^"]*/jammi[-_]ai/Cargo\.toml"' "$artifacts"; then
      echo "::error::the jammi CLI compiled jammi-ai — the strict-client boundary regressed"
      return 1
    fi
    echo "jammi-cli build set carries no jammi-ai edge"
  fi
  return 0
}

# Prints `$1` harmless compiler-artifact lines — padding to reproduce the
# multi-hundred-KB scale the SIGPIPE bug needs (a stream too big for a pipe
# buffer to hold whole, so a `grep -q` that matches near the front of the
# stream genuinely outruns and kills a writer still filling the pipe).
_gen_filler() {
  awk -v n="$1" 'BEGIN{for(i=0;i<n;i++) print "{\"reason\":\"compiler-artifact\",\"manifest_path\":\"/build/filler/Cargo.toml\"}"}'
}

# RED-case-covering self-test: every fixture here is hermetic (no cargo, no
# network) and large enough (~500k lines, tens of MB) to genuinely trigger
# the SIGPIPE this file's grep-a-file-not-a-pipe discipline defends against
# — a smaller fixture fits in one pipe buffer and would pass even on the
# buggy `printf | grep -q` shape, silently regressing this test's coverage.
_self_test() {
  local failures=0
  local work rc out
  work="$(mktemp -d)"

  # Fixture 1: substrate, clean, LARGE — jammi-wire's line sits FIRST
  # (the shape that flips the old `printf | grep -q` package-presence
  # check to a false "never appeared"), admin/client early too, ~500k
  # filler lines trail. Must PASS.
  {
    echo '{"reason":"compiler-artifact","manifest_path":"/w/jammi-wire/Cargo.toml"}'
    echo '{"reason":"compiler-artifact","manifest_path":"/w/jammi-admin/Cargo.toml"}'
    echo '{"reason":"compiler-artifact","manifest_path":"/w/jammi-client/Cargo.toml"}'
    _gen_filler 500000
  } > "$work/substrate.json"
  rc=0
  extract_artifacts "$work/substrate.json" "$work/substrate.artifacts" || rc=$?
  out=""
  if [ "$rc" -eq 0 ]; then
    rc=0
    out="$(assert_boundary substrate "$work/substrate.artifacts" 2>&1)" || rc=$?
  fi
  if [ "$rc" -eq 0 ]; then
    echo "self-test[substrate-clean-large-stream]: OK (exit 0, expected 0)"
  else
    echo "self-test[substrate-clean-large-stream]: FAIL (exit $rc, expected 0)" >&2
    printf '%s\n' "$out" >&2
    failures=$((failures + 1))
  fi

  # Fixture 2: cli, VIOLATION, LARGE — jammi-ai's line sits FIRST (the
  # dangerous direction: the old `printf | grep -q` at the jammi-ai check
  # took the false-GREEN branch here and swallowed a real regression).
  # jammi-cli itself sits LAST, after all the filler, so the
  # package-presence loop's own grep is NOT itself SIGPIPE'd — isolating
  # the jammi-ai check's bug from the package-presence loop's bug so this
  # fixture actually reaches the jammi-ai check. Must FAIL with the
  # jammi-ai error.
  {
    echo '{"reason":"compiler-artifact","manifest_path":"/w/jammi-ai/Cargo.toml"}'
    _gen_filler 500000
    echo '{"reason":"compiler-artifact","manifest_path":"/w/jammi-cli/Cargo.toml"}'
  } > "$work/cli-violation.json"
  rc=0
  extract_artifacts "$work/cli-violation.json" "$work/cli-violation.artifacts" || rc=$?
  out=""
  if [ "$rc" -eq 0 ]; then
    rc=0
    out="$(assert_boundary cli "$work/cli-violation.artifacts" 2>&1)" || rc=$?
  fi
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -qF 'the jammi CLI compiled jammi-ai'; then
    echo "self-test[cli-jammi-ai-violation-large-stream]: OK (exit 1, jammi-ai error present)"
  else
    echo "self-test[cli-jammi-ai-violation-large-stream]: FAIL (exit $rc, expected 1 with the jammi-ai error)" >&2
    printf '%s\n' "$out" >&2
    failures=$((failures + 1))
  fi

  # Fixture 3: cli, CLEAN, LARGE — jammi-cli present early, no jammi-ai
  # anywhere in the large stream. Must still PASS: the fix must not turn
  # every cli run red just because the stream is big.
  {
    echo '{"reason":"compiler-artifact","manifest_path":"/w/jammi-cli/Cargo.toml"}'
    _gen_filler 500000
  } > "$work/cli-clean.json"
  rc=0
  extract_artifacts "$work/cli-clean.json" "$work/cli-clean.artifacts" || rc=$?
  out=""
  if [ "$rc" -eq 0 ]; then
    rc=0
    out="$(assert_boundary cli "$work/cli-clean.artifacts" 2>&1)" || rc=$?
  fi
  if [ "$rc" -eq 0 ]; then
    echo "self-test[cli-clean-large-stream]: OK (exit 0, expected 0)"
  else
    echo "self-test[cli-clean-large-stream]: FAIL (exit $rc, expected 0)" >&2
    printf '%s\n' "$out" >&2
    failures=$((failures + 1))
  fi

  # Fixture 4: substrate, a package genuinely absent (small stream) — a
  # true negative must still error; the fix must not paper over a real
  # miss just because it now reads a file.
  {
    echo '{"reason":"compiler-artifact","manifest_path":"/w/jammi-wire/Cargo.toml"}'
    echo '{"reason":"compiler-artifact","manifest_path":"/w/jammi-admin/Cargo.toml"}'
  } > "$work/substrate-missing.json"
  rc=0
  extract_artifacts "$work/substrate-missing.json" "$work/substrate-missing.artifacts" || rc=$?
  out=""
  if [ "$rc" -eq 0 ]; then
    rc=0
    out="$(assert_boundary substrate "$work/substrate-missing.artifacts" 2>&1)" || rc=$?
  fi
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -qF 'jammi-client never appeared'; then
    echo "self-test[substrate-missing-package]: OK (exit 1, missing-package error present)"
  else
    echo "self-test[substrate-missing-package]: FAIL (exit $rc, expected 1 with the missing-package error)" >&2
    printf '%s\n' "$out" >&2
    failures=$((failures + 1))
  fi

  rm -rf "$work"

  if [ "$failures" -gt 0 ]; then
    echo "check-client-deps --self-test: $failures fixture(s) FAILED" >&2
    return 1
  fi
  echo "check-client-deps --self-test: all 4 fixture(s) passed."
  return 0
}

mode="${1:?usage: check_client_deps.sh substrate|cli|--self-test}"

if [ "$mode" = "--self-test" ]; then
  _self_test
  exit $?
fi

set_mode_vars "$mode" || exit 2

json="$(mktemp)"
artifacts="$(mktemp)"
trap 'rm -f "$json" "$artifacts"' EXIT

# `set -e` aborts here on a failed build — cargo's exit code is the gate,
# never a downstream grep's.
cargo build "${packages[@]}" --message-format=json > "$json"

extract_artifacts "$json" "$artifacts" || {
  echo "::error::no compiler-artifact lines parsed from the isolated build's JSON stream — the guard cannot see the build set; refusing to report the boundary green" >&2
  exit 1
}

assert_boundary "$mode" "$artifacts"
exit $?
