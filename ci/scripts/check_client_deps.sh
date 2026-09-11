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
#     (drives the assertion logic below against synthetic compiler-artifact
#     fixtures, incl. >1 MB streams sized well past one pipe buffer to
#     genuinely reproduce the SIGPIPE class below; no cargo, no network)
#   check_client_deps.sh --assert-boundary <mode> <artifacts-file>
#     (internal only — see `_run_assert_boundary`'s own comment for why
#     `--self-test` invokes this as a genuinely separate process)
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
# never had. Returns (never exits) 1 on an empty/absent stream — a stream
# that parses fine but carries ZERO compiler-artifact lines must refuse
# here, same as a genuinely empty/absent one; --self-test's
# "zero-compiler-artifact-refusal" fixture pins exactly this (a JSON stream
# that is non-empty but carries no `"reason":"compiler-artifact"` line at
# all — a `cat`-shaped mutant that stopped filtering would let it through).
extract_artifacts() {
  grep '"reason":"compiler-artifact"' "$1" > "$2"
}

# The real assertion. `$1` = mode (substrate|cli), `$2` = a file already
# filtered to compiler-artifact lines (by `extract_artifacts`, or a
# synthetic fixture built the same shape by `--self-test`). Returns (never
# exits, so it can be called repeatedly without killing its own shell under
# `set -e` — but see `_run_assert_boundary` below: within THIS process, the
# only safe caller is the bare, standalone production call site near the
# bottom of this file) 0 on a clean boundary, 1 on any violation or missing
# package, 2 on an unknown mode.
#
# Every grep below reads `$2` — a FILE — directly, never a variable piped
# through `printf | grep -q`: on a stream bigger than one pipe buffer,
# `grep -q` exits at its first match and SIGPIPEs the upstream writer,
# which `pipefail` turns into a spurious 141 that flips the verdict — a
# false "package never appeared" (this function's package-presence loop, on
# ANY mode) or a false-GREEN swallowed real violation (the cli-only
# jammi-ai check below, the dangerous direction: an early match reads as no
# match at all). A file has no writer to kill, so no grep here can SIGPIPE.
assert_boundary() {
  local mode="$1" artifacts="$2"
  set_mode_vars "$mode" || return 2

  # The guard must also have SEEN the packages it is guarding (workspace
  # packages live at `crates/<pkg>/Cargo.toml`; a package's targets need
  # not carry its name — jammi-cli's binary target is renamed to `jammi` —
  # so a target-name match would red a clean build and leave the boundary
  # assertions below unreachable).
  #
  # Every grep from here down branches on its OWN exit code (0/1/`other`)
  # explicitly, the same doctrine `pod_target_clone.sh`'s `--verify` branch
  # applies to its own detection grep: exit 1 ("no match" — a real,
  # decided absence) must never be conflated with any OTHER nonzero exit
  # (a read failure on `artifacts` itself — "could not read the evidence").
  # This loop's own read failure is unreachable TODAY only because it is
  # the FIRST grep to touch `artifacts` in this function — the ML-dep and
  # jammi-ai checks below it would never even run — but that is an
  # ordering accident, not a property either sibling site's own branching
  # should rely on to stay correct if this loop, or its package list, is
  # ever reordered or shortened.
  local arg grep_rc
  for arg in "${packages[@]}"; do
    [ "$arg" = "-p" ] && continue
    grep_rc=0
    grep -qE "\"manifest_path\":\"[^\"]*/${arg}/Cargo\.toml\"" "$artifacts" || grep_rc=$?
    if [ "$grep_rc" -eq 1 ]; then
      echo "::error::requested package ${arg} never appeared in the compiler-artifact stream — the guard did not observe the build it is gating" >&2
      return 1
    elif [ "$grep_rc" -ne 0 ]; then
      echo "::error::could not read ${artifacts} while checking for package ${arg} (grep exit ${grep_rc}) — refusing to report the boundary green" >&2
      return 1
    fi
  done

  grep_rc=0
  grep -ioE "$ML_DEP_RE" "$artifacts" || grep_rc=$?
  if [ "$grep_rc" -eq 0 ]; then
    echo "::error::${ml_error}"
    return 1
  elif [ "$grep_rc" -ne 1 ]; then
    echo "::error::could not read ${artifacts} while scanning for an embedded-engine ML dep (grep exit ${grep_rc}) — refusing to report the boundary green" >&2
    return 1
  fi
  echo "$ml_ok"

  if [ "$mode" = "cli" ]; then
    grep_rc=0
    grep -qE '"manifest_path":"[^"]*/jammi[-_]ai/Cargo\.toml"' "$artifacts" || grep_rc=$?
    if [ "$grep_rc" -eq 0 ]; then
      echo "::error::the jammi CLI compiled jammi-ai — the strict-client boundary regressed"
      return 1
    elif [ "$grep_rc" -ne 1 ]; then
      echo "::error::could not read ${artifacts} while scanning for a jammi-ai edge (grep exit ${grep_rc}) — refusing to report the boundary green" >&2
      return 1
    fi
    echo "jammi-cli build set carries no jammi-ai edge"
  fi
  return 0
}

# Prints `$1` harmless compiler-artifact lines — padding to a size that
# genuinely reproduces the SIGPIPE class this file's grep-a-file-not-a-pipe
# discipline defends against (a stream too big for a pipe buffer to hold
# whole, so a `grep -q` that matches near the front of the stream genuinely
# outruns and kills a writer still filling the pipe). A fixture that fits
# in one pipe write would pass even on the buggy `printf | grep -q` shape,
# silently under-covering a reintroduced regression.
_gen_filler() {
  awk -v n="$1" 'BEGIN{for(i=0;i<n;i++) print "{\"reason\":\"compiler-artifact\",\"manifest_path\":\"/build/filler/Cargo.toml\"}"}'
}

# Runs `assert_boundary` the way the PRODUCTION call site (below, near the
# bottom of this file) actually runs it: bare, under a REAL, unsuppressed
# `set -e`, as a standalone statement — never from inside a `cmd || …`
# list.
#
# An in-process `( set -e; assert_boundary … )` subshell does NOT give this
# — bash's `&&`/`||`-list errexit suppression (the special case where a
# non-final command in such a list runs with `-e`'s effect disabled) is
# inherited by ANY subshell forked while that command is being evaluated,
# and re-running `set -e` inside that subshell is a documented no-op: the
# option is already "set", so the statement is a syntactic nop, not a
# state reset. Measured directly (bash 5.3.15): a bare
# `( set -e; /usr/bin/false; echo unreached )` aborts as expected; the
# IDENTICAL subshell called as `out="$( … )" || rc=$?` prints "unreached"
# and reports success. A self-test that only ever invoked `assert_boundary`
# this way would run it under LOOSER error semantics than production ever
# does, silently missing a class of bug (an unexpected failing command
# inside `assert_boundary` that should abort immediately, the way it would
# in the real script).
#
# A genuinely separate PROCESS has no such suppression state to inherit —
# `--assert-boundary` (dispatched near the bottom of this file) is that
# process: a fresh `bash "$0"` invocation whose OWN top-level `set -e` (line
# 32) is never inside anyone else's `||`-list. Callers still wrap THIS
# call in `|| rc=$?` to capture a fixture's PASS/FAIL without killing the
# self-test's own loop — that outer `||` only captures the child PROCESS's
# exit status; it cannot reach back inside the child to suppress ITS
# errexit, because the child is not the same shell.
_run_assert_boundary() {
  bash "$0" --assert-boundary "$1" "$2" 2>&1
}

# RED-case-covering self-test: every fixture here is hermetic (no cargo, no
# network). Each fixture increments `total` and, on failure, `failures` —
# the closing summary derives its count from `total` rather than a
# hardcoded number, so an added/removed fixture can never drift out of
# sync with what actually ran.
_self_test() {
  local failures=0 total=0
  local rc out
  # NOT `local`: the EXIT trap below fires when the whole SCRIPT exits
  # (traps are not function-scoped), which is after `_self_test` itself has
  # already returned — a `local work` would already be unset by then, and
  # the trap's `rm -rf "$work"` would abort on `set -u`'s "unbound
  # variable" instead of ever cleaning up.
  work="$(mktemp -d)"
  trap 'rm -rf "$work"' EXIT

  # The four SIGPIPE-class fixtures below carry ~14,000 filler lines
  # (1,036,000 B measured) after/around the line that matters. The
  # portable trigger for the SIGPIPE'd `printf | grep -q` shape this file
  # replaced is ONE PIPE BUFFER — 64 KiB — not any specific "cargo output"
  # byte count; >1 MB clears that by more than 15x while staying fast. A
  # much smaller fixture (a two-line, ~70-byte stream) fits inside one pipe
  # write and would pass even on the unfixed script, silently regressing
  # this self-test's own coverage.
  local filler=14000

  # Fixture 1: substrate, clean, LARGE — jammi-wire's line sits FIRST
  # (the shape that flips the old `printf | grep -q` package-presence
  # check to a false "never appeared"), admin/client early too. Must PASS.
  {
    echo '{"reason":"compiler-artifact","manifest_path":"/w/jammi-wire/Cargo.toml"}'
    echo '{"reason":"compiler-artifact","manifest_path":"/w/jammi-admin/Cargo.toml"}'
    echo '{"reason":"compiler-artifact","manifest_path":"/w/jammi-client/Cargo.toml"}'
    _gen_filler "$filler"
  } > "$work/substrate.json"
  rc=0
  extract_artifacts "$work/substrate.json" "$work/substrate.artifacts" || rc=$?
  out=""
  if [ "$rc" -eq 0 ]; then
    rc=0
    out="$(_run_assert_boundary substrate "$work/substrate.artifacts")" || rc=$?
  fi
  total=$((total + 1))
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
    _gen_filler "$filler"
    echo '{"reason":"compiler-artifact","manifest_path":"/w/jammi-cli/Cargo.toml"}'
  } > "$work/cli-violation.json"
  rc=0
  extract_artifacts "$work/cli-violation.json" "$work/cli-violation.artifacts" || rc=$?
  out=""
  if [ "$rc" -eq 0 ]; then
    rc=0
    out="$(_run_assert_boundary cli "$work/cli-violation.artifacts")" || rc=$?
  fi
  total=$((total + 1))
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
    _gen_filler "$filler"
  } > "$work/cli-clean.json"
  rc=0
  extract_artifacts "$work/cli-clean.json" "$work/cli-clean.artifacts" || rc=$?
  out=""
  if [ "$rc" -eq 0 ]; then
    rc=0
    out="$(_run_assert_boundary cli "$work/cli-clean.artifacts")" || rc=$?
  fi
  total=$((total + 1))
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
    out="$(_run_assert_boundary substrate "$work/substrate-missing.artifacts")" || rc=$?
  fi
  total=$((total + 1))
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -qF 'jammi-client never appeared'; then
    echo "self-test[substrate-missing-package]: OK (exit 1, missing-package error present)"
  else
    echo "self-test[substrate-missing-package]: FAIL (exit $rc, expected 1 with the missing-package error)" >&2
    printf '%s\n' "$out" >&2
    failures=$((failures + 1))
  fi

  # Fixture 5: substrate, an ML-dep violation (candle-core) alongside a
  # clean package set. Must FAIL with ml_error. Pins the `if grep -ioE
  # "$ML_DEP_RE"` branch itself — a `→ if false` mutant here would leave
  # every fixture above green (none of them carry an ML dep) and only this
  # one catches it.
  {
    echo '{"reason":"compiler-artifact","manifest_path":"/w/jammi-wire/Cargo.toml"}'
    echo '{"reason":"compiler-artifact","manifest_path":"/w/jammi-admin/Cargo.toml"}'
    echo '{"reason":"compiler-artifact","manifest_path":"/w/jammi-client/Cargo.toml"}'
    echo '{"reason":"compiler-artifact","manifest_path":"/home/.cargo/registry/src/index.crates.io/candle-core-0.7.2/Cargo.toml"}'
  } > "$work/substrate-ml-violation.json"
  rc=0
  extract_artifacts "$work/substrate-ml-violation.json" "$work/substrate-ml-violation.artifacts" || rc=$?
  out=""
  if [ "$rc" -eq 0 ]; then
    rc=0
    out="$(_run_assert_boundary substrate "$work/substrate-ml-violation.artifacts")" || rc=$?
  fi
  total=$((total + 1))
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -qF 'the candle-free boundary regressed'; then
    echo "self-test[substrate-ml-dep-violation]: OK (exit 1, ml_error present)"
  else
    echo "self-test[substrate-ml-dep-violation]: FAIL (exit $rc, expected 1 with ml_error)" >&2
    printf '%s\n' "$out" >&2
    failures=$((failures + 1))
  fi

  # Fixture 6: cli, an ML-dep violation (candle-core). Must FAIL with
  # ml_error — the cli-mode arm of the same `grep -ioE "$ML_DEP_RE"` branch,
  # and it must fire BEFORE the jammi-ai check below it (candle-core is not
  # jammi-ai, so a script that skipped straight to the jammi-ai check would
  # wrongly pass this one).
  {
    echo '{"reason":"compiler-artifact","manifest_path":"/w/jammi-cli/Cargo.toml"}'
    echo '{"reason":"compiler-artifact","manifest_path":"/home/.cargo/registry/src/index.crates.io/candle-core-0.7.2/Cargo.toml"}'
  } > "$work/cli-ml-violation.json"
  rc=0
  extract_artifacts "$work/cli-ml-violation.json" "$work/cli-ml-violation.artifacts" || rc=$?
  out=""
  if [ "$rc" -eq 0 ]; then
    rc=0
    out="$(_run_assert_boundary cli "$work/cli-ml-violation.artifacts")" || rc=$?
  fi
  total=$((total + 1))
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -qF 'the strict-client boundary regressed'; then
    echo "self-test[cli-ml-dep-violation]: OK (exit 1, ml_error present)"
  else
    echo "self-test[cli-ml-dep-violation]: FAIL (exit $rc, expected 1 with ml_error)" >&2
    printf '%s\n' "$out" >&2
    failures=$((failures + 1))
  fi

  # Fixture 7: a stream that parses (non-empty) but carries ZERO
  # `"reason":"compiler-artifact"` lines — `extract_artifacts` must refuse
  # (return 1), never silently hand an empty/irrelevant file on to
  # `assert_boundary`. Pins the extract-filter itself: a `→ cat "$1" >
  # "$2"` mutant would copy this fixture through unfiltered and "succeed".
  {
    echo '{"reason":"build-script-executed","package_id":"foo 0.1.0"}'
    echo '{"reason":"compiler-message","message":{"level":"warning"}}'
  } > "$work/zero-artifacts.json"
  rc=0
  extract_artifacts "$work/zero-artifacts.json" "$work/zero-artifacts.artifacts" || rc=$?
  total=$((total + 1))
  if [ "$rc" -eq 1 ]; then
    echo "self-test[zero-compiler-artifact-refusal]: OK (extract_artifacts refused, exit 1)"
  else
    echo "self-test[zero-compiler-artifact-refusal]: FAIL (extract_artifacts exit $rc, expected 1)" >&2
    failures=$((failures + 1))
  fi

  # Fixture 8: an unknown mode is refused with exit 2, from BOTH
  # `set_mode_vars` directly and through `assert_boundary`'s own call to
  # it — never silently treated as a boundary violation (1) or a pass (0).
  rc=0
  set_mode_vars "not-a-real-mode" 2>/dev/null || rc=$?
  total=$((total + 1))
  if [ "$rc" -eq 2 ]; then
    echo "self-test[unknown-mode-set_mode_vars]: OK (exit 2)"
  else
    echo "self-test[unknown-mode-set_mode_vars]: FAIL (exit $rc, expected 2)" >&2
    failures=$((failures + 1))
  fi
  rc=0
  out="$(_run_assert_boundary "not-a-real-mode" /dev/null)" || rc=$?
  total=$((total + 1))
  if [ "$rc" -eq 2 ]; then
    echo "self-test[unknown-mode-assert_boundary]: OK (exit 2)"
  else
    echo "self-test[unknown-mode-assert_boundary]: FAIL (exit $rc, expected 2)" >&2
    printf '%s\n' "$out" >&2
    failures=$((failures + 1))
  fi

  if [ "$failures" -gt 0 ]; then
    echo "check-client-deps --self-test: $failures/$total fixture(s) FAILED" >&2
    return 1
  fi
  echo "check-client-deps --self-test: all $total fixture(s) passed."
  return 0
}

mode="${1:?usage: check_client_deps.sh substrate|cli|--self-test}"

if [ "$mode" = "--assert-boundary" ]; then
  # Internal-only mode, never a caller-facing one (not documented in the
  # module doc's "Usage" list's first line): a genuinely separate PROCESS
  # for `_run_assert_boundary` to invoke via `bash "$0" --assert-boundary
  # <mode> <artifacts-file>` — see that function's own comment for why an
  # in-process `( set -e; … )` subshell cannot give `assert_boundary` the
  # same errexit contract production gives it.
  assert_mode="${2:?usage: check_client_deps.sh --assert-boundary <mode> <artifacts-file>}"
  assert_artifacts="${3:?usage: check_client_deps.sh --assert-boundary <mode> <artifacts-file>}"
  assert_boundary "$assert_mode" "$assert_artifacts"
  exit $?
fi

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
