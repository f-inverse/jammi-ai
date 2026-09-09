#!/usr/bin/env bash
# Extracts every `-C target-feature=<value>` entry from `.cargo/config.toml`'s
# `[target.<triple>]` `rustflags` array — the single source of truth for a
# target's ISA-floor flags, so `.github/actions/setup-rust-ci/action.yml`'s
# CI-side re-application (an exported RUSTFLAGS replaces, never merges with,
# config-file rustflags — see that file's own comment) never carries its own,
# second, driftable copy of the same literal (e.g. `+fp16`).
#
# Same precedent as `rust_pin.sh` reading `rust-toolchain.toml`: a pinned
# value lives in exactly one file, and CI reads it back instead of repeating
# it. Not a `check_*` script (nothing here asserts a repo invariant on every
# run — it is a value extractor a caller consumes), so it carries no
# `check_ci_guard_wiring.py` wiring obligation for the extraction itself, the
# same way `rust_pin.sh` carries none; its `--self-test` IS wired into
# `ci.yml`'s guard matrix (unlike `rust_pin.sh`'s, which has none) because
# this file's own accepted grammar is a real, driftable invariant this repo
# owns and must keep proving on every run — a script with no oracle anywhere
# in CI is exactly the class this repo's own `check_ci_guard_wiring.py`
# exists to catch for `check_*`-named scripts; wiring a non-`check_*`
# script's self-test by name is a stricter choice than that gate requires,
# not a violation of it.
#
# THIS SCRIPT PINS A NARROW, STRICT GRAMMAR — it is not a general TOML
# parser, and does not try to be one:
#   - the target's stanza must be the LITERAL header `[target.<triple>]`
#     (never a `cfg(...)`-keyed stanza for the same underlying arch — this
#     repo does not use that form, and a caller who introduced one without
#     updating this script must not be served a silently-empty answer);
#   - `rustflags` must be a bracketed ARRAY on ONE line, e.g.
#     `rustflags = ["-C", "link-arg=-fuse-ld=mold", "-C",
#     "target-feature=+fp16"]` (the exact shape this repo's config file
#     uses today) — never a bare string value, never a multi-line array;
#   - every array element must be its own plain double-quoted string;
#   - a target-feature flag must be spelled as TWO separate, paired
#     elements — a standalone `"-C"` immediately followed by
#     `"target-feature=<value>"` — never combined into one element
#     (`"-C target-feature=+fp16"`) and never spelled `"-Ctarget-feature=
#     ..."` with no separating space.
#
# ANY shape outside that grammar is a HARD FAILURE (exit 2, message to
# stderr) — including a well-formed stanza that simply has NO
# target-feature entry at all. This script never silently returns empty:
# `setup-rust-ci` relies on getting at least one feature back for
# `aarch64-unknown-linux-gnu`, and a quiet empty result is exactly the
# fail-OPEN shape this script exists to remove. A triple this repo has not
# pinned any feature for is therefore also a hard failure, not a "nothing to
# report" success — there is exactly one caller today (the aarch64-linux
# floor), and it always expects an answer.
#
# Usage:
#   rust_target_features.sh <target-triple>
#     Prints one feature VALUE (e.g. `+fp16`) per line to stdout, taken from
#     THIS REPO's own `.cargo/config.toml`. Exits 2 on any grammar violation
#     or on a well-formed-but-empty result (message to stderr either way).
#   rust_target_features.sh --self-test
#     Asserts against the REAL `.cargo/config.toml` this repo ships (the
#     positive case) plus five synthetic fixtures reproducing every FAIL
#     shape above, driven against ephemeral temp files, never this repo's
#     own file otherwise. No network, no python.
#
# Implemented in awk (POSIX-portable constructs only: gsub/index/split/
# substr/length) plus plain bash string handling — no python/tomllib
# dependency (the CI base image installs no system Python at all; the
# devcontainer's own Python is a `manylinux`-bundled interpreter symlinked
# in for a different reason entirely), and no `< <(process substitution)`
# (a masked exit code is exactly the class of bug this file exists to
# close on the CALLER side; this file holds itself to the same rule).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_CONFIG_TOML="$REPO_ROOT/.cargo/config.toml"

# _extract <triple> <config-toml-path>
# Prints accepted feature values, one per line, to stdout. Returns (not
# exits — callers, including --self-test, need the exit code without
# killing their own shell) awk's own exit status: 0 with at least one
# feature printed, or 2 with a message on stderr for every shape this
# script's pinned grammar (see the module doc) does not accept.
_extract() {
  local triple="$1"
  local config_path="$2"

  if [ ! -f "$config_path" ]; then
    echo "rust_target_features.sh: $config_path does not exist" >&2
    return 2
  fi

  awk -v triple="$triple" -v config_path="$config_path" '
    function trim(s) {
      gsub(/^[ \t]+|[ \t]+$/, "", s)
      return s
    }
    {
      line = trim($0)
      if (line ~ /^\[/) {
        if (in_stanza) {
          in_stanza = 0
        }
        header = line
        gsub(/^\[target\./, "", header)
        gsub(/\]$/, "", header)
        if (header == triple) {
          in_stanza = 1
          found_stanza = 1
        }
        next
      }
      if (in_stanza && rustflags_line == "" && line ~ /^rustflags[ \t]*=/) {
        eq = index(line, "=")
        rhs = trim(substr(line, eq + 1))
        rustflags_line = rhs
      }
    }
    END {
      if (!found_stanza) {
        print "no [target." triple "] stanza found in " config_path > "/dev/stderr"
        exit 2
      }
      if (rustflags_line == "") {
        print "[target." triple "] has no rustflags key in " config_path > "/dev/stderr"
        exit 2
      }
      if (substr(rustflags_line, 1, 1) != "[") {
        print "[target." triple "] rustflags is not a bracketed array (string form or other) in " config_path > "/dev/stderr"
        exit 2
      }
      if (substr(rustflags_line, length(rustflags_line), 1) != "]") {
        print "[target." triple "] rustflags array does not close on the same line (unsupported shape) in " config_path > "/dev/stderr"
        exit 2
      }
      inner = substr(rustflags_line, 2, length(rustflags_line) - 2)
      n = split(inner, raw, ",")
      m = 0
      for (i = 1; i <= n; i++) {
        el = trim(raw[i])
        if (length(el) < 2 || substr(el, 1, 1) != "\"" || substr(el, length(el), 1) != "\"") {
          print "[target." triple "] rustflags array element " i " (\"" el "\") is not a plain double-quoted string in " config_path > "/dev/stderr"
          exit 2
        }
        m++
        elements[m] = substr(el, 2, length(el) - 2)
      }

      found_feature = 0
      for (i = 1; i <= m; i++) {
        val = elements[i]
        if (index(val, "target-feature=") > 0) {
          if (val ~ /^target-feature=.+$/) {
            if (i > 1 && elements[i - 1] == "-C") {
              print substr(val, length("target-feature=") + 1)
              found_feature = 1
            } else {
              print "[target." triple "] element " i " (\"" val "\") is target-feature=... but not preceded by a standalone \"-C\" element in " config_path > "/dev/stderr"
              exit 2
            }
          } else {
            print "[target." triple "] element " i " (\"" val "\") combines a flag and target-feature= in one string -- only the paired \"-C\", \"target-feature=...\" element shape is accepted in " config_path > "/dev/stderr"
            exit 2
          }
        }
      }

      if (!found_feature) {
        print "[target." triple "] rustflags has no target-feature= entry in " config_path > "/dev/stderr"
        exit 2
      }
    }
  ' "$config_path"
}

_self_test() {
  local failures=0
  local got rc
  local tmpdir
  tmpdir="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmpdir'" RETURN

  # 1. Positive: the REAL config file this repo ships.
  set +e
  got="$(_extract aarch64-unknown-linux-gnu "$DEFAULT_CONFIG_TOML" 2>/dev/null)"
  rc=$?
  set -e
  if [ "$rc" -eq 0 ] && [ "$got" = "+fp16" ]; then
    echo "self-test[real config: aarch64-unknown-linux-gnu -> +fp16]: OK"
  else
    echo "self-test[real config: aarch64-unknown-linux-gnu -> +fp16]: FAIL (rc=$rc, got='$got')" >&2
    failures=$((failures + 1))
  fi

  # 2. A well-formed stanza with NO target-feature entry (this repo's own
  # x86_64 shape) -- exit 2, never a quiet empty success.
  cat > "$tmpdir/no-feature.toml" << 'EOF'
[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "link-arg=-fuse-ld=mold"]
EOF
  set +e
  got="$(_extract x86_64-unknown-linux-gnu "$tmpdir/no-feature.toml" 2>/dev/null)"
  rc=$?
  set -e
  if [ "$rc" -eq 2 ]; then
    echo "self-test[stanza with no target-feature -> exit 2]: OK"
  else
    echo "self-test[stanza with no target-feature -> exit 2]: FAIL (rc=$rc, got='$got')" >&2
    failures=$((failures + 1))
  fi

  # 3. Combined-element spelling: flag and value in ONE string.
  cat > "$tmpdir/combined.toml" << 'EOF'
[target.aarch64-unknown-linux-gnu]
rustflags = ["-C target-feature=+fp16"]
EOF
  set +e
  got="$(_extract aarch64-unknown-linux-gnu "$tmpdir/combined.toml" 2>/dev/null)"
  rc=$?
  set -e
  if [ "$rc" -eq 2 ]; then
    echo "self-test[combined-element spelling -> exit 2]: OK"
  else
    echo "self-test[combined-element spelling -> exit 2]: FAIL (rc=$rc, got='$got')" >&2
    failures=$((failures + 1))
  fi

  # 4. String form: rustflags is a bare string, not an array.
  cat > "$tmpdir/string-form.toml" << 'EOF'
[target.aarch64-unknown-linux-gnu]
rustflags = "-C target-feature=+fp16"
EOF
  set +e
  got="$(_extract aarch64-unknown-linux-gnu "$tmpdir/string-form.toml" 2>/dev/null)"
  rc=$?
  set -e
  if [ "$rc" -eq 2 ]; then
    echo "self-test[string-form rustflags -> exit 2]: OK"
  else
    echo "self-test[string-form rustflags -> exit 2]: FAIL (rc=$rc, got='$got')" >&2
    failures=$((failures + 1))
  fi

  # 5. Unknown triple: no [target.<x>] stanza at all in the real file.
  set +e
  got="$(_extract totally-unknown-target-triple "$DEFAULT_CONFIG_TOML" 2>/dev/null)"
  rc=$?
  set -e
  if [ "$rc" -eq 2 ]; then
    echo "self-test[unknown triple -> exit 2]: OK"
  else
    echo "self-test[unknown triple -> exit 2]: FAIL (rc=$rc, got='$got')" >&2
    failures=$((failures + 1))
  fi

  # 6. Remove-the-cause: a copy of the REAL stanza with +fp16 deleted --
  # the exact regression this script exists to catch.
  sed 's/, "-C", "target-feature=+fp16"//' "$DEFAULT_CONFIG_TOML" > "$tmpdir/removed.toml"
  set +e
  got="$(_extract aarch64-unknown-linux-gnu "$tmpdir/removed.toml" 2>/dev/null)"
  rc=$?
  set -e
  if [ "$rc" -eq 2 ]; then
    echo "self-test[remove-the-cause (+fp16 deleted) -> exit 2]: OK"
  else
    echo "self-test[remove-the-cause (+fp16 deleted) -> exit 2]: FAIL (rc=$rc, got='$got')" >&2
    failures=$((failures + 1))
  fi

  if [ "$failures" -gt 0 ]; then
    echo "rust_target_features.sh --self-test: $failures fixture(s) FAILED" >&2
    return 1
  fi
  echo "rust_target_features.sh --self-test: all 6 fixture(s) passed."
  return 0
}

if [ "${1:-}" = "--self-test" ]; then
  _self_test
  exit $?
fi

triple="${1:?usage: rust_target_features.sh <target-triple> | rust_target_features.sh --self-test}"
_extract "$triple" "$DEFAULT_CONFIG_TOML"
