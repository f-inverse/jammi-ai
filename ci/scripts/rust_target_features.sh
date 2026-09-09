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
# not a violation of it. `.github/workflows/ci.yml`'s `arm64-floor-oracle`
# job additionally proves the floor on a REAL `ubuntu-24.04-arm` runner
# (this file only proves the config file parses; that job proves the parsed
# flag actually reaches `RUSTFLAGS` and then `rustc` itself).
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
#     uses today) — never a bare string value, never a multi-line array,
#     never a trailing `# comment` after the closing `]` (the line must
#     literally END with `]`);
#   - every array element must be its own plain double-quoted string, with
#     no trailing comma before the closing `]` (a trailing comma leaves the
#     array's own closing quote missing, which this script rejects the same
#     way as any other malformed boundary);
#   - a target-feature flag must be spelled as TWO separate, paired
#     elements — a standalone `"-C"` immediately followed by
#     `"target-feature=<value>"` — never combined into one element
#     (`"-C target-feature=+fp16"`) and never spelled `"-Ctarget-feature=
#     ..."` with no separating space;
#   - a COMMA INSIDE ONE VALUE is accepted, not rejected: elements are
#     split on the literal `", "` (quote-comma-space) boundary BETWEEN
#     quoted strings, not on every bare comma, so
#     `"target-feature=+fp16,+dotprod"` — Rust's own canonical
#     multi-feature spelling, and the likely next edit to this stanza — is
#     ONE value, `+fp16,+dotprod`, not a parse error.
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
#     positive case) plus eight synthetic fixtures, each reproducing ONE
#     FAIL shape above, driven against ephemeral temp files (a COPY of the
#     real file, in the fixtures that need real content to mutate --
#     never the real file's own path passed to a fixture that expects a
#     different answer than the real file would give). No network, no
#     python.
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
        print "[target." triple "] rustflags array does not close on the same line (a multi-line array or a trailing comment after the closing ] are both unsupported shapes) in " config_path > "/dev/stderr"
        exit 2
      }
      inner = substr(rustflags_line, 2, length(rustflags_line) - 2)
      if (length(inner) == 0) {
        print "[target." triple "] rustflags array is empty in " config_path > "/dev/stderr"
        exit 2
      }
      if (substr(inner, 1, 1) != "\"" || substr(inner, length(inner), 1) != "\"") {
        print "[target." triple "] rustflags array elements must each be a plain double-quoted string (a trailing comma before ] is one way to land here) in " config_path > "/dev/stderr"
        exit 2
      }
      # Split on the literal boundary BETWEEN quoted elements ("\", \"" --
      # quote, comma, space), never on every bare comma: a comma INSIDE one
      # value (e.g. "target-feature=+fp16,+dotprod", Rust'\''s own
      # multi-feature spelling) has no following space and so is never
      # mistaken for an element boundary. The two outer quotes (the
      # array'\''s own first and last) are stripped first so every piece
      # `split` returns is already a bare, dequoted value.
      body = substr(inner, 2, length(inner) - 2)
      n = split(body, raw, /", "/)
      for (i = 1; i <= n; i++) {
        val = raw[i]
        if (index(val, "\"") > 0) {
          print "[target." triple "] rustflags array element " i " (\"" val "\") is not a plain double-quoted string in " config_path > "/dev/stderr"
          exit 2
        }
        elements[i] = val
      }
      m = n

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

  # 1. Positive: the REAL config file this repo ships -- the one fixture
  # that reads it directly, since it is the only one asserting the answer
  # the real file is SUPPOSED to give.
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

  # 5. Unknown triple: no [target.<x>] stanza at all -- against a COPY of
  # the real file (never the real file's own path passed to a fixture
  # expecting an answer the real file does NOT give).
  cp "$DEFAULT_CONFIG_TOML" "$tmpdir/unknown-triple.toml"
  set +e
  got="$(_extract totally-unknown-target-triple "$tmpdir/unknown-triple.toml" 2>/dev/null)"
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

  # 7. Missing rustflags key entirely: the stanza exists, but never
  # declares rustflags at all -- a DIFFERENT shape than #2 (which has a
  # well-formed empty-of-features array); this one has no array at all.
  cat > "$tmpdir/no-rustflags-key.toml" << 'EOF'
[target.aarch64-unknown-linux-gnu]
some-other-key = "irrelevant"
EOF
  set +e
  got="$(_extract aarch64-unknown-linux-gnu "$tmpdir/no-rustflags-key.toml" 2>/dev/null)"
  rc=$?
  set -e
  if [ "$rc" -eq 2 ]; then
    echo "self-test[missing rustflags key -> exit 2]: OK"
  else
    echo "self-test[missing rustflags key -> exit 2]: FAIL (rc=$rc, got='$got')" >&2
    failures=$((failures + 1))
  fi

  # 8. Multi-line array: the closing ] is not on the same line as
  # `rustflags =` -- an unsupported shape this script refuses rather than
  # silently truncates.
  cat > "$tmpdir/multiline.toml" << 'EOF'
[target.aarch64-unknown-linux-gnu]
rustflags = [
  "-C", "target-feature=+fp16"
]
EOF
  set +e
  got="$(_extract aarch64-unknown-linux-gnu "$tmpdir/multiline.toml" 2>/dev/null)"
  rc=$?
  set -e
  if [ "$rc" -eq 2 ]; then
    echo "self-test[multi-line array -> exit 2]: OK"
  else
    echo "self-test[multi-line array -> exit 2]: FAIL (rc=$rc, got='$got')" >&2
    failures=$((failures + 1))
  fi

  # 9. Unpaired target-feature=: a bare target-feature element with no
  # preceding standalone "-C" element at all.
  cat > "$tmpdir/unpaired.toml" << 'EOF'
[target.aarch64-unknown-linux-gnu]
rustflags = ["target-feature=+fp16"]
EOF
  set +e
  got="$(_extract aarch64-unknown-linux-gnu "$tmpdir/unpaired.toml" 2>/dev/null)"
  rc=$?
  set -e
  if [ "$rc" -eq 2 ]; then
    echo "self-test[unpaired target-feature= -> exit 2]: OK"
  else
    echo "self-test[unpaired target-feature= -> exit 2]: FAIL (rc=$rc, got='$got')" >&2
    failures=$((failures + 1))
  fi

  if [ "$failures" -gt 0 ]; then
    echo "rust_target_features.sh --self-test: $failures fixture(s) FAILED" >&2
    return 1
  fi
  echo "rust_target_features.sh --self-test: all 9 fixture(s) passed."
  return 0
}

if [ "${1:-}" = "--self-test" ]; then
  _self_test
  exit $?
fi

triple="${1:?usage: rust_target_features.sh <target-triple> | rust_target_features.sh --self-test}"
_extract "$triple" "$DEFAULT_CONFIG_TOML"
