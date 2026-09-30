#!/usr/bin/env bash
# Asserts every wheel named fits PyPI's per-file upload limit: 100 MiB unless
# PyPI raises it for a project (`MAX_FILESIZE` in warehouse). PyPI refuses a
# larger file with `400 File too large` at the tag's publish, after the other
# wheels of the same lockstep release are already up; this fails the wheel's
# own build instead, on the PR that grew it. One script holds the limit, so
# every workflow that publishes a wheel checks the same number
# (`ci/scripts/check_wheel_gates.py` holds that each one does).
#
# The size is read with `wc -c`, which the Linux and the macOS wheel legs
# both have (`stat`'s size flag differs between them).
#
# Usage:
#   assert_wheel_size.sh <wheel>…
#     Prints each wheel's size and its headroom under the limit; exits 1 if
#     any is over, or if no wheel was named or a named one is missing (an
#     unexpanded glob must not pass as "nothing over the limit").
#   assert_wheel_size.sh --self-test
#     Drives the comparison against synthetic sizes and real files; exits 0
#     iff every fixture matches.
set -euo pipefail

PYPI_FILE_LIMIT=$((100 * 1024 * 1024))

# Returns 0 iff a file of `$1` bytes fits a limit of `$2` bytes. The limit is
# inclusive: warehouse refuses a file only when its size EXCEEDS the limit.
_fits() {
  [ "$1" -le "$2" ]
}

_size_of() {
  wc -c < "$1" | tr -d '[:space:]'
}

# Checks every path against limit `$1`; prints one line per wheel and returns
# 1 if any is missing or over.
_check_wheels() {
  local limit="$1"
  shift
  if [ "$#" -eq 0 ]; then
    echo "::error::assert_wheel_size.sh: no wheel named" >&2
    return 1
  fi
  local wheel size failures=0
  for wheel in "$@"; do
    if [ ! -f "$wheel" ]; then
      echo "::error::assert_wheel_size.sh: no such file: $wheel" >&2
      failures=$((failures + 1))
      continue
    fi
    size="$(_size_of "$wheel")"
    if _fits "$size" "$limit"; then
      echo "assert-wheel-size: OK -- $wheel is $size bytes, $((limit - size)) under PyPI's per-file limit of $limit"
    else
      echo "::error::assert_wheel_size.sh: $wheel is $size bytes, $((size - limit)) over PyPI's per-file limit of $limit" >&2
      failures=$((failures + 1))
    fi
  done
  [ "$failures" -eq 0 ]
}

_self_test() {
  local failures=0 total=0
  _expect() {
    local name="$1" want="$2" got="$3"
    total=$((total + 1))
    if [ "$got" -eq "$want" ]; then
      echo "self-test[$name]: OK"
    else
      echo "self-test[$name]: FAIL (rc=$got, expected $want)" >&2
      failures=$((failures + 1))
    fi
  }
  local rc work
  work="$(mktemp -d)"
  trap 'rm -rf "$work"' RETURN

  rc=0; _fits 104811684 "$PYPI_FILE_LIMIT" || rc=$?
  _expect "under-the-limit-fits" 0 "$rc"
  # The limit is inclusive: a file of exactly the limit is accepted.
  rc=0; _fits "$PYPI_FILE_LIMIT" "$PYPI_FILE_LIMIT" || rc=$?
  _expect "exactly-the-limit-fits" 0 "$rc"
  rc=0; _fits $((PYPI_FILE_LIMIT + 1)) "$PYPI_FILE_LIMIT" || rc=$?
  _expect "one-byte-over-refused" 1 "$rc"
  # The limit is 100 MiB, not 100 MB: a file between the two fits.
  rc=0; _fits 104000000 "$PYPI_FILE_LIMIT" || rc=$?
  _expect "over-100-megabytes-under-100-mebibytes-fits" 0 "$rc"

  printf '%0100d' 0 > "$work/fits.whl"
  printf '%0101d' 0 > "$work/over.whl"
  rc=0; [ "$(_size_of "$work/fits.whl")" -eq 100 ] || rc=$?
  _expect "size-read-is-the-byte-count" 0 "$rc"
  rc=0; _check_wheels 100 "$work/fits.whl" > /dev/null 2>&1 || rc=$?
  _expect "file-at-the-limit-passes" 0 "$rc"
  rc=0; _check_wheels 100 "$work/over.whl" > /dev/null 2>&1 || rc=$?
  _expect "file-over-the-limit-fails" 1 "$rc"
  # One wheel over fails the set, whatever its position.
  rc=0; _check_wheels 100 "$work/over.whl" "$work/fits.whl" > /dev/null 2>&1 || rc=$?
  _expect "one-over-fails-the-set" 1 "$rc"
  rc=0; _check_wheels 100 "$work/dist/*.whl" > /dev/null 2>&1 || rc=$?
  _expect "unexpanded-glob-fails" 1 "$rc"
  rc=0; _check_wheels 100 > /dev/null 2>&1 || rc=$?
  _expect "no-wheel-named-fails" 1 "$rc"

  if [ "$failures" -gt 0 ]; then
    echo "assert-wheel-size --self-test: $failures fixture(s) FAILED" >&2
    return 1
  fi
  echo "assert-wheel-size --self-test: all $total fixture(s) passed."
}

if [ "${1:-}" = "--self-test" ]; then
  _self_test
  exit $?
fi

_check_wheels "$PYPI_FILE_LIMIT" "$@"
