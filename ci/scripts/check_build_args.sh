#!/usr/bin/env bash
# Every line of a `docker build` build-args block is `KEY=VALUE` with a
# non-empty value. An explicitly empty `--build-arg NAME=` overrides a
# Dockerfile ARG's default with blank: a FROM line fails closed on it, but a
# RUN or ENV reading it builds on silently with the wrong value. A caller that
# does not need to override an argument omits the line.
#
#   bash ci/scripts/check_build_args.sh <<< "$BUILD_ARGS"
set -euo pipefail

while IFS= read -r line; do
  [ -n "$line" ] || continue
  case "$line" in
    *=*) ;;
    *) echo "::error::build-args line '$line' is not KEY=VALUE" >&2; exit 1 ;;
  esac
  name="${line%%=*}"
  value="${line#*=}"
  if [ -z "$value" ]; then
    echo "::error::build-arg $name is empty -- omit the line to keep the Dockerfile's default, or pass a value" >&2
    exit 1
  fi
done
