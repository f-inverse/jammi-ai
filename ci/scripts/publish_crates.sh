#!/usr/bin/env bash
# Publishes the workspace crates to crates.io in topological dependency order
# (invoked from crates.yml's `publish` job with CARGO_REGISTRY_TOKEN set from
# the trusted-publishing OIDC exchange, and INPUT_VERSION carrying the manual
# dispatch's version input — empty on a tag push).
set -euo pipefail

# Publish in topological dependency order. Each entry must come
# after every crate it depends on, since `cargo publish` resolves
# workspace deps against crates.io once the path = is stripped.
# `cargo publish -p <crate> --no-verify` carries no explicit
# `--no-default-features`/`--features` flag, so it resolves each
# crate's OWN default feature set — for `jammi-lora` that is
# `default = ["candle"]` (`crates/jammi-lora/Cargo.toml`), which
# enables `dep:jammi-kernels`. The edges below are each
# publishable crate's normal (non-dev, non-build) dependency set
# as reported by `cargo metadata --format-version 1 --locked`
# (`packages[].dependencies`, filtered to `kind == null` and to
# names in this same list) — read off the resolved graph, not
# asserted from the Cargo.toml prose, so this comment cannot
# drift the way the previous version of it did (it called
# `jammi-lora` a leaf, which stopped being true the moment its
# `candle` feature grew an optional `jammi-kernels` dependency,
# and nothing re-verified the publish order against that change):
#   - jammi-numerics -> (none)
#   - jammi-db -> jammi-numerics
#   - jammi-kernels -> (none)
#   - jammi-lora -> jammi-kernels, jammi-numerics
#   - jammi-encoders -> jammi-kernels, jammi-lora, jammi-numerics
#   - jammi-wire -> jammi-db, jammi-lora, jammi-numerics
#   - jammi-admin -> jammi-db, jammi-wire
#   - jammi-client -> jammi-admin, jammi-db, jammi-wire
#   - jammi-ai -> jammi-db, jammi-encoders, jammi-lora, jammi-numerics, jammi-wire
#   - jammi-server -> jammi-ai, jammi-db, jammi-numerics, jammi-wire
#   - jammi-cli -> jammi-admin, jammi-db
# The presence guard skips crates that don't exist at this tag's
# checkout — pre-S2 tags don't have jammi-numerics, and pre-C2
# tags don't have jammi-kernels.
PUBLISH_ORDER=(
  jammi-numerics
  jammi-db
  jammi-kernels
  jammi-lora
  jammi-encoders
  jammi-wire
  jammi-admin
  jammi-client
  jammi-ai
  jammi-server
  jammi-cli
)

# Bounded exponential-backoff retry for a possibly-transient
# failure. Runs CMD (the remaining args), capturing combined
# stdout+stderr. IDEMPOTENT_CHECK, when non-empty, is a function
# name invoked (with no args, via the caller's closed-over
# variables) to detect that CMD's effect already landed even though
# CMD itself reported failure — crates.io can answer a transient
# 5xx *after* a publish that actually succeeded, and a blind retry
# would then hit "already uploaded" and look like a hard failure.
# Classification order: idempotent-check first (already landed —
# done), then version-conflict (never transient — fail fast), then
# transient network/5xx signals (retry), then anything else (fail
# fast). Never retries a version conflict.
retry_transient() {
  local desc="$1" idempotent_check="$2"
  shift 2
  local attempt=1 max_attempts=5 delay=2 output rc

  while true; do
    # `output=$(cmd) || rc=$?` — NOT `if output=$(cmd); then … fi`:
    # under `set -e`, a bare failing `output=$(cmd)` would abort the
    # whole script before we get a chance to classify/retry it; and
    # wrapping it in an `if` with no `else` resets `$?` to 0 once
    # the (false) condition is evaluated, losing cmd's real code.
    # The `||` form both survives `set -e` (the failure is "used")
    # and preserves the real exit code in `rc`.
    rc=0
    output=$("$@" 2>&1) || rc=$?
    if [ "$rc" -eq 0 ]; then
      printf '%s\n' "$output"
      return 0
    fi
    printf '%s\n' "$output" >&2

    if [ -n "$idempotent_check" ] && "$idempotent_check"; then
      echo "${desc}: effect already landed despite the reported failure — treating as success"
      return 0
    fi

    if printf '%s' "$output" | grep -qiE 'already uploaded|already exists'; then
      echo "::error::${desc}: version conflict — not transient, failing fast" >&2
      return "$rc"
    fi

    if ! printf '%s' "$output" | grep -qiE '(^|[^0-9])(500|502|503|504)([^0-9]|$)|service unavailable|spurious network error|failed to get successful http response|could not resolve host|connection reset|connection refused|operation timed out|network is unreachable'; then
      echo "::error::${desc}: non-transient failure — failing fast" >&2
      return "$rc"
    fi

    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "::error::${desc}: still failing after ${attempt} attempts" >&2
      return "$rc"
    fi

    echo "${desc}: transient failure (attempt ${attempt}/${max_attempts}) — retrying in ${delay}s" >&2
    sleep "$delay"
    delay=$(( delay * 2 ))
    attempt=$(( attempt + 1 ))
  done
}

# On a tag push, derive the version from the tag (v0.13.0 -> 0.13.0);
# on a manual dispatch, take it from the input (INPUT_VERSION, set by the
# workflow step's env from the dispatch input).
VERSION="${INPUT_VERSION:-}"
VERSION="${VERSION:-${GITHUB_REF_NAME#v}}"

# VERSION drives the presence probes and index waits below, but `cargo
# publish` publishes whatever the checked-out workspace says — so a VERSION
# that disagrees with the workspace would key every idempotence probe on the
# wrong quantity (a stale dispatch input could read "already published" and
# skip real work, green). Refuse the mismatch instead.
workspace_version="$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')"
if [ -z "$workspace_version" ] || [ "$VERSION" != "$workspace_version" ]; then
  echo "::error::VERSION '${VERSION}' (from the dispatch input or tag ref) does not match the workspace version '${workspace_version}' this checkout would publish — refusing" >&2
  exit 1
fi

# The crates.io sparse index lags `cargo publish` by seconds-to-minutes:
# after publishing crate X, a dependent crate Y's publish reads the index
# to resolve `X = "=<version>"` and fails if X's new version isn't listed
# yet. The presence guard polls the *API* (a different surface) and only
# protects re-runs; first-run propagation needs an *index* wait. So after
# each publish, block until the exact name+version appears in the sparse
# index before moving to the next (dependent) crate — a deterministic
# signal, not a blind sleep.
#
# Index path layout: 1-char names live under `1/`, 2-char under `2/`,
# 3-char under `3/<c1>/`, 4-plus under `<c1c2>/<c3c4>/`.
index_dir_prefix() {
  local name="$1"
  case "${#name}" in
    1) printf '1' ;;
    2) printf '2' ;;
    3) printf '3/%s' "${name:0:1}" ;;
    *) printf '%s/%s' "${name:0:2}" "${name:2:2}" ;;
  esac
}

wait_for_index() {
  local crate="$1" version="$2"
  local url="https://index.crates.io/$(index_dir_prefix "${crate}")/${crate}"
  local deadline=$(( SECONDS + 300 ))  # bound: 5 minutes, then fail loudly
  echo "Waiting for ${crate} ${version} to appear in the crates.io sparse index (${url})..."
  while (( SECONDS < deadline )); do
    if curl -sf -A "jammi-ai release CI (github.com/f-inverse/jammi-ai)" "${url}" \
      | grep -q "\"vers\":\"${version}\""; then
      echo "${crate} ${version} is visible in the sparse index"
      return 0
    fi
    sleep 5
  done
  echo "::error::${crate} ${version} did not appear in the crates.io sparse index within 5 minutes"
  return 1
}
# crates.io's API rejects requests with no User-Agent (403), which a
# bare `curl -sf` cannot distinguish from "not published" — it would
# then try to republish an existing crate and fail. Send a UA so the
# existence check is accurate on a re-run. Shared by the presence
# guard below and by retry_transient's idempotency recheck.
crate_present_on_crates_io() {
  local crate="$1" version="$2"
  curl -sf -A "jammi-ai release CI (github.com/f-inverse/jammi-ai)" \
    "https://crates.io/api/v1/crates/${crate}/${version}" >/dev/null
}

for crate in "${PUBLISH_ORDER[@]}"; do
  if [ ! -f "crates/${crate}/Cargo.toml" ]; then
    echo "${crate} not present at this tag — skipping"
    continue
  fi
  if crate_present_on_crates_io "${crate}" "${VERSION}"; then
    echo "${crate} ${VERSION} already on crates.io — skipping"
  else
    echo "Publishing ${crate} ${VERSION}"
    # --no-verify: the `validate` job already compiled and tested
    # the whole workspace, so cargo's per-crate pre-publish recompile is
    # redundant — and recompiling the full arrow/candle/datafusion tree
    # once per crate overflows the runner's disk on the last crate.
    #
    # publish_idempotent_check closes over $crate/$VERSION so
    # retry_transient can re-run it (by name) with no args after a
    # reported failure, to detect a publish that actually landed
    # despite crates.io answering with a transient error.
    publish_idempotent_check() { crate_present_on_crates_io "${crate}" "${VERSION}"; }
    retry_transient "publish ${crate} ${VERSION}" publish_idempotent_check \
      cargo publish -p "${crate}" --no-verify
    # Block until this exact version is resolvable by the next crate's
    # publish — only for crates this run actually published (the guard's
    # skip branch needs no wait; it's already in the index).
    wait_for_index "${crate}" "${VERSION}"
  fi
done
