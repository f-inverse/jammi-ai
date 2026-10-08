#!/usr/bin/env bash
# A crates.io trusted-publishing token: minted from this job's GitHub OIDC
# identity, revoked when the publish is done. Part of the publish program
# (`publish_crates.sh` consumes the token), so it lives beside it rather than
# in a workflow step.
#
#   bash ci/scripts/crates_io_token.sh mint     # prints token=<masked> for $GITHUB_OUTPUT
#   bash ci/scripts/crates_io_token.sh revoke   # CARGO_REGISTRY_TOKEN names the token
#
# `mint` asks the Actions OIDC endpoint (ACTIONS_ID_TOKEN_REQUEST_URL/_TOKEN,
# present when the job grants `id-token: write`) for a JWT with audience
# `crates.io`, exchanges it at the registry's trusted-publishing endpoint, and
# retries the whole exchange with backoff: a failed or repeated exchange
# creates no state on crates.io, so retrying it is safe. The token is masked
# in the log before it is written.
set -euo pipefail

REGISTRY="${CARGO_REGISTRY_URL:-https://crates.io}"
AUDIENCE="${REGISTRY#https://}"
AUDIENCE="${AUDIENCE#http://}"
ENDPOINT="$REGISTRY/api/v1/trusted_publishing/tokens"
USER_AGENT="jammi-ai release (https://github.com/${GITHUB_REPOSITORY:-f-inverse/jammi-ai})"
ATTEMPTS="${CRATES_IO_TOKEN_ATTEMPTS:-5}"

mint_once() {
  local jwt token
  jwt="$(curl -fsS -H "Authorization: bearer ${ACTIONS_ID_TOKEN_REQUEST_TOKEN:?}" \
    "${ACTIONS_ID_TOKEN_REQUEST_URL:?}&audience=${AUDIENCE}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["value"])')"
  token="$(curl -fsS -X POST -H "Content-Type: application/json" -H "User-Agent: $USER_AGENT" \
    --data "$(python3 -c 'import json,sys; print(json.dumps({"jwt": sys.argv[1]}))' "$jwt")" "$ENDPOINT" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])')"
  [ -n "$token" ] || return 1
  printf '%s' "$token"
}

case "${1:-}" in
  mint)
    delay=2
    for attempt in $(seq 1 "$ATTEMPTS"); do
      if token="$(mint_once)"; then
        echo "::add-mask::$token"
        echo "token=$token" >> "${GITHUB_OUTPUT:-/dev/stdout}"
        echo "crates.io token minted (attempt $attempt)"
        exit 0
      fi
      if [ "$attempt" -lt "$ATTEMPTS" ]; then
        echo "::warning::crates.io token exchange failed (attempt $attempt/$ATTEMPTS); retrying in ${delay}s"
        sleep "$delay"; delay=$((delay * 2))
      fi
    done
    echo "::error::crates.io token exchange failed after $ATTEMPTS attempts" >&2
    exit 1
    ;;
  revoke)
    curl -fsS -X DELETE -H "Authorization: Bearer ${CARGO_REGISTRY_TOKEN:?}" -H "User-Agent: $USER_AGENT" "$ENDPOINT" >/dev/null
    echo "crates.io token revoked"
    ;;
  *) echo "usage: $0 mint|revoke" >&2; exit 2 ;;
esac
