#!/usr/bin/env bash
# Mocks-only, no-network regression suite for the RunPod REST v2 cluster
# primitives added to ci/scripts/runpod_lib.sh (plan #500 U7b, commit c1):
# `_rp_rest`, `_rp_entrypoint_setup` (shared with the pod payload's
# `_rp_deploy_payload`), `_rp_cluster_payload`, `rp_cluster_create`,
# `rp_cluster_get`, `rp_cluster_pods`, `rp_cluster_list`, `rp_cluster_delete`,
# `rp_cluster_sweep`, the shared `_rp_validate_force_hours` / `_rp_parse_ttl_
# seconds` (via `_rp_ttl_parser_pysrc`), and `rp_sweep`'s new cluster-member
# exclusion + terminate-refused reporting.
#
# Covers:
#   (1) the cluster create request body carries EXACTLY the documented
#       CreateClusterRequest keys (`unevaluatedProperties: false` — RunPod's
#       schema; the reviewed key-set fixture under ci/scripts/fixtures/ names
#       its own source/date so a schema drift is caught here, not live).
#   (2) `_rp_entrypoint_setup`'s output is byte-identical whether reached via
#       `_rp_deploy_payload` (the pod leg) or `_rp_cluster_payload` (the
#       cluster leg) for the same RP_TTL_HOURS — ONE watchdog+sshd string,
#       never two that could quietly drift apart.
#   (3) rp_cluster_sweep: an over-TTL cluster is deleted; an under-TTL one is
#       not; a non-jammi-prefixed name is never touched; a failed GET
#       (401/429/500) is rc 1 naming the status; an unparseable body is rc 1;
#       a failed DELETE (non-204) is rc 1 naming the cluster id; a 204 empty
#       body is success; the post-delete re-enumeration is fail-closed the
#       same way as the pre-delete one.
#   (4) `_rp_validate_force_hours` (shared by rp_sweep/rp_cluster_sweep): "0"
#       and "00" refuse (rc 2), a leading zero ("08") is accepted like any
#       other digit string, a non-digit string refuses by name, and an
#       overflow-shaped string still refuses (rc 2, bash's own arithmetic
#       error surfaces under the shared "hours must be > 0" wording).
#   (5) rp_sweep: a pod id in the cluster-member exclusion set is skipped
#       with the named "cluster member, skipped" line and NEVER reaches
#       `rp_terminate`; a `podTerminate` whose GraphQL body carries `errors`
#       is reported as "terminate refused: <reason>" AND — F3(a), plan #500
#       U7b fix round 1 — now fails the sweep's own exit (rc=1, naming
#       every refused id and reason): an unexpected refusal is never folded
#       into a silent 0.
#   (6) rp_cluster_sweep: a cluster with no usable createdAt (F3(b)) is
#       rc=1 naming the cluster id, never a silent `continue` back to a
#       green summary — an unexaminable resource is never "nothing to reap".
#   (7) rp_cluster_create: 201 with an id, 201 with no id (named distinctly
#       from an unparseable body), 201 unparseable, and a non-201 refusal —
#       round-4 audit F2 (this function had zero non-comment invocations
#       anywhere on this tree before this group existed).
#   (8) rp_cluster_get: 200 valid, 200 missing the 'id' key, 200
#       unparseable, 404, 500 — round-4 audit F2, same class as (7).
#   (9) rp_cluster_sweep's post-delete re-enumeration is three-valued: an
#       unparseable/empty/array second-GET body is rc=1 named "could NOT
#       confirm ... is gone", never aliased onto "confirmed gone" by an
#       uncaught Python exception's own default exit code — round-4 audit
#       F1 — and a 429 on that SAME second GET (MOCK_CLUSTER_LIST_STATUS_2,
#       never previously exercised by any fixture) is rc=1 named "could NOT
#       re-enumerate ... after deleting".
#  (10) rp_terminate: an unparseable podTerminate response body (an HTML
#       error page) is a REFUSED terminate via rp_sweep, never a silent
#       "swept" — round-4 audit advisory, elevated to a binding oracle.
#  (11) rp_sweep: a pod with no usable createdAt is rc=1, naming the pod
#       id, mirroring rp_cluster_sweep's own UNAGEABLE handling (6) —
#       round-4 audit P-M1a; this used to be a silent `continue`.
#
# Every network-facing call (`curl` — both the GraphQL pod surface and the
# REST v2 cluster surface) is mocked via a stub `curl` prepended onto PATH.
# `ssh` is never invoked by anything this suite drives.
#
# Run: bash ci/scripts/test_runpod_cluster_lib.sh
# Hermetic: no network, no GPU, no real RunPod account.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../.." && pwd)"

SANDBOX="$(mktemp -d)"
trap 'rm -rf "$SANDBOX"' EXIT

# Groups 3-5 run their assertions INSIDE a `( ... )` subshell (so each
# group's own `export MOCK_*` fixture selection cannot leak into the next
# group without an explicit reset) — a plain in-memory PASS/FAIL counter
# would be silently lost the moment each subshell exits. `RESULTS` is the
# same file-based tally test_gpu_dev_lifecycle.sh's own subshell groups use:
# every `ok`/`bad` call appends a line here regardless of which subshell it
# ran in, and the final tally (bottom of this file) counts lines, never a
# variable a subshell could have dropped.
RESULTS="$SANDBOX/results.log"
: > "$RESULTS"
ok()  { echo "PASS:$*" >> "$RESULTS"; echo "ok   - $*"; }
bad() { echo "FAIL:$*" >> "$RESULTS"; echo "FAIL - $*"; }

# ── shared fixture plumbing ─────────────────────────────────────────────────

STUBBIN="$SANDBOX/bin"
mkdir -p "$STUBBIN"
CALL_LOG="$SANDBOX/calls.log"
: > "$CALL_LOG"

# A stub `curl` answering BOTH shapes this file's functions issue:
#   * REST v2 (`_rp_rest`): recognized by `-X METHOD` + an
#     `https://api.runpod.io/v2/...` URL among the args. Status/body split
#     the same way `_rp_rest` itself splits a real response: `-o "$outfile"`
#     gets the body, this stub's own STDOUT carries the status (mirroring
#     `-w '%{http_code}'`).
#   * GraphQL (`rp_gql`, used by `rp_terminate`/the account query inside
#     rp_sweep): the payload arrives as a plain positional argument.
# Every call is logged (METHOD + URL, or the GraphQL payload) so a test can
# assert both what was and was not sent.
cat > "$STUBBIN/curl" <<'STUB'
#!/usr/bin/env bash
method="" url="" outfile="" payload=""
args=("$@")
i=0
while [ $i -lt ${#args[@]} ]; do
  a="${args[$i]}"
  case "$a" in
    -X) i=$((i + 1)); method="${args[$i]}" ;;
    -o) i=$((i + 1)); outfile="${args[$i]}" ;;
    https://api.runpod.io/v2/*) url="$a" ;;
    *podTerminate*|*'myself{'*) payload="$a" ;;
  esac
  i=$((i + 1))
done
if [ -n "$url" ]; then
  printf 'REST %s %s\n' "$method" "$url" >> "${MOCK_CALL_LOG:-/dev/null}"
  n=0
  case "$url" in
    */v2/clusters)
      case "$method" in
        GET)
          [ -f "${MOCK_LIST_CALL_COUNTER:-/dev/null}" ] && n="$(cat "$MOCK_LIST_CALL_COUNTER")"
          n=$((n + 1))
          [ -n "${MOCK_LIST_CALL_COUNTER:-}" ] && echo "$n" > "$MOCK_LIST_CALL_COUNTER"
          if [ "$n" -le 1 ]; then
            status="${MOCK_CLUSTER_LIST_STATUS:-200}"
            resp="${MOCK_CLUSTER_LIST_RESPONSE:-}"
          else
            status="${MOCK_CLUSTER_LIST_STATUS_2:-${MOCK_CLUSTER_LIST_STATUS:-200}}"
            resp="${MOCK_CLUSTER_LIST_RESPONSE_2:-${MOCK_CLUSTER_LIST_RESPONSE:-}}"
          fi
          if [ -n "$resp" ] && [ -f "$resp" ]; then body="$(cat "$resp")"; else body='{"clusters":[]}'; fi
          ;;
        POST)
          status="${MOCK_CLUSTER_CREATE_STATUS:-201}"
          if [ -n "${MOCK_CLUSTER_CREATE_RESPONSE:-}" ] && [ -f "$MOCK_CLUSTER_CREATE_RESPONSE" ]; then
            body="$(cat "$MOCK_CLUSTER_CREATE_RESPONSE")"
          else
            body='{"id":"cl_test"}'
          fi
          # The REQUEST body (the JSON _rp_cluster_payload built) travels as
          # --data-binary, a plain positional arg -- log it distinctly so a
          # test can inspect exactly what was SENT, not just what came back.
          for a2 in "${args[@]}"; do
            case "$a2" in
              '{"name"'*|'{"'*'"name"'*) printf '%s\n' "$a2" >> "${MOCK_REQUEST_BODY_LOG:-/dev/null}" ;;
            esac
          done
          ;;
      esac ;;
    */v2/clusters/*/pods)
      status="${MOCK_CLUSTER_PODS_STATUS:-200}"
      if [ -n "${MOCK_CLUSTER_PODS_RESPONSE:-}" ] && [ -f "$MOCK_CLUSTER_PODS_RESPONSE" ]; then
        body="$(cat "$MOCK_CLUSTER_PODS_RESPONSE")"
      else
        body='{"pods":[]}'
      fi ;;
    */v2/clusters/*)
      case "$method" in
        DELETE)
          printf '%s\n' "$url" >> "${MOCK_DELETE_CALL_LOG:-/dev/null}"
          status="${MOCK_CLUSTER_DELETE_STATUS:-204}"
          body=""
          ;;
        GET)
          status="${MOCK_CLUSTER_GET_STATUS:-200}"
          if [ -n "${MOCK_CLUSTER_GET_RESPONSE:-}" ] && [ -f "$MOCK_CLUSTER_GET_RESPONSE" ]; then
            body="$(cat "$MOCK_CLUSTER_GET_RESPONSE")"
          else
            body='{"id":"cl_test"}'
          fi ;;
      esac ;;
  esac
  [ -n "$outfile" ] && printf '%s' "$body" > "$outfile"
  printf '%s' "$status"
  exit 0
fi
[ -n "$payload" ] && printf '%s\n' "$payload" >> "${MOCK_CALL_LOG:-/dev/null}"
case "$payload" in
  *podTerminate*)
    n=$(( $(cat "${MOCK_TERM_COUNTER:-/dev/null}" 2>/dev/null || echo 0) + 1 ))
    [ -n "${MOCK_TERM_COUNTER:-}" ] && echo "$n" > "$MOCK_TERM_COUNTER"
    case "$payload" in
      *"${MOCK_TERMINATE_HTML_ID:-__none__}"*) printf '%s' '<html><body>502 Bad Gateway</body></html>' ;;
      *"${MOCK_TERMINATE_REFUSE_ID:-__none__}"*) echo '{"errors":[{"message":"'"${MOCK_TERMINATE_REFUSE_REASON:-refused}"'"}]}' ;;
      *) echo '{"data":{"podTerminate":true}}' ;;
    esac ;;
  *'myself{'*)
    if [ -n "${MOCK_ACCOUNT_RESPONSE:-}" ] && [ -f "$MOCK_ACCOUNT_RESPONSE" ]; then
      cat "$MOCK_ACCOUNT_RESPONSE"
    else
      echo '{"data":{"myself":{"pods":[]}}}'
    fi ;;
  *) echo '{}' ;;
esac
STUB
chmod +x "$STUBBIN/curl"

export PATH="$STUBBIN:$PATH"
export RUNPOD_API_KEY="test-dummy-key"
export MOCK_CALL_LOG="$CALL_LOG"

reset_log() { : > "$CALL_LOG"; }
log_has() { grep -q -- "$1" "$CALL_LOG"; }   # $1=substring

iso_from_epoch() { # $1=epoch seconds -- portable across GNU and BSD `date`
  date -u -d "@$1" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -r "$1" +%Y-%m-%dT%H:%M:%SZ
}

# shellcheck source=ci/scripts/runpod_lib.sh
source "$DIR/runpod_lib.sh"
# shellcheck disable=SC2034  # read by _rp_deploy_payload/_rp_cluster_payload
# inside the sourced runpod_lib.sh (SC1091 above: shellcheck does not follow
# a dynamic `source`), not by anything in this file directly.
RP_PUBKEY="ssh-ed25519 AAAAFIXTURE test@fixture"

echo "=== RunPod cluster primitives: mocks-only regression suite ==="

# ═════════════════════════════════════════════════════════════════════════
# Group 1 — the cluster create request body carries exactly the documented
# CreateClusterRequest keys (a reviewed fixture, not a live openapi.json
# parse — no network in this suite).
# ═════════════════════════════════════════════════════════════════════════
(
  fixture="$REPO_ROOT/ci/scripts/fixtures/runpod_cluster_create_request_keys.json"
  payload="$(_rp_cluster_payload "NVIDIA A100 80GB PCIe")"
  python3 - "$payload" "$fixture" <<'PY'
import json, sys
payload_str, fixture_path = sys.argv[1], sys.argv[2]
payload = json.loads(payload_str)
fixture = json.load(open(fixture_path))
allowed = set(fixture["top_level_keys"])
required = set(fixture["top_level_required"])
compute_allowed = set(fixture["compute_keys"])
compute_required = set(fixture["compute_required"])
got = set(payload.keys())
extra = got - allowed
missing_required = required - got
if extra:
    print("EXTRA:%s" % ",".join(sorted(extra)))
elif missing_required:
    print("MISSING:%s" % ",".join(sorted(missing_required)))
else:
    compute_got = set((payload.get("compute") or {}).keys())
    compute_extra = compute_got - compute_allowed
    compute_missing = compute_required - compute_got
    if compute_extra:
        print("COMPUTE_EXTRA:%s" % ",".join(sorted(compute_extra)))
    elif compute_missing:
        print("COMPUTE_MISSING:%s" % ",".join(sorted(compute_missing)))
    else:
        print("OK")
PY
) > "$SANDBOX/g1.out"
if grep -qx "OK" "$SANDBOX/g1.out"; then
  ok "G1: _rp_cluster_payload emits exactly the documented CreateClusterRequest/ClusterCompute keys"
else
  bad "G1: _rp_cluster_payload's key set disagrees with the reviewed fixture ($(cat "$SANDBOX/g1.out"))"
fi

# ═════════════════════════════════════════════════════════════════════════
# Group 2 — the shared entrypoint: _rp_entrypoint_setup's output is
# byte-identical whether reached via the pod payload or the cluster payload.
# ═════════════════════════════════════════════════════════════════════════
(
  pod_json="$(_rp_deploy_payload SECURE "NVIDIA A100 80GB PCIe")"
  cluster_json="$(_rp_cluster_payload "NVIDIA A100 80GB PCIe")"
  python3 - "$pod_json" "$cluster_json" <<'PY'
import json, re, sys
pod = json.loads(sys.argv[1])
cluster = json.loads(sys.argv[2])
pod_args = pod["variables"]["i"]["dockerArgs"]
cluster_args = cluster["args"]
# Both wrap the SAME setup text in `bash -c '<setup>'` -- strip the wrapper
# and compare the inner text byte-for-byte.
def inner(s):
    m = re.match(r"^bash -c '(.*)'$", s, re.DOTALL)
    assert m, "unexpected wrapper shape: %r" % s
    return m.group(1)
p, c = inner(pod_args), inner(cluster_args)
print("SAME" if p == c else "DIFFERENT")
PY
) > "$SANDBOX/g2.out"
if grep -qx "SAME" "$SANDBOX/g2.out"; then
  ok "G2: the pod payload and the cluster payload share byte-identical entrypoint text (_rp_entrypoint_setup)"
else
  bad "G2: the pod and cluster entrypoint text DIFFER ($(cat "$SANDBOX/g2.out"))"
fi

# ═════════════════════════════════════════════════════════════════════════
# Group 3 — rp_cluster_sweep: age judgement, prefix scoping, failure arms.
# ═════════════════════════════════════════════════════════════════════════
now_epoch="$(date -u +%s)"
stale_iso="$(iso_from_epoch $((now_epoch - 50 * 3600)))"   # 50h old, ttl8, past deadline
fresh_iso="$(iso_from_epoch $((now_epoch - 1 * 3600)))"    # 1h old, ttl8, within deadline

cat > "$SANDBOX/g3-list.json" <<JSON
{"clusters":[
  {"id":"cl-stale","name":"jammi-cluster-ttl8","createdAt":"${stale_iso}"},
  {"id":"cl-fresh","name":"jammi-cluster-ttl8","createdAt":"${fresh_iso}"},
  {"id":"cl-other","name":"someone-else-cluster","createdAt":"${stale_iso}"}
]}
JSON
cat > "$SANDBOX/g3-list-after.json" <<JSON
{"clusters":[
  {"id":"cl-fresh","name":"jammi-cluster-ttl8","createdAt":"${fresh_iso}"},
  {"id":"cl-other","name":"someone-else-cluster","createdAt":"${stale_iso}"}
]}
JSON

(
  export MOCK_CLUSTER_LIST_RESPONSE="$SANDBOX/g3-list.json"
  export MOCK_CLUSTER_LIST_RESPONSE_2="$SANDBOX/g3-list-after.json"
  export MOCK_LIST_CALL_COUNTER="$SANDBOX/g3-list-counter"
  export MOCK_DELETE_CALL_LOG="$SANDBOX/g3-delete.log"
  rm -f "$MOCK_LIST_CALL_COUNTER" "$MOCK_DELETE_CALL_LOG"; : > "$MOCK_DELETE_CALL_LOG"
  out="$(rp_cluster_sweep 2>&1)"; rc=$?
  if [ "$rc" -eq 0 ]; then ok "G3: rp_cluster_sweep exits 0 against a healthy mixed fleet"; else bad "G3: expected rc=0 (got $rc): $out"; fi
  if grep -qx "https://api.runpod.io/v2/clusters/cl-stale" "$MOCK_DELETE_CALL_LOG"; then
    ok "G3: the over-TTL cluster (cl-stale) was deleted"
  else
    bad "G3: expected cl-stale to be deleted ($(cat "$MOCK_DELETE_CALL_LOG"))"
  fi
  if grep -qx "https://api.runpod.io/v2/clusters/cl-fresh" "$MOCK_DELETE_CALL_LOG"; then
    bad "G3: the under-TTL cluster (cl-fresh) must NOT be deleted (regression!)"
  else
    ok "G3: the under-TTL cluster (cl-fresh) was left alone"
  fi
  if grep -qx "https://api.runpod.io/v2/clusters/cl-other" "$MOCK_DELETE_CALL_LOG"; then
    bad "G3: a non-jammi-prefixed cluster (cl-other) must NEVER be touched (regression!)"
  else
    ok "G3: the non-jammi-prefixed cluster (cl-other) was never touched"
  fi
)

for status in 401 429 500; do
(
  export MOCK_CLUSTER_LIST_STATUS="$status"
  out="$(rp_cluster_sweep 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "could NOT enumerate clusters"; then
    ok "G3: a ${status} on the cluster GET is rc=1, named 'could NOT enumerate clusters'"
  else
    bad "G3: expected rc=1 naming the enumeration failure for status ${status} (got rc=$rc): $out"
  fi
)
done

(
  echo 'not json at all' > "$SANDBOX/g3-badjson.json"
  export MOCK_CLUSTER_LIST_RESPONSE="$SANDBOX/g3-badjson.json"
  out="$(rp_cluster_sweep 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "could NOT enumerate clusters"; then
    ok "G3: an unparseable cluster-list body is rc=1, named 'could NOT enumerate clusters'"
  else
    bad "G3: expected rc=1 on an unparseable body (got rc=$rc): $out"
  fi
)

(
  export MOCK_CLUSTER_LIST_RESPONSE="$SANDBOX/g3-list.json"
  export MOCK_CLUSTER_LIST_RESPONSE_2="$SANDBOX/g3-list.json"   # still shows cl-stale after "delete"
  export MOCK_LIST_CALL_COUNTER="$SANDBOX/g3-list-counter-b"
  rm -f "$MOCK_LIST_CALL_COUNTER"
  export MOCK_CLUSTER_DELETE_STATUS="204"
  out="$(rp_cluster_sweep 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "still present after delete"; then
    ok "G3: the post-delete re-enumeration is fail-closed — a cluster still listed after a 204 delete is reported by name, never trusted"
  else
    bad "G3: expected the post-delete confirmation to catch a still-listed cluster (rc=$rc): $out"
  fi
)

(
  export MOCK_CLUSTER_LIST_RESPONSE="$SANDBOX/g3-list.json"
  export MOCK_CLUSTER_DELETE_STATUS="409"
  out="$(rp_cluster_sweep 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "cl-stale"; then
    ok "G3: a non-204 DELETE is rc=1, naming the cluster id (cl-stale)"
  else
    bad "G3: expected rc=1 naming cl-stale on a non-204 delete (rc=$rc): $out"
  fi
)

(
  export MOCK_CLUSTER_LIST_RESPONSE="$SANDBOX/g3-list.json"
  export MOCK_CLUSTER_LIST_RESPONSE_2="$SANDBOX/g3-list-after.json"
  export MOCK_LIST_CALL_COUNTER="$SANDBOX/g3-list-counter-c"
  rm -f "$MOCK_LIST_CALL_COUNTER"
  export MOCK_CLUSTER_DELETE_STATUS="204"
  out="$(rp_cluster_sweep 2>&1)"; rc=$?
  if [ "$rc" -eq 0 ] && printf '%s' "$out" | grep -q "terminated 1 orphaned cluster"; then
    ok "G3: a 204 DELETE followed by a clean re-enumeration is a genuine success (rc=0)"
  else
    bad "G3: expected rc=0 on a clean delete+reconfirm (rc=$rc): $out"
  fi
)

(
  # F3(b): a cluster with no usable createdAt is an UNEXAMINABLE resource
  # -- it is BILLING with no deadline this sweep could establish -- never
  # "nothing to reap". Must be rc=1 naming the cluster id, never a silent
  # `continue` back to a green summary line.
  cat > "$SANDBOX/g3-unageable.json" <<'JSON'
{"clusters":[
  {"id":"cl-unageable","name":"jammi-cluster-ttl8"},
  {"id":"cl-fresh2","name":"jammi-cluster-ttl8","createdAt":"2099-01-01T00:00:00Z"}
]}
JSON
  export MOCK_CLUSTER_LIST_RESPONSE="$SANDBOX/g3-unageable.json"
  export MOCK_DELETE_CALL_LOG="$SANDBOX/g3-unageable-delete.log"
  rm -f "$MOCK_DELETE_CALL_LOG"; : > "$MOCK_DELETE_CALL_LOG"
  out="$(rp_cluster_sweep 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "cl-unageable"; then
    ok "G3 F3(b): a cluster with no usable createdAt is rc=1, naming the cluster id (cl-unageable)"
  else
    bad "G3 F3(b): expected rc=1 naming cl-unageable (got rc=$rc): $out"
  fi
  if [ -s "$MOCK_DELETE_CALL_LOG" ]; then
    bad "G3 F3(b): an unageable cluster must never be deleted on a guess (regression!)"
  else
    ok "G3 F3(b): the unageable cluster was left alone, not deleted on a guess"
  fi
)

# ═════════════════════════════════════════════════════════════════════════
# Group 4 — the shared force_hours validator (_rp_validate_force_hours),
# consulted identically by rp_sweep and rp_cluster_sweep.
# ═════════════════════════════════════════════════════════════════════════
(
  export MOCK_CLUSTER_LIST_RESPONSE="$SANDBOX/g3-list.json"
  export MOCK_CLUSTER_LIST_RESPONSE_2="$SANDBOX/g3-list-after.json"
  export MOCK_LIST_CALL_COUNTER="$SANDBOX/g4-list-counter"
  export MOCK_CLUSTER_DELETE_STATUS="204"
  for bad_val in 0 00; do
    out="$(rp_cluster_sweep "$bad_val" 2>&1)"; rc=$?
    if [ "$rc" -eq 2 ] && printf '%s' "$out" | grep -q "hours must be > 0"; then
      ok "G4: rp_cluster_sweep ${bad_val} refuses (rc=2, 'hours must be > 0')"
    else
      bad "G4: rp_cluster_sweep ${bad_val} expected rc=2 'hours must be > 0' (got rc=$rc): $out"
    fi
  done
  out="$(rp_cluster_sweep abc 2>&1)"; rc=$?
  if [ "$rc" -eq 2 ] && printf '%s' "$out" | grep -q "positive integer"; then
    ok "G4: rp_cluster_sweep abc refuses (rc=2, non-digit named)"
  else
    bad "G4: rp_cluster_sweep abc expected rc=2 naming a non-digit input (got rc=$rc): $out"
  fi
  out="$(rp_cluster_sweep 08 2>&1)"; rc=$?
  # 08 is accepted like any other digit string (decimal 8, never octal/rejected);
  # against g3-list.json (a fresh-relative-to-8h cluster and a 50h-old one), an
  # 8h force ceiling still sweeps the 50h-old one.
  if [ "$rc" -eq 0 ]; then
    ok "G4: rp_cluster_sweep 08 (leading zero) is accepted like '8', not rejected"
  else
    bad "G4: rp_cluster_sweep 08 expected rc=0 (got rc=$rc): $out"
  fi
  overflow="99999999999999999999"
  out="$(rp_cluster_sweep "$overflow" 2>&1)"; rc=$?
  if [ "$rc" -eq 2 ]; then
    ok "G4: rp_cluster_sweep <overflow> still refuses (rc=2)"
  else
    bad "G4: rp_cluster_sweep <overflow> expected rc=2 (got rc=$rc): $out"
  fi
)

# ═════════════════════════════════════════════════════════════════════════
# Group 5 — rp_sweep: cluster-member exclusion + terminate-refused, both
# layered on TOP of the pod-level age judgement (function-level; rp_gql is
# still the real one, going through the stub curl above).
# ═════════════════════════════════════════════════════════════════════════
(
  # Two clusters enumerate to make pod-member the exclusion set trivial via
  # the real _rp_cluster_member_ids -> rp_cluster_list/rp_cluster_pods path
  # (no override needed here).
  cat > "$SANDBOX/g5-clusters.json" <<JSON
{"clusters":[{"id":"cl-live","name":"jammi-cluster-ttl8","createdAt":"${fresh_iso}"}]}
JSON
  cat > "$SANDBOX/g5-pods.json" <<'JSON'
{"pods":[{"id":"pod-member","cluster":{"id":"cl-live","rank":0,"ip":"10.65.0.2"},"ssh":{"direct":{"host":"1.2.3.4","port":22}},"status":"RUNNING"}]}
JSON
  cat > "$SANDBOX/g5-account.json" <<JSON
{"data":{"myself":{"pods":[
  {"id":"pod-stale","name":"jammi-gpu-ttl8","desiredStatus":"RUNNING","createdAt":"${stale_iso}","runtime":{"uptimeInSeconds":180000}},
  {"id":"pod-refused","name":"jammi-gpu-ttl8","desiredStatus":"RUNNING","createdAt":"${stale_iso}","runtime":{"uptimeInSeconds":180000}},
  {"id":"pod-member","name":"jammi-gpu-ttl8","desiredStatus":"RUNNING","createdAt":"${stale_iso}","runtime":{"uptimeInSeconds":180000}}
]}}}
JSON
  export MOCK_CLUSTER_LIST_RESPONSE="$SANDBOX/g5-clusters.json"
  export MOCK_CLUSTER_PODS_RESPONSE="$SANDBOX/g5-pods.json"
  export MOCK_ACCOUNT_RESPONSE="$SANDBOX/g5-account.json"
  export MOCK_TERMINATE_REFUSE_ID="pod-refused"
  export MOCK_TERMINATE_REFUSE_REASON="pod is a cluster member"
  export MOCK_TERM_COUNTER="$SANDBOX/g5-term-counter"
  rm -f "$MOCK_TERM_COUNTER"
  out="$(rp_sweep 2>&1)"; rc=$?
  if printf '%s' "$out" | grep -q "cluster member, skipped: pod-member"; then
    ok "G5: rp_sweep skips a cluster member by name ('cluster member, skipped: pod-member') without ever calling terminate on it"
  else
    bad "G5: expected 'cluster member, skipped: pod-member' in rp_sweep's output: $out"
  fi
  if printf '%s' "$out" | grep -q 'podTerminate.*pod-member' 2>/dev/null; then
    bad "G5: rp_sweep must never issue podTerminate for a known cluster member (regression!)"
  fi
  if printf '%s' "$out" | grep -q "terminate refused: pod is a cluster member (pod pod-refused)"; then
    ok "G5: a refused podTerminate is reported by name ('terminate refused: pod is a cluster member (pod pod-refused)')"
  else
    bad "G5: expected the named terminate-refused line for pod-refused: $out"
  fi
  if printf '%s' "$out" | grep -q "swept pod pod-stale"; then
    ok "G5: an ordinary orphan (pod-stale) is still swept normally alongside the exclusion/refusal handling"
  else
    bad "G5: expected pod-stale to be swept: $out"
  fi
  # F3(a) (plan #500 U7b fix round 1): an unexpected terminate refusal (the
  # cluster-member CASE is already excluded upstream by the "cluster
  # member, skipped" arm above — this is a DIFFERENT pod RunPod's own API
  # refused for its own reason) is never folded into a silent 0: it is
  # counted and named, and the sweep's own exit is now non-zero.
  if [ "$rc" -eq 1 ]; then
    ok "G5: rp_sweep's own exit is now 1 — an unexpected terminate refusal is never folded into a silent 0"
  else
    bad "G5: expected rc=1 (F3a: a refusal is now fatal to the run's own exit) (got rc=$rc)"
  fi
  if printf '%s' "$out" | grep -q "1 terminate(s) refused: pod-refused: pod is a cluster member"; then
    ok "G5: the refusal summary names the pod id and the refused reason ('1 terminate(s) refused: pod-refused: pod is a cluster member')"
  else
    bad "G5: expected the F3(a) refusal summary naming pod-refused and its reason: $out"
  fi
)

# ═════════════════════════════════════════════════════════════════════════
# Group 6 — rp_sweep is FAIL-CLOSED on the cluster-member exclusion set: a
# failed `_rp_cluster_member_ids` (here, a 500 on the cluster GET) must
# terminate NOTHING — not even an ordinary, genuinely-orphaned pod — never
# proceed with an empty/incomplete exclusion set. RunPod's own podTerminate
# refusal on a member is UNMEASURED (`actions: []` is a listing attribute,
# never an observed refusal), so it is not a backstop this doctrine leans
# on; "could not check" never becomes "act anyway" (gpu-reap.yml's own
# doctrine for a failed pod enumeration, restated here for cluster members).
# ═════════════════════════════════════════════════════════════════════════
(
  export MOCK_CLUSTER_LIST_STATUS="500"
  export MOCK_ACCOUNT_RESPONSE="$SANDBOX/g5-account.json"
  reset_log
  out="$(rp_sweep 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ]; then
    ok "G6: rp_sweep returns rc=1 when the cluster-member exclusion set could not be established"
  else
    bad "G6: expected rc=1 on a failed cluster-member enumeration (got rc=$rc): $out"
  fi
  if printf '%s' "$out" | grep -q "sweep could NOT enumerate cluster members; pod sweep skipped"; then
    ok "G6: the refusal is named ('sweep could NOT enumerate cluster members; pod sweep skipped: <reason>')"
  else
    bad "G6: expected the named 'sweep could NOT enumerate cluster members; pod sweep skipped' line: $out"
  fi
  if log_has "podTerminate"; then
    bad "G6: rp_sweep must issue ZERO podTerminate calls when it could not establish the exclusion set (regression!)"
  else
    ok "G6: zero podTerminate calls were recorded — nothing was terminated, not even the ordinary orphan pod-stale"
  fi
  if printf '%s' "$out" | grep -q "swept pod\|terminate refused\|cluster member, skipped"; then
    bad "G6: no pod outcome line (swept/refused/skipped) may appear — the whole pod sweep is skipped, not just the terminations: $out"
  else
    ok "G6: no per-pod outcome line appears — the entire pod sweep was skipped, not merely the terminations within it"
  fi
)

# ═════════════════════════════════════════════════════════════════════════
# Group 7 — rp_cluster_create: every documented arm actually driven through
# the stub (round-4 audit F2: this suite's own header named this function
# since commit c1, but zero non-comment invocations existed anywhere on
# this tree). The 201-with-no-id and 201-unparseable-body arms must be
# named DIFFERENTLY (P-M1a): "no id" is a well-formed body missing a key,
# "unparseable" is a body that could not be read at all.
# ═════════════════════════════════════════════════════════════════════════
(
  echo '{"id":"cl_new"}' > "$SANDBOX/g7-create-ok.json"
  export MOCK_CLUSTER_CREATE_STATUS="201"
  export MOCK_CLUSTER_CREATE_RESPONSE="$SANDBOX/g7-create-ok.json"
  out="$(rp_cluster_create "NVIDIA A100 80GB PCIe" 2>&1)"; rc=$?
  if [ "$rc" -eq 0 ] && [ "$out" = "cl_new" ]; then
    ok "G7: rp_cluster_create prints the id on a 201 with an id"
  else
    bad "G7: expected rc=0 id=cl_new (got rc=$rc): $out"
  fi
)
(
  echo '{"name":"no-id-here"}' > "$SANDBOX/g7-create-noid.json"
  export MOCK_CLUSTER_CREATE_STATUS="201"
  export MOCK_CLUSTER_CREATE_RESPONSE="$SANDBOX/g7-create-noid.json"
  out="$(rp_cluster_create "NVIDIA A100 80GB PCIe" 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "carried no id"; then
    ok "G7: rp_cluster_create names a 201 with no id key ('carried no id')"
  else
    bad "G7: expected rc=1 'carried no id' (got rc=$rc): $out"
  fi
)
(
  echo 'not json at all' > "$SANDBOX/g7-create-badjson.json"
  export MOCK_CLUSTER_CREATE_STATUS="201"
  export MOCK_CLUSTER_CREATE_RESPONSE="$SANDBOX/g7-create-badjson.json"
  out="$(rp_cluster_create "NVIDIA A100 80GB PCIe" 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "unparseable"; then
    ok "G7: rp_cluster_create names a 201 unparseable body distinctly from 'no id' (F1's class)"
  else
    bad "G7: expected rc=1 naming 'unparseable' (got rc=$rc): $out"
  fi
)
(
  export MOCK_CLUSTER_CREATE_STATUS="422"
  out="$(rp_cluster_create "NVIDIA A100 80GB PCIe" 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "refused"; then
    ok "G7: rp_cluster_create refuses a non-201 status by name"
  else
    bad "G7: expected rc=1 'refused' on a non-201 (got rc=$rc): $out"
  fi
)

# ═════════════════════════════════════════════════════════════════════════
# Group 8 — rp_cluster_get: every documented arm actually driven through
# the stub (round-4 audit F2, same class as Group 7).
# ═════════════════════════════════════════════════════════════════════════
(
  echo '{"id":"cl_x","name":"jammi-cluster-ttl8"}' > "$SANDBOX/g8-get-ok.json"
  export MOCK_CLUSTER_GET_STATUS="200"
  export MOCK_CLUSTER_GET_RESPONSE="$SANDBOX/g8-get-ok.json"
  out="$(rp_cluster_get cl_x 2>&1)"; rc=$?
  if [ "$rc" -eq 0 ] && printf '%s' "$out" | grep -q '"id":"cl_x"'; then
    ok "G8: rp_cluster_get prints the body on a 200 with an id"
  else
    bad "G8: expected rc=0 with the body (got rc=$rc): $out"
  fi
)
(
  echo '{"name":"no-id"}' > "$SANDBOX/g8-get-missing.json"
  export MOCK_CLUSTER_GET_STATUS="200"
  export MOCK_CLUSTER_GET_RESPONSE="$SANDBOX/g8-get-missing.json"
  out="$(rp_cluster_get cl_x 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "missing the required 'id' key"; then
    ok "G8: rp_cluster_get names a 200 missing the id key"
  else
    bad "G8: expected rc=1 'missing...id' (got rc=$rc): $out"
  fi
)
(
  echo 'not json' > "$SANDBOX/g8-get-badjson.json"
  export MOCK_CLUSTER_GET_STATUS="200"
  export MOCK_CLUSTER_GET_RESPONSE="$SANDBOX/g8-get-badjson.json"
  out="$(rp_cluster_get cl_x 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "unparseable"; then
    ok "G8: rp_cluster_get names a 200 unparseable body distinctly from 'missing key' (F1's class)"
  else
    bad "G8: expected rc=1 'unparseable' (got rc=$rc): $out"
  fi
)
(
  export MOCK_CLUSTER_GET_STATUS="404"
  out="$(rp_cluster_get cl_missing 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "refused"; then
    ok "G8: rp_cluster_get refuses a 404 by name"
  else
    bad "G8: expected rc=1 'refused' on 404 (got rc=$rc): $out"
  fi
)
(
  export MOCK_CLUSTER_GET_STATUS="500"
  out="$(rp_cluster_get cl_x 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "refused"; then
    ok "G8: rp_cluster_get refuses a 500 by name"
  else
    bad "G8: expected rc=1 'refused' on 500 (got rc=$rc): $out"
  fi
)

# ═════════════════════════════════════════════════════════════════════════
# Group 9 — rp_cluster_sweep's post-delete re-enumeration is three-valued
# (round-4 audit F1): the pre-fix version ran `json.load` with no
# try/except at all, so an unparseable/empty/array second-GET body threw an
# UNCAUGHT Python exception -- Python's own default exit code for an
# uncaught exception is 1, colliding EXACTLY with the "confirmed gone" arm
# and reading a malformed re-enumeration as a clean success (rc=0,
# "terminated N orphaned cluster(s)", a traceback on stderr). Every case
# here must be rc=1, named, and issue NO further DELETE beyond the one
# already made for the genuinely-swept cluster. `MOCK_CLUSTER_LIST_STATUS_2`
# / a malformed `MOCK_CLUSTER_LIST_RESPONSE_2` were never set by any
# pre-existing fixture (round-4 audit's own citation).
# ═════════════════════════════════════════════════════════════════════════
for label_body in "unparseable:not json at all" "empty:" "array:[]"; do
  label="${label_body%%:*}"
  body_content="${label_body#*:}"
(
  printf '%s' "$body_content" > "$SANDBOX/g9-badbody-${label}.json"
  export MOCK_CLUSTER_LIST_RESPONSE="$SANDBOX/g3-list.json"
  export MOCK_CLUSTER_LIST_RESPONSE_2="$SANDBOX/g9-badbody-${label}.json"
  export MOCK_LIST_CALL_COUNTER="$SANDBOX/g9-list-counter-${label}"
  export MOCK_DELETE_CALL_LOG="$SANDBOX/g9-delete-${label}.log"
  rm -f "$MOCK_LIST_CALL_COUNTER" "$MOCK_DELETE_CALL_LOG"; : > "$MOCK_DELETE_CALL_LOG"
  export MOCK_CLUSTER_DELETE_STATUS="204"
  out="$(rp_cluster_sweep 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "could NOT confirm cluster cl-stale is gone"; then
    ok "G9: a ${label} post-delete re-enumeration body is rc=1, named 'could NOT confirm ... is gone' (F1)"
  else
    bad "G9: expected rc=1 naming the unconfirmable delete for a ${label} body (got rc=$rc): $out"
  fi
  n_deletes=0
  [ -f "$MOCK_DELETE_CALL_LOG" ] && n_deletes="$(wc -l < "$MOCK_DELETE_CALL_LOG" | tr -d ' ')"
  if [ "$n_deletes" -eq 1 ]; then
    ok "G9: a ${label} post-delete body issues exactly the ONE delete already made, never a retry"
  else
    bad "G9: expected exactly 1 DELETE call for a ${label} body (got ${n_deletes}): $(cat "$MOCK_DELETE_CALL_LOG")"
  fi
)
done
(
  # A 429 on the SECOND (post-delete) GET specifically -- distinct from the
  # already-covered 401/429/500 on the FIRST (pre-delete) GET (G3 above).
  # The outer "status != 200" check already catches this; this proves it is
  # actually wired to MOCK_CLUSTER_LIST_STATUS_2, never only STATUS_1.
  export MOCK_CLUSTER_LIST_RESPONSE="$SANDBOX/g3-list.json"
  export MOCK_CLUSTER_LIST_STATUS_2="429"
  export MOCK_LIST_CALL_COUNTER="$SANDBOX/g9-list-counter-429"
  export MOCK_DELETE_CALL_LOG="$SANDBOX/g9-delete-429.log"
  rm -f "$MOCK_LIST_CALL_COUNTER" "$MOCK_DELETE_CALL_LOG"; : > "$MOCK_DELETE_CALL_LOG"
  export MOCK_CLUSTER_DELETE_STATUS="204"
  out="$(rp_cluster_sweep 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "could NOT re-enumerate clusters after deleting"; then
    ok "G9: a 429 on the post-delete re-enumeration is rc=1, named 'could NOT re-enumerate ... after deleting'"
  else
    bad "G9: expected rc=1 naming the failed re-enumeration on a 429 (got rc=$rc): $out"
  fi
)

# ═════════════════════════════════════════════════════════════════════════
# Group 10 — rp_terminate: an unparseable podTerminate response body (an
# HTML error page, e.g. a 502 from a misbehaving proxy) is a REFUSED
# terminate, never a silent "swept" (round-4 audit advisory, elevated to a
# binding oracle by P-M1a). The pre-fix version read ANY parse exception as
# `sys.exit(0)` ("no errors" — success).
# ═════════════════════════════════════════════════════════════════════════
(
  cat > "$SANDBOX/g10-account.json" <<JSON
{"data":{"myself":{"pods":[
  {"id":"pod-html","name":"jammi-gpu-ttl8","desiredStatus":"RUNNING","createdAt":"${stale_iso}","runtime":{"uptimeInSeconds":180000}}
]}}}
JSON
  export MOCK_ACCOUNT_RESPONSE="$SANDBOX/g10-account.json"
  export MOCK_TERMINATE_HTML_ID="pod-html"
  export MOCK_TERM_COUNTER="$SANDBOX/g10-term-counter"
  rm -f "$MOCK_TERM_COUNTER"
  out="$(rp_sweep 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ]; then
    ok "G10: rp_sweep returns rc=1 when a podTerminate response is unparseable (never a silent swept)"
  else
    bad "G10: expected rc=1 on an unparseable podTerminate body (got rc=$rc): $out"
  fi
  if printf '%s' "$out" | grep -q "swept pod pod-html"; then
    bad "G10: an unparseable podTerminate body must NEVER read as 'swept pod pod-html' (regression!)"
  else
    ok "G10: pod-html was never reported swept"
  fi
  if printf '%s' "$out" | grep -q "terminate refused:.*pod-html"; then
    ok "G10: the unparseable body is reported as a refused terminate, naming pod-html"
  else
    bad "G10: expected 'terminate refused' naming pod-html: $out"
  fi
)

# ═════════════════════════════════════════════════════════════════════════
# Group 11 — rp_sweep: a pod with no usable createdAt is rc=1, naming the
# pod id — exactly like rp_cluster_sweep's own UNAGEABLE handling (G3
# F3(b) above), round-4 audit P-M1a. Pre-fix, this was a silent `continue`
# that let the sweep finish green (rc=0).
# ═════════════════════════════════════════════════════════════════════════
(
  cat > "$SANDBOX/g11-account.json" <<'JSON'
{"data":{"myself":{"pods":[
  {"id":"pod-unageable","name":"jammi-gpu-ttl8","desiredStatus":"RUNNING","runtime":{"uptimeInSeconds":180000}}
]}}}
JSON
  export MOCK_ACCOUNT_RESPONSE="$SANDBOX/g11-account.json"
  export MOCK_TERM_COUNTER="$SANDBOX/g11-term-counter"
  rm -f "$MOCK_TERM_COUNTER"
  out="$(rp_sweep 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "pod-unageable"; then
    ok "G11: an unageable pod (no usable createdAt) is rc=1, naming the pod id (pod-unageable)"
  else
    bad "G11: expected rc=1 naming pod-unageable (got rc=$rc): $out"
  fi
  if [ -s "$MOCK_TERM_COUNTER" ]; then
    bad "G11: an unageable pod must never be terminated on a guess (regression!)"
  else
    ok "G11: the unageable pod was left alone, not terminated on a guess"
  fi
)

# ═════════════════════════════════════════════════════════════════════════
# Group 12 — round-5 audit F-A: ONE unjudgeable resource must not shield a
# real orphan behind it, and must not abort the run. An unageable pod is
# NAMED (with the by-id remedy) and skipped; the 50h orphan after it IS
# swept THIS run; the sweep still exits 1 so the cron reddens until the
# unageable pod is retired by id. Pre-fix, the loop `return 1`ed on the
# first unageable entry: zero terminates, forever, on every run.
# ═════════════════════════════════════════════════════════════════════════
(
  old="$(iso_from_epoch $(( $(date +%s) - 180000 )))"
  cat > "$SANDBOX/g12-account.json" <<JSON
{"data":{"myself":{"pods":[
  {"id":"pod-unageable","name":"jammi-gpu-ttl8","desiredStatus":"RUNNING","runtime":{"uptimeInSeconds":180000}},
  {"id":"pod-real-orphan","name":"jammi-gpu-ttl8","desiredStatus":"RUNNING","createdAt":"${old}","runtime":{"uptimeInSeconds":180000}}
]}}}
JSON
  export MOCK_ACCOUNT_RESPONSE="$SANDBOX/g12-account.json"
  export MOCK_TERM_COUNTER="$SANDBOX/g12-term-counter"
  rm -f "$MOCK_TERM_COUNTER"; reset_log
  out="$(rp_sweep 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "pod-unageable" && printf '%s' "$out" | grep -q "rp_terminate pod-unageable"; then
    ok "G12: the unageable pod is named with its by-id remedy (rp_terminate <id>) and the sweep exits 1"
  else
    bad "G12: expected rc=1 naming pod-unageable with the rp_terminate remedy (got rc=$rc): $out"
  fi
  if grep -qF "pod-real-orphan" "$CALL_LOG" && [ "$(cat "$MOCK_TERM_COUNTER" 2>/dev/null || echo 0)" -eq 1 ]; then
    ok "G12: the real orphan behind the unageable pod WAS swept this run (exactly one terminate, of pod-real-orphan)"
  else
    bad "G12: the orphan behind an unageable pod must still be swept (terminates=$(cat "$MOCK_TERM_COUNTER" 2>/dev/null || echo 0)): $out"
  fi
  if grep -qF "pod-unageable" "$CALL_LOG"; then
    bad "G12: the unageable pod must never be terminated on a guess (regression!)"
  else
    ok "G12: the unageable pod itself was left alone"
  fi
  # The override cannot help an unageable pod (there is no age to apply it
  # to): same outcome under `rp_sweep 8`, and the remedy stays by-id.
  rm -f "$MOCK_TERM_COUNTER"; reset_log
  out="$(rp_sweep 8 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && ! grep -qF "pod-unageable" "$CALL_LOG" && grep -qF "pod-real-orphan" "$CALL_LOG"; then
    ok "G12: under an override the unageable pod is still named-not-guessed and the orphan is still swept"
  else
    bad "G12: override arm — expected rc=1, orphan swept, unageable untouched (got rc=$rc): $out"
  fi
)
(
  # The cluster arm, same doctrine: an unageable cluster is named with the
  # by-id remedy and the ageable orphan after it is deleted this run.
  old="$(iso_from_epoch $(( $(date +%s) - 180000 )))"
  cat > "$SANDBOX/g12-clusters.json" <<JSON
{"clusters":[
  {"id":"cl-unageable","name":"jammi-cluster-ttl8"},
  {"id":"cl-real-orphan","name":"jammi-cluster-ttl8","createdAt":"${old}"}
]}
JSON
  cat > "$SANDBOX/g12-clusters-after.json" <<'JSON'
{"clusters":[{"id":"cl-unageable","name":"jammi-cluster-ttl8"}]}
JSON
  export MOCK_CLUSTER_LIST_RESPONSE="$SANDBOX/g12-clusters.json"
  export MOCK_CLUSTER_LIST_RESPONSE_2="$SANDBOX/g12-clusters-after.json"
  export MOCK_LIST_CALL_COUNTER="$SANDBOX/g12-list-counter"; rm -f "$MOCK_LIST_CALL_COUNTER"
  export MOCK_DELETE_CALL_LOG="$SANDBOX/g12-deletes"; : > "$MOCK_DELETE_CALL_LOG"
  out="$(rp_cluster_sweep 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "rp_cluster_delete cl-unageable" && grep -q "cl-real-orphan" "$MOCK_DELETE_CALL_LOG" && ! grep -q "cl-unageable" "$MOCK_DELETE_CALL_LOG"; then
    ok "G12: cluster arm — the unageable cluster is named with rp_cluster_delete <id>, the orphan behind it is deleted, rc=1"
  else
    bad "G12: cluster arm — expected rc=1, cl-real-orphan deleted, cl-unageable named-not-deleted (got rc=$rc): $out / deletes: $(cat "$MOCK_DELETE_CALL_LOG")"
  fi
)

# ═════════════════════════════════════════════════════════════════════════
# Group 13 — round-5 audit F-B: a prefixed cluster (or pod) whose name has
# NO parseable -ttl<H> is the same "cannot judge" state as no createdAt.
# Pre-fix it was printed as an `unparseable-deadline` row and DELETED at
# any age (executed by the audit: a 60-second-old jammi-cluster-experiment,
# gone with a green summary). Now: named, never deleted, rc=1.
# ═════════════════════════════════════════════════════════════════════════
(
  young="$(iso_from_epoch $(( $(date +%s) - 60 )))"
  cat > "$SANDBOX/g13-clusters.json" <<JSON
{"clusters":[{"id":"cl-manual","name":"jammi-cluster-experiment","createdAt":"${young}"}]}
JSON
  export MOCK_CLUSTER_LIST_RESPONSE="$SANDBOX/g13-clusters.json"
  export MOCK_DELETE_CALL_LOG="$SANDBOX/g13-deletes"; : > "$MOCK_DELETE_CALL_LOG"
  out="$(rp_cluster_sweep 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "cl-manual" && printf '%s' "$out" | grep -q "no parseable -ttl" && ! grep -q "cl-manual" "$MOCK_DELETE_CALL_LOG"; then
    ok "G13: a prefixed cluster with no parseable -ttl<H> is named (by-id remedy) and NOT deleted; rc=1"
  else
    bad "G13: expected rc=1, named, zero deletes for cl-manual (got rc=$rc): $out / deletes: $(cat "$MOCK_DELETE_CALL_LOG")"
  fi
)
(
  young="$(iso_from_epoch $(( $(date +%s) - 60 )))"
  cat > "$SANDBOX/g13-account.json" <<JSON
{"data":{"myself":{"pods":[{"id":"pod-manual","name":"jammi-gpu-experiment","desiredStatus":"RUNNING","createdAt":"${young}","runtime":{"uptimeInSeconds":60}}]}}}
JSON
  export MOCK_ACCOUNT_RESPONSE="$SANDBOX/g13-account.json"
  export MOCK_TERM_COUNTER="$SANDBOX/g13-term-counter"; rm -f "$MOCK_TERM_COUNTER"; reset_log
  out="$(rp_sweep 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "pod-manual" && ! log_has "podTerminate"; then
    ok "G13: pod mirror — a prefixed pod with no parseable -ttl<H> is named and NOT terminated; rc=1"
  else
    bad "G13: pod mirror — expected rc=1, named, zero terminates (got rc=$rc): $out"
  fi
)

# ═════════════════════════════════════════════════════════════════════════
# Group 14 — round-5 audit F-C: every parse on the reap path is TOTAL. A
# row of the wrong shape is "could not be read" / "could NOT enumerate",
# with the reason — never an uncaught-exception exit 1 that the caller
# reports as "missing the required key" (a confidently wrong diagnosis).
# ═════════════════════════════════════════════════════════════════════════
(
  echo '{"clusters":["oops"]}' > "$SANDBOX/g14-list.json"
  export MOCK_CLUSTER_LIST_RESPONSE="$SANDBOX/g14-list.json"
  out="$(rp_cluster_list 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "could not be read" && ! printf '%s' "$out" | grep -q "missing the required"; then
    ok "G14: rp_cluster_list — a non-object row is 'could not be read', never 'missing the required key'"
  else
    bad "G14: rp_cluster_list on a non-object row (got rc=$rc): $out"
  fi
)
(
  echo '{"pods":[{"id":"p1","ssh":{"direct":{"port":22}}}]}' > "$SANDBOX/g14-pods.json"
  export MOCK_CLUSTER_PODS_RESPONSE="$SANDBOX/g14-pods.json"
  out="$(rp_cluster_pods cl-x 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "could not be read" && ! printf '%s' "$out" | grep -q "missing the required"; then
    ok "G14: rp_cluster_pods — an ssh.direct block without host is 'could not be read', never 'missing the required pods key'"
  else
    bad "G14: rp_cluster_pods on a malformed member row (got rc=$rc): $out"
  fi
)
(
  export MOCK_CLUSTER_PODS_STATUS="503"
  out="$(rp_cluster_pods cl-x 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "refused (status 503)"; then
    ok "G14: rp_cluster_pods — a non-200 is a named refusal (MOCK_CLUSTER_PODS_STATUS drives the arm)"
  else
    bad "G14: rp_cluster_pods non-200 arm (got rc=$rc): $out"
  fi
)
(
  for body in '[{"id":"x"}]' 'null' '42' '"s"'; do
    printf '%s' "$body" > "$SANDBOX/g14-sweep.json"
    export MOCK_CLUSTER_LIST_RESPONSE="$SANDBOX/g14-sweep.json"
    export MOCK_DELETE_CALL_LOG="$SANDBOX/g14-deletes"; : > "$MOCK_DELETE_CALL_LOG"
    out="$(rp_cluster_sweep 2>&1)"; rc=$?
    if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "could NOT enumerate clusters" && printf '%s' "$out" | grep -q "response contained no cluster list\|could not read" && ! [ -s "$MOCK_DELETE_CALL_LOG" ]; then
      ok "G14: rp_cluster_sweep on body ${body} — named 'could NOT enumerate' with a reason, zero deletes"
    else
      bad "G14: rp_cluster_sweep on body ${body} (got rc=$rc): $out"
    fi
  done
)
(
  printf '%s' '[1,2,3]' > "$SANDBOX/g14-account.json"
  export MOCK_ACCOUNT_RESPONSE="$SANDBOX/g14-account.json"
  export MOCK_TERM_COUNTER="$SANDBOX/g14-term-counter"; rm -f "$MOCK_TERM_COUNTER"; reset_log
  out="$(rp_sweep 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "could NOT enumerate pods" && ! log_has "podTerminate"; then
    ok "G14: rp_sweep on an array body — named 'could NOT enumerate pods', zero terminates"
  else
    bad "G14: rp_sweep on an array body (got rc=$rc): $out"
  fi
)
(
  # A pod row of the wrong shape INSIDE an otherwise valid list.
  printf '%s' '{"data":{"myself":{"pods":[null]}}}' > "$SANDBOX/g14-account2.json"
  export MOCK_ACCOUNT_RESPONSE="$SANDBOX/g14-account2.json"
  export MOCK_TERM_COUNTER="$SANDBOX/g14-term-counter2"; rm -f "$MOCK_TERM_COUNTER"; reset_log
  out="$(rp_sweep 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "could not read the pod list" && ! log_has "podTerminate"; then
    ok "G14: rp_sweep on a null pod row — 'could not read the pod list: ...', zero terminates"
  else
    bad "G14: rp_sweep on a null pod row (got rc=$rc): $out"
  fi
)

# ═════════════════════════════════════════════════════════════════════════
# Group 15 — round-5 audit F-D: a cluster member row with NO readable id is
# refused, so the exclusion set is COMPLETE or ABSENT, never short — the
# pod sweep is suspended (zero terminates), exactly like a failed member
# enumeration (G6). Pre-fix the idless member was silently dropped from
# the set and its own pod then terminated as an orphan.
# ═════════════════════════════════════════════════════════════════════════
(
  old="$(iso_from_epoch $(( $(date +%s) - 180001 )))"
  cat > "$SANDBOX/g15-clusters.json" <<'JSON'
{"clusters":[{"id":"cl-1","name":"jammi-cluster-ttl8","createdAt":"2030-01-01T00:00:00Z"}]}
JSON
  cat > "$SANDBOX/g15-pods.json" <<'JSON'
{"pods":[{"cluster":{"rank":0,"ip":"10.0.0.1"}},{"id":"pod-member2","cluster":{"rank":1,"ip":"10.0.0.2"}}]}
JSON
  cat > "$SANDBOX/g15-account.json" <<JSON
{"data":{"myself":{"pods":[
  {"id":"pod-idless-member","name":"jammi-gpu-ttl8","desiredStatus":"RUNNING","createdAt":"${old}","runtime":{"uptimeInSeconds":180001}},
  {"id":"pod-member2","name":"jammi-gpu-ttl8","desiredStatus":"RUNNING","createdAt":"${old}","runtime":{"uptimeInSeconds":180001}}
]}}}
JSON
  export MOCK_CLUSTER_LIST_RESPONSE="$SANDBOX/g15-clusters.json"
  export MOCK_CLUSTER_PODS_RESPONSE="$SANDBOX/g15-pods.json"
  export MOCK_ACCOUNT_RESPONSE="$SANDBOX/g15-account.json"
  export MOCK_TERM_COUNTER="$SANDBOX/g15-term-counter"; rm -f "$MOCK_TERM_COUNTER"; reset_log
  out="$(rp_sweep 2>&1)"; rc=$?
  if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "sweep could NOT enumerate cluster members; pod sweep skipped" && ! log_has "podTerminate"; then
    ok "G15: a member row with no readable id suspends the pod sweep (exclusion set incomplete → nothing terminated)"
  else
    bad "G15: expected rc=1, pod sweep skipped, zero terminates (got rc=$rc): $out"
  fi
)

# ═════════════════════════════════════════════════════════════════════════
# Group 16 — round-5 audit F-E: the request body rp_cluster_create SENDS
# is asserted against the reviewed key-set fixture (MOCK_REQUEST_BODY_LOG,
# previously never exported by any test).
# ═════════════════════════════════════════════════════════════════════════
(
  echo '{"id":"cl_new"}' > "$SANDBOX/g16-create-ok.json"
  export MOCK_CLUSTER_CREATE_STATUS="201"
  export MOCK_CLUSTER_CREATE_RESPONSE="$SANDBOX/g16-create-ok.json"
  export MOCK_REQUEST_BODY_LOG="$SANDBOX/g16-request-body"; : > "$MOCK_REQUEST_BODY_LOG"
  out="$(rp_cluster_create "NVIDIA A100 80GB PCIe" 2>&1)"; rc=$?
  sent_keys="$(python3 -c '
import json, sys
body = open(sys.argv[1]).read().strip().splitlines()[-1]
d = json.loads(body)
print(" ".join(sorted(d.keys())))
print(" ".join(sorted(d["compute"].keys())))
print(d["compute"]["podCount"], d["compute"]["gpuCountPerPod"])
' "$MOCK_REQUEST_BODY_LOG" 2>&1)"
  fixture_keys="$(python3 -c '
import json, sys
f = json.load(open(sys.argv[1]))
print(" ".join(sorted(k for k in f["top_level_keys"])))
print(" ".join(sorted(f["compute_keys"])))
' "$REPO_ROOT/ci/scripts/fixtures/runpod_cluster_create_request_keys.json" 2>/dev/null || echo "fixture-unreadable")"
  if [ "$rc" -eq 0 ] && [ -s "$MOCK_REQUEST_BODY_LOG" ] && printf '%s' "$sent_keys" | tail -n1 | grep -qx "2 1"; then
    ok "G16: the SENT create body was captured and carries the fixed 2×1 shape (podCount 2, gpuCountPerPod 1)"
  else
    bad "G16: expected a captured request body with podCount 2 / gpuCountPerPod 1 (got rc=$rc): sent=[$sent_keys] out=$out"
  fi
  sent_top="$(printf '%s' "$sent_keys" | sed -n 1p)"; sent_compute="$(printf '%s' "$sent_keys" | sed -n 2p)"
  fix_top="$(printf '%s' "$fixture_keys" | sed -n 1p)"; fix_compute="$(printf '%s' "$fixture_keys" | sed -n 2p)"
  if [ "$fixture_keys" != "fixture-unreadable" ] && [ -n "$sent_top" ]; then
    python3 - "$sent_top" "$fix_top" "$sent_compute" "$fix_compute" <<'PY' && ok "G16: every key the SENT body carries is in the reviewed CreateClusterRequest key-set fixture (top level and compute)" || bad "G16: the SENT body carries a key outside the reviewed fixture: sent_top=[$sent_top] fixture_top=[$fix_top] sent_compute=[$sent_compute] fixture_compute=[$fix_compute]"
import sys
sent_top, fix_top, sent_c, fix_c = (set(a.split()) for a in sys.argv[1:5])
sys.exit(0 if sent_top <= fix_top and sent_c <= fix_c else 1)
PY
  else
    bad "G16: could not compare the sent body against the fixture: sent=[$sent_keys] fixture=[$fixture_keys]"
  fi
)

echo
TOTAL_PASS="$(grep -c '^PASS:' "$RESULTS" || true)"
TOTAL_FAIL="$(grep -c '^FAIL:' "$RESULTS" || true)"
echo "runpod-cluster-lib: ${TOTAL_PASS} passed, ${TOTAL_FAIL} failed"
[ "$TOTAL_FAIL" -eq 0 ]
