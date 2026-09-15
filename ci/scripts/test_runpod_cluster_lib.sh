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

echo
TOTAL_PASS="$(grep -c '^PASS:' "$RESULTS" || true)"
TOTAL_FAIL="$(grep -c '^FAIL:' "$RESULTS" || true)"
echo "runpod-cluster-lib: ${TOTAL_PASS} passed, ${TOTAL_FAIL} failed"
[ "$TOTAL_FAIL" -eq 0 ]
