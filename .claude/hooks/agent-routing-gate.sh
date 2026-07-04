#!/bin/sh
# agent-routing-gate.sh — PreToolUse(Task) advisory routing nudge.
#
# Family O/P (LESSONS.md): the rigor chain's phases each clear a NAMED gate agent.
# When a dispatched Task reads as a phase step (audit, pressure-test, fix-verify,
# cookbook re-emit, discipline/boundary check) but is NOT routed to the matching
# gate agent, this nudges toward the right one. It WARNS only — always exit 0. The
# real capability boundary is each agent's native `tools:`, not this hook.
#
# POSIX sh. Fail-open: any parse/tooling error still exits 0. Generic keyword
# heuristic, no incident specifics.

set -u

PAYLOAD="$(cat 2>/dev/null || true)"

json_field() {
  if command -v jq >/dev/null 2>&1; then
    printf '%s' "$PAYLOAD" | jq -r ".${1} // empty" 2>/dev/null && return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    printf '%s' "$PAYLOAD" | python3 -c '
import sys, json
path = sys.argv[1].split(".")
try:
    obj = json.load(sys.stdin)
    for k in path:
        obj = obj.get(k) if isinstance(obj, dict) else None
    if isinstance(obj, str):
        sys.stdout.write(obj)
except Exception:
    pass
' "$1" 2>/dev/null && return 0
  fi
  return 0
}

TOOL="$(json_field tool_name)"
case "$TOOL" in
  Task|"") : ;;
  *) exit 0 ;;
esac

SUB="$(json_field tool_input.subagent_type)"
DESC="$(json_field tool_input.description)"
PROMPT="$(json_field tool_input.prompt)"
# lowercase haystack of intent text for keyword matching
HAY="$(printf '%s\n%s' "$DESC" "$PROMPT" | tr '[:upper:]' '[:lower:]')"

warn() { printf '%s\n' "agent-routing-gate [advisory]: $1" >&2; }

# Nudge only when the intent looks phase-shaped AND the routed agent does not
# already match the expected gate agent (substring match on subagent_type).
nudge_if() {
  # $1 = space-separated intent keywords ; $2 = expected agent ; $3 = message
  kws="$1"; want="$2"; msg="$3"
  case "$SUB" in *"$want"*) return 0 ;; esac   # already routed correctly
  for kw in $kws; do
    case "$HAY" in *"$kw"*) warn "$msg (routed to '${SUB:-unset}')"; return 0 ;; esac
  done
}

nudge_if "adversarial refute the diff audit-the-diff state-collapse band-aid" \
  "adversarial-audit" \
  "this reads like a diff refutation — consider routing to the 'adversarial-audit' verifier (default-BLOCK, principle-level rubric)"

nudge_if "pressure-test attack-the-plan wrong-design before-code feasibility" \
  "pressure-tester" \
  "this reads like a design attack before code — consider routing to the 'pressure-tester' verifier"

nudge_if "red-green tautological non-finite exit-gate closes_escape verify-the-fix" \
  "fix-verifier" \
  "this reads like a fix verification — consider routing to the 'fix-verifier' exit gate (red-green + non-finite control)"

nudge_if "discipline-test names-a-consumer governance-verb boundary open-core-vs-platform" \
  "discipline-test-auditor" \
  "this reads like an engine-not-platform boundary check — consider routing to the 'discipline-test-auditor'"

nudge_if "re-emit cookbook chapter goldens divergence measured-not-asserted" \
  "cookbook" \
  "this reads like a cookbook re-emit — consider routing to the 'cookbook' agent (phase 6.5, block Ship on golden divergence)"

nudge_if "cited path:line re-read-the-citation stale-citation fabricated-citation" \
  "citation-checker" \
  "this reads like a citation re-verification — consider routing to the 'citation-checker'"

nudge_if "frozen-seam lockstep-version migration-monotonic tenant-iso hard-block oracle" \
  "oracle" \
  "this reads like a frozen-seam / lockstep / tenant-iso check — consider routing to the 'oracle' (hard-block, not overridable)"

exit 0
