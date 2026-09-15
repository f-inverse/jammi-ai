#!/usr/bin/env bash
# The merge path, locally: every check a pull request must pass, run in the
# order that keeps the last gate fresh, from the workflow files themselves.
#
#   bash ci/scripts/merge_path.sh [--only STAGE[,STAGE]] [--skip-pg] [--skip-tests]
#
# Stages, in order:
#   static   fmt, the four clippy surfaces, rustdoc -D warnings, the guide build
#            (ci.yml `check`, docs.yml `build`)
#   guards   every `guard` matrix command in ci.yml (read from the file at run
#            time, never a copied list — the list that drifted cost a CI round)
#   swarm    the Swarm-gates workflow's steps EXCEPT the two record checks
#   tests    the hermetic lane (workspace, test-hooks, golden-parity) and the
#            Postgres lane (ci.yml `test`, `test-pg`)
#   records  check_rigor_record.py and check_oracle_gate.py — LAST, because the
#            oracle gate stales on any change outside docs/rigor/**; run the
#            oracle after everything else is green and commit only its record
#
# The Postgres lane needs a live database in JAMMI_TEST_PG_URL (CI's shape:
# user jammi, db jammi_test). Without one the stage FAILS, naming the fix,
# unless --skip-pg is given: a silently skipped lane is how the shared-database
# leak of PR #579 reached CI.
#
# Every command runs with stdin from /dev/null (a stdin-reading guard once
# swallowed the rest of the list) and its own log under $MERGE_PATH_LOG_DIR
# (default: $CARGO_TARGET_DIR/merge-path, never inside the tree — the stamp
# guard walks every in-tree directory but `target`). The exit status is the
# number of failed commands; the summary names each one and its log.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 2

ONLY=""
SKIP_PG=0
SKIP_TESTS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --only=*) ONLY="${1#--only=}"; shift ;;
    --skip-pg) SKIP_PG=1; shift ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "merge_path: unknown argument $1" >&2; exit 2 ;;
  esac
done

if [ -z "${CARGO_TARGET_DIR:-}" ]; then
  echo "merge_path: CARGO_TARGET_DIR is unset — builds land in ./target, which is fine for a" >&2
  echo "merge_path: single checkout; for a worktree set it OUTSIDE the tree (shared targets go stale)." >&2
fi
LOG_DIR="${MERGE_PATH_LOG_DIR:-${CARGO_TARGET_DIR:-$ROOT/target}/merge-path}"
mkdir -p "$LOG_DIR"

HEAD_SHA="$(git rev-parse HEAD)"
BASE_SHA="$(git rev-parse origin/main 2>/dev/null || git rev-parse main)"
HEAD_REF="$(git rev-parse --abbrev-ref HEAD)"

ran=0
failed=0
FAILED_LIST=()

stage_wanted() {
  [ -z "$ONLY" ] && return 0
  case ",$ONLY," in *",$1,"*) return 0 ;; esac
  return 1
}

# run STAGE LABEL CMD... — runs CMD with stdin closed, records rc and log.
run() {
  local stage="$1" label="$2"; shift 2
  ran=$((ran + 1))
  local log
  log="$LOG_DIR/$(printf '%03d' "$ran").log"
  printf '%s\n' "$*" > "$log"
  if "$@" </dev/null >>"$log" 2>&1; then
    printf 'ok    [%s] %s\n' "$stage" "$label"
  else
    local rc=$?
    failed=$((failed + 1))
    FAILED_LIST+=("[$stage] $label  (rc=$rc) -> $log")
    printf 'FAIL  [%s] %s  (rc=%s) -> %s\n' "$stage" "$label" "$rc" "$log"
  fi
}

# run_sh STAGE LABEL 'shell string' — as run, through bash -c (matrix cmds
# carry env prefixes and pipes).
run_sh() {
  local stage="$1" label="$2" cmd="$3"
  cmd="${cmd//\$\{\{ github.event.pull_request.head.sha || \'push\' \}\}/$HEAD_SHA}"
  cmd="${cmd//\$\{\{ github.event.pull_request.base.sha || \'\' \}\}/$BASE_SHA}"
  run "$stage" "$label" bash -c "$cmd"
}

export GITHUB_EVENT_NAME=pull_request GITHUB_BASE_REF=main GITHUB_HEAD_REF="$HEAD_REF" \
  GITHUB_ACTOR="${GITHUB_ACTOR:-$(git config user.name || echo local)}"

# ---------------------------------------------------------------- static
if stage_wanted static; then
  run static "cargo fmt --check" cargo fmt --all -- --check
  run static "clippy workspace" cargo clippy --workspace --all-targets -- -D warnings
  run static "clippy gated test surfaces (jammi-ai)" \
    cargo clippy -p jammi-ai --tests --features live-gpu-tests,live-distributed-tests -- -D warnings
  run static "clippy gated test surfaces (jammi-server)" \
    cargo clippy -p jammi-server --tests --features test-hooks -- -D warnings
  run static "clippy jammi-db postgres,mysql" \
    cargo clippy -p jammi-db --features postgres,mysql --all-targets -- -D warnings
  run static "rustdoc -D warnings" \
    env RUSTDOCFLAGS="-D warnings" cargo doc --workspace --exclude jammi-python --no-deps
  if command -v mdbook >/dev/null 2>&1; then
    run static "mdbook build docs/guide" mdbook build docs/guide
  else
    printf 'skip  [static] mdbook build docs/guide (mdbook not installed; docs.yml runs it)\n'
  fi
fi

# ---------------------------------------------------------------- guards
if stage_wanted guards; then
  GUARD_LIST="$LOG_DIR/guard-matrix.txt"
  python3 - "$GUARD_LIST" <<'PY'
import sys, yaml
ci = yaml.safe_load(open('.github/workflows/ci.yml'))
entries = ci['jobs']['guard']['strategy']['matrix']['include']
with open(sys.argv[1], 'w') as out:
    for e in entries:
        out.write(e['name'] + '\t' + ' '.join(e['cmd'].split()) + '\n')
print(f"merge_path: {len(entries)} guard-matrix commands read from ci.yml")
PY
  while IFS=$'\t' read -r name cmd; do
    run_sh guards "$name" "$cmd"
  done < "$GUARD_LIST"
fi

# ---------------------------------------------------------------- swarm
if stage_wanted swarm; then
  SWARM_LIST="$LOG_DIR/swarm-steps.txt"
  python3 - "$SWARM_LIST" <<'PY'
import sys, yaml
wf = yaml.safe_load(open('.github/workflows/swarm.yml'))
n = 0
with open(sys.argv[1], 'w') as out:
    for job in wf['jobs'].values():
        for st in job.get('steps', []):
            run = st.get('run')
            if not run:
                continue
            for line in run.strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # the two record checks run in the `records` stage, last
                if line in ('python3 ci/scripts/check_rigor_record.py',
                            'python3 ci/scripts/check_oracle_gate.py'):
                    continue
                out.write((st.get('name') or line) + '\t' + line + '\n'); n += 1
print(f"merge_path: {n} swarm-gate steps read from swarm.yml")
PY
  while IFS=$'\t' read -r name cmd; do
    run_sh swarm "$name" "$cmd"
  done < "$SWARM_LIST"
fi

# ---------------------------------------------------------------- tests
if stage_wanted tests && [ "$SKIP_TESTS" = 0 ]; then
  run tests "cargo test --workspace (hermetic lane)" \
    env JAMMI_REQUIRE_MEDIA_SMOKE=1 cargo test --workspace --exclude jammi-python
  run tests "jammi-db test-hooks lane" \
    cargo test -p jammi-db --features test-hooks --test it -- --test-threads=1
  run tests "jammi-encoders golden-parity" \
    cargo test -p jammi-encoders --features golden-parity --test golden_parity
  if [ -n "${JAMMI_TEST_PG_URL:-}" ]; then
    export JAMMI_REQUIRE_PG=1
    run tests "Postgres lane: jammi-db it" \
      cargo test -p jammi-db --features live-postgres-tests --test it -- --test-threads=1
    run tests "Postgres lane: jammi-db it + test-hooks" \
      cargo test -p jammi-db --features live-postgres-tests,test-hooks --test it -- --test-threads=1
    run tests "Postgres lane: jammi-server introspection" \
      cargo test -p jammi-server --test it -- introspection --test-threads=1
  elif [ "$SKIP_PG" = 1 ]; then
    printf 'skip  [tests] Postgres lane (--skip-pg; CI still runs it)\n'
  else
    ran=$((ran + 1)); failed=$((failed + 1))
    FAILED_LIST+=("[tests] Postgres lane: JAMMI_TEST_PG_URL is unset — start a Postgres 16 (user jammi, db jammi_test), export JAMMI_TEST_PG_URL=postgres://jammi@127.0.0.1:PORT/jammi_test, or pass --skip-pg explicitly")
    printf 'FAIL  [tests] Postgres lane: JAMMI_TEST_PG_URL is unset (pass --skip-pg to skip explicitly)\n'
  fi
fi

# ---------------------------------------------------------------- records
if stage_wanted records; then
  run records "check_rigor_record.py" python3 ci/scripts/check_rigor_record.py
  run records "check_oracle_gate.py (must be the last thing that changes)" python3 ci/scripts/check_oracle_gate.py
fi

# ---------------------------------------------------------------- summary
printf '\nmerge_path: RAN %d  FAILED %d  (logs: %s)\n' "$ran" "$failed" "$LOG_DIR"
for f in "${FAILED_LIST[@]:-}"; do [ -n "$f" ] && printf '  %s\n' "$f"; done
exit "$failed"
