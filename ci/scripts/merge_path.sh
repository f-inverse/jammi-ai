#!/usr/bin/env bash
# The merge path, locally: the checks a pull request must pass, read from the
# workflow files themselves and run in one process, with what it does NOT run
# printed by name.
#
#   bash ci/scripts/merge_path.sh [--only STAGE[,STAGE]] [--skip-pg] [--skip-tests] [--skip-mdbook]
#
# Stages, in order:
#   static   fmt, the clippy surfaces the `check` job lints, rustdoc -D warnings, the guide build
#            (ci.yml `check`, docs.yml `build`)
#   guards   the guards this change can affect (`ci/guards.toml`), through the
#            same runner ci.yml's `guard` job calls
#   index    ci.yml's `symbol-index-gates` steps (the guards that build a
#            workspace crate), each step's `run:` block executed WHOLE (a
#            multi-line guard split into lines is never evaluated)
#   tests    the hermetic lane (workspace, test-hooks, golden-parity) and the
#            Postgres lane (ci.yml `test`, `test-pg`)
#
# Refuses to run when HEAD is the base (nothing to check — every diff-scoped
# gate would be vacuously green) or the tree is dirty (the gates read
# committed state; an uncommitted change is invisible to them).
#
# The Postgres lane needs a live database in JAMMI_TEST_PG_URL (CI's shape:
# user jammi, db jammi_test). Without one the stage FAILS, naming the fix,
# unless --skip-pg is given; likewise `mdbook build` FAILS when mdbook is
# absent unless --skip-mdbook is given. A silently skipped lane is how the
# shared-database leak of PR #579 reached CI.
#
# Every command runs with stdin from /dev/null (a stdin-reading guard once
# swallowed the rest of the list) and its own log under $MERGE_PATH_LOG_DIR
# (default: $CARGO_TARGET_DIR/merge-path, never inside the tree — the stamp
# guard walks every in-tree directory but `target`). Every `${{ ... }}`
# workflow expression a command carries is expanded from the local checkout
# or the run fails naming it. The exit status is the number of failed
# commands; the summary names each one and its log, and lists every ci.yml
# job this runner does not cover.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 2

ONLY=""
SKIP_PG=0
SKIP_TESTS=0
SKIP_MDBOOK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --only=*) ONLY="${1#--only=}"; shift ;;
    --skip-pg) SKIP_PG=1; shift ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --skip-mdbook) SKIP_MDBOOK=1; shift ;;
    -h|--help) sed -n '2,42p' "$0"; exit 0 ;;
    *) echo "merge_path: unknown argument $1" >&2; exit 2 ;;
  esac
done

HEAD_SHA="$(git rev-parse HEAD)"
HEAD_REF="$(git rev-parse --abbrev-ref HEAD)"
BASE_REF="${MERGE_PATH_BASE:-main}"
BASE_SHA="$(git rev-parse "origin/$BASE_REF" 2>/dev/null || git rev-parse "$BASE_REF")"
if [ "$HEAD_SHA" = "$BASE_SHA" ]; then
  echo "merge_path: HEAD ($HEAD_SHA) IS the base ($BASE_REF): nothing to check — every diff-scoped gate would be vacuously green. Commit the unit first." >&2
  exit 2
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "merge_path: the tree has uncommitted changes — the gates read committed state, so an uncommitted change is invisible to them. Commit first:" >&2
  git status --short --untracked-files=no >&2
  exit 2
fi

if [ -z "${CARGO_TARGET_DIR:-}" ]; then
  echo "merge_path: CARGO_TARGET_DIR is unset — builds land in ./target, which is fine for a" >&2
  echo "merge_path: single checkout; for a worktree set it OUTSIDE the tree (shared targets go stale)." >&2
fi
LOG_DIR="${MERGE_PATH_LOG_DIR:-${CARGO_TARGET_DIR:-$ROOT/target}/merge-path}"
mkdir -p "$LOG_DIR"

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

# expand every `${{ ... }}` expression a workflow command carries, from the
# local checkout; an expression this runner does not know is a hard stop.
expand() {
  local cmd="$1"
  cmd="${cmd//\$\{\{ github.event.pull_request.head.sha || \'push\' \}\}/$HEAD_SHA}"
  cmd="${cmd//\$\{\{ github.event.pull_request.base.sha || \'\' \}\}/$BASE_SHA}"
  cmd="${cmd//\$\{\{ github.base_ref \}\}/$BASE_REF}"
  cmd="${cmd//\$\{\{ github.head_ref \}\}/$HEAD_REF}"
  cmd="${cmd//\$\{\{ github.event_name \}\}/pull_request}"
  cmd="${cmd//\$\{\{ github.actor \}\}/${GITHUB_ACTOR:-local}}"
  cmd="${cmd//\$\{\{ github.sha \}\}/$HEAD_SHA}"
  if [[ "$cmd" == *'${{'* ]]; then
    echo "merge_path: an unexpanded workflow expression in: $cmd" >&2
    echo "merge_path: teach expand() the expression or the command cannot run locally" >&2
    exit 2
  fi
  printf '%s' "$cmd"
}

# run_sh STAGE LABEL 'shell block' — as run, through `bash -e -c` on the
# WHOLE block (never line by line), with the workflow expressions expanded.
run_sh() {
  local stage="$1" label="$2" cmd
  cmd="$(expand "$3")" || exit 2
  run "$stage" "$label" bash -e -c "$cmd"
}

export GITHUB_EVENT_NAME=pull_request GITHUB_BASE_REF="$BASE_REF" GITHUB_HEAD_REF="$HEAD_REF" \
  GITHUB_ACTOR="${GITHUB_ACTOR:-local}" GITHUB_WORKSPACE="$ROOT"

# provide_in_image STAGE PACKAGE PROBE... — inside the CI image (root, yum),
# install PACKAGE when PROBE fails. The hosted runners a ci.yml job uses carry
# tools the image deliberately leaves out, and some jobs install one for a
# single step; this supplies the same thing to a run inside the image. On a
# developer host it does nothing: PROBE passes, or there is no yum to call.
provide_in_image() {
  local stage="$1" package="$2"; shift 2
  if "$@" >/dev/null 2>&1; then return 0; fi
  if command -v yum >/dev/null 2>&1 && [ "$(id -u)" = 0 ]; then
    run "$stage" "provide $package (absent from the CI image)" yum install -y -q "$package"
  fi
}

# ci_step JOB 'STEP NAME' — the `run:` block of that ci.yml step, so a lane
# this runner shares with CI is written once, in the workflow.
ci_step() {
  python3 - "$1" "$2" <<'PY'
import sys, yaml
job, name = sys.argv[1], sys.argv[2]
steps = yaml.safe_load(open('.github/workflows/ci.yml'))['jobs'][job]['steps']
runs = [st['run'] for st in steps if st.get('name') == name]
if len(runs) != 1:
    sys.exit(f"merge_path: ci.yml job {job!r} has {len(runs)} steps named {name!r}")
print(runs[0])
PY
}

# ---------------------------------------------------------------- coverage
# Which ci.yml jobs this runner covers, printed up front so "green" is never
# read as "every job".
COVERED_JOBS="check test test-pg guard symbol-index-gates"
python3 - "$COVERED_JOBS" <<'PY'
import sys, yaml
ci = yaml.safe_load(open('.github/workflows/ci.yml'))
covered = set(sys.argv[1].split())
jobs = [j for j in ci['jobs'] if j != 'ci-summary']
missing = [j for j in jobs if j not in covered]
print(f"merge_path: covers {len(jobs) - len(missing)} of {len(jobs)} ci.yml jobs (plus docs.yml's build)")
print("merge_path: NOT run here (CI runs them): " + ", ".join(missing))
PY

# ---------------------------------------------------------------- static
if stage_wanted static; then
  run static "cargo fmt --check" cargo fmt --all -- --check
  run static "clippy workspace" cargo clippy --workspace --all-targets -- -D warnings
  run_sh static "clippy feature-gated test surfaces" \
    "$(ci_step check 'Clippy (feature-gated test surfaces)')"
  # The `postgres`/`mysql` source providers pull `openssl-sys`, whose build
  # script needs OpenSSL headers the CI image deliberately omits; ci.yml
  # installs them for this one lint (its "OpenSSL headers" step says why).
  provide_in_image static openssl-devel pkg-config --exists openssl
  run static "clippy jammi-db postgres,mysql" \
    cargo clippy -p jammi-db --features postgres,mysql --all-targets -- -D warnings
  run static "rustdoc -D warnings" \
    env RUSTDOCFLAGS="-D warnings" cargo doc --workspace --exclude jammi-python --no-deps
  if command -v mdbook >/dev/null 2>&1; then
    run static "mdbook build docs/guide" mdbook build docs/guide
  elif [ "$SKIP_MDBOOK" = 1 ]; then
    printf 'skip  [static] mdbook build docs/guide (--skip-mdbook; docs.yml runs it)\n'
  else
    ran=$((ran + 1)); failed=$((failed + 1))
    FAILED_LIST+=("[static] mdbook build docs/guide: mdbook is not installed — install it (cargo install mdbook) or pass --skip-mdbook explicitly")
    printf 'FAIL  [static] mdbook build docs/guide: mdbook not installed (pass --skip-mdbook to skip explicitly)\n'
  fi
fi

# ---------------------------------------------------------------- guards
if stage_wanted guards; then
  run guards "run_guards.py --base $BASE_REF" python3 ci/scripts/run_guards.py --base "$BASE_SHA"
fi

# ---------------------------------------------------------------- index
if stage_wanted index; then
  INDEX_DIR="$LOG_DIR/index-steps"
  rm -rf "$INDEX_DIR"; mkdir -p "$INDEX_DIR"
  python3 - "$INDEX_DIR" <<'PY'
import os, sys, yaml
ci = yaml.safe_load(open('.github/workflows/ci.yml'))
n = 0
for st in ci['jobs']['symbol-index-gates'].get('steps', []):
    run = st.get('run')
    if not run:
        continue
    n += 1
    with open(os.path.join(sys.argv[1], f"{n:02d}.step"), 'w') as out:
        out.write((st.get('name') or run.strip().split('\n')[0]) + '\n')
        out.write(run)
print(f"merge_path: {n} steps read from ci.yml symbol-index-gates (each run as one block)")
PY
  for step in "$INDEX_DIR"/*.step; do
    name="$(head -n1 "$step")"
    body="$(tail -n +2 "$step")"
    run_sh index "$name" "$body"
  done
fi

# ---------------------------------------------------------------- tests
if stage_wanted tests && [ "$SKIP_TESTS" = 0 ]; then
  run tests "cargo test --workspace (hermetic lane)" \
    cargo test --workspace --exclude jammi-python
  run tests "jammi-db test-hooks lane" \
    cargo test -p jammi-db --features test-hooks --test it -- --test-threads=1
  run tests "jammi-encoders golden-parity" \
    cargo test -p jammi-encoders --features golden-parity --test golden_parity
  run_sh tests "permission-fault tests (unprivileged account)" \
    "$(ci_step test 'Permission-fault tests (unprivileged account)')"
  if [ -n "${JAMMI_TEST_PG_URL:-}" ]; then
    run_sh tests "Postgres lane" "$(ci_step test-pg 'Run Postgres tests')"
  elif [ "$SKIP_PG" = 1 ]; then
    printf 'skip  [tests] Postgres lane (--skip-pg; CI still runs it)\n'
  else
    ran=$((ran + 1)); failed=$((failed + 1))
    FAILED_LIST+=("[tests] Postgres lane: JAMMI_TEST_PG_URL is unset — start a Postgres 16 (user jammi, db jammi_test), export JAMMI_TEST_PG_URL=postgres://jammi@127.0.0.1:PORT/jammi_test, or pass --skip-pg explicitly")
    printf 'FAIL  [tests] Postgres lane: JAMMI_TEST_PG_URL is unset (pass --skip-pg to skip explicitly)\n'
  fi
fi

# ---------------------------------------------------------------- summary
printf '\nmerge_path: RAN %d  FAILED %d  (logs: %s)\n' "$ran" "$failed" "$LOG_DIR"
for f in "${FAILED_LIST[@]:-}"; do [ -n "$f" ] && printf '  %s\n' "$f"; done
exit "$failed"
