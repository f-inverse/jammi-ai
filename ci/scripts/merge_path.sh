#!/usr/bin/env bash
# The merge path, locally: the checks a pull request must pass, read from the
# workflow files themselves and run in one process, with what it does NOT run
# printed by name.
#
#   bash ci/scripts/merge_path.sh [--only STAGE[,STAGE]] [--skip-pg] [--skip-tests] [--skip-mdbook]
#
# Stages, in order:
#   static   fmt, the four clippy surfaces, rustdoc -D warnings, the guide build
#            (ci.yml `check`, docs.yml `build`)
#   guards   every `guard` matrix command in ci.yml (read from the file at run
#            time, never a copied list — the list that drifted cost a CI round)
#   swarm    the Swarm-gates workflow's steps, each step's `run:` block executed
#            WHOLE (a multi-line guard split into lines is never evaluated)
#   tests    the hermetic lane (workspace, test-hooks, golden-parity) and the
#            Postgres lane (ci.yml `test`, `test-pg`)
#   records  check_rigor_record.py and check_oracle_gate.py — the committed
#            rigor record and the oracle's PASS row. The oracle gate's
#            freshness is a COMMITTED-content diff between the recorded
#            head_sha and HEAD outside docs/rigor/**, so the order that
#            matters is the commit order: commit every code and doc change,
#            run the oracle, commit only docs/rigor/** after it.
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

# Same as run_sh, with the rustc wrapper pointed at a path that does not
# exist — the ci.yml guard runner's shape for a non-toolchain leg (sccache
# absent while config.toml mandates it; see the guards stage below).
run_sh_nosccache() {
  local stage="$1" label="$2" cmd
  cmd="$(expand "$3")" || exit 2
  run "$stage" "$label" env RUSTC_WRAPPER="$NOSCCACHE_WRAPPER" bash -e -c "$cmd"
}

export GITHUB_EVENT_NAME=pull_request GITHUB_BASE_REF="$BASE_REF" GITHUB_HEAD_REF="$HEAD_REF" \
  GITHUB_ACTOR="${GITHUB_ACTOR:-local}" GITHUB_WORKSPACE="$ROOT"

# ---------------------------------------------------------------- coverage
# Which ci.yml jobs this runner covers, printed up front so "green" is never
# read as "every job".
COVERED_JOBS="check test test-pg guard"
python3 - "$COVERED_JOBS" <<'PY'
import sys, yaml
ci = yaml.safe_load(open('.github/workflows/ci.yml'))
covered = set(sys.argv[1].split())
jobs = [j for j in ci['jobs'] if j != 'ci-summary']
missing = [j for j in jobs if j not in covered]
print(f"merge_path: covers {len(jobs) - len(missing)} of {len(jobs)} ci.yml jobs (plus swarm.yml and docs.yml's build)")
print("merge_path: NOT run here (CI runs them): " + ", ".join(missing))
PY

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
  GUARD_LIST="$LOG_DIR/guard-matrix.tsv"
  python3 - "$GUARD_LIST" <<'PY'
import sys, yaml
ci = yaml.safe_load(open('.github/workflows/ci.yml'))
entries = ci['jobs']['guard']['strategy']['matrix']['include']
with open(sys.argv[1], 'w') as out:
    for e in entries:
        tc = 'true' if e.get('toolchain') is True else 'false'
        out.write(e['name'] + '\t' + tc + '\t' + ' '.join(e['cmd'].split()) + '\n')
print(f"merge_path: {len(entries)} guard-matrix commands read from ci.yml")
PY
  # ci.yml's `guard` runner is bare ubuntu (no container): cargo and rustc
  # are present, `sccache` — which `.cargo/config.toml` makes the mandatory
  # rustc wrapper — is not, and neither is a linker. A guard that shells out
  # to cargo fails there ("could not execute process `sccache ... rustc -vV`")
  # unless its matrix entry declares `toolchain: true` (then `setup-rust-ci`
  # installs sccache — enough for `cargo metadata`, never for a build; a guard
  # that builds Rust lives in ci.yml's container-backed `symbol-index-gates`
  # job, which the swarm stage below runs). A developer
  # machine has sccache, so the same guard passes here. Mirror the runner:
  # every NON-toolchain leg runs with RUSTC_WRAPPER pointed at a path that
  # does not exist (the env var overrides config.toml, so cargo fails exactly
  # as on the runner; sccache shares ~/.cargo/bin with cargo, so hiding it by
  # PATH would hide cargo too, which is NOT the runner's shape); toolchain
  # legs run as-is. The stage refuses to start if that path exists.
  NOSCCACHE_WRAPPER="/nonexistent/sccache-absent-on-the-guard-runner"
  if [ -e "$NOSCCACHE_WRAPPER" ]; then
    printf 'FAIL  [guards] %s exists — the non-toolchain mirror needs an unexecutable wrapper path\n' "$NOSCCACHE_WRAPPER"
    exit 2
  fi
  # A toolchain leg gets sccache and a fetched registry on the runner and
  # links with the runner's own `ld` — which is enough for `cargo metadata`
  # and for building a FIXTURE workspace (`pod build substrate` does both),
  # but not for building a WORKSPACE crate: `.cargo/config.toml` mandates
  # `-fuse-ld=mold` for x86_64 Linux and the bare runner has no mold (PR #587:
  # `collect2: cannot find 'ld'` from the eager-disable sweep's `cargo run -p
  # probed-ops-index`). That divergence has no macOS mirror; the rule is
  # placement — a guard that builds a workspace crate lives in ci.yml's
  # container-backed `symbol-index-gates` job — and the draft PR's CI is the
  # check. Toolchain legs run as-is here.
  while IFS=$'\t' read -r name toolchain cmd; do
    if [ "$toolchain" = "true" ]; then
      run_sh guards "$name" "$cmd"
    else
      run_sh_nosccache guards "$name" "$cmd"
    fi
  done < "$GUARD_LIST"
fi

# ---------------------------------------------------------------- swarm
if stage_wanted swarm; then
  SWARM_DIR="$LOG_DIR/swarm-steps"
  rm -rf "$SWARM_DIR"; mkdir -p "$SWARM_DIR"
  python3 - "$SWARM_DIR" <<'PY'
import os, sys, yaml
wf = yaml.safe_load(open('.github/workflows/swarm.yml'))
ci = yaml.safe_load(open('.github/workflows/ci.yml'))
# the syn-backed gates live in ci.yml's `symbol-index-gates` job (aggregated by
# the required ci-summary); run them here beside the swarm-gates steps
jobs = list(wf['jobs'].values()) + [ci['jobs']['symbol-index-gates']]
n = 0
for job in jobs:
    for st in job.get('steps', []):
        run = st.get('run')
        if not run:
            continue
        stripped = run.strip()
        # the two record checks run in the `records` stage
        if stripped in ('python3 ci/scripts/check_rigor_record.py',
                        'python3 ci/scripts/check_oracle_gate.py'):
            continue
        n += 1
        with open(os.path.join(sys.argv[1], f"{n:02d}.step"), 'w') as out:
            out.write((st.get('name') or stripped.split('\n')[0]) + '\n')
            out.write(run)
print(f"merge_path: {n} swarm-gate steps read from swarm.yml + ci.yml symbol-index-gates (each run as one block)")
PY
  for step in "$SWARM_DIR"/*.step; do
    name="$(head -n1 "$step")"
    body="$(tail -n +2 "$step")"
    run_sh swarm "$name" "$body"
  done
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
  run records "check_oracle_gate.py (fresh PASS at HEAD's committed content)" python3 ci/scripts/check_oracle_gate.py
fi

# ---------------------------------------------------------------- summary
printf '\nmerge_path: RAN %d  FAILED %d  (logs: %s)\n' "$ran" "$failed" "$LOG_DIR"
for f in "${FAILED_LIST[@]:-}"; do [ -n "$f" ] && printf '  %s\n' "$f"; done
exit "$failed"
