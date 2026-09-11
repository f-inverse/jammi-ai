"""Compose smoke (Shape B): RELEASE via SIGINT, then the restarted container
reclaims the released job.

Run by `.github/workflows/compose-smoke.yml` AFTER `shape_b_remote.py`, against
the same `docker compose -f deploy/docker-compose.yml -f
deploy/docker-compose.ci.yml` stack. It proves the two-mode shutdown's fast
arm end to end on the shipped image, against the Postgres catalog:

  1. a long fine-tune (20 000 epochs over the bundled `training_pairs.csv`
     with the bundled `tiny_bert`) is submitted over the wire and observed
     `running`;
  2. `docker compose kill -s SIGINT jammi-server` — the signal an operator's
     `preStop` hook or `jammi-server release` sends — makes the server RELEASE:
     the log carries its "released its leases; exiting now" line (the process
     exits 0 by construction of that line) within 30 s;
  3. `psql` reads the row `lease_expires_at IS NULL AND releases = 1` —
     the lease was handed back, no status invented, no attempt consumed;
  4. `restart: unless-stopped` brings the container back (`/readyz` 200
     again) and its worker reclaims the row within one idle poll:
     `attempts = 2`.

Usage: `python3 tests/compose/shape_b_release.py [--url grpc://127.0.0.1:8081]
[--health-url http://127.0.0.1:8080] [--dry-run]`. `--dry-run` prints the plan
and exits 0 without connecting. Exits 0 on success.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import remote_smoke

COMPOSE_FILES = [
    "deploy/docker-compose.yml",
    "deploy/docker-compose.ci.yml",
]
COMPOSE_SERVICE = "jammi-server"
TRAINING_URL = "file:///fixtures/training_pairs.csv"
MODEL = "local:/fixtures/tiny_bert"
RELEASED_LOG_LINE = "released its leases; exiting now"


def _compose_cmd(*args: str) -> list[str]:
    cmd = ["docker", "compose"]
    for f in COMPOSE_FILES:
        cmd += ["-f", f]
    cmd += list(args)
    return cmd


def _psql(sql: str) -> str:
    out = subprocess.run(
        _compose_cmd("exec", "-T", "postgres", "psql", "-U", "jammi", "-d", "jammi", "-tAc", sql),
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def _server_logs() -> str:
    out = subprocess.run(
        _compose_cmd("logs", "--no-color", COMPOSE_SERVICE),
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout


def _wait_for(predicate, timeout_secs: float, what: str) -> None:
    deadline = time.monotonic() + timeout_secs
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.5)
    raise AssertionError(f"{what} did not happen within {timeout_secs}s")


def run(target: str, health_url: str) -> int:
    import jammi

    remote_smoke.wait_for_ready(health_url, timeout_secs=60)
    db = jammi.connect(target)
    try:
        db.add_source("training", url=TRAINING_URL, format="csv")
        job = db.fine_tune(
            source="training",
            base_model=MODEL,
            columns=["text_a", "text_b", "score"],
            method="lora",
            task="text_embedding",
            epochs=20_000,
            batch_size=8,
            lora_rank=4,
            warmup_steps=0,
        )
        job_id = job.job_id
        print(f"=== submitted long fine-tune {job_id} ===")
        _wait_for(lambda: job.status() == "running", 120, "the job being claimed")
        print("job running; sending SIGINT (RELEASE)")
    finally:
        db.close()

    logs_before = _server_logs().count(RELEASED_LOG_LINE)
    subprocess.run(_compose_cmd("kill", "-s", "SIGINT", COMPOSE_SERVICE), check=True)
    _wait_for(
        lambda: _server_logs().count(RELEASED_LOG_LINE) > logs_before,
        30,
        "the server's RELEASE exit line",
    )
    print("server released and exited")

    row = _psql(
        f"select status, lease_expires_at is null, releases, attempts "
        f"from jobs where job_id = '{job_id}'"
    )
    print(f"row after release: {row}")
    status, lease_null, releases, attempts = row.split("|")
    assert status == "running", row
    assert lease_null == "t", f"the lease must be NULL after a release: {row}"
    assert releases == "1", f"releases must be 1: {row}"
    assert attempts == "1", f"a release costs no attempt: {row}"

    # `restart: unless-stopped` brings the container back; its worker claims.
    remote_smoke.wait_for_ready(health_url, timeout_secs=120)
    print("restarted container ready")
    _wait_for(
        lambda: _psql(f"select attempts from jobs where job_id = '{job_id}'") == "2",
        60,
        "the restarted worker reclaiming the released row (attempts = 2)",
    )
    row = _psql(
        f"select status, releases, attempts from jobs where job_id = '{job_id}'"
    )
    print(f"row after reclaim: {row}")
    assert row.split("|")[0] in ("running", "completed"), row
    assert row.split("|")[1] == "1", row
    print("OK: released job reclaimed by the restarted container")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="grpc://127.0.0.1:8081")
    parser.add_argument("--health-url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit 0 without connecting.",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("shape_b_release --dry-run")
        print(f"  target       = {args.url}")
        print(f"  health url   = {args.health_url}")
        print(f"  training url = {TRAINING_URL}")
        print(f"  model        = {MODEL}")
        print("  fine_tune(source=\"training\", epochs=20000, …) until status == running")
        print(f"  docker compose kill -s SIGINT {COMPOSE_SERVICE}")
        print(f"  logs contain {RELEASED_LOG_LINE!r} within 30s")
        print("  select status, lease_expires_at is null, releases, attempts from jobs where job_id = …")
        print("  wait /readyz (restart: unless-stopped); wait attempts = 2")
        return 0

    return run(args.url, args.health_url)


if __name__ == "__main__":
    sys.exit(main())
