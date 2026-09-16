"""Compose smoke (Shape B): RELEASE via SIGINT, then the restarted container
reclaims the released job.

Run by `.github/workflows/compose-smoke.yml` AFTER `shape_b_remote.py`, against
the same `docker compose -f deploy/docker-compose.yml -f
deploy/docker-compose.ci.yml` stack. It proves the two-mode shutdown's fast
arm end to end on the shipped image, against the Postgres catalog:

  1. a long fine-tune (20 000 epochs over the bundled `training_pairs.csv`
     with the bundled `tiny_bert`) is submitted over the wire and observed
     `running`;
  2. SIGINT delivered to the container's init process from the HOST pid
     namespace (`kill -s SIGINT <pid>` on `docker inspect`'s `.State.Pid`) —
     the signal an operator's `preStop` hook or `jammi-server release` sends —
     makes the server RELEASE: the log carries its "released its leases;
     exiting now" line (the process exits 0 by construction of that line)
     within 30 s. Never `docker kill -s SIGINT`: the daemon records a
     `docker kill` (like a `docker stop`) as a MANUAL stop and then ignores
     `restart: unless-stopped` once the process exits, so step 4 could never
     observe the container come back — this is exactly how this script failed
     on every `main` run from its first (2026-09-14) until it was rewritten;
  3. `restart: unless-stopped` brings the container back (`/readyz` 200
     again) and its worker reclaims the row INSIDE the lease window measured
     from the release (`attempts = 2` within `LEASE_DURATION_SECS` less a
     margin) — a reclaim that fast is explainable only by the lease having
     been handed back, never by expiry;
  4. the end state, read once: `releases = 1` (only the release path writes
     it), `attempts = 2` (one claim plus one reclaim — the release consumed
     none), status `running` or `completed`, and the successor holding the
     lease while running. No snapshot is taken between the release and the
     reclaim: that instant is a few hundred milliseconds wide and any read of
     it races the successor.

Usage: `python3 tests/compose/shape_b_release.py [--url grpc://127.0.0.1:8081]
[--health-url http://127.0.0.1:8080] [--dry-run]`. `--dry-run` prints the plan
and exits 0 without connecting. Exits 0 on success.
"""

from __future__ import annotations

import argparse
import os
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
# The compose stack runs the default `[lease] duration_secs = 30`; the reclaim
# proof below is only discriminating if the successor claims INSIDE that
# window, so its wait is bounded by the window less a margin, never longer.
LEASE_DURATION_SECS = 30
LEASE_WINDOW_MARGIN_SECS = 5


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


def _signal_container_init(signal_name: str) -> None:
    """Deliver `signal_name` to the service container's init process (pid 1
    inside; `.State.Pid` on the host) from the host pid namespace — what an
    operator's `kill` does. This must NOT go through the daemon's kill API:
    `docker kill`/`docker compose kill` flag the container as manually
    stopped, and a manually stopped container is exempt from
    `restart: unless-stopped`, so the restart this script then waits for
    would never happen. A host-side signal leaves the restart policy in
    force. Root is needed to signal the container's root-owned process: the
    CI runner runs as an unprivileged user with passwordless sudo, so a
    non-root caller goes through `sudo -n`; a root caller signals directly.
    Requires a Linux Docker Engine sharing the host pid namespace (the CI
    shape) — Docker Desktop's VM exposes no host pid for a container."""
    container_id = subprocess.run(
        _compose_cmd("ps", "-q", COMPOSE_SERVICE),
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if not container_id:
        raise AssertionError(f"no running container for compose service {COMPOSE_SERVICE!r}")
    pid = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Pid}}", container_id],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if not pid.isdigit() or int(pid) <= 0:
        raise AssertionError(f"container {container_id} has no host pid (State.Pid={pid!r}): not running?")
    kill = ["kill", "-s", signal_name, pid]
    if os.geteuid() != 0:
        kill = ["sudo", "-n", *kill]
    print(f"=== {' '.join(kill)} (container {container_id[:12]}, {COMPOSE_SERVICE}) ===")
    subprocess.run(kill, check=True)


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
    _signal_container_init("SIGINT")
    _wait_for(
        lambda: _server_logs().count(RELEASED_LOG_LINE) > logs_before,
        30,
        "the server's RELEASE exit line",
    )
    print("server released and exited")

    released_at = time.monotonic()

    # No read of the row here. The instant between the release and the
    # successor's reclaim is not observable from outside: the restart policy
    # brings the container back a few hundred milliseconds after the exit
    # line and its worker reclaims on its first poll, so any snapshot taken
    # now races that reclaim. Everything a RELEASE guarantees is carried by
    # the END state, read once the successor has claimed — see below.
    remote_smoke.wait_for_ready(health_url, timeout_secs=120)
    print("restarted container ready")

    # The reclaim must land INSIDE the lease window measured from the release.
    # `claim_next` admits a row only when its lease is NULL or expired; the
    # server's last heartbeat preceded the release, so a reclaim later than
    # `LEASE_DURATION_SECS` after it could also be explained by expiry. A
    # reclaim before that is explainable only by the lease having been handed
    # back — which is the fact step 3 exists to prove.
    _wait_for(
        lambda: _psql(f"select attempts from jobs where job_id = '{job_id}'") == "2",
        LEASE_DURATION_SECS - LEASE_WINDOW_MARGIN_SECS,
        "the restarted worker reclaiming the released row inside the lease window (attempts = 2)",
    )
    reclaimed_after = time.monotonic() - released_at
    print(f"reclaimed {reclaimed_after:.1f}s after the release (lease window {LEASE_DURATION_SECS}s)")

    row = _psql(
        f"select status, lease_expires_at is null, releases, attempts "
        f"from jobs where job_id = '{job_id}'"
    )
    print(f"row after reclaim: {row}")
    status, lease_null, releases, attempts = row.split("|")
    # The end state carries every fact a RELEASE owes:
    #   releases = 1  — only the release path writes it; an expiry-driven
    #                   reclaim leaves it 0;
    #   attempts = 2  — the first claim plus the successor's; the release
    #                   itself consumed none (a failure path would be 2 with
    #                   releases 0, an extra retry 3);
    #   status running or completed, never a status the release invented;
    #   lease set again while running — the successor holds it now.
    assert releases == "1", f"releases must be 1 (the release path wrote it): {row}"
    assert attempts == "2", f"exactly one claim plus one reclaim, the release cost none: {row}"
    assert status in ("running", "completed"), row
    if status == "running":
        assert lease_null == "f", f"the successor must hold the lease while running: {row}"
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
        print(f"  kill -s SIGINT <host pid of {COMPOSE_SERVICE}'s init> (never docker kill: it disarms the restart policy)")
        print(f"  logs contain {RELEASED_LOG_LINE!r} within 30s")
        print("  select status, lease_expires_at is null, releases, attempts from jobs where job_id = …")
        print(f"  wait /readyz (restart: unless-stopped); attempts = 2 within {LEASE_DURATION_SECS - LEASE_WINDOW_MARGIN_SECS}s of the release")
        print("  end state: releases = 1, attempts = 2, running|completed, lease held while running")
        return 0

    return run(args.url, args.health_url)


if __name__ == "__main__":
    sys.exit(main())
