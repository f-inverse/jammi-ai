"""Compose smoke: drive `deploy/docker-compose.yml` (Shape B) from a REMOTE
Python client over the wire, and prove a restart does not lose an existing
result.

Run by `.github/workflows/compose-smoke.yml` AFTER `docker compose -f
deploy/docker-compose.yml -f deploy/docker-compose.ci.yml up --wait` has
brought the stack up healthy. Deliberately outside `tests/uat/` — ci.yml's
`test-python` job globs `tests/uat/shape_b_*.py` / `tests/uat/shape_c_*.py`
on every PR against the EMBEDDED engine; this script needs a running remote
server + Postgres + JetStream, which only this workflow provisions.

This is now a thin driver over `remote_smoke.py`'s shared oracle (same
directory, so `import remote_smoke` resolves whether this script is run
directly or imported): this module supplies only the Compose-specific
`restart` strategy (`docker compose restart`, waiting for `/readyz` again)
and the Compose-specific after-restart property
(`remote_smoke.durable_after_restart` — the segment bundle lives on a
Postgres-backed volume, so it must survive byte-for-byte). See
`remote_smoke.py`'s own module doc for what `run()` actually proves: a
runtime oracle on `get_server_info().broker`, an exact self-hit search, and
(via the callback here) result durability across the restart. The workflow
(`.github/workflows/compose-smoke.yml`) queries the Postgres container
directly after this script exits, as the matching runtime oracle for the
catalog side.

Usage: `python3 tests/compose/shape_b_remote.py [--url grpc://127.0.0.1:8081]
[--health-url http://127.0.0.1:8080] [--dry-run]`. `--dry-run` parses
arguments, prints the plan (target, fixture paths, the SQL literals this
script issues), and exits 0 without connecting — the offline check this
script's own argument parsing and SQL text get when Docker is unavailable.
Exits 0 on success.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import remote_smoke
from remote_smoke import MODEL, SOURCE_URL

COMPOSE_FILES = [
    "deploy/docker-compose.yml",
    "deploy/docker-compose.ci.yml",
]
COMPOSE_SERVICE = "jammi-server"


def _compose_cmd(*args: str) -> list[str]:
    cmd = ["docker", "compose"]
    for f in COMPOSE_FILES:
        cmd += ["-f", f]
    cmd += list(args)
    return cmd


def restart_and_wait(health_url: str, timeout_secs: float) -> None:
    print(f"=== docker compose restart {COMPOSE_SERVICE} ===")
    subprocess.run(
        _compose_cmd("restart", COMPOSE_SERVICE),
        check=True,
    )
    remote_smoke.wait_for_ready(health_url, timeout_secs)
    print(f"{COMPOSE_SERVICE} healthy after restart")


def run(target: str, health_url: str) -> int:
    return remote_smoke.run(
        target,
        health_url,
        restart=lambda: restart_and_wait(health_url, timeout_secs=60),
        after_restart=remote_smoke.durable_after_restart,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="grpc://127.0.0.1:8081")
    parser.add_argument("--health-url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan (target, fixtures, SQL literals) and exit 0 without connecting.",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("shape_b_remote --dry-run")
        print(f"  target       = {args.url}")
        print(f"  health url   = {args.health_url}")
        print(f"  source url   = {SOURCE_URL}")
        print(f"  model        = {MODEL}")
        print("  assert get_server_info().broker == \"jet_stream\"")
        print("  add_source(\"patents\", url=SOURCE_URL, format=\"parquet\")")
        print("  SELECT count(*) FROM patents.public.patents")
        print(
            "  generate_embeddings(source=\"patents\", model=MODEL, "
            "columns=[\"abstract\"], key=\"id\")"
        )
        print('  SELECT count(*) FROM "jammi.{table}"')
        print('  SELECT vector FROM "jammi.{table}" WHERE _row_id = \'1\'')
        print("  search(source=\"patents\", query=vec, k=5)")
        print(f"  docker compose restart {COMPOSE_SERVICE}")
        return 0

    return run(args.url, args.health_url)


if __name__ == "__main__":
    sys.exit(main())
