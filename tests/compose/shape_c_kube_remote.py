"""Kubernetes smoke: drive the `deploy/kubernetes/overlays/ci` Deployment
from a REMOTE Python client over a `kubectl port-forward`, and prove a
rollout restart still shares the catalog and broker with the old pod.

Run by `.github/workflows/kube-smoke.yml` AFTER `kustomize build
deploy/kubernetes/overlays/ci | kubectl apply -f -` has been rolled out and
`kubectl rollout status` reports the Deployment ready.

This is a thin driver over `remote_smoke.py`'s shared oracle (same
directory, so `import remote_smoke` resolves under
`python3 tests/compose/shape_c_kube_remote.py` — the script's own directory
is `sys.path[0]`): this module supplies only the Kubernetes-specific
`restart` strategy (close the port-forward, `kubectl rollout restart` +
`rollout status`, reopen the port-forward) and the Kubernetes-specific
after-restart property (`remote_smoke.shared_catalog_after_restart`).

What this proves: readiness after a rollout restart, the runtime oracle on
`get_server_info().broker`, an exact self-hit search, and — via the
after-restart callback — that the NEW pod shares the OLD pod's catalog
(Postgres) and broker (JetStream): the registered source is still visible,
the sources count is unchanged, and the broker is still `jet_stream`.

What this does NOT prove: result-table DURABILITY across the restart, and
the after-restart callback deliberately never queries the result table to
find that out. The `ci` overlay's Deployment mounts `emptyDir` at
`/var/lib/jammi` (no persistent volume, unlike Shape B's Postgres-backed
compose volume; also no `[storage]` block in the overlay's `jammi.toml`, so
the result root defaults under that same `emptyDir`) — a rollout restart's
new pod starts with an EMPTY local index, and the new pod's
`load_existing_tables` skips registering a `ready` row whose Parquet is
gone. `durable_after_restart` (the Compose driver's callback) would fail
here by construction; that is why this driver passes
`shared_catalog_after_restart` instead. The workflow
(`.github/workflows/kube-smoke.yml`) queries the Postgres StatefulSet
directly after this script exits, as the matching runtime oracle for the
catalog side.

Usage: `python3 tests/compose/shape_c_kube_remote.py
[--namespace jammi-ci] [--deployment jammi-server] [--service jammi-server]
[--url grpc://127.0.0.1:8081] [--health-url http://127.0.0.1:8080]
[--dry-run]`. `--dry-run` parses arguments, prints the plan (namespace,
deployment, the `kubectl` argv this script will issue, the same SQL
literals `remote_smoke.py` issues), and exits 0 without connecting or
requiring `kubectl` on PATH. Exits 0 on success.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import remote_smoke
from remote_smoke import MODEL, SOURCE_URL


class PortForward:
    """Owns one `kubectl port-forward` subprocess for the lifetime of the
    smoke run. `open()`/`close()` are exposed separately (not context-manager
    methods) because the restart strategy below needs to close and reopen
    the SAME forward around a rollout restart."""

    def __init__(self, namespace: str, service: str) -> None:
        self._namespace = namespace
        self._service = service
        self._proc: subprocess.Popen | None = None

    def open(self) -> None:
        argv = [
            "kubectl",
            "-n",
            self._namespace,
            "port-forward",
            f"svc/{self._service}",
            "8081:8081",
            "8080:8080",
        ]
        print(f"=== {' '.join(argv)} ===")
        self._proc = subprocess.Popen(argv)

    def close(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=10)
            self._proc = None


def run(
    *,
    namespace: str,
    deployment: str,
    service: str,
    target: str,
    health_url: str,
) -> int:
    pf = PortForward(namespace, service)
    try:
        pf.open()
        remote_smoke.wait_for_ready(health_url, 60)

        def restart() -> None:
            pf.close()
            print(f"=== kubectl -n {namespace} rollout restart deploy/{deployment} ===")
            subprocess.run(
                ["kubectl", "-n", namespace, "rollout", "restart", f"deploy/{deployment}"],
                check=True,
            )
            print(f"=== kubectl -n {namespace} rollout status deploy/{deployment} ===")
            subprocess.run(
                [
                    "kubectl",
                    "-n",
                    namespace,
                    "rollout",
                    "status",
                    f"deploy/{deployment}",
                    "--timeout=180s",
                ],
                check=True,
            )
            pf.open()
            remote_smoke.wait_for_ready(health_url, 60)
            print(f"{deployment} healthy after rollout restart")

        return remote_smoke.run(
            target,
            health_url,
            restart=restart,
            after_restart=remote_smoke.shared_catalog_after_restart,
        )
    finally:
        pf.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="jammi-ci")
    parser.add_argument("--deployment", default="jammi-server")
    parser.add_argument("--service", default="jammi-server")
    parser.add_argument("--url", default="grpc://127.0.0.1:8081")
    parser.add_argument("--health-url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan (namespace, deployment, kubectl argv, SQL literals) "
        "and exit 0 without connecting.",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("shape_c_kube_remote --dry-run")
        print(f"  namespace    = {args.namespace}")
        print(f"  deployment   = {args.deployment}")
        print(f"  service      = {args.service}")
        print(f"  target       = {args.url}")
        print(f"  health url   = {args.health_url}")
        print(f"  source url   = {SOURCE_URL}")
        print(f"  model        = {MODEL}")
        print(
            f"  kubectl -n {args.namespace} port-forward svc/{args.service} "
            "8081:8081 8080:8080"
        )
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
        print(f"  kubectl -n {args.namespace} rollout restart deploy/{args.deployment}")
        print(
            f"  kubectl -n {args.namespace} rollout status deploy/{args.deployment} "
            "--timeout=180s"
        )
        print("  assert describe_source(\"patents\") is not None")
        print("  assert len(list_sources()) unchanged  # NOT the result table -- emptyDir, see docstring")
        print("  assert get_server_info().broker == \"jet_stream\"  # re-check")
        return 0

    return run(
        namespace=args.namespace,
        deployment=args.deployment,
        service=args.service,
        target=args.url,
        health_url=args.health_url,
    )


if __name__ == "__main__":
    sys.exit(main())
