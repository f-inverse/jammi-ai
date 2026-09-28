#!/usr/bin/env python3
"""Assemble and judge the GPU topology lane's artifact from what the driver
pulled off both fleet hosts.

`runpod_gpu_topology.sh` pulls each host's working directory into
`<dir>/host0` and `<dir>/host1`: the one-host cell's test logs and device
listing (host 0), the fleet program's record per transport
(`fleet-nccl.json`, `fleet-cpu.json`, host 0), and every server's log
(`<collective>-h<host>-g<gpu>.log`, both hosts). This program owns the
cross-run claims no single run can make, and writes `topology.json` — the
`topology` kind `check_cuda_run_artifacts.py`'s rule (k) gates — to stdout:

- both fleet runs passed their own checks (completed, one process per rank,
  ranks spanning every host);
- the coordinator of each run logged the transport its phase configured —
  `Nccl` under `collective = "nccl"`, `Inline` under `"cpu"` — for that job,
  and published it on its FIRST attempt: a gang that only finished after a
  retry hung somewhere, and a retry that publishes must never hide that;
- the two runs' per-epoch losses and probe embeddings are IDENTICAL: the
  transport is invisible in the result;
- the gang's per-epoch loss is within the trainer's pre-registered 1e-4 of
  the single-rank reference at the same global batch.

Exit 0 when every claim holds, 1 otherwise (the reasons are in the JSON and
on stderr). Embeddings enter the artifact as a digest, never verbatim.

Usage:
    gpu_topology_assemble.py DIR --sha SHA --gpu-type TYPE \\
        --data-center DC --gpus-per-host N
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import re
import sys

W_VS_1_LOSS_EPSILON = 1e-4
PHASES = {"nccl": "Nccl", "cpu": "Inline"}
ANSI = re.compile(r"\x1b\[[0-9;]*m")
TEST_OK = re.compile(r"^test (\S+) \.\.\. ok$")
NCCL_VERSION = re.compile(r"NCCL version (\S+)")


def lines(path: pathlib.Path) -> list[str]:
    return [ANSI.sub("", l) for l in path.read_text(errors="replace").splitlines()]


def passed_tests(log: pathlib.Path) -> list[str]:
    return sorted(m.group(1) for l in lines(log) if (m := TEST_OK.match(l.strip())))


def server_logs(dir: pathlib.Path, phase: str) -> list[pathlib.Path]:
    return sorted(dir.glob(f"host*/{phase}-h*-g*.log"))


def selected_transports(logs: list[pathlib.Path], job_id: str) -> list[str]:
    """Every `gang transport selected` line naming `job_id`, as its transport."""
    found = []
    for log in logs:
        for line in lines(log):
            if "gang transport selected" in line and f"job_id={job_id}" in line:
                m = re.search(r"transport=(\w+)", line)
                found.append(m.group(1) if m else "<unnamed>")
    return found


def attempt_ends(logs: list[pathlib.Path], job_id: str) -> list[tuple[int, str]]:
    """Every `coordinator attempt ended` line naming `job_id`, as
    `(attempt, end)` in log order."""
    found = []
    for log in logs:
        for line in lines(log):
            if "coordinator attempt ended" in line and f"job_id={job_id}" in line:
                attempt = re.search(r"attempt=(\d+)", line)
                end = re.search(r" end=(\S+)", line)
                found.append((int(attempt.group(1)) if attempt else 0, end.group(1) if end else "<unnamed>"))
    return found


def digest(embeddings: list[list[float]]) -> str:
    return hashlib.sha256(json.dumps(embeddings).encode()).hexdigest()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dir", type=pathlib.Path)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--gpu-type", required=True)
    parser.add_argument("--data-center", required=True)
    parser.add_argument("--gpus-per-host", type=int, required=True)
    args = parser.parse_args(argv)
    host0 = args.dir / "host0"
    reasons: list[str] = []

    with (host0 / "device.csv").open() as f:
        devices = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]
    one_host = {
        "device_tests": passed_tests(host0 / "one-host-device.log"),
        "product_tests": passed_tests(host0 / "one-host-product.log"),
    }

    runs = {}
    for phase, transport in PHASES.items():
        record = json.loads((host0 / f"fleet-{phase}.json").read_text())
        gang = record["gang"]
        if record["verdict"] != "pass":
            reasons += [f"{phase}: {r}" for r in record["reasons"]]
        selected = selected_transports(server_logs(args.dir, phase), gang["job_id"])
        if selected != [transport]:
            reasons.append(
                f"{phase}: the coordinator must log exactly one {transport} transport "
                f"for job {gang['job_id']}, found {selected}"
            )
        ends = attempt_ends(server_logs(args.dir, phase), gang["job_id"])
        if ends != [(1, "published")]:
            reasons.append(
                f"{phase}: the gang must publish on its first attempt, its coordinator logged {ends}"
            )
        runs[phase] = {"record": record, "selected": selected, "attempts": len(ends)}

    nccl, inline = runs["nccl"]["record"], runs["cpu"]["record"]
    if nccl["gang"]["loss_curve"] != inline["gang"]["loss_curve"]:
        reasons.append(
            f"the transports' loss curves differ: nccl {nccl['gang']['loss_curve']} "
            f"vs inline {inline['gang']['loss_curve']}"
        )
    if nccl["gang"]["embeddings"] != inline["gang"]["embeddings"]:
        reasons.append("the transports' probe embeddings differ")

    reference = nccl["reference"]["loss_curve"]
    gang_curve = nccl["gang"]["loss_curve"]
    if len(reference) != len(gang_curve):
        reasons.append(f"{len(gang_curve)} gang epochs vs {len(reference)} reference epochs")
    deltas = [abs(g - r) for g, r in zip(gang_curve, reference)]
    if any(d > W_VS_1_LOSS_EPSILON for d in deltas):
        reasons.append(
            f"the gang's loss {gang_curve} strays from the single rank's {reference} "
            f"beyond {W_VS_1_LOSS_EPSILON}"
        )
    if nccl["reference"]["status"] != "completed":
        reasons.append(f"the reference job ended {nccl['reference']['status']!r}")

    versions = sorted(
        {m.group(1) for log in server_logs(args.dir, "nccl") for l in lines(log)
         if (m := NCCL_VERSION.search(l))}
    )
    if len(versions) != 1:
        reasons.append(f"the fleet's servers must report one NCCL runtime, found {versions}")
    for key in ("device_tests", "product_tests"):
        if not one_host[key]:
            reasons.append(f"the one-host cell passed no {key.replace('_', ' ')}")

    def run_summary(phase: str) -> dict:
        record, selected = runs[phase]["record"], runs[phase]["selected"]
        gang = record["gang"]
        return {
            "job_id": gang["job_id"],
            "status": gang["status"],
            "transport": selected[0] if len(selected) == 1 else selected,
            "attempts": runs[phase]["attempts"],
            "ranks": [
                {"instance": r, **record["workers"].get(r, {"label": None, "host": None})}
                for r in gang["ranks"]
            ],
            "loss_curve": gang["loss_curve"],
            "embeddings_sha256": digest(gang["embeddings"]),
        }

    topology = {
        "hosts": 2,
        "gpus_per_host": args.gpus_per_host,
        "devices_host0": devices,
        "nccl_version": versions,
        "one_host": one_host,
        "fleet": {
            "world_size": len(nccl["gang"]["ranks"]),
            "nccl": run_summary("nccl"),
            "inline": run_summary("cpu"),
            "reference": {
                "job_id": nccl["reference"]["job_id"],
                "loss_curve": reference,
                "embeddings_sha256": digest(nccl["reference"]["embeddings"]),
            },
            "max_loss_delta_vs_reference": max(deltas, default=None),
            "epsilon": W_VS_1_LOSS_EPSILON,
        },
        "verdict": "fail" if reasons else "pass",
        "reasons": reasons,
    }
    artifact = {
        "schema_version": 1,
        "git_sha": args.sha,
        "box": (
            f"{args.gpu_type}, 2 hosts x {args.gpus_per_host} GPUs in {args.data_center}, "
            "Global Networking"
        ),
        "producer": {
            "path": "ci/scripts/runpod_gpu_topology.sh",
            "kind": "script",
            "invocation": "bash ci/scripts/runpod_gpu_topology.sh",
            "gating": "feature:live-gpu-gang-tests",
        },
        "status": "RED" if reasons else "GREEN",
        "artifact_kind": "topology",
        "topology": topology,
    }
    json.dump(artifact, sys.stdout, indent=2)
    sys.stdout.write("\n")
    for r in reasons:
        print(f"::error::{r}", file=sys.stderr)
    return 1 if reasons else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
