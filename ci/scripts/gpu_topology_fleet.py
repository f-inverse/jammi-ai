#!/usr/bin/env python3
"""The GPU topology lane's fleet program: one fine-tune job across machines.

Run on the fleet's first host by `runpod_gpu_topology.sh`, against `jammi-server`
processes the driver started — one per GPU, on every host, sharing one Postgres
catalog and one object-store result root — through the published Python client,
exactly as any deployment's client would. The training pairs are a file at the
same path on every host (the checkout's `training_triplets.csv`, read as its
`(anchor, positive)` pairs), so whichever server claims the job resolves the
registered source.

It submits the job at `--world-size` ranks (a gang wider than any one process,
so the claiming server coordinates and dials the rest over the compute plane),
waits for it, and records, as JSON on stdout:

- the job's status, `claimed_by` and `ranks` (the instance that ran each rank),
  with each instance's label and host (the machine it runs on) from
  `list_workers`;
- the per-epoch training loss;
- the fine-tuned model's embedding of every probe.

With `--reference` it also trains the single-rank reference at the SAME global
batch on whichever server claims it, and records its loss and embeddings.

The driver runs it once per transport (the fleet restarted between them with
`[worker] collective = "nccl"` and `"cpu"`) and compares the records; this
program asserts only what one run owns: the job completed, and its ranks
spanned `--expect-hosts` hosts.

Usage:
    gpu_topology_fleet.py --endpoint grpc://HOST:PORT --pairs PATH \\
        --model local:/path --world-size 4 --per-rank-batch 2 \\
        --expect-hosts 2 [--reference]
"""

from __future__ import annotations

import argparse
import json
import sys

PROBES = [
    "quantum error correction methods",
    "protein folding with transformers",
    "graph kernels for chemistry",
    "solid electrolyte interphase control",
]


def train(db, model: str, world_size: int, batch_size: int) -> dict:
    """The gang oracle's configuration (`gang_fixtures.rs`'s `gang_config`):
    rank-2 LoRA, no dropout, no warmup, no validation split, a fixed seed —
    over the fixture's `(anchor, positive)` pairs, so the loss is the
    in-batch one whose negatives cross ranks — the gather the W-vs-1 bound
    is registered for. 15 rows at a global batch of 8 end every epoch on a
    7-row remainder step."""
    job = db.fine_tune(
        source="pairs",
        base_model=model,
        columns=["anchor", "positive"],
        method="lora",
        task="text_embedding",
        epochs=3,
        batch_size=batch_size,
        lora_rank=2,
        lora_dropout=0.0,
        learning_rate=1e-4,
        warmup_steps=0,
        gradient_accumulation_steps=1,
        validation_fraction=0.0,
        early_stopping_metric="train_loss",
        early_stopping_patience=10_000,
        seed=99,
        world_size=world_size,
    )
    job.wait()
    listing = {j["job_id"]: j for j in db.list_jobs()}[job.job_id]
    return {
        "job_id": job.job_id,
        "status": listing["status"],
        "claimed_by": listing["claimed_by"],
        "ranks": listing["ranks"],
        "loss_curve": [p["loss"] for p in job.metrics()["train_loss_curve"]],
        "embeddings": [
            list(db.encode_query(model=job.output_model_id, query=p)) for p in PROBES
        ],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--per-rank-batch", type=int, required=True)
    parser.add_argument("--expect-hosts", type=int, required=True)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args(argv)

    import jammi

    with jammi.connect(args.endpoint) as db:
        db.add_source("pairs", url=args.pairs, format="csv")
        workers = {w["instance_id"]: w for w in db.list_workers()}
        gang = train(db, args.model, args.world_size, args.per_rank_batch)
        record = {
            "workers": {i: {"label": w["label"], "host": w["host"]} for i, w in workers.items()},
            "gang": gang,
        }
        if args.reference:
            record["reference"] = train(
                db, args.model, 1, args.world_size * args.per_rank_batch
            )

    failures = []
    if gang["status"] != "completed":
        failures.append(f"the gang job ended {gang['status']!r}")
    if len(gang["ranks"]) != args.world_size:
        failures.append(f"{len(gang['ranks'])} ranks recorded for a world of {args.world_size}")
    unknown = [r for r in gang["ranks"] if r not in record["workers"]]
    if unknown:
        failures.append(f"ranks ran on instances no worker lists: {unknown}")
    hosts = {record["workers"][r]["host"] for r in gang["ranks"] if r in record["workers"]}
    if len(hosts) != args.expect_hosts:
        failures.append(f"the ranks spanned {len(hosts)} host(s), expected {args.expect_hosts}")
    if len(set(gang["ranks"])) != len(gang["ranks"]):
        failures.append("two ranks ran in one process: a fleet rank is one process")
    record["verdict"] = "fail" if failures else "pass"
    record["reasons"] = failures
    json.dump(record, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
