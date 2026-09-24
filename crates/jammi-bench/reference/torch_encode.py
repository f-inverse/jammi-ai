#!/usr/bin/env python3
"""PyTorch reference for jammi-bench's `encode-step` tier.

An ORACLE, not a dependency (see this directory's README): pure Python against
the public `transformers`/`tokenizers`/`pyarrow` APIs, run next to
`jammi-bench encode-step` on the same box over the same files.

It does the SAME work `crates/jammi-bench/src/encode_step.rs` times — rows in
a Parquet table → persisted L2-normalized embeddings — and every choice below
is pinned to what the engine's serving path does, not to what a generic
sentence-embedding recipe would do:

* **Same rows.** It reads the corpus Parquet the jammi leg SERVED
  (`<exchange-dir>/corpus_<rows>.parquet`, written by `encode-step
  --exchange-dir`), never a regenerated one; `corpus_sha256` (the file's own
  bytes) is on both reports.
* **Same tokenizer.** `tokenizers.Tokenizer.from_file(<model-dir>/tokenizer.json)`
  — the Rust library `jammi_ai::model::tokenizer::TokenizerWrapper` wraps,
  through its Python binding — at the same truncation bound
  (`max_position_embeddings`, what the engine's loaded encoders report as
  `max_sequence_length`). `token_lengths_sha256` — the digest of every row's
  real token count — is on both reports, so "the same tokenization" is a
  checked fact.
* **Same pooling, same normalization.** `1_Pooling/config.json` is resolved by
  the rules of `backend/candle.rs`'s `pooling_from_config` (absent → mean; a
  present file must declare exactly one representable strategy), pooled by a
  port of `jammi-encoders`' `pooling.rs`, and L2-normalized by the floor
  `torch_finetune_step.py` already ports.
* **Same dtype, same chunk budget**, as flags; `--batch-size` and
  `--batch-tokens` are `[inference] batch_size` and `batch_tokens`.
* **Same span.** One timed serve reads the corpus file, tokenizes and forwards
  it chunk by chunk, and WRITES the embeddings to Parquet — because the jammi
  leg's span ends at a committed result table, not at a tensor. The engine's
  committed table is also SEARCHABLE: its embeddings sink builds an ANN segment
  beside the Parquet object (a `usearch` HNSW graph — cosine, `f32`, the
  library's default connectivity and expansion, filled row by row on one
  thread). `--ann-index` makes this leg build and save the same graph inside
  its span; without it this leg has done less work than the jammi leg, and
  `ann_index: false` on its report says so.

## Two orders

`--order plan` forwards the chunks the engine's plan cuts
(`jammi_numerics::batch_shape`, ported below): rows ordered by token count,
ties in key order, cut in one pass under `--batch-size` rows and
`--batch-tokens` padded tokens, each chunk padded to the rung of the engine's
shape ladder its longest row rounds up to — hence the same padding. It is the
SEMANTIC twin. `--order length-sorted` forwards them longest-first at the
batch's natural width, `--batch-size` at a time, and restores key order
afterwards — what `sentence-transformers`' `encode()` does by default, and so
the bar a user would actually hold the engine to. `padded_tokens` on each
point says what each order pads.

## Legs, and one instrument per quantity

This script is the `torch` RUNG of the `encode` ladder: it emits LEGS by the
same contract as `jammi-bench encode-step` (`<rung>__rows<N>__r<take>.json`
under `--legs-dir`, the leg block at the top level under `encode_step`, a first
take's vectors beside it as little-endian `f32` in key order) and decides
nothing — every ratio, budget and verdict is `jammi-bench ladder encode`'s.
Each leg carries its per-iteration wall-time series, never only a summary.

Each (unit, take) is measured in a fresh child of this script, because the two
space instruments are per-process and the same for every rung: host memory is
the kernel's resident-set high-water mark (`VmHWM`, which never falls), and
device memory is one external whole-device sampler wrapped around the process
— `jammi-bench sample-device`, given as `--sampler-bin`, the same sampler that
wraps a jammi leg. Without it a leg's `peak_vram_bytes` is unmeasured. Torch's
own allocator counters (`peak_vram_allocator_bytes`) are provenance, never the
compared quantity.

`--dry-run` needs no checkpoint and no jammi leg: it serves the repository's
own `cookbook/fixtures/tiny_bert` through the SAME loader and the same code
path over a small corpus it writes itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import ladder_leg as ll
import shape_ladder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch_finetune_step as tfs  # noqa: E402

RUNG = "torch"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DRY_RUN_MODEL_DIR = os.path.join(REPO_ROOT, "cookbook", "fixtures", "tiny_bert")
DRY_RUN_ROWS = (8, 24)
DRY_RUN_WORDS = ("quantum", "error", "correction", "codes", "of", "the", "ribosome", "structure")

KEY_COLUMN = "_row_id"
TEXT_COLUMN = "text"

# `--dtype`, in the vocabulary jammi's `ComputePrecision` renders
# (`EncodeStepTier::compute_precision`), so the two reports compare verbatim.
# Straight casts of the whole model, as the engine's inference lanes are.
DTYPES = ("f32", "bf16", "f16")

# `backend/candle.rs::pooling_from_config`'s RECOGNIZED table.
# `pooling_mode_mean_sqrt_len_tokens` is a positive multiple of the mean, which
# normalization erases, so it IS mean.
POOLING_FLAGS = {
    "pooling_mode_cls_token": "cls",
    "pooling_mode_mean_tokens": "mean",
    "pooling_mode_max_tokens": "max",
    "pooling_mode_weightedmean_tokens": "weighted_mean",
    "pooling_mode_mean_sqrt_len_tokens": "mean",
}

# `EncodeStepTier::IDENTITY_FIELDS` (`crates/jammi-bench/src/report.rs`), the
# one declaration of what two legs of this workload must agree on, as this
# producer states them. `seed` is the exception: this rung reads the corpus
# from the file the jammi leg served and never generates one, so it carries
# the jammi leg's seed through from `--seed`.
IDENTITY_FIELDS = (
    "task",
    "seed",
    "rows",
    "corpus_sha256",
    "token_lengths_sha256",
    "tokens",
    "batch_size",
    "batch_tokens",
    "max_sequence_length",
    "compute_precision",
    "checkpoint_config_sha256",
    "checkpoint_weights_sha256",
    "checkpoint_weights_size_bytes",
    "checkpoint_tokenizer_sha256",
    "pooling",
    "normalize",
    "iters_measured",
    "checkpoint_pooling_sha256",
    "device_requested",
)


def sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def token_lengths_sha256(row_tokens) -> str:
    """`encode_step.rs::token_lengths_sha256`: the decimal counts, in row
    order, joined by `,`."""
    return hashlib.sha256(",".join(str(n) for n in row_tokens).encode()).hexdigest()


def resolve_pooling(model_dir: str):
    """`(strategy, sha256 of 1_Pooling/config.json or None)` by
    `pooling_from_config`'s rules — a declaration the engine would refuse is
    refused here too, never silently mean-pooled."""
    path = os.path.join(model_dir, "1_Pooling", "config.json")
    if not os.path.exists(path):
        return "mean", None
    with open(path) as fh:
        declared = json.load(fh)
    true_flags = [k for k, v in declared.items() if k.startswith("pooling_mode_") and v is True]
    unsupported = [k for k in true_flags if k not in POOLING_FLAGS]
    strategies = sorted({POOLING_FLAGS[k] for k in true_flags if k in POOLING_FLAGS})
    if not true_flags or unsupported or len(strategies) != 1:
        raise ValueError(
            f"{path} declares {true_flags}: the engine serves exactly one of "
            f"{sorted(set(POOLING_FLAGS.values()))} and refuses anything else"
        )
    return strategies[0], sha256_file(path)


def pool(hidden, attention_mask, strategy: str):
    """Port of `jammi-encoders/src/pooling.rs`'s four strategies, before
    normalization."""
    import torch

    if strategy == "cls":
        return hidden[:, 0]
    mask = attention_mask.to(torch.float32).unsqueeze(-1).expand(hidden.shape)
    if strategy == "mean":
        return (hidden * mask.to(hidden.dtype)).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0).to(hidden.dtype)
    if strategy == "max":
        sentinel = -65504.0 if hidden.dtype == torch.float16 else -1e30
        return torch.where(mask.bool(), hidden, torch.full_like(hidden, sentinel)).max(dim=1).values
    if strategy == "weighted_mean":
        positions = torch.arange(1, hidden.shape[1] + 1, device=hidden.device, dtype=torch.float32)
        effective = mask * positions.view(1, -1, 1)
        return (hidden * effective.to(hidden.dtype)).sum(dim=1) / effective.sum(dim=1).clamp(min=1.0).to(hidden.dtype)
    raise ValueError(f"unknown pooling strategy {strategy!r}")


def read_corpus(path: str):
    """`(keys, texts)` in KEY order — the order the engine's plan forwards
    (`Sort(CAST(key AS Utf8))`)."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=[KEY_COLUMN, TEXT_COLUMN])
    rows = sorted(zip(table.column(KEY_COLUMN).to_pylist(), table.column(TEXT_COLUMN).to_pylist()))
    return [k for k, _ in rows], [t for _, t in rows]


def write_parquet(path: str, keys, vectors) -> None:
    """The persisted table a serve ends at: `_row_id` beside a `vector`
    `FixedSizeList<Float32>`, the shape the engine's result table has."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    flat = pa.array(vectors.reshape(-1), type=pa.float32())
    table = pa.table(
        {
            KEY_COLUMN: pa.array(keys, type=pa.string()),
            "vector": pa.FixedSizeListArray.from_arrays(flat, vectors.shape[1]),
        }
    )
    pq.write_table(table, path)


def write_vectors(path: str, vectors) -> None:
    """The ladder's `vectors_file`: little-endian `f32`, row-major, in key
    order."""
    vectors.astype("<f4").tofile(path)


def outcome_digest(vectors) -> str:
    """`encode_step.rs`'s FNV fold of an embedding artifact: every float's
    bits with a width tag per vector. Comparable between two torch legs of
    one unit on one box; never across stacks."""
    h = 0xCBF29CE484222325
    for row in vectors.astype("<f4"):
        for byte in row.tobytes() + bytes([row.shape[0] & 0xFF]):
            h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def write_ann_index(path: str, vectors) -> None:
    """The graph the engine's embeddings sink builds (`jammi-db`'s
    `index/sidecar.rs`): `usearch`, cosine, `f32`, default connectivity and
    expansion, one `add` per row on one thread — then saved, as the engine
    persists its segment."""
    import numpy as np
    from usearch.index import Index

    index = Index(ndim=vectors.shape[1], metric="cos", dtype="f32")
    index.add(np.arange(len(vectors)), vectors, threads=1)
    index.save(path)


# `ShapeLadder::width`, shared with every twin that pads a batch.
ladder_width = shape_ladder.width


def plan_chunks(costs, batch_size: int, batch_tokens: int, limit: int):
    """`ChunkCutter` over the rows in the plan's order: `(chunk row indices,
    padded width)` per forward. Rows ordered by cost, ties in key order (the
    plan's `(_cost, _ordinal)` sort); one pass, each row joining the open
    chunk unless that would exceed `batch_size` rows or `batch_tokens`
    padded tokens at the ladder width of its longest row; a row alone always
    forms a chunk."""
    chunks, open_rows, longest = [], [], 0
    for row in sorted(range(len(costs)), key=lambda i: costs[i]):
        widest = max(longest, costs[row])
        rows = len(open_rows) + 1
        if open_rows and (rows > batch_size or rows * ladder_width(widest, limit) > batch_tokens):
            chunks.append((open_rows, ladder_width(longest, limit)))
            open_rows, widest = [], costs[row]
        open_rows.append(row)
        longest = widest
    if open_rows:
        chunks.append((open_rows, ladder_width(longest, limit)))
    return chunks


def forward_chunks(costs, texts, args, limit: int):
    """The forwards of one serve as `(row indices, padded width or None for
    the batch's natural width)`. `plan` is the engine's own cut;
    `length-sorted` is `sentence-transformers`' `encode()`: longest text
    first, by character length, ties in input order, `batch_size` at a
    time."""
    if args.order == "plan":
        return plan_chunks(costs, args.batch_size, args.batch_tokens, limit)
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    return [(order[start : start + args.batch_size], None) for start in range(0, len(order), args.batch_size)]


class Encoder:
    """The loaded model, tokenizer and resolved pooling — built once per
    process, outside every timed serve, as the engine's model cache holds its
    loaded model across serves."""

    def __init__(self, args, device):
        import torch
        from tokenizers import Tokenizer
        from transformers import AutoModel

        self.args, self.device = args, device
        config = tfs.load_config_reference_compile_off(args.model_dir)
        self.max_sequence_length = int(config.max_position_embeddings)
        self.model = (
            AutoModel.from_pretrained(
                args.model_dir,
                config=config,
                attn_implementation=args.attn,
                torch_dtype={"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[args.dtype],
            )
            .to(device)
            .eval()
        )
        self.tokenizer = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.max_sequence_length)
        self.pooling, self.pooling_sha256 = resolve_pooling(args.model_dir)

    def encode(self, texts, width):
        """Tokenize one chunk, padded to `width` — the batch's longest row
        when `None`, as the engine pads a chunk to its ladder rung."""
        if width is None:
            self.tokenizer.enable_padding()
        else:
            self.tokenizer.enable_padding(length=width)
        return self.tokenizer.encode_batch(texts)

    def serve(self, corpus_path: str, out_path: str):
        """One serve: corpus file → persisted embeddings (Parquet, as a
        result table is; and the ANN graph with `--ann-index`). Returns
        `(keys, vectors in key order, per-row real token counts in key order,
        padded tokens forwarded)`."""
        import numpy as np
        import torch

        keys, texts = read_corpus(corpus_path)
        # Every row's cost — its truncated token count — is what the plan
        # orders and budgets its chunks by (`RowCostExec`), before any forward.
        self.tokenizer.no_padding()
        row_tokens = [len(e.ids) for e in self.tokenizer.encode_batch(texts)]
        chunks = forward_chunks(row_tokens, texts, self.args, self.max_sequence_length)
        vectors, padded_tokens = [None] * len(texts), 0
        with torch.inference_mode():
            for chunk, width in chunks:
                encodings = self.encode([texts[i] for i in chunk], width)
                input_ids = torch.tensor([e.ids for e in encodings], device=self.device)
                mask = torch.tensor([e.attention_mask for e in encodings], device=self.device)
                hidden = self.model(input_ids=input_ids, attention_mask=mask).last_hidden_state
                pooled = pool(hidden, mask, self.pooling)
                embedded = tfs.l2_normalize(pooled).to(torch.float32).cpu().numpy()
                padded_tokens += input_ids.numel()
                for i, vector in zip(chunk, embedded):
                    vectors[i] = vector
        stacked = np.stack(vectors)
        write_parquet(out_path, keys, stacked)
        if self.args.ann_index:
            write_ann_index(out_path + ".usearch", stacked)
        return keys, stacked, row_tokens, padded_tokens


def nearest_rank(sorted_values, p: float):
    return sorted_values[min(max(math.ceil(p * len(sorted_values)), 1), len(sorted_values)) - 1]


def measure_leg(args, rows: int, take: int) -> dict:
    """Measure one (unit, take) in THIS process; returns the leg block."""
    import torch

    device = torch.device("cpu" if args.cuda is None else f"cuda:{args.cuda}")
    is_cuda = device.type == "cuda"
    corpus_path = os.path.join(args.exchange_dir, f"corpus_{rows}.parquet")
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{leg_stem(rows, take)}.parquet")

    load_start = time.perf_counter()
    encoder = Encoder(args, device)
    if is_cuda:
        torch.cuda.synchronize()
    model_load_ms = (time.perf_counter() - load_start) * 1e3

    # Provenance only: torch's allocator, read with the model resident and
    # at its high-water mark over every serve after that.
    allocator_baseline = torch.cuda.memory_allocated() if is_cuda else None
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    def timed_serve():
        start = time.perf_counter()
        served = encoder.serve(corpus_path, out_path)
        if is_cuda:
            torch.cuda.synchronize()
        return time.perf_counter() - start, served

    first_serve_s, _ = timed_serve()
    # Every serve timed and filed in run order; where the run settled is the
    # ladder's to find.
    samples = [timed_serve() for _ in range(args.iters)]
    iter_wall_s = [s for s, _ in samples]
    keys, vectors, row_tokens, padded_tokens = samples[-1][1]
    first_vectors = samples[0][1][1]
    digest = outcome_digest(vectors)
    if outcome_digest(first_vectors) != digest:
        raise RuntimeError("the torch rung is not deterministic: its first and last serves differ")
    allocator_peak = torch.cuda.max_memory_allocated() if is_cuda else None

    sorted_s = sorted(iter_wall_s)
    p50_ms = nearest_rank(sorted_s, 0.50) * 1e3
    tokens = sum(row_tokens)
    sorted_tokens = sorted(row_tokens)
    vectors_file = None
    if take == 1 and args.legs_dir:
        os.makedirs(args.legs_dir, exist_ok=True)
        vectors_file = f"{leg_stem(rows, take)}.vectors.f32"
        write_vectors(os.path.join(args.legs_dir, vectors_file), vectors)
    checkpoint = tfs.checkpoint_identity(args.model_dir)
    return {
        "task": "embed",
        "seed": args.seed,
        "rows": rows,
        "corpus_sha256": sha256_file(corpus_path),
        "token_lengths_sha256": token_lengths_sha256(row_tokens),
        "tokens": tokens,
        "batch_size": args.batch_size,
        "batch_tokens": args.batch_tokens,
        "max_sequence_length": encoder.max_sequence_length,
        "compute_precision": args.dtype,
        **checkpoint,
        "checkpoint_tokenizer_sha256": sha256_file(os.path.join(args.model_dir, "tokenizer.json")),
        "pooling": encoder.pooling,
        "normalize": True,
        "iters_measured": len(iter_wall_s),
        "checkpoint_pooling_sha256": encoder.pooling_sha256,
        "device_requested": "cpu" if args.cuda is None else f"cuda:{args.cuda}",
        # Provenance: what this leg was, never compared leg to leg.
        "rung": RUNG,
        "take": take,
        "order": args.order,
        "ann_index": args.ann_index,
        "attn_requested": args.attn,
        "attn_implementation": getattr(encoder.model.config, "_attn_implementation", "absent"),
        "torch_num_threads": torch.get_num_threads(),
        "peak_vram_allocator_bytes": None if allocator_peak is None else allocator_peak - allocator_baseline,
        # Measured.
        "iter_wall_s": iter_wall_s,
        "work": rows,
        "peak_rss_bytes": {"value": tfs.peak_rss_bytes(), "unit": "bytes"},
        # Stamped by the parent from the external sampler, when it had one.
        "peak_vram_bytes": {"value": None, "unit": "bytes"},
        "outcome_digest": digest,
        "vectors_file": vectors_file,
        "vector_dim": None if vectors_file is None else int(vectors.shape[1]),
        "padded_tokens": padded_tokens,
        "row_tokens_p50": nearest_rank(sorted_tokens, 0.50),
        "row_tokens_max": sorted_tokens[-1],
        "model_load_ms": model_load_ms,
        "first_serve_ms": first_serve_s * 1e3,
        "serve_ms_p50": p50_ms,
        "serve_ms_min": sorted_s[0] * 1e3,
        "rows_per_s": rows / (p50_ms / 1e3),
        "tokens_per_s": tokens / (p50_ms / 1e3),
    }


def leg_stem(rows: int, take: int) -> str:
    """The ladder's leg name: `<rung>__rows<N>__r<take>`."""
    return f"{RUNG}__rows{rows}__r{take}"


def spawn_leg(args, argv, rows: int, take: int) -> dict:
    """Measure one (unit, take) in a child of this script — under
    `jammi-bench sample-device` when `--sampler-bin` names it, which then
    stamps the leg's `peak_vram_bytes`."""
    command = [sys.executable, os.path.abspath(__file__), *argv, "--leg", f"{rows}:{take}"]
    if args.sampler_bin:
        sampler = [args.sampler_bin, "sample-device"]
        if args.cuda is not None:
            sampler += ["--cuda", str(args.cuda)]
        done = subprocess.run([*sampler, "--", *command], stdout=subprocess.PIPE, text=True, check=False)
        if done.returncode != 0:
            raise RuntimeError(f"torch_encode.py leg {leg_stem(rows, take)} exited {done.returncode} under the sampler")
        sampled = json.loads(done.stdout)
        leg = json.loads(sampled["child_stdout"])
        leg["peak_vram_bytes"] = sampled["peak_vram_bytes"]
        return leg
    done = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"torch_encode.py leg {leg_stem(rows, take)} exited {done.returncode}")
    return json.loads(done.stdout)


def write_dry_run_corpus(exchange_dir: str, rows: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    texts = [" ".join(DRY_RUN_WORDS[(i + w) % len(DRY_RUN_WORDS)] for w in range(1 + (7 * i) % 23)) for i in range(rows)]
    table = pa.table({KEY_COLUMN: [f"row_{i:08}" for i in range(rows)], TEXT_COLUMN: texts})
    pq.write_table(table, os.path.join(exchange_dir, f"corpus_{rows}.parquet"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", help="checkpoint directory (config.json, model.safetensors, tokenizer.json)")
    parser.add_argument("--exchange-dir", help="where `encode-step --exchange-dir` left corpus_<rows>.parquet")
    parser.add_argument("--out-dir", help="where this leg persists torch_vectors_<rows>.parquet (default: the exchange dir)")
    parser.add_argument("--rows", default="16,256", help="the sweep's row counts, comma-separated")
    ll.add_take_argument(parser)
    parser.add_argument("--seed", type=int, default=0, help="the seed the jammi leg generated the corpus with")
    parser.add_argument("--legs-dir", help="where the legs are written (<rung>__rows<N>__r<take>.json)")
    parser.add_argument("--sampler-bin", help="a jammi-bench binary whose `sample-device` wraps each leg's process")
    parser.add_argument("--batch-size", type=int, default=32, help="`[inference] batch_size`: the chunk's row cap")
    parser.add_argument(
        "--batch-tokens", type=int, default=16384, help="`[inference] batch_tokens`: the chunk's padded-token cap"
    )
    parser.add_argument("--iters", type=int, default=32, help="serves timed and filed; the ladder settles no fewer than 32")
    parser.add_argument("--dtype", choices=DTYPES, default="f32")
    parser.add_argument("--attn", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--order", choices=("plan", "length-sorted"), default="plan")
    parser.add_argument(
        "--ann-index",
        action="store_true",
        help="also build and save the ANN graph the engine's sink builds (needs the usearch package)",
    )
    parser.add_argument("--cuda", type=int, default=None, help="CUDA ordinal; omit for CPU")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--leg", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.rows = [int(r) for r in args.rows.split(",")]
    if min(args.rows + [args.batch_size, args.batch_tokens, args.iters]) < 1:
        parser.error("--rows, --batch-size, --batch-tokens and --iters must be >= 1")
    if not args.dry_run and not (args.model_dir and args.exchange_dir):
        parser.error("--model-dir and --exchange-dir are required without --dry-run")
    return args


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    import torch

    if args.cuda is not None and not torch.cuda.is_available():
        # A leg asked for a device must never publish a CPU run under its name.
        print(f"--cuda {args.cuda} requested but torch sees no CUDA device", file=sys.stderr)
        return 2
    if args.leg is not None:
        rows, take = (int(part) for part in args.leg.split(":"))
        args.out_dir = args.out_dir or args.exchange_dir
        tfs.pin_fast_path_globals()
        print(json.dumps(measure_leg(args, rows, take)))
        return 0

    with tempfile.TemporaryDirectory() as scratch:
        if args.dry_run:
            argv += ["--model-dir", DRY_RUN_MODEL_DIR, "--exchange-dir", scratch, "--rows", ",".join(map(str, DRY_RUN_ROWS))]
            argv += ["--batch-size", "4", "--iters", "2"]
            args = parse_args(argv)
            for rows in args.rows:
                write_dry_run_corpus(scratch, rows)
        fast_path_globals = tfs.pin_fast_path_globals()
        provenance = tfs.provenance(torch.device("cpu" if args.cuda is None else f"cuda:{args.cuda}"), fast_path_globals)
        legs = []
        for rows in args.rows:
            for take in args.take:
                leg = spawn_leg(args, argv, rows, take)
                report = {"tool": "torch-encode", "dry_run": args.dry_run, "provenance": provenance, "encode_step": leg}
                if args.legs_dir:
                    os.makedirs(args.legs_dir, exist_ok=True)
                    with open(os.path.join(args.legs_dir, f"{leg_stem(rows, take)}.json"), "w") as fh:
                        json.dump(report, fh, indent=2)
                legs.append(report)
    json.dump({"tool": "torch-encode", "dry_run": args.dry_run, "provenance": provenance, "legs": legs}, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
