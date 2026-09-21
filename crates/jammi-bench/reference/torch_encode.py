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
  through its Python binding — with the same batch-longest padding and the
  same truncation bound (`max_position_embeddings`, what the engine's loaded
  encoders report as `max_sequence_length`). `token_lengths_sha256` — the
  digest of every row's real token count — is on both reports, so "the same
  tokenization" is a checked fact.
* **Same pooling, same normalization.** `1_Pooling/config.json` is resolved by
  the rules of `backend/candle.rs`'s `pooling_from_config` (absent → mean; a
  present file must declare exactly one representable strategy), pooled by a
  port of `jammi-encoders`' `pooling.rs`, and L2-normalized by the floor
  `torch_finetune_step.py` already ports.
* **Same dtype, same batch size**, as flags; both are on both reports.
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

`--order corpus` forwards the rows in key order, `--batch-size` at a time —
the chunks the engine's plan forwards, hence the same padding. It is the
SEMANTIC twin. `--order length-sorted` forwards them longest-first and
restores key order afterwards — what `sentence-transformers`' `encode()` does
by default, and so the bar a user would actually hold the engine to: it pads
far less. `padded_tokens` on each point says how much less.

## Agreement

When the exchange directory holds the jammi leg's `vectors_<rows>.parquet`,
each point reports the cosine between the two stacks' embedding of every row
(`agreement`: min, mean, rows). A throughput ratio means nothing between two
stacks that embedded different things; this is the check that they did not.

## One process per sweep point

Like the jammi leg: `VmHWM` never falls, so each row count is measured in a
fresh child of this script and the parent folds the children's points and
fits `serve_ms = fixed_ms + per_row_ms · rows` over them.

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch_finetune_step as tfs  # noqa: E402

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

# The fields a jammi `encode-step` leg and this producer must agree on before
# their numbers are one comparison — `identity_fields.ENCODE_TWIN_IDENTITY_FIELDS`
# mirrors this tuple, and `test_identity_fields_subset.py` holds the two equal.
IDENTITY_FIELDS = (
    "rows",
    "batch_size",
    "corpus",
    "max_sequence_length",
    "compute_precision",
    "checkpoint_config_sha256",
    "checkpoint_weights_sha256",
    "checkpoint_weights_size_bytes",
    "checkpoint_tokenizer_sha256",
    "pooling",
    "normalize",
    "warmup",
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


def write_vectors(path: str, keys, vectors) -> None:
    """The shape `encode_step.rs::write_vectors` writes: `_row_id` beside a
    `vector` `FixedSizeList<Float32>`."""
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


def read_vectors(path: str):
    """`{key: vector}` of a vectors Parquet either producer wrote."""
    import numpy as np
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=[KEY_COLUMN, "vector"])
    vectors = np.asarray(table.column("vector").combine_chunks().flatten().to_numpy(zero_copy_only=False))
    keys = table.column(KEY_COLUMN).to_pylist()
    return dict(zip(keys, vectors.reshape(len(keys), -1)))


def agreement(jammi_path: str, keys, vectors):
    """Per-row cosine between the jammi leg's persisted vectors and this
    leg's, in float64. `None` when the exchange directory holds no jammi
    vectors for this point."""
    import numpy as np

    if not os.path.exists(jammi_path):
        return None
    theirs = read_vectors(jammi_path)
    if sorted(theirs) != sorted(keys):
        raise ValueError(f"{jammi_path} does not hold exactly the corpus's keys")
    a = np.stack([theirs[k] for k in keys]).astype(np.float64)
    b = vectors.astype(np.float64)
    cosine = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return {"rows": len(keys), "cosine_min": float(cosine.min()), "cosine_mean": float(cosine.mean())}


def forward_order(texts, order: str):
    """The row indices in forward order. `length-sorted` is
    `sentence-transformers`' `encode()`: longest text first, by character
    length, ties in input order."""
    if order == "corpus":
        return list(range(len(texts)))
    return sorted(range(len(texts)), key=lambda i: -len(texts[i]))


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
        self.tokenizer.enable_padding()
        self.pooling, self.pooling_sha256 = resolve_pooling(args.model_dir)

    def serve(self, corpus_path: str, out_path: str):
        """One serve: corpus file → persisted embeddings. Returns `(keys,
        vectors in key order, per-row real token counts in key order, padded
        tokens forwarded)`."""
        import numpy as np
        import torch

        keys, texts = read_corpus(corpus_path)
        order = forward_order(texts, self.args.order)
        vectors, row_tokens, padded_tokens = [None] * len(texts), [0] * len(texts), 0
        with torch.inference_mode():
            for start in range(0, len(order), self.args.batch_size):
                chunk = order[start : start + self.args.batch_size]
                encodings = self.tokenizer.encode_batch([texts[i] for i in chunk])
                input_ids = torch.tensor([e.ids for e in encodings], device=self.device)
                mask = torch.tensor([e.attention_mask for e in encodings], device=self.device)
                hidden = self.model(input_ids=input_ids, attention_mask=mask).last_hidden_state
                pooled = pool(hidden, mask, self.pooling)
                embedded = tfs.l2_normalize(pooled).to(torch.float32).cpu().numpy()
                padded_tokens += input_ids.numel()
                for i, vector, encoding in zip(chunk, embedded, encodings):
                    vectors[i], row_tokens[i] = vector, sum(encoding.attention_mask)
        stacked = np.stack(vectors)
        write_vectors(out_path, keys, stacked)
        if self.args.ann_index:
            write_ann_index(out_path + ".usearch", stacked)
        return keys, stacked, row_tokens, padded_tokens


def nearest_rank(sorted_values, p: float):
    return sorted_values[min(max(math.ceil(p * len(sorted_values)), 1), len(sorted_values)) - 1]


def cost_fit(points):
    """`timing.rs::CostFit::least_squares`: `serve_ms = fixed_ms + per_row_ms ·
    rows` by RELATIVE least squares (weights `1 / ms²`); `None` when the sweep
    cannot determine two terms."""
    if any(not (ms > 0.0 and math.isfinite(ms)) for _, ms in points):
        return None
    sw = swx = swxx = swy = swxy = 0.0
    for rows, ms in points:
        w = 1.0 / (ms * ms)
        sw, swx, swxx, swy, swxy = sw + w, swx + w * rows, swxx + w * rows * rows, swy + w * ms, swxy + w * rows * ms
    determinant = sw * swxx - swx * swx
    if not determinant > 1e-9 * sw * swxx:
        return None
    per_row_ms = (sw * swxy - swx * swy) / determinant
    fixed_ms = (swy - per_row_ms * swx) / sw
    residual = math.sqrt(sum(((fixed_ms + per_row_ms * r - ms) / ms) ** 2 for r, ms in points) / len(points))
    return {"fixed_ms": fixed_ms, "per_row_ms": per_row_ms, "relative_residual_rms": residual}


def measure_point(args, rows: int) -> dict:
    """Measure one row count in THIS process; returns a one-point
    `encode_step` block."""
    import torch

    device = torch.device("cpu" if args.cuda is None else f"cuda:{args.cuda}")
    is_cuda = device.type == "cuda"
    corpus_path = os.path.join(args.exchange_dir, f"corpus_{rows}.parquet")
    out_path = os.path.join(args.out_dir, f"torch_vectors_{rows}.parquet")

    load_start = time.perf_counter()
    encoder = Encoder(args, device)
    if is_cuda:
        torch.cuda.synchronize()
    model_load_ms = (time.perf_counter() - load_start) * 1e3

    # The README's VRAM convention: the baseline is what the allocator holds
    # with the model resident, the peak is its continuous high-water mark over
    # every serve after that.
    vram_baseline = torch.cuda.memory_allocated() if is_cuda else None
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    def timed_serve():
        start = time.perf_counter()
        served = encoder.serve(corpus_path, out_path)
        if is_cuda:
            torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1e3, served

    first_serve_ms, _ = timed_serve()
    for _ in range(args.warmup):
        timed_serve()
    samples = [timed_serve() for _ in range(args.iters)]
    serve_ms = sorted(ms for ms, _ in samples)
    keys, vectors, row_tokens, padded_tokens = samples[-1][1]
    vram_peak = torch.cuda.max_memory_allocated() if is_cuda else None

    p50 = nearest_rank(serve_ms, 0.50)
    tokens = sum(row_tokens)
    sorted_tokens = sorted(row_tokens)
    checkpoint = tfs.checkpoint_identity(args.model_dir)
    return {
        "rows": [rows],
        "batch_size": args.batch_size,
        "corpus": [
            {
                "rows": rows,
                "corpus_sha256": sha256_file(corpus_path),
                "token_lengths_sha256": token_lengths_sha256(row_tokens),
                "tokens": tokens,
            }
        ],
        "max_sequence_length": encoder.max_sequence_length,
        "compute_precision": args.dtype,
        **checkpoint,
        "checkpoint_tokenizer_sha256": sha256_file(os.path.join(args.model_dir, "tokenizer.json")),
        "pooling": encoder.pooling,
        "normalize": True,
        "warmup": args.warmup,
        "iters_measured": args.iters,
        "checkpoint_pooling_sha256": encoder.pooling_sha256,
        "device_requested": "cpu" if args.cuda is None else f"cuda:{args.cuda}",
        # Provenance: what this leg was, never compared leg to leg.
        "order": args.order,
        "ann_index": args.ann_index,
        "attn_requested": args.attn,
        "attn_implementation": getattr(encoder.model.config, "_attn_implementation", "absent"),
        "torch_num_threads": torch.get_num_threads(),
        "points": [
            {
                "rows": rows,
                "padded_tokens": padded_tokens,
                "row_tokens_p50": nearest_rank(sorted_tokens, 0.50),
                "row_tokens_max": sorted_tokens[-1],
                "model_load_ms": model_load_ms,
                "first_serve_ms": first_serve_ms,
                "serve_ms_p50": p50,
                "serve_ms_min": serve_ms[0],
                "rows_per_s": rows / (p50 / 1e3),
                "tokens_per_s": tokens / (p50 / 1e3),
                "peak_rss_bytes": tfs.peak_rss_bytes(),
                "peak_vram_baseline_bytes": vram_baseline,
                "peak_vram_absolute_bytes": vram_peak,
                "peak_vram_delta_bytes": None if vram_peak is None else vram_peak - vram_baseline,
                "agreement": agreement(os.path.join(args.exchange_dir, f"vectors_{rows}.parquet"), keys, vectors),
            }
        ],
        "fit_p50": None,
        "fit_min": None,
    }


PER_POINT_FIELDS = ("rows", "corpus", "points", "fit_p50", "fit_min")


def fold_sweep(children):
    """`encode_step.rs::fold_sweep`: the children's points in sweep order, the
    two fits over them, everything else — which they must agree on — once."""
    shared = lambda block: {k: v for k, v in block.items() if k not in PER_POINT_FIELDS}  # noqa: E731
    tier = children[0]
    for child in children[1:]:
        if shared(child) != shared(tier):
            raise ValueError(f"sweep points disagree on what they measured: {shared(child)} vs {shared(tier)}")
        for field in ("rows", "corpus", "points"):
            tier[field] = tier[field] + child[field]
    tier["fit_p50"] = cost_fit([(p["rows"], p["serve_ms_p50"]) for p in tier["points"]])
    tier["fit_min"] = cost_fit([(p["rows"], p["serve_ms_min"]) for p in tier["points"]])
    return tier


def spawn_point(argv, rows: int) -> dict:
    done = subprocess.run(
        [sys.executable, os.path.abspath(__file__), *argv, "--point", str(rows)],
        stdout=subprocess.PIPE,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise RuntimeError(f"torch_encode.py child ({rows} rows) exited {done.returncode}")
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--dtype", choices=DTYPES, default="f32")
    parser.add_argument("--attn", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--order", choices=("corpus", "length-sorted"), default="corpus")
    parser.add_argument(
        "--ann-index",
        action="store_true",
        help="also build and save the ANN graph the engine's sink builds (needs the usearch package)",
    )
    parser.add_argument("--cuda", type=int, default=None, help="CUDA ordinal; omit for CPU")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--point", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.rows = [int(r) for r in args.rows.split(",")]
    if min(args.rows + [args.batch_size, args.iters]) < 1 or args.warmup < 0:
        parser.error("--rows, --batch-size and --iters must be >= 1 and --warmup >= 0")
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
    if args.point is not None:
        args.out_dir = args.out_dir or args.exchange_dir
        tfs.pin_fast_path_globals()
        print(json.dumps(measure_point(args, args.point)))
        return 0

    with tempfile.TemporaryDirectory() as scratch:
        if args.dry_run:
            argv += ["--model-dir", DRY_RUN_MODEL_DIR, "--exchange-dir", scratch, "--rows", ",".join(map(str, DRY_RUN_ROWS))]
            argv += ["--batch-size", "4", "--warmup", "0", "--iters", "2"]
            args = parse_args(argv)
            for rows in args.rows:
                write_dry_run_corpus(scratch, rows)
        fast_path_globals = tfs.pin_fast_path_globals()
        tier = fold_sweep([spawn_point(argv, rows) for rows in args.rows])
    device = torch.device("cpu" if args.cuda is None else f"cuda:{args.cuda}")
    report = {
        "tool": "torch-encode",
        "dry_run": args.dry_run,
        "provenance": tfs.provenance(device, fast_path_globals),
        "encode_step": tier,
    }
    json.dump(report, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
