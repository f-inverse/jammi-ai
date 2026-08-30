#!/usr/bin/env python3
"""E1's width-evidence tool (CONTRACT `scratchpad/contract-356-profile.md`
v3/v4, "Width pinning" / Artifacts' "E1 width histogram produced OFFLINE by
the tracked producer re-tokenizing the fixture in the same anchors+positives
join and batch order"): re-tokenizes a committed held-out-fixture-shaped
pairs JSONL and reports, per row, the anchor/positive/negative wordpiece
lengths, plus the derived claim E1's width-pinning argument needs.

INPUT: a pairs JSONL in the committed `cookbook/fixtures/finetune_heldout/
heldout_pairs.jsonl` layout (`{"anchor_id","anchor_text","positive_id",
"positive_text","negative_id","negative_text"}` per line -- the same
schema `jammi-bench finetune-run --train-jsonl`/`--heldout-jsonl` consume),
a `tokenizer.json`, and `--cap W` (the leg's `--max-seq-length`).

DERIVED CLAIM ("width uniform at W" / "not provable"): E1's objective is
`Mnrl`, which trains over the (anchor, positive) PROJECTION only
(`crate::finetune_run::project_to_pairs` -- the negative column plays no
role in what a batch's WIDTH is, only anchor/positive do). A row is SHORT
iff NEITHER its anchor nor its positive text reaches the truncation cap
under `--cap W` (i.e. both truncate to fewer than `W` tokens, specials
included) -- a short row supplies no width-`W` entry to any batch it lands
in. Let `k` = the number of short rows out of `N` total. A batch's width
is the MAX over its rows' contributions -- so a batch of `r` rows can only
have width `< W` if EVERY one of its `r` rows is short, which needs at
least `r` DISTINCT short rows to exist in the whole corpus; since only `k`
short rows exist in total, `r > k` (equivalently `r >= k+1`) makes this
combinatorially IMPOSSIBLE, by pigeonhole, regardless of which rows a
particular batching scheme happens to group together. So:

  - `r_threshold = k + 1`: any batch of size `>= r_threshold` is
    GUARANTEED width exactly `W` -- "width uniform at W".
  - a batch of size `<= k` is NOT provably width-`W` this way (it COULD be
    entirely short rows) -- "not provable".

NO SHUFFLE REPLICATION: this max-over-rows pigeonhole argument is a
property of WHICH rows exist and how many are short, never of what ORDER
a particular batcher happens to group them in -- a batch of size `r > k`
is guaranteed width-`W` under ANY partition of the corpus into batches of
that size (random shuffle, `BatchLongest`, sequential, whatever), so this
tool never needs to replicate a specific batch order/shuffle to certify
the claim; it only needs `k` (a property of the row set) and the batch
size actually used (`--batch-size`).

Uses the `tokenizers` package; if it is not importable, this refuses
LOUDLY (exit nonzero) and never writes a report -- a width claim silently
computed some other (unstated, unverified) way would be worse than no
claim at all.

Usage: fixture_width_report.py PAIRS_JSONL --tokenizer TOKENIZER_JSON --cap W
       [--batch-size B] [--out REPORT_JSON]

Hermetic: reads only the two local files named; no network, no build, no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_BATCH_SIZE = 32


def _load_pairs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _truncated_len(tok, text: str, cap: int) -> int:
    """Token count of `text` AFTER truncating to `cap` (specials included)
    -- measures real tokenizer behavior directly (`enable_truncation`),
    never a hand-derived `raw_len + num_specials` arithmetic guess."""
    tok.enable_truncation(max_length=cap)
    return len(tok.encode(text, add_special_tokens=True).ids)


def _raw_len(tok, text: str) -> int:
    """Untruncated, no-specials token count -- the informational
    per-row wordpiece length this tool reports (not what the derived
    claim's short/reaches-cap test uses -- see `_truncated_len`)."""
    tok.no_truncation()
    return len(tok.encode(text, add_special_tokens=False).ids)


def per_row_lengths(pairs: list[dict], tok, cap: int) -> list[dict]:
    rows = []
    for p in pairs:
        anchor_raw = _raw_len(tok, p["anchor_text"])
        positive_raw = _raw_len(tok, p["positive_text"])
        negative_raw = _raw_len(tok, p["negative_text"])
        anchor_reaches_cap = _truncated_len(tok, p["anchor_text"], cap) == cap
        positive_reaches_cap = _truncated_len(tok, p["positive_text"], cap) == cap
        rows.append(
            {
                "anchor_id": p["anchor_id"],
                "anchor_wordpieces": anchor_raw,
                "positive_wordpieces": positive_raw,
                "negative_wordpieces": negative_raw,
                "reaches_cap": anchor_reaches_cap or positive_reaches_cap,
            }
        )
    return rows


def derive_claim(reaches_cap: list[bool], batch_size: int) -> dict:
    """Pure function over the per-row `reaches_cap` booleans (see module
    doc's pigeonhole argument) -- takes no tokenizer/file dependency, so
    the verdict logic itself is testable on a synthetic length table
    without `tokenizers` installed."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    n = len(reaches_cap)
    k = sum(1 for r in reaches_cap if not r)
    r_threshold = k + 1
    verdict = "width uniform at W" if batch_size >= r_threshold else "not provable"
    return {
        "n_rows": n,
        "k_short_rows": k,
        "r_threshold": r_threshold,
        "batch_size": batch_size,
        "verdict": verdict,
    }


def build_report(pairs: list[dict], tok, cap: int, batch_size: int) -> dict:
    rows = per_row_lengths(pairs, tok, cap)
    claim = derive_claim([r["reaches_cap"] for r in rows], batch_size)
    return {"cap": cap, "rows": rows, "claim": claim}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(
        prog="fixture_width_report.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s PAIRS_JSONL --tokenizer TOKENIZER_JSON --cap W "
        "[--batch-size B] [--out REPORT_JSON]",
    )
    ap.add_argument("pairs_jsonl", type=Path, help="committed heldout-fixture-shaped pairs JSONL")
    ap.add_argument("--tokenizer", type=Path, required=True, help="tokenizer.json path")
    ap.add_argument("--cap", type=int, required=True, help="the leg's --max-seq-length (W)")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            f"the leg's batch size in PAIRS (default: {DEFAULT_BATCH_SIZE}, "
            "E1's own committed value)"
        ),
    )
    ap.add_argument(
        "--out", type=Path, default=None, help="write the full report JSON here (optional)"
    )
    args = ap.parse_args(argv)

    try:
        from tokenizers import Tokenizer
    except ImportError as e:
        print(
            f"::error::fixture_width_report: the 'tokenizers' package is not importable "
            f"({e}) -- refusing (no report can be produced without a real tokenizer)",
            file=sys.stderr,
        )
        return 1

    if not args.pairs_jsonl.exists():
        print(f"::error::fixture_width_report: {args.pairs_jsonl} does not exist", file=sys.stderr)
        return 2
    if not args.tokenizer.exists():
        print(f"::error::fixture_width_report: {args.tokenizer} does not exist", file=sys.stderr)
        return 2

    pairs = _load_pairs(args.pairs_jsonl)
    if not pairs:
        print(f"::error::fixture_width_report: {args.pairs_jsonl} has zero rows", file=sys.stderr)
        return 2

    tok = Tokenizer.from_file(str(args.tokenizer))
    try:
        report = build_report(pairs, tok, args.cap, args.batch_size)
    except ValueError as e:
        print(f"::error::fixture_width_report: {e}", file=sys.stderr)
        return 2

    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=1))

    claim = report["claim"]
    print(
        f"fixture_width_report: n_rows={claim['n_rows']} k_short_rows={claim['k_short_rows']} "
        f"r_threshold={claim['r_threshold']} batch_size={claim['batch_size']} "
        f"verdict='{claim['verdict']}'"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
