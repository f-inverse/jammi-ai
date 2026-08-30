#!/usr/bin/env python3
"""Deterministic, seeded generator for a synthetic MNRL train-pairs JSONL
in the EXACT schema `jammi-bench finetune-run --train-jsonl` consumes
(`crates/jammi-bench/src/main.rs::TripletRow`/`load_train_jsonl`, pinned by
reading that source directly, never guessed): one JSON object per line,

    {"anchor_id", "anchor_text", "positive_id", "positive_text",
     "negative_id", "negative_text"}

the same field names the committed `cookbook/fixtures/finetune_heldout/
heldout_pairs.jsonl` fixture uses, so this generator's output is a drop-in
`--train-jsonl` (`Objective::Mnrl` projects to the (anchor, positive) pair
and drops `negative_*`; this generator still emits `negative_*` so the file
is valid for `Objective::Triplet` too).

WIDTH GUARANTEE (CONTRACT `scratchpad/contract-356-profile.md` v3,
"Width pinning"): every leg using this generator needs every text to
tokenize to MORE than `--min-wordpieces W` wordpieces, so that truncating
at `--max-seq-length W` makes EVERY row's contribution to a batch exactly
`W` wide (the contract's own width-pinning argument) -- a batch can only
be narrower than `W` if EVERY row in it independently failed to reach the
cap, which this generator makes false for every row it emits.

Construction and its guarantee: a small fixed vocabulary (`_VOCAB`) of
common, short, lowercase English function/content words, each assumed (or,
with `--verify-tokenizer`, MECHANICALLY CHECKED) to tokenize to exactly one
wordpiece under a standard BERT-style wordpiece tokenizer. A BERT-style
pipeline pre-splits on whitespace/punctuation (`BasicTokenizer`-equivalent)
BEFORE wordpiece matching -- it never merges two space-separated words into
one token -- so a text built by joining `k` such words with single spaces
tokenizes to EXACTLY `k` wordpieces (before any `[CLS]`/`[SEP]` specials).
This generator draws `--min-wordpieces + _BUFFER` words per text (`_BUFFER`
extra words absorb any single edge-case word that turns out not to be a
lone wordpiece in a particular real vocab), so every emitted text's raw
wordpiece count is `>= min_wordpieces + _BUFFER`, comfortably `>
min_wordpieces`. `--verify-tokenizer TOKENIZER_JSON` turns the "assumed"
half of this guarantee into a mechanically checked one (every `_VOCAB` word
re-verified to encode to exactly one id, `add_special_tokens=False`) using
the `tokenizers` package; omitted by default so this generator has NO
network and NO required third-party dependency for its core job.

Determinism: one `random.Random(seed)` instance draws every word for every
row/role in a single, fixed sequential order -- the SAME `(rows,
min_wordpieces, seed)` triple always produces byte-identical output, and
(as a consequence, not a separately-tested guarantee) a smaller `--rows`
run's output is always a literal line-prefix of a larger `--rows` run's
output at the same `(min_wordpieces, seed)`, since later rows' draws never
affect earlier ones.

Usage: gen_fixed_width_corpus.py --rows N --min-wordpieces W --seed S --out PATH
       [--verify-tokenizer TOKENIZER_JSON]

Hermetic: no network. `--verify-tokenizer` reads only the local file named.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Common, short, lowercase English words -- overwhelmingly likely to be
# single whole-word entries in any BERT-family wordpiece vocab of
# meaningful size (they are among the highest-frequency tokens any
# wordpiece trainer sees). Deliberately NOT proper nouns, numerals, or
# anything with punctuation/hyphenation, which are the classes most likely
# to fragment into multiple wordpieces.
_VOCAB = [
    "the", "of", "and", "a", "to", "in", "is", "was", "for", "on",
    "with", "as", "at", "by", "from", "this", "that", "be", "are", "or",
    "an", "but", "not", "have", "has", "will", "can", "one", "two", "new",
    "also", "his", "her", "its", "their", "which", "when", "how", "who",
    "than", "then", "into", "over", "under", "after", "before", "each",
    "some", "any", "all", "most", "other", "such", "no", "only", "same",
    "so", "than", "too", "very", "just", "even", "back", "still", "way",
    "well", "also", "good", "large", "small", "high", "low", "long",
    "many", "much", "used", "made", "used", "world", "system", "data",
    "model", "paper", "study", "result", "method", "approach", "network",
]

# Extra words beyond `min_wordpieces` drawn per text -- absorbs any single
# `_VOCAB` entry that (in a real, unverified deployment) turns out not to
# be a lone wordpiece, without needing the exact count to be provably
# tight; see module doc's "Construction and its guarantee".
_BUFFER = 3


def _row_id(seed: int, idx: int, role: str) -> str:
    return f"synthetic-{seed}-{idx:06d}-{role}"


def _row_text(rng: random.Random, min_wordpieces: int) -> str:
    k = min_wordpieces + _BUFFER
    return " ".join(rng.choices(_VOCAB, k=k))


def generate_rows(rows: int, min_wordpieces: int, seed: int) -> list[dict]:
    if rows <= 0:
        raise ValueError(f"--rows must be positive, got {rows}")
    if min_wordpieces < 0:
        raise ValueError(f"--min-wordpieces must be non-negative, got {min_wordpieces}")
    rng = random.Random(seed)
    out = []
    for i in range(rows):
        out.append(
            {
                "anchor_id": _row_id(seed, i, "a"),
                "anchor_text": _row_text(rng, min_wordpieces),
                "positive_id": _row_id(seed, i, "p"),
                "positive_text": _row_text(rng, min_wordpieces),
                "negative_id": _row_id(seed, i, "n"),
                "negative_text": _row_text(rng, min_wordpieces),
            }
        )
    return out


def write_jsonl(rows: list[dict], out_path: Path) -> None:
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def verify_vocab_is_single_wordpiece(tokenizer_json: Path) -> list[str]:
    """Returns a list of `_VOCAB` words that do NOT encode to exactly one
    id under `tokenizer_json` (`add_special_tokens=False`) -- empty means
    verified. Raises `ImportError` (loudly, uncaught by design) if the
    `tokenizers` package is not importable -- this is an opt-in check, so a
    caller who requests it but lacks the dependency must see a real error,
    never a silent skip."""
    from tokenizers import Tokenizer  # noqa: PLC0415 -- optional dependency, imported lazily

    tok = Tokenizer.from_file(str(tokenizer_json))
    bad = []
    for word in _VOCAB:
        ids = tok.encode(word, add_special_tokens=False).ids
        if len(ids) != 1:
            bad.append(f"{word!r} -> {len(ids)} wordpieces (expected 1)")
    return bad


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(
        prog="gen_fixed_width_corpus.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s --rows N --min-wordpieces W --seed S --out PATH "
        "[--verify-tokenizer TOKENIZER_JSON]",
    )
    ap.add_argument(
        "--rows", type=int, required=True, help="number of (anchor,positive,negative) rows"
    )
    ap.add_argument(
        "--min-wordpieces",
        type=int,
        required=True,
        help="every emitted text's raw wordpiece count exceeds this (see module doc's guarantee)",
    )
    ap.add_argument("--seed", type=int, required=True, help="deterministic RNG seed")
    ap.add_argument("--out", type=Path, required=True, help="output JSONL path")
    ap.add_argument(
        "--verify-tokenizer",
        type=Path,
        default=None,
        help="optional tokenizer.json to mechanically verify the vocab's single-wordpiece "
        "assumption (requires the 'tokenizers' package; refuses loudly if unimportable)",
    )
    args = ap.parse_args(argv)

    if args.verify_tokenizer is not None:
        bad = verify_vocab_is_single_wordpiece(args.verify_tokenizer)
        if bad:
            print(
                "::error::gen_fixed_width_corpus: --verify-tokenizer found vocab word(s) that "
                "are NOT single wordpieces under " + str(args.verify_tokenizer) + ":\n  "
                + "\n  ".join(bad),
                file=sys.stderr,
            )
            return 1
        print(f"gen_fixed_width_corpus: --verify-tokenizer OK -- all {len(_VOCAB)} vocab words "
              f"are single wordpieces under {args.verify_tokenizer}.")

    try:
        rows = generate_rows(args.rows, args.min_wordpieces, args.seed)
    except ValueError as e:
        print(f"::error::gen_fixed_width_corpus: {e}", file=sys.stderr)
        return 2

    write_jsonl(rows, args.out)
    print(
        f"gen_fixed_width_corpus: wrote {len(rows)} rows to {args.out} "
        f"(min_wordpieces={args.min_wordpieces}, seed={args.seed}, "
        f"words_per_text={args.min_wordpieces + _BUFFER})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
