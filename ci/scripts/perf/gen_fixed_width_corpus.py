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

BPE VOCABULARIES (issue #421: the CLIP-text tower): the WIDTH GUARANTEE
above survives a byte-level BPE tokenizer unchanged, for the same reason it
survives wordpiece -- a BPE merge never spans the whitespace boundary
between two words, so `k` space-separated words tokenize to AT LEAST `k`
tokens under CLIP BPE, and `--min-wordpieces 77` therefore pins every text
to `>= 77` CLIP tokens exactly as it pins `>= 77` wordpieces. The bound is
one-sided (a BPE vocab may split one of these words into several tokens,
which only makes a text WIDER), which is all the guarantee needs: the claim
is "no row falls short of the cap", never "every row is exactly `k`".

`--verify-tokenizer` is NOT applicable to a BPE vocab and must not be run
against one: it checks that every `_VOCAB` word encodes to EXACTLY ONE id,
which is a BERT-wordpiece-vocabulary property, and a CLIP tokenizer.json
would fail it on words it merely splits differently -- a refusal that says
nothing about the width guarantee. Pass it only with a wordpiece
`tokenizer.json`.

HELD-OUT SPLIT (issue #421 P1-b(iv)): `--heldout-rows N` additionally emits
`heldout_ids.txt` (TAB-separated `anchor_id\tpositive_id\tnegative_id`, the
SCORING ORDER `finetune-run --heldout-ids` reads) and
`heldout_triplets.jsonl` (the same row schema as `--out`) in `--out`'s own
DIRECTORY, under the SAME two names the media producers emit -- so one
driver handles all three modalities with one pair of paths.

The held-out rows are disjoint from the train rows BY CONSTRUCTION, not by
luck: they are drawn as rows `[rows, rows + heldout_rows)` of the SAME
single RNG stream, so their ids (`synthetic-{seed}-{index}-{role}`, indexed
past every train row) cannot collide and their texts are independent draws.
Because the stream is consumed in row order, the train rows are byte-
identical to what `--rows N` alone emits at the same `(min_wordpieces,
seed)` -- adding a held-out split cannot perturb the train corpus. Every
held-out text carries the SAME width guarantee (it is built by the same
`_row_text`). `--heldout-batch B` is required alongside `--heldout-rows`
and refused unless it divides the held-out row count exactly, mirroring
`finetune-run`'s own "held-out rows must be a nonzero multiple of --batch"
refusal.

Usage: gen_fixed_width_corpus.py --rows N --min-wordpieces W --seed S --out PATH
       [--verify-tokenizer TOKENIZER_JSON]
       [--heldout-rows N --heldout-batch B]

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

# Extra words beyond `min_wordpieces` drawn per text. A `_VOCAB` entry that
# turns out NOT to be a lone wordpiece under a real, unverified tokenizer
# only ever HELPS this guarantee, never hurts it: a BERT-style pipeline
# pre-splits on whitespace/punctuation BEFORE wordpiece matching, so every
# whitespace-separated word independently produces AT LEAST one wordpiece
# (worst case: a single `[UNK]`) -- a word that fragments into two or more
# subword pieces only makes the true count HIGHER than the word count,
# never lower. This buffer is therefore NOT protecting against that
# failure mode (mathematically, `_BUFFER = 1` would already be sufficient
# for `> min_wordpieces` to hold even under worst-case fragmentation). It
# exists purely as a small, cheap defensive margin -- its exact size is
# not load-bearing -- against any OTHER irregularity in a caller-supplied
# vocab entry (e.g. one that normalizes/strips away to nothing under a
# particular tokenizer's own pre-processing); see module doc's
# "Construction and its guarantee".
_BUFFER = 3


def _row_id(seed: int, idx: int, role: str) -> str:
    return f"synthetic-{seed}-{idx:06d}-{role}"


def _row_text(rng: random.Random, min_wordpieces: int) -> str:
    k = min_wordpieces + _BUFFER
    return " ".join(rng.choices(_VOCAB, k=k))


# The names the held-out split is written under, beside `--out` -- the SAME
# two names `gen_fixed_shape_image_corpus.py`/`gen_fixed_length_audio_corpus.py`
# emit, so a driver handles all three modalities with one pair of paths.
_HELDOUT_IDS_NAME = "heldout_ids.txt"
_HELDOUT_JSONL_NAME = "heldout_triplets.jsonl"


def generate_rows(rows: int, min_wordpieces: int, seed: int) -> list[dict]:
    """The train rows alone — [`generate_split`]'s first element, the exact
    shape and values this function returned before `--heldout-rows` existed.
    Kept as the single-split entry point so every existing caller is
    untouched."""
    return generate_split(rows, min_wordpieces, seed)[0]


def generate_split(
    rows: int, min_wordpieces: int, seed: int, heldout_rows: int = 0
) -> tuple[list[dict], list[dict]]:
    """`(train_rows, heldout_rows_list)` — `heldout_rows_list` is EMPTY
    unless `heldout_rows > 0`.

    ONE `random.Random(seed)` stream draws `rows + heldout_rows` rows in
    index order and the split is a slice of it, so (a) the train rows are
    byte-identical to a `heldout_rows == 0` run at the same
    `(rows, min_wordpieces, seed)` — a held-out request cannot perturb the
    train corpus — and (b) the two halves are disjoint by CONSTRUCTION
    (distinct row indices ⇒ distinct ids, independent draws ⇒ independent
    texts), never by a seed-difference argument.
    """
    if rows <= 0:
        raise ValueError(f"--rows must be positive, got {rows}")
    if min_wordpieces < 0:
        raise ValueError(f"--min-wordpieces must be non-negative, got {min_wordpieces}")
    if heldout_rows < 0:
        raise ValueError(f"--heldout-rows must be non-negative, got {heldout_rows}")
    rng = random.Random(seed)
    out = []
    for i in range(rows + heldout_rows):
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
    return out[:rows], out[rows:]


def validate_heldout_split(heldout_rows: int, heldout_batch: int | None) -> None:
    """Refuse a held-out request `finetune-run` itself would refuse, before
    a single row is written."""
    if heldout_rows <= 0:
        raise ValueError(f"--heldout-rows must be positive when stated, got {heldout_rows}")
    if heldout_batch is None:
        raise ValueError(
            "--heldout-batch is required alongside --heldout-rows: `finetune-run` refuses a "
            "held-out fixture whose row count is not a nonzero multiple of --batch, and this "
            "producer cannot check that without being told the divisor"
        )
    if heldout_batch <= 0:
        raise ValueError(f"--heldout-batch must be positive, got {heldout_batch}")
    if heldout_rows % heldout_batch != 0:
        raise ValueError(
            f"--heldout-rows {heldout_rows} is not a multiple of --heldout-batch "
            f"{heldout_batch} (finetune-run refuses a held-out fixture that is not a nonzero "
            f"multiple of --batch)"
        )


def write_heldout(heldout_rows: list[dict], out_dir: Path) -> tuple[Path, Path]:
    """Write `heldout_ids.txt` + `heldout_triplets.jsonl` under `out_dir`,
    in the SAME row order, and return `(ids_path, jsonl_path)`. The ids file
    is the SCORING ORDER; the JSONL is joined to it BY `anchor_id`."""
    ids_path = out_dir / _HELDOUT_IDS_NAME
    with ids_path.open("w") as f:
        for row in heldout_rows:
            f.write(f"{row['anchor_id']}\t{row['positive_id']}\t{row['negative_id']}\n")
    jsonl_path = out_dir / _HELDOUT_JSONL_NAME
    write_jsonl(heldout_rows, jsonl_path)
    return ids_path, jsonl_path


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
        "--heldout-rows",
        type=int,
        default=0,
        help="also emit a disjoint held-out split of this many rows (heldout_ids.txt + "
        "heldout_triplets.jsonl beside --out); 0 (the default) emits nothing extra and "
        "leaves every byte of the train corpus unchanged",
    )
    ap.add_argument(
        "--heldout-batch",
        type=int,
        default=None,
        help="required alongside --heldout-rows: the --batch the consuming finetune-run leg "
        "will use, which the held-out row count must be a nonzero multiple of",
    )
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
        if args.heldout_rows:
            validate_heldout_split(args.heldout_rows, args.heldout_batch)
        rows, heldout_rows = generate_split(
            args.rows, args.min_wordpieces, args.seed, args.heldout_rows
        )
    except ValueError as e:
        print(f"::error::gen_fixed_width_corpus: {e}", file=sys.stderr)
        return 2

    write_jsonl(rows, args.out)
    if heldout_rows:
        ids_path, jsonl_path = write_heldout(heldout_rows, args.out.parent)
        print(
            f"gen_fixed_width_corpus: wrote {len(heldout_rows)} held-out rows to {ids_path} + "
            f"{jsonl_path} (heldout_batch={args.heldout_batch})"
        )
    print(
        f"gen_fixed_width_corpus: wrote {len(rows)} rows to {args.out} "
        f"(min_wordpieces={args.min_wordpieces}, seed={args.seed}, "
        f"words_per_text={args.min_wordpieces + _BUFFER})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
