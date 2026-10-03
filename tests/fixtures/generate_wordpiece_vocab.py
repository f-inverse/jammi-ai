#!/usr/bin/env python3
"""Generate the WordPiece-vocabulary tokenizer fixture and its oracle.

Many BERT-family checkpoints on the Hugging Face Hub ship `vocab.txt` and
`tokenizer_config.json` but no converted `tokenizer.json`; `transformers`
builds the tokenizer from those two files at load time. This script writes two
such checkpoints' tokenizer files — the layouts the Hub's most-used cased and
uncased checkpoints ship — and records the ids `transformers` encodes a set of
probe texts to from exactly those files:

  wordpiece_vocab/cased/    vocab.txt, tokenizer_config.json {"do_lower_case": false}
  wordpiece_vocab/uncased/  vocab.txt, tokenizer_config.json {"do_lower_case": true}
  wordpiece_vocab/expected_ids.json   {variant: [{"text", "ids"}, ...]}

The probes cover what the BERT pipeline does beyond a whitespace split: casing,
accent stripping, CJK characters split one per token, punctuation split off,
control characters dropped, subword continuations, an out-of-vocabulary word,
and special tokens written in the text, which stay whole.

Requires `transformers` (no PyTorch). Deterministic: the same files every run.
"""

import json
import os
import tempfile

from transformers import AutoTokenizer

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wordpiece_vocab")

SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
PUNCTUATION = list(".,!?;:'\"()-/")
WORDS = [
    "the", "a", "of", "and", "in", "to", "is", "was", "for", "on",
    "model", "search", "query", "embed", "token", "engine", "data",
    "cafe", "naive", "resume", "joined", "acme", "corp", "spring",
    "quantum", "computing", "works", "how", "does", "battery", "died",
]
CONTINUATIONS = ["##s", "##ing", "##ed", "##er", "##ly", "##ion"]
CJK = ["中", "文", "字"]
CASED_ONLY = ["The", "Acme", "Corp", "Alice", "Café", "naïve", "ﬁ"]

PROBES = [
    "How does quantum computing work?",
    "Alice joined Acme Corp last spring.",
    "The CAFÉ was naïve; the résumé works.",
    "中文字 search",
    "embedding tokens, queries\tand engines",
    "control\x00char​here",
    "an unseenword appears",
    "the [MASK] token [SEP] stays whole",
    "",
]


def vocab(cased: bool) -> list[str]:
    letters = [chr(c) for c in range(ord("a"), ord("z") + 1)]
    if cased:
        letters += [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    pieces = SPECIAL + PUNCTUATION + letters + [f"##{c}" for c in letters]
    pieces += WORDS + CONTINUATIONS + CJK + (CASED_ONLY if cased else [])
    # One entry per token, in first-seen order: a duplicate line would give the
    # token the later index in both `transformers` and the engine, but the
    # fixture has no reason to exercise that.
    return list(dict.fromkeys(pieces))


def write_variant(name: str, cased: bool, config: dict) -> list[dict]:
    target = os.path.join(OUT, name)
    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, "vocab.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(vocab(cased)) + "\n")
    with open(os.path.join(target, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f)
        f.write("\n")

    # Load from a copy holding ONLY the two files a vocab-only checkpoint ships,
    # so `transformers` converts them exactly as it does for such a Hub repo.
    with tempfile.TemporaryDirectory() as scratch:
        for file in ("vocab.txt", "tokenizer_config.json"):
            with open(os.path.join(target, file), encoding="utf-8") as src:
                with open(os.path.join(scratch, file), "w", encoding="utf-8") as dst:
                    dst.write(src.read())
        with open(os.path.join(scratch, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"model_type": "bert"}, f)
        tokenizer = AutoTokenizer.from_pretrained(scratch)
        return [{"text": text, "ids": tokenizer(text)["input_ids"]} for text in PROBES]


def main() -> None:
    expected = {
        "cased": write_variant("cased", True, {"do_lower_case": False, "max_len": 512}),
        "uncased": write_variant("uncased", False, {"model_max_length": 512, "do_lower_case": True}),
    }
    with open(os.path.join(OUT, "expected_ids.json"), "w", encoding="utf-8") as f:
        json.dump(expected, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
