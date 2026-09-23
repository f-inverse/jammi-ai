#!/usr/bin/env python3
"""The files a checkpoint directory must hold before any leg loads it.

    checkpoint_files.py DIR    exit 0 when DIR is a whole checkpoint, otherwise
                               exit 1 naming every file that is missing, empty
                               or truncated
    checkpoint_files.py --fetch REPO DIR
                               the same, after downloading the Hugging Face
                               repository REPO into DIR when DIR is not yet
                               whole (three attempts; run under an interpreter
                               with `huggingface_hub`, as the torch venv is)

Both stacks load `config.json` and `model.safetensors`; the trainer and the
serving path also tokenize, so `tokenizer.json` is as much a part of the
checkpoint as the weights. A directory with the weights and no tokenizer
fails a leg minutes in, after the build and the reference venv were paid for;
this names it first. The weights file is checked against its own header — the
eight-byte length prefix, the JSON header it announces, and the furthest
`data_offsets` end — so an interrupted download is a finding, not a load
error inside a measured leg.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

REQUIRED = ("config.json", "model.safetensors", "tokenizer.json")


def safetensors_defect(path: Path) -> str | None:
    size = path.stat().st_size
    with path.open("rb") as fh:
        prefix = fh.read(8)
        if len(prefix) < 8:
            return f"{size} bytes: shorter than a safetensors length prefix"
        (header_len,) = struct.unpack("<Q", prefix)
        if 8 + header_len > size:
            return f"announces a {header_len}-byte header in a {size}-byte file"
        try:
            header = json.loads(fh.read(header_len))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return f"its header is not JSON ({exc})"
    ends = [t["data_offsets"][1] for name, t in header.items() if name != "__metadata__"]
    expected = 8 + header_len + max(ends, default=0)
    if expected != size:
        return f"is {size} bytes; its header describes {expected} (an interrupted download?)"
    return None


def defects(directory: Path) -> list[str]:
    found = []
    for name in REQUIRED:
        path = directory / name
        if not path.is_file():
            found.append(f"{name}: missing")
        elif path.stat().st_size == 0:
            found.append(f"{name}: empty")
        elif name.endswith(".safetensors") and (why := safetensors_defect(path)):
            found.append(f"{name}: {why}")
        elif name.endswith(".json"):
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                found.append(f"{name}: not JSON ({exc})")
    return found


def _snapshot_download(repo: str, directory: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo, local_dir=str(directory))


def fetch(repo: str, directory: Path, download=_snapshot_download, attempts: int = 3) -> list[str]:
    """Download `repo` into `directory` until it is a whole checkpoint, at most
    `attempts` times; a directory already whole is left untouched. Returns
    what the directory still lacks — empty once it is whole."""
    found = defects(directory)
    for _ in range(attempts if found else 0):
        directory.mkdir(parents=True, exist_ok=True)
        try:
            download(repo, directory)
        except Exception as exc:  # a failed attempt is retried, and named if it is the last
            found = [f"download of {repo} failed: {exc}"]
            continue
        if not (found := defects(directory)):
            break
    return found


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 3 and args[0] == "--fetch":
        directory = Path(args[2])
        found = fetch(args[1], directory)
    elif len(args) == 1:
        directory = Path(args[0])
        found = defects(directory)
    else:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    if found:
        print(f"{directory} is not a whole checkpoint:", file=sys.stderr)
        for line in found:
            print(f"  {line}", file=sys.stderr)
        sys.exit(1)
