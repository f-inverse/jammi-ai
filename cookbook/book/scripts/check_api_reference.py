#!/usr/bin/env python3
"""Guard the grounded API reference against the installed `jammi_ai` wheel.

The embedded reference (`jammi_cookbook/_api_reference.md`) is only trustworthy if
it tracks the pinned engine. This asserts that every method the chapters call
exists on the installed `jammi_ai` surface and still carries the keyword
arguments the recipes pass. If the pin moves and a signature drifts, CI fails
here — loudly — rather than a chapter calling a stale kwarg at execute time.

Run: `python scripts/check_api_reference.py`
"""

from __future__ import annotations

import inspect
import sys
import tempfile

import jammi_ai

# method -> kwargs the cookbook relies on existing in the signature.
REQUIRED: dict[str, list[str]] = {
    # setup / sources
    "add_source": ["url", "format"],
    "list_sources": [],
    "get_server_info": [],
    "set_tenant": [],
    "tenant_scope": [],
    "tenant": [],
    "generate_embeddings": ["source", "model", "columns", "key", "cache"],
    "encode_query": ["model", "query"],
    "sql": [],
    "rrf_fuse": [],
    # tier 01
    "build_neighbor_graph": ["k", "exact", "cache"],
    # tier 02
    "search": ["query", "k"],
    "assemble_context": ["query", "k", "edge_source", "edge_direction", "edge_hops"],
    "propagate_embeddings": [
        "embedding_table",
        "edge_source",
        "edge_src_column",
        "edge_dst_column",
        "direction",
        "hops",
        "weighting",
        "alpha",
        "output",
        "cache",
    ],
    # tier 03
    "fine_tune": [
        "source", "base_model", "columns", "method", "task", "embedding_loss",
        "mnrl_temperature", "mine_hard_negatives", "hard_negative_k",
        "hard_negative_exclude_hops", "hard_negative_refresh_every", "matryoshka_dims",
    ],
    "fine_tune_graph": [
        "node_source",
        "id_column",
        "text_column",
        "edge_source",
        "src_column",
        "dst_column",
        "base_model",
        "edge_provenance",
    ],
    "eval_embeddings": ["source", "golden_source"],
    "eval_compare": ["embedding_tables", "source", "golden_source"],
    "eval_inference": ["model", "source", "columns", "task", "golden_source", "label_column"],
    "eval_per_query": [],
    # model catalog (control plane)
    "list_models": [],
    "describe_model": ["model_id"],
    "delete_model": ["model_id", "version", "if_exists"],
    # data-plane inference (chapter 14 reference)
    "infer": ["source", "model", "columns", "task", "key", "cache"],
    # tier H4 — point-in-time temporal join + materialization / incremental-recompute
    "asof_join": [
        "spine_by",
        "spine_time",
        "facts_by",
        "facts_time",
        "direction",
        "boundary",
        "tolerance_duration_micros",
        "tie_break_column",
        "project",
    ],
    "verify_materialization": ["expected_definition"],
    "staleness": ["current_definition"],
    "derives_from": [],
    "recompute": ["cascade"],
    # tier 04
    "train_context_predictor": [
        "key_column",
        "task_column",
        "value_column",
        "architecture",
        "output",
        "objective",
        "seed",
    ],
    "predict_with_context_predictor": [
        "source",
        "target_key",
        "edge_source",
        "edge_src_column",
        "edge_dst_column",
        "edge_hops",
        "edge_direction",
    ],
    "conformalize": ["alpha", "score"],
    "conformalize_interval": ["alpha"],
    "conformalize_cqr": ["alpha"],
    "eval_calibration": ["source", "golden_source", "shape"],
    # provenance channels
    "register_channel": ["priority", "columns"],
    "add_channel_columns": ["columns"],
    "list_channels": [],
    # mutable companion tables
    "create_mutable_table": ["schema", "primary_key", "indexes", "order_column", "chunk_size"],
    "drop_mutable_table": ["if_exists"],
    "list_mutable_tables": [],
    # trigger stream / topics
    "register_topic": ["schema", "broker_metadata"],
    "drop_topic": ["if_exists"],
    "list_topics": [],
    "publish_topic": ["batch"],
    "subscribe_collect": ["predicate", "from_offset", "max_batches"],
}

MODULE_FUNCTIONS = ["open_local", "connect"]


def _signature(db: object, name: str) -> inspect.Signature | None:
    """Resolve `name`'s signature the way a CALLER actually invokes it — through
    the composed embedded surface, on an instance.

    `jammi_ai.connect("file://…")` returns a thin `Database` wrapper that holds a
    compiled `_NativeDatabase` handle by composition (see `jammi_ai/_database.py`):
    the migrated verbs (`fine_tune`, `search`, `register_topic`, …) are explicit
    Python methods on the wrapper, while every un-migrated verb (`sql`,
    `drop_mutable_table`, `list_models`, …) lives on the native handle and is
    forwarded at runtime via `__getattr__`. Introspecting the bare wrapper CLASS
    therefore sees the explicit methods without a `__text_signature__` (they are
    plain Python functions) and misses the forwarded verbs entirely.

    Resolving the bound method on an INSTANCE looks through that composition —
    wrapper attribute first, native handle behind it — exactly as the engine's own
    conformance test resolves a verb. `inspect.signature` then reads the real
    signature for both the Python wrapper methods and the PyO3-bound native ones,
    so the keyword surface a recipe relies on is visible again. Returns the parsed
    `Signature` so callers compare parameter names structurally (the rendered form
    carries type annotations, e.g. ``source: 'str'``, that a substring match on the
    bare name would miss).
    """
    fn = getattr(db, name, None)
    if fn is None:
        return None
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


def main() -> int:
    errors: list[str] = []

    for name in MODULE_FUNCTIONS:
        if not hasattr(jammi_ai, name):
            errors.append(f"jammi_ai.{name} is missing from the installed wheel")

    # Introspect through the composition: open the embedded engine and resolve each
    # verb on the live instance, the way a caller invokes it.
    with tempfile.TemporaryDirectory() as artifact_dir:
        db = jammi_ai.connect(f"file://{artifact_dir}")

        for name, kwargs in REQUIRED.items():
            sig = _signature(db, name)
            if sig is None:
                errors.append(f"Database.{name} is missing from the installed wheel")
                continue
            params = set(sig.parameters)
            for kw in kwargs:
                # positional-or-keyword and keyword-only forms both pass: the
                # recipe relies on the name being a real parameter of the verb.
                if kw not in params:
                    errors.append(
                        f"Database.{name}: expected kwarg '{kw}' not found in "
                        f"signature {sig}"
                    )

    if errors:
        print("API reference drifted from the installed jammi_ai wheel:")
        for e in errors:
            print(f"  - {e}")
        return 1

    checked = len(REQUIRED) + len(MODULE_FUNCTIONS)
    print(f"API reference matches installed jammi_ai ({checked} surfaces checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
