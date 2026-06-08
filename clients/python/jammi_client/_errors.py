"""Client-side error types.

The pure-Python `jammi-client` resolves remote targets only. Asked to open a
`file://` (local) target it raises :class:`NoEmbeddedEngineError` — the runtime
echo of the Rust build's `#[cfg(feature = "local")]` gate, where a `wire`-only
build cannot even *name* a local target. The message points at the fix
(`pip install jammi-ai`) rather than failing opaquely.
"""

from __future__ import annotations


class NoEmbeddedEngineError(RuntimeError):
    """Raised when a `file://` (local) target is opened on the pure client.

    `jammi-client` ships no compiled engine, so it cannot run a target
    in-process. Install the embed wheel — `pip install jammi-ai` — whose
    `connect` resolves both local and remote.
    """

    def __init__(self, artifact_dir: str) -> None:
        super().__init__(
            f"no embedded engine in this build: cannot open the local target "
            f"{artifact_dir!r} — `pip install jammi-ai` for the in-process engine, "
            f"or point connect() at a remote https:// / grpc:// target."
        )


class TrainingError(RuntimeError):
    """Raised when a remote training job reaches a ``failed`` terminal state.

    Carries the worker's failure message read off ``TrainingStatus.error`` — the
    same reason the embedded engine's :class:`jammi_ai.TrainingJob.wait` surfaces,
    so a remote ``wait()`` fails for the same cause with the same message.
    """

