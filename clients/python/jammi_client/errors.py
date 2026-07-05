"""The `JammiError` taxonomy — one exception family both backends map onto.

Every failure a Jammi session can raise descends from :class:`JammiError`, so a
caller catches the whole surface with one ``except JammiError``. The taxonomy is
transport-agnostic: the pure-Python remote client raises these classes directly,
and the compiled embedded engine's native converters import this module and
raise the very same classes — so a caller writes one error-handling path and it
holds whether the engine runs in-process or behind a server.

The leaf classes also refine the closest built-in exception where that refinement
is honest, so an existing ``except`` clause keeps working: a bad argument IS a
``ValueError``; a transport or training failure IS a ``RuntimeError``. The extra
base is never a compatibility shim — it states what kind of error each class is.

This module is public (``jammi_client.errors``) because a consumer catches these
types by name, and because the native engine imports them from here — the arrow
points one way, native → client, never the reverse.
"""

from __future__ import annotations


class JammiError(Exception):
    """Base of every Jammi error. One ``except JammiError`` catches them all."""


class InvalidArgument(JammiError, ValueError):
    """A caller supplied an argument outside its valid domain.

    Bad format, an unknown enum token, a malformed id, an out-of-range value.
    Refines :class:`ValueError` so an existing ``except ValueError`` still fires.
    """


class NotSupportedOnBackend(JammiError):
    """A one-sided operation was invoked on a backend that does not carry it.

    Some capabilities exist on only one transport — an in-process audit log or
    ephemeral session on the embedded engine, an explicit channel ``close`` or
    connection ``session_id`` on the remote client. Invoking the wrong one raises
    this typed error rather than a bare ``AttributeError``, and
    :meth:`Session.supports` answers the same question before the call. Carries
    the :class:`~jammi_client.Capability` that was unavailable.
    """

    def __init__(self, capability: object) -> None:
        super().__init__(
            f"capability {capability!s} is not supported on this backend; "
            f"call `supports({capability!s})` to check before invoking it"
        )
        self.capability = capability


class NoEmbeddedEngineError(NotSupportedOnBackend):
    """A `file://` (local) target was opened on a build with no embedded engine.

    The pure `jammi-client` ships no compiled engine, so it cannot run a target
    in-process — a local target is a capability this backend does not carry
    (hence a :class:`NotSupportedOnBackend`). Install the embed wheel — `pip
    install jammi-ai` — whose `connect` resolves both local and remote.
    """

    def __init__(self, artifact_dir: str) -> None:
        # Bypass NotSupportedOnBackend's capability-shaped message: this is the
        # target-relocation form of an unsupported backend, with its own hint.
        JammiError.__init__(
            self,
            f"no embedded engine in this build: cannot open the local target "
            f"{artifact_dir!r} — `pip install jammi-ai` for the in-process engine, "
            f"or point connect() at a remote https:// / grpc:// target.",
        )
        self.artifact_dir = artifact_dir


class TrainingError(JammiError, RuntimeError):
    """A training job reached a ``failed`` terminal state.

    Carries the worker's failure message — read off ``TrainingStatus.error`` on
    the remote transport, surfaced from the engine's `TrainingJob.wait` on the
    embedded one — so a job fails for the same cause with the same message
    regardless of where it ran. Refines :class:`RuntimeError`.
    """


class BackendError(JammiError, RuntimeError):
    """A transport or engine runtime failure that is not a caller error.

    Wraps a remote transport fault or an in-process engine runtime failure — the
    residual bucket for anything that is neither a bad argument, an unsupported
    capability, nor a failed training job. Refines :class:`RuntimeError` for the
    same reason :class:`InvalidArgument` refines :class:`ValueError`.
    """
