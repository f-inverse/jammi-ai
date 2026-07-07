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

This module is public (``jammi.errors``) because a consumer catches these
types by name, and because the native engine imports them from here — the arrow
points one way, native → client, never the reverse.
"""

from __future__ import annotations

from ._capability import Capability


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
    :meth:`Session.supports` answers the same question before the call.

    Constructed one of two ways: with the :class:`~jammi.Capability` a
    caller invoked on the wrong backend (the local, capability-shaped case), or
    with a server ``UNIMPLEMENTED`` detail string (a verb the remote deployment
    did not mount). ``capability`` is the enum member in the first case, ``None``
    in the second.
    """

    def __init__(self, capability_or_detail: object) -> None:
        if isinstance(capability_or_detail, Capability):
            self.capability = capability_or_detail
            super().__init__(
                f"capability {capability_or_detail!s} is not supported on this "
                f"backend; call `supports({capability_or_detail!s})` to check "
                f"before invoking it"
            )
        else:
            self.capability = None
            super().__init__(str(capability_or_detail))


class NoEmbeddedEngineError(NotSupportedOnBackend):
    """The embedded engine was needed on a build that does not carry it.

    `jammi-ai` is the base client; it discovers the compiled engine as an
    optional in-process backend but does not carry it by default. Absent the
    `jammi-ai[embedded]` extra (which pulls `jammi-ai-native`, importable as
    `jammi_native`), two things this build cannot do raise this one error: open a
    `file://` (local) target in-process, and surface an embedded-only value-type
    (`jammi.PerQueryAudit`, …) the engine exports. Either way a capability this
    build does not carry was reached (hence a :class:`NotSupportedOnBackend`).
    Install the extra — `pip install jammi-ai[embedded]` — and the SAME
    `jammi.connect` / `jammi.<Type>` resolve.

    Constructed for whichever thing was reached: the default form names the
    unopenable local target and exposes it as :attr:`artifact_dir`;
    :meth:`for_symbol` names the embedded-only attribute that was accessed. Only
    the target form carries an :attr:`artifact_dir` (``None`` on the symbol form).
    """

    def __init__(self, artifact_dir: str) -> None:
        # Bypass NotSupportedOnBackend's capability-shaped message: this is the
        # target-relocation form of an unsupported backend, with its own hint.
        JammiError.__init__(
            self,
            f"no embedded engine in this build: cannot open the local target "
            f"{artifact_dir!r} — `pip install jammi-ai[embedded]` for the "
            f"in-process engine, or point connect() at a remote https:// / grpc:// "
            f"target.",
        )
        self.artifact_dir = artifact_dir

    @classmethod
    def for_symbol(cls, symbol: str) -> "NoEmbeddedEngineError":
        """An embedded-only attribute (`jammi.<symbol>`) was accessed with no engine.

        The value-types the in-process engine exports (`PerQueryAudit`,
        `TrainingJob`, …) are surfaced lazily on `jammi`; reaching one without the
        `[embedded]` extra is this error, naming the attribute and the extra rather
        than a bare `AttributeError`. Alternate constructor: it bypasses the
        target-shaped ``__init__`` (there is no `artifact_dir` here) and leaves
        :attr:`artifact_dir` ``None``.
        """
        err = cls.__new__(cls)
        JammiError.__init__(
            err,
            f"no embedded engine in this build: `jammi.{symbol}` is an "
            f"embedded-only type provided by the in-process engine — "
            f"`pip install jammi-ai[embedded]` to surface it, or use the remote "
            f"transport, which does not carry it.",
        )
        err.artifact_dir = None
        return err


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
