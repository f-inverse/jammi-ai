"""The `Capability` enum — the closed set of one-sided session features.

A :class:`Session` runs behind one of two transports (an in-process embedded
engine or a remote gRPC channel), and a handful of features exist on only one of
them. :meth:`Session.supports` answers "does this backend carry that feature?"
against this enum, and invoking a feature the backend lacks raises
:class:`~jammi.errors.NotSupportedOnBackend` — never a silent
``AttributeError``.

The set is CLOSED: exactly the features that genuinely diverge between the two
transports. It names no consumer — each value is a generic engine feature
(an audit log, a session-scoped storage context, a model preload, a connection
id), reachable by any user who never heard of a particular one.

"Genuinely diverge" is load-bearing in both directions, and a member LEAVES when
it stops diverging. `close()` is not a member: both transports carry it (the
embedded arm's is the AWAITED catalog-file release under the engine's
`unix-excl` catalog seam, not a release on drop), so it is an ordinary member of
the :class:`~jammi.Session` surface — a flag every backend sets is a predicate
that never discriminates.
"""

from __future__ import annotations

from enum import Enum


class Capability(str, Enum):
    """A one-sided session feature `supports()` predicates over.

    A string enum (``Capability.AUDIT == "audit"``) so a value is both a typed
    member and a plain string in messages and logs. `requires-python >= 3.9`
    predates :class:`enum.StrEnum`, so this spells the same mixin explicitly.
    """

    #: The per-query audit log (`db.audit`) — embedded only.
    AUDIT = "audit"
    #: A session-scoped ephemeral storage context (`db.ephemeral_session`) —
    #: embedded only.
    EPHEMERAL_SESSION = "ephemeral_session"
    #: Preloading a model into the cache (`db.preload_model`) — embedded only.
    PRELOAD_MODEL = "preload_model"
    #: The opaque per-connection session id (`db.session_id`) — remote only.
    SESSION_ID = "session_id"

    def __str__(self) -> str:
        # The bare value ("audit"), not "Capability.AUDIT", so error messages and
        # logs read as the feature name.
        return self.value
