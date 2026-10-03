"""Claims: what a chapter shows, checked on the run that shows it.

The book's measurement rail. A finding ends in the claim it makes::

    claim("rescore recovers what the raw Hamming ranking loses",
          rescored_at_10 > hamming_at_10,
          f"recall@10 {rescored_at_10:.3f} against {hamming_at_10:.3f}")

The claim prints beside the run's own numbers, so the reader sees what the
chapter just showed them; a claim that does not hold raises
:class:`ClaimFailed`, naming it. A claim is a relation the capability
guarantees — an ordering, a bound, an equality the engine defines — so it holds
on any host, at either scale (:mod:`jammi_cookbook.scale`), on any GPU. A frozen
number would only say what one machine measured once.
"""

from __future__ import annotations


class ClaimFailed(AssertionError):
    """A chapter's claim did not hold on this run: what the chapter teaches is
    false here, which is a bug in jammi or in the chapter."""


def claim(statement: str, holds: bool, evidence: str = "") -> None:
    """Check ``statement`` on this run.

    Prints it, with ``evidence`` (the numbers it rests on), when ``holds``;
    raises :class:`ClaimFailed` naming it when not.
    """
    shown = f"{statement} ({evidence})" if evidence else statement
    if not holds:
        raise ClaimFailed(f"does not hold on this run: {shown}")
    print(f"✓ {shown}")
