"""`jammi_numerics::batch_shape::ShapeLadder`, the one padded-width rule of
every forward the engine runs — serving, a training step and an evaluation
pass alike — mirrored once for every reference twin that pads a batch.

Between `2^k` and `2^(k+1)` the rungs are `2^k · (1 + j/N)` for `N =
RUNGS_PER_OCTAVE`, brought up to `RUNG_ALIGNMENT`; a width is padded by under
`1/N` of itself once an octave's divisions are the alignment apart, and by
under eight tokens below that.
"""
from __future__ import annotations

RUNG_ALIGNMENT = 8
RUNGS_PER_OCTAVE = 8


def width(natural: int, limit: int) -> int:
    """`ShapeLadder::width`: the smallest rung at or above `natural`, never
    above `limit`. A `natural` of zero (no rows) or a `limit` of zero (no
    sequence axis) passes through unchanged."""
    if natural == 0 or limit == 0:
        return natural
    natural = min(max(natural, RUNG_ALIGNMENT), max(limit, RUNG_ALIGNMENT))
    octave = 1 << (natural.bit_length() - 1)
    division = max(octave // RUNGS_PER_OCTAVE, 1)
    rung = octave + -(-(natural - octave) // division) * division
    return min(-(-rung // RUNG_ALIGNMENT) * RUNG_ALIGNMENT, limit)
