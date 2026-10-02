"""The eval verbs' requests rank each golden query as a search ranks it.

Hermetic: only the assembled protos, no engine.
"""

from __future__ import annotations

import pytest

from jammi._assembly import build_eval_compare_request, build_eval_embeddings_request

GOLDEN = "golden_rel.public.golden_relevance"


def test_an_eval_is_approximate_unless_told_otherwise():
    embeddings = build_eval_embeddings_request(source="patents", golden_source=GOLDEN)
    compare = build_eval_compare_request(
        embedding_tables=["patents_raw", "patents_tuned"], source="patents", golden_source=GOLDEN
    )
    assert not embeddings.HasField("method")
    assert not compare.HasField("method")


def test_an_exact_eval_scores_every_vector():
    embeddings = build_eval_embeddings_request(source="patents", golden_source=GOLDEN, exact=True)
    compare = build_eval_compare_request(
        embedding_tables=["patents_raw", "patents_tuned"],
        source="patents",
        golden_source=GOLDEN,
        exact=True,
    )
    assert embeddings.method.HasField("exact")
    assert compare.method.HasField("exact")


def test_an_eval_oversample_reaches_the_wire():
    request = build_eval_embeddings_request(source="patents", golden_source=GOLDEN, oversample=8)
    assert request.method.oversample == 8


def test_an_exact_eval_takes_no_oversample():
    with pytest.raises(ValueError, match="no oversample"):
        build_eval_embeddings_request(source="patents", golden_source=GOLDEN, exact=True, oversample=4)
