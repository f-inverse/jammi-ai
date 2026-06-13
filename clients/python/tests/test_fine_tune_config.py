"""Hermetic tests for `RemoteDatabase._fine_tune_config` proto construction.

The builder uses each proto field's explicit presence (`optional`): an omitted
kwarg leaves its field UNSET on the wire so the server resolves it to the engine
default, never a literal `0`. These tests pin that presence shape — in
particular the hard-negative mining knobs, where a literal-zero `refresh_every`
would make the server reject a mining-on config the caller built with only
`mine_hard_negatives=True`.

No channel is dialed: `_fine_tune_config` touches no instance state, so it is
exercised through the unbound method with a sentinel `self`.
"""

from __future__ import annotations

from jammi_client import RemoteDatabase

# All-`None` kwargs: the empty config a caller builds when nothing is set.
_UNSET = dict(
    lora_rank=None,
    lora_alpha=None,
    lora_dropout=None,
    learning_rate=None,
    epochs=None,
    batch_size=None,
    max_seq_length=None,
    validation_fraction=None,
    early_stopping_patience=None,
    warmup_steps=None,
    gradient_accumulation_steps=None,
    triplet_margin=None,
    target_modules=None,
    early_stopping_metric=None,
    backbone_dtype=None,
    weight_decay=None,
    max_grad_norm=None,
    embedding_loss=None,
    mnrl_temperature=None,
    cached=None,
    mine_hard_negatives=None,
    hard_negative_k=None,
    hard_negative_exclude_hops=None,
    hard_negative_refresh_every=None,
    matryoshka_dims=None,
)


def _build(**overrides):
    kwargs = {**_UNSET, **overrides}
    # `_fine_tune_config` reads no instance state; call it unbound.
    return RemoteDatabase._fine_tune_config(object(), **kwargs)


def test_mine_only_leaves_count_knobs_unset() -> None:
    """`mine_hard_negatives=True` alone sends `mine` set and the count knobs
    UNSET, so the server overlays the engine defaults (k=1, exclude_hops=1,
    refresh_every=1) instead of decoding literal zeros."""
    config = _build(mine_hard_negatives=True)

    assert config.HasField("hard_negatives")
    hn = config.hard_negatives
    assert hn.mine is True
    # The count knobs are unset on the wire — the engine fills them.
    assert not hn.HasField("k")
    assert not hn.HasField("exclude_hops")
    assert not hn.HasField("refresh_every")


def test_partial_hard_negatives_sets_only_present_knobs() -> None:
    """A caller that sets `mine` and `k` leaves `exclude_hops`/`refresh_every`
    unset; only the present knobs cross the wire."""
    config = _build(mine_hard_negatives=True, hard_negative_k=5)

    hn = config.hard_negatives
    assert hn.mine is True
    assert hn.HasField("k") and hn.k == 5
    assert not hn.HasField("exclude_hops")
    assert not hn.HasField("refresh_every")


def test_no_hard_negative_kwargs_omits_the_message() -> None:
    """When no mining kwarg is set the nested message is omitted entirely, so the
    server reads mining as off (the engine default)."""
    config = _build()
    assert not config.HasField("hard_negatives")


def _capture_graph_request(**overrides):
    """Build a graph fine-tune request without dialing a channel.

    `fine_tune_graph` touches instance state only at `self._start_training`, so a
    bare instance with that one method stubbed captures the request it assembled.
    """
    db = RemoteDatabase.__new__(RemoteDatabase)
    captured = {}
    db._start_training = lambda request: captured.setdefault("request", request)
    kwargs = dict(
        node_source="papers",
        id_column="paper_id",
        text_column="title",
        edge_source="cites",
        src_column="src",
        dst_column="dst",
        base_model="modernbert",
    )
    kwargs.update(overrides)
    db.fine_tune_graph(**kwargs)
    return captured["request"]


def test_graph_fine_tune_attaches_config_with_hyperparameters() -> None:
    """Regression for #167: remote `fine_tune_graph` must attach the assembled
    `FineTuneConfig` to the `StartTrainingRequest`. Without it the server falls
    back to its built-in defaults and silently drops every hyperparameter the
    caller set — loss, epochs, batch size, learning rate, LoRA rank, matryoshka."""
    request = _capture_graph_request(
        embedding_loss="triplet",
        epochs=7,
        batch_size=16,
        learning_rate=1e-3,
        lora_rank=32,
        matryoshka_dims=[768, 256],
    )

    assert request.HasField("config"), "config must ride the request, not be dropped"
    cfg = request.config
    assert cfg.epochs == 7
    assert cfg.batch_size == 16
    assert abs(cfg.learning_rate - 1e-3) < 1e-9
    assert cfg.lora_rank == 32
    assert list(cfg.matryoshka_dims) == [768, 256]
    # The loss the caller chose carries through, not the server's MNRL default.
    assert cfg.embedding_loss.HasField("triplet")


def test_graph_fine_tune_defaults_still_attach_the_loss() -> None:
    """Even with no hyperparameter kwargs, the config (carrying the default MNRL
    loss) must still attach — the request always speaks the loss explicitly."""
    request = _capture_graph_request()

    assert request.HasField("config")
    assert request.config.embedding_loss.HasField("multiple_negatives_ranking")
