"""Hermetic tests for `build_fine_tune_config` proto construction.

The builder uses each proto field's explicit presence (`optional`): an omitted
kwarg leaves its field UNSET on the wire so the server resolves it to the engine
default, never a literal `0`. These tests pin that presence shape — in
particular the hard-negative mining knobs, where a literal-zero `refresh_every`
would make the server reject a mining-on config the caller built with only
`mine_hard_negatives=True`.

No channel is dialed: `build_fine_tune_config` is a free function in the
assembly layer, so it is exercised directly with no transport.
"""

from __future__ import annotations

from jammi_client import RemoteDatabase
from jammi_client._assembly import build_fine_tune_config

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
    seed=None,
    regression_loss=None,
    regression_beta=None,
    quantile_levels=None,
)


def _build(**overrides):
    kwargs = {**_UNSET, **overrides}
    return build_fine_tune_config(**kwargs)


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


def test_seed_unset_by_default_lets_engine_default_apply() -> None:
    """An omitted `seed` leaves the field UNSET on the wire, so the server
    resolves the engine's fixed default seed rather than decoding a literal 0."""
    config = _build()
    assert not config.HasField("seed")


def test_seed_set_crosses_the_wire() -> None:
    """A caller-supplied `seed` is sent with explicit presence so the engine
    makes the fine-tune bit-reproducible from it."""
    config = _build(seed=12345)
    assert config.HasField("seed")
    assert config.seed == 12345


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


# --- Regression objective (W5-PR4) -----------------------------------------
#
# The embed binding's `fine_tune` decodes `regression_loss` / `regression_beta`
# / `quantile_levels` onto `FineTuneConfig.regression_loss` (proto field 27) and
# `quantile_levels` (field 28). The pure client must thread the SAME three kwargs
# onto the SAME proto fields — a signature-only addition would pass the
# cross-wheel conformance guard while silently dropping the regression config on
# the wire (the #167 config-attach class of bug). These pin that they reach the
# proto, not just the signature.


def test_regression_loss_unset_leaves_field_unset() -> None:
    """No regression kwargs leaves `regression_loss`/`quantile_levels` UNSET so
    the server applies the engine's collapse-resistant β-NLL default."""
    config = _build()
    assert not config.HasField("regression_loss")
    assert list(config.quantile_levels) == []


def test_regression_named_losses_map_to_their_variant() -> None:
    """Each named loss decodes to its `RegressionLoss` oneof variant, matching the
    embed binding's mapping in `crates/jammi-python/src/database.rs`."""
    assert _build(regression_loss="gaussian_nll").regression_loss.HasField("gaussian_nll")
    assert _build(regression_loss="crps").regression_loss.HasField("crps")
    assert _build(regression_loss="pinball").regression_loss.HasField("pinball")
    # `beta_nll` carries its β (explicit, then the 0.5 default).
    cfg = _build(regression_loss="beta_nll", regression_beta=0.25)
    assert cfg.regression_loss.HasField("beta_nll")
    assert abs(cfg.regression_loss.beta_nll.beta - 0.25) < 1e-9
    cfg_default = _build(regression_loss="beta_nll")
    assert abs(cfg_default.regression_loss.beta_nll.beta - 0.5) < 1e-9


def test_bare_regression_beta_implies_beta_nll() -> None:
    """A bare `regression_beta` (no named loss) keeps the beta-implies-β-NLL
    shorthand, mirroring how the embed binding decodes it."""
    cfg = _build(regression_beta=0.3)
    assert cfg.regression_loss.HasField("beta_nll")
    assert abs(cfg.regression_loss.beta_nll.beta - 0.3) < 1e-9


def test_quantile_levels_cross_the_wire() -> None:
    """`quantile_levels` are wired onto field 28 regardless of the named loss so a
    Pinball request is reachable; the server validates them before training."""
    cfg = _build(regression_loss="pinball", quantile_levels=[0.1, 0.5, 0.9])
    assert cfg.regression_loss.HasField("pinball")
    assert list(cfg.quantile_levels) == [0.1, 0.5, 0.9]


def test_unknown_regression_loss_rejected() -> None:
    """An unknown name raises rather than silently sending an empty oneof."""
    import pytest

    with pytest.raises(ValueError, match="Unknown regression_loss"):
        _build(regression_loss="bogus")


def _capture_fine_tune_request(**overrides):
    """Build a (non-graph) `fine_tune` request without dialing a channel.

    `fine_tune` touches instance state only at `self._start_training`; stubbing
    that one method captures the `StartTrainingRequest` it assembled.
    """
    db = RemoteDatabase.__new__(RemoteDatabase)
    captured = {}
    db._start_training = lambda request: captured.setdefault("request", request)
    kwargs = dict(
        source="houses",
        base_model="modernbert",
        columns=["features", "price"],
        method="lora",
        task="regression",
    )
    kwargs.update(overrides)
    db.fine_tune(**kwargs)
    return captured["request"]


def test_remote_fine_tune_threads_regression_config_to_the_proto() -> None:
    """End-to-end #167 guard: a remote `fine_tune(regression_loss=..., quantile_levels=...)`
    must carry that regression config into the `FineTuneConfig` on the built
    proto. A signature-only addition (passing the conformance guard) would drop
    it here — this catches that."""
    request = _capture_fine_tune_request(
        regression_loss="pinball",
        quantile_levels=[0.1, 0.5, 0.9],
    )
    assert request.HasField("config"), "config must ride the request, not be dropped"
    cfg = request.config
    assert cfg.regression_loss.HasField("pinball")
    assert list(cfg.quantile_levels) == [0.1, 0.5, 0.9]


def test_remote_fine_tune_threads_beta_nll_to_the_proto() -> None:
    """A `beta_nll` choice carries its β onto the proto's `RegressionLoss`
    oneof (field 27), not just the signature."""
    request = _capture_fine_tune_request(
        regression_loss="beta_nll",
        regression_beta=0.2,
    )
    cfg = request.config
    assert cfg.regression_loss.HasField("beta_nll")
    assert abs(cfg.regression_loss.beta_nll.beta - 0.2) < 1e-9
