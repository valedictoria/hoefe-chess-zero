"""The prefix mask is load-bearing, not a detail.

Two properties matter.  The board half of the sequence must be bidirectional, or
square a1 cannot see h8 and the position representation is crippled.  The trace
half must be causal, or generation leaks future tokens -- and the engine relies on
that causality to generate a trace with a padded suffix instead of a KV cache.
"""

import torch

from chesszero.model import PRESETS, ChessZeroNet, ModelConfig, build_prefix_mask
from chesszero.vocab import PREFIX_LEN, READOUT_FAST, SEQ_LEN, VOCAB_SIZE


def test_prefix_mask_is_bidirectional_over_the_board():
    mask = build_prefix_mask(SEQ_LEN, PREFIX_LEN)
    assert mask[0, PREFIX_LEN - 1] and mask[PREFIX_LEN - 1, 0]
    assert mask[5, 60] and mask[60, 5]


def test_prefix_mask_is_causal_over_the_trace():
    mask = build_prefix_mask(SEQ_LEN, PREFIX_LEN)
    assert not mask[PREFIX_LEN, PREFIX_LEN + 1]
    assert mask[PREFIX_LEN + 1, PREFIX_LEN]
    assert mask[SEQ_LEN - 1, 5], "trace must still see the board"


def test_presets_build_and_stay_small():
    for name, cfg in PRESETS.items():
        net = ChessZeroNet(cfg)
        assert net.cfg.seq_len == SEQ_LEN
        assert net.n_params() < 8_000_000, f"{name} is not a small model any more"


def test_head_shapes():
    net = ChessZeroNet(PRESETS["nano"]).eval()
    ids = torch.randint(0, VOCAB_SIZE, (3, SEQ_LEN))
    with torch.no_grad():
        out = net(ids)
    assert out["policy"].shape == (3, 4672)
    assert out["value"].shape == (3, 3)
    assert out["mlh"].shape == (3,)
    assert out["policy_reasoned"].shape == (3, 4672)
    assert out["lm"].shape == (3, SEQ_LEN, VOCAB_SIZE)


def test_position_only_heads_cannot_see_the_trace():
    """The MCTS hot path must not depend on trace tokens it never fills in."""
    net = ChessZeroNet(PRESETS["nano"]).eval()
    ids = torch.randint(0, VOCAB_SIZE, (4, SEQ_LEN))
    other = ids.clone()
    other[:, PREFIX_LEN:] = 7
    with torch.no_grad():
        a, b = net(ids), net(other)
    assert torch.allclose(a["policy"], b["policy"], atol=1e-6)
    assert torch.allclose(a["value"], b["value"], atol=1e-6)


def test_hidden_state_ignores_later_tokens():
    """What makes padded-suffix generation exact instead of an approximation."""
    net = ChessZeroNet(PRESETS["nano"]).eval()
    ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
    cut = PREFIX_LEN + 10
    other = ids.clone()
    other[:, cut + 1 :] = 3
    with torch.no_grad():
        assert torch.allclose(net.trunk(ids)[:, cut], net.trunk(other)[:, cut], atol=1e-6)


def test_position_only_path_matches_the_full_forward():
    net = ChessZeroNet(PRESETS["nano"]).eval()
    ids = torch.randint(0, VOCAB_SIZE, (3, SEQ_LEN))
    with torch.no_grad():
        full, cheap = net(ids), net.position_only(ids)
    assert torch.allclose(full["policy"], cheap["policy"], atol=1e-5)
    assert torch.allclose(full["value"], cheap["value"], atol=1e-5)


def test_checkpoint_round_trip(tmp_path):
    net = ChessZeroNet(ModelConfig(d_model=32, n_layer=1, n_head=2, d_ff=64))
    path = tmp_path / "net.pt"
    net.save(path, step=7)
    loaded, blob = ChessZeroNet.load(path)
    assert blob["step"] == 7
    ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
    with torch.no_grad():
        assert torch.allclose(net.eval()(ids)["policy"], loaded(ids)["policy"], atol=1e-6)
