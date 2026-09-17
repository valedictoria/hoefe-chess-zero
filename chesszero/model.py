"""A very small prefix-LM transformer with Lc0-style heads.

One trunk, four heads:

``policy``  4672 AlphaZero-style move logits -- the MCTS prior.
``value``   win/draw/loss logits, like Lc0's WDL head.
``mlh``     moves left, also like Lc0: how long the game still has to run.
``lm``      next-token logits, used to write the reasoning trace.

The attention mask is a *prefix* mask rather than a fully causal one: the 70
position tokens all see each other (a board is not a sequence -- square a1 has
every right to look at h8), while trace tokens see the whole prefix plus the
trace so far.  That makes the trace autoregressive without crippling the board
representation.

The policy and value heads are read at two places: over ``<sep>``, where the
model has seen the position and nothing else, and over the trace's final token,
where it has also seen its own reasoning.  The first is what MCTS uses at every
node; the second is what it uses at the root, after the model has thought.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import N_MOVES
from .vocab import PREFIX_LEN, READOUT_FAST, READOUT_REASONED, SEQ_LEN, VOCAB_SIZE


@dataclass
class ModelConfig:
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_ff: int = 384
    dropout: float = 0.0
    vocab_size: int = VOCAB_SIZE
    seq_len: int = SEQ_LEN
    prefix_len: int = PREFIX_LEN
    n_moves: int = N_MOVES

    def to_dict(self) -> dict:
        return asdict(self)


#: a few ready-made sizes; "small" is the default and is about 1.4M parameters
PRESETS = {
    "nano": ModelConfig(d_model=64, n_layer=2, n_head=2, d_ff=192),
    "tiny": ModelConfig(d_model=96, n_layer=3, n_head=3, d_ff=288),
    "small": ModelConfig(d_model=128, n_layer=4, n_head=4, d_ff=384),
    "base": ModelConfig(d_model=256, n_layer=6, n_head=8, d_ff=768),
}


def build_prefix_mask(seq_len: int, prefix_len: int) -> torch.Tensor:
    """``True`` where query ``i`` may attend to key ``j``."""
    idx = torch.arange(seq_len)
    causal = idx[:, None] >= idx[None, :]
    in_prefix = idx < prefix_len
    bidirectional = in_prefix[:, None] & in_prefix[None, :]
    return causal | bidirectional


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q = q.view(B, S, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, S, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, S, self.n_head, self.d_head).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask[:S, :S],
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


def _head(d_model: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, d_out),
    )


class ChessZeroNet(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        cfg = self.cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.policy_head = _head(cfg.d_model, cfg.n_moves)
        self.value_head = _head(cfg.d_model, 3)
        self.mlh_head = _head(cfg.d_model, 1)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied

        self.register_buffer(
            "attn_mask", build_prefix_mask(cfg.seq_len, cfg.prefix_len), persistent=False
        )
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trunk(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, S = token_ids.shape
        pos = torch.arange(S, device=token_ids.device)
        x = self.drop(self.tok_emb(token_ids) + self.pos_emb(pos)[None, :, :])
        for block in self.blocks:
            x = block(x, self.attn_mask)
        return self.ln_f(x)

    def forward(self, token_ids: torch.Tensor, *, need_lm: bool = True) -> dict:
        """Run the full sequence and read every head.

        ``token_ids`` is ``(B, S)``.  Sequences shorter than ``seq_len`` are
        allowed: the reasoned heads then read the last position present.
        """
        h = self.trunk(token_ids)
        fast = h[:, min(READOUT_FAST, h.shape[1] - 1)]
        reasoned = h[:, -1]
        out = {
            "policy": self.policy_head(fast),
            "value": self.value_head(fast),
            "mlh": self.mlh_head(fast).squeeze(-1),
            "policy_reasoned": self.policy_head(reasoned),
            "value_reasoned": self.value_head(reasoned),
        }
        if need_lm:
            out["lm"] = self.lm_head(h)
        return out

    def position_only(self, token_ids: torch.Tensor) -> dict:
        """Cheap path for MCTS leaves: prefix only, no trace, no LM head."""
        h = self.trunk(token_ids[:, : self.cfg.prefix_len])
        fast = h[:, -1]
        return {
            "policy": self.policy_head(fast),
            "value": self.value_head(fast),
            "mlh": self.mlh_head(fast).squeeze(-1),
        }

    # -- checkpointing -----------------------------------------------------
    def save(self, path, **extra) -> None:
        torch.save({"config": self.cfg.to_dict(), "state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        blob = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(ModelConfig(**blob["config"]))
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model, blob
