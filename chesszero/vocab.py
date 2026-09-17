"""Token vocabulary shared by the board encoder and the reasoning language model.

Everything the network ever sees is a token id from this single vocabulary: the
squares of the board, the little bits of state that do not fit on the board
(castling, en passant, the 50-move counter, repetitions) and the tokens of the
reasoning trace the language-model head emits.

The board is always presented from the side to move's point of view, so an
uppercase piece letter means "a piece belonging to the player to move" and a
lowercase one means "a piece belonging to the opponent".  See
:mod:`chesszero.encoding` for the canonicalisation.
"""

from __future__ import annotations

import chess

# --- special tokens -------------------------------------------------------
PAD = "<pad>"
BOS = "<bos>"
SEP = "<sep>"
EOS = "<eos>"
SPECIAL_TOKENS = [PAD, BOS, SEP, EOS]

# --- board tokens ---------------------------------------------------------
EMPTY_TOKEN = "sq:."
PIECE_TOKENS = [EMPTY_TOKEN] + [f"sq:{c}" for c in "PNBRQKpnbrqk"]
CASTLE_TOKENS = [f"cas:{i}" for i in range(16)]
EP_TOKENS = ["ep:-"] + [f"ep:{f}" for f in "abcdefgh"]
R50_TOKENS = [f"r50:{i}" for i in range(8)]
REP_TOKENS = [f"rep:{i}" for i in range(3)]

# --- tokens the reasoning trace is built from -----------------------------
SQUARE_TOKENS = [f"@{chess.square_name(sq)}" for sq in range(64)]
PROMO_TOKENS = ["pr:-", "pr:n", "pr:b", "pr:r", "pr:q"]

FIELD_TOKENS = [
    "f:MAT",
    "f:PHASE",
    "f:KSAFE",
    "f:THR",
    "f:TAC",
    "f:PLAN",
    "f:CAND",
    "f:BEST",
    "f:EVAL",
]

#: material balance buckets, in pawns, from the mover's point of view
MAT_EDGES = (-5.0, -3.0, -2.0, -0.5, 0.5, 2.0, 3.0, 5.0)
MAT_TOKENS = [f"mat:{i}" for i in range(len(MAT_EDGES) + 1)]

PHASE_TOKENS = ["ph:opening", "ph:middle", "ph:end"]
KSAFE_TOKENS = ["ks:safe", "ks:ok", "ks:loose", "ks:exposed"]
THR_TOKENS = [f"thr:{i}" for i in range(4)]
TAC_TOKENS = [
    "tac:none",
    "tac:mate1",
    "tac:check",
    "tac:hang",
    "tac:fork",
    "tac:pin",
    "tac:promo",
    "tac:cap",
    "tac:quiet",
]
PLAN_TOKENS = [
    "plan:develop",
    "plan:castle",
    "plan:center",
    "plan:attack",
    "plan:trade",
    "plan:push",
    "plan:rooks",
    "plan:defend",
    "plan:convert",
    "plan:hold",
]
#: eval buckets, 0 = losing badly, 10 = winning easily (mover's point of view)
EVAL_TOKENS = [f"ev:{i}" for i in range(11)]

VOCAB: list[str] = (
    SPECIAL_TOKENS
    + PIECE_TOKENS
    + CASTLE_TOKENS
    + EP_TOKENS
    + R50_TOKENS
    + REP_TOKENS
    + SQUARE_TOKENS
    + PROMO_TOKENS
    + FIELD_TOKENS
    + MAT_TOKENS
    + PHASE_TOKENS
    + KSAFE_TOKENS
    + THR_TOKENS
    + TAC_TOKENS
    + PLAN_TOKENS
    + EVAL_TOKENS
)

assert len(VOCAB) == len(set(VOCAB)), "duplicate token in vocabulary"

TOKEN_TO_ID: dict[str, int] = {tok: i for i, tok in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)


def tid(token: str) -> int:
    """Token string -> token id."""
    return TOKEN_TO_ID[token]


def ids(tokens) -> list[int]:
    return [TOKEN_TO_ID[t] for t in tokens]


def tokens(id_seq) -> list[str]:
    return [VOCAB[int(i)] for i in id_seq]


# --- the grammar of a reasoning trace -------------------------------------
# A trace is a *fixed length* sequence of 31 tokens.  Each slot is either a
# literal token or a set of tokens that may appear there.  Keeping the shape
# fixed means batching is trivial and, more importantly, generation can be
# constrained to the grammar so the model can never emit an unparseable trace.

_MOVE_SLOTS = [SQUARE_TOKENS, SQUARE_TOKENS, PROMO_TOKENS]

TRACE_SLOTS: list[list[str]] = (
    [["f:MAT"], MAT_TOKENS]
    + [["f:PHASE"], PHASE_TOKENS]
    + [["f:KSAFE"], KSAFE_TOKENS, KSAFE_TOKENS]
    + [["f:THR"], THR_TOKENS]
    + [["f:TAC"], TAC_TOKENS, TAC_TOKENS]
    + [["f:PLAN"], PLAN_TOKENS]
    + [["f:CAND"]] + _MOVE_SLOTS * 3
    + [["f:BEST"]] + _MOVE_SLOTS
    + [["f:EVAL"], EVAL_TOKENS]
    + [[EOS]]
)

TRACE_LEN = len(TRACE_SLOTS)
assert TRACE_LEN == 31, TRACE_LEN

#: slot index of each interesting field inside a trace
SLOT_MAT = 1
SLOT_PHASE = 3
SLOT_KSAFE_US = 5
SLOT_KSAFE_THEM = 6
SLOT_THR = 8
SLOT_TAC = (10, 11)
SLOT_PLAN = 13
SLOT_CAND = (15, 18, 21)  # each is the from-square slot of a candidate move
SLOT_BEST = 25
SLOT_EVAL = 29

TRACE_SLOT_IDS: list[list[int]] = [ids(slot) for slot in TRACE_SLOTS]

# --- sequence layout ------------------------------------------------------
# <bos> s0..s63 <castle> <ep> <r50> <rep> <sep> | trace...
PREFIX_LEN = 1 + 64 + 4 + 1  # 70, the last token being <sep>
SEQ_LEN = PREFIX_LEN + TRACE_LEN  # 101
READOUT_FAST = PREFIX_LEN - 1  # hidden state over <sep>: position-only heads
READOUT_REASONED = SEQ_LEN - 1  # hidden state over the trace's <eos>
