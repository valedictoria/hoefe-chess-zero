"""The reasoning trace: a tiny controlled language the model speaks about a position.

A trace is 31 tokens long and always has the same shape::

    MAT <bucket> PHASE <p> KSAFE <us> <them> THR <n> TAC <t> <t>
    PLAN <plan> CAND <m1> <m2> <m3> BEST <m> EVAL <bucket> <eos>

During training the target trace is written by :func:`annotate`, a plain
symbolic annotator: the positional fields come from hand-written chess features
and the CAND/BEST/EVAL fields come from whatever search produced the training
sample.  The language-model head is therefore *distilled* from search plus
features -- it is not reasoning that emerged on its own.  At play time the
model writes its own trace, and the heads that read the finished trace produce
the policy and value the engine actually searches with.
"""

from __future__ import annotations

import chess

from . import vocab as V
from .vocab import (
    EOS,
    EVAL_TOKENS,
    KSAFE_TOKENS,
    MAT_EDGES,
    MAT_TOKENS,
    PHASE_TOKENS,
    PLAN_TOKENS,
    PROMO_TOKENS,
    SQUARE_TOKENS,
    TAC_TOKENS,
    THR_TOKENS,
    TRACE_LEN,
)

PIECE_VALUE = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

_PROMO_TOKEN = {
    None: "pr:-",
    chess.KNIGHT: "pr:n",
    chess.BISHOP: "pr:b",
    chess.ROOK: "pr:r",
    chess.QUEEN: "pr:q",
}
_TOKEN_PROMO = {v: k for k, v in _PROMO_TOKEN.items()}

NULL_MOVE_TOKENS = ["@a1", "@a1", "pr:-"]


# --------------------------------------------------------------------------
# moves <-> trace tokens
# --------------------------------------------------------------------------
def move_tokens(move: chess.Move | None) -> list[str]:
    """A canonical move as its three trace tokens (from, to, promotion)."""
    if move is None:
        return list(NULL_MOVE_TOKENS)
    return [
        SQUARE_TOKENS[move.from_square],
        SQUARE_TOKENS[move.to_square],
        _PROMO_TOKEN.get(move.promotion, "pr:-"),
    ]


def tokens_to_move(toks) -> chess.Move | None:
    """Inverse of :func:`move_tokens`; ``None`` for the null placeholder."""
    if list(toks) == NULL_MOVE_TOKENS:
        return None
    try:
        frm = SQUARE_TOKENS.index(toks[0])
        to = SQUARE_TOKENS.index(toks[1])
    except ValueError:
        return None
    return chess.Move(frm, to, promotion=_TOKEN_PROMO.get(toks[2]))


# --------------------------------------------------------------------------
# position features (all computed on a canonical board: white is the mover)
# --------------------------------------------------------------------------
def material_balance(board: chess.Board) -> float:
    """Material in pawns, positive when the side to move is up."""
    total = 0.0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUE[piece.piece_type]
        total += value if piece.color == chess.WHITE else -value
    return total


def material_bucket(balance: float) -> str:
    idx = 0
    for edge in MAT_EDGES:
        if balance >= edge:
            idx += 1
    return MAT_TOKENS[idx]


def phase_token(board: chess.Board) -> str:
    non_pawn = 0.0
    for piece in board.piece_map().values():
        if piece.piece_type not in (chess.PAWN, chess.KING):
            non_pawn += PIECE_VALUE[piece.piece_type]
    if non_pawn >= 55:
        return PHASE_TOKENS[0]
    if non_pawn >= 22:
        return PHASE_TOKENS[1]
    return PHASE_TOKENS[2]


def king_safety(board: chess.Board, color: chess.Color) -> str:
    king = board.king(color)
    if king is None:  # pragma: no cover - only in hand-built positions
        return KSAFE_TOKENS[3]
    danger = 0
    shield = 0
    kf, kr = chess.square_file(king), chess.square_rank(king)
    forward = 1 if color == chess.WHITE else -1
    for df in (-1, 0, 1):
        f = kf + df
        if not 0 <= f < 8:
            continue
        r = kr + forward
        if 0 <= r < 8:
            piece = board.piece_at(chess.square(f, r))
            if piece is not None and piece.piece_type == chess.PAWN and piece.color == color:
                shield += 1
    for df in (-2, -1, 0, 1, 2):
        for dr in (-2, -1, 0, 1, 2):
            f, r = kf + df, kr + dr
            if 0 <= f < 8 and 0 <= r < 8:
                if board.is_attacked_by(not color, chess.square(f, r)):
                    danger += 1
    open_file = not any(
        (p := board.piece_at(chess.square(kf, r))) is not None
        and p.piece_type == chess.PAWN
        and p.color == color
        for r in range(8)
    )
    score = danger - 3 * shield + (3 if open_file else 0)
    if score <= 0:
        return KSAFE_TOKENS[0]
    if score <= 5:
        return KSAFE_TOKENS[1]
    if score <= 10:
        return KSAFE_TOKENS[2]
    return KSAFE_TOKENS[3]


def hanging_pieces(board: chess.Board, color: chess.Color) -> list[int]:
    """Squares of ``color``'s pieces that are attacked and not adequately defended."""
    out = []
    for square, piece in board.piece_map().items():
        if piece.color != color or piece.piece_type == chess.KING:
            continue
        attackers = board.attackers(not color, square)
        if not attackers:
            continue
        defenders = board.attackers(color, square)
        if not defenders:
            out.append(square)
            continue
        cheapest = min(
            PIECE_VALUE[board.piece_type_at(a)] for a in attackers
        )
        if cheapest + 0.5 < PIECE_VALUE[piece.piece_type]:
            out.append(square)
    return out


def _is_fork(board: chess.Board, move: chess.Move) -> bool:
    piece = board.piece_type_at(move.from_square)
    if piece not in (chess.KNIGHT, chess.PAWN):
        return False
    board.push(move)
    try:
        mover_value = PIECE_VALUE[piece]
        targets = 0
        for square in board.attacks(move.to_square):
            victim = board.piece_at(square)
            if victim is not None and victim.color != chess.WHITE:
                if victim.piece_type == chess.KING or PIECE_VALUE[victim.piece_type] > mover_value:
                    targets += 1
        return targets >= 2
    finally:
        board.pop()


def tactic_tags(board: chess.Board) -> list[str]:
    """Up to two tactical tags, most important first."""
    tags: list[str] = []
    legal = list(board.legal_moves)

    mate = check = capture = fork = False
    for move in legal:
        if board.is_capture(move):
            capture = True
        if board.gives_check(move):
            board.push(move)
            mated = board.is_checkmate()
            board.pop()
            if mated:
                mate = True
                break
            check = True
    if mate:
        tags.append("tac:mate1")
    if not mate:
        for move in legal:
            if _is_fork(board, move):
                fork = True
                break

    if hanging_pieces(board, chess.BLACK):
        tags.append("tac:hang")
    if fork:
        tags.append("tac:fork")
    if any(board.is_pinned(chess.BLACK, sq) for sq in board.pieces(chess.QUEEN, chess.BLACK) | board.pieces(chess.ROOK, chess.BLACK)):
        tags.append("tac:pin")
    if board.pieces(chess.PAWN, chess.WHITE) & chess.BB_RANK_7:
        tags.append("tac:promo")
    if check:
        tags.append("tac:check")
    if capture:
        tags.append("tac:cap")
    if not tags:
        tags.append("tac:quiet")
    tags = tags[:2]
    while len(tags) < 2:
        tags.append("tac:none")
    return tags


_BACK_RANK_MINORS = (chess.B1, chess.C1, chess.F1, chess.G1)


def plan_token(board: chess.Board, balance: float, phase: str, threats: int) -> str:
    if board.is_check():
        return "plan:defend"
    if threats >= 2:
        return "plan:defend"
    if phase == PHASE_TOKENS[2]:
        if balance >= 2.0:
            return "plan:convert"
        if board.pieces(chess.PAWN, chess.WHITE):
            return "plan:push"
        return "plan:hold"
    if board.has_castling_rights(chess.WHITE) and king_safety(board, chess.WHITE) != KSAFE_TOKENS[0]:
        return "plan:castle"
    undeveloped = sum(
        1
        for sq in _BACK_RANK_MINORS
        if (p := board.piece_at(sq)) is not None
        and p.color == chess.WHITE
        and p.piece_type in (chess.KNIGHT, chess.BISHOP)
    )
    if phase == PHASE_TOKENS[0] and undeveloped >= 2:
        return "plan:develop"
    if king_safety(board, chess.BLACK) in (KSAFE_TOKENS[2], KSAFE_TOKENS[3]):
        return "plan:attack"
    if balance >= 1.5:
        return "plan:trade"
    rooks = board.pieces(chess.ROOK, chess.WHITE)
    if len(rooks) >= 2 and not any(board.attacks(r) & rooks for r in rooks):
        return "plan:rooks"
    if balance <= -1.5:
        return "plan:hold"
    return "plan:center"


def eval_bucket(q: float) -> str:
    """Map a value in [-1, 1] (mover's point of view) onto an EVAL token."""
    q = max(-1.0, min(1.0, float(q)))
    idx = int(round((q + 1.0) * 0.5 * (len(EVAL_TOKENS) - 1)))
    return EVAL_TOKENS[idx]


# --------------------------------------------------------------------------
# the annotator
# --------------------------------------------------------------------------
def annotate(
    canon_board: chess.Board,
    best_move: chess.Move | None = None,
    candidates=None,
    q: float = 0.0,
) -> list[str]:
    """Write the teacher trace for a canonical position.

    ``best_move`` and ``candidates`` are canonical moves taken from whatever
    search produced this training sample; ``q`` is that search's value estimate
    in [-1, 1] from the mover's point of view.
    """
    balance = material_balance(canon_board)
    phase = phase_token(canon_board)
    threat_squares = hanging_pieces(canon_board, chess.WHITE)
    threats = min(3, len(threat_squares))

    candidates = list(candidates or [])
    if best_move is not None and best_move not in candidates:
        candidates.insert(0, best_move)
    candidates = candidates[:3]
    while len(candidates) < 3:
        candidates.append(None)

    trace: list[str] = ["f:MAT", material_bucket(balance)]
    trace += ["f:PHASE", phase]
    trace += ["f:KSAFE", king_safety(canon_board, chess.WHITE), king_safety(canon_board, chess.BLACK)]
    trace += ["f:THR", THR_TOKENS[threats]]
    trace += ["f:TAC"] + tactic_tags(canon_board)
    trace += ["f:PLAN", plan_token(canon_board, balance, phase, threats)]
    trace += ["f:CAND"]
    for move in candidates:
        trace += move_tokens(move)
    trace += ["f:BEST"] + move_tokens(best_move)
    trace += ["f:EVAL", eval_bucket(q)]
    trace += [EOS]

    assert len(trace) == TRACE_LEN, len(trace)
    _validate(trace)
    return trace


def _validate(trace) -> None:
    for i, (token, allowed) in enumerate(zip(trace, V.TRACE_SLOTS)):
        if token not in allowed:
            raise ValueError(f"trace slot {i}: {token!r} not allowed here")


# --------------------------------------------------------------------------
# reading a trace back
# --------------------------------------------------------------------------
_MAT_LABEL = ["-5+", "-3/-5", "-2", "-1", "level", "+1", "+2", "+3/+5", "+5+"]


def parse_trace(trace) -> dict:
    """Turn a trace into a dict of readable fields plus canonical moves."""
    trace = list(trace)
    cand = []
    for slot in V.SLOT_CAND:
        move = tokens_to_move(trace[slot : slot + 3])
        if move is not None:
            cand.append(move)
    return {
        "material": _MAT_LABEL[MAT_TOKENS.index(trace[V.SLOT_MAT])],
        "phase": trace[V.SLOT_PHASE].split(":")[1],
        "king_us": trace[V.SLOT_KSAFE_US].split(":")[1],
        "king_them": trace[V.SLOT_KSAFE_THEM].split(":")[1],
        "threats": int(trace[V.SLOT_THR].split(":")[1]),
        "tactics": [t.split(":")[1] for t in (trace[V.SLOT_TAC[0]], trace[V.SLOT_TAC[1]]) if t != "tac:none"],
        "plan": trace[V.SLOT_PLAN].split(":")[1],
        "candidates": cand,
        "best": tokens_to_move(trace[V.SLOT_BEST : V.SLOT_BEST + 3]),
        "eval": (EVAL_TOKENS.index(trace[V.SLOT_EVAL]) / (len(EVAL_TOKENS) - 1)) * 2 - 1,
    }


def describe(trace, flipped: bool = False) -> str:
    """One-line human-readable rendering of a trace.

    ``flipped`` un-mirrors the moves so they read in real board coordinates.
    """
    from .encoding import canonical_move

    info = parse_trace(trace)

    def uci(move):
        return canonical_move(move, flipped).uci() if move is not None else "-"

    tactics = ",".join(info["tactics"]) or "none"
    cands = " ".join(uci(m) for m in info["candidates"]) or "-"
    return (
        f"material {info['material']} | {info['phase']} | "
        f"king us={info['king_us']} them={info['king_them']} | "
        f"hanging {info['threats']} | tactics {tactics} | plan {info['plan']} | "
        f"candidates {cands} | best {uci(info['best'])} | eval {info['eval']:+.2f}"
    )
