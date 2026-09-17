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
    MOTIF_TOKENS,
    N_MOTIF_SLOTS,
    SQUARE_TOKENS,
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
        # TACTICAL_VALUE, not PIECE_VALUE: the king is priced at 0 for material
        # counting, which would make it look like the cheapest attacker going.
        # A defended piece attacked only by the enemy king is not hanging --
        # the king is not allowed to take it.
        cheapest = min(TACTICAL_VALUE[board.piece_type_at(a)] for a in attackers)
        if cheapest + 0.5 < PIECE_VALUE[piece.piece_type]:
            out.append(square)
    return out


# --------------------------------------------------------------------------
# motif detectors
#
# Each of these answers one question about a canonical position, using nothing
# but the board.  They serve two masters: the annotator calls them to write
# teacher traces, and the teaching layer's validation gate calls them to decide
# whether a motif the *model* claimed is actually there.  That second use is the
# point -- a motif the model asserts and no detector confirms gets dropped.
#
# They are deliberately conservative.  A false negative costs us a motif we
# could have taught; a false positive teaches a beginner something untrue.
# --------------------------------------------------------------------------

#: piece values for tactical comparisons, with the king priced above everything
TACTICAL_VALUE = dict(PIECE_VALUE)
TACTICAL_VALUE[chess.KING] = 100.0

_DIAG_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_ORTHO_DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0))
_SLIDER_DIRS = {
    chess.BISHOP: _DIAG_DIRS,
    chess.ROOK: _ORTHO_DIRS,
    chess.QUEEN: _DIAG_DIRS + _ORTHO_DIRS,
}


def _is_fork(board: chess.Board, move: chess.Move) -> bool:
    """Does ``move`` attack two or more things the mover outvalues?"""
    piece = board.piece_type_at(move.from_square)
    if piece is None or piece == chess.KING:
        return False
    mover_value = TACTICAL_VALUE[piece]
    board.push(move)
    try:
        # a fork that simply hangs the forking piece is not a fork
        if board.is_attacked_by(chess.BLACK, move.to_square) and not board.attackers(
            chess.WHITE, move.to_square
        ):
            return False
        targets = 0
        for square in board.attacks(move.to_square):
            victim = board.piece_at(square)
            if victim is not None and victim.color == chess.BLACK:
                if TACTICAL_VALUE[victim.piece_type] > mover_value:
                    targets += 1
        return targets >= 2
    finally:
        board.pop()


def has_mate_in_1(board: chess.Board) -> bool:
    for move in board.legal_moves:
        if board.gives_check(move):
            board.push(move)
            mated = board.is_checkmate()
            board.pop()
            if mated:
                return True
    return False


def has_check(board: chess.Board) -> bool:
    return any(board.gives_check(m) for m in board.legal_moves)


def has_capture(board: chess.Board) -> bool:
    return any(board.is_capture(m) for m in board.legal_moves)


def has_hanging(board: chess.Board) -> bool:
    return bool(hanging_pieces(board, chess.BLACK))


def has_fork(board: chess.Board) -> bool:
    return any(_is_fork(board, m) for m in board.legal_moves)


def has_pin(board: chess.Board) -> bool:
    for piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
        for square in board.pieces(piece_type, chess.BLACK):
            if board.is_pinned(chess.BLACK, square):
                return True
    return False


def has_skewer(board: chess.Board) -> bool:
    """A white slider hitting a valuable black piece with a lesser one behind it."""
    for piece_type, dirs in _SLIDER_DIRS.items():
        for square in board.pieces(piece_type, chess.WHITE):
            f0, r0 = chess.square_file(square), chess.square_rank(square)
            for df, dr in dirs:
                f, r, front = f0 + df, r0 + dr, None
                while 0 <= f < 8 and 0 <= r < 8:
                    piece = board.piece_at(chess.square(f, r))
                    if piece is not None:
                        if piece.color == chess.WHITE:
                            break
                        if front is None:
                            front = piece
                        else:
                            # a real skewer: the front piece is valuable enough
                            # that it must move (rook, queen, king) and what is
                            # behind it is actually worth winning.  Without both
                            # tests a bishop x-raying a knight with a pawn behind
                            # reads as a skewer, which it is not.
                            front_value = TACTICAL_VALUE[front.piece_type]
                            back_value = TACTICAL_VALUE[piece.piece_type]
                            if front_value > back_value >= 3.0 and front_value >= 5.0:
                                return True
                            break
                    f += df
                    r += dr
    return False


def has_discovery(board: chess.Board) -> bool:
    """Is there a move that unmasks a *different* white piece onto king or queen?"""
    targets = list(board.pieces(chess.KING, chess.BLACK)) + list(
        board.pieces(chess.QUEEN, chess.BLACK)
    )
    if not targets:
        return False
    before = {t: set(board.attackers(chess.WHITE, t)) for t in targets}
    for move in board.legal_moves:
        board.push(move)
        try:
            for target in targets:
                if target == move.to_square:
                    continue
                revealed = {
                    a for a in board.attackers(chess.WHITE, target) if a != move.to_square
                } - before[target]
                if revealed:
                    return True
        finally:
            board.pop()
    return False


def has_backrank(board: chess.Board) -> bool:
    king = board.king(chess.BLACK)
    if king is None or chess.square_rank(king) != 7:
        return False
    king_file = chess.square_file(king)
    for df in (-1, 0, 1):
        f = king_file + df
        if not 0 <= f < 8:
            continue
        piece = board.piece_at(chess.square(f, 6))
        if piece is None or piece.color != chess.BLACK:
            return False  # the king has air
    # a boxed-in king is only a motif if we can actually get to the back rank --
    # otherwise every starting position would "have" a back-rank weakness
    heavy = board.pieces(chess.ROOK, chess.WHITE) | board.pieces(chess.QUEEN, chess.WHITE)
    return any(board.attacks(sq) & chess.BB_RANK_8 for sq in heavy)


def has_promotion(board: chess.Board) -> bool:
    return bool(board.pieces(chess.PAWN, chess.WHITE) & chess.BB_RANK_7)


def has_overload(board: chess.Board) -> bool:
    """A black piece that is the sole defender of two attacked black units."""
    duties: dict[int, int] = {}
    for square, piece in board.piece_map().items():
        if piece.color != chess.BLACK:
            continue
        if not board.attackers(chess.WHITE, square):
            continue
        defenders = board.attackers(chess.BLACK, square)
        if len(defenders) == 1:
            only = next(iter(defenders))
            duties[only] = duties.get(only, 0) + 1
    return any(count >= 2 for count in duties.values())


def has_trapped(board: chess.Board) -> bool:
    """An attacked black piece with no safe square to run to."""
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for square in board.pieces(piece_type, chess.BLACK):
            if not board.attackers(chess.WHITE, square):
                continue
            for target in board.attacks(square):
                occupant = board.piece_at(target)
                if occupant is not None and occupant.color == chess.BLACK:
                    continue
                if not board.is_attacked_by(chess.WHITE, target):
                    break
            else:
                return True
    return False


def has_passer(board: chess.Board) -> bool:
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)
    for square in board.pieces(chess.PAWN, chess.WHITE):
        f, r = chess.square_file(square), chess.square_rank(square)
        if not any(
            abs(chess.square_file(bp) - f) <= 1 and chess.square_rank(bp) > r
            for bp in black_pawns
        ):
            return True
    return False


def has_outpost(board: chess.Board) -> bool:
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)
    for square in board.pieces(chess.KNIGHT, chess.WHITE):
        f, r = chess.square_file(square), chess.square_rank(square)
        if not 3 <= r <= 5:
            continue
        if not any(
            board.piece_type_at(a) == chess.PAWN for a in board.attackers(chess.WHITE, square)
        ):
            continue
        if not any(
            abs(chess.square_file(bp) - f) == 1 and chess.square_rank(bp) > r
            for bp in black_pawns
        ):
            return True
    return False


def has_rook7th(board: chess.Board) -> bool:
    return bool(board.pieces(chess.ROOK, chess.WHITE) & chess.BB_RANK_7)


def has_openfile(board: chess.Board) -> bool:
    pawns = board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK)
    for rook in board.pieces(chess.ROOK, chess.WHITE):
        f = chess.square_file(rook)
        if not any(chess.square_file(p) == f for p in pawns):
            return True
    return False


def has_battery(board: chess.Board) -> bool:
    """Two white sliders stacked on the same line."""
    queens = board.pieces(chess.QUEEN, chess.WHITE)
    for group, same_line in (
        (list(board.pieces(chess.ROOK, chess.WHITE) | queens), _same_rank_or_file),
        (list(board.pieces(chess.BISHOP, chess.WHITE) | queens), _same_diagonal),
    ):
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                if same_line(a, b) and a in board.attackers(chess.WHITE, b):
                    return True
    return False


def _same_rank_or_file(a: int, b: int) -> bool:
    return chess.square_file(a) == chess.square_file(b) or chess.square_rank(a) == chess.square_rank(b)


def _same_diagonal(a: int, b: int) -> bool:
    return abs(chess.square_file(a) - chess.square_file(b)) == abs(
        chess.square_rank(a) - chess.square_rank(b)
    )


def has_kingattack(board: chess.Board) -> bool:
    king = board.king(chess.BLACK)
    if king is None:
        return False
    kf, kr = chess.square_file(king), chess.square_rank(king)
    attacked = 0
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            f, r = kf + df, kr + dr
            if 0 <= f < 8 and 0 <= r < 8 and board.is_attacked_by(chess.WHITE, chess.square(f, r)):
                attacked += 1
    return attacked >= 4


def has_badbishop(board: chess.Board) -> bool:
    pawns = board.pieces(chess.PAWN, chess.WHITE)
    for bishop in board.pieces(chess.BISHOP, chess.WHITE):
        light = (chess.square_file(bishop) + chess.square_rank(bishop)) % 2 == 1
        same_colour = sum(
            1 for p in pawns if ((chess.square_file(p) + chess.square_rank(p)) % 2 == 1) == light
        )
        if same_colour >= 5:
            return True
    return False


def has_doubled(board: chess.Board) -> bool:
    files = [chess.square_file(p) for p in board.pieces(chess.PAWN, chess.WHITE)]
    return any(files.count(f) >= 2 for f in set(files))


def has_isolated(board: chess.Board) -> bool:
    pawns = board.pieces(chess.PAWN, chess.WHITE)
    files = {chess.square_file(p) for p in pawns}
    return any(f - 1 not in files and f + 1 not in files for f in files)


def has_space(board: chess.Board) -> bool:
    ours = sum(1 for p in board.pieces(chess.PAWN, chess.WHITE) if chess.square_rank(p) >= 3)
    theirs = sum(1 for p in board.pieces(chess.PAWN, chess.BLACK) if chess.square_rank(p) <= 4)
    return ours >= theirs + 2


#: token -> detector.  This mapping *is* the validation oracle.
MOTIF_DETECTORS = {
    "mo:mate1": has_mate_in_1,
    "mo:check": has_check,
    "mo:capture": has_capture,
    "mo:hanging": has_hanging,
    "mo:fork": has_fork,
    "mo:pin": has_pin,
    "mo:skewer": has_skewer,
    "mo:discovery": has_discovery,
    "mo:backrank": has_backrank,
    "mo:promotion": has_promotion,
    "mo:overload": has_overload,
    "mo:trapped": has_trapped,
    "mo:passer": has_passer,
    "mo:outpost": has_outpost,
    "mo:rook7th": has_rook7th,
    "mo:openfile": has_openfile,
    "mo:battery": has_battery,
    "mo:kingattack": has_kingattack,
    "mo:badbishop": has_badbishop,
    "mo:doubled": has_doubled,
    "mo:isolated": has_isolated,
    "mo:space": has_space,
}

#: which motifs are worth naming first when several fire at once
MOTIF_PRIORITY = [
    "mo:mate1",
    "mo:fork",
    "mo:skewer",
    "mo:discovery",
    "mo:backrank",
    "mo:trapped",
    "mo:overload",
    "mo:pin",
    "mo:hanging",
    "mo:promotion",
    "mo:passer",
    "mo:rook7th",
    "mo:outpost",
    "mo:battery",
    "mo:kingattack",
    "mo:openfile",
    "mo:check",
    "mo:capture",
    "mo:badbishop",
    "mo:isolated",
    "mo:doubled",
    "mo:space",
]

assert set(MOTIF_PRIORITY) == set(MOTIF_DETECTORS), "priority list and detectors disagree"
assert set(MOTIF_DETECTORS) | {"mo:none", "mo:quiet"} == set(MOTIF_TOKENS), (
    "every motif token needs a detector"
)


def detect_motifs(board: chess.Board) -> list[str]:
    """Every motif that fires, in priority order."""
    return [m for m in MOTIF_PRIORITY if MOTIF_DETECTORS[m](board)]


def motif_tags(board: chess.Board) -> list[str]:
    """The ``N_MOTIF_SLOTS`` motifs a trace names, padded with ``mo:none``."""
    found = detect_motifs(board)[:N_MOTIF_SLOTS]
    if not found:
        found = ["mo:quiet"]
    return found + ["mo:none"] * (N_MOTIF_SLOTS - len(found))


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
    trace += ["f:TAC"] + motif_tags(canon_board)
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
        "motifs": [trace[i].split(":")[1] for i in V.SLOT_MOTIF if trace[i] != "mo:none"],
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

    motifs = ",".join(info["motifs"]) or "none"
    cands = " ".join(uci(m) for m in info["candidates"]) or "-"
    return (
        f"material {info['material']} | {info['phase']} | "
        f"king us={info['king_us']} them={info['king_them']} | "
        f"hanging {info['threats']} | motifs {motifs} | plan {info['plan']} | "
        f"candidates {cands} | best {uci(info['best'])} | eval {info['eval']:+.2f}"
    )
