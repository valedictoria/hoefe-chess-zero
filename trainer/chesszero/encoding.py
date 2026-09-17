"""Board <-> token encoding and move <-> policy-index encoding.

Two conventions are used everywhere in this package:

**Canonical orientation.**  Every position is shown to the network from the
side to move's point of view.  If black is to move the board is mirrored
(flipped vertically *and* colour-swapped), so as far as the network is
concerned it is always white's turn.  This halves what the network has to
learn and is what Lc0 does too.

**Policy indexing.**  Moves live in an 8x8x73 = 4672 slot AlphaZero-style
policy: 56 "queen" planes (8 directions x 7 distances), 8 knight planes and 9
underpromotion planes (3 files x knight/bishop/rook).  Queen promotions ride
along on the plain queen move.  Indices are always computed on the *canonical*
move.
"""

from __future__ import annotations

import chess

from .vocab import (
    BOS,
    CASTLE_TOKENS,
    EMPTY_TOKEN,
    EP_TOKENS,
    R50_TOKENS,
    REP_TOKENS,
    SEP,
    PREFIX_LEN,
    tid,
)

N_MOVES = 4672
N_PLANES = 73

# (file delta, rank delta) for N, NE, E, SE, S, SW, W, NW
QUEEN_DIRS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
QUEEN_DIR_INDEX = {d: i for i, d in enumerate(QUEEN_DIRS)}

KNIGHT_DIRS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
KNIGHT_DIR_INDEX = {d: i for i, d in enumerate(KNIGHT_DIRS)}

UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
UNDERPROMO_INDEX = {p: i for i, p in enumerate(UNDERPROMO_PIECES)}

_PIECE_CHAR = {
    chess.PAWN: "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK: "R",
    chess.QUEEN: "Q",
    chess.KING: "K",
}


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


# --------------------------------------------------------------------------
# canonicalisation
# --------------------------------------------------------------------------
def canonical_board(board: chess.Board) -> chess.Board:
    """Return ``board`` seen from the side to move (always white to move)."""
    return board if board.turn == chess.WHITE else board.mirror()


def canonical_move(move: chess.Move, flipped: bool) -> chess.Move:
    """Map a move into (or out of) canonical orientation.

    The mapping is an involution, so the same function undoes it.
    """
    if not flipped:
        return move
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
        drop=move.drop,
    )


# --------------------------------------------------------------------------
# moves <-> policy indices
# --------------------------------------------------------------------------
def move_to_index(move: chess.Move) -> int:
    """Policy index of a *canonical* move (white to move)."""
    frm, to = move.from_square, move.to_square
    df = chess.square_file(to) - chess.square_file(frm)
    dr = chess.square_rank(to) - chess.square_rank(frm)

    if move.promotion is not None and move.promotion != chess.QUEEN:
        try:
            piece_idx = UNDERPROMO_INDEX[move.promotion]
        except KeyError:  # pragma: no cover - python-chess never produces these
            raise ValueError(f"cannot encode promotion {move}") from None
        if dr != 1 or df not in (-1, 0, 1):
            raise ValueError(f"cannot encode promotion {move}")
        plane = 64 + piece_idx * 3 + (df + 1)
    elif (abs(df), abs(dr)) in ((1, 2), (2, 1)):
        plane = 56 + KNIGHT_DIR_INDEX[(df, dr)]
    else:
        direction = (_sign(df), _sign(dr))
        if direction not in QUEEN_DIR_INDEX:
            raise ValueError(f"cannot encode move {move}")
        distance = max(abs(df), abs(dr))
        if distance > 7 or (df and dr and abs(df) != abs(dr)):
            raise ValueError(f"cannot encode move {move}")
        plane = QUEEN_DIR_INDEX[direction] * 7 + distance - 1

    return frm * N_PLANES + plane


def index_to_move(index: int, board: chess.Board | None = None) -> chess.Move | None:
    """Inverse of :func:`move_to_index`.

    ``board`` (canonical) is only needed to tell a queen promotion apart from a
    plain pawn push to the last rank.  Returns ``None`` when the index does not
    describe a move that fits on the board.
    """
    frm, plane = divmod(int(index), N_PLANES)
    file_, rank = chess.square_file(frm), chess.square_rank(frm)

    promotion = None
    if plane < 56:
        direction, distance = divmod(plane, 7)
        df, dr = QUEEN_DIRS[direction]
        distance += 1
        df, dr = df * distance, dr * distance
    elif plane < 64:
        df, dr = KNIGHT_DIRS[plane - 56]
    else:
        rest = plane - 64
        promotion = UNDERPROMO_PIECES[rest // 3]
        df, dr = (rest % 3) - 1, 1

    tf, tr = file_ + df, rank + dr
    if not (0 <= tf < 8 and 0 <= tr < 8):
        return None
    to = chess.square(tf, tr)

    if promotion is None and tr == 7 and board is not None:
        piece = board.piece_at(frm)
        if piece is not None and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
            promotion = chess.QUEEN
    return chess.Move(frm, to, promotion=promotion)


def legal_move_indices(canon_board: chess.Board) -> tuple[list[chess.Move], list[int]]:
    """Legal moves of a canonical board together with their policy indices."""
    moves = list(canon_board.legal_moves)
    return moves, [move_to_index(m) for m in moves]


# --------------------------------------------------------------------------
# board -> prefix tokens
# --------------------------------------------------------------------------
def _castle_code(board: chess.Board) -> int:
    rights = board.castling_rights
    code = 0
    if bool(rights & chess.BB_H1):
        code |= 1
    if bool(rights & chess.BB_A1):
        code |= 2
    if bool(rights & chess.BB_H8):
        code |= 4
    if bool(rights & chess.BB_A8):
        code |= 8
    return code


def _r50_bucket(halfmove_clock: int) -> int:
    return min(7, max(0, halfmove_clock) // 13)


def repetition_count(board: chess.Board) -> int:
    """0, 1 or 2: how many times this position has occurred before."""
    if board.is_repetition(3):
        return 2
    if board.is_repetition(2):
        return 1
    return 0


def encode_prefix(canon_board: chess.Board, repetitions: int = 0) -> list[int]:
    """Token ids for the position part of the sequence (ends with ``<sep>``)."""
    out = [tid(BOS)]
    piece_map = canon_board.piece_map()
    for square in range(64):
        piece = piece_map.get(square)
        if piece is None:
            out.append(tid(EMPTY_TOKEN))
        else:
            char = _PIECE_CHAR[piece.piece_type]
            out.append(tid("sq:" + (char if piece.color == chess.WHITE else char.lower())))

    out.append(tid(CASTLE_TOKENS[_castle_code(canon_board)]))
    # Only record an en passant square that can actually be captured on.  A
    # double push always sets `ep_square`, but if no capture is legal the square
    # is not part of the position: encoding it would split two identical
    # positions into different tokens, and it would not survive a round trip
    # through `fen()`, which omits it for exactly the same reason.
    ep = canon_board.ep_square if canon_board.has_legal_en_passant() else None
    out.append(tid(EP_TOKENS[0] if ep is None else EP_TOKENS[1 + chess.square_file(ep)]))
    out.append(tid(R50_TOKENS[_r50_bucket(canon_board.halfmove_clock)]))
    out.append(tid(REP_TOKENS[min(2, max(0, repetitions))]))
    out.append(tid(SEP))
    assert len(out) == PREFIX_LEN
    return out
