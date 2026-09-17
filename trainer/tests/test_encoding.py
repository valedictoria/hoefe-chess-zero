"""The move index and board encoding are the contract the Rust engine mirrors.

If anything here changes, `spec/golden.json` must be regenerated and the Rust
core re-validated, or the two halves will silently disagree.
"""

import random

import chess
import pytest

from chesszero.encoding import (
    N_MOVES,
    canonical_board,
    canonical_move,
    encode_prefix,
    index_to_move,
    legal_move_indices,
    move_to_index,
)
from chesszero.vocab import PREFIX_LEN


def random_positions(n, seed, max_plies=80):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        board = chess.Board()
        for _ in range(rng.randint(0, max_plies)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        if not board.is_game_over():
            out.append(board)
    return out


def test_every_legal_move_round_trips():
    checked = 0
    for board in random_positions(150, seed=7):
        canon = canonical_board(board)
        moves, indices = legal_move_indices(canon)
        assert len(set(indices)) == len(indices), f"index collision in {board.fen()}"
        for move, index in zip(moves, indices):
            assert 0 <= index < N_MOVES
            assert index_to_move(index, canon) == move
            checked += 1
    assert checked > 3000


def test_canonical_move_is_an_involution():
    for board in random_positions(60, seed=11):
        flipped = board.turn == chess.BLACK
        for move in board.legal_moves:
            assert canonical_move(canonical_move(move, flipped), flipped) == move


def test_canonicalisation_always_yields_white_to_move():
    for board in random_positions(60, seed=13):
        assert canonical_board(board).turn == chess.WHITE


def test_black_to_move_mirrors_onto_the_equivalent_white_position():
    white = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    black = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
    assert encode_prefix(canonical_board(white)) == encode_prefix(canonical_board(black))


def test_prefix_length_is_fixed():
    for board in random_positions(40, seed=17):
        assert len(encode_prefix(canonical_board(board))) == PREFIX_LEN


@pytest.mark.parametrize("uci", ["e7e8q", "e7e8n", "e7e8b", "e7e8r"])
def test_promotions_get_distinct_indices(uci):
    board = chess.Board("7k/4P3/8/8/8/8/8/4K3 w - - 0 1")
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves
    assert index_to_move(move_to_index(move), board) == move


#: Random play never produces a 7-square slide or an underpromotion capture, so
#: the rare planes need positions built on purpose.  These were found by search
#: rather than by hand -- a corner queen is easily blocked by its own king, and
#: a pawn on the seventh is easily pinned to answering a check.
PLANE_CORNER_CASES = [
    "8/8/8/8/8/8/2K1k3/Q7 w - - 0 1",      # queen a1: N, NE, E at distance 7
    "Q7/8/8/8/8/8/8/1K1k4 w - - 0 1",      # queen a8: E, SE, S
    "8/8/8/8/8/8/K1k5/7Q w - - 0 1",       # queen h1: N, W, NW
    "7Q/8/8/8/8/8/8/1K1k4 w - - 0 1",      # queen h8: S, SW, W
    "r1r5/1P6/8/8/8/8/8/4K2k w - - 0 1",   # b7 pawn: all nine underpromotions
]


def test_all_73_planes_are_reachable():
    seen = set()
    for board in random_positions(200, seed=23):
        seen.update(i % 73 for i in legal_move_indices(canonical_board(board))[1])
    for fen in PLANE_CORNER_CASES:
        seen.update(i % 73 for i in legal_move_indices(canonical_board(chess.Board(fen)))[1])
    missing = sorted(set(range(73)) - seen)
    assert not missing, f"policy planes never exercised: {missing}"


def test_underpromotion_captures_round_trip():
    board = chess.Board("r1r5/1P6/8/8/8/8/8/4K2k w - - 0 1")
    promos = [m for m in board.legal_moves if m.promotion]
    assert len(promos) == 12  # three destination files x four pieces
    indices = [move_to_index(m) for m in promos]
    assert len(set(indices)) == 12
    for move, index in zip(promos, indices):
        assert index_to_move(index, board) == move
