"""Motif detectors are the validation oracle, so their honesty is the thing under test.

A false negative costs us a motif we could have taught.  A false positive tells a
beginner something untrue about the board, which is the failure this whole design
exists to prevent -- so the negative cases below matter more than the positive ones.
"""

import random

import chess
import pytest

from chesszero.encoding import canonical_board, canonical_move
from chesszero.reasoning import (
    MOTIF_DETECTORS,
    MOTIF_PRIORITY,
    annotate,
    describe,
    detect_motifs,
    motif_tags,
    parse_trace,
)
from chesszero.vocab import MOTIF_TOKENS, N_MOTIF_SLOTS, TRACE_LEN, TRACE_SLOTS

from .test_encoding import random_positions


def test_every_motif_token_has_a_detector():
    """A motif with no detector is a claim nothing can check."""
    assert set(MOTIF_DETECTORS) | {"mo:none", "mo:quiet"} == set(MOTIF_TOKENS)
    assert set(MOTIF_PRIORITY) == set(MOTIF_DETECTORS)


FIRES = [
    ("mo:backrank", "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"),
    ("mo:mate1", "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"),
    ("mo:passer", "8/8/8/3P4/8/7k/8/K7 w - - 0 1"),
    ("mo:rook7th", "8/2R5/8/8/8/7k/8/K7 w - - 0 1"),
    ("mo:doubled", "8/8/8/8/8/3P3k/3P4/K7 w - - 0 1"),
    ("mo:isolated", "8/8/8/8/8/3P3k/3P4/K7 w - - 0 1"),
    ("mo:openfile", "7k/8/8/8/8/8/8/K2R4 w - - 0 1"),
    ("mo:promotion", "8/3P4/8/8/8/7k/8/K7 w - - 0 1"),
    ("mo:skewer", "r7/8/8/q7/8/7k/8/R5K1 w - - 0 1"),
    ("mo:pin", "4k3/8/8/8/8/4n3/8/4R1K1 w - - 0 1"),
    ("mo:battery", "7k/8/8/8/8/8/3Q4/K2R4 w - - 0 1"),
    ("mo:outpost", "4k3/8/8/3N4/4P3/8/8/4K3 w - - 0 1"),
]


@pytest.mark.parametrize("motif,fen", FIRES, ids=[f"{m}-{i}" for i, (m, _) in enumerate(FIRES)])
def test_detector_fires_where_it_should(motif, fen):
    assert MOTIF_DETECTORS[motif](chess.Board(fen))


def test_starting_position_is_quiet():
    """Regression: back-rank and bad-bishop both used to fire on move one.

    The starting position has a boxed-in king and four pawns on each colour, but
    neither is a motif -- no heavy piece can reach the back rank, and four pawns
    is not a majority.  Naming either would teach a beginner a phantom.
    """
    assert detect_motifs(chess.Board()) == []
    assert motif_tags(chess.Board()) == ["mo:quiet", "mo:none", "mo:none"]


def test_x_ray_through_a_knight_is_not_a_skewer():
    """Regression: the Ruy Lopez bishop used to read as a skewer.

    Bb5 x-rays Nc6 with a pawn on d7 behind it.  That is not a skewer: the piece
    in front is not valuable enough to have to move, and what is behind is a pawn.
    """
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bb5"):
        board.push_san(san)
    assert not MOTIF_DETECTORS["mo:skewer"](board)
    assert detect_motifs(board) == []


def test_motif_tags_are_padded_and_ordered():
    for board in random_positions(120, seed=31):
        tags = motif_tags(board if board.turn == chess.WHITE else canonical_board(board))
        assert len(tags) == N_MOTIF_SLOTS
        if "mo:none" in tags:  # padding only ever trails
            first = tags.index("mo:none")
            assert all(t == "mo:none" for t in tags[first:])
        assert "mo:quiet" not in tags or tags[0] == "mo:quiet"


def test_annotate_always_produces_a_grammatical_trace():
    rng = random.Random(41)
    for board in random_positions(120, seed=43):
        canon = canonical_board(board)
        flipped = board.turn == chess.BLACK
        moves = [canonical_move(m, flipped) for m in list(board.legal_moves)[:3]]
        trace = annotate(canon, best_move=moves[0], candidates=moves, q=rng.uniform(-1, 1))
        assert len(trace) == TRACE_LEN
        for i, (token, allowed) in enumerate(zip(trace, TRACE_SLOTS)):
            assert token in allowed, f"slot {i}: {token} not permitted"
        assert parse_trace(trace)["best"] == moves[0]
        assert describe(trace, flipped)


def test_trace_survives_the_round_trip_into_real_coordinates():
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2")
    canon = canonical_board(board)
    real = chess.Move.from_uci("b8c6")
    trace = annotate(canon, best_move=canonical_move(real, True), q=0.0)
    assert canonical_move(parse_trace(trace)["best"], True) == real
    assert "b8c6" in describe(trace, flipped=True)
