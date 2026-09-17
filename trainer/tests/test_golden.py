"""The encoding contract the Rust engine is validated against.

These tests are Python checking its own output, which sounds circular but is not:
each case is replayed *from its FEN*, the way the engine will see it, rather than
from the live board it was generated on.  That distinction already caught a real
bug -- `fen()` drops an en passant square when no capture is legal, so positions
encoded from a live board did not match themselves after a round trip.

If a test here fails after an encoding change, that is the contract working.
Regenerate with `python export.py golden` and re-run the Rust side.
"""

import json
from pathlib import Path

import chess
import pytest

from chesszero import vocab as V
from chesszero.encoding import (
    canonical_board,
    canonical_move,
    encode_prefix,
    legal_move_indices,
)
from chesszero.reasoning import annotate, detect_motifs

GOLDEN_PATH = Path(__file__).resolve().parents[2] / "spec" / "golden.json"


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN_PATH.exists():
        pytest.fail(f"{GOLDEN_PATH} missing -- run `python export.py golden`")
    return json.loads(GOLDEN_PATH.read_text())


def test_spec_block_matches_the_live_vocabulary(golden):
    """If this drifts, every trained checkpoint is silently invalidated."""
    spec = golden["spec"]
    assert spec["vocab"] == V.VOCAB
    assert spec["vocab_size"] == V.VOCAB_SIZE
    assert spec["prefix_len"] == V.PREFIX_LEN
    assert spec["trace_len"] == V.TRACE_LEN
    assert spec["seq_len"] == V.SEQ_LEN
    assert spec["readout_fast"] == V.READOUT_FAST
    assert spec["readout_reasoned"] == V.READOUT_REASONED
    assert spec["trace_slots"] == V.TRACE_SLOT_IDS
    assert spec["motif_tokens"] == V.MOTIF_TOKENS


def test_every_case_replays_from_its_fen(golden):
    from export import deterministic_trace_inputs

    for case in golden["cases"]:
        board = chess.Board(case["fen"])
        canon = canonical_board(board)
        flipped = case["flipped"]

        assert (board.turn == chess.BLACK) == flipped, case["fen"]
        assert encode_prefix(canon, case["repetitions"]) == case["prefix"], case["fen"]

        moves, indices = legal_move_indices(canon)
        replayed = sorted(
            ([canonical_move(m, flipped).uci(), i] for m, i in zip(moves, indices)),
            key=lambda pair: pair[1],
        )
        assert replayed == [list(p) for p in case["moves"]], case["fen"]

        best, candidates, q = deterministic_trace_inputs(canon)
        trace = annotate(canon, best_move=best, candidates=candidates, q=q)
        assert V.ids(trace) == case["trace"], case["fen"]
        assert detect_motifs(canon) == case["motifs"], case["fen"]


def test_traces_obey_the_grammar(golden):
    for case in golden["cases"]:
        assert len(case["trace"]) == V.TRACE_LEN
        for slot, (token_id, allowed) in enumerate(zip(case["trace"], V.TRACE_SLOT_IDS)):
            assert token_id in allowed, f"{case['fen']} slot {slot}"


def test_policy_indices_are_in_range_and_unique(golden):
    for case in golden["cases"]:
        indices = [i for _, i in case["moves"]]
        assert len(set(indices)) == len(indices), case["fen"]
        assert all(0 <= i < 4672 for i in indices), case["fen"]


def test_contract_keeps_its_coverage(golden):
    """Guards against a future regeneration quietly sampling a narrower world."""
    cases = golden["cases"]
    prefixes = [c["prefix"] for c in cases]
    castle_idx, ep_idx, r50_idx, rep_idx = 65, 66, 67, 68

    assert len({p[castle_idx] for p in prefixes}) == 16, "all castling codes"
    assert len({p[r50_idx] for p in prefixes}) == 8, "all rule-50 buckets"
    assert len({p[rep_idx] for p in prefixes}) == 3, "all repetition states"
    assert len({p[ep_idx] for p in prefixes}) >= 4, "several en passant files"

    fired = {m for c in cases for m in c["motifs"]}
    assert len(fired) == len(V.MOTIF_TOKENS) - 2, "every detectable motif appears"

    planes = {i % 73 for c in cases for _, i in c["moves"]}
    assert len(planes) == 73, "every policy plane appears"

    assert sum(1 for c in cases if c["flipped"]) > len(cases) // 4, "both colours to move"
