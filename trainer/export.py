"""Export the two things the Rust engine needs from Python: weights, and the contract.

``golden.json`` is the more important of the two.  The engine re-implements the
encoding -- canonicalisation, the 4672-slot move index, the annotator, the trace
grammar -- and two implementations of one spec drift.  When they do, the failure
is silent: Rust loads Python's weights, feeds them subtly different tokens, and
plays badly for reasons no unit test explains.

So Python writes down exactly what it produces for a few thousand positions and
Rust asserts it reproduces them byte for byte.  Anything that drifts fails loudly
and immediately, pointing at the field that moved.

    python export.py golden --out ../spec/golden.json
    python export.py weights --checkpoint run/net.pt --out ../spec/net.safetensors
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import chess

from chesszero import vocab as V
from chesszero.encoding import (
    canonical_board,
    canonical_move,
    encode_prefix,
    legal_move_indices,
    move_to_index,
    repetition_count,
)
from chesszero.reasoning import annotate, detect_motifs, material_balance

GOLDEN_VERSION = 1

#: Positions that random play will not produce but the encoding must still get
#: right: maximum-distance slides, every underpromotion, en passant, castling
#: rights in each combination, and a handful of decided tactical shapes.
HANDPICKED = [
    chess.STARTING_FEN,
    # long rays in all eight directions
    "8/8/8/8/8/8/2K1k3/Q7 w - - 0 1",
    "Q7/8/8/8/8/8/8/1K1k4 w - - 0 1",
    "8/8/8/8/8/8/K1k5/7Q w - - 0 1",
    "7Q/8/8/8/8/8/8/1K1k4 w - - 0 1",
    # all nine underpromotions, both colours to move
    "r1r5/1P6/8/8/8/8/8/4K2k w - - 0 1",
    "4k2K/8/8/8/8/8/1p6/R1R5 b - - 0 1",
    # en passant, both sides
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "rnbqkbnr/pppp1ppp/8/8/3PpP2/8/PPP1P1PP/RNBQKBNR b KQkq f3 0 3",
    # every castling-rights combination worth sampling
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w Kq - 0 1",
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b Qk - 0 1",
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1",
    # tactical and endgame shapes that light up the detectors
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "4k3/8/8/3N4/4P3/8/8/4K3 w - - 0 1",
    "r7/8/8/q7/8/7k/8/R5K1 w - - 0 1",
    "4k3/8/8/8/8/4n3/8/4R1K1 w - - 0 1",
    "8/2R5/8/8/8/7k/8/K7 w - - 0 1",
    "8/8/8/3P4/8/7k/8/K7 w - - 0 1",
    # more en passant, including a pin that makes the capture illegal (so the
    # square must NOT be encoded) next to the same shape where it is legal
    "rnbqkbnr/1ppppppp/8/p7/P7/8/1PPPPPPP/RNBQKBNR w KQkq - 0 2",
    "4k3/8/8/2pP4/8/8/8/4K3 w - c6 0 2",
    "4k3/8/8/8/2Pp4/8/8/4K3 b - c3 0 2",
]

#: One quiet endgame at eight halfmove clocks, one per r50 bucket.  Random play
#: almost never survives long enough without a capture or pawn move to reach the
#: high buckets, so they have to be asked for.
HANDPICKED += [
    f"8/8/4k3/8/8/4K3/8/7R w - - {clock} {60 + clock}"
    for clock in (0, 13, 26, 39, 52, 65, 78, 91)
]


def sample_positions(n_random: int, seed: int) -> list[tuple[chess.Board, int]]:
    """Hand-picked positions plus random play, each with its repetition count.

    The repetition count travels with the position because a FEN carries no
    history -- Rust cannot recompute it, so it has to be told.
    """
    out: list[tuple[chess.Board, int]] = []
    for fen in HANDPICKED:
        board = chess.Board(fen)
        out.append((board, repetition_count(board)))

    # Repetitions never show up in random play, but the rep token is part of the
    # encoding, so shuffle knights back and forth to exercise it.  The count
    # travels in the case because no FEN can carry it.
    shuffler = chess.Board()
    for san in ("Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"):
        shuffler.push_san(san)
        out.append((shuffler.copy(), repetition_count(shuffler)))

    rng = random.Random(seed)
    while len(out) < len(HANDPICKED) + n_random:
        board = chess.Board()
        for _ in range(rng.randint(0, 120)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
            # sample mid-game as well as at the end of the walk
            if rng.random() < 0.06 and not board.is_game_over():
                out.append((board.copy(), repetition_count(board)))
    return out[: len(HANDPICKED) + n_random]


def deterministic_trace_inputs(canon: chess.Board) -> tuple[chess.Move | None, list[chess.Move], float]:
    """Pick best/candidate moves and a value the same way in any language.

    Ordering by policy index rather than by the move generator's order matters:
    `shakmaty` and python-chess do not enumerate moves in the same sequence, so
    "the first three legal moves" would differ across the two implementations
    and the traces would disagree for no real reason.
    """
    moves, indices = legal_move_indices(canon)
    if not moves:
        return None, [], 0.0
    ordered = [m for _, m in sorted(zip(indices, moves), key=lambda p: p[0])]
    q = math.tanh(material_balance(canon) / 5.0)
    return ordered[0], ordered[:3], round(q, 6)


def build_case(board: chess.Board, repetitions: int) -> dict:
    canon = canonical_board(board)
    flipped = board.turn == chess.BLACK
    prefix = encode_prefix(canon, repetitions)
    moves, indices = legal_move_indices(canon)

    best, candidates, q = deterministic_trace_inputs(canon)
    trace = annotate(canon, best_move=best, candidates=candidates, q=q)

    return {
        "fen": board.fen(),
        "flipped": flipped,
        "repetitions": repetitions,
        "prefix": prefix,
        # canonical uci -> policy index, sorted so the file is stable
        "moves": sorted(
            ([canonical_move(m, flipped).uci(), i] for m, i in zip(moves, indices)),
            key=lambda pair: pair[1],
        ),
        "trace_input": {
            "best": best.uci() if best else None,
            "candidates": [m.uci() for m in candidates],
            "q": q,
        },
        "trace": V.ids(trace),
        "motifs": detect_motifs(canon),
    }


def write_golden(path: Path, n_random: int, seed: int) -> dict:
    cases = [build_case(b, r) for b, r in sample_positions(n_random, seed)]
    blob = {
        "version": GOLDEN_VERSION,
        "spec": {
            "vocab_size": V.VOCAB_SIZE,
            "prefix_len": V.PREFIX_LEN,
            "trace_len": V.TRACE_LEN,
            "seq_len": V.SEQ_LEN,
            "n_motif_slots": V.N_MOTIF_SLOTS,
            "readout_fast": V.READOUT_FAST,
            "readout_reasoned": V.READOUT_REASONED,
            "n_moves": 4672,
            # the full ordered vocabulary, so the engine can assert its own
            # token ids line up rather than discovering it during training
            "vocab": V.VOCAB,
            "trace_slots": V.TRACE_SLOT_IDS,
            "motif_tokens": V.MOTIF_TOKENS,
        },
        "cases": cases,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, separators=(",", ":")))
    return blob


def write_weights(checkpoint: Path, out: Path) -> None:
    import torch
    from safetensors.torch import save_file

    from chesszero.model import ChessZeroNet

    model, _ = ChessZeroNet.load(checkpoint)
    tensors = {k: v.contiguous() for k, v in model.state_dict().items()}
    metadata = {k: str(v) for k, v in model.cfg.to_dict().items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata=metadata)
    total = sum(t.numel() for t in tensors.values())
    print(f"wrote {out} ({len(tensors)} tensors, {total:,} parameters)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("golden", help="write the encoding contract")
    g.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "spec" / "golden.json")
    g.add_argument("--positions", type=int, default=2000)
    g.add_argument("--seed", type=int, default=20260917)

    w = sub.add_parser("weights", help="write safetensors weights")
    w.add_argument("--checkpoint", type=Path, required=True)
    w.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "golden":
        blob = write_golden(args.out, args.positions, args.seed)
        size = args.out.stat().st_size
        print(f"wrote {args.out} ({len(blob['cases'])} cases, {size / 1e6:.1f} MB)")
    else:
        write_weights(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
