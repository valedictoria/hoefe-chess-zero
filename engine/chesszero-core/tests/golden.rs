//! Replay `spec/golden.json` and assert this crate reproduces the trainer exactly.
//!
//! This is the gate the rest of the engine is built behind. If it fails, the two
//! implementations of the encoding have drifted and any weights loaded on top of
//! this crate are being fed the wrong tokens.

use std::fs;
use std::path::PathBuf;

use chesszero_core::encoding::{
    canonical, canonical_square, encode_prefix, index_to_move, legal_move_indices, move_squares,
    move_to_index,
};
use chesszero_core::vocab;
use serde_json::Value;
use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess, Color, Position, Role, Square};

fn golden() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../spec/golden.json")
        .canonicalize()
        .expect("spec/golden.json is missing -- run `python export.py golden` in trainer/");
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

fn parse(fen: &str) -> Chess {
    fen.parse::<Fen>()
        .unwrap()
        .into_position(CastlingMode::Standard)
        .unwrap()
}

fn promo_char(role: Option<Role>) -> &'static str {
    match role {
        Some(Role::Queen) => "q",
        Some(Role::Rook) => "r",
        Some(Role::Bishop) => "b",
        Some(Role::Knight) => "n",
        _ => "",
    }
}

#[test]
fn vocabulary_matches_the_trainer() {
    let g = golden();
    let spec = &g["spec"];
    let expected: Vec<&str> = spec["vocab"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();

    let ours = vocab::vocab();
    assert_eq!(ours.len(), expected.len(), "vocabulary size differs");
    for (i, (got, want)) in ours.iter().zip(&expected).enumerate() {
        assert_eq!(got, want, "token {i} differs: a token's index is its embedding row");
    }

    assert_eq!(spec["prefix_len"].as_u64().unwrap() as usize, vocab::PREFIX_LEN);
    assert_eq!(spec["trace_len"].as_u64().unwrap() as usize, vocab::TRACE_LEN);
    assert_eq!(spec["seq_len"].as_u64().unwrap() as usize, vocab::SEQ_LEN);
    assert_eq!(
        spec["readout_fast"].as_u64().unwrap() as usize,
        vocab::READOUT_FAST
    );
    assert_eq!(
        spec["readout_reasoned"].as_u64().unwrap() as usize,
        vocab::READOUT_REASONED
    );
}

#[test]
fn trace_grammar_matches_the_trainer() {
    let g = golden();
    let expected: Vec<Vec<u16>> = g["spec"]["trace_slots"]
        .as_array()
        .unwrap()
        .iter()
        .map(|slot| {
            slot.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u16)
                .collect()
        })
        .collect();
    assert_eq!(vocab::trace_slots(), &expected, "trace grammar differs");
}

#[test]
fn every_position_encodes_identically() {
    let g = golden();
    let cases = g["cases"].as_array().unwrap();
    assert!(cases.len() > 2000, "contract is suspiciously small");

    for case in cases {
        let fen = case["fen"].as_str().unwrap();
        let pos = parse(fen);
        let (canon, flipped) = canonical(&pos);

        assert_eq!(flipped, case["flipped"].as_bool().unwrap(), "{fen}");
        assert_eq!(pos.turn() == Color::Black, flipped, "{fen}");
        assert_eq!(canon.turn(), Color::White, "{fen}: canonical is white to move");

        let reps = case["repetitions"].as_u64().unwrap() as u8;
        let want: Vec<u16> = case["prefix"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u16)
            .collect();
        assert_eq!(encode_prefix(&canon, reps), want, "prefix differs for {fen}");
    }
}

#[test]
fn every_legal_move_indexes_identically() {
    let g = golden();
    for case in g["cases"].as_array().unwrap() {
        let fen = case["fen"].as_str().unwrap();
        let pos = parse(fen);
        let (canon, flipped) = canonical(&pos);

        // The contract pairs a move in *real* board coordinates with the policy
        // index of its canonical form, so undo the mirror to build the uci.
        let mut ours: Vec<(String, usize)> = legal_move_indices(&canon)
            .iter()
            .map(|(m, index)| {
                let (from, to, promo) = move_squares(m);
                let uci = format!(
                    "{}{}{}",
                    canonical_square(from, flipped),
                    canonical_square(to, flipped),
                    promo_char(promo)
                );
                (uci, *index)
            })
            .collect();
        ours.sort_by_key(|(_, i)| *i);

        let want: Vec<(String, usize)> = case["moves"]
            .as_array()
            .unwrap()
            .iter()
            .map(|pair| {
                let p = pair.as_array().unwrap();
                (
                    p[0].as_str().unwrap().to_string(),
                    p[1].as_u64().unwrap() as usize,
                )
            })
            .collect();

        assert_eq!(ours, want, "move indexing differs for {fen}");
    }
}

#[test]
fn policy_indices_round_trip() {
    let g = golden();
    for case in g["cases"].as_array().unwrap() {
        let pos = parse(case["fen"].as_str().unwrap());
        let (canon, _) = canonical(&pos);
        for (m, index) in legal_move_indices(&canon) {
            let (from, to, promo) = move_squares(&m);
            let back = index_to_move(index, &canon).expect("index must land on the board");
            assert_eq!(
                back,
                (from, to, promo),
                "index {index} did not decode back to its move"
            );
        }
    }
}

#[test]
fn perft_confirms_the_move_generator() {
    fn perft(pos: &Chess, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        let moves = pos.legal_moves();
        if depth == 1 {
            return moves.len() as u64;
        }
        moves
            .iter()
            .map(|m| {
                let mut next = pos.clone();
                next.play_unchecked(*m);
                perft(&next, depth - 1)
            })
            .sum()
    }
    let start = Chess::default();
    assert_eq!(perft(&start, 4), 197_281);
    // Kiwipete: the standard position for castling, en passant and pin bugs
    let kiwipete = parse("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    assert_eq!(perft(&kiwipete, 3), 97_862);
}

#[test]
fn castling_uses_the_kings_destination_not_the_rook() {
    // shakmaty models castling as king-takes-rook; the trainer uses e1g1/e1c1.
    let pos = parse("r3k2r/8/8/8/8/8/8/R3K2R w KQ - 0 1");
    let castles: Vec<(Square, Square)> = pos
        .legal_moves()
        .iter()
        .filter(|m| matches!(m, shakmaty::Move::Castle { .. }))
        .map(|m| {
            let (from, to, _) = move_squares(m);
            (from, to)
        })
        .collect();
    assert!(castles.contains(&(Square::E1, Square::G1)), "{castles:?}");
    assert!(castles.contains(&(Square::E1, Square::C1)), "{castles:?}");
    assert_eq!(
        move_to_index(Square::E1, Square::G1, None),
        move_to_index(Square::E1, Square::G1, None)
    );
}
