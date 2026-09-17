//! Evaluation tests.
//!
//! The symmetry test is the important one. Almost every hand-crafted evaluation
//! bug is an asymmetry -- a term written for White and mirrored wrongly, a rank
//! used absolutely where it should be relative -- and every one of them shows up
//! as the engine quietly believing one colour is better than it is. Mirroring a
//! position swaps the colours *and* who is to move, so an evaluation reported
//! from the mover's point of view must come back identical.

use chesszero_search::eval::{evaluate, evaluate_detailed, phase};
use chesszero_search::score::{Score, MAX_PHASE};
use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess, EnPassantMode, FromSetup, Position};

fn parse(fen: &str) -> Chess {
    fen.parse::<Fen>().unwrap().into_position(CastlingMode::Standard).unwrap()
}

fn mirrored(pos: &Chess) -> Chess {
    let setup = pos.to_setup(EnPassantMode::Always).into_mirrored();
    Chess::from_setup(setup, CastlingMode::Standard).expect("mirror of a legal position is legal")
}

fn random_positions(n: usize, seed: u64) -> Vec<Chess> {
    // xorshift; a dependency for this would be silly
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut out = Vec::new();
    while out.len() < n {
        let mut pos = Chess::default();
        let plies = (next() % 80) as usize;
        for _ in 0..plies {
            let moves = pos.legal_moves();
            if moves.is_empty() {
                break;
            }
            let m = moves[(next() as usize) % moves.len()];
            pos.play_unchecked(m);
        }
        if !pos.legal_moves().is_empty() {
            out.push(pos);
        }
    }
    out
}

#[test]
fn evaluation_is_colour_symmetric() {
    for pos in random_positions(400, 0xC0FFEE) {
        let ours = evaluate(&pos);
        let theirs = evaluate(&mirrored(&pos));
        assert_eq!(
            ours,
            theirs,
            "asymmetric evaluation at {}: {ours} vs {theirs} mirrored",
            Fen::from_position(&pos, EnPassantMode::Always)
        );
    }
}

#[test]
fn every_term_is_colour_symmetric() {
    // Narrows a symmetry failure to the term that caused it.
    for pos in random_positions(200, 0xBEEF) {
        let a = evaluate_detailed(&pos);
        let b = evaluate_detailed(&mirrored(&pos));
        let fen = Fen::from_position(&pos, EnPassantMode::Always);
        assert_eq!(a.material, b.material, "material asymmetric at {fen}");
        assert_eq!(a.placement, b.placement, "placement asymmetric at {fen}");
        assert_eq!(a.mobility, b.mobility, "mobility asymmetric at {fen}");
        assert_eq!(a.pawn_structure, b.pawn_structure, "pawn structure asymmetric at {fen}");
        assert_eq!(a.piece_quality, b.piece_quality, "piece quality asymmetric at {fen}");
        assert_eq!(a.king_safety, b.king_safety, "king safety asymmetric at {fen}");
    }
}

#[test]
fn starting_position_is_level_apart_from_tempo() {
    let start = evaluate_detailed(&Chess::default());
    assert_eq!(start.material, 0);
    assert_eq!(start.placement, 0);
    assert_eq!(start.mobility, 0);
    assert_eq!(start.pawn_structure, 0);
    assert_eq!(start.piece_quality, 0);
    assert_eq!(start.king_safety, 0);
    assert_eq!(start.total, start.tempo, "only tempo should break the tie");
    assert!(start.tempo > 0 && start.tempo < 30, "tempo is {}", start.tempo);
}

#[test]
fn breakdown_sums_to_total() {
    for pos in random_positions(200, 0xD00D) {
        let b = evaluate_detailed(&pos);
        let sum = b.material
            + b.placement
            + b.mobility
            + b.pawn_structure
            + b.piece_quality
            + b.king_safety
            + b.tempo;
        assert_eq!(sum, b.total, "breakdown does not sum to the total it reports");
    }
}

#[test]
fn material_dominates() {
    // A queen up is worth more than any positional term can plausibly offset.
    let up_a_queen = parse("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    assert!(evaluate(&up_a_queen) > 700, "{}", evaluate(&up_a_queen));
    let down_a_queen = parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1");
    assert!(evaluate(&down_a_queen) < -700, "{}", evaluate(&down_a_queen));
}

#[test]
fn phase_falls_as_material_comes_off() {
    assert_eq!(phase(Chess::default().board()), MAX_PHASE);
    let no_queens = parse("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1");
    assert_eq!(phase(no_queens.board()), MAX_PHASE - 8);
    let bare = parse("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    assert_eq!(phase(bare.board()), 0);
}

#[test]
fn tapering_has_no_cliff() {
    // The bug this replaced: the king table switched at a material threshold, so
    // one capture could move the score tens of centipawns with nothing else
    // changing. Interpolation must be continuous in the phase.
    let s = Score::new(100, -100);
    let mut previous = s.taper(0);
    for phase in 1..=MAX_PHASE {
        let current = s.taper(phase);
        assert!(
            (current - previous).abs() <= 200 / MAX_PHASE + 1,
            "phase {phase} jumped from {previous} to {current}"
        );
        previous = current;
    }
    assert_eq!(s.taper(MAX_PHASE), 100);
    assert_eq!(s.taper(0), -100);
}

#[test]
fn passed_pawn_is_worth_more_in_an_endgame() {
    // Same passer, queens on versus queens off.
    let middlegame = parse("r2q1rk1/1p3ppp/8/2P5/8/8/1P3PPP/R2Q1RK1 w - - 0 1");
    let endgame = parse("r4rk1/1p3ppp/8/2P5/8/8/1P3PPP/R4RK1 w - - 0 1");
    let mg = evaluate_detailed(&middlegame).pawn_structure;
    let eg = evaluate_detailed(&endgame).pawn_structure;
    assert!(eg > mg, "passer worth {mg} with queens on, {eg} with them off");
}
