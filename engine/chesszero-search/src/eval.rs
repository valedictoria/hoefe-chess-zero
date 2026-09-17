//! Hand-crafted evaluation: material, placement, mobility, pawn structure,
//! piece quality and king safety, each tapered between middlegame and endgame.
//!
//! Two jobs, and the second is easy to overlook. The first is to *teach the
//! network*: the plan generates training data with alpha-beta rather than
//! self-play, because a randomly initialised network playing itself on a handful
//! of cores learns nothing in any useful time, while this produces dense targets
//! immediately. Better evaluation here means better labels there.
//!
//! The second is that a hand-crafted evaluation *decomposes*, and a neural value
//! head does not. A network says "+0.4"; this says "material level, king safety
//! -0.6, pawn structure +0.2". For a tool whose purpose is teaching a person,
//! that breakdown is the more useful output of the two, and [`Breakdown`] is
//! what the explanation layer will read.
//!
//! Structural features come from `chesszero_core::features` rather than being
//! written again here. The motif detectors need the same facts -- is this pawn
//! passed, is that file open -- and two implementations of one definition drift
//! silently: the detector names a passed pawn while the evaluation scores it as
//! backward, and no test fails.

use chesszero_core::features::{
    doubled_pawns, is_open_file, is_semi_open_file, isolated_pawns, king_attack_pressure,
    mobility, outposts, passed_pawns, pieces, relative_rank,
};
use shakmaty::{Board, Chess, Color, Position, Role, Square};

use crate::score::{Score, MAX_PHASE};

// Material. Pawns and rooks gain in the endgame, bishops slightly so.
pub const MATERIAL: [(Role, Score); 5] = [
    (Role::Pawn, Score::new(100, 120)),
    (Role::Knight, Score::new(320, 320)),
    (Role::Bishop, Score::new(330, 350)),
    (Role::Rook, Score::new(500, 530)),
    (Role::Queen, Score::new(900, 950)),
];

/// Flat centipawn values, for move ordering and see-like comparisons where a
/// phase-aware score would be more precision than the caller can use.
pub fn piece_value(role: Role) -> i32 {
    match role {
        Role::Pawn => 100,
        Role::Knight => 320,
        Role::Bishop => 330,
        Role::Rook => 500,
        Role::Queen => 900,
        Role::King => 0,
    }
}

fn material_score(role: Role) -> Score {
    MATERIAL
        .iter()
        .find(|(r, _)| *r == role)
        .map(|(_, s)| *s)
        .unwrap_or(Score::ZERO)
}

// --- positional weights ---------------------------------------------------
/// Passed-pawn bonus by relative rank. Worth far more in an endgame, where
/// there is nothing left to stop it.
const PASSED_PAWN: [Score; 8] = [
    Score::new(0, 0),
    Score::new(5, 10),
    Score::new(10, 25),
    Score::new(20, 45),
    Score::new(35, 80),
    Score::new(60, 130),
    Score::new(100, 200),
    Score::new(0, 0),
];

const DOUBLED_PAWN: Score = Score::new(-10, -22);
const ISOLATED_PAWN: Score = Score::new(-14, -20);
const BISHOP_PAIR: Score = Score::new(30, 50);
const ROOK_OPEN_FILE: Score = Score::new(26, 14);
const ROOK_SEMI_OPEN_FILE: Score = Score::new(12, 8);
const ROOK_ON_SEVENTH: Score = Score::new(20, 32);
const KNIGHT_OUTPOST: Score = Score::new(26, 14);
const TEMPO: Score = Score::new(12, 4);

/// Mobility is weighted per piece and centred on a typical count, so a piece
/// with nowhere to go is penalised rather than merely unrewarded.
fn mobility_weight(role: Role) -> (Score, i32) {
    match role {
        Role::Knight => (Score::new(4, 4), 4),
        Role::Bishop => (Score::new(5, 5), 6),
        Role::Rook => (Score::new(2, 4), 7),
        Role::Queen => (Score::new(1, 2), 13),
        _ => (Score::ZERO, 0),
    }
}

// Tables are written the way a board is drawn: rank 8 first, a-file on the left.
// Indexing flips them, which is far easier to get right than writing them upside
// down would be.
#[rustfmt::skip]
const PAWN_PST: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
];

#[rustfmt::skip]
const KNIGHT_PST: [i32; 64] = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
];

#[rustfmt::skip]
const BISHOP_PST: [i32; 64] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
];

#[rustfmt::skip]
const ROOK_PST: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
];

#[rustfmt::skip]
const QUEEN_PST: [i32; 64] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
];

#[rustfmt::skip]
const KING_MIDDLEGAME_PST: [i32; 64] = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
];

#[rustfmt::skip]
const KING_ENDGAME_PST: [i32; 64] = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
];


#[rustfmt::skip]
const PAWN_PST_EG: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
    90, 90, 90, 90, 90, 90, 90, 90,
    55, 55, 55, 55, 55, 55, 55, 55,
    30, 30, 30, 30, 30, 30, 30, 30,
    18, 18, 18, 18, 18, 18, 18, 18,
     8,  8,  8,  8,  8,  8,  8,  8,
     5,  5,  5,  5,  5,  5,  5,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
];

/// Read a drawn-board table for a piece of `color` standing on `sq`.
#[inline]
fn pst(table: &[i32; 64], sq: Square, color: Color) -> i32 {
    let file = u32::from(sq.file()) as usize;
    let rank = u32::from(sq.rank()) as usize;
    let row = if color == Color::White { 7 - rank } else { rank };
    table[row * 8 + file]
}

/// How far into the game we are: MAX_PHASE with everything on, 0 with none of
/// it left. Promotions can overshoot, so it is clamped.
pub fn phase(board: &Board) -> i32 {
    let mut phase = 0;
    for sq in board.occupied() {
        phase += match board.role_at(sq) {
            Some(Role::Knight) | Some(Role::Bishop) => 1,
            Some(Role::Rook) => 2,
            Some(Role::Queen) => 4,
            _ => 0,
        };
    }
    phase.min(MAX_PHASE)
}

/// Where an evaluation came from, for explaining it to a person.
///
/// Every field is centipawns from the side to move's point of view, so a
/// negative `king_safety` means the mover's own king is the one in trouble.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Breakdown {
    pub material: i32,
    pub placement: i32,
    pub mobility: i32,
    pub pawn_structure: i32,
    pub piece_quality: i32,
    pub king_safety: i32,
    pub tempo: i32,
    pub total: i32,
    pub phase: i32,
}

fn placement_table(role: Role) -> (&'static [i32; 64], &'static [i32; 64]) {
    match role {
        Role::Pawn => (&PAWN_PST, &PAWN_PST_EG),
        Role::Knight => (&KNIGHT_PST, &KNIGHT_PST),
        Role::Bishop => (&BISHOP_PST, &BISHOP_PST),
        Role::Rook => (&ROOK_PST, &ROOK_PST),
        Role::Queen => (&QUEEN_PST, &QUEEN_PST),
        // The king's two tables are exactly the middlegame/endgame pair the
        // taper wants, so the old threshold switch disappears for free.
        Role::King => (&KING_MIDDLEGAME_PST, &KING_ENDGAME_PST),
    }
}

/// Sum a term over both colours, white positive.
fn both<F: Fn(Color) -> Score>(f: F) -> Score {
    f(Color::White) - f(Color::Black)
}

fn material_and_placement(board: &Board) -> (Score, Score) {
    let mut material = Score::ZERO;
    let mut placement = Score::ZERO;
    for sq in board.occupied() {
        let Some(piece) = board.piece_at(sq) else { continue };
        let sign = if piece.color == Color::White { 1 } else { -1 };
        material += material_score(piece.role) * sign;
        let (mg, eg) = placement_table(piece.role);
        placement += Score::new(pst(mg, sq, piece.color), pst(eg, sq, piece.color)) * sign;
    }
    (material, placement)
}

fn mobility_score(board: &Board, color: Color) -> Score {
    let mut total = Score::ZERO;
    for sq in board.by_color(color) {
        let Some(role) = board.role_at(sq) else { continue };
        let (weight, baseline) = mobility_weight(role);
        if weight == Score::ZERO {
            continue;
        }
        total += weight * (mobility(board, color, sq) as i32 - baseline);
    }
    total
}

fn pawn_structure(board: &Board, color: Color) -> Score {
    let mut total = Score::ZERO;
    for sq in passed_pawns(board, color) {
        total += PASSED_PAWN[relative_rank(color, sq) as usize];
    }
    total += DOUBLED_PAWN * doubled_pawns(board, color).count() as i32;
    total += ISOLATED_PAWN * isolated_pawns(board, color).count() as i32;
    total
}

fn piece_quality(board: &Board, color: Color) -> Score {
    let mut total = Score::ZERO;

    if pieces(board, Role::Bishop, color).count() >= 2 {
        total += BISHOP_PAIR;
    }
    for rook in pieces(board, Role::Rook, color) {
        if is_open_file(board, rook.file()) {
            total += ROOK_OPEN_FILE;
        } else if is_semi_open_file(board, color, rook.file()) {
            total += ROOK_SEMI_OPEN_FILE;
        }
        if relative_rank(color, rook) == 6 {
            total += ROOK_ON_SEVENTH;
        }
    }
    total += KNIGHT_OUTPOST * outposts(board, color).count() as i32;
    total
}

/// Pressure on the enemy king, from `color`'s point of view.
///
/// A lone attacker is almost never dangerous, so the term stays at zero until
/// at least two pieces bear on the zone; past that it grows faster than
/// linearly, because attackers combine.
fn king_safety(board: &Board, color: Color) -> Score {
    let (attackers, weight) = king_attack_pressure(board, color);
    if attackers < 2 {
        return Score::ZERO;
    }
    let pressure = (weight * weight / 12).min(400) as i32;
    // Almost entirely a middlegame concern: with the queens off, a king in the
    // open is an asset rather than a liability.
    Score::new(pressure, pressure / 6)
}

/// Centipawns from the side to move's point of view.
pub fn evaluate(pos: &Chess) -> i32 {
    evaluate_detailed(pos).total
}

/// The same evaluation, itemised.
pub fn evaluate_detailed(pos: &Chess) -> Breakdown {
    let board = pos.board();
    let phase = phase(board);
    let (material, placement) = material_and_placement(board);

    let mobility = both(|c| mobility_score(board, c));
    let pawns = both(|c| pawn_structure(board, c));
    let quality = both(|c| piece_quality(board, c));
    let safety = both(|c| king_safety(board, c));

    // Everything above is white-positive; flip once, at the end.
    let sign = if pos.turn() == Color::White { 1 } else { -1 };
    let term = |s: Score| s.taper(phase) * sign;

    let breakdown = Breakdown {
        material: term(material),
        placement: term(placement),
        mobility: term(mobility),
        pawn_structure: term(pawns),
        piece_quality: term(quality),
        king_safety: term(safety),
        tempo: TEMPO.taper(phase),
        total: 0,
        phase,
    };

    Breakdown {
        total: breakdown.material
            + breakdown.placement
            + breakdown.mobility
            + breakdown.pawn_structure
            + breakdown.piece_quality
            + breakdown.king_safety
            + breakdown.tempo,
        ..breakdown
    }
}
