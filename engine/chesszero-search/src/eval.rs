//! A classical evaluation: material plus piece-square tables.
//!
//! This exists to *teach the network*, not to be the engine. The plan generates
//! training data with alpha-beta rather than self-play because a randomly
//! initialised network playing itself on a handful of cores learns nothing in
//! any reasonable time, while this produces dense, immediately useful targets.

use shakmaty::{Board, Chess, Color, Position, Role, Square};

pub const PAWN: i32 = 100;
pub const KNIGHT: i32 = 320;
pub const BISHOP: i32 = 330;
pub const ROOK: i32 = 500;
pub const QUEEN: i32 = 900;

pub fn piece_value(role: Role) -> i32 {
    match role {
        Role::Pawn => PAWN,
        Role::Knight => KNIGHT,
        Role::Bishop => BISHOP,
        Role::Rook => ROOK,
        Role::Queen => QUEEN,
        Role::King => 0,
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

/// Read a drawn-board table for a piece of `color` standing on `sq`.
#[inline]
fn pst(table: &[i32; 64], sq: Square, color: Color) -> i32 {
    let file = u32::from(sq.file()) as usize;
    let rank = u32::from(sq.rank()) as usize;
    let row = if color == Color::White { 7 - rank } else { rank };
    table[row * 8 + file]
}

/// Non-pawn, non-king material on both sides, used to pick a king table.
pub fn non_pawn_material(board: &Board) -> i32 {
    board
        .occupied()
        .into_iter()
        .filter_map(|sq| board.role_at(sq))
        .filter(|r| !matches!(r, Role::Pawn | Role::King))
        .map(piece_value)
        .sum()
}

/// Centipawns from the side to move's point of view.
pub fn evaluate(pos: &Chess) -> i32 {
    let board = pos.board();
    let endgame = non_pawn_material(board) < 2 * (ROOK + BISHOP);

    let mut score = 0;
    for sq in board.occupied() {
        let piece = board.piece_at(sq).expect("occupied square holds a piece");
        let table = match piece.role {
            Role::Pawn => &PAWN_PST,
            Role::Knight => &KNIGHT_PST,
            Role::Bishop => &BISHOP_PST,
            Role::Rook => &ROOK_PST,
            Role::Queen => &QUEEN_PST,
            Role::King => {
                if endgame {
                    &KING_ENDGAME_PST
                } else {
                    &KING_MIDDLEGAME_PST
                }
            }
        };
        let value = piece_value(piece.role) + pst(table, sq, piece.color);
        score += if piece.color == Color::White { value } else { -value };
    }

    if pos.turn() == Color::White {
        score
    } else {
        -score
    }
}
