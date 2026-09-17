//! Position features the annotator writes into a trace.
//!
//! A faithful port of the feature half of the trainer's `reasoning.py`. Where a
//! constant looks arbitrary it is not: it is whatever the trainer used, and the
//! golden test fails the moment the two disagree.

use shakmaty::attacks;
use shakmaty::{Bitboard, Board, Chess, Color, File, Piece, Position, Rank, Role, Square};

use crate::vocab::{KSAFE_TOKENS, MAT_EDGES, PHASE_TOKENS};

/// Material values in pawns. The king is priced at zero so it does not distort
/// a material count.
pub fn piece_value(role: Role) -> f64 {
    match role {
        Role::Pawn => 1.0,
        Role::Knight => 3.0,
        Role::Bishop => 3.25,
        Role::Rook => 5.0,
        Role::Queen => 9.0,
        Role::King => 0.0,
    }
}

/// Values for tactical comparisons, where the king must outrank everything:
/// pricing it at zero would make it look like the cheapest attacker available,
/// and a defended piece attacked only by the enemy king is not hanging.
pub fn tactical_value(role: Role) -> f64 {
    match role {
        Role::King => 100.0,
        other => piece_value(other),
    }
}

#[inline]
pub fn attackers(board: &Board, color: Color, sq: Square) -> Bitboard {
    board.attacks_to(sq, color, board.occupied())
}

#[inline]
pub fn is_attacked_by(board: &Board, color: Color, sq: Square) -> bool {
    !attackers(board, color, sq).is_empty()
}

#[inline]
pub fn pieces(board: &Board, role: Role, color: Color) -> Bitboard {
    board.by_piece(Piece { color, role })
}

/// Attacks of whatever piece stands on `sq`, empty if the square is empty.
pub fn attacks_from(board: &Board, sq: Square) -> Bitboard {
    match board.piece_at(sq) {
        Some(piece) => attacks::attacks(sq, piece, board.occupied()),
        None => Bitboard::EMPTY,
    }
}

/// Is the piece on `sq` pinned against its own king?
///
/// Implemented as a reveal test: lift the piece and see whether a new enemy
/// attacker of the king appears. Only a sole blocker on the ray can do that,
/// which is exactly what a pin is.
pub fn is_pinned(board: &Board, color: Color, sq: Square) -> bool {
    let Some(king) = board.king_of(color) else {
        return false;
    };
    if king == sq {
        return false;
    }
    let occupied = board.occupied();
    let before = board.attacks_to(king, !color, occupied);
    let after = board.attacks_to(king, !color, occupied.without_const(Bitboard::from_square(sq)));
    !after.without_const(before).is_empty()
}

/// Material in pawns, positive when the side to move is ahead.
pub fn material_balance(board: &Board) -> f64 {
    let mut total = 0.0;
    for sq in board.occupied() {
        let piece = board.piece_at(sq).expect("occupied square holds a piece");
        let value = piece_value(piece.role);
        total += if piece.color == Color::White { value } else { -value };
    }
    total
}

pub fn material_bucket(balance: f64) -> usize {
    MAT_EDGES.iter().filter(|e| balance >= **e as f64).count()
}

pub fn phase(board: &Board) -> &'static str {
    let mut non_pawn = 0.0;
    for sq in board.occupied() {
        let piece = board.piece_at(sq).expect("occupied square holds a piece");
        if !matches!(piece.role, Role::Pawn | Role::King) {
            non_pawn += piece_value(piece.role);
        }
    }
    if non_pawn >= 55.0 {
        PHASE_TOKENS[0]
    } else if non_pawn >= 22.0 {
        PHASE_TOKENS[1]
    } else {
        PHASE_TOKENS[2]
    }
}

pub fn king_safety(board: &Board, color: Color) -> &'static str {
    let Some(king) = board.king_of(color) else {
        return KSAFE_TOKENS[3];
    };
    let kf = u32::from(king.file()) as i32;
    let kr = u32::from(king.rank()) as i32;
    let forward = if color == Color::White { 1 } else { -1 };

    let mut shield = 0i32;
    for df in -1..=1 {
        let (f, r) = (kf + df, kr + forward);
        if (0..8).contains(&f) && (0..8).contains(&r) {
            let sq = Square::from_coords(File::new(f as u32), Rank::new(r as u32));
            if board.piece_at(sq) == Some(Piece { color, role: Role::Pawn }) {
                shield += 1;
            }
        }
    }

    let mut danger = 0i32;
    for df in -2..=2 {
        for dr in -2..=2 {
            let (f, r) = (kf + df, kr + dr);
            if (0..8).contains(&f) && (0..8).contains(&r) {
                let sq = Square::from_coords(File::new(f as u32), Rank::new(r as u32));
                if is_attacked_by(board, !color, sq) {
                    danger += 1;
                }
            }
        }
    }

    let open_file = !(0..8).any(|r| {
        let sq = Square::from_coords(king.file(), Rank::new(r));
        board.piece_at(sq) == Some(Piece { color, role: Role::Pawn })
    });

    let score = danger - 3 * shield + if open_file { 3 } else { 0 };
    if score <= 0 {
        KSAFE_TOKENS[0]
    } else if score <= 5 {
        KSAFE_TOKENS[1]
    } else if score <= 10 {
        KSAFE_TOKENS[2]
    } else {
        KSAFE_TOKENS[3]
    }
}

/// Squares holding pieces of `color` that are attacked and not adequately defended.
pub fn hanging_pieces(board: &Board, color: Color) -> Vec<Square> {
    let mut out = Vec::new();
    for sq in board.by_color(color) {
        let piece = board.piece_at(sq).expect("occupied square holds a piece");
        if piece.role == Role::King {
            continue;
        }
        let attacking = attackers(board, !color, sq);
        if attacking.is_empty() {
            continue;
        }
        if attackers(board, color, sq).is_empty() {
            out.push(sq);
            continue;
        }
        let cheapest = attacking
            .into_iter()
            .map(|a| tactical_value(board.role_at(a).expect("attacker exists")))
            .fold(f64::INFINITY, f64::min);
        if cheapest + 0.5 < piece_value(piece.role) {
            out.push(sq);
        }
    }
    out
}

const BACK_RANK_MINORS: [Square; 4] = [Square::B1, Square::C1, Square::F1, Square::G1];

pub fn plan(pos: &Chess, balance: f64, phase_token: &str, threats: usize) -> &'static str {
    let board = pos.board();
    if pos.is_check() || threats >= 2 {
        return "plan:defend";
    }
    if phase_token == PHASE_TOKENS[2] {
        if balance >= 2.0 {
            return "plan:convert";
        }
        if !pieces(board, Role::Pawn, Color::White).is_empty() {
            return "plan:push";
        }
        return "plan:hold";
    }

    let rights = pos.castles().castling_rights();
    let can_castle = rights.contains(Square::H1) || rights.contains(Square::A1);
    if can_castle && king_safety(board, Color::White) != KSAFE_TOKENS[0] {
        return "plan:castle";
    }

    let undeveloped = BACK_RANK_MINORS
        .iter()
        .filter(|sq| {
            matches!(
                board.piece_at(**sq),
                Some(Piece { color: Color::White, role: Role::Knight })
                    | Some(Piece { color: Color::White, role: Role::Bishop })
            )
        })
        .count();
    if phase_token == PHASE_TOKENS[0] && undeveloped >= 2 {
        return "plan:develop";
    }

    let their_king = king_safety(board, Color::Black);
    if their_king == KSAFE_TOKENS[2] || their_king == KSAFE_TOKENS[3] {
        return "plan:attack";
    }
    if balance >= 1.5 {
        return "plan:trade";
    }

    let rooks = pieces(board, Role::Rook, Color::White);
    if rooks.count() >= 2
        && !rooks
            .into_iter()
            .any(|r| !(attacks_from(board, r) & rooks).is_empty())
    {
        return "plan:rooks";
    }
    if balance <= -1.5 {
        return "plan:hold";
    }
    "plan:center"
}

/// Banker's rounding, matching Python's `round`, which rounds halves to even.
fn round_half_even(x: f64) -> f64 {
    let rounded = x.round();
    if (x - x.trunc()).abs() == 0.5 && rounded % 2.0 != 0.0 {
        rounded - x.signum()
    } else {
        rounded
    }
}

/// Map a value in [-1, 1] from the mover's point of view onto an EVAL bucket.
pub fn eval_bucket(q: f64) -> usize {
    let q = q.clamp(-1.0, 1.0);
    round_half_even((q + 1.0) * 0.5 * 10.0) as usize
}
