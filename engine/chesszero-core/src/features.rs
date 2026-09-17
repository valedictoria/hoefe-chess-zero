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

// --------------------------------------------------------------------------
// Shared structural features
//
// The motif detectors and the engine's evaluation want the same facts: is this
// pawn passed, is that file open, how much can this piece move. Writing them
// twice guarantees the two drift, and the drift is silent -- the detector says
// "passed pawn" while the evaluation scores it as backward, and nothing fails.
//
// So they live here once, colour-generic, and both callers consume them. The
// motif detectors call them with White because they run on canonical positions;
// the evaluation calls them with both colours.
// --------------------------------------------------------------------------

/// Rank as the given colour sees it: 0 is that colour's back rank, 7 promotion.
///
/// Writing every feature in terms of this instead of absolute ranks is what
/// makes one implementation serve both sides without a mirrored copy.
#[inline]
pub fn relative_rank(color: Color, sq: Square) -> i32 {
    let rank = u32::from(sq.rank()) as i32;
    if color == Color::White {
        rank
    } else {
        7 - rank
    }
}

#[inline]
fn file_of(sq: Square) -> i32 {
    u32::from(sq.file()) as i32
}

/// Is `other` further up the board than `sq`, from `color`'s point of view?
#[inline]
fn ahead_of(color: Color, other: Square, sq: Square) -> bool {
    relative_rank(color, other) > relative_rank(color, sq)
}

/// Pawns of `color` with no enemy pawn ahead on their own or an adjacent file.
pub fn passed_pawns(board: &Board, color: Color) -> Bitboard {
    let theirs = pieces(board, Role::Pawn, !color);
    pieces(board, Role::Pawn, color)
        .into_iter()
        .filter(|sq| {
            !theirs.into_iter().any(|enemy| {
                (file_of(enemy) - file_of(*sq)).abs() <= 1 && ahead_of(color, enemy, *sq)
            })
        })
        .collect()
}

/// Pawns of `color` sharing a file with another friendly pawn.
pub fn doubled_pawns(board: &Board, color: Color) -> Bitboard {
    let pawns = pieces(board, Role::Pawn, color);
    pawns
        .into_iter()
        .filter(|sq| pawns.into_iter().filter(|p| p.file() == sq.file()).count() >= 2)
        .collect()
}

/// Pawns of `color` with no friendly pawn on either adjacent file.
pub fn isolated_pawns(board: &Board, color: Color) -> Bitboard {
    let pawns = pieces(board, Role::Pawn, color);
    let files: Vec<i32> = pawns.into_iter().map(file_of).collect();
    pawns
        .into_iter()
        .filter(|sq| {
            let f = file_of(*sq);
            !files.contains(&(f - 1)) && !files.contains(&(f + 1))
        })
        .collect()
}

/// No pawn of either colour stands on this file.
pub fn is_open_file(board: &Board, file: File) -> bool {
    !board
        .by_role(Role::Pawn)
        .into_iter()
        .any(|p| p.file() == file)
}

/// No pawn of `color` stands on this file, though the enemy may have one.
pub fn is_semi_open_file(board: &Board, color: Color, file: File) -> bool {
    !pieces(board, Role::Pawn, color)
        .into_iter()
        .any(|p| p.file() == file)
}

/// Knights of `color` standing on an outpost: advanced, defended by a friendly
/// pawn, and beyond the reach of any enemy pawn.
pub fn outposts(board: &Board, color: Color) -> Bitboard {
    let theirs = pieces(board, Role::Pawn, !color);
    pieces(board, Role::Knight, color)
        .into_iter()
        .filter(|sq| {
            if !(3..=5).contains(&relative_rank(color, *sq)) {
                return false;
            }
            let pawn_defended = attackers(board, color, *sq)
                .into_iter()
                .any(|a| board.role_at(a) == Some(Role::Pawn));
            if !pawn_defended {
                return false;
            }
            !theirs
                .into_iter()
                .any(|enemy| (file_of(enemy) - file_of(*sq)).abs() == 1 && ahead_of(color, enemy, *sq))
        })
        .collect()
}

/// Squares a piece can reach that are neither occupied by its own side nor
/// covered by an enemy pawn. Squares an enemy pawn attacks do not count: a
/// knight cannot usefully stand where a pawn may simply take it.
pub fn mobility(board: &Board, color: Color, sq: Square) -> u32 {
    let own = board.by_color(color);
    let pawn_covered = pawn_attack_span(board, !color);
    attacks_from(board, sq)
        .without_const(own)
        .without_const(pawn_covered)
        .count() as u32
}

/// Every square attacked by a pawn of `color`.
pub fn pawn_attack_span(board: &Board, color: Color) -> Bitboard {
    pieces(board, Role::Pawn, color)
        .into_iter()
        .map(|sq| shakmaty::attacks::pawn_attacks(color, sq))
        .fold(Bitboard::EMPTY, |acc, b| acc | b)
}

/// The king of `color` and the squares around it.
pub fn king_zone(board: &Board, color: Color) -> Bitboard {
    match board.king_of(color) {
        Some(king) => shakmaty::attacks::king_attacks(king) | Bitboard::from_square(king),
        None => Bitboard::EMPTY,
    }
}

/// How many pieces of `attacker` bear on the enemy king zone, and how heavily.
///
/// Returns (attacker count, weighted pressure). The weights are the usual
/// ordering -- a queen near the king matters far more than a knight -- and the
/// count matters because a single attacker is rarely dangerous on its own.
pub fn king_attack_pressure(board: &Board, attacker: Color) -> (u32, u32) {
    let zone = king_zone(board, !attacker);
    if zone.is_empty() {
        return (0, 0);
    }
    let mut count = 0;
    let mut weight = 0;
    for sq in board.by_color(attacker) {
        let Some(role) = board.role_at(sq) else { continue };
        if role == Role::Pawn || role == Role::King {
            continue;
        }
        let hits = (attacks_from(board, sq) & zone).count() as u32;
        if hits > 0 {
            count += 1;
            weight += hits
                * match role {
                    Role::Knight => 2,
                    Role::Bishop => 2,
                    Role::Rook => 3,
                    Role::Queen => 5,
                    _ => 0,
                };
        }
    }
    (count, weight)
}
