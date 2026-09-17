//! The 22 motif detectors, and the oracle the teaching layer validates against.
//!
//! These answer one question each about a canonical position (white is always the
//! side to move) using nothing but the board. They serve two masters: the
//! annotator calls them to write teacher traces, and the teaching layer calls
//! them to decide whether a motif the *model* claimed is really there. That
//! second use is the point -- a motif the model asserts and no detector confirms
//! gets dropped rather than shown.
//!
//! They are deliberately conservative. A false negative costs a motif we could
//! have taught; a false positive tells a beginner something untrue.

use shakmaty::{Bitboard, Chess, Color, File, Move, Position, Rank, Role, Square};

use crate::encoding::move_squares;
use crate::features::{
    attackers, attacks_from, hanging_pieces, is_attacked_by, is_pinned, pieces, tactical_value,
};
use crate::vocab::N_MOTIF_SLOTS;

/// Motifs in the order they are worth naming when several fire at once.
pub const MOTIF_PRIORITY: [&str; 22] = [
    "mo:mate1",
    "mo:fork",
    "mo:skewer",
    "mo:discovery",
    "mo:backrank",
    "mo:trapped",
    "mo:overload",
    "mo:pin",
    "mo:hanging",
    "mo:promotion",
    "mo:passer",
    "mo:rook7th",
    "mo:outpost",
    "mo:battery",
    "mo:kingattack",
    "mo:openfile",
    "mo:check",
    "mo:capture",
    "mo:badbishop",
    "mo:isolated",
    "mo:doubled",
    "mo:space",
];

fn file_i(sq: Square) -> i32 {
    u32::from(sq.file()) as i32
}

fn rank_i(sq: Square) -> i32 {
    u32::from(sq.rank()) as i32
}

pub fn has_mate_in_1(pos: &Chess) -> bool {
    pos.legal_moves().iter().any(|m| {
        let mut next = pos.clone();
        next.play_unchecked(*m);
        next.is_checkmate()
    })
}

pub fn has_check(pos: &Chess) -> bool {
    pos.legal_moves().iter().any(|m| {
        let mut next = pos.clone();
        next.play_unchecked(*m);
        next.is_check()
    })
}

pub fn has_capture(pos: &Chess) -> bool {
    pos.legal_moves().iter().any(|m| m.is_capture())
}

pub fn has_hanging(pos: &Chess) -> bool {
    !hanging_pieces(pos.board(), Color::Black).is_empty()
}

/// Does this move attack two or more things the mover outvalues?
fn is_fork(pos: &Chess, m: &Move) -> bool {
    let (from, to, _) = move_squares(m);
    let Some(role) = pos.board().role_at(from) else {
        return false;
    };
    if role == Role::King {
        return false;
    }
    let mover_value = tactical_value(role);

    let mut next = pos.clone();
    next.play_unchecked(*m);
    let board = next.board();

    // a fork that simply hangs the forking piece is not a fork
    if is_attacked_by(board, Color::Black, to) && attackers(board, Color::White, to).is_empty() {
        return false;
    }
    let targets = attacks_from(board, to)
        .into_iter()
        .filter(|sq| match board.piece_at(*sq) {
            Some(p) if p.color == Color::Black => tactical_value(p.role) > mover_value,
            _ => false,
        })
        .count();
    targets >= 2
}

pub fn has_fork(pos: &Chess) -> bool {
    pos.legal_moves().iter().any(|m| is_fork(pos, m))
}

pub fn has_pin(pos: &Chess) -> bool {
    let board = pos.board();
    for role in [Role::Queen, Role::Rook, Role::Bishop, Role::Knight] {
        for sq in pieces(board, role, Color::Black) {
            if is_pinned(board, Color::Black, sq) {
                return true;
            }
        }
    }
    false
}

const DIAG_DIRS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const ORTHO_DIRS: [(i32, i32); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];

fn slider_dirs(role: Role) -> &'static [(i32, i32)] {
    match role {
        Role::Bishop => &DIAG_DIRS,
        Role::Rook => &ORTHO_DIRS,
        _ => &[(1, 1), (1, -1), (-1, 1), (-1, -1), (0, 1), (0, -1), (1, 0), (-1, 0)],
    }
}

/// A white slider hitting a valuable black piece with a lesser one behind it.
pub fn has_skewer(pos: &Chess) -> bool {
    let board = pos.board();
    for role in [Role::Bishop, Role::Rook, Role::Queen] {
        for origin in pieces(board, role, Color::White) {
            for (df, dr) in slider_dirs(role) {
                let (mut f, mut r) = (file_i(origin) + df, rank_i(origin) + dr);
                let mut front: Option<Role> = None;
                while (0..8).contains(&f) && (0..8).contains(&r) {
                    let sq = Square::from_coords(File::new(f as u32), Rank::new(r as u32));
                    if let Some(piece) = board.piece_at(sq) {
                        if piece.color == Color::White {
                            break;
                        }
                        match front {
                            None => front = Some(piece.role),
                            Some(front_role) => {
                                // A real skewer: the front piece is valuable
                                // enough that it has to move, and what stands
                                // behind it is actually worth winning. Without
                                // both tests a bishop x-raying a knight with a
                                // pawn behind reads as a skewer, which it is not.
                                let front_value = tactical_value(front_role);
                                let back_value = tactical_value(piece.role);
                                if front_value > back_value
                                    && back_value >= 3.0
                                    && front_value >= 5.0
                                {
                                    return true;
                                }
                                break;
                            }
                        }
                    }
                    f += df;
                    r += dr;
                }
            }
        }
    }
    false
}

/// Is there a move that unmasks a *different* white piece onto king or queen?
pub fn has_discovery(pos: &Chess) -> bool {
    let board = pos.board();
    let targets: Vec<Square> = pieces(board, Role::King, Color::Black)
        .into_iter()
        .chain(pieces(board, Role::Queen, Color::Black))
        .collect();
    if targets.is_empty() {
        return false;
    }
    let before: Vec<Bitboard> = targets
        .iter()
        .map(|t| attackers(board, Color::White, *t))
        .collect();

    for m in pos.legal_moves().iter() {
        let (_, to, _) = move_squares(m);
        let mut next = pos.clone();
        next.play_unchecked(*m);
        let after_board = next.board();
        for (i, target) in targets.iter().enumerate() {
            if *target == to {
                continue;
            }
            let revealed = attackers(after_board, Color::White, *target)
                .without_const(Bitboard::from_square(to))
                .without_const(before[i]);
            if !revealed.is_empty() {
                return true;
            }
        }
    }
    false
}

pub fn has_backrank(pos: &Chess) -> bool {
    let board = pos.board();
    let Some(king) = board.king_of(Color::Black) else {
        return false;
    };
    if rank_i(king) != 7 {
        return false;
    }
    for df in -1..=1 {
        let f = file_i(king) + df;
        if !(0..8).contains(&f) {
            continue;
        }
        let sq = Square::from_coords(File::new(f as u32), Rank::Seventh);
        match board.piece_at(sq) {
            Some(p) if p.color == Color::Black => {}
            _ => return false, // the king has air
        }
    }
    // A boxed-in king is only a motif if we can actually get to the back rank,
    // otherwise every starting position would "have" a back-rank weakness.
    let heavy = pieces(board, Role::Rook, Color::White) | pieces(board, Role::Queen, Color::White);
    heavy
        .into_iter()
        .any(|sq| !(attacks_from(board, sq) & Bitboard::from_rank(Rank::Eighth)).is_empty())
}

pub fn has_promotion(pos: &Chess) -> bool {
    !(pieces(pos.board(), Role::Pawn, Color::White) & Bitboard::from_rank(Rank::Seventh)).is_empty()
}

/// A black piece that is the sole defender of two attacked black units.
pub fn has_overload(pos: &Chess) -> bool {
    let board = pos.board();
    let mut duties: Vec<(Square, u32)> = Vec::new();
    for sq in board.by_color(Color::Black) {
        if attackers(board, Color::White, sq).is_empty() {
            continue;
        }
        let defenders = attackers(board, Color::Black, sq);
        if defenders.count() == 1 {
            let only = defenders.into_iter().next().expect("count is one");
            match duties.iter_mut().find(|(s, _)| *s == only) {
                Some(entry) => entry.1 += 1,
                None => duties.push((only, 1)),
            }
        }
    }
    duties.iter().any(|(_, count)| *count >= 2)
}

/// An attacked black piece with no safe square to run to.
pub fn has_trapped(pos: &Chess) -> bool {
    let board = pos.board();
    for role in [Role::Knight, Role::Bishop, Role::Rook, Role::Queen] {
        for sq in pieces(board, role, Color::Black) {
            if attackers(board, Color::White, sq).is_empty() {
                continue;
            }
            let has_escape = attacks_from(board, sq).into_iter().any(|target| {
                match board.piece_at(target) {
                    Some(p) if p.color == Color::Black => false,
                    _ => !is_attacked_by(board, Color::White, target),
                }
            });
            if !has_escape {
                return true;
            }
        }
    }
    false
}

pub fn has_passer(pos: &Chess) -> bool {
    let board = pos.board();
    let black_pawns = pieces(board, Role::Pawn, Color::Black);
    pieces(board, Role::Pawn, Color::White).into_iter().any(|sq| {
        !black_pawns.into_iter().any(|bp| {
            (file_i(bp) - file_i(sq)).abs() <= 1 && rank_i(bp) > rank_i(sq)
        })
    })
}

pub fn has_outpost(pos: &Chess) -> bool {
    let board = pos.board();
    let black_pawns = pieces(board, Role::Pawn, Color::Black);
    pieces(board, Role::Knight, Color::White).into_iter().any(|sq| {
        let r = rank_i(sq);
        if !(3..=5).contains(&r) {
            return false;
        }
        let pawn_defended = attackers(board, Color::White, sq)
            .into_iter()
            .any(|a| board.role_at(a) == Some(Role::Pawn));
        if !pawn_defended {
            return false;
        }
        !black_pawns
            .into_iter()
            .any(|bp| (file_i(bp) - file_i(sq)).abs() == 1 && rank_i(bp) > r)
    })
}

pub fn has_rook7th(pos: &Chess) -> bool {
    !(pieces(pos.board(), Role::Rook, Color::White) & Bitboard::from_rank(Rank::Seventh)).is_empty()
}

pub fn has_openfile(pos: &Chess) -> bool {
    let board = pos.board();
    let pawns = board.by_role(Role::Pawn);
    pieces(board, Role::Rook, Color::White)
        .into_iter()
        .any(|rook| !pawns.into_iter().any(|p| p.file() == rook.file()))
}

/// Two white sliders stacked on the same line.
pub fn has_battery(pos: &Chess) -> bool {
    let board = pos.board();
    let queens = pieces(board, Role::Queen, Color::White);
    let groups: [(Bitboard, fn(Square, Square) -> bool); 2] = [
        (pieces(board, Role::Rook, Color::White) | queens, same_rank_or_file),
        (pieces(board, Role::Bishop, Color::White) | queens, same_diagonal),
    ];
    for (group, aligned) in groups {
        let squares: Vec<Square> = group.into_iter().collect();
        for (i, a) in squares.iter().enumerate() {
            for b in &squares[i + 1..] {
                if aligned(*a, *b) && attackers(board, Color::White, *b).contains(*a) {
                    return true;
                }
            }
        }
    }
    false
}

fn same_rank_or_file(a: Square, b: Square) -> bool {
    a.file() == b.file() || a.rank() == b.rank()
}

fn same_diagonal(a: Square, b: Square) -> bool {
    (file_i(a) - file_i(b)).abs() == (rank_i(a) - rank_i(b)).abs()
}

pub fn has_kingattack(pos: &Chess) -> bool {
    let board = pos.board();
    let Some(king) = board.king_of(Color::Black) else {
        return false;
    };
    let mut attacked = 0;
    for df in -1..=1 {
        for dr in -1..=1 {
            let (f, r) = (file_i(king) + df, rank_i(king) + dr);
            if (0..8).contains(&f) && (0..8).contains(&r) {
                let sq = Square::from_coords(File::new(f as u32), Rank::new(r as u32));
                if is_attacked_by(board, Color::White, sq) {
                    attacked += 1;
                }
            }
        }
    }
    attacked >= 4
}

pub fn has_badbishop(pos: &Chess) -> bool {
    let board = pos.board();
    let pawns = pieces(board, Role::Pawn, Color::White);
    let is_light = |sq: Square| (file_i(sq) + rank_i(sq)) % 2 == 1;
    pieces(board, Role::Bishop, Color::White).into_iter().any(|bishop| {
        pawns
            .into_iter()
            .filter(|p| is_light(*p) == is_light(bishop))
            .count()
            >= 5
    })
}

pub fn has_doubled(pos: &Chess) -> bool {
    let pawns = pieces(pos.board(), Role::Pawn, Color::White);
    File::ALL
        .iter()
        .any(|f| pawns.into_iter().filter(|p| p.file() == *f).count() >= 2)
}

pub fn has_isolated(pos: &Chess) -> bool {
    let pawns = pieces(pos.board(), Role::Pawn, Color::White);
    let files: Vec<i32> = pawns.into_iter().map(file_i).collect();
    files
        .iter()
        .any(|f| !files.contains(&(f - 1)) && !files.contains(&(f + 1)))
}

pub fn has_space(pos: &Chess) -> bool {
    let board = pos.board();
    let ours = pieces(board, Role::Pawn, Color::White)
        .into_iter()
        .filter(|p| rank_i(*p) >= 3)
        .count();
    let theirs = pieces(board, Role::Pawn, Color::Black)
        .into_iter()
        .filter(|p| rank_i(*p) <= 4)
        .count();
    ours >= theirs + 2
}

/// Run one detector by name. This mapping *is* the validation oracle.
pub fn detect(motif: &str, pos: &Chess) -> bool {
    match motif {
        "mo:mate1" => has_mate_in_1(pos),
        "mo:check" => has_check(pos),
        "mo:capture" => has_capture(pos),
        "mo:hanging" => has_hanging(pos),
        "mo:fork" => has_fork(pos),
        "mo:pin" => has_pin(pos),
        "mo:skewer" => has_skewer(pos),
        "mo:discovery" => has_discovery(pos),
        "mo:backrank" => has_backrank(pos),
        "mo:promotion" => has_promotion(pos),
        "mo:overload" => has_overload(pos),
        "mo:trapped" => has_trapped(pos),
        "mo:passer" => has_passer(pos),
        "mo:outpost" => has_outpost(pos),
        "mo:rook7th" => has_rook7th(pos),
        "mo:openfile" => has_openfile(pos),
        "mo:battery" => has_battery(pos),
        "mo:kingattack" => has_kingattack(pos),
        "mo:badbishop" => has_badbishop(pos),
        "mo:doubled" => has_doubled(pos),
        "mo:isolated" => has_isolated(pos),
        "mo:space" => has_space(pos),
        other => panic!("{other} has no detector, so nothing can check it"),
    }
}

/// Every motif that fires, in priority order.
pub fn detect_motifs(pos: &Chess) -> Vec<&'static str> {
    MOTIF_PRIORITY
        .iter()
        .filter(|m| detect(m, pos))
        .copied()
        .collect()
}

/// The motifs a trace names, padded with `mo:none`.
pub fn motif_tags(pos: &Chess) -> Vec<&'static str> {
    let mut found = detect_motifs(pos);
    found.truncate(N_MOTIF_SLOTS);
    if found.is_empty() {
        found.push("mo:quiet");
    }
    while found.len() < N_MOTIF_SLOTS {
        found.push("mo:none");
    }
    found
}
