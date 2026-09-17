//! Canonical orientation and the 4672-slot policy index.
//!
//! **Canonical orientation.** Every position is shown to the network from the
//! side to move's point of view: if black is to move the position is mirrored
//! (flipped vertically *and* colour-swapped), so the network only ever sees
//! white to play. This halves what it has to learn, and is what Lc0 does.
//!
//! **Policy indexing.** AlphaZero's 8x8x73 layout: 56 queen planes (8 directions
//! x 7 distances), 8 knight planes, 9 underpromotion planes (3 files x
//! knight/bishop/rook). Queen promotions ride along on the plain queen move.

use shakmaty::{
    CastlingMode, Chess, Color, EnPassantMode, File, FromSetup, Move, Position, Rank, Role, Square,
};

use crate::vocab::{tid, BOS, SEP};

pub const N_PLANES: usize = 73;
pub const N_MOVES: usize = 64 * N_PLANES;

/// (file delta, rank delta) for N, NE, E, SE, S, SW, W, NW.
pub const QUEEN_DIRS: [(i32, i32); 8] = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
];

pub const KNIGHT_DIRS: [(i32, i32); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];

pub const UNDERPROMO_ROLES: [Role; 3] = [Role::Knight, Role::Bishop, Role::Rook];

#[inline]
fn file_of(sq: Square) -> i32 {
    u32::from(sq.file()) as i32
}

#[inline]
fn rank_of(sq: Square) -> i32 {
    u32::from(sq.rank()) as i32
}

#[inline]
fn signum(x: i32) -> i32 {
    (x > 0) as i32 - (x < 0) as i32
}

/// Mirror a square vertically (a1 <-> a8), matching `chess.square_mirror`.
#[inline]
pub fn canonical_square(sq: Square, flipped: bool) -> Square {
    if flipped {
        sq.flip_vertical()
    } else {
        sq
    }
}

/// The position as the side to move sees it, plus whether it was mirrored.
///
/// `EnPassantMode::Always` is deliberate: the ep square has to survive the
/// mirror so that the *canonical* position can then decide whether a capture is
/// legal on it. Deciding that before mirroring would ask the question of the
/// wrong side.
pub fn canonical(pos: &Chess) -> (Chess, bool) {
    if pos.turn() == Color::White {
        (pos.clone(), false)
    } else {
        let setup = pos.to_setup(EnPassantMode::Always).into_mirrored();
        let mirrored = Chess::from_setup(setup, CastlingMode::Standard)
            .expect("the mirror of a legal position is legal");
        (mirrored, true)
    }
}

/// From-square, to-square and promotion, in python-chess's terms.
///
/// The castling case matters: shakmaty models castling as king-takes-rook, so
/// its `to` is the rook's square, while python-chess uses the king's
/// destination (e1g1 / e1c1). Encoding shakmaty's square directly would put
/// castling on a different policy plane than the trainer did.
pub fn move_squares(m: &Move) -> (Square, Square, Option<Role>) {
    match m {
        Move::Normal {
            from,
            to,
            promotion,
            ..
        } => (*from, *to, *promotion),
        Move::EnPassant { from, to } => (*from, *to, None),
        Move::Castle { king, rook } => {
            let to_file = if rook.file() > king.file() {
                File::G
            } else {
                File::C
            };
            (*king, Square::from_coords(to_file, king.rank()), None)
        }
        Move::Put { .. } => unreachable!("drops do not occur in standard chess"),
    }
}

/// Policy index of a move already in canonical orientation.
pub fn move_to_index(from: Square, to: Square, promotion: Option<Role>) -> usize {
    let df = file_of(to) - file_of(from);
    let dr = rank_of(to) - rank_of(from);

    let plane = match promotion {
        Some(role) if role != Role::Queen => {
            let piece_idx = UNDERPROMO_ROLES
                .iter()
                .position(|r| *r == role)
                .expect("promotion role must be knight, bishop or rook");
            debug_assert_eq!(dr, 1);
            debug_assert!((-1..=1).contains(&df));
            64 + piece_idx * 3 + (df + 1) as usize
        }
        _ => {
            if let Some(k) = KNIGHT_DIRS.iter().position(|d| *d == (df, dr)) {
                56 + k
            } else {
                let dir = (signum(df), signum(dr));
                let idx = QUEEN_DIRS
                    .iter()
                    .position(|d| *d == dir)
                    .expect("move is neither a knight hop nor a ray");
                let distance = df.abs().max(dr.abs());
                idx * 7 + (distance - 1) as usize
            }
        }
    };

    usize::from(from) * N_PLANES + plane
}

/// Inverse of [`move_to_index`]. `board` disambiguates a queen promotion from a
/// plain push to the last rank; `None` means the index is off the board.
pub fn index_to_move(index: usize, board: &Chess) -> Option<(Square, Square, Option<Role>)> {
    let from = Square::new((index / N_PLANES) as u32);
    let plane = index % N_PLANES;
    let (f0, r0) = (file_of(from), rank_of(from));

    let (df, dr, mut promotion) = if plane < 56 {
        let (dir, dist) = (plane / 7, (plane % 7) as i32 + 1);
        let (df, dr) = QUEEN_DIRS[dir];
        (df * dist, dr * dist, None)
    } else if plane < 64 {
        let (df, dr) = KNIGHT_DIRS[plane - 56];
        (df, dr, None)
    } else {
        let rest = plane - 64;
        (rest as i32 % 3 - 1, 1, Some(UNDERPROMO_ROLES[rest / 3]))
    };

    let (tf, tr) = (f0 + df, r0 + dr);
    if !(0..8).contains(&tf) || !(0..8).contains(&tr) {
        return None;
    }
    let to = Square::from_coords(File::new(tf as u32), Rank::new(tr as u32));

    if promotion.is_none() && tr == 7 {
        if let Some(piece) = board.board().piece_at(from) {
            if piece.role == Role::Pawn && piece.color == Color::White {
                promotion = Some(Role::Queen);
            }
        }
    }
    Some((from, to, promotion))
}

/// Legal moves of a canonical position with their policy indices.
pub fn legal_move_indices(canon: &Chess) -> Vec<(Move, usize)> {
    canon
        .legal_moves()
        .iter()
        .map(|m| {
            let (from, to, promo) = move_squares(m);
            (m.clone(), move_to_index(from, to, promo))
        })
        .collect()
}

fn castle_code(pos: &Chess) -> usize {
    let rights = pos.castles().castling_rights();
    let mut code = 0;
    for (bit, square) in [
        (1, Square::H1),
        (2, Square::A1),
        (4, Square::H8),
        (8, Square::A8),
    ] {
        if rights.contains(square) {
            code |= bit;
        }
    }
    code
}

#[inline]
fn r50_bucket(halfmoves: u32) -> usize {
    (halfmoves as usize / 13).min(7)
}

const PIECE_CHAR: [(Role, char); 6] = [
    (Role::Pawn, 'P'),
    (Role::Knight, 'N'),
    (Role::Bishop, 'B'),
    (Role::Rook, 'R'),
    (Role::Queen, 'Q'),
    (Role::King, 'K'),
];

/// Token ids for the position part of the sequence, ending with `<sep>`.
///
/// `repetitions` cannot be derived from the position: a FEN carries no history,
/// so the caller supplies it from the game it is actually playing.
pub fn encode_prefix(canon: &Chess, repetitions: u8) -> Vec<u16> {
    let mut out = Vec::with_capacity(PREFIX_LEN_CHECK);
    out.push(tid(BOS));

    let board = canon.board();
    for sq in Square::ALL {
        match board.piece_at(sq) {
            None => out.push(tid("sq:.")),
            Some(piece) => {
                let base = PIECE_CHAR
                    .iter()
                    .find(|(role, _)| *role == piece.role)
                    .map(|(_, c)| *c)
                    .expect("every role has a letter");
                let c = if piece.color == Color::White {
                    base
                } else {
                    base.to_ascii_lowercase()
                };
                out.push(tid(&format!("sq:{c}")));
            }
        }
    }

    out.push(tid(&format!("cas:{}", castle_code(canon))));

    // Only an en passant square that can actually be captured on is part of the
    // position. A square nobody can take on would split identical positions into
    // different tokens, and it does not survive a FEN round trip either.
    match canon.ep_square(EnPassantMode::Legal) {
        None => out.push(tid("ep:-")),
        Some(sq) => out.push(tid(&format!("ep:{}", sq.file().char()))),
    }

    out.push(tid(&format!("r50:{}", r50_bucket(canon.halfmoves()))));
    out.push(tid(&format!("rep:{}", repetitions.min(2))));
    out.push(tid(SEP));

    debug_assert_eq!(out.len(), PREFIX_LEN_CHECK);
    out
}

const PREFIX_LEN_CHECK: usize = crate::vocab::PREFIX_LEN;
