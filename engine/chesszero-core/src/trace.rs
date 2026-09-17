//! Writing a teacher trace: the 32-token controlled language the model speaks.
//!
//! A trace always has the same shape, which is what lets generation be masked to
//! the grammar and batching stay trivial:
//!
//! ```text
//! MAT <bucket> PHASE <p> KSAFE <us> <them> THR <n> TAC <m> <m> <m>
//! PLAN <plan> CAND <move> <move> <move> BEST <move> EVAL <bucket> <eos>
//! ```
//!
//! The positional fields come from the detectors in [`crate::motifs`] and
//! [`crate::features`]. `best`, `candidates` and `q` are *inputs*: at training
//! time they come from whatever search produced the sample, at play time from
//! the tree. The model never invents them.

use shakmaty::{Chess, Color, Position, Role, Square};

use crate::features::{
    eval_bucket, hanging_pieces, king_safety, material_balance, material_bucket, phase, plan,
};
use crate::motifs::motif_tags;
use crate::vocab::{tid, EOS, TRACE_LEN};

/// A move as the trace sees it: from, to, and an optional promotion.
pub type TraceMove = (Square, Square, Option<Role>);

/// The placeholder for an absent candidate.
const NULL_MOVE_TOKENS: [&str; 3] = ["@a1", "@a1", "pr:-"];

fn promo_token(role: Option<Role>) -> &'static str {
    match role {
        None => "pr:-",
        Some(Role::Knight) => "pr:n",
        Some(Role::Bishop) => "pr:b",
        Some(Role::Rook) => "pr:r",
        Some(Role::Queen) => "pr:q",
        Some(Role::Pawn) | Some(Role::King) => {
            unreachable!("a pawn cannot promote to a pawn or a king")
        }
    }
}

fn push_move(out: &mut Vec<u16>, m: Option<TraceMove>) {
    match m {
        None => out.extend(NULL_MOVE_TOKENS.iter().map(|t| tid(t))),
        Some((from, to, promo)) => {
            out.push(tid(&format!("@{from}")));
            out.push(tid(&format!("@{to}")));
            out.push(tid(promo_token(promo)));
        }
    }
}

/// Write the trace for a canonical position.
///
/// `q` is the search's value estimate in [-1, 1] from the mover's point of view.
pub fn annotate(
    canon: &Chess,
    best: Option<TraceMove>,
    candidates: &[TraceMove],
    q: f64,
) -> Vec<u16> {
    let board = canon.board();
    let balance = material_balance(board);
    let phase_token = phase(board);
    let threats = hanging_pieces(board, Color::White).len().min(3);

    let mut cands: Vec<TraceMove> = candidates.to_vec();
    if let Some(b) = best {
        if !cands.contains(&b) {
            cands.insert(0, b);
        }
    }
    cands.truncate(3);

    let mut out: Vec<u16> = Vec::with_capacity(TRACE_LEN);
    out.push(tid("f:MAT"));
    out.push(tid(&format!("mat:{}", material_bucket(balance))));
    out.push(tid("f:PHASE"));
    out.push(tid(phase_token));
    out.push(tid("f:KSAFE"));
    out.push(tid(king_safety(board, Color::White)));
    out.push(tid(king_safety(board, Color::Black)));
    out.push(tid("f:THR"));
    out.push(tid(&format!("thr:{threats}")));
    out.push(tid("f:TAC"));
    for motif in motif_tags(canon) {
        out.push(tid(motif));
    }
    out.push(tid("f:PLAN"));
    out.push(tid(plan(canon, balance, phase_token, threats)));
    out.push(tid("f:CAND"));
    for i in 0..3 {
        push_move(&mut out, cands.get(i).copied());
    }
    out.push(tid("f:BEST"));
    push_move(&mut out, best);
    out.push(tid("f:EVAL"));
    out.push(tid(&format!("ev:{}", eval_bucket(q))));
    out.push(tid(EOS));

    debug_assert_eq!(out.len(), TRACE_LEN);
    out
}
