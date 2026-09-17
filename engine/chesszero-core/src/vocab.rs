//! The token vocabulary, in the exact order the Python trainer emits it.
//!
//! Order is not cosmetic: a token's index *is* its embedding row, so a vocabulary
//! that differs from the trainer's by a single entry silently corrupts every
//! lookup. `spec/golden.json` carries the authoritative list and the golden test
//! asserts this module reproduces it.

use std::collections::HashMap;
use std::sync::OnceLock;

pub const PAD: &str = "<pad>";
pub const BOS: &str = "<bos>";
pub const SEP: &str = "<sep>";
pub const EOS: &str = "<eos>";

/// `<bos>` + 64 squares + castling + ep + rule50 + repetitions + `<sep>`
pub const PREFIX_LEN: usize = 70;
pub const TRACE_LEN: usize = 32;
pub const SEQ_LEN: usize = PREFIX_LEN + TRACE_LEN;
pub const N_MOTIF_SLOTS: usize = 3;
/// hidden state read by the position-only heads (over `<sep>`)
pub const READOUT_FAST: usize = PREFIX_LEN - 1;
/// hidden state read by the reasoned heads (over the trace's `<eos>`)
pub const READOUT_REASONED: usize = SEQ_LEN - 1;

pub const PIECE_CHARS: &str = "PNBRQKpnbrqk";

pub const PHASE_TOKENS: [&str; 3] = ["ph:opening", "ph:middle", "ph:end"];
pub const KSAFE_TOKENS: [&str; 4] = ["ks:safe", "ks:ok", "ks:loose", "ks:exposed"];

pub const FIELD_TOKENS: [&str; 9] = [
    "f:MAT", "f:PHASE", "f:KSAFE", "f:THR", "f:TAC", "f:PLAN", "f:CAND", "f:BEST", "f:EVAL",
];

pub const PROMO_TOKENS: [&str; 5] = ["pr:-", "pr:n", "pr:b", "pr:r", "pr:q"];

/// Material balance bucket edges, in pawns, from the mover's point of view.
pub const MAT_EDGES: [f32; 8] = [-5.0, -3.0, -2.0, -0.5, 0.5, 2.0, 3.0, 5.0];

pub const MOTIF_TOKENS: [&str; 24] = [
    "mo:none",
    "mo:quiet",
    "mo:mate1",
    "mo:check",
    "mo:capture",
    "mo:hanging",
    "mo:fork",
    "mo:pin",
    "mo:skewer",
    "mo:discovery",
    "mo:backrank",
    "mo:promotion",
    "mo:overload",
    "mo:trapped",
    "mo:passer",
    "mo:outpost",
    "mo:rook7th",
    "mo:openfile",
    "mo:battery",
    "mo:kingattack",
    "mo:badbishop",
    "mo:doubled",
    "mo:isolated",
    "mo:space",
];

pub const PLAN_TOKENS: [&str; 10] = [
    "plan:develop",
    "plan:castle",
    "plan:center",
    "plan:attack",
    "plan:trade",
    "plan:push",
    "plan:rooks",
    "plan:defend",
    "plan:convert",
    "plan:hold",
];

/// Build the vocabulary. Sections appear in the same order as `vocab.py`.
fn build() -> Vec<String> {
    let mut v: Vec<String> = Vec::with_capacity(196);
    v.extend([PAD, BOS, SEP, EOS].iter().map(|s| s.to_string()));

    v.push("sq:.".to_string());
    v.extend(PIECE_CHARS.chars().map(|c| format!("sq:{c}")));

    v.extend((0..16).map(|i| format!("cas:{i}")));

    v.push("ep:-".to_string());
    v.extend("abcdefgh".chars().map(|f| format!("ep:{f}")));

    v.extend((0..8).map(|i| format!("r50:{i}")));
    v.extend((0..3).map(|i| format!("rep:{i}")));

    for sq in shakmaty::Square::ALL {
        v.push(format!("@{sq}"));
    }

    v.extend(PROMO_TOKENS.iter().map(|s| s.to_string()));
    v.extend(FIELD_TOKENS.iter().map(|s| s.to_string()));
    v.extend((0..MAT_EDGES.len() + 1).map(|i| format!("mat:{i}")));
    v.extend(PHASE_TOKENS.iter().map(|s| s.to_string()));
    v.extend(KSAFE_TOKENS.iter().map(|s| s.to_string()));
    v.extend((0..4).map(|i| format!("thr:{i}")));
    v.extend(MOTIF_TOKENS.iter().map(|s| s.to_string()));
    v.extend(PLAN_TOKENS.iter().map(|s| s.to_string()));
    v.extend((0..11).map(|i| format!("ev:{i}")));
    v
}

fn vocab_cell() -> &'static Vec<String> {
    static CELL: OnceLock<Vec<String>> = OnceLock::new();
    CELL.get_or_init(build)
}

fn index_cell() -> &'static HashMap<&'static str, u16> {
    static CELL: OnceLock<HashMap<&'static str, u16>> = OnceLock::new();
    CELL.get_or_init(|| {
        vocab_cell()
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i as u16))
            .collect()
    })
}

pub fn vocab() -> &'static [String] {
    vocab_cell()
}

pub fn vocab_size() -> usize {
    vocab_cell().len()
}

/// Token string to id. Panics on an unknown token: every caller builds tokens
/// from the constants above, so an unknown one is a bug, not bad input.
pub fn tid(token: &str) -> u16 {
    *index_cell()
        .get(token)
        .unwrap_or_else(|| panic!("token {token:?} is not in the vocabulary"))
}

pub fn token(id: u16) -> &'static str {
    &vocab_cell()[id as usize]
}

/// Which tokens may appear in each of the 32 trace slots.
///
/// Generation is masked to these sets, which is why the model cannot emit an
/// ungrammatical trace even with random weights.
pub fn trace_slots() -> &'static Vec<Vec<u16>> {
    static CELL: OnceLock<Vec<Vec<u16>>> = OnceLock::new();
    CELL.get_or_init(|| {
        let lit = |s: &str| vec![tid(s)];
        let set = |xs: &[&str]| xs.iter().map(|s| tid(s)).collect::<Vec<_>>();
        let squares: Vec<u16> = shakmaty::Square::ALL.iter().map(|s| tid(&format!("@{s}"))).collect();
        let promos = set(&PROMO_TOKENS);
        let motifs = set(&MOTIF_TOKENS);
        let move_slots = |out: &mut Vec<Vec<u16>>| {
            out.push(squares.clone());
            out.push(squares.clone());
            out.push(promos.clone());
        };

        let mut slots: Vec<Vec<u16>> = Vec::with_capacity(TRACE_LEN);
        slots.push(lit("f:MAT"));
        slots.push((0..MAT_EDGES.len() + 1).map(|i| tid(&format!("mat:{i}"))).collect());
        slots.push(lit("f:PHASE"));
        slots.push(set(&PHASE_TOKENS));
        slots.push(lit("f:KSAFE"));
        slots.push(set(&KSAFE_TOKENS));
        slots.push(set(&KSAFE_TOKENS));
        slots.push(lit("f:THR"));
        slots.push((0..4).map(|i| tid(&format!("thr:{i}"))).collect());
        slots.push(lit("f:TAC"));
        for _ in 0..N_MOTIF_SLOTS {
            slots.push(motifs.clone());
        }
        slots.push(lit("f:PLAN"));
        slots.push(set(&PLAN_TOKENS));
        slots.push(lit("f:CAND"));
        for _ in 0..3 {
            move_slots(&mut slots);
        }
        slots.push(lit("f:BEST"));
        move_slots(&mut slots);
        slots.push(lit("f:EVAL"));
        slots.push((0..11).map(|i| tid(&format!("ev:{i}"))).collect());
        slots.push(lit(EOS));
        debug_assert_eq!(slots.len(), TRACE_LEN);
        slots
    })
}

// Slot indices of each field inside a trace.
pub const SLOT_MAT: usize = 1;
pub const SLOT_PHASE: usize = 3;
pub const SLOT_KSAFE_US: usize = 5;
pub const SLOT_KSAFE_THEM: usize = 6;
pub const SLOT_THR: usize = 8;
pub const SLOT_MOTIF: [usize; 3] = [10, 11, 12];
pub const SLOT_PLAN: usize = 14;
pub const SLOT_CAND: [usize; 3] = [16, 19, 22];
pub const SLOT_BEST: usize = 26;
pub const SLOT_EVAL: usize = 30;
