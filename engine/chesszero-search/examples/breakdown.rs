//! What a hand-crafted evaluation can say that a value head cannot.
//!
//! A network reports "+0.4". This reports *why*, and "why" is the whole product
//! when the user is trying to learn.

use chesszero_search::eval::evaluate_detailed;
use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess};

fn parse(fen: &str) -> Chess {
    fen.parse::<Fen>().unwrap().into_position(CastlingMode::Standard).unwrap()
}

fn main() {
    let positions = [
        ("start", Chess::default()),
        ("king stripped bare", parse("r2q1rk1/pp3ppp/2n1bn2/3p4/3P4/2N1BN2/PP3PPP/R2QK2R w KQ - 0 1")),
        ("passer, queens off", parse("r4rk1/1p3ppp/8/2P5/8/8/1P3PPP/R4RK1 w - - 0 1")),
        ("shattered pawns", parse("4k3/p1p1p1p1/8/8/8/8/PPP2PPP/4K3 w - - 0 1")),
        ("rooks on the seventh", parse("4k3/1R1R2pp/8/8/8/8/6PP/4K3 w - - 0 1")),
    ];

    println!(
        "{:<22} {:>5} {:>9} {:>6} {:>8} {:>6} {:>7} {:>6} {:>7}",
        "position", "phase", "material", "place", "mobility", "pawns", "pieces", "king", "TOTAL"
    );
    for (name, pos) in &positions {
        let b = evaluate_detailed(pos);
        println!(
            "{:<22} {:>5} {:>9} {:>6} {:>8} {:>6} {:>7} {:>6} {:>7}",
            name, b.phase, b.material, b.placement, b.mobility,
            b.pawn_structure, b.piece_quality, b.king_safety, b.total
        );
    }

    println!("\nSame position, read aloud the way a learner would want it:");
    let b = evaluate_detailed(&parse("4k3/1R1R2pp/8/8/8/8/6PP/4K3 w - - 0 1"));
    let say = |label: &str, cp: i32| {
        if cp.abs() >= 15 {
            println!("  {label}: {:+.2}", cp as f32 / 100.0);
        }
    };
    say("material", b.material);
    say("piece activity", b.piece_quality);
    say("mobility", b.mobility);
    say("pawn structure", b.pawn_structure);
    say("king safety", b.king_safety);
    println!("  -> {:+.2} overall", b.total as f32 / 100.0);
}
