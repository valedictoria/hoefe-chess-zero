//! Where does data-generation time actually go?

use std::time::Instant;

use chesszero_search::{alphabeta::Limits, Searcher};
use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess, Position};

fn parse(fen: &str) -> Chess {
    fen.parse::<Fen>().unwrap().into_position(CastlingMode::Standard).unwrap()
}

fn main() {
    let positions = [
        ("startpos", Chess::default()),
        ("midgame", parse("r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1B3/PPP2PPP/R2Q1RK1 w - - 0 9")),
        ("kiwipete", parse("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")),
        ("endgame", parse("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")),
    ];

    println!("{:<10} {:>6} {:>8} {:>12} {:>10} {:>12} {:>9}",
             "position", "moves", "depth", "one-search", "secs", "root-scores", "secs");
    for (name, pos) in &positions {
        let n_moves = pos.legal_moves().len();
        for depth in [4u32, 6] {
            let mut s = Searcher::default();
            let t = Instant::now();
            let one = s.search(pos, &Limits::depth(depth));
            let one_secs = t.elapsed().as_secs_f64();

            let mut s2 = Searcher::default();
            let t = Instant::now();
            let _ = s2.root_move_scores(pos, &Limits::depth(depth));
            let all_secs = t.elapsed().as_secs_f64();

            println!("{:<10} {:>6} {:>8} {:>12} {:>10.3} {:>12} {:>9.3}  ({:.1}x)",
                     name, n_moves, depth, one.nodes, one_secs, s2.nodes(), all_secs,
                     all_secs / one_secs.max(1e-9));
        }
    }
}
