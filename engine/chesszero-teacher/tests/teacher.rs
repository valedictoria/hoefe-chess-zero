//! Tests against a real UCI engine.
//!
//! They need one, so they skip unless CHESSZERO_TEACHER points at a binary.
//! That keeps the suite green on a machine without the engine, and keeps a
//! strongly-copyleft binary out of the repository.

use std::time::Duration;

use chesszero_teacher::{
    configure_for_datagen, engine_path_from_env, policy_target, value_target, Limit, Teacher,
    UciScore,
};
use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess, Position};

fn parse(fen: &str) -> Chess {
    fen.parse::<Fen>().unwrap().into_position(CastlingMode::Standard).unwrap()
}

/// None means "no engine configured", and the caller should skip.
fn teacher() -> Option<Teacher> {
    let path = engine_path_from_env()?;
    let mut t = Teacher::spawn(&path).expect("engine failed to start");
    configure_for_datagen(&mut t, 4, 64).expect("engine rejected configuration");
    Some(t)
}

macro_rules! engine_or_skip {
    () => {
        match teacher() {
            Some(t) => t,
            None => {
                eprintln!("skipping: set CHESSZERO_TEACHER to a UCI engine binary");
                return;
            }
        }
    };
}

#[test]
fn handshake_reports_a_name() {
    let t = engine_or_skip!();
    assert!(!t.name().is_empty(), "engine did not identify itself");
    t.quit();
}

#[test]
fn analysis_returns_legal_moves_best_first() {
    let mut t = engine_or_skip!();
    let pos = Chess::default();
    let analysis = t.analyse(&pos, Limit::Depth(12)).unwrap();

    assert!(analysis.depth >= 12, "depth {} below request", analysis.depth);
    assert_eq!(analysis.lines.len(), 4, "MultiPV 4 should give four lines");

    let legal = pos.legal_moves();
    for line in &analysis.lines {
        assert!(legal.contains(&line.mv), "{:?} is not legal here", line.mv);
        assert!(!line.pv.is_empty(), "line has no principal variation");
        assert_eq!(line.pv[0], line.mv, "pv must start with the move it scores");
    }

    let scores: Vec<i32> = analysis.lines.iter().map(|l| l.score.to_cp()).collect();
    let mut sorted = scores.clone();
    sorted.sort_by(|a, b| b.cmp(a));
    assert_eq!(scores, sorted, "lines are not ordered best first: {scores:?}");
    t.quit();
}

#[test]
fn forced_mate_is_reported_as_mate_not_centipawns() {
    let mut t = engine_or_skip!();
    // Back-rank mate in one.
    let pos = parse("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1");
    let analysis = t.analyse(&pos, Limit::Depth(10)).unwrap();
    let best = analysis.best().expect("no lines");
    assert_eq!(best.score, UciScore::Mate(1), "expected mate in one, got {:?}", best.score);
    assert_eq!(best.mv.to().to_string(), "d8", "expected Rd8#, got {:?}", best.mv);
    assert!(value_target(best.score) > 0.99);
    t.quit();
}

#[test]
fn the_losing_side_gets_a_negative_score() {
    let mut t = engine_or_skip!();
    // Black a full queen down, with plenty of legal moves.
    let pos = parse("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
    let analysis = t.analyse(&pos, Limit::Depth(12)).unwrap();
    let best = analysis.best().expect("no lines");
    // Deliberately not asserting anything near -900. This engine reports a
    // normalised score, not classical centipawns: a full queen down reads about
    // -490. Writing -700 here originally is exactly the mistake that would have
    // shipped a miscalibrated value target.
    assert!(best.score.to_cp() < -350, "expected a clearly lost score, got {:?}", best.score);
    assert!(value_target(best.score) < -0.5, "value target {}", value_target(best.score));
    t.quit();
}

#[test]
fn a_finished_game_is_reported_as_such() {
    // Regression: this FEN was originally written as "black is getting mated",
    // but black is already mated -- there are no legal moves at all. The engine
    // answers "bestmove (none)" with no info lines, which used to surface as an
    // unhelpful parse error. Data generation walks real games and will meet
    // terminal positions, so it needs to tell "game over" from "engine broke".
    let mut t = engine_or_skip!();
    let mated = parse("3R2k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1");
    assert!(mated.is_checkmate(), "test position should be checkmate");
    assert!(matches!(
        t.analyse(&mated, Limit::Depth(8)),
        Err(chesszero_teacher::TeacherError::NoLegalMoves)
    ));

    let stalemate = parse("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    assert!(stalemate.is_stalemate(), "test position should be stalemate");
    assert!(matches!(
        t.analyse(&stalemate, Limit::Depth(8)),
        Err(chesszero_teacher::TeacherError::NoLegalMoves)
    ));
    t.quit();
}

#[test]
fn node_limited_analysis_is_reproducible() {
    // Data generation depends on this: a label must not change between runs, or
    // the dataset cannot be regenerated or audited.
    let mut t = engine_or_skip!();
    let pos = parse("r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1B3/PPP2PPP/R2Q1RK1 w - - 0 9");

    t.new_game().unwrap();
    let first = t.analyse(&pos, Limit::Nodes(200_000)).unwrap();
    t.new_game().unwrap();
    let second = t.analyse(&pos, Limit::Nodes(200_000)).unwrap();

    let moves = |a: &chesszero_teacher::Analysis| {
        a.lines.iter().map(|l| (l.mv, l.score)).collect::<Vec<_>>()
    };
    assert_eq!(moves(&first), moves(&second), "same position and budget gave different labels");
    t.quit();
}

#[test]
fn policy_target_is_a_distribution_favouring_the_best_move() {
    let mut t = engine_or_skip!();
    let pos = parse("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2");
    let analysis = t.analyse(&pos, Limit::Depth(12)).unwrap();

    let policy = policy_target(&analysis.lines, 100.0);
    assert_eq!(policy.len(), analysis.lines.len());

    let total: f32 = policy.iter().map(|(_, p)| p).sum();
    assert!((total - 1.0).abs() < 1e-4, "probabilities sum to {total}");
    assert!(policy.iter().all(|(_, p)| *p >= 0.0));

    let best = policy[0].1;
    assert!(
        policy.iter().skip(1).all(|(_, p)| *p <= best + 1e-6),
        "a worse move outscored the best one: {policy:?}"
    );
    t.quit();
}

#[test]
fn temperature_controls_how_sharp_the_policy_is() {
    let mut t = engine_or_skip!();
    let pos = Chess::default();
    let analysis = t.analyse(&pos, Limit::Depth(12)).unwrap();

    let sharp = policy_target(&analysis.lines, 20.0)[0].1;
    let soft = policy_target(&analysis.lines, 400.0)[0].1;
    assert!(sharp > soft, "low temperature should concentrate mass: {sharp} vs {soft}");
    t.quit();
}

#[test]
fn value_target_scale_changes_how_harsh_a_score_looks() {
    use chesszero_teacher::value_target_scaled;
    // The calibration constant is load-bearing: the same score means very
    // different things on different engines' scales.
    let score = UciScore::Cp(-490);
    let normalised = value_target_scaled(score, 200.0);
    let classical = value_target_scaled(score, 345.0);
    assert!(normalised < classical, "a tighter scale should read as more decisive");
    assert!(normalised < -0.8, "a queen down should look close to lost: {normalised}");
}

#[test]
fn value_target_is_monotonic_and_bounded() {
    // Pure arithmetic, so it runs without an engine.
    let mut previous = -2.0;
    for cp in [-2000, -800, -300, -100, 0, 100, 300, 800, 2000] {
        let v = value_target(UciScore::Cp(cp));
        assert!(v > previous, "not monotonic at {cp}");
        assert!((-1.0..=1.0).contains(&v), "{v} out of range at {cp}");
        previous = v;
    }
    assert!(value_target(UciScore::Cp(0)).abs() < 1e-6, "an equal position should be 0");
    assert_eq!(value_target(UciScore::Mate(3)), 1.0);
    assert_eq!(value_target(UciScore::Mate(-3)), -1.0);
}

#[test]
fn movetime_limit_is_respected() {
    let mut t = engine_or_skip!();
    let analysis = t
        .analyse(&Chess::default(), Limit::MoveTime(Duration::from_millis(300)))
        .unwrap();
    assert!(analysis.elapsed < Duration::from_secs(5), "took {:?}", analysis.elapsed);
    assert!(!analysis.lines.is_empty());
    t.quit();
}
