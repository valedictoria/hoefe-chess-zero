//! Drive an external UCI engine to label training positions.
//!
//! The network needs targets far stronger than it can produce itself, and far
//! stronger than our own alpha-beta produces. A top engine supplies them: one
//! MultiPV search yields both a value target (the best line's score) and a
//! policy target (the relative scores of the top moves).
//!
//! **This crate deliberately knows nothing about any particular engine.** It
//! spawns a binary and speaks UCI to it over a pipe. That matters for two
//! reasons. It keeps strongly-copyleft engines at arm's length: Reckless is
//! AGPL-3.0, and linking it would pull that licence over this codebase --
//! including its network clause, which reaches anyone who runs the result as a
//! web service. Separate processes exchanging a documented text protocol are
//! the same boundary every chess GUI has relied on for decades. It also means
//! the teacher is swappable: point it at Stockfish, or at a weaker engine when
//! you want labels a beginner could plausibly follow.
//!
//! What the engine *outputs* -- evaluations -- is not a derivative work of it,
//! on the same reading by which compiling with GCC does not make a binary GPL.
//! Worth knowing that whether network weights trained on engine output are
//! derivative has never been tested in court, and has been argued over in the
//! chess engine community more than once.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

use shakmaty::uci::UciMove;
use shakmaty::{Chess, EnPassantMode, Move, Position};

pub type Result<T> = std::result::Result<T, TeacherError>;

#[derive(Debug)]
pub enum TeacherError {
    Spawn(std::io::Error),
    Io(std::io::Error),
    /// The engine closed its output, usually because it crashed.
    EngineGone,
    /// The engine did not answer within the allotted time.
    Timeout(&'static str),
    /// The position is checkmate or stalemate, so there is nothing to analyse.
    /// Data generation should skip these rather than treat it as a failure.
    NoLegalMoves,
    Protocol(String),
}

impl std::fmt::Display for TeacherError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spawn(e) => write!(f, "could not start the engine: {e}"),
            Self::Io(e) => write!(f, "engine io failed: {e}"),
            Self::EngineGone => write!(f, "the engine exited unexpectedly"),
            Self::Timeout(what) => write!(f, "the engine never sent {what}"),
            Self::NoLegalMoves => write!(f, "the position is over; there is nothing to analyse"),
            Self::Protocol(m) => write!(f, "unexpected engine output: {m}"),
        }
    }
}

impl std::error::Error for TeacherError {}

/// A score as UCI reports it: centipawns, or a forced mate in N moves.
///
/// These are kept apart rather than folded into one number because they mean
/// different things. "Mate in 3" is not "+31000 centipawns"; collapsing it
/// loses the distinction between a won position and a forced win, and a policy
/// target built from the collapsed value gets the margins badly wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UciScore {
    Cp(i32),
    Mate(i32),
}

impl UciScore {
    /// A single number for ordering and for softmax targets. Mates map far
    /// outside the centipawn range, nearer mates further out.
    pub fn to_cp(self) -> i32 {
        match self {
            Self::Cp(cp) => cp,
            Self::Mate(n) if n > 0 => 30_000 - n * 100,
            Self::Mate(n) => -30_000 - n * 100,
        }
    }

    pub fn is_mate(self) -> bool {
        matches!(self, Self::Mate(_))
    }
}

/// One MultiPV line.
#[derive(Debug, Clone)]
pub struct Line {
    pub mv: Move,
    pub score: UciScore,
    pub pv: Vec<Move>,
}

/// The result of one analysis.
#[derive(Debug, Clone)]
pub struct Analysis {
    /// Best first. Only the top `multipv` moves appear; everything else the
    /// engine judged worse and did not report.
    pub lines: Vec<Line>,
    pub depth: u32,
    pub nodes: u64,
    pub elapsed: Duration,
}

impl Analysis {
    pub fn best(&self) -> Option<&Line> {
        self.lines.first()
    }
}

/// How long the engine should think.
#[derive(Debug, Clone, Copy)]
pub enum Limit {
    Depth(u32),
    /// Reproducible across machines in a way time is not, which is what data
    /// generation wants.
    Nodes(u64),
    MoveTime(Duration),
}

impl Limit {
    fn to_go(self) -> String {
        match self {
            Self::Depth(d) => format!("go depth {d}"),
            Self::Nodes(n) => format!("go nodes {n}"),
            Self::MoveTime(t) => format!("go movetime {}", t.as_millis()),
        }
    }
}

pub struct Teacher {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    name: String,
    multipv: usize,
}

impl Teacher {
    /// Start an engine and complete the UCI handshake.
    pub fn spawn(path: &Path) -> Result<Self> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(TeacherError::Spawn)?;

        let stdin = child.stdin.take().expect("stdin was piped");
        let stdout = BufReader::new(child.stdout.take().expect("stdout was piped"));
        let mut teacher = Self { child, stdin, stdout, name: String::new(), multipv: 1 };

        teacher.send("uci")?;
        let mut name = String::new();
        teacher.read_until(Duration::from_secs(30), "uciok", |line| {
            if let Some(rest) = line.strip_prefix("id name ") {
                name = rest.trim().to_string();
            }
            line.trim() == "uciok"
        })?;
        teacher.name = name;
        teacher.is_ready()?;
        Ok(teacher)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    fn send(&mut self, command: &str) -> Result<()> {
        writeln!(self.stdin, "{command}").map_err(TeacherError::Io)?;
        self.stdin.flush().map_err(TeacherError::Io)
    }

    /// Read lines until `stop` returns true, feeding each to the caller.
    fn read_until<F: FnMut(&str) -> bool>(
        &mut self,
        timeout: Duration,
        what: &'static str,
        mut stop: F,
    ) -> Result<()> {
        let deadline = Instant::now() + timeout;
        let mut line = String::new();
        loop {
            line.clear();
            match self.stdout.read_line(&mut line) {
                Ok(0) => return Err(TeacherError::EngineGone),
                Ok(_) => {
                    if stop(&line) {
                        return Ok(());
                    }
                }
                Err(e) => return Err(TeacherError::Io(e)),
            }
            if Instant::now() > deadline {
                return Err(TeacherError::Timeout(what));
            }
        }
    }

    pub fn set_option(&mut self, name: &str, value: &str) -> Result<()> {
        if name.eq_ignore_ascii_case("multipv") {
            self.multipv = value.parse().unwrap_or(1);
        }
        self.send(&format!("setoption name {name} value {value}"))?;
        self.is_ready()
    }

    /// Ask for the top `multipv` moves. Threads should be 1 and the limit
    /// node- or depth-based if the labels are to be reproducible.
    pub fn set_multipv(&mut self, lines: usize) -> Result<()> {
        self.set_option("MultiPV", &lines.to_string())
    }

    pub fn is_ready(&mut self) -> Result<()> {
        self.send("isready")?;
        self.read_until(Duration::from_secs(30), "readyok", |l| l.trim() == "readyok")
    }

    /// Clear the transposition table, so one position cannot colour the next.
    ///
    /// Data generation wants every label to depend only on its own position;
    /// leaving the table warm makes a label depend on the order positions
    /// happened to be visited in, which is not reproducible.
    pub fn new_game(&mut self) -> Result<()> {
        self.send("ucinewgame")?;
        self.is_ready()
    }

    pub fn analyse(&mut self, pos: &Chess, limit: Limit) -> Result<Analysis> {
        // Checkmate and stalemate have no moves to report, and engines answer
        // "bestmove (none)" with no info lines. Catching it here gives the
        // caller something it can act on instead of a parse failure.
        if pos.legal_moves().is_empty() {
            return Err(TeacherError::NoLegalMoves);
        }
        let fen = shakmaty::fen::Fen::from_position(pos, EnPassantMode::Legal);
        self.send(&format!("position fen {fen}"))?;
        self.send(&limit.to_go())?;

        // Keep only the deepest completed iteration: shallower ones are
        // superseded, and a partial iteration can be interrupted mid-way.
        let mut by_depth: Vec<(u32, Vec<Option<Line>>)> = Vec::new();
        let mut nodes = 0u64;
        let started = Instant::now();
        let multipv = self.multipv;

        let mut protocol_error = None;
        self.read_until(Duration::from_secs(600), "bestmove", |line| {
            if line.starts_with("bestmove") {
                return true;
            }
            if !line.starts_with("info ") {
                return false;
            }
            match parse_info(line, pos) {
                Ok(Some((depth, index, line_data, seen_nodes))) => {
                    if seen_nodes > 0 {
                        nodes = seen_nodes;
                    }
                    let slot = match by_depth.iter_mut().find(|(d, _)| *d == depth) {
                        Some((_, slot)) => slot,
                        None => {
                            by_depth.push((depth, vec![None; multipv.max(index)]));
                            &mut by_depth.last_mut().expect("just pushed").1
                        }
                    };
                    if slot.len() < index {
                        slot.resize(index, None);
                    }
                    slot[index - 1] = Some(line_data);
                }
                Ok(None) => {}
                Err(e) => protocol_error = Some(e),
            }
            false
        })?;

        if let Some(e) = protocol_error {
            return Err(e);
        }

        let elapsed = started.elapsed();
        let (depth, lines) = by_depth
            .into_iter()
            .filter(|(_, slot)| slot.first().map(Option::is_some).unwrap_or(false))
            .max_by_key(|(d, _)| *d)
            .ok_or_else(|| TeacherError::Protocol("no usable info lines".into()))?;

        Ok(Analysis {
            lines: lines.into_iter().flatten().collect(),
            depth,
            nodes,
            elapsed,
        })
    }

    pub fn quit(mut self) {
        let _ = self.send("quit");
        let _ = self.child.wait();
    }
}

impl Drop for Teacher {
    fn drop(&mut self) {
        // An engine left running would hold a core for the rest of the session.
        let _ = writeln!(self.stdin, "quit");
        let _ = self.stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Parse one `info` line. Returns None for lines without a score or pv, which
/// engines emit constantly (`info currmove`, `info string`, and so on).
fn parse_info(line: &str, pos: &Chess) -> Result<Option<(u32, usize, Line, u64)>> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    let mut depth = None;
    let mut multipv = 1usize;
    let mut score = None;
    let mut nodes = 0u64;
    let mut pv_start = None;

    let mut i = 0;
    while i < tokens.len() {
        match tokens[i] {
            "depth" => {
                depth = tokens.get(i + 1).and_then(|t| t.parse().ok());
                i += 2;
            }
            "multipv" => {
                multipv = tokens.get(i + 1).and_then(|t| t.parse().ok()).unwrap_or(1);
                i += 2;
            }
            "nodes" => {
                nodes = tokens.get(i + 1).and_then(|t| t.parse().ok()).unwrap_or(0);
                i += 2;
            }
            "score" => {
                let value = tokens.get(i + 2).and_then(|t| t.parse::<i32>().ok());
                score = match (tokens.get(i + 1), value) {
                    (Some(&"cp"), Some(v)) => Some(UciScore::Cp(v)),
                    (Some(&"mate"), Some(v)) => Some(UciScore::Mate(v)),
                    // "score cp 12 lowerbound" is a partial result, not a verdict
                    _ => None,
                };
                i += 3;
            }
            "pv" => {
                pv_start = Some(i + 1);
                break;
            }
            _ => i += 1,
        }
    }

    let (Some(depth), Some(score), Some(start)) = (depth, score, pv_start) else {
        return Ok(None);
    };
    if tokens.len() <= start {
        return Ok(None);
    }
    // A bounded score is a search artefact mid-window, not a real evaluation.
    if tokens.contains(&"lowerbound") || tokens.contains(&"upperbound") {
        return Ok(None);
    }

    let mut replay = pos.clone();
    let mut pv = Vec::new();
    for token in &tokens[start..] {
        let Ok(uci) = token.parse::<UciMove>() else { break };
        let Ok(mv) = uci.to_move(&replay) else { break };
        replay.play_unchecked(mv);
        pv.push(mv);
    }
    let Some(&mv) = pv.first() else {
        return Err(TeacherError::Protocol(format!(
            "principal variation did not start with a legal move: {}",
            line.trim()
        )));
    };

    Ok(Some((depth, multipv, Line { mv, score, pv }, nodes)))
}

/// Turn analysed lines into a policy target.
///
/// Scores become probabilities through a softmax in centipawns. `temperature`
/// is in centipawns too: a larger value spreads the distribution, a smaller one
/// sharpens it onto the best move. Moves the engine never reported keep whatever
/// mass the caller leaves for them, since MultiPV only covers the top few.
pub fn policy_target(lines: &[Line], temperature: f32) -> Vec<(Move, f32)> {
    if lines.is_empty() {
        return Vec::new();
    }
    let best = lines[0].score.to_cp() as f32;
    let weights: Vec<f32> = lines
        .iter()
        .map(|l| (((l.score.to_cp() as f32) - best) / temperature.max(1.0)).exp())
        .collect();
    let total: f32 = weights.iter().sum();
    lines
        .iter()
        .zip(weights)
        .map(|(l, w)| (l.mv, w / total))
        .collect()
}

/// How many centipawns correspond to a meaningful shift in win probability.
///
/// **This is engine-specific and must be calibrated.** Modern NNUE engines do
/// not report centipawns on the classical material scale; they report a
/// normalised score tied to win probability. Reckless scores a full queen down
/// at about -490, not -900, so a constant tuned for a classical evaluation
/// compresses every value label toward zero and teaches the network that losing
/// positions are milder than they are.
///
/// The default below is fitted to that observation rather than inherited from
/// classical-engine folklore, but it is still an approximation from a handful of
/// anchor positions. Calibrate it properly by playing games from labelled
/// positions and fitting reported score against actual outcome.
pub const DEFAULT_SCORE_SCALE: f32 = 200.0;

/// Map a score to a value target in [-1, 1] from the mover's point of view.
///
/// `scale` is the engine's centipawn scale; see [`DEFAULT_SCORE_SCALE`] for why
/// it is a parameter rather than a constant.
///
/// Note there is a way to avoid this conversion entirely: when data generation
/// plays whole games rather than labelling isolated positions, the value target
/// can be the game's actual result, which needs no calibration at all. That is
/// what AlphaZero does. Reserve this function for positions with no game
/// attached.
pub fn value_target_scaled(score: UciScore, scale: f32) -> f32 {
    match score {
        UciScore::Mate(n) if n > 0 => 1.0,
        UciScore::Mate(_) => -1.0,
        UciScore::Cp(cp) => {
            let win = 1.0 / (1.0 + (-(cp as f32) / scale.max(1.0)).exp());
            (win * 2.0 - 1.0).clamp(-1.0, 1.0)
        }
    }
}

/// [`value_target_scaled`] with [`DEFAULT_SCORE_SCALE`].
pub fn value_target(score: UciScore) -> f32 {
    value_target_scaled(score, DEFAULT_SCORE_SCALE)
}

/// Convenience: the standard settings for reproducible labelling.
pub fn configure_for_datagen(teacher: &mut Teacher, multipv: usize, hash_mb: usize) -> Result<()> {
    // One thread: multi-threaded search is non-deterministic, so the same
    // position would not always get the same label.
    teacher.set_option("Threads", "1")?;
    teacher.set_option("Hash", &hash_mb.to_string())?;
    teacher.set_multipv(multipv)?;
    Ok(())
}

/// Where the teacher engine lives, from the environment.
pub fn engine_path_from_env() -> Option<std::path::PathBuf> {
    std::env::var_os("CHESSZERO_TEACHER").map(std::path::PathBuf::from)
}
