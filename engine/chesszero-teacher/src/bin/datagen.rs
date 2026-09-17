//! Generate labelled training positions by playing games with a teacher engine.
//!
//! Positions come from *played games*, not random legal boards. Random boards
//! are mostly nonsense no player would ever reach, and a network trained on them
//! spends its capacity learning the wrong distribution. Each game opens with a
//! few random plies for variety, then the teacher plays it out, and every
//! position along the way is labelled with a MultiPV search.
//!
//! **Raw engine output is what gets stored, not derived targets.** Scores are
//! written exactly as the engine reported them, in centipawns or mate distance,
//! along with the eventual game result. Baking a value target into the file
//! would freeze in a centipawn-to-win-probability calibration that is
//! engine-specific and, at the time of writing, only approximately known -- and
//! getting it wrong would mean regenerating everything. Storing the raw numbers
//! means the conversion can be redone at training time for free.
//!
//! The run is resumable. A job of this size will be interrupted, and starting
//! over each time is not viable.

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chesszero_teacher::{configure_for_datagen, Limit, Teacher, TeacherError, UciScore};
use shakmaty::fen::Fen;
use shakmaty::{Chess, EnPassantMode, Position};

struct Config {
    engine: PathBuf,
    out: PathBuf,
    positions: u64,
    workers: usize,
    nodes: u64,
    multipv: usize,
    hash_mb: usize,
    random_plies: usize,
    max_plies: usize,
    shard_size: usize,
    seed: u64,
}

impl Config {
    fn parse() -> Result<Self, String> {
        let mut cfg = Config {
            engine: PathBuf::new(),
            out: PathBuf::from("data"),
            positions: 1_000_000,
            workers: 2,
            nodes: 10_000,
            multipv: 4,
            hash_mb: 64,
            random_plies: 8,
            max_plies: 240,
            shard_size: 50_000,
            seed: 24301,
        };
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < args.len() {
            let value = args.get(i + 1).cloned();
            let need = |v: Option<String>| -> Result<String, String> {
                v.ok_or_else(|| "flag is missing its value".to_string())
            };
            let number = |v: Option<String>| -> Result<u64, String> {
                need(v)?
                    .replace('_', "")
                    .parse()
                    .map_err(|_| "expected a number".to_string())
            };
            match args[i].as_str() {
                "--engine" => cfg.engine = PathBuf::from(need(value)?),
                "--out" => cfg.out = PathBuf::from(need(value)?),
                "--positions" => cfg.positions = number(value)?,
                "--workers" => cfg.workers = number(value)? as usize,
                "--nodes" => cfg.nodes = number(value)?,
                "--multipv" => cfg.multipv = number(value)? as usize,
                "--hash" => cfg.hash_mb = number(value)? as usize,
                "--random-plies" => cfg.random_plies = number(value)? as usize,
                "--max-plies" => cfg.max_plies = number(value)? as usize,
                "--shard-size" => cfg.shard_size = number(value)? as usize,
                "--seed" => cfg.seed = number(value)?,
                "--help" | "-h" => return Err(usage()),
                other => return Err(format!("unknown flag {other}\n\n{}", usage())),
            }
            i += 2;
        }
        if cfg.engine.as_os_str().is_empty() {
            return Err(format!("--engine is required\n\n{}", usage()));
        }
        if cfg.workers == 0 {
            return Err("--workers must be at least 1".into());
        }
        Ok(cfg)
    }
}

fn usage() -> String {
    "\
generate labelled training positions with a teacher engine

  datagen --engine <path> [options]

  --engine PATH        UCI engine binary (required)
  --out DIR            output directory (default: data)
  --positions N        how many positions to label (default: 1_000_000)
  --workers N          parallel engine processes (default: 2)
  --nodes N            search budget per position (default: 10_000)
  --multipv N          moves scored per position (default: 4)
  --hash MB            hash per engine (default: 64)
  --random-plies N     random opening plies per game (default: 8)
  --max-plies N        abandon a game after this many plies (default: 240)
  --shard-size N       records per output file (default: 50_000)
  --seed N             base seed; each worker offsets from it (default: 24301)

Resuming is automatic: existing records in --out are counted and the run
continues from there. Re-running with the same seed and settings reproduces
the same data.
"
    .to_string()
}

/// SplitMix64, used to turn a counter into a well-separated seed.
///
/// Worth the twenty lines. The obvious `seed * K + worker | 1` collapses
/// adjacent workers onto the *same* stream whenever the product is odd -- every
/// worker then plays identical games, and a run with N workers produces one
/// worker's worth of unique data at N times the cost. Measured at exactly 50%
/// duplicates on two workers before this replaced it.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The seed for one game, a pure function of run seed, worker and game index.
///
/// Per-game rather than a running stream so that resuming is exact: worker 3
/// game 900 always opens the same way, whether it is reached in one run or
/// after five interruptions.
fn game_seed(base: u64, worker: usize, game: u64) -> u64 {
    let s = splitmix64(base ^ splitmix64(worker as u64).wrapping_mul(0x1000_0000_0000_01B3))
        ^ splitmix64(game);
    if s == 0 { 0xDEAD_BEEF } else { s }
}

/// xorshift64; a dependency for this would be silly.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
}

/// One labelled position, held until its game finishes and the result is known.
struct Record {
    fen: String,
    game: u64,
    ply: usize,
    depth: u32,
    lines: Vec<(String, UciScore)>,
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

impl Record {
    fn write(&self, out: &mut impl Write, result: &str, nodes: u64) -> std::io::Result<()> {
        write!(
            out,
            "{{\"fen\":\"{}\",\"game\":{},\"ply\":{},\"depth\":{},\"nodes\":{},\"result\":\"{}\",\"lines\":[",
            escape(&self.fen),
            self.game,
            self.ply,
            self.depth,
            nodes,
            result
        )?;
        for (i, (mv, score)) in self.lines.iter().enumerate() {
            if i > 0 {
                write!(out, ",")?;
            }
            match score {
                UciScore::Cp(cp) => write!(out, "{{\"m\":\"{mv}\",\"cp\":{cp}}}")?,
                UciScore::Mate(n) => write!(out, "{{\"m\":\"{mv}\",\"mate\":{n}}}")?,
            }
        }
        writeln!(out, "]}}")
    }
}

/// Where a game ended, in PGN terms, from White's point of view.
fn game_result(pos: &Chess) -> Option<&'static str> {
    if pos.is_checkmate() {
        return Some(if pos.turn() == shakmaty::Color::White { "0-1" } else { "1-0" });
    }
    if pos.is_stalemate() || pos.is_insufficient_material() || pos.halfmoves() >= 100 {
        return Some("1/2-1/2");
    }
    None
}

struct Shards {
    dir: PathBuf,
    worker: usize,
    shard: usize,
    written: usize,
    limit: usize,
    file: Option<BufWriter<File>>,
}

impl Shards {
    fn new(dir: &Path, worker: usize, limit: usize) -> Self {
        Self { dir: dir.to_path_buf(), worker, shard: 0, written: 0, limit, file: None }
    }

    /// Skip past shards this worker already filled, so a resumed run appends
    /// rather than overwriting.
    /// Returns (records already present, the game index to start from).
    ///
    /// The game index matters as much as the count: restarting the generator
    /// from game zero would replay games already on disk, and the duplicates
    /// would be invisible in the output.
    fn resume_state(&mut self) -> std::io::Result<(u64, u64)> {
        let mut existing = 0u64;
        let mut highest_game = None;
        loop {
            let path = self.path();
            if !path.exists() {
                break;
            }
            let (lines, max_game) = scan_shard(&path)?;
            existing += lines;
            if let Some(g) = max_game {
                highest_game = Some(highest_game.map_or(g, |h: u64| h.max(g)));
            }
            if (lines as usize) < self.limit {
                // A partly-filled shard: carry on appending to it.
                self.written = lines as usize;
                break;
            }
            self.shard += 1;
        }
        Ok((existing, highest_game.map_or(0, |g| g + 1)))
    }

    fn path(&self) -> PathBuf {
        self.dir.join(format!("w{:02}-{:05}.jsonl", self.worker, self.shard))
    }

    fn writer(&mut self) -> std::io::Result<&mut BufWriter<File>> {
        if self.written >= self.limit && self.file.is_some() {
            if let Some(mut f) = self.file.take() {
                f.flush()?;
            }
            self.shard += 1;
            self.written = 0;
        }
        if self.file.is_none() {
            let file = OpenOptions::new().create(true).append(true).open(self.path())?;
            self.file = Some(BufWriter::new(file));
        }
        self.written += 1;
        Ok(self.file.as_mut().expect("just created"))
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if let Some(f) = self.file.as_mut() {
            f.flush()?;
        }
        Ok(())
    }
}

/// Count records in a shard and find the highest game index it holds.
fn scan_shard(path: &Path) -> std::io::Result<(u64, Option<u64>)> {
    use std::io::BufRead;
    let file = File::open(path)?;
    let mut count = 0u64;
    let mut max_game = None;
    for line in std::io::BufReader::new(file).lines() {
        let line = line?;
        count += 1;
        // Deliberately not a JSON parse: this runs over every shard at startup,
        // and the field is written by this program in a known position.
        if let Some(rest) = line.split("\"game\":").nth(1) {
            if let Ok(g) = rest.split(&[',', '}'][..]).next().unwrap_or("").trim().parse::<u64>() {
                max_game = Some(max_game.map_or(g, |m: u64| m.max(g)));
            }
        }
    }
    Ok((count, max_game))
}

fn main() {
    let cfg = match Config::parse() {
        Ok(c) => c,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };

    if let Err(e) = fs::create_dir_all(&cfg.out) {
        eprintln!("cannot create {}: {e}", cfg.out.display());
        std::process::exit(1);
    }

    let done = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let started = Instant::now();
    let cfg = Arc::new(cfg);

    println!(
        "labelling {} positions with {} worker(s), {} nodes/position, MultiPV {}",
        cfg.positions, cfg.workers, cfg.nodes, cfg.multipv
    );
    println!("output: {}", cfg.out.display());

    let mut handles = Vec::new();
    for worker in 0..cfg.workers {
        let cfg = Arc::clone(&cfg);
        let done = Arc::clone(&done);
        let stop = Arc::clone(&stop);
        handles.push(std::thread::spawn(move || {
            if let Err(e) = run_worker(worker, &cfg, &done, &stop) {
                eprintln!("worker {worker} stopped: {e}");
                stop.store(true, Ordering::Relaxed);
            }
        }));
    }

    // Progress, so a run measured in days can be steered rather than guessed at.
    let reporter = {
        let done = Arc::clone(&done);
        let stop = Arc::clone(&stop);
        let target = cfg.positions;
        std::thread::spawn(move || {
            let mut last = 0u64;
            while !stop.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_secs(10));
                let now = done.load(Ordering::Relaxed);
                if now >= target {
                    break;
                }
                let elapsed = started.elapsed().as_secs_f64();
                let rate = now as f64 / elapsed.max(1e-9);
                let remaining = (target.saturating_sub(now)) as f64 / rate.max(1e-9);
                if now != last {
                    println!(
                        "  {now}/{target} ({:.1}%)  {rate:.1}/sec  elapsed {}  eta {}",
                        now as f64 * 100.0 / target as f64,
                        human(elapsed),
                        human(remaining)
                    );
                    last = now;
                }
            }
        })
    };

    for h in handles {
        let _ = h.join();
    }
    stop.store(true, Ordering::Relaxed);
    let _ = reporter.join();

    let total = done.load(Ordering::Relaxed);
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "\n{total} positions in {} ({:.1}/sec)",
        human(elapsed),
        total as f64 / elapsed.max(1e-9)
    );
}

fn human(seconds: f64) -> String {
    let s = seconds as u64;
    if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else if s < 86400 {
        format!("{}h{:02}m", s / 3600, (s % 3600) / 60)
    } else {
        format!("{}d{:02}h", s / 86400, (s % 86400) / 3600)
    }
}

fn run_worker(
    worker: usize,
    cfg: &Config,
    done: &AtomicU64,
    stop: &AtomicBool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut shards = Shards::new(&cfg.out, worker, cfg.shard_size);
    let (resumed, mut game) = shards.resume_state()?;
    if resumed > 0 {
        println!("worker {worker}: resuming after {resumed} records, from game {game}");
        done.fetch_add(resumed, Ordering::Relaxed);
    }

    let mut teacher = Teacher::spawn(&cfg.engine)?;
    configure_for_datagen(&mut teacher, cfg.multipv, cfg.hash_mb)?;

    // Each worker gets its own stream, so workers never duplicate each other's
    // games while the whole run still reproduces from one seed.
    let mut skipped = 0u64;

    while !stop.load(Ordering::Relaxed) && done.load(Ordering::Relaxed) < cfg.positions {
        let mut rng = Rng(game_seed(cfg.seed, worker, game));
        teacher.new_game()?;
        let mut pos = Chess::default();

        // Random opening plies: without them every game is the same game.
        //
        // The count is jittered by one so the engine does not always take over
        // with White to move. A fixed count gives White first crack at punishing
        // whatever the random phase hung, and the generated games come out
        // lopsidedly won by White -- roughly six to one, measured -- which would
        // teach the network a first-move advantage far larger than the real one.
        let plies = cfg.random_plies + rng.below(2);
        for _ in 0..plies {
            let moves = pos.legal_moves();
            if moves.is_empty() {
                break;
            }
            pos.play_unchecked(moves[rng.below(moves.len())]);
        }

        let mut pending: Vec<Record> = Vec::new();
        let mut result = "*";
        let mut ply = 0usize;

        while ply < cfg.max_plies {
            if let Some(r) = game_result(&pos) {
                result = r;
                break;
            }
            if stop.load(Ordering::Relaxed) || done.load(Ordering::Relaxed) >= cfg.positions {
                break;
            }

            let analysis = match teacher.analyse(&pos, Limit::Nodes(cfg.nodes)) {
                Ok(a) => a,
                Err(TeacherError::NoLegalMoves) => break,
                Err(e) => return Err(Box::new(e)),
            };
            let Some(best) = analysis.best_move.or_else(|| analysis.best().map(|l| l.mv)) else {
                break;
            };

            // A budget that cuts the final iteration short can leave too few
            // scored moves to build a policy target from. Play on so the game
            // still produces positions, but do not label this one from a single
            // move -- a one-hot target teaches the network to be certain about
            // something the engine never actually compared.
            if analysis.lines.len() < 2 {
                skipped += 1;
                pos.play_unchecked(best);
                ply += 1;
                continue;
            }

            pending.push(Record {
                fen: Fen::from_position(&pos, EnPassantMode::Legal).to_string(),
                game,
                ply,
                depth: analysis.depth,
                lines: analysis
                    .lines
                    .iter()
                    .map(|l| {
                        (
                            l.mv.to_uci(shakmaty::CastlingMode::Standard).to_string(),
                            l.score,
                        )
                    })
                    .collect(),
            });

            pos.play_unchecked(best);
            ply += 1;
        }

        // The result is only known once the game ends, and it is the one value
        // target that needs no calibration, so every position waits for it.
        let writer = shards.writer()?;
        let mut written = 0u64;
        for record in &pending {
            record.write(writer, result, cfg.nodes)?;
            written += 1;
        }
        shards.flush()?;
        done.fetch_add(written, Ordering::Relaxed);
        game += 1;
    }

    shards.flush()?;
    if skipped > 0 {
        println!("worker {worker}: skipped {skipped} positions with too few scored moves");
    }
    teacher.quit();
    Ok(())
}
