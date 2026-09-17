//! Negamax alpha-beta with quiescence, a transposition table and MVV/LVA ordering.
//!
//! Its job is data generation: it labels positions fast and deterministically so
//! the network has something real to learn from long before self-play could
//! produce anything useful. Live play uses MCTS over the trained network.

use std::time::{Duration, Instant};

use shakmaty::zobrist::Zobrist64;
use shakmaty::{Chess, Color, EnPassantMode, Move, MoveList, Position};

use crate::eval::{evaluate, piece_value};

pub const MATE: i32 = 32_000;
/// Anything beyond this is a forced mate rather than an evaluation.
pub const MATE_THRESHOLD: i32 = MATE - 1_000;
const INFINITY: i32 = MATE + 1;
const MAX_PLY: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
struct TtEntry {
    key: u64,
    depth: i32,
    score: i32,
    bound: Bound,
    best: Option<Move>,
}

#[derive(Debug, Clone)]
pub struct Limits {
    pub depth: u32,
    pub nodes: u64,
    pub time: Option<Duration>,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            depth: 6,
            nodes: u64::MAX,
            time: None,
        }
    }
}

impl Limits {
    pub fn depth(depth: u32) -> Self {
        Self {
            depth,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub best: Option<Move>,
    pub score: i32,
    pub pv: Vec<Move>,
    pub nodes: u64,
    pub depth: u32,
}

pub struct Searcher {
    table: Vec<Option<TtEntry>>,
    killers: [[Option<Move>; 2]; MAX_PLY],
    history: Vec<u64>,
    nodes: u64,
    limit_nodes: u64,
    deadline: Option<Instant>,
    aborted: bool,
}

impl Default for Searcher {
    fn default() -> Self {
        Self::new(1 << 20)
    }
}

impl Searcher {
    pub fn new(tt_entries: usize) -> Self {
        Self {
            table: vec![None; tt_entries.next_power_of_two()],
            killers: [[None; 2]; MAX_PLY],
            history: Vec::new(),
            nodes: 0,
            limit_nodes: u64::MAX,
            deadline: None,
            aborted: false,
        }
    }

    /// Positions already seen in the game, so repetitions score as draws.
    pub fn set_history(&mut self, history: Vec<u64>) {
        self.history = history;
    }

    pub fn nodes(&self) -> u64 {
        self.nodes
    }

    fn key(pos: &Chess) -> u64 {
        let hash: Zobrist64 = pos.zobrist_hash(EnPassantMode::Legal);
        hash.0
    }

    fn out_of_time(&mut self) -> bool {
        if self.aborted {
            return true;
        }
        if self.nodes >= self.limit_nodes {
            self.aborted = true;
            return true;
        }
        // checking the clock every node is itself measurable
        if self.nodes % 2048 == 0 {
            if let Some(deadline) = self.deadline {
                if Instant::now() >= deadline {
                    self.aborted = true;
                    return true;
                }
            }
        }
        false
    }

    /// Iterative deepening. Shallow results order the next iteration's moves,
    /// which is what makes deepening cheaper than searching the full depth flat.
    pub fn search(&mut self, pos: &Chess, limits: &Limits) -> SearchResult {
        self.nodes = 0;
        self.aborted = false;
        self.limit_nodes = limits.nodes;
        self.deadline = limits.time.map(|t| Instant::now() + t);
        self.killers = [[None; 2]; MAX_PLY];

        let mut result = SearchResult {
            best: pos.legal_moves().first().cloned(),
            score: 0,
            pv: Vec::new(),
            nodes: 0,
            depth: 0,
        };

        for depth in 1..=limits.depth {
            let score = self.alphabeta(pos, depth as i32, -INFINITY, INFINITY, 0);
            if self.aborted && depth > 1 {
                break; // an interrupted iteration is not trustworthy
            }
            let pv = self.principal_variation(pos, depth as usize);
            result = SearchResult {
                best: pv.first().cloned().or(result.best),
                score,
                pv,
                nodes: self.nodes,
                depth,
            };
            if score.abs() >= MATE_THRESHOLD {
                break; // a forced mate will not improve
            }
        }
        result.nodes = self.nodes;
        result
    }

    fn probe(&self, key: u64) -> Option<&TtEntry> {
        let slot = (key as usize) & (self.table.len() - 1);
        self.table[slot].as_ref().filter(|e| e.key == key)
    }

    fn store(&mut self, key: u64, depth: i32, score: i32, bound: Bound, best: Option<Move>) {
        let slot = (key as usize) & (self.table.len() - 1);
        let replace = match &self.table[slot] {
            Some(existing) => existing.key != key || depth >= existing.depth,
            None => true,
        };
        if replace {
            self.table[slot] = Some(TtEntry { key, depth, score, bound, best });
        }
    }

    fn is_repetition(&self, key: u64) -> bool {
        self.history.iter().filter(|k| **k == key).count() >= 2
    }

    fn alphabeta(&mut self, pos: &Chess, depth: i32, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        self.nodes += 1;
        if self.out_of_time() {
            return 0;
        }

        if ply > 0 {
            if pos.halfmoves() >= 100 || pos.is_insufficient_material() {
                return 0;
            }
            let key = Self::key(pos);
            if self.is_repetition(key) {
                return 0;
            }
        }

        let key = Self::key(pos);
        let mut tt_move = None;
        if let Some(entry) = self.probe(key) {
            tt_move = entry.best.clone();
            if ply > 0 && entry.depth >= depth {
                match entry.bound {
                    Bound::Exact => return entry.score,
                    Bound::Lower if entry.score >= beta => return entry.score,
                    Bound::Upper if entry.score <= alpha => return entry.score,
                    _ => {}
                }
            }
        }

        if depth <= 0 {
            return self.quiesce(pos, alpha, beta, ply);
        }

        let mut moves = pos.legal_moves();
        if moves.is_empty() {
            return if pos.is_check() { -MATE + ply as i32 } else { 0 };
        }
        self.order(&mut moves, tt_move.as_ref(), ply);

        let original_alpha = alpha;
        let mut best_move = None;
        let mut best_score = -INFINITY;

        for m in moves.iter() {
            let mut child = pos.clone();
            child.play_unchecked(*m);
            self.history.push(Self::key(&child));
            let score = -self.alphabeta(&child, depth - 1, -beta, -alpha, ply + 1);
            self.history.pop();

            if score > best_score {
                best_score = score;
                best_move = Some(m.clone());
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                // a quiet move that causes a cutoff is worth trying first next time
                if !m.is_capture() && ply < MAX_PLY {
                    self.killers[ply][1] = self.killers[ply][0].take();
                    self.killers[ply][0] = Some(m.clone());
                }
                break;
            }
        }

        let bound = if best_score <= original_alpha {
            Bound::Upper
        } else if best_score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.store(key, depth, best_score, bound, best_move);
        best_score
    }

    /// Search only captures until the position is quiet, so the evaluation is
    /// never read in the middle of an exchange.
    fn quiesce(&mut self, pos: &Chess, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        self.nodes += 1;
        if self.out_of_time() {
            return 0;
        }

        let stand_pat = evaluate(pos);
        if stand_pat >= beta {
            return stand_pat;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }
        if ply >= MAX_PLY - 1 {
            return stand_pat;
        }

        let mut captures: MoveList = pos.legal_moves();
        captures.retain(|m| m.is_capture() || m.promotion().is_some());
        if captures.is_empty() {
            return alpha;
        }
        self.order(&mut captures, None, ply);

        let mut best = stand_pat;
        for m in captures.iter() {
            let mut child = pos.clone();
            child.play_unchecked(*m);
            let score = -self.quiesce(&child, -beta, -alpha, ply + 1);
            if score > best {
                best = score;
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                break;
            }
        }
        best
    }

    /// Transposition-table move first, then winning captures by MVV/LVA, then
    /// killers, then the rest.
    fn order(&self, moves: &mut MoveList, tt_move: Option<&Move>, ply: usize) {
        let killers = if ply < MAX_PLY {
            self.killers[ply]
        } else {
            [None, None]
        };
        moves.sort_by_key(|m| {
            if Some(m) == tt_move {
                return -1_000_000;
            }
            if let Some(victim) = m.capture() {
                let attacker = m.role();
                // most valuable victim, least valuable attacker
                return -100_000 - (piece_value(victim) * 16 - piece_value(attacker));
            }
            if m.promotion().is_some() {
                return -90_000;
            }
            if killers[0].as_ref() == Some(m) {
                return -80_000;
            }
            if killers[1].as_ref() == Some(m) {
                return -79_000;
            }
            0
        });
    }

    fn principal_variation(&self, pos: &Chess, max_len: usize) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut current = pos.clone();
        let mut seen = Vec::new();
        for _ in 0..max_len {
            let key = Self::key(&current);
            if seen.contains(&key) {
                break; // the table can cycle; a PV must not
            }
            seen.push(key);
            let Some(entry) = self.probe(key) else { break };
            let Some(best) = entry.best.clone() else { break };
            if !current.legal_moves().contains(&best) {
                break; // a key collision handed us a move from another position
            }
            current.play_unchecked(best);
            pv.push(best);
        }
        pv
    }

    /// Score every root move, for turning a search into a policy target.
    pub fn root_move_scores(&mut self, pos: &Chess, limits: &Limits) -> Vec<(Move, i32)> {
        self.nodes = 0;
        self.aborted = false;
        self.limit_nodes = limits.nodes;
        self.deadline = limits.time.map(|t| Instant::now() + t);

        let mut scored: Vec<(Move, i32)> = Vec::new();
        for m in pos.legal_moves().iter() {
            let mut child = pos.clone();
            child.play_unchecked(*m);
            self.history.push(Self::key(&child));
            let score = -self.alphabeta(&child, limits.depth as i32 - 1, -INFINITY, INFINITY, 1);
            self.history.pop();
            scored.push((m.clone(), score));
        }
        scored.sort_by_key(|(_, score)| -score);
        scored
    }
}

/// Describe a score the way UCI does: mate distance, or centipawns.
pub fn score_string(score: i32) -> String {
    if score.abs() >= MATE_THRESHOLD {
        let plies = MATE - score.abs();
        let moves = (plies + 1) / 2;
        format!("mate {}", if score > 0 { moves } else { -moves })
    } else {
        format!("cp {score}")
    }
}

/// Side-to-move colour helper for callers building training targets.
pub fn perspective(pos: &Chess) -> Color {
    pos.turn()
}
