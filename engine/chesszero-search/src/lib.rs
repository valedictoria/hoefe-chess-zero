//! Search for the chesszero engine.
//!
//! Two searchers with different jobs. Alpha-beta generates training data: it is
//! fast, deterministic and needs no network, so the trainer has dense targets
//! from the first minute. MCTS plays games with the trained network.

pub mod alphabeta;
pub mod eval;

pub use alphabeta::{score_string, Limits, SearchResult, Searcher, MATE, MATE_THRESHOLD};
pub use eval::evaluate;
