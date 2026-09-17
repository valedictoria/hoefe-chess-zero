//! Board encoding, move indexing and motif detection for the chesszero engine.
//!
//! This crate is a deliberate re-implementation of the Python trainer's
//! `vocab.py`, `encoding.py` and the feature half of `reasoning.py`. Two
//! implementations of one spec drift, and when they do the failure is silent:
//! the engine loads the trainer's weights, feeds them subtly different tokens,
//! and plays badly for reasons no unit test explains.
//!
//! `spec/golden.json` exists to make that impossible. It records exactly what
//! Python produces for a few thousand positions, and `tests/golden.rs` asserts
//! this crate reproduces it byte for byte.

pub mod encoding;
pub mod features;
pub mod motifs;
pub mod trace;
pub mod vocab;

pub use encoding::{
    canonical, canonical_square, encode_prefix, index_to_move, move_squares, move_to_index,
    legal_move_indices, N_MOVES, N_PLANES,
};
pub use vocab::{tid, token, vocab, vocab_size, PREFIX_LEN, SEQ_LEN, TRACE_LEN};
