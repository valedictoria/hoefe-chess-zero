//! A (middlegame, endgame) score pair, and the taper that collapses it.
//!
//! Every evaluation term is worth different amounts at different points in the
//! game -- a king wants a pawn shield with queens on and wants to march to the
//! centre once they come off. The usual mistake is to switch tables at a
//! material threshold, which makes the evaluation *discontinuous*: one capture
//! crosses the line and the score jumps tens of centipawns with nothing else
//! about the position having changed. Search then sees a phantom gain and plays
//! for it. Carrying both values and interpolating removes the cliff.

use std::ops::{Add, AddAssign, Mul, Neg, Sub};

/// Phase units: the material on a full board, used to interpolate.
pub const MAX_PHASE: i32 = 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Score {
    pub mg: i32,
    pub eg: i32,
}

impl Score {
    pub const ZERO: Score = Score { mg: 0, eg: 0 };

    pub const fn new(mg: i32, eg: i32) -> Self {
        Self { mg, eg }
    }

    /// The same value regardless of phase.
    pub const fn flat(value: i32) -> Self {
        Self { mg: value, eg: value }
    }

    /// Collapse to centipawns. `phase` is MAX_PHASE with everything on the
    /// board and 0 in a bare king-and-pawn endgame.
    pub fn taper(self, phase: i32) -> i32 {
        let phase = phase.clamp(0, MAX_PHASE);
        (self.mg * phase + self.eg * (MAX_PHASE - phase)) / MAX_PHASE
    }
}

impl Add for Score {
    type Output = Score;
    fn add(self, rhs: Score) -> Score {
        Score::new(self.mg + rhs.mg, self.eg + rhs.eg)
    }
}

impl Sub for Score {
    type Output = Score;
    fn sub(self, rhs: Score) -> Score {
        Score::new(self.mg - rhs.mg, self.eg - rhs.eg)
    }
}

impl Neg for Score {
    type Output = Score;
    fn neg(self) -> Score {
        Score::new(-self.mg, -self.eg)
    }
}

impl Mul<i32> for Score {
    type Output = Score;
    fn mul(self, rhs: i32) -> Score {
        Score::new(self.mg * rhs, self.eg * rhs)
    }
}

impl AddAssign for Score {
    fn add_assign(&mut self, rhs: Score) {
        self.mg += rhs.mg;
        self.eg += rhs.eg;
    }
}
