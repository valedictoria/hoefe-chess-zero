# Where this project is

An Lc0-style chess engine whose priors come from a small language model that
writes a structured trace about the position. **The purpose is teaching a human,
not maximising playing strength**, and several design decisions only make sense
in that light.

Branch: `engine/lc0-lm`. Everything below is pushed.

## The shape of it

```
trainer/     Python + PyTorch: model definition, teacher annotator, weight export
engine/      Rust workspace: the actual engine
spec/        contracts that keep the two halves honest
```

| crate | what it does | state |
|---|---|---|
| `chesszero-core` | encoding, 4672 move index, 22 motif detectors | verified, 10 tests |
| `chesszero-nn` | candle transformer, loads trainer weights | verified, 6 tests |
| `chesszero-search` | tapered evaluation + alpha-beta | **eval tested (8), search UNTESTED** |
| `chesszero-teacher` | drives an external UCI engine; `datagen` binary | verified, 11 tests |

Python side: 42 tests, `.venv/bin/python -m pytest trainer/tests/`.

## Two contracts, and why they exist

The engine re-implements the trainer's encoding. Two implementations of one spec
drift, and the failure is silent: Rust loads Python's weights, feeds them subtly
different tokens, and plays badly for reasons no test explains.

- **`spec/golden.json`** — 2031 positions with exact prefix tokens, the policy
  index of every legal move, and annotator traces. Rust replays it and asserts
  byte-identical output. Regenerate with `cd trainer && python export.py golden`.
- **`spec/nn_reference.{json,safetensors}`** — a small network and its exact
  outputs, so the candle port is proven to match PyTorch to 1e-4.

Both have already caught real bugs. **Do not weaken them.** Note the nn fixture
uses *amplified* weights on purpose: built on the training initialisation, every
activation sits near zero where all GELU variants agree to ~1e-7, and the test
passed while the port used the wrong activation. Mutation-test after changing
either contract — a green suite that cannot fail is not evidence.

## Decisions already made

- **Rust engine, Python training.** Weights cross as safetensors.
- **Data generation uses alpha-beta / an external engine; MCTS is for live play.**
- **Reasoning runs at the root and the top-K nodes**, not every node.
- **The LM emits a fixed 32-token structured trace, never prose.** Its vocabulary
  is 196 tokens; there is no token for "the" or "because". It cannot write a
  sentence, which deletes the dominant hallucination mode by construction.
- **Every motif the model can name has a symbolic detector** behind it
  (`chesszero_core::motifs::detect`). That mapping is the validation oracle: a
  motif the model asserts and no detector confirms gets dropped, not shown.
- **The hand-crafted evaluation stays** even though the teacher is far stronger.
  It decomposes — "material level, king safety -0.6, pawn structure +0.2" —
  and a neural value head structurally cannot. For a teaching tool that
  breakdown is the more useful output. See `examples/breakdown.rs`.

## Not done

- `chesszero-search`'s alpha-beta has **no tests**. Evaluation is tested; the
  search, mate scoring and repetition detection are not. Do not trust it.
- `root_move_scores` does a full-window search per root move, costing 3-8x a
  single search. Mostly moot now that MultiPV does that job, but the fallback
  path still wants fixing.
- No MCTS, no UCI binary, no training loop, no teaching layer yet.
- The 12.5M-position data generation run has not been done.

## Open questions

**Teacher search depth.** Datagen at 10k nodes reaches mean depth 8.3, while
NNUE engines play at depth ~22. Depth costs roughly exponentially and label
quality improves sub-linearly, so datagen normally runs far shallower than match
play — but the right budget here has *not* been measured. Measure it before
committing to a long run.

**Value-target calibration.** Modern NNUE engines do not report classical
centipawns: Reckless scores a full queen down at about -490, not -900. A
constant tuned for classical evaluations compresses every value label toward
zero. `DEFAULT_SCORE_SCALE` is a first approximation from a few anchor
positions. Either calibrate it against real game outcomes, or sidestep it
entirely by using the game result as the value target — datagen already records
the result for exactly this reason.

**Does the reasoning earn its place?** Untested. Plan is `ReasonBoost=0` versus
default over ~200 games. If it does not help, the README should say so.

## Running things

```bash
# Rust
cd engine && cargo test --release
CHESSZERO_TEACHER=/path/to/engine cargo test --release   # includes teacher tests

# Python (needs the venv; see gotchas)
.venv/bin/python -m pytest trainer/tests/

# regenerate contracts after any encoding change
cd trainer && python export.py golden && python export.py reference

# data generation
cargo build --release -p chesszero-teacher --bin datagen
./engine/target/release/datagen --engine /path/to/reckless \
  --out data/teacher --positions 12_500_000 --workers 2 --nodes 10000 --multipv 4
```

Datagen is resumable: re-run the identical command and it continues. Measured
123 positions/sec on two workers, 176 on four, ~216 bytes per record.

## Environment gotchas

- **Use a virtualenv for Python.** Debian's patched setuptools cannot build
  `python-chess` from the system pip.
- **`shakmaty` is pinned to 0.30.0** in `Cargo.lock`; 0.30.1 requires rustc 1.95.
- **The teacher engine is never linked, only spawned.** Reckless is AGPL-3.0, and
  linking would pull that licence over this codebase including its network
  clause, which reaches anyone running the result as a web service. Separate
  processes over a pipe is the boundary every chess GUI relies on. Do not vendor
  the binary, and do not add it as a crate dependency.
- Teacher tests skip cleanly when `CHESSZERO_TEACHER` is unset, so the suite
  stays green without a copyleft binary present.

## Bugs found by testing, worth not reintroducing

Each of these passed review and failed only under measurement:

- En passant was encoded from the live board, but `fen()` omits it when no
  capture is legal — 161 of 2022 positions did not match themselves after a FEN
  round trip, and the engine only ever sees FENs.
- `hanging_pieces` valued attackers with the king priced at 0 (correct for
  material counting), making the king look like the cheapest attacker, so any
  defended piece beside the enemy king read as hanging.
- Back rank and bad bishop both fired on the *starting position*; skewer fired on
  a Ruy Lopez.
- Datagen worker seeds collided (`seed * K + worker | 1` gives adjacent workers
  the same stream when the product is odd) — exactly 50% duplicate records.
- Datagen resume replayed games already on disk.
- MultiPV selection took the deepest iteration even when a node budget cut it off
  after one line, so most records carried a one-move policy target.
- A fixed random-opening ply count made the engine always take over with White to
  move; generated games came out won by White roughly six to one.
