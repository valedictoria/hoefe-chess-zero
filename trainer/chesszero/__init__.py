"""chesszero trainer: the PyTorch half of an Lc0-style engine with a reasoning LM.

The engine itself lives in ``engine/`` and is written in Rust.  This package owns
the model definition, the teacher annotator and the training loop, and exports
both the weights and ``spec/golden.json`` -- the encoding contract the Rust side
is validated against.
"""

from .model import ChessZeroNet, ModelConfig, PRESETS
from .vocab import SEQ_LEN, PREFIX_LEN, TRACE_LEN, VOCAB_SIZE

__version__ = "0.1.0"

__all__ = [
    "ChessZeroNet",
    "ModelConfig",
    "PRESETS",
    "SEQ_LEN",
    "PREFIX_LEN",
    "TRACE_LEN",
    "VOCAB_SIZE",
    "__version__",
]
