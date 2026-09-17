//! The prefix-LM transformer, in candle, loading the trainer's safetensors.
//!
//! This has to match PyTorch numerically, not just structurally. Two details
//! bite in practice and both are pinned by `tests/parity.rs`:
//!
//! * PyTorch's `nn.GELU()` defaults to the exact erf formulation, while the
//!   obvious candle call (`gelu`) is the tanh approximation. They differ by
//!   enough to move a move choice.
//! * The attention mask is a *prefix* mask, not a causal one: the 70 position
//!   tokens all see each other, while trace tokens stay causal. Getting this
//!   wrong still loads, still runs, and only shows up as an engine that plays
//!   badly.

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{
    embedding, layer_norm, linear, linear_no_bias, ops::softmax_last_dim, Embedding, LayerNorm,
    LayerNormConfig, Linear, Module, VarBuilder,
};

use chesszero_core::vocab::{PREFIX_LEN, SEQ_LEN};

/// Matches the trainer's `ModelConfig`; read back from safetensors metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelConfig {
    pub d_model: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub d_ff: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub prefix_len: usize,
    pub n_moves: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            d_model: 128,
            n_layer: 4,
            n_head: 4,
            d_ff: 384,
            vocab_size: chesszero_core::vocab::vocab_size(),
            seq_len: SEQ_LEN,
            prefix_len: PREFIX_LEN,
            n_moves: 4672,
        }
    }
}

/// `true` where query `i` may attend to key `j`.
///
/// A board is not a sequence: square a1 has every right to look at h8, so the
/// prefix is bidirectional. Only the trace is autoregressive.
pub fn prefix_mask(seq_len: usize, prefix_len: usize) -> Vec<bool> {
    let mut mask = vec![false; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let causal = j <= i;
            let both_in_prefix = i < prefix_len && j < prefix_len;
            mask[i * seq_len + j] = causal || both_in_prefix;
        }
    }
    mask
}

/// The mask as an additive bias: 0 where allowed, -inf where not, exactly as
/// PyTorch turns a boolean `attn_mask` into scores.
fn mask_bias(seq_len: usize, prefix_len: usize, device: &Device) -> Result<Tensor> {
    let allowed = prefix_mask(seq_len, prefix_len);
    let bias: Vec<f32> = allowed
        .iter()
        .map(|ok| if *ok { 0.0 } else { f32::NEG_INFINITY })
        .collect();
    Tensor::from_vec(bias, (seq_len, seq_len), device)
}

struct Attention {
    qkv: Linear,
    proj: Linear,
    n_head: usize,
    d_head: usize,
}

impl Attention {
    fn load(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            qkv: linear_no_bias(cfg.d_model, 3 * cfg.d_model, vb.pp("qkv"))?,
            proj: linear_no_bias(cfg.d_model, cfg.d_model, vb.pp("proj"))?,
            n_head: cfg.n_head,
            d_head: cfg.d_model / cfg.n_head,
        })
    }

    fn forward(&self, x: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let (b, s, d) = x.dims3()?;
        let qkv = self.qkv.forward(x)?;

        let split = |i: usize| -> Result<Tensor> {
            qkv.narrow(D::Minus1, i * d, d)?
                .reshape((b, s, self.n_head, self.d_head))?
                .transpose(1, 2)?
                .contiguous()
        };
        let (q, k, v) = (split(0)?, split(1)?, split(2)?);

        let scale = 1.0 / (self.d_head as f64).sqrt();
        let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        // (s, s) broadcasts over batch and heads
        let scores = scores.broadcast_add(&bias.i((..s, ..s))?)?;
        let attn = softmax_last_dim(&scores)?;

        attn.matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, s, d))?
            .apply(&self.proj)
    }
}

struct Block {
    ln1: LayerNorm,
    attn: Attention,
    ln2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

fn norm(size: usize, vb: VarBuilder) -> Result<LayerNorm> {
    // PyTorch's nn.LayerNorm default eps
    layer_norm(size, LayerNormConfig { eps: 1e-5, ..Default::default() }, vb)
}

impl Block {
    fn load(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln1: norm(cfg.d_model, vb.pp("ln1"))?,
            attn: Attention::load(cfg, vb.pp("attn"))?,
            ln2: norm(cfg.d_model, vb.pp("ln2"))?,
            // nn.Sequential indices: 0 = Linear, 1 = GELU, 2 = Linear, 3 = Dropout
            fc1: linear(cfg.d_model, cfg.d_ff, vb.pp("mlp").pp("0"))?,
            fc2: linear(cfg.d_ff, cfg.d_model, vb.pp("mlp").pp("2"))?,
        })
    }

    fn forward(&self, x: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.ln1.forward(x)?, bias)?)?;
        let h = self
            .fc1
            .forward(&self.ln2.forward(&x)?)?
            .gelu_erf()? // PyTorch nn.GELU(): exact, not the tanh approximation
            .apply(&self.fc2)?;
        x + h
    }
}

/// LayerNorm -> Linear -> GELU -> Linear, matching the trainer's `_head`.
struct Head {
    ln: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl Head {
    fn load(d_model: usize, d_out: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln: norm(d_model, vb.pp("0"))?,
            fc1: linear(d_model, d_model, vb.pp("1"))?,
            fc2: linear(d_model, d_out, vb.pp("3"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.ln
            .forward(x)?
            .apply(&self.fc1)?
            .gelu_erf()?
            .apply(&self.fc2)
    }
}

/// What one forward pass yields.
pub struct Outputs {
    pub policy: Tensor,
    pub value: Tensor,
    pub mlh: Tensor,
    pub policy_reasoned: Tensor,
    pub value_reasoned: Tensor,
}

pub struct ChessZeroNet {
    cfg: ModelConfig,
    tok_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    policy_head: Head,
    value_head: Head,
    mlh_head: Head,
    bias: Tensor,
    device: Device,
}

impl ChessZeroNet {
    pub fn load(vb: VarBuilder, cfg: ModelConfig, device: &Device) -> Result<Self> {
        let blocks = (0..cfg.n_layer)
            .map(|i| Block::load(&cfg, vb.pp("blocks").pp(i.to_string())))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            tok_emb: embedding(cfg.vocab_size, cfg.d_model, vb.pp("tok_emb"))?,
            pos_emb: embedding(cfg.seq_len, cfg.d_model, vb.pp("pos_emb"))?,
            blocks,
            ln_f: norm(cfg.d_model, vb.pp("ln_f"))?,
            policy_head: Head::load(cfg.d_model, cfg.n_moves, vb.pp("policy_head"))?,
            value_head: Head::load(cfg.d_model, 3, vb.pp("value_head"))?,
            mlh_head: Head::load(cfg.d_model, 1, vb.pp("mlh_head"))?,
            bias: mask_bias(cfg.seq_len, cfg.prefix_len, device)?,
            device: device.clone(),
            cfg,
        })
    }

    /// Load from a safetensors file written by `export.py weights`.
    pub fn from_safetensors(path: &std::path::Path, device: &Device) -> Result<Self> {
        let cfg = read_config(path)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)?
        };
        Self::load(vb, cfg, device)
    }

    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Hidden states for a `(batch, seq)` block of token ids.
    pub fn trunk(&self, ids: &Tensor) -> Result<Tensor> {
        let (_, s) = ids.dims2()?;
        let positions = Tensor::arange(0u32, s as u32, &self.device)?;
        let mut x = self
            .tok_emb
            .forward(ids)?
            .broadcast_add(&self.pos_emb.forward(&positions)?)?;
        for block in &self.blocks {
            x = block.forward(&x, &self.bias)?;
        }
        self.ln_f.forward(&x)
    }

    /// Full forward: both the position-only and the reasoned readouts.
    pub fn forward(&self, ids: &Tensor) -> Result<Outputs> {
        let h = self.trunk(ids)?;
        let s = h.dim(1)?;
        let fast = h.i((.., (self.cfg.prefix_len - 1).min(s - 1), ..))?;
        let reasoned = h.i((.., s - 1, ..))?;
        Ok(Outputs {
            policy: self.policy_head.forward(&fast)?,
            value: self.value_head.forward(&fast)?,
            mlh: self.mlh_head.forward(&fast)?.squeeze(D::Minus1)?,
            policy_reasoned: self.policy_head.forward(&reasoned)?,
            value_reasoned: self.value_head.forward(&reasoned)?,
        })
    }

    /// The hot path for search leaves: prefix only, no trace, no language head.
    pub fn position_only(&self, ids: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let prefix = ids.narrow(1, 0, self.cfg.prefix_len)?;
        let h = self.trunk(&prefix)?;
        let fast = h.i((.., self.cfg.prefix_len - 1, ..))?;
        Ok((
            self.policy_head.forward(&fast)?,
            self.value_head.forward(&fast)?,
            self.mlh_head.forward(&fast)?.squeeze(D::Minus1)?,
        ))
    }

    /// Language-model logits, reusing the token embedding: the head is tied, so
    /// the trainer does not ship a separate matrix for it.
    pub fn lm_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        hidden.broadcast_matmul(&self.tok_emb.embeddings().t()?)
    }
}

/// Read the trainer's config out of the safetensors metadata.
///
/// The metadata is authoritative, not the tensor shapes: `n_head` cannot be
/// recovered from shapes at all (it only partitions `d_model`), and guessing it
/// would silently produce a different attention pattern than the one trained.
pub fn read_config(path: &std::path::Path) -> Result<ModelConfig> {
    use candle_core::Error;

    let bytes = std::fs::read(path).map_err(Error::wrap)?;
    let (_, metadata) = safetensors::SafeTensors::read_metadata(&bytes).map_err(Error::wrap)?;
    let map = metadata
        .metadata()
        .as_ref()
        .ok_or_else(|| Error::Msg(format!("{path:?} carries no metadata; re-export it")))?;

    let get = |key: &str| -> Result<usize> {
        map.get(key)
            .ok_or_else(|| Error::Msg(format!("{path:?} metadata is missing {key}")))?
            .parse::<usize>()
            .map_err(|e| Error::Msg(format!("{path:?} metadata {key}: {e}")))
    };

    let cfg = ModelConfig {
        d_model: get("d_model")?,
        n_layer: get("n_layer")?,
        n_head: get("n_head")?,
        d_ff: get("d_ff")?,
        vocab_size: get("vocab_size")?,
        seq_len: get("seq_len")?,
        prefix_len: get("prefix_len")?,
        n_moves: get("n_moves")?,
    };

    // The engine and the trainer must agree about the shape of a sequence, or
    // the readout positions point at the wrong hidden states.
    if cfg.prefix_len != PREFIX_LEN || cfg.seq_len != SEQ_LEN {
        return Err(Error::Msg(format!(
            "weights were trained with prefix_len={} seq_len={}, but this engine uses {PREFIX_LEN} and {SEQ_LEN}",
            cfg.prefix_len, cfg.seq_len
        )));
    }
    if cfg.vocab_size != chesszero_core::vocab::vocab_size() {
        return Err(Error::Msg(format!(
            "weights have a {}-token vocabulary, this engine has {}",
            cfg.vocab_size,
            chesszero_core::vocab::vocab_size()
        )));
    }
    if cfg.d_model % cfg.n_head != 0 {
        return Err(Error::Msg(format!(
            "d_model {} is not divisible by n_head {}",
            cfg.d_model, cfg.n_head
        )));
    }
    Ok(cfg)
}
