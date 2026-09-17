//! Assert the candle port reproduces PyTorch on the same weights.
//!
//! `golden.json` pins the encoding; this pins the network. Without it a port
//! that gets GELU or the attention mask subtly wrong still loads, still runs,
//! and the only symptom is an engine that plays badly for no visible reason.

use std::path::PathBuf;

use candle_core::{Device, IndexOp, Tensor};
use chesszero_nn::{prefix_mask, read_config, ChessZeroNet};
use chesszero_core::vocab::{PREFIX_LEN, SEQ_LEN};
use serde_json::Value;

/// Generous next to f32 noise, tight enough to catch a wrong activation:
/// tanh-approximate GELU differs from the exact form by ~1e-3.
const TOL: f32 = 1e-4;

fn spec_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../spec")
}

fn reference() -> Value {
    let path = spec_dir().join("nn_reference.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("{path:?} missing -- run `python export.py reference`"));
    serde_json::from_str(&text).unwrap()
}

fn net() -> ChessZeroNet {
    ChessZeroNet::from_safetensors(&spec_dir().join("nn_reference.safetensors"), &Device::Cpu)
        .expect("reference weights failed to load")
}

fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

fn assert_close(ours: &[f32], want: &[f32], what: &str) {
    assert_eq!(ours.len(), want.len(), "{what}: length differs");
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for (i, (a, b)) in ours.iter().zip(want).enumerate() {
        let diff = (a - b).abs();
        if diff > worst {
            worst = diff;
            worst_at = i;
        }
    }
    assert!(
        worst <= TOL,
        "{what}: worst difference {worst:.3e} at index {worst_at} (ours {}, want {})",
        ours[worst_at],
        want[worst_at]
    );
}

fn input_tensor(reference: &Value) -> Tensor {
    let rows: Vec<Vec<u32>> = reference["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect())
        .collect();
    let batch = rows.len();
    let seq = rows[0].len();
    Tensor::from_vec(rows.concat(), (batch, seq), &Device::Cpu).unwrap()
}

#[test]
fn config_round_trips_through_metadata() {
    let reference = reference();
    let cfg = read_config(&spec_dir().join("nn_reference.safetensors")).unwrap();
    let want = &reference["config"];
    assert_eq!(cfg.d_model, want["d_model"].as_u64().unwrap() as usize);
    assert_eq!(cfg.n_layer, want["n_layer"].as_u64().unwrap() as usize);
    assert_eq!(cfg.n_head, want["n_head"].as_u64().unwrap() as usize);
    assert_eq!(cfg.d_ff, want["d_ff"].as_u64().unwrap() as usize);
    assert_eq!(cfg.vocab_size, want["vocab_size"].as_u64().unwrap() as usize);
    assert_eq!(cfg.seq_len, SEQ_LEN);
    assert_eq!(cfg.prefix_len, PREFIX_LEN);
}

#[test]
fn heads_match_pytorch() {
    let reference = reference();
    let net = net();
    let ids = input_tensor(&reference);
    let out = net.forward(&ids).unwrap();
    let expected = &reference["outputs"];

    for (i, want) in expected["policy"].as_array().unwrap().iter().enumerate() {
        let ours = out.policy.i(i).unwrap().to_vec1::<f32>().unwrap();
        assert_close(&ours, &floats(want), &format!("policy[{i}]"));
    }
    for (i, want) in expected["value"].as_array().unwrap().iter().enumerate() {
        let ours = out.value.i(i).unwrap().to_vec1::<f32>().unwrap();
        assert_close(&ours, &floats(want), &format!("value[{i}]"));
    }
    for (i, want) in expected["policy_reasoned"].as_array().unwrap().iter().enumerate() {
        let ours = out.policy_reasoned.i(i).unwrap().to_vec1::<f32>().unwrap();
        assert_close(&ours, &floats(want), &format!("policy_reasoned[{i}]"));
    }
    for (i, want) in expected["value_reasoned"].as_array().unwrap().iter().enumerate() {
        let ours = out.value_reasoned.i(i).unwrap().to_vec1::<f32>().unwrap();
        assert_close(&ours, &floats(want), &format!("value_reasoned[{i}]"));
    }
    assert_close(
        &out.mlh.to_vec1::<f32>().unwrap(),
        &floats(&expected["mlh"]),
        "mlh",
    );
}

#[test]
fn language_head_matches_pytorch() {
    // The tied head is not in the file at all, so this also proves the engine
    // reuses the token embedding the way the trainer does.
    let reference = reference();
    let net = net();
    let ids = input_tensor(&reference);
    let hidden = net.trunk(&ids).unwrap();
    let logits = net.lm_logits(&hidden).unwrap();
    let expected = &reference["outputs"];

    for (i, want) in expected["lm_at_sep"].as_array().unwrap().iter().enumerate() {
        let ours = logits.i((i, PREFIX_LEN - 1)).unwrap().to_vec1::<f32>().unwrap();
        assert_close(&ours, &floats(want), &format!("lm_at_sep[{i}]"));
    }
    for (i, want) in expected["lm_at_last"].as_array().unwrap().iter().enumerate() {
        let ours = logits.i((i, SEQ_LEN - 1)).unwrap().to_vec1::<f32>().unwrap();
        assert_close(&ours, &floats(want), &format!("lm_at_last[{i}]"));
    }
}

#[test]
fn position_only_path_matches_the_full_forward() {
    let reference = reference();
    let net = net();
    let ids = input_tensor(&reference);
    let (policy, _, _) = net.position_only(&ids).unwrap();
    for (i, want) in reference["outputs"]["policy_position_only"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let ours = policy.i(i).unwrap().to_vec1::<f32>().unwrap();
        assert_close(&ours, &floats(want), &format!("policy_position_only[{i}]"));
    }
}

#[test]
fn position_only_ignores_the_trace() {
    // The MCTS hot path must not depend on trace tokens it never fills in.
    let net = net();
    let ids = input_tensor(&reference());
    let (batch, seq) = ids.dims2().unwrap();

    let mut scrambled: Vec<u32> = ids.flatten_all().unwrap().to_vec1().unwrap();
    for b in 0..batch {
        for t in PREFIX_LEN..seq {
            scrambled[b * seq + t] = 7;
        }
    }
    let other = Tensor::from_vec(scrambled, (batch, seq), &Device::Cpu).unwrap();

    let a = net.forward(&ids).unwrap().policy.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = net.forward(&other).unwrap().policy.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_close(&a, &b, "policy must not see the trace");
}

#[test]
fn mask_is_bidirectional_over_the_board_and_causal_over_the_trace() {
    let mask = prefix_mask(SEQ_LEN, PREFIX_LEN);
    let at = |i: usize, j: usize| mask[i * SEQ_LEN + j];
    assert!(at(0, PREFIX_LEN - 1) && at(PREFIX_LEN - 1, 0), "board is bidirectional");
    assert!(at(5, 60) && at(60, 5));
    assert!(!at(PREFIX_LEN, PREFIX_LEN + 1), "trace is causal");
    assert!(at(PREFIX_LEN + 1, PREFIX_LEN));
    assert!(at(SEQ_LEN - 1, 5), "trace still sees the board");
}
