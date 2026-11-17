#![cfg(feature = "hf-hub")]
#![allow(clippy::excessive_precision)]

use fastembed::{
    get_cache_dir, Bgem3Embedding, Bgem3InitOptions, Bgem3Model, InitOptionsUserDefined,
    TokenizerFiles, UserDefinedBgem3Model,
};
use ort::environment::Environment;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

static MODEL_LOCK: Mutex<()> = Mutex::new(());
const EPS: f32 = 2e-2;
/// How far the int8 graph's sparse weights may lie from the fp32 model's.
///
/// `MatMulInteger` multiplies dynamically quantized `u8` activations by `s8` weights, and ONNX
/// Runtime's CPU kernels for it differ by instruction set: AVX512-VNNI accumulates exactly, while
/// AVX2 without VNNI saturates its 16-bit intermediate sums. The int8 weights therefore vary by
/// machine: for these sentences two x64 machines differ by up to 0.031, while each stays within
/// 0.025 of the fp32 model. The weights are checked against the fp32 reference, which no kernel
/// changes, within this bound.
const INT8_SPARSE_BOUND: f32 = 3e-2;

fn environment() -> Environment {
    static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
    ENVIRONMENT
        .get_or_init(|| ort::init().build().unwrap())
        .clone()
}

#[test]
fn test_bgem3_joint_embeddings_match_python() {
    let _guard = MODEL_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let mut model = Bgem3Embedding::try_new(
        Bgem3InitOptions::new(Bgem3Model::BGEM3Q).with_environment(environment()),
    )
    .expect("Failed to initialize BGEM3Q model");

    let sentences = vec![
        "fastembed-rs is licensed under Apache  2.0",
        "Superman is the best superhero of all time",
    ];

    let output = model.embed(sentences, None).expect("Embedding failed");

    // 1. Verify Dense Embeddings
    assert_eq!(output.dense.len(), 2);
    assert_eq!(output.dense[0].len(), 1024);
    assert_eq!(output.dense[1].len(), 1024);

    let expected_dense_0 = [
        -0.018156249076128006,
        -0.017044715583324432,
        0.000982290250249207,
        0.0583689846098423,
        -0.01868816465139389,
    ];
    let expected_dense_1 = [
        -0.011247089132666588,
        0.031949788331985474,
        -0.02800164744257927,
        -0.009801163338124752,
        -0.014338407665491104,
    ];

    for (i, val) in expected_dense_0.iter().enumerate() {
        assert!((output.dense[0][i] - val).abs() < EPS);
    }
    for (i, val) in expected_dense_1.iter().enumerate() {
        assert!((output.dense[1][i] - val).abs() < EPS);
    }

    // 2. Verify Sparse Embeddings
    assert_eq!(output.sparse.len(), 2);

    // The fp32 model's weights, from FlagEmbedding: `BGEM3FlagModel('BAAI/bge-m3')`, as
    // `bgem3_comparison.rs` pins them for the same sentences.
    let expected_sparse_0: HashMap<usize, f32> = [
        (9, 0.04153823),
        (71, 0.08359783),
        (83, 0.07778944),
        (195, 0.16260204),
        (1379, 0.10846229),
        (1430, 0.15291117),
        (4271, 0.17456965),
        (4295, 0.24856839),
        (9795, 0.1580239),
        (13482, 0.28582922),
        (16655, 0.223301),
        (86872, 0.17708361),
    ]
    .into_iter()
    .collect();

    let expected_sparse_1: HashMap<usize, f32> = [
        (70, 0.16743071),
        (83, 0.19853045),
        (111, 0.11962792),
        (756, 0.13541803),
        (1601, 0.17550871),
        (1733, 0.18622744),
        (2965, 0.24451455),
        (90865, 0.25476876),
        (183497, 0.32012847),
    ]
    .into_iter()
    .collect();

    assert_eq!(output.sparse[0].indices.len(), expected_sparse_0.len());
    for (idx, val) in output.sparse[0]
        .indices
        .iter()
        .zip(output.sparse[0].values.iter())
    {
        let expected_val = expected_sparse_0
            .get(idx)
            .expect("Unexpected index in sparse 0");
        assert!(
            (val - expected_val).abs() < INT8_SPARSE_BOUND,
            "Sparse 0 index {}: expected {}, got {}",
            idx,
            expected_val,
            val
        );
    }

    assert_eq!(output.sparse[1].indices.len(), expected_sparse_1.len());
    for (idx, val) in output.sparse[1]
        .indices
        .iter()
        .zip(output.sparse[1].values.iter())
    {
        let expected_val = expected_sparse_1
            .get(idx)
            .expect("Unexpected index in sparse 1");
        assert!(
            (val - expected_val).abs() < INT8_SPARSE_BOUND,
            "Sparse 1 index {}: expected {}, got {}",
            idx,
            expected_val,
            val
        );
    }

    // 3. Verify ColBERT Embeddings
    assert_eq!(output.colbert.len(), 2);
    assert_eq!(output.colbert[0].len(), 13);
    assert_eq!(output.colbert[1].len(), 10);

    let expected_colbert_0_tok1 = [
        -0.02416383847594261,
        -0.0405534990131855,
        -0.0560004822909832,
    ];
    let expected_colbert_0_tok2 = [
        -0.01845022290945053,
        -0.042646653950214386,
        -0.033078353852033615,
    ];
    let expected_colbert_1_tok1 = [
        -0.013565482571721077,
        -0.04746083542704582,
        -0.027890587225556374,
    ];
    let expected_colbert_1_tok2 = [
        0.004542498383671045,
        -0.05220562964677811,
        -0.045384544879198074,
    ];

    for (i, val) in expected_colbert_0_tok1.iter().enumerate() {
        assert!((output.colbert[0][0][i] - val).abs() < EPS);
    }
    for (i, val) in expected_colbert_0_tok2.iter().enumerate() {
        assert!((output.colbert[0][1][i] - val).abs() < EPS);
    }
    for (i, val) in expected_colbert_1_tok1.iter().enumerate() {
        assert!((output.colbert[1][0][i] - val).abs() < EPS);
    }
    for (i, val) in expected_colbert_1_tok2.iter().enumerate() {
        assert!((output.colbert[1][1][i] - val).abs() < EPS);
    }
}

#[test]
fn test_bgem3_user_defined_model() {
    let _guard = MODEL_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    // We will verify the user-defined loader by pulling the files from HF and feeding them manually to simulate a local deployment

    // Reuse fastembed's cache, the model is already downloaded by test_bgem3_joint_embeddings_match_python
    let cache = hf_hub::Cache::new(std::path::PathBuf::from(get_cache_dir()));
    let model_repo = hf_hub::api::sync::ApiBuilder::from_cache(cache)
        .with_progress(false)
        .build()
        .expect("Failed to build API client")
        .model(Bgem3Model::BGEM3Q.to_string());

    let onnx_file = std::fs::read(
        model_repo
            .get("model_quantized.onnx")
            .expect("Failed to get model file"),
    )
    .expect("Failed to read model file");

    let tokenizer_files = TokenizerFiles {
        tokenizer_file: std::fs::read(model_repo.get("tokenizer.json").unwrap()).unwrap(),
        config_file: std::fs::read(model_repo.get("config.json").unwrap()).unwrap(),
        special_tokens_map_file: std::fs::read(model_repo.get("special_tokens_map.json").unwrap())
            .unwrap(),
        tokenizer_config_file: std::fs::read(model_repo.get("tokenizer_config.json").unwrap())
            .unwrap(),
    };

    let user_model = UserDefinedBgem3Model::new(onnx_file, tokenizer_files);
    let mut model = Bgem3Embedding::try_new_from_user_defined(
        user_model,
        InitOptionsUserDefined::default().with_environment(environment()),
    )
    .expect("Failed to build user defined BGEM3 model");

    let sentences = vec![
        "fastembed-rs is licensed under Apache  2.0",
        "Superman is the best superhero of all time",
    ];

    let output = model.embed(sentences, None).expect("Embedding failed");

    assert_eq!(output.dense.len(), 2);
    assert_eq!(output.sparse.len(), 2);
    assert_eq!(output.colbert.len(), 2);

    // Verify a simple weight from the first sentence to ensure it produces identical numbers
    let expected_dense_0 = [
        -0.018156249076128006,
        -0.017044715583324432,
        0.000982290250249207,
        0.0583689846098423,
        -0.01868816465139389,
    ];
    for (i, val) in expected_dense_0.iter().enumerate() {
        assert!((output.dense[0][i] - val).abs() < EPS);
    }
}

#[test]
fn test_bgem3_custom_max_length() {
    let _guard = MODEL_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    // Verify that the user can override the max length (e.g. to 5 tokens) and it successfully truncates
    let mut model = Bgem3Embedding::try_new(
        Bgem3InitOptions::new(Bgem3Model::BGEM3Q)
            .with_environment(environment())
            .with_max_length(5),
    )
    .expect("Failed to initialize BGEM3Q model with custom max length");

    let sentences = vec!["fastembed-rs is licensed under Apache 2.0 and is very cool"];

    let output = model.embed(sentences, None).expect("Embedding failed");

    // The sentence "fastembed-rs is licensed under Apache 2.0 and is very cool" has more than 5 tokens.
    // If truncated to 5 tokens, the ColBERT output token count (excluding CLS) should be exactly 4 tokens
    // since the total tokens with CLS/EOS is limited by max_length = 5.
    assert_eq!(output.colbert[0].len(), 4);
}
