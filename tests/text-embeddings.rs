#![cfg(feature = "hf-hub")]
#![allow(clippy::excessive_precision)]

use std::path::PathBuf;
use std::{
    fs,
    sync::{Mutex, OnceLock},
};

use hf_hub::Repo;
use ort::environment::Environment;

use fastembed::{
    get_cache_dir, Embedding, EmbeddingModel, InitOptionsUserDefined, OnnxSource, Pooling,
    QuantizationMode, RerankInitOptions, RerankInitOptionsUserDefined, RerankerModel,
    RerankerModelInfo, SparseInitOptions, SparseTextEmbedding, TextEmbedding, TextInitOptions,
    TextRerank, TokenizerFiles, UserDefinedEmbeddingModel, UserDefinedRerankingModel,
};

fn environment() -> Environment {
    static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
    ENVIRONMENT
        .get_or_init(|| ort::init().build().unwrap())
        .clone()
}

/// A small epsilon value for floating point comparisons.
const EPS: f32 = 2e-2;

/// A quantized model and the full-precision model it was quantized from.
///
/// ONNX Runtime runs quantized graphs' integer matrix products with kernels that differ by
/// instruction set: AVX512-VNNI accumulates exactly, AVX2 without VNNI saturates 16-bit sums, and
/// 4-bit blocks dequantize per platform. A quantized model's output therefore varies by machine,
/// by more than [`EPS`] in the embedding sums `verify_embeddings` pins, so it is checked against
/// its full-precision model instead, whose output no such kernel changes.
struct QuantizedVariant {
    full_precision: EmbeddingModel,
    /// The least cosine similarity each embedding keeps to the full-precision model's embedding
    /// of the same text: the least measured on an AVX512-VNNI x64 machine with ONNX Runtime 1.31,
    /// less 0.03 for other kernels, rounded down.
    min_cosine: f32,
}

fn quantized_variant(model: &EmbeddingModel) -> Option<QuantizedVariant> {
    let (full_precision, min_cosine) = match model {
        EmbeddingModel::AllMiniLML6V2Q => (EmbeddingModel::AllMiniLML6V2, 0.95),
        EmbeddingModel::AllMiniLML12V2Q => (EmbeddingModel::AllMiniLML12V2, 0.95),
        EmbeddingModel::BGEBaseENV15Q => (EmbeddingModel::BGEBaseENV15, 0.96),
        EmbeddingModel::BGELargeENV15Q => (EmbeddingModel::BGELargeENV15, 0.96),
        EmbeddingModel::BGESmallENV15Q => (EmbeddingModel::BGESmallENV15, 0.96),
        EmbeddingModel::GTEBaseENV15Q => (EmbeddingModel::GTEBaseENV15, 0.93),
        // The int8 GTE large keeps a cosine of only 0.79 to 0.90 on these texts.
        EmbeddingModel::GTELargeENV15Q => (EmbeddingModel::GTELargeENV15, 0.76),
        EmbeddingModel::MxbaiEmbedLargeV1Q => (EmbeddingModel::MxbaiEmbedLargeV1, 0.95),
        EmbeddingModel::NomicEmbedTextV15Q => (EmbeddingModel::NomicEmbedTextV15, 0.92),
        EmbeddingModel::ParaphraseMLMiniLML12V2Q => (EmbeddingModel::ParaphraseMLMiniLML12V2, 0.96),
        EmbeddingModel::EmbeddingGemma300MQ4 => (EmbeddingModel::EmbeddingGemma300M, 0.93),
        EmbeddingModel::EmbeddingGemma300MQ => (EmbeddingModel::EmbeddingGemma300M, 0.96),
        EmbeddingModel::SnowflakeArcticEmbedXSQ => (EmbeddingModel::SnowflakeArcticEmbedXS, 0.96),
        EmbeddingModel::SnowflakeArcticEmbedSQ => (EmbeddingModel::SnowflakeArcticEmbedS, 0.96),
        EmbeddingModel::SnowflakeArcticEmbedMQ => (EmbeddingModel::SnowflakeArcticEmbedM, 0.96),
        EmbeddingModel::SnowflakeArcticEmbedMLongQ => {
            (EmbeddingModel::SnowflakeArcticEmbedMLong, 0.95)
        }
        EmbeddingModel::SnowflakeArcticEmbedLQ => (EmbeddingModel::SnowflakeArcticEmbedL, 0.96),
        _ => return None,
    };
    Some(QuantizedVariant {
        full_precision,
        min_cosine,
    })
}

/// The cosine similarity of two embeddings.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

/// Precalculated embeddings for the supported models using #99
/// (4f09b6842ce1fcfaf6362678afcad9a176e05304).
///
/// These are the sum of all embedding values for each document. While not
/// perfect, they should be good enough to verify that the embeddings are being
/// generated correctly.
///
/// If you have just inserted a new `EmbeddingModel` variant, please update the
/// expected embeddings.
///
/// # Returns
///
/// If the embeddings are correct, this function returns `Ok(())`. If there are
/// any mismatches, it returns `Err(Vec<usize>)` with the indices of the
/// mismatched embeddings.
#[allow(unreachable_patterns)]
fn verify_embeddings(model: &EmbeddingModel, embeddings: &[Embedding]) -> Result<(), Vec<usize>> {
    let expected = match model {
        EmbeddingModel::AllMiniLML12V2 => [-0.12147753, 0.30144796, -0.06882502, -0.6303331],
        EmbeddingModel::AllMiniLML6V2 => [0.59605527, 0.36542925, -0.16450031, -0.40903988],
        EmbeddingModel::AllMpnetBaseV2 => [-0.21253541, -0.050802127, 0.14072442, -0.2908188],
        EmbeddingModel::BGEBaseENV15 => [-0.51290065, -0.4844747, -0.53036124, -0.5337459],
        EmbeddingModel::BGELargeENV15 => [-0.19347441, -0.28394595, -0.1549195, -0.22201893],
        EmbeddingModel::BGESmallENV15 => [0.09881669, 0.15151203, 0.12057499, 0.13641948],
        EmbeddingModel::BGESmallZHV15 => [-1.1194772, -1.0928253, -1.0325904, -1.0050416],
        EmbeddingModel::BGELargeZHV15 => [-0.62066114, -0.76666945, -0.7013123, -0.86202735],
        EmbeddingModel::BGEM3 => [-0.7138151, -0.69116485, -0.7932898, -0.6727733],
        EmbeddingModel::GTEBaseENV15 => [-1.6900877, -1.7148916, -1.7333382, -1.5121834],
        EmbeddingModel::GTELargeENV15 => [-1.6457459, -1.6582386, -1.6809471, -1.6070237],
        EmbeddingModel::ModernBertEmbedLarge => [ 0.24799639, 0.32174295, 0.17255782, 0.32919246],
        EmbeddingModel::MultilingualE5Base => [-0.057211064, -0.14287914, -0.071678676, -0.17549144],
        EmbeddingModel::MultilingualE5Large => [-0.7473163, -0.76040405, -0.7537941, -0.72920954],
        EmbeddingModel::MultilingualE5Small => [-0.2640718, -0.13929011, -0.08091972, -0.12388548],
        EmbeddingModel::MxbaiEmbedLargeV1 => [-0.2032495, -0.29803938, -0.15803768, -0.23155808],
        EmbeddingModel::NomicEmbedTextV1 => [0.13788113, 0.10750078, 0.050809078, 0.09284662],
        EmbeddingModel::NomicEmbedTextV15 => [0.1932303, 0.13795732, 0.14700879, 0.14940643],
        EmbeddingModel::ParaphraseMLMiniLML12V2 => [-0.07795018, -0.059113946, -0.043668486, -0.1880083],
        EmbeddingModel::ParaphraseMLMpnetBaseV2 => [0.39132136, 0.49490625, 0.65497226, 0.34237382],
        EmbeddingModel::ClipVitB32 => [0.7057363, 1.3549932, 0.46823958, 0.52351093],
        EmbeddingModel::JinaEmbeddingsV2BaseCode => [-0.31383067, -0.3758629, -0.24878195, -0.35373706],
        EmbeddingModel::JinaEmbeddingsV2BaseEN => [-0.055866606, -0.033922599, 0.012131551, -0.0132129812],
        EmbeddingModel::EmbeddingGemma300M => [0.22703816, 0.6947083, 0.07579082, 1.6958784],
        EmbeddingModel::SnowflakeArcticEmbedXS => [0.4418098, 0.46424747, 0.37932625, 0.44663674],
        EmbeddingModel::SnowflakeArcticEmbedS => [-0.64302516, -0.63146704, -0.57860875, -0.5829098],
        EmbeddingModel::SnowflakeArcticEmbedM => [-0.16999032, -0.109130904, -0.016444799, -0.108033374],
        EmbeddingModel::SnowflakeArcticEmbedMLong => [0.20396729, 0.18245143, 0.13489585, 0.15486401],
        EmbeddingModel::SnowflakeArcticEmbedL => [0.4049112, 0.42825335, 0.46401042, 0.4064963],
        _ => panic!("Model {model} not found. If you have just inserted this `EmbeddingModel` variant, please update the expected embeddings."),
    };

    let mismatched_indices = embeddings
        .iter()
        .map(|embedding| embedding.iter().sum::<f32>())
        .zip(expected.iter())
        .enumerate()
        .filter_map(|(i, (sum, &expected))| {
            if (sum - expected).abs() > EPS {
                eprintln!(
                    "Mismatched embeddings for model {model:?} at index {i}: {sum} != {expected} (expected)",
                    model = model,
                    i = i,
                    sum = sum,
                    expected = expected
                );
                Some(i)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    if mismatched_indices.is_empty() {
        Ok(())
    } else {
        Err(mismatched_indices)
    }
}

macro_rules! create_embeddings_test {
    (
        name: $name:ident,
        batch_size: $batch_size:expr,
    ) => {
        #[test]
        fn $name() {
            TextEmbedding::list_supported_models()
                .iter()
                .for_each(|supported_model| {
                    let mut model: TextEmbedding = TextEmbedding::try_new(
                        TextInitOptions::new(supported_model.model.clone()).with_environment(environment())
                    )
                    .unwrap();

                    let documents = vec![
                        "Hello, World!",
                        "This is an example passage.",
                        "fastembed-rs is licensed under Apache-2.0",
                        "Some other short text here blah blah blah",
                    ];

                    // Generate embeddings with the default batch size, 256
                    let batch_size = $batch_size;
                    let embeddings = model.embed(documents.clone(), batch_size);

                    if matches!(
                        (batch_size, TextEmbedding::get_quantization_mode(&supported_model.model)),
                        (Some(n), QuantizationMode::Dynamic) if n < documents.len()
                    ) {
                        // For Dynamic quantization, the batch size must be greater than or equal to the number of documents
                        // Otherwise, an error is expected
                        assert!(embeddings.is_err(), "Expected error for batch size < document count for {model} using dynamic quantization.", model=supported_model.model);
                    } else {
                        let embeddings = embeddings.unwrap_or_else(
                            |exc| panic!("Expected embeddings for {model} to be generated successfully: {exc}", model=supported_model.model, exc=exc),
                        );
                        assert_eq!(embeddings.len(), documents.len());

                        for embedding in &embeddings {
                            assert_eq!(embedding.len(), supported_model.dim);
                        }

                        if let Some(QuantizedVariant { full_precision, min_cosine }) =
                            quantized_variant(&supported_model.model)
                        {
                            let reference = TextEmbedding::try_new(
                                TextInitOptions::new(full_precision.clone())
                                    .with_environment(environment()),
                            )
                            .unwrap()
                            .embed(documents.clone(), batch_size)
                            .unwrap();
                            for (document, (embedding, reference)) in
                                documents.iter().zip(embeddings.iter().zip(&reference))
                            {
                                let similarity = cosine(embedding, reference);
                                assert!(
                                    similarity >= min_cosine,
                                    "{model} embeds {document:?} at cosine {similarity} to {full_precision}, below {min_cosine}",
                                    model = supported_model.model,
                                );
                            }
                        } else if let Err(mismatched_indices) =
                            verify_embeddings(&supported_model.model, &embeddings)
                        {
                            panic!(
                                "Mismatched embeddings for model {model}: {sentences:?}",
                                model = supported_model.model,
                                sentences = &mismatched_indices
                                    .iter()
                                    .map(|&i| documents[i])
                                    .collect::<Vec<_>>()
                            );
                        }
                    }
                });
        }

    };
}

create_embeddings_test!(
    name: test_batch_size_default,
    batch_size: None,
);

#[test]
fn test_sparse_embeddings() {
    SparseTextEmbedding::list_supported_models()
        .iter()
        // The inference-free models produce far denser documents and have a query path of their
        // own, they are covered by the `if_splade` suite instead.
        .filter(|supported_model| {
            supported_model
                .additional_files
                .iter()
                .all(|f| f != "idf.json")
        })
        .for_each(|supported_model| {
            let mut model: SparseTextEmbedding = SparseTextEmbedding::try_new(
                SparseInitOptions::new(supported_model.model.clone())
                    .with_environment(environment()),
            )
            .unwrap();

            let documents = vec![
                "Hello, World!",
                "This is an example passage.",
                "fastembed-rs is licensed under Apache-2.0",
                "Some other short text here blah blah blah",
            ];

            // Generate embeddings with the default batch size, 256
            let embeddings = model.embed(documents.clone(), None).unwrap();

            assert_eq!(embeddings.len(), documents.len());
            embeddings.into_iter().for_each(|embedding| {
                assert!(embedding.values.iter().all(|&v| v > 0.0));
                assert!(embedding.indices.len() < 100);
                assert_eq!(embedding.indices.len(), embedding.values.len());
            });

            // Symmetric models have no separate query representation
            assert!(model.query_embed(documents).is_err());

            // Clear the model cache to avoid running out of space on GitHub Actions.
            if std::env::var("CI").is_ok() {
                clean_cache(supported_model.model_code.clone())
            }
        });
}

#[test]
fn test_user_defined_embedding_model() {
    // Constitute the model in order to ensure it's downloaded and cached
    let test_model_info = TextEmbedding::get_model_info(&EmbeddingModel::AllMiniLML6V2).unwrap();

    TextEmbedding::try_new(
        TextInitOptions::new(test_model_info.model.clone()).with_environment(environment()),
    )
    .unwrap();

    // Get the directory of the model
    let model_name = test_model_info.model_code.replace('/', "--");
    let model_dir = model_cache_dir().join(format!("models--{}", model_name));

    // Find the "snapshots" sub-directory
    let snapshots_dir = model_dir.join("snapshots");

    // Get the first sub-directory in snapshots
    let model_files_dir = snapshots_dir
        .read_dir()
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();

    // Find the onnx file - it will be any file ending with .onnx
    let onnx_file = std::fs::read(
        model_files_dir
            .read_dir()
            .unwrap()
            .find(|entry| {
                entry
                    .as_ref()
                    .unwrap()
                    .path()
                    .extension()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    == "onnx"
            })
            .unwrap()
            .unwrap()
            .path(),
    )
    .expect("Could not read onnx file");

    // Load the tokenizer files
    let tokenizer_files = TokenizerFiles {
        tokenizer_file: std::fs::read(model_files_dir.join("tokenizer.json"))
            .expect("Could not read tokenizer.json"),
        config_file: std::fs::read(model_files_dir.join("config.json"))
            .expect("Could not read config.json"),
        special_tokens_map_file: std::fs::read(model_files_dir.join("special_tokens_map.json"))
            .expect("Could not read special_tokens_map.json"),
        tokenizer_config_file: std::fs::read(model_files_dir.join("tokenizer_config.json"))
            .expect("Could not read tokenizer_config.json"),
    };
    // Create a UserDefinedEmbeddingModel
    let user_defined_model =
        UserDefinedEmbeddingModel::new(onnx_file, tokenizer_files).with_pooling(Pooling::Mean);

    // Try creating a TextEmbedding instance from the user-defined model
    let mut user_defined_text_embedding = TextEmbedding::try_new_from_user_defined(
        user_defined_model,
        InitOptionsUserDefined::default().with_environment(environment()),
    )
    .unwrap();

    let documents = vec![
        "Hello, World!",
        "This is an example passage.",
        "fastembed-rs is licensed under Apache-2.0",
        "Some other short text here blah blah blah",
    ];

    // Generate embeddings over documents
    let embeddings = user_defined_text_embedding
        .embed(documents.clone(), None)
        .unwrap();
    assert_eq!(embeddings.len(), documents.len());
    for embedding in embeddings {
        assert_eq!(embedding.len(), test_model_info.dim);
    }
}

/// Held by the tests that use the reranker cache, which `test_rerank` clears on CI while another
/// test may read a model from it.
static RERANKER_CACHE: Mutex<()> = Mutex::new(());

#[test]
fn test_rerank() {
    let _cache = RERANKER_CACHE.lock().unwrap_or_else(|err| err.into_inner());
    let test_one_model = |supported_model: &RerankerModelInfo| {
        println!("supported_model: {:?}", supported_model);

        let mut result = TextRerank::try_new(
            RerankInitOptions::new(supported_model.model.clone()).with_environment(environment()),
        )
        .unwrap();

        let documents = vec![
            "hi",
            "The giant panda, sometimes called a panda bear or simply panda, is a bear species endemic to China.",
            "panda is an animal",
            "i dont know",
            "kind of mammal",
        ];

        let results = result
            .rerank("what is panda?", documents.clone(), true, None)
            .unwrap();

        assert_eq!(
            results.len(),
            documents.len(),
            "rerank model {:?} failed",
            supported_model
        );

        let option_a = "panda is an animal";
        let option_b = "The giant panda, sometimes called a panda bear or simply panda, is a bear species endemic to China.";

        assert!(
            results[0].document.as_ref().unwrap() == option_a
                || results[0].document.as_ref().unwrap() == option_b
        );
        assert!(
            results[1].document.as_ref().unwrap() == option_a
                || results[1].document.as_ref().unwrap() == option_b
        );
        assert_ne!(
            results[0].document, results[1].document,
            "The top two results should be different"
        );

        // Clear the model cache to avoid running out of space on GitHub Actions.
        if std::env::var("CI").is_ok() {
            clean_cache(supported_model.model_code.clone())
        }
    };
    TextRerank::list_supported_models()
        .iter()
        .for_each(test_one_model);
}

#[ignore]
#[test]
fn test_user_defined_reranking_large_model() {
    // Setup model to download from Hugging Face
    let cache = hf_hub::Cache::new(std::path::PathBuf::from(&fastembed::get_cache_dir()));
    let api = hf_hub::api::sync::ApiBuilder::from_cache(cache)
        .with_progress(true)
        .build()
        .expect("Failed to build API from cache");
    let model_repo = api.model("rozgo/bge-reranker-v2-m3".to_string());

    // Download the onnx model file
    let onnx_file = model_repo.download("model.onnx").unwrap();
    // Onnx model exceeds the limit of 2GB for a file, so we need to download the data file separately
    let _onnx_data_file = model_repo.get("model.onnx.data").unwrap();

    // OnnxSource::File is used to load the onnx file using onnx session builder commit_from_file
    let onnx_source = OnnxSource::File(onnx_file);

    // Load the tokenizer files
    let tokenizer_files: TokenizerFiles = TokenizerFiles {
        tokenizer_file: std::fs::read(model_repo.get("tokenizer.json").unwrap()).unwrap(),
        config_file: std::fs::read(model_repo.get("config.json").unwrap()).unwrap(),
        special_tokens_map_file: std::fs::read(model_repo.get("special_tokens_map.json").unwrap())
            .unwrap(),

        tokenizer_config_file: std::fs::read(model_repo.get("tokenizer_config.json").unwrap())
            .unwrap(),
    };

    let model = UserDefinedRerankingModel::new(onnx_source, tokenizer_files);

    let mut user_defined_reranker = TextRerank::try_new_from_user_defined(
        model,
        RerankInitOptionsUserDefined::default().with_environment(environment()),
    )
    .unwrap();

    let documents = vec![
        "Hello, World!",
        "This is an example passage.",
        "fastembed-rs is licensed under Apache-2.0",
        "Some other short text here blah blah blah",
    ];

    let results = user_defined_reranker
        .rerank("Ciao, Earth!", documents.clone(), false, None)
        .unwrap();

    assert_eq!(results.len(), documents.len());
    assert_eq!(results.first().unwrap().index, 0);
}

#[test]
fn test_user_defined_reranking_model() {
    let _cache = RERANKER_CACHE.lock().unwrap_or_else(|err| err.into_inner());
    // Constitute the model in order to ensure it's downloaded and cached
    let test_model_info: fastembed::RerankerModelInfo =
        TextRerank::get_model_info(&RerankerModel::JINARerankerV1TurboEn);

    TextRerank::try_new(
        RerankInitOptions::new(test_model_info.model).with_environment(environment()),
    )
    .unwrap();

    // Get the directory of the model
    let model_name = test_model_info.model_code.replace('/', "--");
    let model_dir = model_cache_dir().join(format!("models--{}", model_name));

    // Find the "snapshots" sub-directory
    let snapshots_dir = model_dir.join("snapshots");

    // Get the first sub-directory in snapshots
    let model_files_dir = snapshots_dir
        .read_dir()
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();

    // Find the onnx file - it will be any file in ./onnx ending with .onnx
    let onnx_file = std::fs::read(
        model_files_dir
            .join("onnx")
            .read_dir()
            .unwrap()
            .find(|entry| {
                entry
                    .as_ref()
                    .unwrap()
                    .path()
                    .extension()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    == "onnx"
            })
            .unwrap()
            .unwrap()
            .path(),
    )
    .expect("Could not read onnx file");

    // Load the tokenizer files
    let tokenizer_files = TokenizerFiles {
        tokenizer_file: std::fs::read(model_files_dir.join("tokenizer.json"))
            .expect("Could not read tokenizer.json"),
        config_file: std::fs::read(model_files_dir.join("config.json"))
            .expect("Could not read config.json"),
        special_tokens_map_file: std::fs::read(model_files_dir.join("special_tokens_map.json"))
            .expect("Could not read special_tokens_map.json"),
        tokenizer_config_file: std::fs::read(model_files_dir.join("tokenizer_config.json"))
            .expect("Could not read tokenizer_config.json"),
    };
    // Create a UserDefinedEmbeddingModel
    let user_defined_model = UserDefinedRerankingModel::new(onnx_file, tokenizer_files);

    // Try creating a TextEmbedding instance from the user-defined model
    let mut user_defined_reranker = TextRerank::try_new_from_user_defined(
        user_defined_model,
        RerankInitOptionsUserDefined::default().with_environment(environment()),
    )
    .unwrap();

    let documents = vec![
        "Hello, World!",
        "This is an example passage.",
        "fastembed-rs is licensed under Apache-2.0",
        "Some other short text here blah blah blah",
    ];

    // Generate embeddings over documents
    let results = user_defined_reranker
        .rerank("Ciao, Earth!", documents.clone(), false, None)
        .unwrap();

    assert_eq!(results.len(), documents.len());
    assert_eq!(results.first().unwrap().index, 0);
}

/// The directory fastembed downloads models into: `HF_HOME` when it is set, as fastembed's Hub
/// client prefers it, and the fastembed cache directory otherwise.
fn model_cache_dir() -> PathBuf {
    std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(get_cache_dir()))
}

fn clean_cache(model_code: String) {
    let repo = Repo::model(model_code);
    fs::remove_dir_all(model_cache_dir().join(repo.folder_name())).ok();
}

// This is item "test-environment-aeghhgwpe-pro02a" of the [Aguana corpus](http://argumentation.bplaced.net/arguana/data)
fn get_sample_text() -> String {
    let t = include_str!("assets/sample_text.txt");
    t.to_string()
}

#[test]
fn test_batch_size_does_not_change_output() {
    let mut model = TextEmbedding::try_new(
        TextInitOptions::new(EmbeddingModel::AllMiniLML6V2)
            .with_environment(environment())
            .with_max_length(384),
    )
    .expect("Create model successfully");

    let sentences = vec![
        "Books are no more threatened by Kindle than stairs by elevators.",
        "You are who you are when nobody's watching.",
        "An original idea. That can't be too hard. The library must be full of them.",
        "Gaia visited her daughter Mnemosyne, who was busy being unpronounceable.",
        "You can never be overdressed or overeducated.",
        "I don't want to go to heaven. None of my friends are there.",
        "I never travel without my diary. One should always have something sensational to read in the train.",
        "I can resist anything except temptation.",
        "It is absurd to divide people into good and bad. People are either charming or tedious."
    ];

    let single_batch = model
        .embed(sentences.clone(), None)
        .expect("create successfully");
    let small_batch = model
        .embed(sentences, Some(3))
        .expect("create successfully");

    assert_eq!(single_batch.len(), small_batch.len());
    for (a, b) in single_batch.into_iter().zip(small_batch) {
        assert!(a == b, "Expect each sentence embedding are equal.");
    }
}

#[test]
fn test_bgesmallen1point5_match_python_counterpart() {
    let mut model = TextEmbedding::try_new(
        TextInitOptions::new(EmbeddingModel::BGESmallENV15)
            .with_environment(environment())
            .with_max_length(384),
    )
    .expect("Create model successfully");

    let text = get_sample_text();

    // baseline is generated in python using Xenova/bge-small-en-v1.5.onnx
    // Tokenize with python SentenceTransformer("BAAI/bge-small-en-v1.5") default tokenizer
    // with (text, padding="max_length",max_length=384,truncation=True, return_tensors="np").
    // Normalized and pooled with SentenceTransformer("BAAI/bge-small-en-v1.5") default pooling settings.
    // We only take 10 items to keep the test file small.
    let baseline: Vec<f32> = vec![
        4.208_193_7e-2,
        -2.748_133_2e-2,
        6.742_810_5e-2,
        2.282_790_5e-2,
        4.257_192e-2,
        -4.163_983_5e-2,
        6.814_807_4e-6,
        -9.643_933e-3,
        -3.475_583e-3,
        6.606_272e-2,
    ];

    let embeddings = model.embed(vec![text], None).expect("create successfully");
    let tolerance: f32 = 1e-3;
    for (expected, actual) in embeddings[0]
        .clone()
        .into_iter()
        .take(baseline.len())
        .zip(baseline)
    {
        assert!((expected - actual).abs() < tolerance);
    }
}

#[test]
fn test_allminilml6v2_match_python_counterpart() {
    let mut model = TextEmbedding::try_new(
        TextInitOptions::new(EmbeddingModel::AllMiniLML6V2)
            .with_environment(environment())
            .with_max_length(384),
    )
    .expect("Create model successfully");

    let text = get_sample_text();

    // The model's tokenizer declares `model_max_length` 256, which caps the requested 384, so the
    // text is truncated at 256 tokens. The baseline is generated in Python by
    // `tests/assets/allminilml6v2_baseline.py`: the same graph through onnxruntime, the tokenizer
    // truncating at 256 tokens, mean pooling and normalization. We only take 10 items to keep the
    // test file small.
    let baseline: Vec<f32> = vec![
        2.613_225_6e-2,
        4.206_722_4e-2,
        5.250_315_7e-2,
        7.876_658_1e-2,
        4.569_452_4e-2,
        2.103_072_1e-3,
        -2.805_862_1e-2,
        -2.223_154_5e-2,
        2.083_955_5e-3,
        2.926_191_8e-2,
    ];

    let embeddings = model.embed(vec![text], None).expect("create successfully");
    let tolerance: f32 = 1e-6;
    for (expected, actual) in embeddings[0]
        .clone()
        .into_iter()
        .take(baseline.len())
        .zip(baseline)
    {
        assert!((expected - actual).abs() < tolerance);
    }
}

// Ref: https://github.com/Anush008/fastembed-rs/issues/171#issue-3209484009
#[test]
fn clip_vit_b32_deterministic_across_calls() {
    let q = "red car";
    let mut fe = TextEmbedding::try_new(
        TextInitOptions::new(EmbeddingModel::ClipVitB32).with_environment(environment()),
    )
    .unwrap();
    let first = fe.embed(vec![q], None).unwrap()[0].clone();
    for i in 1..100 {
        let vecs = fe.embed(vec![q], None).unwrap();
        assert_eq!(vecs[0], first, "Embedding changed after {i} iterations");
    }
}
