pub(crate) use fastembed_pipeline::{
    encode_batch, load_tokenizer, EncodedBatch, Error, Result, TokenizerFiles,
};
#[cfg(feature = "hf-hub")]
use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use ort::{
    environment::Environment,
    ep::ExecutionProviderDispatch,
    session::{
        builder::{GraphOptimizationLevel, SessionBuilder},
        SessionInputValue,
    },
    value::Value,
};
use std::borrow::Cow;
#[cfg(feature = "hf-hub")]
use std::path::PathBuf;
#[cfg(feature = "hf-hub")]
use tokenizers::Tokenizer;

const DEFAULT_CACHE_DIR: &str = ".fastembed_cache";

pub fn get_cache_dir() -> String {
    std::env::var("FASTEMBED_CACHE_DIR").unwrap_or(DEFAULT_CACHE_DIR.into())
}

#[cfg(feature = "hf-hub")]
fn hf_hub_error(e: hf_hub::api::sync::ApiError) -> Error {
    Error::Other(format!("HuggingFace API error: {e}"))
}

/// An ONNX Runtime error as the pipeline's [`Error::Ort`].
pub(crate) fn ort_error(e: ort::Error) -> Error {
    Error::Ort(Box::new(e))
}

/// Loading tokenizer files from the Hugging Face Hub is separate from the main
/// `load_tokenizer` function, which expects bytes from any source.
#[cfg(feature = "hf-hub")]
pub fn load_tokenizer_hf_hub(model_repo: ApiRepo, max_length: usize) -> Result<Tokenizer> {
    load_tokenizer(tokenizer_files_hf_hub(&model_repo)?, max_length)
}

/// The tokenizer files of a Hugging Face Hub model repository.
#[cfg(feature = "hf-hub")]
pub(crate) fn tokenizer_files_hf_hub(model_repo: &ApiRepo) -> Result<TokenizerFiles> {
    let read = |file: &str| -> Result<Vec<u8>> {
        Ok(std::fs::read(model_repo.get(file).map_err(hf_hub_error)?)?)
    };
    Ok(TokenizerFiles {
        tokenizer_file: read("tokenizer.json")?,
        config_file: read("config.json")?,
        special_tokens_map_file: read("special_tokens_map.json")?,
        tokenizer_config_file: read("tokenizer_config.json")?,
    })
}

/// Pulls a model repo from HuggingFace.
/// HF_HOME decides the location of the cache folder
/// HF_ENDPOINT modifies the URL for the HuggingFace location.
/// HF_TOKEN authenticates the requests. Without it the token written by
/// `huggingface-cli login` is used when present.
#[cfg(feature = "hf-hub")]
pub fn pull_from_hf(
    model_name: String,
    default_cache_dir: PathBuf,
    show_download_progress: bool,
) -> Result<ApiRepo> {
    use std::env;

    let cache_dir = env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or(default_cache_dir);

    let endpoint = env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());

    let token = env::var("HF_TOKEN")
        .ok()
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
        .or_else(default_hf_token);

    let api = ApiBuilder::new()
        .with_cache_dir(cache_dir)
        .with_endpoint(endpoint)
        .with_token(token)
        .with_progress(show_download_progress)
        .build()
        .map_err(|e| Error::Other(format!("Failed to initialize HuggingFace API: {e}")))?;

    let repo = api.model(model_name);
    Ok(repo)
}

/// Token written by `huggingface-cli login`. `Cache::default` panics without a home dir.
#[cfg(feature = "hf-hub")]
fn default_hf_token() -> Option<String> {
    let has_home = std::env::var_os("HOME").is_some() || std::env::var_os("USERPROFILE").is_some();
    has_home.then(|| hf_hub::Cache::default().token()).flatten()
}

/// Safetensors files of a repo, resolving `model.safetensors.index.json` for sharded checkpoints.
#[cfg(any(feature = "qwen3", feature = "nomic-v2-moe"))]
pub(crate) fn safetensors_weight_files(repo: &ApiRepo) -> Result<Vec<PathBuf>> {
    const SINGLE_FILE: &str = "model.safetensors";
    const INDEX_FILE: &str = "model.safetensors.index.json";

    if let Ok(path) = repo.get(SINGLE_FILE) {
        return Ok(vec![path]);
    }

    let index_path = repo.get(INDEX_FILE).map_err(|e| Error::ModelRetrieval {
        file: format!("{SINGLE_FILE} or {INDEX_FILE}"),
        source: Box::new(e),
    })?;
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(index_path)?)
        .map_err(|e| Error::Other(format!("Failed to parse {INDEX_FILE}: {e}")))?;

    let mut shards: Vec<String> = index["weight_map"]
        .as_object()
        .ok_or_else(|| Error::Other(format!("{INDEX_FILE} has no `weight_map` object")))?
        .values()
        .filter_map(|file| file.as_str().map(str::to_string))
        .collect();
    shards.sort_unstable();
    shards.dedup();

    shards
        .iter()
        .map(|file| {
            repo.get(file).map_err(|e| Error::ModelRetrieval {
                file: file.clone(),
                source: Box::new(e),
            })
        })
        .collect()
}

/// ONNX session inputs of `batch`.
///
/// `token_type_ids` is moved out and attached only when the graph declares it, while
/// `input_ids` and `attention_mask` stay available for post-processing.
pub(crate) fn session_inputs(
    batch: &mut EncodedBatch,
    need_token_type_ids: bool,
) -> Result<Vec<(Cow<'static, str>, SessionInputValue<'static>)>> {
    let mut inputs = ort::inputs![
        "input_ids" => Value::from_array(batch.input_ids.clone()).map_err(ort_error)?,
        "attention_mask" => Value::from_array(batch.attention_mask.clone()).map_err(ort_error)?,
    ];
    if need_token_type_ids {
        let token_type_ids = std::mem::take(&mut batch.token_type_ids);
        inputs.push((
            "token_type_ids".into(),
            Value::from_array(token_type_ids).map_err(ort_error)?.into(),
        ));
    }
    Ok(inputs)
}

pub(crate) fn init_session_builder(
    environment: Option<Environment>,
    execution_providers: Vec<ExecutionProviderDispatch>,
    intra_threads: Option<usize>,
    session_config: Vec<(String, String)>,
) -> Result<SessionBuilder> {
    let threads = match intra_threads {
        Some(n) => n,
        None => std::thread::available_parallelism()?.get(),
    };

    #[cfg(feature = "directml")]
    let has_directml = execution_providers
        .iter()
        .any(|ep| ep.name() == <ort::ep::DirectML as ort::ep::ExecutionProvider>::NAME);
    #[cfg(not(feature = "directml"))]
    let has_directml = false;

    let builder_error = |err: ort::Error<SessionBuilder>| Error::OrtBuilder(err.to_string());

    let environment = match environment {
        Some(environment) => environment,
        None => ort::init().build().map_err(|e| {
            if e.message().contains("only one environment") {
                Error::EnvironmentRequired
            } else {
                ort_error(e)
            }
        })?,
    };

    let mut builder = ort::session::Session::builder(&environment)
        .map_err(ort_error)?
        .with_execution_providers(execution_providers)
        .map_err(builder_error)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(builder_error)?
        .with_intra_threads(threads)
        .map_err(builder_error)?;

    for (key, value) in session_config {
        builder = builder
            .with_config_entry(&key, &value)
            .map_err(builder_error)?;
    }

    if has_directml {
        builder = builder
            .with_memory_pattern(false)
            .map_err(builder_error)?
            .with_parallel_execution(false)
            .map_err(builder_error)?;
    }

    Ok(builder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_session_builder_applies_config_entry() {
        let builder = init_session_builder(
            None,
            vec![],
            Some(1),
            vec![("session.disable_prepacking".into(), "1".into())],
        );
        assert!(builder.is_ok());
    }
}
