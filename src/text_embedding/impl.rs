//! The definition of the main struct for text embeddings - [`TextEmbedding`].

#[cfg(feature = "hf-hub")]
use crate::common::{init_session_builder, ort_error, tokenizer_files_hf_hub};
use crate::{
    common::{Error, Result},
    models::{text_embedding::models_list, ModelTrait},
    ort_session::{ModelSession, OrtGraphSession},
    Embedding, EmbeddingModel, EmbeddingOutput, GraphSession, ModelInfo, Pooling, QuantizationMode,
    UserDefinedGraphEmbeddingModel,
};
#[cfg(feature = "hf-hub")]
use hf_hub::api::sync::ApiRepo;
#[cfg(feature = "hf-hub")]
use std::path::PathBuf;

#[cfg(feature = "hf-hub")]
use super::TextInitOptions;
use super::{InitOptionsUserDefined, TextEmbedding, UserDefinedEmbeddingModel};

impl TextEmbedding {
    /// Try to generate a new TextEmbedding Instance
    ///
    /// Uses the highest level of Graph optimization
    ///
    /// Uses the total number of CPUs available as the number of intra-threads
    #[cfg(feature = "hf-hub")]
    pub fn try_new(options: TextInitOptions) -> Result<Self> {
        let TextInitOptions {
            max_length,
            model_name,
            environment,
            execution_providers,
            cache_dir,
            show_download_progress,
            intra_threads,
            session_config,
        } = options;

        let model_repo = TextEmbedding::retrieve_model(
            model_name.clone(),
            cache_dir.clone(),
            show_download_progress,
        )?;

        let model_info = TextEmbedding::get_model_info(&model_name)?;
        let model_file_name = &model_info.model_file;
        let model_file_reference =
            model_repo
                .get(model_file_name)
                .map_err(|e| Error::ModelRetrieval {
                    file: model_file_name.clone(),
                    source: Box::new(e),
                })?;

        for file in &model_info.additional_files {
            model_repo.get(file).map_err(|e| Error::ModelRetrieval {
                file: file.clone(),
                source: Box::new(e),
            })?;
        }

        // Prioritize loading the pooling config if available; otherwise, look for it in the hardcoded defaults.
        let post_processing = TextEmbedding::get_default_pooling_method(&model_name);

        let session = init_session_builder(
            environment,
            execution_providers,
            intra_threads,
            session_config,
        )?
        .commit_from_file(model_file_reference)
        .map_err(ort_error)?;

        let mut model = UserDefinedGraphEmbeddingModel::new(tokenizer_files_hf_hub(&model_repo)?)
            .with_quantization(TextEmbedding::get_quantization_mode(&model_name))
            .with_max_length(max_length);
        model.pooling = post_processing;
        model.output_key = model_info.output_key.clone();
        Self::new(ModelSession::ort(session), model)
    }

    /// Create a TextEmbedding instance from model files provided by the user.
    ///
    /// This can be used for 'bring your own' embedding models
    pub fn try_new_from_user_defined(
        model: UserDefinedEmbeddingModel,
        options: InitOptionsUserDefined,
    ) -> Result<Self> {
        let UserDefinedEmbeddingModel {
            onnx_file,
            external_initializers,
            tokenizer_files,
            pooling,
            quantization,
            output_key,
        } = model;
        let max_length = options.max_length;
        let session = OrtGraphSession::from_memory(&onnx_file, external_initializers, options)?;
        let mut model = UserDefinedGraphEmbeddingModel::new(tokenizer_files)
            .with_quantization(quantization)
            .with_max_length(max_length);
        model.pooling = pooling;
        model.output_key = output_key;
        Self::new(ModelSession::Ort(session), model)
    }

    /// Create a TextEmbedding instance whose graph runs through a caller-supplied session.
    ///
    /// Tokenization, batching, output selection, pooling and normalization behave exactly as
    /// for [`TextEmbedding::try_new_from_user_defined`]; only the graph call goes to `session`.
    ///
    /// # Errors
    ///
    /// Returns an error when the tokenizer files cannot be loaded.
    pub fn try_new_from_graph_session(
        session: Box<dyn GraphSession>,
        model: UserDefinedGraphEmbeddingModel,
    ) -> Result<Self> {
        Self::new(ModelSession::Graph(session), model)
    }

    fn new(session: ModelSession, model: UserDefinedGraphEmbeddingModel) -> Result<Self> {
        Ok(Self {
            pipeline: fastembed_pipeline::TextEmbedding::try_new_from_graph_session(
                session, model,
            )?,
        })
    }

    /// Return the TextEmbedding model's directory from cache or remote retrieval
    #[cfg(feature = "hf-hub")]
    fn retrieve_model(
        model: EmbeddingModel,
        cache_dir: PathBuf,
        show_download_progress: bool,
    ) -> Result<ApiRepo> {
        use crate::common::pull_from_hf;

        let model_code = TextEmbedding::get_model_info(&model)?.model_code.clone();
        pull_from_hf(model_code, cache_dir, show_download_progress)
    }

    pub fn get_default_pooling_method(model_name: &EmbeddingModel) -> Option<Pooling> {
        match model_name {
            EmbeddingModel::AllMiniLML6V2 => Some(Pooling::Mean),
            EmbeddingModel::AllMiniLML6V2Q => Some(Pooling::Mean),
            EmbeddingModel::AllMiniLML12V2 => Some(Pooling::Mean),
            EmbeddingModel::AllMiniLML12V2Q => Some(Pooling::Mean),

            EmbeddingModel::BGEBaseENV15 => Some(Pooling::Cls),
            EmbeddingModel::BGEBaseENV15Q => Some(Pooling::Cls),
            EmbeddingModel::BGELargeENV15 => Some(Pooling::Cls),
            EmbeddingModel::BGELargeENV15Q => Some(Pooling::Cls),
            EmbeddingModel::BGESmallENV15 => Some(Pooling::Cls),
            EmbeddingModel::BGESmallENV15Q => Some(Pooling::Cls),
            EmbeddingModel::BGESmallZHV15 => Some(Pooling::Cls),
            EmbeddingModel::BGELargeZHV15 => Some(Pooling::Cls),
            EmbeddingModel::BGEM3 => Some(Pooling::Cls),

            EmbeddingModel::NomicEmbedTextV1 => Some(Pooling::Mean),
            EmbeddingModel::NomicEmbedTextV15 => Some(Pooling::Mean),
            EmbeddingModel::NomicEmbedTextV15Q => Some(Pooling::Mean),

            EmbeddingModel::ParaphraseMLMiniLML12V2 => Some(Pooling::Mean),
            EmbeddingModel::ParaphraseMLMiniLML12V2Q => Some(Pooling::Mean),
            EmbeddingModel::ParaphraseMLMpnetBaseV2 => Some(Pooling::Mean),
            EmbeddingModel::AllMpnetBaseV2 => Some(Pooling::Mean),

            EmbeddingModel::ModernBertEmbedLarge => Some(Pooling::Mean),

            EmbeddingModel::MultilingualE5Base => Some(Pooling::Mean),
            EmbeddingModel::MultilingualE5Small => Some(Pooling::Mean),
            EmbeddingModel::MultilingualE5Large => Some(Pooling::Mean),

            EmbeddingModel::MxbaiEmbedLargeV1 => Some(Pooling::Cls),
            EmbeddingModel::MxbaiEmbedLargeV1Q => Some(Pooling::Cls),

            EmbeddingModel::GTEBaseENV15 => Some(Pooling::Cls),
            EmbeddingModel::GTEBaseENV15Q => Some(Pooling::Cls),
            EmbeddingModel::GTELargeENV15 => Some(Pooling::Cls),
            EmbeddingModel::GTELargeENV15Q => Some(Pooling::Cls),

            EmbeddingModel::ClipVitB32 => Some(Pooling::Mean),

            EmbeddingModel::JinaEmbeddingsV2BaseCode => Some(Pooling::Mean),
            EmbeddingModel::JinaEmbeddingsV2BaseEN => Some(Pooling::Mean),

            EmbeddingModel::EmbeddingGemma300M => Some(Pooling::Mean),
            EmbeddingModel::EmbeddingGemma300MQ4 => Some(Pooling::Mean),
            EmbeddingModel::EmbeddingGemma300MQ => Some(Pooling::Mean),

            EmbeddingModel::SnowflakeArcticEmbedXS => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedXSQ => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedS => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedSQ => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedM => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedMQ => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedMLong => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedMLongQ => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedL => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedLQ => Some(Pooling::Cls),
        }
    }

    /// Get the quantization mode of the model.
    ///
    /// Any models with a `Q` suffix in their name are quantized models.
    ///
    /// Currently only 6 supported models have dynamic quantization:
    /// - Alibaba-NLP/gte-base-en-v1.5
    /// - Alibaba-NLP/gte-large-en-v1.5
    /// - mixedbread-ai/mxbai-embed-large-v1
    /// - nomic-ai/nomic-embed-text-v1.5
    /// - Xenova/all-MiniLM-L12-v2
    /// - Xenova/all-MiniLM-L6-v2
    ///
    // TODO: Update this list when more models are added
    pub fn get_quantization_mode(model_name: &EmbeddingModel) -> QuantizationMode {
        match model_name {
            EmbeddingModel::AllMiniLML6V2Q => QuantizationMode::Dynamic,
            EmbeddingModel::AllMiniLML12V2Q => QuantizationMode::Dynamic,
            EmbeddingModel::BGEBaseENV15Q => QuantizationMode::Static,
            EmbeddingModel::BGELargeENV15Q => QuantizationMode::Static,
            EmbeddingModel::BGESmallENV15Q => QuantizationMode::Static,
            EmbeddingModel::NomicEmbedTextV15Q => QuantizationMode::Dynamic,
            EmbeddingModel::ParaphraseMLMiniLML12V2Q => QuantizationMode::Static,
            EmbeddingModel::MxbaiEmbedLargeV1Q => QuantizationMode::Dynamic,
            EmbeddingModel::GTEBaseENV15Q => QuantizationMode::Dynamic,
            EmbeddingModel::GTELargeENV15Q => QuantizationMode::Dynamic,
            EmbeddingModel::SnowflakeArcticEmbedXSQ => QuantizationMode::Dynamic,
            EmbeddingModel::SnowflakeArcticEmbedSQ => QuantizationMode::Dynamic,
            EmbeddingModel::SnowflakeArcticEmbedMQ => QuantizationMode::Dynamic,
            EmbeddingModel::SnowflakeArcticEmbedMLongQ => QuantizationMode::Dynamic,
            EmbeddingModel::SnowflakeArcticEmbedLQ => QuantizationMode::Dynamic,
            EmbeddingModel::EmbeddingGemma300MQ => QuantizationMode::Dynamic,
            // 4-bit static quantization: batching-safe, so not Dynamic
            EmbeddingModel::EmbeddingGemma300MQ4 => QuantizationMode::None,
            EmbeddingModel::AllMiniLML6V2
            | EmbeddingModel::AllMiniLML12V2
            | EmbeddingModel::AllMpnetBaseV2
            | EmbeddingModel::BGEBaseENV15
            | EmbeddingModel::BGELargeENV15
            | EmbeddingModel::BGESmallENV15
            | EmbeddingModel::BGESmallZHV15
            | EmbeddingModel::BGELargeZHV15
            | EmbeddingModel::BGEM3
            | EmbeddingModel::NomicEmbedTextV1
            | EmbeddingModel::NomicEmbedTextV15
            | EmbeddingModel::ParaphraseMLMiniLML12V2
            | EmbeddingModel::ParaphraseMLMpnetBaseV2
            | EmbeddingModel::ModernBertEmbedLarge
            | EmbeddingModel::MultilingualE5Small
            | EmbeddingModel::MultilingualE5Base
            | EmbeddingModel::MultilingualE5Large
            | EmbeddingModel::MxbaiEmbedLargeV1
            | EmbeddingModel::GTEBaseENV15
            | EmbeddingModel::GTELargeENV15
            | EmbeddingModel::ClipVitB32
            | EmbeddingModel::JinaEmbeddingsV2BaseCode
            | EmbeddingModel::JinaEmbeddingsV2BaseEN
            | EmbeddingModel::EmbeddingGemma300M
            | EmbeddingModel::SnowflakeArcticEmbedXS
            | EmbeddingModel::SnowflakeArcticEmbedS
            | EmbeddingModel::SnowflakeArcticEmbedM
            | EmbeddingModel::SnowflakeArcticEmbedMLong
            | EmbeddingModel::SnowflakeArcticEmbedL => QuantizationMode::None,
        }
    }

    /// Retrieve a list of supported models
    pub fn list_supported_models() -> Vec<ModelInfo<EmbeddingModel>> {
        models_list()
    }

    /// Get ModelInfo from EmbeddingModel
    pub fn get_model_info(model: &EmbeddingModel) -> Result<&ModelInfo<EmbeddingModel>> {
        EmbeddingModel::get_model_info(model).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "Model {model:?} not found. Please check if the model is supported \
                by the current version."
            ))
        })
    }

    /// Method to generate the raw session outputs wrapped in an [`EmbeddingOutput`]
    /// instance, which can be used to extract the embeddings with default or custom
    /// methods as well as output key precedence.
    ///
    /// Metadata that could be useful for creating the array transformer is
    /// returned alongside the [`EmbeddingOutput`] instance, such as pooling methods
    /// etc.
    ///
    /// # Note
    ///
    /// This is a lower level method than [`TextEmbedding::embed`], and is useful
    /// when you need to extract the session outputs in a custom way.
    ///
    /// If you want to extract the embeddings directly, use [`TextEmbedding::embed`].
    ///
    /// If you want to use the raw session outputs, use [`EmbeddingOutput::into_raw`]
    /// on the output of this method.
    ///
    /// If you want to choose a different export key or customize the way the batch
    /// arrays are aggregated, you can define your own array transformer
    /// and use it on [`EmbeddingOutput::export_with_transformer`] to extract the
    /// embeddings with your custom output type.
    pub fn transform<S: AsRef<str> + Send + Sync>(
        &mut self,
        texts: impl AsRef<[S]>,
        batch_size: Option<usize>,
    ) -> Result<EmbeddingOutput> {
        self.pipeline.transform(texts, batch_size)
    }

    /// Method to generate sentence embeddings for a collection of texts.
    ///
    /// Accepts anything that can be referenced as a slice of elements implementing
    /// [`AsRef<str>`], such as `Vec<String>`, `Vec<&str>`, `&[String]`, or `&[&str]`.
    ///
    /// The output is a [`Vec`] of [`Embedding`]s.
    ///
    /// # Note
    ///
    /// This method is a higher level method than [`TextEmbedding::transform`] by utilizing
    /// the default output precedence and array transformer for the [`TextEmbedding`] model.
    pub fn embed<S: AsRef<str> + Send + Sync>(
        &mut self,
        texts: impl AsRef<[S]>,
        batch_size: Option<usize>,
    ) -> Result<Vec<Embedding>> {
        self.pipeline.embed(texts, batch_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_variants_have_explicit_pooling() {
        for variant in crate::models::text_embedding::all_variants() {
            let _ = TextEmbedding::get_default_pooling_method(&variant);
            let _ = TextEmbedding::get_quantization_mode(&variant);
        }
    }
}
