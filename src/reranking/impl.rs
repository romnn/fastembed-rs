#[cfg(feature = "hf-hub")]
use crate::common::tokenizer_files_hf_hub;
use crate::{
    common::{init_session_builder, ort_error, Error, Result},
    models::reranking::reranker_model_list,
    ort_session::ModelSession,
    GraphSession, RerankResult, RerankerModel, RerankerModelInfo, UserDefinedGraphRerankingModel,
};

#[cfg(feature = "hf-hub")]
use super::RerankInitOptions;
use super::{OnnxSource, RerankInitOptionsUserDefined, TextRerank, UserDefinedRerankingModel};

impl TextRerank {
    fn new(session: ModelSession, model: UserDefinedGraphRerankingModel) -> Result<Self> {
        Ok(Self {
            pipeline: fastembed_pipeline::TextRerank::try_new_from_graph_session(session, model)?,
        })
    }

    pub fn get_model_info(model: &RerankerModel) -> RerankerModelInfo {
        TextRerank::list_supported_models()
            .into_iter()
            .find(|m| &m.model == model)
            .expect("Model not found in supported models list. This is a bug - please report it.")
    }

    pub fn list_supported_models() -> Vec<RerankerModelInfo> {
        reranker_model_list()
    }

    #[cfg(feature = "hf-hub")]
    pub fn try_new(options: RerankInitOptions) -> Result<TextRerank> {
        use super::RerankInitOptions;
        use crate::common::pull_from_hf;

        let RerankInitOptions {
            max_length,
            model_name,
            environment,
            execution_providers,
            cache_dir,
            show_download_progress,
            intra_threads,
            session_config,
        } = options;

        let model_repo = pull_from_hf(model_name.to_string(), cache_dir, show_download_progress)?;

        let model_info = TextRerank::get_model_info(&model_name);
        let model_file_reference =
            model_repo
                .get(&model_info.model_file)
                .map_err(|e| Error::ModelRetrieval {
                    file: model_info.model_file.clone(),
                    source: Box::new(e),
                })?;
        for additional_file in &model_info.additional_files {
            model_repo
                .get(additional_file)
                .map_err(|e| Error::ModelRetrieval {
                    file: additional_file.clone(),
                    source: Box::new(e),
                })?;
        }

        let session = init_session_builder(
            environment,
            execution_providers,
            intra_threads,
            session_config,
        )?
        .commit_from_file(model_file_reference)
        .map_err(ort_error)?;

        let model = UserDefinedGraphRerankingModel::new(tokenizer_files_hf_hub(&model_repo)?)
            .with_max_length(max_length);
        Self::new(ModelSession::ort(session), model)
    }

    /// Create a TextRerank instance from model files provided by the user.
    ///
    /// This can be used for 'bring your own' reranking models
    pub fn try_new_from_user_defined(
        model: UserDefinedRerankingModel,
        options: RerankInitOptionsUserDefined,
    ) -> Result<Self> {
        let RerankInitOptionsUserDefined {
            environment,
            execution_providers,
            max_length,
            intra_threads,
            disable_cpu_fallback,
            dimension_overrides,
            session_config,
        } = options;

        let mut session_builder = init_session_builder(
            environment,
            execution_providers,
            intra_threads,
            session_config,
        )?;
        let builder_error = |err: ort::Error<ort::session::builder::SessionBuilder>| {
            Error::OrtBuilder(err.to_string())
        };
        if disable_cpu_fallback {
            session_builder = session_builder
                .with_disable_cpu_fallback()
                .map_err(builder_error)?;
        }
        for (name, size) in dimension_overrides {
            session_builder = session_builder
                .with_dimension_override(name, size)
                .map_err(builder_error)?;
        }
        let session = match &model.onnx_source {
            OnnxSource::Memory(bytes) => session_builder.commit_from_memory(bytes),
            OnnxSource::File(path) => session_builder.commit_from_file(path),
        }
        .map_err(ort_error)?;

        let model =
            UserDefinedGraphRerankingModel::new(model.tokenizer_files).with_max_length(max_length);
        Self::new(ModelSession::ort(session), model)
    }

    /// Create a TextRerank instance whose graph runs through a caller-supplied session.
    ///
    /// Tokenization, batching and scoring behave exactly as for
    /// [`TextRerank::try_new_from_user_defined`]; only the graph call goes to `session`, which
    /// must return a `[batch, n]` `logits` output whose first column holds the scores.
    ///
    /// # Errors
    ///
    /// Returns an error when the tokenizer files cannot be loaded.
    pub fn try_new_from_graph_session(
        session: Box<dyn GraphSession>,
        model: UserDefinedGraphRerankingModel,
    ) -> Result<Self> {
        Self::new(ModelSession::Graph(session), model)
    }

    /// Rerank documents using the reranker model and returns the results sorted by score in descending order.
    ///
    /// Accepts a query and a collection of documents implementing [`AsRef<str>`].
    pub fn rerank<S: AsRef<str> + Send + Sync>(
        &mut self,
        query: S,
        documents: impl AsRef<[S]>,
        return_documents: bool,
        batch_size: Option<usize>,
    ) -> Result<Vec<RerankResult>> {
        self.pipeline
            .rerank(query, documents, return_documents, batch_size)
    }
}
