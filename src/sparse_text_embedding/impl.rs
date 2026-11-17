#[cfg(feature = "hf-hub")]
use crate::common::Error;
#[cfg(feature = "hf-hub")]
use crate::common::{init_session_builder, tokenizer_files_hf_hub};
#[cfg(feature = "hf-hub")]
use crate::models::sparse::IDF_FILE;
use crate::{
    common::{ort_error, Result},
    models::sparse::{models_list, SparseModel},
    ort_session::ModelSession,
    text_embedding::InitOptionsUserDefined,
    ModelInfo, SparseEmbedding, SparseOutput, UserDefinedGraphSparseModel,
};
#[cfg(feature = "hf-hub")]
use hf_hub::api::sync::ApiRepo;
#[cfg(feature = "hf-hub")]
use std::path::PathBuf;

#[cfg(feature = "hf-hub")]
use super::SparseInitOptions;
use super::{SparseTextEmbedding, UserDefinedSparseModel};

impl From<SparseModel> for SparseOutput {
    /// What the catalog model's graph returns: token logits for the SPLADE models, hidden states
    /// for BGE-M3.
    fn from(model: SparseModel) -> Self {
        match model {
            SparseModel::SPLADEPPV1 => SparseOutput::Splade,
            SparseModel::BGEM3 => SparseOutput::Bgem3,
            SparseModel::OpenSearchNeuralSparseDocV3Gte => SparseOutput::InferenceFreeSplade,
        }
    }
}

impl SparseTextEmbedding {
    /// Try to generate a new SparseTextEmbedding Instance
    ///
    /// Uses the highest level of Graph optimization
    ///
    /// Uses the total number of CPUs available as the number of intra-threads
    #[cfg(feature = "hf-hub")]
    pub fn try_new(options: SparseInitOptions) -> Result<Self> {
        let SparseInitOptions {
            max_length,
            model_name,
            cache_dir,
            show_download_progress,
            environment,
            execution_providers,
            intra_threads,
            session_config,
        } = options;

        let model_repo = SparseTextEmbedding::retrieve_model(
            model_name.clone(),
            cache_dir.clone(),
            show_download_progress,
        )?;

        let model_info = SparseTextEmbedding::get_model_info(&model_name);
        let model_file_name = &model_info.model_file;
        let model_file_reference =
            model_repo
                .get(model_file_name)
                .map_err(|e| Error::ModelRetrieval {
                    file: model_file_name.clone(),
                    source: Box::new(e),
                })?;

        // Download additional files if needed (e.g., model.onnx.data for large models)
        let mut idf_file_reference: Option<PathBuf> = None;
        for file in &model_info.additional_files {
            let reference = model_repo.get(file).map_err(|e| Error::ModelRetrieval {
                file: file.clone(),
                source: Box::new(e),
            })?;
            if file == IDF_FILE {
                idf_file_reference = Some(reference);
            }
        }

        let session = init_session_builder(
            environment,
            execution_providers,
            intra_threads,
            session_config,
        )?
        .commit_from_file(model_file_reference)
        .map_err(ort_error)?;

        let mut model = UserDefinedGraphSparseModel::new(
            tokenizer_files_hf_hub(&model_repo)?,
            model_name.into(),
        )
        .with_max_length(max_length);
        // Models declaring an `idf.json` embed queries from a lookup table instead of
        // running inference, so the table is loaded up front alongside the tokenizer.
        model.idf_file = idf_file_reference.map(std::fs::read).transpose()?;
        Self::new(ModelSession::ort(session), model)
    }

    /// Create a SparseTextEmbedding instance from model files provided by the user.
    ///
    /// This can be used for 'bring your own' sparse embedding models
    pub fn try_new_from_user_defined(
        model: UserDefinedSparseModel,
        options: InitOptionsUserDefined,
    ) -> Result<Self> {
        let (mut session_builder, max_length) = options.into_session_builder()?;
        let session = session_builder
            .commit_from_memory(&model.onnx_file)
            .map_err(ort_error)?;

        let UserDefinedSparseModel {
            tokenizer_files,
            model,
            idf_file,
            ..
        } = model;
        let mut model = UserDefinedGraphSparseModel::new(tokenizer_files, model.into())
            .with_max_length(max_length);
        model.idf_file = idf_file;
        Self::new(ModelSession::ort(session), model)
    }

    fn new(session: ModelSession, model: UserDefinedGraphSparseModel) -> Result<Self> {
        Ok(Self {
            pipeline: fastembed_pipeline::SparseTextEmbedding::try_new_from_graph_session(
                session, model,
            )?,
        })
    }

    /// Return the SparseTextEmbedding model's directory from cache or remote retrieval
    #[cfg(feature = "hf-hub")]
    fn retrieve_model(
        model: SparseModel,
        cache_dir: PathBuf,
        show_download_progress: bool,
    ) -> Result<ApiRepo> {
        use crate::common::pull_from_hf;

        pull_from_hf(model.to_string(), cache_dir, show_download_progress)
    }

    /// Retrieve a list of supported models
    pub fn list_supported_models() -> Vec<ModelInfo<SparseModel>> {
        models_list()
    }

    /// Get ModelInfo from SparseModel
    pub fn get_model_info(model: &SparseModel) -> ModelInfo<SparseModel> {
        SparseTextEmbedding::list_supported_models()
            .into_iter()
            .find(|m| &m.model == model)
            .expect("Model not found in supported models list. This is a bug - please report it.")
    }

    /// Method to generate sentence embeddings for a collection of texts.
    ///
    /// Accepts anything that can be referenced as a slice of elements implementing
    /// [`AsRef<str>`], such as `Vec<String>`, `Vec<&str>`, `&[String]`, or `&[&str]`.
    pub fn embed<S: AsRef<str> + Send + Sync>(
        &mut self,
        texts: impl AsRef<[S]>,
        batch_size: Option<usize>,
    ) -> Result<Vec<SparseEmbedding>> {
        self.pipeline.embed(texts, batch_size)
    }

    /// Method to generate sparse query embeddings without running any model inference.
    ///
    /// Only available for the inference-free (asymmetric) models, such as
    /// [`SparseModel::OpenSearchNeuralSparseDocV3Gte`].
    ///
    /// Accepts anything that can be referenced as a slice of elements implementing
    /// [`AsRef<str>`], such as `Vec<String>`, `Vec<&str>`, `&[String]`, or `&[&str]`.
    pub fn query_embed<S: AsRef<str> + Send + Sync>(
        &self,
        texts: impl AsRef<[S]>,
    ) -> Result<Vec<SparseEmbedding>> {
        self.pipeline.query_embed(texts)
    }
}
