use crate::{
    init::{HasMaxLength, InitOptionsWithLength},
    models::sparse::SparseModel,
    ort_session::ModelSession,
    TokenizerFiles,
};

use super::DEFAULT_MAX_LENGTH;

impl HasMaxLength for SparseModel {
    const MAX_LENGTH: usize = DEFAULT_MAX_LENGTH;
}

/// Options for initializing the SparseTextEmbedding model
pub type SparseInitOptions = InitOptionsWithLength<SparseModel>;

/// Struct for "bring your own" sparse embedding models
///
/// The `onnx_file` and `tokenizer_files` fields contain the file contents as bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct UserDefinedSparseModel {
    pub onnx_file: Vec<u8>,
    pub tokenizer_files: TokenizerFiles,
    /// Model family of the checkpoint, selects the post-processing of the ONNX outputs.
    pub model: SparseModel,
    /// `idf.json` bytes for inference-free models, enables [`SparseTextEmbedding::query_embed`].
    pub idf_file: Option<Vec<u8>>,
}

impl UserDefinedSparseModel {
    pub fn new(onnx_file: Vec<u8>, tokenizer_files: TokenizerFiles, model: SparseModel) -> Self {
        Self {
            onnx_file,
            tokenizer_files,
            model,
            idf_file: None,
        }
    }

    pub fn with_idf_file(mut self, idf_file: Vec<u8>) -> Self {
        self.idf_file = Some(idf_file);
        self
    }
}

/// Rust representation of the SparseTextEmbedding model.
///
/// It derefs to the pipeline it runs, whose tokenizer and session it exposes.
pub struct SparseTextEmbedding {
    pub(crate) pipeline: fastembed_pipeline::SparseTextEmbedding<ModelSession>,
}

impl std::ops::Deref for SparseTextEmbedding {
    type Target = fastembed_pipeline::SparseTextEmbedding<ModelSession>;

    fn deref(&self) -> &Self::Target {
        &self.pipeline
    }
}

impl std::ops::DerefMut for SparseTextEmbedding {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.pipeline
    }
}
