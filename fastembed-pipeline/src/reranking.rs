//! Cross-encoder reranking: [`TextRerank`] and the files it is built from.

use crate::{
    error::{Error, Result},
    graph_session::{needs_token_type_ids, run_checked, GraphSession},
    tokenizer::{encode_batch, fixed_padding, load_tokenizer, TokenizerFiles},
};
use ndarray::Ix2;
use tokenizers::Tokenizer;

const DEFAULT_BATCH_SIZE: usize = 256;
const DEFAULT_MAX_LENGTH: usize = 512;

/// Struct for "bring your own" reranking models whose graph runs through a [`GraphSession`].
///
/// It holds everything about the model except its ONNX graph: the tokenizer files and the
/// tokenizer max length.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct UserDefinedGraphRerankingModel {
    pub tokenizer_files: TokenizerFiles,
    /// Upper bound on tokens per query/document pair, further capped by the tokenizer's
    /// `model_max_length`.
    pub max_length: usize,
}

impl UserDefinedGraphRerankingModel {
    pub fn new(tokenizer_files: TokenizerFiles) -> Self {
        Self {
            tokenizer_files,
            max_length: DEFAULT_MAX_LENGTH,
        }
    }

    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }
}

/// Rerank result.
#[derive(Debug, PartialEq, Clone)]
pub struct RerankResult {
    pub document: Option<String>,
    pub score: f32,
    pub index: usize,
}

/// A cross-encoder reranker whose graph runs through the [`GraphSession`] `S`.
#[derive(Debug)]
pub struct TextRerank<S = Box<dyn GraphSession>> {
    pub tokenizer: Tokenizer,
    session: S,
    need_token_type_ids: bool,
}

impl<S: GraphSession> TextRerank<S> {
    /// Create a TextRerank instance whose graph runs through `session`.
    ///
    /// Tokenization, batching and scoring run here; only the graph call goes to `session`, which
    /// must return a `[batch, n]` `logits` output whose first column holds the scores.
    ///
    /// # Errors
    ///
    /// Returns an error when the tokenizer files cannot be loaded.
    pub fn try_new_from_graph_session(
        session: S,
        model: UserDefinedGraphRerankingModel,
    ) -> Result<Self> {
        let UserDefinedGraphRerankingModel {
            tokenizer_files,
            max_length,
        } = model;
        let tokenizer = load_tokenizer(tokenizer_files, max_length)?;
        Ok(Self {
            tokenizer,
            need_token_type_ids: needs_token_type_ids(&session),
            session,
        })
    }

    /// The session the graph runs through.
    pub fn session(&self) -> &S {
        &self.session
    }

    /// Fix padding and complete query/document-pair truncation, preserving tokenizer semantics.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid length or missing tokenizer configuration.
    pub fn set_fixed_padding(&mut self, length: usize) -> Result<()> {
        fixed_padding(&mut self.tokenizer, length)
    }

    /// Rerank documents using the reranker model and returns the results sorted by score in descending order.
    ///
    /// Accepts a query and a collection of documents implementing [`AsRef<str>`].
    pub fn rerank<T: AsRef<str> + Send + Sync>(
        &mut self,
        query: T,
        documents: impl AsRef<[T]>,
        return_documents: bool,
        batch_size: Option<usize>,
    ) -> Result<Vec<RerankResult>> {
        let documents = documents.as_ref();
        let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE);
        if batch_size == 0 {
            return Err(Error::InvalidArgument(
                "batch_size must be greater than 0".into(),
            ));
        }
        let q = query.as_ref();

        let mut scores: Vec<f32> = Vec::with_capacity(documents.len());
        for batch in documents.chunks(batch_size) {
            let inputs = batch.iter().map(|d| (q, d.as_ref())).collect();
            let mut encoded = encode_batch(&self.tokenizer, inputs)?;
            let graph_inputs = encoded.graph_inputs(self.need_token_type_ids);

            let outputs = run_checked(&mut self.session, graph_inputs)?;
            let logits = outputs
                .iter()
                .find(|output| output.name == "logits")
                .ok_or_else(|| Error::OutputKeyMissing {
                    key: "logits".into(),
                })?
                .tensor
                .view();
            let shape = logits.shape().to_vec();
            // One score per pair keeps scores aligned with `documents`.
            let logits = logits
                .into_dimensionality::<Ix2>()
                .ok()
                .filter(|logits| logits.nrows() == batch.len() && logits.ncols() > 0)
                .ok_or_else(|| {
                    Error::InvalidShape(format!(
                        "logits have shape {shape:?}, expected [{}, n]",
                        batch.len()
                    ))
                })?;
            scores.extend(logits.column(0).iter().copied());
        }

        // Return top_n_result of type Vec<RerankResult> ordered by score in descending order, don't use binary heap
        let mut top_n_result: Vec<RerankResult> = scores
            .into_iter()
            .enumerate()
            .map(|(index, score)| RerankResult {
                document: return_documents.then(|| documents[index].as_ref().to_string()),
                score,
                index,
            })
            .collect();
        top_n_result.sort_by(|a, b| a.score.total_cmp(&b.score).reverse());
        Ok(top_n_result)
    }
}
