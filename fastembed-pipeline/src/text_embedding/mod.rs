//! Dense text embeddings: [`TextEmbedding`] and the files it is built from.

pub mod output;

use crate::{
    error::{Error, Result},
    graph_session::{needs_token_type_ids, run_checked, GraphSession},
    output::{EmbeddingOutput, OutputKey, SingleBatchOutput},
    pooling::Pooling,
    tokenizer::{encode_batch, fixed_padding, load_tokenizer, TokenizerFiles},
    Embedding, QuantizationMode,
};
use tokenizers::Tokenizer;

const DEFAULT_BATCH_SIZE: usize = 256;
const DEFAULT_MAX_LENGTH: usize = 512;

/// Struct for "bring your own" embedding models whose graph runs through a
/// [`GraphSession`].
///
/// It holds everything about the model except its ONNX graph: the tokenizer files, how outputs
/// are selected and pooled, and the tokenizer max length.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct UserDefinedGraphEmbeddingModel {
    pub tokenizer_files: TokenizerFiles,
    pub pooling: Option<Pooling>,
    pub quantization: QuantizationMode,
    pub output_key: Option<OutputKey>,
    /// Upper bound on tokens per input, further capped by the tokenizer's `model_max_length`.
    pub max_length: usize,
}

impl UserDefinedGraphEmbeddingModel {
    pub fn new(tokenizer_files: TokenizerFiles) -> Self {
        Self {
            tokenizer_files,
            pooling: None,
            quantization: QuantizationMode::None,
            output_key: None,
            max_length: DEFAULT_MAX_LENGTH,
        }
    }

    pub fn with_quantization(mut self, quantization: QuantizationMode) -> Self {
        self.quantization = quantization;
        self
    }

    pub fn with_pooling(mut self, pooling: Pooling) -> Self {
        self.pooling = Some(pooling);
        self
    }

    pub fn with_output_key(mut self, output_key: OutputKey) -> Self {
        self.output_key = Some(output_key);
        self
    }

    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }
}

/// A dense text embedding model whose graph runs through the [`GraphSession`] `S`.
pub struct TextEmbedding<S = Box<dyn GraphSession>> {
    pub tokenizer: Tokenizer,
    pooling: Option<Pooling>,
    session: S,
    need_token_type_ids: bool,
    quantization: QuantizationMode,
    output_key: Option<OutputKey>,
}

impl<S: GraphSession> TextEmbedding<S> {
    /// Create a TextEmbedding instance whose graph runs through `session`.
    ///
    /// Tokenization, batching, output selection, pooling and normalization run here; only the
    /// graph call goes to `session`.
    ///
    /// # Errors
    ///
    /// Returns an error when the tokenizer files cannot be loaded.
    pub fn try_new_from_graph_session(
        session: S,
        model: UserDefinedGraphEmbeddingModel,
    ) -> Result<Self> {
        let UserDefinedGraphEmbeddingModel {
            tokenizer_files,
            pooling,
            quantization,
            output_key,
            max_length,
        } = model;
        let tokenizer = load_tokenizer(tokenizer_files, max_length)?;
        Ok(Self {
            tokenizer,
            pooling,
            need_token_type_ids: needs_token_type_ids(&session),
            session,
            quantization,
            output_key,
        })
    }

    /// The session the graph runs through.
    pub fn session(&self) -> &S {
        &self.session
    }

    /// Fix padding and truncation without changing pad IDs, special tokens or padding direction.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid length or missing tokenizer configuration.
    pub fn set_fixed_padding(&mut self, length: usize) -> Result<()> {
        fixed_padding(&mut self.tokenizer, length)
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
    pub fn transform<T: AsRef<str> + Send + Sync>(
        &mut self,
        texts: impl AsRef<[T]>,
        batch_size: Option<usize>,
    ) -> Result<EmbeddingOutput> {
        let texts = texts.as_ref();
        // Determine the batch size according to the quantization method used.
        // Default if not specified
        let batch_size = match self.quantization {
            QuantizationMode::Dynamic => {
                if let Some(batch_size) = batch_size {
                    if batch_size < texts.len() {
                        Err(Error::InvalidArgument(
                            "Dynamic quantization cannot be used with batching. \
                            This is due to the dynamic quantization process adjusting \
                            the data range to fit each batch, making the embeddings \
                            incompatible across batches. Try specifying a batch size \
                            of `None`, or use a model with static or no quantization."
                                .into(),
                        ))
                    } else {
                        Ok(texts.len())
                    }
                } else {
                    Ok(texts.len())
                }
            }
            _ => Ok(batch_size.unwrap_or(DEFAULT_BATCH_SIZE)),
        }?;
        if batch_size == 0 {
            return Err(Error::InvalidArgument(
                "batch_size must be greater than 0".into(),
            ));
        }

        let batches = texts
            .chunks(batch_size)
            .map(|batch| {
                let inputs = batch.iter().map(|text| text.as_ref()).collect();
                let mut encoded = encode_batch(&self.tokenizer, inputs)?;
                let graph_inputs = encoded.graph_inputs(self.need_token_type_ids);
                Ok(SingleBatchOutput {
                    outputs: run_checked(&mut self.session, graph_inputs)?,
                    attention_mask_array: encoded.attention_mask,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(EmbeddingOutput::new(batches))
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
    pub fn embed<T: AsRef<str> + Send + Sync>(
        &mut self,
        texts: impl AsRef<[T]>,
        batch_size: Option<usize>,
    ) -> Result<Vec<Embedding>> {
        let batches = self.transform(texts.as_ref(), batch_size)?;
        if let Some(output_key) = &self.output_key {
            batches.export_with_transformer(output::transformer_with_precedence(
                output_key,
                self.pooling.clone(),
            ))
        } else {
            batches.export_with_transformer(output::transformer_with_precedence(
                output::OUTPUT_TYPE_PRECEDENCE,
                self.pooling.clone(),
            ))
        }
    }
}
