//! Sparse text embeddings: [`SparseTextEmbedding`] and the files it is built from.

mod bgem3_weights;

use crate::{
    error::{Error, Result},
    graph_session::{needs_token_type_ids, run_checked, GraphOutput, GraphSession},
    tokenizer::{encode_batch, load_tokenizer, TokenizerFiles},
};
use ndarray::{Array2, ArrayView2, ArrayViewD, Ix2};
use std::collections::{HashMap, HashSet};
use tokenizers::Tokenizer;

const DEFAULT_BATCH_SIZE: usize = 256;
const DEFAULT_MAX_LENGTH: usize = 512;

/// A sparse embedding: the vocabulary entries a text activates and their weights.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseEmbedding {
    pub indices: Vec<usize>,
    pub values: Vec<f32>,
}

/// What a sparse model's graph returns, which decides how its output becomes a
/// [`SparseEmbedding`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseOutput {
    /// SPLADE++ token logits, `[batch, sequence, vocab]`, which the pipeline max-pools over the
    /// unmasked positions as `log(1 + relu(logits))`.
    Splade,
    /// SPLADE++ weights the graph has already activated and pooled, `[batch, vocab]`: the graph
    /// computes `log(1 + relu(logits))`, zeroes the masked positions and takes the maximum over
    /// the sequence, so the pipeline only selects each row's nonzero entries.
    PooledSplade,
    /// BGE-M3 hidden states, `[batch, sequence, hidden]`, which the pipeline projects onto token
    /// weights.
    Bgem3,
    /// Token logits, `[batch, sequence, vocab]`, of an inference-free SPLADE document encoder,
    /// which the pipeline max-pools with a double log activation.
    InferenceFreeSplade,
}

/// Struct for "bring your own" sparse embedding models whose graph runs through a
/// [`GraphSession`].
///
/// It holds everything about the model except its ONNX graph: the tokenizer files, what the
/// graph returns, the optional IDF table and the tokenizer max length.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct UserDefinedGraphSparseModel {
    pub tokenizer_files: TokenizerFiles,
    /// What the graph returns, which selects the post-processing of its output.
    pub output: SparseOutput,
    /// `idf.json` bytes for inference-free models, enables [`SparseTextEmbedding::query_embed`].
    pub idf_file: Option<Vec<u8>>,
    /// Upper bound on tokens per input, further capped by the tokenizer's `model_max_length`.
    pub max_length: usize,
}

impl UserDefinedGraphSparseModel {
    pub fn new(tokenizer_files: TokenizerFiles, output: SparseOutput) -> Self {
        Self {
            tokenizer_files,
            output,
            idf_file: None,
            max_length: DEFAULT_MAX_LENGTH,
        }
    }

    pub fn with_idf_file(mut self, idf_file: Vec<u8>) -> Self {
        self.idf_file = Some(idf_file);
        self
    }

    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }
}

/// A sparse text embedding model whose graph runs through the [`GraphSession`] `S`.
pub struct SparseTextEmbedding<S = Box<dyn GraphSession>> {
    pub tokenizer: Tokenizer,
    session: S,
    need_token_type_ids: bool,
    output: SparseOutput,
    /// Ids of the tokenizer's special tokens, excluded from every embedding produced by
    /// the inference-free models.
    special_token_ids: HashSet<usize>,
    /// Token id to IDF weight, read from the `idf.json` shipped with inference-free models.
    /// [`None`] for models which do not have a separate query representation.
    token_id_to_idf: Option<HashMap<usize, f32>>,
}

impl<S: GraphSession> SparseTextEmbedding<S> {
    /// Create a SparseTextEmbedding instance whose graph runs through `session`.
    ///
    /// Tokenization, batching and post-processing run here; only the graph call goes to
    /// `session`.
    ///
    /// # Errors
    ///
    /// Returns an error when the tokenizer files or the IDF table cannot be loaded.
    pub fn try_new_from_graph_session(
        session: S,
        model: UserDefinedGraphSparseModel,
    ) -> Result<Self> {
        let UserDefinedGraphSparseModel {
            tokenizer_files,
            output,
            idf_file,
            max_length,
        } = model;
        let tokenizer = load_tokenizer(tokenizer_files, max_length)?;
        // Models declaring an `idf.json` embed queries from a lookup table instead of
        // running inference, so the table is loaded up front alongside the tokenizer.
        let token_id_to_idf = idf_file
            .map(|bytes| load_idf(&bytes, &tokenizer))
            .transpose()?;
        let special_token_ids = tokenizer
            .get_added_tokens_decoder()
            .iter()
            .filter(|(_, token)| token.special)
            .map(|(id, _)| *id as usize)
            .collect();
        Ok(Self {
            tokenizer,
            need_token_type_ids: needs_token_type_ids(&session),
            session,
            output,
            special_token_ids,
            token_id_to_idf,
        })
    }

    /// The session the graph runs through.
    pub fn session(&self) -> &S {
        &self.session
    }

    /// Method to generate sentence embeddings for a collection of texts.
    ///
    /// Accepts anything that can be referenced as a slice of elements implementing
    /// [`AsRef<str>`], such as `Vec<String>`, `Vec<&str>`, `&[String]`, or `&[&str]`.
    pub fn embed<T: AsRef<str> + Send + Sync>(
        &mut self,
        texts: impl AsRef<[T]>,
        batch_size: Option<usize>,
    ) -> Result<Vec<SparseEmbedding>> {
        let texts = texts.as_ref();
        // Determine the batch size, default if not specified
        let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE);
        if batch_size == 0 {
            return Err(Error::InvalidArgument(
                "batch_size must be greater than 0".into(),
            ));
        }

        let mut output = Vec::with_capacity(texts.len());
        for batch in texts.chunks(batch_size) {
            let inputs = batch.iter().map(|text| text.as_ref()).collect();
            let mut encoded = encode_batch(&self.tokenizer, inputs)?;
            let input_ids = encoded.input_ids.clone();
            let graph_inputs = encoded.graph_inputs(self.need_token_type_ids);
            let outputs = run_checked(&mut self.session, graph_inputs)?;

            let embeddings = match self.output {
                SparseOutput::Splade => {
                    let logits = token_output(select_output(&outputs, "last_hidden_state")?)?;
                    post_process_splade(&logits, &encoded.attention_mask)
                }
                SparseOutput::PooledSplade => {
                    let output = select_output(&outputs, "sparse_embedding")?;
                    let pooled = pooled_output(output, encoded.attention_mask.nrows())?;
                    post_process_pooled_splade(&pooled)
                }
                SparseOutput::Bgem3 => {
                    let output = outputs.first().ok_or_else(|| Error::OutputKeyMissing {
                        key: "<first output>".into(),
                    })?;
                    let hidden_states = token_output(output)?;
                    post_process_bgem3(&hidden_states, &input_ids, &encoded.attention_mask)
                }
                SparseOutput::InferenceFreeSplade => {
                    let logits = token_output(select_output(&outputs, "logits")?)?;
                    post_process_if_splade(
                        &logits,
                        &encoded.attention_mask,
                        &self.special_token_ids,
                    )
                }
            };
            output.extend(embeddings);
        }

        Ok(output)
    }

    /// Method to generate sparse query embeddings without running any model inference.
    ///
    /// Only available for the inference-free (asymmetric) models, which ship an IDF table.
    ///
    /// Accepts anything that can be referenced as a slice of elements implementing
    /// [`AsRef<str>`], such as `Vec<String>`, `Vec<&str>`, `&[String]`, or `&[&str]`.
    pub fn query_embed<T: AsRef<str> + Send + Sync>(
        &self,
        texts: impl AsRef<[T]>,
    ) -> Result<Vec<SparseEmbedding>> {
        let token_id_to_idf = self.token_id_to_idf.as_ref().ok_or_else(|| {
            Error::InvalidArgument(
                "the model has no IDF table and no separate query representation, use `embed` \
                 instead"
                    .into(),
            )
        })?;

        texts
            .as_ref()
            .iter()
            .map(|text| {
                let encoding = self
                    .tokenizer
                    .encode(text.as_ref(), true)
                    .map_err(|e| Error::Tokenization(format!("Failed to encode the query: {e}")))?;

                // Every unique token contributes its IDF weight exactly once, ordered by token id
                let mut token_ids: Vec<usize> = encoding
                    .get_ids()
                    .iter()
                    .map(|&id| id as usize)
                    .filter(|id| !self.special_token_ids.contains(id))
                    .collect();
                token_ids.sort_unstable();
                token_ids.dedup();

                let mut indices = Vec::with_capacity(token_ids.len());
                let mut values = Vec::with_capacity(token_ids.len());
                for token_id in token_ids {
                    if let Some(&idf) = token_id_to_idf.get(&token_id) {
                        indices.push(token_id);
                        values.push(idf);
                    }
                }

                Ok(SparseEmbedding { values, indices })
            })
            .collect()
    }
}

/// Parse `idf.json`, resolving token strings to ids. Unknown tokens are dropped.
fn load_idf(idf_json: &[u8], tokenizer: &Tokenizer) -> Result<HashMap<usize, f32>> {
    let token_to_idf: HashMap<String, f32> = serde_json::from_slice(idf_json)
        .map_err(|e| Error::Other(format!("Failed to parse the idf.json of the model: {e}")))?;

    let vocab = tokenizer.get_vocab(true);
    Ok(token_to_idf
        .into_iter()
        .filter_map(|(token, idf)| vocab.get(&token).map(|&id| (id as usize, idf)))
        .collect())
}

/// The sole output, or the one named `preferred_key`.
fn select_output<'a>(outputs: &'a [GraphOutput], preferred_key: &str) -> Result<&'a GraphOutput> {
    let found = match outputs {
        [only] => Some(only),
        _ => outputs.iter().find(|output| output.name == preferred_key),
    };
    found.ok_or_else(|| Error::OutputKeyMissing {
        key: preferred_key.to_string(),
    })
}

/// `output` as a rank-3 `[batch, sequence, features]` view of per-token values.
fn token_output(output: &GraphOutput) -> Result<ArrayViewD<'_, f32>> {
    let shape = output.tensor.shape();
    if shape.len() != 3 {
        return Err(Error::InvalidShape(format!(
            "Output '{}' must be rank-3 [batch, sequence, vocab], got shape {shape:?}",
            output.name
        )));
    }
    Ok(output.tensor.view())
}

/// `output` as a `[batch, vocab]` view of weights the graph has already pooled, one row per input.
fn pooled_output(output: &GraphOutput, batch_size: usize) -> Result<ArrayView2<'_, f32>> {
    let shape = output.tensor.shape();
    output
        .tensor
        .view()
        .into_dimensionality::<Ix2>()
        .ok()
        .filter(|pooled| pooled.nrows() == batch_size)
        .ok_or_else(|| {
            Error::InvalidShape(format!(
                "Output '{}' of a pooled model must be rank-2 [{batch_size}, vocab], got shape \
                 {shape:?}",
                output.name
            ))
        })
}

/// SPLADE++: `log(1 + relu(logits))` max-pooled over the unmasked positions.
fn post_process_splade(
    model_output: &ArrayViewD<f32>,
    attention_mask: &Array2<i64>,
) -> Vec<SparseEmbedding> {
    let batch_size = attention_mask.shape()[0];
    let seq_len = attention_mask.shape()[1];
    let vocab_size = model_output.shape()[2];

    (0..batch_size)
        .map(|batch_idx| {
            let mut pooled = vec![0.0f32; vocab_size];
            for seq_idx in 0..seq_len {
                if attention_mask[[batch_idx, seq_idx]] == 0 {
                    continue;
                }
                let token_logits = model_output.slice(ndarray::s![batch_idx, seq_idx, ..]);
                for (score, &logit) in pooled.iter_mut().zip(token_logits.iter()) {
                    *score = score.max(logit);
                }
            }

            let mut values = Vec::new();
            let mut indices = Vec::new();
            for (idx, &score) in pooled.iter().enumerate() {
                if score > 0.0 {
                    values.push((1.0 + score).ln());
                    indices.push(idx);
                }
            }
            SparseEmbedding { values, indices }
        })
        .collect()
}

/// SPLADE++ weights the graph has already activated and pooled: each row's nonzero entries, as
/// they are.
fn post_process_pooled_splade(pooled: &ArrayView2<f32>) -> Vec<SparseEmbedding> {
    pooled
        .rows()
        .into_iter()
        .map(|row| {
            let (indices, values) = row
                .iter()
                .enumerate()
                .filter(|(_, &weight)| weight != 0.0)
                .unzip();
            SparseEmbedding { indices, values }
        })
        .collect()
}

fn post_process_bgem3(
    hidden_states: &ArrayViewD<f32>,
    input_ids: &Array2<i64>,
    attention_mask: &Array2<i64>,
) -> Vec<SparseEmbedding> {
    use ndarray::ArrayView1;

    // Special tokens to skip (XLM-RoBERTa: CLS=0, PAD=1, EOS=2, UNK=3)
    const SPECIAL_TOKENS: [i64; 4] = [0, 1, 2, 3];

    let sparse_weights = bgem3_weights::get_weights();
    let weights = ArrayView1::from(&sparse_weights.weight[..]);
    let bias = sparse_weights.bias;
    let batch_size = input_ids.shape()[0];
    let seq_len = input_ids.shape()[1];

    (0..batch_size)
        .map(|batch_idx| {
            let mut token_weights: HashMap<usize, f32> = HashMap::new();

            for seq_idx in 0..seq_len {
                if attention_mask[[batch_idx, seq_idx]] == 0 {
                    continue;
                }

                let token_id = input_ids[[batch_idx, seq_idx]];
                if SPECIAL_TOKENS.contains(&token_id) {
                    continue;
                }

                let hidden = hidden_states.slice(ndarray::s![batch_idx, seq_idx, ..]);
                let weight = (hidden.dot(&weights) + bias).max(0.0);

                if weight > 0.0 {
                    token_weights
                        .entry(token_id as usize)
                        .and_modify(|w| *w = w.max(weight))
                        .or_insert(weight);
                }
            }

            let mut indices: Vec<_> = token_weights.keys().copied().collect();
            indices.sort_unstable();
            let values: Vec<_> = indices.iter().map(|i| token_weights[i]).collect();

            SparseEmbedding { values, indices }
        })
        .collect()
}

/// Post-processing for the inference-free SPLADE document encoder.
///
/// The token logits are max-pooled over the unmasked positions and squashed with a double
/// log activation, `log(1 + log(1 + relu(x)))`, which the v3 models of the
/// opensearch-neural-sparse family use to make document embeddings sparser than the single
/// `log(1 + relu(x))` of SPLADE++.
fn post_process_if_splade(
    logits: &ArrayViewD<f32>,
    attention_mask: &Array2<i64>,
    special_token_ids: &HashSet<usize>,
) -> Vec<SparseEmbedding> {
    let batch_size = attention_mask.shape()[0];
    let seq_len = attention_mask.shape()[1];
    let vocab_size = logits.shape()[2];

    (0..batch_size)
        .map(|batch_idx| {
            // Starting the accumulator at `0.0` encodes both the ReLU floor and the zero
            // contribution of the padded positions, which are skipped altogether.
            let mut pooled = vec![0.0f32; vocab_size];

            for seq_idx in 0..seq_len {
                if attention_mask[[batch_idx, seq_idx]] == 0 {
                    continue;
                }

                let token_logits = logits.slice(ndarray::s![batch_idx, seq_idx, ..]);
                for (score, &logit) in pooled.iter_mut().zip(token_logits.iter()) {
                    *score = score.max(logit);
                }
            }

            let mut values: Vec<f32> = Vec::new();
            let mut indices: Vec<usize> = Vec::new();

            for (token_id, &score) in pooled.iter().enumerate() {
                // Special tokens are dropped from the document side as well, otherwise they
                // would match every query
                if score <= 0.0 || special_token_ids.contains(&token_id) {
                    continue;
                }
                values.push((1.0 + (1.0 + score).ln()).ln());
                indices.push(token_id);
            }

            SparseEmbedding { values, indices }
        })
        .collect()
}

#[cfg(test)]
mod tests;
