//! The text pipeline of [FastEmbed](https://github.com/Anush008/fastembed-rs), without ONNX
//! Runtime.
//!
//! Everything a fastembed text model does around its ONNX graph lives here: tokenizers loaded from
//! file bytes, padding (fixed padding included), batching, output selection, pooling,
//! normalization, sparse post-processing and rerank scores. The graph itself runs through a
//! [`GraphSession`], so the same pipeline serves a graph that runs in process, which `fastembed`
//! builds with ONNX Runtime, and one that runs elsewhere.
//!
//! - [`TextEmbedding`] - dense text embeddings
//! - [`SparseTextEmbedding`] - sparse (SPLADE, BGE-M3) embeddings for lexical search
//! - [`TextRerank`] - cross-encoder reranking of candidates
//!
//! Each is built with `try_new_from_graph_session` from a session and the model's tokenizer
//! files. `fastembed` re-exports this crate and adds the in-process sessions, the model catalog
//! and the Hugging Face downloads.

mod common;
mod error;
mod graph_session;
pub mod output;
mod pooling;
mod quantization;
mod reranking;
mod sparse_text_embedding;
#[cfg(test)]
mod test_support;
mod text_embedding;
mod tokenizer;

pub use crate::common::{normalize, Embedding};
pub use crate::error::{BoxError, Error, Result};
pub use crate::graph_session::{GraphInput, GraphOutput, GraphSession};
pub use crate::output::{EmbeddingOutput, OutputKey, OutputPrecedence, SingleBatchOutput};
pub use crate::pooling::Pooling;
pub use crate::quantization::QuantizationMode;
pub use crate::reranking::{RerankResult, TextRerank, UserDefinedGraphRerankingModel};
pub use crate::sparse_text_embedding::{
    SparseEmbedding, SparseOutput, SparseTextEmbedding, UserDefinedGraphSparseModel,
};
pub use crate::text_embedding::{TextEmbedding, UserDefinedGraphEmbeddingModel};
pub use crate::tokenizer::{encode_batch, load_tokenizer, EncodedBatch, TokenizerFiles};
