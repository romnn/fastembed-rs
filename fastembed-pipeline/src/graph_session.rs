//! Pluggable execution of the ONNX graph behind [`TextEmbedding`], [`TextRerank`] and
//! [`SparseTextEmbedding`].
//!
//! Tokenization, padding, truncation, batching, output selection, pooling and normalization stay
//! in this crate.
//! Only the graph call itself goes through a [`GraphSession`], so a caller can run the graph
//! elsewhere, for example in another process, and still get the same pipeline around it.
//! `fastembed` implements it over an in-process ONNX Runtime session.

use crate::error::{Error, Result};
use ndarray::{Array2, ArrayD};
use std::fmt;

#[cfg(doc)]
use crate::{SparseTextEmbedding, TextEmbedding, TextRerank};

/// One named `[batch, sequence]` token tensor fed to the graph.
///
/// The pipeline sends `input_ids` and `attention_mask`, plus `token_type_ids` when the session
/// lists it among [`GraphSession::input_names`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphInput {
    /// The graph input this tensor binds to.
    pub name: String,
    /// Token values, one row per input of the batch.
    pub tensor: Array2<i64>,
}

/// One named output tensor of the graph, with its shape.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphOutput {
    /// The graph output this tensor came from.
    pub name: String,
    /// Output values, batch first.
    pub tensor: ArrayD<f32>,
}

/// Runs one ONNX graph on token tensors.
///
/// The pipeline reads [`input_names`](Self::input_names) once, when the model is built, to decide
/// whether to send `token_type_ids`.
/// Output selection by [`OutputKey`](crate::OutputKey) works on the outputs [`run`](Self::run)
/// returns, so `OutputKey::OnlyOne` and `OutputKey::ByOrder` depend on their count and order.
pub trait GraphSession: Send + Sync {
    /// Names of the graph inputs, in graph order.
    fn input_names(&self) -> &[String];

    /// Names of the graph outputs, in graph order.
    fn output_names(&self) -> &[String];

    /// Runs the graph on one batch and returns every output.
    ///
    /// The result holds exactly one output per entry of [`output_names`](Self::output_names), in
    /// that order; the pipeline rejects any other result.
    ///
    /// # Errors
    ///
    /// Returns an error when the graph cannot run or an output is not an `f32` tensor.
    /// [`Error::GraphSession`] carries an implementation's own error as its source.
    fn run(&mut self, inputs: Vec<GraphInput>) -> Result<Vec<GraphOutput>>;
}

impl fmt::Debug for dyn GraphSession + '_ {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphSession")
            .field("inputs", &self.input_names())
            .field("outputs", &self.output_names())
            .finish_non_exhaustive()
    }
}

impl<G: GraphSession + ?Sized> GraphSession for Box<G> {
    fn input_names(&self) -> &[String] {
        (**self).input_names()
    }

    fn output_names(&self) -> &[String] {
        (**self).output_names()
    }

    fn run(&mut self, inputs: Vec<GraphInput>) -> Result<Vec<GraphOutput>> {
        (**self).run(inputs)
    }
}

/// Whether `session`'s graph takes `token_type_ids`.
pub(crate) fn needs_token_type_ids(session: &impl GraphSession) -> bool {
    session
        .input_names()
        .iter()
        .any(|name| name == "token_type_ids")
}

/// Runs one batch and checks that the outputs match the declared output names.
pub(crate) fn run_checked(
    session: &mut impl GraphSession,
    inputs: Vec<GraphInput>,
) -> Result<Vec<GraphOutput>> {
    let outputs = session.run(inputs)?;
    let declared = session.output_names();
    if !outputs
        .iter()
        .map(|output| &output.name)
        .eq(declared.iter())
    {
        return Err(Error::GraphSession(
            format!(
                "graph session returned outputs {:?}, but declares {declared:?}",
                outputs
                    .iter()
                    .map(|output| &output.name)
                    .collect::<Vec<_>>()
            )
            .into(),
        ));
    }
    Ok(outputs)
}
