//! The in-process [`GraphSession`]: an ONNX Runtime session.

use crate::common::{ort_error, Error, Result};
use crate::text_embedding::{ExternalInitializerFile, InitOptionsUserDefined};
use fastembed_pipeline::{GraphInput, GraphOutput, GraphSession};
use ort::{
    session::{Session, SessionInputValue},
    value::Value,
};
use std::borrow::Cow;

/// A [`GraphSession`] over an in-process ONNX Runtime session.
#[derive(Debug)]
pub struct OrtGraphSession {
    session: Session,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl OrtGraphSession {
    /// Loads the graph `onnx_file` with its `external_initializers` from memory under
    /// `options`, whose `max_length` belongs to the tokenizer and is not read here.
    ///
    /// # Errors
    ///
    /// Returns an error when ONNX Runtime cannot build a session from the bytes under `options`.
    pub fn from_memory(
        onnx_file: &[u8],
        external_initializers: Vec<ExternalInitializerFile>,
        options: InitOptionsUserDefined,
    ) -> Result<Self> {
        let (mut session_builder, _max_length) = options.into_session_builder()?;
        for ExternalInitializerFile { file_name, buffer } in external_initializers {
            session_builder = session_builder
                .with_external_initializer_file_in_memory(file_name, buffer.into())
                .map_err(|err| Error::OrtBuilder(err.to_string()))?;
        }
        let session = session_builder
            .commit_from_memory(onnx_file)
            .map_err(ort_error)?;
        Ok(Self::new(session))
    }

    /// Runs the graph of `session`.
    pub fn new(session: Session) -> Self {
        let names = |outlets: &[ort::value::Outlet]| {
            outlets
                .iter()
                .map(|outlet| outlet.name().to_string())
                .collect()
        };
        Self {
            input_names: names(session.inputs()),
            output_names: names(session.outputs()),
            session,
        }
    }
}

impl GraphSession for OrtGraphSession {
    fn input_names(&self) -> &[String] {
        &self.input_names
    }

    fn output_names(&self) -> &[String] {
        &self.output_names
    }

    fn run(&mut self, inputs: Vec<GraphInput>) -> Result<Vec<GraphOutput>> {
        let inputs = inputs
            .into_iter()
            .map(|GraphInput { name, tensor }| {
                let value = SessionInputValue::from(Value::from_array(tensor).map_err(ort_error)?);
                Ok((Cow::Owned(name), value))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut outputs = self
            .session
            .run(inputs)
            .map_err(|e| Error::OrtSession(e.to_string()))?;
        self.output_names
            .iter()
            .map(|name| {
                let value = outputs
                    .remove(name)
                    .ok_or_else(|| Error::OutputKeyMissing { key: name.clone() })?;
                let tensor = value
                    .try_extract_array::<f32>()
                    .map_err(|e| {
                        Error::TensorExtraction(format!("Failed to extract output '{name}': {e}"))
                    })?
                    .to_owned();
                Ok(GraphOutput {
                    name: name.clone(),
                    tensor,
                })
            })
            .collect()
    }
}

/// The graph session a fastembed model runs, keeping the in-process session reachable for
/// diagnostics.
#[derive(Debug)]
pub enum ModelSession {
    /// The graph runs in this process.
    Ort(OrtGraphSession),
    /// The graph runs through a caller's session.
    Graph(Box<dyn GraphSession>),
}

impl ModelSession {
    pub(crate) fn ort(session: Session) -> Self {
        Self::Ort(OrtGraphSession::new(session))
    }

    /// The in-process ONNX Runtime session, if the graph runs in this process.
    pub fn ort_session(&self) -> Option<&Session> {
        match self {
            Self::Ort(session) => Some(&session.session),
            Self::Graph(_) => None,
        }
    }
}

impl GraphSession for ModelSession {
    fn input_names(&self) -> &[String] {
        match self {
            Self::Ort(session) => session.input_names(),
            Self::Graph(session) => session.input_names(),
        }
    }

    fn output_names(&self) -> &[String] {
        match self {
            Self::Ort(session) => session.output_names(),
            Self::Graph(session) => session.output_names(),
        }
    }

    fn run(&mut self, inputs: Vec<GraphInput>) -> Result<Vec<GraphOutput>> {
        match self {
            Self::Ort(session) => session.run(inputs),
            Self::Graph(session) => session.run(inputs),
        }
    }
}
