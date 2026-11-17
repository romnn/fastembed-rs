/// Type-erased error of a dependency that exposes none this crate can name.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Error type returned by fastembed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Failed to retrieve model file '{file}'")]
    ModelRetrieval {
        file: String,
        #[source]
        source: BoxError,
    },

    #[error("Invalid tokenizer configuration: {0}")]
    TokenizerConfig(String),

    #[error("Failed to tokenize input: {0}")]
    Tokenization(String),

    #[error("Tokenizer returned empty encodings for the batch")]
    EmptyTokenizations,

    /// An error of the ONNX Runtime that runs the graph in process.
    ///
    /// This crate links no ONNX Runtime, so the error is carried as its source.
    #[error("ONNX runtime error: {0}")]
    Ort(#[source] BoxError),

    #[error("Failed to build ONNX session: {0}")]
    OrtBuilder(String),

    #[error("ONNX session error: {0}")]
    OrtSession(String),

    /// A [`GraphSession`](crate::GraphSession) failed or broke its contract.
    #[error("Graph session error: {0}")]
    GraphSession(#[source] BoxError),

    #[error("Output tensor '{key}' not found")]
    OutputKeyMissing { key: String },

    #[error("Failed to extract tensor: {0}")]
    TensorExtraction(String),

    #[error("Failed to decode image: {0}")]
    ImageDecode(String),

    #[error("Invalid preprocessor configuration: {0}")]
    PreprocessorConfig(String),

    #[error("Image transform error: {0}")]
    ImageTransform(String),

    #[error("Invalid tensor shape: {0}")]
    InvalidShape(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("`environment` must be specified if using multiple models per process")]
    EnvironmentRequired,

    #[error("{0}")]
    Other(String),
}

/// Type alias for `Result` returning the fastembed [`Error`] type.
pub type Result<T> = std::result::Result<T, Error>;
