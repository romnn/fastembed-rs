//! Text embedding module, containing the main struct [TextEmbedding] and its
//! initialization options.

// Constants.
const DEFAULT_MAX_LENGTH: usize = 512;

// Initialization options.
mod init;
pub use init::*;

// The implementation of the embedding models.
mod r#impl;
