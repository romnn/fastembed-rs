//! Tokenizer files and graph sessions for the crate's tests.

use crate::{GraphInput, GraphOutput, GraphSession, Result, TokenizerFiles};
use ndarray::ArrayD;

/// Files of a whitespace word-level tokenizer over `[UNK]`, `[PAD]`, `hello` and `world`, which
/// pads with `[PAD]` and adds no special tokens.
pub(crate) fn word_level_tokenizer_files() -> TokenizerFiles {
    TokenizerFiles {
        tokenizer_file: br#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "unk_token": "[UNK]",
                "vocab": {"[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3}
            }
        }"#
        .to_vec(),
        config_file: br#"{"pad_token_id": 1}"#.to_vec(),
        special_tokens_map_file: b"{}".to_vec(),
        tokenizer_config_file: br#"{"model_max_length": 512, "pad_token": "[PAD]"}"#.to_vec(),
    }
}

/// A session with the inputs `input_ids` and `attention_mask` that returns `output` for any
/// batch.
pub(crate) struct ScriptedSession {
    input_names: Vec<String>,
    output_names: Vec<String>,
    output: GraphOutput,
}

impl ScriptedSession {
    pub(crate) fn returning(name: &str, tensor: ArrayD<f32>) -> Self {
        Self {
            input_names: vec!["input_ids".to_owned(), "attention_mask".to_owned()],
            output_names: vec![name.to_owned()],
            output: GraphOutput {
                name: name.to_owned(),
                tensor,
            },
        }
    }
}

impl GraphSession for ScriptedSession {
    fn input_names(&self) -> &[String] {
        &self.input_names
    }

    fn output_names(&self) -> &[String] {
        &self.output_names
    }

    fn run(&mut self, _inputs: Vec<GraphInput>) -> Result<Vec<GraphOutput>> {
        Ok(vec![self.output.clone()])
    }
}
