//! Tokenizers loaded from file bytes, and the padded token tensors they produce.

use crate::error::{Error, Result};
use crate::graph_session::GraphInput;
use ndarray::Array2;
use tokenizers::{
    AddedToken, EncodeInput, Encoding, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams,
};

/// Tokenizer files for "bring your own" models.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerFiles {
    pub tokenizer_file: Vec<u8>,
    pub config_file: Vec<u8>,
    pub special_tokens_map_file: Vec<u8>,
    pub tokenizer_config_file: Vec<u8>,
}

/// Function can be called directly from the try_new_from_user_defined function (providing file bytes)
///
/// Or indirectly from the try_new function via load_tokenizer_hf_hub (converting HF files to bytes)
pub fn load_tokenizer(tokenizer_files: TokenizerFiles, max_length: usize) -> Result<Tokenizer> {
    let base_error_message =
        "Error building TokenizerFiles for UserDefinedEmbeddingModel. Could not read {} file.";

    // Deserialize each tokenizer file
    let config: serde_json::Value =
        serde_json::from_slice(&tokenizer_files.config_file).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                base_error_message.replace("{}", "config.json"),
            )
        })?;
    let special_tokens_map: serde_json::Value =
        serde_json::from_slice(&tokenizer_files.special_tokens_map_file).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                base_error_message.replace("{}", "special_tokens_map.json"),
            )
        })?;
    let tokenizer_config: serde_json::Value =
        serde_json::from_slice(&tokenizer_files.tokenizer_config_file).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                base_error_message.replace("{}", "tokenizer_config.json"),
            )
        })?;
    let mut tokenizer: tokenizers::Tokenizer =
        tokenizers::Tokenizer::from_bytes(tokenizer_files.tokenizer_file).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                base_error_message.replace("{}", "tokenizer.json"),
            )
        })?;

    // For BGEBaseSmall, model_max_length is set to 1000000000000000019884624838656.
    // This value fits in an f64.
    let model_max_length = tokenizer_config["model_max_length"]
        .as_f64()
        .ok_or_else(|| {
            Error::TokenizerConfig(
                "tokenizer_config.json is missing a numeric `model_max_length` field".into(),
            )
        })? as f32;
    let max_length = max_length.min(model_max_length as usize);
    let pad_token_value = &tokenizer_config["pad_token"];
    let pad_token: String = pad_token_value
        .as_str()
        .or_else(|| pad_token_value["content"].as_str())
        .ok_or_else(|| {
            Error::TokenizerConfig(
                "tokenizer_config.json is missing a string `pad_token` field".into(),
            )
        })?
        .into();
    let pad_id = config["pad_token_id"]
        .as_u64()
        .map(|id| id as u32)
        .or_else(|| tokenizer.token_to_id(&pad_token))
        .unwrap_or(0);

    let mut tokenizer = tokenizer
        .with_padding(Some(PaddingParams {
            // TODO: the user should be able to choose the padding strategy
            strategy: PaddingStrategy::BatchLongest,
            pad_token,
            pad_id,
            ..Default::default()
        }))
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(|e| Error::TokenizerConfig(e.to_string()))?
        .clone();
    if let serde_json::Value::Object(root_object) = special_tokens_map {
        for (_, value) in root_object.iter() {
            if value.is_string() {
                if let Some(content) = value.as_str() {
                    tokenizer
                        .add_special_tokens([AddedToken {
                            content: content.into(),
                            special: true,
                            ..Default::default()
                        }])
                        .map_err(|e| Error::TokenizerConfig(e.to_string()))?;
                }
            } else if value.is_object() {
                if let (
                    Some(content),
                    Some(single_word),
                    Some(lstrip),
                    Some(rstrip),
                    Some(normalized),
                ) = (
                    value["content"].as_str(),
                    value["single_word"].as_bool(),
                    value["lstrip"].as_bool(),
                    value["rstrip"].as_bool(),
                    value["normalized"].as_bool(),
                ) {
                    tokenizer
                        .add_special_tokens([AddedToken {
                            content: content.into(),
                            special: true,
                            single_word,
                            lstrip,
                            rstrip,
                            normalized,
                        }])
                        .map_err(|e| Error::TokenizerConfig(e.to_string()))?;
                }
            }
        }
    }
    Ok(tokenizer.into())
}

/// One tokenized batch as `[batch, sequence]` tensors.
pub struct EncodedBatch {
    /// Token ids, padded with the tokenizer's pad id.
    pub input_ids: Array2<i64>,
    /// 1 for a real token and 0 for padding.
    pub attention_mask: Array2<i64>,
    /// Segment ids, distinguishing the two texts of a pair.
    pub token_type_ids: Array2<i64>,
}

impl EncodedBatch {
    /// [`GraphSession`](crate::GraphSession) inputs.
    ///
    /// `input_ids` and `token_type_ids` are moved out, the latter attached only when the graph
    /// declares it, while `attention_mask` stays available for pooling.
    pub fn graph_inputs(&mut self, need_token_type_ids: bool) -> Vec<GraphInput> {
        let mut inputs = vec![
            GraphInput {
                name: "input_ids".into(),
                tensor: std::mem::take(&mut self.input_ids),
            },
            GraphInput {
                name: "attention_mask".into(),
                tensor: self.attention_mask.clone(),
            },
        ];
        if need_token_type_ids {
            inputs.push(GraphInput {
                name: "token_type_ids".into(),
                tensor: std::mem::take(&mut self.token_type_ids),
            });
        }
        inputs
    }
}

/// Tokenize a batch into padded `[batch, sequence]` tensors.
///
/// # Errors
///
/// Returns an error when the tokenizer fails or the batch is empty.
pub fn encode_batch<'s, E>(tokenizer: &Tokenizer, inputs: Vec<E>) -> Result<EncodedBatch>
where
    E: Into<EncodeInput<'s>> + Send,
{
    let encodings = tokenizer
        .encode_batch(inputs, true)
        .map_err(|e| Error::Tokenization(format!("Failed to encode the batch: {e}")))?;
    encodings_to_batch(&encodings)
}

fn encodings_to_batch(encodings: &[Encoding]) -> Result<EncodedBatch> {
    let encoding_length = encodings.first().ok_or(Error::EmptyTokenizations)?.len();
    let batch_size = encodings.len();
    let max_size = encoding_length * batch_size;

    let mut ids = Vec::with_capacity(max_size);
    let mut mask = Vec::with_capacity(max_size);
    let mut type_ids = Vec::with_capacity(max_size);
    for encoding in encodings {
        ids.extend(encoding.get_ids().iter().map(|&x| x as i64));
        mask.extend(encoding.get_attention_mask().iter().map(|&x| x as i64));
        type_ids.extend(encoding.get_type_ids().iter().map(|&x| x as i64));
    }

    let shape = (batch_size, encoding_length);
    let to_array = |data: Vec<i64>| {
        Array2::from_shape_vec(shape, data).map_err(|e| Error::InvalidShape(e.to_string()))
    };
    Ok(EncodedBatch {
        input_ids: to_array(ids)?,
        attention_mask: to_array(mask)?,
        token_type_ids: to_array(type_ids)?,
    })
}

/// Pads and truncates every encoding to exactly `length` tokens, keeping pad ids, special tokens
/// and the padding direction.
pub(crate) fn fixed_padding(tokenizer: &mut Tokenizer, length: usize) -> Result<()> {
    if length == 0 {
        return Err(Error::InvalidArgument(
            "sequence length must be positive".into(),
        ));
    }
    let mut padding = tokenizer
        .get_padding()
        .cloned()
        .ok_or_else(|| Error::TokenizerConfig("tokenizer has no padding configuration".into()))?;
    let mut truncation = tokenizer.get_truncation().cloned().ok_or_else(|| {
        Error::TokenizerConfig("tokenizer has no truncation configuration".into())
    })?;
    padding.strategy = PaddingStrategy::Fixed(length);
    padding.pad_to_multiple_of = None;
    truncation.max_length = length;
    tokenizer
        .with_truncation(Some(truncation))
        .map_err(|error| Error::TokenizerConfig(error.to_string()))?;
    tokenizer.with_padding(Some(padding));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::{
        models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
        processors::template::TemplateProcessing, PaddingDirection, TruncationDirection,
    };

    type TestResult<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

    fn minimal_tokenizer_bytes() -> Vec<u8> {
        // Minimal valid tokenizer.json (word-level model with a tiny vocab).
        br#"{
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
                "vocab": {"[UNK]": 0, "[PAD]": 1, "hello": 2}
            }
        }"#
        .to_vec()
    }

    fn tokenizer_files(tokenizer_config: &str) -> TokenizerFiles {
        TokenizerFiles {
            tokenizer_file: minimal_tokenizer_bytes(),
            config_file: br#"{"pad_token_id": 0}"#.to_vec(),
            special_tokens_map_file: b"{}".to_vec(),
            tokenizer_config_file: tokenizer_config.as_bytes().to_vec(),
        }
    }

    #[test]
    fn load_tokenizer_ok_with_complete_config() {
        let files = tokenizer_files(r#"{"model_max_length": 512, "pad_token": "[PAD]"}"#);
        assert!(load_tokenizer(files, 512).is_ok());
    }

    #[test]
    fn load_tokenizer_errors_on_missing_pad_token() {
        let files = tokenizer_files(r#"{"model_max_length": 512}"#);
        let err = load_tokenizer(files, 512).unwrap_err();
        assert!(
            err.to_string().contains("pad_token"),
            "error message was: {err}"
        );
    }

    #[test]
    fn load_tokenizer_errors_on_missing_model_max_length() {
        let files = tokenizer_files(r#"{"pad_token": "[PAD]"}"#);
        let err = load_tokenizer(files, 512).unwrap_err();
        assert!(
            err.to_string().contains("model_max_length"),
            "error message was: {err}"
        );
    }
    #[test]
    fn load_tokenizer_accepts_added_token_object_as_pad_token() {
        let files = tokenizer_files(
            r#"{"model_max_length": 512, "pad_token": {"__type": "AddedToken", "content": "[PAD]", "lstrip": false}}"#,
        );
        let tokenizer = load_tokenizer(files, 512).unwrap();
        assert_eq!(tokenizer.get_padding().unwrap().pad_token, "[PAD]");
    }

    #[test]
    fn load_tokenizer_resolves_pad_id_from_vocab_when_config_lacks_it() {
        let mut files = tokenizer_files(r#"{"model_max_length": 512, "pad_token": "[PAD]"}"#);
        files.config_file = b"{}".to_vec();
        let tokenizer = load_tokenizer(files, 512).unwrap();
        assert_eq!(tokenizer.get_padding().unwrap().pad_id, 1);
    }

    #[test]
    fn encode_batch_pads_to_longest_and_masks_padding() {
        let files = tokenizer_files(r#"{"model_max_length": 512, "pad_token": "[PAD]"}"#);
        let tokenizer = load_tokenizer(files, 512).unwrap();
        let batch = encode_batch(&tokenizer, vec!["hello", "hello hello"]).unwrap();
        assert_eq!(batch.input_ids.dim(), (2, 2));
        assert_eq!(batch.attention_mask.row(0).to_vec(), vec![1, 0]);
        assert_eq!(batch.attention_mask.row(1).to_vec(), vec![1, 1]);
        assert_eq!(batch.input_ids[[0, 1]], 0);
        assert_eq!(batch.input_ids.row(1).to_vec(), vec![2, 2]);
        assert!(encode_batch::<&str>(&tokenizer, vec![]).is_err());
    }

    fn tokenizer() -> TestResult<Tokenizer> {
        let model = WordLevel::builder()
            .vocab(
                [
                    ("[UNK]".into(), 0),
                    ("one".into(), 1),
                    ("two".into(), 2),
                    ("[PAD]".into(), 3),
                    ("[CLS]".into(), 4),
                    ("[SEP]".into(), 5),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("[UNK]".into())
            .build()?;
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("[CLS] $A [SEP]")?
                .try_pair("[CLS] $A [SEP] $B:1 [SEP]:1")?
                .special_tokens(vec![("[CLS]", 4), ("[SEP]", 5)])
                .build()?,
        ));
        tokenizer.with_padding(Some(PaddingParams {
            pad_id: 3,
            pad_token: "[PAD]".into(),
            pad_type_id: 9,
            direction: PaddingDirection::Left,
            pad_to_multiple_of: Some(8),
            ..Default::default()
        }));
        tokenizer.with_truncation(Some(TruncationParams {
            max_length: 16,
            direction: TruncationDirection::Left,
            ..Default::default()
        }))?;
        Ok(tokenizer)
    }

    /// Fixed profiles retain pad identity and direction while disabling multiple-based rounding.
    #[test]
    fn fixed_padding_preserves_tokenizer_semantics() -> TestResult<()> {
        let mut tokenizer = tokenizer()?;
        fixed_padding(&mut tokenizer, 6)?;
        let padding = tokenizer.get_padding().ok_or("missing padding")?;
        assert_eq!(padding.pad_id, 3);
        assert_eq!(padding.pad_type_id, 9);
        assert_eq!(padding.pad_token, "[PAD]");
        assert!(matches!(padding.direction, PaddingDirection::Left));
        assert_eq!(padding.pad_to_multiple_of, None);
        assert_eq!(
            tokenizer
                .get_truncation()
                .ok_or("missing truncation")?
                .direction,
            TruncationDirection::Left
        );

        let encoding = tokenizer.encode("one", true)?;
        assert_eq!(encoding.len(), 6);
        assert_eq!(encoding.get_ids().first(), Some(&3));
        assert_eq!(encoding.get_attention_mask().first(), Some(&0));
        Ok(())
    }

    /// Pair truncation includes its special tokens, so long query/document pairs stay in profile.
    #[test]
    fn fixed_padding_bounds_complete_pairs() -> TestResult<()> {
        let mut tokenizer = tokenizer()?;
        fixed_padding(&mut tokenizer, 6)?;
        let encoding = tokenizer.encode(("one two one two", "two one two one"), true)?;
        assert_eq!(encoding.len(), 6);
        assert_eq!(encoding.get_attention_mask().iter().sum::<u32>(), 6);
        assert_eq!(encoding.get_ids().iter().filter(|id| **id == 5).count(), 2);
        Ok(())
    }

    /// Invalid profile sizes fail before mutating an existing tokenizer configuration.
    #[test]
    fn invalid_profile_does_not_mutate_tokenizer() -> TestResult<()> {
        let mut tokenizer = tokenizer()?;
        let padding = serde_json::to_value(tokenizer.get_padding())?;
        let truncation = serde_json::to_value(tokenizer.get_truncation())?;
        assert!(matches!(
            fixed_padding(&mut tokenizer, 0),
            Err(Error::InvalidArgument(_))
        ));
        assert_eq!(serde_json::to_value(tokenizer.get_padding())?, padding);
        assert_eq!(
            serde_json::to_value(tokenizer.get_truncation())?,
            truncation
        );
        Ok(())
    }
}
