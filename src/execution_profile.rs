//! Fixed token shapes, run configuration and session-owned CUDA allocator diagnostics.

use crate::{Error, SparseTextEmbedding, TextEmbedding, TextRerank};
use ort::{
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::RunOptions,
    AsPointer,
};
use std::ffi::CStr;
use tokenizers::{PaddingStrategy, Tokenizer};

/// Statistics from one session's CUDA arena, after synchronous execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaAllocatorStats {
    /// Live allocation bytes.
    pub in_use: u64,
    /// Bytes retained by the allocator, including reserved initializers.
    pub total_allocated: u64,
    /// Number of reusable arena extensions.
    pub num_arena_extensions: u64,
    /// Number of direct reservations, including temporary kernel scratch.
    pub num_reserves: u64,
    /// Largest individual allocation in bytes.
    pub max_alloc_size: u64,
}

/// The run options every run of a session takes, built from its run configuration.
pub(crate) struct RunConfig(pub(crate) RunOptions);

impl std::fmt::Debug for RunConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("RunConfig").finish_non_exhaustive()
    }
}

/// Run options holding every `(key, value)` entry of `config`, or `None` when it holds none, so a
/// session without run configuration keeps the plain run.
fn run_options<K: AsRef<str>, V: AsRef<str>>(
    config: &[(K, V)],
) -> crate::Result<Option<RunConfig>> {
    if config.is_empty() {
        return Ok(None);
    }
    let mut options = RunOptions::new().map_err(|e| Error::OrtSession(e.to_string()))?;
    for (key, value) in config {
        options
            .set(key, value)
            .map_err(|e| Error::OrtSession(e.to_string()))?;
    }
    Ok(Some(RunConfig(options)))
}

fn fixed_padding(tokenizer: &mut Tokenizer, length: usize) -> crate::Result<()> {
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

fn cuda_stats(session: &ort::session::Session) -> ort::Result<CudaAllocatorStats> {
    let allocator = Allocator::new(
        session,
        MemoryInfo::new(
            AllocationDevice::CUDA,
            0,
            AllocatorType::Arena,
            MemoryType::Default,
        )?,
    )?;
    let mut pairs = std::ptr::null_mut();
    // SAFETY: The session-backed allocator stays alive throughout the call and ORT owns the
    // returned key/value object until ReleaseKeyValuePairs below.
    unsafe {
        ort::Error::result_from_status((ort::api().AllocatorGetStats)(
            allocator.ptr(),
            &mut pairs,
        ))?;
    }
    if pairs.is_null() {
        return Err(ort::Error::new("CUDA allocator returned no statistics"));
    }
    let read = |key: &CStr| -> ort::Result<u64> {
        // SAFETY: The key is NUL-terminated and pairs remains alive until every value is copied.
        let value = unsafe { (ort::api().GetKeyValue)(pairs, key.as_ptr()) };
        if value.is_null() {
            return Err(ort::Error::new(format!(
                "missing allocator statistic {key:?}"
            )));
        }
        // SAFETY: ORT returns a NUL-terminated string owned by the live key/value object.
        unsafe { CStr::from_ptr(value) }
            .to_str()
            .map_err(ort::Error::wrap)?
            .parse::<u64>()
            .map_err(ort::Error::wrap)
    };
    let result = (|| {
        Ok(CudaAllocatorStats {
            in_use: read(c"InUse")?,
            total_allocated: read(c"TotalAllocated")?,
            num_arena_extensions: read(c"NumArenaExtensions")?,
            num_reserves: read(c"NumReserves")?,
            max_alloc_size: read(c"MaxAllocSize")?,
        })
    })();
    // SAFETY: This releases the single object returned by AllocatorGetStats, after all reads.
    unsafe {
        (ort::api().ReleaseKeyValuePairs)(pairs);
    }
    result
}

impl TextEmbedding {
    /// Set the ONNX Runtime run configuration entries every later run of the session takes,
    /// replacing any set before; an empty `config` restores plain runs.
    /// Example: `.set_run_config(&[("memory.enable_memory_arena_shrinkage", "gpu:0")])`.
    ///
    /// # Errors
    ///
    /// Returns an error if ONNX Runtime refuses an entry.
    pub fn set_run_config<K: AsRef<str>, V: AsRef<str>>(
        &mut self,
        config: &[(K, V)],
    ) -> crate::Result<()> {
        self.run_options = run_options(config)?;
        Ok(())
    }

    /// Fix padding and truncation without changing pad IDs, special tokens or padding direction.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid length or missing tokenizer configuration.
    pub fn set_fixed_padding(&mut self, length: usize) -> crate::Result<()> {
        fixed_padding(&mut self.tokenizer, length)
    }

    /// Read the CUDA device-zero allocator belonging to this session.
    ///
    /// # Errors
    ///
    /// Returns an error if the session has no CUDA arena or statistics are unavailable.
    pub fn cuda_allocator_stats(&self) -> ort::Result<CudaAllocatorStats> {
        cuda_stats(&self.session)
    }
}

impl TextRerank {
    /// Set the ONNX Runtime run configuration entries every later run of the session takes,
    /// replacing any set before; an empty `config` restores plain runs.
    /// Example: `.set_run_config(&[("memory.enable_memory_arena_shrinkage", "gpu:0")])`.
    ///
    /// # Errors
    ///
    /// Returns an error if ONNX Runtime refuses an entry.
    pub fn set_run_config<K: AsRef<str>, V: AsRef<str>>(
        &mut self,
        config: &[(K, V)],
    ) -> crate::Result<()> {
        self.run_options = run_options(config)?;
        Ok(())
    }

    /// Fix padding and complete query/document-pair truncation, preserving tokenizer semantics.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid length or missing tokenizer configuration.
    pub fn set_fixed_padding(&mut self, length: usize) -> crate::Result<()> {
        fixed_padding(&mut self.tokenizer, length)
    }

    /// Read the CUDA device-zero allocator belonging to this session.
    ///
    /// # Errors
    ///
    /// Returns an error if the session has no CUDA arena or statistics are unavailable.
    pub fn cuda_allocator_stats(&self) -> ort::Result<CudaAllocatorStats> {
        cuda_stats(&self.session)
    }
}

impl SparseTextEmbedding {
    /// Set the ONNX Runtime run configuration entries every later run of the session takes,
    /// replacing any set before; an empty `config` restores plain runs.
    /// Example: `.set_run_config(&[("memory.enable_memory_arena_shrinkage", "gpu:0")])`.
    ///
    /// # Errors
    ///
    /// Returns an error if ONNX Runtime refuses an entry.
    pub fn set_run_config<K: AsRef<str>, V: AsRef<str>>(
        &mut self,
        config: &[(K, V)],
    ) -> crate::Result<()> {
        self.run_options = run_options(config)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::{
        models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
        processors::template::TemplateProcessing, PaddingDirection, PaddingParams,
        TruncationDirection, TruncationParams,
    };

    #[test]
    fn an_empty_run_config_keeps_plain_runs() -> crate::Result<()> {
        assert!(run_options::<&str, &str>(&[])?.is_none());
        Ok(())
    }

    #[test]
    fn a_run_config_sets_every_entry() -> crate::Result<()> {
        let config = run_options(&[
            ("memory.enable_memory_arena_shrinkage", "cpu:0"),
            ("run.example", "1"),
        ])?;
        let Some(RunConfig(mut options)) = config else {
            panic!("a non-empty run config builds run options");
        };
        assert_eq!(
            options
                .get("memory.enable_memory_arena_shrinkage")
                .as_deref(),
            Some("cpu:0")
        );
        assert_eq!(options.get("run.example").as_deref(), Some("1"));
        Ok(())
    }

    fn tokenizer() -> Result<Tokenizer, Box<dyn std::error::Error + Send + Sync>> {
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
    fn fixed_padding_preserves_tokenizer_semantics(
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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
    fn fixed_padding_bounds_complete_pairs() -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    {
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
    fn invalid_profile_does_not_mutate_tokenizer(
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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
