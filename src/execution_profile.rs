//! Fixed token shapes and session-owned CUDA allocator diagnostics.

use crate::{common::ort_error, ort_session::ModelSession, Error, TextEmbedding, TextRerank};
use ort::{
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    AsPointer,
};
use std::ffi::CStr;

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

impl ModelSession {
    /// Read the CUDA device-zero allocator belonging to this session.
    ///
    /// # Errors
    ///
    /// Returns an error if the session has no CUDA arena or statistics are unavailable, and
    /// [`Error::Other`] if the graph runs through a [`GraphSession`](crate::GraphSession) rather
    /// than in process.
    pub fn cuda_allocator_stats(&self) -> crate::Result<CudaAllocatorStats> {
        session_cuda_stats(self)
    }
}

fn session_cuda_stats(session: &ModelSession) -> crate::Result<CudaAllocatorStats> {
    // Build this error without ONNX Runtime, which a process running remote graphs may not load.
    let session = session.ort_session().ok_or_else(|| {
        Error::Other(
            "CUDA allocator statistics are only available for in-process ONNX Runtime sessions"
                .into(),
        )
    })?;
    cuda_stats(session).map_err(ort_error)
}

impl TextEmbedding {
    /// Fix padding and truncation without changing pad IDs, special tokens or padding direction.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid length or missing tokenizer configuration.
    pub fn set_fixed_padding(&mut self, length: usize) -> crate::Result<()> {
        self.pipeline.set_fixed_padding(length)
    }

    /// Read the CUDA device-zero allocator belonging to this session.
    ///
    /// # Errors
    ///
    /// Returns an error if the session has no CUDA arena or statistics are unavailable, and
    /// [`Error::Other`] if the graph runs through a [`GraphSession`](crate::GraphSession) rather
    /// than an in-process ONNX Runtime session.
    pub fn cuda_allocator_stats(&self) -> crate::Result<CudaAllocatorStats> {
        session_cuda_stats(self.pipeline.session())
    }
}

impl TextRerank {
    /// Fix padding and complete query/document-pair truncation, preserving tokenizer semantics.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid length or missing tokenizer configuration.
    pub fn set_fixed_padding(&mut self, length: usize) -> crate::Result<()> {
        self.pipeline.set_fixed_padding(length)
    }

    /// Read the CUDA device-zero allocator belonging to this session.
    ///
    /// # Errors
    ///
    /// Returns an error if the session has no CUDA arena or statistics are unavailable, and
    /// [`Error::Other`] if the graph runs through a [`GraphSession`](crate::GraphSession) rather
    /// than an in-process ONNX Runtime session.
    pub fn cuda_allocator_stats(&self) -> crate::Result<CudaAllocatorStats> {
        session_cuda_stats(self.pipeline.session())
    }
}
