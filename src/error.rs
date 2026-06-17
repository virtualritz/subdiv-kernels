use thiserror::Error;

/// Error type returned by kernel APIs.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum KernelError {
    /// The mesh topology arrays are inconsistent.
    #[error("invalid topology: {0}")]
    InvalidTopology(&'static str),

    /// A requested feature is not implemented yet.
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),

    /// A GPU (wgpu) operation failed or was misconfigured.
    #[error("gpu error: {0}")]
    Gpu(String),
}
