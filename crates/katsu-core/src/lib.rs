//! Tensores CPU contíguos com autograd reverso em grafo dinâmico.
#![deny(unsafe_code)]

mod backend;
#[allow(unsafe_code)]
mod simd;
pub use backend::{create_cuda_backend, Backend, CpuBackend, GpuBackend};
#[cfg(feature = "cuda")]
#[allow(unsafe_code)]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;

mod tensor;
pub use tensor::{Op, Tensor, TensorError};

mod conv2d;
pub use conv2d::Conv2D;

mod ops;
pub use ops::LAYER_NORM_EPS;

mod optim;
pub use optim::{fit, Adam, EpochMetrics, Optimizer, Sgd};

mod lm;
pub use lm::{
    load_params, penalize_repetitions, save_params, CharTokenizer, GenerationOptions,
    KatsuLmConfig, SplitMix64,
};

mod bpe;
pub use bpe::{BpeTokenizer, INSTRUCTION_TOKENS};
