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
pub use cuda::{CudaBackend, CudaParamStore};
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
mod accelerate;
#[cfg(target_os = "macos")]
pub use accelerate::AccelerateBackend;

mod tensor;
pub use tensor::{Op, Tensor, TensorError};

mod conv2d;
pub use conv2d::Conv2D;

mod ops;
pub use ops::LAYER_NORM_EPS;

mod optim;
pub use optim::{fit, step_partial, Adam, AdamStep, EpochMetrics, Optimizer, Sgd};

mod resident;
pub use resident::{
    arena_len, materialize, upload, Buffer, HostParamStore, ParamStore, Transfers, View,
};

mod lm;
mod vision;
pub use lm::{
    load_params, penalize_repetitions, save_params, CharTokenizer, GenerationOptions,
    KatsuLmConfig, SplitMix64,
};
pub use vision::Projector;

mod bpe;
pub use bpe::{BpeTokenizer, OrdinaryEncoder, INSTRUCTION_TOKENS};

mod gpu_layout;
pub use gpu_layout::{GpuLayout, LayerSlots, SftBatch, Slot};
#[cfg(feature = "cuda")]
#[allow(unsafe_code)]
mod gpu_train;
#[cfg(feature = "cuda")]
pub use gpu_train::{GemmKernel, GpuTrainOptions, GpuTrainer};
