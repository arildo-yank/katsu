//! Tensores CPU contíguos com autograd reverso em grafo dinâmico.
#![forbid(unsafe_code)]

mod tensor;
pub use tensor::{Op, Tensor, TensorError};

mod conv2d;
pub use conv2d::Conv2D;

mod ops;
pub use ops::LAYER_NORM_EPS;

mod optim;
pub use optim::{fit, Adam, EpochMetrics, Optimizer, Sgd};

mod lm;
pub use lm::{load_params, save_params, CharTokenizer, KatsuLmConfig, SplitMix64};
