//! Tensores CPU contíguos com autograd reverso em grafo dinâmico.
#![forbid(unsafe_code)]

mod tensor;
pub use tensor::{Op, Tensor, TensorError};

mod conv2d;
pub use conv2d::Conv2D;
