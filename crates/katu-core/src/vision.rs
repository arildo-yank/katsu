//! Arandu-Ve: projetor que leva vetores de um codificador de imagem congelado (ex.: SigLIP,
//! 768 por patch) para o espaço de embeddings do modelo de linguagem. Os vetores projetados
//! entram em `KatuLmConfig::forward_with_prefix` no lugar dos primeiros tokens.
use crate::{lm::SplitMix64, Tensor, TensorError};
use std::sync::Arc;

type Result<T> = std::result::Result<T, TensorError>;

/// MLP de duas camadas, `x·W1 + b1 → GELU → ·W2 + b2`.
#[derive(Debug, Clone)]
pub struct Projector {
    pub w1: Tensor,
    pub b1: Tensor,
    pub w2: Tensor,
    pub b2: Tensor,
}

impl Projector {
    /// Pesos uniformes com desvio-padrão 0,02, como o resto do modelo; vieses em zero.
    pub fn new(input: usize, hidden: usize, output: usize, seed: u64) -> Result<Self> {
        if input == 0 || hidden == 0 || output == 0 {
            return Err(TensorError::InvalidShape(
                "projector sizes must be positive",
            ));
        }
        let mut rng = SplitMix64(seed);
        Ok(Self {
            w1: rng.tensor(&[input, hidden], 0.02)?,
            b1: Tensor::zeros(&[hidden])?,
            w2: rng.tensor(&[hidden, output], 0.02)?,
            b2: Tensor::zeros(&[output])?,
        })
    }

    /// `[w1, b1, w2, b2]`, na ordem de `from_params` (para o otimizador e o checkpoint).
    pub fn params(&self) -> Vec<Tensor> {
        vec![
            self.w1.clone(),
            self.b1.clone(),
            self.w2.clone(),
            self.b2.clone(),
        ]
    }

    pub fn from_params(params: Vec<Tensor>) -> Result<Self> {
        let [w1, b1, w2, b2]: [Tensor; 4] = params
            .try_into()
            .map_err(|_| TensorError::InvalidShape("projector needs 4 tensors"))?;
        if w1.rank() != 2
            || w2.rank() != 2
            || b1.shape() != [w1.shape()[1]]
            || w2.shape()[0] != w1.shape()[1]
            || b2.shape() != [w2.shape()[1]]
        {
            return Err(TensorError::InvalidShape("projector shapes do not chain"));
        }
        Ok(Self { w1, b1, w2, b2 })
    }

    /// `features` `[linhas, entrada]` → `[linhas, saída]`, com gradiente para os pesos.
    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let cpu: Arc<dyn crate::Backend> = Arc::new(crate::CpuBackend::default());
        features
            .linear_with_backend(&self.w1, &self.b1, cpu.clone())?
            .gelu()?
            .linear_with_backend(&self.w2, &self.b2, cpu)
    }
}
