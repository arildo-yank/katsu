//! Otimização e loop de treino mínimo.
//!
//! Os dados de um Tensor são imutáveis (o backward depende disso). Por isso o
//! SGD não altera parâmetros no lugar: cada passo devolve novas folhas com
//! `p - lr * grad`. Grafos anteriores continuam válidos e não há gradiente
//! acumulado entre passos, então zero_grad não é necessário.
use std::fmt;

use crate::{Tensor, TensorError};

type Result<T> = std::result::Result<T, TensorError>;

/// Descida de gradiente estocástica com taxa de aprendizado fixa.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sgd {
    learning_rate: f32,
}

impl Sgd {
    /// A taxa precisa ser finita e positiva.
    pub fn new(learning_rate: f32) -> Result<Self> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(TensorError::InvalidLearningRate);
        }
        Ok(Self { learning_rate })
    }

    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Devolve novas folhas `p - lr * grad`, na mesma ordem e shapes.
    /// Todos os parâmetros precisam ter gradiente; nada é criado se algum faltar.
    pub fn step(&self, params: &[Tensor]) -> Result<Vec<Tensor>> {
        let mut updated = Vec::new();
        updated.try_reserve_exact(params.len())?;
        for (index, param) in params.iter().enumerate() {
            let grad = param.grad()?;
            let grad = grad
                .as_ref()
                .ok_or(TensorError::MissingGradient { index })?;
            let mut data = Vec::new();
            data.try_reserve_exact(param.numel())?;
            data.extend(
                param
                    .data()
                    .iter()
                    .zip(grad)
                    .map(|(value, g)| value - self.learning_rate * g),
            );
            updated.push(Tensor::new(data, param.shape())?);
        }
        Ok(updated)
    }
}

/// Telemetria de uma época, no formato consumido pelo monitor arandu.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochMetrics {
    /// Começa em 1.
    pub epoch: usize,
    /// Perda calculada antes da atualização desta época.
    pub loss: f32,
    pub learning_rate: f32,
}

impl fmt::Display for EpochMetrics {
    /// Uma linha JSON (JSON Lines). `loss` é sempre finito: fit rejeita NaN/Inf.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{"epoch":{},"loss":{},"learning_rate":{}}}"#,
            self.epoch, self.loss, self.learning_rate
        )
    }
}

/// Executa `epochs` passos de SGD e devolve os parâmetros finais.
///
/// Em cada época: `loss = loss_fn(&params)`, backward, `on_epoch(métricas)` e
/// atualização. A perda precisa ser escalar e finita; caso contrário o treino
/// para com erro e nenhuma métrica é emitida para essa época.
pub fn fit<F, C>(
    mut params: Vec<Tensor>,
    optimizer: &Sgd,
    epochs: usize,
    mut loss_fn: F,
    mut on_epoch: C,
) -> Result<Vec<Tensor>>
where
    F: FnMut(&[Tensor]) -> Result<Tensor>,
    C: FnMut(&EpochMetrics),
{
    for epoch in 1..=epochs {
        let loss = loss_fn(&params)?;
        if loss.numel() != 1 {
            return Err(TensorError::NonScalarBackward);
        }
        let value = loss.data()[0];
        if !value.is_finite() {
            return Err(TensorError::NonFiniteLoss { epoch });
        }
        loss.backward()?;
        on_epoch(&EpochMetrics {
            epoch,
            loss: value,
            learning_rate: optimizer.learning_rate(),
        });
        params = optimizer.step(&params)?;
    }
    Ok(params)
}
