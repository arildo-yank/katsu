//! Otimização e loop de treino mínimo.
//!
//! Os dados de um Tensor são imutáveis (o backward depende disso). Por isso o
//! SGD não altera parâmetros no lugar: cada passo devolve novas folhas com
//! `p - lr * grad`. Grafos anteriores continuam válidos e não há gradiente
//! acumulado entre passos, então zero_grad não é necessário.
use std::fmt;

use crate::{Tensor, TensorError};

type Result<T> = std::result::Result<T, TensorError>;

/// Otimizador que transforma parâmetros com gradiente em novas folhas.
pub trait Optimizer {
    fn step(&mut self, params: &[Tensor]) -> Result<Vec<Tensor>>;
    fn learning_rate(&self) -> f32;
}

fn gradients(params: &[Tensor]) -> Result<Vec<Vec<f32>>> {
    let mut grads = Vec::new();
    grads.try_reserve_exact(params.len())?;
    for (index, param) in params.iter().enumerate() {
        let grad = param.grad()?;
        let grad = grad
            .as_ref()
            .ok_or(TensorError::MissingGradient { index })?;
        let mut copy = Vec::new();
        copy.try_reserve_exact(grad.len())?;
        copy.extend_from_slice(grad);
        grads.push(copy);
    }
    Ok(grads)
}

fn validate_learning_rate(learning_rate: f32) -> Result<()> {
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err(TensorError::InvalidLearningRate);
    }
    Ok(())
}

/// Descida de gradiente estocástica com taxa de aprendizado fixa.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sgd {
    learning_rate: f32,
}

impl Sgd {
    /// A taxa precisa ser finita e positiva.
    pub fn new(learning_rate: f32) -> Result<Self> {
        validate_learning_rate(learning_rate)?;
        Ok(Self { learning_rate })
    }

    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Devolve novas folhas `p - lr * grad`, na mesma ordem e shapes.
    /// Todos os parâmetros precisam ter gradiente; nada é criado se algum faltar.
    pub fn step(&self, params: &[Tensor]) -> Result<Vec<Tensor>> {
        let grads = gradients(params)?;
        let mut updated = Vec::new();
        updated.try_reserve_exact(params.len())?;
        for (param, grad) in params.iter().zip(&grads) {
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

impl Optimizer for Sgd {
    fn step(&mut self, params: &[Tensor]) -> Result<Vec<Tensor>> {
        Sgd::step(self, params)
    }
    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
}

/// Adam (Kingma & Ba, 2015) com correção de viés. Guarda médias móveis por
/// parâmetro; a ordem e os shapes dos parâmetros não podem mudar entre passos.
#[derive(Debug, Clone)]
pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step: i32,
    moments: Vec<(Vec<f32>, Vec<f32>)>,
}

impl Adam {
    /// Padrões usuais: beta1 = 0.9, beta2 = 0.999, eps = 1e-8.
    pub fn new(learning_rate: f32) -> Result<Self> {
        Self::with_betas(learning_rate, 0.9, 0.999, 1e-8)
    }

    pub fn with_betas(learning_rate: f32, beta1: f32, beta2: f32, eps: f32) -> Result<Self> {
        validate_learning_rate(learning_rate)?;
        let valid_beta = |b: f32| (0.0..1.0).contains(&b);
        if !valid_beta(beta1) || !valid_beta(beta2) || !eps.is_finite() || eps <= 0.0 {
            return Err(TensorError::InvalidLearningRate);
        }
        Ok(Self {
            learning_rate,
            beta1,
            beta2,
            eps,
            step: 0,
            moments: Vec::new(),
        })
    }
}

impl Optimizer for Adam {
    /// Valida tudo antes de alterar o estado: um erro não consome um passo.
    fn step(&mut self, params: &[Tensor]) -> Result<Vec<Tensor>> {
        let grads = gradients(params)?;
        if self.moments.is_empty() {
            for param in params {
                self.moments
                    .push((vec![0.0; param.numel()], vec![0.0; param.numel()]));
            }
        } else if self.moments.len() != params.len()
            || self
                .moments
                .iter()
                .zip(params)
                .any(|(m, p)| m.0.len() != p.numel())
        {
            return Err(TensorError::OptimizerState);
        }
        self.step = self.step.saturating_add(1);
        let correction1 = 1.0 - self.beta1.powi(self.step);
        let correction2 = 1.0 - self.beta2.powi(self.step);
        let mut updated = Vec::new();
        updated.try_reserve_exact(params.len())?;
        for ((param, grad), (m, v)) in params.iter().zip(&grads).zip(&mut self.moments) {
            let mut data = Vec::new();
            data.try_reserve_exact(param.numel())?;
            for (((&value, &g), m), v) in param.data().iter().zip(grad).zip(m).zip(v) {
                *m = self.beta1 * *m + (1.0 - self.beta1) * g;
                *v = self.beta2 * *v + (1.0 - self.beta2) * g * g;
                let m_hat = *m / correction1;
                let v_hat = *v / correction2;
                data.push(value - self.learning_rate * m_hat / (v_hat.sqrt() + self.eps));
            }
            updated.push(Tensor::new(data, param.shape())?);
        }
        Ok(updated)
    }
    fn learning_rate(&self) -> f32 {
        self.learning_rate
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
pub fn fit<O, F, C>(
    mut params: Vec<Tensor>,
    optimizer: &mut O,
    epochs: usize,
    mut loss_fn: F,
    mut on_epoch: C,
) -> Result<Vec<Tensor>>
where
    O: Optimizer,
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
