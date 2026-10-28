//! Adam com momentos de 8 bits, por blocos (ideia de Dettmers et al., "8-bit Optimizers via
//! Block-wise Quantization", 2022; aqui com um mapa de compansão mais simples).
//!
//! Cada bloco de `BLOCK` números guarda um fator de escala FP32 e um byte por número:
//! - primeiro momento `m` (com sinal): `q = round(127·sinal·√(|m|/escala))`;
//! - segundo momento `v` (≥ 0): guarda-se `u = √v`, e `q = round(255·√(u/escala))`.
//!
//! A raiz dá mais resolução perto de zero, onde a maioria dos momentos fica; sem ela, a
//! quantização linear zera os valores pequenos de um bloco com um valor grande.
//! Estado: 2 bytes + 8/BLOCK bytes por parâmetro (~2,03), contra 8 do Adam FP32.
//! O passo dequantiza o bloco, faz a conta do Adam em FP32 e requantiza; pesos-mestre e
//! gradientes continuam FP32.
use rayon::prelude::*;

use crate::{AdamStep, Optimizer, Tensor, TensorError};

type Result<T> = std::result::Result<T, TensorError>;

pub const ADAM8_BLOCK: usize = 256;

#[derive(Debug, Clone, Default)]
struct State {
    m: Vec<i8>,
    v: Vec<u8>,
    m_scale: Vec<f32>,
    v_scale: Vec<f32>,
}

impl State {
    fn zeros(len: usize) -> Result<Self> {
        let blocks = len.div_ceil(ADAM8_BLOCK);
        let mut state = Self::default();
        state.m.try_reserve_exact(len)?;
        state.v.try_reserve_exact(len)?;
        state.m_scale.try_reserve_exact(blocks)?;
        state.v_scale.try_reserve_exact(blocks)?;
        state.m.resize(len, 0);
        state.v.resize(len, 0);
        state.m_scale.resize(blocks, 0.0);
        state.v_scale.resize(blocks, 0.0);
        Ok(state)
    }

    fn bytes(&self) -> usize {
        self.m.len() + self.v.len() + 4 * (self.m_scale.len() + self.v_scale.len())
    }
}

#[inline]
fn decode_m(q: i8, scale: f32) -> f32 {
    let r = f32::from(q) / 127.0;
    r * r.abs() * scale
}

#[inline]
fn encode_m(x: f32, scale: f32) -> i8 {
    if scale == 0.0 {
        return 0;
    }
    (127.0 * x.signum() * (x.abs() / scale).sqrt())
        .round()
        .clamp(-127.0, 127.0) as i8
}

#[inline]
fn decode_v(q: u8, scale: f32) -> f32 {
    let r = f32::from(q) / 255.0;
    let u = r * r * scale;
    u * u
}

#[inline]
fn encode_v(u: f32, scale: f32) -> u8 {
    if scale == 0.0 {
        return 0;
    }
    (255.0 * (u / scale).sqrt()).round().clamp(0.0, 255.0) as u8
}

/// Um bloco: dequantiza, aplica o Adam em FP32 sobre `weights` e requantiza.
fn update_block(
    c: AdamStep,
    weights: &mut [f32],
    grad: &[f32],
    m: &mut [i8],
    v: &mut [u8],
    m_scale: &mut f32,
    v_scale: &mut f32,
) {
    let mut new_m = [0.0f32; ADAM8_BLOCK];
    let mut new_u = [0.0f32; ADAM8_BLOCK];
    let (mut max_m, mut max_u) = (0.0f32, 0.0f32);
    for i in 0..weights.len() {
        let g = grad[i];
        let mi = c.beta1 * decode_m(m[i], *m_scale) + (1.0 - c.beta1) * g;
        let vi = c.beta2 * decode_v(v[i], *v_scale) + (1.0 - c.beta2) * g * g;
        weights[i] -= c.lr * (mi / c.correction1) / ((vi / c.correction2).sqrt() + c.eps);
        new_m[i] = mi;
        new_u[i] = vi.sqrt();
        max_m = max_m.max(mi.abs());
        max_u = max_u.max(new_u[i]);
    }
    *m_scale = max_m;
    *v_scale = max_u;
    for i in 0..weights.len() {
        m[i] = encode_m(new_m[i], max_m);
        v[i] = encode_v(new_u[i], max_u);
    }
}

/// Adam com momentos de 8 bits. Não aceita parâmetros residentes (os do `ParamStore`).
#[derive(Debug, Clone)]
pub struct Adam8bit {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step: i32,
    states: Vec<State>,
}

impl Adam8bit {
    /// Padrões usuais: beta1 = 0.9, beta2 = 0.999, eps = 1e-8.
    pub fn new(learning_rate: f32) -> Result<Self> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(TensorError::InvalidLearningRate);
        }
        Ok(Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            step: 0,
            states: Vec::new(),
        })
    }

    pub fn set_learning_rate(&mut self, learning_rate: f32) -> Result<()> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(TensorError::InvalidLearningRate);
        }
        self.learning_rate = learning_rate;
        Ok(())
    }

    /// Bytes do estado do otimizador (momentos e escalas), sem cabeçalhos de Vec.
    pub fn state_bytes(&self) -> usize {
        self.states.iter().map(State::bytes).sum()
    }

    fn prepare(&mut self, params: &[Tensor]) -> Result<()> {
        if params.iter().any(Tensor::is_resident) {
            return Err(TensorError::OptimizerState);
        }
        if self.states.is_empty() {
            for param in params {
                self.states.push(State::zeros(param.numel())?);
            }
        } else if self.states.len() != params.len()
            || self
                .states
                .iter()
                .zip(params)
                .any(|(s, p)| s.m.len() != p.numel())
        {
            return Err(TensorError::OptimizerState);
        }
        Ok(())
    }

    fn advance(&mut self) -> AdamStep {
        self.step = self.step.saturating_add(1);
        AdamStep {
            beta1: self.beta1,
            beta2: self.beta2,
            lr: self.learning_rate,
            eps: self.eps,
            correction1: 1.0 - self.beta1.powi(self.step),
            correction2: 1.0 - self.beta2.powi(self.step),
        }
    }
}

fn apply(c: AdamStep, weights: &mut [f32], grad: &[f32], state: &mut State) {
    weights
        .par_chunks_mut(ADAM8_BLOCK)
        .zip(grad.par_chunks(ADAM8_BLOCK))
        .zip(state.m.par_chunks_mut(ADAM8_BLOCK))
        .zip(state.v.par_chunks_mut(ADAM8_BLOCK))
        .zip(
            state
                .m_scale
                .par_iter_mut()
                .zip(state.v_scale.par_iter_mut()),
        )
        .for_each(|((((w, g), m), v), (ms, vs))| update_block(c, w, g, m, v, ms, vs));
}

impl Optimizer for Adam8bit {
    fn step(&mut self, params: &[Tensor]) -> Result<Vec<Tensor>> {
        for (index, param) in params.iter().enumerate() {
            if param.grad()?.is_none() {
                return Err(TensorError::MissingGradient { index });
            }
        }
        self.prepare(params)?;
        let c = self.advance();
        let mut updated = Vec::new();
        updated.try_reserve_exact(params.len())?;
        for (param, state) in params.iter().zip(&mut self.states) {
            let mut data = param.data().to_vec();
            let guard = param.grad()?;
            let grad = guard.as_deref().ok_or(TensorError::OptimizerState)?;
            apply(c, &mut data, grad, state);
            updated.push(Tensor::new(data, param.shape())?);
        }
        Ok(updated)
    }

    /// Pesos sem outro dono são atualizados no próprio buffer, como no `Adam::step_owned`.
    fn step_owned(&mut self, params: Vec<Tensor>) -> Result<Vec<Tensor>> {
        for (index, param) in params.iter().enumerate() {
            if param.grad()?.is_none() {
                return Err(TensorError::MissingGradient { index });
            }
        }
        self.prepare(&params)?;
        let c = self.advance();
        let mut updated = Vec::new();
        updated.try_reserve_exact(params.len())?;
        for (param, state) in params.into_iter().zip(&mut self.states) {
            match param.try_into_parts() {
                Ok(mut parts) => {
                    let grad = parts.grad.take().ok_or(TensorError::OptimizerState)?;
                    apply(c, &mut parts.data, &grad, state);
                    drop(grad);
                    updated.push(Tensor::new(parts.data, &parts.shape)?);
                }
                Err(param) => {
                    let mut data = param.data().to_vec();
                    let guard = param.grad()?;
                    let grad = guard.as_deref().ok_or(TensorError::OptimizerState)?;
                    apply(c, &mut data, grad, state);
                    updated.push(Tensor::new(data, param.shape())?);
                }
            }
        }
        Ok(updated)
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
}
