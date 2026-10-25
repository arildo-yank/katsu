//! Otimização e loop de treino mínimo.
//!
//! Os dados de um Tensor são imutáveis (o backward depende disso). Por isso o
//! SGD não altera parâmetros no lugar: cada passo devolve novas folhas com
//! `p - lr * grad`. Grafos anteriores continuam válidos e não há gradiente
//! acumulado entre passos, então zero_grad não é necessário.
use std::fmt;

use rayon::prelude::*;

use crate::{tensor::zeros_buffer, Tensor, TensorError};

type Result<T> = std::result::Result<T, TensorError>;

/// Otimizador que transforma parâmetros com gradiente em novas folhas.
pub trait Optimizer {
    fn step(&mut self, params: &[Tensor]) -> Result<Vec<Tensor>>;
    /// Como `step`, mas recebe os parâmetros por valor: o otimizador pode reaproveitar o
    /// buffer de quem não tem outro dono (sem grafo vivo apontando para ele) em vez de
    /// alocar um conjunto novo de pesos. Em erro, os parâmetros recebidos são descartados.
    fn step_owned(&mut self, params: Vec<Tensor>) -> Result<Vec<Tensor>> {
        self.step(&params)
    }
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
    /// Change LR without discarding Adam moments or its bias-correction step.
    pub fn set_learning_rate(&mut self, learning_rate: f32) -> Result<()> {
        validate_learning_rate(learning_rate)?;
        self.learning_rate = learning_rate;
        Ok(())
    }

    pub fn save_state(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_all(b"KATADAM1")?;
        for value in [self.learning_rate, self.beta1, self.beta2, self.eps] {
            w.write_all(&value.to_le_bytes())?;
        }
        w.write_all(&self.step.to_le_bytes())?;
        w.write_all(&(self.moments.len() as u64).to_le_bytes())?;
        for (m, v) in &self.moments {
            w.write_all(&(m.len() as u64).to_le_bytes())?;
            for values in [m, v] {
                for value in values {
                    w.write_all(&value.to_le_bytes())?;
                }
            }
        }
        Ok(())
    }

    pub fn load_state(r: &mut impl std::io::Read, params: &[Tensor]) -> std::io::Result<Self> {
        use std::io::{Error, ErrorKind};
        let bad = || Error::new(ErrorKind::InvalidData, "invalid Adam checkpoint");
        let mut magic = [0; 8];
        r.read_exact(&mut magic)?;
        if &magic != b"KATADAM1" {
            return Err(bad());
        }
        let mut values = [0.0; 4];
        for value in &mut values {
            let mut b = [0; 4];
            r.read_exact(&mut b)?;
            *value = f32::from_le_bytes(b);
        }
        let mut opt =
            Self::with_betas(values[0], values[1], values[2], values[3]).map_err(|_| bad())?;
        let mut b = [0; 4];
        r.read_exact(&mut b)?;
        opt.step = i32::from_le_bytes(b);
        let mut b = [0; 8];
        r.read_exact(&mut b)?;
        let count = u64::from_le_bytes(b);
        if opt.step < 0 || (count != params.len() as u64 && !(count == 0 && opt.step == 0)) {
            return Err(bad());
        }
        for p in params.iter().take(count as usize) {
            r.read_exact(&mut b)?;
            if u64::from_le_bytes(b) != p.numel() as u64 {
                return Err(bad());
            }
            let mut pair = (vec![0.0; p.numel()], vec![0.0; p.numel()]);
            for (index, values) in [&mut pair.0, &mut pair.1].into_iter().enumerate() {
                for value in values {
                    let mut b = [0; 4];
                    r.read_exact(&mut b)?;
                    *value = f32::from_le_bytes(b);
                    if !value.is_finite() || (index == 1 && *value < 0.0) {
                        return Err(bad());
                    }
                }
            }
            opt.moments.push(pair);
        }
        Ok(opt)
    }
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

/// Constantes de um passo do Adam, já com a correção de viés. Públicas para que um
/// `ParamStore` faça exatamente a mesma conta no dispositivo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamStep {
    pub beta1: f32,
    pub beta2: f32,
    pub lr: f32,
    pub eps: f32,
    pub correction1: f32,
    pub correction2: f32,
}

impl AdamStep {
    /// Atualiza momentos e escreve o peso novo em `out`. Sem `source`, o peso atual é o
    /// próprio `out` (atualização no lugar). Cada elemento é independente: dividir em
    /// blocos entre threads não muda o resultado, e os dois caminhos fazem a mesma conta.
    pub(crate) fn apply(
        self,
        out: &mut [f32],
        source: Option<&[f32]>,
        grad: &[f32],
        m: &mut [f32],
        v: &mut [f32],
    ) {
        const CHUNK: usize = 1 << 14;
        let update = |out: &mut [f32],
                      values: Option<&[f32]>,
                      grad: &[f32],
                      m: &mut [f32],
                      v: &mut [f32]| {
            for (i, (((out, &g), m), v)) in out.iter_mut().zip(grad).zip(m).zip(v).enumerate() {
                let value = values.map_or(*out, |values| values[i]);
                *m = self.beta1 * *m + (1.0 - self.beta1) * g;
                *v = self.beta2 * *v + (1.0 - self.beta2) * g * g;
                let m_hat = *m / self.correction1;
                let v_hat = *v / self.correction2;
                *out = value - self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        };
        let chunks = out
            .par_chunks_mut(CHUNK)
            .zip(grad.par_chunks(CHUNK))
            .zip(m.par_chunks_mut(CHUNK).zip(v.par_chunks_mut(CHUNK)))
            .enumerate();
        chunks.for_each(|(i, ((out, grad), (m, v)))| {
            let values = source.map(|s| &s[i * CHUNK..i * CHUNK + out.len()]);
            update(out, values, grad, m, v);
        });
    }
}

impl Adam {
    /// Cria os momentos no primeiro passo ou confere que batem com os parâmetros. Os de um
    /// parâmetro residente ficam no store: aqui, vazios.
    fn prepare(&mut self, params: &[Tensor]) -> Result<()> {
        let size = |p: &Tensor| if p.is_resident() { 0 } else { p.numel() };
        if self.moments.is_empty() {
            for param in params {
                self.moments
                    .push((vec![0.0; size(param)], vec![0.0; size(param)]));
            }
        } else if self.moments.len() != params.len()
            || self
                .moments
                .iter()
                .zip(params)
                .any(|(m, p)| m.0.len() != size(p))
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

impl Optimizer for Adam {
    /// Valida tudo antes de alterar o estado: um erro não consome um passo.
    fn step(&mut self, params: &[Tensor]) -> Result<Vec<Tensor>> {
        // Lê os gradientes sem copiar; os guards ficam abertos só durante o passo.
        let mut guards = Vec::new();
        guards.try_reserve_exact(params.len())?;
        for param in params {
            guards.push(param.grad()?);
        }
        let mut grads = Vec::new();
        grads.try_reserve_exact(params.len())?;
        for (index, guard) in guards.iter().enumerate() {
            grads.push(
                guard
                    .as_deref()
                    .ok_or(TensorError::MissingGradient { index })?,
            );
        }
        self.prepare(params)?;
        let mut buffers = Vec::new();
        buffers.try_reserve_exact(params.len())?;
        for param in params {
            buffers.push(zeros_buffer(param.numel())?);
        }
        let constants = self.advance();
        for (((param, grad), (m, v)), data) in params
            .iter()
            .zip(&grads)
            .zip(&mut self.moments)
            .zip(&mut buffers)
        {
            constants.apply(data, Some(param.data()), grad, m, v);
        }
        drop(grads);
        drop(guards);
        let mut updated = Vec::new();
        updated.try_reserve_exact(params.len())?;
        for (param, data) in params.iter().zip(buffers) {
            updated.push(Tensor::new(data, param.shape())?);
        }
        Ok(updated)
    }

    /// Pesos sem outro dono são atualizados no próprio buffer e perdem o gradiente na
    /// hora; só os compartilhados ganham buffer novo. Valida e aloca antes de alterar
    /// os momentos, como `step`.
    fn step_owned(&mut self, params: Vec<Tensor>) -> Result<Vec<Tensor>> {
        for (index, param) in params.iter().enumerate() {
            if param.grad()?.is_none() {
                return Err(TensorError::MissingGradient { index });
            }
        }
        self.prepare(&params)?;
        let constants = self.advance();
        self.update_owned(params, constants)
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
}

impl Adam {
    /// Corpo do `step_owned`, com as constantes do passo já calculadas. Parâmetros residentes
    /// passam intactos: o store os atualiza.
    fn update_owned(&mut self, params: Vec<Tensor>, constants: AdamStep) -> Result<Vec<Tensor>> {
        let mut spare = Vec::new();
        spare.try_reserve_exact(params.len())?;
        for param in &params {
            spare.push(if param.is_unique() {
                None
            } else {
                Some(zeros_buffer(param.numel())?)
            });
        }
        let mut updated = Vec::new();
        updated.try_reserve_exact(params.len())?;
        for ((param, (m, v)), spare) in params.into_iter().zip(&mut self.moments).zip(spare) {
            if param.is_resident() {
                updated.push(param);
                continue;
            }
            match param.try_into_parts() {
                Ok(mut parts) => {
                    let grad = parts.grad.take().ok_or(TensorError::OptimizerState)?;
                    constants.apply(&mut parts.data, None, &grad, m, v);
                    drop(grad);
                    updated.push(Tensor::new(parts.data, &parts.shape)?);
                }
                Err(param) => {
                    let mut data = match spare {
                        Some(data) => data,
                        None => zeros_buffer(param.numel())?,
                    };
                    let guard = param.grad()?;
                    let grad = guard.as_deref().ok_or(TensorError::OptimizerState)?;
                    constants.apply(&mut data, Some(param.data()), grad, m, v);
                    updated.push(Tensor::new(data, param.shape())?);
                }
            }
        }
        Ok(updated)
    }

    /// Como `step_owned`, mas os parâmetros residentes são atualizados no próprio store, num
    /// passo sobre a arena inteira, com as mesmas constantes dos demais. Devolve os parâmetros
    /// e quantos pesos residentes ficaram não finitos.
    pub fn step_resident(
        &mut self,
        params: Vec<Tensor>,
        store: &dyn crate::ParamStore,
    ) -> Result<(Vec<Tensor>, usize)> {
        for (index, param) in params.iter().enumerate() {
            match param.residency() {
                Some((owner, _)) if !same_store(owner.as_ref(), store) => {
                    return Err(TensorError::InvalidShape(
                        "resident parameter belongs to another store",
                    ))
                }
                Some(_) => {}
                None if param.grad()?.is_none() => {
                    return Err(TensorError::MissingGradient { index })
                }
                None => {}
            }
        }
        if self.moments.is_empty() {
            // Adam novo: os momentos de um treino anterior no mesmo store não valem mais.
            for param in &params {
                if let Some((_, view)) = param.residency() {
                    let zeros = zeros_buffer(view.len)?;
                    store.write(crate::Buffer::M, view, &zeros)?;
                    store.write(crate::Buffer::V, view, &zeros)?;
                }
            }
        }
        self.prepare(&params)?;
        let constants = self.advance();
        let nonfinite = store.adam_step(constants)?;
        Ok((self.update_owned(params, constants)?, nonfinite))
    }

    /// Leva para o store os momentos dos parâmetros residentes (ex.: ao retomar um checkpoint
    /// carregado no host) e os solta da memória do host.
    pub fn offload_moments(&mut self, params: &[Tensor]) -> Result<()> {
        if self.moments.is_empty() {
            return Ok(());
        }
        if self.moments.len() != params.len() {
            return Err(TensorError::OptimizerState);
        }
        for (param, (m, v)) in params.iter().zip(&mut self.moments) {
            let Some((store, view)) = param.residency() else {
                continue;
            };
            if m.is_empty() {
                continue;
            }
            if m.len() != view.len {
                return Err(TensorError::OptimizerState);
            }
            store.write(crate::Buffer::M, view, m)?;
            store.write(crate::Buffer::V, view, v)?;
            *m = Vec::new();
            *v = Vec::new();
        }
        Ok(())
    }

    /// Mesmo formato do `save_state`, lendo do store os momentos dos parâmetros residentes:
    /// o checkpoint não sabe onde o treino rodou.
    pub fn save_state_resident(
        &self,
        w: &mut impl std::io::Write,
        params: &[Tensor],
    ) -> std::io::Result<()> {
        use std::io::{Error, ErrorKind};
        if !self.moments.is_empty() && self.moments.len() != params.len() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "params do not match Adam",
            ));
        }
        w.write_all(b"KATADAM1")?;
        for value in [self.learning_rate, self.beta1, self.beta2, self.eps] {
            w.write_all(&value.to_le_bytes())?;
        }
        w.write_all(&self.step.to_le_bytes())?;
        w.write_all(&(self.moments.len() as u64).to_le_bytes())?;
        for (param, (m, v)) in params.iter().zip(&self.moments) {
            let read;
            let (m, v) = match param.residency() {
                Some((store, view)) => {
                    let load = |b| store.read(b, view).map_err(|e| Error::other(e.to_string()));
                    read = (load(crate::Buffer::M)?, load(crate::Buffer::V)?);
                    (&read.0, &read.1)
                }
                None => (m, v),
            };
            w.write_all(&(m.len() as u64).to_le_bytes())?;
            for values in [m, v] {
                for value in values {
                    w.write_all(&value.to_le_bytes())?;
                }
            }
        }
        Ok(())
    }
}

/// Mesmo objeto (compara só o endereço, não a vtable).
fn same_store(a: &dyn crate::ParamStore, b: &dyn crate::ParamStore) -> bool {
    std::ptr::eq(
        a as *const dyn crate::ParamStore as *const (),
        b as *const dyn crate::ParamStore as *const (),
    )
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
