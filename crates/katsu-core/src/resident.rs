//! Parâmetros residentes: pesos, gradientes e momentos do Adam ficam num store (a VRAM, no
//! CUDA) durante todo o treino; por passo só as ativações cruzam a fronteira host/dispositivo.
//!
//! Os parâmetros residentes ocupam uma arena contígua por buffer (pesos, gradiente, m, v), cada
//! um numa faixa `View`. Faixas vizinhas viram uma só sem cópia: é assim que as projeções de
//! saída das cabeças da atenção (`Tensor::concat`) continuam no store. O Adam roda na arena
//! inteira de uma vez. `HostParamStore` é a referência: faz as mesmas contas do caminho CPU,
//! bit a bit, e conta os bytes que cruzariam a fronteira num dispositivo de verdade.
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

use crate::{optim::AdamStep, tensor::zeros_buffer, Backend, Tensor, TensorError};

type Result<T> = std::result::Result<T, TensorError>;

/// Faixa `[offset, offset + len)` da arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct View {
    pub offset: usize,
    pub len: usize,
}

impl View {
    pub(crate) fn range(self) -> std::ops::Range<usize> {
        self.offset..self.offset + self.len
    }
}

/// Qual buffer da arena ler ou escrever.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Buffer {
    Weights,
    Grad,
    M,
    V,
}

/// Bytes enviados ao dispositivo e trazidos de volta desde a criação do store.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Transfers {
    pub to_device: u64,
    pub to_host: u64,
}

pub trait ParamStore: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &'static str;
    /// Tamanho da arena, em números.
    fn capacity(&self) -> usize;
    /// `a [m,k]` (host) · `arena[rhs] [k,n]` → `[m,n]` (host).
    fn matmul(&self, a: &[f32], rhs: View, m: usize, k: usize, n: usize) -> Result<Vec<f32>>;
    /// Devolve `dA = g·Bᵀ` e soma `Aᵀ·g` no gradiente residente de `rhs`.
    fn matmul_backward(
        &self,
        a: &[f32],
        rhs: View,
        g: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>>;
    /// Vários pesos `[k,n]` com a mesma entrada, que sobe uma vez só: saída `[m, R·n]`, as
    /// colunas de cada peso lado a lado (as projeções Q, K e V de todas as cabeças).
    fn matmul_many(
        &self,
        a: &[f32],
        rhs: &[View],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>>;
    /// Backward do `matmul_many`: `g` é `[m, R·n]`. Devolve `dA = Σᵢ gᵢ·Bᵢᵀ`, somado na ordem
    /// dos pesos, e soma `Aᵀ·gᵢ` no gradiente residente de cada um.
    fn matmul_many_backward(
        &self,
        a: &[f32],
        rhs: &[View],
        g: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>>;
    /// Soma dos quadrados do gradiente (em f64) e quantos valores não finitos ele tem.
    fn grad_stats(&self) -> Result<(f64, usize)>;
    /// Um passo do Adam na arena inteira; zera o gradiente e devolve quantos pesos ficaram
    /// não finitos.
    fn adam_step(&self, step: AdamStep) -> Result<usize>;
    fn read(&self, buffer: Buffer, view: View) -> Result<Vec<f32>>;
    fn write(&self, buffer: Buffer, view: View, data: &[f32]) -> Result<()>;
    fn transfers(&self) -> Transfers;
}

pub(crate) fn check_view(capacity: usize, view: View) -> Result<()> {
    match view.offset.checked_add(view.len) {
        Some(end) if end <= capacity => Ok(()),
        _ => Err(TensorError::InvalidShape("view outside the resident arena")),
    }
}

pub(crate) fn check_matmul(
    a: &[f32],
    rhs: View,
    m: usize,
    k: usize,
    n: usize,
    capacity: usize,
) -> Result<()> {
    check_view(capacity, rhs)?;
    crate::backend::check(a, &[m, k])?;
    if Some(rhs.len) != k.checked_mul(n) {
        return Err(TensorError::InvalidShape(
            "resident rhs does not match [k,n]",
        ));
    }
    Ok(())
}

/// Colunas `[i·n, (i+1)·n)` de uma matriz `[m, R·n]`, como matriz `[m, n]` contígua.
pub(crate) fn columns_of(g: &[f32], m: usize, n: usize, count: usize, i: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(m * n);
    for row in 0..m {
        let start = row * count * n + i * n;
        out.extend_from_slice(&g[start..start + n]);
    }
    out
}

/// Escreve `part [m, n]` nas colunas `[i·n, (i+1)·n)` de `out [m, R·n]`.
pub(crate) fn put_columns(
    out: &mut [f32],
    part: &[f32],
    m: usize,
    n: usize,
    count: usize,
    i: usize,
) {
    for row in 0..m {
        let start = row * count * n + i * n;
        out[start..start + n].copy_from_slice(&part[row * n..(row + 1) * n]);
    }
}

#[derive(Debug)]
struct Arena {
    weights: Vec<f32>,
    grad: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl Arena {
    fn buffer(&mut self, buffer: Buffer) -> &mut Vec<f32> {
        match buffer {
            Buffer::Weights => &mut self.weights,
            Buffer::Grad => &mut self.grad,
            Buffer::M => &mut self.m,
            Buffer::V => &mut self.v,
        }
    }
}

/// Store de referência em memória do host: as contas do `Backend` dado, sem cópia real, com
/// a contagem dos bytes que um dispositivo trocaria. Serve para testar o caminho residente
/// contra o caminho CPU sem GPU.
#[derive(Debug)]
pub struct HostParamStore {
    backend: Arc<dyn Backend>,
    arena: Mutex<Arena>,
    capacity: usize,
    to_device: AtomicU64,
    to_host: AtomicU64,
}

impl HostParamStore {
    pub fn new(capacity: usize, backend: Arc<dyn Backend>) -> Result<Self> {
        Ok(Self {
            backend,
            arena: Mutex::new(Arena {
                weights: zeros_buffer(capacity)?,
                grad: zeros_buffer(capacity)?,
                m: zeros_buffer(capacity)?,
                v: zeros_buffer(capacity)?,
            }),
            capacity,
            to_device: AtomicU64::new(0),
            to_host: AtomicU64::new(0),
        })
    }

    fn arena(&self) -> Result<std::sync::MutexGuard<'_, Arena>> {
        self.arena.lock().map_err(|_| TensorError::LockPoisoned)
    }

    fn sent(&self, floats: usize) {
        self.to_device
            .fetch_add(4 * floats as u64, Ordering::Relaxed);
    }

    fn received(&self, bytes: usize) {
        self.to_host.fetch_add(bytes as u64, Ordering::Relaxed);
    }
}

impl ParamStore for HostParamStore {
    fn name(&self) -> &'static str {
        "host-resident"
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn matmul(&self, a: &[f32], rhs: View, m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        check_matmul(a, rhs, m, k, n, self.capacity)?;
        let arena = self.arena()?;
        self.sent(a.len());
        let out = self
            .backend
            .matmul(a, &arena.weights[rhs.range()], m, k, n)?;
        self.received(4 * out.len());
        Ok(out)
    }

    fn matmul_backward(
        &self,
        a: &[f32],
        rhs: View,
        g: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        check_matmul(a, rhs, m, k, n, self.capacity)?;
        crate::backend::check(g, &[m, n])?;
        let mut arena = self.arena()?;
        self.sent(a.len() + g.len());
        let (da, db) = self
            .backend
            .matmul_backward(a, &arena.weights[rhs.range()], g, m, k, n)?;
        if da.len() != m * k || db.len() != rhs.len {
            return Err(TensorError::InvalidShape(
                "backend returned wrong gradient size",
            ));
        }
        // Primeira contribuição: 0 + x = x, o mesmo número que o adjunto do caminho CPU.
        for (out, value) in arena.grad[rhs.range()].iter_mut().zip(db) {
            *out += value;
        }
        self.received(4 * da.len());
        Ok(da)
    }

    fn matmul_many(
        &self,
        a: &[f32],
        rhs: &[View],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        for &view in rhs {
            check_matmul(a, view, m, k, n, self.capacity)?;
        }
        let arena = self.arena()?;
        self.sent(a.len());
        let mut out = zeros_buffer(m * n * rhs.len())?;
        for (i, view) in rhs.iter().enumerate() {
            let part = self
                .backend
                .matmul(a, &arena.weights[view.range()], m, k, n)?;
            put_columns(&mut out, &part, m, n, rhs.len(), i);
        }
        self.received(4 * out.len());
        Ok(out)
    }

    fn matmul_many_backward(
        &self,
        a: &[f32],
        rhs: &[View],
        g: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        for &view in rhs {
            check_matmul(a, view, m, k, n, self.capacity)?;
        }
        crate::backend::check(g, &[m, n * rhs.len()])?;
        let mut arena = self.arena()?;
        self.sent(a.len() + g.len());
        let mut da = zeros_buffer(m * k)?;
        for (i, view) in rhs.iter().enumerate() {
            let gi = columns_of(g, m, n, rhs.len(), i);
            let (dai, db) =
                self.backend
                    .matmul_backward(a, &arena.weights[view.range()], &gi, m, k, n)?;
            for (out, value) in arena.grad[view.range()].iter_mut().zip(db) {
                *out += value;
            }
            for (out, value) in da.iter_mut().zip(dai) {
                *out += value;
            }
        }
        self.received(4 * da.len());
        Ok(da)
    }

    fn grad_stats(&self) -> Result<(f64, usize)> {
        let arena = self.arena()?;
        let (mut sum, mut nonfinite) = (0.0, 0);
        for &g in &arena.grad {
            if g.is_finite() {
                sum += f64::from(g).powi(2);
            } else {
                nonfinite += 1;
            }
        }
        self.received(16);
        Ok((sum, nonfinite))
    }

    fn adam_step(&self, step: AdamStep) -> Result<usize> {
        let mut arena = self.arena()?;
        let Arena {
            weights,
            grad,
            m,
            v,
        } = &mut *arena;
        // Mesma atualização no lugar do `Adam::step_owned`, elemento a elemento.
        step.apply(weights, None, grad, m, v);
        grad.fill(0.0);
        self.received(4);
        Ok(weights.iter().filter(|w| !w.is_finite()).count())
    }

    fn read(&self, buffer: Buffer, view: View) -> Result<Vec<f32>> {
        check_view(self.capacity, view)?;
        let mut arena = self.arena()?;
        self.received(4 * view.len);
        Ok(arena.buffer(buffer)[view.range()].to_vec())
    }

    fn write(&self, buffer: Buffer, view: View, data: &[f32]) -> Result<()> {
        check_view(self.capacity, view)?;
        if data.len() != view.len {
            return Err(TensorError::DataLength {
                expected: view.len,
                actual: data.len(),
            });
        }
        let mut arena = self.arena()?;
        self.sent(data.len());
        arena.buffer(buffer)[view.range()].copy_from_slice(data);
        Ok(())
    }

    fn transfers(&self) -> Transfers {
        Transfers {
            to_device: self.to_device.load(Ordering::Relaxed),
            to_host: self.to_host.load(Ordering::Relaxed),
        }
    }
}

/// Números que os parâmetros de `layout` ocupam na arena.
pub fn arena_len(params: &[Tensor], layout: &[usize]) -> Result<usize> {
    layout.iter().try_fold(0usize, |total, &i| {
        let param = params.get(i).ok_or(TensorError::InvalidShape(
            "resident layout index out of range",
        ))?;
        total
            .checked_add(param.numel())
            .ok_or(TensorError::ShapeOverflow)
    })
}

/// Sobe os parâmetros de `layout` para o store, na ordem dada (vizinhos na arena), e os troca
/// por tensores residentes. Os demais continuam no host.
pub fn upload(
    mut params: Vec<Tensor>,
    layout: &[usize],
    store: Arc<dyn ParamStore>,
) -> Result<Vec<Tensor>> {
    if arena_len(&params, layout)? > store.capacity() {
        return Err(TensorError::InvalidShape(
            "resident layout exceeds the arena",
        ));
    }
    let mut seen = vec![false; params.len()];
    let mut offset = 0;
    for &i in layout {
        if std::mem::replace(&mut seen[i], true) || params[i].is_resident() {
            return Err(TensorError::InvalidShape("parameter uploaded twice"));
        }
        let view = View {
            offset,
            len: params[i].numel(),
        };
        store.write(Buffer::Weights, view, params[i].data())?;
        params[i] = Tensor::resident(store.clone(), view, params[i].shape())?;
        offset += view.len;
    }
    Ok(params)
}

/// Cópia no host de todos os parâmetros (para checkpoint ou inferência); os residentes são
/// lidos do store.
pub fn materialize(params: &[Tensor]) -> Result<Vec<Tensor>> {
    params
        .iter()
        .map(|p| match p.residency() {
            Some((store, view)) => Tensor::new(store.read(Buffer::Weights, view)?, p.shape()),
            None => Ok(p.clone()),
        })
        .collect()
}
