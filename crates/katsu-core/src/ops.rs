//! Operações de transformer: broadcasting, forma, ativações e normalização.
//! Cada forward registra seu Op; as derivadas locais ficam em `unary_backward`.
use crate::{
    tensor::{element_count, zeros_buffer},
    Op, Tensor, TensorError,
};

type Result<T> = std::result::Result<T, TensorError>;

/// Épsilon da LayerNorm, como no PyTorch.
pub const LAYER_NORM_EPS: f32 = 1e-5;
const GELU_C: f32 = 0.797_884_6; // sqrt(2/pi)
const GELU_K: f32 = 0.044_715;

/// Shape resultante pelas regras do NumPy: alinha à direita; dimensão 1 expande.
pub(crate) fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut shape = Vec::new();
    shape.try_reserve_exact(rank)?;
    for k in 0..rank {
        let da = (k + a.len()).checked_sub(rank).map_or(1, |i| a[i]);
        let db = (k + b.len()).checked_sub(rank).map_or(1, |i| b[i]);
        shape.push(match (da, db) {
            _ if da == db => da,
            (1, d) | (d, 1) => d,
            _ => return Err(TensorError::ShapeMismatch),
        });
    }
    Ok(shape)
}

/// Índice de entrada para cada elemento da saída; None quando os shapes são iguais.
pub(crate) fn broadcast_map(out: &[usize], input: &[usize]) -> Result<Option<Vec<usize>>> {
    if out == input {
        return Ok(None);
    }
    let offset = out.len() - input.len();
    // Stride 0 nas dimensões ausentes ou expandidas.
    let mut strides = vec![0; out.len()];
    let mut stride = 1;
    for k in (0..input.len()).rev() {
        if input[k] != 1 {
            strides[offset + k] = stride;
        }
        stride *= input[k];
    }
    let count = element_count(out)?;
    let mut map = Vec::new();
    map.try_reserve_exact(count)?;
    let mut index = vec![0; out.len()];
    let mut source = 0;
    for _ in 0..count {
        map.push(source);
        for k in (0..out.len()).rev() {
            index[k] += 1;
            source += strides[k];
            if index[k] < out[k] {
                break;
            }
            source -= strides[k] * index[k];
            index[k] = 0;
        }
    }
    Ok(Some(map))
}

pub(crate) fn at(map: &Option<Vec<usize>>, i: usize) -> usize {
    map.as_ref().map_or(i, |m| m[i])
}

/// Linhas da última dimensão: (quantidade de linhas, largura).
fn rows(tensor: &Tensor) -> Result<(usize, usize)> {
    let width = *tensor
        .shape()
        .last()
        .ok_or(TensorError::InvalidShape("operation needs rank >= 1"))?;
    Ok((
        if width == 0 {
            0
        } else {
            tensor.numel() / width
        },
        width,
    ))
}

impl Tensor {
    /// Escalar de shape `[]`, útil para escalar perdas via broadcasting.
    pub fn scalar(value: f32) -> Result<Self> {
        Self::new(vec![value], &[])
    }

    /// Matriz `[n, classes]` com 1 na coluna de cada índice (folha constante).
    pub fn one_hot(indices: &[usize], classes: usize) -> Result<Self> {
        let mut data = zeros_buffer(element_count(&[indices.len(), classes])?)?;
        for (row, &index) in indices.iter().enumerate() {
            if index >= classes {
                return Err(TensorError::IndexOutOfRange { index, classes });
            }
            data[row * classes + index] = 1.0;
        }
        Self::new(data, &[indices.len(), classes])
    }

    /// Máscara causal `[t, t]`: 0 na diagonal e abaixo, -inf acima.
    /// Some-a aos scores de atenção antes do softmax.
    pub fn causal_mask(size: usize) -> Result<Self> {
        let mut data = zeros_buffer(element_count(&[size, size])?)?;
        for i in 0..size {
            for value in &mut data[i * size + i + 1..(i + 1) * size] {
                *value = f32::NEG_INFINITY;
            }
        }
        Self::new(data, &[size, size])
    }

    /// Transposta de uma matriz `[m, n] -> [n, m]` (copia os dados).
    pub fn transpose(&self) -> Result<Self> {
        let &[m, n] = self.shape() else {
            return Err(TensorError::InvalidShape("transpose needs rank 2"));
        };
        let mut data = zeros_buffer(self.numel())?;
        for i in 0..m {
            for j in 0..n {
                data[j * m + i] = self.data()[i * n + j];
            }
        }
        Ok(Self::node(
            data,
            vec![n, m],
            Op::Transpose,
            vec![self.clone()],
        ))
    }

    /// Mesmos dados com outro shape; a quantidade de elementos precisa coincidir.
    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        let expected = element_count(shape)?;
        if expected != self.numel() {
            return Err(TensorError::DataLength {
                expected,
                actual: self.numel(),
            });
        }
        let mut data = zeros_buffer(self.numel())?;
        data.copy_from_slice(self.data());
        Ok(Self::node(
            data,
            shape.to_vec(),
            Op::Reshape,
            vec![self.clone()],
        ))
    }

    /// GELU com aproximação tanh (GPT-2/BERT).
    pub fn gelu(&self) -> Result<Self> {
        let mut data = zeros_buffer(self.numel())?;
        for (out, &x) in data.iter_mut().zip(self.data()) {
            *out = 0.5 * x * (1.0 + (GELU_C * (x + GELU_K * x * x * x)).tanh());
        }
        Ok(self.unary(data, Op::Gelu))
    }

    /// Softmax na última dimensão, estável (subtrai o máximo da linha).
    /// Entradas -inf viram probabilidade 0; uma linha toda -inf vira NaN.
    pub fn softmax(&self) -> Result<Self> {
        let mut data = self.row_log_softmax()?;
        for value in &mut data {
            *value = value.exp();
        }
        Ok(self.unary(data, Op::Softmax))
    }

    /// log(softmax) na última dimensão, sem calcular o log de probabilidades pequenas.
    pub fn log_softmax(&self) -> Result<Self> {
        let data = self.row_log_softmax()?;
        Ok(self.unary(data, Op::LogSoftmax))
    }

    /// Normaliza cada linha da última dimensão para média 0 e variância 1.
    /// Sem escala/bias: aplique-os com mul/add via broadcasting.
    pub fn layer_norm(&self) -> Result<Self> {
        let (count, width) = rows(self)?;
        let mut data = zeros_buffer(self.numel())?;
        for r in 0..count {
            let x = &self.data()[r * width..(r + 1) * width];
            let (mean, inv_std) = row_stats(x);
            for (out, &value) in data[r * width..(r + 1) * width].iter_mut().zip(x) {
                *out = (value - mean) * inv_std;
            }
        }
        Ok(self.unary(data, Op::LayerNorm))
    }

    /// Entropia cruzada média de logits `[n, classes]` contra índices de classe.
    pub fn cross_entropy(&self, targets: &[usize]) -> Result<Self> {
        let &[n, classes] = self.shape() else {
            return Err(TensorError::InvalidShape(
                "cross_entropy needs [n, classes] logits",
            ));
        };
        if targets.len() != n {
            return Err(TensorError::DataLength {
                expected: n,
                actual: targets.len(),
            });
        }
        let picked = self
            .log_softmax()?
            .mul(&Self::one_hot(targets, classes)?)?
            .sum()?;
        picked.mul(&Self::scalar(-1.0 / n.max(1) as f32)?)
    }

    fn unary(&self, data: Vec<f32>, op: Op) -> Self {
        Self::node(data, self.shape().to_vec(), op, vec![self.clone()])
    }

    fn row_log_softmax(&self) -> Result<Vec<f32>> {
        let (count, width) = rows(self)?;
        let mut data = zeros_buffer(self.numel())?;
        for r in 0..count {
            let x = &self.data()[r * width..(r + 1) * width];
            let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            // Subtrai o máximo antes do log: somar max ao ln perderia precisão em f32.
            let ln_sum = x.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
            for (out, &value) in data[r * width..(r + 1) * width].iter_mut().zip(x) {
                *out = (value - max) - ln_sum;
            }
        }
        Ok(data)
    }
}

fn row_stats(x: &[f32]) -> (f32, f32) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    (mean, 1.0 / (var + LAYER_NORM_EPS).sqrt())
}

/// VJP das operações de um pai. `target` é o adjunto do pai; `g`, o da saída.
pub(crate) fn unary_backward(
    op: Op,
    parent: &Tensor,
    node: &Tensor,
    g: &[f32],
    target: &mut [f32],
) {
    match op {
        Op::Reshape => {
            for (out, grad) in target.iter_mut().zip(g) {
                *out += grad;
            }
        }
        Op::Transpose => {
            let (m, n) = (parent.shape()[0], parent.shape()[1]);
            for i in 0..m {
                for j in 0..n {
                    target[i * n + j] += g[j * m + i];
                }
            }
        }
        Op::Gelu => {
            for ((out, grad), &x) in target.iter_mut().zip(g).zip(parent.data()) {
                let t = (GELU_C * (x + GELU_K * x * x * x)).tanh();
                let du = GELU_C * (1.0 + 3.0 * GELU_K * x * x);
                *out += grad * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du);
            }
        }
        Op::Softmax | Op::LogSoftmax | Op::LayerNorm => {
            let width = *node.shape().last().expect("row ops have rank >= 1");
            if width == 0 {
                return;
            }
            for r in 0..node.numel() / width {
                let span = r * width..(r + 1) * width;
                let (y, gr, out) = (
                    &node.data()[span.clone()],
                    &g[span.clone()],
                    &mut target[span.clone()],
                );
                match op {
                    // dx = y * (g - Σ g·y)
                    Op::Softmax => {
                        let dot: f32 = gr.iter().zip(y).map(|(a, b)| a * b).sum();
                        for ((o, &gi), &yi) in out.iter_mut().zip(gr).zip(y) {
                            *o += yi * (gi - dot);
                        }
                    }
                    // dx = g - softmax · Σ g
                    Op::LogSoftmax => {
                        let total: f32 = gr.iter().sum();
                        for ((o, &gi), &yi) in out.iter_mut().zip(gr).zip(y) {
                            *o += gi - yi.exp() * total;
                        }
                    }
                    // dx = inv_std · (g - média(g) - x̂ · média(g·x̂)), com y = x̂
                    _ => {
                        let (_, inv_std) = row_stats(&parent.data()[span]);
                        let n = width as f32;
                        let mean_g = gr.iter().sum::<f32>() / n;
                        let mean_gy = gr.iter().zip(y).map(|(a, b)| a * b).sum::<f32>() / n;
                        for ((o, &gi), &yi) in out.iter_mut().zip(gr).zip(y) {
                            *o += inv_std * (gi - mean_g - yi * mean_gy);
                        }
                    }
                }
            }
        }
        _ => unreachable!("not a unary op: {op:?}"),
    }
}
