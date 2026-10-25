//! Operações de transformer: broadcasting, forma, ativações e normalização.
//! Cada forward registra seu Op; as derivadas locais ficam em `unary_backward`.
use rayon::prelude::*;

use crate::{
    tensor::{element_count, zeros_buffer},
    Op, Tensor, TensorError,
};

type Result<T> = std::result::Result<T, TensorError>;

/// Épsilon da LayerNorm, como no PyTorch.
pub const LAYER_NORM_EPS: f32 = 1e-5;
pub(crate) const GELU_C: f32 = 0.797_884_6; // sqrt(2/pi)
pub(crate) const GELU_K: f32 = 0.044_715;

/// Cópia transposta de uma matriz `[rows, cols]` contígua.
pub(crate) fn transposed(data: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
    let mut out = zeros_buffer(rows * cols)?;
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = data[i * cols + j];
        }
    }
    Ok(out)
}

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

    /// Transpõe as duas últimas dimensões de uma matriz ou batch (copia os dados).
    pub fn transpose(&self) -> Result<Self> {
        if self.rank() < 2 {
            return Err(TensorError::InvalidShape("transpose needs rank >= 2"));
        }
        let rank = self.rank();
        let (m, n) = (self.shape()[rank - 2], self.shape()[rank - 1]);
        let mut shape = self.shape().to_vec();
        shape.swap(rank - 2, rank - 1);
        let mut data = zeros_buffer(self.numel())?;
        if m != 0 && n != 0 {
            for (out, input) in data.chunks_mut(m * n).zip(self.data().chunks(m * n)) {
                for i in 0..m {
                    for j in 0..n {
                        out[j * m + i] = input[i * n + j];
                    }
                }
            }
        }
        Ok(Self::node(data, shape, Op::Transpose, vec![self.clone()]))
    }

    /// `x·W + b` com x `[m, k]`, W `[k, n]` e b `[n]`, num nó só: mesmas contas de
    /// `x.matmul(W).add(b)`, sem guardar o produto antes do bias (que o backward não usa).
    pub fn linear_with_backend(
        &self,
        weight: &Self,
        bias: &Self,
        backend: std::sync::Arc<dyn crate::Backend>,
    ) -> Result<Self> {
        if self.rank() != 2 || weight.rank() != 2 || weight.shape()[0] != self.shape()[1] {
            return Err(TensorError::InvalidShape(
                "linear needs x [m,k] and W [k,n]",
            ));
        }
        let (m, k, n) = (self.shape()[0], self.shape()[1], weight.shape()[1]);
        if bias.shape() != [n] {
            return Err(TensorError::InvalidShape("linear bias must be [n]"));
        }
        if self.is_resident() || bias.is_resident() {
            return Err(TensorError::InvalidShape(
                "resident tensors only as the linear weight",
            ));
        }
        let mut data = match weight.residency() {
            Some((store, view)) => store.matmul(self.data(), view, m, k, n)?,
            None => backend.matmul(self.data(), weight.data(), m, k, n)?,
        };
        if data.len() != m * n {
            return Err(TensorError::InvalidShape(
                "backend returned wrong output size",
            ));
        }
        if n != 0 {
            for row in data.chunks_mut(n) {
                for (value, &b) in row.iter_mut().zip(bias.data()) {
                    *value += b;
                }
            }
        }
        Ok(Self::node(
            data,
            vec![m, n],
            Op::Linear,
            vec![self.clone(), weight.clone(), bias.clone()],
        )
        .with_backend(backend))
    }

    /// `x·Wᵢ` para vários pesos `[k,n]` com a mesma entrada, saídas lado a lado: `[m, R·n]`.
    /// Mesmas contas de R matmuls; num store residente a entrada sobe uma vez só e o
    /// gradiente dela desce uma vez só (as projeções Q, K e V de todas as cabeças).
    pub fn matmul_many_with_backend(
        &self,
        weights: &[Self],
        backend: std::sync::Arc<dyn crate::Backend>,
    ) -> Result<Self> {
        let first = weights
            .first()
            .ok_or(TensorError::InvalidShape("matmul_many needs weights"))?;
        if self.rank() != 2
            || self.is_resident()
            || first.rank() != 2
            || first.shape()[0] != self.shape()[1]
        {
            return Err(TensorError::InvalidShape(
                "matmul_many needs x [m,k] and W [k,n]",
            ));
        }
        if weights.iter().any(|w| w.shape() != first.shape()) {
            return Err(TensorError::ShapeMismatch);
        }
        let (m, k, n) = (self.shape()[0], self.shape()[1], first.shape()[1]);
        let count = weights.len();
        let data = match first.residency() {
            Some((store, _)) => {
                let mut views = Vec::with_capacity(count);
                for w in weights {
                    match w.residency() {
                        Some((owner, view)) if std::sync::Arc::ptr_eq(owner, store) => {
                            views.push(view)
                        }
                        _ => {
                            return Err(TensorError::InvalidShape(
                                "matmul_many weights must all be resident in one store",
                            ))
                        }
                    }
                }
                store.matmul_many(self.data(), &views, m, k, n)?
            }
            None => {
                if weights.iter().any(Self::is_resident) {
                    return Err(TensorError::InvalidShape(
                        "matmul_many weights must all be resident in one store",
                    ));
                }
                let mut out = zeros_buffer(element_count(&[m, n * count])?)?;
                for (i, w) in weights.iter().enumerate() {
                    let part = backend.matmul(self.data(), w.data(), m, k, n)?;
                    if part.len() != m * n {
                        return Err(TensorError::InvalidShape(
                            "backend returned wrong output size",
                        ));
                    }
                    crate::resident::put_columns(&mut out, &part, m, n, count, i);
                }
                out
            }
        };
        if data.len() != m * n * count {
            return Err(TensorError::InvalidShape(
                "store returned wrong output size",
            ));
        }
        let mut parents = Vec::with_capacity(count + 1);
        parents.push(self.clone());
        parents.extend(weights.iter().cloned());
        Ok(Self::node(data, vec![m, n * count], Op::MatmulMany, parents).with_backend(backend))
    }

    /// Colunas `[start, start + width)` de uma matriz, numa matriz nova `[m, width]`.
    pub fn columns(&self, start: usize, width: usize) -> Result<Self> {
        if self.rank() != 2
            || start
                .checked_add(width)
                .is_none_or(|end| end > self.shape()[1])
        {
            return Err(TensorError::InvalidShape("columns out of range"));
        }
        let (rows, total) = (self.shape()[0], self.shape()[1]);
        let mut data = Vec::new();
        data.try_reserve_exact(rows * width)?;
        for row in 0..rows {
            data.extend_from_slice(&self.data()[row * total + start..row * total + start + width]);
        }
        Ok(
            Self::node(data, vec![rows, width], Op::Columns, vec![self.clone()])
                .with_indices(&[start, width]),
        )
    }

    /// `layer_norm(x)·γ + β` por linha, num nó só: mesmas contas de
    /// `x.layer_norm().mul(γ).add(β)`, guardando só a saída. O backward refaz x̂ a partir de x.
    pub fn layer_norm_affine(&self, gamma: &Self, beta: &Self) -> Result<Self> {
        let (count, width) = rows(self)?;
        if gamma.shape() != [width] || beta.shape() != [width] {
            return Err(TensorError::InvalidShape(
                "layer norm γ and β must be [width]",
            ));
        }
        let mut data = zeros_buffer(self.numel())?;
        for r in 0..count {
            let x = &self.data()[r * width..(r + 1) * width];
            let (mean, inv_std) = row_stats(x);
            for (((out, &value), &g), &b) in data[r * width..(r + 1) * width]
                .iter_mut()
                .zip(x)
                .zip(gamma.data())
                .zip(beta.data())
            {
                *out = (value - mean) * inv_std * g + b;
            }
        }
        Ok(Self::node(
            data,
            self.shape().to_vec(),
            Op::LayerNormAffine,
            vec![self.clone(), gamma.clone(), beta.clone()],
        ))
    }

    /// Atenção causal escalada sobre `[batch, t, dh]` (self = q): `softmax(q·kᵀ/√dh + máscara)`.
    /// Faz as mesmas contas, na mesma ordem, que
    /// `q.bmm(&k.transpose()).mul(&escala).add(&causal_mask(t)).softmax()`, mas o grafo guarda
    /// só as probabilidades `[batch, t, t]`: kᵀ, scores, escala e máscara não viram nós e
    /// são refeitos no backward. Corta três matrizes `t×t` por cabeça das ativações.
    pub fn causal_attention_with_backend(
        &self,
        k: &Self,
        backend: std::sync::Arc<dyn crate::Backend>,
    ) -> Result<Self> {
        if self.rank() != 3 || k.shape() != self.shape() {
            return Err(TensorError::InvalidShape(
                "causal attention needs q and k with the same [batch, t, dh] shape",
            ));
        }
        let (batch, t, dh) = (self.shape()[0], self.shape()[1], self.shape()[2]);
        let scale = 1.0 / (dh as f32).sqrt();
        let mut data = zeros_buffer(element_count(&[batch, t, t])?)?;
        if t != 0 && dh != 0 {
            for b in 0..batch {
                let kt = transposed(&k.data()[b * t * dh..(b + 1) * t * dh], t, dh)?;
                let scores =
                    backend.matmul(&self.data()[b * t * dh..(b + 1) * t * dh], &kt, t, dh, t)?;
                if scores.len() != t * t {
                    return Err(TensorError::InvalidShape(
                        "backend returned wrong output size",
                    ));
                }
                let out = &mut data[b * t * t..(b + 1) * t * t];
                // Linhas independentes: em paralelo, com as mesmas contas por linha.
                out.par_chunks_mut(t)
                    .zip(scores.par_chunks(t))
                    .enumerate()
                    .for_each(|(i, (row, scores))| {
                        // Mesma ordem de antes: × escala, + máscara (0 ou -inf), log-softmax, exp.
                        for (j, (value, &score)) in row.iter_mut().zip(scores).enumerate() {
                            *value = score * scale + if j > i { f32::NEG_INFINITY } else { 0.0 };
                        }
                        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let ln_sum = row.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
                        for value in row.iter_mut() {
                            *value = ((*value - max) - ln_sum).exp();
                        }
                    });
            }
        }
        Ok(Self::node(
            data,
            vec![batch, t, t],
            Op::CausalAttention,
            vec![self.clone(), k.clone()],
        )
        .with_backend(backend))
    }

    /// Junta matrizes ao longo de `axis` (0: empilha linhas, 1: lado a lado nas colunas).
    /// Junta as cabeças da atenção numa matriz só, para uma única projeção de saída.
    pub fn concat(parts: &[Self], axis: usize) -> Result<Self> {
        if parts.is_empty() || axis > 1 || parts.iter().any(|p| p.rank() != 2) {
            return Err(TensorError::InvalidShape(
                "concat needs matrices and axis 0 or 1",
            ));
        }
        let fixed = parts[0].shape()[1 - axis];
        if parts.iter().any(|p| p.shape()[1 - axis] != fixed) {
            return Err(TensorError::ShapeMismatch);
        }
        let joined = parts.iter().map(|p| p.shape()[axis]).sum::<usize>();
        let shape = if axis == 0 {
            vec![joined, fixed]
        } else {
            vec![fixed, joined]
        };
        if parts.iter().any(Self::is_resident) {
            return resident_concat(parts, axis, &shape);
        }
        let mut data = Vec::new();
        data.try_reserve_exact(element_count(&shape)?)?;
        if axis == 0 {
            for part in parts {
                data.extend_from_slice(part.data());
            }
        } else {
            for row in 0..fixed {
                for part in parts {
                    let width = part.shape()[1];
                    data.extend_from_slice(&part.data()[row * width..(row + 1) * width]);
                }
            }
        }
        Ok(Self::node(data, shape, Op::Concat { axis }, parts.to_vec()))
    }

    /// Mesmos dados com outro shape; a quantidade de elementos precisa coincidir.
    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        if self.is_resident() {
            return Err(TensorError::InvalidShape("reshape of a resident tensor"));
        }
        let expected = element_count(shape)?;
        if expected != self.numel() {
            return Err(TensorError::DataLength {
                expected,
                actual: self.numel(),
            });
        }
        // Os dados são imutáveis: o nó novo compartilha o buffer do pai.
        Ok(Self::shared_node(
            std::sync::Arc::clone(self.storage()),
            shape.to_vec(),
            Op::Reshape,
            vec![self.clone()],
        ))
    }

    /// GELU com aproximação tanh (GPT-2/BERT).
    pub fn gelu(&self) -> Result<Self> {
        let mut data = zeros_buffer(self.numel())?;
        // Elemento a elemento: em paralelo, mesmo resultado (tanh é o custo dominante).
        data.par_iter_mut()
            .zip(self.data().par_iter())
            .for_each(|(out, &x)| {
                *out = 0.5 * x * (1.0 + (GELU_C * (x + GELU_K * x * x * x)).tanh());
            });
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
        self.cross_entropy_with_backend(targets, std::sync::Arc::new(crate::CpuBackend::default()))
    }

    pub fn cross_entropy_with_backend(
        &self,
        targets: &[usize],
        backend: std::sync::Arc<dyn crate::Backend>,
    ) -> Result<Self> {
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
        let loss = backend.cross_entropy(self.data(), targets, classes)?;
        Ok(
            Self::node(vec![loss], vec![], Op::CrossEntropy, vec![self.clone()])
                .with_indices(targets)
                .with_backend(backend),
        )
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

pub(crate) fn row_stats(x: &[f32]) -> (f32, f32) {
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
            let rank = parent.rank();
            let (m, n) = (parent.shape()[rank - 2], parent.shape()[rank - 1]);
            if m != 0 && n != 0 {
                for (out, gr) in target.chunks_mut(m * n).zip(g.chunks(m * n)) {
                    for i in 0..m {
                        for j in 0..n {
                            out[i * n + j] += gr[j * m + i];
                        }
                    }
                }
            }
        }
        Op::Gelu => {
            target
                .par_iter_mut()
                .zip(g.par_iter())
                .zip(parent.data().par_iter())
                .for_each(|((out, grad), &x)| {
                    let t = (GELU_C * (x + GELU_K * x * x * x)).tanh();
                    let du = GELU_C * (1.0 + 3.0 * GELU_K * x * x);
                    *out += grad * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du);
                });
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

/// Linhas de faixas vizinhas do mesmo store já são uma matriz contígua na arena: o concat
/// vira uma folha residente sobre a faixa toda, sem copiar. Os gradientes caem direto nas
/// faixas de cada parte.
fn resident_concat(parts: &[Tensor], axis: usize, shape: &[usize]) -> Result<Tensor> {
    let not_adjacent =
        || TensorError::InvalidShape("resident concat needs adjacent views on axis 0");
    let (store, first) = parts[0].residency().ok_or_else(not_adjacent)?;
    let mut end = first.offset;
    for part in parts {
        match part.residency() {
            Some((owner, view))
                if axis == 0 && std::sync::Arc::ptr_eq(owner, store) && view.offset == end =>
            {
                end += view.len;
            }
            _ => return Err(not_adjacent()),
        }
    }
    let view = crate::View {
        offset: first.offset,
        len: end - first.offset,
    };
    Tensor::resident(store.clone(), view, shape)
}
