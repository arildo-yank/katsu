//! Fronteira inicial de kernels de matmul. Buffers ainda são f32 em memória host.
use crate::{
    simd::Kernel,
    tensor::{element_count, zeros_buffer},
    TensorError,
};
use rayon::prelude::*;

type Result<T> = std::result::Result<T, TensorError>;

pub trait Backend: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &'static str;
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>>;
    fn matmul_backward(
        &self,
        a: &[f32],
        b: &[f32],
        g: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)>;
    fn add(&self, _a: &[f32], _b: &[f32]) -> Result<Vec<f32>> {
        Err(TensorError::BackendUnavailable(
            "add kernel not implemented",
        ))
    }
    fn add_backward(
        &self,
        _g: &[f32],
        _left_map: &[usize],
        _right_map: &[usize],
        _left_len: usize,
        _right_len: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        Err(TensorError::BackendUnavailable(
            "add backward not implemented",
        ))
    }
    fn cross_entropy(&self, _logits: &[f32], _targets: &[usize], _classes: usize) -> Result<f32> {
        Err(TensorError::BackendUnavailable(
            "cross entropy not implemented",
        ))
    }
    fn cross_entropy_backward(
        &self,
        _logits: &[f32],
        _targets: &[usize],
        _classes: usize,
        _seed: f32,
    ) -> Result<Vec<f32>> {
        Err(TensorError::BackendUnavailable(
            "cross entropy backward not implemented",
        ))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CpuBackend {
    pub parallel: bool,
    pub simd: bool,
}
impl Default for CpuBackend {
    fn default() -> Self {
        Self {
            parallel: true,
            simd: true,
        }
    }
}
impl CpuBackend {
    pub fn serial() -> Self {
        Self {
            parallel: false,
            simd: false,
        }
    }
    pub fn kernel_name(&self) -> &'static str {
        Kernel::detect(self.simd).name()
    }
}
pub(crate) fn check(data: &[f32], shape: &[usize]) -> Result<()> {
    let expected = element_count(shape)?;
    if data.len() != expected {
        return Err(TensorError::DataLength {
            expected,
            actual: data.len(),
        });
    }
    Ok(())
}
fn parallel_work(cpu: &CpuBackend, m: usize, k: usize, n: usize) -> bool {
    cpu.parallel && m.saturating_mul(k).saturating_mul(n) >= 32_768
}
impl Backend for CpuBackend {
    fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        check(b, &[a.len()])?;
        Ok(a.iter().zip(b).map(|(a, b)| a + b).collect())
    }
    fn add_backward(
        &self,
        g: &[f32],
        left_map: &[usize],
        right_map: &[usize],
        left_len: usize,
        right_len: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        validate_maps(g, left_map, right_map, left_len, right_len)?;
        let mut left = zeros_buffer(left_len)?;
        let mut right = zeros_buffer(right_len)?;
        for (i, &value) in g.iter().enumerate() {
            left[left_map[i]] += value;
            right[right_map[i]] += value;
        }
        Ok((left, right))
    }
    fn cross_entropy(&self, logits: &[f32], targets: &[usize], classes: usize) -> Result<f32> {
        validate_ce(logits, targets, classes)?;
        let mut total = 0.0;
        for (r, &label) in targets.iter().enumerate() {
            let row = &logits[r * classes..(r + 1) * classes];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
            total += sum.ln() - (row[label] - max);
        }
        Ok(total / targets.len().max(1) as f32)
    }
    fn cross_entropy_backward(
        &self,
        logits: &[f32],
        targets: &[usize],
        classes: usize,
        seed: f32,
    ) -> Result<Vec<f32>> {
        validate_ce(logits, targets, classes)?;
        let mut result = zeros_buffer(logits.len())?;
        for (r, &label) in targets.iter().enumerate() {
            let row = &logits[r * classes..(r + 1) * classes];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
            for (c, &x) in row.iter().enumerate() {
                result[r * classes + c] = seed
                    * ((x - max).exp() / sum - if c == label { 1.0 } else { 0.0 })
                    / targets.len() as f32;
            }
        }
        Ok(result)
    }
    fn name(&self) -> &'static str {
        "cpu"
    }
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        check(a, &[m, k])?;
        check(b, &[k, n])?;
        let mut out = zeros_buffer(element_count(&[m, n])?)?;
        if out.is_empty() || k == 0 {
            return Ok(out);
        }
        let kernel = Kernel::detect(self.simd);
        let row = |(i, out): (usize, &mut [f32])| {
            for (p, &scale) in a[i * k..(i + 1) * k].iter().enumerate() {
                kernel.axpy(out, &b[p * n..(p + 1) * n], scale);
            }
        };
        if parallel_work(self, m, k, n) {
            out.par_chunks_mut(n).enumerate().for_each(row);
        } else {
            out.chunks_mut(n).enumerate().for_each(row);
        }
        Ok(out)
    }
    fn matmul_backward(
        &self,
        a: &[f32],
        b: &[f32],
        g: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        check(a, &[m, k])?;
        check(b, &[k, n])?;
        check(g, &[m, n])?;
        let mut da = zeros_buffer(a.len())?;
        let mut db = zeros_buffer(b.len())?;
        if m == 0 || k == 0 || n == 0 {
            return Ok((da, db));
        }
        let kernel = Kernel::detect(self.simd);
        let row_a = |(i, out): (usize, &mut [f32])| {
            let g_row = &g[i * n..(i + 1) * n];
            for (p, result) in out.iter_mut().enumerate() {
                *result = kernel.dot(g_row, &b[p * n..(p + 1) * n]);
            }
        };
        let row_b = |(p, out): (usize, &mut [f32])| {
            for i in 0..m {
                kernel.axpy(out, &g[i * n..(i + 1) * n], a[i * k + p]);
            }
        };
        if parallel_work(self, m, k, n) {
            da.par_chunks_mut(k).enumerate().for_each(row_a);
            db.par_chunks_mut(n).enumerate().for_each(row_b);
        } else {
            da.chunks_mut(k).enumerate().for_each(row_a);
            db.chunks_mut(n).enumerate().for_each(row_b);
        }
        Ok((da, db))
    }
}

/// Reservado para CUDA. Não simula GPU nem cai silenciosamente para CPU.
#[derive(Debug, Clone, Copy)]
pub struct GpuBackend;
impl Backend for GpuBackend {
    fn name(&self) -> &'static str {
        "cuda-unavailable"
    }
    fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Result<Vec<f32>> {
        Err(TensorError::BackendUnavailable(
            "CUDA kernels/device storage are not implemented",
        ))
    }
    fn matmul_backward(
        &self,
        _: &[f32],
        _: &[f32],
        _: &[f32],
        _: usize,
        _: usize,
        _: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        Err(TensorError::BackendUnavailable(
            "CUDA backward is not implemented",
        ))
    }
}

pub(crate) fn validate_ce(logits: &[f32], targets: &[usize], classes: usize) -> Result<()> {
    check(logits, &[targets.len(), classes])?;
    for &index in targets {
        if index >= classes {
            return Err(TensorError::IndexOutOfRange { index, classes });
        }
    }
    Ok(())
}
pub(crate) fn validate_maps(
    g: &[f32],
    left: &[usize],
    right: &[usize],
    left_len: usize,
    right_len: usize,
) -> Result<()> {
    if left.len() != g.len()
        || right.len() != g.len()
        || left.iter().any(|&i| i >= left_len)
        || right.iter().any(|&i| i >= right_len)
    {
        return Err(TensorError::InvalidShape("invalid broadcast gradient maps"));
    }
    Ok(())
}

pub fn create_cuda_backend(device: usize) -> Result<std::sync::Arc<dyn Backend>> {
    #[cfg(feature = "cuda")]
    {
        Ok(std::sync::Arc::new(crate::cuda::CudaBackend::new(device)?))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = device;
        Err(TensorError::BackendUnavailable(
            "rebuild with --features cuda on Linux/NVIDIA",
        ))
    }
}
