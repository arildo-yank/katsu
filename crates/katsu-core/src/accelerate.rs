//! Backend macOS: matmul pelo `cblas_sgemm` do Accelerate (coprocessador de matrizes da Apple).
//! No M3, 8 a 16 vezes o kernel CPU próprio nos formatos do 134M. Só a ordem das somas muda
//! (erro relativo ~1e-6); as demais operações ficam no `CpuBackend`, que segue como referência.
use crate::{backend::check, tensor::zeros_buffer, Backend, CpuBackend, TensorError};

type Result<T> = std::result::Result<T, TensorError>;

const ROW_MAJOR: i32 = 101;
const NO_TRANS: i32 = 111;
const TRANS: i32 = 112;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/// `C[m,n] = op(A)·op(B)`, em linhas. `a` e `b` já no layout que `trans_*` indica.
#[allow(clippy::too_many_arguments)]
fn sgemm(
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
) -> Result<Vec<f32>> {
    let mut c = zeros_buffer(m * n)?;
    if m == 0 || n == 0 || k == 0 {
        return Ok(c);
    }
    let dim = |value: usize| i32::try_from(value).map_err(|_| TensorError::ShapeOverflow);
    let (lda, ldb) = (if trans_a { m } else { k }, if trans_b { k } else { n });
    if a.len() < m * k || b.len() < k * n {
        return Err(TensorError::InvalidShape("sgemm operands too small"));
    }
    // SAFETY: os tamanhos foram conferidos acima; o Accelerate lê a e b e escreve só em c,
    // que tem m·n elementos e ldc = n.
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            if trans_a { TRANS } else { NO_TRANS },
            if trans_b { TRANS } else { NO_TRANS },
            dim(m)?,
            dim(n)?,
            dim(k)?,
            1.0,
            a.as_ptr(),
            dim(lda)?,
            b.as_ptr(),
            dim(ldb)?,
            0.0,
            c.as_mut_ptr(),
            dim(n)?,
        );
    }
    Ok(c)
}

#[derive(Debug, Default, Clone, Copy)]
pub struct AccelerateBackend {
    cpu: CpuBackend,
}

impl Backend for AccelerateBackend {
    fn name(&self) -> &'static str {
        "accelerate"
    }
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        check(a, &[m, k])?;
        check(b, &[k, n])?;
        sgemm(false, false, m, n, k, a, b)
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
        // dA = g·Bᵀ e dB = Aᵀ·g, com as transposições feitas pelo próprio sgemm.
        Ok((
            sgemm(false, true, m, k, n, g, b)?,
            sgemm(true, false, k, n, m, a, g)?,
        ))
    }
    fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.cpu.add(a, b)
    }
    fn add_backward(
        &self,
        g: &[f32],
        left_map: &[usize],
        right_map: &[usize],
        left_len: usize,
        right_len: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.cpu
            .add_backward(g, left_map, right_map, left_len, right_len)
    }
    fn cross_entropy(&self, logits: &[f32], targets: &[usize], classes: usize) -> Result<f32> {
        self.cpu.cross_entropy(logits, targets, classes)
    }
    fn cross_entropy_backward(
        &self,
        logits: &[f32],
        targets: &[usize],
        classes: usize,
        seed: f32,
    ) -> Result<Vec<f32>> {
        self.cpu
            .cross_entropy_backward(logits, targets, classes, seed)
    }
}
