//! BF16 (bfloat16): 1 bit de sinal, 8 de expoente e 7 de mantissa. Mesmo alcance do FP32, com
//! ~3 dígitos decimais de precisão; é o formato dos Tensor Cores no treino de precisão mista.
//!
//! Precisão mista, como no treino de referência (Micikevicius et al., 2018):
//! - os operandos de cada matmul (pesos e ativações) são guardados em BF16;
//! - o produto acumula em FP32 (o produto de duas mantissas de 8 bits cabe exato em FP32);
//! - a saída da matmul (ativação) volta a BF16; o gradiente de peso fica em FP32;
//! - pesos-mestre, gradientes acumulados e momentos do Adam continuam em FP32;
//! - LayerNorm, softmax, GELU e a entropia cruzada rodam em FP32.
//!
//! Na CPU o BF16 é numérico, não de memória: o grafo do autograd guarda `Vec<f32>`, então os
//! valores BF16 ocupam 4 bytes cada e há custo de conversão. `Bf16Buffer` é o armazenamento de
//! 2 bytes; a economia de memória de ativações só acontece num executor que guarde esse formato
//! no dispositivo (os kernels CUDA em BF16 ainda não existem).
use std::sync::Arc;

use rayon::prelude::*;

use crate::{backend::check, tensor::zeros_buffer, Backend, TensorError};

type Result<T> = std::result::Result<T, TensorError>;

const PARALLEL: usize = 1 << 16;

/// Arredonda para o BF16 mais próximo (empate para o par). NaN continua NaN; valores acima do
/// maior BF16 finito viram infinito, como no hardware.
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        // Mantém o sinal e força um NaN silencioso (a mantissa truncada poderia zerar).
        return ((bits >> 16) as u16) | 0x0040;
    }
    let lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7fff + lsb) >> 16) as u16
}

/// Conversão exata: todo BF16 é um FP32.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// O FP32 que o BF16 mais próximo representa.
#[inline]
pub fn round_bf16(x: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(x))
}

/// Cópia de `values` arredondada para BF16, ainda em FP32 (para kernels FP32 que acumulam em
/// FP32, como o `cblas_sgemm`: o resultado é o de uma matmul BF16 com acumulação FP32).
pub fn rounded(values: &[f32]) -> Result<Vec<f32>> {
    let mut out = zeros_buffer(values.len())?;
    round_into(&mut out, values);
    Ok(out)
}

fn round_into(out: &mut [f32], values: &[f32]) {
    if values.len() >= PARALLEL {
        out.par_chunks_mut(PARALLEL / 4)
            .zip(values.par_chunks(PARALLEL / 4))
            .for_each(|(o, v)| o.iter_mut().zip(v).for_each(|(o, &v)| *o = round_bf16(v)));
    } else {
        out.iter_mut()
            .zip(values)
            .for_each(|(o, &v)| *o = round_bf16(v));
    }
}

fn round_in_place(values: &mut [f32]) {
    if values.len() >= PARALLEL {
        values
            .par_chunks_mut(PARALLEL / 4)
            .for_each(|c| c.iter_mut().for_each(|v| *v = round_bf16(*v)));
    } else {
        values.iter_mut().for_each(|v| *v = round_bf16(*v));
    }
}

/// Armazenamento BF16: 2 bytes por número.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bf16Buffer {
    bits: Vec<u16>,
}

impl Bf16Buffer {
    pub fn from_f32(values: &[f32]) -> Result<Self> {
        let mut bits = Vec::new();
        bits.try_reserve_exact(values.len())?;
        if values.len() >= PARALLEL {
            values
                .par_iter()
                .map(|&v| f32_to_bf16(v))
                .collect_into_vec(&mut bits);
        } else {
            bits.extend(values.iter().map(|&v| f32_to_bf16(v)));
        }
        Ok(Self { bits })
    }

    pub fn from_bits(bits: Vec<u16>) -> Self {
        Self { bits }
    }

    pub fn to_f32(&self) -> Result<Vec<f32>> {
        let mut out = zeros_buffer(self.bits.len())?;
        out.iter_mut()
            .zip(&self.bits)
            .for_each(|(o, &b)| *o = bf16_to_f32(b));
        Ok(out)
    }

    pub fn bits(&self) -> &[u16] {
        &self.bits
    }

    pub fn len(&self) -> usize {
        self.bits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// Bytes ocupados pelos números (sem o cabeçalho do Vec).
    pub fn bytes(&self) -> usize {
        self.bits.len() * std::mem::size_of::<u16>()
    }
}

/// Kernel de referência: `C[m,n] = A[m,k]·B[k,n]` com A e B em BF16 e acumulação em FP32.
/// Lento (escalar por linha, em paralelo); serve de oráculo para `Bf16Backend` e para os
/// futuros kernels de GPU.
pub fn matmul_bf16(
    a: &Bf16Buffer,
    b: &Bf16Buffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    if a.len() != m * k || b.len() != k * n {
        return Err(TensorError::MatmulMismatch {
            left_columns: a.len().checked_div(m).unwrap_or(0),
            right_rows: b.len().checked_div(n).unwrap_or(0),
        });
    }
    let mut out = zeros_buffer(m * n)?;
    if out.is_empty() || k == 0 {
        return Ok(out);
    }
    let (a, b) = (a.bits(), b.bits());
    out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for p in 0..k {
            let scale = bf16_to_f32(a[i * k + p]);
            for (o, &w) in row.iter_mut().zip(&b[p * n..(p + 1) * n]) {
                *o += scale * bf16_to_f32(w);
            }
        }
    });
    Ok(out)
}

/// Precisão mista sobre um backend FP32: arredonda os operandos de cada matmul para BF16, deixa
/// o `inner` acumular em FP32 e devolve a saída (ativação) em BF16. No backward, o gradiente da
/// entrada (ativação) volta em BF16 e o do lado direito (peso, na maioria das chamadas) fica em
/// FP32, para somar nos pesos-mestre. Soma e entropia cruzada passam direto para o `inner`.
#[derive(Debug, Clone)]
pub struct Bf16Backend {
    inner: Arc<dyn Backend>,
}

impl Bf16Backend {
    pub fn new(inner: Arc<dyn Backend>) -> Self {
        Self { inner }
    }
}

impl Backend for Bf16Backend {
    fn name(&self) -> &'static str {
        "bf16-mixed"
    }
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        check(a, &[m, k])?;
        check(b, &[k, n])?;
        let mut out = self.inner.matmul(&rounded(a)?, &rounded(b)?, m, k, n)?;
        round_in_place(&mut out);
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
        let (mut da, db) =
            self.inner
                .matmul_backward(&rounded(a)?, &rounded(b)?, &rounded(g)?, m, k, n)?;
        round_in_place(&mut da);
        Ok((da, db))
    }
    fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.inner.add(a, b)
    }
    fn add_backward(
        &self,
        g: &[f32],
        left_map: &[usize],
        right_map: &[usize],
        left_len: usize,
        right_len: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.inner
            .add_backward(g, left_map, right_map, left_len, right_len)
    }
    fn cross_entropy(&self, logits: &[f32], targets: &[usize], classes: usize) -> Result<f32> {
        self.inner.cross_entropy(logits, targets, classes)
    }
    fn cross_entropy_backward(
        &self,
        logits: &[f32],
        targets: &[usize],
        classes: usize,
        seed: f32,
    ) -> Result<Vec<f32>> {
        self.inner
            .cross_entropy_backward(logits, targets, classes, seed)
    }
}
