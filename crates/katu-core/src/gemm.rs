//! GEMM de f32 em blocos, no esquema do BLIS: B é empacotado em fatias de NR colunas, A em fatias de
//! MR linhas, e um microkernel com os acumuladores nos registradores calcula cada bloco MR×NR de C.
//! Microkernels: NEON 8×12 (aarch64), AVX2+FMA 6×16 (x86_64) e um escalar 4×8 de reserva.
//! Fase 1 de `docs/KATU_ARRAY_PLAN.md`. Ainda não está ligado ao backend de treino: a ordem das somas
//! é outra, e trocar agora mudaria os números do motor antes da avaliação do piloto de 1B.

use rayon::prelude::*;

/// Profundidade de um bloco: KC elementos de uma linha de A e de uma coluna de B cabem no L1.
const KC: usize = 256;
/// Linhas de A por tarefa paralela (arredondado para múltiplo de MR).
const MC: usize = 64;
/// Colunas de B empacotadas de uma vez.
const NC: usize = 4096;
/// Maior bloco de C de um microkernel (8×12 e 6×16 dão 96).
const TILE: usize = 96;

type MicroFn = unsafe fn(usize, *const f32, *const f32, &mut [f32; TILE]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Micro {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx2,
}

impl Micro {
    /// O melhor microkernel desta máquina.
    pub fn detect() -> Self {
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Micro::Neon;
        }
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return Micro::Avx2;
        }
        Micro::Scalar
    }

    /// Todos os microkernels que esta máquina consegue rodar, para os testes.
    pub fn available() -> Vec<Self> {
        let mut all = vec![Micro::Scalar];
        let best = Self::detect();
        if best != Micro::Scalar {
            all.push(best);
        }
        all
    }

    pub fn name(self) -> &'static str {
        match self {
            Micro::Scalar => "scalar 4x8",
            #[cfg(target_arch = "aarch64")]
            Micro::Neon => "neon 8x12",
            #[cfg(target_arch = "x86_64")]
            Micro::Avx2 => "avx2 6x16",
        }
    }

    fn shape(self) -> (usize, usize, MicroFn) {
        match self {
            Micro::Scalar => (4, 8, scalar_4x8 as MicroFn),
            #[cfg(target_arch = "aarch64")]
            Micro::Neon => (8, 12, neon_8x12 as MicroFn),
            #[cfg(target_arch = "x86_64")]
            Micro::Avx2 => (6, 16, avx2_6x16 as MicroFn),
        }
    }
}

/// C = A·B, com A de m×k, B de k×n e C de m×n, todos em ordem de linhas. C é sobrescrito.
pub fn sgemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    sgemm_with(Micro::detect(), a, b, c, m, k, n);
}

/// Como [`sgemm`], com o microkernel escolhido. Um microkernel que a máquina não tem cai no escalar.
pub fn sgemm_with(micro: Micro, a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(a.len(), m * k, "A precisa ter m×k elementos");
    assert_eq!(b.len(), k * n, "B precisa ter k×n elementos");
    assert_eq!(c.len(), m * n, "C precisa ter m×n elementos");
    let micro = if Micro::available().contains(&micro) {
        micro
    } else {
        Micro::Scalar
    };
    let (mr, nr, kernel) = micro.shape();
    c.fill(0.0);
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let mc = (MC / mr).max(1) * mr;
    for jc in (0..n).step_by(NC) {
        let nc = NC.min(n - jc);
        for pc in (0..k).step_by(KC) {
            let kc = KC.min(k - pc);
            let packed_b = pack_b(b, n, pc, jc, kc, nc, nr);
            c.par_chunks_mut(mc * n)
                .enumerate()
                .for_each(|(block, rows)| {
                    let ic = block * mc;
                    let mcur = rows.len() / n;
                    let packed_a = pack_a(a, k, ic, pc, mcur, kc, mr);
                    let mut tile = [0.0f32; TILE];
                    for jr in (0..nc).step_by(nr) {
                        let b_sliver = &packed_b[(jr / nr) * nr * kc..][..nr * kc];
                        for ir in (0..mcur).step_by(mr) {
                            let a_sliver = &packed_a[(ir / mr) * mr * kc..][..mr * kc];
                            // SAFETY: as fatias têm mr·kc e nr·kc elementos, que é o que o microkernel lê,
                            // e o microkernel só foi escolhido se a máquina tem a extensão.
                            unsafe { kernel(kc, a_sliver.as_ptr(), b_sliver.as_ptr(), &mut tile) };
                            for i in 0..mr.min(mcur - ir) {
                                let out = &mut rows[(ir + i) * n + jc + jr..][..nr.min(nc - jr)];
                                for (o, &t) in out.iter_mut().zip(&tile[i * nr..]) {
                                    *o += t;
                                }
                            }
                        }
                    }
                });
        }
    }
}

/// Bloco de A em fatias de `mr` linhas; cada fatia guarda, para cada p, as `mr` linhas lado a lado.
/// Linhas que faltam no fim viram zero.
fn pack_a(
    a: &[f32],
    k: usize,
    ic: usize,
    pc: usize,
    mcur: usize,
    kc: usize,
    mr: usize,
) -> Vec<f32> {
    let slivers = mcur.div_ceil(mr);
    let mut out = vec![0.0f32; slivers * mr * kc];
    for s in 0..slivers {
        let sliver = &mut out[s * mr * kc..][..mr * kc];
        for i in 0..mr.min(mcur - s * mr) {
            let row = &a[(ic + s * mr + i) * k + pc..][..kc];
            for (p, &value) in row.iter().enumerate() {
                sliver[p * mr + i] = value;
            }
        }
    }
    out
}

/// Bloco de B em fatias de `nr` colunas; cada fatia guarda, para cada p, as `nr` colunas lado a lado.
fn pack_b(b: &[f32], n: usize, pc: usize, jc: usize, kc: usize, nc: usize, nr: usize) -> Vec<f32> {
    let slivers = nc.div_ceil(nr);
    let mut out = vec![0.0f32; slivers * nr * kc];
    out.par_chunks_mut(nr * kc)
        .enumerate()
        .for_each(|(t, sliver)| {
            let width = nr.min(nc - t * nr);
            for p in 0..kc {
                let row = &b[(pc + p) * n + jc + t * nr..][..width];
                sliver[p * nr..p * nr + width].copy_from_slice(row);
            }
        });
    out
}

/// Reserva sem SIMD: 4×8, para máquinas sem as extensões e para conferir os outros.
///
/// # Safety
/// `a` precisa ter 4·kc e `b` 8·kc elementos legíveis.
unsafe fn scalar_4x8(kc: usize, a: *const f32, b: *const f32, out: &mut [f32; TILE]) {
    let (a, b) = unsafe {
        (
            std::slice::from_raw_parts(a, 4 * kc),
            std::slice::from_raw_parts(b, 8 * kc),
        )
    };
    let mut c = [[0.0f32; 8]; 4];
    for p in 0..kc {
        for (i, row) in c.iter_mut().enumerate() {
            let x = a[p * 4 + i];
            for (j, value) in row.iter_mut().enumerate() {
                *value += x * b[p * 8 + j];
            }
        }
    }
    for (i, row) in c.iter().enumerate() {
        out[i * 8..i * 8 + 8].copy_from_slice(row);
    }
}

/// NEON 8×12: 24 acumuladores de 4 lanes, A lido em 2 vetores e multiplicado por lane.
///
/// # Safety
/// A máquina precisa ter NEON; `a` precisa ter 8·kc e `b` 12·kc elementos legíveis.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_8x12(kc: usize, a: *const f32, b: *const f32, out: &mut [f32; TILE]) {
    use std::arch::aarch64::*;
    let mut c = [[vdupq_n_f32(0.0); 3]; 8];
    for p in 0..kc {
        let (a_lo, a_hi, b0, b1, b2) = unsafe {
            (
                vld1q_f32(a.add(p * 8)),
                vld1q_f32(a.add(p * 8 + 4)),
                vld1q_f32(b.add(p * 12)),
                vld1q_f32(b.add(p * 12 + 4)),
                vld1q_f32(b.add(p * 12 + 8)),
            )
        };
        macro_rules! row {
            ($i:literal, $a:ident, $lane:literal) => {
                c[$i][0] = vfmaq_laneq_f32::<$lane>(c[$i][0], b0, $a);
                c[$i][1] = vfmaq_laneq_f32::<$lane>(c[$i][1], b1, $a);
                c[$i][2] = vfmaq_laneq_f32::<$lane>(c[$i][2], b2, $a);
            };
        }
        row!(0, a_lo, 0);
        row!(1, a_lo, 1);
        row!(2, a_lo, 2);
        row!(3, a_lo, 3);
        row!(4, a_hi, 0);
        row!(5, a_hi, 1);
        row!(6, a_hi, 2);
        row!(7, a_hi, 3);
    }
    for (i, row) in c.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            unsafe { vst1q_f32(out.as_mut_ptr().add(i * 12 + j * 4), v) };
        }
    }
}

/// AVX2+FMA 6×16: 12 acumuladores de 8 lanes, A difundido valor a valor.
///
/// # Safety
/// A máquina precisa ter AVX2 e FMA; `a` precisa ter 6·kc e `b` 16·kc elementos legíveis.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_6x16(kc: usize, a: *const f32, b: *const f32, out: &mut [f32; TILE]) {
    use std::arch::x86_64::*;
    let mut c = [[_mm256_setzero_ps(); 2]; 6];
    for p in 0..kc {
        let (b0, b1) = unsafe {
            (
                _mm256_loadu_ps(b.add(p * 16)),
                _mm256_loadu_ps(b.add(p * 16 + 8)),
            )
        };
        for (i, row) in c.iter_mut().enumerate() {
            let x = unsafe { _mm256_broadcast_ss(&*a.add(p * 6 + i)) };
            row[0] = _mm256_fmadd_ps(x, b0, row[0]);
            row[1] = _mm256_fmadd_ps(x, b1, row[1]);
        }
    }
    for (i, row) in c.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(i * 16 + j * 8), v) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random(len: usize, seed: &mut u64) -> Vec<f32> {
        (0..len)
            .map(|_| {
                *seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Referência em f64, com a cota de erro de cada elemento: soma de |a|·|b| vezes a precisão do f32.
    fn reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut exact = vec![0.0; m * n];
        let mut bound = vec![0.0; m * n];
        for i in 0..m {
            for p in 0..k {
                let x = a[i * k + p] as f64;
                for j in 0..n {
                    let y = b[p * n + j] as f64;
                    exact[i * n + j] += x * y;
                    bound[i * n + j] += (x * y).abs();
                }
            }
        }
        (exact, bound)
    }

    #[test]
    fn every_kernel_matches_the_reference_on_awkward_shapes() {
        let mut seed = 7;
        let shapes = [
            (1, 1, 1),
            (1, 300, 1),
            (7, 13, 5),
            (8, 256, 12),
            (9, 257, 13),
            (65, 70, 129),
            (130, 513, 37),
            (3, 2, 4100),
        ];
        for micro in Micro::available() {
            for &(m, k, n) in &shapes {
                let a = random(m * k, &mut seed);
                let b = random(k * n, &mut seed);
                let mut c = vec![f32::NAN; m * n];
                sgemm_with(micro, &a, &b, &mut c, m, k, n);
                let (exact, bound) = reference(&a, &b, m, k, n);
                for (i, ((&got, &want), &limit)) in c.iter().zip(&exact).zip(&bound).enumerate() {
                    let tolerance = 2.0 * k as f64 * f32::EPSILON as f64 * limit + 1e-30;
                    assert!(
                        (got as f64 - want).abs() <= tolerance,
                        "{} {m}x{k}x{n}[{i}]: {got} contra {want}",
                        micro.name()
                    );
                }
            }
        }
    }

    #[test]
    fn empty_shapes_give_zeros() {
        let mut c = vec![1.0; 6];
        sgemm(&[], &[], &mut c, 2, 0, 3);
        assert_eq!(c, vec![0.0; 6]);
        sgemm(&[], &[1.0; 3], &mut [], 0, 1, 3);
    }
}
