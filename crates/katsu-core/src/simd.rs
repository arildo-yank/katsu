//! Intrinsics isolados: despacho por arquitetura/feature, fatias verificadas e cauda escalar.
#[derive(Clone, Copy, Debug)]
pub enum Kernel {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx2,
}
impl Kernel {
    pub fn detect(enabled: bool) -> Self {
        if enabled {
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Self::Neon;
            }
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx2") {
                return Self::Avx2;
            }
        }
        Self::Scalar
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            #[cfg(target_arch = "aarch64")]
            Self::Neon => "neon",
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => "avx2",
        }
    }
    #[inline]
    pub fn axpy(self, out: &mut [f32], x: &[f32], scale: f32) {
        assert_eq!(out.len(), x.len());
        // SAFETY: feature detection precedes construction of this private kernel;
        // each intrinsic loop checks its full vector width and uses unaligned loads.
        match self {
            Self::Scalar => scalar(out, x, scale),
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { neon(out, x, scale) },
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => unsafe { avx2(out, x, scale) },
        }
    }
    /// Produto interno com ordem de soma fixa por kernel (o resultado não depende de threads).
    #[inline]
    pub fn dot(self, x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len());
        // SAFETY: como em axpy, a feature foi detectada antes de construir o kernel.
        match self {
            Self::Scalar => scalar_dot(x, y),
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { neon_dot(x, y) },
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => unsafe { avx2_dot(x, y) },
        }
    }
}
fn scalar_dot(x: &[f32], y: &[f32]) -> f32 {
    // Oito acumuladores independentes, como antes no matmul_backward.
    let mut lanes = [0.0f32; 8];
    let (xc, yc) = (x.chunks_exact(8), y.chunks_exact(8));
    let (xt, yt) = (xc.remainder(), yc.remainder());
    for (xs, ys) in xc.zip(yc) {
        for l in 0..8 {
            lanes[l] += xs[l] * ys[l];
        }
    }
    for (l, (&a, &b)) in xt.iter().zip(yt).enumerate() {
        lanes[l] += a * b;
    }
    lanes.iter().sum()
}
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn neon_dot(x: &[f32], y: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let mut i = 0;
    let mut acc = [vdupq_n_f32(0.0); 4];
    while i + 16 <= x.len() {
        for (lane, acc) in acc.iter_mut().enumerate() {
            let a = vld1q_f32(x.as_ptr().add(i + 4 * lane));
            let b = vld1q_f32(y.as_ptr().add(i + 4 * lane));
            *acc = vfmaq_f32(*acc, a, b);
        }
        i += 16;
    }
    while i + 4 <= x.len() {
        acc[0] = vfmaq_f32(
            acc[0],
            vld1q_f32(x.as_ptr().add(i)),
            vld1q_f32(y.as_ptr().add(i)),
        );
        i += 4;
    }
    let sum = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
    vaddvq_f32(sum) + scalar_dot(&x[i..], &y[i..])
}
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn avx2_dot(x: &[f32], y: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut i = 0;
    let mut acc = [_mm256_setzero_ps(); 4];
    while i + 32 <= x.len() {
        for (lane, acc) in acc.iter_mut().enumerate() {
            let a = _mm256_loadu_ps(x.as_ptr().add(i + 8 * lane));
            let b = _mm256_loadu_ps(y.as_ptr().add(i + 8 * lane));
            *acc = _mm256_add_ps(*acc, _mm256_mul_ps(a, b));
        }
        i += 32;
    }
    while i + 8 <= x.len() {
        let a = _mm256_loadu_ps(x.as_ptr().add(i));
        let b = _mm256_loadu_ps(y.as_ptr().add(i));
        acc[0] = _mm256_add_ps(acc[0], _mm256_mul_ps(a, b));
        i += 8;
    }
    let sum = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
    lanes.iter().sum::<f32>() + scalar_dot(&x[i..], &y[i..])
}
fn scalar(out: &mut [f32], x: &[f32], scale: f32) {
    for (o, &v) in out.iter_mut().zip(x) {
        *o += scale * v;
    }
}
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn neon(out: &mut [f32], x: &[f32], scale: f32) {
    use std::arch::aarch64::*;
    let mut i = 0;
    let a = vdupq_n_f32(scale);
    // Quatro vetores por iteração reduzem o custo de controle do laço.
    while i + 16 <= out.len() {
        macro_rules! lane {
            ($offset:expr) => {{
                let y = vld1q_f32(out.as_ptr().add(i + $offset));
                let v = vld1q_f32(x.as_ptr().add(i + $offset));
                vst1q_f32(
                    out.as_mut_ptr().add(i + $offset),
                    vaddq_f32(y, vmulq_f32(a, v)),
                );
            }};
        }
        lane!(0);
        lane!(4);
        lane!(8);
        lane!(12);
        i += 16;
    }
    while i + 4 <= out.len() {
        let y = vld1q_f32(out.as_ptr().add(i));
        let v = vld1q_f32(x.as_ptr().add(i));
        vst1q_f32(out.as_mut_ptr().add(i), vaddq_f32(y, vmulq_f32(a, v)));
        i += 4;
    }
    scalar(&mut out[i..], &x[i..], scale);
}
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn avx2(out: &mut [f32], x: &[f32], scale: f32) {
    use std::arch::x86_64::*;
    let mut i = 0;
    let a = _mm256_set1_ps(scale);
    while i + 32 <= out.len() {
        macro_rules! lane {
            ($offset:expr) => {{
                let y = _mm256_loadu_ps(out.as_ptr().add(i + $offset));
                let v = _mm256_loadu_ps(x.as_ptr().add(i + $offset));
                _mm256_storeu_ps(
                    out.as_mut_ptr().add(i + $offset),
                    _mm256_add_ps(y, _mm256_mul_ps(a, v)),
                );
            }};
        }
        lane!(0);
        lane!(8);
        lane!(16);
        lane!(24);
        i += 32;
    }
    while i + 8 <= out.len() {
        let y = _mm256_loadu_ps(out.as_ptr().add(i));
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        _mm256_storeu_ps(
            out.as_mut_ptr().add(i),
            _mm256_add_ps(y, _mm256_mul_ps(a, v)),
        );
        i += 8;
    }
    scalar(&mut out[i..], &x[i..], scale);
}
