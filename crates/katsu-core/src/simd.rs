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
