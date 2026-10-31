//! Inferência sem autograd com KV cache.
//!
//! Repete a aritmética de `forward_batch` na mesma ordem (matmul por `axpy`,
//! LayerNorm, GELU tanh, softmax via log-softmax), então os logits batem com o
//! caminho do grafo; a diferença é que cada token novo só calcula a própria linha,
//! reaproveita K/V das posições anteriores e projeta no vocabulário só a última.
use rayon::prelude::*;

use super::{KatsuLmConfig, Result, HEAD_START, LN1_B, LN1_G, PER_HEAD};
use crate::{
    ops::{GELU_C, GELU_K},
    simd::Kernel,
    Backend, CpuBackend, Tensor, TensorError,
};

/// K e V já calculados, por camada e cabeça, em `[posições, d/h]` contíguo.
pub(super) struct KvCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    len: usize,
}

impl KvCache {
    pub(super) fn new(config: &KatsuLmConfig) -> Self {
        let slots = config.layers * config.heads;
        Self {
            keys: vec![Vec::new(); slots],
            values: vec![Vec::new(); slots],
            len: 0,
        }
    }
    pub(super) fn len(&self) -> usize {
        self.len
    }
    pub(super) fn clear(&mut self) {
        self.keys
            .iter_mut()
            .chain(&mut self.values)
            .for_each(Vec::clear);
        self.len = 0;
    }
}

/// `a[m,k] @ b[k,n]`. Com uma linha (decode) divide as colunas entre threads;
/// cada saída ainda soma `a[p]·b[p,j]` em ordem de `p`, como no `CpuBackend`.
fn matmul(
    cpu: &CpuBackend,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    if m != 1 {
        return cpu.matmul(a, b, m, k, n);
    }
    const COLUMNS: usize = 512;
    let kernel = Kernel::detect(true);
    let mut out = vec![0.0; n];
    out.par_chunks_mut(COLUMNS)
        .enumerate()
        .for_each(|(c, chunk)| {
            let start = c * COLUMNS;
            for (p, &scale) in a.iter().enumerate() {
                kernel.axpy(chunk, &b[p * n + start..p * n + start + chunk.len()], scale);
            }
        });
    Ok(out)
}

/// `(x - média)/σ · γ + β` por linha, com os mesmos arredondamentos de `layer_norm().mul().add()`.
fn layer_norm(x: &[f32], width: usize, gamma: &Tensor, beta: &Tensor) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(width) {
        let n = width as f32;
        let mean = row.iter().sum::<f32>() / n;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv_std = 1.0 / (var + crate::LAYER_NORM_EPS).sqrt();
        for ((&v, &g), &b) in row.iter().zip(gamma.data()).zip(beta.data()) {
            out.push((v - mean) * inv_std * g + b);
        }
    }
    out
}

fn add_bias(x: &mut [f32], bias: &Tensor) {
    for row in x.chunks_exact_mut(bias.numel()) {
        for (v, &b) in row.iter_mut().zip(bias.data()) {
            *v += b;
        }
    }
}

fn add_into(x: &mut [f32], y: &[f32]) {
    for (v, &d) in x.iter_mut().zip(y) {
        *v += d;
    }
}

/// Softmax como `Tensor::softmax`: exp(log_softmax), estável pelo máximo.
fn softmax(scores: &mut [f32]) {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let ln_sum = scores.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
    for v in scores.iter_mut() {
        *v = ((*v - max) - ln_sum).exp();
    }
}

impl KatsuLmConfig {
    /// Processa `tokens` nas posições `cache.len()..` e devolve os logits do último.
    pub(super) fn forward_cached(
        &self,
        params: &[Tensor],
        cache: &mut KvCache,
        tokens: &[usize],
    ) -> Result<Vec<f32>> {
        self.check_params(params)?;
        let (m, d, start) = (tokens.len(), self.embed_dim, cache.len);
        if m == 0 || start + m > self.block_size {
            return Err(TensorError::InvalidShape("cached step exceeds block_size"));
        }
        if tokens.iter().any(|&t| t >= self.vocab_size) {
            return Err(TensorError::InvalidShape("token outside vocabulary"));
        }
        let cpu = CpuBackend::default();
        let (wte, wpe) = (params[0].data(), params[1].data());
        let mut x = Vec::with_capacity(m * d);
        for (i, &token) in tokens.iter().enumerate() {
            let pos = start + i;
            x.extend(
                wte[token * d..(token + 1) * d]
                    .iter()
                    .zip(&wpe[pos * d..(pos + 1) * d])
                    .map(|(a, b)| a + b),
            );
        }
        let dh = d / self.heads;
        let scale = 1.0 / (dh as f32).sqrt();
        for layer in 0..self.layers {
            let p = &params[2 + layer * self.per_layer()..2 + (layer + 1) * self.per_layer()];
            let h = layer_norm(&x, d, &p[LN1_G], &p[LN1_B]);
            for (head, w) in p[HEAD_START..HEAD_START + self.heads * PER_HEAD]
                .as_chunks::<PER_HEAD>()
                .0
                .iter()
                .enumerate()
            {
                let slot = layer * self.heads + head;
                let q = matmul(&cpu, &h, w[0].data(), m, d, dh)?;
                cache.keys[slot].extend(matmul(&cpu, &h, w[1].data(), m, d, dh)?);
                cache.values[slot].extend(matmul(&cpu, &h, w[2].data(), m, d, dh)?);
                let (keys, values) = (&cache.keys[slot], &cache.values[slot]);
                let mut mixed = vec![0.0; m * dh];
                for i in 0..m {
                    let qi = &q[i * dh..(i + 1) * dh];
                    let visible = start + i + 1;
                    let mut scores: Vec<f32> = keys[..visible * dh]
                        .chunks_exact(dh)
                        .map(|k| qi.iter().zip(k).fold(0.0, |acc, (a, b)| acc + a * b) * scale)
                        .collect();
                    softmax(&mut scores);
                    let out = &mut mixed[i * dh..(i + 1) * dh];
                    for (&weight, v) in scores.iter().zip(values.chunks_exact(dh)) {
                        for (o, &value) in out.iter_mut().zip(v) {
                            *o += weight * value;
                        }
                    }
                }
                add_into(&mut x, &matmul(&cpu, &mixed, w[3].data(), m, dh, d)?);
            }
            let p = &p[HEAD_START + self.heads * PER_HEAD..];
            let h = layer_norm(&x, d, &p[0], &p[1]);
            let mut mlp = matmul(&cpu, &h, p[2].data(), m, d, 4 * d)?;
            add_bias(&mut mlp, &p[3]);
            for v in &mut mlp {
                *v = 0.5 * *v * (1.0 + (GELU_C * (*v + GELU_K * *v * *v * *v)).tanh());
            }
            let mut out = matmul(&cpu, &mlp, p[4].data(), m, 4 * d, d)?;
            add_bias(&mut out, &p[5]);
            add_into(&mut x, &out);
        }
        cache.len += m;
        let head = &params[2 + self.layers * self.per_layer()..];
        let last = layer_norm(&x[(m - 1) * d..], d, &head[0], &head[1]);
        matmul(&cpu, &last, head[2].data(), 1, d, self.vocab_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> (KatsuLmConfig, Vec<Tensor>) {
        let config = KatsuLmConfig {
            vocab_size: 37,
            block_size: 12,
            embed_dim: 16,
            layers: 2,
            heads: 4,
        };
        let params = config.init(5).unwrap();
        (config, params)
    }

    fn last_row(config: &KatsuLmConfig, params: &[Tensor], tokens: &[usize]) -> Vec<f32> {
        let logits = config.forward(params, tokens).unwrap();
        logits.data()[logits.numel() - config.vocab_size..].to_vec()
    }

    fn assert_close(a: &[f32], b: &[f32]) {
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= 1e-5 * (1.0 + y.abs()), "{x} vs {y}");
        }
    }

    #[test]
    fn prefill_and_decode_match_full_forward() {
        let (config, params) = model();
        let tokens = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8];
        let mut cache = KvCache::new(&config);
        let prefill = config
            .forward_cached(&params, &mut cache, &tokens[..5])
            .unwrap();
        assert_close(&prefill, &last_row(&config, &params, &tokens[..5]));
        for n in 6..=tokens.len() {
            let step = config
                .forward_cached(&params, &mut cache, &tokens[n - 1..n])
                .unwrap();
            assert_close(&step, &last_row(&config, &params, &tokens[..n]));
        }
        assert!(config.forward_cached(&params, &mut cache, &[0]).is_err());
    }

    #[test]
    fn cached_generation_matches_uncached_reference_past_the_window() {
        let (config, params) = model();
        let options = crate::GenerationOptions {
            temperature: 0.8,
            repetition_penalty: 1.3,
            seed: 9,
            stop_tokens: vec![],
        };
        // Prompt curto (cruza a janela de 12 várias vezes) e prompt maior que a janela.
        let long: Vec<usize> = (0..17).map(|i| (i * 7 + 3) % config.vocab_size).collect();
        for prompt in [vec![7, 2, 2], long] {
            // Referência sem cache: a janela começa em 0 e, quando enche, salta para os
            // últimos block_size - block_size/2 tokens; cada token recalcula a janela inteira.
            let keep = config.block_size - config.block_size / 2;
            let mut rng = crate::SplitMix64(options.seed);
            let mut expected = prompt.clone();
            let (mut start, mut slides) = (0, 0);
            for _ in 0..30 {
                if expected.len() - start > config.block_size {
                    start = expected.len() - keep;
                    slides += 1;
                }
                let mut last = last_row(&config, &params, &expected[start..]);
                crate::penalize_repetitions(
                    &mut last,
                    &expected[start..],
                    options.repetition_penalty,
                )
                .unwrap();
                expected.push(super::super::sample(
                    &last,
                    options.temperature,
                    rng.uniform(),
                ));
            }
            assert!(
                slides >= 3,
                "o teste precisa deslizar a janela várias vezes"
            );
            assert_eq!(
                config
                    .generate_with_options(&params, &prompt, 30, &options)
                    .unwrap(),
                expected
            );
        }
    }

    #[test]
    fn filtered_generation_never_emits_vetoed_tokens() {
        let (config, params) = model();
        let options = crate::GenerationOptions {
            temperature: 0.8,
            repetition_penalty: 1.0,
            seed: 3,
            stop_tokens: vec![],
        };
        let even = config
            .generate_filtered(&params, &[1, 2], 20, &options, &mut |_, t| t % 2 == 0)
            .unwrap();
        assert_eq!(even.len(), 22);
        assert!(even[2..].iter().all(|t| t % 2 == 0), "{even:?}");
        // Sem filtro, o mesmo caminho dá o mesmo que generate_with_options.
        let all = config
            .generate_filtered(&params, &[1, 2], 20, &options, &mut |_, _| true)
            .unwrap();
        assert_eq!(
            all,
            config
                .generate_with_options(&params, &[1, 2], 20, &options)
                .unwrap()
        );
        // Tudo vetado: com stop_tokens, para neles em vez de travar.
        let stop = crate::GenerationOptions {
            stop_tokens: vec![0],
            ..options
        };
        let out = config
            .generate_filtered(&params, &[1, 2], 20, &stop, &mut |_, _| false)
            .unwrap();
        assert_eq!(out, vec![1, 2, 0]);
    }
}
