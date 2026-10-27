//! Katsu LM: modelo de linguagem de caracteres, transformer decoder-only com
//! atenção causal de várias cabeças.
//!
//! Os parâmetros são um `Vec<Tensor>` plano (o formato que `fit` e os otimizadores
//! usam); `KatsuLmConfig` define o layout. Cada sequência passa pelo grafo como uma
//! matriz `[t, d]`: embeddings e posições entram por `one_hot @ tabela`, sem
//! operação de indexação dedicada.
use std::{collections::BTreeSet, sync::Arc};

use crate::{Backend, CpuBackend, Tensor, TensorError};

type Result<T> = std::result::Result<T, TensorError>;

mod infer;

/// Vocabulário de caracteres, ordenado para ser reproduzível.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CharTokenizer {
    chars: Vec<char>,
}

impl CharTokenizer {
    /// Um token por caractere distinto do texto.
    pub fn from_text(text: &str) -> Self {
        let chars: BTreeSet<char> = text.chars().collect();
        Self {
            chars: chars.into_iter().collect(),
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.chars.len()
    }

    /// Falha com `IndexOutOfRange` em caracteres fora do vocabulário.
    pub fn encode(&self, text: &str) -> Result<Vec<usize>> {
        text.chars()
            .map(|c| {
                self.chars
                    .binary_search(&c)
                    .map_err(|_| TensorError::IndexOutOfRange {
                        index: c as usize,
                        classes: self.chars.len(),
                    })
            })
            .collect()
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .map(|&t| self.chars.get(t).copied().unwrap_or('\u{FFFD}'))
            .collect()
    }
}

/// Hiperparâmetros do modelo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KatsuLmConfig {
    pub vocab_size: usize,
    /// Comprimento máximo de contexto (tabela de posições).
    pub block_size: usize,
    /// Largura do modelo (d).
    pub embed_dim: usize,
    pub layers: usize,
    /// Cabeças de atenção; `embed_dim` precisa ser múltiplo delas.
    pub heads: usize,
}

// Layout de cada camada em `params`:
// [ln1_g, ln1_b, (wq, wk, wv, wo) × cabeças, ln2_g, ln2_b, w1, b1, w2, b2].
// Cada cabeça projeta d -> d/h; wo [d/h, d] devolve a soma ao resíduo, o que
// equivale a concatenar as cabeças e multiplicar por uma Wo [d, d].
// Com uma cabeça, o layout (e a sequência da inicialização) é o do v0.
const LN1_G: usize = 0;
const LN1_B: usize = 1;
const HEAD_START: usize = 2;
const PER_HEAD: usize = 4;

/// Configuração de amostragem; stop_tokens são IDs reais do tokenizer carregado.
#[derive(Debug, Clone)]
pub struct GenerationOptions {
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
    pub stop_tokens: Vec<usize>,
}
impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            repetition_penalty: 1.2,
            seed: 0,
            stop_tokens: Vec::new(),
        }
    }
}
/// Penaliza uma vez cada ID já observado; logits negativos ficam mais negativos.
pub fn penalize_repetitions(logits: &mut [f32], history: &[usize], penalty: f32) -> Result<()> {
    if !penalty.is_finite() || penalty < 1.0 {
        return Err(TensorError::InvalidShape(
            "repetition_penalty must be finite and >= 1",
        ));
    }
    for id in history.iter().copied().collect::<BTreeSet<_>>() {
        let value = logits.get_mut(id).ok_or(TensorError::InvalidShape(
            "history token outside vocabulary",
        ))?;
        *value = if *value < 0.0 {
            *value * penalty
        } else {
            *value / penalty
        };
    }
    Ok(())
}

impl KatsuLmConfig {
    fn validate(&self) -> Result<()> {
        if self.vocab_size == 0 || self.block_size == 0 || self.embed_dim == 0 {
            return Err(TensorError::InvalidShape(
                "vocab_size, block_size and embed_dim must be positive",
            ));
        }
        if self.heads == 0 || self.embed_dim % self.heads != 0 {
            return Err(TensorError::InvalidShape(
                "heads must be positive and divide embed_dim",
            ));
        }
        Ok(())
    }

    fn per_layer(&self) -> usize {
        HEAD_START + self.heads * PER_HEAD + 6
    }

    /// Quantidade de tensores em `params`: wte, wpe, camadas, LN final e cabeça.
    pub fn param_tensors(&self) -> usize {
        2 + self.layers * self.per_layer() + 3
    }

    /// Índices dos parâmetros que só aparecem como lado direito de matmul (Q, K, V e Wo de
    /// cada cabeça, as duas camadas do MLP e a cabeça de saída), na ordem da arena residente:
    /// os Wo de uma camada ficam vizinhos, para o concat das cabeças virar uma faixa só.
    /// Embeddings, LayerNorm e bias ficam fora: são usados por operações que rodam no host.
    pub fn resident_layout(&self) -> Vec<usize> {
        let mut layout = Vec::new();
        for layer in 0..self.layers {
            let base = 2 + layer * self.per_layer() + HEAD_START;
            for head in 0..self.heads {
                layout.extend((0..3).map(|i| base + head * PER_HEAD + i));
            }
            layout.extend((0..self.heads).map(|head| base + head * PER_HEAD + 3));
            let mlp = base + self.heads * PER_HEAD;
            layout.extend([mlp + 2, mlp + 4]);
        }
        layout.push(self.param_tensors() - 1);
        layout
    }

    /// Total de números treináveis.
    pub fn param_count(&self) -> usize {
        let (v, t, d) = (self.vocab_size, self.block_size, self.embed_dim);
        let layer = 4 * d + 4 * d * d + 4 * d * d + 4 * d + 4 * d * d + d;
        v * d + t * d + self.layers * layer + 2 * d + d * v
    }

    /// Shapes de `params`, na ordem de `init`, sem alocar os pesos.
    pub fn init_shapes(&self) -> Vec<Vec<usize>> {
        let (v, t, d) = (self.vocab_size, self.block_size, self.embed_dim);
        let dh = d / self.heads.max(1);
        let mut shapes = vec![vec![v, d], vec![t, d]];
        for _ in 0..self.layers {
            shapes.extend([vec![d], vec![d]]);
            for _ in 0..self.heads {
                shapes.extend([vec![d, dh], vec![d, dh], vec![d, dh], vec![dh, d]]);
            }
            shapes.extend([
                vec![d],
                vec![d],
                vec![d, 4 * d],
                vec![4 * d],
                vec![4 * d, d],
                vec![d],
            ]);
        }
        shapes.extend([vec![d], vec![d], vec![d, v]]);
        shapes
    }

    /// Inicialização reproduzível no estilo GPT-2: pesos ~ U(-a, a) com desvio 0.02,
    /// projeções de saída escaladas por 1/√(2·camadas), LayerNorm com γ=1 e β=0.
    pub fn init(&self, seed: u64) -> Result<Vec<Tensor>> {
        self.validate()?;
        let (v, t, d) = (self.vocab_size, self.block_size, self.embed_dim);
        let mut rng = SplitMix64(seed);
        let std = 0.02;
        let residual_std = std / (2.0 * self.layers.max(1) as f32).sqrt();
        let mut params = Vec::with_capacity(self.param_tensors());
        params.push(rng.tensor(&[v, d], std)?);
        params.push(rng.tensor(&[t, d], std)?);
        let dh = d / self.heads;
        for _ in 0..self.layers {
            params.push(Tensor::new(vec![1.0; d], &[d])?);
            params.push(Tensor::zeros(&[d])?);
            for _ in 0..self.heads {
                for _ in 0..3 {
                    params.push(rng.tensor(&[d, dh], std)?);
                }
                params.push(rng.tensor(&[dh, d], residual_std)?);
            }
            params.push(Tensor::new(vec![1.0; d], &[d])?);
            params.push(Tensor::zeros(&[d])?);
            params.push(rng.tensor(&[d, 4 * d], std)?);
            params.push(Tensor::zeros(&[4 * d])?);
            params.push(rng.tensor(&[4 * d, d], residual_std)?);
            params.push(Tensor::zeros(&[d])?);
        }
        params.push(Tensor::new(vec![1.0; d], &[d])?);
        params.push(Tensor::zeros(&[d])?);
        params.push(rng.tensor(&[d, v], std)?);
        Ok(params)
    }

    fn check_params(&self, params: &[Tensor]) -> Result<()> {
        self.validate()?;
        if params.len() != self.param_tensors() {
            return Err(TensorError::InvalidShape(
                "params do not match KatsuLmConfig",
            ));
        }
        Ok(())
    }

    /// ~139M com vocabulário 30k (embeddings e cabeça independentes).
    pub fn native_139m(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            block_size: 512,
            embed_dim: 768,
            layers: 13,
            heads: 12,
        }
    }

    /// ~349M com vocabulário 34816; sem alocar pesos ao criar o perfil.
    pub fn native_349m(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            block_size: 512,
            embed_dim: 1024,
            layers: 22,
            heads: 16,
        }
    }

    /// Logits `[t,vocab]`; preserva o layout dos checkpoints v0/v1.
    pub fn forward(&self, params: &[Tensor], tokens: &[usize]) -> Result<Tensor> {
        self.forward_batch(params, tokens, 1)?
            .reshape(&[tokens.len(), self.vocab_size])
    }

    /// Tokens achatados de um batch retangular; logits `[batch,t,vocab]`.
    pub fn forward_batch(
        &self,
        params: &[Tensor],
        tokens: &[usize],
        batch: usize,
    ) -> Result<Tensor> {
        self.forward_batch_with_backend(params, tokens, batch, Arc::new(CpuBackend::default()))
    }

    pub fn forward_batch_with_backend(
        &self,
        params: &[Tensor],
        tokens: &[usize],
        batch: usize,
        backend: Arc<dyn Backend>,
    ) -> Result<Tensor> {
        self.check_params(params)?;
        if batch == 0 || tokens.is_empty() || tokens.len() % batch != 0 {
            return Err(TensorError::InvalidShape(
                "batch must be positive and rectangular",
            ));
        }
        let t = tokens.len() / batch;
        if t > self.block_size {
            return Err(TensorError::InvalidShape("sequence exceeds block_size"));
        }
        let positions: Vec<usize> = (0..tokens.len()).map(|i| i % t).collect();
        let x = params[0]
            .gather_rows(tokens)?
            .add_with_backend(&params[1].gather_rows(&positions)?, backend.clone())?;
        self.forward_embedded(params, x, batch, t, backend)
    }

    /// Entrada mista do Arandu-Ve: as `n` primeiras posições de cada sequência recebem vetores
    /// (`prefix`, `[batch·n, d]`, ex.: a imagem projetada) no lugar do embedding de token; as
    /// demais, os `tokens` (`batch·m`, achatados). Toda posição soma o embedding de posição.
    /// Logits `[batch, n+m, vocab]`; o gradiente chega a `prefix`.
    pub fn forward_with_prefix(
        &self,
        params: &[Tensor],
        prefix: &Tensor,
        n: usize,
        tokens: &[usize],
        batch: usize,
        backend: Arc<dyn Backend>,
    ) -> Result<Tensor> {
        self.check_params(params)?;
        if n == 0 || batch == 0 || tokens.is_empty() || tokens.len() % batch != 0 {
            return Err(TensorError::InvalidShape(
                "prefix needs n > 0 and a rectangular token batch",
            ));
        }
        if prefix.shape() != [batch * n, self.embed_dim] {
            return Err(TensorError::InvalidShape(
                "prefix must be [batch·n, embed_dim]",
            ));
        }
        let m = tokens.len() / batch;
        let t = n + m;
        if t > self.block_size {
            return Err(TensorError::InvalidShape(
                "prefix + tokens exceed block_size",
            ));
        }
        // Empilha [prefixo; embeddings dos tokens] e reordena por sequência: o gather_rows
        // devolve o gradiente para cada origem, vetor ou linha da tabela de tokens.
        let stacked = Tensor::concat(&[prefix.clone(), params[0].gather_rows(tokens)?], 0)?;
        let order: Vec<usize> = (0..batch)
            .flat_map(|b| (b * n..(b + 1) * n).chain(batch * n + b * m..batch * n + (b + 1) * m))
            .collect();
        let positions: Vec<usize> = (0..batch * t).map(|i| i % t).collect();
        let x = stacked
            .gather_rows(&order)?
            .add_with_backend(&params[1].gather_rows(&positions)?, backend.clone())?;
        self.forward_embedded(params, x, batch, t, backend)
    }

    /// Perda do SFT com prefixo de vetores: média, entre os exemplos, da entropia cruzada só nos
    /// tokens da resposta (`ids[start..]`). As posições do prefixo nunca entram na perda. Cada
    /// exemplo usa as linhas `i·n..(i+1)·n` de `prefix`. Mesma conta de `SftBatch::with_prefix`.
    pub fn prefix_loss(
        &self,
        params: &[Tensor],
        prefix: &Tensor,
        n: usize,
        examples: &[(&[usize], usize)],
        backend: Arc<dyn Backend>,
    ) -> Result<Tensor> {
        if examples.is_empty()
            || examples
                .iter()
                .any(|(ids, start)| *start == 0 || *start >= ids.len())
        {
            return Err(TensorError::InvalidShape(
                "prefix SFT needs answers after a prompt",
            ));
        }
        if prefix.shape().first() != Some(&(examples.len() * n)) {
            return Err(TensorError::InvalidShape(
                "prefix rows must be examples × n",
            ));
        }
        let mut total: Option<Tensor> = None;
        for (i, (ids, start)) in examples.iter().enumerate() {
            let rows: Vec<usize> = (i * n..(i + 1) * n).collect();
            let inputs = &ids[..ids.len() - 1];
            let logits = self
                .forward_with_prefix(
                    params,
                    &prefix.gather_rows(&rows)?,
                    n,
                    inputs,
                    1,
                    backend.clone(),
                )?
                .reshape(&[n + inputs.len(), self.vocab_size])?;
            let answer: Vec<usize> = (n + start - 1..n + ids.len() - 1).collect();
            let ce = logits
                .gather_rows(&answer)?
                .cross_entropy_with_backend(&ids[*start..], backend.clone())?;
            total = Some(match total {
                Some(sum) => sum.add(&ce)?,
                None => ce,
            });
        }
        total
            .expect("at least one example")
            .mul(&Tensor::scalar(1.0 / examples.len() as f32)?)
    }

    /// Camadas e cabeça sobre embeddings já montados (`x`, `[batch·t, d]`).
    fn forward_embedded(
        &self,
        params: &[Tensor],
        mut x: Tensor,
        batch: usize,
        t: usize,
        backend: Arc<dyn Backend>,
    ) -> Result<Tensor> {
        let add = |a: &Tensor, b: &Tensor| a.add_with_backend(b, backend.clone());
        let dh = self.embed_dim / self.heads;
        let mm = |a: &Tensor, b: &Tensor| a.matmul_with_backend(b, backend.clone());
        for layer in 0..self.layers {
            let p = &params[2 + layer * self.per_layer()..2 + (layer + 1) * self.per_layer()];
            let h = x.layer_norm_affine(&p[LN1_G], &p[LN1_B])?;
            // Σ_h values_h·Wo_h = [values_1 … values_H]·[Wo_1; …; Wo_H]: uma projeção só,
            // sem guardar uma saída [B·T, d] e uma soma residual por cabeça.
            let mut heads = Vec::with_capacity(self.heads);
            let mut out_weights = Vec::with_capacity(self.heads);
            let per_head = p[HEAD_START..HEAD_START + self.heads * PER_HEAD].chunks_exact(PER_HEAD);
            // Q, K e V de todas as cabeças numa chamada: a entrada h sobe uma vez só.
            let projections: Vec<Tensor> = per_head.clone().flat_map(|w| w[..3].to_vec()).collect();
            let qkv = h.matmul_many_with_backend(&projections, backend.clone())?;
            for (head, w) in per_head.enumerate() {
                let part = |j: usize| {
                    qkv.columns((3 * head + j) * dh, dh)?
                        .reshape(&[batch, t, dh])
                };
                let (q, k, v) = (part(0)?, part(1)?, part(2)?);
                let attention = q.causal_attention_with_backend(&k, backend.clone())?;
                heads.push(
                    attention
                        .bmm_with_backend(&v, backend.clone())?
                        .reshape(&[batch * t, dh])?,
                );
                out_weights.push(w[3].clone());
            }
            let projection = mm(
                &Tensor::concat(&heads, 1)?,
                &Tensor::concat(&out_weights, 0)?,
            )?;
            drop(heads);
            x = add(&x, &projection)?;
            let p = &p[HEAD_START + self.heads * PER_HEAD..];
            let h = x.layer_norm_affine(&p[0], &p[1])?;
            let mlp = h
                .linear_with_backend(&p[2], &p[3], backend.clone())?
                .gelu()?;
            x = add(&x, &mlp.linear_with_backend(&p[4], &p[5], backend.clone())?)?;
        }
        let head = &params[2 + self.layers * self.per_layer()..];
        let normalized = x.layer_norm_affine(&head[0], &head[1])?;
        mm(&normalized, &head[2])?.reshape(&[batch, t, self.vocab_size])
    }

    /// Perda de minibatch retangular; uma única projeção linear para B×T tokens.
    pub fn loss_batch_with_backend(
        &self,
        params: &[Tensor],
        windows: &[&[usize]],
        backend: Arc<dyn Backend>,
    ) -> Result<Tensor> {
        let len = windows.first().map_or(0, |w| w.len());
        if len < 2 || windows.iter().any(|w| w.len() != len) {
            return Err(TensorError::InvalidShape(
                "batch needs equal-length windows >= 2",
            ));
        }
        let inputs: Vec<usize> = windows
            .iter()
            .flat_map(|w| w[..len - 1].iter().copied())
            .collect();
        let targets: Vec<usize> = windows
            .iter()
            .flat_map(|w| w[1..].iter().copied())
            .collect();
        self.forward_batch_with_backend(params, &inputs, windows.len(), backend.clone())?
            .reshape(&[targets.len(), self.vocab_size])?
            .cross_entropy_with_backend(&targets, backend)
    }

    /// Perda média de previsão do próximo caractere sobre um lote de janelas.
    /// Cada janela tem `len >= 2`: entrada `w[..n-1]`, alvo `w[1..]`.
    pub fn loss(&self, params: &[Tensor], windows: &[&[usize]]) -> Result<Tensor> {
        if windows.is_empty() {
            return Err(TensorError::InvalidShape("loss needs at least one window"));
        }
        if windows.iter().all(|w| w.len() == windows[0].len()) {
            return self.loss_batch_with_backend(params, windows, Arc::new(CpuBackend::default()));
        }
        let mut total: Option<Tensor> = None;
        for window in windows {
            if window.len() < 2 {
                return Err(TensorError::InvalidShape("windows need at least 2 tokens"));
            }
            let n = window.len() - 1;
            let loss = self
                .forward(params, &window[..n])?
                .cross_entropy(&window[1..])?;
            total = Some(match total {
                Some(sum) => sum.add(&loss)?,
                None => loss,
            });
        }
        total
            .expect("at least one window")
            .mul(&Tensor::scalar(1.0 / windows.len() as f32)?)
    }

    /// Continua `prompt` por `count` tokens, amostrando com temperatura.
    /// `temperature <= 0` escolhe sempre o mais provável. O contexto cabe em
    /// `block_size` posições; quando enche, recomeça dos últimos `block_size/2`
    /// (arredondado para cima) tokens. Não registra gradientes úteis.
    pub fn generate(
        &self,
        params: &[Tensor],
        prompt: &[usize],
        count: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<Vec<usize>> {
        self.generate_with_options(
            params,
            prompt,
            count,
            &GenerationOptions {
                temperature,
                seed,
                repetition_penalty: 1.0,
                stop_tokens: Vec::new(),
            },
        )
    }

    pub fn generate_with_options(
        &self,
        params: &[Tensor],
        prompt: &[usize],
        count: usize,
        options: &GenerationOptions,
    ) -> Result<Vec<usize>> {
        self.generate_filtered(params, prompt, count, options, &mut |_, _| true)
    }

    /// Como `generate_with_options`, mas `allow(gerados, candidato)` pode vetar tokens: o vetado
    /// sai da distribuição e a amostragem se repete (decodificação restrita). Se `allow` vetar
    /// `MAX_VETOES` candidatos seguidos, o passo para em `stop_tokens` (ou, sem eles, aceita o
    /// último candidato), para nunca travar.
    pub fn generate_filtered(
        &self,
        params: &[Tensor],
        prompt: &[usize],
        count: usize,
        options: &GenerationOptions,
        allow: &mut dyn FnMut(&[usize], usize) -> bool,
    ) -> Result<Vec<usize>> {
        const MAX_VETOES: usize = 512;
        if prompt.is_empty() || !options.temperature.is_finite() {
            return Err(TensorError::InvalidShape(
                "nonempty prompt and finite temperature required",
            ));
        }
        let mut probe = Vec::new();
        penalize_repetitions(&mut probe, &[], options.repetition_penalty)?;
        let mut rng = SplitMix64(options.seed);
        let mut tokens = prompt.to_vec();
        // KV cache: cada passo processa só os tokens que o cache ainda não viu. As posições
        // são absolutas, então deslizar a janela invalida o cache; quando ela enche, a janela
        // pula para os últimos `keep` tokens e o prefill é refeito uma vez, não a cada token.
        let keep = self.block_size - self.block_size / 2;
        let mut cache = infer::KvCache::new(self);
        let mut start = 0;
        for _ in 0..count {
            if tokens.len() - start > self.block_size {
                start = tokens.len() - keep;
                cache.clear();
            }
            let seen = start + cache.len();
            let mut last = self.forward_cached(params, &mut cache, &tokens[seen..])?;
            penalize_repetitions(&mut last, &tokens[start..], options.repetition_penalty)?;
            let mut next = sample(&last, options.temperature, rng.uniform());
            let mut vetoes = 0;
            while !allow(&tokens[prompt.len()..], next) {
                vetoes += 1;
                if vetoes >= MAX_VETOES {
                    if !options.stop_tokens.is_empty() {
                        tokens.extend_from_slice(&options.stop_tokens);
                        return Ok(tokens);
                    }
                    break;
                }
                last[next] = f32::NEG_INFINITY;
                next = sample(&last, options.temperature, rng.uniform());
            }
            tokens.push(next);
            if !options.stop_tokens.is_empty()
                && tokens[prompt.len()..].ends_with(&options.stop_tokens)
            {
                break;
            }
        }
        Ok(tokens)
    }

    /// Acrescenta entradas e colunas de saída preservando os logits do vocabulário antigo.
    pub fn extend_vocab(
        &self,
        params: &[Tensor],
        vocab_size: usize,
        seed: u64,
    ) -> Result<(Self, Vec<Tensor>)> {
        if vocab_size < self.vocab_size {
            return Err(TensorError::InvalidShape("vocabulary must not shrink"));
        }
        let (_, mut next) = self.extend_context(params, self.block_size, seed)?;
        let mut rng = SplitMix64(seed);
        let mut embedding = rng
            .tensor(&[vocab_size, self.embed_dim], 0.02)?
            .data()
            .to_vec();
        embedding[..params[0].numel()].copy_from_slice(params[0].data());
        next[0] = Tensor::new(embedding, &[vocab_size, self.embed_dim])?;
        let last = params.len() - 1;
        let mut head = rng
            .tensor(&[self.embed_dim, vocab_size], 0.02)?
            .data()
            .to_vec();
        for row in 0..self.embed_dim {
            head[row * vocab_size..row * vocab_size + self.vocab_size].copy_from_slice(
                &params[last].data()[row * self.vocab_size..(row + 1) * self.vocab_size],
            );
        }
        next[last] = Tensor::new(head, &[self.embed_dim, vocab_size])?;
        let mut config = *self;
        config.vocab_size = vocab_size;
        Ok((config, next))
    }

    /// Expande só a tabela posicional. Preserva exatamente os outros pesos e as posições antigas.
    pub fn extend_context(
        &self,
        params: &[Tensor],
        block_size: usize,
        seed: u64,
    ) -> Result<(Self, Vec<Tensor>)> {
        self.validate()?;
        if block_size < self.block_size {
            return Err(TensorError::InvalidShape("new context must not shrink"));
        }
        let expected = self.init(seed)?;
        if params.len() != expected.len()
            || params
                .iter()
                .zip(&expected)
                .any(|(p, e)| p.shape() != e.shape())
        {
            return Err(TensorError::InvalidShape(
                "checkpoint does not match source config",
            ));
        }
        let mut config = *self;
        config.block_size = block_size;
        let mut migrated = params
            .iter()
            .map(|p| Tensor::new(p.data().to_vec(), p.shape()))
            .collect::<Result<Vec<_>>>()?;
        let mut rng = SplitMix64(seed);
        let mut positions = rng
            .tensor(&[block_size, self.embed_dim], 0.02)?
            .data()
            .to_vec();
        positions[..params[1].numel()].copy_from_slice(params[1].data());
        migrated[1] = Tensor::new(positions, &[block_size, self.embed_dim])?;
        Ok((config, migrated))
    }
}

/// Amostra um índice de logits; `u` em [0, 1).
fn sample(logits: &[f32], temperature: f32, u: f32) -> usize {
    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i);
    if temperature <= 0.0 {
        return argmax;
    }
    let max = logits[argmax];
    let weights: Vec<f32> = logits
        .iter()
        .map(|l| ((l - max) / temperature).exp())
        .collect();
    let mut target = u * weights.iter().sum::<f32>();
    for (i, w) in weights.iter().enumerate() {
        target -= w;
        if target <= 0.0 {
            return i;
        }
    }
    argmax
}

/// SplitMix64: PRNG local e reproduzível; não serve para criptografia.
#[derive(Debug, Clone)]
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniforme em [0, 1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16_777_216.0
    }

    /// Inteiro uniforme em [0, n); n > 0.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Uniforme em [-a, a] com desvio-padrão `std` (a = std·√3).
    pub(crate) fn tensor(&mut self, shape: &[usize], std: f32) -> Result<Tensor> {
        let bound = std * 3f32.sqrt();
        let count = shape.iter().product();
        let data = (0..count)
            .map(|_| (2.0 * self.uniform() - 1.0) * bound)
            .collect();
        Tensor::new(data, shape)
    }
}

const CHECKPOINT_MAGIC: &[u8; 8] = b"KATSUCK1";

/// Serializa parâmetros: magic, quantidade, e por tensor rank, dims (u64) e dados f32,
/// tudo little-endian. Independe da plataforma.
pub fn save_params(params: &[Tensor], writer: &mut impl std::io::Write) -> std::io::Result<()> {
    writer.write_all(CHECKPOINT_MAGIC)?;
    writer.write_all(&(params.len() as u64).to_le_bytes())?;
    for tensor in params {
        writer.write_all(&(tensor.rank() as u64).to_le_bytes())?;
        for &dim in tensor.shape() {
            writer.write_all(&(dim as u64).to_le_bytes())?;
        }
        for value in tensor.data() {
            writer.write_all(&value.to_le_bytes())?;
        }
    }
    Ok(())
}

/// Lê o formato de `save_params`. Rejeita arquivos truncados ou com magic errado.
pub fn load_params(reader: &mut impl std::io::Read) -> std::io::Result<Vec<Tensor>> {
    use std::io::{Error, ErrorKind};
    let invalid = |message: &str| Error::new(ErrorKind::InvalidData, message.to_string());
    let mut magic = [0; 8];
    reader.read_exact(&mut magic)?;
    if &magic != CHECKPOINT_MAGIC {
        return Err(invalid("not a Katsu checkpoint"));
    }
    let mut word = [0; 8];
    let mut read_u64 = |reader: &mut dyn std::io::Read| -> std::io::Result<usize> {
        reader.read_exact(&mut word)?;
        usize::try_from(u64::from_le_bytes(word)).map_err(|_| invalid("value too large"))
    };
    let count = read_u64(reader)?;
    let mut params = Vec::new();
    for _ in 0..count {
        let rank = read_u64(reader)?;
        if rank > 8 {
            return Err(invalid("unsupported tensor rank"));
        }
        let shape = (0..rank)
            .map(|_| read_u64(reader))
            .collect::<std::io::Result<Vec<_>>>()?;
        let numel = crate::tensor::element_count(&shape).map_err(|e| invalid(&e.to_string()))?;
        // Reserva falível: um cabeçalho corrompido não pode abortar o processo.
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(numel * 4)
            .map_err(|_| invalid("tensor too large"))?;
        bytes.resize(numel * 4, 0);
        reader.read_exact(&mut bytes)?;
        let data = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        params.push(Tensor::new(data, &shape).map_err(|e| invalid(&e.to_string()))?);
    }
    Ok(params)
}
