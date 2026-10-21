//! GPT de caracteres: transformer decoder-only, uma cabeça de atenção por camada.
//!
//! Os parâmetros são um `Vec<Tensor>` plano (o formato que `fit` e os otimizadores
//! usam); `GptConfig` define o layout. Cada sequência passa pelo grafo como uma
//! matriz `[t, d]`: embeddings e posições entram por `one_hot @ tabela`, sem
//! operação de indexação dedicada.
use std::collections::BTreeSet;

use crate::{Tensor, TensorError};

type Result<T> = std::result::Result<T, TensorError>;

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
pub struct GptConfig {
    pub vocab_size: usize,
    /// Comprimento máximo de contexto (tabela de posições).
    pub block_size: usize,
    /// Largura do modelo (d).
    pub embed_dim: usize,
    pub layers: usize,
}

// Layout de cada camada em `params`.
const LN1_G: usize = 0;
const LN1_B: usize = 1;
const WQ: usize = 2;
const WK: usize = 3;
const WV: usize = 4;
const WO: usize = 5;
const LN2_G: usize = 6;
const LN2_B: usize = 7;
const W1: usize = 8;
const B1: usize = 9;
const W2: usize = 10;
const B2: usize = 11;
const PER_LAYER: usize = 12;

impl GptConfig {
    fn validate(&self) -> Result<()> {
        if self.vocab_size == 0 || self.block_size == 0 || self.embed_dim == 0 {
            return Err(TensorError::InvalidShape(
                "vocab_size, block_size and embed_dim must be positive",
            ));
        }
        Ok(())
    }

    /// Quantidade de tensores em `params`: wte, wpe, camadas, LN final e cabeça.
    pub fn param_tensors(&self) -> usize {
        2 + self.layers * PER_LAYER + 3
    }

    /// Total de números treináveis.
    pub fn param_count(&self) -> usize {
        let (v, t, d) = (self.vocab_size, self.block_size, self.embed_dim);
        let layer = 4 * d + 4 * d * d + 4 * d * d + 4 * d + 4 * d * d + d;
        v * d + t * d + self.layers * layer + 2 * d + d * v
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
        for _ in 0..self.layers {
            params.push(Tensor::new(vec![1.0; d], &[d])?);
            params.push(Tensor::zeros(&[d])?);
            for _ in 0..3 {
                params.push(rng.tensor(&[d, d], std)?);
            }
            params.push(rng.tensor(&[d, d], residual_std)?);
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
        if params.len() != self.param_tensors() {
            return Err(TensorError::InvalidShape("params do not match GptConfig"));
        }
        Ok(())
    }

    /// Logits `[t, vocab]` para uma sequência de até `block_size` tokens.
    /// A posição i só enxerga tokens 0..=i (máscara causal).
    pub fn forward(&self, params: &[Tensor], tokens: &[usize]) -> Result<Tensor> {
        self.check_params(params)?;
        let t = tokens.len();
        if t == 0 || t > self.block_size {
            return Err(TensorError::InvalidShape(
                "sequence length must be in 1..=block_size",
            ));
        }
        let positions: Vec<usize> = (0..t).collect();
        let mut x = Tensor::one_hot(tokens, self.vocab_size)?
            .matmul(&params[0])?
            .add(&Tensor::one_hot(&positions, self.block_size)?.matmul(&params[1])?)?;
        let mask = Tensor::causal_mask(t)?;
        let scale = Tensor::scalar(1.0 / (self.embed_dim as f32).sqrt())?;
        for layer in 0..self.layers {
            let p = &params[2 + layer * PER_LAYER..2 + (layer + 1) * PER_LAYER];
            let h = x.layer_norm()?.mul(&p[LN1_G])?.add(&p[LN1_B])?;
            let (q, k, v) = (h.matmul(&p[WQ])?, h.matmul(&p[WK])?, h.matmul(&p[WV])?);
            let attention = q
                .matmul(&k.transpose()?)?
                .mul(&scale)?
                .add(&mask)?
                .softmax()?;
            x = x.add(&attention.matmul(&v)?.matmul(&p[WO])?)?;
            let h = x.layer_norm()?.mul(&p[LN2_G])?.add(&p[LN2_B])?;
            let mlp = h.matmul(&p[W1])?.add(&p[B1])?.gelu()?;
            x = x.add(&mlp.matmul(&p[W2])?.add(&p[B2])?)?;
        }
        let head = &params[2 + self.layers * PER_LAYER..];
        x.layer_norm()?
            .mul(&head[0])?
            .add(&head[1])?
            .matmul(&head[2])
    }

    /// Perda média de previsão do próximo caractere sobre um lote de janelas.
    /// Cada janela tem `len >= 2`: entrada `w[..n-1]`, alvo `w[1..]`.
    pub fn loss(&self, params: &[Tensor], windows: &[&[usize]]) -> Result<Tensor> {
        if windows.is_empty() {
            return Err(TensorError::InvalidShape("loss needs at least one window"));
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
    /// `temperature <= 0` escolhe sempre o mais provável. Usa as últimas
    /// `block_size` posições como contexto. Não registra gradientes úteis.
    pub fn generate(
        &self,
        params: &[Tensor],
        prompt: &[usize],
        count: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<Vec<usize>> {
        if prompt.is_empty() {
            return Err(TensorError::InvalidShape("prompt needs at least one token"));
        }
        let mut rng = SplitMix64(seed);
        let mut tokens = prompt.to_vec();
        for _ in 0..count {
            let start = tokens.len().saturating_sub(self.block_size);
            let logits = self.forward(params, &tokens[start..])?;
            let last = &logits.data()[logits.numel() - self.vocab_size..];
            tokens.push(sample(last, temperature, rng.uniform()));
        }
        Ok(tokens)
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
    fn tensor(&mut self, shape: &[usize], std: f32) -> Result<Tensor> {
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
