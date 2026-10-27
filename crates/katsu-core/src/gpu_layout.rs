//! Layout dos parâmetros do Katsu LM numa arena plana para o treino inteiro na GPU.
//!
//! O checkpoint guarda Q, K, V e Wo de cada cabeça como tensores separados. Na GPU eles
//! viram uma matriz Wqkv `[d, 3d]` por camada (colunas `[q_0..q_H | k_0..k_H | v_0..v_H]`)
//! e uma Wo `[d, d]` (linhas da cabeça h em `h·dh..`), para que cada projeção seja um GEMM
//! só. Cada tensor do host vira um retângulo na arena: elemento `(r, c)` em
//! `offset + r·ld + c`. A mesma permutação serve para pesos, gradientes e momentos do Adam.
//! Sem dependência de CUDA: os testes rodam em qualquer máquina.
use crate::{KatsuLmConfig, Tensor, TensorError};

type Result<T> = std::result::Result<T, TensorError>;

/// Onde um tensor do host mora na arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slot {
    pub offset: usize,
    pub rows: usize,
    pub cols: usize,
    pub ld: usize,
}

/// Deslocamentos (em floats) dos blocos de uma camada na arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerSlots {
    pub ln1_g: usize,
    pub ln1_b: usize,
    pub wqkv: usize,
    pub wo: usize,
    pub ln2_g: usize,
    pub ln2_b: usize,
    pub w1: usize,
    pub b1: usize,
    pub w2: usize,
    pub b2: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuLayout {
    pub config: KatsuLmConfig,
    pub wte: usize,
    pub wpe: usize,
    pub layers: Vec<LayerSlots>,
    pub lnf_g: usize,
    pub lnf_b: usize,
    pub head: usize,
    /// Total de floats da arena (igual a `config.param_count()`).
    pub total: usize,
    /// Um slot por tensor do host, na ordem de `KatsuLmConfig::init`.
    pub slots: Vec<Slot>,
}

impl GpuLayout {
    pub fn new(config: KatsuLmConfig) -> Result<Self> {
        let (v, t, d, h) = (
            config.vocab_size,
            config.block_size,
            config.embed_dim,
            config.heads,
        );
        if v == 0 || t == 0 || d == 0 || h == 0 || d % h != 0 {
            return Err(TensorError::InvalidShape(
                "invalid KatsuLmConfig for GPU layout",
            ));
        }
        let dh = d / h;
        let mut next = 0;
        let mut take = |n: usize| {
            let at = next;
            next += n;
            at
        };
        let wte = take(v * d);
        let wpe = take(t * d);
        let mut layers = Vec::with_capacity(config.layers);
        for _ in 0..config.layers {
            layers.push(LayerSlots {
                ln1_g: take(d),
                ln1_b: take(d),
                wqkv: take(d * 3 * d),
                wo: take(d * d),
                ln2_g: take(d),
                ln2_b: take(d),
                w1: take(d * 4 * d),
                b1: take(4 * d),
                w2: take(4 * d * d),
                b2: take(d),
            });
        }
        let lnf_g = take(d);
        let lnf_b = take(d);
        let head = take(d * v);
        let total = next;

        let flat = |offset, n| Slot {
            offset,
            rows: 1,
            cols: n,
            ld: n,
        };
        let matrix = |offset, rows, cols| Slot {
            offset,
            rows,
            cols,
            ld: cols,
        };
        let mut slots = vec![matrix(wte, v, d), matrix(wpe, t, d)];
        for l in &layers {
            slots.push(flat(l.ln1_g, d));
            slots.push(flat(l.ln1_b, d));
            for head in 0..h {
                for j in 0..3 {
                    slots.push(Slot {
                        offset: l.wqkv + j * d + head * dh,
                        rows: d,
                        cols: dh,
                        ld: 3 * d,
                    });
                }
                slots.push(matrix(l.wo + head * dh * d, dh, d));
            }
            slots.push(flat(l.ln2_g, d));
            slots.push(flat(l.ln2_b, d));
            slots.push(matrix(l.w1, d, 4 * d));
            slots.push(flat(l.b1, 4 * d));
            slots.push(matrix(l.w2, 4 * d, d));
            slots.push(flat(l.b2, d));
        }
        slots.push(flat(lnf_g, d));
        slots.push(flat(lnf_b, d));
        slots.push(matrix(head, d, v));
        debug_assert_eq!(slots.len(), config.param_tensors());
        Ok(Self {
            config,
            wte,
            wpe,
            layers,
            lnf_g,
            lnf_b,
            head,
            total,
            slots,
        })
    }

    fn check(&self, sizes: impl ExactSizeIterator<Item = usize>) -> Result<()> {
        if sizes.len() != self.slots.len() {
            return Err(TensorError::InvalidShape("params do not match GPU layout"));
        }
        for (size, slot) in sizes.zip(&self.slots) {
            if size != slot.rows * slot.cols {
                return Err(TensorError::InvalidShape(
                    "param size does not match GPU layout",
                ));
            }
        }
        Ok(())
    }

    /// Junta buffers do host (um por tensor, na ordem do checkpoint) na arena.
    pub fn pack(&self, parts: &[&[f32]]) -> Result<Vec<f32>> {
        self.check(parts.iter().map(|p| p.len()))?;
        let mut out = vec![0.0; self.total];
        for (part, slot) in parts.iter().zip(&self.slots) {
            for r in 0..slot.rows {
                let at = slot.offset + r * slot.ld;
                out[at..at + slot.cols].copy_from_slice(&part[r * slot.cols..(r + 1) * slot.cols]);
            }
        }
        Ok(out)
    }

    pub fn pack_params(&self, params: &[Tensor]) -> Result<Vec<f32>> {
        let parts: Vec<&[f32]> = params.iter().map(Tensor::data).collect();
        self.pack(&parts)
    }

    /// Separa a arena em buffers do host, um por tensor, na ordem do checkpoint.
    pub fn unpack(&self, arena: &[f32]) -> Result<Vec<Vec<f32>>> {
        if arena.len() != self.total {
            return Err(TensorError::DataLength {
                expected: self.total,
                actual: arena.len(),
            });
        }
        Ok(self
            .slots
            .iter()
            .map(|slot| {
                let mut part = Vec::with_capacity(slot.rows * slot.cols);
                for r in 0..slot.rows {
                    let at = slot.offset + r * slot.ld;
                    part.extend_from_slice(&arena[at..at + slot.cols]);
                }
                part
            })
            .collect())
    }

    /// Tensores com os shapes de `KatsuLmConfig::init`.
    pub fn unpack_params(&self, arena: &[f32]) -> Result<Vec<Tensor>> {
        let shapes = self.config.init_shapes();
        self.unpack(arena)?
            .into_iter()
            .zip(shapes)
            .map(|(data, shape)| Tensor::new(data, &shape))
            .collect()
    }
}

/// Um lote de SFT pronto para a GPU, com a mesma perda do `batch_loss` do train_sft: média
/// por exemplo das entropias cruzadas da resposta, e média entre os exemplos. Cada exemplo é
/// completado com o token 0 até o maior do lote; a linha r tem peso `1/(B·n_e)` se prevê um
/// token da resposta do exemplo e, senão, peso 0.
#[derive(Debug, Clone, PartialEq)]
pub struct SftBatch {
    pub batch: usize,
    pub width: usize,
    pub tokens: Vec<usize>,
    pub targets: Vec<usize>,
    pub weights: Vec<f32>,
    /// Tokens reais de entrada (sem o padding), para medir tokens/s.
    pub input_tokens: usize,
    /// Arandu-Ve: quantas posições do começo de cada sequência recebem vetores (0 = nenhuma).
    pub prefix_len: usize,
    /// Vetores do prefixo, `[batch·prefix_len·d]`, em ordem de sequência.
    pub prefix: Vec<f32>,
}

impl SftBatch {
    /// `examples`: (ids, início da resposta), com `0 < answer_start < ids.len()`.
    pub fn new(examples: &[(&[usize], usize)]) -> Result<Self> {
        if examples.is_empty()
            || examples
                .iter()
                .any(|(ids, start)| *start == 0 || *start >= ids.len())
        {
            return Err(TensorError::InvalidShape(
                "SFT batch needs answers after a prompt",
            ));
        }
        let batch = examples.len();
        let width = examples
            .iter()
            .map(|(ids, _)| ids.len() - 1)
            .max()
            .unwrap_or(0);
        let mut tokens = vec![0; batch * width];
        let mut targets = vec![0; batch * width];
        let mut weights = vec![0.0; batch * width];
        let mut input_tokens = 0;
        for (i, (ids, start)) in examples.iter().enumerate() {
            let n = ids.len() - 1;
            input_tokens += n;
            tokens[i * width..i * width + n].copy_from_slice(&ids[..n]);
            let answer = ids.len() - start;
            let weight = 1.0 / (batch as f32 * answer as f32);
            for r in start - 1..n {
                targets[i * width + r] = ids[r + 1];
                weights[i * width + r] = weight;
            }
        }
        Ok(Self {
            batch,
            width,
            tokens,
            targets,
            weights,
            input_tokens,
            prefix_len: 0,
            prefix: Vec::new(),
        })
    }

    /// Lote do Arandu-Ve: cada exemplo começa com `n` vetores (`prefix`, `[exemplos·n·d]`) e
    /// segue com os tokens; só a resposta tem peso, como em `new` (mesma conta de
    /// `KatsuLmConfig::prefix_loss`). As posições do prefixo levam o token 0, que não é lido, e
    /// peso 0: nunca entram na perda.
    pub fn with_prefix(
        examples: &[(&[usize], usize)],
        prefix: &[f32],
        n: usize,
        d: usize,
    ) -> Result<Self> {
        if n == 0 || d == 0 || prefix.len() != examples.len() * n * d {
            return Err(TensorError::InvalidShape(
                "prefix must be examples × n × d with n, d > 0",
            ));
        }
        let mut batch = Self::new(examples)?;
        let (b, text) = (batch.batch, batch.width);
        let width = n + text;
        let shift = |v: &[usize]| -> Vec<usize> {
            (0..b)
                .flat_map(|i| {
                    std::iter::repeat_n(0, n).chain(v[i * text..(i + 1) * text].iter().copied())
                })
                .collect()
        };
        batch.tokens = shift(&batch.tokens);
        batch.targets = shift(&batch.targets);
        batch.weights = (0..b)
            .flat_map(|i| {
                std::iter::repeat_n(0.0, n)
                    .chain(batch.weights[i * text..(i + 1) * text].iter().copied())
            })
            .collect();
        batch.width = width;
        batch.prefix_len = n;
        batch.prefix = prefix.to_vec();
        Ok(batch)
    }

    /// Janelas de mesmo tamanho do pré-treino (`loss_batch_with_backend`): todas as linhas
    /// contam, peso `1/(B·T)`.
    pub fn windows(windows: &[&[usize]]) -> Result<Self> {
        let len = windows.first().map_or(0, |w| w.len());
        if len < 2 || windows.iter().any(|w| w.len() != len) {
            return Err(TensorError::InvalidShape(
                "batch needs equal-length windows >= 2",
            ));
        }
        let (batch, width) = (windows.len(), len - 1);
        let weight = 1.0 / (batch * width) as f32;
        Ok(Self {
            batch,
            width,
            tokens: windows
                .iter()
                .flat_map(|w| w[..width].iter().copied())
                .collect(),
            targets: windows
                .iter()
                .flat_map(|w| w[1..].iter().copied())
                .collect(),
            weights: vec![weight; batch * width],
            input_tokens: batch * width,
            prefix_len: 0,
            prefix: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> KatsuLmConfig {
        KatsuLmConfig {
            vocab_size: 11,
            block_size: 7,
            embed_dim: 12,
            layers: 2,
            heads: 3,
        }
    }

    #[test]
    fn every_parameter_lands_once_and_round_trips() {
        let c = tiny();
        let layout = GpuLayout::new(c).unwrap();
        assert_eq!(layout.total, c.param_count());
        let params = c.init(5).unwrap();
        let shapes: Vec<Vec<usize>> = params.iter().map(|p| p.shape().to_vec()).collect();
        assert_eq!(c.init_shapes(), shapes);
        // marca cada elemento com um valor único para pegar sobreposição ou buraco
        let marked: Vec<Tensor> = params
            .iter()
            .scan(0.0f32, |n, p| {
                let data: Vec<f32> = (0..p.numel())
                    .map(|_| {
                        *n += 1.0;
                        *n
                    })
                    .collect();
                Some(Tensor::new(data, p.shape()).unwrap())
            })
            .collect();
        let arena = layout.pack_params(&marked).unwrap();
        let mut seen: Vec<f32> = arena.clone();
        seen.sort_by(f32::total_cmp);
        let expected: Vec<f32> = (1..=layout.total).map(|i| i as f32).collect();
        assert_eq!(seen, expected);
        let back = layout.unpack_params(&arena).unwrap();
        for (a, b) in marked.iter().zip(&back) {
            assert_eq!(a.shape(), b.shape());
            assert_eq!(a.data(), b.data());
        }
    }

    #[test]
    fn heads_are_packed_as_qkv_columns_and_wo_rows() {
        let c = tiny();
        let layout = GpuLayout::new(c).unwrap();
        let (d, dh) = (12, 4);
        let l = layout.layers[0];
        // camada 0 começa no índice 2: ln1_g, ln1_b, depois (wq, wk, wv, wo) da cabeça 0
        assert_eq!(
            layout.slots[4 + 1],
            Slot {
                offset: l.wqkv + d,
                rows: d,
                cols: dh,
                ld: 3 * d
            }
        );
        let head2_v = 4 + 2 * 4 + 2;
        assert_eq!(layout.slots[head2_v].offset, l.wqkv + 2 * d + 2 * dh);
        let head1_o = 4 + 4 + 3;
        assert_eq!(
            layout.slots[head1_o],
            Slot {
                offset: l.wo + dh * d,
                rows: dh,
                cols: d,
                ld: d
            }
        );
    }

    #[test]
    fn sft_batch_weights_match_the_mean_of_example_means() {
        let a = [5, 6, 7, 8, 9];
        let b = [1, 2, 3];
        let batch = SftBatch::new(&[(&a, 3), (&b, 1)]).unwrap();
        assert_eq!((batch.batch, batch.width, batch.input_tokens), (2, 4, 6));
        assert_eq!(batch.tokens, vec![5, 6, 7, 8, 1, 2, 0, 0]);
        // exemplo a: linhas 2 e 3 preveem 8 e 9 (resposta de 2 tokens) -> peso 1/(2·2)
        assert_eq!(batch.targets, vec![0, 0, 8, 9, 2, 3, 0, 0]);
        assert_eq!(
            batch.weights,
            vec![0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.0, 0.0]
        );
        let total: f32 = batch.weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }
}
