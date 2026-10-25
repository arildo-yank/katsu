//! Paridade do treino inteiro na GPU com o autograd da CPU: perda, gradiente de cada tensor
//! e pesos depois de um passo do Adam. Sem GPU (ou fora do Linux) os testes são pulados.
#![cfg(feature = "cuda")]
use katsu_core::{
    Adam, GemmKernel, GpuTrainOptions, GpuTrainer, KatsuLmConfig, Optimizer, SftBatch, Tensor,
    TensorError,
};

type Example<'a> = (&'a [usize], usize);

fn trainer(
    c: KatsuLmConfig,
    params: &[Tensor],
    adam: &Adam,
    batch: usize,
    options: GpuTrainOptions,
) -> Option<GpuTrainer> {
    match GpuTrainer::new(c, params, adam, batch, c.block_size, 0, options) {
        Ok(t) => Some(t),
        Err(TensorError::BackendUnavailable(why)) => {
            eprintln!("sem GPU, teste pulado: {why}");
            None
        }
        Err(e) => panic!("GpuTrainer::new: {e}"),
    }
}

/// A mesma perda do `batch_loss` do train_sft, pelo autograd da CPU.
fn cpu_loss(c: &KatsuLmConfig, p: &[Tensor], examples: &[Example]) -> Tensor {
    let b = examples.len();
    let width = examples.iter().map(|(ids, _)| ids.len() - 1).max().unwrap();
    let mut inputs = Vec::new();
    for (ids, _) in examples {
        inputs.extend_from_slice(&ids[..ids.len() - 1]);
        inputs.resize(inputs.len() + width + 1 - ids.len(), 0);
    }
    let logits = c
        .forward_batch(p, &inputs, b)
        .unwrap()
        .reshape(&[b * width, c.vocab_size])
        .unwrap();
    let mut total: Option<Tensor> = None;
    for (i, (ids, start)) in examples.iter().enumerate() {
        let rows: Vec<usize> = (start - 1..ids.len() - 1).map(|r| i * width + r).collect();
        let ce = logits
            .gather_rows(&rows)
            .unwrap()
            .cross_entropy(&ids[*start..])
            .unwrap();
        total = Some(match total {
            Some(sum) => sum.add(&ce).unwrap(),
            None => ce,
        });
    }
    total
        .unwrap()
        .mul(&Tensor::scalar(1.0 / b as f32).unwrap())
        .unwrap()
}

fn tokens(seed: u64, n: usize, vocab: usize) -> Vec<usize> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as usize % vocab
        })
        .collect()
}

/// max |a−b| ≤ rtol·max|b| + atol, tensor a tensor.
fn assert_close(name: &str, gpu: &[f32], cpu: &[f32], rtol: f32, atol: f32) {
    assert_eq!(gpu.len(), cpu.len(), "{name}: tamanhos");
    let scale = cpu.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let diff = gpu
        .iter()
        .zip(cpu)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    if std::env::var_os("KATSU_PARITY_VERBOSE").is_some() {
        eprintln!("{name}: max|Δ| {diff:e}, max|cpu| {scale:e}");
    }
    assert!(
        diff <= rtol * scale + atol,
        "{name}: diferença {diff:e} acima de {rtol}·{scale:e} + {atol:e}"
    );
}

/// Tolerâncias relativas: FP32 (`STRICT`) ou produtos em TF32 (`TF32`, 10 bits de mantissa).
#[derive(Clone, Copy)]
struct Tol {
    loss: f64,
    grad: f32,
    /// fração máxima de pesos que podem divergir no 1º passo do Adam (sinal de gradiente ~0)
    flips: f64,
}
const STRICT: Tol = Tol {
    loss: 1e-5,
    grad: 1e-5,
    flips: 1e-3,
};
const TF32: Tol = Tol {
    loss: 2e-3,
    grad: 3e-2,
    flips: 5e-2,
};

fn check_parity(c: KatsuLmConfig, options: GpuTrainOptions) {
    let tol = if options.gemm == GemmKernel::Tf32 {
        TF32
    } else {
        STRICT
    };
    let params = c.init(11).unwrap();
    let (a, b, d) = (
        tokens(1, 11, c.vocab_size),
        tokens(2, 7, c.vocab_size),
        tokens(3, 13, c.vocab_size),
    );
    let examples: Vec<Example> = vec![(&a, 4), (&b, 2), (&d, 9)];
    let batch = SftBatch::new(&examples).unwrap();
    let lr = 1e-3;
    let Some(mut gpu) = trainer(c, &params, &Adam::new(lr).unwrap(), 4, options) else {
        return;
    };

    // perda só de forward não mexe nos gradientes
    let probe = gpu.loss(&batch).unwrap();
    let loss = gpu.loss_and_grad(&batch).unwrap();
    assert!(
        (probe - loss).abs() <= 1e-6 * loss.abs(),
        "forward {probe} vs treino {loss}"
    );

    let cpu = cpu_loss(&c, &params, &examples);
    let expected = f64::from(cpu.data()[0]);
    assert!(
        (loss - expected).abs() <= tol.loss * expected.abs(),
        "perda GPU {loss} vs CPU {expected}"
    );
    cpu.backward().unwrap();
    let grads = gpu.grads().unwrap();
    for (i, (p, g)) in params.iter().zip(&grads).enumerate() {
        let host = p.grad().unwrap();
        let host = host.as_ref().expect("gradiente da CPU");
        assert_close(
            &format!("grad[{i}] {:?}", p.shape()),
            g,
            host,
            tol.grad,
            1e-10,
        );
    }

    // um passo do Adam: mesmas constantes dos dois lados
    let (norm, nonfinite) = gpu.grad_stats().unwrap();
    assert_eq!(nonfinite, 0);
    let expected_norm: f64 = params
        .iter()
        .flat_map(|p| p.grad().unwrap().clone().unwrap())
        .map(|g| f64::from(g).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        (norm - expected_norm).abs() <= (tol.grad as f64).max(1e-4) * expected_norm,
        "norma {norm} vs {expected_norm}"
    );
    assert_eq!(gpu.adam_step().unwrap(), 0);
    let mut adam = Adam::new(lr).unwrap();
    let stepped = adam.step(&params).unwrap();
    let after = gpu.params().unwrap();
    for (i, (x, y)) in after.iter().zip(&stepped).enumerate() {
        // no 1º passo o Adam anda ±lr por elemento: só um gradiente quase zero com sinal
        // trocado pelo arredondamento difere, e no máximo 2·lr
        let flipped = x
            .data()
            .iter()
            .zip(y.data())
            .filter(|(a, b)| (*a - *b).abs() > 1e-6)
            .count();
        assert!(
            flipped as f64 <= tol.flips * x.numel().max(1000) as f64,
            "param[{i}]: {flipped} elementos divergem"
        );
        assert_close(
            &format!("param[{i}]"),
            x.data(),
            y.data(),
            0.0,
            2.0 * lr + 1e-6,
        );
    }
    // estado do Adam exportado no formato do host
    let exported = gpu.adam().unwrap();
    assert_eq!(exported.hyperparameters().4, 1);
    assert_eq!(exported.moments().len(), params.len());
    // depois do passo os gradientes voltam a zero
    assert!(gpu
        .grads()
        .unwrap()
        .iter()
        .all(|g| g.iter().all(|&x| x == 0.0)));
}

fn tiny() -> KatsuLmConfig {
    KatsuLmConfig {
        vocab_size: 37,
        block_size: 16,
        embed_dim: 32,
        layers: 2,
        heads: 4,
    }
}

#[test]
fn reference_kernels_match_cpu_autograd() {
    check_parity(
        tiny(),
        GpuTrainOptions {
            gemm: GemmKernel::Reference16,
            fused: false,
        },
    );
}

#[test]
fn fused_epilogues_match_cpu_autograd() {
    check_parity(
        tiny(),
        GpuTrainOptions {
            gemm: GemmKernel::Reference16,
            fused: true,
        },
    );
}

#[test]
fn tiled64_matches_cpu_autograd() {
    check_parity(
        tiny(),
        GpuTrainOptions {
            gemm: GemmKernel::Tiled64,
            fused: true,
        },
    );
}

#[test]
fn odd_sizes_tiled64_match_cpu_autograd() {
    let c = KatsuLmConfig {
        vocab_size: 53,
        block_size: 20,
        embed_dim: 48,
        layers: 1,
        heads: 3,
    };
    check_parity(
        c,
        GpuTrainOptions {
            gemm: GemmKernel::Tiled64,
            fused: true,
        },
    );
}

#[test]
fn tf32_tensor_cores_match_cpu_autograd_within_tf32_precision() {
    check_parity(
        tiny(),
        GpuTrainOptions {
            gemm: GemmKernel::Tf32,
            fused: true,
        },
    );
}

#[test]
fn odd_sizes_match_cpu_autograd() {
    // d e vocabulário fora de múltiplos de 16/32, cabeça de 16
    let c = KatsuLmConfig {
        vocab_size: 53,
        block_size: 20,
        embed_dim: 48,
        layers: 1,
        heads: 3,
    };
    check_parity(
        c,
        GpuTrainOptions {
            gemm: GemmKernel::Reference16,
            fused: false,
        },
    );
}
