//! Paridade do treino inteiro na GPU com o autograd da CPU: perda, gradiente de cada tensor
//! e pesos depois de um passo do Adam. Sem GPU (ou fora do Linux) os testes são pulados.
#![cfg(feature = "cuda")]
use katsu_core::{
    step_partial, Adam, Adam8bit, CpuBackend, GemmKernel, GpuTrainOptions, GpuTrainer,
    KatsuLmConfig, Optimizer, SftBatch, Tensor, TensorError,
};
use std::sync::Arc;

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
/// BF16 (7 bits de mantissa, ~8× o erro do TF32) nos operandos e nas ativações dos GEMMs. O
/// critério principal da V2 é a curva de perda (`v2_options_together_track_fp32_cpu_loss`).
const BF16: Tol = Tol {
    loss: 1e-2,
    grad: 0.25,
    flips: 0.2,
};

fn tolerance(options: GpuTrainOptions) -> Tol {
    match options.gemm {
        GemmKernel::Tf32 => TF32,
        GemmKernel::Bf16 => BF16,
        _ => STRICT,
    }
}

fn check_parity(c: KatsuLmConfig, options: GpuTrainOptions) {
    let tol = tolerance(options);
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
        },
    );
}

// ---------------------------------------------------------------------------------------
// Arandu-Ve, etapa 0: entrada por vetores (prefixo) e congelamento, contra a CPU.

fn prefix_data(rows: usize, d: usize, seed: u64) -> Vec<f32> {
    tokens(seed, rows * d, 2001)
        .into_iter()
        .map(|x| (x as f32 / 1000.0 - 1.0) * 0.5)
        .collect()
}

/// Perda, gradiente de cada tensor e gradiente dos vetores do prefixo iguais ao `prefix_loss`
/// da CPU. Com `perturb`, um vetor do lote da GPU muda 10% e a comparação do gradiente do
/// prefixo tem de falhar (controle negativo).
fn check_prefix_parity(c: KatsuLmConfig, options: GpuTrainOptions, perturb: bool) {
    let tol = tolerance(options);
    let params = c.init(13).unwrap();
    let (a, b, e) = (
        tokens(4, 9, c.vocab_size),
        tokens(5, 6, c.vocab_size),
        tokens(6, 12, c.vocab_size),
    );
    let examples: Vec<Example> = vec![(&a, 3), (&b, 1), (&e, 7)];
    let n = 3;
    let d = c.embed_dim;
    let data = prefix_data(examples.len() * n, d, 7);
    let prefix = Tensor::new(data.clone(), &[examples.len() * n, d]).unwrap();
    let Some(mut gpu) = trainer(c, &params, &Adam::new(1e-3).unwrap(), 4, options) else {
        return;
    };
    let mut sent = data.clone();
    if perturb {
        sent[d + 2] *= 1.1;
    }
    let batch = SftBatch::with_prefix(&examples, &sent, n, d).unwrap();
    let loss = gpu.loss_and_grad(&batch).unwrap();
    let cpu = c
        .prefix_loss(
            &params,
            &prefix,
            n,
            &examples,
            Arc::new(CpuBackend::default()),
        )
        .unwrap();
    let expected = f64::from(cpu.data()[0]);
    cpu.backward().unwrap();
    let host_prefix = prefix.grad().unwrap().as_ref().unwrap().clone();
    let check_prefix = || {
        assert_close(
            "prefixo",
            &gpu.prefix_grad().unwrap(),
            &host_prefix,
            tol.grad,
            1e-10,
        )
    };
    if perturb {
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(check_prefix)).is_err();
        assert!(
            failed,
            "o controle negativo passou: o teste não enxerga um prefixo errado"
        );
        return;
    }
    assert!(
        (loss - expected).abs() <= tol.loss * expected.abs(),
        "perda GPU {loss} vs CPU {expected}"
    );
    check_prefix();
    for (i, (p, g)) in params.iter().zip(gpu.grads().unwrap()).enumerate() {
        let host = p.grad().unwrap();
        let host = host.as_ref().expect("gradiente da CPU");
        assert_close(
            &format!("grad[{i}] {:?}", p.shape()),
            &g,
            host,
            tol.grad,
            1e-10,
        );
    }
}

#[test]
fn prefix_input_matches_cpu_autograd() {
    check_prefix_parity(
        tiny(),
        GpuTrainOptions {
            gemm: GemmKernel::Tiled64,
            fused: true,
            ..Default::default()
        },
        false,
    );
}

#[test]
fn prefix_input_matches_cpu_autograd_within_tf32_precision() {
    check_prefix_parity(
        tiny(),
        GpuTrainOptions {
            gemm: GemmKernel::Tf32,
            fused: true,
            ..Default::default()
        },
        false,
    );
}

#[test]
fn prefix_parity_detects_a_wrong_vector() {
    check_prefix_parity(
        tiny(),
        GpuTrainOptions {
            gemm: GemmKernel::Tiled64,
            fused: true,
            ..Default::default()
        },
        true,
    );
}

#[test]
fn frozen_tensors_stay_bit_identical_on_gpu() {
    let c = tiny();
    let params = c.init(17).unwrap();
    let (a, b) = (tokens(8, 10, c.vocab_size), tokens(9, 7, c.vocab_size));
    let examples: Vec<Example> = vec![(&a, 4), (&b, 2)];
    let lr = 1e-3;
    let options = GpuTrainOptions {
        gemm: GemmKernel::Tiled64,
        fused: true,
        ..Default::default()
    };
    let Some(mut gpu) = trainer(c, &params, &Adam::new(lr).unwrap(), 4, options) else {
        return;
    };
    // tudo congelado menos o embedding de posição, uma projeção Q (fatia intercalada da
    // matriz QKV) e a cabeça de saída
    let q = 2 + 2; // depois de wte, wpe, ln1_g, ln1_b vem o Q da cabeça 0
    let trainable = [1, q, params.len() - 1];
    let frozen: Vec<bool> = (0..params.len()).map(|i| !trainable.contains(&i)).collect();
    gpu.set_frozen(&frozen).unwrap();
    gpu.loss_and_grad(&SftBatch::new(&examples).unwrap())
        .unwrap();
    assert_eq!(gpu.adam_step().unwrap(), 0);
    let after = gpu.params().unwrap();
    cpu_loss(&c, &params, &examples).backward().unwrap();
    let stepped = step_partial(&mut Adam::new(lr).unwrap(), &params, &frozen).unwrap();
    for (i, (x, y)) in after.iter().zip(&stepped).enumerate() {
        if frozen[i] {
            assert_eq!(
                x.data(),
                params[i].data(),
                "tensor congelado {i} mudou na GPU"
            );
        } else {
            assert_ne!(x.data(), params[i].data(), "tensor treinável {i} não mudou");
            assert_close(
                &format!("param[{i}]"),
                x.data(),
                y.data(),
                0.0,
                2.0 * lr + 1e-6,
            );
        }
    }
    assert!(gpu.grads().unwrap().iter().flatten().all(|&g| g == 0.0));
    assert!(gpu.set_frozen(&frozen[1..]).is_err());
    assert!(
        gpu.prefix_grad().is_err(),
        "o último lote não tinha prefixo"
    );
}

// ---------------------------------------------------------------------------------------
// V2: BF16 nos Tensor Cores, Adam de 8 bits e recomputação de ativações, contra a CPU.

#[test]
fn bf16_tensor_cores_match_cpu_autograd_within_bf16_precision() {
    check_parity(
        tiny(),
        GpuTrainOptions {
            gemm: GemmKernel::Bf16,
            ..Default::default()
        },
    );
}

#[test]
fn recompute_matches_cpu_autograd() {
    for gemm in [GemmKernel::Tiled64, GemmKernel::Bf16] {
        check_parity(
            tiny(),
            GpuTrainOptions {
                gemm,
                recompute: true,
                ..Default::default()
            },
        );
    }
}

/// Com e sem recomputação, na mesma GPU: a mesma perda e os mesmos gradientes (só a ordem dos
/// atomicAdd do backward muda).
#[test]
fn recompute_gives_the_same_gradients_as_stored_activations() {
    let c = KatsuLmConfig {
        layers: 3,
        ..tiny()
    };
    let params = c.init(19).unwrap();
    let (a, b) = (tokens(10, 14, c.vocab_size), tokens(11, 9, c.vocab_size));
    let examples: Vec<Example> = vec![(&a, 5), (&b, 3)];
    let batch = SftBatch::new(&examples).unwrap();
    let mut results = Vec::new();
    for recompute in [false, true] {
        let options = GpuTrainOptions {
            gemm: GemmKernel::Tiled64,
            recompute,
            ..Default::default()
        };
        let Some(mut gpu) = trainer(c, &params, &Adam::new(1e-3).unwrap(), 2, options) else {
            return;
        };
        let loss = gpu.loss_and_grad(&batch).unwrap();
        results.push((loss, gpu.grads().unwrap()));
    }
    let ((plain_loss, plain), (loss, grads)) = (&results[0], &results[1]);
    assert_eq!(
        plain_loss, loss,
        "a perda do forward não depende da recomputação"
    );
    for (i, (x, y)) in grads.iter().zip(plain).enumerate() {
        assert_close(&format!("grad[{i}]"), x, y, 1e-6, 1e-12);
    }
}

/// Perdas da GPU e da CPU, pesos finais da GPU e da CPU.
type Trained = (Vec<f64>, Vec<f64>, Vec<Tensor>, Vec<Tensor>);

/// `steps` passos no mesmo lote, na GPU e na CPU, a partir dos mesmos pesos; devolve as perdas
/// e os pesos finais dos dois lados.
fn train_both(
    c: KatsuLmConfig,
    options: GpuTrainOptions,
    cpu_optimizer: &mut dyn Optimizer,
    lr: f32,
    steps: usize,
) -> Option<Trained> {
    let params = c.init(23).unwrap();
    let (a, b, d) = (
        tokens(12, 15, c.vocab_size),
        tokens(13, 10, c.vocab_size),
        tokens(14, 16, c.vocab_size),
    );
    let examples: Vec<Example> = vec![(&a, 3), (&b, 2), (&d, 6)];
    let batch = SftBatch::new(&examples).unwrap();
    let mut gpu = trainer(c, &params, &Adam::new(lr).unwrap(), 4, options)?;
    let mut gpu_losses = Vec::new();
    for _ in 0..steps {
        gpu_losses.push(gpu.loss_and_grad(&batch).unwrap());
        assert_eq!(gpu.adam_step().unwrap(), 0);
    }
    let mut host = params;
    let mut cpu_losses = Vec::new();
    for _ in 0..steps {
        let loss = cpu_loss(&c, &host, &examples);
        cpu_losses.push(f64::from(loss.data()[0]));
        loss.backward_release().unwrap();
        host = cpu_optimizer.step_owned(host).unwrap();
    }
    Some((gpu_losses, cpu_losses, gpu.params().unwrap(), host))
}

/// Adam de 8 bits na GPU contra o `Adam8bit` da CPU, 8 passos. Os blocos de quantização não
/// coincidem (na GPU cobrem a arena, na CPU cada tensor), então não é bit a bit (isso é
/// conferido no teste do kernel, em gpu_train.rs): perda a 0,5% e pesos a menos de meio passo
/// máximo do Adam por passo.
#[test]
fn adam8_on_gpu_tracks_cpu_adam8bit() {
    let (lr, steps) = (3e-3, 8);
    let options = GpuTrainOptions {
        gemm: GemmKernel::Tiled64,
        adam8: true,
        ..Default::default()
    };
    let mut cpu = Adam8bit::new(lr).unwrap();
    let Some((gpu, host, after, expected)) = train_both(tiny(), options, &mut cpu, lr, steps)
    else {
        return;
    };
    for (s, (x, y)) in gpu.iter().zip(&host).enumerate() {
        assert!(
            (x - y).abs() <= 5e-3 * y.abs(),
            "passo {s}: GPU {x}, CPU {y}"
        );
    }
    assert!(gpu[steps - 1] < gpu[0], "a perda não caiu: {gpu:?}");
    for (i, (x, y)) in after.iter().zip(&expected).enumerate() {
        assert_close(
            &format!("param[{i}]"),
            x.data(),
            y.data(),
            0.0,
            0.5 * lr * steps as f32,
        );
    }
}

/// Exportar o estado de 8 bits (dequantizado) e carregá-lo de volta continua o treino.
#[test]
fn adam8_state_survives_export_and_reload() {
    let c = tiny();
    let params = c.init(29).unwrap();
    let a = tokens(15, 12, c.vocab_size);
    let batch = SftBatch::new(&[(&a, 4)]).unwrap();
    let options = GpuTrainOptions {
        gemm: GemmKernel::Tiled64,
        adam8: true,
        ..Default::default()
    };
    let Some(mut gpu) = trainer(c, &params, &Adam::new(1e-3).unwrap(), 1, options) else {
        return;
    };
    for _ in 0..3 {
        gpu.loss_and_grad(&batch).unwrap();
        gpu.adam_step().unwrap();
    }
    let exported = gpu.adam().unwrap();
    assert_eq!(exported.hyperparameters().4, 3);
    assert_eq!(exported.moments().len(), params.len());
    assert!(exported
        .moments()
        .iter()
        .any(|(m, _)| m.iter().any(|&x| x != 0.0)));
    let weights = gpu.params().unwrap();
    let mut resumed = trainer(c, &weights, &exported, 1, options).unwrap();
    // requantizar o estado exportado devolve os mesmos momentos, a menos de um degrau
    let reloaded = resumed.adam().unwrap();
    assert_eq!(reloaded.hyperparameters().4, 3);
    for (i, ((m, v), (m2, v2))) in exported
        .moments()
        .iter()
        .zip(reloaded.moments())
        .enumerate()
    {
        assert_close(&format!("m[{i}]"), m2, m, 2.0 / 127.0, 1e-20);
        assert_close(&format!("v[{i}]"), v2, v, 4.0 / 255.0, 1e-20);
    }
    let (x, y) = (
        gpu.loss_and_grad(&batch).unwrap(),
        resumed.loss_and_grad(&batch).unwrap(),
    );
    assert_eq!(x, y, "mesmos pesos, mesma perda");
    gpu.adam_step().unwrap();
    resumed.adam_step().unwrap();
    for (i, (p, q)) in gpu
        .params()
        .unwrap()
        .iter()
        .zip(resumed.params().unwrap())
        .enumerate()
    {
        assert_close(&format!("param[{i}]"), p.data(), q.data(), 0.0, 0.5e-3);
    }
}

/// As três opções da V2 juntas (BF16, Adam de 8 bits, recomputação) contra o treino FP32 da
/// CPU com Adam FP32, como o teste de paridade do 134M no Mac: 8 passos, perda a 1% em cada.
#[test]
fn v2_options_together_track_fp32_cpu_loss() {
    let (lr, steps) = (1e-3, 8);
    let options = GpuTrainOptions {
        gemm: GemmKernel::Bf16,
        adam8: true,
        recompute: true,
        ..Default::default()
    };
    let c = KatsuLmConfig {
        embed_dim: 64,
        layers: 3,
        ..tiny()
    };
    let mut cpu = Adam::new(lr).unwrap();
    let Some((gpu, host, _, _)) = train_both(c, options, &mut cpu, lr, steps) else {
        return;
    };
    for (s, (x, y)) in gpu.iter().zip(&host).enumerate() {
        if std::env::var_os("KATSU_PARITY_VERBOSE").is_some() {
            eprintln!(
                "passo {s}: GPU {x:.6}, CPU {y:.6}, {:.3}%",
                100.0 * (x - y).abs() / y
            );
        }
        assert!(
            (x - y).abs() <= 1e-2 * y.abs(),
            "passo {s}: GPU {x}, CPU {y}"
        );
    }
    assert!(gpu[steps - 1] < gpu[0], "a perda não caiu: {gpu:?}");
}
