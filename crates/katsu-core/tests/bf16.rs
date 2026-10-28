use std::sync::Arc;

use katsu_core::{
    bf16_to_f32, f32_to_bf16, matmul_bf16, round_bf16, Adam, Backend, Bf16Backend, Bf16Buffer,
    CpuBackend, KatsuLmConfig, Optimizer, SplitMix64, TensorError,
};

#[test]
fn conversion_rounds_to_nearest_even_and_keeps_specials() {
    assert_eq!(f32_to_bf16(1.0), 0x3f80);
    assert_eq!(bf16_to_f32(0x3f80), 1.0);
    // 1 + 2^-8 fica no meio entre 1 e 1 + 2^-7: empate vai para o par (1.0).
    assert_eq!(round_bf16(1.0 + 1.0 / 256.0), 1.0);
    // 1 + 3·2^-8 empata entre 1 + 2^-7 (ímpar) e 1 + 2^-6 (par).
    assert_eq!(round_bf16(1.0 + 3.0 / 256.0), 1.0 + 1.0 / 64.0);
    assert_eq!(round_bf16(-2.5), -2.5);
    assert!(round_bf16(f32::NAN).is_nan());
    assert_eq!(round_bf16(f32::INFINITY), f32::INFINITY);
    assert_eq!(round_bf16(f32::MAX), f32::INFINITY);
    assert_eq!(round_bf16(-0.0).to_bits(), (-0.0f32).to_bits());
    // Erro relativo de arredondamento <= 2^-8.
    let mut rng = SplitMix64(3);
    for _ in 0..10_000 {
        let x = (rng.next_u64() as f32 / u64::MAX as f32 - 0.5) * 1e3;
        assert!((round_bf16(x) - x).abs() <= x.abs() / 256.0);
    }
}

#[test]
fn buffer_uses_two_bytes_per_value() -> Result<(), TensorError> {
    let values = vec![0.1f32; 1000];
    let buffer = Bf16Buffer::from_f32(&values)?;
    assert_eq!(buffer.bytes(), 2000);
    assert!(buffer.to_f32()?.iter().all(|&v| v == round_bf16(0.1)));
    Ok(())
}

#[test]
fn mixed_backend_matches_reference_bf16_kernel() -> Result<(), TensorError> {
    let (m, k, n) = (17, 300, 23);
    let mut rng = SplitMix64(9);
    let mut random = |len: usize| -> Vec<f32> {
        (0..len)
            .map(|_| rng.next_u64() as f32 / u64::MAX as f32 - 0.5)
            .collect()
    };
    let (a, b) = (random(m * k), random(k * n));
    let reference = matmul_bf16(
        &Bf16Buffer::from_f32(&a)?,
        &Bf16Buffer::from_f32(&b)?,
        m,
        k,
        n,
    )?;
    let mixed = Bf16Backend::new(Arc::new(CpuBackend::default())).matmul(&a, &b, m, k, n)?;
    let exact = CpuBackend::default().matmul(&a, &b, m, k, n)?;
    for ((&r, &x), &e) in reference.iter().zip(&mixed).zip(&exact) {
        // A saída do backend é a referência arredondada para BF16.
        assert!((round_bf16(r) - x).abs() <= r.abs() / 128.0 + 1e-6);
        // Contra o FP32 puro, o erro é o dos operandos BF16 (~2^-8 relativo por termo).
        assert!((x - e).abs() <= 0.05 * (k as f32).sqrt() / 16.0);
    }
    Ok(())
}

/// Paridade de treino: o mesmo modelo, a mesma inicialização e os mesmos lotes, em FP32 e em
/// precisão mista BF16 (pesos-mestre e Adam em FP32). Tolerância: perda BF16 a 1% da FP32 em
/// cada passo, e as duas caindo.
#[test]
fn bf16_training_tracks_fp32_within_one_percent() -> Result<(), TensorError> {
    let config = KatsuLmConfig {
        vocab_size: 97,
        block_size: 24,
        embed_dim: 48,
        layers: 2,
        heads: 4,
    };
    let mut rng = SplitMix64(11);
    let windows: Vec<Vec<usize>> = (0..4)
        .map(|_| {
            (0..config.block_size + 1)
                .map(|_| (rng.next_u64() % config.vocab_size as u64) as usize)
                .collect()
        })
        .collect();
    let refs: Vec<&[usize]> = windows.iter().map(Vec::as_slice).collect();
    let run = |backend: Arc<dyn Backend>| -> Result<Vec<f32>, TensorError> {
        let mut params = config.init(5)?;
        let mut adam = Adam::new(3e-3)?;
        let mut losses = Vec::new();
        for _ in 0..8 {
            let loss = config.loss_batch_with_backend(&params, &refs, backend.clone())?;
            losses.push(loss.data()[0]);
            loss.backward_release()?;
            params = adam.step_owned(params)?;
        }
        Ok(losses)
    };
    let cpu: Arc<dyn Backend> = Arc::new(CpuBackend::default());
    let fp32 = run(cpu.clone())?;
    let bf16 = run(Arc::new(Bf16Backend::new(cpu)))?;
    for (step, (a, b)) in fp32.iter().zip(&bf16).enumerate() {
        assert!(
            (a - b).abs() / a <= 0.01,
            "passo {step}: fp32 {a}, bf16 {b}"
        );
    }
    assert!(fp32[7] < fp32[0] * 0.9 && bf16[7] < bf16[0] * 0.9);
    Ok(())
}
