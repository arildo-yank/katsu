use std::sync::Arc;

use katsu_core::{
    Adam, Adam8bit, Backend, CpuBackend, KatsuLmConfig, Optimizer, SplitMix64, Tensor, TensorError,
};

fn config() -> KatsuLmConfig {
    KatsuLmConfig {
        vocab_size: 97,
        block_size: 24,
        embed_dim: 48,
        layers: 3,
        heads: 4,
    }
}

fn batch(config: &KatsuLmConfig, seed: u64) -> Vec<Vec<usize>> {
    let mut rng = SplitMix64(seed);
    (0..3)
        .map(|_| {
            (0..config.block_size + 1)
                .map(|_| (rng.next_u64() % config.vocab_size as u64) as usize)
                .collect()
        })
        .collect()
}

fn grads(params: &[Tensor]) -> Result<Vec<Vec<f32>>, TensorError> {
    params
        .iter()
        .map(|p| Ok(p.grad()?.clone().expect("gradient")))
        .collect()
}

#[test]
fn recompute_gives_the_same_loss_and_gradients() -> Result<(), TensorError> {
    let config = config();
    let windows = batch(&config, 1);
    let refs: Vec<&[usize]> = windows.iter().map(Vec::as_slice).collect();
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::default());
    let plain = config.init(2)?;
    let loss = config.loss_batch_with_backend(&plain, &refs, backend.clone())?;
    let expected = loss.data()[0];
    loss.backward_release()?;
    let checkpointed = config.init(2)?;
    let value = config.loss_backward_recompute(&checkpointed, &refs, backend)?;
    assert!((value - expected).abs() <= 1e-6 * expected.abs());
    for (i, (a, b)) in grads(&plain)?.iter().zip(grads(&checkpointed)?).enumerate() {
        let scale = a.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
        let diff = a
            .iter()
            .zip(&b)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(
            diff <= 1e-4 * scale,
            "parâmetro {i}: diferença {diff}, escala {scale}"
        );
    }
    Ok(())
}

#[test]
fn adam8bit_state_is_about_two_bytes_per_parameter() -> Result<(), TensorError> {
    let config = config();
    let windows = batch(&config, 3);
    let refs: Vec<&[usize]> = windows.iter().map(Vec::as_slice).collect();
    let params = config.init(4)?;
    config.loss(&params, &refs)?.backward()?;
    let mut adam = Adam8bit::new(1e-3)?;
    adam.step(&params)?;
    let per_param = adam.state_bytes() as f64 / config.param_count() as f64;
    // 2 bytes + 2 escalas FP32 por bloco de 256 (tensores pequenos arredondam para cima).
    assert!((2.0..2.2).contains(&per_param), "{per_param} bytes/param");
    Ok(())
}

/// O Adam de 8 bits segue o Adam FP32: a mesma inicialização e os mesmos lotes, perda a 3% em
/// cada passo e queda de pelo menos 30% nos dois.
#[test]
fn adam8bit_tracks_fp32_adam() -> Result<(), TensorError> {
    let config = config();
    let windows = batch(&config, 5);
    let refs: Vec<&[usize]> = windows.iter().map(Vec::as_slice).collect();
    let run = |optimizer: &mut dyn Optimizer| -> Result<Vec<f32>, TensorError> {
        let mut params = config.init(6)?;
        let mut losses = Vec::new();
        for _ in 0..25 {
            let loss = config.loss(&params, &refs)?;
            losses.push(loss.data()[0]);
            loss.backward_release()?;
            params = optimizer.step_owned(params)?;
        }
        Ok(losses)
    };
    let fp32 = run(&mut Adam::new(3e-3)?)?;
    let q8 = run(&mut Adam8bit::new(3e-3)?)?;
    for (step, (a, b)) in fp32.iter().zip(&q8).enumerate() {
        assert!(
            (a - b).abs() / a <= 0.03,
            "passo {step}: fp32 {a}, 8 bits {b}"
        );
    }
    assert!(fp32[24] < 0.7 * fp32[0] && q8[24] < 0.7 * q8[0]);
    Ok(())
}
